// Range-safe BF16 GEMM (docs/vulkan-bf16.md): BF16 operands whose A rows run
// from 1e-6 to past 1e6 (FP16 tops out at 65504), through every matrix op
// and epilogue the cooperative-matrix kernel serves, against the CPU op in
// FP32 on the same BF16-rounded inputs. Shapes cover partial tiles, K not a
// multiple of 8 (the scalar loads), an unaligned A view, batched / broadcast
// / padded strides, the transposed-A layout (the backwards) and the GEMV
// kernel; each runs on the cooperative-matrix kernel and on the SIMT
// fallback. With the scaling switched off the same inputs must overflow, so
// the cases do exercise the range.
//
// Tolerance, per output row: rel * max|row of want| + two BF16 ulps of the
// element (the output's rounding). The scaled staging is exact for BF16 A
// (its 8 significant bits fit FP16's 11 at any power-of-two scale), so what
// remains is the FP32 summation order.

#include "test_vulkan_common.h"

#include "detail/gemm.h"
#include "detail/device.h"

#include <brotensor/ops/linear.h>

#include <array>
#include <cmath>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

Tensor cpu_t(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

// Row r scaled by 10^(((r * 5) % 13) - 6), i.e. 1e-6 .. 1e6, every fourth row
// with one outlier column at 30x its scale (T5's activation pattern: a few
// channels far above the rest), one all-zero row. BF16-rounded.
std::vector<float> wide_rows(int rows, int cols, std::uint64_t seed) {
    auto v = random_values(std::size_t(rows) * cols, seed, -1.0f, 1.0f, Dtype::FP32);
    for (int r = 0; r < rows; ++r) {
        const float s = std::pow(10.0f, float((r * 5) % 13 - 6));
        for (int c = 0; c < cols; ++c) v[std::size_t(r) * cols + c] *= s;
        if (r % 4 == 1) v[std::size_t(r) * cols + (r * 7) % cols] = 30.0f * s;
        if (rows > 2 && r == rows / 2) {
            for (int c = 0; c < cols; ++c) v[std::size_t(r) * cols + c] = 0.0f;
        }
    }
    for (float& x : v) x = round_to(Dtype::BF16, x);
    return v;
}

float max_abs(const std::vector<float>& v) {
    float m = 0;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

bool any_inf_nan(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return true;
    return false;
}

// got / want are (rows, ld) row-major; columns [0, cols) are compared.
void check_rows(const std::vector<float>& got, const std::vector<float>& want, int rows, int cols, int ld,
                float rel, const std::string& tag) {
    std::vector<float> g, w;
    std::vector<float> tol;
    for (int r = 0; r < rows; ++r) {
        float m = 0;
        for (int c = 0; c < cols; ++c) {
            const float x = want[std::size_t(r) * ld + c];
            if (std::isfinite(x)) m = std::max(m, std::fabs(x));
        }
        for (int c = 0; c < cols; ++c) {
            const std::size_t i = std::size_t(r) * ld + c;
            g.push_back(got[i]);
            w.push_back(round_to(Dtype::BF16, want[i]));
            tol.push_back(rel * m + 2.0f * dtype_eps(Dtype::BF16) * std::fabs(want[i]) + 1e-30f);
        }
    }
    // expect_close with a per-element atol: scale both sides by 1 / tol.
    std::vector<float> gs(g.size()), ws(w.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        gs[i] = std::isfinite(w[i]) ? g[i] / tol[i] : g[i];
        ws[i] = std::isfinite(w[i]) ? w[i] / tol[i] : w[i];
    }
    expect_close(gs, ws, 1.0f, 0.0f, tag + " (in units of the row tolerance)");
}

const char* mode_name(int mode) { return mode == 0 ? "coopmat" : mode == 1 ? "no-gemv" : "simt"; }

// ─── matmul_abt ────────────────────────────────────────────────────────────

struct Case {
    int batch, M, N, K;
    long long sa, sb, sc;   // -1 = packed
    bool bias;
    int act;
    const char* what;
    int a_offset;           // elements: an unaligned A view (scalar loads)
};

void run_abt(const Case& c, int mode, std::uint64_t seed) {
    const long long sa = c.sa < 0 ? (long long)c.M * c.K : c.sa;
    const long long sb = c.sb < 0 ? (long long)c.N * c.K : c.sb;
    const long long sc = c.sc < 0 ? (long long)c.M * c.N : c.sc;
    const int arows = sa == 0 ? c.M : int(((c.batch - 1) * sa + (long long)c.M * c.K + c.K - 1) / c.K);
    const auto av = wide_rows(arows, c.K, seed);
    const std::size_t nb = std::size_t((c.batch - 1) * sb + (long long)c.N * c.K);
    const std::size_t nc = std::size_t((c.batch - 1) * sc + (long long)c.M * c.N);
    const auto bv = random_values(nb, seed + 1, -1, 1, Dtype::BF16);
    const auto biasv = random_values(std::size_t(c.N), seed + 2, -0.5f, 0.5f, Dtype::BF16);
    std::vector<float> want(nc, 0.0f);
    Tensor bc = cpu_t(biasv, c.N, 1);
    for (int z = 0; z < c.batch; ++z) {
        std::vector<float> as(av.begin() + z * sa, av.begin() + z * sa + std::size_t(c.M) * c.K);
        std::vector<float> bs(bv.begin() + z * sb, bv.begin() + z * sb + std::size_t(c.N) * c.K);
        Tensor y;
        brotensor::linear_forward_batched_ex(cpu_t(bs, c.N, c.K), c.bias ? &bc : nullptr, cpu_t(as, c.M, c.K), c.act,
                                             brotensor::kLinearEpiStore, nullptr, y);
        const auto yv = y.to_host_vector();
        std::copy(yv.begin(), yv.end(), want.begin() + z * sc);
    }
    std::vector<float> apad(std::size_t(c.a_offset), 0.0f);
    apad.insert(apad.end(), av.begin(), av.end());
    Tensor abuf = upload(apad, int(apad.size()), 1, Dtype::BF16);
    Tensor A = Tensor::view(vk(), static_cast<char*>(abuf.data) + c.a_offset * 2, int(av.size()), 1, Dtype::BF16);
    Tensor B = upload(bv, int(nb), 1, Dtype::BF16), bias = upload(biasv, c.N, 1, Dtype::BF16);
    Tensor C = Tensor::zeros_on(vk(), int(nc), 1, Dtype::BF16);
    brotensor::matmul_abt(A, B, C, c.batch, c.M, c.N, c.K, sa, sb, sc, c.bias ? &bias : nullptr, c.act);
    const auto got = download(C);
    char tag[160];
    std::snprintf(tag, sizeof tag, "wide bf16 matmul_abt %dx%dx%d b%d %s [%s]", c.M, c.N, c.K, c.batch, c.what,
                  mode_name(mode));
    // Batch slices with padded strides: compare row by row over the whole C
    // (the gaps are zero on both sides).
    check_rows(got, want, int(nc / c.N), c.N, c.N, 2e-5f, tag);
}

void test_abt() {
    std::printf("range-safe BF16: matmul_abt\n");
    using brotensor::kLinearActGeluExact;
    using brotensor::kLinearActGeluTanh;
    using brotensor::kLinearActQuickGelu;
    using brotensor::kLinearActRelu;
    using brotensor::kLinearActSilu;
    const Case cases[] = {
        {1, 300, 200, 1000, -1, -1, -1, false, 0, "plain", 0},
        {1, 129, 257, 1000, -1, -1, -1, true, kLinearActGeluExact, "bias gelu_exact", 0},
        {1, 64, 96, 64, -1, -1, -1, true, kLinearActRelu, "bias relu", 0},
        {1, 77, 130, 136, -1, -1, -1, true, kLinearActSilu, "bias silu", 0},
        {1, 513, 384, 256, -1, -1, -1, false, kLinearActQuickGelu, "quick_gelu", 0},
        {1, 40, 72, 99, -1, -1, -1, true, kLinearActGeluTanh, "K%8 != 0", 0},
        {1, 70, 50, 72, -1, -1, -1, true, 0, "unaligned A", 3},
        {8, 33, 47, 64, -1, -1, -1, false, 0, "heads", 0},
        {6, 17, 40, 64, -1, 0, -1, false, 0, "B broadcast", 0},
        {3, 48, 40, 64, 0, -1, -1, true, 0, "A broadcast", 0},
        {3, 20, 24, 40, 20 * 40 + 8, 24 * 40 + 3, 20 * 24 + 7, true, 0, "padded strides", 0},
    };
    std::uint64_t seed = 9000;
    for (int mode : {0, 2}) {
        dv::set_gemm_override(mode);
        for (const Case& c : cases) run_abt(c, mode, seed += 10);
    }
    dv::set_gemm_override(0);
}

// ─── linear family: every epilogue, and the GEMV kernel ────────────────────

void test_linear() {
    std::printf("range-safe BF16: linear_forward_batched_ex (all epilogues)\n");
    const char* epi_names[] = {"store", "accum", "geglu", "swiglu"};
    std::uint64_t seed = 9500;
    for (int mode : {0, 1, 2}) {
        dv::set_gemm_override(mode);
        for (int B : {3, 64, 200}) {
            if (mode != 0 && B != 64) continue;
            for (const auto& kd : {std::array<int, 2>{1000, 136}, std::array<int, 2>{72, 96}}) {
                const int K = kd[0], N = kd[1];
                for (int epi = 0; epi < 4; ++epi) {
                    const int act = epi == 0 ? (B % 5) + 1 : 0;
                    const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, Dtype::BF16);
                    const auto xv = wide_rows(B, K, seed++);
                    const auto bv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, Dtype::BF16);
                    const auto y0 = wide_rows(B, N, seed++);
                    const int out = epi >= 2 ? N / 2 : N;
                    Tensor bc = cpu_t(bv, N, 1), yc = epi == 1 ? cpu_t(y0, B, N) : Tensor();
                    brotensor::linear_forward_batched_ex(cpu_t(wv, N, K), &bc, cpu_t(xv, B, K), act, epi, nullptr, yc);
                    Tensor W = upload(wv, N, K, Dtype::BF16), X = upload(xv, B, K, Dtype::BF16);
                    Tensor bias = upload(bv, N, 1, Dtype::BF16);
                    Tensor Y = epi == 1 ? upload(y0, B, N, Dtype::BF16) : Tensor();
                    brotensor::linear_forward_batched_ex(W, &bias, X, act, epi, nullptr, Y);
                    char tag[128];
                    std::snprintf(tag, sizeof tag, "wide bf16 linear_ex %s B%d K%d N%d act%d [%s]", epi_names[epi], B, K,
                                  N, act, mode_name(mode));
                    // A GLU multiplies one half's summation error by the other.
                    check_rows(download(Y), yc.to_host_vector(), B, out, out, epi >= 2 ? 1e-4f : 2e-5f, tag);
                }
            }
        }
    }
    dv::set_gemm_override(0);
}

// ─── matmul (NN) and the backwards (transposed A) ──────────────────────────

void test_nn_and_backward() {
    std::printf("range-safe BF16: matmul (NN), linear_backward_batched (transposed A)\n");
    std::uint64_t seed = 9800;
    for (int mode : {0, 2}) {
        dv::set_gemm_override(mode);
        {
            const int M = 150, K = 264, N = 96;
            const auto av = wide_rows(M, K, seed++);
            const auto bv = random_values(std::size_t(K) * N, seed++, -1, 1, Dtype::BF16);
            Tensor cc, cg;
            brotensor::matmul(cpu_t(av, M, K), cpu_t(bv, K, N), cc);
            brotensor::matmul(upload(av, M, K, Dtype::BF16), upload(bv, K, N, Dtype::BF16), cg);
            check_rows(download(cg), cc.to_host_vector(), M, N, N, 2e-5f,
                       std::string("wide bf16 matmul NN [") + mode_name(mode) + "]");
        }
        {
            // dY with per-row and per-column magnitudes: dX = dY W scales A
            // per row; dW += dY^T X has A = dY^T, so its rows are dY's columns.
            const int B = 136, in = 80, out = 72;
            auto gv = wide_rows(B, out, seed++);
            for (int b = 0; b < B; ++b)
                for (int o = 0; o < out; ++o)
                    gv[std::size_t(b) * out + o] =
                        round_to(Dtype::BF16, gv[std::size_t(b) * out + o] * std::pow(10.0f, float(o % 5 - 2)));
            const auto wv = random_values(std::size_t(out) * in, seed++, -1, 1, Dtype::BF16);
            const auto xv = random_values(std::size_t(B) * in, seed++, -1, 1, Dtype::BF16);
            const auto dw0 = random_values(std::size_t(out) * in, seed++, -1, 1, Dtype::BF16);
            const std::vector<float> db0(std::size_t(out), 0.0f);
            Tensor dxc, dwc = cpu_t(dw0, out, in), dbc = cpu_t(db0, out, 1);
            brotensor::linear_backward_batched(cpu_t(wv, out, in), cpu_t(xv, B, in), cpu_t(gv, B, out), dxc, dwc, dbc);
            Tensor dxg, dwg = upload(dw0, out, in, Dtype::BF16), dbg = upload(db0, out, 1, Dtype::BF16);
            brotensor::linear_backward_batched(upload(wv, out, in, Dtype::BF16), upload(xv, B, in, Dtype::BF16),
                                               upload(gv, B, out, Dtype::BF16), dxg, dwg, dbg);
            const std::string m = std::string(" [") + mode_name(mode) + "]";
            check_rows(download(dxg), dxc.to_host_vector(), B, in, in, 2e-5f, "wide bf16 linear_backward dX" + m);
            check_rows(download(dwg), dwc.to_host_vector(), out, in, in, 2e-5f, "wide bf16 linear_backward dW" + m);
        }
    }
    dv::set_gemm_override(0);
}

// ─── the range is real: without the scaling the same GEMM overflows ────────

void test_unscaled_overflows() {
    dv::DeviceCtx& d = dv::device(0);
    if (!d.info().coopmat_f16) {
        std::printf("range-safe BF16: no cooperative matrix, unscaled check skipped\n");
        return;
    }
    std::printf("range-safe BF16: scaling off overflows, scaling on does not\n");
    const int M = 96, N = 64, K = 128;
    const auto av = wide_rows(M, K, 9900);
    const auto bv = random_values(std::size_t(N) * K, 9901, -1, 1, Dtype::BF16);
    VKT_CHECK(max_abs(av) > 65504.0f);
    Tensor A = upload(av, M, K, Dtype::BF16), B = upload(bv, N, K, Dtype::BF16), C;
    dv::set_gemm_scaling(0);
    brotensor::matmul_abt(A, B, C, 1, M, N, K, 0, 0, 0, nullptr, 0);
    VKT_CHECK(any_inf_nan(download(C)));
    dv::set_gemm_scaling(1);
    brotensor::matmul_abt(A, B, C, 1, M, N, K, 0, 0, 0, nullptr, 0);
    VKT_CHECK(!any_inf_nan(download(C)));
    std::printf("  PASS  unscaled staging overflows, scaled does not\n");
}

// A row holding inf keeps its exponent at 0 and propagates the inf; the
// other rows are unaffected.
void test_nonfinite_row() {
    std::printf("range-safe BF16: a non-finite row\n");
    const int M = 40, N = 32, K = 64;
    auto av = wide_rows(M, K, 9950);
    av[std::size_t(5) * K + 3] = INFINITY;
    const auto bv = random_values(std::size_t(N) * K, 9951, 0.25f, 1, Dtype::BF16);   // positive: no inf - inf
    Tensor cc;   // (the CPU matmul_abt is 16-bit only)
    brotensor::linear_forward_batched_ex(cpu_t(bv, N, K), nullptr, cpu_t(av, M, K), 0, brotensor::kLinearEpiStore,
                                         nullptr, cc);
    Tensor C;
    brotensor::matmul_abt(upload(av, M, K, Dtype::BF16), upload(bv, N, K, Dtype::BF16), C, 1, M, N, K, 0, 0, 0,
                          nullptr, 0);
    check_rows(download(C), cc.to_host_vector(), M, N, N, 2e-5f, "wide bf16 inf row");
}

// ─── the forms the attention ops use: 16-bit operands with an FP32 C, and an
// FP32 A (its own intermediate, round_a) against a 16-bit B ───────────────

void test_fp32_forms() {
    std::printf("range-safe GEMM: FP32 C from 16-bit operands, FP32 A with round_a\n");
    dv::DeviceCtx& d = dv::device(0);
    const int M = 77, N = 130, K = 136;
    std::uint64_t seed = 9970;
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (bool wide_a : {false, true}) {
            if (wide_a && dt == Dtype::FP16) continue;
            // 16-bit A, B and bias; FP32 C (the bias shares the operands' dtype).
            const auto av = wide_a ? wide_rows(M, K, seed++) : random_values(std::size_t(M) * K, seed++, -1, 1, dt);
            const auto bv = random_values(std::size_t(N) * K, seed++, -1, 1, dt);
            const auto biasv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, dt);
            Tensor bc = cpu_t(biasv, N, 1), yc;
            brotensor::linear_forward_batched_ex(cpu_t(bv, N, K), &bc, cpu_t(av, M, K), brotensor::kLinearActSilu,
                                                 brotensor::kLinearEpiStore, nullptr, yc);
            Tensor A = upload(av, M, K, dt), B = upload(bv, N, K, dt), bias = upload(biasv, N, 1, dt);
            Tensor C = Tensor::empty_on(vk(), M, N, Dtype::FP32);
            dv::GemmArgs g;
            g.a = dv::addr(A.data); g.b = dv::addr(B.data); g.c = dv::addr(C.data); g.bias = dv::addr(bias.data);
            g.da = g.db = dt; g.dc = Dtype::FP32;
            g.m = M; g.n = N; g.k = K; g.lda = g.ldb = K; g.ldc = N;
            g.act = brotensor::kLinearActSilu;
            const std::string path = dv::gemm_path(d, g);
            dv::gemm(d, g);
            const std::string tag = std::string("gemm ") + dt_name(dt) + " operands, f32 C" + (wide_a ? ", wide A" : "") +
                                    " [" + path + "]";
            // FP32 output: compare unrounded (check_rows rounds want to BF16, so
            // use the FP32 tolerance directly).
            const auto got = download(C), want = yc.to_host_vector();
            std::vector<float> gs(got.size()), ws(want.size());
            for (int r = 0; r < M; ++r) {
                float m = 0;
                for (int n = 0; n < N; ++n) m = std::max(m, std::fabs(want[std::size_t(r) * N + n]));
                for (int n = 0; n < N; ++n) {
                    const std::size_t i = std::size_t(r) * N + n;
                    gs[i] = got[i] / (2e-5f * m + 1e-30f);
                    ws[i] = want[i] / (2e-5f * m + 1e-30f);
                }
            }
            expect_close(gs, ws, 1.0f, 0.0f, tag + " (in units of the row tolerance)");
        }
    }
    // FP32 A (rows 1e-6 .. 3e7) against FP16 / BF16 B, FP32 C, FP32 bias:
    // coopmat only with round_a, A rounded to FP16's 11 bits at its row's
    // scale (relative 2^-12 per product), so 2e-3 of the row's largest output.
    for (Dtype wdt : {Dtype::FP16, Dtype::BF16}) {
        auto av = wide_rows(M, K, seed++);
        for (std::size_t i = 0; i < av.size(); ++i) av[i] *= 1.0f + 0.001f * float(i % 7);   // FP32 bits below BF16's
        const auto bv = random_values(std::size_t(N) * K, seed++, -1, 1, wdt);
        const auto biasv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, Dtype::FP32);
        Tensor bc = cpu_t(biasv, N, 1), yc;
        brotensor::linear_forward_batched_ex(cpu_t(bv, N, K), &bc, cpu_t(av, M, K), 0, brotensor::kLinearEpiStore,
                                             nullptr, yc);
        Tensor A = upload(av, M, K, Dtype::FP32), B = upload(bv, N, K, wdt), bias = upload(biasv, N, 1, Dtype::FP32);
        for (bool round_a : {false, true}) {
            Tensor C = Tensor::empty_on(vk(), M, N, Dtype::FP32);
            dv::GemmArgs g;
            g.a = dv::addr(A.data); g.b = dv::addr(B.data); g.c = dv::addr(C.data); g.bias = dv::addr(bias.data);
            g.da = Dtype::FP32; g.db = wdt; g.dc = Dtype::FP32;
            g.m = M; g.n = N; g.k = K; g.lda = g.ldb = K; g.ldc = N;
            g.round_a = round_a;
            const std::string path = dv::gemm_path(d, g);
            VKT_CHECK(path == (round_a && d.info().coopmat_f16 ? "coopmat" : "simt"));
            dv::gemm(d, g);
            const auto got = download(C), want = yc.to_host_vector();
            std::vector<float> gs(got.size()), ws(want.size());
            const float rel = round_a ? 2e-3f : 2e-5f;
            for (int r = 0; r < M; ++r) {
                float m = 0;
                for (int n = 0; n < N; ++n) m = std::max(m, std::fabs(want[std::size_t(r) * N + n]));
                for (int n = 0; n < N; ++n) {
                    const std::size_t i = std::size_t(r) * N + n;
                    gs[i] = got[i] / (rel * m + 1e-30f);
                    ws[i] = want[i] / (rel * m + 1e-30f);
                }
            }
            expect_close(gs, ws, 1.0f, 0.0f,
                         std::string("gemm f32 A x ") + dt_name(wdt) + " B" + (round_a ? " round_a" : "") + " [" + path +
                             "] (in units of the row tolerance)");
        }
    }
}

}  // namespace

void run_gemm_range_tests() {
    test_unscaled_overflows();
    test_abt();
    test_linear();
    test_nn_and_backward();
    test_nonfinite_row();
    test_fp32_forms();
}

}  // namespace vkt
