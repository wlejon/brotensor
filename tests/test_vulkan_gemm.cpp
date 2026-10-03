// Vulkan matrix-op parity against the CPU backend: matmul, matmul_abt
// (batched, broadcast and padded strides, bias + every activation), the
// linear family with all four epilogues, and the GEMM backwards. Every case
// runs on FP32, FP16 and BF16 storage; the 16-bit ones through the
// cooperative-matrix kernel, the GEMV kernel (at most 8 rows) and, forced
// through the test hook, the SIMT fallback a device without cooperative
// matrix would use. Shapes include odd sizes (partial tiles, K not a
// multiple of 8, so the scalar-load path), unaligned views, and the GQA /
// attention layouts the transformer siblings feed matmul_abt.
//
// Reference: the CPU op on the same (dtype-rounded) inputs in FP32. The
// allowed difference is the FP32 summation-order term (scaled by K) plus,
// for 16-bit outputs, two units in the last place of the dtype.

#include "test_vulkan_common.h"

#include "detail/gemm.h"

#include <brotensor/ops/fused.h>
#include <brotensor/ops/linear.h>
#include <brotensor/vulkan.h>

#include <array>
#include <cmath>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

const Dtype kFloatTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

Tensor cpu_tensor(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

std::vector<float> rounded(std::vector<float> v, Dtype dt) {
    for (float& x : v) x = round_to(dt, x);
    return v;
}

void close_for(Dtype dt, const std::vector<float>& got, const std::vector<float>& want, float atol, float rtol,
               const std::string& t) {
    if (dt == Dtype::FP32) {
        expect_close(got, want, atol, rtol, t);
    } else {
        expect_close(got, rounded(want, dt), atol + 1e-6f, rtol + 2.0f * dtype_eps(dt), t);
    }
}

// Summation-order allowance for a K-term dot product of values in [-1, 1].
float k_atol(int K) { return 2e-6f * std::sqrt(static_cast<float>(K) + 1.0f) + 1e-6f; }

std::string shape_tag(const char* op, Dtype dt, int a, int b, int c, const char* extra = "") {
    return std::string(op) + " " + dt_name(dt) + " " + std::to_string(a) + "x" + std::to_string(b) + "x" +
           std::to_string(c) + extra;
}

// ─── matmul (NN) ───────────────────────────────────────────────────────────

void test_matmul() {
    std::printf("matmul\n");
    const int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {64, 64, 64}, {33, 70, 129}, {257, 96, 300}, {128, 512, 256}};
    std::uint64_t seed = 100;
    for (Dtype dt : kFloatTypes) {
        for (const auto& s : shapes) {
            const int M = s[0], N = s[1], K = s[2];
            const auto av = random_values(std::size_t(M) * K, seed++, -1, 1, dt);
            const auto bv = random_values(std::size_t(K) * N, seed++, -1, 1, dt);
            Tensor cc;
            brotensor::matmul(cpu_tensor(av, M, K), cpu_tensor(bv, K, N), cc);
            Tensor cg;
            brotensor::matmul(upload(av, M, K, dt), upload(bv, K, N, dt), cg);
            VKT_CHECK(cg.rows == M && cg.cols == N && cg.dtype == dt);
            close_for(dt, download(cg), cc.to_host_vector(), k_atol(K), 1e-5f, shape_tag("matmul", dt, M, N, K));
        }
    }
}

// ─── matmul_abt ────────────────────────────────────────────────────────────

struct AbtCase {
    int batch, M, N, K;
    long long sa, sb, sc;   // -1 = tightly packed
    bool bias;
    int act;
    const char* what;
};

// CPU reference: per batch slice, linear_forward_batched_ex in FP32 (the same
// r = act(A B^T + bias) contract).
std::vector<float> cpu_abt(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>* bias,
                           const AbtCase& c, long long sa, long long sb, long long sc, std::size_t c_size) {
    std::vector<float> out(c_size, 0.0f);
    Tensor bc = bias ? cpu_tensor(*bias, c.N, 1) : Tensor();
    for (int z = 0; z < c.batch; ++z) {
        std::vector<float> as(a.begin() + z * sa, a.begin() + z * sa + std::size_t(c.M) * c.K);
        std::vector<float> bs(b.begin() + z * sb, b.begin() + z * sb + std::size_t(c.N) * c.K);
        Tensor y;
        brotensor::linear_forward_batched_ex(cpu_tensor(bs, c.N, c.K), bias ? &bc : nullptr,
                                             cpu_tensor(as, c.M, c.K), c.act, brotensor::kLinearEpiStore, nullptr, y);
        const auto yv = y.to_host_vector();
        std::copy(yv.begin(), yv.end(), out.begin() + z * sc);
    }
    return out;
}

void run_abt(const AbtCase& c, Dtype dt, std::uint64_t& seed, int a_offset = 0) {
    const long long sa = c.sa < 0 ? (long long)c.M * c.K : c.sa;
    const long long sb = c.sb < 0 ? (long long)c.N * c.K : c.sb;
    const long long sc = c.sc < 0 ? (long long)c.M * c.N : c.sc;
    const std::size_t na = std::size_t((c.batch - 1) * sa + (long long)c.M * c.K);
    const std::size_t nb = std::size_t((c.batch - 1) * sb + (long long)c.N * c.K);
    const std::size_t nc = std::size_t((c.batch - 1) * sc + (long long)c.M * c.N);
    const auto av = random_values(na, seed++, -1, 1, dt);
    const auto bv = random_values(nb, seed++, -1, 1, dt);
    const auto biasv = random_values(std::size_t(c.N), seed++, -0.5f, 0.5f, dt);
    const auto want = cpu_abt(av, bv, c.bias ? &biasv : nullptr, c, sa, sb, sc, nc);
    // a_offset > 0: A is a view a few elements into a larger buffer, which
    // breaks 16-byte alignment and forces the scalar loads.
    std::vector<float> apad(std::size_t(a_offset), 0.0f);
    apad.insert(apad.end(), av.begin(), av.end());
    Tensor abuf = upload(apad, int(apad.size()), 1, dt);
    const std::size_t es = dt == Dtype::FP32 ? 4 : 2;
    Tensor A = Tensor::view(vk(), static_cast<char*>(abuf.data) + a_offset * es, int(na), 1, dt);
    Tensor B = upload(bv, int(nb), 1, dt), bias = upload(biasv, c.N, 1, dt);
    Tensor C = Tensor::zeros_on(vk(), int(nc), 1, dt);
    brotensor::matmul_abt(A, B, C, c.batch, c.M, c.N, c.K, sa, sb, sc, c.bias ? &bias : nullptr, c.act);
    // Padded strides leave gaps in C that the op must not write.
    const auto got = download(C);
    const std::string t = shape_tag("matmul_abt", dt, c.M, c.N, c.K,
                                    (std::string(" b") + std::to_string(c.batch) + " " + c.what +
                                     (a_offset ? " unaligned" : "")).c_str());
    close_for(dt, got, want, k_atol(c.K) + 2e-6f, 2e-5f, t);
}

void test_matmul_abt() {
    std::printf("matmul_abt\n");
    using brotensor::kLinearActGeluExact;
    using brotensor::kLinearActGeluTanh;
    using brotensor::kLinearActQuickGelu;
    using brotensor::kLinearActRelu;
    using brotensor::kLinearActSilu;
    const AbtCase cases[] = {
        {1, 1, 1, 1, -1, -1, -1, false, 0, "plain"},
        {1, 7, 13, 9, -1, -1, -1, true, kLinearActRelu, "bias relu"},
        {1, 300, 200, 96, -1, -1, -1, false, kLinearActGeluTanh, "gelu_tanh"},
        {1, 64, 64, 64, -1, -1, -1, true, kLinearActSilu, "bias silu"},
        {1, 129, 257, 1000, -1, -1, -1, true, kLinearActGeluExact, "bias gelu_exact"},
        {1, 512, 384, 256, -1, -1, -1, false, kLinearActQuickGelu, "quick_gelu"},
        // Attention scores Q K^T per head: batch = heads, packed.
        {8, 33, 47, 64, -1, -1, -1, false, 0, "heads"},
        // GQA: 6 query heads against one shared K (stride 0 broadcasts).
        {6, 17, 40, 64, -1, 0, -1, false, 0, "gqa broadcast"},
        // Padded batch strides (gaps between slices, unaligned slice starts).
        {3, 20, 24, 40, 20 * 40 + 5, 24 * 40 + 3, 20 * 24 + 7, true, 0, "padded strides"},
        {2, 520, 260, 136, -1, -1, -1, true, 0, "two big slices"},
    };
    std::uint64_t seed = 200;
    for (Dtype dt : kFloatTypes) {
        for (const AbtCase& c : cases) run_abt(c, dt, seed);
        run_abt({1, 70, 50, 72, -1, -1, -1, true, 0, "view"}, dt, seed, 3);
    }
    // The SIMT kernel on 16-bit operands (a device without cooperative matrix).
    dv::set_gemm_override(2);
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        run_abt({1, 129, 257, 300, -1, -1, -1, true, kLinearActGeluTanh, "simt"}, dt, seed);
        run_abt({6, 17, 40, 64, -1, 0, -1, false, 0, "simt gqa"}, dt, seed);
    }
    dv::set_gemm_override(0);
}

// ─── linear family ─────────────────────────────────────────────────────────

// CPU FP32 reference for the ex contract.
std::vector<float> cpu_linear_ex(const std::vector<float>& w, const std::vector<float>* bias,
                                 const std::vector<float>& x, int B, int N, int K, int act, int epi,
                                 const std::vector<float>* y0) {
    Tensor bc = bias ? cpu_tensor(*bias, N, 1) : Tensor();
    Tensor y = y0 ? cpu_tensor(*y0, B, N) : Tensor();
    brotensor::linear_forward_batched_ex(cpu_tensor(w, N, K), bias ? &bc : nullptr, cpu_tensor(x, B, K), act, epi,
                                         nullptr, y);
    return y.to_host_vector();
}

void test_linear_ex() {
    std::printf("linear_forward_batched_ex / _fp16 / _fp16_act\n");
    const int rows[] = {1, 3, 8, 9, 64, 200};
    const int dims[][2] = {{64, 96}, {200, 130}, {1000, 64}};   // (K, N), N even
    const char* epi_names[] = {"store", "accum", "geglu", "swiglu"};
    std::uint64_t seed = 300;
    for (Dtype dt : kFloatTypes) {
        for (int B : rows) {
            for (const auto& kd : dims) {
                const int K = kd[0], N = kd[1];
                for (int epi = 0; epi < 4; ++epi) {
                    const int act = epi == 0 ? (B % 5) + 1 : 0;   // a different activation per row count
                    const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, dt);
                    const auto xv = random_values(std::size_t(B) * K, seed++, -1, 1, dt);
                    const auto bv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, dt);
                    const auto y0 = random_values(std::size_t(B) * N, seed++, -1, 1, dt);
                    const auto want = cpu_linear_ex(wv, &bv, xv, B, N, K, act, epi, epi == 1 ? &y0 : nullptr);
                    Tensor W = upload(wv, N, K, dt), X = upload(xv, B, K, dt), bias = upload(bv, N, 1, dt);
                    Tensor Y = epi == 1 ? upload(y0, B, N, dt) : Tensor();
                    brotensor::linear_forward_batched_ex(W, &bias, X, act, epi, nullptr, Y);
                    const int out = epi >= 2 ? N / 2 : N;
                    VKT_CHECK(Y.rows == B && Y.cols == out && Y.dtype == dt);
                    char buf[96];
                    std::snprintf(buf, sizeof buf, "linear_ex %s B%d K%d N%d act%d %s", epi_names[epi], B, K, N, act,
                                  dt_name(dt));
                    // A GLU multiplies the gate's summation-order error by the
                    // other half (|r| ~ sqrt(K) / 3 for these inputs).
                    const float glu = epi >= 2 ? 0.5f * std::sqrt(float(K)) : 1.0f;
                    close_for(dt, download(Y), want, k_atol(K) * glu + 2e-6f, 3e-5f, buf);
                }
            }
        }
    }
    // _fp16 / _fp16_act are the store epilogue on 16-bit storage.
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (int B : {1, 5, 77}) {
            const int K = 136, N = 72;
            const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, dt);
            const auto xv = random_values(std::size_t(B) * K, seed++, -1, 1, dt);
            const auto bv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, dt);
            Tensor W = upload(wv, N, K, dt), X = upload(xv, B, K, dt), bias = upload(bv, N, 1, dt);
            Tensor Y, Z;
            brotensor::linear_forward_batched_fp16(W, &bias, X, Y);
            close_for(dt, download(Y), cpu_linear_ex(wv, &bv, xv, B, N, K, 0, 0, nullptr), k_atol(K), 2e-5f,
                      shape_tag("linear_fp16", dt, B, N, K));
            brotensor::linear_forward_batched_fp16_act(W, nullptr, X, brotensor::kLinearActGeluTanh, Z);
            close_for(dt, download(Z), cpu_linear_ex(wv, nullptr, xv, B, N, K, 2, 0, nullptr), k_atol(K), 3e-5f,
                      shape_tag("linear_fp16_act gelu", dt, B, N, K));
        }
    }
    // The SIMT kernel's separate GLU pass and the cooperative-matrix GLU at
    // the edge of a tile, with the GEMV kernel disabled.
    for (int mode : {1, 2}) {
        dv::set_gemm_override(mode);
        for (Dtype dt : kFloatTypes) {
            for (int epi : {2, 3}) {
                const int B = 5, K = 72, N = 136;
                const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, dt);
                const auto xv = random_values(std::size_t(B) * K, seed++, -1, 1, dt);
                const auto bv = random_values(std::size_t(N), seed++, -0.5f, 0.5f, dt);
                Tensor W = upload(wv, N, K, dt), X = upload(xv, B, K, dt), bias = upload(bv, N, 1, dt), Y;
                brotensor::linear_forward_batched_ex(W, &bias, X, 0, epi, nullptr, Y);
                close_for(dt, download(Y), cpu_linear_ex(wv, &bv, xv, B, N, K, 0, epi, nullptr),
                          k_atol(K) * 0.5f * std::sqrt(float(K)), 3e-5f,
                          std::string("linear_ex ") + epi_names[epi] + (mode == 1 ? " no-gemv " : " simt ") + dt_name(dt));
            }
        }
    }
    dv::set_gemm_override(0);
    // fused_gemv_swiglu over the two halves of one stacked [gate; up] weight
    // (the fused GEMV), and over separate weights (the composite).
    for (Dtype dt : kFloatTypes) {
        const int K = 256, N = 72;
        const auto wv = random_values(std::size_t(2 * N) * K, seed++, -1, 1, dt);
        const auto xv = random_values(std::size_t(K), seed++, -1, 1, dt);
        const std::vector<float> wg(wv.begin(), wv.begin() + N * K), wu(wv.begin() + N * K, wv.end());
        const auto want = cpu_linear_ex(wv, nullptr, xv, 1, 2 * N, K, 0, brotensor::kLinearEpiSwiglu, nullptr);
        Tensor W = upload(wv, 2 * N, K, dt);
        const std::size_t es = dt == Dtype::FP32 ? 4 : 2;
        Tensor Wg = Tensor::view(vk(), W.data, N, K, dt);
        Tensor Wu = Tensor::view(vk(), static_cast<char*>(W.data) + std::size_t(N) * K * es, N, K, dt);
        Tensor x = upload(xv, 1, K, dt), y, y2;
        brotensor::fused_gemv_swiglu(x, Wg, Wu, y);
        close_for(dt, download(y), want, k_atol(K) * 8.0f, 3e-5f, std::string("fused_gemv_swiglu stacked ") + dt_name(dt));
        brotensor::fused_gemv_swiglu(x, upload(wg, N, K, dt), upload(wu, N, K, dt), y2);
        // The composite rounds both projections to the dtype before gating.
        close_for(dt, download(y2), want, k_atol(K) * 8.0f, 3e-5f + 4.0f * dtype_eps(dt),
                  std::string("fused_gemv_swiglu separate ") + dt_name(dt));
    }
}

void test_linear_batched() {
    std::printf("linear_forward_batched / linear_forward\n");
    std::uint64_t seed = 400;
    // FP32 activations against FP32 / FP16 / BF16 weights: FP32 accumulation
    // over the unrounded activations, so the tolerance is FP32's.
    for (Dtype wdt : kFloatTypes) {
        for (int B : {1, 2, 8, 9, 130}) {
            for (const auto& kd : {std::array<int, 2>{512, 96}, std::array<int, 2>{1000, 37}}) {
                const int K = kd[0], N = kd[1];
                const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, wdt);
                const auto xv = random_values(std::size_t(B) * K, seed++, -1, 1, Dtype::FP32);
                const auto bv = random_values(std::size_t(N), seed++, -1, 1, Dtype::FP32);
                Tensor yc;
                brotensor::linear_forward_batched(cpu_tensor(wv, N, K), cpu_tensor(bv, N, 1), cpu_tensor(xv, B, K), yc);
                Tensor yg;
                brotensor::linear_forward_batched(upload(wv, N, K, wdt), upload(bv, N, 1, Dtype::FP32),
                                                  upload(xv, B, K, Dtype::FP32), yg);
                VKT_CHECK(yg.dtype == Dtype::FP32 && yg.rows == B && yg.cols == N);
                expect_close(download(yg), yc.to_host_vector(), k_atol(K), 1e-5f,
                             std::string("linear_batched X f32 W ") + dt_name(wdt) + " B" + std::to_string(B) +
                                 " K" + std::to_string(K) + " N" + std::to_string(N));
            }
        }
    }
    // Same-dtype 16-bit, and linear_forward on a single vector.
    for (Dtype dt : kFloatTypes) {
        const int K = 257, N = 45;
        const auto wv = random_values(std::size_t(N) * K, seed++, -1, 1, dt);
        const auto xv = random_values(std::size_t(K), seed++, -1, 1, dt);
        const auto bv = random_values(std::size_t(N), seed++, -1, 1, dt);
        Tensor yc;
        brotensor::linear_forward(cpu_tensor(wv, N, K), cpu_tensor(bv, N, 1), cpu_tensor(xv, K, 1), yc);
        Tensor yg;
        brotensor::linear_forward(upload(wv, N, K, dt), upload(bv, N, 1, dt), upload(xv, K, 1, dt), yg);
        VKT_CHECK(yg.rows == N && yg.cols == 1 && yg.dtype == dt);
        close_for(dt, download(yg), yc.to_host_vector(), k_atol(K), 1e-5f, shape_tag("linear_forward", dt, N, K, 1));
    }
}

// ─── backwards ─────────────────────────────────────────────────────────────

void test_backwards() {
    std::printf("matmul_backward / linear_backward(_batched)\n");
    std::uint64_t seed = 500;
    for (Dtype dt : kFloatTypes) {
        // Gradients accumulate into non-zero buffers; in 16-bit the
        // accumulator's own rounding adds an ulp.
        for (const auto& s : {std::array<int, 3>{1, 1, 1}, std::array<int, 3>{17, 33, 65}, std::array<int, 3>{128, 96, 200}}) {
            const int M = s[0], N = s[1], K = s[2];
            const auto av = random_values(std::size_t(M) * K, seed++, -1, 1, dt);
            const auto bv = random_values(std::size_t(K) * N, seed++, -1, 1, dt);
            const auto gv = random_values(std::size_t(M) * N, seed++, -1, 1, dt);
            const auto da0 = random_values(std::size_t(M) * K, seed++, -1, 1, dt);
            const auto db0 = random_values(std::size_t(K) * N, seed++, -1, 1, dt);
            Tensor dac = cpu_tensor(da0, M, K), dbc = cpu_tensor(db0, K, N);
            brotensor::matmul_backward(cpu_tensor(av, M, K), cpu_tensor(bv, K, N), cpu_tensor(gv, M, N), dac, dbc);
            Tensor dag = upload(da0, M, K, dt), dbg = upload(db0, K, N, dt);
            brotensor::matmul_backward(upload(av, M, K, dt), upload(bv, K, N, dt), upload(gv, M, N, dt), dag, dbg);
            close_for(dt, download(dag), dac.to_host_vector(), k_atol(N) + 2e-6f, 2e-5f,
                      shape_tag("matmul_backward dA", dt, M, N, K));
            close_for(dt, download(dbg), dbc.to_host_vector(), k_atol(M) + 2e-6f, 2e-5f,
                      shape_tag("matmul_backward dB", dt, M, N, K));
        }
        for (int B : {1, 6, 70}) {
            const int in = 90, out = 41;
            const auto wv = random_values(std::size_t(out) * in, seed++, -1, 1, dt);
            const auto xv = random_values(std::size_t(B) * in, seed++, -1, 1, dt);
            const auto gv = random_values(std::size_t(B) * out, seed++, -1, 1, dt);
            const auto dw0 = random_values(std::size_t(out) * in, seed++, -1, 1, dt);
            const auto db0 = random_values(std::size_t(out), seed++, -1, 1, dt);
            Tensor dxc, dwc = cpu_tensor(dw0, out, in), dbc = cpu_tensor(db0, out, 1);
            Tensor dxg, dwg = upload(dw0, out, in, dt), dbg = upload(db0, out, 1, dt);
            if (B == 1) {
                brotensor::linear_backward(cpu_tensor(wv, out, in), cpu_tensor(xv, in, 1), cpu_tensor(gv, out, 1),
                                           dxc, dwc, dbc);
                brotensor::linear_backward(upload(wv, out, in, dt), upload(xv, in, 1, dt), upload(gv, out, 1, dt),
                                           dxg, dwg, dbg);
            } else {
                brotensor::linear_backward_batched(cpu_tensor(wv, out, in), cpu_tensor(xv, B, in), cpu_tensor(gv, B, out),
                                                   dxc, dwc, dbc);
                brotensor::linear_backward_batched(upload(wv, out, in, dt), upload(xv, B, in, dt),
                                                   upload(gv, B, out, dt), dxg, dwg, dbg);
            }
            const std::string t = std::string(B == 1 ? "linear_backward " : "linear_backward_batched ") + dt_name(dt) +
                                  " B" + std::to_string(B);
            close_for(dt, download(dxg), dxc.to_host_vector(), k_atol(out), 2e-5f, t + " dX");
            close_for(dt, download(dwg), dwc.to_host_vector(), k_atol(B) + 2e-6f, 2e-5f, t + " dW");
            close_for(dt, download(dbg), dbc.to_host_vector(), k_atol(B) + 2e-6f, 2e-5f, t + " dB");
        }
    }
}

// ─── kernel choice ─────────────────────────────────────────────────────────

void test_paths() {
    std::printf("gemm kernel choice\n");
    dv::DeviceCtx& d = dv::device(0);
    dv::GemmArgs g;
    g.m = 4; g.n = 64; g.k = 64; g.lda = g.ldb = g.ldc = 64;
    g.da = g.db = g.dc = Dtype::FP16;
    VKT_CHECK(std::string(dv::gemm_path(d, g)) == "gemv");
    g.m = 9;
    const bool cm = brotensor::vulkan::device_info(vk()).cooperative_matrix && d.info().coopmat_f16;
    VKT_CHECK(std::string(dv::gemm_path(d, g)) == (cm ? "coopmat" : "simt"));
    std::printf("  coopmat GEMM: %s\n", cm ? "yes" : "no (SIMT fallback)");
    g.da = g.db = g.dc = Dtype::FP32;
    VKT_CHECK(std::string(dv::gemm_path(d, g)) == "simt");
    g.db = Dtype::BF16;
    g.m = 3;
    VKT_CHECK(std::string(dv::gemm_path(d, g)) == "gemv");
}

}  // namespace

void run_gemm_tests() {
    test_paths();
    test_matmul();
    test_matmul_abt();
    test_linear_ex();
    test_linear_batched();
    test_backwards();
}

}  // namespace vkt
