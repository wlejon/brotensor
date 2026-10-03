// Vulkan op parity: every op the Vulkan backend registers, against the CPU
// backend (the FP32 reference) on several shapes and on FP32 / FP16 / BF16
// storage. For 16-bit dtypes both sides see the same rounded inputs, the CPU
// computes in FP32 and its result is rounded to the dtype, so the allowed
// difference is the op's FP32 tolerance plus two units in the last place of
// the dtype. Data movement (cast, copies, transposes) must be bit-exact.

#include "test_vulkan_common.h"

#include <algorithm>
#include <functional>

namespace vkt {

namespace {

const Dtype kFloatTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};
const int kShapes[][2] = {{1, 1}, {3, 5}, {64, 257}, {513, 1031}};

Tensor cpu_tensor(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

std::vector<float> rounded(std::vector<float> v, Dtype dt) {
    for (float& x : v) x = round_to(dt, x);
    return v;
}

std::string tag(const char* op, Dtype dt, int r, int c) {
    return std::string(op) + " " + dt_name(dt) + " " + std::to_string(r) + "x" + std::to_string(c);
}

void close_for(Dtype dt, const std::vector<float>& got, const std::vector<float>& want, float atol,
               float rtol, const std::string& t) {
    if (dt == Dtype::FP32) {
        expect_close(got, want, atol, rtol, t);
    } else {
        expect_close(got, rounded(want, dt), atol + 1e-6f, rtol + 2.0f * dtype_eps(dt), t);
    }
}

// ─── unary forward (incl. the in-place scalar ops) ─────────────────────────

struct UnaryCase {
    const char* name;
    std::function<void(const Tensor&, Tensor&)> fn;
    float lo, hi, atol, rtol;
};

void test_unary() {
    std::printf("unary forward\n");
    const std::vector<UnaryCase> cases = {
        {"relu", [](const Tensor& x, Tensor& y) { brotensor::relu_forward(x, y); }, -4, 4, 0, 0},
        {"relu_batched", [](const Tensor& x, Tensor& y) { brotensor::relu_forward_batched(x, y); }, -4, 4, 0, 0},
        {"tanh", [](const Tensor& x, Tensor& y) { brotensor::tanh_forward(x, y); }, -6, 6, 2e-7f, 2e-6f},
        {"sigmoid", [](const Tensor& x, Tensor& y) { brotensor::sigmoid_forward(x, y); }, -8, 8, 2e-7f, 2e-6f},
        {"silu", [](const Tensor& x, Tensor& y) { brotensor::silu_forward(x, y); }, -8, 8, 2e-7f, 2e-6f},
        {"gelu", [](const Tensor& x, Tensor& y) { brotensor::gelu_forward(x, y); }, -6, 6, 2e-7f, 2e-6f},
        {"gelu_exact", [](const Tensor& x, Tensor& y) { brotensor::gelu_exact_forward(x, y); }, -6, 6, 5e-7f, 2e-6f},
        {"quick_gelu", [](const Tensor& x, Tensor& y) { brotensor::quick_gelu_forward(x, y); }, -6, 6, 2e-7f, 2e-6f},
        {"exp", [](const Tensor& x, Tensor& y) { brotensor::exp_forward(x, y); }, -10, 10, 0, 4e-6f},
        {"log", [](const Tensor& x, Tensor& y) { brotensor::log_forward(x, y); }, 0.01f, 100, 5e-7f, 1e-6f},
        {"sin", [](const Tensor& x, Tensor& y) { brotensor::sin_forward(x, y); }, -4, 4, 2e-6f, 0},
        {"cos", [](const Tensor& x, Tensor& y) { brotensor::cos_forward(x, y); }, -4, 4, 2e-6f, 0},
        {"rsqrt", [](const Tensor& x, Tensor& y) { brotensor::rsqrt_forward(x, y); }, 0.01f, 100, 0, 1e-6f},
        {"elu", [](const Tensor& x, Tensor& y) { brotensor::elu_forward(x, 0.7f, y); }, -4, 4, 2e-7f, 2e-6f},
        {"leaky_relu", [](const Tensor& x, Tensor& y) { brotensor::leaky_relu_forward(x, 0.1f, y); }, -4, 4, 0, 0},
        {"add_scalar", [](const Tensor& x, Tensor& y) { y = x.clone(); brotensor::add_scalar_inplace(y, 0.37f); }, -4, 4, 0, 0},
        {"scale", [](const Tensor& x, Tensor& y) { y = x.clone(); brotensor::scale_inplace(y, -1.7f); }, -4, 4, 0, 0},
        {"clamp", [](const Tensor& x, Tensor& y) { y = x.clone(); brotensor::clamp(y, -0.5f, 0.8f); }, -4, 4, 0, 0},
    };
    std::uint64_t seed = 1;
    for (const auto& c : cases) {
        for (Dtype dt : kFloatTypes) {
            for (const auto& s : kShapes) {
                const int r = s[0], k = s[1];
                const auto xv = random_values(std::size_t(r) * k, seed++, c.lo, c.hi, dt);
                Tensor xc = cpu_tensor(xv, r, k), yc;
                c.fn(xc, yc);
                Tensor xg = upload(xv, r, k, dt), yg;
                c.fn(xg, yg);
                VKT_CHECK(yg.device == vk() && yg.dtype == dt && yg.rows == r && yg.cols == k);
                close_for(dt, download(yg), yc.to_host_vector(), c.atol, c.rtol, tag(c.name, dt, r, k));
            }
        }
    }
    // round: ties to even, so feed exact halves.
    for (Dtype dt : kFloatTypes) {
        std::vector<float> xv;
        for (int i = -40; i <= 40; ++i) xv.push_back(i * 0.5f + (i % 3 == 0 ? 0.25f : 0.0f));
        const int n = static_cast<int>(xv.size());
        Tensor xc = cpu_tensor(xv, n, 1), yc;
        brotensor::round_forward(xc, yc);
        Tensor yg;
        brotensor::round_forward(upload(xv, n, 1, dt), yg);
        close_for(dt, download(yg), yc.to_host_vector(), 0, 0, tag("round", dt, n, 1));
    }
    // In place (y aliases x) and a grid larger than one dispatch's worth of
    // invocations (65535 workgroups x 256), so the grid-stride loop wraps.
    {
        const int r = 2048, k = 9000;
        const auto xv = random_values(std::size_t(r) * k, 99, -3, 3, Dtype::FP32);
        Tensor xg = upload(xv, r, k, Dtype::FP32);
        brotensor::relu_forward(xg, xg);
        std::vector<float> want(xv);
        for (float& v : want) v = v > 0 ? v : 0;
        expect_close(download(xg), want, 0, 0, "relu in place 2048x9000 (grid stride)");
    }
}

// ─── binary in place ───────────────────────────────────────────────────────

void test_binary() {
    std::printf("binary in place\n");
    // axpby's atol: the CPU build contracts a*y + b*x into an FMA (one
    // rounding), the GPU kernels evaluate it unfused like CUDA (two), so
    // where the terms cancel they differ by an ulp of the terms (|terms| <= 4).
    struct Case {
        const char* name;
        std::function<void(Tensor&, const Tensor&)> fn;
        float rtol, atol = 0.0f;
    };
    const std::vector<Case> cases = {
        {"add", [](Tensor& y, const Tensor& x) { brotensor::add_inplace(y, x); }, 0},
        {"add_batched", [](Tensor& y, const Tensor& x) { brotensor::add_inplace_batched(y, x); }, 0},
        {"mul", [](Tensor& y, const Tensor& x) { brotensor::mul_inplace(y, x); }, 0},
        {"div", [](Tensor& y, const Tensor& x) { brotensor::div_inplace(y, x); }, 3e-7f},
        {"axpby", [](Tensor& y, const Tensor& x) { brotensor::axpby_inplace(y, x, 0.3f, -1.2f); }, 3e-7f, 5e-7f},
    };
    std::uint64_t seed = 1000;
    for (const auto& c : cases) {
        for (Dtype dt : kFloatTypes) {
            for (const auto& s : kShapes) {
                const int r = s[0], k = s[1];
                const auto yv = random_values(std::size_t(r) * k, seed++, -3, 3, dt);
                const auto xv = random_values(std::size_t(r) * k, seed++, 0.5f, 2.0f, dt);
                Tensor yc = cpu_tensor(yv, r, k);
                c.fn(yc, cpu_tensor(xv, r, k));
                Tensor yg = upload(yv, r, k, dt);
                c.fn(yg, upload(xv, r, k, dt));
                close_for(dt, download(yg), yc.to_host_vector(), c.atol, c.rtol, tag(c.name, dt, r, k));
            }
        }
    }
    // FP32 y += FP16 x, the one mixed pair add_inplace takes.
    {
        const auto yv = random_values(1000, 7, -3, 3, Dtype::FP32);
        const auto xv = random_values(1000, 8, -3, 3, Dtype::FP16);
        Tensor yg = upload(yv, 10, 100, Dtype::FP32);
        brotensor::add_inplace(yg, upload(xv, 10, 100, Dtype::FP16));
        std::vector<float> want(yv);
        for (std::size_t i = 0; i < want.size(); ++i) want[i] += xv[i];
        expect_close(download(yg), want, 0, 0, "add f32 += f16");
    }
    // Contract errors.
    Tensor a = upload(random_values(6, 1, 0, 1, Dtype::FP32), 2, 3, Dtype::FP32);
    Tensor b = upload(random_values(4, 2, 0, 1, Dtype::FP32), 2, 2, Dtype::FP32);
    Tensor h = upload(random_values(6, 3, 0, 1, Dtype::FP16), 2, 3, Dtype::FP16);
    VKT_CHECK(throws([&] { brotensor::add_inplace(a, b); }));
    VKT_CHECK(throws([&] { brotensor::mul_inplace(a, h); }));
    VKT_CHECK(throws([&] { brotensor::add_inplace(a, cpu_tensor({1, 2, 3, 4, 5, 6}, 2, 3)); }));
}

// ─── activation backwards ──────────────────────────────────────────────────

void test_backward() {
    std::printf("activation backward\n");
    // uses_y: the op takes the forward output, so p is computed by the CPU
    // forward (and rounded to the dtype like any other input).
    struct Case {
        const char* name;
        std::function<void(const Tensor&, Tensor&)> fwd;
        std::function<void(const Tensor&, const Tensor&, Tensor&)> bwd;
        bool uses_y;
        float lo, hi, atol, rtol;
    };
    using namespace brotensor;
    const std::vector<Case> cases = {
        {"relu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { relu_backward(p, g, d); }, false, -4, 4, 0, 0},
        {"relu_batched", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { relu_backward_batched(p, g, d); }, false, -4, 4, 0, 0},
        {"tanh", [](const Tensor& x, Tensor& y) { tanh_forward(x, y); }, [](const Tensor& p, const Tensor& g, Tensor& d) { tanh_backward(p, g, d); }, true, -3, 3, 1e-7f, 1e-6f},
        {"tanh_batched", [](const Tensor& x, Tensor& y) { tanh_forward(x, y); }, [](const Tensor& p, const Tensor& g, Tensor& d) { tanh_backward_batched(p, g, d); }, true, -3, 3, 1e-7f, 1e-6f},
        {"sigmoid", [](const Tensor& x, Tensor& y) { sigmoid_forward(x, y); }, [](const Tensor& p, const Tensor& g, Tensor& d) { sigmoid_backward(p, g, d); }, true, -6, 6, 1e-7f, 1e-6f},
        {"silu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { silu_backward(p, g, d); }, false, -6, 6, 2e-7f, 2e-6f},
        // atol: the CPU reference evaluates 0.5*(1 + tanh u) + 0.5*x*(1 - tanh^2 u)*u',
        // which cancels for x < -3 (1 + tanh u ~ 1e-6 against tanh's 6e-8
        // rounding), an absolute error of up to ~1e-6 that the GPU's sigmoid
        // form does not have.
        {"gelu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { gelu_backward(p, g, d); }, false, -5, 5, 1.5e-6f, 2e-6f},
        {"gelu_exact", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { gelu_exact_backward(p, g, d); }, false, -5, 5, 5e-7f, 2e-6f},
        {"quick_gelu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { quick_gelu_backward(p, g, d); }, false, -5, 5, 2e-7f, 2e-6f},
        {"exp", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { exp_backward(p, g, d); }, false, -8, 8, 0, 4e-6f},
        {"log", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { log_backward(p, g, d); }, false, 0.05f, 20, 0, 3e-7f},
        {"sin", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { sin_backward(p, g, d); }, false, -4, 4, 2e-6f, 1e-7f},
        {"cos", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { cos_backward(p, g, d); }, false, -4, 4, 2e-6f, 1e-7f},
        {"rsqrt", [](const Tensor& x, Tensor& y) { rsqrt_forward(x, y); }, [](const Tensor& p, const Tensor& g, Tensor& d) { rsqrt_backward(p, g, d); }, true, 0.1f, 10, 0, 1e-6f},
        {"elu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { elu_backward(p, g, 0.7f, d); }, false, -4, 4, 1e-7f, 2e-6f},
        {"leaky_relu", nullptr, [](const Tensor& p, const Tensor& g, Tensor& d) { leaky_relu_backward(p, g, 0.1f, d); }, false, -4, 4, 0, 0},
    };
    std::uint64_t seed = 5000;
    for (const auto& c : cases) {
        for (Dtype dt : kFloatTypes) {
            for (const auto& s : kShapes) {
                const int r = s[0], k = s[1];
                const std::size_t n = std::size_t(r) * k;
                std::vector<float> pv = random_values(n, seed++, c.lo, c.hi, dt);
                if (c.uses_y) {
                    Tensor y;
                    c.fwd(cpu_tensor(pv, r, k), y);
                    pv = rounded(y.to_host_vector(), dt);
                }
                const auto gv = random_values(n, seed++, -2, 2, dt);
                Tensor dc;
                c.bwd(cpu_tensor(pv, r, k), cpu_tensor(gv, r, k), dc);
                Tensor dg;
                c.bwd(upload(pv, r, k, dt), upload(gv, r, k, dt), dg);
                close_for(dt, download(dg), dc.to_host_vector(), c.atol, c.rtol, tag((std::string(c.name) + "_bwd").c_str(), dt, r, k));
            }
        }
    }
    // dX aliasing dY, and round's straight-through estimator.
    {
        const auto xv = random_values(4096, 1, -3, 3, Dtype::FP32);
        const auto gv = random_values(4096, 2, -3, 3, Dtype::FP32);
        Tensor g = upload(gv, 64, 64, Dtype::FP32);
        brotensor::relu_backward(upload(xv, 64, 64, Dtype::FP32), g, g);
        std::vector<float> want(gv);
        for (std::size_t i = 0; i < want.size(); ++i) want[i] = xv[i] > 0 ? gv[i] : 0.0f * gv[i];
        expect_close(download(g), want, 0, 0, "relu_bwd dX aliases dY");
        Tensor d;
        brotensor::round_backward(upload(gv, 64, 64, Dtype::BF16), d);
        expect_close(download(d), rounded(gv, Dtype::BF16), 0, 0, "round_bwd bf16 (pass-through)");
    }
}

// ─── bias ──────────────────────────────────────────────────────────────────

void test_bias() {
    std::printf("bias\n");
    for (Dtype dt : kFloatTypes) {
        const int R = 37, D = 129;
        const auto yv = random_values(std::size_t(R) * D, 11, -2, 2, dt);
        const auto bv = random_values(D, 12, -2, 2, dt);
        Tensor yc = cpu_tensor(yv, R, D);
        Tensor yg = upload(yv, R, D, dt);
        if (dt == Dtype::FP32) brotensor::add_row_bias_inplace(yc, cpu_tensor(bv, D, 1));
        brotensor::add_row_bias_inplace(yg, upload(bv, D, 1, dt));
        std::vector<float> want(yv);
        for (int r = 0; r < R; ++r)
            for (int d = 0; d < D; ++d) want[std::size_t(r) * D + d] += bv[d];
        close_for(dt, download(yg), want, 0, 0, tag("add_row_bias", dt, R, D));
        if (dt == Dtype::FP32) expect_close(download(yg), yc.to_host_vector(), 0, 0, "add_row_bias vs cpu");

        const int C = 24, L = 77;
        const auto cv = random_values(std::size_t(C) * L, 13, -2, 2, dt);
        const auto cb = random_values(C, 14, -2, 2, dt);
        Tensor cg = upload(cv, C * L, 1, dt);
        brotensor::add_channel_bias_inplace(cg, upload(cb, C, 1, dt), C, L);
        Tensor cc = cpu_tensor(cv, C * L, 1);
        brotensor::add_channel_bias_inplace(cc, cpu_tensor(cb, C, 1), C, L);
        close_for(dt, download(cg), cc.to_host_vector(), 0, 0, tag("add_channel_bias", dt, C, L));
    }
}

// ─── cast / copies / transposes ────────────────────────────────────────────

void test_cast() {
    std::printf("cast\n");
    // Ordinary values plus the cases where rounding modes differ: FP16 ties,
    // 65520 (rounds to inf under RNE, to 65504 under truncation), subnormal
    // FP16 results, infinities and NaN.
    std::vector<float> v = random_values(10000, 21, -1000, 1000, Dtype::FP32);
    const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, 65504.0f, 65519.0f, 65520.0f, -65520.0f,
                              1e-5f, -3e-6f, 6.1e-5f, 5.96e-8f, 1.0009765625f, 1.00048828125f,
                              2.0009765625f, 3.4e38f, -3.4e38f, INFINITY, -INFINITY, NAN,
                              1.00390625f, 1.01171875f, 1e-30f};
    v.insert(v.end(), std::begin(specials), std::end(specials));
    const int n = static_cast<int>(v.size());
    std::vector<std::uint16_t> f16(v.size()), bf16(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        f16[i] = brotensor::fp32_to_fp16_bits(v[i]);
        bf16[i] = brotensor::fp32_to_bf16_bits(v[i]);
    }
    Tensor x32 = upload(v, n, 1, Dtype::FP32);
    Tensor h, b, back32, h2b, b2h;
    brotensor::cast(x32, h, Dtype::FP16);
    brotensor::cast(x32, b, Dtype::BF16);
    const auto hb = h.to_host_vector_fp16();
    const auto bb = b.to_host_vector_bf16();
    expect_equal_bits(hb.data(), f16.data(), f16.size() * 2, "cast f32->f16 bit-exact (RNE, specials)");
    expect_equal_bits(bb.data(), bf16.data(), bf16.size() * 2, "cast f32->bf16 bit-exact (RNE, specials)");

    brotensor::cast(h, back32, Dtype::FP32);
    std::vector<float> want(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) want[i] = brotensor::fp16_bits_to_fp32(f16[i]);
    auto got = back32.to_host_vector();
    expect_equal_bits(got.data(), want.data(), want.size() * 4, "cast f16->f32 bit-exact");
    brotensor::cast(b, back32, Dtype::FP32);
    for (std::size_t i = 0; i < v.size(); ++i) want[i] = brotensor::bf16_bits_to_fp32(bf16[i]);
    got = back32.to_host_vector();
    expect_equal_bits(got.data(), want.data(), want.size() * 4, "cast bf16->f32 bit-exact");

    brotensor::cast(h, h2b, Dtype::BF16);
    brotensor::cast(b, b2h, Dtype::FP16);
    std::vector<std::uint16_t> want16(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) want16[i] = brotensor::fp32_to_bf16_bits(brotensor::fp16_bits_to_fp32(f16[i]));
    auto g16 = h2b.to_host_vector_bf16();
    expect_equal_bits(g16.data(), want16.data(), want16.size() * 2, "cast f16->bf16 bit-exact");
    for (std::size_t i = 0; i < v.size(); ++i) want16[i] = brotensor::fp32_to_fp16_bits(brotensor::bf16_bits_to_fp32(bf16[i]));
    g16 = b2h.to_host_vector_fp16();
    expect_equal_bits(g16.data(), want16.data(), want16.size() * 2, "cast bf16->f16 bit-exact");

    Tensor same;
    brotensor::cast(x32, same, Dtype::FP32);
    got = same.to_host_vector();
    expect_equal_bits(got.data(), v.data(), v.size() * 4, "cast f32->f32 copies");
    VKT_CHECK(throws([&] { Tensor o; brotensor::cast(x32, o, Dtype::INT32); }));
}

void test_copies() {
    std::printf("copies / transposes\n");
    for (Dtype dt : kFloatTypes) {
        const auto v = random_values(5000, 31, -5, 5, dt);
        Tensor src = upload(v, 50, 100, dt);
        Tensor dst = Tensor::zeros_on(vk(), 50, 100, dt);
        brotensor::copy_d2d(src, 17, dst, 333, 1234);
        std::vector<float> want(5000, 0.0f);
        std::copy(v.begin() + 17, v.begin() + 17 + 1234, want.begin() + 333);
        expect_close(download(dst), want, 0, 0, std::string("copy_d2d ") + dt_name(dt));

        // 2D: a 13 x 37 window from pitch 100 into pitch 41.
        Tensor d2 = Tensor::zeros_on(vk(), 20, 41, dt);
        brotensor::copy_d2d_strided(src, 205, 100, d2, 3, 41, 37, 13);
        std::vector<float> w2(20 * 41, 0.0f);
        for (int r = 0; r < 13; ++r)
            for (int c = 0; c < 37; ++c) w2[3 + r * 41 + c] = v[205 + r * 100 + c];
        expect_close(download(d2), w2, 0, 0, std::string("copy_d2d_strided ") + dt_name(dt));

        // NCHW <-> sequence round trip against the CPU (FP32) / host loops.
        const int N = 2, C = 37, H = 9, W = 11;
        const auto xv = random_values(std::size_t(N) * C * H * W, 32, -5, 5, dt);
        Tensor seq, back;
        brotensor::nchw_to_sequence(upload(xv, N, C * H * W, dt), N, C, H, W, seq);
        Tensor seqc;
        brotensor::nchw_to_sequence(cpu_tensor(xv, N, C * H * W), N, C, H, W, seqc);
        VKT_CHECK(seq.rows == N * H * W && seq.cols == C && seq.dtype == dt);
        expect_close(download(seq), seqc.to_host_vector(), 0, 0, std::string("nchw_to_sequence ") + dt_name(dt));
        brotensor::sequence_to_nchw(seq, N, C, H, W, back);
        expect_close(download(back), xv, 0, 0, std::string("sequence_to_nchw ") + dt_name(dt));
    }
    // INT8 strided copy (the 1-byte kernel).
    {
        std::vector<std::int8_t> b(64 * 64);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::int8_t>(i * 7);
        Tensor s = Tensor::from_host_int8_on(vk(), b.data(), 64, 64);
        Tensor d = Tensor::zeros_on(vk(), 64, 64, Dtype::INT8);
        brotensor::copy_d2d_strided(s, 1, 64, d, 0, 64, 63, 64);
        std::vector<std::int8_t> got(b.size()), want(b.size(), 0);
        d.copy_to_host_raw(got.data(), got.size());
        for (int r = 0; r < 64; ++r)
            for (int c = 0; c < 63; ++c) want[r * 64 + c] = b[r * 64 + c + 1];
        expect_equal_bits(got.data(), want.data(), want.size(), "copy_d2d_strided int8");
    }
    // concat_rows / split_rows.
    {
        const auto a = random_values(10, 41, -1, 1, Dtype::FP32);
        const auto b = random_values(25, 42, -1, 1, Dtype::FP32);
        Tensor ta = upload(a, 10, 1, Dtype::FP32), tb = upload(b, 5, 5, Dtype::FP32), out;
        brotensor::concat_rows({&ta, &tb}, out);
        std::vector<float> want(a);
        want.insert(want.end(), b.begin(), b.end());
        VKT_CHECK(out.rows == 35 && out.cols == 1);
        expect_close(download(out), want, 0, 0, "concat_rows");
        Tensor pa = Tensor::empty_on(vk(), 10, 1), pb = Tensor::empty_on(vk(), 25, 1);
        brotensor::split_rows(out, {&pa, &pb});
        expect_close(download(pa), a, 0, 0, "split_rows part 0");
        expect_close(download(pb), b, 0, 0, "split_rows part 1");
    }
}

// ─── reductions ────────────────────────────────────────────────────────────

void test_reductions() {
    std::printf("reductions\n");
    std::uint64_t seed = 9000;
    for (Dtype dt : kFloatTypes) {
        for (const auto& s : kShapes) {
            const int r = s[0], k = s[1];
            const auto xv = random_values(std::size_t(r) * k, seed++, -1, 1, dt);
            Tensor xc = cpu_tensor(xv, r, k), xg = upload(xv, r, k, dt);
            Tensor sc, sg, cc, cg;
            brotensor::sum_rows(xc, sc);
            brotensor::sum_rows(xg, sg);
            VKT_CHECK(sg.rows == r && sg.cols == 1 && sg.dtype == dt);
            close_for(dt, download(sg), sc.to_host_vector(), 2e-7f * k, 1e-6f, tag("sum_rows", dt, r, k));
            brotensor::sum_cols(xc, cc);
            brotensor::sum_cols(xg, cg);
            VKT_CHECK(cg.rows == 1 && cg.cols == k && cg.dtype == dt);
            close_for(dt, download(cg), cc.to_host_vector(), 2e-7f * r, 1e-6f, tag("sum_cols", dt, r, k));

            Tensor ic, ig, ii = Tensor::empty_on(vk(), r, 1, Dtype::INT32);
            brotensor::argmax_rows(xc, ic);
            brotensor::argmax_rows(xg, ig);
            brotensor::argmax_rows(xg, ii);
            VKT_CHECK(ig.dtype == Dtype::FP32 && ii.dtype == Dtype::INT32);
            expect_close(download(ig), ic.to_host_vector(), 0, 0, tag("argmax_rows", dt, r, k));
            std::vector<std::int32_t> iv(static_cast<std::size_t>(r));
            ii.copy_to_host_raw(iv.data(), iv.size() * 4);
            std::vector<float> ivf(iv.begin(), iv.end());
            expect_close(ivf, ic.to_host_vector(), 0, 0, tag("argmax_rows int32", dt, r, k));
        }
    }
    // Ties keep the lowest index; a row of -inf reports 0 like the CPU loop.
    std::vector<float> t = {1, 5, 3, 5, 5, 0, 2, 9, 9, 1, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    Tensor ic, ig;
    brotensor::argmax_rows(cpu_tensor(t, 3, 5), ic);
    brotensor::argmax_rows(upload(t, 3, 5, Dtype::FP32), ig);
    expect_close(download(ig), ic.to_host_vector(), 0, 0, "argmax_rows ties / -inf row");
    expect_close(download(ig), {1, 2, 0}, 0, 0, "argmax_rows ties keep lowest index");
}

}  // namespace

void run_op_tests() {
    test_unary();
    test_binary();
    test_backward();
    test_bias();
    test_cast();
    test_copies();
    test_reductions();
}

}  // namespace vkt
