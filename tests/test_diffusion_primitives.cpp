// Coverage for the diffusion-targeted additions:
//   - clamp (FP32 + FP16)
//   - add_scalar_inplace FP16 dispatch
//   - embedding_lookup_forward FP16
//   - flash_attention_forward with causal=true (compared vs causal mask)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return out;
}

static void test_clamp_fp32() {
    std::printf("  clamp fp32\n");
    const int N = 128;
    std::vector<float> host(N);
    std::mt19937 rng(0xC1A1);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : host) v = dist(rng);

    Tensor y = Tensor::from_host_on(Device::CUDA, host.data(), N, 1);
    brotensor::clamp(y, -1.0f, 1.0f);

    std::vector<float> got(N);
    y.copy_to_host(got.data());
    brotensor::sync_all();

    for (int i = 0; i < N; ++i) {
        float r = host[i];
        if (r < -1.0f) r = -1.0f;
        if (r >  1.0f) r =  1.0f;
        CHECK(std::fabs(got[i] - r) < 1e-6f);
    }
}

static void test_clamp_fp16() {
    std::printf("  clamp fp16\n");
    const int N = 128;
    std::vector<float> host_f(N);
    std::mt19937 rng(0xC1A2);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : host_f) v = dist(rng);
    auto host_h = to_fp16(host_f);

    Tensor y = Tensor::from_host_fp16_on(Device::CUDA, host_h.data(), N, 1);
    brotensor::clamp(y, -1.0f, 1.0f);

    std::vector<uint16_t> got_h(N);
    y.copy_to_host_fp16(got_h.data());
    brotensor::sync_all();

    for (int i = 0; i < N; ++i) {
        float r = brotensor::fp16_bits_to_fp32(host_h[i]);
        if (r < -1.0f) r = -1.0f;
        if (r >  1.0f) r =  1.0f;
        const float g = brotensor::fp16_bits_to_fp32(got_h[i]);
        CHECK(std::fabs(g - r) < 1e-3f);
    }
}

static void test_add_scalar_fp16() {
    std::printf("  add_scalar_inplace fp16\n");
    const int N = 64;
    std::vector<float> host_f(N);
    for (int i = 0; i < N; ++i) host_f[i] = static_cast<float>(i) * 0.1f;
    auto host_h = to_fp16(host_f);

    Tensor y = Tensor::from_host_fp16_on(Device::CUDA, host_h.data(), N, 1);
    brotensor::add_scalar_inplace(y, 3.5f);

    std::vector<uint16_t> got_h(N);
    y.copy_to_host_fp16(got_h.data());
    brotensor::sync_all();

    for (int i = 0; i < N; ++i) {
        const float r = host_f[i] + 3.5f;
        const float g = brotensor::fp16_bits_to_fp32(got_h[i]);
        CHECK(std::fabs(g - r) < 5e-3f);
    }
}

static void test_embedding_fp16() {
    std::printf("  embedding_lookup_forward fp16\n");
    const int V = 7, D = 5, B = 4;
    std::vector<float> table_f(V * D);
    for (int i = 0; i < V * D; ++i) table_f[i] = 0.01f * static_cast<float>(i);
    auto table_h = to_fp16(table_f);
    Tensor table = Tensor::from_host_fp16_on(Device::CUDA, table_h.data(), V, D);

    std::vector<int32_t> idx_host = {0, 3, 1, 6};
    // Stage indices on device via a tiny upload using a B*1 FP32 tensor as
    // storage, reinterpreting the float buffer as B int32s by writing them
    // with memcpy at upload time.
    std::vector<float> idx_as_float(B);
    for (int i = 0; i < B; ++i) {
        int32_t v = idx_host[i];
        std::memcpy(&idx_as_float[i], &v, sizeof(int32_t));
    }
    Tensor idx_buf = Tensor::from_host_on(Device::CUDA, idx_as_float.data(), B, 1);

    Tensor out;
    brotensor::embedding_lookup_forward(
        table, reinterpret_cast<const int32_t*>(idx_buf.data), B, out);
    CHECK(out.rows == B && out.cols == D && out.dtype == Dtype::FP16);

    std::vector<uint16_t> got_h(out.size());
    out.copy_to_host_fp16(got_h.data());
    brotensor::sync_all();

    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < D; ++j) {
            const float r = table_f[idx_host[b] * D + j];
            const float g = brotensor::fp16_bits_to_fp32(got_h[b * D + j]);
            CHECK(std::fabs(g - r) < 1e-3f);
        }
    }
}

// Reference attention with explicit causal mask via build_causal_mask_row
// per query — slow but reuses an already-tested path.
static void test_flash_causal() {
    std::printf("  flash_attention causal\n");
    const int L = 13, D = 32, nh = 2;
    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(L*D), K(L*D), V(L*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V);
    Tensor Qg = Tensor::from_host_fp16_on(Device::CUDA, Qh.data(), L, D);
    Tensor Kg = Tensor::from_host_fp16_on(Device::CUDA, Kh.data(), L, D);
    Tensor Vg = Tensor::from_host_fp16_on(Device::CUDA, Vh.data(), L, D);

    // causal=true path.
    Tensor O_causal;
    brotensor::flash_attention_forward(Qg, Kg, Vg, nullptr, nh,
                                       /*causal=*/true, O_causal);
    std::vector<uint16_t> got_h(O_causal.size());
    O_causal.copy_to_host_fp16(got_h.data());

    // Reference: same flash kernel but with a per-query mask, run one query at
    // a time by zeroing rows of K/V outside the causal window via a mask
    // tensor. The mask path is shared logic from the non-causal kernel.
    Tensor O_ref = Tensor::zeros_on(Device::CUDA, L, D, Dtype::FP16);
    for (int q = 0; q < L; ++q) {
        Tensor Qrow = Tensor::zeros_on(Device::CUDA, 1, D, Dtype::FP16);
        brotensor::copy_d2d(Qg, q * D, Qrow, 0, D);

        Tensor mask = Tensor::zeros_on(Device::CUDA, L, 1);
        brotensor::build_causal_mask_row(L, q, mask);

        Tensor Orow;
        brotensor::flash_attention_forward(
            Qrow, Kg, Vg, static_cast<const float*>(mask.data), nh,
            /*causal=*/false, Orow);
        brotensor::copy_d2d(Orow, 0, O_ref, q * D, D);
    }
    std::vector<uint16_t> ref_h(O_ref.size());
    O_ref.copy_to_host_fp16(ref_h.data());
    brotensor::sync_all();

    float max_err = 0.0f;
    int bad = 0;
    for (int i = 0; i < L * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_h[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref_h[i]);
        const float e = std::fabs(g - r);
        if (e > max_err) max_err = e;
        // FP16 tolerance; per-query path and tiled path take different
        // numerical routes through the online softmax.
        if (e > 5e-3f + 5e-3f * std::fabs(r)) ++bad;
    }
    std::printf("    max_err=%g bad=%d\n", max_err, bad);
    CHECK(bad == 0);
}

static float gelu_ref(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}
static float gelu_grad_ref(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = std::tanh(u);
    const float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}

static void test_geglu_fp32() {
    std::printf("  geglu fp32 fwd+bwd\n");
    const int B = 3, D = 9;
    const int two_d = 2 * D;
    std::mt19937 rng(0xCEFA);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> hx(B * two_d), hdy(B * D);
    for (auto& v : hx)  v = dist(rng);
    for (auto& v : hdy) v = dist(rng);

    Tensor X  = Tensor::from_host_on(Device::CUDA, hx.data(),  B, two_d);
    Tensor dY = Tensor::from_host_on(Device::CUDA, hdy.data(), B, D);

    Tensor Y;
    brotensor::geglu_forward(X, Y);
    CHECK(Y.rows == B && Y.cols == D && Y.dtype == Dtype::FP32);
    std::vector<float> gotY(B * D);
    Y.copy_to_host(gotY.data());
    brotensor::sync_all();
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a  = hx[b * two_d + d];
            const float bh = hx[b * two_d + D + d];
            const float r  = a * gelu_ref(bh);
            CHECK(std::fabs(gotY[b * D + d] - r) < 1e-5f + 1e-5f * std::fabs(r));
        }
    }

    Tensor dX;
    brotensor::geglu_backward(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP32);
    std::vector<float> gotdX(B * two_d);
    dX.copy_to_host(gotdX.data());
    brotensor::sync_all();
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a   = hx[b * two_d + d];
            const float bh  = hx[b * two_d + D + d];
            const float dy  = hdy[b * D + d];
            const float dA  = dy * gelu_ref(bh);
            const float dBh = dy * a * gelu_grad_ref(bh);
            const float ga  = gotdX[b * two_d + d];
            const float gb  = gotdX[b * two_d + D + d];
            CHECK(std::fabs(ga - dA)  < 1e-5f + 1e-5f * std::fabs(dA));
            CHECK(std::fabs(gb - dBh) < 1e-5f + 1e-5f * std::fabs(dBh));
        }
    }
}

static float gelu_exact_ref(float v) {
    return 0.5f * v * (1.0f + std::erf(v * 0.70710678118f));
}
static float gelu_exact_grad_ref(float v) {
    return 0.5f * (1.0f + std::erf(v * 0.70710678118f))
         + v * std::exp(-0.5f * v * v) * 0.39894228040f;
}

static void test_geglu_exact_fp32() {
    std::printf("  geglu_exact fp32 fwd+bwd\n");
    const int B = 3, D = 9;
    const int two_d = 2 * D;
    std::mt19937 rng(0xCEFC);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> hx(B * two_d), hdy(B * D);
    for (auto& v : hx)  v = dist(rng);
    for (auto& v : hdy) v = dist(rng);

    Tensor X  = Tensor::from_host_on(Device::CUDA, hx.data(),  B, two_d);
    Tensor dY = Tensor::from_host_on(Device::CUDA, hdy.data(), B, D);

    Tensor Y;
    brotensor::geglu_exact_forward(X, Y);
    CHECK(Y.rows == B && Y.cols == D && Y.dtype == Dtype::FP32);
    std::vector<float> gotY(B * D);
    Y.copy_to_host(gotY.data());
    brotensor::sync_all();
    float max_err_fwd = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a  = hx[b * two_d + d];
            const float bh = hx[b * two_d + D + d];
            const float r  = a * gelu_exact_ref(bh);
            const float e  = std::fabs(gotY[b * D + d] - r);
            if (e > max_err_fwd) max_err_fwd = e;
            CHECK(e < 1e-5f + 1e-5f * std::fabs(r));
        }
    }
    std::printf("    fwd max_err=%g\n", max_err_fwd);

    Tensor dX;
    brotensor::geglu_exact_backward(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP32);
    std::vector<float> gotdX(B * two_d);
    dX.copy_to_host(gotdX.data());
    brotensor::sync_all();
    float max_err_bwd = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a   = hx[b * two_d + d];
            const float bh  = hx[b * two_d + D + d];
            const float dy  = hdy[b * D + d];
            const float dA  = dy * gelu_exact_ref(bh);
            const float dBh = dy * a * gelu_exact_grad_ref(bh);
            const float ga  = gotdX[b * two_d + d];
            const float gb  = gotdX[b * two_d + D + d];
            const float ea  = std::fabs(ga - dA);
            const float eb  = std::fabs(gb - dBh);
            if (ea > max_err_bwd) max_err_bwd = ea;
            if (eb > max_err_bwd) max_err_bwd = eb;
            CHECK(ea < 1e-5f + 1e-5f * std::fabs(dA));
            CHECK(eb < 1e-5f + 1e-5f * std::fabs(dBh));
        }
    }
    std::printf("    bwd max_err=%g\n", max_err_bwd);
}

static void test_geglu_exact_fp16() {
    std::printf("  geglu_exact fp16 fwd+bwd\n");
    const int B = 3, D = 9;
    const int two_d = 2 * D;
    std::mt19937 rng(0xCEFD);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> hx_f(B * two_d), hdy_f(B * D);
    for (auto& v : hx_f)  v = dist(rng);
    for (auto& v : hdy_f) v = dist(rng);
    auto hx_h  = to_fp16(hx_f);
    auto hdy_h = to_fp16(hdy_f);

    Tensor X  = Tensor::from_host_fp16_on(Device::CUDA, hx_h.data(),  B, two_d);
    Tensor dY = Tensor::from_host_fp16_on(Device::CUDA, hdy_h.data(), B, D);

    Tensor Y;
    brotensor::geglu_exact_forward(X, Y);
    CHECK(Y.rows == B && Y.cols == D && Y.dtype == Dtype::FP16);
    std::vector<uint16_t> gotY_h(B * D);
    Y.copy_to_host_fp16(gotY_h.data());
    brotensor::sync_all();
    float max_err_fwd = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a  = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + d]);
            const float bh = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + D + d]);
            const float r  = a * gelu_exact_ref(bh);
            const float g  = brotensor::fp16_bits_to_fp32(gotY_h[b * D + d]);
            const float e  = std::fabs(g - r);
            if (e > max_err_fwd) max_err_fwd = e;
            CHECK(e < 1e-2f + 1e-2f * std::fabs(r));
        }
    }
    std::printf("    fwd max_err=%g\n", max_err_fwd);

    Tensor dX;
    brotensor::geglu_exact_backward(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP16);
    std::vector<uint16_t> got_h(B * two_d);
    dX.copy_to_host_fp16(got_h.data());
    brotensor::sync_all();
    float max_err_bwd = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a   = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + d]);
            const float bh  = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + D + d]);
            const float dy  = brotensor::fp16_bits_to_fp32(hdy_h[b * D + d]);
            const float dA  = dy * gelu_exact_ref(bh);
            const float dBh = dy * a * gelu_exact_grad_ref(bh);
            const float ga  = brotensor::fp16_bits_to_fp32(got_h[b * two_d + d]);
            const float gb  = brotensor::fp16_bits_to_fp32(got_h[b * two_d + D + d]);
            const float ea  = std::fabs(ga - dA);
            const float eb  = std::fabs(gb - dBh);
            if (ea > max_err_bwd) max_err_bwd = ea;
            if (eb > max_err_bwd) max_err_bwd = eb;
            CHECK(ea < 1e-2f + 1e-2f * std::fabs(dA));
            CHECK(eb < 1e-2f + 1e-2f * std::fabs(dBh));
        }
    }
    std::printf("    bwd max_err=%g\n", max_err_bwd);
}

static void test_geglu_fp16_bwd() {
    std::printf("  geglu fp16 bwd\n");
    const int B = 3, D = 9;
    const int two_d = 2 * D;
    std::mt19937 rng(0xCEFB);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> hx_f(B * two_d), hdy_f(B * D);
    for (auto& v : hx_f)  v = dist(rng);
    for (auto& v : hdy_f) v = dist(rng);
    auto hx_h  = to_fp16(hx_f);
    auto hdy_h = to_fp16(hdy_f);

    Tensor X  = Tensor::from_host_fp16_on(Device::CUDA, hx_h.data(),  B, two_d);
    Tensor dY = Tensor::from_host_fp16_on(Device::CUDA, hdy_h.data(), B, D);

    Tensor dX;
    brotensor::geglu_backward(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP16);
    std::vector<uint16_t> got_h(B * two_d);
    dX.copy_to_host_fp16(got_h.data());
    brotensor::sync_all();

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a   = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + d]);
            const float bh  = brotensor::fp16_bits_to_fp32(hx_h[b * two_d + D + d]);
            const float dy  = brotensor::fp16_bits_to_fp32(hdy_h[b * D + d]);
            const float dA  = dy * gelu_ref(bh);
            const float dBh = dy * a * gelu_grad_ref(bh);
            const float ga  = brotensor::fp16_bits_to_fp32(got_h[b * two_d + d]);
            const float gb  = brotensor::fp16_bits_to_fp32(got_h[b * two_d + D + d]);
            CHECK(std::fabs(ga - dA)  < 1e-2f + 1e-2f * std::fabs(dA));
            CHECK(std::fabs(gb - dBh) < 1e-2f + 1e-2f * std::fabs(dBh));
        }
    }
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_diffusion_primitives\n");

    test_clamp_fp32();
    test_clamp_fp16();
    test_add_scalar_fp16();
    test_embedding_fp16();
    test_flash_causal();
    test_geglu_fp32();
    test_geglu_fp16_bwd();
    test_geglu_exact_fp32();
    test_geglu_exact_fp16();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll diffusion-primitive checks passed.\n");
    return 0;
}
