// Coverage for the diffusion-targeted additions:
//   - clamp_gpu (FP32 + FP16)
//   - add_scalar_inplace_gpu FP16 dispatch
//   - embedding_lookup_forward_gpu FP16
//   - flash_attention_forward_gpu with causal=true (compared vs causal mask)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

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

    GpuTensor y;
    brotensor::upload(host.data(), N, 1, y);
    brotensor::clamp_gpu(y, -1.0f, 1.0f);

    std::vector<float> got(N);
    brotensor::download(y, got.data());
    brotensor::cuda_sync();

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

    GpuTensor y;
    brotensor::upload_fp16(host_h.data(), N, 1, y);
    brotensor::clamp_gpu(y, -1.0f, 1.0f);

    std::vector<uint16_t> got_h(N);
    brotensor::download_fp16(y, got_h.data());
    brotensor::cuda_sync();

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

    GpuTensor y;
    brotensor::upload_fp16(host_h.data(), N, 1, y);
    brotensor::add_scalar_inplace_gpu(y, 3.5f);

    std::vector<uint16_t> got_h(N);
    brotensor::download_fp16(y, got_h.data());
    brotensor::cuda_sync();

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
    GpuTensor table;
    brotensor::upload_fp16(table_h.data(), V, D, table);

    std::vector<int32_t> idx_host = {0, 3, 1, 6};
    // Stage indices on device via a tiny upload using upload(..) into FP32 tensor
    // for storage, then reinterpret. Since GpuTensor is float*, we treat a B*1
    // FP32 buffer as B int32s by writing them with bit_cast at upload time.
    GpuTensor idx_buf(B, 1, Dtype::FP32);
    std::vector<float> idx_as_float(B);
    for (int i = 0; i < B; ++i) {
        int32_t v = idx_host[i];
        std::memcpy(&idx_as_float[i], &v, sizeof(int32_t));
    }
    brotensor::upload(idx_as_float.data(), B, 1, idx_buf);

    GpuTensor out;
    brotensor::embedding_lookup_forward_gpu(
        table, reinterpret_cast<const int32_t*>(idx_buf.data), B, out);
    CHECK(out.rows == B && out.cols == D && out.dtype == Dtype::FP16);

    std::vector<uint16_t> got_h(out.size());
    brotensor::download_fp16(out, got_h.data());
    brotensor::cuda_sync();

    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < D; ++j) {
            const float r = table_f[idx_host[b] * D + j];
            const float g = brotensor::fp16_bits_to_fp32(got_h[b * D + j]);
            CHECK(std::fabs(g - r) < 1e-3f);
        }
    }
}

// Reference attention with explicit causal mask via build_causal_mask_row_gpu
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
    GpuTensor Qg, Kg, Vg;
    brotensor::upload_fp16(Qh.data(), L, D, Qg);
    brotensor::upload_fp16(Kh.data(), L, D, Kg);
    brotensor::upload_fp16(Vh.data(), L, D, Vg);

    // causal=true path.
    GpuTensor O_causal;
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, nullptr, nh,
                                           /*causal=*/true, O_causal);
    std::vector<uint16_t> got_h(O_causal.size());
    brotensor::download_fp16(O_causal, got_h.data());

    // Reference: same flash kernel but with a per-query mask, run one query at
    // a time by zeroing rows of K/V outside the causal window via a mask
    // tensor. The mask path is shared logic from the non-causal kernel.
    GpuTensor O_ref(L, D, Dtype::FP16);
    for (int q = 0; q < L; ++q) {
        GpuTensor Qrow(1, D, Dtype::FP16);
        brotensor::copy_d2d_gpu(Qg, q * D, Qrow, 0, D);

        GpuTensor mask;
        brotensor::build_causal_mask_row_gpu(L, q, mask);

        GpuTensor Orow;
        brotensor::flash_attention_forward_gpu(Qrow, Kg, Vg, mask.data, nh,
                                               /*causal=*/false, Orow);
        brotensor::copy_d2d_gpu(Orow, 0, O_ref, q * D, D);
    }
    std::vector<uint16_t> ref_h(O_ref.size());
    brotensor::download_fp16(O_ref, ref_h.data());
    brotensor::cuda_sync();

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

    GpuTensor X, dY;
    brotensor::upload(hx.data(),  B, two_d, X);
    brotensor::upload(hdy.data(), B, D,    dY);

    GpuTensor Y;
    brotensor::geglu_forward_gpu(X, Y);
    CHECK(Y.rows == B && Y.cols == D && Y.dtype == Dtype::FP32);
    std::vector<float> gotY(B * D);
    brotensor::download(Y, gotY.data());
    brotensor::cuda_sync();
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a  = hx[b * two_d + d];
            const float bh = hx[b * two_d + D + d];
            const float r  = a * gelu_ref(bh);
            CHECK(std::fabs(gotY[b * D + d] - r) < 1e-5f + 1e-5f * std::fabs(r));
        }
    }

    GpuTensor dX;
    brotensor::geglu_backward_gpu(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP32);
    std::vector<float> gotdX(B * two_d);
    brotensor::download(dX, gotdX.data());
    brotensor::cuda_sync();
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

    GpuTensor X, dY;
    brotensor::upload_fp16(hx_h.data(),  B, two_d, X);
    brotensor::upload_fp16(hdy_h.data(), B, D,    dY);

    GpuTensor dX;
    brotensor::geglu_backward_gpu(X, dY, dX);
    CHECK(dX.rows == B && dX.cols == two_d && dX.dtype == Dtype::FP16);
    std::vector<uint16_t> got_h(B * two_d);
    brotensor::download_fp16(dX, got_h.data());
    brotensor::cuda_sync();

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
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_diffusion_primitives\n");

    test_clamp_fp32();
    test_clamp_fp16();
    test_add_scalar_fp16();
    test_embedding_fp16();
    test_flash_causal();
    test_geglu_fp32();
    test_geglu_fp16_bwd();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll diffusion-primitive checks passed.\n");
    return 0;
}
