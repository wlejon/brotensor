// W8A16 parity for the new batched-linear primitive and the three fused
// flash-attention INT8 wrappers, vs the FP16 reference paths:
//   linear_forward_batched_int8w_fp16_gpu  vs  linear_forward_batched_fp16_gpu(W_deq, X)
//   flash_attention_project_kv_int8w_fp16_gpu  vs  flash_attention_project_kv_gpu
//   flash_attention_q_with_kv_cached_int8w_fp16_gpu  vs  FP16 cached path
//   flash_attention_qkvo_int8w_fp16_gpu  vs  flash_attention_qkvo_forward_gpu

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
static inline void cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n);
}
#define cudaMemcpyHostToDevice 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

// Quantise W (out, in) FP16 → (W_int8, scales) and produce the matching
// FP16 dequant-reference. Uploads both representations.
static void prepare_w8a16(int out, int in,
                          const std::vector<uint16_t>& Wh,
                          GpuTensor& W_int8, GpuTensor& S,
                          GpuTensor& W_deq) {
    std::vector<int8_t> Wq(out * in);
    std::vector<float>  scales(out);
    brotensor::quantize_int8_per_row_host(Wh.data(), out, in, Wq.data(), scales.data());
    std::vector<uint16_t> Wdeq(out * in);
    for (int r = 0; r < out; ++r) {
        const float s = scales[r];
        for (int c = 0; c < in; ++c) {
            Wdeq[r * in + c] = brotensor::fp32_to_fp16_bits(
                static_cast<float>(Wq[r * in + c]) * s);
        }
    }
    W_int8.resize(out, in, Dtype::INT8);
    cudaMemcpy(W_int8.data, Wq.data(), out * in * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    brotensor::upload(scales.data(), out, 1, S);
    brotensor::upload_fp16(Wdeq.data(), out, in, W_deq);
}

static float max_abs_err(const std::vector<uint16_t>& got,
                         const std::vector<uint16_t>& ref) {
    float m = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref[i]);
        const float e = std::fabs(g - r);
        if (e > m) m = e;
    }
    return m;
}

static void test_linear_batched_int8w() {
    std::printf("  linear_forward_batched_int8w_fp16_gpu B=12 in=128 out=192 (+bias)\n");
    const int B = 12, IN = 128, OUT = 192;
    std::mt19937 rng(0xAB);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Wf(OUT * IN), Xf(B * IN), bf(OUT);
    for (auto& v : Wf) v = dist(rng);
    for (auto& v : Xf) v = dist(rng);
    for (auto& v : bf) v = dist(rng);
    auto Wh = to_fp16(Wf), Xh = to_fp16(Xf), bh = to_fp16(bf);

    GpuTensor W_int8, S, W_deq, X_g, B_g, Y_ref, Y_int8;
    prepare_w8a16(OUT, IN, Wh, W_int8, S, W_deq);
    brotensor::upload_fp16(Xh.data(), B,   IN, X_g);
    brotensor::upload_fp16(bh.data(), OUT, 1,  B_g);

    brotensor::linear_forward_batched_fp16_gpu(W_deq, &B_g, X_g, Y_ref);
    brotensor::linear_forward_batched_int8w_fp16_gpu(W_int8, S, &B_g, X_g, Y_int8);
    CHECK(Y_int8.dtype == Dtype::FP16 && Y_int8.rows == B && Y_int8.cols == OUT);
    std::vector<uint16_t> ref(B * OUT), got(B * OUT);
    brotensor::download_fp16(Y_ref,  ref.data());
    brotensor::download_fp16(Y_int8, got.data());
    brotensor::cuda_sync();
    const float me = max_abs_err(got, ref);
    std::printf("    max_err=%g\n", me);
    CHECK(me < 1e-2f);
}

// Cross-attention-ish shape; one head_dim=64 head, masking off.
static void test_flash_qkvo_int8w() {
    std::printf("  flash_attention_qkvo_int8w_fp16_gpu Lq=8 Lk=12 D=64 H=2 (cross)\n");
    const int Lq = 8, Lk = 12, D = 64, D_ctx = 80;
    const int num_heads = 2;
    std::mt19937 rng(0xCD);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> Xf(Lq * D), Cf(Lk * D_ctx);
    std::vector<float> Wqf(D * D), Wkf(D * D_ctx), Wvf(D * D_ctx), Wof(D * D);
    std::vector<float> bqf(D), bkf(D), bvf(D), bof(D);
    for (auto& v : Xf)  v = dist(rng);
    for (auto& v : Cf)  v = dist(rng);
    for (auto& v : Wqf) v = dist(rng);
    for (auto& v : Wkf) v = dist(rng);
    for (auto& v : Wvf) v = dist(rng);
    for (auto& v : Wof) v = dist(rng);
    for (auto& v : bqf) v = dist(rng);
    for (auto& v : bkf) v = dist(rng);
    for (auto& v : bvf) v = dist(rng);
    for (auto& v : bof) v = dist(rng);
    auto Xh  = to_fp16(Xf);
    auto Ch  = to_fp16(Cf);
    auto Wqh = to_fp16(Wqf), Wkh = to_fp16(Wkf), Wvh = to_fp16(Wvf), Woh = to_fp16(Wof);
    auto bqh = to_fp16(bqf), bkh = to_fp16(bkf), bvh = to_fp16(bvf), boh = to_fp16(bof);

    GpuTensor X_g, C_g, Bq, Bk, Bv, Bo;
    brotensor::upload_fp16(Xh.data(),  Lq, D,     X_g);
    brotensor::upload_fp16(Ch.data(),  Lk, D_ctx, C_g);
    brotensor::upload_fp16(bqh.data(), D, 1, Bq);
    brotensor::upload_fp16(bkh.data(), D, 1, Bk);
    brotensor::upload_fp16(bvh.data(), D, 1, Bv);
    brotensor::upload_fp16(boh.data(), D, 1, Bo);

    GpuTensor Wq_i8, Wk_i8, Wv_i8, Wo_i8;
    GpuTensor sq, sk, sv, so;
    GpuTensor Wq_deq, Wk_deq, Wv_deq, Wo_deq;
    prepare_w8a16(D, D,     Wqh, Wq_i8, sq, Wq_deq);
    prepare_w8a16(D, D_ctx, Wkh, Wk_i8, sk, Wk_deq);
    prepare_w8a16(D, D_ctx, Wvh, Wv_i8, sv, Wv_deq);
    prepare_w8a16(D, D,     Woh, Wo_i8, so, Wo_deq);

    GpuTensor O_ref, O_int8;
    brotensor::flash_attention_qkvo_forward_gpu(
        X_g, &C_g,
        Wq_deq, &Bq, Wk_deq, &Bk, Wv_deq, &Bv, Wo_deq, &Bo,
        nullptr, num_heads, /*causal=*/false, O_ref);
    brotensor::flash_attention_qkvo_int8w_fp16_gpu(
        X_g, &C_g,
        Wq_i8, sq, &Bq, Wk_i8, sk, &Bk, Wv_i8, sv, &Bv, Wo_i8, so, &Bo,
        nullptr, num_heads, /*causal=*/false, O_int8);
    CHECK(O_int8.dtype == Dtype::FP16 && O_int8.rows == Lq && O_int8.cols == D);

    std::vector<uint16_t> ref(Lq * D), got(Lq * D);
    brotensor::download_fp16(O_ref,  ref.data());
    brotensor::download_fp16(O_int8, got.data());
    brotensor::cuda_sync();
    const float me = max_abs_err(got, ref);
    std::printf("    max_err=%g\n", me);
    // Whole-attention path: the quantised weights are identical numerically
    // to the dequant FP16 path inside the kernel, and the FP16 attention core
    // is shared bitwise. Any divergence is FP16 rounding only.
    CHECK(me < 5e-2f);
}

// Self-attention split-path: project_kv then q_with_kv_cached (the path
// flash_attention_qkvo composes internally).
static void test_flash_split_int8w() {
    std::printf("  flash_attention_project_kv + q_with_kv_cached (INT8 path) self-attn L=10 D=64\n");
    const int L = 10, D = 64;
    const int num_heads = 4;
    std::mt19937 rng(0x77);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> Xf(L * D);
    std::vector<float> Wqf(D * D), Wkf(D * D), Wvf(D * D), Wof(D * D);
    for (auto& v : Xf)  v = dist(rng);
    for (auto& v : Wqf) v = dist(rng);
    for (auto& v : Wkf) v = dist(rng);
    for (auto& v : Wvf) v = dist(rng);
    for (auto& v : Wof) v = dist(rng);
    auto Xh = to_fp16(Xf);
    auto Wqh = to_fp16(Wqf), Wkh = to_fp16(Wkf), Wvh = to_fp16(Wvf), Woh = to_fp16(Wof);

    GpuTensor X_g;
    brotensor::upload_fp16(Xh.data(), L, D, X_g);

    GpuTensor Wq_i8, Wk_i8, Wv_i8, Wo_i8, sq, sk, sv, so;
    GpuTensor Wq_deq, Wk_deq, Wv_deq, Wo_deq;
    prepare_w8a16(D, D, Wqh, Wq_i8, sq, Wq_deq);
    prepare_w8a16(D, D, Wkh, Wk_i8, sk, Wk_deq);
    prepare_w8a16(D, D, Wvh, Wv_i8, sv, Wv_deq);
    prepare_w8a16(D, D, Woh, Wo_i8, so, Wo_deq);

    // Reference: project_kv + q_with_kv_cached using dequantised FP16 weights.
    GpuTensor K_ref, V_ref, O_ref;
    brotensor::flash_attention_project_kv_gpu(X_g, Wk_deq, nullptr, Wv_deq, nullptr,
                                              K_ref, V_ref);
    brotensor::flash_attention_q_with_kv_cached_forward_gpu(
        X_g, K_ref, V_ref, Wq_deq, nullptr, Wo_deq, nullptr,
        nullptr, num_heads, /*causal=*/false, O_ref);

    // INT8 path.
    GpuTensor K_i8, V_i8, O_i8;
    brotensor::flash_attention_project_kv_int8w_fp16_gpu(
        X_g, Wk_i8, sk, nullptr, Wv_i8, sv, nullptr, K_i8, V_i8);
    brotensor::flash_attention_q_with_kv_cached_int8w_fp16_gpu(
        X_g, K_i8, V_i8, Wq_i8, sq, nullptr, Wo_i8, so, nullptr,
        nullptr, num_heads, /*causal=*/false, O_i8);
    CHECK(O_i8.dtype == Dtype::FP16 && O_i8.rows == L && O_i8.cols == D);

    std::vector<uint16_t> ref(L * D), got(L * D);
    brotensor::download_fp16(O_ref, ref.data());
    brotensor::download_fp16(O_i8,  got.data());
    brotensor::cuda_sync();
    const float me = max_abs_err(got, ref);
    std::printf("    max_err=%g\n", me);
    CHECK(me < 5e-2f);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_int8_attention\n");
    test_linear_batched_int8w();
    test_flash_split_int8w();
    test_flash_qkvo_int8w();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
