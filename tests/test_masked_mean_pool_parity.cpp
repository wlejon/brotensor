// CPU↔GPU parity tests for brotensor::masked_mean_pool_forward / _backward.
//
// The pooled text-embedding op (llm2vec encode_pooled). CPU is FP32-only;
// the GPU backends dispatch on X.dtype (FP32 / FP16 / BF16) with FP32
// accumulation — the 16-bit runs quantise the FP32 inputs first so both
// sides start from the same values, then compare with a loose tolerance.
// The mask is always FP32 and device-resident (or null = all rows valid).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

std::vector<float> make_mask(int K, uint64_t seed, bool all_zero = false) {
    SplitMix64 rng(seed);
    std::vector<float> m(static_cast<size_t>(K));
    bool any = false;
    for (int k = 0; k < K; ++k) {
        m[k] = all_zero ? 0.0f : (rng.next_f01() < 0.5f ? 0.0f : 1.0f);
        any = any || (m[k] != 0.0f);
    }
    if (!all_zero && !any) m[0] = 1.0f;   // keep num_valid > 0 deterministic
    return m;
}

void run_fwd(int K, int D, const std::vector<float>* mask, uint64_t seed,
             brotensor::Dtype dt, float atol, float rtol) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(K, D);
    fill_random(X, rng);
    if (dt == brotensor::Dtype::FP16) X = fp16_host_to_f32(to_fp16_host(X));
    if (dt == brotensor::Dtype::BF16) X = bf16_host_to_f32(to_bf16_host(X));

    Tensor cpu_y;
    brotensor::masked_mean_pool_forward(X, mask ? mask->data() : nullptr,
                                        cpu_y);

    Tensor gX = (dt == brotensor::Dtype::FP16) ? to_fp16_gpu(X)
              : (dt == brotensor::Dtype::BF16) ? to_bf16_gpu(X)
                                               : X.to(gpu_device());
    Tensor d_mask = upload_mask(mask);
    Tensor gpu_y;
    brotensor::masked_mean_pool_forward(
        gX, mask ? static_cast<const float*>(d_mask.data) : nullptr, gpu_y);

    Tensor host_y = download_to_host(gpu_y);
    if (dt == brotensor::Dtype::FP16) host_y = fp16_host_to_f32(host_y);
    if (dt == brotensor::Dtype::BF16) host_y = bf16_host_to_f32(host_y);
    compare_tensors(cpu_y, host_y, "masked_mean_pool_fwd", atol, rtol);
}

void run_bwd(int K, int D, const std::vector<float>* mask, uint64_t seed,
             brotensor::Dtype dt, float atol, float rtol) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(D, 1);
    fill_random(dY, rng);
    if (dt == brotensor::Dtype::FP16) dY = fp16_host_to_f32(to_fp16_host(dY));
    if (dt == brotensor::Dtype::BF16) dY = bf16_host_to_f32(to_bf16_host(dY));

    Tensor cpu_dX;
    brotensor::masked_mean_pool_backward(dY, mask ? mask->data() : nullptr,
                                         K, cpu_dX);

    Tensor gdY = (dt == brotensor::Dtype::FP16) ? to_fp16_gpu(dY)
               : (dt == brotensor::Dtype::BF16) ? to_bf16_gpu(dY)
                                                : dY.to(gpu_device());
    Tensor d_mask = upload_mask(mask);
    Tensor gpu_dX;
    brotensor::masked_mean_pool_backward(
        gdY, mask ? static_cast<const float*>(d_mask.data) : nullptr,
        K, gpu_dX);

    Tensor host_dX = download_to_host(gpu_dX);
    if (dt == brotensor::Dtype::FP16) host_dX = fp16_host_to_f32(host_dX);
    if (dt == brotensor::Dtype::BF16) host_dX = bf16_host_to_f32(host_dX);
    compare_tensors(cpu_dX, host_dX, "masked_mean_pool_bwd", atol, rtol);
}

constexpr auto FP32 = brotensor::Dtype::FP32;
constexpr auto FP16 = brotensor::Dtype::FP16;
constexpr auto BF16 = brotensor::Dtype::BF16;

} // namespace

// ── forward ────────────────────────────────────────────────────────────────

BT_PARITY_TEST(mmp_fwd_fp32_nomask_8x16) {
    run_fwd(8, 16, nullptr, 0x9100ull, FP32, 1e-6f, 1e-5f);
}
BT_PARITY_TEST(mmp_fwd_fp32_mask_33x64) {
    auto m = make_mask(33, 0x9101ull);
    run_fwd(33, 64, &m, 0x9102ull, FP32, 1e-6f, 1e-5f);
}
BT_PARITY_TEST(mmp_fwd_fp32_mask_all_zero) {
    auto m = make_mask(9, 0x9103ull, /*all_zero=*/true);
    run_fwd(9, 32, &m, 0x9104ull, FP32, 0.0f, 0.0f);
}
// The llm2vec shape class: FP16 hidden states pooled over the sequence.
// The Metal backend used to read these FP16 rows as FP32 bits (silent
// garbage); this is the regression case.
BT_PARITY_TEST(mmp_fwd_fp16_nomask_17x96) {
    run_fwd(17, 96, nullptr, 0x9110ull, FP16, 1e-2f, 1e-2f);
}
BT_PARITY_TEST(mmp_fwd_fp16_mask_64x128) {
    auto m = make_mask(64, 0x9111ull);
    run_fwd(64, 128, &m, 0x9112ull, FP16, 1e-2f, 1e-2f);
}
BT_PARITY_TEST(mmp_fwd_bf16_mask_31x48) {
    auto m = make_mask(31, 0x9113ull);
    run_fwd(31, 48, &m, 0x9114ull, BF16, 2e-2f, 2e-2f);
}

// ── backward ───────────────────────────────────────────────────────────────

BT_PARITY_TEST(mmp_bwd_fp32_nomask_8x16) {
    run_bwd(8, 16, nullptr, 0x9120ull, FP32, 1e-6f, 1e-5f);
}
BT_PARITY_TEST(mmp_bwd_fp32_mask_33x64) {
    auto m = make_mask(33, 0x9121ull);
    run_bwd(33, 64, &m, 0x9122ull, FP32, 1e-6f, 1e-5f);
}
BT_PARITY_TEST(mmp_bwd_fp16_mask_64x128) {
    auto m = make_mask(64, 0x9123ull);
    run_bwd(64, 128, &m, 0x9124ull, FP16, 1e-2f, 1e-2f);
}
BT_PARITY_TEST(mmp_bwd_bf16_nomask_17x96) {
    run_bwd(17, 96, nullptr, 0x9125ull, BF16, 2e-2f, 2e-2f);
}

int main() { return run_all("masked_mean_pool cpu/gpu parity"); }
