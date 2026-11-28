// Pure C++ dispatchers for Wave 2 Brass MIR Tensor Core kernels replacing:
//   q4k_wmma.cu, q6k_wmma.cu, q8_0_wmma.cu, linear_int8w_wmma.cu, conv2d_int8w_wmma.cu.

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "../kernels/mir_gemm_quant_mma.h"
#include "../vulkan/shaders/op_codes.h"

#include <atomic>
#include <cstdint>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

std::atomic<unsigned long long> g_q4k_mma_calls{0};
std::atomic<unsigned long long> g_q6k_mma_calls{0};
std::atomic<unsigned long long> g_q8_0_mma_calls{0};

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

bool launch_quant_gemm_mma(std::uint8_t fmt, Dtype dt,
                           const void* X, const uint8_t* W, const float* scales,
                           const void* bias, void* Y,
                           int B, int M, int K, int row_bytes) {
    if (!supports_sm80()) return false;
    const std::uint8_t cfg = (B > 64) ? 1u : 0u;
    const int bm = static_cast<int>(kernels::kGemmQuantMmaCfgs[cfg].bm);
    const int bn = static_cast<int>(kernels::kGemmQuantMmaCfgs[cfg].bn);
    const int gx = (M + bn - 1) / bn;
    const int gy = (B + bm - 1) / bm;
    const bool aligned_mnk = (B % bm == 0) && (M % bn == 0) && ((K & 31) == 0);
    const bool swizzle_l2 = (gx >= 4 && gy >= 4);

    const std::uint32_t op = static_cast<std::uint32_t>(fmt & 0x0fu) |
                             (static_cast<std::uint32_t>(cfg & 0x0fu) << 4);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (bias != nullptr ? 1u : 0u) |
        (aligned_mnk ? 2u : 0u) |
        (swizzle_l2 ? 4u : 0u));

    CUfunction fn = mir_function(
        {MirKind::GemmQuantMma, op, dt, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uintptr_t ptrs[] = {addr(X), addr(W), addr(scales), addr(bias), addr(Y)};
    const std::int32_t ints[] = {B, M, K, row_bytes, gy};
    mir_dispatch_args_3d(
        fn, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy), 1u,
        kernels::kGemmQuantMmaCfgs[cfg].threads, ptrs, ints, {});
    return true;
}

}  // namespace

namespace q4k_wmma_internal {

bool launch_linear_q4k_fp16_wmma(const Fp16* X, const uint8_t* W_q4k,
                                 const Fp16* bias, Fp16* Y,
                                 int B, int M, int K, CUstream /*stream*/) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % 256 != 0) return false;
    if (B < 4) return false;
    g_q4k_mma_calls.fetch_add(1ull, std::memory_order_relaxed);
    return launch_quant_gemm_mma(
        QF_Q4K, Dtype::FP16, X, W_q4k, nullptr, bias, Y, B, M, K, (K / 256) * 144);
}

}  // namespace q4k_wmma_internal

namespace q6k_wmma_internal {

bool launch_linear_q6k_fp16_wmma(const Fp16* X, const uint8_t* W_q6k,
                                 const Fp16* bias, Fp16* Y,
                                 int B, int M, int K, CUstream /*stream*/) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % 256 != 0) return false;
    if (B < 4) return false;
    g_q6k_mma_calls.fetch_add(1ull, std::memory_order_relaxed);
    return launch_quant_gemm_mma(
        QF_Q6K, Dtype::FP16, X, W_q6k, nullptr, bias, Y, B, M, K, (K / 256) * 210);
}

}  // namespace q6k_wmma_internal

namespace q8_0_wmma_internal {

bool launch_linear_q8_0_fp16_wmma(const Fp16* X, const uint8_t* W_q8,
                                  const Fp16* bias, Fp16* Y,
                                  int B, int M, int K, CUstream /*stream*/) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % 32 != 0) return false;
    if (B < 4) return false;
    g_q8_0_mma_calls.fetch_add(1ull, std::memory_order_relaxed);
    return launch_quant_gemm_mma(
        QF_Q8_0, Dtype::FP16, X, W_q8, nullptr, bias, Y, B, M, K, (K / 32) * 34);
}

}  // namespace q8_0_wmma_internal

namespace linear_int8w_wmma_internal {

bool launch_linear_int8w_fp16_wmma(
        const Fp16* X, const int8_t* W, const float* scales,
        const Fp16* bias, Fp16* Y,
        int B, int M, int K) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if ((K & 7) != 0) return false;
    if (K < 32) return false;
    if (static_cast<std::size_t>(B) * static_cast<std::size_t>(M) < 1024u) return false;
    return launch_quant_gemm_mma(
        QF_INT8, Dtype::FP16, X, reinterpret_cast<const uint8_t*>(W), scales, bias, Y, B, M, K, K);
}

bool launch_linear_int8w_bf16_wmma(
        const Bf16* X, const int8_t* W, const float* scales,
        const Bf16* bias, Bf16* Y,
        int B, int M, int K) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if ((K & 7) != 0) return false;
    if (K < 32) return false;
    if (static_cast<std::size_t>(B) * static_cast<std::size_t>(M) < 1024u) return false;
    return launch_quant_gemm_mma(
        QF_INT8, Dtype::BF16, X, reinterpret_cast<const uint8_t*>(W), scales, bias, Y, B, M, K, K);
}

}  // namespace linear_int8w_wmma_internal

namespace conv2d_int8w_wmma_internal {

bool launch_conv2d_int8w_implicit_gemm_wmma(
        const Fp16* X, const int8_t* W_int8, const float* scales,
        const Fp16* bias, Fp16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    if (dil_h != 1 || dil_w != 1) return false;
    const int M = N * H_out * W_out;
    if (M <= 0 || C_out <= 0 || C_in <= 0) return false;
    if (static_cast<std::size_t>(M) * static_cast<std::size_t>(C_out) < 1024u) return false;

    std::uint32_t cfg_id = 99u;
    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1) cfg_id = 0u;
    else if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) cfg_id = 1u;
    else if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 2 && stride_w == 2) cfg_id = 2u;
    else return false;

    const int HW_out = H_out * W_out;
    const int K_tot = C_in * kH * kW;
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (bias != nullptr ? 1u : 0u) | ((K_tot & 7) == 0 ? 2u : 0u));
    CUfunction fn = mir_function(
        {MirKind::Conv2dInt8wMma, cfg_id, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = static_cast<std::uint32_t>((C_out + 63) / 64);
    const std::uint32_t gy = static_cast<std::uint32_t>((M + 63) / 64);
    const std::uintptr_t ptrs[] = {addr(X), addr(W_int8), addr(scales), addr(bias), addr(Y)};
    const std::int32_t ints[] = {N, C_in, H, W, C_out, H_out, W_out, M, HW_out, K_tot};
    mir_dispatch_args_3d(fn, gx, gy, 1u, 128u, ptrs, ints, {});
    return true;
}

}  // namespace conv2d_int8w_wmma_internal

}  // namespace brotensor::detail::cuda

extern "C" unsigned long long brotensor_q4k_wmma_calls_consume() {
    return brotensor::detail::cuda::g_q4k_mma_calls.exchange(0ull, std::memory_order_relaxed);
}

extern "C" unsigned long long brotensor_q6k_wmma_calls_consume() {
    return brotensor::detail::cuda::g_q6k_mma_calls.exchange(0ull, std::memory_order_relaxed);
}

extern "C" unsigned long long brotensor_q8_0_wmma_calls_consume() {
    return brotensor::detail::cuda::g_q8_0_mma_calls.exchange(0ull, std::memory_order_relaxed);
}
