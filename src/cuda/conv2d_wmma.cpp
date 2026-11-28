// Pure C++ dispatcher for Brass MIR Tensor Core implicit-GEMM Conv2D (FP16 / BF16),
// replacing src/cuda/conv2d_wmma.cu.

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "../kernels/mir_conv_mma.h"

#include <cstdint>

namespace brotensor::conv2d_wmma_internal {

using ::brotensor::detail::cuda::Bf16;
using ::brotensor::detail::cuda::Fp16;

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

bool launch_conv2d_mma_impl(
        ::brotensor::Dtype dt,
        const void* X, const void* Wt, const void* bias, void* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    using namespace ::brotensor::detail::cuda;
    if (dil_h != 1 || dil_w != 1) return false;
    const int M = N * H_out * W_out;
    if (M <= 0 || C_out <= 0 || C_in <= 0) return false;
    if (static_cast<std::size_t>(M) * static_cast<std::size_t>(C_out) < 1024u) return false;
    if (C_out < 16) return false;
    const long long K = static_cast<long long>(C_in) * kH * kW;
    const int ctas_64 = ((C_out + 63) / 64) * ((M + 63) / 64);
    if (ctas_64 < 40 && K >= 1024) return false;

    std::uint8_t shape_id = 99u;
    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1) shape_id = 0u;
    else if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) shape_id = 1u;
    else if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 2 && stride_w == 2) shape_id = 2u;
    else if (kH == 3 && kW == 3 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) shape_id = 3u;
    else if (kH == 7 && kW == 7 && pad_h == 3 && pad_w == 3 && stride_h == 1 && stride_w == 1) shape_id = 4u;
    else if (kH == 7 && kW == 7 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) shape_id = 5u;
    else if (kH == 5 && kW == 5 && pad_h == 2 && pad_w == 2 && stride_h == 1 && stride_w == 1) shape_id = 6u;
    else return false;

    const int ctas_128 = ((C_out + 63) / 64) * ((M + 127) / 128);
    const std::uint8_t cfg = (M >= 128 && ctas_128 >= 256) ? 1u : 0u;
    const auto& tc = kernels::kConv2dMmaCfgs[cfg];
    const std::uint32_t gx = static_cast<std::uint32_t>((C_out + tc.bn - 1u) / tc.bn);
    const std::uint32_t gy = static_cast<std::uint32_t>((M + tc.bm - 1u) / tc.bm);
    const int HW_out = H_out * W_out;
    const int K_tot = static_cast<int>(K);
    const bool aligned_k8 = (K_tot & 7) == 0;

    const std::uint32_t op = static_cast<std::uint32_t>(shape_id & 0x0fu) |
                             (static_cast<std::uint32_t>(cfg & 0x0fu) << 4);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (bias != nullptr ? 1u : 0u) | (aligned_k8 ? 2u : 0u));

    CUfunction fn = mir_function(
        {MirKind::Conv2dMma, op, dt, ::brotensor::Dtype::FP32, 1, ::brotensor::Dtype::FP32, flags});
    const std::uintptr_t ptrs[] = {addr(X), addr(Wt), addr(bias), addr(Y)};
    const std::int32_t ints[] = {N, C_in, H, W, C_out, H_out, W_out, M, HW_out, K_tot};
    mir_dispatch_args_3d(fn, gx, gy, 1u, tc.threads, ptrs, ints, {});
    return true;
}

}  // namespace

bool launch_conv2d_implicit_gemm_wmma(
        const Fp16* X, const Fp16* Wt, const Fp16* bias, Fp16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    return launch_conv2d_mma_impl(
        ::brotensor::Dtype::FP16, X, Wt, bias, Y,
        N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out);
}

bool launch_conv2d_implicit_gemm_wmma_bf16(
        const Bf16* X, const Bf16* Wt,
        const Bf16* bias, Bf16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    return launch_conv2d_mma_impl(
        ::brotensor::Dtype::BF16, X, Wt, bias, Y,
        N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out);
}

}  // namespace brotensor::conv2d_wmma_internal
