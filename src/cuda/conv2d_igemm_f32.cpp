// Pure C++ Brass MIR dispatcher for FP32 implicit-GEMM conv2d and conv_transpose2d,
// replacing src/cuda/conv2d_igemm_f32.cu.

#include "cu_mir_cache.h"
#include "cuda_driver.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace brotensor::detail::cuda {

void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::uint32_t grid1d_blocks(long long n, int block) {
    long long b = (n + block - 1) / block;
    if (b > 65535LL * 32) b = 65535LL * 32;
    return static_cast<std::uint32_t>(b < 1 ? 1 : b);
}

}  // namespace

bool conv2d_igemm_f32(const float* X, const float* Wt, const float* bias, float* Y, int N,
                      int Cin, int H, int W, int Cout, int kH, int kW, int sh, int sw, int ph,
                      int pw, int dh, int dw, int Ho, int Wo, CUstream /*stream*/) {
    const long long M = static_cast<long long>(N) * Ho * Wo;
    const long long K = static_cast<long long>(Cin) * kH * kW;
    if (M <= 0 || K <= 0 || M > 0x7fffffffLL || K > 0x7fffffffLL) return false;
    if (static_cast<long long>(N) * Cin * H * W > 0x7fffffffLL * 4LL) return false;
    const int HWo = Ho * Wo;
    const bool vec_store = (HWo % 4 == 0) && (addr(Y) % 16u == 0u);
    const long long gx = (M + 127) / 128;
    const long long gy = (Cout + 63) / 64;
    if (gx > 0x7fffffffLL || gy > 65535) return false;

    const std::uint8_t flags = static_cast<std::uint8_t>(
        (bias != nullptr ? 1u : 0u) | (vec_store ? 2u : 0u));
    CUfunction fn = mir_function(
        {MirKind::Conv2dIgemmF32, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uintptr_t ptrs[] = {addr(X), addr(Wt), addr(bias), addr(Y)};
    const std::int32_t ints[] = {
        Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, Ho, Wo,
        static_cast<int>(M), static_cast<int>(K), kH * kW, HWo};
    mir_dispatch_args_3d(
        fn, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy), 1u, 256u, ptrs, ints, {});
    return true;
}

bool conv_transpose2d_f32_gemm(const float* X, const float* Wt, const float* bias, float* Y,
                               int N, int Cin, int H, int W, int Cout, int kH, int kW, int sh,
                               int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                               CUstream stream) {
    const long long R = static_cast<long long>(Cout) * kH * kW;
    if (R > 0x7fffffffLL || static_cast<long long>(N) * R * H * W > 0x7fffffffLL * 4LL) return false;
    float* wT = static_cast<float*>(cuda_alloc(static_cast<std::size_t>(R) * Cin * sizeof(float)));
    float* cols = static_cast<float*>(cuda_alloc(static_cast<std::size_t>(N) * R * H * W * sizeof(float)));

    {
        const std::uint32_t r = static_cast<std::uint32_t>(Cin);
        const std::uint32_t c = static_cast<std::uint32_t>(R);
        const std::uint32_t gx = (std::max)((c + 31u) / 32u, 1u);
        const std::uint32_t gy = std::clamp((r + 31u) / 32u, 1u, 65535u);
        CUfunction tfn = mir_function({MirKind::Transpose, 4u});
        mir_dispatch_transpose(tfn, gx, gy, 1u, Wt, wT, 1u, r, c);
    }

    const bool ok = conv2d_igemm_f32(X, wT, nullptr, cols, N, Cin, H, W, static_cast<int>(R), 1, 1,
                                     1, 1, 0, 0, 1, 1, H, W, stream);
    if (ok) {
        const long long total = static_cast<long long>(N) * Cout * Ho * Wo;
        CUfunction cfn = mir_function(
            {MirKind::ConvAux, 4u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32,
             static_cast<std::uint8_t>(bias != nullptr ? 1u : 0u)});
        const std::uintptr_t ptrs[] = {addr(cols), addr(bias), addr(Y)};
        const std::int32_t ints[] = {N, Cout, H, W, kH, kW, sh, sw, ph, pw, dh, dw, Ho, Wo};
        mir_dispatch_args_3d(cfn, grid1d_blocks(total, 256), 1u, 1u, 256u, ptrs, ints, {});
    }
    cuda_free(cols);
    cuda_free(wT);
    return ok;
}

}  // namespace brotensor::detail::cuda
