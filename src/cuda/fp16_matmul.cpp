#include "fp16_internal.cuh"

#include "detail/cuda_check.h"
#include "gemm_mma.cuh"
#include "ops_gemm_mir.h"
#include "ops_gemm_mma_mir.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace brotensor {
void* cuda_current_stream();
namespace fp16_internal {

namespace {

bool mma_gemm_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("BROTENSOR_MMA_GEMM");
        return !(e && e[0] == '0');
    }();
    return on;
}

int sm_count_and_major(int& major) {
    static int cached_sms[16] = {0}, cached_major[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (cached_sms[dev] == 0) {
        cudaDeviceGetAttribute(&cached_sms[dev], cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&cached_major[dev], cudaDevAttrComputeCapabilityMajor, dev);
        if (cached_sms[dev] <= 0) cached_sms[dev] = 1;
    }
    major = cached_major[dev];
    return cached_sms[dev];
}

template <typename T>
void launch_abt(const T* A, const T* B, T* C,
                int batch, int M, int N, int K,
                std::size_t strideA, std::size_t strideB, std::size_t strideC,
                const T* bias, int act) {
    if (batch <= 0 || M == 0 || N == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    const Dtype dt = std::is_same_v<T, __half> ? Dtype::FP16 : Dtype::BF16;

    if (batch == 1 && static_cast<std::size_t>(M) * static_cast<std::size_t>(N) >= 256 &&
        mma_gemm_enabled() &&
        ::brotensor::detail::cuda::mma_gemm::launch(
            A, B, C, M, N, K, bias, act,
            ::brotensor::detail::cuda::mma_gemm::kStore, nullptr, 0, stream)) {
        return;
    }

    if (K == 0) {
        for (int b = 0; b < batch; ++b) {
            cudaMemsetAsync(const_cast<T*>(C) + static_cast<std::size_t>(b) * strideC, 0,
                            sizeof(T) * static_cast<std::size_t>(M) * N, stream);
        }
        return;
    }

    const auto aligned16 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0u;
    };
    const bool strides_aligned =
        batch == 1 || (((strideA | strideB | strideC) & 7u) == 0u);
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major >= 8 && mma_gemm_enabled() &&
        static_cast<std::size_t>(M) * static_cast<std::size_t>(N) >= 256 &&
        K >= 16 && (K & 7) == 0 && (N & 7) == 0 && strides_aligned &&
        aligned16(A) && aligned16(B) && aligned16(C) &&
        (!bias || (reinterpret_cast<std::uintptr_t>(bias) & 3u) == 0u)) {
        if (::brotensor::detail::cuda::mma_gemm::launch_mir_gemm_mma_batched(
                dt, A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, act, sms)) {
            return;
        }
    }

    ::brotensor::detail::cuda::dispatch_mir_gemm_simt(
        A, dt, B, dt, C, dt, bias, dt,
        batch, M, N, K, K, K, N,
        static_cast<std::int64_t>(strideA),
        static_cast<std::int64_t>(strideB),
        static_cast<std::int64_t>(strideC),
        false, false, 0, act, ::brotensor::Device::cuda(0));
}

}  // namespace

void launch_matmul_ABT_impl(const __half* A, const __half* B, __half* C,
                            int M, int N, int K,
                            const __half* bias, int act) {
    launch_abt<__half>(A, B, C, 1, M, N, K, 0, 0, 0, bias, act);
}

void launch_matmul_ABT_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                            __nv_bfloat16* C,
                            int M, int N, int K,
                            const __nv_bfloat16* bias, int act) {
    launch_abt<__nv_bfloat16>(A, B, C, 1, M, N, K, 0, 0, 0, bias, act);
}

void launch_matmul_ABT_batched_impl(const __half* A, const __half* B, __half* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const __half* bias, int act) {
    launch_abt<__half>(A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, act);
}

void launch_matmul_ABT_batched_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                    __nv_bfloat16* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const __nv_bfloat16* bias, int act) {
    launch_abt<__nv_bfloat16>(A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, act);
}

}  // namespace fp16_internal
}  // namespace brotensor
