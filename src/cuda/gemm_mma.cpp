#include "gemm_mma.cuh"

#include "detail/cuda_check.h"
#include "ops_gemm_mma_mir.h"

#include <brotensor/tensor.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace brotensor {
void* cuda_current_stream();
namespace detail::cuda {
void cuda_set_stream(void* stream);
}
}  // namespace brotensor

namespace brotensor::detail::cuda::mma_gemm {

namespace {

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

struct ScopedStream {
    void* prev = nullptr;
    bool swapped = false;
    explicit ScopedStream(cudaStream_t stream) {
        prev = ::brotensor::cuda_current_stream();
        if (reinterpret_cast<void*>(stream) != prev) {
            ::brotensor::detail::cuda::cuda_set_stream(reinterpret_cast<void*>(stream));
            swapped = true;
        }
    }
    ~ScopedStream() {
        if (swapped) ::brotensor::detail::cuda::cuda_set_stream(prev);
    }
};

template <typename T>
bool launch_t(const T* A, const T* W, T* C, int M, int N, int K, const T* bias, int act, int epi,
              float* ws, std::size_t ws_floats, cudaStream_t stream, int acc_mode) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if ((K & 7) || (N & 7)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) != 0u; };
    if (misaligned(A) || misaligned(W) || misaligned(C) || misaligned(ws)) return false;
    if (bias && (reinterpret_cast<std::uintptr_t>(bias) & 3u) != 0u) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    const Dtype dt = std::is_same_v<T, __half> ? Dtype::FP16 : Dtype::BF16;
    const bool fast_acc = (acc_mode == kAccHybrid) && (dt == Dtype::FP16);
    ScopedStream guard(stream);
    launch_mir_gemm_mma(dt, A, W, C, M, N, K, bias, act, epi, ws, ws_floats, sms, fast_acc);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_f32out_t(const T* A, const T* W, float* C, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if ((K & 7) || (N & 7)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) != 0u; };
    if (misaligned(A) || misaligned(W) || misaligned(C)) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    const Dtype dt = std::is_same_v<T, __half> ? Dtype::FP16 : Dtype::BF16;
    ScopedStream guard(stream);
    launch_mir_gemm_mma_f32out(dt, A, W, C, M, N, K, sms);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace

std::size_t workspace_floats(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    int major = 0;
    const int sms = sm_count_and_major(major);
    const MirSplitPlan p = plan_mir_gemm_mma(M, N, K, sms, true);
    return p.splits > 1 ? static_cast<std::size_t>(p.splits) * M * N : 0;
}

bool launch(const __half* A, const __half* W, __half* C, int M, int N, int K, const __half* bias, int act,
            int epi, float* ws, std::size_t ws_floats, cudaStream_t stream, int acc_mode) {
    return launch_t<__half>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream, acc_mode);
}

bool launch(const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* C, int M, int N, int K,
            const __nv_bfloat16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            cudaStream_t stream, int acc_mode) {
    return launch_t<__nv_bfloat16>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream, acc_mode);
}

bool launch_f32out(const __half* A, const __half* W, float* C, int M, int N, int K, cudaStream_t stream) {
    return launch_f32out_t<__half>(A, W, C, M, N, K, stream);
}

bool launch_f32out(const __nv_bfloat16* A, const __nv_bfloat16* W, float* C, int M, int N, int K,
                   cudaStream_t stream) {
    return launch_f32out_t<__nv_bfloat16>(A, W, C, M, N, K, stream);
}

}  // namespace brotensor::detail::cuda::mma_gemm
