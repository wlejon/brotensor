#include "gemm_mma.h"

#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "ops_gemm_mma_mir.h"

#include <brotensor/tensor.h>

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
    int dev = current_device();
    if (dev < 0 || dev >= 16) dev = 0;
    if (cached_sms[dev] == 0) {
        cached_sms[dev] = device_attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        cached_major[dev] = device_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        if (cached_sms[dev] <= 0) cached_sms[dev] = 1;
    }
    major = cached_major[dev];
    return cached_sms[dev];
}

struct ScopedStream {
    void* prev = nullptr;
    bool swapped = false;
    explicit ScopedStream(CUstream stream) {
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
              float* ws, std::size_t ws_floats, CUstream stream, int acc_mode) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if ((K & 7) || (N & 7)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) != 0u; };
    if (misaligned(A) || misaligned(W) || misaligned(C) || misaligned(ws)) return false;
    if (bias && (reinterpret_cast<std::uintptr_t>(bias) & 3u) != 0u) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    const Dtype dt = std::is_same_v<T, Fp16> ? Dtype::FP16 : Dtype::BF16;
    const bool fast_acc = (acc_mode == kAccHybrid) && (dt == Dtype::FP16);
    ScopedStream guard(stream);
    launch_mir_gemm_mma(dt, A, W, C, M, N, K, bias, act, epi, ws, ws_floats, sms, fast_acc);
    return true;
}

template <typename T>
bool launch_f32out_t(const T* A, const T* W, float* C, int M, int N, int K, CUstream stream) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if ((K & 7) || (N & 7)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) != 0u; };
    if (misaligned(A) || misaligned(W) || misaligned(C)) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    const Dtype dt = std::is_same_v<T, Fp16> ? Dtype::FP16 : Dtype::BF16;
    ScopedStream guard(stream);
    launch_mir_gemm_mma_f32out(dt, A, W, C, M, N, K, sms);
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

bool launch(const Fp16* A, const Fp16* W, Fp16* C, int M, int N, int K, const Fp16* bias, int act,
            int epi, float* ws, std::size_t ws_floats, CUstream stream, int acc_mode) {
    return launch_t<Fp16>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream, acc_mode);
}

bool launch(const Bf16* A, const Bf16* W, Bf16* C, int M, int N, int K,
            const Bf16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            CUstream stream, int acc_mode) {
    return launch_t<Bf16>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream, acc_mode);
}

bool launch_f32out(const Fp16* A, const Fp16* W, float* C, int M, int N, int K, CUstream stream) {
    return launch_f32out_t<Fp16>(A, W, C, M, N, K, stream);
}

bool launch_f32out(const Bf16* A, const Bf16* W, float* C, int M, int N, int K,
                   CUstream stream) {
    return launch_f32out_t<Bf16>(A, W, C, M, N, K, stream);
}

}  // namespace brotensor::detail::cuda::mma_gemm
