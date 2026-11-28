#include "gemm_skinny.h"

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "ops_gemm_mma_mir.h"
#include "../kernels/mir_gemm_skinny.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace brotensor {
void* cuda_current_stream();
namespace detail::cuda {
void cuda_set_stream(void* stream);
}
}  // namespace brotensor

namespace brotensor::detail::cuda::gemm_skinny {

namespace {

constexpr int kCtaCols = 128;
constexpr int kChunk = 64;
constexpr int kThreads = 256;

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

int col_blocks(int N, int epi) {
    return (epi == 3) ? ((N / 2 + kCtaCols / 2 - 1) / (kCtaCols / 2))
                      : ((N + kCtaCols - 1) / kCtaCols);
}

int chunks_per_split(int N, int K, int epi, int sms) {
    const int n_chunks = (K + kChunk - 1) / kChunk;
    const int blocks = col_blocks(N, epi);
    const int splits = std::clamp(sms / std::max(1, blocks), 1, std::max(1, n_chunks / 4));
    return (n_chunks + splits - 1) / splits;
}

int split_count(int N, int K, int epi, int sms) {
    const int n_chunks = (K + kChunk - 1) / kChunk;
    const int cps = chunks_per_split(N, K, epi, sms);
    return (n_chunks + cps - 1) / cps;
}

bool shape_ok(int M, int N, int K, int epi) {
    if (M < 9 || M > 64 || N <= 0 || K <= 0 || (K & 7)) return false;
    if (epi < 0 || epi > 3 || (epi >= 2 && (N & 1))) return false;
    return true;
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
bool launch_t(const T* X, const T* W, T* Y, int M, int N, int K, const T* bias, int act, int epi,
              float* ws, std::size_t ws_floats, CUstream stream) {
    if (!shape_ok(M, N, K, epi)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) != 0u; };
    if (misaligned(X) || misaligned(W)) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    int splits = split_count(N, K, epi, sms);
    int cps = chunks_per_split(N, K, epi, sms);
    if (splits > 1 && (!ws || ws_floats < static_cast<std::size_t>(splits) * M * N)) {
        splits = 1;
        cps = (K + kChunk - 1) / kChunk;
    }
    float* partial = (splits > 1) ? ws : nullptr;
    const std::uint8_t nt = static_cast<std::uint8_t>(std::clamp((M + 7) / 8, 2, 8));
    const Dtype dt = std::is_same_v<T, Fp16> ? Dtype::FP16 : Dtype::BF16;
    const bool has_bias = (bias != nullptr);
    const bool has_partial = (partial != nullptr);

    const std::uint32_t op = static_cast<std::uint32_t>(nt) |
                             (static_cast<std::uint32_t>(epi & 0x0f) << 4) |
                             (static_cast<std::uint32_t>(act & 0x0f) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (has_bias ? 1u : 0u) | (has_partial ? 2u : 0u));

    ScopedStream guard(stream);
    CUfunction fn = mir_function({MirKind::GemmSkinny, op, dt, dt, 1u, dt, flags});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(X),
        reinterpret_cast<std::uintptr_t>(W),
        reinterpret_cast<std::uintptr_t>(Y),
        has_bias ? reinterpret_cast<std::uintptr_t>(bias) : 0u,
        has_partial ? reinterpret_cast<std::uintptr_t>(partial) : 0u,
    };
    const std::int32_t ints[] = {M, N, K, cps};
    const std::uint32_t smem = kernels::gemm_skinny_smem_bytes(nt);
    mir_dispatch_args_3d(
        fn, static_cast<std::uint32_t>(col_blocks(N, epi)),
        static_cast<std::uint32_t>(splits), 1u,
        kThreads, ptrs, ints, {}, smem);

    if (partial) {
        mma_gemm::launch_mir_splitk_reduce(dt, partial, splits, M, N, Y, bias, act, epi);
    }
    return true;
}

}  // namespace

std::size_t workspace_floats(int M, int N, int K, int epi) {
    if (!shape_ok(M, N, K, epi)) return 0;
    int major = 0;
    const int splits = split_count(N, K, epi, sm_count_and_major(major));
    return splits > 1 ? static_cast<std::size_t>(splits) * M * N : 0;
}

bool launch(const Fp16* X, const Fp16* W, Fp16* Y, int M, int N, int K, const Fp16* bias, int act, int epi,
            float* ws, std::size_t ws_floats, CUstream stream) {
    return launch_t<Fp16>(X, W, Y, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

bool launch(const Bf16* X, const Bf16* W, Bf16* Y, int M, int N, int K,
            const Bf16* bias, int act, int epi, float* ws, std::size_t ws_floats, CUstream stream) {
    return launch_t<Bf16>(X, W, Y, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

}  // namespace brotensor::detail::cuda::gemm_skinny
