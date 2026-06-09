// CUDA port of src/cpu/ops_impl.cpp::xavier_init.
//
// The CPU impl walks splitmix64 sequentially: each element i consumes one
// splitmix step. The accumulator update `s += K` is associative under
// repeated application, so element i's input state is `rng_state + (i+1)*K`
// — which we can compute independently per thread in parallel. After
// dispatching n elements the host advances `rng_state` by `n*K` to match the
// CPU's final state byte-for-byte.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr uint64_t SPLITMIX_K = 0x9E3779B97F4A7C15ULL;

__device__ inline uint64_t splitmix_from(uint64_t s_in) {
    // Mirror src/cpu/ops_impl.cpp::splitmix exactly. The CPU adds K to the
    // running state *before* mixing; we receive the already-bumped state.
    uint64_t z = s_in;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__global__ void xavier_init_kernel(float* __restrict__ W,
                                   int n, float limit,
                                   uint64_t base_state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Element i: state = base_state + (i+1)*K, matching CPU sequential order.
    const uint64_t s = base_state + static_cast<uint64_t>(i + 1) * SPLITMIX_K;
    const uint64_t z = splitmix_from(s);
    // top 24 bits → [0, 1), then map to [-limit, +limit].
    const float u = static_cast<float>(z >> 40) / 16777216.0f;
    W[i] = (u * 2.0f - 1.0f) * limit;
}

} // namespace

void xavier_init(::brotensor::Tensor& W, uint64_t& rng_state) {
    const int n = W.size();
    if (n <= 0) return;
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    const uint64_t base = rng_state;

    constexpr int BLOCK = 256;
    const int grid = (n + BLOCK - 1) / BLOCK;
    xavier_init_kernel<<<grid, BLOCK, 0, cur_stream()>>>(static_cast<float*>(W.data),
                                        n, limit, base);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Advance rng_state by n splitmix steps to match the CPU's final state.
    rng_state = base + static_cast<uint64_t>(n) * SPLITMIX_K;
}

void fill_cuda_vtable_xavier_init(::brotensor::detail::OpsVTable& v) {
    v.xavier_init = &xavier_init;
}

} // namespace brotensor::detail::cuda
