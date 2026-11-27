#include "cu_mir_cache.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace brotensor::detail::cuda {

namespace {

constexpr int kBQ = 64;
constexpr int kHD = 64;
constexpr std::uint32_t kSmemBytes = (64 * 64 + 64 * 68 + 64 * 64) * sizeof(float);

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

}  // namespace

bool flash_attention_varlen_f32_hd64(const float* Q, const float* K, const float* V,
                                     const int* cu_q, const int* cu_k, float* O,
                                     int total_q, int batch, int num_heads, int head_dim,
                                     bool causal, cudaStream_t /*stream*/) {
    const int D = num_heads * head_dim;
    if (head_dim != kHD || causal || batch < 1 || num_heads > 65535) return false;
    const auto aligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (!aligned(Q) || !aligned(K) || !aligned(V) || !aligned(O)) return false;

    CUfunction fn = mir_function({MirKind::FaVarlenF32, 0u, Dtype::FP32, Dtype::FP32, 1u});
    const float q_scale = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(kHD)));
    const std::array<std::uintptr_t, 6> ptrs = {
        addr(Q), addr(K), addr(V), addr(cu_q), addr(cu_k), addr(O)};
    const std::array<std::int32_t, 3> ints = {total_q, batch, D};
    const std::array<float, 1> flts = {q_scale};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>((total_q + kBQ - 1) / kBQ),
                         static_cast<std::uint32_t>(num_heads), 1u, 256u, ptrs, ints, flts, kSmemBytes);
    return true;
}

}  // namespace brotensor::detail::cuda
