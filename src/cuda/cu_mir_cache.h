#pragma once

// CUDA runtime JIT cache for backend-neutral brass MIR elementwise kernels.

#include <brotensor/tensor.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>

namespace brotensor::detail::cuda {

enum class MirKind : std::uint8_t {
    Unary = 0,
    Binary = 1,
    ActBwd = 2,
    Bias = 3,
    Cast = 4,
    Glu = 5,
};

struct MirKey {
    MirKind kind = MirKind::Unary;
    std::uint32_t op = 0;
    Dtype dt0 = Dtype::FP32;
    Dtype dt1 = Dtype::FP32;
    std::uint8_t lanes = 1;

    bool operator==(const MirKey& o) const noexcept = default;
};

// Lowers, compiles (on first use per key), and caches CUfunction via CudaJitEngine.
CUfunction mir_function(const MirKey& key);

// Chooses 128-bit (lanes=4 f32 / lanes=8 f16,bf16), 64-bit (lanes=4 f16,bf16),
// or scalar (lanes=1) based on element count and pointer alignment.
int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uintptr_t a0, std::uintptr_t a1 = 0, std::uintptr_t a2 = 0);
int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n, std::uintptr_t a0, std::uintptr_t a1);

inline std::uint32_t mir_groups_1d(std::uint32_t n_elems, int lanes, std::uint32_t block_x = 256) {
    const std::uint64_t nv = static_cast<std::uint64_t>(n_elems) / static_cast<std::uint64_t>(lanes);
    const std::uint64_t g = (nv + block_x - 1) / block_x;
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(g, 65535)));
}

void mir_dispatch_5(CUfunction fn, std::uint32_t gx,
                    const void* p0, const void* p1, std::uint32_t p2, float p3, float p4);
void mir_dispatch_act_bwd(CUfunction fn, std::uint32_t gx,
                          const void* p, const void* dy, const void* dx, std::uint32_t n, float a);
void mir_dispatch_bias(CUfunction fn, std::uint32_t gx,
                       const void* y, const void* bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l);
void mir_dispatch_cast(CUfunction fn, std::uint32_t gx,
                       const void* src, const void* dst, std::uint32_t n);
void mir_dispatch_glu(CUfunction fn, std::uint32_t gx,
                      const void* x, const void* dy, const void* y, std::uint32_t n, std::uint32_t dim);

}  // namespace brotensor::detail::cuda
