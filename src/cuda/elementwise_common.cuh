#pragma once

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

constexpr int EW_BLOCK = 256;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

// Fast 32-bit unsigned index division and modulo for tensors with
// total_elems <= INT32_MAX, replacing 20-instruction integer div/rem inside
// inner grid-stride loops with bitwise shifts/masks (for powers of two) or a
// single __umulhi + multiply-subtract pair.
struct FastDivmodU32 {
    std::uint32_t d;
    std::uint32_t mul;
    std::uint32_t pre_shift;
    std::uint32_t post_shift;
    std::uint32_t is_pow2;

    __host__ explicit FastDivmodU32(int div)
        : d(static_cast<std::uint32_t>(div > 0 ? div : 1)),
          mul(1u),
          pre_shift(0u),
          post_shift(0u),
          is_pow2(0u) {
        std::uint32_t u = d;
        if (u <= 1u) {
            is_pow2 = 1u;
            return;
        }
        while ((u & 1u) == 0u) {
            u >>= 1u;
            ++pre_shift;
        }
        if (u == 1u) {
            is_pow2 = 1u;
            return;
        }
        const int prec = (pre_shift > 0u) ? 31 : 32;
        const std::uint64_t two_n_m1 = 1ULL << (prec - 1);
        const std::uint64_t nc = ((two_n_m1 + ((std::uint64_t(u) - 1ULL) & 1ULL)) / u) * u - 1ULL;
        for (int p = 0; p <= 32; ++p) {
            const std::uint64_t pow_p = 1ULL << p;
            if (two_n_m1 * pow_p > nc * (u - 1ULL - (two_n_m1 * pow_p - 1ULL) % u)) {
                const std::uint64_t e = u - 1ULL - (two_n_m1 * pow_p - 1ULL) % u;
                const std::uint64_t m_full = (two_n_m1 * pow_p + e) / u;
                if (m_full < (1ULL << 32)) {
                    mul = static_cast<std::uint32_t>(m_full);
                    post_shift = static_cast<std::uint32_t>(p);
                } else {
                    mul = 0u;
                }
                return;
            }
        }
    }

    __device__ __forceinline__ void divmod(std::uint32_t idx, std::uint32_t& q,
                                           std::uint32_t& r) const {
        if (is_pow2) {
            q = idx >> pre_shift;
            r = idx & (d - 1u);
        } else if (mul != 0u) {
            q = __umulhi(idx >> pre_shift, mul) >> post_shift;
            r = idx - q * d;
        } else {
            q = idx / d;
            r = idx - q * d;
        }
    }

    __device__ __forceinline__ std::uint32_t div(std::uint32_t idx) const {
        if (is_pow2) return idx >> pre_shift;
        if (mul != 0u) return __umulhi(idx >> pre_shift, mul) >> post_shift;
        return idx / d;
    }

    __device__ __forceinline__ std::uint32_t mod(std::uint32_t idx) const {
        std::uint32_t q = 0u, r = 0u;
        divmod(idx, q, r);
        return r;
    }
};

template <typename T>
__device__ __forceinline__ int vec_head(const T* p, int n) {
    const unsigned off = static_cast<unsigned>(reinterpret_cast<uintptr_t>(p) & 15u);
    const int head = off ? static_cast<int>((16u - off) / sizeof(T)) : 0;
    return head < n ? head : n;
}

template <typename T, typename Cvt, typename Op>
__device__ __forceinline__ void vec_apply_inplace(T* __restrict__ y, int n, Op op) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int head = vec_head(y, n);
    const int nvec = (n - head) >> 3;
    const int tail0 = head + (nvec << 3);

    int4* v4 = reinterpret_cast<int4*>(y + head);
    for (int i = tid; i < nvec; i += stride) {
        int4 v = v4[i];
        auto* h = reinterpret_cast<typename Cvt::pair*>(&v);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            h[k] = Cvt::pack(op(Cvt::lo(h[k])), op(Cvt::hi(h[k])));
        v4[i] = v;
    }
    for (int i = tid; i < head; i += stride)
        y[i] = Cvt::from_f32(op(Cvt::to_f32(y[i])));
    for (int i = tail0 + tid; i < n; i += stride)
        y[i] = Cvt::from_f32(op(Cvt::to_f32(y[i])));
}

struct HalfCvt {
    using pair = __half2;
    __device__ static float lo(pair v) { return __low2float(v); }
    __device__ static float hi(pair v) { return __high2float(v); }
    __device__ static pair pack(float a, float b) { return __floats2half2_rn(a, b); }
    __device__ static float to_f32(__half v) { return __half2float(v); }
    __device__ static __half from_f32(float v) { return __float2half(v); }
};

struct Bf16Cvt {
    using pair = __nv_bfloat162;
    __device__ static float lo(pair v) { return __low2float(v); }
    __device__ static float hi(pair v) { return __high2float(v); }
    __device__ static pair pack(float a, float b) { return __floats2bfloat162_rn(a, b); }
    __device__ static float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
};

template <typename T, typename Cvt, typename Op>
__device__ __forceinline__ void vec_apply_inplace2(T* __restrict__ y,
                                                   const T* __restrict__ x,
                                                   int n, Op op) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const bool congruent = ((reinterpret_cast<uintptr_t>(y) ^
                             reinterpret_cast<uintptr_t>(x)) & 15u) == 0;
    const int head = congruent ? vec_head(y, n) : n;
    const int nvec = (n - head) >> 3;
    const int tail0 = head + (nvec << 3);

    int4* v4 = reinterpret_cast<int4*>(y + head);
    const int4* u4 = reinterpret_cast<const int4*>(x + head);
    for (int i = tid; i < nvec; i += stride) {
        int4 v = v4[i];
        int4 u = u4[i];
        auto* h = reinterpret_cast<typename Cvt::pair*>(&v);
        auto* g = reinterpret_cast<typename Cvt::pair*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            h[k] = Cvt::pack(op(Cvt::lo(h[k]), Cvt::lo(g[k])),
                             op(Cvt::hi(h[k]), Cvt::hi(g[k])));
        v4[i] = v;
    }
    for (int i = tid; i < head; i += stride)
        y[i] = Cvt::from_f32(op(Cvt::to_f32(y[i]), Cvt::to_f32(x[i])));
    for (int i = tail0 + tid; i < n; i += stride)
        y[i] = Cvt::from_f32(op(Cvt::to_f32(y[i]), Cvt::to_f32(x[i])));
}

} // namespace brotensor::detail::cuda
