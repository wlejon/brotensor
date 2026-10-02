#pragma once

// Internal header — the pieces the native HIP attention kernels share
// (flash_attention.hip, flash_attention_dense.hip, flash_attention_extra.hip):
// block geometry, the head_dim chunking rule, Wave32 reductions and the
// per-dtype scalar load/convert/dot helpers.
//
// head_dim chunking. Every scalar kernel keeps a register accumulator of
// kMaxHdPerThread output columns per thread, so one block covers at most
// kFaHdChunk = kFaBlock * kMaxHdPerThread columns of a head. Wider heads
// launch fa_hd_chunks(head_dim) blocks along grid.z, block z owning columns
// [z * kFaHdChunk, min(head_dim, (z + 1) * kFaHdChunk)). Each of those blocks
// recomputes the full-width Q·K scores (the dot product needs every column)
// and so arrives at bit-identical softmax statistics; only the P·V half is
// split. head_dim <= kFaHdChunk is one z-block, i.e. exactly the old launch.

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace fa {

constexpr int kFaBlock        = 128; // 4 Wave32 waves per block
constexpr int kMaxHdPerThread = 8;
constexpr int kFaHdChunk      = kFaBlock * kMaxHdPerThread; // columns per z-block

inline int fa_hd_chunks(int head_dim) {
    return head_dim <= 0 ? 1 : (head_dim + kFaHdChunk - 1) / kFaHdChunk;
}

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

// Largest dynamic LDS allocation one block may take on `dev` (64 KB on RDNA 3.5).
size_t max_lds_per_block(int dev);

// The reductions below are 32-lane shuffles and the kernels index `red[warp]`
// by tid >> 5: they assume Wave32, which runtime.hip enforces per device.
__device__ __forceinline__ float wave32_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, 32));
    }
    return val;
}

__device__ __forceinline__ float wave32_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 32);
    }
    return val;
}

template <typename T> struct FaScalar;

template <> struct FaScalar<float> {
    __device__ static __forceinline__ float to_f32(float v) { return v; }
    __device__ static __forceinline__ float from_f32(float v) { return v; }
    __device__ static __forceinline__ float dot(const float* a, const float* b, int hd) {
        float acc = 0.0f;
        int d = 0;
        if (hd % 4 == 0) {
            const float4* a4 = reinterpret_cast<const float4*>(a);
            const float4* b4 = reinterpret_cast<const float4*>(b);
            const int n4 = hd / 4;
            for (int i = 0; i < n4; ++i) {
                float4 va = a4[i];
                float4 vb = b4[i];
                acc += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
            }
            d = n4 * 4;
        }
        for (; d < hd; ++d) acc += a[d] * b[d];
        return acc;
    }
};

template <> struct FaScalar<__half> {
    __device__ static __forceinline__ float to_f32(__half v) { return __half2float(v); }
    __device__ static __forceinline__ __half from_f32(float v) { return __float2half(v); }
    __device__ static __forceinline__ float dot(const __half* a, const __half* b, int hd) {
        float acc = 0.0f;
        int d = 0;
        if (hd % 8 == 0) {
            // 128-bit vectorized load (8 x 16-bit half)
            const uint4* a4 = reinterpret_cast<const uint4*>(a);
            const uint4* b4 = reinterpret_cast<const uint4*>(b);
            const int n8 = hd / 8;
            for (int i = 0; i < n8; ++i) {
                uint4 ra = a4[i];
                uint4 rb = b4[i];
                const __half* ha = reinterpret_cast<const __half*>(&ra);
                const __half* hb = reinterpret_cast<const __half*>(&rb);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc += __half2float(ha[j]) * __half2float(hb[j]);
                }
            }
            d = n8 * 8;
        }
        for (; d < hd; ++d) acc += __half2float(a[d]) * __half2float(b[d]);
        return acc;
    }
};

template <> struct FaScalar<hip_bfloat16> {
    __device__ static __forceinline__ float to_f32(hip_bfloat16 v) { return float(v); }
    __device__ static __forceinline__ hip_bfloat16 from_f32(float v) { return hip_bfloat16(v); }
    __device__ static __forceinline__ float dot(const hip_bfloat16* a, const hip_bfloat16* b, int hd) {
        float acc = 0.0f;
        int d = 0;
        if (hd % 8 == 0) {
            // 128-bit vectorized load (8 x 16-bit bfloat16)
            const uint4* a4 = reinterpret_cast<const uint4*>(a);
            const uint4* b4 = reinterpret_cast<const uint4*>(b);
            const int n8 = hd / 8;
            for (int i = 0; i < n8; ++i) {
                uint4 ra = a4[i];
                uint4 rb = b4[i];
                const hip_bfloat16* ha = reinterpret_cast<const hip_bfloat16*>(&ra);
                const hip_bfloat16* hb = reinterpret_cast<const hip_bfloat16*>(&rb);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc += float(ha[j]) * float(hb[j]);
                }
            }
            d = n8 * 8;
        }
        for (; d < hd; ++d) acc += float(a[d]) * float(b[d]);
        return acc;
    }
};

} // namespace fa

// Non-causal, unwindowed attention as two hipBLAS GEMMs around an FP32-score
// softmax (flash_attention_dense.hip). Any head_dim, optional key mask, GQA
// (num_heads a multiple of n_kv). Q: (Lq, num_heads*head_dim), K/V:
// (Lk, n_kv*head_dim), O pre-sized (Lq, num_heads*head_dim), all one dtype.
void hip_dense_attention(const ::brotensor::Tensor& Q,
                         const ::brotensor::Tensor& K,
                         const ::brotensor::Tensor& V,
                         const float* d_mask,
                         int num_heads, int n_kv, int head_dim,
                         ::brotensor::Tensor& O);

} // namespace brotensor::detail::hip
