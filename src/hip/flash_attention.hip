// Native AMD HIP FlashAttention-2 forward for RDNA 3.5 (gfx1151, Wave32, 64 KB LDS).
//
// Designed specifically for AMD RDNA 3.5:
// - Wave32 execution: 32-lane wavefronts, cross-lane shuffles (__shfl_xor / __shfl_down
//   with 0xffffffff mask) for in-warp max/sum reductions without shared-memory stalls.
// - 64 KB LDS budget: the K/V tile depth is picked per launch (fa_pick_ktile) so the
//   staged Q row + K tile + V tile fit; heads too wide to stage read K/V from global.
// - Unified support for causal, bidirectional, GQA, windowed, and varlen sequences,
//   at any head_dim (columns split across grid.z, see detail/flash_internal.h).
// - Non-causal unwindowed attention of any width takes the hipBLAS path
//   (flash_attention_dense.hip) when it pays for itself.

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/flash_internal.h"
#include "detail/hip_check.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace brotensor::detail::hip {

void hip_linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     ::brotensor::Tensor& Y_BD);

// Implemented in flash_attention_extra.hip
void hip_flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV,
                                            const ::brotensor::Tensor& seq_bounds,
                                            int num_heads, int window,
                                            ::brotensor::Tensor& O);
void hip_flash_attention_decode(const ::brotensor::Tensor& Q,
                                const ::brotensor::Tensor& K_cache,
                                const ::brotensor::Tensor& V_cache,
                                int valid_len, int num_q_heads, int num_kv_heads,
                                ::brotensor::Tensor& O,
                                float attn_softcap, int window);
void hip_flash_attention_decode_masked(const ::brotensor::Tensor& Q,
                                       const ::brotensor::Tensor& K_cache,
                                       const ::brotensor::Tensor& V_cache,
                                       const float* d_mask,
                                       int num_q_heads, int num_kv_heads,
                                       ::brotensor::Tensor& O,
                                       float attn_softcap, int window);

namespace fa {

size_t max_lds_per_block(int dev) {
    constexpr int kMaxDev = 16;
    static size_t cache[kMaxDev] = {};
    static std::once_flag once[kMaxDev];
    if (dev < 0 || dev >= kMaxDev) dev = 0;
    std::call_once(once[dev], [dev] {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerBlock, dev) != hipSuccess || v <= 0) {
            (void)hipGetLastError();
            v = 64 * 1024;
        }
        cache[dev] = static_cast<size_t>(v);
    });
    return cache[dev];
}

} // namespace fa

namespace {

using fa::FaScalar;
using fa::kFaBlock;
using fa::kFaHdChunk;
using fa::kMaxHdPerThread;
using fa::wave32_max;
using fa::wave32_sum;
using fa::cur_stream;

constexpr int kFaKtile = 32;    // Deepest K tile (keeps the common head_dims at >= 2 blocks per WGP)
constexpr int kFaMinKtile = 8;  // Below this the scalar kernel stops staging K/V in LDS

// LDS layout of the scalar kernels: scores[ktile] + red[32] (FP32), then the
// staged Q row (head_dim), the K tile (ktile x head_dim) and the V tile for
// this block's column chunk (ktile x vchunk). Unstaged launches keep only the
// FP32 part and read Q/K/V straight from global memory.
template <typename T>
inline size_t calc_fa_shmem(int head_dim, int vchunk, int ktile, bool stage) {
    const size_t f32 = sizeof(float) * static_cast<size_t>(kFaKtile + 32);
    if (!stage) return f32;
    return f32 + sizeof(T) * (static_cast<size_t>(head_dim) +
                              static_cast<size_t>(ktile) * head_dim +
                              static_cast<size_t>(ktile) * vchunk);
}

struct FaLaunch {
    int ktile;
    int stage;
    size_t shmem;
};

// Deepest K tile whose staged footprint fits the device's per-block LDS;
// unstaged (global reads, full-depth tile) when even kFaMinKtile does not fit.
template <typename T>
FaLaunch fa_pick_ktile(int head_dim, int dev) {
    const int vchunk = std::min(head_dim, kFaHdChunk);
    const size_t limit = fa::max_lds_per_block(dev);
    for (int kt = kFaKtile; kt >= kFaMinKtile; kt >>= 1) {
        const size_t sh = calc_fa_shmem<T>(head_dim, vchunk, kt, true);
        if (sh <= limit) return {kt, 1, sh};
    }
    return {kFaKtile, 0, calc_fa_shmem<T>(head_dim, vchunk, kFaKtile, false)};
}

// ─── Native AMD FlashAttention-2 Forward Kernel (Tiled Online Softmax) ────────
//
// One block per (query row, head, column chunk). `stage` selects whether the
// Q row and the K/V tiles are copied to LDS first (ktile deep) or read from
// global memory in place (wide heads).

template <typename T>
__global__ void hip_flash_attention_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,
        T* __restrict__ Out,
        int Lk, int Dq, int Dkv, int head_dim,
        int window, int q_offset, int group, int causal,
        int ktile, int stage) {
    extern __shared__ char s_raw[];
    float* scores = reinterpret_cast<float*>(s_raw);
    float* red    = scores + kFaKtile;

    const int q           = blockIdx.x;
    const int h           = blockIdx.y;
    const int d_base      = blockIdx.z * kFaHdChunk;
    const int d_end       = min(head_dim, d_base + kFaHdChunk);
    const int vchunk      = d_end - d_base;
    const int tid         = threadIdx.x;
    const int warp_id     = tid >> 5;
    const int lane        = tid & 31;
    const int num_warps   = blockDim.x >> 5;

    const int h_kv        = h / group;
    const int head_off_q  = h * head_dim;
    const int head_off_kv = h_kv * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));

    T* s_q = reinterpret_cast<T*>(red + 32);
    T* s_k = s_q + head_dim;
    T* s_v = s_k + ktile * head_dim;

    const int aq = q + q_offset;
    int lo = 0;
    int k_hi = Lk - 1;
    if (causal) {
        lo = (window > 0 && aq - window + 1 > 0) ? (aq - window + 1) : 0;
        k_hi = (aq < Lk) ? aq : (Lk - 1);
    } else if (window > 0) {
        lo = (aq - window / 2 > 0) ? (aq - window / 2) : 0;
        k_hi = (aq + window / 2 < Lk) ? (aq + window / 2) : (Lk - 1);
    }

    if (causal && aq < 0) {
        for (int d = d_base + tid; d < d_end; d += blockDim.x) {
            Out[static_cast<size_t>(q) * Dq + head_off_q + d] = FaScalar<T>::from_f32(0.0f);
        }
        return;
    }

    const T* q_vec = Q + static_cast<size_t>(q) * Dq + head_off_q;
    if (stage) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            s_q[d] = q_vec[d];
        }
        __syncthreads();
    }
    const T* q_src = stage ? s_q : q_vec;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    const int k0_start = (lo / ktile) * ktile;

    for (int k0 = k0_start; k0 <= k_hi; k0 += ktile) {
        int klen = ktile;
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;

        // Load the K tile (full width) and this chunk's V columns into LDS
        if (stage) {
            const int k_elems = klen * head_dim;
            for (int idx = tid; idx < k_elems; idx += blockDim.x) {
                const int t = idx / head_dim;
                const int d = idx % head_dim;
                s_k[idx] = K[static_cast<size_t>(k0 + t) * Dkv + head_off_kv + d];
            }
            const int v_elems = klen * vchunk;
            for (int idx = tid; idx < v_elems; idx += blockDim.x) {
                const int t = idx / vchunk;
                const int d = idx % vchunk;
                s_v[idx] = V[static_cast<size_t>(k0 + t) * Dkv + head_off_kv + d_base + d];
            }
            __syncthreads();
        }

        // 1. Compute dot products: q . k * inv_sqrt
        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float s;
            if (kg < lo || (mask && mask[kg] <= 0.5f)) {
                s = -1e30f;
            } else {
                float dot = 0.0f;
                const T* k_ptr = stage ? s_k + t * head_dim
                                       : K + static_cast<size_t>(kg) * Dkv + head_off_kv;
                for (int d = 0; d < head_dim; ++d) {
                    dot += FaScalar<T>::to_f32(q_src[d]) * FaScalar<T>::to_f32(k_ptr[d]);
                }
                s = dot * inv_sqrt;
            }
            scores[t] = s;
        }
        __syncthreads();

        // 2. Tile max reduction using Wave32 shuffles + warp leaders
        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            local_max = fmaxf(local_max, scores[t]);
        }
        local_max = wave32_max(local_max);
        if (lane == 0) red[warp_id] = local_max;
        __syncthreads();

        float tile_max = -1e30f;
        if (tid < num_warps) tile_max = red[tid];
        tile_max = wave32_max(tile_max);
        if (warp_id == 0) red[lane] = tile_max;
        __syncthreads();
        tile_max = red[0];
        // Every warp must have read the max before the sum reduction below
        // reuses red: a fast warp writing red[warp_id] raced a slow one here.
        __syncthreads();

        const float m_new = fmaxf(tile_max, run_max);

        // 3. Exponentiate and sum reduction
        const bool tile_empty = (m_new <= -1e29f);
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
            local_sum += e;
        }
        local_sum = wave32_sum(local_sum);
        if (lane == 0) red[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = 0.0f;
        if (tid < num_warps) tile_sum = red[tid];
        tile_sum = wave32_sum(tile_sum);
        if (warp_id == 0) red[lane] = tile_sum;
        __syncthreads();
        tile_sum = red[0];

        if (tile_sum == 0.0f && tile_max <= -1e29f) {
            __syncthreads();
            continue;
        }

        // 4. Rescale running accumulator
        float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

        // 5. Update partial output for this thread's column slots of the chunk
        int slot = 0;
        for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                const T v = stage ? s_v[t * vchunk + (d - d_base)]
                                  : V[static_cast<size_t>(k0 + t) * Dkv + head_off_kv + d];
                acc += scores[t] * FaScalar<T>::to_f32(v);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    // 6. Normalization and write out
    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
        if (slot >= kMaxHdPerThread) break;
        Out[static_cast<size_t>(q) * Dq + head_off_q + d] = FaScalar<T>::from_f32(partial[slot] * inv);
    }
}

// ─── Variable-Length Flash Attention Kernel ──────────────────────────────────

template <typename T>
__global__ void hip_flash_attention_varlen_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const int* __restrict__ cu_q,
        const int* __restrict__ cu_k,
        T* __restrict__ Out,
        int B, int D, int head_dim,
        int causal, int ktile, int stage) {
    extern __shared__ char s_raw[];
    float* scores = reinterpret_cast<float*>(s_raw);
    float* red    = scores + kFaKtile;

    const int q_global  = blockIdx.x;
    const int h         = blockIdx.y;
    const int d_base    = blockIdx.z * kFaHdChunk;
    const int d_end     = min(head_dim, d_base + kFaHdChunk);
    const int vchunk    = d_end - d_base;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int num_warps = blockDim.x >> 5;

    T* s_q = reinterpret_cast<T*>(red + 32);
    T* s_k = s_q + head_dim;
    T* s_v = s_k + ktile * head_dim;

    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    int b = 0;
    while (b < B && cu_q[b + 1] <= q_global) ++b;
    if (b >= B) return;

    const int q_beg = cu_q[b];
    const int k_beg = cu_k[b];
    const int k_end = cu_k[b + 1];
    const int Lk    = k_end - k_beg;
    if (Lk <= 0) {
        for (int d = d_base + tid; d < d_end; d += blockDim.x) {
            Out[static_cast<size_t>(q_global) * D + head_off + d] = FaScalar<T>::from_f32(0.0f);
        }
        return;
    }
    const int q_local = q_global - q_beg;

    const T* q_vec = Q + static_cast<size_t>(q_global) * D + head_off;
    if (stage) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            s_q[d] = q_vec[d];
        }
        __syncthreads();
    }
    const T* q_src = stage ? s_q : q_vec;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    for (int k0 = 0; k0 < Lk; k0 += ktile) {
        if (causal && k0 > q_local) break;
        int klen = (Lk - k0) < ktile ? (Lk - k0) : ktile;
        if (causal && k0 + klen - 1 > q_local) klen = q_local - k0 + 1;

        // Load the K tile (full width) and this chunk's V columns into LDS
        if (stage) {
            const int k_elems = klen * head_dim;
            for (int idx = tid; idx < k_elems; idx += blockDim.x) {
                const int t = idx / head_dim;
                const int d = idx % head_dim;
                s_k[idx] = K[static_cast<size_t>(k_beg + k0 + t) * D + head_off + d];
            }
            const int v_elems = klen * vchunk;
            for (int idx = tid; idx < v_elems; idx += blockDim.x) {
                const int t = idx / vchunk;
                const int d = idx % vchunk;
                s_v[idx] = V[static_cast<size_t>(k_beg + k0 + t) * D + head_off + d_base + d];
            }
            __syncthreads();
        }

        for (int t = tid; t < klen; t += blockDim.x) {
            float dot = 0.0f;
            const T* k_ptr = stage ? s_k + t * head_dim
                                   : K + static_cast<size_t>(k_beg + k0 + t) * D + head_off;
            for (int d = 0; d < head_dim; ++d) {
                dot += FaScalar<T>::to_f32(q_src[d]) * FaScalar<T>::to_f32(k_ptr[d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) local_max = fmaxf(local_max, scores[t]);
        local_max = wave32_max(local_max);
        if (lane == 0) red[warp_id] = local_max;
        __syncthreads();

        float tile_max = -1e30f;
        if (tid < num_warps) tile_max = red[tid];
        tile_max = wave32_max(tile_max);
        if (warp_id == 0) red[lane] = tile_max;
        __syncthreads();
        tile_max = red[0];
        // Every warp must have read the max before the sum reduction below
        // reuses red: a fast warp writing red[warp_id] raced a slow one here.
        __syncthreads();

        const float m_new = fmaxf(tile_max, run_max);
        const bool tile_empty = (m_new <= -1e29f);

        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
            local_sum += e;
        }
        local_sum = wave32_sum(local_sum);
        if (lane == 0) red[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = 0.0f;
        if (tid < num_warps) tile_sum = red[tid];
        tile_sum = wave32_sum(tile_sum);
        if (warp_id == 0) red[lane] = tile_sum;
        __syncthreads();
        tile_sum = red[0];

        float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

        int slot = 0;
        for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                const T v = stage ? s_v[t * vchunk + (d - d_base)]
                                  : V[static_cast<size_t>(k_beg + k0 + t) * D + head_off + d];
                acc += scores[t] * FaScalar<T>::to_f32(v);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
        if (slot >= kMaxHdPerThread) break;
        Out[static_cast<size_t>(q_global) * D + head_off + d] = FaScalar<T>::from_f32(partial[slot] * inv);
    }
}

template <typename T>
void launch_fa_windowed(const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                        const ::brotensor::Tensor& V, const float* d_mask,
                        ::brotensor::Tensor& O, int Lq, int Lk, int Dq, int Dkv,
                        int num_heads, int head_dim, int window, int q_offset,
                        int group, int causal) {
    const FaLaunch L = fa_pick_ktile<T>(head_dim, Q.device.index);
    dim3 grid(Lq, num_heads, fa::fa_hd_chunks(head_dim));
    hip_flash_attention_kernel<T><<<grid, kFaBlock, L.shmem, cur_stream()>>>(
        static_cast<const T*>(Q.data), static_cast<const T*>(K.data), static_cast<const T*>(V.data),
        d_mask, static_cast<T*>(O.data),
        Lk, Dq, Dkv, head_dim, window, q_offset, group, causal, L.ktile, L.stage);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename T>
void launch_fa_varlen(const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                      const ::brotensor::Tensor& V, const int32_t* cu_q, const int32_t* cu_k,
                      ::brotensor::Tensor& O, int total_q, int batch_size, int num_heads,
                      int D, int head_dim, int causal) {
    const FaLaunch L = fa_pick_ktile<T>(head_dim, Q.device.index);
    dim3 grid(total_q, num_heads, fa::fa_hd_chunks(head_dim));
    hip_flash_attention_varlen_kernel<T><<<grid, kFaBlock, L.shmem, cur_stream()>>>(
        static_cast<const T*>(Q.data), static_cast<const T*>(K.data), static_cast<const T*>(V.data),
        cu_q, cu_k, static_cast<T*>(O.data), batch_size, D, head_dim, causal, L.ktile, L.stage);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace

// ─── Public API Implementations ──────────────────────────────────────────────

void hip_flash_attention_windowed_forward(const ::brotensor::Tensor& Q,
                                          const ::brotensor::Tensor& K,
                                          const ::brotensor::Tensor& V,
                                          const float* d_mask,
                                          int num_heads,
                                          int window,
                                          ::brotensor::Tensor& O,
                                          bool causal) {
    using ::brotensor::Dtype;
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("hip_flash_attention_windowed_forward: Q, K, V must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("hip_flash_attention_windowed_forward: Q, K, V dtype must match");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_flash_attention_windowed_forward: num_heads must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;
    const int Dkv = K.cols;
    if (Dq % num_heads != 0) throw std::runtime_error("hip_flash_attention_windowed_forward: num_heads must divide Dq");
    const int head_dim = Dq / num_heads;
    if (V.cols != Dkv || V.rows != Lk) throw std::runtime_error("hip_flash_attention_windowed_forward: shape mismatch");
    if (Dkv == 0 || Dkv % head_dim != 0) throw std::runtime_error("hip_flash_attention_windowed_forward: K/V width mismatch");
    const int n_kv = Dkv / head_dim;
    if (num_heads % n_kv != 0) throw std::runtime_error("hip_flash_attention_windowed_forward: num_heads must be multiple of n_kv");
    if ((causal || window > 0) && Lk < Lq) {
        throw std::runtime_error("hip_flash_attention_windowed_forward: causal/windowed requires Lk >= Lq");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt) {
        O.resize(Lq, Dq, dt);
    }
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    const int group = num_heads / n_kv;
    // Unwindowed bidirectional attention goes through hipBLAS once the
    // problem is big enough to amortise the GEMM setup, or the head is wide
    // enough that the scalar kernel's one-thread-per-key Q.K dot dominates
    // (past kFaHdChunk it would also recompute that dot per column chunk).
    // Everything else, causal and windowed included, runs the online-softmax
    // kernel, which takes any head_dim.
    if (!causal && window <= 0 &&
        (head_dim > 128 || (Lq >= 64 && Lk >= 64))) {
        hip_dense_attention(Q, K, V, d_mask, num_heads, n_kv, head_dim, O);
        return;
    }

    const int q_offset  = (causal || window > 0) ? (Lk - Lq) : 0;
    const int causal_i  = causal ? 1 : 0;

    if (dt == Dtype::FP16) {
        launch_fa_windowed<__half>(Q, K, V, d_mask, O, Lq, Lk, Dq, Dkv, num_heads, head_dim,
                                   window, q_offset, group, causal_i);
    } else if (dt == Dtype::BF16) {
        launch_fa_windowed<hip_bfloat16>(Q, K, V, d_mask, O, Lq, Lk, Dq, Dkv, num_heads, head_dim,
                                         window, q_offset, group, causal_i);
    } else {
        launch_fa_windowed<float>(Q, K, V, d_mask, O, Lq, Lk, Dq, Dkv, num_heads, head_dim,
                                  window, q_offset, group, causal_i);
    }
}

void hip_flash_attention_forward(const ::brotensor::Tensor& Q,
                                 const ::brotensor::Tensor& K,
                                 const ::brotensor::Tensor& V,
                                 const float* d_mask,
                                 int num_heads,
                                 bool causal,
                                 ::brotensor::Tensor& O) {
    hip_flash_attention_windowed_forward(Q, K, V, d_mask, num_heads, /*window=*/0, O, causal);
}

void hip_flash_attention_gqa_forward(const ::brotensor::Tensor& Q,
                                     const ::brotensor::Tensor& K,
                                     const ::brotensor::Tensor& V,
                                     const float* d_mask,
                                     int num_q_heads,
                                     int num_kv_heads,
                                     bool causal,
                                     ::brotensor::Tensor& O) {
    hip_flash_attention_windowed_forward(Q, K, V, d_mask, num_q_heads, /*window=*/0, O, causal);
}

void hip_flash_attention_varlen_forward(const ::brotensor::Tensor& Q,
                                        const ::brotensor::Tensor& K,
                                        const ::brotensor::Tensor& V,
                                        const int32_t* cu_seqlens_q,
                                        const int32_t* cu_seqlens_k,
                                        int batch_size,
                                        int max_seqlen_q,
                                        int max_seqlen_k,
                                        int num_heads,
                                        int head_dim,
                                        bool causal,
                                        ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("hip_flash_attention_varlen_forward: Q, K, V must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("hip_flash_attention_varlen_forward: Q, K, V dtype must match");
    }
    const int total_q = Q.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D) {
        throw std::runtime_error("hip_flash_attention_varlen_forward: Q/K/V cols != num_heads * head_dim");
    }
    if (O.rows != total_q || O.cols != D || O.dtype != dt) {
        O.resize(total_q, D, dt);
    }
    if (total_q == 0 || batch_size <= 0) return;

    if (!causal && batch_size == 1 && total_q >= 64 && K.rows >= 64 && head_dim <= 128) {
        hip_dense_attention(Q, K, V, nullptr, num_heads, num_heads, head_dim, O);
        return;
    }

    const int causal_i = causal ? 1 : 0;
    if (dt == Dtype::FP16) {
        launch_fa_varlen<__half>(Q, K, V, cu_seqlens_q, cu_seqlens_k, O, total_q, batch_size,
                                 num_heads, D, head_dim, causal_i);
    } else if (dt == Dtype::BF16) {
        launch_fa_varlen<hip_bfloat16>(Q, K, V, cu_seqlens_q, cu_seqlens_k, O, total_q, batch_size,
                                       num_heads, D, head_dim, causal_i);
    } else {
        launch_fa_varlen<float>(Q, K, V, cu_seqlens_q, cu_seqlens_k, O, total_q, batch_size,
                                num_heads, D, head_dim, causal_i);
    }
}

void hip_flash_attention_project_kv(const ::brotensor::Tensor& ctx,
                                    const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
                                    const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
                                    ::brotensor::Tensor& K_out,
                                    ::brotensor::Tensor& V_out) {
    const int Lk = ctx.rows;
    const int D = Wk.rows;
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != ctx.dtype) K_out.resize(Lk, D, ctx.dtype);
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != ctx.dtype) V_out.resize(Lk, D, ctx.dtype);
    if (Lk == 0 || D == 0) return;
    hip_linear_forward_batched_fp16(Wk, bk, ctx, K_out);
    hip_linear_forward_batched_fp16(Wv, bv, ctx, V_out);
}

void hip_flash_attention_q_with_kv_cached_forward(const ::brotensor::Tensor& X,
                                                  const ::brotensor::Tensor& K,
                                                  const ::brotensor::Tensor& V,
                                                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
                                                  const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
                                                  const float* d_mask,
                                                  int num_heads,
                                                  bool causal,
                                                  ::brotensor::Tensor& O) {
    const int Lq = X.rows;
    const int D = Wq.rows;
    ::brotensor::Tensor Q = ::brotensor::Tensor::empty_on(X.device, Lq, D, X.dtype);
    hip_linear_forward_batched_fp16(Wq, bq, X, Q);

    ::brotensor::Tensor Attn_out = ::brotensor::Tensor::empty_on(X.device, Lq, D, X.dtype);
    hip_flash_attention_forward(Q, K, V, d_mask, num_heads, causal, Attn_out);

    if (O.rows != Lq || O.cols != D || O.dtype != X.dtype) O.resize(Lq, D, X.dtype);
    hip_linear_forward_batched_fp16(Wo, bo, Attn_out, O);
}

void hip_flash_attention_qkvo_forward(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor* Ctx,
                                      const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
                                      const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
                                      const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
                                      const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
                                      const float* d_mask,
                                      int num_heads,
                                      bool causal,
                                      ::brotensor::Tensor& O) {
    const int Lq = X.rows;
    const int D = Wq.rows;
    const ::brotensor::Tensor& kv_in = (Ctx && Ctx->size() > 0) ? *Ctx : X;
    const int Lk = kv_in.rows;

    ::brotensor::Tensor Q = ::brotensor::Tensor::empty_on(X.device, Lq, D, X.dtype);
    ::brotensor::Tensor K = ::brotensor::Tensor::empty_on(X.device, Lk, D, X.dtype);
    ::brotensor::Tensor V = ::brotensor::Tensor::empty_on(X.device, Lk, D, X.dtype);

    hip_linear_forward_batched_fp16(Wq, bq, X, Q);
    hip_linear_forward_batched_fp16(Wk, bk, kv_in, K);
    hip_linear_forward_batched_fp16(Wv, bv, kv_in, V);

    ::brotensor::Tensor Attn_out = ::brotensor::Tensor::empty_on(X.device, Lq, D, X.dtype);
    hip_flash_attention_forward(Q, K, V, d_mask, num_heads, causal, Attn_out);

    if (O.rows != Lq || O.cols != D || O.dtype != X.dtype) O.resize(Lq, D, X.dtype);
    hip_linear_forward_batched_fp16(Wo, bo, Attn_out, O);
}

} // namespace brotensor::detail::hip

// Forward declare fallback backward passes
namespace brotensor::detail::cuda {
void flash_attention_backward(const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O, const ::brotensor::Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              ::brotensor::Tensor& dQ, ::brotensor::Tensor& dK, ::brotensor::Tensor& dV);
void flash_attention_varlen_backward(const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,
                                     const ::brotensor::Tensor& O, const ::brotensor::Tensor& dO,
                                     const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     ::brotensor::Tensor& dQ, ::brotensor::Tensor& dK, ::brotensor::Tensor& dV);
void flash_attention_packed_qkv_backward(const ::brotensor::Tensor& QKV, const ::brotensor::Tensor& dO,
                                         const ::brotensor::Tensor& seq_bounds, int num_heads, int window,
                                         ::brotensor::Tensor& dQKV);
void flash_attention_qkvo_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor* Ctx,
                                   const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
                                   const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
                                   const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
                                   const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
                                   const float* d_mask, int num_heads, bool causal,
                                   const ::brotensor::Tensor& dO, ::brotensor::Tensor& dX, ::brotensor::Tensor* dCtx,
                                   ::brotensor::Tensor& dWq, ::brotensor::Tensor* dbq,
                                   ::brotensor::Tensor& dWk, ::brotensor::Tensor* dbk,
                                   ::brotensor::Tensor& dWv, ::brotensor::Tensor* dbv,
                                   ::brotensor::Tensor& dWo, ::brotensor::Tensor* dbo);
} // namespace brotensor::detail::cuda

namespace brotensor::detail::hip {

void fill_hip_vtable_flash_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward                  = &hip_flash_attention_forward;
    v.flash_attention_gqa_forward              = &hip_flash_attention_gqa_forward;
    v.flash_attention_windowed_forward         = &hip_flash_attention_windowed_forward;
    v.flash_attention_varlen_forward           = &hip_flash_attention_varlen_forward;
    v.flash_attention_packed_qkv_forward       = &hip_flash_attention_packed_qkv_forward;
    v.flash_attention_decode                   = &hip_flash_attention_decode;
    v.flash_attention_decode_masked            = &hip_flash_attention_decode_masked;
    v.flash_attention_project_kv               = &hip_flash_attention_project_kv;
    v.flash_attention_q_with_kv_cached_forward = &hip_flash_attention_q_with_kv_cached_forward;
    v.flash_attention_qkvo_forward             = &hip_flash_attention_qkvo_forward;

    // Backward passes
    v.flash_attention_backward                 = &::brotensor::detail::cuda::flash_attention_backward;
    v.flash_attention_varlen_backward          = &::brotensor::detail::cuda::flash_attention_varlen_backward;
    v.flash_attention_packed_qkv_backward      = &::brotensor::detail::cuda::flash_attention_packed_qkv_backward;
    v.flash_attention_qkvo_backward            = &::brotensor::detail::cuda::flash_attention_qkvo_backward;
}

} // namespace brotensor::detail::hip
