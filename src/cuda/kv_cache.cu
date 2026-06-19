// KV-cache append + causal flash-attention decode.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache) {
    const Dtype dt = K_new.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("kv_cache_append: tensors must be FP16 or BF16");
    }
    if (V_new.dtype != dt || K_cache.dtype != dt || V_cache.dtype != dt) {
        throw std::runtime_error("kv_cache_append: all tensors must share the same dtype");
    }
    if (K_new.cols != V_new.cols || K_new.cols != K_cache.cols ||
        K_cache.cols != V_cache.cols) {
        throw std::runtime_error("kv_cache_append: column mismatch");
    }
    if (K_new.rows != V_new.rows) {
        throw std::runtime_error("kv_cache_append: K_new/V_new row mismatch");
    }
    if (K_cache.rows != V_cache.rows) {
        throw std::runtime_error("kv_cache_append: K_cache/V_cache row mismatch");
    }
    const int L_new = K_new.rows;
    const int L_max = K_cache.rows;
    const int D     = K_new.cols;
    if (cur_len < 0 || cur_len + L_new > L_max) {
        throw std::runtime_error("kv_cache_append: cur_len + L_new exceeds cache capacity");
    }
    if (L_new == 0 || D == 0) return;

    const size_t elem = 2; // FP16 and BF16 are both 2 bytes
    const size_t bytes = static_cast<size_t>(L_new) * D * elem;
    const size_t dst_off = static_cast<size_t>(cur_len) * D * elem;

    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(K_cache.data) + dst_off,
        K_new.data, bytes, cudaMemcpyDeviceToDevice, cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(V_cache.data) + dst_off,
        V_new.data, bytes, cudaMemcpyDeviceToDevice, cur_stream()));
}

namespace {

constexpr int FAD_BLOCK = 128;
constexpr int FAD_KTILE = 64;

// Gemma-2 tanh logit soft-cap on an already-scaled score. softcap <= 0 is a
// no-op (returns s untouched), so the default path stays bit-identical.
__device__ __forceinline__ float fad_softcap(float s, float softcap) {
    return softcap > 0.0f ? softcap * tanhf(s / softcap) : s;
}

__global__ void flash_attention_decode_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ Out,
        int Lq, int valid_len, int Dq, int Dkv, int head_dim, int seq_offset,
        int group, float softcap, int window) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FAD_KTILE;

    const int q   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_head_off  = h * head_dim;             // Q/Out (Dq-wide)
    const int kv_head_off = (h / group) * head_dim;   // K/V (Dkv-wide), GQA group
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
    const int p_q = seq_offset + q;
    // Sliding-window lower bound: keys below `lo` are out of band. window <= 0
    // leaves lo == 0 (unbounded causal) — bit-identical to the pre-window path.
    const int lo = (window > 0) ? (p_q - window + 1 < 0 ? 0 : p_q - window + 1) : 0;

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    float run_max = -1e30f;
    float run_sum = 0.0f;

    for (int k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        int klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1 > p_q) klen = p_q - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            if (kg < lo) { scores[t] = -1e30f; continue; }   // out of window
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(Q[q * Dq + q_head_off + d]) *
                       __half2float(K[kg * Dkv + kv_head_off + d]);
            }
            scores[t] = fad_softcap(dot * inv_sqrt, softcap);
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;
        const bool tile_empty = (m_new <= -1e29f);

        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       __half2float(V[(k0 + t) * Dkv + kv_head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * Dq + q_head_off + d] = __float2half(partial[slot] * inv);
    }
}

// Masked fixed-capacity single-query decode — the CUDA-graph-capturable twin
// of the kernel above. Iterates every tile of the full (cap, ·) cache with NO
// host-side length anywhere in the launch, so the captured launch replays
// unchanged as generation advances; a device-resident FP32 key mask carries
// validity instead. Masked keys score -1e30 (their softmax weights underflow
// to exact 0.0f, and adding exact zeros leaves every reduction bit-identical
// to the length-truncated kernel), and fully-masked tiles are skipped before
// any K/V traffic. Templated over storage type instead of the twin-copy
// pattern; numerics are unchanged — all math stays FP32.

template <typename T> __device__ inline float fadm_ld(const T& v);
template <> __device__ inline float fadm_ld<__half>(const __half& v) { return __half2float(v); }
template <> __device__ inline float fadm_ld<__nv_bfloat16>(const __nv_bfloat16& v) { return __bfloat162float(v); }
template <typename T> __device__ inline void fadm_st(T& dst, float v);
template <> __device__ inline void fadm_st<__half>(__half& dst, float v) { dst = __float2half(v); }
template <> __device__ inline void fadm_st<__nv_bfloat16>(__nv_bfloat16& dst, float v) { dst = __float2bfloat16(v); }
template <typename T> struct FadmT2;
template <> struct FadmT2<__half> { using type = __half2; };
template <> struct FadmT2<__nv_bfloat16> { using type = __nv_bfloat162; };

template <typename T>
__global__ void flash_attention_decode_masked_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,
        T* __restrict__ Out,
        int cap, int Dq, int Dkv, int head_dim, int group,
        float softcap, int window) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FAD_KTILE;

    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_head_off  = h * head_dim;             // Q/Out (Dq-wide)
    const int kv_head_off = (h / group) * head_dim;   // K/V (Dkv-wide), GQA group
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    // Sliding-window lower bound. window <= 0 keeps the full valid set (lo == 0,
    // bit-identical to the pre-window path). Otherwise the query sits at the
    // highest valid key index p_max; keep keys in (p_max - window, p_max].
    int lo = 0;
    if (window > 0) {
        float local_pmax = -1.0f;
        for (int kg = tid; kg < cap; kg += blockDim.x) {
            if (mask[kg] > 0.5f) local_pmax = fmaxf(local_pmax, static_cast<float>(kg));
        }
        red[tid] = local_pmax;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
            __syncthreads();
        }
        const int p_max = static_cast<int>(red[0]);
        __syncthreads();
        if (p_max >= 0) { lo = p_max - window + 1; if (lo < 0) lo = 0; }
    }

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    float run_max = -1e30f;
    float run_sum = 0.0f;

    for (int k0 = 0; k0 < cap; k0 += FAD_KTILE) {
        const int klen = (cap - k0) < FAD_KTILE ? (cap - k0) : FAD_KTILE;

        // Tile skip: reduce "any valid key here?" before touching K/V. A
        // generation cache is a valid prefix, so every tile past the cursor
        // costs one strided mask scan and one block reduce, nothing more.
        float any = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (mask[k0 + t] > 0.5f) any = 1.0f;
        }
        red[tid] = any;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const bool tile_any = red[0] > 0.0f;
        __syncthreads();   // red[] is reused below
        if (!tile_any) continue;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            if (mask[kg] <= 0.5f || kg < lo) {
                scores[t] = -1e30f;   // dropped key — weight becomes exact 0
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += fadm_ld(Q[q_head_off + d]) *
                       fadm_ld(K[static_cast<size_t>(kg) * Dkv + kv_head_off + d]);
            }
            scores[t] = fad_softcap(dot * inv_sqrt, softcap);
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;
        const bool tile_empty = (m_new <= -1e29f);

        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                const float s = scores[t];
                // Zero-weight keys add exactly nothing for any finite V, and
                // masked rows may hold garbage (even NaN bit patterns) that
                // 0*v would otherwise propagate — skip the V read entirely.
                if (s == 0.0f) continue;
                acc += s *
                       fadm_ld(V[static_cast<size_t>(k0 + t) * Dkv + kv_head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        fadm_st(Out[q_head_off + d], partial[slot] * inv);
    }
}

// BF16 twin: verbatim copy of flash_attention_decode_kernel with
// __half -> __nv_bfloat16, __half2float -> __bfloat162float,
// __float2half -> __float2bfloat16.
__global__ void flash_attention_decode_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        __nv_bfloat16* __restrict__ Out,
        int Lq, int valid_len, int Dq, int Dkv, int head_dim, int seq_offset,
        int group, float softcap, int window) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FAD_KTILE;

    const int q   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_head_off  = h * head_dim;             // Q/Out (Dq-wide)
    const int kv_head_off = (h / group) * head_dim;   // K/V (Dkv-wide), GQA group
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
    const int p_q = seq_offset + q;
    const int lo = (window > 0) ? (p_q - window + 1 < 0 ? 0 : p_q - window + 1) : 0;

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    float run_max = -1e30f;
    float run_sum = 0.0f;

    for (int k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        int klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1 > p_q) klen = p_q - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            if (kg < lo) { scores[t] = -1e30f; continue; }   // out of window
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __bfloat162float(Q[q * Dq + q_head_off + d]) *
                       __bfloat162float(K[kg * Dkv + kv_head_off + d]);
            }
            scores[t] = fad_softcap(dot * inv_sqrt, softcap);
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;
        const bool tile_empty = (m_new <= -1e29f);

        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       __bfloat162float(V[(k0 + t) * Dkv + kv_head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * Dq + q_head_off + d] = __float2bfloat16(partial[slot] * inv);
    }
}

// ---------------------------------------------------------------------------
// Split-K decode attention (L_q == 1, head_dim % 32 == 0, head_dim <= 256).
//
// The single-block-per-head kernels above leave a 128-SM GPU almost idle at
// decode (16 q heads = 16 blocks) and crawl each K row with one thread, so
// they run at ~2% of HBM bandwidth. This pair fixes both:
//
//   stage 1  grid (n_splits, n_q_heads) — each block runs the online-softmax
//            core over its slice of the key range and writes an unnormalized
//            partial record [m, sum, acc[head_dim]] to a workspace. Keys are
//            scored a warp at a time: the 32 lanes split head_dim, so a K row
//            is one coalesced 2*head_dim-byte read instead of a serial crawl.
//   stage 2  grid (n_q_heads) — rescales the split partials to the global max
//            and combines them (the standard flash-decoding merge).
//
// One template serves both the length-truncated and the masked op; validity
// is `kg < valid_len` or `mask[kg] > 0.5f` per key. The arithmetic sequence
// is otherwise identical, and split boundaries derive from the cache
// CAPACITY (not valid_len) in both, so masked-with-prefix-mask stays
// bit-identical to truncated — the contract the graph decode path relies on.
// Masked-out keys score -1e30 (softmax weight exactly 0.0f) and their K/V
// rows are never read, so garbage (even NaN) in dead rows cannot propagate.

constexpr int FAD_SPLIT_BLOCK  = 128;
constexpr int FAD_MAX_SPLITS   = 16;
constexpr int FAD_SPLIT_MAX_HD = 256;

template <typename T, bool MASKED>
__global__ void fad_split_partials_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,   // MASKED only, else unused
        float* __restrict__ ws,           // (n_q_heads*n_splits) x (head_dim+2)
        int cap, int valid_len,           // valid_len used when !MASKED
        int Dq, int Dkv, int head_dim, int group,
        int tiles_per_split, int n_splits,
        float softcap, int window) {
    __shared__ float scores[FAD_KTILE];
    __shared__ float bcast[2];

    const int h    = blockIdx.y;
    const int spl  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int N_WARPS = FAD_SPLIT_BLOCK / 32;
    constexpr int KPW     = FAD_KTILE / N_WARPS;   // keys per warp per tile

    const int q_head_off  = h * head_dim;
    const int kv_head_off = (h / group) * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));
    const int e = head_dim / 32;                   // dot elems per lane

    // Sliding-window lower bound, shared by all splits of this head. window <= 0
    // leaves lo == 0 (no band) — bit-identical to the pre-window path. For the
    // masked op the query position is the highest valid key index (a full-mask
    // block reduction); for the truncated op it is valid_len - 1 (L_q == 1).
    int lo = 0;
    if (window > 0) {
        int p_max = MASKED ? -1 : (valid_len - 1);
        if (MASKED) {
            float lpm = -1.0f;
            for (int kg = tid; kg < cap; kg += FAD_SPLIT_BLOCK) {
                if (mask[kg] > 0.5f) lpm = fmaxf(lpm, static_cast<float>(kg));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lpm = fmaxf(lpm, __shfl_down_sync(0xffffffffu, lpm, off));
            }
            if (lane == 0) scores[warp] = lpm;
            __syncthreads();
            if (warp == 0) {
                float m = (lane < N_WARPS) ? scores[lane] : -1.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
                }
                if (lane == 0) bcast[0] = m;
            }
            __syncthreads();
            p_max = static_cast<int>(bcast[0]);
            __syncthreads();
        }
        if (p_max >= 0) { lo = p_max - window + 1; if (lo < 0) lo = 0; }
    }

    // This lane's slice of the query row, kept in registers across all keys.
    float qreg[FAD_SPLIT_MAX_HD / 32];
    #pragma unroll
    for (int i = 0; i < FAD_SPLIT_MAX_HD / 32; ++i) qreg[i] = 0.0f;
    for (int i = 0; i < e; ++i) {
        qreg[i] = fadm_ld(Q[q_head_off + lane * e + i]);
    }

    constexpr int MAX_SLOTS = FAD_SPLIT_MAX_HD / FAD_SPLIT_BLOCK;
    float partial[MAX_SLOTS];
    #pragma unroll
    for (int i = 0; i < MAX_SLOTS; ++i) partial[i] = 0.0f;
    float run_max = -1e30f;
    float run_sum = 0.0f;

    const int k_begin = spl * tiles_per_split * FAD_KTILE;
    int k_end = k_begin + tiles_per_split * FAD_KTILE;
    if (k_end > cap) k_end = cap;

    for (int k0 = k_begin; k0 < k_end; k0 += FAD_KTILE) {
        if (!MASKED && k0 >= valid_len) break;
        const int klen = (k_end - k0) < FAD_KTILE ? (k_end - k0) : FAD_KTILE;

        for (int j = 0; j < KPW; ++j) {
            const int t = warp * KPW + j;
            if (t >= klen) break;
            const int kg = k0 + t;
            const bool valid = (kg >= lo) &&
                               (MASKED ? (mask[kg] > 0.5f) : (kg < valid_len));
            if (!valid) {
                if (lane == 0) scores[t] = -1e30f;
                continue;
            }
            const T* krow = K + static_cast<size_t>(kg) * Dkv + kv_head_off +
                            lane * e;
            float dot = 0.0f;
            if ((e & 1) == 0) {
                using T2 = typename FadmT2<T>::type;
                #pragma unroll
                for (int i = 0; i < FAD_SPLIT_MAX_HD / 32; i += 2) {
                    if (i >= e) break;
                    const T2 kv2 = *reinterpret_cast<const T2*>(krow + i);
                    dot += qreg[i] * fadm_ld(kv2.x) +
                           qreg[i + 1] * fadm_ld(kv2.y);
                }
            } else {
                for (int i = 0; i < e; ++i) dot += qreg[i] * fadm_ld(krow[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            }
            if (lane == 0) scores[t] = fad_softcap(dot * inv_sqrt, softcap);
        }
        __syncthreads();

        if (warp == 0) {   // tile max
            float m = -1e30f;
            for (int t = lane; t < klen; t += 32) m = fmaxf(m, scores[t]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
            }
            if (lane == 0) bcast[0] = m;
        }
        __syncthreads();
        const float m_new = fmaxf(run_max, bcast[0]);
        const bool tile_empty = (m_new <= -1e29f);

        if (tid < klen) {
            scores[tid] = tile_empty ? 0.0f : __expf(scores[tid] - m_new);
        }
        __syncthreads();
        if (warp == 0) {   // tile weight sum
            float s = 0.0f;
            for (int t = lane; t < klen; t += 32) s += scores[t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            if (lane == 0) bcast[1] = s;
        }
        __syncthreads();
        const float tile_sum = bcast[1];
        const float alpha = (run_max <= -1e29f) ? 0.0f
                                                : __expf(run_max - m_new);

        int slot = 0;
        for (int d = tid; d < head_dim; d += FAD_SPLIT_BLOCK, ++slot) {
            float acc = alpha * partial[slot];
            #pragma unroll 4
            for (int t = 0; t < klen; ++t) {
                const float s = scores[t];
                // Zero-weight keys contribute exactly nothing for finite V,
                // and dead rows may hold garbage — skip the read entirely.
                if (s == 0.0f) continue;
                acc += s * fadm_ld(V[static_cast<size_t>(k0 + t) * Dkv +
                                     kv_head_off + d]);
            }
            partial[slot] = acc;
        }
        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    float* rec = ws + (static_cast<size_t>(h) * n_splits + spl) *
                          (head_dim + 2);
    if (tid == 0) {
        rec[0] = run_max;
        rec[1] = run_sum;
    }
    int slot = 0;
    for (int d = tid; d < head_dim; d += FAD_SPLIT_BLOCK, ++slot) {
        rec[2 + d] = partial[slot];
    }
}

template <typename T>
__global__ void fad_split_reduce_kernel(const float* __restrict__ ws,
                                        T* __restrict__ Out,
                                        int head_dim, int n_splits) {
    const int h       = blockIdx.x;
    const int tid     = threadIdx.x;
    const int rec_len = head_dim + 2;
    const float* base = ws + static_cast<size_t>(h) * n_splits * rec_len;

    float m_g = -1e30f;
    for (int s = 0; s < n_splits; ++s) m_g = fmaxf(m_g, base[s * rec_len]);

    float f[FAD_MAX_SPLITS];
    float den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float m_s = base[s * rec_len];
        const float w   = (m_s <= -1e29f) ? 0.0f : __expf(m_s - m_g);
        f[s] = w;
        den += w * base[s * rec_len + 1];
    }
    const float inv = (den > 0.0f) ? (1.0f / den) : 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float num = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            if (f[s] == 0.0f) continue;   // empty split: acc rows are stale
            num += f[s] * base[s * rec_len + 2 + d];
        }
        fadm_st(Out[h * head_dim + d], num * inv);
    }
}

// Per-device partials scratch for the split kernels, grown monotonically and
// deliberately never freed (a few hundred KB at most). Guarded by the
// library's single-host-thread-per-device usage contract — no locking.
// Growing requires cudaMalloc, which is illegal mid-capture, so a capture
// that arrives with a too-small workspace throws: run the op once eagerly
// first (the warm-up-then-capture contract the graph API already documents,
// and what every existing graph user does).
float* fad_split_workspace(size_t need_floats) {
    constexpr int MAX_DEVICES = 16;
    static float* buf[MAX_DEVICES]   = {};
    static size_t cap_f[MAX_DEVICES] = {};
    int dev = 0;
    BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= MAX_DEVICES) {
        throw std::runtime_error(
            "flash_attention_decode: device ordinal out of range");
    }
    if (cap_f[dev] < need_floats) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        BROTENSOR_CUDA_CHECK(cudaStreamIsCapturing(cur_stream(), &st));
        if (st != cudaStreamCaptureStatusNone) {
            throw std::runtime_error(
                "flash_attention_decode(_masked): split-K workspace cannot "
                "grow during CUDA graph capture — run the op once eagerly "
                "(warm-up) before capturing");
        }
        if (buf[dev] != nullptr) BROTENSOR_CUDA_CHECK(cudaFree(buf[dev]));
        buf[dev]   = nullptr;
        cap_f[dev] = 0;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf[dev]),
                                        need_floats * sizeof(float)));
        cap_f[dev] = need_floats;
    }
    return buf[dev];
}

// Split geometry depends on cache CAPACITY only, so the truncated and masked
// ops always agree (bit-identity) and a captured masked launch stays valid as
// the cursor advances.
inline void fad_split_dims(int cap, int& n_splits, int& tiles_per_split) {
    const int n_tiles = (cap + FAD_KTILE - 1) / FAD_KTILE;
    n_splits = n_tiles < FAD_MAX_SPLITS ? n_tiles : FAD_MAX_SPLITS;
    tiles_per_split = (n_tiles + n_splits - 1) / n_splits;
}

inline bool fad_split_eligible(int head_dim) {
    return head_dim % 32 == 0 && head_dim <= FAD_SPLIT_MAX_HD;
}

} // namespace

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_q_heads, int num_kv_heads,
                            Tensor& O, float attn_softcap, int window) {
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) {
        throw std::runtime_error("flash_attention_decode: tensors must be FP16 or BF16");
    }
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode: all tensors must share the same dtype");
    }
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;            // num_q_heads  * head_dim
    const int Dkv = K_cache.cols;      // num_kv_heads * head_dim
    if (V_cache.cols != Dkv) {
        throw std::runtime_error("flash_attention_decode: K_cache.cols != V_cache.cols");
    }
    if (valid_len < 0 || valid_len > K_cache.rows || valid_len > V_cache.rows) {
        throw std::runtime_error("flash_attention_decode: invalid valid_len");
    }
    if (valid_len < Lq) {
        throw std::runtime_error("flash_attention_decode: valid_len must be >= Lq");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention_decode: num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode: num_kv_heads must divide num_q_heads");
    }
    if (Dq % num_q_heads != 0 || Dkv % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode: head_dim does not divide cols cleanly");
    }
    const int head_dim = Dq / num_q_heads;
    if (Dkv / num_kv_heads != head_dim) {
        throw std::runtime_error("flash_attention_decode: head_dim mismatch between Q and K/V");
    }
    const int group = num_q_heads / num_kv_heads;   // q heads served per kv head
    if ((head_dim + FAD_BLOCK - 1) / FAD_BLOCK > 8) {
        throw std::runtime_error("flash_attention_decode: head_dim too large (max 8 * FAD_BLOCK = 1024)");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != Q.dtype) {
        O.resize(Lq, Dq, Q.dtype);
    }
    if (Lq == 0 || Dq == 0 || valid_len == 0) return;

    if (Lq == 1 && fad_split_eligible(head_dim)) {
        const int cap = K_cache.rows;
        int n_splits = 0, tps = 0;
        fad_split_dims(cap, n_splits, tps);
        float* ws = fad_split_workspace(static_cast<size_t>(num_q_heads) *
                                        n_splits * (head_dim + 2));
        const dim3 g1(n_splits, num_q_heads, 1);
        if (Q.dtype == Dtype::FP16) {
            fad_split_partials_kernel<__half, false>
                <<<g1, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __half*>(Q.data),
                    static_cast<const __half*>(K_cache.data),
                    static_cast<const __half*>(V_cache.data),
                    nullptr, ws, cap, valid_len, Dq, Dkv, head_dim, group,
                    tps, n_splits, attn_softcap, window);
            fad_split_reduce_kernel<__half>
                <<<num_q_heads, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    ws, static_cast<__half*>(O.data), head_dim, n_splits);
        } else {
            fad_split_partials_kernel<__nv_bfloat16, false>
                <<<g1, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(Q.data),
                    static_cast<const __nv_bfloat16*>(K_cache.data),
                    static_cast<const __nv_bfloat16*>(V_cache.data),
                    nullptr, ws, cap, valid_len, Dq, Dkv, head_dim, group,
                    tps, n_splits, attn_softcap, window);
            fad_split_reduce_kernel<__nv_bfloat16>
                <<<num_q_heads, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    ws, static_cast<__nv_bfloat16*>(O.data), head_dim,
                    n_splits);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int seq_offset = valid_len - Lq;
    const size_t shmem = (static_cast<size_t>(FAD_KTILE) + FAD_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_q_heads, 1);
    if (Q.dtype == Dtype::FP16) {
        flash_attention_decode_kernel<<<grid, FAD_BLOCK, shmem, cur_stream()>>>(
            static_cast<const __half*>(Q.data),
            static_cast<const __half*>(K_cache.data),
            static_cast<const __half*>(V_cache.data),
            static_cast<__half*>(O.data),
            Lq, valid_len, Dq, Dkv, head_dim, seq_offset, group,
            attn_softcap, window);
    } else {
        flash_attention_decode_bf16_kernel<<<grid, FAD_BLOCK, shmem, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(Q.data),
            static_cast<const __nv_bfloat16*>(K_cache.data),
            static_cast<const __nv_bfloat16*>(V_cache.data),
            static_cast<__nv_bfloat16*>(O.data),
            Lq, valid_len, Dq, Dkv, head_dim, seq_offset, group,
            attn_softcap, window);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void flash_attention_decode_masked(const Tensor& Q,
                                   const Tensor& K_cache,
                                   const Tensor& V_cache,
                                   const float* d_mask,
                                   int num_q_heads, int num_kv_heads,
                                   Tensor& O, float attn_softcap, int window) {
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) {
        throw std::runtime_error("flash_attention_decode_masked: tensors must be FP16 or BF16");
    }
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode_masked: all tensors must share the same dtype");
    }
    if (d_mask == nullptr) {
        throw std::runtime_error("flash_attention_decode_masked: d_mask must not be null");
    }
    if (Q.rows != 1) {
        throw std::runtime_error("flash_attention_decode_masked: Q must be a single row (L_q == 1)");
    }
    const int Dq  = Q.cols;
    const int Dkv = K_cache.cols;
    const int cap = K_cache.rows;
    if (V_cache.cols != Dkv) {
        throw std::runtime_error("flash_attention_decode_masked: K_cache.cols != V_cache.cols");
    }
    if (V_cache.rows != cap) {
        throw std::runtime_error("flash_attention_decode_masked: K_cache/V_cache row mismatch");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention_decode_masked: num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_masked: num_kv_heads must divide num_q_heads");
    }
    if (Dq % num_q_heads != 0 || Dkv % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_masked: head_dim does not divide cols cleanly");
    }
    const int head_dim = Dq / num_q_heads;
    if (Dkv / num_kv_heads != head_dim) {
        throw std::runtime_error("flash_attention_decode_masked: head_dim mismatch between Q and K/V");
    }
    const int group = num_q_heads / num_kv_heads;
    if ((head_dim + FAD_BLOCK - 1) / FAD_BLOCK > 8) {
        throw std::runtime_error("flash_attention_decode_masked: head_dim too large (max 8 * FAD_BLOCK = 1024)");
    }
    if (O.rows != 1 || O.cols != Dq || O.dtype != Q.dtype) {
        O.resize(1, Dq, Q.dtype);
    }
    if (Dq == 0 || cap == 0) return;

    if (fad_split_eligible(head_dim)) {
        int n_splits = 0, tps = 0;
        fad_split_dims(cap, n_splits, tps);
        float* ws = fad_split_workspace(static_cast<size_t>(num_q_heads) *
                                        n_splits * (head_dim + 2));
        const dim3 g1(n_splits, num_q_heads, 1);
        if (Q.dtype == Dtype::FP16) {
            fad_split_partials_kernel<__half, true>
                <<<g1, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __half*>(Q.data),
                    static_cast<const __half*>(K_cache.data),
                    static_cast<const __half*>(V_cache.data),
                    d_mask, ws, cap, /*valid_len=*/cap, Dq, Dkv, head_dim,
                    group, tps, n_splits, attn_softcap, window);
            fad_split_reduce_kernel<__half>
                <<<num_q_heads, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    ws, static_cast<__half*>(O.data), head_dim, n_splits);
        } else {
            fad_split_partials_kernel<__nv_bfloat16, true>
                <<<g1, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(Q.data),
                    static_cast<const __nv_bfloat16*>(K_cache.data),
                    static_cast<const __nv_bfloat16*>(V_cache.data),
                    d_mask, ws, cap, /*valid_len=*/cap, Dq, Dkv, head_dim,
                    group, tps, n_splits, attn_softcap, window);
            fad_split_reduce_kernel<__nv_bfloat16>
                <<<num_q_heads, FAD_SPLIT_BLOCK, 0, cur_stream()>>>(
                    ws, static_cast<__nv_bfloat16*>(O.data), head_dim,
                    n_splits);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const size_t shmem = (static_cast<size_t>(FAD_KTILE) + FAD_BLOCK) * sizeof(float);
    dim3 grid(1, num_q_heads, 1);
    if (Q.dtype == Dtype::FP16) {
        flash_attention_decode_masked_kernel<__half><<<grid, FAD_BLOCK, shmem, cur_stream()>>>(
            static_cast<const __half*>(Q.data),
            static_cast<const __half*>(K_cache.data),
            static_cast<const __half*>(V_cache.data),
            d_mask,
            static_cast<__half*>(O.data),
            cap, Dq, Dkv, head_dim, group, attn_softcap, window);
    } else {
        flash_attention_decode_masked_kernel<__nv_bfloat16><<<grid, FAD_BLOCK, shmem, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(Q.data),
            static_cast<const __nv_bfloat16*>(K_cache.data),
            static_cast<const __nv_bfloat16*>(V_cache.data),
            d_mask,
            static_cast<__nv_bfloat16*>(O.data),
            cap, Dq, Dkv, head_dim, group, attn_softcap, window);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
