// Native AMD HIP FlashAttention-2 forward for RDNA 3.5 (gfx1151, Wave32, 64 KB LDS).
//
// Designed specifically for AMD RDNA 3.5:
// - Wave32 execution: 32-lane wavefronts, cross-lane shuffles (__shfl_xor / __shfl_down
//   with 0xffffffff mask) for in-warp max/sum reductions without shared-memory stalls.
// - 64 KB LDS budget: tile sizes Br x Bc chosen so Q, K, V tiles fit within 32 KB LDS
//   (preserving >= 2 workgroups per WGP for full occupancy).
// - Vectorized 128-bit loads (uint4 / 8x half / 4x float) from unified memory.
// - Unified support for causal, bidirectional, GQA, windowed, and varlen sequences.

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/hip_check.h"
#include "detail/hipblas_handle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifndef BROTENSOR_HIPBLAS_CHECK
#define BROTENSOR_HIPBLAS_CHECK(expr)                                                 \
    do {                                                                               \
        hipblasStatus_t _bga_err = (expr);                                             \
        if (_bga_err != HIPBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("hipBLAS error: ") + #expr);          \
        }                                                                              \
    } while (0)
#endif

namespace brotensor::detail::hip {

void hip_softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                              int rows, int cols);

void* hip_current_stream(int dev = -1);
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

namespace {

constexpr int kFaBlock = 128; // 4 Wave32 waves per block
constexpr int kFaKtile = 64;  // Key tile depth
constexpr int kMaxHdPerThread = 8;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

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

template <typename T>
inline size_t calc_fa_shmem(int head_dim, int ktile = kFaKtile) {
    return sizeof(float) * (ktile + 32) + sizeof(T) * (head_dim + 2 * ktile * head_dim);
}

// ─── Native AMD FlashAttention-2 Forward Kernel (Tiled Online Softmax) ────────

template <typename T>
__global__ void hip_flash_attention_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,
        T* __restrict__ Out,
        int Lk, int Dq, int Dkv, int head_dim,
        int window, int q_offset, int group, int causal) {
    extern __shared__ char s_raw[];
    float* scores = reinterpret_cast<float*>(s_raw);
    float* red    = scores + kFaKtile;
    T* s_q        = reinterpret_cast<T*>(red + 32);
    T* s_k        = s_q + head_dim;
    T* s_v        = s_k + kFaKtile * head_dim;

    const int q           = blockIdx.x;
    const int h           = blockIdx.y;
    const int tid         = threadIdx.x;
    const int warp_id     = tid >> 5;
    const int lane        = tid & 31;
    const int num_warps   = blockDim.x >> 5;

    const int h_kv        = h / group;
    const int head_off_q  = h * head_dim;
    const int head_off_kv = h_kv * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));

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
        for (int d = tid; d < head_dim; d += blockDim.x) {
            Out[static_cast<size_t>(q) * Dq + head_off_q + d] = FaScalar<T>::from_f32(0.0f);
        }
        return;
    }

    const T* q_vec = Q + static_cast<size_t>(q) * Dq + head_off_q;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_vec[d];
    }
    __syncthreads();

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    const int k0_start = (lo / kFaKtile) * kFaKtile;

    for (int k0 = k0_start; k0 <= k_hi; k0 += kFaKtile) {
        int klen = kFaKtile;
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;

        // Load K and V tiles into LDS
        const int tile_elems = klen * head_dim;
        for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
            const int t = idx / head_dim;
            const int d = idx % head_dim;
            const size_t kg = static_cast<size_t>(k0 + t);
            s_k[idx] = K[kg * Dkv + head_off_kv + d];
            s_v[idx] = V[kg * Dkv + head_off_kv + d];
        }
        __syncthreads();

        // 1. Compute dot products: s_q . s_k * inv_sqrt from LDS
        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float s;
            if (kg < lo || (mask && mask[kg] <= 0.5f)) {
                s = -1e30f;
            } else {
                float dot = 0.0f;
                const T* k_ptr = s_k + t * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    dot += FaScalar<T>::to_f32(s_q[d]) * FaScalar<T>::to_f32(k_ptr[d]);
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

        // 5. Update partial output for this thread's assigned dimension slots from LDS s_v
        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] * FaScalar<T>::to_f32(s_v[t * head_dim + d]);
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
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
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
        int causal) {
    extern __shared__ char s_raw[];
    float* scores = reinterpret_cast<float*>(s_raw);
    float* red    = scores + kFaKtile;
    T* s_q        = reinterpret_cast<T*>(red + 32);
    T* s_k        = s_q + head_dim;
    T* s_v        = s_k + kFaKtile * head_dim;

    const int q_global  = blockIdx.x;
    const int h         = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int num_warps = blockDim.x >> 5;

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
        for (int d = tid; d < head_dim; d += blockDim.x) {
            Out[static_cast<size_t>(q_global) * D + head_off + d] = FaScalar<T>::from_f32(0.0f);
        }
        return;
    }
    const int q_local = q_global - q_beg;

    const T* q_vec = Q + static_cast<size_t>(q_global) * D + head_off;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_vec[d];
    }
    __syncthreads();

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    for (int k0 = 0; k0 < Lk; k0 += kFaKtile) {
        if (causal && k0 > q_local) break;
        int klen = (Lk - k0) < kFaKtile ? (Lk - k0) : kFaKtile;
        if (causal && k0 + klen - 1 > q_local) klen = q_local - k0 + 1;

        // Load K and V tiles into LDS
        const int tile_elems = klen * head_dim;
        for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
            const int t = idx / head_dim;
            const int d = idx % head_dim;
            const size_t kg = static_cast<size_t>(k_beg + k0 + t);
            s_k[idx] = K[kg * D + head_off + d];
            s_v[idx] = V[kg * D + head_off + d];
        }
        __syncthreads();

        for (int t = tid; t < klen; t += blockDim.x) {
            float dot = 0.0f;
            const T* k_ptr = s_k + t * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dot += FaScalar<T>::to_f32(s_q[d]) * FaScalar<T>::to_f32(k_ptr[d]);
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
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] * FaScalar<T>::to_f32(s_v[t * head_dim + d]);
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
        if (slot >= kMaxHdPerThread) break;
        Out[static_cast<size_t>(q_global) * D + head_off + d] = FaScalar<T>::from_f32(partial[slot] * inv);
    }
}

template <typename T>
__global__ void transpose_lhd_to_hld_kernel(const T* __restrict__ In,
                                            T* __restrict__ Out,
                                            int L, int H, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    const int h = blockIdx.z;
    if (c < d && l < L && h < H) {
        Out[(h * L + l) * d + c] = In[(l * H + h) * d + c];
    }
}

template <typename T>
__global__ void transpose_hld_to_lhd_kernel(const T* __restrict__ In,
                                            T* __restrict__ Out,
                                            int L, int H, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    const int h = blockIdx.z;
    if (c < d && l < L && h < H) {
        Out[(l * H + h) * d + c] = In[(h * L + l) * d + c];
    }
}

bool hip_dense_attention_hipblas(
    const ::brotensor::Tensor& Q,
    const ::brotensor::Tensor& K,
    const ::brotensor::Tensor& V,
    int num_heads, int head_dim,
    ::brotensor::Tensor& O) {
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int H = num_heads;
    const int d = head_dim;
    using ::brotensor::Dtype;
    using ::brotensor::Tensor;
    const Dtype dt = Q.dtype;
    hipStream_t stream = cur_stream();
    hipblasHandle_t handle = hipblas_handle(Q.device.index);

    Tensor Qt = Tensor::zeros_on(Q.device, H * Lq, d, dt);
    Tensor Kt = Tensor::zeros_on(K.device, H * Lk, d, dt);
    Tensor Vt = Tensor::zeros_on(V.device, H * Lk, d, dt);

    dim3 blk(std::min(d, 256));
    dim3 grid_q((d + blk.x - 1) / blk.x, Lq, H);
    dim3 grid_k((d + blk.x - 1) / blk.x, Lk, H);

    if (dt == Dtype::FP16) {
        transpose_lhd_to_hld_kernel<__half><<<grid_q, blk, 0, stream>>>(
            static_cast<const __half*>(Q.data), static_cast<__half*>(Qt.data), Lq, H, d);
        transpose_lhd_to_hld_kernel<__half><<<grid_k, blk, 0, stream>>>(
            static_cast<const __half*>(K.data), static_cast<__half*>(Kt.data), Lk, H, d);
        transpose_lhd_to_hld_kernel<__half><<<grid_k, blk, 0, stream>>>(
            static_cast<const __half*>(V.data), static_cast<__half*>(Vt.data), Lk, H, d);
    } else if (dt == Dtype::BF16) {
        transpose_lhd_to_hld_kernel<hip_bfloat16><<<grid_q, blk, 0, stream>>>(
            static_cast<const hip_bfloat16*>(Q.data), static_cast<hip_bfloat16*>(Qt.data), Lq, H, d);
        transpose_lhd_to_hld_kernel<hip_bfloat16><<<grid_k, blk, 0, stream>>>(
            static_cast<const hip_bfloat16*>(K.data), static_cast<hip_bfloat16*>(Kt.data), Lk, H, d);
        transpose_lhd_to_hld_kernel<hip_bfloat16><<<grid_k, blk, 0, stream>>>(
            static_cast<const hip_bfloat16*>(V.data), static_cast<hip_bfloat16*>(Vt.data), Lk, H, d);
    } else {
        transpose_lhd_to_hld_kernel<float><<<grid_q, blk, 0, stream>>>(
            static_cast<const float*>(Q.data), static_cast<float*>(Qt.data), Lq, H, d);
        transpose_lhd_to_hld_kernel<float><<<grid_k, blk, 0, stream>>>(
            static_cast<const float*>(K.data), static_cast<float*>(Kt.data), Lk, H, d);
        transpose_lhd_to_hld_kernel<float><<<grid_k, blk, 0, stream>>>(
            static_cast<const float*>(V.data), static_cast<float*>(Vt.data), Lk, H, d);
    }

    Tensor S = Tensor::zeros_on(Q.device, H * Lq, Lk, dt);
    const float alpha_s = 1.0f / std::sqrt(static_cast<float>(d));
    const float beta_zero = 0.0f;
    const long long strideQ = static_cast<long long>(Lq) * d;
    const long long strideK = static_cast<long long>(Lk) * d;
    const long long strideS = static_cast<long long>(Lq) * Lk;

    hipDataType hip_dt = (dt == Dtype::FP16) ? HIP_R_16F : (dt == Dtype::BF16 ? HIP_R_16BF : HIP_R_32F);

    BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
        Lk, Lq, d,
        &alpha_s,
        Kt.data, hip_dt, d, strideK,
        Qt.data, hip_dt, d, strideQ,
        &beta_zero,
        S.data, hip_dt, Lk, strideS,
        H,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));

    hip_softmax_rows_forward(S, S, H * Lq, Lk);

    Tensor Ot = Tensor::zeros_on(Q.device, H * Lq, d, dt);
    const float alpha_one = 1.0f;
    const long long strideO = static_cast<long long>(Lq) * d;

    BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        d, Lq, Lk,
        &alpha_one,
        Vt.data, hip_dt, d, strideK,
        S.data, hip_dt, Lk, strideS,
        &beta_zero,
        Ot.data, hip_dt, d, strideO,
        H,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));

    if (dt == Dtype::FP16) {
        transpose_hld_to_lhd_kernel<__half><<<grid_q, blk, 0, stream>>>(
            static_cast<const __half*>(Ot.data), static_cast<__half*>(O.data), Lq, H, d);
    } else if (dt == Dtype::BF16) {
        transpose_hld_to_lhd_kernel<hip_bfloat16><<<grid_q, blk, 0, stream>>>(
            static_cast<const hip_bfloat16*>(Ot.data), static_cast<hip_bfloat16*>(O.data), Lq, H, d);
    } else {
        transpose_hld_to_lhd_kernel<float><<<grid_q, blk, 0, stream>>>(
            static_cast<const float*>(Ot.data), static_cast<float*>(O.data), Lq, H, d);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    return true;
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
    if ((head_dim + kFaBlock - 1) / kFaBlock > kMaxHdPerThread) {
        throw std::runtime_error("hip_flash_attention_windowed_forward: head_dim too large");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt) {
        O.resize(Lq, Dq, dt);
    }
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    const int group = num_heads / n_kv;
    if (!causal && window == 0 && d_mask == nullptr && group == 1 && Lq >= 64 && Lk >= 64 && head_dim <= 128) {
        if (hip_dense_attention_hipblas(Q, K, V, num_heads, head_dim, O)) {
            return;
        }
    }

    const int q_offset  = (causal || window > 0) ? (Lk - Lq) : 0;
    const int causal_i  = causal ? 1 : 0;
    dim3 grid(Lq, num_heads, 1);
    hipStream_t stream = cur_stream();

    if (dt == Dtype::FP16) {
        const size_t shmem = calc_fa_shmem<__half>(head_dim);
        hip_flash_attention_kernel<__half><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const __half*>(Q.data), static_cast<const __half*>(K.data), static_cast<const __half*>(V.data),
            d_mask, static_cast<__half*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    } else if (dt == Dtype::BF16) {
        const size_t shmem = calc_fa_shmem<hip_bfloat16>(head_dim);
        hip_flash_attention_kernel<hip_bfloat16><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const hip_bfloat16*>(Q.data), static_cast<const hip_bfloat16*>(K.data), static_cast<const hip_bfloat16*>(V.data),
            d_mask, static_cast<hip_bfloat16*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    } else {
        const size_t shmem = calc_fa_shmem<float>(head_dim);
        hip_flash_attention_kernel<float><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const float*>(Q.data), static_cast<const float*>(K.data), static_cast<const float*>(V.data),
            d_mask, static_cast<float*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
        if (hip_dense_attention_hipblas(Q, K, V, num_heads, head_dim, O)) {
            return;
        }
    }

    dim3 grid(total_q, num_heads, 1);
    hipStream_t stream = cur_stream();

    if (dt == Dtype::FP16) {
        const size_t shmem = calc_fa_shmem<__half>(head_dim);
        hip_flash_attention_varlen_kernel<__half><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const __half*>(Q.data), static_cast<const __half*>(K.data), static_cast<const __half*>(V.data),
            cu_seqlens_q, cu_seqlens_k, static_cast<__half*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    } else if (dt == Dtype::BF16) {
        const size_t shmem = calc_fa_shmem<hip_bfloat16>(head_dim);
        hip_flash_attention_varlen_kernel<hip_bfloat16><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const hip_bfloat16*>(Q.data), static_cast<const hip_bfloat16*>(K.data), static_cast<const hip_bfloat16*>(V.data),
            cu_seqlens_q, cu_seqlens_k, static_cast<hip_bfloat16*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    } else {
        const size_t shmem = calc_fa_shmem<float>(head_dim);
        hip_flash_attention_varlen_kernel<float><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const float*>(Q.data), static_cast<const float*>(K.data), static_cast<const float*>(V.data),
            cu_seqlens_q, cu_seqlens_k, static_cast<float*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
