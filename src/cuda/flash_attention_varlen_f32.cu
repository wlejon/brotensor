// FP32 flash_attention_varlen_forward for head_dim 64, non-causal.
//
// The generic varlen kernel runs one block per (query row, head) and walks
// the keys one dot product per thread; at a few thousand tokens that is two
// orders of magnitude off the FMA rate. This is a FlashAttention-2 style
// tiled kernel on plain FP32 FMAs (no tensor cores, so FP32 throughout):
//
//   * one CTA per (64-row query block, head), 256 threads as 16 x 16;
//   * each thread owns 4 query rows; it computes their scores against 4 keys
//     of the 64-key tile (S = Q K^T from shared memory, float4 reads) and
//     4 head dims of the output (O += P V);
//   * the online softmax (exp2, log2 e folded into the Q scale) runs in
//     registers, row statistics reduced over the 16 threads sharing the rows
//     with warp shuffles; P goes through shared memory (over the K tile,
//     which is dead by then) to feed P V.
//
// Varlen: a query block may straddle sequences. Each row looks up its own key
// range [cu_k[b], cu_k[b+1]); the block walks the union and masks every score
// against its row's range, so straddling blocks are exact.

#include "detail/cuda_check.h"
#include "detail/smem_opt_in.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>

namespace brotensor::detail::cuda {

namespace {

constexpr int kBQ = 64, kBK = 64, kHD = 64;
constexpr int kKS = kHD + 4;             // K tile row stride: conflict-free float4 reads
constexpr int kThreads = 256;
constexpr int kSmemFloats = kBQ * kHD + kBK * kKS + kBK * kHD;
constexpr size_t kSmemBytes = kSmemFloats * sizeof(float);

__device__ __forceinline__ float fma4(float4 a, float4 b, float acc) {
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    return fmaf(a.w, b.w, acc);
}

__global__ void __launch_bounds__(kThreads, 2)
fa_varlen_f32_hd64_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                          const float* __restrict__ V, const int* __restrict__ cu_q,
                          const int* __restrict__ cu_k, float* __restrict__ O,
                          int total_q, int B, int D, float q_scale) {
    extern __shared__ float4 smem4[];
    float* Qs = reinterpret_cast<float*>(smem4);
    float* Ks = Qs + kBQ * kHD;
    float* Vs = Ks + kBK * kKS;
    float* Ps = Ks;                       // P (kBQ x kBK, stride kBK) reuses the K tile
    __shared__ int s_kb[kBQ], s_ke[kBQ];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int tx = lane & 15;
    const int ty = (tid >> 5) * 2 + (lane >> 4);
    const int q0 = blockIdx.x * kBQ;
    const int hoff = blockIdx.y * kHD;

    if (tid < kBQ) {
        const int r = q0 + tid;
        int kb = 0, ke = 0;
        if (r < total_q) {
            int lo = 0, hi = B - 1;       // last b with cu_q[b] <= r
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (cu_q[mid] <= r) lo = mid; else hi = mid - 1;
            }
            kb = cu_k[lo];
            ke = cu_k[lo + 1];
        }
        s_kb[tid] = kb;
        s_ke[tid] = ke;
    }
    for (int idx = tid; idx < kBQ * (kHD / 4); idx += kThreads) {
        const int r = idx >> 4, d4 = idx & 15;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < total_q) {
            v = *reinterpret_cast<const float4*>(Q + (long long)(q0 + r) * D + hoff + d4 * 4);
            v.x *= q_scale; v.y *= q_scale; v.z *= q_scale; v.w *= q_scale;
        }
        *reinterpret_cast<float4*>(Qs + r * kHD + d4 * 4) = v;
    }
    __syncthreads();

    const int last = min(kBQ, total_q - q0) - 1;
    const int kmin = s_kb[0], kmax = s_ke[last];
    int kb[4], ke[4];
    float m[4], l[4], o[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kb[i] = s_kb[ty * 4 + i];
        ke[i] = s_ke[ty * 4 + i];
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    }

    for (int k0 = kmin; k0 < kmax; k0 += kBK) {
        __syncthreads();                  // the previous tile's P and V reads are done
        for (int idx = tid; idx < kBK * (kHD / 4); idx += kThreads) {
            const int c = idx >> 4, d4 = idx & 15;
            const int g = k0 + c;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (g < kmax) {
                kv = *reinterpret_cast<const float4*>(K + (long long)g * D + hoff + d4 * 4);
                vv = *reinterpret_cast<const float4*>(V + (long long)g * D + hoff + d4 * 4);
            }
            *reinterpret_cast<float4*>(Ks + c * kKS + d4 * 4) = kv;
            *reinterpret_cast<float4*>(Vs + c * kHD + d4 * 4) = vv;
        }
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll 4
        for (int d = 0; d < kHD; d += 4) {
            float4 qa[4], kv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                qa[i] = *reinterpret_cast<const float4*>(Qs + (ty * 4 + i) * kHD + d);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                kv[j] = *reinterpret_cast<const float4*>(Ks + (tx + 16 * j) * kKS + d);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[i][j] = fma4(qa[i], kv[j], s[i][j]);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int g = k0 + tx + 16 * j;
                if (g < kb[i] || g >= ke[i]) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m[i], mx);
            float sum = 0.f;
            if (m_new == -INFINITY) {
#pragma unroll
                for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[i][j] = exp2f(s[i][j] - m_new);
                    sum += s[i][j];
                }
                const float alpha = exp2f(m[i] - m_new);   // 0 while m[i] is -inf
                l[i] *= alpha;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[i][j] *= alpha;
                m[i] = m_new;
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            l[i] += sum;
        }

        __syncthreads();                  // every thread is done reading the K tile
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) Ps[(ty * 4 + i) * kBK + tx + 16 * j] = s[i][j];
        __syncthreads();

#pragma unroll 4
        for (int k = 0; k < kBK; k += 4) {
            float4 pa[4], vb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                pa[i] = *reinterpret_cast<const float4*>(Ps + (ty * 4 + i) * kBK + k);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                vb[e] = *reinterpret_cast<const float4*>(Vs + (k + e) * kHD + tx * 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p[4] = {pa[i].x, pa[i].y, pa[i].z, pa[i].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[i][0] = fmaf(p[e], vb[e].x, o[i][0]);
                    o[i][1] = fmaf(p[e], vb[e].y, o[i][1]);
                    o[i][2] = fmaf(p[e], vb[e].z, o[i][2]);
                    o[i][3] = fmaf(p[e], vb[e].w, o[i][3]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = q0 + ty * 4 + i;
        if (r >= total_q) continue;
        const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        *reinterpret_cast<float4*>(O + (long long)r * D + hoff + tx * 4) =
            make_float4(o[i][0] * inv, o[i][1] * inv, o[i][2] * inv, o[i][3] * inv);
    }
}

}  // namespace

// Returns false (nothing issued) for shapes this kernel does not take.
bool flash_attention_varlen_f32_hd64(const float* Q, const float* K, const float* V,
                                     const int* cu_q, const int* cu_k, float* O,
                                     int total_q, int batch, int num_heads, int head_dim,
                                     bool causal, cudaStream_t stream) {
    const int D = num_heads * head_dim;
    if (head_dim != kHD || causal || batch < 1 || num_heads > 65535) return false;
    const auto aligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (!aligned(Q) || !aligned(K) || !aligned(V) || !aligned(O)) return false;
    static std::atomic<std::uint32_t> opted{0};
    opt_in_dynamic_smem(fa_varlen_f32_hd64_kernel, static_cast<int>(kSmemBytes), opted);
    const float q_scale = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(kHD)));
    const dim3 grid(static_cast<unsigned>((total_q + kBQ - 1) / kBQ), static_cast<unsigned>(num_heads));
    fa_varlen_f32_hd64_kernel<<<grid, kThreads, kSmemBytes, stream>>>(Q, K, V, cu_q, cu_k, O,
                                                                      total_q, batch, D, q_scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace brotensor::detail::cuda
