// Grouped-query single-token decode attention, length read on the device —
// flash_attention_decode_len's fast path.
//
// Decode attention is a stream over the KV cache, so the kernel is shaped for
// bandwidth:
//   * one CTA per (key split, KV head) serves the KV head's whole query group,
//     so each K/V row is read once however many query heads share it (the
//     per-query-head kernel read it `group` times);
//   * a K or V row moves as 16-byte loads, head_dim/8 lanes per row, so a warp
//     covers 32*8/head_dim keys per load and two key steps are in flight per
//     lane;
//   * each lane keeps its own online softmax (max, sum, 8 output columns per
//     query head) over the keys it saw; lanes, warps, then splits merge with
//     the flash-decoding rescale. Scores and accumulation are FP32.
// The split count follows the capacity (a captured graph keeps its launch
// shape) and every split divides the VALID keys, *d_len of them.

#include "decode_gqa.cuh"

#include "detail/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace brotensor::detail::cuda::decode_gqa {

namespace {

constexpr int BLOCK = 128;
constexpr int WARPS = BLOCK / 32;
constexpr int MAX_SPLITS = 64;

template <typename T> __device__ __forceinline__ void load8(const T* p, float (&v)[8]);
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}
template <> __device__ __forceinline__ void load8<__half>(const __half* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

// Running softmax state of one lane for G query heads.
template <int G>
struct State {
    float m[G], l[G], acc[G][8];
    __device__ void init() {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            m[g] = -1e30f;
            l[g] = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[g][i] = 0.0f;
        }
    }
    // Fold another state (m2, l2, a2) for head g into this one.
    __device__ void merge(int g, float m2, float l2, const float (&a2)[8]) {
        const float mn = fmaxf(m[g], m2);
        const float ca = __expf(m[g] - mn), cb = __expf(m2 - mn);
        l[g] = l[g] * ca + l2 * cb;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[g][i] = acc[g][i] * ca + a2[i] * cb;
        m[g] = mn;
    }
};

template <typename T, int HD, int G>
__global__ void __launch_bounds__(BLOCK)
partials_kernel(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
                const int32_t* __restrict__ d_len, float* __restrict__ ws, int cap, int Dkv, int n_splits) {
    constexpr int LPK = HD / 8;         // lanes per key row
    constexpr int KPS = 32 / LPK;       // keys per warp per step
    constexpr int STRIDE = WARPS * KPS; // keys per CTA per step
    const int spl = blockIdx.x, kvh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int sub = lane / LPK, sl = lane % LPK;

    int len = *d_len;
    len = len < 0 ? 0 : (len > cap ? cap : len);
    const int chunk = (len + n_splits - 1) / n_splits;
    const int k_begin = spl * chunk;
    const int k_end = min(len, k_begin + chunk);

    const float inv_sqrt = rsqrtf(static_cast<float>(HD));
    float q[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        load8<T>(Q + static_cast<size_t>(kvh * G + g) * HD + sl * 8, q[g]);
#pragma unroll
        for (int i = 0; i < 8; ++i) q[g][i] *= inv_sqrt;
    }
    State<G> st;
    st.init();

    const size_t col = static_cast<size_t>(kvh) * HD + sl * 8;
    // Warp-uniform loop: two key steps per iteration, both loaded first.
    for (int base = k_begin + warp * KPS; base < k_end; base += 2 * STRIDE) {
        const int k0 = base + sub, k1 = base + STRIDE + sub;
        const bool v0 = k0 < k_end, v1 = k1 < k_end;
        float ka[8], va[8], kb[8], vb[8];
        if (v0) {
            load8<T>(K + static_cast<size_t>(k0) * Dkv + col, ka);
            load8<T>(V + static_cast<size_t>(k0) * Dkv + col, va);
        }
        if (v1) {
            load8<T>(K + static_cast<size_t>(k1) * Dkv + col, kb);
            load8<T>(V + static_cast<size_t>(k1) * Dkv + col, vb);
        }
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const bool valid = step == 0 ? v0 : v1;
            const float(&kk)[8] = step == 0 ? ka : kb;
            const float(&vv)[8] = step == 0 ? va : vb;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float s = 0.0f;
                if (valid) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) s += q[g][i] * kk[i];
                }
#pragma unroll
                for (int off = LPK / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
                if (valid) {
                    const float mn = fmaxf(st.m[g], s);
                    const float c = __expf(st.m[g] - mn), p = __expf(s - mn);
                    st.l[g] = st.l[g] * c + p;
#pragma unroll
                    for (int i = 0; i < 8; ++i) st.acc[g][i] = st.acc[g][i] * c + p * vv[i];
                    st.m[g] = mn;
                }
            }
        }
    }

    // Merge the key slots of the warp (lanes with the same sl).
#pragma unroll
    for (int off = LPK; off < 32; off <<= 1) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float m2 = __shfl_xor_sync(0xffffffffu, st.m[g], off);
            const float l2 = __shfl_xor_sync(0xffffffffu, st.l[g], off);
            float a2[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a2[i] = __shfl_xor_sync(0xffffffffu, st.acc[g][i], off);
            st.merge(g, m2, l2, a2);
        }
    }

    // Merge the warps through shared memory, then write the split record
    // [m, l, acc[HD]] per query head (fad_split_reduce's layout).
    __shared__ float s_ml[WARPS][G][2];
    __shared__ float s_acc[WARPS][G][HD];
    if (sub == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (sl == 0) {
                s_ml[warp][g][0] = st.m[g];
                s_ml[warp][g][1] = st.l[g];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) s_acc[warp][g][sl * 8 + i] = st.acc[g][i];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * (HD + 2); idx += BLOCK) {
        const int g = idx / (HD + 2), e = idx % (HD + 2);
        float m = -1e30f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) m = fmaxf(m, s_ml[w][g][0]);
        float out = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float c = __expf(s_ml[w][g][0] - m);
            out += c * (e == 0 ? 0.0f : e == 1 ? s_ml[w][g][1] : s_acc[w][g][e - 2]);
        }
        if (e == 0) out = m;
        const int h = kvh * G + g;
        ws[(static_cast<size_t>(h) * n_splits + spl) * (HD + 2) + e] = out;
    }
}

template <typename T>
__global__ void reduce_kernel(const float* __restrict__ ws, T* __restrict__ O, int hd, int n_splits) {
    const int h = blockIdx.x;
    const int rec = hd + 2;
    const float* base = ws + static_cast<size_t>(h) * n_splits * rec;
    float m = -1e30f;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, base[s * rec]);
    float den = 0.0f;
    for (int s = 0; s < n_splits; ++s) den += __expf(base[s * rec] - m) * base[s * rec + 1];
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float num = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float w = __expf(base[s * rec] - m);
            if (w != 0.0f) num += w * base[s * rec + 2 + d];
        }
        O[static_cast<size_t>(h) * hd + d] = from_f<T>(num * inv);
    }
}

// Grow-only per-device record scratch (never freed); growth is refused
// during a graph capture (warm up eagerly first).
float* workspace(size_t floats, cudaStream_t stream) {
    constexpr int MAX_DEVICES = 16;
    static float* buf[MAX_DEVICES] = {};
    static size_t capf[MAX_DEVICES] = {};
    int dev = 0;
    BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= MAX_DEVICES) throw std::runtime_error("flash_attention_decode_len: device out of range");
    if (capf[dev] >= floats) return buf[dev];
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    BROTENSOR_CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
    if (cs != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "flash_attention_decode_len: workspace cannot grow during graph capture; run the op once eagerly first");
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (buf[dev]) BROTENSOR_CUDA_CHECK(cudaFree(buf[dev]));
    buf[dev] = nullptr;
    capf[dev] = 0;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf[dev]), floats * sizeof(float)));
    capf[dev] = floats;
    return buf[dev];
}

int sm_count() {
    constexpr int MAX_DEVICES = 16;
    static int n[MAX_DEVICES] = {};
    int dev = 0;
    BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= MAX_DEVICES) return 80;
    if (!n[dev]) BROTENSOR_CUDA_CHECK(cudaDeviceGetAttribute(&n[dev], cudaDevAttrMultiProcessorCount, dev));
    return n[dev];
}

template <typename T, int HD, int G>
void run(const T* Q, const T* K, const T* V, const int32_t* d_len, T* O, int cap, int nq, int nkv,
         cudaStream_t stream) {
    // Splits: about two CTAs per SM over the KV heads, at most one per 32
    // cache rows; fixed by the capacity so a captured launch stays valid.
    int n_splits = (2 * sm_count() + nkv - 1) / nkv;
    n_splits = std::max(1, std::min({n_splits, MAX_SPLITS, (cap + 31) / 32}));
    float* ws = workspace(static_cast<size_t>(nq) * n_splits * (HD + 2), stream);
    partials_kernel<T, HD, G><<<dim3(n_splits, nkv), BLOCK, 0, stream>>>(Q, K, V, d_len, ws, cap, nkv * HD,
                                                                         n_splits);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    reduce_kernel<T><<<nq, BLOCK, 0, stream>>>(ws, O, HD, n_splits);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int HD>
void by_group(const T* Q, const T* K, const T* V, const int32_t* d_len, T* O, int cap, int nq, int nkv,
              cudaStream_t s) {
    switch (nq / nkv) {
        case 1: run<T, HD, 1>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
        case 2: run<T, HD, 2>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
        case 4: run<T, HD, 4>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
        default: run<T, HD, 8>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
    }
}

template <typename T>
void by_hd(const T* Q, const T* K, const T* V, const int32_t* d_len, T* O, int cap, int nq, int nkv, int hd,
           cudaStream_t s) {
    switch (hd) {
        case 64: by_group<T, 64>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
        case 128: by_group<T, 128>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
        default: by_group<T, 256>(Q, K, V, d_len, O, cap, nq, nkv, s); break;
    }
}

}  // namespace

bool supported(int head_dim, int group) {
    return (head_dim == 64 || head_dim == 128 || head_dim == 256) &&
           (group == 1 || group == 2 || group == 4 || group == 8);
}

void launch(const void* Q, const void* K, const void* V, const int32_t* d_len, void* O, int cap, int nq, int nkv,
            int hd, bool bf16, cudaStream_t stream) {
    if (bf16)
        by_hd(static_cast<const __nv_bfloat16*>(Q), static_cast<const __nv_bfloat16*>(K),
              static_cast<const __nv_bfloat16*>(V), d_len, static_cast<__nv_bfloat16*>(O), cap, nq, nkv, hd, stream);
    else
        by_hd(static_cast<const __half*>(Q), static_cast<const __half*>(K), static_cast<const __half*>(V), d_len,
              static_cast<__half*>(O), cap, nq, nkv, hd, stream);
}

}  // namespace brotensor::detail::cuda::decode_gqa
