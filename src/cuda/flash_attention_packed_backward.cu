// Backward of the packed, windowed bidirectional attention
// (flash_attention_packed_qkv_forward): the gradient of O w.r.t. the fused
// (L, 3*H*hd) QKV buffer, under the same per-row sequence bounds and window.
//
// Recompute-based, two launches, one warp per (row, head), FP32 math:
//
//   1. stats: each query row re-runs its online softmax over the keys it
//      attends, keeping the row max m_r and normaliser l_r, and the output
//      O_r it produced, then D_r = dO_r . O_r (= sum_j P_rj dP_rj).
//   2. grads: each row plays both roles.
//        as a query:  dQ_r = scale * sum_j dS_rj k_j
//        as a key:    dK_j = scale * sum_r dS_rj q_r,  dV_j = sum_r P_rj dO_r
//      with P_rj = exp(s_rj - m_r) / l_r and dS_rj = P_rj (dO_r . v_j - D_r).
//      The window is symmetric (|r - j| <= window/2 inside one sequence), so
//      the queries attending key j are found by the same bounds as the keys
//      row j attends.
//
// No P / dS matrix is stored and nothing is accumulated atomically, so the
// result is deterministic and the scratch is 3 floats per (row, head). Each
// lane holds head_dim/32 (rounded up) dimensions of every vector; dot
// products reduce across the warp.
//
// Built for the short sequences of a packed encoder batch (tens to a few
// hundred rows); cost is O(L * span * hd) with span the keys per row.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

constexpr const char* kOp = "flash_attention_packed_qkv_backward";

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + kOp + ": " + reason);
}

cudaStream_t cur_stream() { return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()); }

template <typename T> __device__ __forceinline__ float ld(const T* p);
template <> __device__ __forceinline__ float ld<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
template <typename T> __device__ __forceinline__ T st(float v);
template <> __device__ __forceinline__ float st<float>(float v) { return v; }
template <> __device__ __forceinline__ __half st<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 st<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Lane-strided slice of one head vector: element i of the lane is d = lane + 32 i.
template <typename T, int NPL>
__device__ __forceinline__ void load_vec(const T* base, int lane, int hd, float (&x)[NPL]) {
#pragma unroll
    for (int i = 0; i < NPL; ++i) {
        const int d = lane + 32 * i;
        x[i] = d < hd ? ld<T>(base + d) : 0.0f;
    }
}

template <int NPL>
__device__ __forceinline__ float dot(const float (&a)[NPL], const float (&b)[NPL]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NPL; ++i) s += a[i] * b[i];
    return warp_sum(s);
}

// Keys row r attends (equivalently, queries attending key r): [lo, hi).
__device__ __forceinline__ void span_of(const int* __restrict__ bounds, int r, int hw, int& lo, int& hi) {
    const int start = bounds[2 * r], end = bounds[2 * r + 1];
    lo = hw >= 0 ? max(start, r - hw) : start;
    hi = hw >= 0 ? min(end, r + hw + 1) : end;
}

template <typename T, int NPL>
__global__ void packed_attn_bwd_stats_kernel(const T* __restrict__ qkv, const T* __restrict__ dO,
                                             const int* __restrict__ bounds, float* __restrict__ stats, int L,
                                             int H, int hd, int hw, float scale) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= L * H) return;
    const int r = warp / H, h = warp - r * H;
    const int D = H * hd;
    int lo, hi;
    span_of(bounds, r, hw, lo, hi);

    float q[NPL], k[NPL], v[NPL], o[NPL];
    load_vec<T, NPL>(qkv + static_cast<size_t>(r) * 3 * D + h * hd, lane, hd, q);
#pragma unroll
    for (int i = 0; i < NPL; ++i) o[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int j = lo; j < hi; ++j) {
        const T* row = qkv + static_cast<size_t>(j) * 3 * D;
        load_vec<T, NPL>(row + D + h * hd, lane, hd, k);
        load_vec<T, NPL>(row + 2 * D + h * hd, lane, hd, v);
        const float s = dot<NPL>(q, k) * scale;
        if (s > m) {
            const float f = __expf(m - s);  // exp(-inf) = 0 on the first key
            l = l * f + 1.0f;
#pragma unroll
            for (int i = 0; i < NPL; ++i) o[i] = o[i] * f + v[i];
            m = s;
        } else {
            const float e = __expf(s - m);
            l += e;
#pragma unroll
            for (int i = 0; i < NPL; ++i) o[i] += e * v[i];
        }
    }
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int i = 0; i < NPL; ++i) o[i] *= inv_l;
    float g[NPL];
    load_vec<T, NPL>(dO + static_cast<size_t>(r) * D + h * hd, lane, hd, g);
    const float drow = dot<NPL>(g, o);
    if (lane == 0) {
        float* s = stats + 3 * static_cast<size_t>(warp);
        s[0] = m;
        s[1] = inv_l;
        s[2] = drow;
    }
}

template <typename T, int NPL>
__global__ void packed_attn_bwd_grads_kernel(const T* __restrict__ qkv, const T* __restrict__ dO,
                                             const int* __restrict__ bounds, const float* __restrict__ stats,
                                             T* __restrict__ dqkv, int L, int H, int hd, int hw, float scale) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= L * H) return;
    const int r = warp / H, h = warp - r * H;
    const int D = H * hd;
    int lo, hi;
    span_of(bounds, r, hw, lo, hi);

    float a[NPL], b[NPL];
    const T* rrow = qkv + static_cast<size_t>(r) * 3 * D;
    T* grow = dqkv + static_cast<size_t>(r) * 3 * D;

    // ── row r as a query: dQ_r ──
    {
        float q[NPL], g[NPL], dq[NPL];
        load_vec<T, NPL>(rrow + h * hd, lane, hd, q);
        load_vec<T, NPL>(dO + static_cast<size_t>(r) * D + h * hd, lane, hd, g);
        const float* s = stats + 3 * static_cast<size_t>(warp);
        const float m = s[0], inv_l = s[1], drow = s[2];
#pragma unroll
        for (int i = 0; i < NPL; ++i) dq[i] = 0.0f;
        for (int j = lo; j < hi; ++j) {
            const T* row = qkv + static_cast<size_t>(j) * 3 * D;
            load_vec<T, NPL>(row + D + h * hd, lane, hd, a);      // k_j
            load_vec<T, NPL>(row + 2 * D + h * hd, lane, hd, b);  // v_j
            const float p = __expf(dot<NPL>(q, a) * scale - m) * inv_l;
            const float ds = p * (dot<NPL>(g, b) - drow) * scale;
#pragma unroll
            for (int i = 0; i < NPL; ++i) dq[i] += ds * a[i];
        }
#pragma unroll
        for (int i = 0; i < NPL; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) grow[h * hd + d] = st<T>(dq[i]);
        }
    }

    // ── row r as a key: dK_r, dV_r ──
    {
        float k[NPL], v[NPL], dk[NPL], dv[NPL];
        load_vec<T, NPL>(rrow + D + h * hd, lane, hd, k);
        load_vec<T, NPL>(rrow + 2 * D + h * hd, lane, hd, v);
#pragma unroll
        for (int i = 0; i < NPL; ++i) dk[i] = dv[i] = 0.0f;
        for (int qr = lo; qr < hi; ++qr) {
            load_vec<T, NPL>(qkv + static_cast<size_t>(qr) * 3 * D + h * hd, lane, hd, a);  // q_qr
            load_vec<T, NPL>(dO + static_cast<size_t>(qr) * D + h * hd, lane, hd, b);       // dO_qr
            const float* s = stats + 3 * (static_cast<size_t>(qr) * H + h);
            const float p = __expf(dot<NPL>(a, k) * scale - s[0]) * s[1];
            const float ds = p * (dot<NPL>(b, v) - s[2]) * scale;
#pragma unroll
            for (int i = 0; i < NPL; ++i) {
                dk[i] += ds * a[i];
                dv[i] += p * b[i];
            }
        }
#pragma unroll
        for (int i = 0; i < NPL; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) {
                grow[D + h * hd + d] = st<T>(dk[i]);
                grow[2 * D + h * hd + d] = st<T>(dv[i]);
            }
        }
    }
}

template <typename T, int NPL>
void launch(const Tensor& QKV, const Tensor& dO, const Tensor& bounds, float* stats, Tensor& dQKV, int L, int H,
            int hd, int hw) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    constexpr int kWarps = 4;
    const long long warps = static_cast<long long>(L) * H;
    const int blocks = static_cast<int>((warps + kWarps - 1) / kWarps);
    const T* q = static_cast<const T*>(QKV.data);
    const T* g = static_cast<const T*>(dO.data);
    const int* b = static_cast<const int*>(bounds.data);
    packed_attn_bwd_stats_kernel<T, NPL><<<blocks, 32 * kWarps, 0, cur_stream()>>>(q, g, b, stats, L, H, hd, hw,
                                                                                   scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    packed_attn_bwd_grads_kernel<T, NPL><<<blocks, 32 * kWarps, 0, cur_stream()>>>(
        q, g, b, stats, static_cast<T*>(dQKV.data), L, H, hd, hw, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_dt(const Tensor& QKV, const Tensor& dO, const Tensor& bounds, float* stats, Tensor& dQKV, int L, int H,
               int hd, int hw) {
    if (hd <= 32) launch<T, 1>(QKV, dO, bounds, stats, dQKV, L, H, hd, hw);
    else if (hd <= 64) launch<T, 2>(QKV, dO, bounds, stats, dQKV, L, H, hd, hw);
    else if (hd <= 128) launch<T, 4>(QKV, dO, bounds, stats, dQKV, L, H, hd, hw);
    else if (hd <= 256) launch<T, 8>(QKV, dO, bounds, stats, dQKV, L, H, hd, hw);
    else fail("head_dim above 256 is not supported");
}

}  // namespace

void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV) {
    const Dtype dt = QKV.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail("QKV must be FP16, BF16 or FP32");
    if (dO.dtype != dt) fail("dO must share QKV's dtype");
    if (seq_bounds.dtype != Dtype::INT32) fail("seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail("QKV.cols must be 3 * num_heads * head_dim");
    const int L = QKV.rows;
    const int D = QKV.cols / 3;
    const int hd = D / num_heads;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail("seq_bounds must be (L, 2)");
    if (dO.rows != L || dO.cols != D) fail("dO must be (L, num_heads*head_dim)");
    if (dQKV.rows != L || dQKV.cols != 3 * D || dQKV.dtype != dt) dQKV.resize(L, 3 * D, dt);
    if (L == 0) return;

    Tensor stats = Tensor::empty_on(QKV.device, L * num_heads, 3, Dtype::FP32);
    float* s = static_cast<float*>(stats.data);
    const int hw = window > 0 ? window / 2 : -1;
    if (dt == Dtype::FP16) launch_dt<__half>(QKV, dO, seq_bounds, s, dQKV, L, num_heads, hd, hw);
    else if (dt == Dtype::BF16) launch_dt<__nv_bfloat16>(QKV, dO, seq_bounds, s, dQKV, L, num_heads, hd, hw);
    else launch_dt<float>(QKV, dO, seq_bounds, s, dQKV, L, num_heads, hd, hw);
}

}  // namespace brotensor::detail::cuda
