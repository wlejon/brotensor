// Backward of the bare attention core (flash_attention_backward) and of its
// packed variable-length form (flash_attention_varlen_backward). Both are
// recompute-based: per head (and per sequence) the scores are re-derived from
// Q and K, the softmax is reversed, and dQ / dK / dV are written. O is not
// read. flash_attention_qkvo_backward (flash_attention.cu) runs its attention
// core through flash_attention_backward as well.
//
// 16-bit inputs take a tensor-core pipeline whose precision-sensitive half is
// FP32 throughout:
//
//   S  = Q_h K_h^T      FP32 out of the mma GEMM (mma_gemm::launch_f32out)
//   dP = dO_h V_h^T     FP32 likewise
//   row statistics      max, normaliser and D_q = sum_k P dP, all FP32
//   P, dS               formed in FP32: dS = P (dP - D_q) / sqrt(hd)
//   dV = P^T dO_h,  dQ = dS K_h,  dK = dS^T Q_h
//                       16-bit operands, FP32 accumulation and FP32 output,
//                       narrowed once when scattered into dQ / dK / dV
//
// P and dS are the only 16-bit intermediates. Each is a GEMM operand whose
// rounding is an independent relative error per term, as P is in the
// forward's P@V. What this path used to hold in 16 bits instead -- the raw
// score ahead of the scale and the exp, dP ahead of the dP - D_q difference
// (where most of its bits cancel), D_q's own reduction over 16-bit P and dP,
// and each head's gradient before it was packed -- put the rounding inside the
// softmax and inside that cancellation: 1-2e-2 of the gradient row's scale in
// FP16 and 5e-2 to 1 in BF16, where this pipeline sits at the output rounding.
//
// That pipeline launches per head and per sequence, so short sequences and
// small score matrices take the row path instead (prefer_rows): one warp per
// (row, head), two launches for the whole call, every score, P and dS in FP32
// registers. The row path is also the FP32 path (varlen only) and the fallback
// on devices where the FP32-output GEMM does not run (pre-sm_80).

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"
#include "gemm_mma.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace brotensor {

void* cuda_current_stream();

namespace {

inline int grid_for(long long n, int block) {
    long long b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return static_cast<int>(b);
}

inline int round8(int n) { return (n + 7) & ~7; }

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T> constexpr Dtype dtype_of();
template <> constexpr Dtype dtype_of<__half>() { return Dtype::FP16; }
template <> constexpr Dtype dtype_of<__nv_bfloat16>() { return Dtype::BF16; }

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Block-wide max / sum over blockDim.x (a multiple of 32) threads; every
// thread gets the result. `red` holds 32 floats.
__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : neg_inf();
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    return v;
}

// ─── Tensor-core pipeline (FP16 / BF16) ─────────────────────────────────────

// Y[r, d] (row stride ldY) = X[r, col0 + d] for r < rows, d < hd; the pad rows
// [rows, rows_pad) and pad columns [hd, ldY) are written as zeros.
template <typename T>
__global__ void bw_gather_kernel(const T* __restrict__ X, int ldX, int col0, int rows, int hd,
                                 T* __restrict__ Y, int rows_pad, int ldY) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows_pad) * ldY) return;
    const int r = static_cast<int>(idx / ldY), d = static_cast<int>(idx % ldY);
    Y[idx] = (r < rows && d < hd) ? X[static_cast<size_t>(r) * ldX + col0 + d] : from_f32<T>(0.0f);
}

// Transposed gather: Y[d, r] (row stride ldY) = X[r, col0 + d] for d < hdp,
// r < rows_pad, zeros in the pads. dup != 0 writes a second copy at column
// offset rows_pad -- the [X; X] operand the hi / lo dS pair multiplies.
template <typename T>
__global__ void bw_gather_T_kernel(const T* __restrict__ X, int ldX, int col0, int rows, int hd,
                                   T* __restrict__ Y, int rows_pad, int hdp, int ldY, int dup) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(hdp) * rows_pad) return;
    const int d = static_cast<int>(idx / rows_pad), r = static_cast<int>(idx % rows_pad);
    const T v = (r < rows && d < hd) ? X[static_cast<size_t>(r) * ldX + col0 + d] : from_f32<T>(0.0f);
    T* y = Y + static_cast<size_t>(d) * ldY + r;
    y[0] = v;
    if (dup) y[rows_pad] = v;
}

// The power-of-two scale that puts the chunk's largest |dS| bound near 2^14:
// dS * scale is far from FP16's overflow and keeps ~28 binades of normal range
// below it. `bits` is the bound's FP32 bit pattern (bw_row_stats_kernel).
__device__ __forceinline__ float bw_ds_scale(const unsigned int* bits) {
    const float b = __uint_as_float(*bits);
    if (!(b > 0.0f) || !isfinite(b)) return 1.0f;
    int e = 0;
    frexpf(b, &e);                              // b < 2^e
    return ldexpf(1.0f, max(-120, min(120, 14 - e)));
}

// Fixed scale of P in its 16-bit GEMM operand: P <= 1 becomes <= 2^14, so FP16
// holds probabilities down to 2^-28 at full precision instead of 2^-14.
constexpr float kPScale = 16384.0f;

// Out[r, col0 + d] = Y[r, d] / scale (FP32, row stride ldY) for r < rows,
// d < hd, with scale from bw_ds_scale.
template <typename T>
__global__ void bw_scatter_kernel(const float* __restrict__ Y, int ldY, int rows, int hd,
                                  T* __restrict__ Out, int ldO, int col0, const unsigned int* __restrict__ sbits) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows) * hd) return;
    const int r = static_cast<int>(idx / hd), d = static_cast<int>(idx % hd);
    const float inv = sbits ? 1.0f / bw_ds_scale(sbits) : 1.0f;
    Out[static_cast<size_t>(r) * ldO + col0 + d] = from_f32<T>(Y[static_cast<size_t>(r) * ldY + d] * inv);
}

// acc = (first ? 0 : acc) + src * inv, inv = 1 / bw_ds_scale(sbits) when sbits
// is given, else inv_const. Every scale is a power of two, so the unscaling is
// exact.
__global__ void bw_acc_kernel(float* __restrict__ acc, const float* __restrict__ src, long long n,
                              const unsigned int* __restrict__ sbits, float inv_const, int first) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float inv = sbits ? 1.0f / bw_ds_scale(sbits) : inv_const;
    const float v = src[idx] * inv;
    acc[idx] = first ? v : acc[idx] + v;
}

// Key k is attended by chunk row i (absolute query q0 + i).
__device__ __forceinline__ bool bw_key_on(const float* mask, int causal, int k, int qa) {
    return !(mask && mask[k] <= 0.5f) && !(causal && k > qa);
}

// Per chunk row i (one block each): the row's max of the scaled log2 scores,
// 1 / normaliser, and D_q = sum_k P dP -- all from the FP32 S / dP rows.
// A row with no key attended gets stats (0, 0, 0), so its P and dS are zero.
// Also folds scale * (max_k |dP| + |D_q|), a bound on the row's |dS| (P <= 1),
// into *bound_bits with an integer max (valid for non-negative floats).
__global__ void bw_row_stats_kernel(const float* __restrict__ S, const float* __restrict__ dP, int ld, int Lk,
                                    int q0, float scale_log2, float scale, const float* __restrict__ mask,
                                    int causal, float* __restrict__ stats, unsigned int* __restrict__ bound_bits) {
    __shared__ float red[32];
    const int i = blockIdx.x;
    const float* s = S + static_cast<size_t>(i) * ld;
    const float* dp = dP + static_cast<size_t>(i) * ld;
    const int qa = q0 + i;
    const int kend = causal ? min(Lk, qa + 1) : Lk;
    float mx = neg_inf();
    for (int k = threadIdx.x; k < kend; k += blockDim.x)
        if (bw_key_on(mask, 0, k, qa)) mx = fmaxf(mx, s[k] * scale_log2);
    mx = block_max(mx, red);
    float l = 0.0f, a = 0.0f, g = 0.0f;
    if (mx != neg_inf()) {
        for (int k = threadIdx.x; k < kend; k += blockDim.x) {
            if (!bw_key_on(mask, 0, k, qa)) continue;
            const float e = exp2f(fmaf(s[k], scale_log2, -mx));
            l += e;
            a = fmaf(e, dp[k], a);
            g = fmaxf(g, fabsf(dp[k]));
        }
    }
    l = block_sum(l, red);
    a = block_sum(a, red);
    g = block_max(g, red);
    if (threadIdx.x == 0) {
        const bool any = l > 0.0f;
        const float Dq = any ? a / l : 0.0f;
        stats[3 * i + 0] = any ? mx : 0.0f;
        stats[3 * i + 1] = any ? 1.0f / l : 0.0f;
        stats[3 * i + 2] = Dq;
        const float b = any ? scale * (g + fabsf(Dq)) : 0.0f;
        if (isfinite(b)) atomicMax(bound_bits, __float_as_uint(b));
    }
}

// One 32 x 32 tile of (chunk row i, key j) per block of 32 x 8 threads:
// P = exp(s - max) / l and dS = P (dP - D_q) * scale in FP32, then narrowed for
// the three GEMMs that consume them. dS, scaled by bw_ds_scale, is split into
// a 16-bit hi part and the 16-bit rounding of its remainder (lo): the dQ / dK
// GEMMs multiply [hi | lo] by [K ; K] (resp. [Q ; Q]), which carries dS to
// ~2^-22 (FP16) / 2^-16 (BF16) instead of one 16-bit rounding. dS enters dQ and
// dK through sums whose terms cancel (sum_k dS = 0), so a single rounding is
// amplified exactly where the softmax is peaked.
//   dS  (m, 2 Lk_pad):     row i = [hi | lo] for dQ
//   dST (Lk_pad, 2 m_pad): row j = [hi | lo] for dK
//   PT  (Lk_pad, m_pad):   P * kPScale for dV
// The transposes go through shared memory so both stores coalesce. Masked,
// causal-excluded and pad positions are exact zeros.
template <typename T>
__global__ void bw_ds_kernel(const float* __restrict__ S, const float* __restrict__ dP, int ldS,
                             const float* __restrict__ stats, const unsigned int* __restrict__ sbits, int m,
                             int m_pad, int Lk, int Lk_pad, int q0, float scale_log2, float scale,
                             const float* __restrict__ mask, int causal, T* __restrict__ dS, T* __restrict__ dST,
                             T* __restrict__ PT) {
    __shared__ float tp[32][33], tds[32][33];
    const int j0 = blockIdx.x * 32, i0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int j = j0 + tx;
    const bool jon = j < Lk && !(mask && mask[j] <= 0.5f);
    const float dsc = bw_ds_scale(sbits) * scale;
    for (int r = ty; r < 32; r += 8) {
        const int i = i0 + r;
        float p = 0.0f, ds = 0.0f;
        if (i < m && jon && !(causal && j > q0 + i)) {
            const float* st = stats + 3 * i;
            const size_t o = static_cast<size_t>(i) * ldS + j;
            if (st[1] > 0.0f) {
                p = exp2f(fmaf(S[o], scale_log2, -st[0])) * st[1];
                ds = p * (dP[o] - st[2]) * dsc;
            }
        }
        if (i < m && j < Lk_pad) {
            const T hi = from_f32<T>(ds);
            T* row = dS + static_cast<size_t>(i) * 2 * Lk_pad;
            row[j] = hi;
            row[Lk_pad + j] = from_f32<T>(ds - to_f32(hi));
        }
        tp[r][tx] = p;
        tds[r][tx] = ds;
    }
    __syncthreads();
    for (int r = ty; r < 32; r += 8) {
        const int jj = j0 + r, ii = i0 + tx;
        if (jj < Lk_pad && ii < m_pad) {
            PT[static_cast<size_t>(jj) * m_pad + ii] = from_f32<T>(tp[tx][r] * kPScale);
            const float ds = tds[tx][r];
            const T hi = from_f32<T>(ds);
            T* row = dST + static_cast<size_t>(jj) * 2 * m_pad;
            row[ii] = hi;
            row[m_pad + ii] = from_f32<T>(ds - to_f32(hi));
        }
    }
}

// Score elements (rows x Lk_pad) per query chunk. Each costs 18 bytes of
// scratch (FP32 S and dP, the 16-bit hi / lo dS and dS^T, 16-bit P^T):
// 8 M -> 144 MB.
constexpr size_t kBwdScoreBudget = size_t(8) << 20;

// dQ / dK / dV of one block of rows: Q, dO (Lq rows), K, V (Lk rows), row
// stride ld, head h at columns [h*hd, (h+1)*hd); the gradients are written at
// the same strides. Returns false, having written nothing, when the
// FP32-output GEMM cannot run on this device.
template <typename T>
bool tc_backward(const T* Q, const T* K, const T* V, const T* dO, const float* mask, bool causal, T* dQ, T* dK,
                 T* dV, int Lq, int Lk, int ld, int nh, int hd, cudaStream_t stream) {
    namespace mg = detail::cuda::mma_gemm;
    const Dtype dt = dtype_of<T>();
    const int hdp = round8(hd), Lk_pad = round8(Lk);
    int rows = Lq;
    if (static_cast<size_t>(Lq) * Lk_pad > kBwdScoreBudget) {
        rows = static_cast<int>(kBwdScoreBudget / Lk_pad) / 128 * 128;
        if (rows < 128) rows = 128;
        if (rows > Lq) rows = Lq;
    }
    const int rows_pad = round8(rows);

    auto buf = [&](size_t n, Dtype d) { return Tensor::empty_on(Device::CUDA, static_cast<int>(n), 1, d); };
    const size_t kd = static_cast<size_t>(Lk_pad) * hdp, qd = static_cast<size_t>(rows_pad) * hdp;
    const size_t sc = static_cast<size_t>(rows) * Lk_pad, tc = static_cast<size_t>(Lk_pad) * rows_pad;
    Tensor Kh = buf(kd, dt), Vh = buf(kd, dt), KhT = buf(2 * kd, dt);
    Tensor Qh = buf(qd, dt), dOh = buf(qd, dt), QhT = buf(2 * qd, dt), dOhT = buf(qd, dt);
    Tensor Sb = buf(sc, Dtype::FP32), dPb = buf(sc, Dtype::FP32), dSb = buf(2 * sc, dt);
    Tensor dSTb = buf(2 * tc, dt), PTb = buf(tc, dt), stats = buf(static_cast<size_t>(rows) * 3, Dtype::FP32);
    Tensor dq32 = buf(qd, Dtype::FP32), dk32 = buf(kd, Dtype::FP32), dv32 = buf(kd, Dtype::FP32);
    Tensor tmp = buf(2 * kd, Dtype::FP32), bound = buf(1, Dtype::FP32);

    T* kh = static_cast<T*>(Kh.data);
    T* vh = static_cast<T*>(Vh.data);
    T* khT = static_cast<T*>(KhT.data);
    T* qh = static_cast<T*>(Qh.data);
    T* doh = static_cast<T*>(dOh.data);
    T* qhT = static_cast<T*>(QhT.data);
    T* dohT = static_cast<T*>(dOhT.data);
    float* s = static_cast<float*>(Sb.data);
    float* dp = static_cast<float*>(dPb.data);
    T* ds = static_cast<T*>(dSb.data);
    T* dsT = static_cast<T*>(dSTb.data);
    T* pT = static_cast<T*>(PTb.data);
    float* st = static_cast<float*>(stats.data);
    float* dq = static_cast<float*>(dq32.data);
    float* dk = static_cast<float*>(dk32.data);
    float* dv = static_cast<float*>(dv32.data);
    float* tk = static_cast<float*>(tmp.data);
    float* tv = tk + kd;
    unsigned int* bb = static_cast<unsigned int*>(bound.data);

    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const float scale_log2 = scale * 1.4426950408889634f;
    constexpr int CP = 256;
    const int cz = causal ? 1 : 0;
    const long long kdl = static_cast<long long>(kd);

    for (int h = 0; h < nh; ++h) {
        const int c0 = h * hd;
        bw_gather_kernel<T><<<grid_for(kd, CP), CP, 0, stream>>>(K, ld, c0, Lk, hd, kh, Lk_pad, hdp);
        bw_gather_kernel<T><<<grid_for(kd, CP), CP, 0, stream>>>(V, ld, c0, Lk, hd, vh, Lk_pad, hdp);
        bw_gather_T_kernel<T><<<grid_for(kd, CP), CP, 0, stream>>>(K, ld, c0, Lk, hd, khT, Lk_pad, hdp,
                                                                   2 * Lk_pad, 1);
        for (int q0 = 0; q0 < Lq; q0 += rows) {
            const int m = Lq - q0 < rows ? Lq - q0 : rows;
            const int m_pad = round8(m);
            const size_t mo = static_cast<size_t>(q0) * ld;
            const long long mq = static_cast<long long>(m) * hdp, mqT = static_cast<long long>(m_pad) * hdp;
            bw_gather_kernel<T><<<grid_for(mq, CP), CP, 0, stream>>>(Q + mo, ld, c0, m, hd, qh, m, hdp);
            bw_gather_kernel<T><<<grid_for(mq, CP), CP, 0, stream>>>(dO + mo, ld, c0, m, hd, doh, m, hdp);
            bw_gather_T_kernel<T><<<grid_for(mqT, CP), CP, 0, stream>>>(Q + mo, ld, c0, m, hd, qhT, m_pad, hdp,
                                                                        2 * m_pad, 1);
            bw_gather_T_kernel<T><<<grid_for(mqT, CP), CP, 0, stream>>>(dO + mo, ld, c0, m, hd, dohT, m_pad, hdp,
                                                                        m_pad, 0);

            // Only the first GEMM of the call can refuse (the device check);
            // nothing has been written to dQ / dK / dV by then.
            if (!mg::launch_f32out(qh, kh, s, m, Lk_pad, hdp, stream)) return false;
            mg::launch_f32out(doh, vh, dp, m, Lk_pad, hdp, stream);
            BROTENSOR_CUDA_CHECK(cudaMemsetAsync(bb, 0, sizeof(unsigned int), stream));
            bw_row_stats_kernel<<<m, 256, 0, stream>>>(s, dp, Lk_pad, Lk, q0, scale_log2, scale, mask, cz, st, bb);
            {
                dim3 block(32, 8);
                dim3 grid((Lk_pad + 31) / 32, (m_pad + 31) / 32);
                bw_ds_kernel<T><<<grid, block, 0, stream>>>(s, dp, Lk_pad, st, bb, m, m_pad, Lk, Lk_pad, q0,
                                                            scale_log2, scale, mask, cz, ds, dsT, pT);
            }
            mg::launch_f32out(ds, khT, dq, m, hdp, 2 * Lk_pad, stream);
            bw_scatter_kernel<T><<<grid_for(static_cast<long long>(m) * hd, CP), CP, 0, stream>>>(
                dq, hdp, m, hd, dQ + mo, ld, c0, bb);
            const int first = q0 == 0;
            mg::launch_f32out(dsT, qhT, tk, Lk_pad, hdp, 2 * m_pad, stream);
            mg::launch_f32out(pT, dohT, tv, Lk_pad, hdp, m_pad, stream);
            bw_acc_kernel<<<grid_for(kd, CP), CP, 0, stream>>>(dk, tk, kdl, bb, 1.0f, first);
            bw_acc_kernel<<<grid_for(kd, CP), CP, 0, stream>>>(dv, tv, kdl, nullptr, 1.0f / kPScale, first);
        }
        const long long kh_n = static_cast<long long>(Lk) * hd;
        bw_scatter_kernel<T><<<grid_for(kh_n, CP), CP, 0, stream>>>(dk, hdp, Lk, hd, dK, ld, c0, nullptr);
        bw_scatter_kernel<T><<<grid_for(kh_n, CP), CP, 0, stream>>>(dv, hdp, Lk, hd, dV, ld, c0, nullptr);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ─── Row path (FP32 math, 3 floats of scratch per row and head) ─────────────
//
// One warp per (row, head), after flash_attention_packed_qkv_backward:
//   rw_query_kernel  each query row re-runs its softmax online (max, normaliser
//                    and its output O_r), stores (max, 1/l, D_r = dO_r . O_r),
//                    then walks its keys again for dQ_r = scale sum_k dS_rk k_k.
//   rw_key_kernel    each key row walks the queries attending it for
//                    dK_k = scale sum_r dS_rk q_r and dV_k = sum_r P_rk dO_r.
// Scores, P and dS live in FP32 registers; nothing is accumulated atomically,
// so the result is deterministic. Two launches cover every head and every
// varlen sequence, where the tensor-core pipeline launches per head and per
// sequence: this path is the faster one until the per-sequence score matrix
// is large (kRowsMaxScores). It is also the FP32 path, and the fallback
// wherever the FP32-output GEMM does not run.

// Warps per block. Consecutive warps take consecutive rows of one head, so a
// block's warps walk the same keys (queries) together through L1.
constexpr int kRowWarps = 8;

// The sequence of packed row r under offsets cu[0..B]: the b with
// cu[b] <= r < cu[b + 1] (an empty sequence shares its offset with the next).
__device__ __forceinline__ int rw_seq_of(const int32_t* __restrict__ cu, int B, int r) {
    int lo = 0, hi = B;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu[mid] <= r) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float rw_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Lane-strided slice of one head vector: element i of the lane is d = lane + 32 i.
template <typename T, int NPL>
__device__ __forceinline__ void rw_load(const T* __restrict__ p, int lane, int hd, float (&x)[NPL]) {
#pragma unroll
    for (int i = 0; i < NPL; ++i) {
        const int d = lane + 32 * i;
        x[i] = d < hd ? to_f32(p[d]) : 0.0f;
    }
}

template <int NPL>
__device__ __forceinline__ float rw_dot(const float (&a)[NPL], const float (&b)[NPL]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NPL; ++i) s = fmaf(a[i], b[i], s);
    return rw_warp_sum(s);
}

template <typename T, int NPL>
__device__ __forceinline__ void rw_store(T* __restrict__ p, int lane, int hd, const float (&x)[NPL], float mul) {
#pragma unroll
    for (int i = 0; i < NPL; ++i) {
        const int d = lane + 32 * i;
        if (d < hd) p[d] = from_f32<T>(x[i] * mul);
    }
}

// Rows [q0, q1) of the packed query axis attend keys [k0, k1). cu_q == nullptr
// is the single block (0, nq) x (0, nk), where the key mask applies.
struct RwSeg {
    int q0, q1, k0, k1;
};
__device__ __forceinline__ RwSeg rw_seg(const int32_t* cu_q, const int32_t* cu_k, int b, int nq, int nk) {
    if (!cu_q) return {0, nq, 0, nk};
    return {cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]};
}

template <typename T, int NPL>
__global__ void rw_query_kernel(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
                                const T* __restrict__ dO, int ld, int H, int hd, const int32_t* __restrict__ cu_q,
                                const int32_t* __restrict__ cu_k, int B, int nq, int nk,
                                const float* __restrict__ mask, int causal, float scale, float scale_log2,
                                float* __restrict__ stats, T* __restrict__ dQ) {
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= static_cast<long long>(nq) * H) return;
    const int h = static_cast<int>(warp / nq), r = static_cast<int>(warp % nq);
    const RwSeg sg = rw_seg(cu_q, cu_k, cu_q ? rw_seq_of(cu_q, B, r) : 0, nq, nk);
    const int k1 = causal ? min(sg.k1, sg.k0 + (r - sg.q0) + 1) : sg.k1;
    const int c0 = h * hd;

    float q[NPL], g[NPL], kk[NPL], vv[NPL], acc[NPL];
    rw_load<T, NPL>(Q + static_cast<size_t>(r) * ld + c0, lane, hd, q);
    rw_load<T, NPL>(dO + static_cast<size_t>(r) * ld + c0, lane, hd, g);
#pragma unroll
    for (int i = 0; i < NPL; ++i) acc[i] = 0.0f;

    // Pass 1: max (log2 domain), normaliser and O_r, online.
    float m = neg_inf(), l = 0.0f;
    for (int k = sg.k0; k < k1; ++k) {
        if (mask && mask[k - sg.k0] <= 0.5f) continue;
        const T* kr = K + static_cast<size_t>(k) * ld + c0;
        rw_load<T, NPL>(kr, lane, hd, kk);
        rw_load<T, NPL>(V + static_cast<size_t>(k) * ld + c0, lane, hd, vv);
        const float s = rw_dot<NPL>(q, kk) * scale_log2;
        if (s > m) {
            const float f = exp2f(m - s);  // 0 on the first key
            l = fmaf(l, f, 1.0f);
#pragma unroll
            for (int i = 0; i < NPL; ++i) acc[i] = fmaf(acc[i], f, vv[i]);
            m = s;
        } else {
            const float e = exp2f(s - m);
            l += e;
#pragma unroll
            for (int i = 0; i < NPL; ++i) acc[i] = fmaf(e, vv[i], acc[i]);
        }
    }
    const bool any = l > 0.0f;
    const float inv_l = any ? 1.0f / l : 0.0f;
    const float D = rw_dot<NPL>(g, acc) * inv_l;
    if (!any) m = 0.0f;
    if (lane == 0) {
        float* st = stats + 3 * static_cast<size_t>(warp);
        st[0] = m;
        st[1] = inv_l;
        st[2] = D;
    }

    // Pass 2: dQ_r = scale * sum_k P (dP - D) k_k.
#pragma unroll
    for (int i = 0; i < NPL; ++i) acc[i] = 0.0f;
    if (any) {
        for (int k = sg.k0; k < k1; ++k) {
            if (mask && mask[k - sg.k0] <= 0.5f) continue;
            rw_load<T, NPL>(K + static_cast<size_t>(k) * ld + c0, lane, hd, kk);
            rw_load<T, NPL>(V + static_cast<size_t>(k) * ld + c0, lane, hd, vv);
            const float p = exp2f(fmaf(rw_dot<NPL>(q, kk), scale_log2, -m)) * inv_l;
            const float ds = p * (rw_dot<NPL>(g, vv) - D);
#pragma unroll
            for (int i = 0; i < NPL; ++i) acc[i] = fmaf(ds, kk[i], acc[i]);
        }
    }
    rw_store<T, NPL>(dQ + static_cast<size_t>(r) * ld + c0, lane, hd, acc, scale);
}

template <typename T, int NPL>
__global__ void rw_key_kernel(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
                              const T* __restrict__ dO, int ld, int H, int hd, const int32_t* __restrict__ cu_q,
                              const int32_t* __restrict__ cu_k, int B, int nq, int nk,
                              const float* __restrict__ mask, int causal, float scale, float scale_log2,
                              const float* __restrict__ stats, T* __restrict__ dK, T* __restrict__ dV) {
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= static_cast<long long>(nk) * H) return;
    const int h = static_cast<int>(warp / nk), j = static_cast<int>(warp % nk);
    const RwSeg sg = rw_seg(cu_q, cu_k, cu_k ? rw_seq_of(cu_k, B, j) : 0, nq, nk);
    const int kj = j - sg.k0;
    const int c0 = h * hd;
    const float* sth = stats + 3 * static_cast<size_t>(h) * nq;

    float kk[NPL], vv[NPL], q[NPL], g[NPL], dk[NPL], dv[NPL];
#pragma unroll
    for (int i = 0; i < NPL; ++i) dk[i] = dv[i] = 0.0f;
    if (!(mask && mask[kj] <= 0.5f)) {
        rw_load<T, NPL>(K + static_cast<size_t>(j) * ld + c0, lane, hd, kk);
        rw_load<T, NPL>(V + static_cast<size_t>(j) * ld + c0, lane, hd, vv);
        for (int r = causal ? sg.q0 + kj : sg.q0; r < sg.q1; ++r) {
            const float* st = sth + 3 * static_cast<size_t>(r);
            rw_load<T, NPL>(Q + static_cast<size_t>(r) * ld + c0, lane, hd, q);
            rw_load<T, NPL>(dO + static_cast<size_t>(r) * ld + c0, lane, hd, g);
            const float p = exp2f(fmaf(rw_dot<NPL>(q, kk), scale_log2, -st[0])) * st[1];
            const float ds = p * (rw_dot<NPL>(g, vv) - st[2]);
#pragma unroll
            for (int i = 0; i < NPL; ++i) {
                dk[i] = fmaf(ds, q[i], dk[i]);
                dv[i] = fmaf(p, g[i], dv[i]);
            }
        }
    }
    rw_store<T, NPL>(dK + static_cast<size_t>(j) * ld + c0, lane, hd, dk, scale);
    rw_store<T, NPL>(dV + static_cast<size_t>(j) * ld + c0, lane, hd, dv, 1.0f);
}

template <typename T, int NPL>
void rows_launch(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& dO, const int32_t* cu_q,
                 const int32_t* cu_k, int B, const float* mask, bool causal, int H, int hd, Tensor& dQ, Tensor& dK,
                 Tensor& dV, cudaStream_t stream) {
    const int nq = Q.rows, nk = K.rows, ld = Q.cols;
    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const float scale_log2 = scale * 1.4426950408889634f;
    Tensor stats = Tensor::empty_on(Device::CUDA, nq * H, 3, Dtype::FP32);
    const T* q = static_cast<const T*>(Q.data);
    const T* k = static_cast<const T*>(K.data);
    const T* v = static_cast<const T*>(V.data);
    const T* g = static_cast<const T*>(dO.data);
    float* st = static_cast<float*>(stats.data);
    constexpr int threads = 32 * kRowWarps;
    const int cz = causal ? 1 : 0;
    rw_query_kernel<T, NPL><<<grid_for(static_cast<long long>(nq) * H * 32, threads), threads, 0, stream>>>(
        q, k, v, g, ld, H, hd, cu_q, cu_k, B, nq, nk, mask, cz, scale, scale_log2, st, static_cast<T*>(dQ.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    rw_key_kernel<T, NPL><<<grid_for(static_cast<long long>(nk) * H * 32, threads), threads, 0, stream>>>(
        q, k, v, g, ld, H, hd, cu_q, cu_k, B, nq, nk, mask, cz, scale, scale_log2, st, static_cast<T*>(dK.data),
        static_cast<T*>(dV.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void rows_backward_t(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& dO, const int32_t* cu_q,
                     const int32_t* cu_k, int B, const float* mask, bool causal, int H, int hd, Tensor& dQ,
                     Tensor& dK, Tensor& dV, cudaStream_t stream) {
#define BT_RW(N) rows_launch<T, N>(Q, K, V, dO, cu_q, cu_k, B, mask, causal, H, hd, dQ, dK, dV, stream)
    switch ((hd + 31) / 32) {
    case 1: BT_RW(1); break;
    case 2: BT_RW(2); break;
    case 3: BT_RW(3); break;
    case 4: BT_RW(4); break;
    case 5: BT_RW(5); break;
    case 6: BT_RW(6); break;
    case 7:
    case 8: BT_RW(8); break;
    default:
        throw std::runtime_error("flash_attention_backward: head_dim above 256 needs the tensor-core path (sm_80+)");
    }
#undef BT_RW
}

// The row path over the whole call: Q / dO / dQ have Q.rows rows, K / V / dK /
// dV K.rows, row stride Q.cols. cu_q / cu_k (device, B + 1 offsets) or, when
// null, one block with the key mask.
void rows_backward(Dtype dt, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& dO,
                   const int32_t* cu_q, const int32_t* cu_k, int B, const float* mask, bool causal, int H, int hd,
                   Tensor& dQ, Tensor& dK, Tensor& dV, cudaStream_t stream) {
    if (dt == Dtype::FP32)
        rows_backward_t<float>(Q, K, V, dO, cu_q, cu_k, B, mask, causal, H, hd, dQ, dK, dV, stream);
    else if (dt == Dtype::BF16)
        rows_backward_t<__nv_bfloat16>(Q, K, V, dO, cu_q, cu_k, B, mask, causal, H, hd, dQ, dK, dV, stream);
    else
        rows_backward_t<__half>(Q, K, V, dO, cu_q, cu_k, B, mask, causal, H, hd, dQ, dK, dV, stream);
}

// One block of rows (query rows from q_row, key rows from k_row) through the
// tensor-core pipeline; false, having written nothing, when it cannot run.
bool tc_block(Dtype dt, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& dO, const float* mask,
              bool causal, Tensor& dQ, Tensor& dK, Tensor& dV, size_t q_row, size_t k_row, int Lq, int Lk, int D,
              int nh, int hd, cudaStream_t stream) {
    const size_t qo = q_row * D, ko = k_row * D;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        return tc_backward<T>(static_cast<const T*>(Q.data) + qo, static_cast<const T*>(K.data) + ko,
                              static_cast<const T*>(V.data) + ko, static_cast<const T*>(dO.data) + qo, mask,
                              causal, static_cast<T*>(dQ.data) + qo, static_cast<T*>(dK.data) + ko,
                              static_cast<T*>(dV.data) + ko, Lq, Lk, D, nh, hd, stream);
    };
    if (dt == Dtype::BF16) return run(__nv_bfloat16{});
    if (dt == Dtype::FP16) return run(__half{});
    return false;  // no FP32 tensor-core GEMM
}

// Row path or tensor-core pipeline, from a cost model fitted to
// bench_attention_head_dims bwd (RTX 4090, Windows):
//   rows  ~ c(hd) ns per attended (query, key, head) triple, c = 0.26 up to
//           hd 64 and +0.1 per further 32 dims (shuffle-bound dot products),
//           plus ~0.2 us per step of the longest serial walk (a query row walks
//           its keys twice, a key row its queries once);
//   tc    ~ (0.055 + 1e-4 hd) ns per (query, key, head) of the full score
//           matrix, but never below ~0.2 ms per (sequence, head): that
//           block's ~19 launches and scratch allocations.
// So short sequences, few queries against many heads' worth of launches, and
// small score matrices take the rows; large per-head score matrices the GEMMs.
// BROTENSOR_ATTN_BWD_PATH=rows|tc forces one (A/B comparisons and tests); read
// per call so a process can flip it.
bool prefer_rows(const std::vector<int32_t>& cq, const std::vector<int32_t>& ck, int nh, int hd, bool causal) {
    if (const char* e = std::getenv("BROTENSOR_ATTN_BWD_PATH")) {
        if (e[0] == 'r') return true;
        if (e[0] == 't') return false;
    }
    const int npl = (hd + 31) / 32;
    if (npl > 8) return false;
    double rows_pairs = 0.0, tc_pairs = 0.0, blocks = 0.0;
    int max_q = 0, max_k = 0;
    for (size_t b = 0; b + 1 < cq.size(); ++b) {
        const double lq = cq[b + 1] - cq[b], lk = ck[b + 1] - ck[b];
        if (lq <= 0.0 || lk <= 0.0) continue;
        tc_pairs += lq * lk;
        rows_pairs += causal ? lq * (lq + 1.0) * 0.5 : lq * lk;
        blocks += 1.0;
        max_q = std::max(max_q, cq[b + 1] - cq[b]);
        max_k = std::max(max_k, ck[b + 1] - ck[b]);
    }
    const double c_rows = 0.26 + 0.1 * std::max(0, npl - 2);
    const double rows_ms = rows_pairs * nh * c_rows * 1e-6 + (2.0 * max_k + max_q) * 0.2e-3;
    const double tc_ms = std::max(tc_pairs * nh * (0.055 + 1e-4 * hd) * 1e-6, blocks * nh * 0.2);
    return rows_ms <= tc_ms;
}

} // namespace

namespace detail::cuda {

// O is not read: the backward recomputes the softmax from Q and K. It stays in
// the signature to mirror standard flash-attention backward APIs.
void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV) {
    (void)O;
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO dtype must match");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != dt) dQ.resize(Lq, D, dt);
    if (dK.rows != Lk || dK.cols != D || dK.dtype != dt) dK.resize(Lk, D, dt);
    if (dV.rows != Lk || dV.cols != D || dV.dtype != dt) dV.resize(Lk, D, dt);
    if (Lq == 0 || Lk == 0 || D == 0) {
        dQ.zero();
        dK.zero();
        dV.zero();
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (!prefer_rows({0, Lq}, {0, Lk}, num_heads, hd, causal) &&
        tc_block(dt, Q, K, V, dO, d_mask, causal, dQ, dK, dV, 0, 0, Lq, Lk, D, num_heads, hd, stream))
        return;
    rows_backward(dt, Q, K, V, dO, nullptr, nullptr, 1, d_mask, causal, num_heads, hd, dQ, dK, dV, stream);
}

// ─── flash_attention_varlen_backward ────────────────────────────────────────
//
// Packed variable-length backward. The row path takes the whole batch in its
// two launches (each row finds its sequence in cu_seqlens); the tensor-core
// pipeline runs each sequence as one block (pointer offsets into the
// (total_tokens, D) buffers). Either way the math per sequence is
// flash_attention_backward's.
//
// cu_seqlens_q/k are DEVICE pointers (matching the varlen forward); they are
// copied host-side once per call (B+1 INT32s) to validate them and to choose
// the path.
void flash_attention_varlen_backward(const Tensor& Q,
                                     const Tensor& K,
                                     const Tensor& V,
                                     const Tensor& O,
                                     const Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     Tensor& dQ,
                                     Tensor& dK,
                                     Tensor& dV) {
    (void)O;

    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO dtype must match");
    }
    const int total_q = Q.rows;
    const int total_k = K.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_backward: shape mismatch");
    }
    if (dO.rows != total_q || dO.cols != D) {
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: max_seqlen_q/k must be non-negative");
    }

    if (dQ.rows != total_q || dQ.cols != D || dQ.dtype != dt) dQ.resize(total_q, D, dt);
    if (dK.rows != total_k || dK.cols != D || dK.dtype != dt) dK.resize(total_k, D, dt);
    if (dV.rows != total_k || dV.cols != D || dV.dtype != dt) dV.resize(total_k, D, dt);
    // Rows of empty sequences (and any rows outside cu_seqlens) stay zero.
    dQ.zero();
    dK.zero();
    dV.zero();

    if (D == 0 || batch_size == 0) return;
    if (total_q == 0 && total_k == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    std::vector<int32_t> cq(batch_size + 1);
    std::vector<int32_t> ck(batch_size + 1);
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(cq.data(), cu_seqlens_q, sizeof(int32_t) * (batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(ck.data(), cu_seqlens_k, sizeof(int32_t) * (batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (cq[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[0] must be 0");
    if (ck[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[0] must be 0");
    if (cq[batch_size] != total_q)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[B] != total_tokens_q");
    if (ck[batch_size] != total_k)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[B] != total_tokens_k");

    // max_seqlen_q/k are advisory: the kernels bound each sequence by cu_seqlens.
    bool any = false;
    for (int b = 0; b < batch_size; ++b) {
        const int Lq_b = cq[b + 1] - cq[b];
        const int Lk_b = ck[b + 1] - ck[b];
        if (Lq_b < 0 || Lk_b < 0)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens must be non-decreasing");
        if (causal && Lq_b != Lk_b)
            throw std::runtime_error("flash_attention_varlen_backward: causal requires per-sequence Lq == Lk");
        any = any || (Lq_b > 0 && Lk_b > 0);
    }
    if (!any) return;  // every sequence empty.

    if (!prefer_rows(cq, ck, num_heads, head_dim, causal)) {
        // Only the first block can refuse (the device check), before writing.
        bool ran = false, refused = false;
        for (int b = 0; b < batch_size && !refused; ++b) {
            const int Lq = cq[b + 1] - cq[b];
            const int Lk = ck[b + 1] - ck[b];
            if (Lq == 0 || Lk == 0) continue;  // grad rows already zero.
            if (tc_block(dt, Q, K, V, dO, nullptr, causal, dQ, dK, dV, static_cast<size_t>(cq[b]),
                         static_cast<size_t>(ck[b]), Lq, Lk, D, num_heads, head_dim, stream))
                ran = true;
            else
                refused = true;
        }
        if (!refused) return;
        if (ran) throw std::runtime_error("flash_attention_varlen_backward: tensor-core path refused mid-call");
    }
    // Rows of a sequence with no keys (or no queries) write zeros here too.
    rows_backward(dt, Q, K, V, dO, cu_seqlens_q, cu_seqlens_k, batch_size, nullptr, causal, num_heads, head_dim, dQ,
                  dK, dV, stream);
}

} // namespace detail::cuda

} // namespace brotensor
