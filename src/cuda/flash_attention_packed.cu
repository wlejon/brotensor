// Packed variable-length bidirectional self-attention off a fused QKV buffer
// (flash_attention_packed_qkv_forward) — the attention of a packed, unpadded
// encoder batch: many independent sequences of any lengths back to back in one
// (L, 3*H*hd) tensor, each row carrying its own [start, end) sequence bounds.
//
// head_dim 64 at FP16/BF16 runs a FlashAttention-2 style tensor-core kernel:
//
//   * one CTA per (64-row query block, head); 4 warps, 16 query rows each;
//   * the block's key range is the union of its rows' ranges (sequence bounds
//     clipped to the optional |i - j| <= window/2 band), walked in 64-key tiles
//     double-buffered through shared memory with cp.async;
//   * S = Q K^T and O += P V are mma.sync m16n8k16 (FP32 accumulate) fed by
//     ldmatrix; the online softmax runs in registers (exp2, log2e folded into
//     the scale) and P is re-packed from the S accumulators straight into
//     A-operand fragments, so no score ever touches shared memory;
//   * every score is masked against its own row's range, so a block that
//     straddles two sequences is exact; a warp skips the math of any tile
//     outside its 16 rows' union range.
//
// The grid depends only on L (never on how the rows are split into
// sequences), so a CUDA graph captured once per L replays for any packing.
//
// Every other (dtype, head_dim) takes a generic one-warp-per-(row, head)
// kernel with the same semantics (FP32 math).

#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

cudaStream_t cur_stream() { return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()); }

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("brotensor: flash_attention_packed_qkv_forward: " + reason);
}

constexpr int PK_HD = 64;
constexpr int PK_BQ = 64;             // query rows per CTA
constexpr int PK_BK = 64;             // keys per tile
constexpr int PK_LDS = PK_HD + 8;     // shared row stride (halves): conflict-free ldmatrix
constexpr int PK_THREADS = 128;
constexpr float kLog2e = 1.4426950408889634f;
// -inf without MSVC's overflowing INFINITY macro.
#define kNegInf __int_as_float(0xff800000)

// Row r's key range [lo, hi) — its sequence, clipped to the window band.
__device__ __forceinline__ void row_range(const int* __restrict__ bounds, int r, int L, int hw,
                                          int& lo, int& hi) {
    if (r >= L) {
        lo = 0;
        hi = 0;
        return;
    }
    const int s = bounds[2 * r], e = bounds[2 * r + 1];
    lo = hw >= 0 ? max(s, r - hw) : s;
    hi = hw >= 0 ? min(e, r + hw + 1) : e;
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(smem)), "l"(gmem),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

template <typename T> struct mma_t;
template <> struct mma_t<__half> {
    __device__ static void run(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static uint32_t pack(float lo, float hi) {
        __half2 h = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&h);
    }
};
template <> struct mma_t<__nv_bfloat16> {
    __device__ static void run(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static uint32_t pack(float lo, float hi) {
        __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&h);
    }
};

// Stage a 64-row x 64-col tile (rows [row0, row0+64) of one head section)
// into shared; rows at or past `row_end` are zero-filled.
template <typename T>
__device__ __forceinline__ void load_tile(T (*dst)[PK_LDS], const T* __restrict__ src, int ld, int col0,
                                          int row0, int row_end, int tid) {
    // 64 rows x 8 16-byte chunks = 512 chunks, 4 per thread.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = tid + i * PK_THREADS;
        const int row = c >> 3, ch = c & 7;
        const int gr = row0 + row;
        const bool ok = gr < row_end;
        const T* g = src + (ok ? static_cast<size_t>(gr) * ld + col0 + ch * 8 : 0);
        cp_async16(&dst[row][ch * 8], g, ok);
    }
}

template <typename T>
__global__ void __launch_bounds__(PK_THREADS)
    packed_attn_hd64_kernel(const T* __restrict__ qkv, const int* __restrict__ bounds, T* __restrict__ out,
                            int L, int H, int hw, float scale_log2) {
    __shared__ alignas(16) T sQ[PK_BQ][PK_LDS];
    __shared__ alignas(16) T sK[2][PK_BK][PK_LDS];
    __shared__ alignas(16) T sV[2][PK_BK][PK_LDS];
    __shared__ int s_lo, s_hi;

    const int D = H * PK_HD;
    const int ld = 3 * D;
    const int h = blockIdx.y;
    const int q0 = blockIdx.x * PK_BQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;

    if (tid == 0) {
        s_lo = INT_MAX;
        s_hi = INT_MIN;
    }
    __syncthreads();
    if (tid < PK_BQ) {
        int lo, hi;
        row_range(bounds, q0 + tid, L, hw, lo, hi);
        if (lo < hi) {
            atomicMin(&s_lo, lo);
            atomicMax(&s_hi, hi);
        }
    }

    // This thread's two rows and the warp's union range.
    const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    int lo0, hi0, lo1, hi1;
    row_range(bounds, r0, L, hw, lo0, hi0);
    row_range(bounds, r1, L, hw, lo1, hi1);
    int wlo = min(lo0 < hi0 ? lo0 : INT_MAX, lo1 < hi1 ? lo1 : INT_MAX);
    int whi = max(lo0 < hi0 ? hi0 : INT_MIN, lo1 < hi1 ? hi1 : INT_MIN);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        wlo = min(wlo, __shfl_xor_sync(0xffffffffu, wlo, o));
        whi = max(whi, __shfl_xor_sync(0xffffffffu, whi, o));
    }

    load_tile<T>(sQ, qkv, ld, h * PK_HD, q0, L, tid);
    __syncthreads();
    const int kmin = s_lo, kmax = s_hi;
    if (kmin >= kmax) return;  // no valid row in this block
    const int ntiles = (kmax - kmin + PK_BK - 1) / PK_BK;
    load_tile<T>(sK[0], qkv, ld, D + h * PK_HD, kmin, kmax, tid);
    load_tile<T>(sV[0], qkv, ld, 2 * D + h * PK_HD, kmin, kmax, tid);
    cp_async_commit();

    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.0f, l1 = 0.0f;
    uint32_t qf[4][4];

    for (int j = 0; j < ntiles; ++j) {
        const int buf = j & 1;
        if (j + 1 < ntiles) {
            const int nb = kmin + (j + 1) * PK_BK;
            load_tile<T>(sK[buf ^ 1], qkv, ld, D + h * PK_HD, nb, kmax, tid);
            load_tile<T>(sV[buf ^ 1], qkv, ld, 2 * D + h * PK_HD, nb, kmax, tid);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                ldsm_x4(qf[kc], &sQ[warp * 16 + (lane & 15)][kc * 16 + (lane >> 4) * 8]);
            }
        }
        const int kb = kmin + j * PK_BK;
        if (kb < whi && kb + PK_BK > wlo) {
            float s[8][4];
#pragma unroll
            for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t b[4];
                    const int mi = lane >> 3;
                    ldsm_x4(b, &sK[buf][np * 16 + (mi >> 1) * 8 + (lane & 7)][kc * 16 + (mi & 1) * 8]);
                    mma_t<T>::run(s[2 * np], qf[kc], b[0], b[1]);
                    mma_t<T>::run(s[2 * np + 1], qf[kc], b[2], b[3]);
                }
            }
            // Mask + scale, then the online-softmax update (rows r0 / r1).
            float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int key = kb + nt * 8 + 2 * t;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int kk = key + e;
                    s[nt][e] = (kk >= lo0 && kk < hi0) ? s[nt][e] * scale_log2 : kNegInf;
                    s[nt][2 + e] = (kk >= lo1 && kk < hi1) ? s[nt][2 + e] * scale_log2 : kNegInf;
                    mx0 = fmaxf(mx0, s[nt][e]);
                    mx1 = fmaxf(mx1, s[nt][2 + e]);
                }
            }
#pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
            }
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float mu0 = mn0 == kNegInf ? 0.0f : mn0;
            const float mu1 = mn1 == kNegInf ? 0.0f : mn1;
            const float a0 = exp2f(m0 - mu0), a1 = exp2f(m1 - mu1);
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - mu0);
                s[nt][1] = exp2f(s[nt][1] - mu0);
                s[nt][2] = exp2f(s[nt][2] - mu1);
                s[nt][3] = exp2f(s[nt][3] - mu1);
                ps0 += s[nt][0] + s[nt][1];
                ps1 += s[nt][2] + s[nt][3];
            }
            l0 = l0 * a0 + ps0;
            l1 = l1 * a1 + ps1;
            m0 = mn0;
            m1 = mn1;
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                o[dt][0] *= a0;
                o[dt][1] *= a0;
                o[dt][2] *= a1;
                o[dt][3] *= a1;
            }
            // O += P V
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                uint32_t a[4];
                a[0] = mma_t<T>::pack(s[2 * kc][0], s[2 * kc][1]);
                a[1] = mma_t<T>::pack(s[2 * kc][2], s[2 * kc][3]);
                a[2] = mma_t<T>::pack(s[2 * kc + 1][0], s[2 * kc + 1][1]);
                a[3] = mma_t<T>::pack(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
                for (int dp = 0; dp < 4; ++dp) {
                    uint32_t b[4];
                    const int mi = lane >> 3;
                    ldsm_x4_t(b, &sV[buf][kc * 16 + (mi & 1) * 8 + (lane & 7)][dp * 16 + (mi >> 1) * 8]);
                    mma_t<T>::run(o[2 * dp], a, b[0], b[1]);
                    mma_t<T>::run(o[2 * dp + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();  // everyone is done with `buf` before it is refilled
    }

#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        const int col = h * PK_HD + dt * 8 + 2 * t;
        if (r0 < L) {
            *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(r0) * D + col) =
                mma_t<T>::pack(o[dt][0] * inv0, o[dt][1] * inv0);
        }
        if (r1 < L) {
            *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(r1) * D + col) =
                mma_t<T>::pack(o[dt][2] * inv1, o[dt][3] * inv1);
        }
    }
}

// ─── Generic fallback: one warp per (row, head), any head_dim ──────────────

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

constexpr int GEN_MAX_PER_LANE = 8;  // head_dim <= 256

template <typename T>
__global__ void packed_attn_generic_kernel(const T* __restrict__ qkv, const int* __restrict__ bounds,
                                           T* __restrict__ out, int L, int H, int hd, int hw, float scale) {
    const int r = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (h >= H) return;
    const int D = H * hd;
    int lo, hi;
    row_range(bounds, r, L, hw, lo, hi);
    float q[GEN_MAX_PER_LANE], acc[GEN_MAX_PER_LANE];
    const T* qp = qkv + static_cast<size_t>(r) * 3 * D + h * hd;
#pragma unroll
    for (int i = 0; i < GEN_MAX_PER_LANE; ++i) {
        const int d = lane + 32 * i;
        q[i] = d < hd ? to_f(qp[d]) * scale : 0.0f;
        acc[i] = 0.0f;
    }
    float m = kNegInf, l = 0.0f;
    for (int j = lo; j < hi; ++j) {
        const T* kp = qkv + static_cast<size_t>(j) * 3 * D + D + h * hd;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < GEN_MAX_PER_LANE; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) dot += q[i] * to_f(kp[d]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const float mn = fmaxf(m, dot);
        const float a = expf(m - mn), p = expf(dot - mn);
        l = l * a + p;
        m = mn;
        const T* vp = kp + D;
#pragma unroll
        for (int i = 0; i < GEN_MAX_PER_LANE; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) acc[i] = acc[i] * a + p * to_f(vp[d]);
        }
    }
    T* op = out + static_cast<size_t>(r) * D + h * hd;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int i = 0; i < GEN_MAX_PER_LANE; ++i) {
        const int d = lane + 32 * i;
        if (d < hd) op[d] = from_f<T>(acc[i] * inv);
    }
}

template <typename T>
void launch(const ::brotensor::Tensor& QKV, const int* bounds, ::brotensor::Tensor& O, int L, int H, int hd,
            int hw) {
    const T* qkv = static_cast<const T*>(QKV.data);
    T* out = static_cast<T*>(O.data);
    constexpr bool tc = !std::is_same<T, float>::value;
    if (tc && hd == PK_HD) {
        if constexpr (tc) {
            const dim3 grid((L + PK_BQ - 1) / PK_BQ, H);
            const float scale_log2 = kLog2e / sqrtf(static_cast<float>(hd));
            packed_attn_hd64_kernel<T><<<grid, PK_THREADS, 0, cur_stream()>>>(qkv, bounds, out, L, H, hw,
                                                                               scale_log2);
        }
    } else {
        constexpr int warps = 4;
        const dim3 grid(L, (H + warps - 1) / warps);
        packed_attn_generic_kernel<T><<<grid, 32 * warps, 0, cur_stream()>>>(
            qkv, bounds, out, L, H, hd, hw, 1.0f / sqrtf(static_cast<float>(hd)));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV, const ::brotensor::Tensor& seq_bounds,
                                        int num_heads, int window, ::brotensor::Tensor& O) {
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail("QKV.cols must be 3 * num_heads * head_dim");
    if (seq_bounds.dtype != Dtype::INT32) fail("seq_bounds must be INT32");
    const int L = QKV.rows;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail("seq_bounds must be (L, 2)");
    const int D = QKV.cols / 3;
    const int hd = D / num_heads;
    if (hd > 32 * GEN_MAX_PER_LANE) fail("head_dim > 256 is not supported");
    const Dtype dt = QKV.dtype;
    if (O.rows != L || O.cols != D || O.dtype != dt) O.resize(L, D, dt);
    if (L == 0) return;
    const int hw = window > 0 ? window / 2 : -1;
    const int* b = static_cast<const int*>(seq_bounds.data);
    switch (dt) {
        case Dtype::FP16: launch<__half>(QKV, b, O, L, num_heads, hd, hw); break;
        case Dtype::BF16: launch<__nv_bfloat16>(QKV, b, O, L, num_heads, hd, hw); break;
        case Dtype::FP32: launch<float>(QKV, b, O, L, num_heads, hd, hw); break;
        default: fail("QKV must be FP32/FP16/BF16");
    }
}

}  // namespace brotensor::detail::cuda
