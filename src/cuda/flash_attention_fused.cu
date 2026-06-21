// Fused flash-attention forward (FlashAttention-2 style).
//
// One CTA per (head, BR-row query tile). Q is staged to shared once per CTA
// (pre-scaled by 1/sqrt(head_dim)); K/V stream through shared in BC-row tiles.
// Each warp owns 16 query rows end to end:
//
//   S_strip(16, BC) = Q_strip @ K_tile^T        WMMA, FP32 accum -> shared
//   online softmax over the strip's rows        running (max, sum) + rescale
//   O_strip += P_strip @ V_tile                 WMMA, FP32 accum
//
// against running per-row statistics, so no (Lq, Lk) score matrix ever exists
// and the kernel reads Q/K/V directly from the interleaved (L, num_heads*hd)
// layout (no per-head extraction, no pack-back). The FP32 S staging buffer is
// reused to ferry the P@V accumulator fragments back to per-thread registers.
//
// Both 16-bit storage types share the one templated kernel (sm_80+ has BF16
// WMMA); accumulation and softmax are FP32 regardless.
//
// head_dim is a template parameter. WMMA contracts in 16-wide tiles, so a
// head_dim that is not a multiple of 16 is padded up to HD_PAD = round16(hd):
// Q and K/V pad columns are zero-filled in shared (so the padded QK^T scores and
// the padded P@V outputs are bit-identical to the unpadded math), and only the
// real hd output columns are written back. Shared usage grows with HD_PAD, so
// the query-tile height BR is also a template parameter — the wide heads use a
// shorter tile to stay under the per-block shared-memory cap (sm_89 ≈ 99 KB).
// Add (head_dim, BR) pairs in supported()/launch() as callers appear:
//   head_dim 64 -> BR 128 (DINOv3, TripoSplat flow, SD-class self-attention)
//   head_dim 72 -> BR 64  (PixArt-Sigma DiT self-attention)

#include "flash_fused_internal.cuh"
#include "detail/cuda_check.h"

#include <mma.h>

#include <cmath>
#include <cstdint>

namespace brotensor {
namespace flash_fused {

namespace {

using namespace nvcuda;

constexpr int BC = 64;        // key rows per tile
constexpr int NT = BC / 16;   // n-tiles across the key tile

// Per-head_dim derived layout. WMMA wants the head-dim contraction in 16-wide
// tiles, so everything is sized to HD_PAD = round-up-16(HD). Strides carry a
// little extra (the historical +8 / +16 padding) to keep WMMA's ldm % 8 == 0
// for 16-bit fragments, the FP32 S buffer's ldm % 4 == 0, and the int4 K/V
// stores 16-byte aligned. For HD == 64 these reduce to the original 72 / 80 / 72.
template <int HD>
struct hd_dims {
    static constexpr int PAD  = (HD + 15) / 16 * 16;        // contraction width
    static constexpr int LDQ  = PAD + 8;                    // Q row stride
    static constexpr int LDKV = PAD + 16;                   // K/V row stride
    static constexpr int LDS  = (BC > PAD ? BC : PAD) + 8;  // S / P / O staging
    static constexpr int KT   = PAD / 16;                   // head-dim tiles
    static constexpr int OCOL = PAD / 2;                    // O cols / thread
};

template <typename T> struct ff_traits;
template <> struct ff_traits<__half> {
    __device__ static float to_f32(__half v) { return __half2float(v); }
    __device__ static __half from_f32(float v) { return __float2half(v); }
};
template <> struct ff_traits<__nv_bfloat16> {
    __device__ static float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
};

// Dynamic shared layout (manual carve so the two 16-bit tile types and the
// FP32 staging buffer pack without alignment surprises).
template <typename T, int HD, int BR>
struct Smem {
    using D = hd_dims<HD>;
    T* q;        // (BR, LDQ)
    T* k;        // (BC, LDKV)
    T* v;        // (BC, LDKV)
    float* s;    // (BR, LDS)  S staging, then P@V staging
    T* p;        // (BR, LDS)

    __device__ static size_t bytes() {
        return (size_t(BR) * D::LDQ + 2 * size_t(BC) * D::LDKV) * sizeof(T) +
               size_t(BR) * D::LDS * sizeof(float) +
               size_t(BR) * D::LDS * sizeof(T);
    }
    __device__ explicit Smem(char* base) {
        s = reinterpret_cast<float*>(base);          // f32 first: strictest align
        base += size_t(BR) * D::LDS * sizeof(float);
        q = reinterpret_cast<T*>(base);
        base += size_t(BR) * D::LDQ * sizeof(T);
        k = reinterpret_cast<T*>(base);
        base += size_t(BC) * D::LDKV * sizeof(T);
        v = reinterpret_cast<T*>(base);
        base += size_t(BC) * D::LDKV * sizeof(T);
        p = reinterpret_cast<T*>(base);
    }
};

template <typename T, int HD, int BR>
size_t smem_bytes_host() {
    using D = hd_dims<HD>;
    return (size_t(BR) * D::LDQ + 2 * size_t(BC) * D::LDKV) * sizeof(T) +
           size_t(BR) * D::LDS * sizeof(float) +
           size_t(BR) * D::LDS * sizeof(T);
}

// Cooperatively load a (rows<=BC, HD) tile from the interleaved (L, D) source
// into shared (LDKV stride), zero-filling rows past Lk AND the pad columns
// [HD, HD_PAD) (so the padded WMMA reads exact zeros there). int4 = 8 halves
// per thread per step; D % 8 == 0 always holds (D = num_heads * HD, HD % 8 == 0).
template <typename T, int HD, int BR>
__device__ void load_kv_tile(const T* __restrict__ src, T* __restrict__ dst,
                             int l0, int Lk, int D, int head_off) {
    constexpr int NTHREADS = (BR / 16) * 32;
    constexpr int SEGS     = hd_dims<HD>::PAD / 8;   // 8-element segments incl pad
    constexpr int SEGS_REAL = HD / 8;                // segments backed by src
    constexpr int LDKV     = hd_dims<HD>::LDKV;
    constexpr int TOTAL    = BC * SEGS;
    const int4 zero4 = make_int4(0, 0, 0, 0);
    for (int idx = threadIdx.x; idx < TOTAL; idx += NTHREADS) {
        const int r   = idx / SEGS;
        const int seg = idx % SEGS;
        const int c   = seg * 8;
        int4* d = reinterpret_cast<int4*>(dst + size_t(r) * LDKV + c);
        if (seg < SEGS_REAL && l0 + r < Lk) {
            *d = *reinterpret_cast<const int4*>(
                src + size_t(l0 + r) * D + head_off + c);
        } else {
            *d = zero4;
        }
    }
}

template <typename T, int HD, int BR>
__global__ void __launch_bounds__((BR / 16) * 32)
flash_fused_kernel(const T* __restrict__ Q,
                   const T* __restrict__ K,
                   const T* __restrict__ V,
                   const float* __restrict__ mask,
                   T* __restrict__ O,
                   int Lq, int Lk, int D, float scale) {
    using DM = hd_dims<HD>;
    constexpr int WARPS    = BR / 16;
    constexpr int NTHREADS = WARPS * 32;
    constexpr int PAD  = DM::PAD;
    constexpr int LDQ  = DM::LDQ;
    constexpr int LDKV = DM::LDKV;
    constexpr int LDS  = DM::LDS;
    constexpr int KT   = DM::KT;       // head-dim tiles (contraction / output)
    constexpr int OCOL = DM::OCOL;     // O columns owned per thread
    constexpr int SCOL = BC / 2;       // softmax (key) columns owned per thread

    // 32-byte alignment: WMMA tile pointers must be 256-bit aligned, and the
    // carve in Smem keeps every tile offset a multiple of 32 relative to base.
    extern __shared__ __align__(32) char smem_raw[];
    Smem<T, HD, BR> sm(smem_raw);

    const int head = blockIdx.y;
    const int head_off = head * HD;
    const int q0 = blockIdx.x * BR;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wrow0 = warp * 16;             // warp's first row within the tile

    // Per-thread ownership within the warp's 16-row strip: two threads per row.
    // Softmax owns SCOL key columns each; O owns OCOL head-dim columns each.
    const int trow  = lane / 2;               // 0..15
    const int tcol_s = (lane % 2) * SCOL;     // key-column offset (softmax)
    const int tcol_o = (lane % 2) * OCOL;     // head-dim-column offset (output)
    const int orow  = q0 + wrow0 + trow;      // global query row

    // ── Stage Q (pre-scaled), zero-filling rows past Lq and pad cols [HD,PAD) ──
    for (int idx = threadIdx.x; idx < BR * PAD; idx += NTHREADS) {
        const int r = idx / PAD;
        const int c = idx % PAD;
        const int l = q0 + r;
        sm.q[size_t(r) * LDQ + c] = ff_traits<T>::from_f32(
            (l < Lq && c < HD)
                ? ff_traits<T>::to_f32(Q[size_t(l) * D + head_off + c]) * scale
                : 0.0f);
    }

    float o_acc[OCOL];
#pragma unroll
    for (int i = 0; i < OCOL; ++i) o_acc[i] = 0.0f;
    float m_run = -1e30f;
    float l_run = 0.0f;

    for (int j0 = 0; j0 < Lk; j0 += BC) {
        __syncthreads();  // everyone done with the previous K/V tile
        load_kv_tile<T, HD, BR>(K, sm.k, j0, Lk, D, head_off);
        load_kv_tile<T, HD, BR>(V, sm.v, j0, Lk, D, head_off);
        __syncthreads();

        // ── S_strip(16, BC) = Q_strip @ K_tile^T (WMMA, FP32 accum) ──
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) wmma::fill_fragment(c_frag[n], 0.0f);
#pragma unroll
            for (int k = 0; k < KT; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, sm.q + size_t(wrow0) * LDQ + k * 16, LDQ);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    // col_major view of the row-major K tile = K^T.
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, sm.k + size_t(n) * 16 * LDKV + k * 16, LDKV);
                    wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
                }
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                wmma::store_matrix_sync(sm.s + size_t(wrow0) * LDS + n * 16,
                                        c_frag[n], LDS, wmma::mem_row_major);
            }
        }
        __syncwarp();

        // ── Online softmax over the strip's rows (this thread: SCOL columns of
        //    one row; its lane^1 partner holds the other SCOL) ──
        const float* srow = sm.s + size_t(wrow0 + trow) * LDS + tcol_s;
        float s_val[SCOL];
        float tile_max = -1e30f;
#pragma unroll
        for (int c = 0; c < SCOL; ++c) {
            const int j = j0 + tcol_s + c;
            const bool valid = j < Lk && (!mask || mask[j] > 0.5f);
            s_val[c] = valid ? srow[c] : -1e30f;
            tile_max = fmaxf(tile_max, s_val[c]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));

        const float m_new = fmaxf(m_run, tile_max);
        const float corr = __expf(m_run - m_new);   // m_run <= m_new, finite diff
        T* prow = sm.p + size_t(wrow0 + trow) * LDS + tcol_s;
        float tile_sum = 0.0f;
#pragma unroll
        for (int c = 0; c < SCOL; ++c) {
            const float p = s_val[c] > -1e29f ? __expf(s_val[c] - m_new) : 0.0f;
            prow[c] = ff_traits<T>::from_f32(p);
            tile_sum += p;
        }
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
        l_run = l_run * corr + tile_sum;
        m_run = m_new;
        __syncwarp();

        // ── O_strip += P_strip(16, BC) @ V_tile(BC, HD_PAD) (WMMA, FP32 accum),
        //    staged back through the (now free) S buffer ──
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[KT];
#pragma unroll
            for (int n = 0; n < KT; ++n) wmma::fill_fragment(o_frag[n], 0.0f);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, sm.p + size_t(wrow0) * LDS + k * 16, LDS);
#pragma unroll
                for (int n = 0; n < KT; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, sm.v + size_t(k) * 16 * LDKV + n * 16, LDKV);
                    wmma::mma_sync(o_frag[n], a_frag, b_frag, o_frag[n]);
                }
            }
#pragma unroll
            for (int n = 0; n < KT; ++n) {
                wmma::store_matrix_sync(sm.s + size_t(wrow0) * LDS + n * 16,
                                        o_frag[n], LDS, wmma::mem_row_major);
            }
        }
        __syncwarp();

        const float* orow_s = sm.s + size_t(wrow0 + trow) * LDS + tcol_o;
#pragma unroll
        for (int c = 0; c < OCOL; ++c) {
            o_acc[c] = o_acc[c] * corr + orow_s[c];
        }
        __syncwarp();  // o_acc reads done before next tile's S overwrites sm.s
    }

    // ── Normalise and write the strip back into the interleaved output (only
    //    the real head_dim columns; the pad columns [HD, HD_PAD) are dropped) ──
    if (orow < Lq) {
        const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
        T* out = O + size_t(orow) * D + head_off + tcol_o;
#pragma unroll
        for (int c = 0; c < OCOL; ++c) {
            if (tcol_o + c < HD) out[c] = ff_traits<T>::from_f32(o_acc[c] * inv_l);
        }
    }
}

template <typename T, int HD, int BR>
void launch_impl(const T* Q, const T* K, const T* V, const float* mask, T* O,
                 int Lq, int Lk, int D, int num_heads, cudaStream_t stream) {
    constexpr int NTHREADS = (BR / 16) * 32;
    const size_t shmem = smem_bytes_host<T, HD, BR>();
    static bool attr_set = false;   // one-time opt-in past the 48KB default
    if (!attr_set) {
        BROTENSOR_CUDA_CHECK(cudaFuncSetAttribute(
            flash_fused_kernel<T, HD, BR>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem)));
        attr_set = true;
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(HD));
    dim3 grid((Lq + BR - 1) / BR, num_heads);
    flash_fused_kernel<T, HD, BR><<<grid, NTHREADS, shmem, stream>>>(
        Q, K, V, mask, O, Lq, Lk, D, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_dispatch(const T* Q, const T* K, const T* V, const float* mask, T* O,
                     int Lq, int Lk, int D, int num_heads, int head_dim,
                     cudaStream_t stream) {
    switch (head_dim) {
        case 64:
            launch_impl<T, 64, 128>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        case 72:
            launch_impl<T, 72, 64>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        default:
            return;  // guarded by supported(); unreachable
    }
}

}  // namespace

bool supported(int head_dim) {
    return head_dim == 64 || head_dim == 72;
}

void launch(const __half* Q, const __half* K, const __half* V,
            const float* mask, __half* O,
            int Lq, int Lk, int D, int num_heads, int head_dim,
            cudaStream_t stream) {
    launch_dispatch<__half>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
}

void launch(const __nv_bfloat16* Q, const __nv_bfloat16* K,
            const __nv_bfloat16* V,
            const float* mask, __nv_bfloat16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim,
            cudaStream_t stream) {
    launch_dispatch<__nv_bfloat16>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
}

}  // namespace flash_fused
}  // namespace brotensor
