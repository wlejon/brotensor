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
// layout (no per-head extraction, no pack-back). The P@V accumulator stays in
// WMMA accumulator fragments for the entire key loop — the per-row softmax
// correction is applied to fragment elements in place, using a row map the
// kernel probes from the API rather than assumes. Only the final normalised
// write-back is ferried out through a slot in the (by then free) FP32 S
// staging buffer, one 16-column fragment at a time, so that buffer only ever
// needs BC (score) columns — not head_dim columns. That matters for the wide
// heads: head_dim 128 with a full-width ferry needed a 136-column FP32 buffer,
// which capped the query tile at BR = 48 (3 warps — one of the four warp
// schedulers permanently idle, 1 CTA/SM).
//
// K/V tiles are prefetched a tile ahead through registers, so a tile's global
// reads are in flight while the previous tile's math runs.
//
// Scores carry a factor of log2(e) folded into the Q staging scale so the
// softmax is exp2 (one ex2.approx) rather than exp (ex2.approx plus a mul).
//
// Both 16-bit storage types share the one templated kernel (sm_80+ has BF16
// WMMA); accumulation and softmax are FP32 regardless.
//
// head_dim is a template parameter. WMMA contracts in 16-wide tiles, so a
// head_dim that is not a multiple of 16 is padded up to HD_PAD = round16(hd):
// Q and K/V pad columns are zero-filled in shared (so the padded QK^T scores and
// the padded P@V outputs are bit-identical to the unpadded math), and only the
// real hd output columns are written back. Shared usage grows with HD_PAD, so
// the query-tile height BR and key-tile depth BC are also template parameters —
// the wide heads use a shallower key tile to afford a taller query tile under
// the per-block shared-memory cap (sm_89 ≈ 99 KB).
// Add (head_dim, BR, BC) triples in supported()/launch() as callers appear:
//   head_dim 64  -> BR 128, BC 64 (DINOv3, TripoSplat flow, SD-class self-attn)
//   head_dim 72  -> BR 64,  BC 64 (PixArt-Sigma DiT self-attention)
//   head_dim 128 -> BR 128, BC 32 (Krea 2 / Flux-class DiT self-attention)
//
// CAUSAL is a template parameter too, so the non-causal path keeps exactly the
// codegen it had before causal existed - no runtime branch in the softmax inner
// loop, no extra register pressure. A causal CTA stops its key loop at its own
// query tile (kv_end) and masks j > i per element only in the one tile that
// straddles the diagonal, which is where the ~2x arithmetic saving comes from.

#include "flash_fused_internal.cuh"
#include "detail/cuda_check.h"

#include <mma.h>

#include <cmath>
#include <cstdint>

namespace brotensor {
namespace flash_fused {

namespace {

using namespace nvcuda;

// Per-(head_dim, BC) derived layout. WMMA wants the head-dim contraction in
// 16-wide tiles, so everything is sized to HD_PAD = round-up-16(HD). Strides
// carry a little extra (the historical +8 / +16 padding) to keep WMMA's
// ldm % 8 == 0 for 16-bit fragments, the FP32 S buffer's ldm % 4 == 0, and the
// int4 K/V stores 16-byte aligned.
template <int HD, int BC>
struct hd_dims {
    static constexpr int PAD  = (HD + 15) / 16 * 16;  // contraction width
    static constexpr int LDQ  = PAD + 8;              // Q row stride
    static constexpr int LDKV = PAD + 16;             // K/V row stride
    static constexpr int LDS  = BC + 8;               // S scores / P staging
    static constexpr int KT   = PAD / 16;             // head-dim tiles
    static constexpr int NT   = BC / 16;              // key tiles across BC
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
template <typename T, int HD, int BR, int BC>
struct Smem {
    using D = hd_dims<HD, BC>;
    T* q;        // (BR, LDQ)
    T* k;        // (BC, LDKV)
    T* v;        // (BC, LDKV)
    float* s;    // (BR, LDS)  S staging, then the P@V ferry slot
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

template <typename T, int HD, int BR, int BC>
size_t smem_bytes_host() {
    using D = hd_dims<HD, BC>;
    return (size_t(BR) * D::LDQ + 2 * size_t(BC) * D::LDKV) * sizeof(T) +
           size_t(BR) * D::LDS * sizeof(float) +
           size_t(BR) * D::LDS * sizeof(T);
}

// K/V tile staging. A (rows<=BC, HD) tile of the interleaved (L, D) source is
// moved to shared (LDKV stride) in two halves — global->register, then
// register->shared — so the key loop can issue tile j+1's global reads before
// doing tile j's math and let the DRAM latency retire underneath it. int4 = 8
// halves per thread per step; D % 8 == 0 always holds (D = num_heads * HD,
// HD % 8 == 0).
//
// The alternative shape, a second shared K/V buffer filled by cp.async, does
// not fit: head_dim 64 already sits at 92 of the ~99 KB per-block cap, and
// double-buffering K+V costs another 20 KB. Registers are the free resource
// here — PER is 2 for three of the four instantiations, 5 for head_dim 72.
template <int HD, int BR, int BC>
struct kv_stage {
    static constexpr int NTHREADS  = (BR / 16) * 32;
    static constexpr int SEGS      = hd_dims<HD, BC>::PAD / 8;  // incl. pad cols
    static constexpr int SEGS_REAL = HD / 8;                    // backed by src
    static constexpr int TOTAL     = BC * SEGS;
    static constexpr int PER       = (TOTAL + NTHREADS - 1) / NTHREADS;
    // Only the head_dim 40 tiling leaves a partial final step (TOTAL 384 over
    // 256 threads); everywhere else the guard folds away at compile time.
    static constexpr bool GUARD    = (PER * NTHREADS != TOTAL);
};

// Read one tile into registers, materialising the zero-fill (rows past Lk and
// the pad columns [HD, HD_PAD)) here rather than at the shared store, so the
// padded WMMA reads exact zeros without a second pass.
template <typename T, int HD, int BR, int BC>
__device__ __forceinline__
void load_kv_regs(const T* __restrict__ src,
                  int4 (&reg)[kv_stage<HD, BR, BC>::PER],
                  int l0, int Lk, int D, int head_off) {
    using S = kv_stage<HD, BR, BC>;
#pragma unroll
    for (int i = 0; i < S::PER; ++i) {
        const int idx = threadIdx.x + i * S::NTHREADS;
        if (S::GUARD && idx >= S::TOTAL) continue;
        const int r   = idx / S::SEGS;
        const int seg = idx % S::SEGS;
        if (seg < S::SEGS_REAL && l0 + r < Lk) {
            reg[i] = *reinterpret_cast<const int4*>(
                src + size_t(l0 + r) * D + head_off + seg * 8);
        } else {
            reg[i] = make_int4(0, 0, 0, 0);
        }
    }
}

template <typename T, int HD, int BR, int BC>
__device__ __forceinline__
void store_kv_regs(const int4 (&reg)[kv_stage<HD, BR, BC>::PER],
                   T* __restrict__ dst) {
    using S = kv_stage<HD, BR, BC>;
    constexpr int LDKV = hd_dims<HD, BC>::LDKV;
#pragma unroll
    for (int i = 0; i < S::PER; ++i) {
        const int idx = threadIdx.x + i * S::NTHREADS;
        if (S::GUARD && idx >= S::TOTAL) continue;
        const int r   = idx / S::SEGS;
        const int seg = idx % S::SEGS;
        *reinterpret_cast<int4*>(dst + size_t(r) * LDKV + seg * 8) = reg[i];
    }
}

template <typename T, int HD, int BR, int BC, bool CAUSAL>
__global__ void __launch_bounds__((BR / 16) * 32)
flash_fused_kernel(const T* __restrict__ Q,
                   const T* __restrict__ K,
                   const T* __restrict__ V,
                   const float* __restrict__ mask,
                   T* __restrict__ O,
                   int Lq, int Lk, int D, float scale) {
    using DM = hd_dims<HD, BC>;
    constexpr int WARPS    = BR / 16;
    constexpr int NTHREADS = WARPS * 32;
    constexpr int PAD  = DM::PAD;
    constexpr int LDQ  = DM::LDQ;
    constexpr int LDKV = DM::LDKV;
    constexpr int LDS  = DM::LDS;
    constexpr int KT   = DM::KT;       // head-dim tiles (contraction / output)
    constexpr int NT   = DM::NT;       // key tiles across the BC key rows
    constexpr int SCOL = BC / 2;       // softmax (key) columns owned per thread

    using OFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    constexpr int OFE = OFrag::num_elements;   // 8 on every arch so far

    // 32-byte alignment: WMMA tile pointers must be 256-bit aligned, and the
    // carve in Smem keeps every tile offset a multiple of 32 relative to base.
    extern __shared__ __align__(32) char smem_raw[];
    Smem<T, HD, BR, BC> sm(smem_raw);

    const int head = blockIdx.y;
    const int head_off = head * HD;
    // Causal CTAs do work proportional to their query tile index: tile 0 walks
    // one key tile, the last walks Lk/BC of them. Blocks are dispatched in
    // increasing linear id, so issuing the tiles in reverse puts the longest
    // CTAs into the first wave and leaves the short ones to fill the tail.
    // In-order dispatch strands the heaviest CTA in the final wave with the
    // rest of the machine idle behind it.
    const int qtile = CAUSAL ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const int q0 = qtile * BR;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wrow0 = warp * 16;             // warp's first row within the tile

    // Per-thread ownership within the warp's 16-row strip: two threads per row.
    // Softmax owns SCOL key columns each; O owns 8 columns of each 16-wide
    // head-dim fragment (KT fragments * 8 = OCOL total).
    const int trow  = lane / 2;               // 0..15
    const int tcol_s = (lane % 2) * SCOL;     // key-column offset (softmax)
    const int half_o = (lane % 2) * 8;        // per-fragment output half
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

    // ── Probe the accumulator fragment's element -> row mapping ──
    // O now lives in accumulator fragments for the whole key loop, so the
    // per-row online-softmax correction has to be applied to fragment elements
    // directly — and the WMMA API does not document which element belongs to
    // which row. Rather than hard-code the arrangement Volta through Ada happen
    // to use, ask for it: fill a 16x16 tile with its own row indices, load it
    // as an accumulator fragment, and read back where each of this thread's
    // elements came from. One warp-private shared round-trip, once per CTA.
    int frag_row[OFE];
    {
        float* probe_slot = sm.s + size_t(wrow0) * LDS;
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            probe_slot[size_t(idx / 16) * LDS + (idx % 16)] =
                static_cast<float>(idx / 16);
        }
        __syncwarp();
        OFrag probe;
        wmma::load_matrix_sync(probe, probe_slot, LDS, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < OFE; ++i) frag_row[i] = static_cast<int>(probe.x[i]);
        __syncwarp();
    }

    OFrag o_frag[KT];
#pragma unroll
    for (int t = 0; t < KT; ++t) wmma::fill_fragment(o_frag[t], 0.0f);
    float m_run = -1e30f;
    float l_run = 0.0f;

    // Causal: this CTA's tallest query row is q0 + BR - 1, so every key tile
    // starting at or past q0 + BR is entirely above the diagonal. Skipping
    // them outright is where the ~2x arithmetic saving comes from — the
    // partially-masked tile straddling the diagonal is still computed in full
    // and masked per element below.
    const int kv_end = CAUSAL ? min(Lk, q0 + BR) : Lk;

    // Tile 0's reads are the only ones that cannot be hidden: there is no
    // preceding tile to hide them behind.
    using KVS = kv_stage<HD, BR, BC>;
    int4 kreg[KVS::PER], vreg[KVS::PER];
    load_kv_regs<T, HD, BR, BC>(K, kreg, 0, Lk, D, head_off);
    load_kv_regs<T, HD, BR, BC>(V, vreg, 0, Lk, D, head_off);

    for (int j0 = 0; j0 < kv_end; j0 += BC) {
        __syncthreads();  // everyone done with the previous K/V tile
        store_kv_regs<T, HD, BR, BC>(kreg, sm.k);
        store_kv_regs<T, HD, BR, BC>(vreg, sm.v);

        // Issue the next tile's global reads before waiting on the barrier and
        // before any of this tile's math. They land in registers and are not
        // consumed until the store at the top of the next iteration, so the
        // DRAM latency retires underneath the QK^T / softmax / P@V below. The
        // previous shape stalled the whole CTA on these loads between two
        // barriers with nothing else in flight.
        const int jnext = j0 + BC;
        if (jnext < kv_end) {
            load_kv_regs<T, HD, BR, BC>(K, kreg, jnext, Lk, D, head_off);
            load_kv_regs<T, HD, BR, BC>(V, vreg, jnext, Lk, D, head_off);
        }
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
            // `orow` is this thread's global query row. Masked-out scores go to
            // -1e30, so their exp() below is written as an exact 0.0f into the
            // P tile — the P@V contraction then runs over the whole BC tile
            // unchanged and still adds nothing for them.
            const bool valid = j < Lk && (!CAUSAL || j <= orow) &&
                               (!mask || mask[j] > 0.5f);
            s_val[c] = valid ? srow[c] : -1e30f;
            tile_max = fmaxf(tile_max, s_val[c]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));

        const float m_new = fmaxf(m_run, tile_max);
        // exp2, not exp: the scores arrive pre-multiplied by log2(e) (folded
        // into the Q staging scale), so every softmax exponential here is one
        // MUFU with no accompanying multiply. __expf() is ex2.approx preceded
        // by a mul by log2(e), and there are BR*BC of them per key tile.
        const float corr = exp2f(m_run - m_new);   // m_run <= m_new, finite diff
        T* prow = sm.p + size_t(wrow0 + trow) * LDS + tcol_s;
        float tile_sum = 0.0f;
#pragma unroll
        for (int c = 0; c < SCOL; ++c) {
            const float p = s_val[c] > -1e29f ? exp2f(s_val[c] - m_new) : 0.0f;
            prow[c] = ff_traits<T>::from_f32(p);
            tile_sum += p;
        }
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
        l_run = l_run * corr + tile_sum;
        m_run = m_new;
        __syncwarp();

        // ── O_strip = corr * O_strip + P_strip(16, BC) @ V_tile(BC, HD_PAD) ──
        // The running O stays in accumulator fragments across the entire key
        // loop and the rescale is applied to fragment elements in place, so
        // nothing round-trips through shared here. `corr` for query row r sits
        // in lanes 2r and 2r+1 (softmax pairs two threads per row and equalises
        // them across lane^1), so one shuffle per element position fetches it.
        // The element->row map is a property of the fragment shape, not of the
        // output tile, so this is OFE shuffles per key tile rather than per
        // output tile — replacing KT shared stores, KT*8 shared loads and
        // 2*KT __syncwarp()s that the per-tile ferry paid on every iteration.
        {
            float corr_e[OFE];
#pragma unroll
            for (int i = 0; i < OFE; ++i)
                corr_e[i] = __shfl_sync(0xffffffffu, corr, frag_row[i] * 2);

            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> p_frag[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                wmma::load_matrix_sync(p_frag[k], sm.p + size_t(wrow0) * LDS + k * 16, LDS);
            }
#pragma unroll
            for (int t = 0; t < KT; ++t) {
#pragma unroll
                for (int i = 0; i < OFE; ++i) o_frag[t].x[i] *= corr_e[i];
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, sm.v + size_t(k) * 16 * LDKV + t * 16, LDKV);
                    wmma::mma_sync(o_frag[t], p_frag[k], b_frag, o_frag[t]);
                }
            }
        }
    }

    // ── Normalise and write the strip back into the interleaved output (only
    //    the real head_dim columns; the pad columns [HD, HD_PAD) are dropped) ──
    // The probe recovered rows, not columns, so the write-back still ferries
    // each output fragment through the warp's slot in the (now free) S buffer
    // — but KT times for the whole CTA rather than KT times per key tile.
    {
        const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
        float* slot = sm.s + size_t(wrow0) * LDS;
        const float* slot_rd = sm.s + size_t(wrow0 + trow) * LDS + half_o;
        T* out = O + size_t(orow) * D + head_off;
#pragma unroll
        for (int t = 0; t < KT; ++t) {
            __syncwarp();  // previous slot reads done before it is overwritten
            wmma::store_matrix_sync(slot, o_frag[t], LDS, wmma::mem_row_major);
            __syncwarp();
            if (orow < Lq) {
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int col = t * 16 + half_o + c;
                    if (col < HD)
                        out[col] = ff_traits<T>::from_f32(slot_rd[c] * inv_l);
                }
            }
        }
    }
}

template <typename T, int HD, int BR, int BC, bool CAUSAL>
void launch_impl(const T* Q, const T* K, const T* V, const float* mask, T* O,
                 int Lq, int Lk, int D, int num_heads, cudaStream_t stream) {
    constexpr int NTHREADS = (BR / 16) * 32;
    const size_t shmem = smem_bytes_host<T, HD, BR, BC>();
    static bool attr_set = false;   // one-time opt-in past the 48KB default
    if (!attr_set) {
        BROTENSOR_CUDA_CHECK(cudaFuncSetAttribute(
            flash_fused_kernel<T, HD, BR, BC, CAUSAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem)));
        attr_set = true;
    }
    // log2(e) is folded into the score scale so the kernel's softmax can be
    // exp2 rather than exp. exp2(x*log2e) == exp(x) exactly in the algebra;
    // the running max is scaled by the same constant, so every comparison and
    // correction stays consistent. Q is staged pre-scaled, so this costs
    // nothing at run time.
    const float scale = 1.4426950408889634f / sqrtf(static_cast<float>(HD));
    dim3 grid((Lq + BR - 1) / BR, num_heads);
    flash_fused_kernel<T, HD, BR, BC, CAUSAL><<<grid, NTHREADS, shmem, stream>>>(
        Q, K, V, mask, O, Lq, Lk, D, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T, bool CAUSAL>
void launch_dispatch(const T* Q, const T* K, const T* V, const float* mask, T* O,
                     int Lq, int Lk, int D, int num_heads, int head_dim,
                     cudaStream_t stream) {
    switch (head_dim) {
        case 40:
            // PAD = round16(40) = 48, smaller than head_dim 64's PAD (64), so
            // shared-memory pressure is lower here than the 64/BR=128 case —
            // BR=128 fits comfortably. SD1.5-class self-attention (Lq=Lk=4096,
            // head_dim=40) is the motivating shape: previously fell through to
            // the O(Lq*Lk) scalar path since this switch had no case for it.
            launch_impl<T, 40, 128, 64, CAUSAL>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        case 64:
            launch_impl<T, 64, 128, 64, CAUSAL>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        case 72:
            launch_impl<T, 72, 64, 64, CAUSAL>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        case 128:
            // The wide head doubles the Q footprint per query row, so the key
            // tile drops to BC = 32 to afford a full BR = 128 (8 warps) tile:
            // 34816 (Q) + 18432 (K+V) + 20480 (S) + 10240 (P) = 82 KB, under
            // sm_89's ~99 KB dynamic-shared cap. The previous full-width P@V
            // ferry forced a PAD-wide FP32 S buffer, capping this shape at
            // BR = 48 / 3 warps / 1 CTA per SM (~32 TFLOPS). Krea 2 /
            // Flux-class DiT self-attention (head_dim 128) is the motivating
            // shape.
            launch_impl<T, 128, 128, 32, CAUSAL>(Q, K, V, mask, O, Lq, Lk, D, num_heads, stream);
            return;
        default:
            return;  // guarded by supported(); unreachable
    }
}

}  // namespace

bool supported(int head_dim) {
    return head_dim == 40 || head_dim == 64 || head_dim == 72 ||
           head_dim == 128;
}

void launch(const __half* Q, const __half* K, const __half* V,
            const float* mask, __half* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream) {
    if (causal) {
        launch_dispatch<__half, true>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
    } else {
        launch_dispatch<__half, false>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
    }
}

void launch(const __nv_bfloat16* Q, const __nv_bfloat16* K,
            const __nv_bfloat16* V,
            const float* mask, __nv_bfloat16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream) {
    if (causal) {
        launch_dispatch<__nv_bfloat16, true>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
    } else {
        launch_dispatch<__nv_bfloat16, false>(Q, K, V, mask, O, Lq, Lk, D, num_heads, head_dim, stream);
    }
}

}  // namespace flash_fused
}  // namespace brotensor
