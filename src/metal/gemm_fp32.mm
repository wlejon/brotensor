// FP32-multiply GEMM on Metal: C[b](M, N) (op)= epilogue(alpha * op(A) @ op(B) + bias)
// for FP32 or BF16 operands (BF16 widens to FP32 on load; exact), FP32
// accumulate, FP16 / BF16 / FP32 output. This is the fast path behind
// launch_matmul_abt_mixed for every non-FP16 input type (see fp16_matmul.h for
// the AbtMixed contract) — it carries the FP32 matmul, the FP32 batched
// linears and the 1x1 conv GEMMs.
//
// Two kernels, each compiled per (transA, transB, in, out) variant by
// prepending #defines to one source (a layout that is a compile-time constant
// keeps the loads and the simdgroup_load transposes branch-free):
//
//   k_gemm_tiled   64x64 output tile per threadgroup, BK = 16, four simdgroups
//                  in a 2x2 grid each owning 32x32 = 4x4 simdgroup_float8x8
//                  accumulators; ~3.5 TFLOP/s on M2 Pro at 4096^3 (MPS: ~5).
//                  Global loads are float4 along each operand's contiguous
//                  axis when its rows are 16-byte aligned, scalar + masked at
//                  the M/N/K edges, so any shape and any stride is valid. The
//                  epilogue works straight from the accumulator registers
//                  (thread_elements), no staging tile. Measured and rejected
//                  on M2 Pro: a double-buffered tile, a register prefetch of
//                  the next K step (both ~10-20% slower: they cost occupancy),
//                  BK 32, and 128-wide or 32-wide tiles.
//   k_gemm_skinny  M <= kSkinnyMaxM (a GEMV-like decode row block), keeping
//                  all M rows' partial sums in registers so B streams once:
//                  B stored (K, N) is read in float4 column groups by 16 K
//                  lanes reduced through threadgroup memory; B stored (N, K)
//                  gives each output column a simdgroup reducing along K.
//
// The long-K / small-output shape (1x1 conv weight gradient) is split over K
// by the caller (conv2d.mm) through the batch axis, then folded.

#include "fp16_matmul.h"

#import "internal.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brotensor::metal_impl {

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Prepended per variant: TA, TB (0/1 — A stored (K, M) / B stored (K, N)),
// TI (float / bfloat), TO (half / bfloat / float).

struct GemmParams {
    uint  M, N, K, has_bias;
    int   act, epi;
    float alpha;
    uint  vec;       // bit 0: A rows 4-element aligned; bit 1: B rows likewise
    ulong lda, ldb, ldc, sA, sB, sC;
};

// A & S 7.1.26, |err| < 1.5e-7 (MSL has no erf).
inline float g_erf(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return x < 0.0f ? -y : y;
}
inline float g_gelu_erf(float v) { return 0.5f * v * (1.0f + g_erf(v * 0.70710678118654752f)); }
inline float g_act(int act, float v) {
    switch (act) {
        case 1: return max(v, 0.0f);
        case 2: {
            const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: return g_gelu_erf(v);
        case 4: return v / (1.0f + exp(-v));
        case 5: return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}
inline float g_pre(float acc, uint n, device const TI* bias, constant GemmParams& p) {
    float v = acc * p.alpha;
    if (p.has_bias) v += float(bias[n]);
    return g_act(p.act, v);
}

// Writes result r for output column n (or, for GeGLU, the pair n, n + 1 with
// n even) of row m. The caller has checked m < M and n < N.
inline void g_store1(device TO* Cb, uint m, uint n, float r, device const TI* bias, constant GemmParams& p) {
    device TO* out = Cb + ulong(m) * p.ldc + n;
    float v = g_pre(r, n, bias, p);
    if (p.epi == 1) v += float(*out);
    *out = TO(v);
}
inline void g_store2(device TO* Cb, uint m, uint n, float r0, float r1, device const TI* bias,
                     constant GemmParams& p) {
    if (p.epi == 2) {
        Cb[ulong(m) * p.ldc + n / 2] = TO(g_pre(r0, n, bias, p) * g_gelu_erf(g_pre(r1, n + 1, bias, p)));
        return;
    }
    g_store1(Cb, m, n, r0, bias, p);
    if (n + 1 < p.N) g_store1(Cb, m, n + 1, r1, bias, p);
}

// Element (row, k) of an operand stored row-major (row, k), or with T, (k, row).
#define G_AT(T, row, k, ld) ((T) ? ulong(k) * (ld) + (row) : ulong(row) * (ld) + (k))

// Fixed-trip loops must fully unroll: left to the compiler, the rolled
// accumulator loops measured ~6x slower.
#define G_UNROLL _Pragma("clang loop unroll(full)")

// 64x64x16 over a 2x2 simdgroup grid measured best on M2 Pro (see the file
// comment).
constant constexpr int BM = 64, BN = 64, BK = 16;
constant constexpr int WM = 2, WN = 2;            // simdgroup grid
constant constexpr int NT = WM * WN * 32;
constant constexpr int SG_M = BM / WM, SG_N = BN / WN;   // per-simdgroup output block
constant constexpr int FM = SG_M / 8, FN = SG_N / 8;
constant constexpr int PAD = 4;                   // keeps float4 rows 16-byte aligned
// A tile: row-major (BM, BK + PAD) when A is (M, K); (BK, BM + PAD) when (K, M).
constant constexpr int LDA_S = TA ? BM + PAD : BK + PAD;
constant constexpr int LDB_S = TB ? BN + PAD : BK + PAD;
constant constexpr int A_TILE = TA ? BK * (BM + PAD) : BM * (BK + PAD);
constant constexpr int B_TILE = TB ? BK * (BN + PAD) : BN * (BK + PAD);
// float4 slots per tile and per thread.
constant constexpr int A_V = BM * BK / 4 / NT;
constant constexpr int B_V = BN * BK / 4 / NT;

typedef vec<TI, 4> TI4;

// Loads the float4 slot `s` (of the TR x BK tile) at (row0, k0) of an operand
// with R rows; zeros past the edges. The slot's four elements run along the
// operand's contiguous axis: k when stored (rows, K), rows when stored (K, rows).
template <bool T, int TR>
inline float4 g_load4(device const TI* X, ulong ld, bool vec_ok, uint R, uint K, uint row0, uint k0, int s) {
    constexpr int per = T ? TR / 4 : BK / 4;
    const int a = s / per, b = (s % per) * 4;
    const uint row = T ? row0 + uint(b) : row0 + uint(a);
    const uint k   = T ? k0 + uint(a)   : k0 + uint(b);
    device const TI* src = X + G_AT(T, row, k, ld);
    const bool in4 = T ? (k < K && row + 3 < R) : (row < R && k + 3 < K);
    if (vec_ok && in4) return float4(*reinterpret_cast<device const TI4*>(src));
    float4 v = 0.0f;
    G_UNROLL for (int i = 0; i < 4; ++i) {
        const bool ok = T ? (k < K && row + uint(i) < R) : (row < R && k + uint(i) < K);
        if (ok) v[i] = float(src[i]);
    }
    return v;
}

// Stores slot `s` (the same slot g_load4 read) into the tile.
template <bool T, int TR>
inline void g_stash4(threadgroup float* S, int ld, int s, float4 v) {
    constexpr int per = T ? TR / 4 : BK / 4;
    const int a = s / per, b = (s % per) * 4;
    *reinterpret_cast<threadgroup float4*>(S + a * ld + b) = v;
}

[[max_total_threads_per_threadgroup(NT)]]
kernel void k_gemm_tiled(device const TI* A    [[buffer(0)]],
                         device const TI* B    [[buffer(1)]],
                         device TO*       C    [[buffer(2)]],
                         device const TI* bias [[buffer(3)]],
                         constant GemmParams& p [[buffer(4)]],
                         uint3  tg_pos [[threadgroup_position_in_grid]],
                         ushort tid    [[thread_index_in_threadgroup]],
                         ushort sg_id  [[simdgroup_index_in_threadgroup]],
                         ushort lane   [[thread_index_in_simdgroup]]) {
    const uint M = p.M, N = p.N, K = p.K;
    device const TI* Ab = A + ulong(tg_pos.z) * p.sA;
    device const TI* Bb = B + ulong(tg_pos.z) * p.sB;
    device TO*       Cb = C + ulong(tg_pos.z) * p.sC;
    const uint block_m = tg_pos.y * BM, block_n = tg_pos.x * BN;
    const bool va = (p.vec & 1u) != 0u, vb = (p.vec & 2u) != 0u;

    threadgroup float As[A_TILE];
    threadgroup float Bs[B_TILE];

    const int sm = int(sg_id) / WN, sn = int(sg_id) % WN;
    simdgroup_float8x8 acc[FM][FN];
    G_UNROLL for (int i = 0; i < FM; ++i)
        G_UNROLL for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Global -> registers -> the tile. Loading straight into the tile after
        // the previous step's trailing barrier beats a register prefetch of
        // the next tile or a second tile buffer: both cost occupancy.
        float4 ra[A_V], rb[B_V];
        G_UNROLL for (int i = 0; i < A_V; ++i) ra[i] = g_load4<TA, BM>(Ab, p.lda, va, M, K, block_m, k0, int(tid) + i * NT);
        G_UNROLL for (int i = 0; i < B_V; ++i) rb[i] = g_load4<TB, BN>(Bb, p.ldb, vb, N, K, block_n, k0, int(tid) + i * NT);
        G_UNROLL for (int i = 0; i < A_V; ++i) g_stash4<TA, BM>(As, LDA_S, int(tid) + i * NT, ra[i]);
        G_UNROLL for (int i = 0; i < B_V; ++i) g_stash4<TB, BN>(Bs, LDB_S, int(tid) + i * NT, rb[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        G_UNROLL for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 af[FM], bf[FN];
            G_UNROLL for (int i = 0; i < FM; ++i) {
                const int r = sm * SG_M + i * 8;
                if (TA) simdgroup_load(af[i], As + kk * LDA_S + r, LDA_S, ulong2(0, 0), true);
                else    simdgroup_load(af[i], As + r * LDA_S + kk, LDA_S, ulong2(0, 0), false);
            }
            G_UNROLL for (int j = 0; j < FN; ++j) {
                const int c = sn * SG_N + j * 8;
                if (TB) simdgroup_load(bf[j], Bs + kk * LDB_S + c, LDB_S, ulong2(0, 0), false);
                else    simdgroup_load(bf[j], Bs + c * LDB_S + kk, LDB_S, ulong2(0, 0), true);
            }
            G_UNROLL for (int i = 0; i < FM; ++i)
                G_UNROLL for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The 8x8 accumulator's two elements this lane holds: (fr, fc) and (fr, fc + 1).
    const ushort q = lane / 4;
    const uint fr = uint((q & 4) + (lane / 2) % 4);
    const uint fc = uint((q & 2) * 2 + (lane % 2) * 2);
    G_UNROLL for (int i = 0; i < FM; ++i) {
        const uint m = block_m + uint(sm * SG_M + i * 8) + fr;
        G_UNROLL for (int j = 0; j < FN; ++j) {
            const uint n = block_n + uint(sn * SG_N + j * 8) + fc;
            if (m >= M || n >= N) continue;
            const thread auto& e = acc[i][j].thread_elements();
            g_store2(Cb, m, n, e[0], e[1], bias, p);
        }
    }
}

// ---- Skinny M (decode rows): B streams once, all M rows ride in registers ----
constant constexpr int SK_MAX_M = 8;
constant constexpr int SK_NT = 256;
constant constexpr int SK_COLS = 64;            // (K, N) B: columns per threadgroup
constant constexpr int SK_KL = SK_NT / (SK_COLS / 4);   // 16 K lanes
constant constexpr int SK_UNROLL = 4;

// src[0 .. 3] as floats, zeros past the first n (n may exceed 4).
inline float4 g_row4(device const TI* src, bool vec_ok, uint n) {
    if (vec_ok && n >= 4) return float4(*reinterpret_cast<device const TI4*>(src));
    float4 v = 0.0f;
    G_UNROLL for (int i = 0; i < 4; ++i) if (uint(i) < n) v[i] = float(src[i]);
    return v;
}

[[max_total_threads_per_threadgroup(SK_NT)]]
kernel void k_gemm_skinny(device const TI* A    [[buffer(0)]],
                          device const TI* B    [[buffer(1)]],
                          device TO*       C    [[buffer(2)]],
                          device const TI* bias [[buffer(3)]],
                          constant GemmParams& p [[buffer(4)]],
                          uint3  tg_pos [[threadgroup_position_in_grid]],
                          ushort tid    [[thread_index_in_threadgroup]],
                          ushort sg_id  [[simdgroup_index_in_threadgroup]],
                          ushort lane   [[thread_index_in_simdgroup]]) {
    const uint M = p.M, N = p.N, K = p.K;
    device const TI* Ab = A + ulong(tg_pos.z) * p.sA;
    device const TI* Bb = B + ulong(tg_pos.z) * p.sB;
    device TO*       Cb = C + ulong(tg_pos.z) * p.sC;
    const bool va = (p.vec & 1u) != 0u, vb = (p.vec & 2u) != 0u;
    threadgroup float red[TB ? SK_KL : 1][SK_COLS];   // (K, N) B only
    if (TB) {
        // B (K, N): thread (kl, cg) owns columns n0 .. n0 + 3 over the K rows
        // kl, kl + 16, ...; a simdgroup reads two whole 256-byte row segments.
        // The 16 K lanes then reduce through threadgroup memory.
        const uint kl = tid / (SK_COLS / 4), cg = tid % (SK_COLS / 4);
        const uint n0 = tg_pos.x * SK_COLS + cg * 4;
        const uint nv = n0 < N ? N - n0 : 0u;
        float4 acc[SK_MAX_M];
        G_UNROLL for (int m = 0; m < SK_MAX_M; ++m) acc[m] = 0.0f;
        uint k = kl;
        for (; k + (SK_UNROLL - 1) * SK_KL < K; k += SK_UNROLL * SK_KL) {
            float4 b[SK_UNROLL];
            G_UNROLL for (int u = 0; u < SK_UNROLL; ++u)
                b[u] = g_row4(Bb + ulong(k + u * SK_KL) * p.ldb + n0, vb, nv);
            G_UNROLL for (int u = 0; u < SK_UNROLL; ++u)
                G_UNROLL for (int m = 0; m < SK_MAX_M; ++m)
                    if (uint(m) < M) acc[m] += float(Ab[G_AT(TA, uint(m), k + u * SK_KL, p.lda)]) * b[u];
        }
        for (; k < K; k += SK_KL) {
            const float4 b = g_row4(Bb + ulong(k) * p.ldb + n0, vb, nv);
            G_UNROLL for (int m = 0; m < SK_MAX_M; ++m)
                if (uint(m) < M) acc[m] += float(Ab[G_AT(TA, uint(m), k, p.lda)]) * b;
        }
        for (uint m = 0; m < M; ++m) {
            *reinterpret_cast<threadgroup float4*>(&red[kl][cg * 4]) = acc[m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < SK_COLS) {
                float r = 0.0f;
                G_UNROLL for (int l = 0; l < SK_KL; ++l) r += red[l][tid];
                const uint n = tg_pos.x * SK_COLS + tid;
                if (n < N) g_store1(Cb, m, n, r, bias, p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }
    // B (N, K): one simdgroup per column n; lanes take four K each, then reduce.
    const uint n = tg_pos.x * (SK_NT / 32) + sg_id;
    if (n >= N) return;
    device const TI* brow = Bb + ulong(n) * p.ldb;
    float acc[SK_MAX_M];
    G_UNROLL for (int m = 0; m < SK_MAX_M; ++m) acc[m] = 0.0f;
    for (uint k = lane * 4; k < K; k += 128) {
        const float4 b = g_row4(brow + k, vb, K - k);
        G_UNROLL for (int m = 0; m < SK_MAX_M; ++m) {
            if (uint(m) >= M) continue;
            float4 a;
            if (TA) {
                a = 0.0f;
                G_UNROLL for (int i = 0; i < 4; ++i) if (k + uint(i) < K) a[i] = float(Ab[ulong(k + i) * p.lda + m]);
            } else {
                a = g_row4(Ab + ulong(m) * p.lda + k, va, K - k);
            }
            acc[m] += dot(a, b);
        }
    }
    for (uint m = 0; m < M; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0) g_store1(Cb, m, n, s, bias, p);
    }
}
)msl";

// Must match the MSL `GemmParams` byte-for-byte.
struct GemmParams {
    uint32_t M, N, K, has_bias;
    int32_t  act, epi;
    float    alpha;
    uint32_t vec;
    uint64_t lda, ldb, ldc, sA, sB, sC;
};

constexpr int kBM = 64, kBN = 64, kThreads = 128;
constexpr int kSkinnyMaxM = 8, kSkinnyThreads = 256, kSkinnyCols = 64;

const char* msl_type(AbtType t) {
    return t == kAbtF16 ? "half" : t == kAbtBF16 ? "bfloat" : "float";
}

id<MTLComputePipelineState> pso_for(bool skinny, bool ta, bool tb, AbtType in, AbtType out) {
    static std::mutex mu;
    static std::unordered_map<int, id<MTLComputePipelineState>> cache;
    const int key = (skinny ? 1 : 0) | (ta ? 2 : 0) | (tb ? 4 : 0) | (int(in) << 3) | (int(out) << 5);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    NSString* src = [NSString stringWithFormat:@"#define TA %d\n#define TB %d\n#define TI %s\n#define TO %s\n%@",
                                               ta ? 1 : 0, tb ? 1 : 0, msl_type(in), msl_type(out), kSrc];
    id<MTLComputePipelineState> pso = compile_pipeline(src, skinny ? @"k_gemm_skinny" : @"k_gemm_tiled");
    cache.emplace(key, pso);
    return pso;
}

// True when every batch slice's rows of an operand start on a 4-element
// boundary, so the kernel may read it four elements at a time.
bool rows_vec4(NSUInteger ofs_bytes, size_t esz, uint64_t ld, uint64_t stride, int batch) {
    if (ofs_bytes % (4 * esz) != 0 || ld % 4 != 0) return false;
    return batch <= 1 || stride % 4 == 0;
}

} // namespace

void launch_gemm_fp32(const AbtMixed& g) {
    if (g.in == kAbtF16) throw std::runtime_error("brotensor (Metal): launch_gemm_fp32: FP16 input takes the half GEMM");
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0 || g.K <= 0) return;
    GemmParams p{};
    p.M = static_cast<uint32_t>(g.M);
    p.N = static_cast<uint32_t>(g.N);
    p.K = static_cast<uint32_t>(g.K);
    p.has_bias = g.bias ? 1u : 0u;
    p.act = g.act;
    p.epi = g.epilogue;
    p.alpha = g.alpha;
    p.lda = g.lda;
    p.ldb = g.ldb;
    p.ldc = g.ldc;
    p.sA = g.strideA;
    p.sB = g.strideB;
    p.sC = g.strideC;
    const size_t esz = g.in == kAbtBF16 ? 2 : 4;
    p.vec = (rows_vec4(g.ofs_A, esz, g.lda, g.strideA, g.batch) ? 1u : 0u) |
            (rows_vec4(g.ofs_B, esz, g.ldb, g.strideB, g.batch) ? 2u : 0u);
    id<MTLBuffer> bBias = g.bias ? g.bias : g.A;
    const NSUInteger oBias = g.bias ? g.ofs_bias : g.ofs_A;

    const bool skinny = g.M <= kSkinnyMaxM && g.epilogue != kAbtGeglu;
    id<MTLComputePipelineState> pso = pso_for(skinny, g.transA, g.transB, g.in, g.out);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:g.A offset:g.ofs_A atIndex:0];
        [enc setBuffer:g.B offset:g.ofs_B atIndex:1];
        [enc setBuffer:g.C offset:g.ofs_C atIndex:2];
        [enc setBuffer:bBias offset:oBias atIndex:3];
        [enc setBytes:&p length:sizeof(GemmParams) atIndex:4];
        const NSUInteger nb = static_cast<NSUInteger>(g.batch);
        if (skinny) {
            const int cols_per_tg = g.transB ? kSkinnyCols : kSkinnyThreads / 32;
            const NSUInteger gx = static_cast<NSUInteger>((g.N + cols_per_tg - 1) / cols_per_tg);
            [enc dispatchThreadgroups:MTLSizeMake(gx, 1, nb) threadsPerThreadgroup:MTLSizeMake(kSkinnyThreads, 1, 1)];
        } else {
            const NSUInteger gx = static_cast<NSUInteger>((g.N + kBN - 1) / kBN);
            const NSUInteger gy = static_cast<NSUInteger>((g.M + kBM - 1) / kBM);
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, nb) threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
        }
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::metal_impl
