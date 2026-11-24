// FP32-accumulate GEMM on Metal: C[b](M, N) (op)= epilogue(alpha * op(A) @
// op(B) + bias) for FP16, BF16 or FP32 operands, FP16 / BF16 / FP32 output.
// This is the fast path behind launch_matmul_abt_mixed for every input type
// (see fp16_matmul.h for the AbtMixed contract) — it carries matmul (+ bwd),
// matmul_abt, the batched linears (fwd + bwd, FP16-weight / FP32-activation
// included) and the 1x1 conv GEMMs. FP16 x FP16 stages half tiles and
// multiplies simdgroup_half8x8 into simdgroup_float8x8 accumulators (the CUDA
// WMMA contract: half inputs, FP32 accumulate); anything else widens to FP32
// on load (exact for BF16 and FP16) and multiplies simdgroup_float8x8.
//
// Two kernels, each compiled per (transA, transB, A / B / out type) variant by
// prepending #defines to one source (a layout that is a compile-time constant
// keeps the loads and the simdgroup_load transposes branch-free):
//
//   k_gemm_tiled   64x64 output tile per threadgroup, BK = 16 (or 32 for
//                  half tiles on a large grid), four simdgroups in a 2x2 grid each
//                  owning 32x32 = 4x4 simdgroup_float8x8 accumulators; FP32
//                  ~3.5 TFLOP/s on M2 Pro at 4096^3 (MPS: ~5).
//                  Global loads are 4-wide along each operand's contiguous
//                  axis when its rows are 4-element aligned, scalar + masked at
//                  the M/N/K edges, so any shape and any stride is valid. The
//                  epilogue works straight from the accumulator registers
//                  (thread_elements), no staging tile. Measured and rejected
//                  on M2 Pro: a double-buffered tile, a register prefetch of
//                  the next K step (both ~10-20% slower: they cost occupancy),
//                  BK 32 for float tiles, and 128-wide or 32-wide tiles (for
//                  half tiles too). FP16 reaches ~4.2 TFLOP/s at 4096^3 (MPS:
//                  ~5.1; M2 Pro's FP16 and FP32 simdgroup rates are equal).
//   k_gemm_skinny  M <= kSkinnyMaxM (a GEMV-like decode row block), keeping
//                  all M rows' partial sums in registers so B streams once:
//                  B stored (K, N) is read in 4-wide column groups by 16 K
//                  lanes reduced through threadgroup memory; B stored (N, K)
//                  gives each output column a simdgroup reducing along K,
//                  with A staged through threadgroup memory when M > 1. The
//                  FP16 M = 1 GEMV streams B at ~165 GB/s (near DRAM rate).
//
// The long-K / small-output shape (1x1 conv weight gradient) is split over K
// by the caller (conv2d.mm) through the batch axis, then folded.

#include "fp16_matmul.h"
#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#import "internal.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace brotensor::metal_impl {

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Prepended per variant: TA, TB (0/1 — A stored (K, M) / B stored (K, N)),
// TIA / TIB (A's / B's element type: half / bfloat / float; bias is TIA),
// TO (half / bfloat / float), TS (the tile + multiply type: half when both
// inputs are half, else float) and BKV (the K step: 16, or 32 for half tiles
// on a large grid — the same bytes per tile row as float's 16).

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
inline float g_pre(float acc, uint n, device const TIA* bias, constant GemmParams& p) {
    float v = acc * p.alpha;
    if (p.has_bias) v += float(bias[n]);
    return g_act(p.act, v);
}

// Writes result r for output column n (or, for GeGLU, the pair n, n + 1 with
// n even) of row m. The caller has checked m < M and n < N.
inline void g_store1(device TO* Cb, uint m, uint n, float r, device const TIA* bias, constant GemmParams& p) {
    device TO* out = Cb + ulong(m) * p.ldc + n;
    float v = g_pre(r, n, bias, p);
    if (p.epi == 1) v += float(*out);
    *out = TO(v);
}
inline void g_store2(device TO* Cb, uint m, uint n, float r0, float r1, device const TIA* bias,
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

// 64x64 over a 2x2 simdgroup grid measured best on M2 Pro (see the file
// comment).
constant constexpr int BM = 64, BN = 64, BK = BKV;
constant constexpr int WM = 2, WN = 2;            // simdgroup grid
constant constexpr int NT = WM * WN * 32;
constant constexpr int SG_M = BM / WM, SG_N = BN / WN;   // per-simdgroup output block
constant constexpr int FM = SG_M / 8, FN = SG_N / 8;
constant constexpr int PAD = 4;                   // keeps 4-wide tile rows vector-aligned
// A tile: row-major (BM, BK + PAD) when A is (M, K); (BK, BM + PAD) when (K, M).
constant constexpr int LDA_S = TA ? BM + PAD : BK + PAD;
constant constexpr int LDB_S = TB ? BN + PAD : BK + PAD;
constant constexpr int A_TILE = TA ? BK * (BM + PAD) : BM * (BK + PAD);
constant constexpr int B_TILE = TB ? BK * (BN + PAD) : BN * (BK + PAD);
// 4-element slots per tile and per thread.
constant constexpr int A_V = BM * BK / 4 / NT;
constant constexpr int B_V = BN * BK / 4 / NT;

typedef vec<TS, 4> TS4;

// Loads the 4-element slot `s` (of the TR x BK tile) at (row0, k0) of an
// operand with R rows; zeros past the edges. The slot's four elements run
// along the operand's contiguous axis: k when stored (rows, K), rows when
// stored (K, rows).
template <typename TX, bool T, int TR>
inline TS4 g_load4(device const TX* X, ulong ld, bool vec_ok, uint R, uint K, uint row0, uint k0, int s) {
    constexpr int per = T ? TR / 4 : BK / 4;
    const int a = s / per, b = (s % per) * 4;
    const uint row = T ? row0 + uint(b) : row0 + uint(a);
    const uint k   = T ? k0 + uint(a)   : k0 + uint(b);
    device const TX* src = X + G_AT(T, row, k, ld);
    const bool in4 = T ? (k < K && row + 3 < R) : (row < R && k + 3 < K);
    if (vec_ok && in4) return TS4(*reinterpret_cast<device const vec<TX, 4>*>(src));
    TS4 v = TS4(0);
    G_UNROLL for (int i = 0; i < 4; ++i) {
        const bool ok = T ? (k < K && row + uint(i) < R) : (row < R && k + uint(i) < K);
        if (ok) v[i] = TS(src[i]);
    }
    return v;
}

// Stores slot `s` (the same slot g_load4 read) into the tile.
template <bool T, int TR>
inline void g_stash4(threadgroup TS* S, int ld, int s, TS4 v) {
    constexpr int per = T ? TR / 4 : BK / 4;
    const int a = s / per, b = (s % per) * 4;
    *reinterpret_cast<threadgroup TS4*>(S + a * ld + b) = v;
}

[[max_total_threads_per_threadgroup(NT)]]
kernel void k_gemm_tiled(device const TIA* A    [[buffer(0)]],
                         device const TIB* B    [[buffer(1)]],
                         device TO*        C    [[buffer(2)]],
                         device const TIA* bias [[buffer(3)]],
                         constant GemmParams& p [[buffer(4)]],
                         uint3  tg_pos [[threadgroup_position_in_grid]],
                         ushort tid    [[thread_index_in_threadgroup]],
                         ushort sg_id  [[simdgroup_index_in_threadgroup]],
                         ushort lane   [[thread_index_in_simdgroup]]) {
    const uint M = p.M, N = p.N, K = p.K;
    device const TIA* Ab = A + ulong(tg_pos.z) * p.sA;
    device const TIB* Bb = B + ulong(tg_pos.z) * p.sB;
    device TO*        Cb = C + ulong(tg_pos.z) * p.sC;
    const uint block_m = tg_pos.y * BM, block_n = tg_pos.x * BN;
    const bool va = (p.vec & 1u) != 0u, vb = (p.vec & 2u) != 0u;

    threadgroup TS As[A_TILE];
    threadgroup TS Bs[B_TILE];

    const int sm = int(sg_id) / WN, sn = int(sg_id) % WN;
    simdgroup_float8x8 acc[FM][FN];
    G_UNROLL for (int i = 0; i < FM; ++i)
        G_UNROLL for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Global -> registers -> the tile. Loading straight into the tile after
        // the previous step's trailing barrier beats a register prefetch of
        // the next tile or a second tile buffer: both cost occupancy.
        TS4 ra[A_V], rb[B_V];
        G_UNROLL for (int i = 0; i < A_V; ++i) ra[i] = g_load4<TIA, TA, BM>(Ab, p.lda, va, M, K, block_m, k0, int(tid) + i * NT);
        G_UNROLL for (int i = 0; i < B_V; ++i) rb[i] = g_load4<TIB, TB, BN>(Bb, p.ldb, vb, N, K, block_n, k0, int(tid) + i * NT);
        G_UNROLL for (int i = 0; i < A_V; ++i) g_stash4<TA, BM>(As, LDA_S, int(tid) + i * NT, ra[i]);
        G_UNROLL for (int i = 0; i < B_V; ++i) g_stash4<TB, BN>(Bs, LDB_S, int(tid) + i * NT, rb[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        G_UNROLL for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<TS, 8, 8> af[FM], bf[FN];
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
constexpr size_t kBk32MinGroups = 192;

const char* msl_type(AbtType t) {
    return t == kAbtF16 ? "half" : t == kAbtBF16 ? "bfloat" : "float";
}

::brotensor::Dtype abt_to_dtype(AbtType t) {
    return t == kAbtF16 ? ::brotensor::Dtype::FP16
         : t == kAbtBF16 ? ::brotensor::Dtype::BF16
         : ::brotensor::Dtype::FP32;
}

// One pipeline per (layout, A / B / output type, K step) variant. bk is 16 or
// 32 (32 only for half tiles on a large grid).
id<MTLComputePipelineState> pso_for(bool ta, bool tb, AbtType in_a, AbtType in_b, AbtType out, int bk) {
    const bool half_tiles = in_a == kAbtF16 && in_b == kAbtF16;
    if (!half_tiles) bk = 16;
    static std::mutex mu;
    static std::unordered_map<int, id<MTLComputePipelineState>> cache;
    const int key = (ta ? 2 : 0) | (tb ? 4 : 0) | (int(in_a) << 3) | (int(in_b) << 5) |
                    (int(out) << 7) | (bk == 32 ? 512 : 0);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    NSString* src = [NSString stringWithFormat:@"#define TA %d\n#define TB %d\n#define TIA %s\n#define TIB %s\n"
                                               @"#define TO %s\n#define TS %s\n#define BKV %d\n%@",
                                               ta ? 1 : 0, tb ? 1 : 0, msl_type(in_a), msl_type(in_b), msl_type(out),
                                               half_tiles ? "half" : "float", bk, kSrc];
    id<MTLComputePipelineState> pso = compile_pipeline(src, @"k_gemm_tiled");
    cache.emplace(key, pso);
    return pso;
}

bool rows_vec4(NSUInteger ofs_bytes, size_t esz, uint64_t ld, uint64_t stride, int batch) {
    if (ofs_bytes % (4 * esz) != 0 || ld % 4 != 0) return false;
    return batch <= 1 || stride % 4 == 0;
}

size_t type_size(AbtType t) { return t == kAbtF32 ? 4 : 2; }

bool try_mir_gemv(const AbtMixed& g, AbtType in_b) {
    if (g.transA || g.transB || g.batch != 1 || g.M < 1 || g.M > 8 || g.K <= 0 || g.alpha != 1.0f)
        return false;
    if (g.epilogue < kAbtStore || g.epilogue > kAbtSwiglu) return false;
    const ::brotensor::Dtype da = abt_to_dtype(g.in);
    const ::brotensor::Dtype db = abt_to_dtype(in_b);
    const ::brotensor::Dtype dc = abt_to_dtype(g.out);
    if (da != dc) return false;
    if (db != da && !(da == ::brotensor::Dtype::FP32 && db != ::brotensor::Dtype::FP32)) return false;
    if (g.lda > 0x7fffffffULL || g.ldb > 0x7fffffffULL || g.ldc > 0x7fffffffULL) return false;

    using namespace ::brotensor::detail::metal;
    const std::uint32_t nout =
        static_cast<std::uint32_t>((g.epilogue == kAbtGeglu || g.epilogue == kAbtSwiglu) ? g.N / 2 : g.N);
    if (nout == 0) return true;
    const bool vec = (g.ofs_A % 16u == 0u) && (g.ofs_B % 16u == 0u) &&
                     (g.lda % 8u == 0u) && (g.ldb % 8u == 0u) && (g.K % 8 == 0);
    const std::uint32_t op = static_cast<std::uint32_t>(g.M) |
                             (static_cast<std::uint32_t>(g.epilogue) << 4) |
                             (static_cast<std::uint32_t>(g.act) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (g.bias ? 2u : 0u));
    const MirKernel mk = mir_kernel({MirKind::Gemv, op, da, db, 1, dc, flags});
    const std::uint64_t groups = (static_cast<std::uint64_t>(nout) + 3u) / 4u;
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(1u, std::min<std::uint64_t>(groups, 65535u)));
    const std::uint32_t gy = static_cast<std::uint32_t>((groups + gx - 1u) / gx);
    const void* bufs[4] = {
        (__bridge const void*)g.A, (__bridge const void*)g.B,
        (__bridge const void*)g.C, (__bridge const void*)g.bias,
    };
    const std::size_t offs[4] = {g.ofs_A, g.ofs_B, g.ofs_C, g.ofs_bias};
    const std::int32_t ints[7] = {
        g.M, g.N, g.K,
        static_cast<std::int32_t>(g.lda), static_cast<std::int32_t>(g.ldb),
        static_cast<std::int32_t>(g.ldc), g.N / 2,
    };
    mir_dispatch_mtl_3d(mk, gx, gy, 1u, bufs, offs, 4, ints, {});
    return true;
}

} // namespace

void launch_gemm_fp32(const AbtMixed& g) {
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0 || g.K <= 0) return;
    const AbtType in_b = g.in_B < 0 ? g.in : static_cast<AbtType>(g.in_B);
    if (try_mir_gemv(g, in_b)) return;

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
    p.vec = (rows_vec4(g.ofs_A, type_size(g.in), g.lda, g.strideA, g.batch) ? 1u : 0u) |
            (rows_vec4(g.ofs_B, type_size(in_b), g.ldb, g.strideB, g.batch) ? 2u : 0u);
    id<MTLBuffer> bBias = g.bias ? g.bias : g.A;
    const NSUInteger oBias = g.bias ? g.ofs_bias : g.ofs_A;

    const size_t tgs = static_cast<size_t>((g.N + kBN - 1) / kBN) * ((g.M + kBM - 1) / kBM) * g.batch;
    const int bk = tgs >= kBk32MinGroups ? 32 : 16;
    id<MTLComputePipelineState> pso = pso_for(g.transA, g.transB, g.in, in_b, g.out, bk);
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
        const NSUInteger gx = static_cast<NSUInteger>((g.N + kBN - 1) / kBN);
        const NSUInteger gy = static_cast<NSUInteger>((g.M + kBM - 1) / kBM);
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, nb) threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::metal_impl
