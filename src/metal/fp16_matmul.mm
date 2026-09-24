// Tiled FP16 matmul on Metal using simdgroup_matrix<half, 8, 8>.
//
// Computes C(M, N) = A(M, K) @ B(N, K)^T, FP16 storage with FP32 accumulator,
// optionally batched (per-matrix strides), with a fused per-N bias and an
// epilogue activation. Memory layout within a batch: A row-major (M, K),
// B row-major (N, K), C row-major (M, N).
//
// Threadgroup tile: BM=64, BN=64, BK=32 — mirrors the CUDA WMMA GEMM and the
// in-tree conv2d_wmma implicit-GEMM (src/metal/conv2d_wmma.mm), the best
// reference kernel here. 4 simdgroups in a 2x2 grid, 32 threads each = 128
// threads/threadgroup; each simdgroup owns a 32x32 output region covered by a
// 4x4 grid of 8x8 simdgroup_matrix tiles. Shared A/B tiles carry an 8-element
// pad on the leading dim (LDA/LDB = BK+8) to avoid threadgroup-memory bank
// conflicts; the FP32 epilogue stages through Cs (LDC = BN+8).
//
// The tiled kernel masks all three of M, N, K by zero-filling the shared tiles
// on out-of-range global loads (partial K-tiles contribute zero, partial N/M
// rows are skipped on store), so it is correct for ARBITRARY M, N, K — no
// alignment precondition. Small problems (M*N below kTiledMin) and K==0 take
// the one-thread-per-output naive kernel, which also carries batch/bias/act.
//
// BF16 has no simdgroup_matrix form, so it keeps a naive per-thread GEMM.

#include "fp16_matmul.h"

#import "internal.h"

#include <mutex>
#include <stdexcept>

namespace brotensor::metal_impl {

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct AbtParams {
    uint  M, N, K, has_bias;
    int   act;
    uint  _pad;
    ulong sA, sB, sC;
};

// ---- Fused activation epilogue (matches src/cuda/fp16_matmul.cu act codes) --
// MSL has no built-in erf; Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
inline float abt_erf_approx(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign_x = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t  = 1.0f / (1.0f + p * ax);
    float y  = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}

inline float abt_apply_act(int act, float v) {
    switch (act) {
        case 1:  return max(v, 0.0f);
        case 2: {
            constexpr float kSqrt2OverPi = 0.7978845608f;
            float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: {
            constexpr float kInvSqrt2 = 0.70710678118654752440f;
            return 0.5f * v * (1.0f + abt_erf_approx(v * kInvSqrt2));
        }
        case 4:  return v / (1.0f + exp(-v));
        case 5:  return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}

// ---------------- Naive fallback (one thread per output, batched) -----------
kernel void k_matmul_abt_fp16_naive(device const half*   A    [[buffer(0)]],
                                    device const half*   B    [[buffer(1)]],
                                    device half*         C    [[buffer(2)]],
                                    device const half*   bias [[buffer(3)]],
                                    constant AbtParams&  p    [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint idx = gid.x;
    const uint b   = gid.y;
    const uint total = p.M * p.N;
    if (idx >= total) return;
    device const half* Ab = A + b * p.sA;
    device const half* Bb = B + b * p.sB;
    device half*       Cb = C + b * p.sC;
    const uint m = idx / p.N;
    const uint n = idx % p.N;
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        acc += float(Ab[m * p.K + k]) * float(Bb[n * p.K + k]);
    }
    if (p.has_bias) acc += float(bias[n]);
    acc = abt_apply_act(p.act, acc);
    Cb[idx] = half(acc);
}

// ---------------- BF16 naive (no simdgroup form for bfloat) -----------------
kernel void k_matmul_abt_bf16_naive(device const bfloat* A    [[buffer(0)]],
                                    device const bfloat* B    [[buffer(1)]],
                                    device bfloat*       C    [[buffer(2)]],
                                    device const bfloat* bias [[buffer(3)]],
                                    constant AbtParams&  p    [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint idx = gid.x;
    const uint b   = gid.y;
    const uint total = p.M * p.N;
    if (idx >= total) return;
    device const bfloat* Ab = A + b * p.sA;
    device const bfloat* Bb = B + b * p.sB;
    device bfloat*       Cb = C + b * p.sC;
    const uint m = idx / p.N;
    const uint n = idx % p.N;
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        acc += float(Ab[m * p.K + k]) * float(Bb[n * p.K + k]);
    }
    if (p.has_bias) acc += float(bias[n]);
    acc = abt_apply_act(p.act, acc);
    Cb[idx] = bfloat(acc);
}

// ---------------- Tiled simdgroup-matrix kernel (batched) -------------------
constant constexpr int BM = 64;
constant constexpr int BN = 64;
constant constexpr int BK = 32;
constant constexpr int WARPS_M = 2;
constant constexpr int WARPS_N = 2;
constant constexpr int THREADS_PER_TG = 128;   // WARPS_M * WARPS_N * 32
constant constexpr int WM = BM / WARPS_M;       // 32
constant constexpr int WN = BN / WARPS_N;       // 32
constant constexpr int FRAGS_M = WM / 8;        // 4
constant constexpr int FRAGS_N = WN / 8;        // 4
constant constexpr int FRAGS_K = BK / 8;        // 4
constant constexpr int LDA = BK + 8;            // pad to dodge bank conflicts
constant constexpr int LDB = BK + 8;
constant constexpr int LDC = BN + 8;

[[max_total_threads_per_threadgroup(THREADS_PER_TG)]]
kernel void k_matmul_abt_fp16_simdgroup(device const half*  A    [[buffer(0)]],
                                        device const half*  B    [[buffer(1)]],
                                        device half*        C    [[buffer(2)]],
                                        device const half*  bias [[buffer(3)]],
                                        constant AbtParams& p    [[buffer(4)]],
                                        uint3 tg_pos [[threadgroup_position_in_grid]],
                                        uint  tid    [[thread_index_in_threadgroup]],
                                        uint  sg_id  [[simdgroup_index_in_threadgroup]]) {
    const uint M = p.M, N = p.N, K = p.K;
    const uint b = tg_pos.z;
    device const half* Ab = A + (ulong)b * p.sA;
    device const half* Bb = B + (ulong)b * p.sB;
    device half*       Cb = C + (ulong)b * p.sC;

    const int block_m = int(tg_pos.y) * BM;
    const int block_n = int(tg_pos.x) * BN;
    const int warp_m = int(sg_id) / WARPS_N;
    const int warp_n = int(sg_id) % WARPS_N;

    threadgroup half As[BM * LDA];
    threadgroup half Bs[BN * LDB];

    simdgroup_matrix<float, 8, 8> c_frag[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i)
        for (int j = 0; j < FRAGS_N; ++j)
            c_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // A tile (BM x BK), row = M index, col = K index. Masked on M and K.
        {
            constexpr int kPerThr = (BM * BK) / THREADS_PER_TG;   // 16
            for (int li = 0; li < kPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int r   = lin / BK;              // 0..BM-1
                const int c   = lin - r * BK;          // 0..BK-1
                const int m_g = block_m + r;
                const uint gk = k0 + uint(c);
                half v = half(0);
                if (m_g < int(M) && gk < K) v = Ab[uint(m_g) * K + gk];
                As[r * LDA + c] = v;
            }
        }
        // B tile (BN x BK), row = N index, col = K index. Masked on N and K.
        {
            constexpr int kPerThr = (BN * BK) / THREADS_PER_TG;   // 16
            for (int li = 0; li < kPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int r   = lin / BK;              // 0..BN-1
                const int c   = lin - r * BK;          // 0..BK-1
                const int n_g = block_n + r;
                const uint gk = k0 + uint(c);
                half v = half(0);
                if (n_g < int(N) && gk < K) v = Bb[uint(n_g) * K + gk];
                Bs[r * LDB + c] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < FRAGS_K; ++kk) {
            simdgroup_matrix<half, 8, 8> a_frag[FRAGS_M];
            simdgroup_matrix<half, 8, 8> b_frag[FRAGS_N];
            for (int i = 0; i < FRAGS_M; ++i) {
                const int a_row = warp_m * WM + i * 8;
                simdgroup_load(a_frag[i], As + a_row * LDA + kk * 8, LDA,
                               ulong2(0, 0), false);
            }
            // B stored row-major (N, K). The MMA right operand wants (K, N):
            // load with transpose=true so the 8x8 frag is B^T[k, n].
            for (int j = 0; j < FRAGS_N; ++j) {
                const int b_row = warp_n * WN + j * 8;   // n
                simdgroup_load(b_frag[j], Bs + b_row * LDB + kk * 8, LDB,
                               ulong2(0, 0), true);
            }
            for (int i = 0; i < FRAGS_M; ++i)
                for (int j = 0; j < FRAGS_N; ++j)
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i],
                                                  b_frag[j], c_frag[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: stage FP32 frags to Cs, then write to C with bias + activation.
    threadgroup float Cs[BM * LDC];
    for (int i = 0; i < FRAGS_M; ++i)
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_row = warp_m * WM + i * 8;
            const int c_col = warp_n * WN + j * 8;
            simdgroup_store(c_frag[i][j], Cs + c_row * LDC + c_col, LDC,
                            ulong2(0, 0), false);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        constexpr int kPerThr = (BM * BN) / THREADS_PER_TG;   // 32
        for (int si = 0; si < kPerThr; ++si) {
            const int lin = int(tid) + si * THREADS_PER_TG;
            const int r   = lin / BN;
            const int c   = lin - r * BN;
            const int m_g = block_m + r;
            const int n_g = block_n + c;
            if (m_g >= int(M) || n_g >= int(N)) continue;
            float v = Cs[r * LDC + c];
            if (p.has_bias) v += float(bias[n_g]);
            v = abt_apply_act(p.act, v);
            Cb[uint(m_g) * N + uint(n_g)] = half(v);
        }
    }
}
)msl";

id<MTLComputePipelineState> pso_tiled() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_abt_fp16_simdgroup"); });
    return pso;
}
id<MTLComputePipelineState> pso_naive() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_abt_fp16_naive"); });
    return pso;
}
id<MTLComputePipelineState> pso_naive_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_abt_bf16_naive"); });
    return pso;
}

// Parameter block — must match the MSL `AbtParams` struct byte-for-byte.
struct AbtParams {
    uint32_t M, N, K, has_bias;
    int32_t  act;
    uint32_t _pad;
    uint64_t sA, sB, sC;
};

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kThreadsPerTG = 128;   // WARPS_M * WARPS_N * 32, matches the MSL
constexpr size_t kTiledMin = 1024;   // below this M*N, the naive path wins

} // namespace

void launch_matmul_abt_fp16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    if (batch <= 0 || M == 0 || N == 0) return;

    AbtParams p{};
    p.M = static_cast<uint32_t>(M);
    p.N = static_cast<uint32_t>(N);
    p.K = static_cast<uint32_t>(K < 0 ? 0 : K);
    p.has_bias = has_bias ? 1u : 0u;
    p.act = static_cast<int32_t>(act);
    p.sA = strideA;
    p.sB = strideB;
    p.sC = strideC;

    // Dummy bias bind when has_bias=false; the kernel never reads it.
    id<MTLBuffer> bBias = has_bias ? bias : A;
    const NSUInteger oBias = has_bias ? ofs_bias : ofs_A;

    const bool tiled = (K > 0) &&
        (static_cast<size_t>(M) * static_cast<size_t>(N) >= kTiledMin);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (tiled) {
            id<MTLComputePipelineState> pso = pso_tiled();
            [enc setComputePipelineState:pso];
            [enc setBuffer:A offset:ofs_A atIndex:0];
            [enc setBuffer:B offset:ofs_B atIndex:1];
            [enc setBuffer:C offset:ofs_C atIndex:2];
            [enc setBuffer:bBias offset:oBias atIndex:3];
            [enc setBytes:&p length:sizeof(AbtParams) atIndex:4];
            const NSUInteger gx = static_cast<NSUInteger>((N + kBN - 1) / kBN);
            const NSUInteger gy = static_cast<NSUInteger>((M + kBM - 1) / kBM);
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, static_cast<NSUInteger>(batch))
                threadsPerThreadgroup:MTLSizeMake(kThreadsPerTG, 1, 1)];
        } else {
            id<MTLComputePipelineState> pso = pso_naive();
            [enc setComputePipelineState:pso];
            [enc setBuffer:A offset:ofs_A atIndex:0];
            [enc setBuffer:B offset:ofs_B atIndex:1];
            [enc setBuffer:C offset:ofs_C atIndex:2];
            [enc setBuffer:bBias offset:oBias atIndex:3];
            [enc setBytes:&p length:sizeof(AbtParams) atIndex:4];
            const NSUInteger total = static_cast<NSUInteger>(M) * static_cast<NSUInteger>(N);
            NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(total, static_cast<NSUInteger>(batch), 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void launch_matmul_abt_bf16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    // BF16 has no simdgroup_matrix form, so this is naive-only (one thread per
    // output). It still carries batch strides + bias + activation so the public
    // matmul_abt op has a real BF16 path.
    if (batch <= 0 || M == 0 || N == 0) return;
    AbtParams p{};
    p.M = static_cast<uint32_t>(M);
    p.N = static_cast<uint32_t>(N);
    p.K = static_cast<uint32_t>(K < 0 ? 0 : K);
    p.has_bias = has_bias ? 1u : 0u;
    p.act = static_cast<int32_t>(act);
    p.sA = strideA;
    p.sB = strideB;
    p.sC = strideC;
    id<MTLBuffer> bBias = has_bias ? bias : A;
    const NSUInteger oBias = has_bias ? ofs_bias : ofs_A;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_naive_bf16();
        [enc setComputePipelineState:pso];
        [enc setBuffer:A offset:ofs_A atIndex:0];
        [enc setBuffer:B offset:ofs_B atIndex:1];
        [enc setBuffer:C offset:ofs_C atIndex:2];
        [enc setBuffer:bBias offset:oBias atIndex:3];
        [enc setBytes:&p length:sizeof(AbtParams) atIndex:4];
        const NSUInteger total = static_cast<NSUInteger>(M) * static_cast<NSUInteger>(N);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, static_cast<NSUInteger>(batch), 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// ─── Mixed-precision A @ B^T (see fp16_matmul.h) ────────────────────────────

namespace {

NSString* const kMixSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MixParams {
    uint  M, N, K, has_bias;
    int   act, epi;
    float alpha;
    uint  trans;     // bit 0: A stored (K, M); bit 1: B stored (K, N)
    ulong lda, ldb, ldc, sA, sB, sC;
};

// Element (row, k) of an operand stored row-major, or transposed.
inline ulong mix_at(bool t, uint row, uint k, ulong ld) { return t ? ulong(k) * ld + row : ulong(row) * ld + k; }

// A & S 7.1.26, |err| < 1.5e-7 (MSL has no erf).
inline float mix_erf(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return x < 0.0f ? -y : y;
}
inline float mix_gelu_erf(float v) { return 0.5f * v * (1.0f + mix_erf(v * 0.70710678118654752f)); }
inline float mix_act(int act, float v) {
    switch (act) {
        case 1: return max(v, 0.0f);
        case 2: {
            const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: return mix_gelu_erf(v);
        case 4: return v / (1.0f + exp(-v));
        case 5: return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}

// The simdgroup_matrix element type an input type is multiplied in: FP16
// natively, BF16 / FP32 widened to FP32 (exact for both).
template <typename T> struct MixMma { typedef float type; };
template <> struct MixMma<half> { typedef half type; };

// r = act(alpha * acc + bias[n]); the epilogue then stores / adds / gates it.
template <typename TI>
inline float mix_pre(float acc, uint n, device const TI* bias, constant MixParams& p) {
    float v = acc * p.alpha;
    if (p.has_bias) v += float(bias[n]);
    return mix_act(p.act, v);
}

template <typename TI, typename TO>
kernel void k_abt_mixed_naive(device const TI* A    [[buffer(0)]],
                              device const TI* B    [[buffer(1)]],
                              device TO*       C    [[buffer(2)]],
                              device const TI* bias [[buffer(3)]],
                              constant MixParams& p [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const bool geglu = p.epi == 2;
    const uint cols = geglu ? p.N / 2 : p.N;
    if (gid.x >= p.M * cols) return;
    const uint m = gid.x / cols, j = gid.x % cols;
    const bool ta = (p.trans & 1u) != 0u, tb = (p.trans & 2u) != 0u;
    device const TI* Ab = A + (ulong)gid.y * p.sA;
    device const TI* Bb = B + (ulong)gid.y * p.sB;
    device TO* c = C + (ulong)gid.y * p.sC + (ulong)m * p.ldc + j;
    if (geglu) {
        const uint n0 = 2 * j;
        float a0 = 0.0f, a1 = 0.0f;
        for (uint k = 0; k < p.K; ++k) {
            const float a = float(Ab[mix_at(ta, m, k, p.lda)]);
            a0 += a * float(Bb[mix_at(tb, n0, k, p.ldb)]);
            a1 += a * float(Bb[mix_at(tb, n0 + 1, k, p.ldb)]);
        }
        *c = TO(mix_pre(a0, n0, bias, p) * mix_gelu_erf(mix_pre(a1, n0 + 1, bias, p)));
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) acc += float(Ab[mix_at(ta, m, k, p.lda)]) * float(Bb[mix_at(tb, j, k, p.ldb)]);
    float v = mix_pre(acc, j, bias, p);
    if (p.epi == 1) v += float(*c);
    *c = TO(v);
}

constant constexpr int MBM = 64, MBN = 64, MBK = 32;
constant constexpr int MWARPS_N = 2;
constant constexpr int MTHREADS = 128;
constant constexpr int MWM = 32, MWN = 32;
constant constexpr int MLDA = MBK + 8, MLDB = MBK + 8, MLDC = MBN + 8;
// A + B tiles (widest case: FP32) and the FP32 C staging tile share storage.
constant constexpr int MSMEM = (MBM * MLDA + MBN * MLDB) > (MBM * MLDC)
                                   ? (MBM * MLDA + MBN * MLDB) : (MBM * MLDC);

template <typename TI, typename TO>
[[max_total_threads_per_threadgroup(128)]]
kernel void k_abt_mixed_tiled(device const TI* A    [[buffer(0)]],
                              device const TI* B    [[buffer(1)]],
                              device TO*       C    [[buffer(2)]],
                              device const TI* bias [[buffer(3)]],
                              constant MixParams& p [[buffer(4)]],
                              uint3 tg_pos [[threadgroup_position_in_grid]],
                              uint  tid    [[thread_index_in_threadgroup]],
                              uint  sg_id  [[simdgroup_index_in_threadgroup]]) {
    typedef typename MixMma<TI>::type TM;
    const uint M = p.M, N = p.N, K = p.K;
    device const TI* Ab = A + (ulong)tg_pos.z * p.sA;
    device const TI* Bb = B + (ulong)tg_pos.z * p.sB;
    device TO*       Cb = C + (ulong)tg_pos.z * p.sC;
    const int block_m = int(tg_pos.y) * MBM;
    const int block_n = int(tg_pos.x) * MBN;
    const int warp_m = int(sg_id) / MWARPS_N;
    const int warp_n = int(sg_id) % MWARPS_N;
    const bool ta = (p.trans & 1u) != 0u, tb = (p.trans & 2u) != 0u;

    threadgroup float smem[MSMEM];
    threadgroup TM* As = reinterpret_cast<threadgroup TM*>(smem);
    threadgroup TM* Bs = As + MBM * MLDA;

    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += MBK) {
        for (int li = 0; li < (MBM * MBK) / MTHREADS; ++li) {
            const int lin = int(tid) + li * MTHREADS;
            const int r = lin / MBK, c = lin - r * MBK;
            const int m_g = block_m + r;
            const uint gk = k0 + uint(c);
            As[r * MLDA + c] = (m_g < int(M) && gk < K) ? TM(float(Ab[mix_at(ta, uint(m_g), gk, p.lda)])) : TM(0);
        }
        for (int li = 0; li < (MBN * MBK) / MTHREADS; ++li) {
            const int lin = int(tid) + li * MTHREADS;
            const int r = lin / MBK, c = lin - r * MBK;
            const int n_g = block_n + r;
            const uint gk = k0 + uint(c);
            Bs[r * MLDB + c] = (n_g < int(N) && gk < K) ? TM(float(Bb[mix_at(tb, uint(n_g), gk, p.ldb)])) : TM(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < MBK / 8; ++kk) {
            simdgroup_matrix<TM, 8, 8> af[4], bf[4];
            for (int i = 0; i < 4; ++i)
                simdgroup_load(af[i], As + (warp_m * MWM + i * 8) * MLDA + kk * 8, MLDA, ulong2(0, 0), false);
            for (int j = 0; j < 4; ++j)
                simdgroup_load(bf[j], Bs + (warp_n * MWN + j * 8) * MLDB + kk * 8, MLDB, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = smem;   // the operand tiles are dead past the last barrier
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Cs + (warp_m * MWM + i * 8) * MLDC + warp_n * MWN + j * 8, MLDC,
                            ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.epi == 2) {
        // GeGLU: column pairs (2j, 2j+1) never straddle a tile (MBN and block_n even).
        for (int si = 0; si < (MBM * MBN / 2) / MTHREADS; ++si) {
            const int lin = int(tid) + si * MTHREADS;
            const int r = lin / (MBN / 2), cp = lin - r * (MBN / 2);
            const int m_g = block_m + r, n0 = block_n + 2 * cp;
            if (m_g >= int(M) || n0 >= int(N)) continue;
            const float a = mix_pre(Cs[r * MLDC + 2 * cp], uint(n0), bias, p);
            const float g = mix_pre(Cs[r * MLDC + 2 * cp + 1], uint(n0 + 1), bias, p);
            Cb[(ulong)m_g * p.ldc + uint(n0 / 2)] = TO(a * mix_gelu_erf(g));
        }
        return;
    }
    for (int si = 0; si < (MBM * MBN) / MTHREADS; ++si) {
        const int lin = int(tid) + si * MTHREADS;
        const int r = lin / MBN, c = lin - r * MBN;
        const int m_g = block_m + r, n_g = block_n + c;
        if (m_g >= int(M) || n_g >= int(N)) continue;
        device TO* out = Cb + (ulong)m_g * p.ldc + uint(n_g);
        float v = mix_pre(Cs[r * MLDC + c], uint(n_g), bias, p);
        if (p.epi == 1) v += float(*out);
        *out = TO(v);
    }
}

#define MIX_INST(TI, TO, SUF)                                                                           \
template [[host_name("k_abt_mixed_naive_" SUF)]] kernel void k_abt_mixed_naive<TI, TO>(                 \
    device const TI*, device const TI*, device TO*, device const TI*, constant MixParams&, uint2);       \
template [[host_name("k_abt_mixed_tiled_" SUF)]] kernel void k_abt_mixed_tiled<TI, TO>(                 \
    device const TI*, device const TI*, device TO*, device const TI*, constant MixParams&, uint3, uint, uint);
MIX_INST(half, half, "0_0")
MIX_INST(half, bfloat, "0_1")
MIX_INST(half, float, "0_2")
MIX_INST(bfloat, half, "1_0")
MIX_INST(bfloat, bfloat, "1_1")
MIX_INST(bfloat, float, "1_2")
MIX_INST(float, half, "2_0")
MIX_INST(float, bfloat, "2_1")
MIX_INST(float, float, "2_2")
)msl";

// Must match the MSL `MixParams` byte-for-byte.
struct MixParams {
    uint32_t M, N, K, has_bias;
    int32_t  act, epi;
    float    alpha;
    uint32_t trans;
    uint64_t lda, ldb, ldc, sA, sB, sC;
};

id<MTLComputePipelineState> pso_mixed(bool tiled, int in, int out) {
    static std::once_flag once[2][3][3];
    static id<MTLComputePipelineState> pso[2][3][3];
    std::call_once(once[tiled][in][out], [&] {
        NSString* name = [NSString stringWithFormat:@"k_abt_mixed_%s_%d_%d", tiled ? "tiled" : "naive", in, out];
        pso[tiled][in][out] = compile_pipeline(kMixSrc, name);
    });
    return pso[tiled][in][out];
}

} // namespace

AbtType abt_type(::brotensor::Dtype dt) {
    switch (dt) {
        case ::brotensor::Dtype::FP16: return kAbtF16;
        case ::brotensor::Dtype::BF16: return kAbtBF16;
        case ::brotensor::Dtype::FP32: return kAbtF32;
        default: throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: dtype must be FP16, BF16 or FP32");
    }
}

void launch_matmul_abt_mixed(const AbtMixed& g) {
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0) return;
    if (g.epilogue == kAbtGeglu && (g.N % 2) != 0)
        throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: GeGLU needs an even N");
    MixParams p{};
    p.M = static_cast<uint32_t>(g.M);
    p.N = static_cast<uint32_t>(g.N);
    p.K = static_cast<uint32_t>(g.K < 0 ? 0 : g.K);
    p.has_bias = g.bias ? 1u : 0u;
    p.act = g.act;
    p.epi = g.epilogue;
    p.alpha = g.alpha;
    p.trans = (g.transA ? 1u : 0u) | (g.transB ? 2u : 0u);
    p.lda = g.lda;
    p.ldb = g.ldb;
    p.ldc = g.ldc;
    p.sA = g.strideA;
    p.sB = g.strideB;
    p.sC = g.strideC;
    id<MTLBuffer> bBias = g.bias ? g.bias : g.A;
    const NSUInteger oBias = g.bias ? g.ofs_bias : g.ofs_A;

    const bool tiled = p.K > 0 && static_cast<size_t>(g.M) * static_cast<size_t>(g.N) >= kTiledMin;
    id<MTLComputePipelineState> pso = pso_mixed(tiled, g.in, g.out);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:g.A offset:g.ofs_A atIndex:0];
        [enc setBuffer:g.B offset:g.ofs_B atIndex:1];
        [enc setBuffer:g.C offset:g.ofs_C atIndex:2];
        [enc setBuffer:bBias offset:oBias atIndex:3];
        [enc setBytes:&p length:sizeof(MixParams) atIndex:4];
        if (tiled) {
            const NSUInteger gx = static_cast<NSUInteger>((g.N + kBN - 1) / kBN);
            const NSUInteger gy = static_cast<NSUInteger>((g.M + kBM - 1) / kBM);
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, static_cast<NSUInteger>(g.batch))
                threadsPerThreadgroup:MTLSizeMake(kThreadsPerTG, 1, 1)];
        } else {
            const NSUInteger cols = g.epilogue == kAbtGeglu ? static_cast<NSUInteger>(g.N / 2)
                                                             : static_cast<NSUInteger>(g.N);
            NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(g.M) * cols, static_cast<NSUInteger>(g.batch), 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::metal_impl
