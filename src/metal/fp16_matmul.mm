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

void launch_matmul_abt_fp16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K) {
    // Single (non-batched) contiguous case, no bias, no activation — the shape
    // the attention QK^T/PV and Linear-forward callers use. Routes through the
    // batched tiled path so those callers get the simdgroup fast path.
    launch_matmul_abt_fp16_ex(A, ofs_A, B, ofs_B, C, ofs_C,
                              /*batch=*/1, M, N, K,
                              static_cast<uint64_t>(M) * static_cast<uint64_t>(K < 0 ? 0 : K),
                              static_cast<uint64_t>(N) * static_cast<uint64_t>(K < 0 ? 0 : K),
                              static_cast<uint64_t>(M) * static_cast<uint64_t>(N),
                              /*bias=*/nil, /*ofs_bias=*/0, /*has_bias=*/false,
                              /*act=*/0);
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

void launch_matmul_abt_bf16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K) {
    const uint64_t Ku = static_cast<uint64_t>(K < 0 ? 0 : K);
    launch_matmul_abt_bf16_ex(A, ofs_A, B, ofs_B, C, ofs_C,
                              /*batch=*/1, M, N, K,
                              static_cast<uint64_t>(M) * Ku,
                              static_cast<uint64_t>(N) * Ku,
                              static_cast<uint64_t>(M) * static_cast<uint64_t>(N),
                              /*bias=*/nil, /*ofs_bias=*/0, /*has_bias=*/false,
                              /*act=*/0);
}

} // namespace brotensor::metal_impl
