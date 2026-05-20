// Tiled FP16 matmul on Metal using simdgroup_matrix<half, 8, 8>.
//
// Computes C(M, N) = A(M, K) @ B(N, K)^T, FP16 storage with FP32 accumulator.
// Memory layout: A row-major (M, K). B row-major (N, K). C row-major (M, N).
//
// CTA (threadgroup) tile: BM=32, BN=32, BK=16. 4 simdgroups in a 2x2 grid;
// each simdgroup owns a 16x16 sub-tile of C (= 2x2 grid of 8x8 fragments).
// Threadgroup size = 128 (4 simdgroups * 32 threads/simdgroup on Apple).
//
// Fast-path safety conditions (otherwise fall back to naive kernel):
//   K >= 16 && (K % 16 == 0) && (N % 32 == 0) && M*N >= 256
// M may be arbitrary — M-edge tiles mask both load (zero-fill rows of A)
// and store (skip rows >= M). N and K must be aligned so that 8x8
// simdgroup loads/stores never straddle the tile boundary.

#include "fp16_matmul.h"

#import "internal.h"

#include <stdexcept>

namespace brotensor::metal_impl {

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------- Naive fallback (one thread per output) ----------------
kernel void k_matmul_abt_fp16_naive(device const half* A [[buffer(0)]],
                                    device const half* B [[buffer(1)]],
                                    device half*       C [[buffer(2)]],
                                    constant uint& M     [[buffer(3)]],
                                    constant uint& N     [[buffer(4)]],
                                    constant uint& K     [[buffer(5)]],
                                    uint idx [[thread_position_in_grid]]) {
    uint total = M * N;
    if (idx >= total) return;
    uint m = idx / N;
    uint n = idx % N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(A[m * K + k]) * float(B[n * K + k]);
    }
    C[idx] = half(acc);
}

// ---------------- Zero-fill (used when K == 0) ----------------
kernel void k_fp16_zero(device half* C [[buffer(0)]],
                        constant uint& total [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    C[idx] = half(0.0f);
}

// ---------------- Tiled simdgroup-matrix kernel ----------------
constant constexpr int BM = 32;
constant constexpr int BN = 32;
constant constexpr int BK = 16;
constant constexpr int WARPS_M = 2;
constant constexpr int WARPS_N = 2;
constant constexpr int THREADS_PER_TG = 128;  // WARPS_M * WARPS_N * 32
constant constexpr int WM = BM / WARPS_M;     // 16
constant constexpr int WN = BN / WARPS_N;     // 16
constant constexpr int FRAGS_M = WM / 8;      // 2
constant constexpr int FRAGS_N = WN / 8;      // 2
constant constexpr int FRAGS_K = BK / 8;      // 2

[[max_total_threads_per_threadgroup(THREADS_PER_TG)]]
kernel void k_matmul_abt_fp16_simdgroup(device const half* A [[buffer(0)]],
                                        device const half* B [[buffer(1)]],
                                        device half*       C [[buffer(2)]],
                                        constant uint& M     [[buffer(3)]],
                                        constant uint& N     [[buffer(4)]],
                                        constant uint& K     [[buffer(5)]],
                                        uint3 tid_in_tg3 [[thread_position_in_threadgroup]],
                                        uint3 tg_pos3    [[threadgroup_position_in_grid]]) {
    const uint  tid_in_tg = tid_in_tg3.x;
    const uint2 tg_pos    = uint2(tg_pos3.x, tg_pos3.y);
    const uint  sg_id     = tid_in_tg / 32u;
    threadgroup half As[BM * BK];          // (BM, BK) row-major
    threadgroup half Bs[BN * BK];          // (BN, BK) row-major
    threadgroup float Cs[BM * BN];         // staging for fp32 -> fp16 store

    const uint block_m = tg_pos.y * BM;
    const uint block_n = tg_pos.x * BN;

    const uint warp_m = sg_id / WARPS_N;   // 0 or 1
    const uint warp_n = sg_id % WARPS_N;   // 0 or 1

    // Accumulator fragments (fp32).
    simdgroup_matrix<float, 8, 8> c_frag[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            c_frag[i][j] = simdgroup_matrix<float, 8, 8>(0);
        }
    }

    // Loop over K in chunks of BK.
    for (uint k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile (BM x BK = 32 x 16 = 512 halves; 128 threads => 4/thread)
        // Layout: each thread loads 4 halves along K (one row-chunk).
        // We use one half per load with row-major linearization.
        for (int li = 0; li < 4; ++li) {
            int lin = int(tid_in_tg) + li * THREADS_PER_TG;  // 0..511
            int r = lin / BK;                                // 0..31
            int c = lin % BK;                                // 0..15
            int gr = int(block_m) + r;
            int gk = int(k0) + c;
            half v = half(0.0f);
            if (gr < int(M)) {
                v = A[uint(gr) * K + uint(gk)];
            }
            As[r * BK + c] = v;
        }

        // ---- Load B tile (BN x BK = 32 x 16) — N and K are aligned, no bounds check.
        for (int li = 0; li < 4; ++li) {
            int lin = int(tid_in_tg) + li * THREADS_PER_TG;
            int r = lin / BK;                                // row in N
            int c = lin % BK;                                // col in K
            int gr = int(block_n) + r;
            int gk = int(k0) + c;
            Bs[r * BK + c] = B[uint(gr) * K + uint(gk)];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Compute on tile ----
        // A frag (matrix_a) loaded from As, row-major stride BK.
        // B frag (matrix_b) is (K, N) view: B[k,n] = Bs[n,k]. We load via
        // simdgroup_load with transpose=true reading from &Bs[n_col][kk], so
        // the loaded 8x8 frag holds Bs[n_col:n_col+8, kk:kk+8] transposed,
        // i.e. B^T[kk:kk+8, n_col:n_col+8] = the matrix_b slice we want.
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[FRAGS_M];
            simdgroup_matrix<half, 8, 8> b_frag[FRAGS_N];

            for (int i = 0; i < FRAGS_M; ++i) {
                ulong a_row = ulong(warp_m * WM + i * 8);
                ulong a_col = ulong(kk);
                simdgroup_load(a_frag[i],
                               (const threadgroup half*)As,
                               ulong(BK),
                               ulong2(a_col, a_row),
                               false);
            }
            for (int j = 0; j < FRAGS_N; ++j) {
                // Want frag = B^T[kk:kk+8, n0:n0+8] where n0 = warp_n*WN + j*8.
                // Source in Bs is at row=n0, col=kk; load with transpose=true.
                ulong n0 = ulong(warp_n * WN + j * 8);
                simdgroup_load(b_frag[j],
                               (const threadgroup half*)Bs,
                               ulong(BK),
                               ulong2(ulong(kk), n0),
                               true);
            }

            for (int i = 0; i < FRAGS_M; ++i) {
                for (int j = 0; j < FRAGS_N; ++j) {
                    simdgroup_multiply_accumulate(c_frag[i][j],
                                                  a_frag[i],
                                                  b_frag[j],
                                                  c_frag[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Store fp32 frags to threadgroup Cs, then convert + write to C ----
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            ulong row = ulong(warp_m * WM + i * 8);
            ulong col = ulong(warp_n * WN + j * 8);
            simdgroup_store(c_frag[i][j],
                            (threadgroup float*)Cs,
                            ulong(BN),
                            ulong2(col, row),
                            false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative write Cs -> C global (BM*BN = 1024 halves, 128 thr => 8/thr).
    // N is aligned to BN, so no N-bounds check. M-edge: skip out-of-range rows.
    for (int li = 0; li < 8; ++li) {
        int lin = int(tid_in_tg) + li * THREADS_PER_TG;  // 0..1023
        int r = lin / BN;                                // 0..31
        int c = lin % BN;                                // 0..31
        int gr = int(block_m) + r;
        int gn = int(block_n) + c;
        if (gr < int(M)) {
            C[uint(gr) * N + uint(gn)] = half(Cs[r * BN + c]);
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
id<MTLComputePipelineState> pso_zero() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fp16_zero"); });
    return pso;
}

constexpr int kBM = 32;
constexpr int kBN = 32;
constexpr int kBK = 16;
constexpr int kThreadsPerTG = 128;

} // namespace

void launch_matmul_abt_fp16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K) {
    if (M == 0 || N == 0) return;

    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Ku = static_cast<uint32_t>(K);

    if (K == 0) {
        // Zero the (M,N) output.
        const uint32_t total = Mu * Nu;
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            id<MTLComputePipelineState> pso = pso_zero();
            [enc setComputePipelineState:pso];
            [enc setBuffer:C offset:ofs_C atIndex:0];
            [enc setBytes:&total length:sizeof(uint32_t) atIndex:1];
            NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    const bool tiled_ok = (static_cast<size_t>(M) * static_cast<size_t>(N) >= 256)
                       && (K >= kBK)
                       && (K % kBK == 0)
                       && (N % kBN == 0);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        if (tiled_ok) {
            id<MTLComputePipelineState> pso = pso_tiled();
            [enc setComputePipelineState:pso];
            [enc setBuffer:A offset:ofs_A atIndex:0];
            [enc setBuffer:B offset:ofs_B atIndex:1];
            [enc setBuffer:C offset:ofs_C atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            const NSUInteger grid_x = static_cast<NSUInteger>((N + kBN - 1) / kBN);
            const NSUInteger grid_y = static_cast<NSUInteger>((M + kBM - 1) / kBM);
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadsPerTG, 1, 1)];
        } else {
            id<MTLComputePipelineState> pso = pso_naive();
            [enc setComputePipelineState:pso];
            [enc setBuffer:A offset:ofs_A atIndex:0];
            [enc setBuffer:B offset:ofs_B atIndex:1];
            [enc setBuffer:C offset:ofs_C atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            const NSUInteger total = static_cast<NSUInteger>(Mu) * static_cast<NSUInteger>(Nu);
            NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::metal_impl
