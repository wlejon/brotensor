// Tiled simdgroup-matrix FP16 implicit-GEMM conv2d forward (Metal mirror of
// src/cuda/conv2d_wmma.cu). FP32 accumulator. The A-tile rows are gathered
// directly from X on the fly (no materialised im2col).
//
// Tile shape (mirrors the CUDA WMMA kernel):
//   BM=64, BN=64, BK=32, 4 simdgroups arranged 2x2, 32 threads per simdgroup
//   → 128 threads per threadgroup. Each simdgroup owns a 32x32 output region
//   covered by 4x4 simdgroup_matrix<half,8,8> tiles. FP32 accumulator,
//   downcast to half + bias add in the epilogue.
//
// SD1.5 fast-path shapes (PSO specialised via MSL function constants):
//   * (kH=3, kW=3, pad=1, stride=1)
//   * (kH=1, kW=1, pad=0, stride=1)
//   * (kH=3, kW=3, pad=1, stride=2)

#include "conv2d_wmma.h"
#include "internal.h"

#include <stdexcept>
#include <string>

namespace brotensor {
namespace conv2d_wmma_internal {

using metal_impl::device;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Function constants — specialised per-PSO.
constant int KH       [[function_constant(0)]];
constant int KW       [[function_constant(1)]];
constant int PAD_H    [[function_constant(2)]];
constant int PAD_W    [[function_constant(3)]];
constant int STRIDE_H [[function_constant(4)]];
constant int STRIDE_W [[function_constant(5)]];

// Tile dimensions.
constant int BM = 64;
constant int BN = 64;
constant int BK = 32;
constant int WARPS_M = 2;
constant int WARPS_N = 2;
constant int WARPS_PER_TG = WARPS_M * WARPS_N;       // 4
constant int THREADS_PER_TG = WARPS_PER_TG * 32;     // 128
constant int WM = BM / WARPS_M;                      // 32
constant int WN = BN / WARPS_N;                      // 32
constant int FRAGS_M = WM / 8;                       // 4
constant int FRAGS_N = WN / 8;                       // 4
constant int FRAGS_K = BK / 8;                       // 4

struct ConvWmmaParams {
    int N, C_in, H, W;
    int C_out, H_out, W_out;
    uint has_bias;
};

kernel void k_conv2d_implicit_gemm_simdgroup(
        device const half* X     [[buffer(0)]],
        device const half* Wt    [[buffer(1)]],
        device const half* bias  [[buffer(2)]],
        device half*       Y     [[buffer(3)]],
        constant ConvWmmaParams& p [[buffer(4)]],
        uint3 tg_pos [[threadgroup_position_in_grid]],
        uint  tid    [[thread_index_in_threadgroup]],
        uint  sg_id  [[simdgroup_index_in_threadgroup]]) {
    const int KHW = KH * KW;
    const int HW_out  = p.H_out * p.W_out;
    const int K_total = p.C_in * KHW;

    const int block_m = int(tg_pos.y) * BM;
    const int block_n = int(tg_pos.x) * BN;

    const int warp_m = int(sg_id) / WARPS_N;
    const int warp_n = int(sg_id) % WARPS_N;

    constexpr int LDA = BK + 8;
    constexpr int LDB = BK + 8;
    constexpr int LDC = BN + 8;
    threadgroup half As[BM * LDA];
    threadgroup half Bs[BN * LDB];

    // FP32 accumulator fragments.
    simdgroup_matrix<float, 8, 8> c_frag[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            c_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < K_total; k0 += BK) {
        // ---- A tile (BM x BK) gathered from X ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BM * BK;       // 2048
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;  // 16
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;        // 0..BM-1
                const int col = lin - row * kElemsPerRow;  // 0..BK-1
                const int gk  = k0 + col;
                const int m_g = block_m + row;

                half v = half(0);
                if (m_g < p.N * HW_out && gk < K_total) {
                    const int n     = m_g / HW_out;
                    const int sp    = m_g - n * HW_out;
                    const int oh    = sp / p.W_out;
                    const int ow    = sp - oh * p.W_out;

                    const int ic    = gk / KHW;
                    const int khw   = gk - ic * KHW;
                    const int kh    = khw / KW;
                    const int kw    = khw - kh * KW;

                    const int in_h  = oh * STRIDE_H - PAD_H + kh;
                    const int in_w  = ow * STRIDE_W - PAD_W + kw;
                    if (in_h >= 0 && in_h < p.H && in_w >= 0 && in_w < p.W) {
                        v = X[((n * p.C_in + ic) * p.H + in_h) * p.W + in_w];
                    }
                }
                As[row * LDA + col] = v;
            }
        }

        // ---- B tile (BN x BK) copied from Wt[oc, k] ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BN * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;  // 16
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int oc  = block_n + row;
                half v = half(0);
                if (oc < p.C_out && gk < K_total) {
                    v = Wt[oc * K_total + gk];
                }
                Bs[row * LDB + col] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup-matrix compute ----
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            simdgroup_matrix<half, 8, 8> a_frag[FRAGS_M];
            simdgroup_matrix<half, 8, 8> b_frag[FRAGS_N];

            for (int i = 0; i < FRAGS_M; ++i) {
                const int a_row = warp_m * WM + i * 8;
                const int a_col = kk * 8;
                simdgroup_load(a_frag[i],
                               As + a_row * LDA + a_col,
                               LDA,
                               ulong2(0, 0),
                               false);
            }
            // B is stored row-major (BN=oc, BK=k). The MMA right operand wants
            // (K x N). Load with transpose=true to feed K rows of 8 (k) x 8 (oc).
            for (int j = 0; j < FRAGS_N; ++j) {
                const int b_row = warp_n * WN + j * 8;   // oc
                const int b_col = kk * 8;                 // k
                simdgroup_load(b_frag[j],
                               Bs + b_row * LDB + b_col,
                               LDB,
                               ulong2(0, 0),
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

    // Epilogue: stage FP32 fragments to threadgroup mem, then scatter to Y
    // (NCHW layout) with bias add + FP16 cast.
    threadgroup float Cs[BM * LDC];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_row = warp_m * WM + i * 8;
            const int c_col = warp_n * WN + j * 8;
            simdgroup_store(c_frag[i][j],
                            Cs + c_row * LDC + c_col,
                            LDC,
                            ulong2(0, 0),
                            false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        constexpr int kElemsPerCol = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;  // 32
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = int(tid) + si * THREADS_PER_TG;
            const int row = lin / kElemsPerCol;
            const int col = lin - row * kElemsPerCol;

            const int m_g = block_m + row;
            const int oc  = block_n + col;
            if (oc >= p.C_out) continue;
            if (m_g >= p.N * HW_out) continue;

            const int n  = m_g / HW_out;
            const int sp = m_g - n * HW_out;
            float v = Cs[row * LDC + col];
            if (p.has_bias != 0u) {
                v += float(bias[oc]);
            }
            Y[(n * p.C_out + oc) * HW_out + sp] = half(v);
        }
    }
}
)msl";

struct PsoKey {
    int kH, kW, pad_h, pad_w, stride_h, stride_w;
};

id<MTLComputePipelineState> build_pso(const PsoKey& k) {
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [device() newLibraryWithSource:kSrc
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            std::string m = "Metal: MSL compile failed for conv2d_implicit_gemm_simdgroup";
            if (err) { m += ": "; m += [[err localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
        int v;
        v = k.kH;        [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:0];
        v = k.kW;        [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:1];
        v = k.pad_h;     [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:2];
        v = k.pad_w;     [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:3];
        v = k.stride_h;  [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:4];
        v = k.stride_w;  [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:5];

        NSError* ferr = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"k_conv2d_implicit_gemm_simdgroup"
                                       constantValues:fcv
                                                error:&ferr];
        if (!fn) {
            std::string m = "Metal: function specialise failed for conv2d_implicit_gemm_simdgroup";
            if (ferr) { m += ": "; m += [[ferr localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        NSError* perr = nil;
        id<MTLComputePipelineState> pso =
            [device() newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
            std::string m = "Metal: pipeline build failed for conv2d_implicit_gemm_simdgroup";
            if (perr) { m += ": "; m += [[perr localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        return pso;
    }
}

id<MTLComputePipelineState> pso_3x3_s1() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = build_pso(PsoKey{3,3,1,1,1,1}); });
    return pso;
}
id<MTLComputePipelineState> pso_1x1_s1() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = build_pso(PsoKey{1,1,0,0,1,1}); });
    return pso;
}
id<MTLComputePipelineState> pso_3x3_s2() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = build_pso(PsoKey{3,3,1,1,2,2}); });
    return pso;
}

struct ConvWmmaParams {
    int32_t N, C_in, H, W;
    int32_t C_out, H_out, W_out;
    uint32_t has_bias;
};

} // namespace

bool launch_conv2d_implicit_gemm_simdgroup(
        id<MTLBuffer> X, NSUInteger ofs_X,
        id<MTLBuffer> Wt, NSUInteger ofs_Wt,
        id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
        id<MTLBuffer> Y, NSUInteger ofs_Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    if (dil_h != 1 || dil_w != 1) return false;
    const int M = N * H_out * W_out;
    if (M <= 0 || C_out <= 0 || C_in <= 0) return false;
    if (static_cast<size_t>(M) * static_cast<size_t>(C_out) < 1024) return false;

    id<MTLComputePipelineState> pso = nil;
    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1) {
        pso = pso_3x3_s1();
    } else if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) {
        pso = pso_1x1_s1();
    } else if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 2 && stride_w == 2) {
        pso = pso_3x3_s2();
    } else {
        return false;
    }

    ConvWmmaParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.H_out = H_out; p.W_out = W_out;
    p.has_bias = has_bias ? 1u : 0u;

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int THREADS_PER_TG = 128;

    NSUInteger grid_x = static_cast<NSUInteger>((C_out + BN - 1) / BN);
    NSUInteger grid_y = static_cast<NSUInteger>((M + BM - 1) / BM);

    // Dummy bind for bias when has_bias=false; reuse X buffer to satisfy the
    // encoder. The kernel won't read bias when has_bias=0.
    id<MTLBuffer> bb = has_bias ? bias : X;
    NSUInteger ob = has_bias ? ofs_bias : ofs_X;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X  offset:ofs_X  atIndex:0];
        [enc setBuffer:Wt offset:ofs_Wt atIndex:1];
        [enc setBuffer:bb offset:ob     atIndex:2];
        [enc setBuffer:Y  offset:ofs_Y  atIndex:3];
        [enc setBytes:&p length:sizeof(ConvWmmaParams) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
             threadsPerThreadgroup:MTLSizeMake(THREADS_PER_TG, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
    return true;
}

} // namespace conv2d_wmma_internal
} // namespace brotensor
