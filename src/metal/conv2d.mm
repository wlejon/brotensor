#include <brotensor/runtime.h>

#include <algorithm>
#include <stdexcept>

#import "internal.h"
#import "conv2d_wmma.h"
#import "fp16_matmul.h"
#import "conv2d_msl.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

id<MTLComputePipelineState> pso_conv_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_input_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_input_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_weight_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_bias_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_input_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_input_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_weight_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_weight_fp32_scratch() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp32_to_scratch"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_bias_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_bias_fp32_scratch() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp32_to_scratch"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_add_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_add_fp32_into_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_add_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_add_fp32_into_fp32"); });
    return pso;
}
#define SPLITK_PSO(NAME, FN)                                                  \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
SPLITK_PSO(pso_splitk_fold_fp32, @"k_conv2d_splitk_fold_fp32")
SPLITK_PSO(pso_splitk_fold_fp16, @"k_conv2d_splitk_fold_fp16")
SPLITK_PSO(pso_splitk_fold_bf16, @"k_conv2d_splitk_fold_bf16")
#undef SPLITK_PSO
id<MTLComputePipelineState> pso_conv_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_input_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_input_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_weight_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_bias_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_add_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_add_fp32_into_bf16"); });
    return pso;
}

struct ConvParams {
    uint32_t N, C_in, H, W;
    uint32_t C_out, kH, kW;
    uint32_t H_out, W_out;
    int32_t  stride_h, stride_w;
    int32_t  pad_h, pad_w;
    int32_t  dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
    uint32_t groups;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

} // namespace

// ─── Pointwise (1x1, stride 1, no padding, ungrouped) GEMM fast path ────────
//
// A 1x1 conv over NCHW is a plain GEMM per image: Y[n](C_out, HW) =
// W(C_out, C_in) @ X[n](C_in, HW), and its two backwards are GEMMs too. The
// direct-conv kernels below run one thread per output with a serial K loop
// (~50-130 GFLOP/s); routing these shapes to the tiled simdgroup GEMM
// (launch_matmul_abt_mixed: FP32 accumulate, any M/N/K) is an order of
// magnitude faster. StyleGAN3-R's modulated convs are all 1x1, which is what
// made its Metal inversion (forward + backward) slow.
namespace {

bool conv2d_is_pointwise(int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int groups) {
    return kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 &&
           pad_h == 0 && pad_w == 0 && groups == 1;
}

} // namespace

void conv2d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dil_h, int dil_w,
                        int groups,
                        Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_forward: X must be FP16, BF16, or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(Wt);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx; // dummy bind if no bias
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(Wt);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    // Fast path: tiled simdgroup-matrix implicit-GEMM for FP16 SD1.5 shapes.
    // BF16 uses the naive direct-conv path (no simdgroup fast path for BF16).
    if (X.dtype == Dtype::FP16) {
        if (conv2d_wmma_internal::launch_conv2d_implicit_gemm_simdgroup(
                bx, ox, bw, ow_, bb, ob, bias != nullptr,
                by, oy,
                N, C_in, H, W,
                C_out, kH, kW,
                stride_h, stride_w,
                pad_h, pad_w,
                dil_h, dil_w,
                H_out, W_out)) {
            return;
        }
    }

    if (!bias && conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
        // Y[n](C_out, HW) = W(C_out, C_in) @ X[n], X[n] stored (C_in, HW) = (K, N).
        const uint64_t HW = static_cast<uint64_t>(H) * W;
        metal_impl::AbtMixed g;
        g.A = bw; g.ofs_A = ow_; g.lda = static_cast<uint64_t>(C_in);
        g.B = bx; g.ofs_B = ox;  g.ldb = HW; g.transB = true;
        g.C = by; g.ofs_C = oy;  g.ldc = HW;
        g.M = C_out; g.N = static_cast<int>(HW); g.K = C_in;
        g.in = g.out = metal_impl::abt_type(X.dtype);
        g.batch = N;
        g.strideA = 0;
        g.strideB = static_cast<uint64_t>(C_in) * HW;
        g.strideC = static_cast<uint64_t>(C_out) * HW;
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_conv_fp16()
                                   : (X.dtype == Dtype::BF16) ? pso_conv_bf16()
                                                               : pso_conv_fp32();

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void conv2d_backward_input(const Tensor& Wt,
                           const Tensor& dY,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w,
                           int groups,
                           Tensor& dX) {
    if (Wt.dtype != Dtype::FP16 && Wt.dtype != Dtype::FP32 && Wt.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_input: Wt must be FP16, BF16, or FP32");
    }
    if (dY.dtype != Wt.dtype) {
        throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_input: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(in_cols);
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
        // dX[n](C_in, HW) = W^T @ dY[n]: W stored (C_out, C_in) = (K, M),
        // dY[n] stored (C_out, HW) = (K, N). Overwrites dX, as the kernel does.
        const uint64_t HW = static_cast<uint64_t>(H) * W;
        metal_impl::AbtMixed g;
        g.A = buffer_for(Wt); g.ofs_A = buffer_offset_for(Wt);
        g.lda = static_cast<uint64_t>(C_in); g.transA = true;
        g.B = buffer_for(dY); g.ofs_B = buffer_offset_for(dY); g.ldb = HW; g.transB = true;
        g.C = buffer_for(dX); g.ofs_C = buffer_offset_for(dX); g.ldc = HW;
        g.M = C_in; g.N = static_cast<int>(HW); g.K = C_out;
        g.in = g.out = metal_impl::abt_type(Wt.dtype);
        g.batch = N;
        g.strideA = 0;
        g.strideB = static_cast<uint64_t>(C_out) * HW;
        g.strideC = static_cast<uint64_t>(C_in) * HW;
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    id<MTLComputePipelineState> pso = (Wt.dtype == Dtype::FP16) ? pso_conv_bwd_input_fp16()
                                   : (Wt.dtype == Dtype::BF16) ? pso_conv_bwd_input_bf16()
                                                                : pso_conv_bwd_input_fp32();
    id<MTLBuffer> bw  = buffer_for(Wt);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ow_  = buffer_offset_for(Wt);
    const NSUInteger ody  = buffer_offset_for(dY);
    const NSUInteger odx  = buffer_offset_for(dX);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bw  offset:ow_ atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void conv2d_backward_weight(const Tensor& X,
                            const Tensor& dY,
                                int N, int C_in, int H, int W,
                                int C_out, int kH, int kW,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dil_h, int dil_w,
                            int groups,
                            Tensor& dWt) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_weight: X must be FP16, BF16, or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight: X, dY, dWt dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_weight: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    const uint32_t total = static_cast<uint32_t>(C_out) * Cg_in * kH * kW;
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    // Pointwise: dW(C_out, C_in) += sum_n dY[n](C_out, HW) @ X[n](C_in, HW)^T.
    // The reduction runs over the long HW axis while the output is only
    // C_out x C_in, so the GEMM has few output tiles (e.g. 128x181 -> 6 64x64
    // threadgroups over K = 276^2) and would leave the GPU nearly idle. Split
    // K: batch the K chunks into FP32 partials (one GEMM launch, the chunks as
    // its batch axis), then fold them into dWt in one pass — a single rounding
    // per image, like the scratch kernel. With a 16-bit dWt a multi-image batch
    // would still round once per image, so that case keeps the kernel below.
    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups) &&
        (X.dtype == Dtype::FP32 || N == 1)) {
        const int64_t HW = static_cast<int64_t>(H) * W;
        const int64_t tiles = static_cast<int64_t>((C_out + 63) / 64) * ((C_in + 63) / 64);
        // Aim for >= ~256 threadgroups, but keep each chunk >= 1024 deep.
        int64_t S = tiles >= 256 ? 1 : (256 + tiles - 1) / tiles;
        S = std::min<int64_t>(S, std::max<int64_t>(1, HW / 1024));
        const int64_t Kc = ((HW + S - 1) / S + 31) / 32 * 32;   // whole BK tiles
        S = (HW + Kc - 1) / Kc;
        const int64_t S_full = HW / Kc;                         // chunks of exactly Kc
        const int64_t K_tail = HW - S_full * Kc;                // 0 or one short chunk
        const uint32_t MN = static_cast<uint32_t>(C_out) * static_cast<uint32_t>(C_in);
        const metal_impl::AbtType ty = metal_impl::abt_type(X.dtype);
        const size_t esz = dtype_size_bytes(X.dtype);
        Tensor part;
        if (S > 1) part = Tensor::empty_on(Device::Metal, static_cast<int>(S), static_cast<int>(MN), Dtype::FP32);
        for (int n = 0; n < N; ++n) {
            const NSUInteger oA = buffer_offset_for(dY) +
                static_cast<NSUInteger>(static_cast<uint64_t>(n) * C_out * HW * esz);
            const NSUInteger oB = buffer_offset_for(X) +
                static_cast<NSUInteger>(static_cast<uint64_t>(n) * C_in * HW * esz);
            metal_impl::AbtMixed g;
            g.A = buffer_for(dY); g.lda = static_cast<uint64_t>(HW);
            g.B = buffer_for(X);  g.ldb = static_cast<uint64_t>(HW);
            g.ldc = static_cast<uint64_t>(C_in);
            g.M = C_out; g.N = C_in;
            g.in = ty;
            if (S == 1) {
                g.ofs_A = oA; g.ofs_B = oB;
                g.C = buffer_for(dWt); g.ofs_C = buffer_offset_for(dWt);
                g.K = static_cast<int>(HW);
                g.out = ty;
                g.epilogue = metal_impl::kAbtAccumulate;
                metal_impl::launch_matmul_abt_mixed(g);
                continue;
            }
            g.C = buffer_for(part);
            g.out = metal_impl::kAbtF32;
            g.epilogue = metal_impl::kAbtStore;
            if (S_full > 0) {
                g.ofs_A = oA; g.ofs_B = oB; g.ofs_C = buffer_offset_for(part);
                g.K = static_cast<int>(Kc);
                g.batch = static_cast<int>(S_full);
                g.strideA = g.strideB = static_cast<uint64_t>(Kc);
                g.strideC = MN;
                metal_impl::launch_matmul_abt_mixed(g);
            }
            if (K_tail > 0) {
                const NSUInteger koff = static_cast<NSUInteger>(S_full * Kc * esz);
                g.ofs_A = oA + koff; g.ofs_B = oB + koff;
                g.ofs_C = buffer_offset_for(part) +
                          static_cast<NSUInteger>(static_cast<uint64_t>(S_full) * MN * sizeof(float));
                g.K = static_cast<int>(K_tail);
                g.batch = 1;
                g.strideA = g.strideB = g.strideC = 0;
                metal_impl::launch_matmul_abt_mixed(g);
            }
            id<MTLComputePipelineState> fold =
                X.dtype == Dtype::FP16 ? pso_splitk_fold_fp16()
              : X.dtype == Dtype::BF16 ? pso_splitk_fold_bf16() : pso_splitk_fold_fp32();
            const uint32_t Su = static_cast<uint32_t>(S);
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = new_command_buffer();
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:fold];
                [enc setBuffer:buffer_for(part) offset:buffer_offset_for(part) atIndex:0];
                [enc setBuffer:buffer_for(dWt) offset:buffer_offset_for(dWt) atIndex:1];
                [enc setBytes:&MN length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&Su length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(MN, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                ::brotensor::metal_impl::submit(cmd);
            }
        }
        return;
    }

    const bool is_fp16 = (X.dtype == Dtype::FP16);
    const bool is_bf16 = (X.dtype == Dtype::BF16);
    id<MTLComputePipelineState> pso = is_fp16 ? pso_conv_bwd_weight_fp16()
                                   : is_bf16  ? pso_conv_bwd_weight_bf16()
                                              : pso_conv_bwd_weight_fp32_scratch();
    id<MTLBuffer> bx   = buffer_for(X);
    id<MTLBuffer> bdy  = buffer_for(dY);
    id<MTLBuffer> bdwt = buffer_for(dWt);
    const NSUInteger ox   = buffer_offset_for(X);
    const NSUInteger ody  = buffer_offset_for(dY);
    const NSUInteger odwt = buffer_offset_for(dWt);

    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:total * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx   offset:ox   atIndex:0];
        [enc setBuffer:bdy  offset:ody  atIndex:1];
        [enc setBuffer:scratch offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        // Fold scratch into dWt.
        id<MTLComputePipelineState> add_pso = is_fp16 ? pso_conv_add_fp16()
                                            : is_bf16  ? pso_conv_add_bf16()
                                                       : pso_conv_add_fp32();
        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:bdwt offset:odwt atIndex:1];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg2 = [add_pso maxTotalThreadsPerThreadgroup];
        if (tg2 > 256) tg2 = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_bias: dY must be FP16, BF16, or FP32");
    }
    if (dB.dtype != dY.dtype) {
        throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = 0; p.H = 0; p.W = 0;
    p.C_out = C_out; p.kH = 0; p.kW = 0;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = 0; p.stride_w = 0;
    p.pad_h = 0; p.pad_w = 0;
    p.dil_h = 0; p.dil_w = 0;
    p.has_bias = 0u;
    p.total = 0u;

    const bool is_fp16 = (dY.dtype == Dtype::FP16);
    const bool is_bf16 = (dY.dtype == Dtype::BF16);
    id<MTLComputePipelineState> pso = is_fp16 ? pso_conv_bwd_bias_fp16()
                                   : is_bf16  ? pso_conv_bwd_bias_bf16()
                                              : pso_conv_bwd_bias_fp32_scratch();
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdb = buffer_for(dB);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odb = buffer_offset_for(dB);

    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:C_out * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        // One threadgroup per c_out; tg threads per group.
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(C_out), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        id<MTLComputePipelineState> add_pso = is_fp16 ? pso_conv_add_fp16()
                                            : is_bf16  ? pso_conv_add_bf16()
                                                       : pso_conv_add_fp32();
        const uint32_t Cn = static_cast<uint32_t>(C_out);
        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:bdb offset:odb atIndex:1];
        [enc setBytes:&Cn length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg2 = [add_pso maxTotalThreadsPerThreadgroup];
        if (tg2 > 256) tg2 = 256;
        [enc dispatchThreads:MTLSizeMake(Cn, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
