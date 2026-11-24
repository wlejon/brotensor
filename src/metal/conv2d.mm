// Metal 2D convolution:
//   conv2d_forward         — FP16 WMMA simdgroup fast path (groups == 1),
//                            pointwise 1x1 GEMM fast path, else MirKind::ConvDirect (mode 0)
//   conv2d_backward_input  — pointwise 1x1 GEMM fast path, else MirKind::ConvDirect (mode 1)
//   conv2d_backward_weight — pointwise 1x1 split-K GEMM fast path, else direct scratch + fold
//   conv2d_backward_bias   — MirKind::Gnorm (GN_BIAS_GRAD)

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#import "internal.h"
#import "conv2d_wmma.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

struct ConvWParams {
    uint32_t N, C_in, H, W;
    uint32_t C_out, kH, kW;
    uint32_t H_out, W_out;
    int32_t  stride_h, stride_w;
    int32_t  pad_h, pad_w;
    int32_t  dil_h, dil_w;
    uint32_t total;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

NSString* const kConvWeightSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct ConvWParams {
    uint N, C_in, H, W;
    uint C_out, kH, kW;
    uint H_out, W_out;
    int  stride_h, stride_w;
    int  pad_h, pad_w;
    int  dil_h, dil_w;
    uint total;
    uint Cg_in;
    uint Cg_out;
};

template <typename T>
kernel void k_conv2d_bwd_w(device const T* X           [[buffer(0)]],
                           device const T* dY          [[buffer(1)]],
                           device float*   dWt_scratch [[buffer(2)]],
                           constant ConvWParams& p     [[buffer(3)]],
                           uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint kw = idx % p.kW;
    uint t  = idx / p.kW;
    uint kh = t % p.kH;
    t /= p.kH;
    uint c_in_local = t % p.Cg_in;
    uint c_out      = t / p.Cg_in;
    uint g    = c_out / p.Cg_out;
    uint c_in = g * p.Cg_in + c_in_local;
    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        for (uint i_out = 0; i_out < p.H_out; ++i_out) {
            int in_h = int(i_out) * p.stride_h - p.pad_h + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint j_out = 0; j_out < p.W_out; ++j_out) {
                int in_w = int(j_out) * p.stride_w - p.pad_w + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                uint x_idx  = ((n * p.C_in  + c_in)  * p.H     + uint(in_h)) * p.W     + uint(in_w);
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + i_out)      * p.W_out + j_out;
                acc += float(dY[dy_idx]) * float(X[x_idx]);
            }
        }
    }
    dWt_scratch[idx] = acc;
}
template [[host_name("k_conv2d_bwd_w_fp32")]] kernel void k_conv2d_bwd_w<float>(
    device const float*, device const float*, device float*, constant ConvWParams&, uint);
template [[host_name("k_conv2d_bwd_w_fp16")]] kernel void k_conv2d_bwd_w<half>(
    device const half*, device const half*, device float*, constant ConvWParams&, uint);
template [[host_name("k_conv2d_bwd_w_bf16")]] kernel void k_conv2d_bwd_w<bfloat>(
    device const bfloat*, device const bfloat*, device float*, constant ConvWParams&, uint);

template <typename T>
kernel void k_conv2d_add_f32(device const float* src [[buffer(0)]],
                             device T*           dst [[buffer(1)]],
                             constant uint&      n   [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = T(float(dst[i]) + src[i]);
}
template [[host_name("k_conv2d_add_fp32")]] kernel void k_conv2d_add_f32<float>(
    device const float*, device float*, constant uint&, uint);
template [[host_name("k_conv2d_add_fp16")]] kernel void k_conv2d_add_f32<half>(
    device const float*, device half*, constant uint&, uint);
template [[host_name("k_conv2d_add_bf16")]] kernel void k_conv2d_add_f32<bfloat>(
    device const float*, device bfloat*, constant uint&, uint);

template <typename T>
kernel void k_conv2d_splitk_fold(device const float* part [[buffer(0)]],
                                 device T*           dst  [[buffer(1)]],
                                 constant uint&      n    [[buffer(2)]],
                                 constant uint&      S    [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float acc = 0.0f;
    for (uint s = 0; s < S; ++s) acc += part[(ulong)s * n + i];
    dst[i] = T(float(dst[i]) + acc);
}
template [[host_name("k_conv2d_splitk_fold_fp32")]] kernel void k_conv2d_splitk_fold<float>(
    device const float*, device float*, constant uint&, constant uint&, uint);
template [[host_name("k_conv2d_splitk_fold_fp16")]] kernel void k_conv2d_splitk_fold<half>(
    device const float*, device half*, constant uint&, constant uint&, uint);
template [[host_name("k_conv2d_splitk_fold_bf16")]] kernel void k_conv2d_splitk_fold<bfloat>(
    device const float*, device bfloat*, constant uint&, constant uint&, uint);
)msl";

#define DEF_PSO(NAME, FN)                                                          \
    id<MTLComputePipelineState> NAME() {                                           \
        static dispatch_once_t once;                                               \
        static id<MTLComputePipelineState> pso;                                    \
        dispatch_once(&once, ^{ pso = compile_pipeline(kConvWeightSrc, FN); });    \
        return pso;                                                                \
    }
DEF_PSO(pso_bwd_w_fp32,       @"k_conv2d_bwd_w_fp32")
DEF_PSO(pso_bwd_w_fp16,       @"k_conv2d_bwd_w_fp16")
DEF_PSO(pso_bwd_w_bf16,       @"k_conv2d_bwd_w_bf16")
DEF_PSO(pso_add_fp32,         @"k_conv2d_add_fp32")
DEF_PSO(pso_add_fp16,         @"k_conv2d_add_fp16")
DEF_PSO(pso_add_bf16,         @"k_conv2d_add_bf16")
DEF_PSO(pso_splitk_fold_fp32, @"k_conv2d_splitk_fold_fp32")
DEF_PSO(pso_splitk_fold_fp16, @"k_conv2d_splitk_fold_fp16")
DEF_PSO(pso_splitk_fold_bf16, @"k_conv2d_splitk_fold_bf16")
#undef DEF_PSO

bool conv2d_is_pointwise(int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int groups) {
    return kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 &&
           pad_h == 0 && pad_w == 0 && groups == 1;
}

}  // namespace

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
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    // Fast path 1: tiled simdgroup-matrix implicit-GEMM for FP16 SD1.5 shapes.
    if (X.dtype == Dtype::FP16 && groups == 1) {
        id<MTLBuffer> bx = buffer_for(X);
        id<MTLBuffer> bw = buffer_for(Wt);
        id<MTLBuffer> by = buffer_for(Y);
        id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
        if (conv2d_wmma_internal::launch_conv2d_implicit_gemm_simdgroup(
                bx, buffer_offset_for(X), bw, buffer_offset_for(Wt),
                bb, bias ? buffer_offset_for(*bias) : 0, bias != nullptr,
                by, buffer_offset_for(Y),
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out)) {
            return;
        }
    }

    // Fast path 2: pointwise 1x1 GEMM.
    if (!bias && conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
        const uint64_t HW = static_cast<uint64_t>(H) * W;
        metal_impl::AbtMixed g;
        g.A = buffer_for(Wt); g.ofs_A = buffer_offset_for(Wt); g.lda = static_cast<uint64_t>(C_in);
        g.B = buffer_for(X);  g.ofs_B = buffer_offset_for(X);  g.ldb = HW; g.transB = true;
        g.C = buffer_for(Y);  g.ofs_C = buffer_offset_for(Y);  g.ldc = HW;
        g.M = C_out; g.N = static_cast<int>(HW); g.K = C_in;
        g.in = g.out = metal_impl::abt_type(X.dtype);
        g.batch = N;
        g.strideA = 0;
        g.strideB = static_cast<uint64_t>(C_in) * HW;
        g.strideC = static_cast<uint64_t>(C_out) * HW;
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    mir_dispatch_conv_direct(
        0u, X.dtype, "conv2d_forward",
        X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, H_out, W_out,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
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
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
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

    mir_dispatch_conv_direct(
        1u, dY.dtype, "conv2d_backward_input",
        dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, H_out, W_out, C_in, 1, H, W,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
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

    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups) &&
        (X.dtype == Dtype::FP32 || N == 1)) {
        const int64_t HW = static_cast<int64_t>(H) * W;
        const int64_t tiles = static_cast<int64_t>((C_out + 63) / 64) * ((C_in + 63) / 64);
        int64_t S = tiles >= 256 ? 1 : (256 + tiles - 1) / tiles;
        S = std::min<int64_t>(S, std::max<int64_t>(1, HW / 1024));
        const int64_t Kc = ((HW + S - 1) / S + 31) / 32 * 32;
        S = (HW + Kc - 1) / Kc;
        const int64_t S_full = HW / Kc;
        const int64_t K_tail = HW - S_full * Kc;
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

    ConvWParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.total = total;
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    const bool is_fp16 = (X.dtype == Dtype::FP16);
    const bool is_bf16 = (X.dtype == Dtype::BF16);
    id<MTLComputePipelineState> pso = is_fp16 ? pso_bwd_w_fp16()
                                    : is_bf16 ? pso_bwd_w_bf16() : pso_bwd_w_fp32();
    id<MTLComputePipelineState> add_pso = is_fp16 ? pso_add_fp16()
                                        : is_bf16 ? pso_add_bf16() : pso_add_fp32();
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:total * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:0];
        [enc setBuffer:buffer_for(dY)  offset:buffer_offset_for(dY)  atIndex:1];
        [enc setBuffer:scratch         offset:0                      atIndex:2];
        [enc setBytes:&p length:sizeof(ConvWParams) atIndex:3];
        NSUInteger tg = std::min<NSUInteger>([pso maxTotalThreadsPerThreadgroup], 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch           offset:0                      atIndex:0];
        [enc setBuffer:buffer_for(dWt)   offset:buffer_offset_for(dWt) atIndex:1];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg2 = std::min<NSUInteger>([add_pso maxTotalThreadsPerThreadgroup], 256);
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

    const MirKernel mk = mir_kernel({MirKind::Gnorm, std::uint32_t(GN_BIAS_GRAD),
                                     dY.dtype, dY.dtype, 1});
    const std::uint32_t uc = static_cast<std::uint32_t>(C_out);
    mir_dispatch_gnorm(mk, std::min<std::uint32_t>(uc, 65535u), (uc + 65534u) / 65535u,
                       "conv2d_backward_bias",
                       dY.data, dB.data, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0u, static_cast<std::uint32_t>(H_out * W_out), 0u, 0u, 0u, 0u,
                       static_cast<std::uint32_t>(N), uc, 0.0f);
}

}  // namespace brotensor::detail::metal
