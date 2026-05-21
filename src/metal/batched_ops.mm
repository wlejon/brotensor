#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// Y[b, i] = bias[i] + sum_k W[i, k] * X[b, k]
kernel void k_linear_fw_batched(device const float* W    [[buffer(0)]],
                                device const float* bias [[buffer(1)]],
                                device const float* X    [[buffer(2)]],
                                device float*       Y    [[buffer(3)]],
                                constant uint& B         [[buffer(4)]],
                                constant uint& out_dim   [[buffer(5)]],
                                constant uint& in_dim    [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x; // out
    uint b = gid.y;
    if (b >= B || i >= out_dim) return;
    device const float* xrow = X + b * in_dim;
    device const float* wrow = W + i * in_dim;
    float acc = 0.0f;
    for (uint k = 0; k < in_dim; ++k) acc += wrow[k] * xrow[k];
    Y[b * out_dim + i] = bias[i] + acc;
}

kernel void k_relu_fw_batched(device const float* x [[buffer(0)]],
                              device float*       y [[buffer(1)]],
                              constant uint& n      [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i];
    y[i] = v > 0.0f ? v : 0.0f;
}

kernel void k_tanh_fw_batched(device const float* x [[buffer(0)]],
                              device float*       y [[buffer(1)]],
                              constant uint& n      [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = tanh(x[i]);
}

kernel void k_add_inplace_batched(device float*       y [[buffer(0)]],
                                  device const float* x [[buffer(1)]],
                                  constant uint& n      [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += x[i];
}

kernel void k_relu_bw_batched(device const float* x  [[buffer(0)]],
                              device const float* dY [[buffer(1)]],
                              device float*       dX [[buffer(2)]],
                              constant uint& n       [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
}

kernel void k_tanh_bw_batched(device const float* y  [[buffer(0)]],
                              device const float* dY [[buffer(1)]],
                              device float*       dX [[buffer(2)]],
                              constant uint& n       [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float yv = y[i];
    dX[i] = dY[i] * (1.0f - yv * yv);
}

// dX[b, j] = sum_i W[i, j] * dY[b, i]
kernel void k_linear_bw_batched_dx(device const float* W   [[buffer(0)]],
                                   device const float* dY  [[buffer(1)]],
                                   device float*       dX  [[buffer(2)]],
                                   constant uint& B        [[buffer(3)]],
                                   constant uint& out_dim  [[buffer(4)]],
                                   constant uint& in_dim   [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint b = gid.y;
    if (b >= B || j >= in_dim) return;
    device const float* dY_row = dY + b * out_dim;
    float acc = 0.0f;
    for (uint i = 0; i < out_dim; ++i) acc += W[i * in_dim + j] * dY_row[i];
    dX[b * in_dim + j] = acc;
}

// dW[i, j] += sum_b dY[b, i] * X[b, j]
kernel void k_linear_bw_batched_dw(device const float* dY  [[buffer(0)]],
                                   device const float* X   [[buffer(1)]],
                                   device float*       dW  [[buffer(2)]],
                                   constant uint& B        [[buffer(3)]],
                                   constant uint& out_dim  [[buffer(4)]],
                                   constant uint& in_dim   [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) {
        acc += dY[b * out_dim + i] * X[b * in_dim + j];
    }
    dW[i * in_dim + j] += acc;
}

// dB[i] += sum_b dY[b, i]
kernel void k_linear_bw_batched_db(device const float* dY  [[buffer(0)]],
                                   device float*       dB  [[buffer(1)]],
                                   constant uint& B        [[buffer(2)]],
                                   constant uint& out_dim  [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) acc += dY[b * out_dim + i];
    dB[i] += acc;
}

// ─── FP16 variants ────────────────────────────────────────────────────────

kernel void k_linear_bw_batched_dx_fp16(device const half* W   [[buffer(0)]],
                                        device const half* dY  [[buffer(1)]],
                                        device half*       dX  [[buffer(2)]],
                                        constant uint& B       [[buffer(3)]],
                                        constant uint& out_dim [[buffer(4)]],
                                        constant uint& in_dim  [[buffer(5)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint b = gid.y;
    if (b >= B || j >= in_dim) return;
    device const half* dY_row = dY + b * out_dim;
    float acc = 0.0f;
    for (uint i = 0; i < out_dim; ++i)
        acc += float(W[i * in_dim + j]) * float(dY_row[i]);
    dX[b * in_dim + j] = half(acc);
}

// dW_scratch[i, j] = sum_b dY[b, i] * X[b, j] (FP32 scratch)
kernel void k_linear_bw_batched_dw_fp16(device const half*  dY [[buffer(0)]],
                                        device const half*  X  [[buffer(1)]],
                                        device float*       dW_scratch [[buffer(2)]],
                                        constant uint& B       [[buffer(3)]],
                                        constant uint& out_dim [[buffer(4)]],
                                        constant uint& in_dim  [[buffer(5)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) {
        acc += float(dY[b * out_dim + i]) * float(X[b * in_dim + j]);
    }
    dW_scratch[i * in_dim + j] = acc;
}

// FP32 variant of dw to scratch (parity with FP16 path).
kernel void k_linear_bw_batched_dw_fp32_to_scratch(
        device const float* dY [[buffer(0)]],
        device const float* X  [[buffer(1)]],
        device float*       dW_scratch [[buffer(2)]],
        constant uint& B       [[buffer(3)]],
        constant uint& out_dim [[buffer(4)]],
        constant uint& in_dim  [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) {
        acc += dY[b * out_dim + i] * X[b * in_dim + j];
    }
    dW_scratch[i * in_dim + j] = acc;
}

kernel void k_linear_bw_batched_db_fp16(device const half*  dY [[buffer(0)]],
                                        device float*       dB_scratch [[buffer(1)]],
                                        constant uint& B       [[buffer(2)]],
                                        constant uint& out_dim [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) acc += float(dY[b * out_dim + i]);
    dB_scratch[i] = acc;
}

kernel void k_linear_bw_batched_db_fp32_to_scratch(
        device const float* dY [[buffer(0)]],
        device float*       dB_scratch [[buffer(1)]],
        constant uint& B       [[buffer(2)]],
        constant uint& out_dim [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) acc += dY[b * out_dim + i];
    dB_scratch[i] = acc;
}

kernel void k_lbb_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                     device half*        dst [[buffer(1)]],
                                     constant uint& n        [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}
kernel void k_lbb_add_fp32_into_fp32(device const float* src [[buffer(0)]],
                                     device float*       dst [[buffer(1)]],
                                     constant uint& n        [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] += src[i];
}

// ─── BF16 variants ────────────────────────────────────────────────────────

kernel void k_linear_bw_batched_dx_bf16(device const bfloat* W   [[buffer(0)]],
                                        device const bfloat* dY  [[buffer(1)]],
                                        device bfloat*       dX  [[buffer(2)]],
                                        constant uint& B         [[buffer(3)]],
                                        constant uint& out_dim   [[buffer(4)]],
                                        constant uint& in_dim    [[buffer(5)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint b = gid.y;
    if (b >= B || j >= in_dim) return;
    device const bfloat* dY_row = dY + b * out_dim;
    float acc = 0.0f;
    for (uint i = 0; i < out_dim; ++i)
        acc += float(W[i * in_dim + j]) * float(dY_row[i]);
    dX[b * in_dim + j] = bfloat(acc);
}

// dW_scratch[i, j] = sum_b dY[b, i] * X[b, j] (FP32 scratch)
kernel void k_linear_bw_batched_dw_bf16(device const bfloat* dY [[buffer(0)]],
                                        device const bfloat* X  [[buffer(1)]],
                                        device float*       dW_scratch [[buffer(2)]],
                                        constant uint& B        [[buffer(3)]],
                                        constant uint& out_dim  [[buffer(4)]],
                                        constant uint& in_dim   [[buffer(5)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) {
        acc += float(dY[b * out_dim + i]) * float(X[b * in_dim + j]);
    }
    dW_scratch[i * in_dim + j] = acc;
}

kernel void k_linear_bw_batched_db_bf16(device const bfloat* dY [[buffer(0)]],
                                        device float*       dB_scratch [[buffer(1)]],
                                        constant uint& B        [[buffer(2)]],
                                        constant uint& out_dim  [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) acc += float(dY[b * out_dim + i]);
    dB_scratch[i] = acc;
}

kernel void k_lbb_add_fp32_into_bf16(device const float* src [[buffer(0)]],
                                     device bfloat*      dst [[buffer(1)]],
                                     constant uint& n        [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_lin_fw, @"k_linear_fw_batched")
DEF_PSO(pso_relu_fw, @"k_relu_fw_batched")
DEF_PSO(pso_tanh_fw, @"k_tanh_fw_batched")
DEF_PSO(pso_add_ip, @"k_add_inplace_batched")
DEF_PSO(pso_relu_bw, @"k_relu_bw_batched")
DEF_PSO(pso_tanh_bw, @"k_tanh_bw_batched")
DEF_PSO(pso_lin_bw_dx, @"k_linear_bw_batched_dx")
DEF_PSO(pso_lin_bw_dw, @"k_linear_bw_batched_dw")
DEF_PSO(pso_lin_bw_db, @"k_linear_bw_batched_db")
DEF_PSO(pso_lin_bw_dx_fp16,  @"k_linear_bw_batched_dx_fp16")
DEF_PSO(pso_lin_bw_dw_fp16,  @"k_linear_bw_batched_dw_fp16")
DEF_PSO(pso_lin_bw_dw_fp32_scratch, @"k_linear_bw_batched_dw_fp32_to_scratch")
DEF_PSO(pso_lin_bw_db_fp16,  @"k_linear_bw_batched_db_fp16")
DEF_PSO(pso_lin_bw_db_fp32_scratch, @"k_linear_bw_batched_db_fp32_to_scratch")
DEF_PSO(pso_lbb_add_fp16,    @"k_lbb_add_fp32_into_fp16")
DEF_PSO(pso_lbb_add_fp32,    @"k_lbb_add_fp32_into_fp32")
DEF_PSO(pso_lin_bw_dx_bf16,  @"k_linear_bw_batched_dx_bf16")
DEF_PSO(pso_lin_bw_dw_bf16,  @"k_linear_bw_batched_dw_bf16")
DEF_PSO(pso_lin_bw_db_bf16,  @"k_linear_bw_batched_db_bf16")
DEF_PSO(pso_lbb_add_bf16,    @"k_lbb_add_fp32_into_bf16")
#undef DEF_PSO

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger n,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void dispatch2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 16, h = 16;
        if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim) Y_BD.resize(B, out_dim);
    if (B == 0 || out_dim == 0) return;
    id<MTLBuffer> bw = buffer_for(W);
    NSUInteger ow = buffer_offset_for(W);
    id<MTLBuffer> bb = buffer_for(bias);
    NSUInteger ob = buffer_offset_for(bias);
    id<MTLBuffer> bx = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    id<MTLBuffer> by = buffer_for(Y_BD);
    NSUInteger oy = buffer_offset_for(Y_BD);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Ou = static_cast<uint32_t>(out_dim);
    const uint32_t Iu = static_cast<uint32_t>(in_dim);
    dispatch2d(pso_lin_fw(), out_dim, B, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bw offset:ow atIndex:0];
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:bx offset:ox atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Iu length:sizeof(uint32_t) atIndex:6];
    });
}

void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const uint32_t n = static_cast<uint32_t>(X_BD.size());
    if (n == 0) return;
    id<MTLBuffer> bx = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    id<MTLBuffer> by = buffer_for(Y_BD);
    NSUInteger oy = buffer_offset_for(Y_BD);
    dispatch1d(pso_relu_fw(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const uint32_t n = static_cast<uint32_t>(X_BD.size());
    if (n == 0) return;
    id<MTLBuffer> bx = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    id<MTLBuffer> by = buffer_for(Y_BD);
    NSUInteger oy = buffer_offset_for(Y_BD);
    dispatch1d(pso_tanh_fw(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD) {
    const uint32_t n = static_cast<uint32_t>(Y_BD.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(Y_BD);
    NSUInteger oy = buffer_offset_for(Y_BD);
    id<MTLBuffer> bx = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    dispatch1d(pso_add_ip(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD,
                           Tensor& dX_BD) {
    if (dX_BD.rows != X_BD.rows || dX_BD.cols != X_BD.cols)
        dX_BD.resize(X_BD.rows, X_BD.cols);
    const uint32_t n = static_cast<uint32_t>(X_BD.size());
    if (n == 0) return;
    id<MTLBuffer> bx  = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    id<MTLBuffer> bdy = buffer_for(dY_BD);
    NSUInteger ody = buffer_offset_for(dY_BD);
    id<MTLBuffer> bdx = buffer_for(dX_BD);
    NSUInteger odx = buffer_offset_for(dX_BD);
    dispatch1d(pso_relu_bw(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD,
                           Tensor& dX_BD) {
    if (dX_BD.rows != Y_BD.rows || dX_BD.cols != Y_BD.cols)
        dX_BD.resize(Y_BD.rows, Y_BD.cols);
    const uint32_t n = static_cast<uint32_t>(Y_BD.size());
    if (n == 0) return;
    id<MTLBuffer> by  = buffer_for(Y_BD);
    NSUInteger oy = buffer_offset_for(Y_BD);
    id<MTLBuffer> bdy = buffer_for(dY_BD);
    NSUInteger ody = buffer_offset_for(dY_BD);
    id<MTLBuffer> bdx = buffer_for(dX_BD);
    NSUInteger odx = buffer_offset_for(dX_BD);
    dispatch1d(pso_tanh_bw(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB) {
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::FP32 && W.dtype != Dtype::BF16) {
        throw std::runtime_error("linear_backward_batched: W must be FP16, BF16, or FP32");
    }
    if (X_BD.dtype != W.dtype || dY_BD.dtype != W.dtype ||
        dW.dtype != W.dtype || dB.dtype != W.dtype) {
        throw std::runtime_error("linear_backward_batched: all tensors must share dtype");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != W.dtype) {
        dX_BD.resize(B, in_dim, W.dtype);
    }
    if (B == 0) return;
    const bool is_fp16 = (W.dtype == Dtype::FP16);
    const bool is_bf16 = (W.dtype == Dtype::BF16);

    id<MTLBuffer> bw  = buffer_for(W);
    NSUInteger ow = buffer_offset_for(W);
    id<MTLBuffer> bx  = buffer_for(X_BD);
    NSUInteger ox = buffer_offset_for(X_BD);
    id<MTLBuffer> bdy = buffer_for(dY_BD);
    NSUInteger ody = buffer_offset_for(dY_BD);
    id<MTLBuffer> bdx = buffer_for(dX_BD);
    NSUInteger odx = buffer_offset_for(dX_BD);
    id<MTLBuffer> bdw = buffer_for(dW);
    NSUInteger odw = buffer_offset_for(dW);
    id<MTLBuffer> bdb = buffer_for(dB);
    NSUInteger odb = buffer_offset_for(dB);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Ou = static_cast<uint32_t>(out_dim);
    const uint32_t Iu = static_cast<uint32_t>(in_dim);

    if (in_dim > 0 && out_dim > 0) {
        id<MTLComputePipelineState> pso = is_fp16 ? pso_lin_bw_dx_fp16()
                                        : is_bf16 ? pso_lin_bw_dx_bf16()
                                        : pso_lin_bw_dx();
        dispatch2d(pso, in_dim, B, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bw offset:ow atIndex:0];
            [enc setBuffer:bdy offset:ody atIndex:1];
            [enc setBuffer:bdx offset:odx atIndex:2];
            [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Iu length:sizeof(uint32_t) atIndex:5];
        });
    }
    if (out_dim > 0 && in_dim > 0) {
        const NSUInteger dw_n = static_cast<NSUInteger>(out_dim) * in_dim;
        @autoreleasepool {
            id<MTLBuffer> scratch = [metal_impl::device()
                newBufferWithLength:dw_n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pso = is_fp16
                ? pso_lin_bw_dw_fp16()
                : is_bf16 ? pso_lin_bw_dw_bf16() : pso_lin_bw_dw_fp32_scratch();
            dispatch2d(pso, in_dim, out_dim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:bdy offset:ody atIndex:0];
                [enc setBuffer:bx offset:ox atIndex:1];
                [enc setBuffer:scratch offset:0 atIndex:2];
                [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&Iu length:sizeof(uint32_t) atIndex:5];
            });
            const uint32_t n = static_cast<uint32_t>(dw_n);
            id<MTLComputePipelineState> add_pso = is_fp16
                ? pso_lbb_add_fp16()
                : is_bf16 ? pso_lbb_add_bf16() : pso_lbb_add_fp32();
            dispatch1d(add_pso, n, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:scratch offset:0 atIndex:0];
                [enc setBuffer:bdw offset:odw atIndex:1];
                [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
            });
        }
    }
    if (out_dim > 0) {
        @autoreleasepool {
            id<MTLBuffer> scratch = [metal_impl::device()
                newBufferWithLength:out_dim * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pso = is_fp16
                ? pso_lin_bw_db_fp16()
                : is_bf16 ? pso_lin_bw_db_bf16() : pso_lin_bw_db_fp32_scratch();
            dispatch1d(pso, out_dim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:bdy offset:ody atIndex:0];
                [enc setBuffer:scratch offset:0 atIndex:1];
                [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:3];
            });
            const uint32_t n = static_cast<uint32_t>(out_dim);
            id<MTLComputePipelineState> add_pso = is_fp16
                ? pso_lbb_add_fp16()
                : is_bf16 ? pso_lbb_add_bf16() : pso_lbb_add_fp32();
            dispatch1d(add_pso, n, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:scratch offset:0 atIndex:0];
                [enc setBuffer:bdb offset:odb atIndex:1];
                [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
            });
        }
    }
}

} // namespace brotensor::detail::metal
