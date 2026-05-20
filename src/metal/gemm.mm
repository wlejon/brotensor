#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::dispatch1d_sync;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// y = W * x + b   (W: out×in row-major, x: in×1, b: out×1, y: out×1)
kernel void k_linear_fw(device const float* W [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device const float* x [[buffer(2)]],
                        device float*       y [[buffer(3)]],
                        constant uint& out_dim [[buffer(4)]],
                        constant uint& in_dim  [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    device const float* wr = W + i * in_dim;
    float acc = 0.0f;
    for (uint k = 0; k < in_dim; ++k) acc += wr[k] * x[k];
    y[i] = acc + b[i];
}

// dX = W^T * dY   (W: out×in, dY: out, dX: in)
kernel void k_linear_bw_dx(device const float* W   [[buffer(0)]],
                           device const float* dY  [[buffer(1)]],
                           device float*       dX  [[buffer(2)]],
                           constant uint& out_dim  [[buffer(3)]],
                           constant uint& in_dim   [[buffer(4)]],
                           uint j [[thread_position_in_grid]]) {
    if (j >= in_dim) return;
    float acc = 0.0f;
    for (uint i = 0; i < out_dim; ++i) acc += W[i * in_dim + j] * dY[i];
    dX[j] = acc;
}
)msl";

id<MTLComputePipelineState> pso_fw() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_fw"); });
    return pso;
}
id<MTLComputePipelineState> pso_bw_dx() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_bw_dx"); });
    return pso;
}

void run1d(id<MTLComputePipelineState> pso, NSUInteger n,
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
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1) y.resize(out_dim, 1);
    if (out_dim == 0) return;

    id<MTLBuffer> bW = buffer_for(W);
    id<MTLBuffer> bB = buffer_for(b);
    id<MTLBuffer> bX = buffer_for(x);
    id<MTLBuffer> bY = buffer_for(y);
    const NSUInteger oW = buffer_offset_for(W);
    const NSUInteger oB = buffer_offset_for(b);
    const NSUInteger oX = buffer_offset_for(x);
    const NSUInteger oY = buffer_offset_for(y);
    const uint32_t out_u = static_cast<uint32_t>(out_dim);
    const uint32_t in_u  = static_cast<uint32_t>(in_dim);

    run1d(pso_fw(), out_dim, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bW offset:oW atIndex:0];
        [enc setBuffer:bB offset:oB atIndex:1];
        [enc setBuffer:bX offset:oX atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&out_u length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&in_u  length:sizeof(uint32_t) atIndex:5];
    });
}

void linear_backward(const Tensor& W, const Tensor& x,
                     const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (dX.rows != in_dim || dX.cols != 1) dX.resize(in_dim, 1);

    // dX = W^T * dY
    if (in_dim > 0 && out_dim > 0) {
        id<MTLBuffer> bW  = buffer_for(W);
        id<MTLBuffer> bdY = buffer_for(dY);
        id<MTLBuffer> bdX = buffer_for(dX);
        const NSUInteger oW = buffer_offset_for(W);
        const NSUInteger odY = buffer_offset_for(dY);
        const NSUInteger odX = buffer_offset_for(dX);
        const uint32_t out_u = static_cast<uint32_t>(out_dim);
        const uint32_t in_u  = static_cast<uint32_t>(in_dim);
        run1d(pso_bw_dx(), in_dim, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bW  offset:oW  atIndex:0];
            [enc setBuffer:bdY offset:odY atIndex:1];
            [enc setBuffer:bdX offset:odX atIndex:2];
            [enc setBytes:&out_u length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&in_u  length:sizeof(uint32_t) atIndex:4];
        });
    }

    // dW += dY * x^T  (rank-1 update, accumulated)
    if (out_dim > 0 && in_dim > 0) {
        const uint32_t out_u = static_cast<uint32_t>(out_dim);
        const uint32_t in_u  = static_cast<uint32_t>(in_dim);
        id<MTLBuffer> bdy = buffer_for(dY);
        id<MTLBuffer> bx  = buffer_for(x);
        id<MTLBuffer> bdw = buffer_for(dW);
        @autoreleasepool {
            id<MTLComputePipelineState> pso = metal_impl::pipeline(@"k_linear_backward_dw");
            id<MTLCommandBuffer> cmd = metal_impl::new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bdy offset:buffer_offset_for(dY) atIndex:0];
            [enc setBuffer:bx  offset:buffer_offset_for(x)  atIndex:1];
            [enc setBuffer:bdw offset:buffer_offset_for(dW) atIndex:2];
            [enc setBytes:&out_u length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&in_u  length:sizeof(uint32_t) atIndex:4];
            MTLSize grid = MTLSizeMake(in_u, out_u, 1);
            NSUInteger w = 16, h = 16;
            if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
            MTLSize tg   = MTLSizeMake(w, h, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    // dB += dY
    if (out_dim > 0) {
        const uint32_t out_u = static_cast<uint32_t>(out_dim);
        id<MTLBuffer> bdy = buffer_for(dY);
        id<MTLBuffer> bdb = buffer_for(dB);
        const NSUInteger off_dy = buffer_offset_for(dY);
        const NSUInteger off_db = buffer_offset_for(dB);
        dispatch1d_sync(@"k_linear_backward_db", out_u,
                        ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bdy offset:off_dy atIndex:0];
            [enc setBuffer:bdb offset:off_db atIndex:1];
            [enc setBytes:&out_u length:sizeof(uint32_t) atIndex:2];
        });
    }
}

namespace {

NSString* const kFp16LinearSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_fp16_bias_add(device half*       Y    [[buffer(0)]],
                            device const half* bias [[buffer(1)]],
                            constant uint& B        [[buffer(2)]],
                            constant uint& out_dim  [[buffer(3)]],
                            uint idx [[thread_position_in_grid]]) {
    uint total = B * out_dim;
    if (idx >= total) return;
    uint j = idx % out_dim;
    Y[idx] = half(float(Y[idx]) + float(bias[j]));
}
)msl";

id<MTLComputePipelineState> pso_bias_add() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kFp16LinearSrc, @"k_fp16_bias_add"); });
    return pso;
}

void launch_1d(id<MTLComputePipelineState> pso, NSUInteger n,
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
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD) {
    if (W.dtype != Dtype::FP16 || X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_fp16: W and X must be FP16");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_fp16: bias must be FP16");
    }
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out_dim, Dtype::FP16);
    }
    if (B == 0 || out_dim == 0) return;

    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(out_dim);
    metal_impl::launch_matmul_abt_fp16(buffer_for(X_BD), buffer_offset_for(X_BD),
                                       buffer_for(W),    buffer_offset_for(W),
                                       buffer_for(Y_BD), buffer_offset_for(Y_BD),
                                       B, out_dim, in_dim);

    if (bias && bias->size() > 0) {
        id<MTLBuffer> bY = buffer_for(Y_BD);
        id<MTLBuffer> bb = buffer_for(*bias);
        const NSUInteger oY = buffer_offset_for(Y_BD);
        const NSUInteger ob = buffer_offset_for(*bias);
        const uint32_t Bu = static_cast<uint32_t>(B);
        const uint32_t Ou = static_cast<uint32_t>(out_dim);
        launch_1d(pso_bias_add(), total, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bY offset:oY atIndex:0];
            [enc setBuffer:bb offset:ob atIndex:1];
            [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:3];
        });
    }
}

} // namespace brotensor::detail::metal
