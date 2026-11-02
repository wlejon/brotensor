#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"
#import "elementwise_msl.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::dispatch1d_sync;
using metal_impl::new_command_buffer;

// relu/tanh/sigmoid dispatch on dtype like the silu/gelu family below —
// their wrappers live after the kActivationSrc DEF_PSOs.

// add_inplace forward decls (FP16 dispatcher lives further down with the
// other FP16-extension kernels).
namespace { void launch_fp16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_f32_f16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_fp16_scale_inplace(Tensor&, float, uint32_t);
             void launch_bf16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_bf16_scale_inplace(Tensor&, float, uint32_t); }

void add_inplace(Tensor& y, const Tensor& x) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        if (x.dtype != Dtype::FP16) {
            throw std::runtime_error("add_inplace_gpu: dtype mismatch");
        }
        launch_fp16_add_inplace(y, x, n);
        return;
    }
    if (y.dtype == Dtype::BF16) {
        if (x.dtype != Dtype::BF16) {
            throw std::runtime_error("add_inplace_gpu: dtype mismatch");
        }
        launch_bf16_add_inplace(y, x, n);
        return;
    }
    if (y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16) {
        // FP32 y += FP16 x (as on CUDA and Vulkan).
        launch_f32_f16_add_inplace(y, x, n);
        return;
    }
    if (y.dtype != Dtype::FP32 || x.dtype != Dtype::FP32) {
        throw std::runtime_error("add_inplace_gpu: dtype mismatch");
    }
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger off_y = buffer_offset_for(y);
    const NSUInteger off_x = buffer_offset_for(x);
    dispatch1d_sync(@"k_add_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBuffer:bx offset:off_x atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

// Forward decls for FP16/BF16 PSOs defined later in this TU.
namespace {
id<MTLComputePipelineState> pso_add_scalar_fp16();
id<MTLComputePipelineState> pso_add_scalar_bf16();
id<MTLComputePipelineState> pso_scale_fp16();
id<MTLComputePipelineState> pso_clamp_fp32();
id<MTLComputePipelineState> pso_clamp_fp16();
id<MTLComputePipelineState> pso_clamp_bf16();
}

namespace {
void dispatch_scalar_inplace(id<MTLComputePipelineState> pso,
                             id<MTLBuffer> by, NSUInteger off,
                             float s, uint32_t n) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:by offset:off atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}
} // namespace

void add_scalar_inplace(Tensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    if (y.dtype == Dtype::FP16) {
        dispatch_scalar_inplace(pso_add_scalar_fp16(), by, off_y, s, n);
        return;
    }
    if (y.dtype == Dtype::BF16) {
        dispatch_scalar_inplace(pso_add_scalar_bf16(), by, off_y, s, n);
        return;
    }
    dispatch1d_sync(@"k_add_scalar_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void scale_inplace(Tensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        launch_fp16_scale_inplace(y, s, n);
        return;
    }
    if (y.dtype == Dtype::BF16) {
        launch_bf16_scale_inplace(y, s, n);
        return;
    }
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    if (y.dtype == Dtype::FP16) {
        dispatch_scalar_inplace(pso_scale_fp16(), by, off_y, s, n);
        return;
    }
    dispatch1d_sync(@"k_scale_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void clamp(Tensor& y, float lo, float hi) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::BF16) ? pso_clamp_bf16()
      : (y.dtype == Dtype::FP16) ? pso_clamp_fp16()
      : pso_clamp_fp32();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&lo length:sizeof(float) atIndex:1];
        [enc setBytes:&hi length:sizeof(float) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask) {
    if (mask.rows != K || mask.cols != 1 || mask.dtype != Dtype::FP32) mask.resize(K, 1, Dtype::FP32);
    if (K <= 0) return;
    id<MTLBuffer> bx = buffer_for(x);
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger off_x = buffer_offset_for(x);
    const NSUInteger off_m = buffer_offset_for(mask);
    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Ou = static_cast<uint32_t>(offset);
    const uint32_t Su = static_cast<uint32_t>(stride);
    dispatch1d_sync(@"k_build_slot_mask", Ku, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:off_x atIndex:0];
        [enc setBuffer:bm offset:off_m atIndex:1];
        [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:4];
    });
}

namespace {

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kActivationSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_silu_fp32, @"k_silu_forward_fp32")
DEF_PSO(pso_silu_fp16, @"k_silu_forward_fp16")
DEF_PSO(pso_silu_bf16, @"k_silu_forward_bf16")
DEF_PSO(pso_gelu_fp32, @"k_gelu_forward_fp32")
DEF_PSO(pso_gelu_fp16, @"k_gelu_forward_fp16")
DEF_PSO(pso_gelu_bf16, @"k_gelu_forward_bf16")
DEF_PSO(pso_quick_gelu_fp32, @"k_quick_gelu_forward_fp32")
DEF_PSO(pso_quick_gelu_fp16, @"k_quick_gelu_forward_fp16")
DEF_PSO(pso_quick_gelu_bf16, @"k_quick_gelu_forward_bf16")
DEF_PSO(pso_silu_bwd_fp32,       @"k_silu_backward_fp32")
DEF_PSO(pso_silu_bwd_fp16,       @"k_silu_backward_fp16")
DEF_PSO(pso_silu_bwd_bf16,       @"k_silu_backward_bf16")
DEF_PSO(pso_gelu_bwd_fp32,       @"k_gelu_backward_fp32")
DEF_PSO(pso_gelu_bwd_fp16,       @"k_gelu_backward_fp16")
DEF_PSO(pso_gelu_bwd_bf16,       @"k_gelu_backward_bf16")
DEF_PSO(pso_quick_gelu_bwd_fp32, @"k_quick_gelu_backward_fp32")
DEF_PSO(pso_quick_gelu_bwd_fp16, @"k_quick_gelu_backward_fp16")
DEF_PSO(pso_quick_gelu_bwd_bf16, @"k_quick_gelu_backward_bf16")
DEF_PSO(pso_gelu_exact_fp32,     @"k_gelu_exact_forward_fp32")
DEF_PSO(pso_gelu_exact_fp16,     @"k_gelu_exact_forward_fp16")
DEF_PSO(pso_gelu_exact_bf16,     @"k_gelu_exact_forward_bf16")
DEF_PSO(pso_gelu_exact_bwd_fp32, @"k_gelu_exact_backward_fp32")
DEF_PSO(pso_gelu_exact_bwd_fp16, @"k_gelu_exact_backward_fp16")
DEF_PSO(pso_gelu_exact_bwd_bf16, @"k_gelu_exact_backward_bf16")
DEF_PSO(pso_add_scalar_fp16,  @"k_add_scalar_inplace_fp16")
DEF_PSO(pso_add_scalar_bf16,  @"k_add_scalar_inplace_bf16")
DEF_PSO(pso_scale_fp16,       @"k_scale_inplace_fp16")
DEF_PSO(pso_clamp_fp32,       @"k_clamp_fp32")
DEF_PSO(pso_clamp_fp16,       @"k_clamp_fp16")
DEF_PSO(pso_clamp_bf16,       @"k_clamp_bf16")
DEF_PSO(pso_relu_fp32,        @"k_relu_forward_fp32")
DEF_PSO(pso_relu_fp16,        @"k_relu_forward_fp16")
DEF_PSO(pso_relu_bf16,        @"k_relu_forward_bf16")
DEF_PSO(pso_tanh_fp32,        @"k_tanh_forward_fp32")
DEF_PSO(pso_tanh_fp16,        @"k_tanh_forward_fp16")
DEF_PSO(pso_tanh_bf16,        @"k_tanh_forward_bf16")
DEF_PSO(pso_sigmoid_fp32,     @"k_sigmoid_forward_fp32")
DEF_PSO(pso_sigmoid_fp16,     @"k_sigmoid_forward_fp16")
DEF_PSO(pso_sigmoid_bf16,     @"k_sigmoid_forward_bf16")
DEF_PSO(pso_relu_bwd_fp32,    @"k_relu_backward_fp32")
DEF_PSO(pso_relu_bwd_fp16,    @"k_relu_backward_fp16")
DEF_PSO(pso_relu_bwd_bf16,    @"k_relu_backward_bf16")
DEF_PSO(pso_tanh_bwd_fp32,    @"k_tanh_backward_fp32")
DEF_PSO(pso_tanh_bwd_fp16,    @"k_tanh_backward_fp16")
DEF_PSO(pso_tanh_bwd_bf16,    @"k_tanh_backward_bf16")
DEF_PSO(pso_sigmoid_bwd_fp32, @"k_sigmoid_backward_fp32")
DEF_PSO(pso_sigmoid_bwd_fp16, @"k_sigmoid_backward_fp16")
DEF_PSO(pso_sigmoid_bwd_bf16, @"k_sigmoid_backward_bf16")
#undef DEF_PSO

void launch_activation_unary(id<MTLComputePipelineState> pso,
                             const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const uint32_t n = static_cast<uint32_t>(x.size());
    if (n == 0) return;
    id<MTLBuffer> bx = buffer_for(x);
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void silu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_silu_fp16()
      : x.dtype == Dtype::BF16 ? pso_silu_bf16()
      : pso_silu_fp32(), x, y);
}
void gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_gelu_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_bf16()
      : pso_gelu_fp32(), x, y);
}
void quick_gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_quick_gelu_fp16()
      : x.dtype == Dtype::BF16 ? pso_quick_gelu_bf16()
      : pso_quick_gelu_fp32(), x, y);
}

namespace {
void launch_activation_bwd(id<MTLComputePipelineState> pso,
                           const Tensor& x, const Tensor& dY,
                           Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const uint32_t n = static_cast<uint32_t>(x.size());
    if (n == 0) return;
    id<MTLBuffer> bx  = buffer_for(x);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(x);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}
} // namespace

void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_silu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_silu_bwd_bf16()
      : pso_silu_bwd_fp32(), x, dY, dX);
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_gelu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_bwd_bf16()
      : pso_gelu_bwd_fp32(), x, dY, dX);
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_quick_gelu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_quick_gelu_bwd_bf16()
      : pso_quick_gelu_bwd_fp32(), x, dY, dX);
}
void gelu_exact_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_gelu_exact_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_exact_bf16()
      : pso_gelu_exact_fp32(), x, y);
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_gelu_exact_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_exact_bwd_bf16()
      : pso_gelu_exact_bwd_fp32(), x, dY, dX);
}
void relu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_relu_fp16()
      : x.dtype == Dtype::BF16 ? pso_relu_bf16()
      : pso_relu_fp32(), x, y);
}
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_relu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_relu_bwd_bf16()
      : pso_relu_bwd_fp32(), x, dY, dX);
}
void tanh_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_tanh_fp16()
      : x.dtype == Dtype::BF16 ? pso_tanh_bf16()
      : pso_tanh_fp32(), x, y);
}
// tanh/sigmoid backward take the forward OUTPUT y, not x (same as CUDA).
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        y.dtype == Dtype::FP16 ? pso_tanh_bwd_fp16()
      : y.dtype == Dtype::BF16 ? pso_tanh_bwd_bf16()
      : pso_tanh_bwd_fp32(), y, dY, dX);
}
void sigmoid_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_sigmoid_fp16()
      : x.dtype == Dtype::BF16 ? pso_sigmoid_bf16()
      : pso_sigmoid_fp32(), x, y);
}
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        y.dtype == Dtype::FP16 ? pso_sigmoid_bwd_fp16()
      : y.dtype == Dtype::BF16 ? pso_sigmoid_bwd_bf16()
      : pso_sigmoid_bwd_fp32(), y, dY, dX);
}

namespace {

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kFp16ExtSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_add_inplace_fp16,   @"k_add_inplace_fp16")
DEF_PSO(pso_add_inplace_bf16,   @"k_add_inplace_bf16")
DEF_PSO(pso_add_inplace_f32_f16, @"k_add_inplace_f32_f16")
DEF_PSO(pso_scale_inplace_fp16, @"k_scale_inplace_fp16")
DEF_PSO(pso_scale_inplace_bf16, @"k_scale_inplace_bf16")
DEF_PSO(pso_mul_inplace_fp32,   @"k_mul_inplace_fp32")
DEF_PSO(pso_mul_inplace_fp16,   @"k_mul_inplace_fp16")
DEF_PSO(pso_mul_inplace_bf16,   @"k_mul_inplace_bf16")
DEF_PSO(pso_geglu_fp16,         @"k_geglu_forward_fp16")
DEF_PSO(pso_geglu_fp32,         @"k_geglu_forward_fp32")
DEF_PSO(pso_geglu_bf16,         @"k_geglu_forward_bf16")
DEF_PSO(pso_geglu_bwd_fp32,     @"k_geglu_backward_fp32")
DEF_PSO(pso_geglu_bwd_fp16,     @"k_geglu_backward_fp16")
DEF_PSO(pso_geglu_bwd_bf16,     @"k_geglu_backward_bf16")
DEF_PSO(pso_geglu_exact_fp32,     @"k_geglu_exact_forward_fp32")
DEF_PSO(pso_geglu_exact_fp16,     @"k_geglu_exact_forward_fp16")
DEF_PSO(pso_geglu_exact_bf16,     @"k_geglu_exact_forward_bf16")
DEF_PSO(pso_geglu_exact_bwd_fp32, @"k_geglu_exact_backward_fp32")
DEF_PSO(pso_geglu_exact_bwd_fp16, @"k_geglu_exact_backward_fp16")
DEF_PSO(pso_geglu_exact_bwd_bf16, @"k_geglu_exact_backward_bf16")
DEF_PSO(pso_causal_mask_row,    @"k_causal_mask_row")
#undef DEF_PSO

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
        ::brotensor::metal_impl::submit(cmd);
    }
}

void launch_fp16_add_inplace(Tensor& y, const Tensor& x, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso_add_inplace_fp16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_f32_f16_add_inplace(Tensor& y, const Tensor& x, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso_add_inplace_f32_f16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_bf16_add_inplace(Tensor& y, const Tensor& x, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso_add_inplace_bf16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_fp16_scale_inplace(Tensor& y, float s, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger oy = buffer_offset_for(y);
    launch_1d(pso_scale_inplace_fp16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_bf16_scale_inplace(Tensor& y, float s, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger oy = buffer_offset_for(y);
    launch_1d(pso_scale_inplace_bf16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace

void mul_inplace(Tensor& y, const Tensor& x) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace_gpu: shape/dtype mismatch");
    }
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_mul_inplace_fp16()
      : (y.dtype == Dtype::BF16) ? pso_mul_inplace_bf16()
      : pso_mul_inplace_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void geglu_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_bf16()
      : pso_geglu_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
    });
}

void geglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx  = buffer_for(X);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(X);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_bwd_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_bwd_bf16()
      : pso_geglu_bwd_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_exact_bf16()
      : pso_geglu_exact_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
    });
}

void geglu_exact_backward(const Tensor& X, const Tensor& dY,
                          Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx  = buffer_for(X);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(X);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_bwd_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_exact_bwd_bf16()
      : pso_geglu_exact_bwd_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger om = buffer_offset_for(mask);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t qu = static_cast<uint32_t>(q);
    launch_1d(pso_causal_mask_row(), Lu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bm offset:om atIndex:0];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&qu length:sizeof(uint32_t) atIndex:2];
    });
}

// ─── cast: FP32 <-> FP16 dtype conversion ──────────────────────────────────

namespace {

id<MTLComputePipelineState> pso_cast_f2h() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_f2h"); });
    return pso;
}
id<MTLComputePipelineState> pso_cast_h2f() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_h2f"); });
    return pso;
}
id<MTLComputePipelineState> pso_cast_f2b() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_f2b"); });
    return pso;
}
id<MTLComputePipelineState> pso_cast_b2f() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_b2f"); });
    return pso;
}

} // namespace

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const uint32_t n = static_cast<uint32_t>(src.size());
    if (n == 0) return;
    id<MTLBuffer> bs = buffer_for(src);
    id<MTLBuffer> bd = buffer_for(dst);
    const NSUInteger os = buffer_offset_for(src);
    const NSUInteger od = buffer_offset_for(dst);

    // Ensure BF16 is handled properly (src.bytes() uses dtype_size_bytes).
    if (src.dtype == out_dtype) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:bs sourceOffset:os
                        toBuffer:bd destinationOffset:od
                            size:src.bytes()];
            [blit endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    id<MTLComputePipelineState> pso;
    if (src.dtype == Dtype::FP32 && out_dtype == Dtype::FP16) {
        pso = pso_cast_f2h();
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::FP32) {
        pso = pso_cast_h2f();
    } else if (src.dtype == Dtype::FP32 && out_dtype == Dtype::BF16) {
        pso = pso_cast_f2b();
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP32) {
        pso = pso_cast_b2f();
    } else {
        throw std::runtime_error(
            "cast: unsupported dtype pair (Metal supports FP32<->FP16, FP32<->BF16)");
    }
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bs offset:os atIndex:0];
        [enc setBuffer:bd offset:od atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

// ─── axpby_inplace / add_channel_bias_inplace / threshold_u8 ──────────────
//
// CUDA-only until now (see src/metal/register.mm's parity-gap comment); these
// three are plain elementwise/broadcast kernels, so they follow the same
// compile_pipeline + launch_1d pattern as the FP16-extension block above.

namespace {

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kExtra2Src, FN); }); \
        return pso; \
    }
DEF_PSO(pso_axpby_fp32, @"k_axpby_inplace_fp32")
DEF_PSO(pso_axpby_fp16, @"k_axpby_inplace_fp16")
DEF_PSO(pso_axpby_bf16, @"k_axpby_inplace_bf16")
DEF_PSO(pso_add_channel_bias_fp32, @"k_add_channel_bias_inplace_fp32")
DEF_PSO(pso_add_channel_bias_fp16, @"k_add_channel_bias_inplace_fp16")
DEF_PSO(pso_add_channel_bias_bf16, @"k_add_channel_bias_inplace_bf16")
DEF_PSO(pso_add_row_bias_fp32, @"k_add_row_bias_inplace_fp32")
DEF_PSO(pso_add_row_bias_fp16, @"k_add_row_bias_inplace_fp16")
DEF_PSO(pso_add_row_bias_bf16, @"k_add_row_bias_inplace_bf16")
DEF_PSO(pso_threshold_u8_fp32, @"k_threshold_u8_fp32")
DEF_PSO(pso_threshold_u8_fp16, @"k_threshold_u8_fp16")
#undef DEF_PSO

} // namespace

void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("axpby_inplace: shape/dtype mismatch");
    }
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_axpby_fp16()
      : (y.dtype == Dtype::BF16) ? pso_axpby_bf16()
      : pso_axpby_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&a length:sizeof(float) atIndex:2];
        [enc setBytes:&b length:sizeof(float) atIndex:3];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:4];
    });
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    const uint32_t n = static_cast<uint32_t>(C) * static_cast<uint32_t>(L);
    if (n == 0) return;
    if (y.dtype != bias.dtype) {
        throw std::runtime_error("add_channel_bias_inplace: dtype mismatch");
    }
    if (static_cast<uint32_t>(y.size()) != n) {
        throw std::runtime_error("add_channel_bias_inplace: y size != C*L");
    }
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_add_channel_bias_fp16()
      : (y.dtype == Dtype::BF16) ? pso_add_channel_bias_bf16()
      : pso_add_channel_bias_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bb = buffer_for(bias);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ob = buffer_offset_for(bias);
    const uint32_t Lu = static_cast<uint32_t>(L);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&n  length:sizeof(uint32_t) atIndex:3];
    });
}

void add_row_bias_inplace(Tensor& y, const Tensor& bias) {
    if (y.dtype != bias.dtype) {
        throw std::runtime_error("add_row_bias_inplace: dtype mismatch");
    }
    if (bias.size() != y.cols) {
        throw std::runtime_error("add_row_bias_inplace: bias size != y.cols");
    }
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    const uint32_t D = static_cast<uint32_t>(y.cols);
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_add_row_bias_fp16()
      : (y.dtype == Dtype::BF16) ? pso_add_row_bias_bf16()
      : pso_add_row_bias_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bb = buffer_for(bias);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ob = buffer_offset_for(bias);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("threshold_u8: X must be FP32 or FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const uint32_t n = static_cast<uint32_t>(X.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_threshold_u8_fp16() : pso_threshold_u8_fp32();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBytes:&t length:sizeof(float) atIndex:1];
        [enc setBuffer:by offset:oy atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

} // namespace brotensor::detail::metal
