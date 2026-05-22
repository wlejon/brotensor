// ─── Metal log / exp / round elementwise ops (CHUNK 6, family G) ────────────
//
// Metal counterpart of src/cpu/log_exp_round.cpp. FP32-only, matching the CPU
// contract (these audio-pipeline elementwise maps are FP32 on every backend):
//   log_forward / log_backward    y = log(x);   dX = dY / x
//   exp_forward / exp_backward    y = exp(x);   dX = dY * exp(x)
//   round_forward                 y = round-half-to-even(x)  (torch.round)
//   round_backward                straight-through estimator: dX = dY
//
// log_forward / log_backward do NOT guard the x > 0 precondition — for x <= 0
// they return the IEEE result (log(0) = -inf, log(<0) = NaN; 1/x for backward),
// matching the CPU op so a mis-clamped pipeline fails loudly on both backends.
//
// All ops are plain elementwise; the output is resized + dtype-set to match the
// input; x/y and dX/dY may alias. None has a learnable parameter, so every
// backward OVERWRITES dX.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name)
                     + " must be FP32 (log/exp/round are FP32-only)");
    }
}

// Parameter block — must match the MSL struct below.
struct EwParams {
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct EwParams { uint total; };

// ── log ─────────────────────────────────────────────────────────────────────
kernel void k_log_forward(device const float* x [[buffer(0)]],
                          device float*       y [[buffer(1)]],
                          constant EwParams&  P [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    y[gid] = precise::log(x[gid]);
}

kernel void k_log_backward(device const float* x  [[buffer(0)]],
                           device const float* dY [[buffer(1)]],
                           device float*       dX [[buffer(2)]],
                           constant EwParams&  P  [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    dX[gid] = dY[gid] / x[gid];
}

// ── exp ─────────────────────────────────────────────────────────────────────
kernel void k_exp_forward(device const float* x [[buffer(0)]],
                          device float*       y [[buffer(1)]],
                          constant EwParams&  P [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    y[gid] = precise::exp(x[gid]);
}

kernel void k_exp_backward(device const float* x  [[buffer(0)]],
                           device const float* dY [[buffer(1)]],
                           device float*       dX [[buffer(2)]],
                           constant EwParams&  P  [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    dX[gid] = dY[gid] * precise::exp(x[gid]);
}

// ── round ───────────────────────────────────────────────────────────────────
// rint() rounds half-to-even under the default rounding mode — matches the
// CPU op's std::nearbyint (torch.round / numpy.round).
kernel void k_round_forward(device const float* x [[buffer(0)]],
                            device float*       y [[buffer(1)]],
                            constant EwParams&  P [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    y[gid] = rint(x[gid]);
}

// Straight-through estimator: round() has zero gradient a.e. and is
// non-differentiable at the half-integers, so dY passes straight through.
kernel void k_round_backward(device const float* dY [[buffer(0)]],
                             device float*       dX [[buffer(1)]],
                             constant EwParams&  P  [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    dX[gid] = dY[gid];
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_log_forward,   @"k_log_forward")
DEF_PSO(pso_log_backward,  @"k_log_backward")
DEF_PSO(pso_exp_forward,   @"k_exp_forward")
DEF_PSO(pso_exp_backward,  @"k_exp_backward")
DEF_PSO(pso_round_forward, @"k_round_forward")
DEF_PSO(pso_round_backward,@"k_round_backward")
#undef DEF_PSO

// Encode + submit a 1-D-grid dispatch covering `total` elements.
void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger total,
                void (^binders)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        binders(enc);
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Shared body for a unary forward (y = f(x)) — resize y to match x, dispatch.
void unary_forward(id<MTLComputePipelineState> pso, const char* op,
                   const Tensor& x, Tensor& y) {
    req_fp32(op, x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    EwParams p{static_cast<uint32_t>(n)};
    dispatch1d(pso, static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&p length:sizeof(EwParams) atIndex:2];
    });
}

// Shared body for a backward that reads x + dY (dX = f(x, dY)).
void xdy_backward(id<MTLComputePipelineState> pso, const char* op,
                  const Tensor& x, const Tensor& dY, Tensor& dX) {
    req_fp32(op, x, "x");
    req_fp32(op, dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != Dtype::FP32) {
        dX.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    EwParams p{static_cast<uint32_t>(n)};
    dispatch1d(pso, static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x)  offset:buffer_offset_for(x)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&p length:sizeof(EwParams) atIndex:3];
    });
}

} // namespace

// ─── log ─────────────────────────────────────────────────────────────────────
void log_forward(const Tensor& x, Tensor& y) {
    unary_forward(pso_log_forward(), "log_forward", x, y);
}
void log_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    xdy_backward(pso_log_backward(), "log_backward", x, dY, dX);
}

// ─── exp ─────────────────────────────────────────────────────────────────────
void exp_forward(const Tensor& x, Tensor& y) {
    unary_forward(pso_exp_forward(), "exp_forward", x, y);
}
void exp_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    xdy_backward(pso_exp_backward(), "exp_backward", x, dY, dX);
}

// ─── round ───────────────────────────────────────────────────────────────────
void round_forward(const Tensor& x, Tensor& y) {
    unary_forward(pso_round_forward(), "round_forward", x, y);
}

void round_backward(const Tensor& dY, Tensor& dX) {
    req_fp32("round_backward", dY, "dY");
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != Dtype::FP32) {
        dX.resize(dY.rows, dY.cols, Dtype::FP32);
    }
    const int n = dY.size();
    if (n == 0) return;
    EwParams p{static_cast<uint32_t>(n)};
    dispatch1d(pso_round_backward(), static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(EwParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
