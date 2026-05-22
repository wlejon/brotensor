// ─── Metal vocoder / codec activations (brosoundml CHUNK 4, family C) ───────
//
// Metal counterpart of src/cpu/vocoder_activations.cpp. FP32-only, matching
// the CPU contract (these audio ops are FP32 on every backend). Ops here:
//   snake_forward / snake_backward         — BigVGAN / DAC snake + snakebeta
//   elu_forward / elu_backward             — EnCodec ELU
//   leaky_relu_forward / leaky_relu_backward — HiFi-GAN leaky ReLU
//
// ── Layout ──────────────────────────────────────────────────────────────────
//   snake is per-channel over an NCL tensor: element (n, c, l) at flat index
//   (n*C + c)*L + l. alpha / beta carry one scalar per channel c, broadcast
//   across the (n, l) plane. elu / leaky_relu are plain elementwise.
//
// ── Accumulation (matches the group_norm_backward contract) ─────────────────
//   snake_forward            — Y  OVERWRITTEN.
//   snake_backward           — dX OVERWRITTEN; dAlpha / dBeta ACCUMULATE (+=).
//   elu / leaky_relu forward — y  OVERWRITTEN.
//   elu / leaky_relu backward— dX OVERWRITTEN (no learnable params).
//
// snake_backward's dX is one-thread-per-element; the dAlpha/dBeta reduction is
// one-thread-per-channel, so the accumulating += needs no atomics. Read
// src/cpu/vocoder_activations.cpp for the gradient derivations.

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
                     + " must be FP32 (vocoder activations are FP32-only)");
    }
}

// Parameter blocks — must match the MSL structs below.
struct SnakeParams {
    uint32_t N, C, L;
    uint32_t has_beta;
    uint32_t total;
};

struct ActParams {
    uint32_t total;
    float    param;   // elu alpha / leaky_relu negative_slope
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct SnakeParams {
    uint N, C, L;
    uint has_beta;
    uint total;
};

struct ActParams {
    uint  total;
    float param;
};

// Sign-preserving floor on a reciprocal denominator: keeps |d| >= 1e-9 so a
// near-zero alpha/beta degrades gracefully instead of producing NaN/Inf.
inline float guard_denom(float d) {
    const float kMin = 1e-9f;
    if (d >= 0.0f) return d < kMin ? kMin : d;
    return d > -kMin ? -kMin : d;
}

// ── snake_forward: y = x + (1/denom) * sin^2(alpha*x) ───────────────────────
// One thread per element (n, c, l); channel c indexes alpha / beta.
kernel void k_snake_forward(device const float* X     [[buffer(0)]],
                            device const float* alpha [[buffer(1)]],
                            device const float* beta  [[buffer(2)]],
                            device float*       Y     [[buffer(3)]],
                            constant SnakeParams& P   [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint c = (gid / P.L) % P.C;
    float a = alpha[c];
    float denom = guard_denom((P.has_beta != 0u) ? beta[c] : a);
    float r = 1.0f / denom;
    float x = X[gid];
    float s = precise::sin(a * x);
    Y[gid] = x + r * s * s;
}

// ── snake_backward, part 1: dX (overwrite) ──────────────────────────────────
// dy/dx = 1 + 2*a*r*sin(ax)*cos(ax).
kernel void k_snake_backward_dx(device const float* X     [[buffer(0)]],
                                device const float* alpha [[buffer(1)]],
                                device const float* beta  [[buffer(2)]],
                                device const float* dY    [[buffer(3)]],
                                device float*       dX    [[buffer(4)]],
                                constant SnakeParams& P   [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint c = (gid / P.L) % P.C;
    float a = alpha[c];
    float denom = guard_denom((P.has_beta != 0u) ? beta[c] : a);
    float r = 1.0f / denom;
    float x = X[gid];
    float s  = precise::sin(a * x);
    float co = precise::cos(a * x);
    dX[gid] = dY[gid] * (1.0f + 2.0f * a * r * s * co);
}

// ── snake_backward, part 2: dAlpha / dBeta (accumulate) ─────────────────────
// One thread per channel c reduces over the (n, l) plane.
kernel void k_snake_backward_params(device const float* X      [[buffer(0)]],
                                    device const float* alpha  [[buffer(1)]],
                                    device const float* beta   [[buffer(2)]],
                                    device const float* dY     [[buffer(3)]],
                                    device float*       dAlpha [[buffer(4)]],
                                    device float*       dBeta  [[buffer(5)]],
                                    constant SnakeParams& P    [[buffer(6)]],
                                    uint c [[thread_position_in_grid]]) {
    if (c >= P.C) return;
    float a = alpha[c];
    bool has_beta = (P.has_beta != 0u);
    float denom = guard_denom(has_beta ? beta[c] : a);
    float r = 1.0f / denom;
    float dalpha_acc = 0.0f;
    float dbeta_acc  = 0.0f;
    for (uint n = 0u; n < P.N; ++n) {
        uint base = (n * P.C + c) * P.L;
        for (uint l = 0u; l < P.L; ++l) {
            float x  = X[base + l];
            float dy = dY[base + l];
            float s  = precise::sin(a * x);
            float co = precise::cos(a * x);
            float sc = s * co;
            // dy/dalpha (frequency term) = 2*r*x*sin*cos.
            dalpha_acc += dy * (2.0f * r * x * sc);
            float recip_term = dy * (-r * r * s * s);
            if (has_beta) {
                dbeta_acc += recip_term;            // dy/dbeta = -r^2*sin^2.
            } else {
                dalpha_acc += recip_term;           // alpha drives denom too.
            }
        }
    }
    dAlpha[c] += dalpha_acc;                         // accumulate
    if (has_beta) dBeta[c] += dbeta_acc;             // accumulate
}

// ── elu ─────────────────────────────────────────────────────────────────────
kernel void k_elu_forward(device const float* x   [[buffer(0)]],
                          device float*       y   [[buffer(1)]],
                          constant ActParams& P   [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    float v = x[gid];
    y[gid] = v > 0.0f ? v : P.param * (precise::exp(v) - 1.0f);
}

kernel void k_elu_backward(device const float* x   [[buffer(0)]],
                           device const float* dY  [[buffer(1)]],
                           device float*       dX  [[buffer(2)]],
                           constant ActParams& P   [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    float v = x[gid];
    float g = v > 0.0f ? 1.0f : P.param * precise::exp(v);
    dX[gid] = dY[gid] * g;
}

// ── leaky_relu ──────────────────────────────────────────────────────────────
kernel void k_leaky_relu_forward(device const float* x  [[buffer(0)]],
                                 device float*       y  [[buffer(1)]],
                                 constant ActParams& P  [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    float v = x[gid];
    y[gid] = v > 0.0f ? v : P.param * v;
}

kernel void k_leaky_relu_backward(device const float* x  [[buffer(0)]],
                                  device const float* dY [[buffer(1)]],
                                  device float*       dX [[buffer(2)]],
                                  constant ActParams& P  [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    float g = x[gid] > 0.0f ? 1.0f : P.param;
    dX[gid] = dY[gid] * g;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_snake_forward,         @"k_snake_forward")
DEF_PSO(pso_snake_backward_dx,     @"k_snake_backward_dx")
DEF_PSO(pso_snake_backward_params, @"k_snake_backward_params")
DEF_PSO(pso_elu_forward,           @"k_elu_forward")
DEF_PSO(pso_elu_backward,          @"k_elu_backward")
DEF_PSO(pso_leaky_relu_forward,    @"k_leaky_relu_forward")
DEF_PSO(pso_leaky_relu_backward,   @"k_leaky_relu_backward")
#undef DEF_PSO

// Encode + submit a 1-D-grid dispatch.
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

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  snake_forward
// ════════════════════════════════════════════════════════════════════════════
void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y) {
    const char* op = "snake_forward";
    req_fp32(op, X, "X");
    req_fp32(op, alpha, "alpha");
    if (beta) req_fp32(op, *beta, "beta");
    if (N < 0 || C < 0 || L < 0) fail(op, "N, C, L must be non-negative");
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) fail(op, "X must be shaped (N, C*L)");
    if (alpha.size() != C) fail(op, "alpha must have C elements");
    if (beta && beta->size() != C) fail(op, "beta must have C elements");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    if (N == 0 || cols == 0) return;

    SnakeParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.L        = static_cast<uint32_t>(L);
    p.has_beta = beta ? 1u : 0u;
    p.total    = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);

    id<MTLBuffer> bb = beta ? buffer_for(*beta) : buffer_for(alpha); // dummy
    const NSUInteger ob =
        beta ? buffer_offset_for(*beta) : buffer_offset_for(alpha);
    dispatch1d(pso_snake_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)     offset:buffer_offset_for(X)     atIndex:0];
        [enc setBuffer:buffer_for(alpha) offset:buffer_offset_for(alpha) atIndex:1];
        [enc setBuffer:bb                offset:ob                       atIndex:2];
        [enc setBuffer:buffer_for(Y)     offset:buffer_offset_for(Y)     atIndex:3];
        [enc setBytes:&p length:sizeof(SnakeParams) atIndex:4];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  snake_backward  (dX overwrite; dAlpha / dBeta accumulate)
// ════════════════════════════════════════════════════════════════════════════
void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta) {
    const char* op = "snake_backward";
    req_fp32(op, X, "X");
    req_fp32(op, alpha, "alpha");
    req_fp32(op, dY, "dY");
    req_fp32(op, dAlpha, "dAlpha");
    if (beta) req_fp32(op, *beta, "beta");
    if (dBeta) req_fp32(op, *dBeta, "dBeta");
    if ((beta == nullptr) != (dBeta == nullptr)) {
        fail(op, "dBeta must be non-null exactly when beta is non-null");
    }
    if (N < 0 || C < 0 || L < 0) fail(op, "N, C, L must be non-negative");
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) fail(op, "X must be shaped (N, C*L)");
    if (dY.rows != N || dY.cols != cols) fail(op, "dY must be shaped (N, C*L)");
    if (alpha.size() != C) fail(op, "alpha must have C elements");
    if (beta && beta->size() != C) fail(op, "beta must have C elements");
    if (dAlpha.rows != C || dAlpha.cols != 1) fail(op, "dAlpha must be (C, 1)");
    if (dBeta && (dBeta->rows != C || dBeta->cols != 1)) {
        fail(op, "dBeta must be (C, 1)");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, static_cast<int>(cols), X.dtype);
    }
    if (N == 0 || cols == 0) return;

    SnakeParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.L        = static_cast<uint32_t>(L);
    p.has_beta = beta ? 1u : 0u;
    p.total    = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);

    id<MTLBuffer> bbeta = beta ? buffer_for(*beta) : buffer_for(alpha);  // dummy
    const NSUInteger obeta =
        beta ? buffer_offset_for(*beta) : buffer_offset_for(alpha);
    // dBeta is bound even when null (dummy → dAlpha); the kernel only writes it
    // when has_beta != 0.
    id<MTLBuffer> bdbeta = dBeta ? buffer_for(*dBeta) : buffer_for(dAlpha);
    const NSUInteger odbeta =
        dBeta ? buffer_offset_for(*dBeta) : buffer_offset_for(dAlpha);

    // Part 1: per-element dX.
    dispatch1d(pso_snake_backward_dx(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)     offset:buffer_offset_for(X)     atIndex:0];
        [enc setBuffer:buffer_for(alpha) offset:buffer_offset_for(alpha) atIndex:1];
        [enc setBuffer:bbeta             offset:obeta                    atIndex:2];
        [enc setBuffer:buffer_for(dY)    offset:buffer_offset_for(dY)    atIndex:3];
        [enc setBuffer:buffer_for(dX)    offset:buffer_offset_for(dX)    atIndex:4];
        [enc setBytes:&p length:sizeof(SnakeParams) atIndex:5];
    });
    // Part 2: per-channel dAlpha / dBeta reduction (submitted after part 1, so
    // it observes nothing of part 1 — independent — but ordering is harmless).
    dispatch1d(pso_snake_backward_params(), static_cast<NSUInteger>(C),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)      offset:buffer_offset_for(X)      atIndex:0];
        [enc setBuffer:buffer_for(alpha)  offset:buffer_offset_for(alpha)  atIndex:1];
        [enc setBuffer:bbeta              offset:obeta                     atIndex:2];
        [enc setBuffer:buffer_for(dY)     offset:buffer_offset_for(dY)     atIndex:3];
        [enc setBuffer:buffer_for(dAlpha) offset:buffer_offset_for(dAlpha) atIndex:4];
        [enc setBuffer:bdbeta             offset:odbeta                    atIndex:5];
        [enc setBytes:&p length:sizeof(SnakeParams) atIndex:6];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  elu
// ════════════════════════════════════════════════════════════════════════════
void elu_forward(const Tensor& x, float alpha, Tensor& y) {
    req_fp32("elu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    ActParams p{static_cast<uint32_t>(n), alpha};
    dispatch1d(pso_elu_forward(), static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&p length:sizeof(ActParams) atIndex:2];
    });
}

void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX) {
    req_fp32("elu_backward", x, "x");
    req_fp32("elu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    ActParams p{static_cast<uint32_t>(n), alpha};
    dispatch1d(pso_elu_backward(), static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x)  offset:buffer_offset_for(x)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&p length:sizeof(ActParams) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  leaky_relu
// ════════════════════════════════════════════════════════════════════════════
void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y) {
    req_fp32("leaky_relu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    ActParams p{static_cast<uint32_t>(n), negative_slope};
    dispatch1d(pso_leaky_relu_forward(), static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&p length:sizeof(ActParams) atIndex:2];
    });
}

void leaky_relu_backward(const Tensor& x, const Tensor& dY,
                         float negative_slope, Tensor& dX) {
    req_fp32("leaky_relu_backward", x, "x");
    req_fp32("leaky_relu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    ActParams p{static_cast<uint32_t>(n), negative_slope};
    dispatch1d(pso_leaky_relu_backward(), static_cast<NSUInteger>(n),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x)  offset:buffer_offset_for(x)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&p length:sizeof(ActParams) atIndex:3];
    });
}

} // namespace brotensor::detail::metal
