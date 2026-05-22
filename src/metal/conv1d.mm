// ─── Metal 1D-convolution family (brosoundml CHUNK 3) ──────────────────────
//
// Metal counterpart of src/cpu/conv1d.cpp. FP32-only, matching the CPU
// contract (these audio ops are FP32 on every backend). Ops implemented here:
//   conv_transpose1d_forward / _backward_input / _backward_weight / _backward_bias
//   causal_conv1d_update
//   pad1d_forward / pad1d_backward
//
// ── Layout (NCL) ────────────────────────────────────────────────────────────
//   X / Y : NCL — ((n*C + c) * L + l). N batched signals folded into rows.
//   conv_transpose1d weights: OIL, input-channel-major (transposed-conv
//     convention): Wt[(c_in*Cg_out + c_out_local) * kL + kl], Cg_out = C_out/groups.
//   causal_conv1d_update weights: depthwise, one row per channel: Wt[c*kL + kl].
//
// ── Strategy ────────────────────────────────────────────────────────────────
// Each op is one GPU kernel whose threads each compute one *output* element by
// gathering every contribution — no scratch tensors, no atomics. The
// transposed-conv scatter is inverted to a gather (stride-divisibility test);
// pad1d_backward likewise gathers. The contract, accumulation semantics, and
// reflect/replicate padding all mirror src/cpu/conv1d.cpp — read that file for
// the derivations.
//
// ── Accumulation (matches the conv2d contract) ──────────────────────────────
//   *_forward / *_backward_input / pad1d_*   — output OVERWRITTEN.
//   conv_transpose1d_backward_weight / _bias — dWt / dB ACCUMULATE (+=);
//                                              caller zeros them first. One
//                                              thread owns each output, so the
//                                              += needs no atomics.

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
        fail(op, std::string(name) + " must be FP32 (conv1d ops are FP32-only)");
    }
}

void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

// L_out of a 1D transposed convolution (torch ConvTranspose1d formula).
int convt1d_out_len(int L, int stride, int padding, int output_padding,
                    int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

// Parameter blocks — must match the MSL structs below.
struct ConvT1dParams {
    uint32_t N, C_in, L, C_out, kL;
    uint32_t L_out;
    int32_t  stride, padding, dilation;
    uint32_t Cg_in, Cg_out;
    uint32_t has_bias;
    uint32_t total;
};

struct CausalParams {
    uint32_t N, C, L_step, kL;
    int32_t  dilation;
    uint32_t hist;
    uint32_t has_bias;
    uint32_t total;
};

struct Pad1dParams {
    uint32_t N, C, L, L_pad;
    int32_t  pad_left, mode;
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct ConvT1dParams {
    uint N, C_in, L, C_out, kL;
    uint L_out;
    int  stride, padding, dilation;
    uint Cg_in, Cg_out;
    uint has_bias;
    uint total;
};

struct CausalParams {
    uint N, C, L_step, kL;
    int  dilation;
    uint hist;
    uint has_bias;
    uint total;
};

struct Pad1dParams {
    uint N, C, L, L_pad;
    int  pad_left, mode;
    uint total;
};

// ── conv_transpose1d_forward: scatter inverted to a gather ───────────────────
// One thread per output element (n, oc, lo). Input (n, c_in, l) reaches output
// lo = l*stride - padding + kl*dilation, so l = (lo + padding - kl*dilation) /
// stride must be a non-negative in-range integer.
kernel void k_convt1d_forward(device const float* X      [[buffer(0)]],
                              device const float* Wt     [[buffer(1)]],
                              device const float* bias   [[buffer(2)]],
                              device float*       Y      [[buffer(3)]],
                              constant ConvT1dParams& P  [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint lo  = gid % P.L_out;
    uint tmp = gid / P.L_out;
    uint oc  = tmp % P.C_out;
    uint n   = tmp / P.C_out;
    uint g        = oc / P.Cg_out;
    uint oc_local = oc - g * P.Cg_out;
    float acc = (P.has_bias != 0u) ? bias[oc] : 0.0f;
    for (uint ci_local = 0u; ci_local < P.Cg_in; ++ci_local) {
        uint c_in = g * P.Cg_in + ci_local;
        for (uint kl = 0u; kl < P.kL; ++kl) {
            int num = int(lo) + P.padding - int(kl) * P.dilation;
            if (num < 0 || (num % P.stride) != 0) continue;
            int l = num / P.stride;
            if (l >= int(P.L)) continue;
            uint w_idx = (c_in * P.Cg_out + oc_local) * P.kL + kl;
            uint x_idx = (n * P.C_in + c_in) * P.L + uint(l);
            acc += X[x_idx] * Wt[w_idx];
        }
    }
    Y[gid] = acc;
}

// ── conv_transpose1d_backward_input: adjoint scatter is a plain gather conv ──
// One thread per input gradient element (n, c_in, l).
kernel void k_convt1d_backward_input(device const float* Wt    [[buffer(0)]],
                                     device const float* dY    [[buffer(1)]],
                                     device float*       dX    [[buffer(2)]],
                                     constant ConvT1dParams& P [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint l   = gid % P.L;
    uint tmp = gid / P.L;
    uint c_in = tmp % P.C_in;
    uint n    = tmp / P.C_in;
    uint g       = c_in / P.Cg_in;
    uint oc_base = g * P.Cg_out;
    int  lo_origin = int(l) * P.stride - P.padding;
    float acc = 0.0f;
    for (uint kl = 0u; kl < P.kL; ++kl) {
        int lo = lo_origin + int(kl) * P.dilation;
        if (lo < 0 || lo >= int(P.L_out)) continue;
        for (uint oc_local = 0u; oc_local < P.Cg_out; ++oc_local) {
            uint oc = oc_base + oc_local;
            uint w_idx  = (c_in * P.Cg_out + oc_local) * P.kL + kl;
            uint dy_idx = (n * P.C_out + oc) * P.L_out + uint(lo);
            acc += dY[dy_idx] * Wt[w_idx];
        }
    }
    dX[gid] = acc;
}

// ── conv_transpose1d_backward_weight: one thread per weight element ──────────
// gid is exactly the weight flat index (c_in*Cg_out + oc_local)*kL + kl.
// dWt ACCUMULATES (+=); caller zeroed it.
kernel void k_convt1d_backward_weight(device const float* X     [[buffer(0)]],
                                      device const float* dY    [[buffer(1)]],
                                      device float*       dWt   [[buffer(2)]],
                                      constant ConvT1dParams& P [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint kl       = gid % P.kL;
    uint tmp      = gid / P.kL;
    uint oc_local = tmp % P.Cg_out;
    uint c_in     = tmp / P.Cg_out;
    uint g  = c_in / P.Cg_in;
    uint oc = g * P.Cg_out + oc_local;
    float acc = 0.0f;
    for (uint n = 0u; n < P.N; ++n) {
        uint x_base  = (n * P.C_in  + c_in) * P.L;
        uint dy_base = (n * P.C_out + oc)   * P.L_out;
        for (uint l = 0u; l < P.L; ++l) {
            int lo = int(l) * P.stride - P.padding + int(kl) * P.dilation;
            if (lo < 0 || lo >= int(P.L_out)) continue;
            acc += X[x_base + l] * dY[dy_base + uint(lo)];
        }
    }
    dWt[gid] += acc;
}

// ── conv_transpose1d_backward_bias: one thread per output channel ───────────
// dB ACCUMULATES (+=); caller zeroed it.
kernel void k_convt1d_backward_bias(device const float* dY    [[buffer(0)]],
                                    device float*       dB    [[buffer(1)]],
                                    constant ConvT1dParams& P [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= P.C_out) return;
    float acc = 0.0f;
    for (uint n = 0u; n < P.N; ++n) {
        uint base = (n * P.C_out + gid) * P.L_out;
        for (uint lo = 0u; lo < P.L_out; ++lo) acc += dY[base + lo];
    }
    dB[gid] += acc;
}

// ── causal_conv1d_update: one thread per (n, c) channel ─────────────────────
// Each thread owns one state row + one Y row: computes all L_step outputs from
// the [state ++ new] window, then rolls the state forward in place. No
// cross-thread sharing, so the in-place state update is race-free.
kernel void k_causal_conv1d_update(device const float* X     [[buffer(0)]],
                                   device const float* Wt    [[buffer(1)]],
                                   device const float* bias  [[buffer(2)]],
                                   device float*       state [[buffer(3)]],
                                   device float*       Y     [[buffer(4)]],
                                   constant CausalParams& P  [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;          // total = N * C
    uint c = gid % P.C;
    uint s_base = gid * P.hist;
    uint x_base = gid * P.L_step;
    uint w_base = c * P.kL;
    float bv = (P.has_bias != 0u) ? bias[c] : 0.0f;
    uint buf_len = P.hist + P.L_step;
    // Output sample t convolves buf[t .. t + (kL-1)*dilation] (causal).
    for (uint t = 0u; t < P.L_step; ++t) {
        float acc = bv;
        for (uint kl = 0u; kl < P.kL; ++kl) {
            uint idx = t + kl * uint(P.dilation);
            float v = (idx < P.hist) ? state[s_base + idx]
                                     : X[x_base + (idx - P.hist)];
            acc += Wt[w_base + kl] * v;
        }
        Y[x_base + t] = acc;
    }
    // Roll the state: new state = last `hist` samples of the window. idx grows
    // with i, so state[s_base+idx] is read before it is overwritten.
    for (uint i = 0u; i < P.hist; ++i) {
        uint idx = buf_len - P.hist + i;
        state[s_base + i] = (idx < P.hist)
                                ? state[s_base + idx]
                                : X[x_base + (idx - P.hist)];
    }
}

// Map an output position p in [0, L_pad) to a source index in [0, L), or -1
// for a zero-padded position. Mirrors pad1d_src() in src/cpu/conv1d.cpp.
inline int pad1d_src(int p, int L, int pad_left, int mode) {
    int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;     // interior — straight copy
    if (mode == 0) return -1;                // zero
    if (mode == 2) return rel < 0 ? 0 : L - 1;  // replicate
    // mode 1: numpy 'reflect' — no repeated edge sample.
    if (L == 1) return 0;
    int period = 2 * (L - 1);
    int q = rel % period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

// ── pad1d_forward: one thread per output element (n, c, p) ───────────────────
kernel void k_pad1d_forward(device const float* X    [[buffer(0)]],
                            device float*       Y    [[buffer(1)]],
                            constant Pad1dParams& P  [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint p   = gid % P.L_pad;
    uint tmp = gid / P.L_pad;
    uint c   = tmp % P.C;
    uint n   = tmp / P.C;
    int src = pad1d_src(int(p), int(P.L), P.pad_left, P.mode);
    Y[gid] = (src < 0) ? 0.0f : X[(n * P.C + c) * P.L + uint(src)];
}

// ── pad1d_backward: one thread per input gradient element (n, c, l) ──────────
// Adjoint of the (possibly many-to-one, reflect/replicate) read map: gather
// every output position that sourced this input sample.
kernel void k_pad1d_backward(device const float* dY   [[buffer(0)]],
                             device float*       dX   [[buffer(1)]],
                             constant Pad1dParams& P  [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint l   = gid % P.L;
    uint tmp = gid / P.L;
    uint c   = tmp % P.C;
    uint n   = tmp / P.C;
    uint dy_base = (n * P.C + c) * P.L_pad;
    float acc = 0.0f;
    for (uint p = 0u; p < P.L_pad; ++p) {
        int src = pad1d_src(int(p), int(P.L), P.pad_left, P.mode);
        if (src == int(l)) acc += dY[dy_base + p];
    }
    dX[gid] = acc;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_convt1d_forward,         @"k_convt1d_forward")
DEF_PSO(pso_convt1d_backward_input,  @"k_convt1d_backward_input")
DEF_PSO(pso_convt1d_backward_weight, @"k_convt1d_backward_weight")
DEF_PSO(pso_convt1d_backward_bias,   @"k_convt1d_backward_bias")
DEF_PSO(pso_causal_conv1d_update,    @"k_causal_conv1d_update")
DEF_PSO(pso_pad1d_forward,           @"k_pad1d_forward")
DEF_PSO(pso_pad1d_backward,          @"k_pad1d_backward")
#undef DEF_PSO

// Encode + submit a 1-D-grid dispatch. `binders` runs against the encoder to
// bind buffers and the constant param block.
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

ConvT1dParams make_convt1d_params(int N, int C_in, int L, int C_out, int kL,
                                  int L_out, int stride, int padding,
                                  int dilation, int groups) {
    ConvT1dParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C_in     = static_cast<uint32_t>(C_in);
    p.L        = static_cast<uint32_t>(L);
    p.C_out    = static_cast<uint32_t>(C_out);
    p.kL       = static_cast<uint32_t>(kL);
    p.L_out    = static_cast<uint32_t>(L_out);
    p.stride   = stride;
    p.padding  = padding;
    p.dilation = dilation;
    p.Cg_in    = static_cast<uint32_t>(C_in / groups);
    p.Cg_out   = static_cast<uint32_t>(C_out / groups);
    p.has_bias = 0u;
    p.total    = 0u;
    return p;
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose1d_forward
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, Tensor& Y) {
    const char* op = "conv_transpose1d_forward";
    req_fp32(op, X, "X");
    req_fp32(op, Wt, "Wt");
    if (bias) req_fp32(op, *bias, "bias");
    check_groups(op, C_in, C_out, groups);
    if (kL < 1 || stride < 1 || dilation < 1 || padding < 0
        || output_padding < 0) {
        fail(op, "kL/stride/dilation must be >=1 and padding/output_padding >=0");
    }
    if (output_padding >= stride && output_padding >= dilation) {
        fail(op, "output_padding must be < stride or < dilation");
    }
    const int Cg_out = C_out / groups;
    const int L_out  = convt1d_out_len(L, stride, padding, output_padding,
                                       dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * L_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, out_cols, Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;

    ConvT1dParams p = make_convt1d_params(N, C_in, L, C_out, kL, L_out,
                                          stride, padding, dilation, groups);
    p.has_bias = bias ? 1u : 0u;
    p.total    = static_cast<uint32_t>(N) * C_out * L_out;

    id<MTLBuffer> bb = bias ? buffer_for(*bias) : buffer_for(X); // dummy bind
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : buffer_offset_for(X);
    dispatch1d(pso_convt1d_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)  offset:buffer_offset_for(X)  atIndex:0];
        [enc setBuffer:buffer_for(Wt) offset:buffer_offset_for(Wt) atIndex:1];
        [enc setBuffer:bb             offset:ob                    atIndex:2];
        [enc setBuffer:buffer_for(Y)  offset:buffer_offset_for(Y)  atIndex:3];
        [enc setBytes:&p length:sizeof(ConvT1dParams) atIndex:4];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose1d_backward_input
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, Tensor& dX) {
    const char* op = "conv_transpose1d_backward_input";
    req_fp32(op, Wt, "Wt");
    req_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    const int Cg_out = C_out / groups;
    const int L_out  = convt1d_out_len(L, stride, padding, output_padding,
                                       dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    const int in_cols = C_in * L;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, in_cols, Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    ConvT1dParams p = make_convt1d_params(N, C_in, L, C_out, kL, L_out,
                                          stride, padding, dilation, groups);
    p.total = static_cast<uint32_t>(N) * C_in * L;

    dispatch1d(pso_convt1d_backward_input(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(Wt) offset:buffer_offset_for(Wt) atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&p length:sizeof(ConvT1dParams) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose1d_backward_weight  (dWt accumulates)
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, Tensor& dWt) {
    const char* op = "conv_transpose1d_backward_weight";
    req_fp32(op, X, "X");
    req_fp32(op, dY, "dY");
    req_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    const int Cg_out = C_out / groups;
    const int L_out  = convt1d_out_len(L, stride, padding, output_padding,
                                       dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (dWt.rows != C_in || dWt.cols != Cg_out * kL) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (X.rows != N || X.cols != C_in * L) {
        fail(op, "X shape must be (N, C_in*L)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kL == 0) return;

    ConvT1dParams p = make_convt1d_params(N, C_in, L, C_out, kL, L_out,
                                          stride, padding, dilation, groups);
    p.total = static_cast<uint32_t>(C_in) * Cg_out * kL;

    dispatch1d(pso_convt1d_backward_weight(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:0];
        [enc setBuffer:buffer_for(dY)  offset:buffer_offset_for(dY)  atIndex:1];
        [enc setBuffer:buffer_for(dWt) offset:buffer_offset_for(dWt) atIndex:2];
        [enc setBytes:&p length:sizeof(ConvT1dParams) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose1d_backward_bias  (dB accumulates)
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int L_out, Tensor& dB) {
    const char* op = "conv_transpose1d_backward_bias";
    req_fp32(op, dY, "dY");
    req_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_out == 0 || N == 0 || L_out == 0) return;

    ConvT1dParams p{};
    p.N     = static_cast<uint32_t>(N);
    p.C_out = static_cast<uint32_t>(C_out);
    p.L_out = static_cast<uint32_t>(L_out);

    dispatch1d(pso_convt1d_backward_bias(), static_cast<NSUInteger>(C_out),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dB) offset:buffer_offset_for(dB) atIndex:1];
        [enc setBytes:&p length:sizeof(ConvT1dParams) atIndex:2];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  causal_conv1d_update
// ════════════════════════════════════════════════════════════════════════════
void causal_conv1d_update(const Tensor& X, const Tensor& Wt,
                          const Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          Tensor& state, Tensor& Y) {
    const char* op = "causal_conv1d_update";
    req_fp32(op, X, "X");
    req_fp32(op, Wt, "Wt");
    if (bias) req_fp32(op, *bias, "bias");
    req_fp32(op, state, "state");
    if (kL < 1 || dilation < 1 || L_step < 1 || N < 0 || C < 1) {
        fail(op, "kL/dilation/L_step/C must be >=1 and N >=0");
    }
    if (Wt.rows != C || Wt.cols != kL) {
        fail(op, "Wt shape must be (C, kL) — one depthwise filter per channel");
    }
    if (bias && (bias->rows != C || bias->cols != 1)) {
        fail(op, "bias shape must be (C, 1)");
    }
    const int hist = (kL - 1) * dilation;
    if (state.rows != N || state.cols != C * hist) {
        fail(op, "state shape must be (N, C*(kL-1)*dilation)");
    }
    if (Y.rows != N || Y.cols != C * L_step || Y.dtype != Dtype::FP32) {
        Y.resize(N, C * L_step, Dtype::FP32);
    }
    if (N == 0 || C == 0 || L_step == 0) return;

    CausalParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.L_step   = static_cast<uint32_t>(L_step);
    p.kL       = static_cast<uint32_t>(kL);
    p.dilation = dilation;
    p.hist     = static_cast<uint32_t>(hist);
    p.has_bias = bias ? 1u : 0u;
    p.total    = static_cast<uint32_t>(N) * C;

    id<MTLBuffer> bb = bias ? buffer_for(*bias) : buffer_for(X); // dummy bind
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : buffer_offset_for(X);
    dispatch1d(pso_causal_conv1d_update(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)     offset:buffer_offset_for(X)     atIndex:0];
        [enc setBuffer:buffer_for(Wt)    offset:buffer_offset_for(Wt)    atIndex:1];
        [enc setBuffer:bb                offset:ob                       atIndex:2];
        [enc setBuffer:buffer_for(state) offset:buffer_offset_for(state) atIndex:3];
        [enc setBuffer:buffer_for(Y)     offset:buffer_offset_for(Y)     atIndex:4];
        [enc setBytes:&p length:sizeof(CausalParams) atIndex:5];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  pad1d_forward
// ════════════════════════════════════════════════════════════════════════════
void pad1d_forward(const Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode, Tensor& Y) {
    const char* op = "pad1d_forward";
    req_fp32(op, X, "X");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    if (X.rows != N || X.cols != C * L) fail(op, "X shape must be (N, C*L)");
    const int L_pad = L + pad_left + pad_right;
    if (Y.rows != N || Y.cols != C * L_pad || Y.dtype != Dtype::FP32) {
        Y.resize(N, C * L_pad, Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    Pad1dParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.L        = static_cast<uint32_t>(L);
    p.L_pad    = static_cast<uint32_t>(L_pad);
    p.pad_left = pad_left;
    p.mode     = mode;
    p.total    = static_cast<uint32_t>(N) * C * L_pad;

    dispatch1d(pso_pad1d_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(Pad1dParams) atIndex:2];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  pad1d_backward
// ════════════════════════════════════════════════════════════════════════════
void pad1d_backward(const Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode, Tensor& dX) {
    const char* op = "pad1d_backward";
    req_fp32(op, dY, "dY");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    const int L_pad = L + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * L_pad) {
        fail(op, "dY shape must be (N, C*(L+pad_left+pad_right))");
    }
    if (dX.rows != N || dX.cols != C * L || dX.dtype != Dtype::FP32) {
        dX.resize(N, C * L, Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    Pad1dParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.L        = static_cast<uint32_t>(L);
    p.L_pad    = static_cast<uint32_t>(L_pad);
    p.pad_left = pad_left;
    p.mode     = mode;
    p.total    = static_cast<uint32_t>(N) * C * L;

    dispatch1d(pso_pad1d_backward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(Pad1dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
