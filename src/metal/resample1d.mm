// ─── Metal 1D resampling ops (CHUNK 6, family E) ────────────────────────────
//
// Metal counterpart of src/cpu/resample1d.cpp. FP32-only. Arbitrary-scale
// resampling along the length axis of an NCL audio tensor.
//
// Memory layout (NCL flat — consistent with conv1d / snake):
//   element (n, c, l) at flat index (n*C + c)*L + l
//   resample1d_forward : (N, C, L_in) -> (N, C, L_out)
//
// Sampling convention — PyTorch align_corners=False:
//   src = (dst + 0.5) * (L_in / L_out) - 0.5
//   nearest : Y[dst] = X[ clamp(round_half_to_even(src), 0, L_in-1) ]
//   linear  : s = clamp(src, 0, L_in-1), x0 = floor(s),
//             x1 = min(x0+1, L_in-1), f = s - x0
//             Y[dst] = (1-f)*X[x0] + f*X[x1]
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   resample1d_forward  — Y  OVERWRITTEN  (one thread per output element).
//   resample1d_backward — dX OVERWRITTEN. The CPU op is a zero-then-scatter
//     adjoint; Metal instead runs the exact transpose as a *gather*: one thread
//     per input element loops over the L_out outputs and sums the contributions
//     that sampled it. That removes the scatter's write contention (no atomics)
//     and, by iterating dst in ascending order, sums in the same order as the
//     CPU scatter — so the FP32 result is bit-identical.

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
        fail(op, std::string(name) + " must be FP32 (resample1d is FP32-only)");
    }
}

void check_args(const char* op, int N, int C, int L_in, int L_out, int mode) {
    if (N < 0 || C < 0 || L_in < 0 || L_out < 0) {
        fail(op, "N, C, L_in, L_out must be non-negative");
    }
    if (mode != 0 && mode != 1) {
        fail(op, "mode must be 0 (nearest) or 1 (linear)");
    }
    if (L_out > 0 && L_in == 0) {
        fail(op, "L_in must be > 0 when L_out > 0");
    }
}

// Parameter block — must match the MSL struct below.
struct R1dParams {
    uint32_t N, C, L_in, L_out;
    uint32_t mode;     // 0 = nearest, 1 = linear
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct R1dParams {
    uint N, C, L_in, L_out;
    uint mode;
    uint total;
};

// ── forward: one thread per output element (n, c, dst) ──────────────────────
kernel void k_resample1d_forward(device const float* X [[buffer(0)]],
                                 device float*       Y [[buffer(1)]],
                                 constant R1dParams& P [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint dst   = gid % P.L_out;
    uint nc    = gid / P.L_out;          // n*C + c
    uint xbase = nc * P.L_in;
    float scale = float(P.L_in) / float(P.L_out);
    float src   = (float(dst) + 0.5f) * scale - 0.5f;
    if (P.mode == 0u) {
        int idx = int(rint(src));        // round half-to-even
        idx = max(0, min(idx, int(P.L_in) - 1));
        Y[gid] = X[xbase + uint(idx)];
    } else {
        float s = clamp(src, 0.0f, float(int(P.L_in) - 1));
        int x0  = int(floor(s));
        int x1  = (x0 + 1 < int(P.L_in)) ? x0 + 1 : int(P.L_in) - 1;
        float f = s - float(x0);
        Y[gid] = (1.0f - f) * X[xbase + uint(x0)] + f * X[xbase + uint(x1)];
    }
}

// ── backward: one thread per input element (n, c, xi) — gather adjoint ──────
kernel void k_resample1d_backward(device const float* dY [[buffer(0)]],
                                  device float*       dX [[buffer(1)]],
                                  constant R1dParams& P  [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint xi    = gid % P.L_in;
    uint nc    = gid / P.L_in;           // n*C + c
    uint ybase = nc * P.L_out;
    float scale = float(P.L_in) / float(P.L_out);
    float acc = 0.0f;
    for (uint dst = 0u; dst < P.L_out; ++dst) {
        float src = (float(dst) + 0.5f) * scale - 0.5f;
        float g   = dY[ybase + dst];
        if (P.mode == 0u) {
            int idx = int(rint(src));
            idx = max(0, min(idx, int(P.L_in) - 1));
            if (uint(idx) == xi) acc += g;
        } else {
            float s = clamp(src, 0.0f, float(int(P.L_in) - 1));
            int x0  = int(floor(s));
            int x1  = (x0 + 1 < int(P.L_in)) ? x0 + 1 : int(P.L_in) - 1;
            float f = s - float(x0);
            // Match the CPU scatter order: x0 contribution, then x1.
            if (uint(x0) == xi) acc += (1.0f - f) * g;
            if (uint(x1) == xi) acc += f * g;
        }
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
DEF_PSO(pso_resample1d_forward,  @"k_resample1d_forward")
DEF_PSO(pso_resample1d_backward, @"k_resample1d_backward")
#undef DEF_PSO

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

// ─── Forward ─────────────────────────────────────────────────────────────────
void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out,
                        int mode, Tensor& Y) {
    const char* op = "resample1d_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, L_in, L_out, mode);

    const int cols = C * L_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    R1dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.L_in = static_cast<uint32_t>(L_in);
    p.L_out = static_cast<uint32_t>(L_out);
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);

    dispatch1d(pso_resample1d_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(R1dParams) atIndex:2];
    });
}

// ─── Backward ────────────────────────────────────────────────────────────────
void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out,
                         int mode, Tensor& dX) {
    const char* op = "resample1d_backward";
    req_fp32(op, dY, "dY");
    check_args(op, N, C, L_in, L_out, mode);

    const int cols_in = C * L_in;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    // L_out == 0: there is no upstream gradient to gather (and dY is an empty
    // tensor with no backing buffer), so every input gradient is zero.
    if (L_out == 0) { dX.zero(); return; }

    R1dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.L_in = static_cast<uint32_t>(L_in);
    p.L_out = static_cast<uint32_t>(L_out);
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    // One thread per input element gathers the L_out outputs that sampled it.
    dispatch1d(pso_resample1d_backward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(R1dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
