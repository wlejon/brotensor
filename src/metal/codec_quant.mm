// ─── Metal codec quantization ops (brosoundml CHUNK 5, family D) ────────────
//
// Metal counterpart of src/cpu/codec_quant.cpp. The quantization bottlenecks of
// neural audio codecs:
//   vq_encode_forward / vq_encode_backward     — vector quantization (EnCodec /
//                                                DAC residual-VQ encode step)
//   fsq_quantize_forward / fsq_quantize_backward — finite scalar quantization
//                                                (NanoCodec)
//
// ── INT32 outputs ───────────────────────────────────────────────────────────
//   vq_encode_forward.indices           — (N, 1) INT32 codeword indices.
//   fsq_quantize_forward.packed_indices — (N, 1) INT32 mixed-radix codes.
//   fsq_quantize_forward.levels         — (D,)  INT32 per-coordinate level
//                                         counts (an *input*).
//
// ── Accumulation ────────────────────────────────────────────────────────────
//   *_forward             — all outputs OVERWRITTEN. One thread per row.
//   vq_encode_backward    — dX OVERWRITTEN (straight-through identity).
//   fsq_quantize_backward — dX OVERWRITTEN (straight-through identity).
//   Neither backward accumulates and neither produces a codebook gradient.
//
// FP32-only for the arithmetic, exactly as the CPU op.

#include <brotensor/runtime.h>

#include <cstdint>
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
        fail(op, std::string(name) + " must be FP32");
    }
}

void req_int32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::INT32) {
        fail(op, std::string(name) + " must be INT32");
    }
}

// Parameter blocks — must match the MSL structs below.
struct VqParams {
    uint32_t N, D, K;
};
struct FsqParams {
    uint32_t N, D;
};
struct CopyParams {
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct VqParams   { uint N, D, K; };
struct FsqParams  { uint N, D; };
struct CopyParams { uint total; };

// ── vq_encode_forward: one thread per input row n ───────────────────────────
// Picks the codeword k minimising ||x[n] - codebook[k]||^2; strict `<` keeps
// the lowest index on ties (matches the CPU op).
kernel void k_vq_encode_forward(device const float* x         [[buffer(0)]],
                                device const float* codebook  [[buffer(1)]],
                                device int*         indices   [[buffer(2)]],
                                device float*       quantized [[buffer(3)]],
                                constant VqParams&  P         [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N) return;
    device const float* x_row = x + (ulong)gid * P.D;
    float best_d2 = 3.4028235e38f;   // +FLT_MAX
    uint  best_k  = 0u;
    for (uint k = 0u; k < P.K; ++k) {
        device const float* c_row = codebook + (ulong)k * P.D;
        float d2 = 0.0f;
        for (uint j = 0u; j < P.D; ++j) {
            float diff = x_row[j] - c_row[j];
            d2 += diff * diff;
        }
        if (d2 < best_d2) { best_d2 = d2; best_k = k; }
    }
    indices[gid] = int(best_k);
    device const float* c_best = codebook + (ulong)best_k * P.D;
    device float*       q_row  = quantized + (ulong)gid * P.D;
    for (uint j = 0u; j < P.D; ++j) q_row[j] = c_best[j];
}

// ── fsq_quantize_forward: one thread per input row n ────────────────────────
// Each coordinate is snapped to one of L_d evenly spaced levels in [-1, 1];
// the per-coordinate level indices are packed mixed-radix (dim 0 = least
// significant digit) via Horner from the most-significant digit down.
kernel void k_fsq_quantize_forward(device const float* x       [[buffer(0)]],
                                   device const int*   levels  [[buffer(1)]],
                                   device float*       quant   [[buffer(2)]],
                                   device int*         packed  [[buffer(3)]],
                                   constant FsqParams& P       [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N) return;
    device const float* x_row = x + (ulong)gid * P.D;
    device float*       q_row = quant + (ulong)gid * P.D;
    long acc = 0;
    for (int d = int(P.D) - 1; d >= 0; --d) {
        int   L = levels[d];
        float h = float(L - 1) * 0.5f;        // half-width
        float v = clamp(x_row[d], -1.0f, 1.0f);
        // map to a level index in [0, L-1] (round half away from zero).
        float idx_f = round((v + 1.0f) * 0.5f * float(L - 1));
        int   idx   = int(idx_f);
        idx = max(0, min(idx, L - 1));
        q_row[d] = float(idx) / h - 1.0f;     // dequantise back into [-1, 1]
        acc = acc * long(L) + long(idx);
    }
    packed[gid] = int(acc);
}

// ── straight-through backward: dX = dQuantized (identity copy) ───────────────
kernel void k_ste_copy(device const float* dQ [[buffer(0)]],
                       device float*       dX [[buffer(1)]],
                       constant CopyParams& P [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    dX[gid] = dQ[gid];
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_vq_encode_forward,    @"k_vq_encode_forward")
DEF_PSO(pso_fsq_quantize_forward, @"k_fsq_quantize_forward")
DEF_PSO(pso_ste_copy,             @"k_ste_copy")
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

// Straight-through estimator shared by both backward ops.
void ste_copy(const char* op, const Tensor& dQuantized, Tensor& dX) {
    req_fp32(op, dQuantized, "dQuantized");
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols ||
        dX.dtype != Dtype::FP32) {
        dX.resize(dQuantized.rows, dQuantized.cols, Dtype::FP32);
    }
    const int total = dQuantized.size();
    if (total == 0) return;
    CopyParams p{static_cast<uint32_t>(total)};
    dispatch1d(pso_ste_copy(), static_cast<NSUInteger>(total),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dQuantized)
                offset:buffer_offset_for(dQuantized) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(CopyParams) atIndex:2];
    });
}

} // namespace

// ─── vq_encode ───────────────────────────────────────────────────────────────
void vq_encode_forward(const Tensor& x, const Tensor& codebook,
                       Tensor& indices, Tensor& quantized) {
    const char* op = "vq_encode_forward";
    req_fp32(op, x, "x");
    req_fp32(op, codebook, "codebook");
    const int N = x.rows;
    const int D = x.cols;
    const int K = codebook.rows;
    if (codebook.cols != D) {
        fail(op, "codebook must have the same column count as x");
    }
    if (K == 0 && N != 0) {
        fail(op, "codebook must have at least one codeword");
    }
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != Dtype::INT32) {
        indices.resize(N, 1, Dtype::INT32);
    }
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != Dtype::FP32) {
        quantized.resize(N, D, Dtype::FP32);
    }
    if (N == 0) return;

    VqParams p{static_cast<uint32_t>(N), static_cast<uint32_t>(D),
               static_cast<uint32_t>(K)};
    dispatch1d(pso_vq_encode_forward(), static_cast<NSUInteger>(N),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(codebook)
                offset:buffer_offset_for(codebook) atIndex:1];
        [enc setBuffer:buffer_for(indices)
                offset:buffer_offset_for(indices) atIndex:2];
        [enc setBuffer:buffer_for(quantized)
                offset:buffer_offset_for(quantized) atIndex:3];
        [enc setBytes:&p length:sizeof(VqParams) atIndex:4];
    });
}

void vq_encode_backward(const Tensor& dQuantized, Tensor& dX) {
    ste_copy("vq_encode_backward", dQuantized, dX);
}

// ─── fsq_quantize ────────────────────────────────────────────────────────────
void fsq_quantize_forward(const Tensor& x, const Tensor& levels,
                          Tensor& quantized, Tensor& packed_indices) {
    const char* op = "fsq_quantize_forward";
    req_fp32(op, x, "x");
    req_int32(op, levels, "levels");
    const int N = x.rows;
    const int D = x.cols;
    if (levels.size() != D) {
        fail(op, "levels must have D elements (one per column of x)");
    }
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != Dtype::FP32) {
        quantized.resize(N, D, Dtype::FP32);
    }
    if (packed_indices.rows != N || packed_indices.cols != 1 ||
        packed_indices.dtype != Dtype::INT32) {
        packed_indices.resize(N, 1, Dtype::INT32);
    }

    // Validate every level count >= 2. levels is a small (D-element) device
    // tensor; pull it to the host once so the op fails loudly — exactly as the
    // CPU op does — rather than silently producing inf/NaN in the kernel.
    if (D > 0) {
        Tensor levels_host = levels.to(Device::CPU);
        const int32_t* Lp =
            static_cast<const int32_t*>(levels_host.host_raw());
        for (int d = 0; d < D; ++d) {
            if (Lp[d] < 2) fail(op, "every level count must be >= 2");
        }
    }
    if (N == 0 || D == 0) return;

    FsqParams p{static_cast<uint32_t>(N), static_cast<uint32_t>(D)};
    dispatch1d(pso_fsq_quantize_forward(), static_cast<NSUInteger>(N),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(levels)
                offset:buffer_offset_for(levels) atIndex:1];
        [enc setBuffer:buffer_for(quantized)
                offset:buffer_offset_for(quantized) atIndex:2];
        [enc setBuffer:buffer_for(packed_indices)
                offset:buffer_offset_for(packed_indices) atIndex:3];
        [enc setBytes:&p length:sizeof(FsqParams) atIndex:4];
    });
}

void fsq_quantize_backward(const Tensor& dQuantized, Tensor& dX) {
    ste_copy("fsq_quantize_backward", dQuantized, dX);
}

} // namespace brotensor::detail::metal
