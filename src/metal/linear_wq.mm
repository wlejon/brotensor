// Weight-only INT8 / FP8 E4M3 on Metal: quantize_rows and
// linear_forward_batched_wq_ex. Mirrors src/cuda/linear_wq.cu's contract;
// the CPU twin (src/cpu/linear_wq.cpp) is the parity oracle.
//
// quantize_rows: one threadgroup per weight row reduces max|W| and writes the
// row's scale and codes. The codes must be the CPU's byte for byte, so the
// division is IEEE (precise::divide: the library is built with fast math),
// INT8 rounds half to even (rint), and FP8 E4M3 is the host encoder's
// bracket-and-round, ported as is.
//
// linear_forward_batched_wq_ex: decode-shaped. One simdgroup per output
// column reads its weight row once, a byte per weight, and dots it against up
// to eight input rows at a time (more rows in slices of eight), FP32
// accumulation; the row scale multiplies the dot product, then bias,
// activation and the epilogue (store, accumulate, GeGLU over the interleaved
// pair 2j / 2j + 1, SwiGLU over the stacked halves j / j + N/2), as
// linear_forward_batched_ex has them.

#include "mtl_mir_cache.h"

#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <algorithm>
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

constexpr NSUInteger kColsPerTg = 8;      // simdgroups (output columns) per GEMV threadgroup

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// ── FP8 E4M3 (OCP e4m3fn), as src/tensor.cpp's host codec ──────────────────
inline float e4m3_decode(uint b) {
    const uint s = b >> 7, e = (b >> 3) & 15u, m = b & 7u;
    float mag;
    if (e == 0u) mag = float(m) * 0.001953125f;                          // m/8 * 2^-6
    else         mag = as_type<float>(((e + 120u) << 23) | (m << 20));   // (1 + m/8) * 2^(e-7)
    return s ? -mag : mag;
}

// ── linear_forward_batched_wq_ex: one simdgroup per output column ──────────
struct WqParams {
    uint M, N, K, out_cols;
    uint fp8, has_bias;
    int  act, epi;   // epi: 0 store, 1 accumulate, 2 GeGLU, 3 SwiGLU
    uint vec;        // K % 8 == 0 and Wq / X aligned: 8 weights a lane a step
};

// A & S 7.1.26, |err| < 1.5e-7 (as gemm_fp32.mm).
inline float w_erf(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return x < 0.0f ? -y : y;
}
inline float w_gelu_erf(float v) { return 0.5f * v * (1.0f + w_erf(v * 0.70710678118654752f)); }
inline float w_act(int act, float v) {
    switch (act) {
        case 1: return max(v, 0.0f);
        case 2: {
            const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: return w_gelu_erf(v);
        case 4: return v / (1.0f + exp(-v));
        case 5: return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}
inline float w_decode(uchar b, uint fp8) { return fp8 ? e4m3_decode(uint(b)) : float(char(b)); }
// Four codes packed in a uint (little-endian: the first code lowest).
inline float4 w_decode4(uint b, uint fp8) {
    if (!fp8) return float4(as_type<char4>(b));
    return float4(e4m3_decode(b & 255u), e4m3_decode((b >> 8) & 255u), e4m3_decode((b >> 16) & 255u),
                  e4m3_decode(b >> 24));
}

constant constexpr uint ROWS = 8;   // input rows a weight row is read for at once

template <typename T>
void wq_gemv(device const uchar* Wq, device const float* scales, device const T* bias, device const T* X,
             device T* Y, constant WqParams& p, uint col, uint lane) {
    if (col >= p.out_cols) return;
    const bool gated = p.epi >= 2;
    // The weight rows this column reads: one, or the gated pair.
    const uint n0 = p.epi == 2 ? 2u * col : col;
    const uint n1 = p.epi == 2 ? 2u * col + 1u : col + p.out_cols;
    device const uchar* w0 = Wq + ulong(n0) * p.K;
    device const uchar* w1 = Wq + ulong(n1) * p.K;
    for (uint m0 = 0; m0 < p.M; m0 += ROWS) {
        const uint rows = min(ROWS, p.M - m0);
        float a0[ROWS], a1[ROWS];
        for (uint r = 0; r < ROWS; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
        // Eight weights a lane a step (one 8-byte load, the simdgroup a
        // 256-byte run of the row) and the eight inputs under them in two
        // 4-wide loads; what K leaves past the last multiple of 8 a weight a
        // lane, as before.
        const uint K8 = p.vec ? (p.K & ~7u) : 0u;
        for (uint k = lane * 8u; k < K8; k += 256u) {
            const uint2 b0 = *reinterpret_cast<device const uint2*>(w0 + k);
            const float4 w0a = w_decode4(b0.x, p.fp8), w0b = w_decode4(b0.y, p.fp8);
            float4 w1a = 0.0f, w1b = 0.0f;
            if (gated) {
                const uint2 b1 = *reinterpret_cast<device const uint2*>(w1 + k);
                w1a = w_decode4(b1.x, p.fp8);
                w1b = w_decode4(b1.y, p.fp8);
            }
            for (uint r = 0; r < rows; ++r) {
                device const vec<T, 4>* x4 = reinterpret_cast<device const vec<T, 4>*>(X + ulong(m0 + r) * p.K + k);
                const float4 xa = float4(x4[0]), xb = float4(x4[1]);
                a0[r] += dot(w0a, xa) + dot(w0b, xb);
                if (gated) a1[r] += dot(w1a, xa) + dot(w1b, xb);
            }
        }
        for (uint k = K8 + lane; k < p.K; k += 32u) {
            const float wv0 = w_decode(w0[k], p.fp8);
            const float wv1 = gated ? w_decode(w1[k], p.fp8) : 0.0f;
            for (uint r = 0; r < rows; ++r) {
                const float x = float(X[ulong(m0 + r) * p.K + k]);
                a0[r] += wv0 * x;
                if (gated) a1[r] += wv1 * x;
            }
        }
        for (uint r = 0; r < rows; ++r) {
            const float d0 = simd_sum(a0[r]);
            const float d1 = gated ? simd_sum(a1[r]) : 0.0f;
            if (lane != 0) continue;
            float v0 = d0 * scales[n0];
            if (p.has_bias) v0 += float(bias[n0]);
            v0 = w_act(p.act, v0);
            device T* out = Y + ulong(m0 + r) * p.out_cols + col;
            if (gated) {
                float v1 = d1 * scales[n1];
                if (p.has_bias) v1 += float(bias[n1]);
                v1 = w_act(p.act, v1);
                *out = T(p.epi == 2 ? v0 * w_gelu_erf(v1) : (v0 / (1.0f + exp(-v0))) * v1);
            } else if (p.epi == 1) {
                *out = T(float(*out) + v0);
            } else {
                *out = T(v0);
            }
        }
    }
}

#define WQ_KERNEL(NAME, T)                                                      \
kernel void NAME(device const uchar* Wq [[buffer(0)]], device const float* scales [[buffer(1)]], \
                 device const T* bias [[buffer(2)]], device const T* X [[buffer(3)]],          \
                 device T* Y [[buffer(4)]], constant WqParams& p [[buffer(5)]],               \
                 uint tg [[threadgroup_position_in_grid]],                       \
                 uint sg [[simdgroup_index_in_threadgroup]],                     \
                 uint lane [[thread_index_in_simdgroup]],                        \
                 uint sgs [[simdgroups_per_threadgroup]]) {                      \
    wq_gemv<T>(Wq, scales, bias, X, Y, p, tg * sgs + sg, lane);                  \
}
WQ_KERNEL(k_linear_wq_fp32, float)
WQ_KERNEL(k_linear_wq_fp16, half)
WQ_KERNEL(k_linear_wq_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                            \
    }
DEF_PSO(pso_wq_fp32, @"k_linear_wq_fp32")
DEF_PSO(pso_wq_fp16, @"k_linear_wq_fp16")
DEF_PSO(pso_wq_bf16, @"k_linear_wq_bf16")
#undef DEF_PSO

// Mirrors the MSL struct.
struct WqParams {
    uint32_t M, N, K, out_cols;
    uint32_t fp8, has_bias;
    int32_t act, epi;
    uint32_t vec;
};

}  // namespace

void quantize_rows(const Tensor& W, Dtype fmt, Tensor& Wq, Tensor& scales) {
    constexpr const char* op = "quantize_rows";
    if (fmt != Dtype::INT8 && fmt != Dtype::FP8_E4M3) fail(op, "fmt must be INT8 or FP8_E4M3");
    if (W.dtype != Dtype::FP32 && W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16)
        fail(op, "W must be FP32, FP16 or BF16");
    const int N = W.rows, K = W.cols;
    if (Wq.rows != N || Wq.cols != K || Wq.dtype != fmt) Wq.resize(N, K, fmt);
    if (scales.rows != N || scales.cols != 1 || scales.dtype != Dtype::FP32) scales.resize(N, 1, Dtype::FP32);
    if (N == 0) return;
    const MirKernel mk = mir_kernel(
        {MirKind::WqRows, 0u, W.dtype, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(fmt == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min(N, 65535));
    const std::uint32_t gy = static_cast<std::uint32_t>((static_cast<std::uint64_t>(N) + gx - 1u) / gx);
    const void* ptrs[] = {W.data, Wq.data, scales.data};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args_3d(mk, gx, gy, 1u, op, ptrs, ints, {});
}

void dequantize_rows(const Tensor& Wq, const Tensor& scales, Dtype out, Tensor& W) {
    constexpr const char* op = "dequantize_rows";
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    if (out != Dtype::FP16 && out != Dtype::BF16 && out != Dtype::FP32) fail(op, "out must be FP16, BF16 or FP32");
    const int N = Wq.rows, K = Wq.cols;
    if (W.rows != N || W.cols != K || W.dtype != out) W.resize(N, K, out);
    const std::uint64_t n = static_cast<std::uint64_t>(N) * static_cast<std::uint64_t>(K);
    if (n == 0) return;
    const MirKernel mk = mir_kernel(
        {MirKind::WqRows, 1u, out, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(Wq.dtype == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((n + 255u) / 256u, 1u, 65535u));
    const void* ptrs[] = {W.data, Wq.data, scales.data};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args(mk, gx, op, ptrs, ints, {});
}

void linear_forward_batched_wq_ex(const Tensor& Wq, const Tensor& scales, const Tensor* bias, const Tensor& X,
                                  int act, int epilogue_flags, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_wq_ex";
    const int epi = epilogue_flags & ~kLinearEpiFastAccum;   // accumulation is FP32 regardless
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    const int N = Wq.rows, K = Wq.cols, M = X.rows;
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != N)
        fail(op, "scales must be (out,1) FP32");
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail(op, "X must be FP16, BF16 or FP32");
    if (X.cols != K) fail(op, "X.cols must equal Wq.cols");
    if (bias && (bias->dtype != dt || static_cast<long long>(bias->rows) * bias->cols != N))
        fail(op, "bias must be (out,1) in X's dtype");
    if (epi < kLinearEpiStore || epi > kLinearEpiSwiglu) fail(op, "unknown epilogue");
    const bool gated = epi == kLinearEpiGeglu || epi == kLinearEpiSwiglu;
    if (gated && (act != 0 || N % 2 != 0)) fail(op, "geglu/swiglu needs act 0 and even Wq.rows");
    const int out_cols = gated ? N / 2 : N;
    if (epi == kLinearEpiAccumulate) {
        if (Y.rows != M || Y.cols != N || Y.dtype != dt) fail(op, "accumulate needs Y (B, out) in X's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(M, out_cols, dt);
    }
    if (M == 0 || out_cols == 0) return;

    WqParams p{};
    p.M = static_cast<uint32_t>(M);
    p.N = static_cast<uint32_t>(N);
    p.K = static_cast<uint32_t>(K);
    p.out_cols = static_cast<uint32_t>(out_cols);
    p.fp8 = Wq.dtype == Dtype::FP8_E4M3 ? 1u : 0u;
    p.has_bias = bias ? 1u : 0u;
    p.act = act;
    p.epi = epi;
    // The 8-wide loads: every Wq row and X row starts 8-byte / 4-element
    // aligned when K is a multiple of 8 and the buffers' offsets are.
    const NSUInteger xAlign = 4 * static_cast<NSUInteger>(dtype_size_bytes(dt));
    p.vec = K % 8 == 0 && buffer_offset_for(Wq) % 8 == 0 && buffer_offset_for(X) % xAlign == 0 ? 1u : 0u;
    id<MTLComputePipelineState> pso = dt == Dtype::FP16 ? pso_wq_fp16() : dt == Dtype::BF16 ? pso_wq_bf16()
                                                                                          : pso_wq_fp32();
    const NSUInteger tgs = (static_cast<NSUInteger>(out_cols) + kColsPerTg - 1) / kColsPerTg;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(Wq) offset:buffer_offset_for(Wq) atIndex:0];
        [enc setBuffer:buffer_for(scales) offset:buffer_offset_for(scales) atIndex:1];
        // No bias: X stands in at its index, never read (has_bias 0).
        [enc setBuffer:buffer_for(bias ? *bias : X) offset:buffer_offset_for(bias ? *bias : X) atIndex:2];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:3];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:4];
        [enc setBytes:&p length:sizeof(WqParams) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * kColsPerTg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace brotensor::detail::metal
