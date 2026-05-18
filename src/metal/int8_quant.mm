// Metal implementation of W8A16 (int8 weight, fp16 activation) matmul + conv2d.
// Mirrors src/cuda/int8_quant.cu. The host quantiser is portable; only the two
// device ops dispatch Metal kernels.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

// ─── Host quantiser ────────────────────────────────────────────────────────

void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    for (int r = 0; r < out; ++r) {
        const uint16_t* row = W_fp16 + static_cast<size_t>(r) * static_cast<size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + static_cast<size_t>(r) * static_cast<size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    }
}

// ─── Device kernels ────────────────────────────────────────────────────────

namespace {

constexpr NSUInteger MM_TILE = 16;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint MM_TILE = 16;

// Y(M, N) = (W_int8(M, K) * scale[row]) @ X_fp16(K, N). Tiled in MM_TILE.
kernel void k_matmul_int8w_fp16(device const char*  W      [[buffer(0)]],
                                device const float* scales [[buffer(1)]],
                                device const half*  X      [[buffer(2)]],
                                device half*        Y      [[buffer(3)]],
                                constant uint& M           [[buffer(4)]],
                                constant uint& N           [[buffer(5)]],
                                constant uint& K           [[buffer(6)]],
                                threadgroup float* smem    [[threadgroup(0)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint2 li [[thread_position_in_threadgroup]]) {
    threadgroup float* Ws = smem;                       // MM_TILE * MM_TILE
    threadgroup float* Xs = smem + MM_TILE * MM_TILE;   // MM_TILE * MM_TILE

    uint row = tg.y * MM_TILE + li.y;
    uint col = tg.x * MM_TILE + li.x;

    float row_scale = (row < M) ? scales[row] : 0.0f;

    float acc = 0.0f;
    uint n_tiles = (K + MM_TILE - 1u) / MM_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint w_col = t * MM_TILE + li.x;
        uint x_row = t * MM_TILE + li.y;

        float w_val = 0.0f;
        if (row < M && w_col < K) {
            int8_t wi = ((device const int8_t*)W)[row * K + w_col];
            w_val = float(wi) * row_scale;
        }
        Ws[li.y * MM_TILE + li.x] = w_val;

        float x_val = 0.0f;
        if (x_row < K && col < N) {
            x_val = float(X[x_row * N + col]);
        }
        Xs[li.y * MM_TILE + li.x] = x_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MM_TILE; ++k) {
            acc += Ws[li.y * MM_TILE + k] * Xs[k * MM_TILE + li.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        Y[row * N + col] = half(acc);
    }
}

// Y(B, out) = X_fp16(B, K) @ (W_int8(out, K) * scale[out])^T + bias[out].
// (B,in) → (B,out) layout — mirrors linear_forward_batched_fp16_gpu.
kernel void k_linear_batched_int8w_fp16(device const half*  X      [[buffer(0)]],
                                        device const char*  W      [[buffer(1)]],
                                        device const float* scales [[buffer(2)]],
                                        device const half*  bias   [[buffer(3)]],
                                        device half*        Y      [[buffer(4)]],
                                        constant uint& B           [[buffer(5)]],
                                        constant uint& M           [[buffer(6)]],
                                        constant uint& K           [[buffer(7)]],
                                        constant uint& has_bias    [[buffer(8)]],
                                        threadgroup float* smem    [[threadgroup(0)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint2 li [[thread_position_in_threadgroup]]) {
    threadgroup float* Xs = smem;                       // MM_TILE * MM_TILE
    threadgroup float* Ws = smem + MM_TILE * MM_TILE;   // MM_TILE * MM_TILE

    uint b = tg.y * MM_TILE + li.y;
    uint m = tg.x * MM_TILE + li.x;
    float m_scale = (m < M) ? scales[m] : 0.0f;

    float acc = 0.0f;
    uint n_tiles = (K + MM_TILE - 1u) / MM_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint x_k = t * MM_TILE + li.x;
        uint w_k = t * MM_TILE + li.y;

        float x_val = 0.0f;
        if (b < B && x_k < K) {
            x_val = float(X[b * K + x_k]);
        }
        Xs[li.y * MM_TILE + li.x] = x_val;

        float w_val = 0.0f;
        if (m < M && w_k < K) {
            int8_t wi = ((device const int8_t*)W)[m * K + w_k];
            w_val = float(wi) * m_scale;
        }
        Ws[li.y * MM_TILE + li.x] = w_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MM_TILE; ++k) {
            acc += Xs[li.y * MM_TILE + k] * Ws[k * MM_TILE + li.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (b < B && m < M) {
        if (has_bias != 0u) acc += float(bias[m]);
        Y[b * M + m] = half(acc);
    }
}

struct ConvI8Params {
    uint N, C_in, H, W;
    uint C_out, kH, kW;
    uint H_out, W_out;
    int  stride_h, stride_w;
    int  pad_h, pad_w;
    int  dil_h, dil_w;
    uint has_bias;
    uint total;
    uint groups;
    uint Cg_in;
    uint Cg_out;
};

// One thread per output element. W is INT8; per-row (per-c_out) FP32 scale.
kernel void k_conv2d_int8w_fp16_forward(device const half*  X      [[buffer(0)]],
                                        device const char*  W      [[buffer(1)]],
                                        device const float* scales [[buffer(2)]],
                                        device const half*  bias   [[buffer(3)]],
                                        device half*        Y      [[buffer(4)]],
                                        constant ConvI8Params& p   [[buffer(5)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float scale = scales[oc];

    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.H * p.W;

    float acc = 0.0f;
    device const int8_t* W_i8 = (device const int8_t*)W;
    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.H * p.W;
        for (uint kh = 0; kh < p.kH; ++kh) {
            int in_h = in_h_origin + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint kw = 0; kw < p.kW; ++kw) {
                int in_w = in_w_origin + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                float x_v = float(X[x_ic_base + uint(in_h) * p.W + uint(in_w)]);
                float w_v = float(W_i8[w_ic_base + kh * p.kW + kw]) * scale;
                acc += x_v * w_v;
            }
        }
    }
    if (p.has_bias != 0u) {
        acc += float(bias[oc]);
    }
    Y[idx] = half(acc);
}
)msl";

id<MTLComputePipelineState> pso_matmul() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_int8w_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_int8w_fp16_forward"); });
    return pso;
}

id<MTLComputePipelineState> pso_linear_batched() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_batched_int8w_fp16"); });
    return pso;
}

struct ConvI8Params {
    uint32_t N, C_in, H, W;
    uint32_t C_out, kH, kW;
    uint32_t H_out, W_out;
    int32_t  stride_h, stride_w;
    int32_t  pad_h, pad_w;
    int32_t  dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
    uint32_t groups;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

} // namespace

void matmul_int8w_fp16_gpu(const GpuTensor& W_int8,
                           const GpuTensor& scales,
                           const GpuTensor& X,
                           GpuTensor& Y) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: W_int8 must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: scales must be FP32");
    }
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: X must be FP16");
    }
    const int M = W_int8.rows;
    const int K = W_int8.cols;
    if (X.rows != K) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: K mismatch (W.cols != X.rows)");
    }
    if (scales.rows != M || scales.cols != 1) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: scales shape must be (out, 1)");
    }
    const int Nb = X.cols;
    if (Y.rows != M || Y.cols != Nb || Y.dtype != Dtype::FP16) {
        Y.resize(M, Nb, Dtype::FP16);
    }
    if (M == 0 || Nb == 0) return;
    if (K == 0) {
        Y.zero();
        return;
    }

    id<MTLComputePipelineState> pso = pso_matmul();
    id<MTLBuffer> bW = buffer_for(W_int8);
    id<MTLBuffer> bS = buffer_for(scales);
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oW = buffer_offset_for(W_int8);
    const NSUInteger oS = buffer_offset_for(scales);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oY = buffer_offset_for(Y);

    const uint32_t uM = static_cast<uint32_t>(M);
    const uint32_t uN = static_cast<uint32_t>(Nb);
    const uint32_t uK = static_cast<uint32_t>(K);

    const NSUInteger shmem = 2 * MM_TILE * MM_TILE * sizeof(float);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:oW atIndex:0];
        [enc setBuffer:bS offset:oS atIndex:1];
        [enc setBuffer:bX offset:oX atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&uM length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&uN length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&uK length:sizeof(uint32_t) atIndex:6];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        NSUInteger grid_x = (static_cast<NSUInteger>(Nb) + MM_TILE - 1) / MM_TILE;
        NSUInteger grid_y = (static_cast<NSUInteger>(M)  + MM_TILE - 1) / MM_TILE;
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(MM_TILE, MM_TILE, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void conv2d_int8w_fp16_forward_gpu(const GpuTensor& X,
                                   const GpuTensor& W_int8,
                                   const GpuTensor& scales,
                                   const GpuTensor* bias,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dil_h, int dil_w, int groups,
                                   GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_int8w_fp16_forward_gpu: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: scales shape mismatch");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    ConvI8Params p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLComputePipelineState> pso = pso_conv();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(W_int8);
    id<MTLBuffer> bs = buffer_for(scales);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(W_int8);
    const NSUInteger os = buffer_offset_for(scales);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bs offset:os atIndex:2];
        [enc setBuffer:bb offset:ob atIndex:3];
        [enc setBuffer:by offset:oy atIndex:4];
        [enc setBytes:&p length:sizeof(ConvI8Params) atIndex:5];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void linear_forward_batched_int8w_fp16_gpu(const GpuTensor& W_int8,
                                           const GpuTensor& scales,
                                           const GpuTensor* bias,
                                           const GpuTensor& X_BD,
                                           GpuTensor& Y_BD) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: scales must be FP32");
    }
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: X must be FP16");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: bias must be FP16");
    }
    const int B   = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W_int8.rows;
    if (W_int8.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: shape mismatch (W.cols != X.cols)");
    }
    if (scales.rows != out_dim || scales.cols != 1) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16_gpu: scales shape must be (out, 1)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out_dim, Dtype::FP16);
    }
    if (B == 0 || out_dim == 0) return;
    if (in_dim == 0) { Y_BD.zero(); return; }

    id<MTLComputePipelineState> pso = pso_linear_batched();
    id<MTLBuffer> bx = buffer_for(X_BD);
    id<MTLBuffer> bw = buffer_for(W_int8);
    id<MTLBuffer> bs = buffer_for(scales);
    id<MTLBuffer> by = buffer_for(Y_BD);
    id<MTLBuffer> bb = (bias && bias->size() > 0) ? buffer_for(*bias) : bx;
    const NSUInteger ox = buffer_offset_for(X_BD);
    const NSUInteger ow_ = buffer_offset_for(W_int8);
    const NSUInteger os = buffer_offset_for(scales);
    const NSUInteger oy = buffer_offset_for(Y_BD);
    const NSUInteger ob = (bias && bias->size() > 0) ? buffer_offset_for(*bias) : 0;

    const uint32_t uB = static_cast<uint32_t>(B);
    const uint32_t uM = static_cast<uint32_t>(out_dim);
    const uint32_t uK = static_cast<uint32_t>(in_dim);
    const uint32_t uHas = (bias && bias->size() > 0) ? 1u : 0u;

    const NSUInteger shmem = 2 * MM_TILE * MM_TILE * sizeof(float);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bs offset:os atIndex:2];
        [enc setBuffer:bb offset:ob atIndex:3];
        [enc setBuffer:by offset:oy atIndex:4];
        [enc setBytes:&uB   length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&uM   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&uK   length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&uHas length:sizeof(uint32_t) atIndex:8];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        NSUInteger grid_x = (static_cast<NSUInteger>(out_dim) + MM_TILE - 1) / MM_TILE;
        NSUInteger grid_y = (static_cast<NSUInteger>(B)       + MM_TILE - 1) / MM_TILE;
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(MM_TILE, MM_TILE, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
