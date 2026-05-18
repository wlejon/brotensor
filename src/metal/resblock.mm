#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger RB_GN_BLOCK   = 256;
constexpr NSUInteger RB_CONV_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint RB_GN_BLOCK = 256;

// Fused GroupNorm + SiLU. One threadgroup per (sample, group).
kernel void k_gn_silu_fused(
        device const half* X     [[buffer(0)]],
        device const half* gamma [[buffer(1)]],
        device const half* beta  [[buffer(2)]],
        device half*       Y     [[buffer(3)]],
        constant uint& C                  [[buffer(4)]],
        constant uint& spatial            [[buffer(5)]],
        constant uint& channels_per_group [[buffer(6)]],
        constant float& eps               [[buffer(7)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    threadgroup float s_sum[RB_GN_BLOCK];
    threadgroup float s_sumsq[RB_GN_BLOCK];
    threadgroup float s_mean;
    threadgroup float s_rstd;
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    uint g = gid.x;
    uint n = gid.y;
    uint tile_size = channels_per_group * spatial;
    uint chan_base = g * channels_per_group;
    uint sample_stride = C * spatial;
    device const half* x_tile = X + n * sample_stride + chan_base * spatial;
    device       half* y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        float v = float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float inv_n = 1.0f / float(tile_size);
        float mean = s_sum[0] * inv_n;
        float var  = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;
    float rstd = s_rstd;

    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv = float(gamma[channel]);
        float bv = float(beta[channel]);
        float v  = float(x_tile[i]);
        float yn = (v - mean) * rstd * gv + bv;
        float silu = yn / (1.0f + exp(-yn));
        y_tile[i] = half(silu);
    }
}

struct ConvParams {
    uint N, C_in, C_out, H, Wd;
    uint has_bias, has_skip;
    uint total;
};

// 3x3 same-padding conv with optional bias and optional skip-add epilogue.
kernel void k_conv3x3_same_add(
        device const half* X    [[buffer(0)]],
        device const half* W    [[buffer(1)]],
        device const half* bias [[buffer(2)]],   // dummy when has_bias==0
        device const half* skip [[buffer(3)]],   // dummy when has_skip==0
        device half*       Y    [[buffer(4)]],
        constant ConvParams& p  [[buffer(5)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.Wd;
    uint t  = idx / p.Wd;
    uint oh = t % p.H;
    t /= p.H;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    float acc = 0.0f;
    uint w_oc_base = oc * p.C_in * 9u;
    uint x_n_base  = n * p.C_in * p.H * p.Wd;
    for (uint ic = 0; ic < p.C_in; ++ic) {
        uint w_ic_base = w_oc_base + ic * 9u;
        uint x_ic_base = x_n_base  + ic * p.H * p.Wd;
        for (uint kh = 0; kh < 3; ++kh) {
            int in_h = int(oh) + int(kh) - 1;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint kw = 0; kw < 3; ++kw) {
                int in_w = int(ow) + int(kw) - 1;
                if (in_w < 0 || in_w >= int(p.Wd)) continue;
                float xv = float(X[x_ic_base + uint(in_h) * p.Wd + uint(in_w)]);
                float wv = float(W[w_ic_base + kh * 3u + kw]);
                acc += xv * wv;
            }
        }
    }
    if (p.has_bias != 0u) acc += float(bias[oc]);
    if (p.has_skip != 0u) acc += float(skip[idx]);
    Y[idx] = half(acc);
}

struct Conv1x1Params {
    uint N, C_in, C_out, spatial;
    uint has_bias;
    uint total;
};

kernel void k_conv1x1(device const half* X    [[buffer(0)]],
                      device const half* W    [[buffer(1)]],
                      device const half* bias [[buffer(2)]],
                      device half*       Y    [[buffer(3)]],
                      constant Conv1x1Params& p [[buffer(4)]],
                      uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint s  = idx % p.spatial;
    uint t  = idx / p.spatial;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;
    float acc = 0.0f;
    uint w_base    = oc * p.C_in;
    uint x_n_base  = n  * p.C_in * p.spatial;
    for (uint ic = 0; ic < p.C_in; ++ic) {
        acc += float(X[x_n_base + ic * p.spatial + s]) *
               float(W[w_base + ic]);
    }
    if (p.has_bias != 0u) acc += float(bias[oc]);
    Y[idx] = half(acc);
}

struct ShiftParams {
    uint N, C, spatial, has_N, total;
};

kernel void k_add_NC_shift(device half*       Y     [[buffer(0)]],
                           device const half* shift [[buffer(1)]],
                           constant ShiftParams& p  [[buffer(2)]],
                           uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint t = idx / p.spatial;
    uint c = t % p.C;
    uint n = t / p.C;
    uint sidx = (p.has_N != 0u) ? (n * p.C + c) : c;
    Y[idx] = half(float(Y[idx]) + float(shift[sidx]));
}

// Per-(n, c) HW reduction of an NCHW FP16 tensor, accumulated (folded) into
// d_shift[n, c]. One threadgroup per (n, c); RB_GN_BLOCK threads reduce.
struct ReduceNCParams { uint N, C, spatial; };

kernel void k_sum_hw_per_NC(
        device const half* dh2     [[buffer(0)]],
        device half*       d_shift [[buffer(1)]],
        constant ReduceNCParams& p [[buffer(2)]],
        uint3 gid  [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgs3 [[threads_per_threadgroup]]) {
    threadgroup float s_buf[RB_GN_BLOCK];
    uint nc = gid.x;
    if (nc >= p.N * p.C) return;
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    device const half* row = dh2 + nc * p.spatial;
    float acc = 0.0f;
    for (uint i = tid; i < p.spatial; i += tg_size) {
        acc += float(row[i]);
    }
    s_buf[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float prev = float(d_shift[nc]);
        d_shift[nc] = half(prev + s_buf[0]);
    }
}
)msl";

id<MTLComputePipelineState> pso_gn_silu() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_gn_silu_fused"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv3x3() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv3x3_same_add"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv1x1_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv1x1"); });
    return pso;
}
id<MTLComputePipelineState> pso_add_shift() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_add_NC_shift"); });
    return pso;
}
id<MTLComputePipelineState> pso_sum_hw_per_NC() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_sum_hw_per_NC"); });
    return pso;
}

struct ConvParams {
    uint32_t N, C_in, C_out, H, Wd;
    uint32_t has_bias, has_skip;
    uint32_t total;
};
struct Conv1x1Params {
    uint32_t N, C_in, C_out, spatial;
    uint32_t has_bias;
    uint32_t total;
};
struct ShiftParams {
    uint32_t N, C, spatial, has_N, total;
};
struct ReduceNCParams {
    uint32_t N, C, spatial;
};

void launch_gn_silu(const GpuTensor& X,
                    const GpuTensor& gamma, const GpuTensor& beta,
                    GpuTensor& Y,
                    int N, int C, int spatial, int channels_per_group, float eps) {
    id<MTLComputePipelineState> pso = pso_gn_silu();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bg = buffer_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger og = buffer_offset_for(gamma);
    const NSUInteger ob = buffer_offset_for(beta);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Cu = C, Su = spatial, cpg = channels_per_group;
    const int num_groups = C / channels_per_group;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Cu  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Su  length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&cpg length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps length:sizeof(float)    atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, N, 1)
            threadsPerThreadgroup:MTLSizeMake(RB_GN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void launch_conv3x3(const GpuTensor& X, const GpuTensor& W,
                    const GpuTensor* bias, const GpuTensor* skip,
                    GpuTensor& Y,
                    int N, int C_in, int C_out, int H, int Wd) {
    id<MTLComputePipelineState> pso = pso_conv3x3();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(W);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
    id<MTLBuffer> bs = skip ? buffer_for(*skip) : bx;
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(W);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;
    const NSUInteger os = skip ? buffer_offset_for(*skip) : 0;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.C_out = C_out; p.H = H; p.Wd = Wd;
    p.has_bias = bias ? 1u : 0u;
    p.has_skip = skip ? 1u : 0u;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(C_out) *
              static_cast<uint32_t>(H) * static_cast<uint32_t>(Wd);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:bs offset:os atIndex:3];
        [enc setBuffer:by offset:oy atIndex:4];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:5];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > RB_CONV_BLOCK) tg = RB_CONV_BLOCK;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void launch_conv1x1(const GpuTensor& X, const GpuTensor& W,
                    const GpuTensor* bias,
                    GpuTensor& Y,
                    int N, int C_in, int C_out, int spatial) {
    id<MTLComputePipelineState> pso = pso_conv1x1_pso();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(W);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(W);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    Conv1x1Params p{};
    p.N = N; p.C_in = C_in; p.C_out = C_out; p.spatial = spatial;
    p.has_bias = bias ? 1u : 0u;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(C_out) *
              static_cast<uint32_t>(spatial);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&p length:sizeof(Conv1x1Params) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > RB_CONV_BLOCK) tg = RB_CONV_BLOCK;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void launch_add_shift(GpuTensor& Y, const GpuTensor& shift,
                      int N, int C, int spatial, int has_N) {
    id<MTLComputePipelineState> pso = pso_add_shift();
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bs = buffer_for(shift);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger os = buffer_offset_for(shift);

    ShiftParams p{};
    p.N = N; p.C = C; p.spatial = spatial;
    p.has_N = static_cast<uint32_t>(has_N);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(C) *
              static_cast<uint32_t>(spatial);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bs offset:os atIndex:1];
        [enc setBytes:&p length:sizeof(ShiftParams) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > RB_CONV_BLOCK) tg = RB_CONV_BLOCK;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void launch_sum_hw_per_NC(const GpuTensor& dh2, GpuTensor& d_shift,
                          int N, int C, int spatial) {
    id<MTLComputePipelineState> pso = pso_sum_hw_per_NC();
    id<MTLBuffer> bx = buffer_for(dh2);
    id<MTLBuffer> bs = buffer_for(d_shift);
    const NSUInteger ox = buffer_offset_for(dh2);
    const NSUInteger os = buffer_offset_for(d_shift);

    ReduceNCParams p{};
    p.N = N; p.C = C; p.spatial = spatial;
    const uint32_t blocks = static_cast<uint32_t>(N) * static_cast<uint32_t>(C);
    if (blocks == 0 || spatial == 0) return;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bs offset:os atIndex:1];
        [enc setBytes:&p length:sizeof(ReduceNCParams) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(RB_GN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void resblock_forward_gpu(const GpuTensor& X,
                          const GpuTensor& gamma1, const GpuTensor& beta1,
                          const GpuTensor& W1, const GpuTensor* b1,
                          const GpuTensor* t_emb_shift,
                          const GpuTensor& gamma2, const GpuTensor& beta2,
                          const GpuTensor& W2, const GpuTensor* b2,
                          const GpuTensor* Wskip, const GpuTensor* bskip,
                          int N, int C_in, int C_out, int H, int Wd,
                          int num_groups, float eps,
                          GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || W1.dtype != Dtype::FP16 ||
        gamma2.dtype != Dtype::FP16 || beta2.dtype != Dtype::FP16 ||
        W2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_gpu: all required tensors must be FP16");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_gpu: num_groups must divide C_in and C_out");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_forward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    GpuTensor h1(N, C_in  * spatial, Dtype::FP16);
    GpuTensor h2(N, C_out * spatial, Dtype::FP16);
    GpuTensor h3(N, C_out * spatial, Dtype::FP16);

    launch_gn_silu(X, gamma1, beta1, h1, N, C_in, spatial,
                   C_in / num_groups, eps);
    launch_conv3x3(h1, W1, b1, nullptr, h2, N, C_in, C_out, H, Wd);

    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    launch_gn_silu(h2, gamma2, beta2, h3, N, C_out, spatial,
                   C_out / num_groups, eps);

    GpuTensor skip_scratch;
    const GpuTensor* skip_ptr = nullptr;
    if (Wskip == nullptr) {
        skip_ptr = &X;
    } else {
        if (Wskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_gpu: Wskip must be FP16");
        }
        skip_scratch.resize(N, C_out * spatial, Dtype::FP16);
        launch_conv1x1(X, *Wskip, bskip, skip_scratch, N, C_in, C_out, spatial);
        skip_ptr = &skip_scratch;
    }

    launch_conv3x3(h3, W2, b2, skip_ptr, Y, N, C_out, C_out, H, Wd);
}

void resblock_backward_gpu(const GpuTensor& X,
                           const GpuTensor& gamma1, const GpuTensor& beta1,
                           const GpuTensor& W1, const GpuTensor* b1,
                           const GpuTensor* t_emb_shift,
                           const GpuTensor& gamma2, const GpuTensor& beta2,
                           const GpuTensor& W2, const GpuTensor* b2,
                           const GpuTensor* Wskip, const GpuTensor* bskip,
                           int N, int C_in, int C_out, int H, int Wd,
                           int num_groups, float eps,
                           const GpuTensor& dY,
                           GpuTensor& dX,
                           GpuTensor& dGamma1, GpuTensor& dBeta1,
                           GpuTensor& dW1, GpuTensor* db1,
                           GpuTensor* dt_emb_shift,
                           GpuTensor& dGamma2, GpuTensor& dBeta2,
                           GpuTensor& dW2, GpuTensor* db2,
                           GpuTensor* dWskip, GpuTensor* dbskip) {
    if (X.dtype != Dtype::FP16 || dY.dtype != Dtype::FP16 ||
        gamma1.dtype != Dtype::FP16 || beta1.dtype != Dtype::FP16 ||
        W1.dtype != Dtype::FP16 ||
        gamma2.dtype != Dtype::FP16 || beta2.dtype != Dtype::FP16 ||
        W2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_backward_gpu: all required tensors must be FP16");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_backward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    if (dY.rows != N || dY.cols != C_out * spatial) {
        throw std::runtime_error("resblock_backward_gpu: dY shape mismatch");
    }
    if (dX.rows != N || dX.cols != C_in * spatial || dX.dtype != Dtype::FP16) {
        dX.resize(N, C_in * spatial, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    // Recompute forward intermediates via public ops.
    GpuTensor h1_pre_silu, h1;
    group_norm_forward_gpu(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps,
                           h1_pre_silu);
    silu_forward_gpu(h1_pre_silu, h1);

    GpuTensor h2;
    conv2d_forward_gpu(h1, W1, b1, N, C_in, H, Wd,
                       C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    GpuTensor h3_pre_silu, h3;
    group_norm_forward_gpu(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps,
                           h3_pre_silu);
    silu_forward_gpu(h3_pre_silu, h3);

    // Conv2 backward.
    GpuTensor dh3(N, C_out * spatial, Dtype::FP16);
    conv2d_backward_input_gpu(W2, dY, N, C_out, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh3);
    conv2d_backward_weight_gpu(h3, dY, N, C_out, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) {
        conv2d_backward_bias_gpu(dY, N, C_out, H, Wd, *db2);
    }

    // SiLU2 backward.
    GpuTensor dh3_pre_silu;
    silu_backward_gpu(h3_pre_silu, dh3, dh3_pre_silu);

    // GN2 backward.
    GpuTensor dh2;
    group_norm_backward_gpu(h2, gamma2, dh3_pre_silu, N, C_out, H, Wd,
                            num_groups, eps, dh2, dGamma2, dBeta2);

    // t_emb_shift backward.
    if (t_emb_shift && dt_emb_shift) {
        if (dt_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: dt_emb_shift must be FP16");
        }
        const bool has_N = (t_emb_shift->rows == N && t_emb_shift->cols == C_out);
        if (has_N) {
            launch_sum_hw_per_NC(dh2, *dt_emb_shift, N, C_out, spatial);
        } else {
            conv2d_backward_bias_gpu(dh2, N, C_out, H, Wd, *dt_emb_shift);
        }
    }

    // Conv1 backward.
    GpuTensor dh1(N, C_in * spatial, Dtype::FP16);
    conv2d_backward_input_gpu(W1, dh2, N, C_in, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh1);
    conv2d_backward_weight_gpu(h1, dh2, N, C_in, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) {
        conv2d_backward_bias_gpu(dh2, N, C_out, H, Wd, *db1);
    }

    // SiLU1 backward.
    GpuTensor dh1_pre_silu;
    silu_backward_gpu(h1_pre_silu, dh1, dh1_pre_silu);

    // GN1 backward (writes dX).
    group_norm_backward_gpu(X, gamma1, dh1_pre_silu, N, C_in, H, Wd,
                            num_groups, eps, dX, dGamma1, dBeta1);

    // Skip path backward, then sum into dX.
    if (Wskip == nullptr) {
        add_inplace_gpu(dX, dY);
    } else {
        if (Wskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: Wskip must be FP16");
        }
        GpuTensor dX_skip(N, C_in * spatial, Dtype::FP16);
        conv2d_backward_input_gpu(*Wskip, dY, N, C_in, H, Wd,
                                  C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX_skip);
        if (dWskip) {
            conv2d_backward_weight_gpu(X, dY, N, C_in, H, Wd,
                                       C_out, 1, 1, 1, 1, 0, 0, 1, 1, *dWskip);
        }
        if (dbskip) {
            conv2d_backward_bias_gpu(dY, N, C_out, H, Wd, *dbskip);
        }
        add_inplace_gpu(dX, dX_skip);
    }
}

} // namespace brotensor
