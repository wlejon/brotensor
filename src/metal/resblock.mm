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

    // Reuse scratch across calls. SD1.5 invokes ~30 resblocks per UNet step,
    // so a fresh allocation per call burns Metal heap traffic; GpuTensor::resize
    // is a no-op when shape/dtype already match (tensor.mm:180). Inference is
    // single-threaded against the device, so thread_local lifetime is fine.
    thread_local GpuTensor h1, h2, h3;
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    launch_gn_silu(X, gamma1, beta1, h1, N, C_in, spatial,
                   C_in / num_groups, eps);

    // Conv1: 3x3 same. Dispatch through the public conv2d (simdgroup fast path).
    conv2d_forward_gpu(h1, W1, b1,
                       N, C_in, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       h2);

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

    // Prepare the skip tensor for the conv2 epilogue (post-conv2 add).
    thread_local GpuTensor skip_scratch;
    if (Wskip != nullptr) {
        if (Wskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_gpu: Wskip must be FP16");
        }
        // 1x1 conv through the public path.
        conv2d_forward_gpu(X, *Wskip, bskip,
                           N, C_in, H, Wd,
                           C_out, 1, 1,
                           /*stride*/1, 1,
                           /*pad*/0, 0,
                           /*dil*/1, 1,
                           skip_scratch);
    }

    // Conv2 (3x3 same) → Y, then fuse-in the skip via add_inplace_gpu.
    conv2d_forward_gpu(h3, W2, b2,
                       N, C_out, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       Y);
    if (Wskip == nullptr) {
        // skip is X itself (C_in == C_out so shapes match).
        add_inplace_gpu(Y, X);
    } else {
        add_inplace_gpu(Y, skip_scratch);
    }
}

void resblock_forward_int8w_fp16_gpu(const GpuTensor& X,
                                     const GpuTensor& gamma1, const GpuTensor& beta1,
                                     const GpuTensor& W1_int8, const GpuTensor& s1,
                                     const GpuTensor* b1,
                                     const GpuTensor* t_emb_shift,
                                     const GpuTensor& gamma2, const GpuTensor& beta2,
                                     const GpuTensor& W2_int8, const GpuTensor& s2,
                                     const GpuTensor* b2,
                                     const GpuTensor* Wskip_int8, const GpuTensor* sskip,
                                     const GpuTensor* bskip,
                                     int N, int C_in, int C_out, int H, int Wd,
                                     int num_groups, float eps,
                                     GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || gamma2.dtype != Dtype::FP16 ||
        beta2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: activation/norm tensors must be FP16");
    }
    if (W1_int8.dtype != Dtype::INT8 || W2_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1/W2 must be INT8");
    }
    if (s1.dtype != Dtype::FP32 || s2.dtype != Dtype::FP32) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: scales s1/s2 must be FP32");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: num_groups must divide C_in and C_out");
    }
    if (W1_int8.rows != C_out || W1_int8.cols != C_in * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1_int8 shape mismatch");
    }
    if (W2_int8.rows != C_out || W2_int8.cols != C_out * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W2_int8 shape mismatch");
    }
    if (s1.rows != C_out || s1.cols != 1 || s2.rows != C_out || s2.cols != 1) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: s1/s2 must be (C_out, 1)");
    }
    if (b1 && b1->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b1 must be FP16");
    }
    if (b2 && b2->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b2 must be FP16");
    }
    if (Wskip_int8 == nullptr) {
        if (C_in != C_out) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip required when C_in != C_out");
        }
    } else {
        if (Wskip_int8->dtype != Dtype::INT8) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 must be INT8");
        }
        if (Wskip_int8->rows != C_out || Wskip_int8->cols != C_in) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 shape mismatch");
        }
        if (sskip == nullptr || sskip->dtype != Dtype::FP32 ||
            sskip->rows != C_out || sskip->cols != 1) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: sskip must be FP32 (C_out, 1)");
        }
        if (bskip && bskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: bskip must be FP16");
        }
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    thread_local GpuTensor h1, h2, h3;
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    launch_gn_silu(X, gamma1, beta1, h1, N, C_in, spatial,
                   C_in / num_groups, eps);

    conv2d_int8w_fp16_forward_gpu(h1, W1_int8, s1, b1,
                                  N, C_in, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  h2);

    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    launch_gn_silu(h2, gamma2, beta2, h3, N, C_out, spatial,
                   C_out / num_groups, eps);

    thread_local GpuTensor skip_scratch;
    if (Wskip_int8 != nullptr) {
        conv2d_int8w_fp16_forward_gpu(X, *Wskip_int8, *sskip, bskip,
                                      N, C_in, H, Wd,
                                      C_out, 1, 1,
                                      1, 1, 0, 0, 1, 1, /*groups*/1,
                                      skip_scratch);
    }

    conv2d_int8w_fp16_forward_gpu(h3, W2_int8, s2, b2,
                                  N, C_out, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  Y);
    if (Wskip_int8 == nullptr) {
        add_inplace_gpu(Y, X);
    } else {
        add_inplace_gpu(Y, skip_scratch);
    }
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
