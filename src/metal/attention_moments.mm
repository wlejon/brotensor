// Per-text-token spatial moments of an FP16 cross-attention map. One
// threadgroup per text token k; threads stride over q = y * w_lat + x and
// reduce three running sums (mass, y-mass, x-mass) in threadgroup memory.
// Thread 0 divides and writes (mass, centroid_y, centroid_x).

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

constexpr NSUInteger MOM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint MOM_BLOCK = 256;
constant float MOM_EPS  = 1e-8;

struct MomentsParams {
    uint Lq;
    uint Lk;
    uint h_lat;
    uint w_lat;
};

kernel void k_attn_token_moments(device const half*  Attn      [[buffer(0)]],
                                 device float*       mass      [[buffer(1)]],
                                 device float*       centroid  [[buffer(2)]],
                                 constant MomentsParams& p     [[buffer(3)]],
                                 uint3 gid  [[threadgroup_position_in_grid]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tgs3 [[threads_per_threadgroup]]) {
    threadgroup float sm_m[MOM_BLOCK];
    threadgroup float sm_y[MOM_BLOCK];
    threadgroup float sm_x[MOM_BLOCK];

    uint k = gid.x;
    if (k >= p.Lk) return;
    uint tid = tid3.x;
    uint tg_size = tgs3.x;

    float am = 0.0, ay = 0.0, ax = 0.0;
    for (uint q = tid; q < p.Lq; q += tg_size) {
        float a = float(Attn[q * p.Lk + k]);
        uint y = q / p.w_lat;
        uint x = q - y * p.w_lat;
        am += a;
        ay += float(y) * a;
        ax += float(x) * a;
    }
    sm_m[tid] = am;
    sm_y[tid] = ay;
    sm_x[tid] = ax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm_m[tid] += sm_m[tid + s];
            sm_y[tid] += sm_y[tid + s];
            sm_x[tid] += sm_x[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = sm_m[0];
        mass[k] = m;
        if (m > MOM_EPS) {
            float inv = 1.0 / m;
            centroid[k * 2 + 0] = sm_y[0] * inv;
            centroid[k * 2 + 1] = sm_x[0] * inv;
        } else {
            centroid[k * 2 + 0] = 0.0;
            centroid[k * 2 + 1] = 0.0;
        }
    }
}
)msl";

struct MomentsParams {
    uint32_t Lq, Lk, h_lat, w_lat;
};

id<MTLComputePipelineState> pso_moments() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_attn_token_moments"); });
    return pso;
}

} // namespace

void attention_token_moments_gpu(const GpuTensor& Attn,
                                 int h_lat, int w_lat,
                                 GpuTensor& mass,
                                 GpuTensor& centroid) {
    if (Attn.dtype != Dtype::FP16) {
        throw std::runtime_error("attention_token_moments_gpu: Attn must be FP16");
    }
    if (h_lat <= 0 || w_lat <= 0) {
        throw std::runtime_error("attention_token_moments_gpu: h_lat and w_lat must be positive");
    }
    const int Lq = h_lat * w_lat;
    const int Lk = Attn.cols;
    if (Attn.rows != Lq) {
        throw std::runtime_error("attention_token_moments_gpu: Attn.rows must equal h_lat * w_lat");
    }
    if (mass.rows != Lk || mass.cols != 1 || mass.dtype != Dtype::FP32) {
        mass.resize(Lk, 1, Dtype::FP32);
    }
    if (centroid.rows != Lk || centroid.cols != 2 || centroid.dtype != Dtype::FP32) {
        centroid.resize(Lk, 2, Dtype::FP32);
    }
    if (Lk == 0) return;

    id<MTLComputePipelineState> pso = pso_moments();
    id<MTLBuffer> bA = buffer_for(Attn);
    id<MTLBuffer> bM = buffer_for(mass);
    id<MTLBuffer> bC = buffer_for(centroid);
    const NSUInteger oA = buffer_offset_for(Attn);
    const NSUInteger oM = buffer_offset_for(mass);
    const NSUInteger oC = buffer_offset_for(centroid);

    MomentsParams p{};
    p.Lq    = static_cast<uint32_t>(Lq);
    p.Lk    = static_cast<uint32_t>(Lk);
    p.h_lat = static_cast<uint32_t>(h_lat);
    p.w_lat = static_cast<uint32_t>(w_lat);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bM offset:oM atIndex:1];
        [enc setBuffer:bC offset:oC atIndex:2];
        [enc setBytes:&p length:sizeof(MomentsParams) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(Lk, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MOM_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
