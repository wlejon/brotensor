// Metal loss ops:
//   softmax_xent / softmax_xent_fused / softmax_xent_segment /
//   softmax_xent_fused_batched / bce_with_logits_fused_batched
//       -> single-source brass MIR kernel (src/kernels/mir_xent.cpp)
//   mse_vec_forward / mse_vec_backward / mse_vec_per_sample / mse_scalar
//       -> small MSE kernels (Wave 6+ misc)

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_require;

namespace {

constexpr NSUInteger LOSS_BLOCK = 256;

NSString* const kMseSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LOSS_BLOCK = 256;

kernel void k_mse_forward(device const float* pred    [[buffer(0)]],
                          device const float* target  [[buffer(1)]],
                          device float*       out_sum [[buffer(2)]],
                          constant uint& n            [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LOSS_BLOCK];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = pred[i] - target[i];
        local += d * d;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_sum[0] = sdata[0];
}

kernel void k_mse_backward(device const float* pred   [[buffer(0)]],
                           device const float* target [[buffer(1)]],
                           device float*       dPred  [[buffer(2)]],
                           constant uint& n           [[buffer(3)]],
                           constant float& scale      [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dPred[i] = scale * (pred[i] - target[i]);
}

kernel void k_mse_per_sample(device const float* pred   [[buffer(0)]],
                             device const float* target [[buffer(1)]],
                             device float*       dPred  [[buffer(2)]],
                             device float*       loss   [[buffer(3)]],
                             constant uint& B           [[buffer(4)]],
                             uint b [[thread_position_in_grid]]) {
    if (b >= B) return;
    float d = pred[b] - target[b];
    dPred[b] = d;
    loss[b] = 0.5f * d * d;
}
)msl";

#define DEF_PSO(NAME, FN)                                                 \
    id<MTLComputePipelineState> NAME() {                                  \
        static dispatch_once_t once;                                      \
        static id<MTLComputePipelineState> pso;                           \
        dispatch_once(&once, ^{ pso = compile_pipeline(kMseSrc, FN); });  \
        return pso;                                                       \
    }
DEF_PSO(pso_mse_fw, @"k_mse_forward")
DEF_PSO(pso_mse_bw, @"k_mse_backward")
DEF_PSO(pso_mse_per_sample, @"k_mse_per_sample")
#undef DEF_PSO

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void need_f32(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(what) + " must be FP32");
}

void ensure_f32_like(const Tensor& like, Tensor& t) {
    if (t.rows != like.rows || t.cols != like.cols || t.dtype != Dtype::FP32) {
        t.resize(like.rows, like.cols, Dtype::FP32);
    }
}

void require_ptr(const void* p, const char* op, const char* name) {
    if (!p) return;
    NSUInteger ofs = 0;
    pool_require(p, ofs, op, name);
}

void run_xent_mir(const char* op, const void* logits, const void* target,
                  const float* mask, const int* off, int rows, int n, int heads,
                  void* probs, void* dlog, void* loss, int mode = 0,
                  float pos_weight = 1.0f) {
    const std::uint8_t flags =
        static_cast<std::uint8_t>((mask ? 1u : 0u) | (off ? 2u : 0u));
    const MirKernel mk = mir_kernel(
        {MirKind::Xent, static_cast<std::uint32_t>(mode), Dtype::FP32,
         Dtype::FP32, 1, Dtype::FP32, flags});
    mir_dispatch_xent(mk, static_cast<std::uint32_t>(std::min(rows, 65535)), op,
                      logits, target, mask, off, probs, dlog, loss,
                      static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(n),
                      static_cast<std::uint32_t>(heads), pos_weight);
}

float run_xent_single(const char* op, const Tensor& logits, const Tensor& target,
                      const float* mask, Tensor& probs, Tensor& dlog) {
    need_f32(logits, op, "logits");
    need_f32(target, op, "target");
    if (target.size() != logits.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits, probs);
    ensure_f32_like(logits, dlog);
    const long long n = logits.size();
    if (n == 0) return 0.0f;
    if (n > 0x7fffffffLL) fail(op, "too large");
    require_ptr(mask, op, "mask");
    Tensor loss = Tensor::empty_on(logits.device, 1, 1, Dtype::FP32);
    run_xent_mir(op, logits.data, target.data, mask, nullptr, 1,
                 static_cast<int>(n), 1, probs.data, dlog.data, loss.data);
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

}  // namespace

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    const int n = pred.size();
    if (n == 0) return 0.0f;
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:sizeof(float)
                        options:MTLResourceStorageModeShared];
        float* sptr = static_cast<float*>([scratch contents]);
        sptr[0] = 0.0f;
        id<MTLComputePipelineState> pso = pso_mse_fw();
        id<MTLBuffer> bp = buffer_for(pred);
        NSUInteger op = buffer_offset_for(pred);
        id<MTLBuffer> bt = buffer_for(target);
        NSUInteger ot = buffer_offset_for(target);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:scratch offset:0 atIndex:2];
        const uint32_t nu = static_cast<uint32_t>(n);
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LOSS_BLOCK, 1, 1)];
        [enc endEncoding];
        metal_impl::flush();
        return sptr[0] / static_cast<float>(n);
    }
}

void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    const int n = pred.size();
    ensure_f32_like(pred, dPred);
    if (n == 0) return;
    const float scale = 2.0f / static_cast<float>(n);
    const uint32_t nu = static_cast<uint32_t>(n);
    id<MTLComputePipelineState> pso = pso_mse_bw();
    id<MTLBuffer> bp = buffer_for(pred);
    NSUInteger op = buffer_offset_for(pred);
    id<MTLBuffer> bt = buffer_for(target);
    NSUInteger ot = buffer_offset_for(target);
    id<MTLBuffer> bdp = buffer_for(dPred);
    NSUInteger odp = buffer_offset_for(dPred);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:bdp offset:odp atIndex:2];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&scale length:sizeof(float) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample) {
    const int B = pred.size();
    ensure_f32_like(pred, dPred);
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(B, 1, Dtype::FP32);
    }
    if (B == 0) return;
    id<MTLComputePipelineState> pso = pso_mse_per_sample();
    id<MTLBuffer> bp = buffer_for(pred);
    NSUInteger op = buffer_offset_for(pred);
    id<MTLBuffer> bt = buffer_for(target);
    NSUInteger ot = buffer_offset_for(target);
    id<MTLBuffer> bdp = buffer_for(dPred);
    NSUInteger odp = buffer_offset_for(dPred);
    id<MTLBuffer> bL = buffer_for(loss_per_sample);
    NSUInteger oL = buffer_offset_for(loss_per_sample);
    const uint32_t Bu = static_cast<uint32_t>(B);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:bdp offset:odp atIndex:2];
        [enc setBuffer:bL offset:oL atIndex:3];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask, Tensor& probs, Tensor& dLogits) {
    return run_xent_single("softmax_xent_fused", logits, target, d_mask, probs, dLogits);
}

float softmax_xent(const Tensor& logits, const Tensor& target, Tensor& probs,
                   Tensor& dLogits, const float* mask) {
    return run_xent_single("softmax_xent", logits, target, mask, probs, dLogits);
}

float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits, int n,
                           const float* mask) {
    constexpr const char* op = "softmax_xent_segment";
    if (n <= 0) return 0.0f;
    require_ptr(logits, op, "logits");
    require_ptr(target, op, "target");
    require_ptr(probs, op, "probs");
    require_ptr(dLogits, op, "dLogits");
    require_ptr(mask, op, "mask");
    Tensor loss = Tensor::empty_on(Device::Metal, 1, 1, Dtype::FP32);
    run_xent_mir(op, logits, target, mask, nullptr, 1, n, 1, probs, dLogits, loss.data);
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets, int n_heads,
                                Tensor& probs_BL, Tensor& dLogits_BL,
                                Tensor& loss_per_sample) {
    constexpr const char* op = "softmax_xent_fused_batched";
    need_f32(logits_BL, op, "logits");
    need_f32(target_BL, op, "target");
    if (target_BL.size() != logits_BL.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits_BL, probs_BL);
    ensure_f32_like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0 || n_heads <= 0) return;
    if (!d_head_offsets) fail(op, "d_head_offsets is null");
    require_ptr(d_mask_BL, op, "mask");
    require_ptr(d_head_offsets, op, "head_offsets");
    run_xent_mir(op, logits_BL.data, target_BL.data, d_mask_BL, d_head_offsets,
                 logits_BL.rows, logits_BL.cols, n_heads,
                 probs_BL.data, dLogits_BL.data, loss_per_sample.data);
}

void bce_with_logits_fused_batched(const Tensor& logits_BL,
                                   const Tensor& target_BL,
                                   const float* d_mask_BL, float pos_weight,
                                   Tensor& probs_BL, Tensor& dLogits_BL,
                                   Tensor& loss_per_sample) {
    constexpr const char* op = "bce_with_logits_fused_batched";
    need_f32(logits_BL, op, "logits");
    need_f32(target_BL, op, "target");
    if (target_BL.size() != logits_BL.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits_BL, probs_BL);
    ensure_f32_like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0) return;
    require_ptr(d_mask_BL, op, "mask");
    run_xent_mir(op, logits_BL.data, target_BL.data, d_mask_BL, nullptr,
                 logits_BL.rows, logits_BL.cols, 1,
                 probs_BL.data, dLogits_BL.data, loss_per_sample.data,
                 /*mode=*/1, pos_weight);
}

}  // namespace brotensor::detail::metal
