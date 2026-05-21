// ─── Metal self-attention with additive pre-softmax bias ───────────────────
//
// Metal counterpart of src/cuda/self_attention_bias.cu — multi-head
// self-attention that adds an optional per-head (L, L) bias to the attention
// logits before softmax (the primitive behind T5 relative-position bias and
// ALiBi-style biases).
//
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Scores are materialised (L, L) per head — intended for encoder-length
// sequences, not long-context decoding. Dispatched on X.dtype (FP32 / FP16 /
// BF16): the projection inputs/outputs are typed, every intermediate
// (Q/K/V/scores/softmax) is FP32 scratch, math is FP32. attn_bias is FP32.

#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

constexpr NSUInteger kSabSoftmaxBlock = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

#define SAB_SM_BLOCK 256u

// Per-head projection: Out[(hh*L+i), j] = sum_k In[i,k] * W[hh*dh+j, k].
// In: (L, Din) typed, W: (D, Din) typed, Out: (H*L, dh) FP32.
#define SAB_PROJ_KERNEL(NAME, T)                                              \
kernel void NAME(device const T*     In  [[buffer(0)]],                       \
                 device const T*     W   [[buffer(1)]],                       \
                 device float*       Out [[buffer(2)]],                       \
                 constant uint& L   [[buffer(3)]],                            \
                 constant uint& Din [[buffer(4)]],                            \
                 constant uint& dh  [[buffer(5)]],                            \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint j = gid.x, i = gid.y, hh = gid.z;                                    \
    if (i >= L || j >= dh) return;                                            \
    device const T* xr = In + (ulong)i * Din;                                 \
    device const T* wr = W  + (ulong)(hh * dh + j) * Din;                     \
    float acc = 0.0f;                                                         \
    for (uint k = 0; k < Din; ++k) acc += float(xr[k]) * float(wr[k]);        \
    Out[((ulong)hh * L + i) * dh + j] = acc;                                  \
}

SAB_PROJ_KERNEL(k_sab_proj_fp32, float)
SAB_PROJ_KERNEL(k_sab_proj_fp16, half)
SAB_PROJ_KERNEL(k_sab_proj_bf16, bfloat)

// S[(hh*L+i), j] = scale * (Q_h[i] . K_h[j]) + bias[(hh*L+i), j].
kernel void k_sab_scores(device const float* Qh   [[buffer(0)]],
                         device const float* Kh   [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         constant uint& has_bias  [[buffer(3)]],
                         device float*       S    [[buffer(4)]],
                         constant uint& L     [[buffer(5)]],
                         constant uint& dh    [[buffer(6)]],
                         constant float& scale [[buffer(7)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x, i = gid.y, hh = gid.z;
    if (i >= L || j >= L) return;
    ulong qrow = ((ulong)hh * L + i) * dh;
    ulong krow = ((ulong)hh * L + j) * dh;
    float s = 0.0f;
    for (uint k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    ulong srow = ((ulong)hh * L + i) * L;
    s *= scale;
    if (has_bias) s += bias[srow + j];
    S[srow + j] = s;
}

// Per-row masked softmax over (H*L, L). One threadgroup per (head, query row).
kernel void k_sab_softmax(device const float* scores [[buffer(0)]],
                          device float*       Attn   [[buffer(1)]],
                          device const float* mask   [[buffer(2)]],
                          constant uint& has_mask    [[buffer(3)]],
                          constant uint& L           [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[SAB_SM_BLOCK];
    uint i_within = row % L;
    device const float* srow = scores + (ulong)row * L;
    device float*       arow = Attn   + (ulong)row * L;

    if (has_mask && mask[i_within] < 0.5f) {
        for (uint j = tid; j < L; j += tg_size) arow[j] = 0.0f;
        return;
    }
    float local_max = -1e30f;
    for (uint j = tid; j < L; j += tg_size) {
        if (has_mask && mask[j] < 0.5f) continue;
        float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = sdata[0];

    float local_sum = 0.0f;
    for (uint j = tid; j < L; j += tg_size) {
        if (has_mask && mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
        float e = exp(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = sdata[0];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint j = tid; j < L; j += tg_size) arow[j] *= inv;
}

// Yconcat[i, hh*dh+k] = sum_j Attn[(hh*L+i), j] * Vh[(hh*L+j), k].
kernel void k_sab_apply_v(device const float* Attn    [[buffer(0)]],
                          device const float* Vh      [[buffer(1)]],
                          device float*       Yconcat [[buffer(2)]],
                          constant uint& L  [[buffer(3)]],
                          constant uint& dh [[buffer(4)]],
                          constant uint& D  [[buffer(5)]],
                          uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x, i = gid.y, hh = gid.z;
    if (i >= L || k >= dh) return;
    ulong arow = ((ulong)hh * L + i) * L;
    float acc = 0.0f;
    for (uint j = 0; j < L; ++j) {
        ulong vrow = ((ulong)hh * L + j) * dh;
        acc += Attn[arow + j] * Vh[vrow + k];
    }
    Yconcat[(ulong)i * D + (hh * dh + k)] = acc;
}

// O[i, c] = mask[i] ? sum_k Yconcat[i,k] * Wo[c,k] : 0.
#define SAB_OUTPUT_KERNEL(NAME, T)                                            \
kernel void NAME(device const float* Y    [[buffer(0)]],                      \
                 device const T*     Wo   [[buffer(1)]],                      \
                 device const float* mask [[buffer(2)]],                      \
                 constant uint& has_mask  [[buffer(3)]],                      \
                 device T*           O    [[buffer(4)]],                      \
                 constant uint& L [[buffer(5)]],                              \
                 constant uint& D [[buffer(6)]],                              \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint c = gid.x, i = gid.y;                                                \
    if (i >= L || c >= D) return;                                             \
    if (has_mask && mask[i] < 0.5f) { O[(ulong)i * D + c] = T(0.0f); return; }\
    device const float* yr = Y  + (ulong)i * D;                               \
    device const T*     wr = Wo + (ulong)c * D;                               \
    float acc = 0.0f;                                                         \
    for (uint k = 0; k < D; ++k) acc += yr[k] * float(wr[k]);                 \
    O[(ulong)i * D + c] = T(acc);                                             \
}

SAB_OUTPUT_KERNEL(k_sab_output_fp32, float)
SAB_OUTPUT_KERNEL(k_sab_output_fp16, half)
SAB_OUTPUT_KERNEL(k_sab_output_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_proj_fp32,   @"k_sab_proj_fp32")
DEF_PSO(pso_proj_fp16,   @"k_sab_proj_fp16")
DEF_PSO(pso_proj_bf16,   @"k_sab_proj_bf16")
DEF_PSO(pso_scores,      @"k_sab_scores")
DEF_PSO(pso_softmax,     @"k_sab_softmax")
DEF_PSO(pso_apply_v,     @"k_sab_apply_v")
DEF_PSO(pso_output_fp32, @"k_sab_output_fp32")
DEF_PSO(pso_output_fp16, @"k_sab_output_fp16")
DEF_PSO(pso_output_bf16, @"k_sab_output_bf16")
#undef DEF_PSO

// Dispatch a 3-D thread grid (nx, ny, nz), threadgroup capped at 256 threads.
void run3d(id<MTLComputePipelineState> pso,
           NSUInteger nx, NSUInteger ny, NSUInteger nz,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tx = nx < 16 ? nx : 16;
        NSUInteger ty = ny < 16 ? ny : 16;
        [enc dispatchThreads:MTLSizeMake(nx, ny, nz)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// One threadgroup per row; `kSabSoftmaxBlock` threads each.
void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (rows == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kSabSoftmaxBlock, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq,
                                 const Tensor& Wk, const Tensor& Wv,
                                 const Tensor& Wo, const float* d_mask,
                                 const Tensor* attn_bias, int num_heads,
                                 float scale, Tensor& O) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("self_attention_bias_forward: X must be FP32, FP16, or BF16");
    }
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo dtype must match X");
    }
    const int L = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    if (H <= 0 || D % H != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const int dh = D / H;

    bool has_bias = false;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be FP32");
        }
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        has_bias = true;
    }
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    // FP32 scratch for every intermediate.
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t has_mask = (bM != nil) ? 1u : 0u;

    id<MTLBuffer> bB = has_bias ? buffer_for(*attn_bias) : nil;
    NSUInteger oB = has_bias ? buffer_offset_for(*attn_bias) : 0;
    id<MTLBuffer> bB_arg = bB ? bB : bX;
    NSUInteger oB_arg = bB ? oB : oX;
    const uint32_t has_bias_u = has_bias ? 1u : 0u;

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);

    id<MTLComputePipelineState> proj_pso =
        (X.dtype == Dtype::FP16) ? pso_proj_fp16()
      : (X.dtype == Dtype::BF16) ? pso_proj_bf16()
      : pso_proj_fp32();

    // Q / K / V per-head projections.
    auto proj = ^(id<MTLBuffer> bW, NSUInteger oW,
                  id<MTLBuffer> bOut, NSUInteger oOut) {
        run3d(proj_pso, dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bOut offset:oOut atIndex:2];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:5];
        });
    };
    proj(buffer_for(Wq), buffer_offset_for(Wq), bQh, oQh);
    proj(buffer_for(Wk), buffer_offset_for(Wk), bKh, oKh);
    proj(buffer_for(Wv), buffer_offset_for(Wv), bVh, oVh);

    // Scores: scale * (Q.K) + bias.
    run3d(pso_scores(), Lu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh   offset:oQh   atIndex:0];
        [enc setBuffer:bKh   offset:oKh   atIndex:1];
        [enc setBuffer:bB_arg offset:oB_arg atIndex:2];
        [enc setBytes:&has_bias_u length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bS    offset:oS    atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
    });

    // Row-wise masked softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS     offset:oS     atIndex:0];
        [enc setBuffer:bA     offset:oA     atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat.
    run3d(pso_apply_v(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // Output projection by Wo.
    id<MTLComputePipelineState> out_pso =
        (X.dtype == Dtype::FP16) ? pso_output_fp16()
      : (X.dtype == Dtype::BF16) ? pso_output_bf16()
      : pso_output_fp32();
    id<MTLBuffer> bWo = buffer_for(Wo);
    NSUInteger oWo = buffer_offset_for(Wo);
    run3d(out_pso, Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc    offset:oYc    atIndex:0];
        [enc setBuffer:bWo    offset:oWo    atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bO     offset:oO     atIndex:4];
        [enc setBytes:&Lu     length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:6];
    });
}

} // namespace brotensor::detail::metal
