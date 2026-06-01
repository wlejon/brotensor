// ─── Metal flash_attention_windowed_forward ─────────────────────────────────
//
// Metal counterpart of the CUDA flash_attention_windowed_kernel
// (src/cuda/flash_attention.cu). Sliding-window causal self-attention with GQA.
// The Lq queries occupy the last Lq positions of a length-Lk causal sequence
// (q_offset = Lk - Lq): query row r is at absolute position aq = r + q_offset
// and attends keys [lo, aq] with lo = max(0, aq-window+1) (window <= 0 => plain
// causal). One threadgroup per (query, head); online softmax tiled along Lk in
// FA_KTILE chunks, per-thread output partials in registers. d_mask is an
// optional length-Lk key mask (1 valid / 0 invalid). FP32 / FP16 / BF16, matching
// the CUDA contract; math runs in float, only load/store differ by dtype.

#include <brotensor/runtime.h>

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

constexpr NSUInteger FAW_BLOCK = 128;
constexpr NSUInteger FAW_KTILE = 64;

// Parameter block — must match the MSL struct below.
struct WinParams {
    uint32_t Lk, Dq, Dkv, head_dim;
    int32_t  window;
    uint32_t q_offset, group, has_mask;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint FAW_BLOCK = 128;
constant uint FAW_KTILE = 64;
constant uint MAX_HD_PER_THREAD = 8;

struct WinParams {
    uint Lk, Dq, Dkv, head_dim;
    int  window;
    uint q_offset, group, has_mask;
};

// Online-softmax core. One threadgroup per (q, head). scores[] holds one
// FAW_KTILE tile of scores; red[] is reduction scratch (FAW_BLOCK wide).
template <typename T>
static inline void flash_windowed_core(device const T* Q,
                                       device const T* K,
                                       device const T* V,
                                       device const float* mask,
                                       device T* Out,
                                       constant WinParams& p,
                                       threadgroup float* scores,
                                       threadgroup float* red,
                                       uint q, uint h, uint tid, uint nthreads) {
    const uint head_off    = h * p.head_dim;                 // Q/Out (Dq-wide)
    const uint head_off_kv = (h / p.group) * p.head_dim;     // K/V (Dkv-wide), GQA
    const float inv_sqrt   = rsqrt(float(p.head_dim));

    const int aq = int(q) + int(p.q_offset);                 // absolute causal pos
    int lo = (p.window > 0) ? (aq - p.window + 1) : 0;
    if (lo < 0) lo = 0;
    const int k_hi = aq;                                      // inclusive upper bound

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    const int k0_start = (lo / int(FAW_KTILE)) * int(FAW_KTILE);
    for (int k0 = k0_start; k0 <= k_hi; k0 += int(FAW_KTILE)) {
        int klen = int(FAW_KTILE);
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;      // causal trim

        // 1. Scores (below window or masked out -> -inf).
        for (uint t = tid; t < uint(klen); t += nthreads) {
            const int kg = k0 + int(t);
            float s;
            if (kg < lo || (p.has_mask != 0u && mask[kg] <= 0.5f)) {
                s = -1e30f;
            } else {
                float dot = 0.0f;
                for (uint d = 0; d < p.head_dim; ++d) {
                    dot += float(Q[q * p.Dq + head_off + d]) *
                           float(K[uint(kg) * p.Dkv + head_off_kv + d]);
                }
                s = dot * inv_sqrt;
            }
            scores[t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Tile max.
        float local_max = -1e30f;
        for (uint t = tid; t < uint(klen); t += nthreads)
            if (scores[t] > local_max) local_max = scores[t];
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;

        // 3. Exponentiate, sum.
        const bool tile_empty = (m_new <= -1e29f);
        for (uint t = tid; t < uint(klen); t += nthreads) {
            const float e = tile_empty ? 0.0f : exp(scores[t] - m_new);
            scores[t] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_sum = 0.0f;
        for (uint t = tid; t < uint(klen); t += nthreads) local_sum += scores[t];
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = red[0];

        // 4. Rescale factor for the running accumulators.
        const float alpha = (run_max <= -1e29f) ? 0.0f : exp(run_max - m_new);

        // 5. Update partial output.
        uint slot = 0;
        for (uint d = tid; d < p.head_dim; d += nthreads, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       float(V[uint(k0 + t) * p.Dkv + head_off_kv + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < p.head_dim; d += nthreads, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * p.Dq + head_off + d] = T(partial[slot] * inv);
    }
}

#define FAW_KERNEL(NAME, T)                                                   \
kernel void NAME(device const T* Q       [[buffer(0)]],                       \
                 device const T* K       [[buffer(1)]],                       \
                 device const T* V       [[buffer(2)]],                       \
                 device const float* mask[[buffer(3)]],                       \
                 device T* Out           [[buffer(4)]],                       \
                 constant WinParams& p   [[buffer(5)]],                       \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    threadgroup float scores[FAW_KTILE];                                      \
    threadgroup float red[FAW_BLOCK];                                         \
    flash_windowed_core<T>(Q, K, V, mask, Out, p, scores, red,               \
                           gid.x, gid.y, tid3.x, tgs3.x);                     \
}

FAW_KERNEL(k_flash_windowed_fp32, float)
FAW_KERNEL(k_flash_windowed_fp16, half)
FAW_KERNEL(k_flash_windowed_bf16, bfloat)
#undef FAW_KERNEL
)msl";

id<MTLComputePipelineState> pso_fp32() {
    static dispatch_once_t once; static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_flash_windowed_fp32"); });
    return p;
}
id<MTLComputePipelineState> pso_fp16() {
    static dispatch_once_t once; static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_flash_windowed_fp16"); });
    return p;
}
id<MTLComputePipelineState> pso_bf16() {
    static dispatch_once_t once; static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_flash_windowed_bf16"); });
    return p;
}

} // namespace

void flash_attention_windowed_forward(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V,
                                      const float* d_mask,
                                      int num_heads,
                                      int window,
                                      Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V must be FP16, BF16, or FP32");
    if (K.dtype != dt || V.dtype != dt)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V dtype must match");
    if (num_heads <= 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;        // num_heads * head_dim
    const int Dkv = K.cols;        // n_kv * head_dim (GQA when < Dq)
    if (Dq % num_heads != 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must divide D");
    const int head_dim = Dq / num_heads;
    if (V.cols != Dkv || V.rows != Lk)
        throw std::runtime_error("flash_attention_windowed_forward: shape mismatch");
    if (Dkv == 0 || Dkv % head_dim != 0)
        throw std::runtime_error("flash_attention_windowed_forward: K/V width must be a head_dim multiple");
    const int n_kv = Dkv / head_dim;
    if (num_heads % n_kv != 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must be a multiple of n_kv");
    if (Lk < Lq)
        throw std::runtime_error("flash_attention_windowed_forward: requires Lk >= Lq");
    if ((head_dim + (int)FAW_BLOCK - 1) / (int)FAW_BLOCK > 8)
        throw std::runtime_error("flash_attention_windowed_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    WinParams p{};
    p.Lk       = static_cast<uint32_t>(Lk);
    p.Dq       = static_cast<uint32_t>(Dq);
    p.Dkv      = static_cast<uint32_t>(Dkv);
    p.head_dim = static_cast<uint32_t>(head_dim);
    p.window   = window;
    p.q_offset = static_cast<uint32_t>(Lk - Lq);
    p.group    = static_cast<uint32_t>(num_heads / n_kv);
    p.has_mask = d_mask ? 1u : 0u;

    id<MTLComputePipelineState> pso =
        (dt == Dtype::BF16) ? pso_bf16()
      : (dt == Dtype::FP32) ? pso_fp32()
                            : pso_fp16();

    id<MTLBuffer> bQ = buffer_for(Q);   NSUInteger oQ = buffer_offset_for(Q);
    id<MTLBuffer> bK = buffer_for(K);   NSUInteger oK = buffer_offset_for(K);
    id<MTLBuffer> bV = buffer_for(V);   NSUInteger oV = buffer_offset_for(V);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bQ;     // dummy bind when no mask
    NSUInteger oM_arg = bM ? oM : oQ;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:oQ atIndex:0];
        [enc setBuffer:bK offset:oK atIndex:1];
        [enc setBuffer:bV offset:oV atIndex:2];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:3];
        [enc setBuffer:bO offset:oO atIndex:4];
        [enc setBytes:&p length:sizeof(WinParams) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(Lq, num_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(FAW_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
