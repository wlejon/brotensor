// Metal implementation of KV-cache append + causal flash-attention decode.
// Mirrors src/cuda/kv_cache.cu.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger FAD_BLOCK = 128;
constexpr NSUInteger FAD_KTILE = 64;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint FAD_BLOCK = 128;
constant uint FAD_KTILE = 64;
constant uint MAX_HD_PER_THREAD = 8;

// Copy L_new * D halves from src into dst starting at dst_off_halves.
kernel void k_kv_append_copy(device const half* src    [[buffer(0)]],
                             device half*       dst    [[buffer(1)]],
                             constant uint& n          [[buffer(2)]],
                             constant uint& dst_off    [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[dst_off + i] = src[i];
}

kernel void k_kv_append_copy_bf16(device const bfloat* src [[buffer(0)]],
                                  device bfloat*       dst [[buffer(1)]],
                                  constant uint& n         [[buffer(2)]],
                                  constant uint& dst_off   [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[dst_off + i] = src[i];
}

// Causal flash-attention decode kernel. One threadgroup per (q, head).
kernel void k_flash_attention_decode(
        device const half* Q       [[buffer(0)]],   // (Lq, D)
        device const half* Kk      [[buffer(1)]],   // (L_max, D), valid_len valid
        device const half* V       [[buffer(2)]],   // (L_max, D)
        device half*       Out     [[buffer(3)]],   // (Lq, D)
        constant uint& Lq          [[buffer(4)]],
        constant uint& valid_len   [[buffer(5)]],
        constant uint& Dq          [[buffer(6)]],
        constant uint& head_dim    [[buffer(7)]],
        constant uint& seq_offset  [[buffer(8)]],
        constant uint& Dkv         [[buffer(9)]],
        constant uint& group       [[buffer(10)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* red    = scratch + FAD_KTILE;

    uint q = gid.x;
    uint h = gid.y;
    uint q_head_off  = h * head_dim;             // Q/Out (Dq-wide)
    uint kv_head_off = (h / group) * head_dim;   // K/V (Dkv-wide), GQA group
    float inv_sqrt = rsqrt(float(head_dim));
    uint p_q = seq_offset + q;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        uint klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1u > p_q) klen = p_q - k0 + 1u;

        // 1. scores
        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q[q * Dq + q_head_off + d]) *
                       float(Kk[kg * Dkv + kv_head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. tile max
        float local_max = -1e30f;
        for (uint t = tid; t < klen; t += tg_size) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = red[tid + s];
                if (other > red[tid]) red[tid] = other;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_max = red[0];
        float m_new = (tile_max > run_max) ? tile_max : run_max;
        bool tile_empty = (m_new <= -1e29f);

        // 3. exponentiate + sum
        for (uint t = tid; t < klen; t += tg_size) {
            float e = tile_empty ? 0.0f : exp(scores[t] - m_new);
            scores[t] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_sum = 0.0f;
        for (uint t = tid; t < klen; t += tg_size) local_sum += scores[t];
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_sum = red[0];

        // 4. rescale
        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        // 5. update partial output
        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * float(V[(k0 + t) * Dkv + kv_head_off + d]);
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * Dq + q_head_off + d] = half(partial[slot] * inv);
        ++slot;
    }
}

// BF16 variant — identical logic, all float accumulators.
kernel void k_flash_attention_decode_bf16(
        device const bfloat* Q       [[buffer(0)]],
        device const bfloat* Kk      [[buffer(1)]],
        device const bfloat* V       [[buffer(2)]],
        device bfloat*       Out     [[buffer(3)]],
        constant uint& Lq          [[buffer(4)]],
        constant uint& valid_len   [[buffer(5)]],
        constant uint& Dq          [[buffer(6)]],
        constant uint& head_dim    [[buffer(7)]],
        constant uint& seq_offset  [[buffer(8)]],
        constant uint& Dkv         [[buffer(9)]],
        constant uint& group       [[buffer(10)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* red    = scratch + FAD_KTILE;

    uint q = gid.x;
    uint h = gid.y;
    uint q_head_off  = h * head_dim;             // Q/Out (Dq-wide)
    uint kv_head_off = (h / group) * head_dim;   // K/V (Dkv-wide), GQA group
    float inv_sqrt = rsqrt(float(head_dim));
    uint p_q = seq_offset + q;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        uint klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1u > p_q) klen = p_q - k0 + 1u;

        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q[q * Dq + q_head_off + d]) *
                       float(Kk[kg * Dkv + kv_head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -1e30f;
        for (uint t = tid; t < klen; t += tg_size) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = red[tid + s];
                if (other > red[tid]) red[tid] = other;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_max = red[0];
        float m_new = (tile_max > run_max) ? tile_max : run_max;
        bool tile_empty = (m_new <= -1e29f);

        for (uint t = tid; t < klen; t += tg_size) {
            float e = tile_empty ? 0.0f : exp(scores[t] - m_new);
            scores[t] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_sum = 0.0f;
        for (uint t = tid; t < klen; t += tg_size) local_sum += scores[t];
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_sum = red[0];

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * float(V[(k0 + t) * Dkv + kv_head_off + d]);
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * Dq + q_head_off + d] = bfloat(partial[slot] * inv);
        ++slot;
    }
}
)msl";

id<MTLComputePipelineState> pso_append() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_kv_append_copy"); });
    return pso;
}
id<MTLComputePipelineState> pso_append_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_kv_append_copy_bf16"); });
    return pso;
}

id<MTLComputePipelineState> pso_decode() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_flash_attention_decode"); });
    return pso;
}
id<MTLComputePipelineState> pso_decode_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_flash_attention_decode_bf16"); });
    return pso;
}

void run_copy(id<MTLComputePipelineState> copy_pso,
              id<MTLBuffer> src_buf, NSUInteger src_off_bytes,
              id<MTLBuffer> dst_buf, NSUInteger dst_off_bytes,
              uint32_t n_halves, uint32_t dst_extra_halves) {
    // dst element index = dst_extra_halves + i; we already account for cur_len
    // by passing dst_extra_halves.
    id<MTLComputePipelineState> pso = copy_pso;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src_buf offset:src_off_bytes atIndex:0];
        [enc setBuffer:dst_buf offset:dst_off_bytes atIndex:1];
        [enc setBytes:&n_halves length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dst_extra_halves length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n_halves, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache) {
    const bool is_bf16 = (K_new.dtype == Dtype::BF16);
    if ((K_new.dtype != Dtype::FP16 && K_new.dtype != Dtype::BF16) ||
        V_new.dtype != K_new.dtype ||
        K_cache.dtype != K_new.dtype || V_cache.dtype != K_new.dtype) {
        throw std::runtime_error("kv_cache_append_gpu: all tensors must be FP16 or all BF16");
    }
    if (K_new.cols != V_new.cols || K_new.cols != K_cache.cols ||
        K_cache.cols != V_cache.cols) {
        throw std::runtime_error("kv_cache_append_gpu: column mismatch");
    }
    if (K_new.rows != V_new.rows) {
        throw std::runtime_error("kv_cache_append_gpu: K_new/V_new row mismatch");
    }
    if (K_cache.rows != V_cache.rows) {
        throw std::runtime_error("kv_cache_append_gpu: K_cache/V_cache row mismatch");
    }
    const int L_new = K_new.rows;
    const int L_max = K_cache.rows;
    const int D     = K_new.cols;
    if (cur_len < 0 || cur_len + L_new > L_max) {
        throw std::runtime_error("kv_cache_append_gpu: cur_len + L_new exceeds cache capacity");
    }
    if (L_new == 0 || D == 0) return;

    const uint32_t n_halves = static_cast<uint32_t>(L_new) * static_cast<uint32_t>(D);
    const uint32_t dst_extra_halves = static_cast<uint32_t>(cur_len) * static_cast<uint32_t>(D);

    id<MTLBuffer> bK_new = buffer_for(K_new);
    id<MTLBuffer> bV_new = buffer_for(V_new);
    id<MTLBuffer> bK_c   = buffer_for(K_cache);
    id<MTLBuffer> bV_c   = buffer_for(V_cache);
    const NSUInteger oK_new = buffer_offset_for(K_new);
    const NSUInteger oV_new = buffer_offset_for(V_new);
    const NSUInteger oK_c   = buffer_offset_for(K_cache);
    const NSUInteger oV_c   = buffer_offset_for(V_cache);

    id<MTLComputePipelineState> copy_pso = is_bf16 ? pso_append_bf16() : pso_append();
    run_copy(copy_pso, bK_new, oK_new, bK_c, oK_c, n_halves, dst_extra_halves);
    run_copy(copy_pso, bV_new, oV_new, bV_c, oV_c, n_halves, dst_extra_halves);
}

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_q_heads, int num_kv_heads,
                            Tensor& O) {
    const bool is_bf16 = (Q.dtype == Dtype::BF16);
    if ((Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) ||
        K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode_gpu: all tensors must be FP16 or all BF16");
    }
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;            // num_q_heads  * head_dim
    const int Dkv = K_cache.cols;      // num_kv_heads * head_dim
    if (V_cache.cols != Dkv) {
        throw std::runtime_error("flash_attention_decode_gpu: K_cache.cols != V_cache.cols");
    }
    if (valid_len < 0 || valid_len > K_cache.rows || valid_len > V_cache.rows) {
        throw std::runtime_error("flash_attention_decode_gpu: invalid valid_len");
    }
    if (valid_len < Lq) {
        throw std::runtime_error("flash_attention_decode_gpu: valid_len must be >= Lq");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention_decode_gpu: num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_gpu: num_kv_heads must divide num_q_heads");
    }
    if (Dq % num_q_heads != 0 || Dkv % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_gpu: head_dim does not divide cols cleanly");
    }
    const int head_dim = Dq / num_q_heads;
    if (Dkv / num_kv_heads != head_dim) {
        throw std::runtime_error("flash_attention_decode_gpu: head_dim mismatch between Q and K/V");
    }
    const int group = num_q_heads / num_kv_heads;   // q heads served per kv head
    if ((head_dim + static_cast<int>(FAD_BLOCK) - 1) / static_cast<int>(FAD_BLOCK) > 8) {
        throw std::runtime_error("flash_attention_decode_gpu: head_dim too large (max 8 * FAD_BLOCK = 1024)");
    }
    Dtype out_dtype = is_bf16 ? Dtype::BF16 : Dtype::FP16;
    if (O.rows != Lq || O.cols != Dq || O.dtype != out_dtype) {
        O.resize(Lq, Dq, out_dtype);
    }
    if (Lq == 0 || Dq == 0 || valid_len == 0) return;

    const uint32_t seq_offset = static_cast<uint32_t>(valid_len - Lq);
    const uint32_t uLq = static_cast<uint32_t>(Lq);
    const uint32_t uValid = static_cast<uint32_t>(valid_len);
    const uint32_t uDq = static_cast<uint32_t>(Dq);
    const uint32_t uDkv = static_cast<uint32_t>(Dkv);
    const uint32_t uHd = static_cast<uint32_t>(head_dim);
    const uint32_t uGroup = static_cast<uint32_t>(group);

    id<MTLComputePipelineState> pso = is_bf16 ? pso_decode_bf16() : pso_decode();
    id<MTLBuffer> bQ = buffer_for(Q);
    id<MTLBuffer> bK = buffer_for(K_cache);
    id<MTLBuffer> bV = buffer_for(V_cache);
    id<MTLBuffer> bO = buffer_for(O);
    const NSUInteger oQ = buffer_offset_for(Q);
    const NSUInteger oK = buffer_offset_for(K_cache);
    const NSUInteger oV = buffer_offset_for(V_cache);
    const NSUInteger oO = buffer_offset_for(O);

    const NSUInteger shmem_bytes = (FAD_KTILE + FAD_BLOCK) * sizeof(float);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:oQ atIndex:0];
        [enc setBuffer:bK offset:oK atIndex:1];
        [enc setBuffer:bV offset:oV atIndex:2];
        [enc setBuffer:bO offset:oO atIndex:3];
        [enc setBytes:&uLq      length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&uValid   length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&uDq      length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&uHd      length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&seq_offset length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&uDkv     length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&uGroup   length:sizeof(uint32_t) atIndex:10];
        [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(Lq),
                                              static_cast<NSUInteger>(num_q_heads), 1)
            threadsPerThreadgroup:MTLSizeMake(FAD_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
