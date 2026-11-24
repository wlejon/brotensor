// Metal implementation of KV-cache append (MTLBlitCommandEncoder) + causal /
// masked / device-length flash-attention decode (MirKind::FaRows).
// Mirrors src/cuda/kv_cache.cu.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#include <cstddef>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::new_command_buffer;

void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache) {
    if ((K_new.dtype != Dtype::FP16 && K_new.dtype != Dtype::BF16 && K_new.dtype != Dtype::FP32) ||
        V_new.dtype != K_new.dtype ||
        K_cache.dtype != K_new.dtype || V_cache.dtype != K_new.dtype) {
        throw std::runtime_error("kv_cache_append_gpu: all tensors must share one float dtype");
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

    const std::size_t esz = dtype_size_bytes(K_new.dtype);
    const NSUInteger nbytes = static_cast<NSUInteger>(static_cast<std::size_t>(L_new) * D * esz);
    const NSUInteger dst_off = static_cast<NSUInteger>(static_cast<std::size_t>(cur_len) * D * esz);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:buffer_for(K_new)
                sourceOffset:buffer_offset_for(K_new)
                    toBuffer:buffer_for(K_cache)
           destinationOffset:buffer_offset_for(K_cache) + dst_off
                        size:nbytes];
        [blit copyFromBuffer:buffer_for(V_new)
                sourceOffset:buffer_offset_for(V_new)
                    toBuffer:buffer_for(V_cache)
           destinationOffset:buffer_offset_for(V_cache) + dst_off
                        size:nbytes];
        [blit endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

namespace {

void check_decode_heads(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
                        int num_q_heads, int num_kv_heads, const char* op) {
    if ((Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16 && Q.dtype != Dtype::FP32) ||
        K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error(std::string(op) + ": all tensors must share one float dtype");
    }
    if (V_cache.cols != K_cache.cols) {
        throw std::runtime_error(std::string(op) + ": K_cache.cols != V_cache.cols");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error(std::string(op) + ": num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error(std::string(op) + ": num_kv_heads must divide num_q_heads");
    }
    if (Q.cols % num_q_heads != 0 || K_cache.cols % num_kv_heads != 0) {
        throw std::runtime_error(std::string(op) + ": head_dim does not divide cols cleanly");
    }
    if (K_cache.cols / num_kv_heads != Q.cols / num_q_heads) {
        throw std::runtime_error(std::string(op) + ": head_dim mismatch between Q and K/V");
    }
}

}  // namespace

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_q_heads, int num_kv_heads,
                            Tensor& O, float attn_softcap, int window) {
    constexpr const char* op = "flash_attention_decode_gpu";
    if (valid_len < 0) throw std::runtime_error("flash_attention_decode_gpu: invalid valid_len");
    check_decode_heads(Q, K_cache, V_cache, num_q_heads, num_kv_heads, op);
    const int Lq = Q.rows, Dq = Q.cols, Dkv = K_cache.cols;
    if (valid_len > K_cache.rows || valid_len > V_cache.rows) {
        throw std::runtime_error("flash_attention_decode_gpu: invalid valid_len");
    }
    if (valid_len < Lq) {
        throw std::runtime_error("flash_attention_decode_gpu: valid_len must be >= Lq");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != Q.dtype) O.resize(Lq, Dq, Q.dtype);
    if (Lq == 0 || Dq == 0) return;
    if (valid_len == 0) { O.zero(); return; }

    FaRowsProblem p;
    p.op = op;
    p.dt = Q.dtype;
    p.q = Q.data; p.k = K_cache.data; p.v = V_cache.data; p.o = O.data;
    p.lq = Lq; p.lk = valid_len;
    p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
    p.hd = Dq / num_q_heads; p.hq = num_q_heads; p.hkv = num_kv_heads;
    p.mode = FA_MODE_ROWS;
    p.causal = true;
    p.window = window;
    p.q_offset = valid_len - Lq;
    p.softcap = attn_softcap;
    fa_rows_metal(p);
}

void flash_attention_decode_len(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
                                const int32_t* d_len, int num_q_heads, int num_kv_heads, Tensor& O) {
    constexpr const char* op = "flash_attention_decode_len";
    check_decode_heads(Q, K_cache, V_cache, num_q_heads, num_kv_heads, op);
    if (d_len == nullptr) throw std::runtime_error("flash_attention_decode_len: d_len must not be null");
    if (V_cache.rows != K_cache.rows) {
        throw std::runtime_error("flash_attention_decode_len: K_cache/V_cache row mismatch");
    }
    const int Lq = Q.rows, Dq = Q.cols, Dkv = K_cache.cols;
    if (Lq > K_cache.rows) {
        throw std::runtime_error("flash_attention_decode_len: more Q rows than the cache holds");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != Q.dtype) O.resize(Lq, Dq, Q.dtype);
    if (Lq == 0 || Dq == 0) return;
    if (K_cache.rows == 0) { O.zero(); return; }

    FaRowsProblem p;
    p.op = op;
    p.dt = Q.dtype;
    p.q = Q.data; p.k = K_cache.data; p.v = V_cache.data; p.o = O.data;
    p.aux = d_len;
    p.lq = Lq; p.lk = K_cache.rows;
    p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
    p.hd = Dq / num_q_heads; p.hq = num_q_heads; p.hkv = num_kv_heads;
    p.mode = FA_MODE_LEN;
    fa_rows_metal(p);
}

void flash_attention_decode_masked(const Tensor& Q,
                                   const Tensor& K_cache, const Tensor& V_cache,
                                   const float* d_mask,
                                   int num_q_heads, int num_kv_heads,
                                   Tensor& O, float attn_softcap, int window) {
    constexpr const char* op = "flash_attention_decode_masked";
    check_decode_heads(Q, K_cache, V_cache, num_q_heads, num_kv_heads, op);
    if (d_mask == nullptr) {
        throw std::runtime_error("flash_attention_decode_masked: d_mask must not be null");
    }
    if (Q.rows != 1) {
        throw std::runtime_error("flash_attention_decode_masked: Q must be a single row (L_q == 1)");
    }
    const int Dq = Q.cols, Dkv = K_cache.cols, cap = K_cache.rows;
    if (V_cache.rows != cap) {
        throw std::runtime_error("flash_attention_decode_masked: K_cache/V_cache row mismatch");
    }
    if (O.rows != 1 || O.cols != Dq || O.dtype != Q.dtype) O.resize(1, Dq, Q.dtype);
    if (Dq == 0) return;
    if (cap == 0) { O.zero(); return; }

    FaRowsProblem p;
    p.op = op;
    p.dt = Q.dtype;
    p.q = Q.data; p.k = K_cache.data; p.v = V_cache.data; p.o = O.data;
    p.mask = d_mask;
    p.lq = 1; p.lk = cap;
    p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
    p.hd = Dq / num_q_heads; p.hq = num_q_heads; p.hkv = num_kv_heads;
    p.mode = FA_MODE_ROWS;
    p.window = window;
    p.softcap = attn_softcap;
    p.maskpos = window > 0;
    fa_rows_metal(p);
}

}  // namespace brotensor::detail::metal
