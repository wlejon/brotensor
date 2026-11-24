// Metal flash_attention_windowed_forward and flash_attention_gqa_forward —
// routed onto MirKind::FaRows (and dense_attention_metal for large bidirectional GQA).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#include <stdexcept>

namespace brotensor::detail::metal {

void flash_attention_windowed_forward(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V,
                                      const float* d_mask,
                                      int num_heads,
                                      int window,
                                      Tensor& O,
                                      bool causal) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V must be FP16, BF16, or FP32");
    if (K.dtype != dt || V.dtype != dt)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V dtype must match");
    if (num_heads <= 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;
    const int Dkv = K.cols;
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
    if ((causal || window > 0) && Lk < Lq)
        throw std::runtime_error("flash_attention_windowed_forward: requires Lk >= Lq");
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Dq == 0) return;
    if (Lk == 0) { O.zero(); return; }

    FaRowsProblem p;
    p.op = "flash_attention_windowed_forward";
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
    p.mask = d_mask;
    p.lq = Lq; p.lk = Lk;
    p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
    p.hd = head_dim; p.hq = num_heads; p.hkv = n_kv;
    p.mode = FA_MODE_ROWS;
    p.window = window;
    p.q_offset = Lk - Lq;
    p.causal = causal;
    fa_rows_metal(p);
}

void flash_attention_gqa_forward(const Tensor& Q,
                                 const Tensor& K,
                                 const Tensor& V,
                                 const float* d_mask,
                                 int num_q_heads,
                                 int num_kv_heads,
                                 bool causal,
                                 Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        throw std::runtime_error("flash_attention_gqa_forward: Q, K, V must be FP16, BF16, or FP32");
    if (K.dtype != dt || V.dtype != dt)
        throw std::runtime_error("flash_attention_gqa_forward: Q, K, V dtype must match");
    if (num_q_heads <= 0 || num_kv_heads <= 0)
        throw std::runtime_error("flash_attention_gqa_forward: head counts must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;
    const int Dkv = K.cols;
    if (Dq % num_q_heads != 0)
        throw std::runtime_error("flash_attention_gqa_forward: num_q_heads must divide Q.cols");
    const int head_dim = Dq / num_q_heads;
    if (V.cols != Dkv || V.rows != Lk)
        throw std::runtime_error("flash_attention_gqa_forward: shape mismatch");
    if (Dkv != num_kv_heads * head_dim)
        throw std::runtime_error("flash_attention_gqa_forward: K/V width must be num_kv_heads*head_dim");
    if (num_q_heads % num_kv_heads != 0)
        throw std::runtime_error("flash_attention_gqa_forward: num_kv_heads must divide num_q_heads");
    if (Lk < Lq)
        throw std::runtime_error("flash_attention_gqa_forward: requires Lk >= Lq");
    if (causal && Lq != Lk)
        throw std::runtime_error("flash_attention_gqa_forward: causal requires Lq == Lk");
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Dq == 0) return;
    if (Lk == 0) { O.zero(); return; }

    if (!causal && Lq >= 64 && Lk >= 64) {
        DenseAttnProblem p;
        p.op = "flash_attention_gqa_forward";
        p.dev = Q.device;
        p.dt = dt;
        p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
        p.mask = d_mask;
        p.lq = Lq; p.lk = Lk;
        p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
        p.hd = head_dim; p.hq = num_q_heads; p.hkv = num_kv_heads;
        DenseAttnExtras x;
        x.auto_scale = true;
        x.mask_ge = false;
        dense_attention_metal(p, x);
        return;
    }

    FaRowsProblem p;
    p.op = "flash_attention_gqa_forward";
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
    p.mask = d_mask;
    p.lq = Lq; p.lk = Lk;
    p.ldq = Dq; p.ldk = Dkv; p.ldo = Dq;
    p.hd = head_dim; p.hq = num_q_heads; p.hkv = num_kv_heads;
    p.mode = FA_MODE_ROWS;
    p.q_offset = Lk - Lq;
    p.causal = causal;
    fa_rows_metal(p);
}

}  // namespace brotensor::detail::metal
