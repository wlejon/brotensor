// brotensor public op wrappers — attention, matmul, LSTM, RoPE, RMSNorm, SwiGLU, reductions.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor {

// ─── FP16 linear + GEGLU ───────────────────────────────────────────────────

void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W, X_BD, {bias, &Y_BD});
    if (!v.linear_forward_batched_fp16) detail::throw_not_implemented("linear_forward_batched_fp16", W.device);
    detail::adopt_output(Y_BD, W.device);
    v.linear_forward_batched_fp16(W, bias, X_BD, Y_BD);
}

void linear_forward_batched_fp16_act(const Tensor& W, const Tensor* bias,
                                     const Tensor& X_BD, int act, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W, X_BD, {bias, &Y_BD});
    if (!v.linear_forward_batched_fp16_act)
        detail::throw_not_implemented("linear_forward_batched_fp16_act", W.device);
    detail::adopt_output(Y_BD, W.device);
    v.linear_forward_batched_fp16_act(W, bias, X_BD, act, Y_BD);
}

void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X_BD,
                               int act, int epilogue, Tensor* workspace, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W, X_BD, {bias, &Y_BD, workspace});
    if (!v.linear_forward_batched_ex)
        detail::throw_not_implemented("linear_forward_batched_ex", W.device);
    detail::adopt_output(Y_BD, W.device);
    if (workspace) detail::adopt_output(*workspace, W.device);
    v.linear_forward_batched_ex(W, bias, X_BD, act, epilogue, workspace, Y_BD);
}

void geglu_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.geglu_forward) detail::throw_not_implemented("geglu_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.geglu_forward(X, Y);
}
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.geglu_backward) detail::throw_not_implemented("geglu_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.geglu_backward(X, dY, dX);
}
void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.geglu_exact_forward) detail::throw_not_implemented("geglu_exact_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.geglu_exact_forward(X, Y);
}
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.geglu_exact_backward) detail::throw_not_implemented("geglu_exact_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.geglu_exact_backward(X, dY, dX);
}

// ─── Causal mask helper ────────────────────────────────────────────────────

void build_causal_mask_row(int L, int q, Tensor& mask) {
    const auto& v = detail::dispatch(mask);
    if (!v.build_causal_mask_row) detail::throw_not_implemented("build_causal_mask_row", mask.device);
    detail::adopt_output(mask, mask.device);
    v.build_causal_mask_row(L, q, mask);
}

// ─── Cross-attention family ────────────────────────────────────────────────

void cross_attention_forward(const Tensor& X, const Tensor& Ctx,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads, Tensor& O) {
    const auto& v = detail::dispatch(X, Ctx, Wq, Wk, Wv, Wo, O);
    if (!v.cross_attention_forward) detail::throw_not_implemented("cross_attention_forward", X.device);
    detail::adopt_output(O, X.device);
    v.cross_attention_forward(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads, O);
}

void cross_attention_forward_with_attn(const Tensor& X, const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask,
                                       const Tensor* attn_logit_bias,
                                       int num_heads, Tensor& O, Tensor& AttnAvg) {
    const auto& v = detail::dispatch_with_opts(X, Ctx, {&Wq, &Wk, &Wv, &Wo, attn_logit_bias, &O, &AttnAvg});
    if (!v.cross_attention_forward_with_attn)
        detail::throw_not_implemented("cross_attention_forward_with_attn", X.device);
    detail::adopt_output(O, X.device);
    detail::adopt_output(AttnAvg, X.device);
    v.cross_attention_forward_with_attn(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                        attn_logit_bias, num_heads, O, AttnAvg);
}

void self_attention_forward_train(const Tensor& X,
                                  const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask, int num_heads,
                                  Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, Qh, Kh, Vh);
    if (!v.self_attention_forward_train)
        detail::throw_not_implemented("self_attention_forward_train", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_forward_train(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                   Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_backward(const Tensor& dO, const Tensor& X,
                             const Tensor& Qh, const Tensor& Kh,
                             const Tensor& Vh, const Tensor& Attnh,
                             const Tensor& Yconcat,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads,
                             Tensor& dX,
                             Tensor& dWq, Tensor& dWk,
                             Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch(dO, X, Qh, Kh, Vh, Attnh, Yconcat);
    if (!v.self_attention_backward)
        detail::throw_not_implemented("self_attention_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.self_attention_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                              Wq, Wk, Wv, Wo, d_mask, num_heads,
                              dX, dWq, dWk, dWv, dWo);
}

void attention_token_moments(const Tensor& Attn, int h_lat, int w_lat,
                             Tensor& mass, Tensor& centroid) {
    if (h_lat <= 0 || w_lat <= 0 || Attn.rows != h_lat * w_lat) {
        detail::throw_spatial_2d_error("attention_token_moments", "Attn",
                                       h_lat * w_lat, Attn.cols, h_lat, w_lat,
                                       Attn.rows, Attn.cols);
    }
    const auto& v = detail::dispatch(Attn, mass, centroid);
    if (!v.attention_token_moments) detail::throw_not_implemented("attention_token_moments", Attn.device);
    detail::adopt_output(mass, Attn.device);
    detail::adopt_output(centroid, Attn.device);
    v.attention_token_moments(Attn, h_lat, w_lat, mass, centroid);
}

void cross_attention_forward_train(const Tensor& X, const Tensor& Ctx,
                                   const Tensor& Wq, const Tensor& Wk,
                                   const Tensor& Wv, const Tensor& Wo,
                                   const float* d_mask, int num_heads,
                                   Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                   Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch(X, Ctx, Wq, Wk, Wv, Wo, Qh, Kh);
    if (!v.cross_attention_forward_train)
        detail::throw_not_implemented("cross_attention_forward_train", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.cross_attention_forward_train(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                    Qh, Kh, Vh, Attnh, Yconcat, O);
}

void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh,
                              const Tensor& Vh, const Tensor& Attnh,
                              const Tensor& Yconcat,
                              const Tensor& Wq, const Tensor& Wk,
                              const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads,
                              Tensor& dX, Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk,
                              Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch(dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat);
    if (!v.cross_attention_backward)
        detail::throw_not_implemented("cross_attention_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dCtx, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.cross_attention_backward(dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat,
                               Wq, Wk, Wv, Wo, d_mask, num_heads,
                               dX, dCtx, dWq, dWk, dWv, dWo);
}

// ─── FP16 LN inference + FP16 self-attention ───────────────────────────────

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD, float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, beta, Y_RD);
    if (!v.layernorm_forward_inference_batched_fp16)
        detail::throw_not_implemented("layernorm_forward_inference_batched_fp16", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    v.layernorm_forward_inference_batched_fp16(X_RD, gamma, &beta, Y_RD, eps);
}

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              Tensor& Y_RD, float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, Y_RD);
    if (!v.layernorm_forward_inference_batched_fp16)
        detail::throw_not_implemented("layernorm_forward_inference_batched_fp16", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    v.layernorm_forward_inference_batched_fp16(X_RD, gamma, nullptr, Y_RD, eps);
}

void self_attention_forward(const Tensor& X,
                            const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask, int num_heads, Tensor& O) {
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, O);
    if (!v.self_attention_forward) detail::throw_not_implemented("self_attention_forward", X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads, O);
}

void self_attention_bias_forward(const Tensor& X,
                                 const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo,
                                 const Tensor* bq, const Tensor* bk,
                                 const Tensor* bv, const Tensor* bo,
                                 const float* d_mask,
                                 const Tensor* attn_bias,
                                 int num_heads, float scale, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(X, Wq, {&Wk, &Wv, &Wo, bq, bk, bv, bo, attn_bias, &O});
    if (!v.self_attention_bias_forward)
        detail::throw_not_implemented("self_attention_bias_forward", X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_bias_forward(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, d_mask, attn_bias,
                                  num_heads, scale, O);
}

void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk,
                             int num_heads, int head_dim, Tensor& Bias) {
    const auto& v = detail::dispatch_with_opts(Qv, Pk, {&Bias});
    if (!v.rel_pos_bias_xl_forward)
        detail::throw_not_implemented("rel_pos_bias_xl_forward", Qv.device);
    detail::adopt_output(Bias, Qv.device);
    v.rel_pos_bias_xl_forward(Qv, Pk, num_heads, head_dim, Bias);
}

void self_attention_decomposed_rel_pos_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale, Tensor& O) {
    if (grid_h <= 0 || grid_w <= 0 || X.rows != grid_h * grid_w) {
        detail::throw_spatial_2d_error("self_attention_decomposed_rel_pos_forward", "X",
                                       grid_h * grid_w, X.cols, grid_h, grid_w,
                                       X.rows, X.cols);
    }
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {bq, &Wk, bk, &Wv, bv, &Wo, bo, &rel_pos_h, &rel_pos_w, &O});
    if (!v.self_attention_decomposed_rel_pos_forward)
        detail::throw_not_implemented("self_attention_decomposed_rel_pos_forward",
                                      X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_decomposed_rel_pos_forward(
        X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
        num_heads, grid_h, grid_w, scale, O);
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        Tensor& O) {
    if (grid_h <= 0 || grid_w <= 0 || X.rows != grid_h * grid_w) {
        detail::throw_spatial_2d_error(
            "self_attention_decomposed_rel_pos_windowed_forward", "X",
            grid_h * grid_w, X.cols, grid_h, grid_w, X.rows, X.cols);
    }
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {bq, &Wk, bk, &Wv, bv, &Wo, bo, &rel_pos_h, &rel_pos_w, &O});
    if (!v.self_attention_decomposed_rel_pos_windowed_forward)
        detail::throw_not_implemented(
            "self_attention_decomposed_rel_pos_windowed_forward", X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_decomposed_rel_pos_windowed_forward(
        X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
        num_heads, grid_h, grid_w, window, scale, O);
}

void self_attention_bias_int8w_fp16(const Tensor& X,
                                    const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask,
                                    const Tensor* attn_bias,
                                    int num_heads, float scale, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq_int8, {&sq, &Wk_int8, &sk, &Wv_int8, &sv, &Wo_int8, &so,
                     attn_bias, &O});
    if (!v.self_attention_bias_int8w_fp16)
        detail::throw_not_implemented("self_attention_bias_int8w_fp16", X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_bias_int8w_fp16(X, Wq_int8, sq, Wk_int8, sk,
                                     Wv_int8, sv, Wo_int8, so,
                                     d_mask, attn_bias, num_heads, scale, O);
}

// ─── Flash attention family ────────────────────────────────────────────────

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                             const float* d_mask, int num_heads, bool causal,
                             Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, O);
    if (!v.flash_attention_forward) detail::throw_not_implemented("flash_attention_forward", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_forward(Q, K, V, d_mask, num_heads, causal, O);
}

void flash_attention_gqa_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 const float* d_mask, int num_q_heads, int num_kv_heads,
                                 bool causal, Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, O);
    if (!v.flash_attention_gqa_forward)
        detail::throw_not_implemented("flash_attention_gqa_forward", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_gqa_forward(Q, K, V, d_mask, num_q_heads, num_kv_heads, causal, O);
}

void flash_attention_windowed_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                      const float* d_mask, int num_heads, int window,
                                      Tensor& O, bool causal) {
    const auto& v = detail::dispatch(Q, K, V, O);
    if (!v.flash_attention_windowed_forward)
        detail::throw_not_implemented("flash_attention_windowed_forward", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_windowed_forward(Q, K, V, d_mask, num_heads, window, O, causal);
}

void flash_attention_varlen_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size, int max_seqlen_q, int max_seqlen_k,
                                    int num_heads, int head_dim, bool causal,
                                    Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, O);
    if (!v.flash_attention_varlen_forward)
        detail::throw_not_implemented("flash_attention_varlen_forward", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_varlen_forward(Q, K, V, cu_seqlens_q, cu_seqlens_k,
                                     batch_size, max_seqlen_q, max_seqlen_k,
                                     num_heads, head_dim, causal, O);
}

void flash_attention_varlen_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                     const Tensor& O, const Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     Tensor& dQ, Tensor& dK, Tensor& dV) {
    const auto& v = detail::dispatch(Q, K, V, O, dO, dQ, dK, dV);
    if (!v.flash_attention_varlen_backward)
        detail::throw_not_implemented("flash_attention_varlen_backward", Q.device);
    detail::adopt_output(dQ, Q.device);
    detail::adopt_output(dK, Q.device);
    detail::adopt_output(dV, Q.device);
    v.flash_attention_varlen_backward(Q, K, V, O, dO,
                                      cu_seqlens_q, cu_seqlens_k,
                                      batch_size, max_seqlen_q, max_seqlen_k,
                                      num_heads, head_dim, causal,
                                      dQ, dK, dV);
}

void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask, int num_heads,
                                  bool causal, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {Ctx, bq, &Wk, bk, &Wv, bv, &Wo, bo, &O});
    if (!v.flash_attention_qkvo_forward)
        detail::throw_not_implemented("flash_attention_qkvo_forward", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_qkvo_forward(X, Ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                                   d_mask, num_heads, causal, O);
}

void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask, int num_heads, bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {Ctx, bq, &Wk, bk, &Wv, bv, &Wo, bo, &dO,
                &dX, dCtx, &dWq, dbq, &dWk, dbk, &dWv, dbv, &dWo, dbo});
    if (!v.flash_attention_qkvo_backward)
        detail::throw_not_implemented("flash_attention_qkvo_backward", X.device);
    detail::adopt_output(dX, X.device);
    if (dCtx) detail::adopt_output(*dCtx, X.device);
    detail::adopt_output(dWq, X.device);
    if (dbq) detail::adopt_output(*dbq, X.device);
    detail::adopt_output(dWk, X.device);
    if (dbk) detail::adopt_output(*dbk, X.device);
    detail::adopt_output(dWv, X.device);
    if (dbv) detail::adopt_output(*dbv, X.device);
    detail::adopt_output(dWo, X.device);
    if (dbo) detail::adopt_output(*dbo, X.device);
    v.flash_attention_qkvo_backward(X, Ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                                    d_mask, num_heads, causal, dO,
                                    dX, dCtx, dWq, dbq, dWk, dbk,
                                    dWv, dbv, dWo, dbo);
}

void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& O, const Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              Tensor& dQ, Tensor& dK, Tensor& dV) {
    const auto& v = detail::dispatch(Q, K, V, O, dO, dQ, dK, dV);
    if (!v.flash_attention_backward) detail::throw_not_implemented("flash_attention_backward", Q.device);
    detail::adopt_output(dQ, Q.device);
    detail::adopt_output(dK, Q.device);
    detail::adopt_output(dV, Q.device);
    v.flash_attention_backward(Q, K, V, O, dO, d_mask, num_heads, causal,
                               dQ, dK, dV);
}

void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out, Tensor& V_out) {
    const auto& v = detail::dispatch_with_opts(ctx, Wk, {bk, &Wv, bv, &K_out, &V_out});
    if (!v.flash_attention_project_kv)
        detail::throw_not_implemented("flash_attention_project_kv", ctx.device);
    detail::adopt_output(K_out, ctx.device);
    detail::adopt_output(V_out, ctx.device);
    v.flash_attention_project_kv(ctx, Wk, bk, Wv, bv, K_out, V_out);
}

void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K, const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask, int num_heads,
                                              bool causal, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(X, K, {&V, &Wq, bq, &Wo, bo, &O});
    if (!v.flash_attention_q_with_kv_cached_forward)
        detail::throw_not_implemented("flash_attention_q_with_kv_cached_forward", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_q_with_kv_cached_forward(X, K, V, Wq, bq, Wo, bo,
                                               d_mask, num_heads, causal, O);
}

// ─── Matmul + RoPE + RMSNorm + SwiGLU + KV-cache + Llama ───────────────────

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    const auto& v = detail::dispatch(A, B, C);
    if (!v.matmul) detail::throw_not_implemented("matmul", A.device);
    detail::adopt_output(C, A.device);
    v.matmul(A, B, C);
}

void matmul_abt(const Tensor& A, const Tensor& B, Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const Tensor* bias, int act) {
    const auto& v = detail::dispatch(A, B, C);
    if (!v.matmul_abt) detail::throw_not_implemented("matmul_abt", A.device);
    detail::adopt_output(C, A.device);
    v.matmul_abt(A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, act);
}

void matmul_backward(const Tensor& A, const Tensor& B, const Tensor& dC,
                     Tensor& dA, Tensor& dB) {
    const auto& v = detail::dispatch(A, B, dC, dA, dB);
    if (!v.matmul_backward) detail::throw_not_implemented("matmul_backward", A.device);
    detail::adopt_output(dA, A.device);
    detail::adopt_output(dB, A.device);
    v.matmul_backward(A, B, dC, dA, dB);
}

void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                        const Tensor* b_ih, const Tensor* b_hh,
                        const Tensor* h0, const Tensor* c0, int T, int B,
                        Tensor& Y, Tensor& gates, Tensor& C,
                        Tensor* hT, Tensor* cT) {
    const auto& v = detail::dispatch(X);
    if (!v.lstm_forward_train) detail::throw_not_implemented("lstm_forward_train", X.device);
    detail::adopt_output(Y, X.device);
    detail::adopt_output(gates, X.device);
    detail::adopt_output(C, X.device);
    if (hT) detail::adopt_output(*hT, X.device);
    if (cT) detail::adopt_output(*cT, X.device);
    v.lstm_forward_train(X, W_ih, W_hh, b_ih, b_hh, h0, c0, T, B, Y, gates, C, hT, cT);
}

void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                   const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C,
                   const Tensor& dY, int T, int B,
                   Tensor& dX, Tensor& dW_ih, Tensor& dW_hh,
                   Tensor* db_ih, Tensor* db_hh,
                   Tensor* dh0, Tensor* dc0) {
    const auto& v = detail::dispatch(X);
    if (!v.lstm_backward) detail::throw_not_implemented("lstm_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dW_ih, X.device);
    detail::adopt_output(dW_hh, X.device);
    if (db_ih) detail::adopt_output(*db_ih, X.device);
    if (db_hh) detail::adopt_output(*db_hh, X.device);
    if (dh0) detail::adopt_output(*dh0, X.device);
    if (dc0) detail::adopt_output(*dc0, X.device);
    v.lstm_backward(X, W_ih, W_hh, h0, c0, Y, gates, C, dY, T, B,
                    dX, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0);
}

void rope_forward(const Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.rope_forward) detail::throw_not_implemented("rope_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.rope_forward(X, head_dim, num_heads, seq_offset, theta_base, Y);
}

void rope_backward(const Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.rope_backward) detail::throw_not_implemented("rope_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.rope_backward(dY, head_dim, num_heads, seq_offset, theta_base, dX);
}

void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y) {
    const auto& v = detail::dispatch(X, cos_tbl, sin_tbl, Y);
    if (!v.rope_apply) detail::throw_not_implemented("rope_apply", X.device);
    detail::adopt_output(Y, X.device);
    v.rope_apply(X, cos_tbl, sin_tbl, head_dim, num_heads, Y);
}

void rope_apply_perhead(const Tensor& X, const Tensor& cos_tbl,
                        const Tensor& sin_tbl, int head_dim, int num_heads,
                        Tensor& Y) {
    const auto& v = detail::dispatch(X, cos_tbl, sin_tbl, Y);
    if (!v.rope_apply_perhead)
        detail::throw_not_implemented("rope_apply_perhead", X.device);
    detail::adopt_output(Y, X.device);
    v.rope_apply_perhead(X, cos_tbl, sin_tbl, head_dim, num_heads, Y);
}

void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl,
                         const Tensor& sin_tbl, int head_dim, int num_heads,
                         Tensor& dX) {
    const auto& v = detail::dispatch(dY, cos_tbl, sin_tbl, dX);
    if (!v.rope_apply_backward)
        detail::throw_not_implemented("rope_apply_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.rope_apply_backward(dY, cos_tbl, sin_tbl, head_dim, num_heads, dX);
}

void rope_apply_mrope(const Tensor& X,
                      const Tensor& cos_t, const Tensor& sin_t,
                      const Tensor& cos_h, const Tensor& sin_h,
                      const Tensor& cos_w, const Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      Tensor& Y) {
    const auto& v = detail::dispatch(X, cos_t, sin_t, cos_h, sin_h, cos_w,
                                     sin_w, Y);
    if (!v.rope_apply_mrope)
        detail::throw_not_implemented("rope_apply_mrope", X.device);
    detail::adopt_output(Y, X.device);
    v.rope_apply_mrope(X, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                       pos_t, pos_h, pos_w, head_dim, num_heads,
                       d_t, d_h, d_w, Y);
}

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    const auto& v = detail::dispatch(X, gamma, Y);
    if (!v.rms_norm_forward) detail::throw_not_implemented("rms_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.rms_norm_forward(X, gamma, eps, Y);
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma) {
    const auto& v = detail::dispatch(X, gamma, dY, dX, dGamma);
    if (!v.rms_norm_backward) detail::throw_not_implemented("rms_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma, X.device);
    v.rms_norm_backward(X, gamma, dY, eps, dX, dGamma);
}

void swiglu_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.swiglu_forward) detail::throw_not_implemented("swiglu_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.swiglu_forward(X, Y);
}
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.swiglu_backward) detail::throw_not_implemented("swiglu_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.swiglu_backward(X, dY, dX);
}

void kv_cache_append(const Tensor& K_new, const Tensor& V_new, int cur_len,
                     Tensor& K_cache, Tensor& V_cache) {
    const auto& v = detail::dispatch(K_new, V_new, K_cache, V_cache);
    if (!v.kv_cache_append) detail::throw_not_implemented("kv_cache_append", K_new.device);
    detail::adopt_output(K_cache, K_new.device);
    detail::adopt_output(V_cache, K_new.device);
    v.kv_cache_append(K_new, V_new, cur_len, K_cache, V_cache);
}

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_q_heads, int num_kv_heads,
                            Tensor& O, float attn_softcap, int window) {
    const auto& v = detail::dispatch(Q, K_cache, V_cache, O);
    if (!v.flash_attention_decode) detail::throw_not_implemented("flash_attention_decode", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_decode(Q, K_cache, V_cache, valid_len,
                             num_q_heads, num_kv_heads, O, attn_softcap, window);
}

void flash_attention_decode_masked(const Tensor& Q,
                                   const Tensor& K_cache,
                                   const Tensor& V_cache,
                                   const float* d_mask,
                                   int num_q_heads, int num_kv_heads,
                                   Tensor& O, float attn_softcap, int window) {
    const auto& v = detail::dispatch(Q, K_cache, V_cache, O);
    if (!v.flash_attention_decode_masked) {
        detail::throw_not_implemented("flash_attention_decode_masked", Q.device);
    }
    detail::adopt_output(O, Q.device);
    v.flash_attention_decode_masked(Q, K_cache, V_cache, d_mask,
                                    num_q_heads, num_kv_heads, O,
                                    attn_softcap, window);
}

void flash_attention_decode_len(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
                                const int32_t* d_len, int num_q_heads, int num_kv_heads, Tensor& O) {
    const auto& v = detail::dispatch(Q, K_cache, V_cache, O);
    if (!v.flash_attention_decode_len) detail::throw_not_implemented("flash_attention_decode_len", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_decode_len(Q, K_cache, V_cache, d_len, num_q_heads, num_kv_heads, O);
}

// ─── Public reductions ─────────────────────────────────────────────────────

void sum_rows(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.sum_rows) detail::throw_not_implemented("sum_rows", X.device);
    detail::adopt_output(Y, X.device);
    v.sum_rows(X, Y);
}
void sum_cols(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.sum_cols) detail::throw_not_implemented("sum_cols", X.device);
    detail::adopt_output(Y, X.device);
    v.sum_cols(X, Y);
}
void argmax_rows(const Tensor& X, Tensor& Idx) {
    const auto& v = detail::dispatch(X, Idx);
    if (!v.argmax_rows) detail::throw_not_implemented("argmax_rows", X.device);
    detail::adopt_output(Idx, X.device);
    v.argmax_rows(X, Idx);
}
void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets, Tensor& out) {
    const auto& v = detail::dispatch(logits, seg_offsets, out);
    if (!v.segment_softmax_stats)
        detail::throw_not_implemented("segment_softmax_stats", logits.device);
    detail::adopt_output(out, logits.device);
    v.segment_softmax_stats(logits, seg_offsets, out);
}
void flash_attention_packed_qkv_forward(const Tensor& QKV, const Tensor& seq_bounds,
                                        int num_heads, int window, Tensor& O) {
    const auto& v = detail::dispatch(QKV, seq_bounds, O);
    if (!v.flash_attention_packed_qkv_forward)
        detail::throw_not_implemented("flash_attention_packed_qkv_forward", QKV.device);
    detail::adopt_output(O, QKV.device);
    v.flash_attention_packed_qkv_forward(QKV, seq_bounds, num_heads, window, O);
}
void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV) {
    const auto& v = detail::dispatch(QKV, dO, seq_bounds, dQKV);
    if (!v.flash_attention_packed_qkv_backward)
        detail::throw_not_implemented("flash_attention_packed_qkv_backward", QKV.device);
    detail::adopt_output(dQKV, QKV.device);
    v.flash_attention_packed_qkv_backward(QKV, dO, seq_bounds, num_heads, window, dQKV);
}
void rope_qkv_packed_inplace(Tensor& QKV, const Tensor& cos_tbl, const Tensor& sin_tbl,
                             const Tensor& pos, int num_heads, int head_dim) {
    const auto& v = detail::dispatch(QKV, cos_tbl, sin_tbl, pos);
    if (!v.rope_qkv_packed_inplace)
        detail::throw_not_implemented("rope_qkv_packed_inplace", QKV.device);
    v.rope_qkv_packed_inplace(QKV, cos_tbl, sin_tbl, pos, num_heads, head_dim);
}
void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts) {
    const auto& v = detail::dispatch(X, counts);
    if (!v.rows_count_above)
        detail::throw_not_implemented("rows_count_above", X.device);
    detail::adopt_output(counts, X.device);
    v.rows_count_above(X, t_lo, t_hi, counts);
}
void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.threshold_u8) detail::throw_not_implemented("threshold_u8", X.device);
    detail::adopt_output(Y, X.device);
    v.threshold_u8(X, t, Y);
}

} // namespace brotensor
