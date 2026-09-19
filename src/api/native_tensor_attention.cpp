// The attention family of bro.tensor, ported from the QuickJS binding
// (tensor_bindings.cpp: attention/mha; tensor_bindings_attention.cpp: self /
// cross / bias attention, the flash-attention family, resblock). Every
// nullable tensor (mask|null, the projection biases, Ctx|null, the optional
// gradient outputs) crosses as `dynamic`; js/tensor_ext.js has already
// checked each one is null or a GpuTensor, and unpacks the options-object
// forms (flashAttentionQkvoBackward(opts), resblockForward(opts),
// resblockBackward(opts)) into these positional calls.

#include "native_tensor_decl.h"
#include "api_internal.h"

using namespace brotensor::api;

extern "C" {

void bro_tensor_attentionForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, void* Q, void* K, void* V, void* Attn, void* Y_pre_Wo, void* O) {
    if (!need("attentionForward", {X, Wq, Wk, Wv, Wo, Q, K, V, Attn, Y_pre_Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::attention_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                     maskPtr(mask_bits),
                                     *toTensor(Q), *toTensor(K), *toTensor(V), *toTensor(Attn), *toTensor(Y_pre_Wo),
                                     *toTensor(O));
    BROTENSOR_API_CATCH("attentionForward")
}

void bro_tensor_attentionBackward(void* dO, void* X, void* Q, void* K, void* V, void* Attn, void* Y_pre_Wo, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    if (!need("attentionBackward", {dO, X, Q, K, V, Attn, Y_pre_Wo, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    BROTENSOR_API_TRY
        brotensor::attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Q), *toTensor(K), *toTensor(V),
                                      *toTensor(Attn), *toTensor(Y_pre_Wo),
                                      *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                      maskPtr(mask_bits),
                                      *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH("attentionBackward")
}

void bro_tensor_mhaForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    if (!need("mhaForward", {X, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    BROTENSOR_API_TRY
        brotensor::mha_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                               maskPtr(mask_bits), numHeads,
                               *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh), *toTensor(Yconcat),
                               *toTensor(O));
    BROTENSOR_API_CATCH("mhaForward")
}

void bro_tensor_mhaBackward(void* dO, void* X, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    if (!need("mhaBackward", {dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    BROTENSOR_API_TRY
        brotensor::mha_backward(*toTensor(dO), *toTensor(X), *toTensor(Qh), *toTensor(Kh), *toTensor(Vh),
                                *toTensor(Attnh), *toTensor(Yconcat),
                                *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                maskPtr(mask_bits), numHeads,
                                *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH("mhaBackward")
}

void bro_tensor_selfAttentionForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* O) {
    if (!need("selfAttentionForward", {X, Wq, Wk, Wv, Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                          maskPtr(mask_bits), numHeads, *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionForward")
}

void bro_tensor_selfAttentionForwardTrain(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    if (!need("selfAttentionForwardTrain", {X, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_forward_train(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                                maskPtr(mask_bits), numHeads,
                                                *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh),
                                                *toTensor(Yconcat), *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionForwardTrain")
}

void bro_tensor_selfAttentionBackward(void* dO, void* X, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    if (!need("selfAttentionBackward", {dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Qh), *toTensor(Kh), *toTensor(Vh),
                                           *toTensor(Attnh), *toTensor(Yconcat),
                                           *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                           maskPtr(mask_bits), numHeads,
                                           *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH("selfAttentionBackward")
}

void bro_tensor_selfAttentionBiasForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, uint64_t attnBias_bits, int32_t numHeads, double scale, void* O) {
    if (!need("selfAttentionBiasForward", {X, Wq, Wk, Wv, Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_bias_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                               nullptr, nullptr, nullptr, nullptr,
                                               maskPtr(mask_bits), tensorFromValue(attnBias_bits),
                                               numHeads, static_cast<float>(scale), *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionBiasForward")
}

void bro_tensor_crossAttentionForward(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* O) {
    if (!need("crossAttentionForward", {X, Ctx, Wq, Wk, Wv, Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv),
                                           *toTensor(Wo), maskPtr(mask_bits), numHeads, *toTensor(O));
    BROTENSOR_API_CATCH("crossAttentionForward")
}

void bro_tensor_crossAttentionForwardWithAttn(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, uint64_t attnLogitBias_bits, int32_t numHeads, void* O, void* AttnAvg) {
    if (!need("crossAttentionForwardWithAttn", {X, Ctx, Wq, Wk, Wv, Wo, O, AttnAvg})) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward_with_attn(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk),
                                                     *toTensor(Wv), *toTensor(Wo),
                                                     maskPtr(mask_bits), tensorFromValue(attnLogitBias_bits),
                                                     numHeads, *toTensor(O), *toTensor(AttnAvg));
    BROTENSOR_API_CATCH("crossAttentionForwardWithAttn")
}

void bro_tensor_crossAttentionForwardTrain(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    if (!need("crossAttentionForwardTrain", {X, Ctx, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward_train(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk),
                                                 *toTensor(Wv), *toTensor(Wo), maskPtr(mask_bits), numHeads,
                                                 *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh),
                                                 *toTensor(Yconcat), *toTensor(O));
    BROTENSOR_API_CATCH("crossAttentionForwardTrain")
}

void bro_tensor_crossAttentionBackward(void* dO, void* X, void* Ctx, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dCtx, void* dWq, void* dWk, void* dWv, void* dWo) {
    if (!need("crossAttentionBackward", {dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dCtx, dWq, dWk, dWv, dWo})) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Ctx), *toTensor(Qh), *toTensor(Kh),
                                            *toTensor(Vh), *toTensor(Attnh), *toTensor(Yconcat),
                                            *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                            maskPtr(mask_bits), numHeads,
                                            *toTensor(dX), *toTensor(dCtx),
                                            *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH("crossAttentionBackward")
}

void bro_tensor_flashAttentionForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    if (!need("flashAttentionForward", {Q, K, V, O})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_forward(*toTensor(Q), *toTensor(K), *toTensor(V), maskPtr(mask_bits),
                                           numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionForward")
}

void bro_tensor_flashAttentionWindowedForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numHeads, int32_t window, void* O) {
    if (!need("flashAttentionWindowedForward", {Q, K, V, O})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_windowed_forward(*toTensor(Q), *toTensor(K), *toTensor(V), maskPtr(mask_bits),
                                                    numHeads, window, *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionWindowedForward")
}

void bro_tensor_flashAttentionBackward(void* Q, void* K, void* V, void* O, void* dO, uint64_t mask_bits, int32_t numHeads, bool causal, void* dQ, void* dK, void* dV) {
    if (!need("flashAttentionBackward", {Q, K, V, O, dO, dQ, dK, dV})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_backward(*toTensor(Q), *toTensor(K), *toTensor(V), *toTensor(O), *toTensor(dO),
                                            maskPtr(mask_bits), numHeads, causal,
                                            *toTensor(dQ), *toTensor(dK), *toTensor(dV));
    BROTENSOR_API_CATCH("flashAttentionBackward")
}

void bro_tensor_flashAttentionQkvoForward(void* X, uint64_t Ctx_bits, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    if (!need("flashAttentionQkvoForward", {X, Wq, Wk, Wv, Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_qkvo_forward(*toTensor(X), tensorFromValue(Ctx_bits),
                                                *toTensor(Wq), tensorFromValue(bq_bits),
                                                *toTensor(Wk), tensorFromValue(bk_bits),
                                                *toTensor(Wv), tensorFromValue(bv_bits),
                                                *toTensor(Wo), tensorFromValue(bo_bits),
                                                maskPtr(mask_bits), numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionQkvoForward")
}

void bro_tensor_flashAttentionQkvoBackward(void* X, uint64_t Ctx_bits, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* dO, void* dX, uint64_t dCtx_bits, void* dWq, uint64_t dbq_bits, void* dWk, uint64_t dbk_bits, void* dWv, uint64_t dbv_bits, void* dWo, uint64_t dbo_bits) {
    if (!need("flashAttentionQkvoBackward", {X, Wq, Wk, Wv, Wo, dO, dX, dWq, dWk, dWv, dWo})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_qkvo_backward(*toTensor(X), tensorFromValue(Ctx_bits),
                                                 *toTensor(Wq), tensorFromValue(bq_bits),
                                                 *toTensor(Wk), tensorFromValue(bk_bits),
                                                 *toTensor(Wv), tensorFromValue(bv_bits),
                                                 *toTensor(Wo), tensorFromValue(bo_bits),
                                                 maskPtr(mask_bits), numHeads, causal,
                                                 *toTensor(dO),
                                                 *toTensor(dX), tensorFromValue(dCtx_bits),
                                                 *toTensor(dWq), tensorFromValue(dbq_bits),
                                                 *toTensor(dWk), tensorFromValue(dbk_bits),
                                                 *toTensor(dWv), tensorFromValue(dbv_bits),
                                                 *toTensor(dWo), tensorFromValue(dbo_bits));
    BROTENSOR_API_CATCH("flashAttentionQkvoBackward")
}

void bro_tensor_flashAttentionProjectKv(void* ctx, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* K_out, void* V_out) {
    if (!need("flashAttentionProjectKv", {ctx, Wk, Wv, K_out, V_out})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_project_kv(*toTensor(ctx), *toTensor(Wk), tensorFromValue(bk_bits),
                                              *toTensor(Wv), tensorFromValue(bv_bits),
                                              *toTensor(K_out), *toTensor(V_out));
    BROTENSOR_API_CATCH("flashAttentionProjectKv")
}

void bro_tensor_flashAttentionQWithKvCachedForward(void* X, void* K, void* V, void* Wq, uint64_t bq_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    if (!need("flashAttentionQWithKvCachedForward", {X, K, V, Wq, Wo, O})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_q_with_kv_cached_forward(*toTensor(X), *toTensor(K), *toTensor(V),
                                                            *toTensor(Wq), tensorFromValue(bq_bits),
                                                            *toTensor(Wo), tensorFromValue(bo_bits),
                                                            maskPtr(mask_bits), numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionQWithKvCachedForward")
}

void bro_tensor_resblockForward(void* X, void* gamma1, void* beta1, void* W1, uint64_t b1_bits, uint64_t t_emb_shift_bits, void* gamma2, void* beta2, void* W2, uint64_t b2_bits, uint64_t Wskip_bits, uint64_t bskip_bits, int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t W, int32_t numGroups, double eps, void* Y) {
    if (!need("resblockForward", {X, gamma1, beta1, W1, gamma2, beta2, W2, Y})) return;
    BROTENSOR_API_TRY
        brotensor::resblock_forward(*toTensor(X), *toTensor(gamma1), *toTensor(beta1),
                                    *toTensor(W1), tensorFromValue(b1_bits), tensorFromValue(t_emb_shift_bits),
                                    *toTensor(gamma2), *toTensor(beta2),
                                    *toTensor(W2), tensorFromValue(b2_bits),
                                    tensorFromValue(Wskip_bits), tensorFromValue(bskip_bits),
                                    N, C_in, C_out, H, W, numGroups, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("resblockForward")
}

void bro_tensor_resblockBackward(void* X, void* gamma1, void* beta1, void* W1, uint64_t b1_bits, uint64_t t_emb_shift_bits, void* gamma2, void* beta2, void* W2, uint64_t b2_bits, uint64_t Wskip_bits, uint64_t bskip_bits, int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t W, int32_t numGroups, double eps, void* dY, void* dX, void* dGamma1, void* dBeta1, void* dW1, uint64_t db1_bits, uint64_t dt_emb_shift_bits, void* dGamma2, void* dBeta2, void* dW2, uint64_t db2_bits, uint64_t dWskip_bits, uint64_t dbskip_bits) {
    if (!need("resblockBackward", {X, gamma1, beta1, W1, gamma2, beta2, W2, dY, dX, dGamma1, dBeta1, dW1, dGamma2, dBeta2, dW2})) return;
    BROTENSOR_API_TRY
        brotensor::resblock_backward(*toTensor(X), *toTensor(gamma1), *toTensor(beta1),
                                     *toTensor(W1), tensorFromValue(b1_bits), tensorFromValue(t_emb_shift_bits),
                                     *toTensor(gamma2), *toTensor(beta2),
                                     *toTensor(W2), tensorFromValue(b2_bits),
                                     tensorFromValue(Wskip_bits), tensorFromValue(bskip_bits),
                                     N, C_in, C_out, H, W, numGroups, static_cast<float>(eps),
                                     *toTensor(dY), *toTensor(dX),
                                     *toTensor(dGamma1), *toTensor(dBeta1),
                                     *toTensor(dW1), tensorFromValue(db1_bits), tensorFromValue(dt_emb_shift_bits),
                                     *toTensor(dGamma2), *toTensor(dBeta2),
                                     *toTensor(dW2), tensorFromValue(db2_bits),
                                     tensorFromValue(dWskip_bits), tensorFromValue(dbskip_bits));
    BROTENSOR_API_CATCH("resblockBackward")
}

} // extern "C"
