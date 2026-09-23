// The attention family of bro.tensor, ported from the QuickJS binding
// (tensor_bindings.cpp: attention/mha; tensor_bindings_attention.cpp: self /
// cross / bias attention, the flash-attention family, resblock). Every
// nullable tensor (mask|null, the projection biases, Ctx|null, the optional
// gradient outputs) crosses as `dynamic`; js/tensor_ext.js has already
// checked each one is null or a GpuTensor, and unpacks the options-object
// forms (flashAttentionQkvoBackward(opts), resblockForward(opts),
// resblockBackward(opts)) into these positional calls.
//
// Each body checks its operands against the shapes X (and Ctx) imply before
// the op runs (api_internal.h "input-size contract"): the weights are (D,D)
// or (D,D_ctx), the caches and upstream gradients match the forward, the
// accumulated weight gradients match their weights, and a mask covers every
// key.

#include "native_tensor_decl.h"
#include "api_internal.h"

using namespace brotensor::api;
using brotensor::Tensor;

namespace {

Tensor* T(void* p) { return toTensor(p); }

// Square (D,D) projections Wq/Wk/Wv/Wo in X's dtype.
bool needSquareProj(const char* L, const Tensor* X, void* Wq, void* Wk, void* Wv, void* Wo) {
    const int64_t DD = elems({X->cols, X->cols});
    return needOperands(L, X, {{"Wq", T(Wq), DD}, {"Wk", T(Wk), DD}, {"Wv", T(Wv), DD}, {"Wo", T(Wo), DD}});
}

// Cross projections: Wq/Wo (D,D), Wk/Wv (D,D_ctx).
bool needCrossProj(const char* L, const Tensor* X, const Tensor* Ctx, void* Wq, void* Wk, void* Wv, void* Wo) {
    const int64_t DD = elems({X->cols, X->cols});
    const int64_t DC = elems({X->cols, Ctx->cols});
    if (!needSameDtype(L, "Ctx", Ctx, X)) return false;
    return needOperands(L, X, {{"Wq", T(Wq), DD}, {"Wk", T(Wk), DC}, {"Wv", T(Wv), DC}, {"Wo", T(Wo), DD}});
}

// The multi-head backward caches of a self-attention forward over X (K,D).
bool needMhaCaches(const char* L, const Tensor* X, int32_t numHeads,
                   void* dO, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat) {
    const int64_t KD = elems({X->rows, X->cols});
    return needOperands(L, X, {{"dO", T(dO), KD}, {"Qh", T(Qh), KD}, {"Kh", T(Kh), KD}, {"Vh", T(Vh), KD},
                               {"Attnh", T(Attnh), elems({numHeads, X->rows, X->rows})},
                               {"Yconcat", T(Yconcat), KD}});
}

// Accumulated (D,D) weight gradients.
bool needSquareGrads(const char* L, const Tensor* X, void* dWq, void* dWk, void* dWv, void* dWo) {
    const int64_t DD = elems({X->cols, X->cols});
    return needOperands(L, X, {{"dWq", T(dWq), DD}, {"dWk", T(dWk), DD}, {"dWv", T(dWv), DD}, {"dWo", T(dWo), DD}});
}

// Pre-projected flash attention: Q (Lq,D), K/V (Lk,D), numHeads | D, a
// causal run needs Lq == Lk.
bool needFlashQkv(const char* L, const Tensor* Q, const Tensor* K, const Tensor* V, int32_t numHeads, bool causal) {
    if (!needHeads(L, Q->cols, numHeads)) return false;
    if (K->cols != Q->cols || V->cols != Q->cols || V->rows != K->rows) {
        setError(std::string(L) + ": Q is (Lq,D); K and V must both be (Lk,D)");
        return false;
    }
    if (causal && Q->rows != K->rows) {
        setError(std::string(L) + ": causal attention needs Lq == Lk");
        return false;
    }
    return needSameDtype(L, "K", K, Q) && needSameDtype(L, "V", V, Q);
}

// The optional (D,1) projection biases of the qkvo family, in X's dtype.
bool needBiases(const char* L, const Tensor* X, std::initializer_list<Operand> biases) {
    return needOperands(L, X, biases);
}

} // namespace

extern "C" {

void bro_tensor_attentionForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, void* Q, void* K, void* V, void* Attn, void* Y_pre_Wo, void* O) {
    const char* L = "attentionForward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, Q, K, V, Attn, Y_pre_Wo, O})) return;
    const Tensor* x = T(X);
    if (!needSquareProj(L, x, Wq, Wk, Wv, Wo) || !needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::attention_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                     maskPtr(mask_bits),
                                     *toTensor(Q), *toTensor(K), *toTensor(V), *toTensor(Attn), *toTensor(Y_pre_Wo),
                                     *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_attentionBackward(void* dO, void* X, void* Q, void* K, void* V, void* Attn, void* Y_pre_Wo, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    const char* L = "attentionBackward";
    if (!need(L, {dO, X, Q, K, V, Attn, Y_pre_Wo, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    const Tensor* x = T(X);
    const int64_t ND = elems({x->rows, x->cols});
    if (!needSquareProj(L, x, Wq, Wk, Wv, Wo) || !needSquareGrads(L, x, dWq, dWk, dWv, dWo)) return;
    if (!needOperands(L, x, {{"dO", T(dO), ND}, {"Q", T(Q), ND}, {"K", T(K), ND}, {"V", T(V), ND},
                             {"Attn", T(Attn), elems({x->rows, x->rows})}, {"Y_pre_Wo", T(Y_pre_Wo), ND}})) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Q), *toTensor(K), *toTensor(V),
                                      *toTensor(Attn), *toTensor(Y_pre_Wo),
                                      *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                      maskPtr(mask_bits),
                                      *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_mhaForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    const char* L = "mhaForward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::mha_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                               maskPtr(mask_bits), numHeads,
                               *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh), *toTensor(Yconcat),
                               *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_mhaBackward(void* dO, void* X, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    const char* L = "mhaBackward";
    if (!need(L, {dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needSquareGrads(L, x, dWq, dWk, dWv, dWo) || !needMhaCaches(L, x, numHeads, dO, Qh, Kh, Vh, Attnh, Yconcat)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::mha_backward(*toTensor(dO), *toTensor(X), *toTensor(Qh), *toTensor(Kh), *toTensor(Vh),
                                *toTensor(Attnh), *toTensor(Yconcat),
                                *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                maskPtr(mask_bits), numHeads,
                                *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_selfAttentionForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* O) {
    const char* L = "selfAttentionForward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, O})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                          maskPtr(mask_bits), numHeads, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_selfAttentionForwardTrain(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    const char* L = "selfAttentionForwardTrain";
    if (!need(L, {X, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_forward_train(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                                maskPtr(mask_bits), numHeads,
                                                *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh),
                                                *toTensor(Yconcat), *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_selfAttentionBackward(void* dO, void* X, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dWq, void* dWk, void* dWv, void* dWo) {
    const char* L = "selfAttentionBackward";
    if (!need(L, {dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dWq, dWk, dWv, dWo})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needSquareGrads(L, x, dWq, dWk, dWv, dWo) || !needMhaCaches(L, x, numHeads, dO, Qh, Kh, Vh, Attnh, Yconcat)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Qh), *toTensor(Kh), *toTensor(Vh),
                                           *toTensor(Attnh), *toTensor(Yconcat),
                                           *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                           maskPtr(mask_bits), numHeads,
                                           *toTensor(dX), *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_selfAttentionBiasForward(void* X, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, uint64_t attnBias_bits, int32_t numHeads, double scale, void* O) {
    const char* L = "selfAttentionBiasForward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, O})) return;
    const Tensor* x = T(X);
    if (!needHeads(L, x->cols, numHeads) || !needSquareProj(L, x, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), x->rows)) return;
    // attn_bias: (numHeads*L, L) FP32.
    if (!needMask(L, tensorFromValue(attnBias_bits), elems({numHeads, x->rows, x->rows}), "attnBias")) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_bias_forward(*toTensor(X), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                               nullptr, nullptr, nullptr, nullptr,
                                               maskPtr(mask_bits), tensorFromValue(attnBias_bits),
                                               numHeads, static_cast<float>(scale), *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_crossAttentionForward(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* O) {
    const char* L = "crossAttentionForward";
    if (!need(L, {X, Ctx, Wq, Wk, Wv, Wo, O})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = T(Ctx);
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, ctx, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), ctx->rows)) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk), *toTensor(Wv),
                                           *toTensor(Wo), maskPtr(mask_bits), numHeads, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_crossAttentionForwardWithAttn(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, uint64_t attnLogitBias_bits, int32_t numHeads, void* O, void* AttnAvg) {
    const char* L = "crossAttentionForwardWithAttn";
    if (!need(L, {X, Ctx, Wq, Wk, Wv, Wo, O, AttnAvg})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = T(Ctx);
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, ctx, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), ctx->rows)) return;
    // attn_logit_bias: (Lq, Lk) FP32.
    if (!needMask(L, tensorFromValue(attnLogitBias_bits), elems({x->rows, ctx->rows}), "attnLogitBias")) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward_with_attn(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk),
                                                     *toTensor(Wv), *toTensor(Wo),
                                                     maskPtr(mask_bits), tensorFromValue(attnLogitBias_bits),
                                                     numHeads, *toTensor(O), *toTensor(AttnAvg));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_crossAttentionForwardTrain(void* X, void* Ctx, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* O) {
    const char* L = "crossAttentionForwardTrain";
    if (!need(L, {X, Ctx, Wq, Wk, Wv, Wo, Qh, Kh, Vh, Attnh, Yconcat, O})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = T(Ctx);
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, ctx, Wq, Wk, Wv, Wo)) return;
    if (!needMask(L, tensorFromValue(mask_bits), ctx->rows)) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_forward_train(*toTensor(X), *toTensor(Ctx), *toTensor(Wq), *toTensor(Wk),
                                                 *toTensor(Wv), *toTensor(Wo), maskPtr(mask_bits), numHeads,
                                                 *toTensor(Qh), *toTensor(Kh), *toTensor(Vh), *toTensor(Attnh),
                                                 *toTensor(Yconcat), *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_crossAttentionBackward(void* dO, void* X, void* Ctx, void* Qh, void* Kh, void* Vh, void* Attnh, void* Yconcat, void* Wq, void* Wk, void* Wv, void* Wo, uint64_t mask_bits, int32_t numHeads, void* dX, void* dCtx, void* dWq, void* dWk, void* dWv, void* dWo) {
    const char* L = "crossAttentionBackward";
    if (!need(L, {dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, dX, dCtx, dWq, dWk, dWv, dWo})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = T(Ctx);
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, ctx, Wq, Wk, Wv, Wo)) return;
    const int64_t QD = elems({x->rows, x->cols});
    const int64_t KD = elems({ctx->rows, x->cols});
    const int64_t DD = elems({x->cols, x->cols});
    const int64_t DC = elems({x->cols, ctx->cols});
    if (!needOperands(L, x, {{"dO", T(dO), QD}, {"Qh", T(Qh), QD}, {"Kh", T(Kh), KD}, {"Vh", T(Vh), KD},
                             {"Attnh", T(Attnh), elems({numHeads, x->rows, ctx->rows})}, {"Yconcat", T(Yconcat), QD},
                             {"dWq", T(dWq), DD}, {"dWk", T(dWk), DC}, {"dWv", T(dWv), DC}, {"dWo", T(dWo), DD}})) return;
    if (!needMask(L, tensorFromValue(mask_bits), ctx->rows)) return;
    BROTENSOR_API_TRY
        brotensor::cross_attention_backward(*toTensor(dO), *toTensor(X), *toTensor(Ctx), *toTensor(Qh), *toTensor(Kh),
                                            *toTensor(Vh), *toTensor(Attnh), *toTensor(Yconcat),
                                            *toTensor(Wq), *toTensor(Wk), *toTensor(Wv), *toTensor(Wo),
                                            maskPtr(mask_bits), numHeads,
                                            *toTensor(dX), *toTensor(dCtx),
                                            *toTensor(dWq), *toTensor(dWk), *toTensor(dWv), *toTensor(dWo));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    const char* L = "flashAttentionForward";
    if (!need(L, {Q, K, V, O})) return;
    if (!needFlashQkv(L, T(Q), T(K), T(V), numHeads, causal)) return;
    if (!needMask(L, tensorFromValue(mask_bits), T(K)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_forward(*toTensor(Q), *toTensor(K), *toTensor(V), maskPtr(mask_bits),
                                           numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionWindowedForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numHeads, int32_t window, void* O) {
    const char* L = "flashAttentionWindowedForward";
    if (!need(L, {Q, K, V, O})) return;
    if (!needFlashQkv(L, T(Q), T(K), T(V), numHeads, false)) return;
    if (T(K)->rows < T(Q)->rows) { setError("flashAttentionWindowedForward: needs Lk >= Lq"); return; }
    if (!needMask(L, tensorFromValue(mask_bits), T(K)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_windowed_forward(*toTensor(Q), *toTensor(K), *toTensor(V), maskPtr(mask_bits),
                                                    numHeads, window, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionBackward(void* Q, void* K, void* V, void* O, void* dO, uint64_t mask_bits, int32_t numHeads, bool causal, void* dQ, void* dK, void* dV) {
    const char* L = "flashAttentionBackward";
    if (!need(L, {Q, K, V, O, dO, dQ, dK, dV})) return;
    if (!needFlashQkv(L, T(Q), T(K), T(V), numHeads, causal)) return;
    if (!needPair(L, "dO", T(dO), T(Q))) return;
    if (!needMask(L, tensorFromValue(mask_bits), T(K)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_backward(*toTensor(Q), *toTensor(K), *toTensor(V), *toTensor(O), *toTensor(dO),
                                            maskPtr(mask_bits), numHeads, causal,
                                            *toTensor(dQ), *toTensor(dK), *toTensor(dV));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionQkvoForward(void* X, uint64_t Ctx_bits, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    const char* L = "flashAttentionQkvoForward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, O})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = tensorFromValue(Ctx_bits);
    const Tensor* kv = ctx ? ctx : x;
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, kv, Wq, Wk, Wv, Wo)) return;
    if (!needBiases(L, x, {{"bq", tensorFromValue(bq_bits), x->cols}, {"bk", tensorFromValue(bk_bits), x->cols},
                           {"bv", tensorFromValue(bv_bits), x->cols}, {"bo", tensorFromValue(bo_bits), x->cols}})) return;
    if (!needMask(L, tensorFromValue(mask_bits), kv->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_qkvo_forward(*toTensor(X), tensorFromValue(Ctx_bits),
                                                *toTensor(Wq), tensorFromValue(bq_bits),
                                                *toTensor(Wk), tensorFromValue(bk_bits),
                                                *toTensor(Wv), tensorFromValue(bv_bits),
                                                *toTensor(Wo), tensorFromValue(bo_bits),
                                                maskPtr(mask_bits), numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionQkvoBackward(void* X, uint64_t Ctx_bits, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* dO, void* dX, uint64_t dCtx_bits, void* dWq, uint64_t dbq_bits, void* dWk, uint64_t dbk_bits, void* dWv, uint64_t dbv_bits, void* dWo, uint64_t dbo_bits) {
    const char* L = "flashAttentionQkvoBackward";
    if (!need(L, {X, Wq, Wk, Wv, Wo, dO, dX, dWq, dWk, dWv, dWo})) return;
    const Tensor* x = T(X);
    const Tensor* ctx = tensorFromValue(Ctx_bits);
    const Tensor* kv = ctx ? ctx : x;
    if (!needHeads(L, x->cols, numHeads) || !needCrossProj(L, x, kv, Wq, Wk, Wv, Wo)) return;
    const int64_t D = x->cols;
    const int64_t DD = elems({D, D});
    const int64_t DC = elems({D, kv->cols});
    if (!needBiases(L, x, {{"bq", tensorFromValue(bq_bits), D}, {"bk", tensorFromValue(bk_bits), D},
                           {"bv", tensorFromValue(bv_bits), D}, {"bo", tensorFromValue(bo_bits), D},
                           {"dbq", tensorFromValue(dbq_bits), D}, {"dbk", tensorFromValue(dbk_bits), D},
                           {"dbv", tensorFromValue(dbv_bits), D}, {"dbo", tensorFromValue(dbo_bits), D}})) return;
    if (!needOperands(L, x, {{"dO", T(dO), elems({x->rows, D})},
                             {"dWq", T(dWq), DD}, {"dWk", T(dWk), DC}, {"dWv", T(dWv), DC}, {"dWo", T(dWo), DD}})) return;
    if (!needMask(L, tensorFromValue(mask_bits), kv->rows)) return;
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
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionProjectKv(void* ctx, void* Wk, uint64_t bk_bits, void* Wv, uint64_t bv_bits, void* K_out, void* V_out) {
    const char* L = "flashAttentionProjectKv";
    if (!need(L, {ctx, Wk, Wv, K_out, V_out})) return;
    // Wk/Wv are (D, D_ctx); D is Wk's row count.
    const Tensor* c = T(ctx);
    const Tensor* wk = T(Wk);
    const int64_t D = wk->rows;
    if (!needOperands(L, c, {{"Wk", wk, elems({D, c->cols})}, {"Wv", T(Wv), elems({D, c->cols})},
                             {"bk", tensorFromValue(bk_bits), D}, {"bv", tensorFromValue(bv_bits), D}})) return;
    if (wk->cols != c->cols) { setError("flashAttentionProjectKv: Wk must be (D, ctx.cols)"); return; }
    BROTENSOR_API_TRY
        brotensor::flash_attention_project_kv(*toTensor(ctx), *toTensor(Wk), tensorFromValue(bk_bits),
                                              *toTensor(Wv), tensorFromValue(bv_bits),
                                              *toTensor(K_out), *toTensor(V_out));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionQWithKvCachedForward(void* X, void* K, void* V, void* Wq, uint64_t bq_bits, void* Wo, uint64_t bo_bits, uint64_t mask_bits, int32_t numHeads, bool causal, void* O) {
    const char* L = "flashAttentionQWithKvCachedForward";
    if (!need(L, {X, K, V, Wq, Wo, O})) return;
    const Tensor* x = T(X);
    const int64_t D = x->cols;
    if (!needHeads(L, D, numHeads)) return;
    if (T(K)->cols != D || T(V)->cols != D || T(V)->rows != T(K)->rows) {
        setError("flashAttentionQWithKvCachedForward: K and V must both be (Lk, D)");
        return;
    }
    if (causal && x->rows != T(K)->rows) { setError("flashAttentionQWithKvCachedForward: causal attention needs Lq == Lk"); return; }
    if (!needOperands(L, x, {{"K", T(K), 0}, {"V", T(V), 0}, {"Wq", T(Wq), elems({D, D})}, {"Wo", T(Wo), elems({D, D})},
                             {"bq", tensorFromValue(bq_bits), D}, {"bo", tensorFromValue(bo_bits), D}})) return;
    if (!needMask(L, tensorFromValue(mask_bits), T(K)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_q_with_kv_cached_forward(*toTensor(X), *toTensor(K), *toTensor(V),
                                                            *toTensor(Wq), tensorFromValue(bq_bits),
                                                            *toTensor(Wo), tensorFromValue(bo_bits),
                                                            maskPtr(mask_bits), numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

// The fused SD ResBlock: 3x3 convs W1 (C_out, C_in*9) and W2 (C_out, C_out*9),
// per-channel GN params, an optional 1x1 skip.
static bool needResblock(const char* L, const Tensor* x, void* gamma1, void* beta1, void* W1, uint64_t b1_bits,
                         uint64_t t_emb_shift_bits, void* gamma2, void* beta2, void* W2, uint64_t b2_bits,
                         uint64_t Wskip_bits, uint64_t bskip_bits,
                         int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t W, int32_t numGroups) {
    if (!needPositive(L, {numGroups}) || !needNonNegative(L, {N, C_in, C_out, H, W})) return false;
    if (C_in % numGroups != 0 || C_out % numGroups != 0) {
        setError(std::string(L) + ": numGroups must divide C_in and C_out");
        return false;
    }
    if (C_in != C_out && !tensorFromValue(Wskip_bits)) {
        setError(std::string(L) + ": C_in != C_out needs a Wskip projection");
        return false;
    }
    return needOperands(L, x, {{"X", x, elems({N, C_in, H, W})},
                               {"gamma1", T(gamma1), C_in}, {"beta1", T(beta1), C_in},
                               {"W1", T(W1), elems({C_out, C_in, 9})}, {"b1", tensorFromValue(b1_bits), C_out},
                               {"t_emb_shift", tensorFromValue(t_emb_shift_bits), C_out},
                               {"gamma2", T(gamma2), C_out}, {"beta2", T(beta2), C_out},
                               {"W2", T(W2), elems({C_out, C_out, 9})}, {"b2", tensorFromValue(b2_bits), C_out},
                               {"Wskip", tensorFromValue(Wskip_bits), elems({C_out, C_in})},
                               {"bskip", tensorFromValue(bskip_bits), C_out}});
}

void bro_tensor_resblockForward(void* X, void* gamma1, void* beta1, void* W1, uint64_t b1_bits, uint64_t t_emb_shift_bits, void* gamma2, void* beta2, void* W2, uint64_t b2_bits, uint64_t Wskip_bits, uint64_t bskip_bits, int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t W, int32_t numGroups, double eps, void* Y) {
    const char* L = "resblockForward";
    if (!need(L, {X, gamma1, beta1, W1, gamma2, beta2, W2, Y})) return;
    if (!needResblock(L, T(X), gamma1, beta1, W1, b1_bits, t_emb_shift_bits, gamma2, beta2, W2, b2_bits,
                      Wskip_bits, bskip_bits, N, C_in, C_out, H, W, numGroups)) return;
    BROTENSOR_API_TRY
        brotensor::resblock_forward(*toTensor(X), *toTensor(gamma1), *toTensor(beta1),
                                    *toTensor(W1), tensorFromValue(b1_bits), tensorFromValue(t_emb_shift_bits),
                                    *toTensor(gamma2), *toTensor(beta2),
                                    *toTensor(W2), tensorFromValue(b2_bits),
                                    tensorFromValue(Wskip_bits), tensorFromValue(bskip_bits),
                                    N, C_in, C_out, H, W, numGroups, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_resblockBackward(void* X, void* gamma1, void* beta1, void* W1, uint64_t b1_bits, uint64_t t_emb_shift_bits, void* gamma2, void* beta2, void* W2, uint64_t b2_bits, uint64_t Wskip_bits, uint64_t bskip_bits, int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t W, int32_t numGroups, double eps, void* dY, void* dX, void* dGamma1, void* dBeta1, void* dW1, uint64_t db1_bits, uint64_t dt_emb_shift_bits, void* dGamma2, void* dBeta2, void* dW2, uint64_t db2_bits, uint64_t dWskip_bits, uint64_t dbskip_bits) {
    const char* L = "resblockBackward";
    if (!need(L, {X, gamma1, beta1, W1, gamma2, beta2, W2, dY, dX, dGamma1, dBeta1, dW1, dGamma2, dBeta2, dW2})) return;
    const Tensor* x = T(X);
    if (!needResblock(L, x, gamma1, beta1, W1, b1_bits, t_emb_shift_bits, gamma2, beta2, W2, b2_bits,
                      Wskip_bits, bskip_bits, N, C_in, C_out, H, W, numGroups)) return;
    // The gradients mirror their parameters.
    if (!needOperands(L, x, {{"dY", T(dY), elems({N, C_out, H, W})},
                             {"dGamma1", T(dGamma1), C_in}, {"dBeta1", T(dBeta1), C_in},
                             {"dW1", T(dW1), elems({C_out, C_in, 9})}, {"db1", tensorFromValue(db1_bits), C_out},
                             {"dt_emb_shift", tensorFromValue(dt_emb_shift_bits), C_out},
                             {"dGamma2", T(dGamma2), C_out}, {"dBeta2", T(dBeta2), C_out},
                             {"dW2", T(dW2), elems({C_out, C_out, 9})}, {"db2", tensorFromValue(db2_bits), C_out},
                             {"dWskip", tensorFromValue(dWskip_bits), elems({C_out, C_in})},
                             {"dbskip", tensorFromValue(dbskip_bits), C_out}})) return;
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
    BROTENSOR_API_CATCH(L)
}

} // extern "C"
