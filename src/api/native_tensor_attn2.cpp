// Native bodies for the second attention family (see
// native_tensor_attn2_decl.h): decomposed 2D rel-pos self-attention (global +
// windowed), packed variable-length flash attention, the gated delta rule and
// M-RoPE. js/tensor_attn2.js has already checked every argument, so a body
// only has to unwrap, call and record any thrown contract violation.

#include "native_tensor_attn2_decl.h"
#include "api_internal.h"
#include "native_register.h"

using namespace brotensor::api;

namespace {

// The `const int32_t*` convention of the varlen / M-RoPE ops: the raw storage
// of an optional GpuTensor viewed as INT32 (a device pointer on CUDA/Metal, a
// host pointer on CPU), or nullptr. Mirrors maskPtr() for FP32 masks and
// embeddingLookupForward's idx handling.
inline const int32_t* idxPtr(uint64_t bits) {
    auto* t = tensorFromValue(bits);
    return t ? static_cast<const int32_t*>(t->data) : nullptr;
}

} // namespace

extern "C" {

// ---- SAM / ViTDet decomposed 2D rel-pos self-attention ---------------------

void bro_tensor_selfAttentionDecomposedRelPosForward(void* X, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits,
                                                     void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits,
                                                     void* relPosH, void* relPosW,
                                                     int32_t numHeads, int32_t gridH, int32_t gridW, double scale,
                                                     void* O) {
    if (!need("selfAttentionDecomposedRelPosForward", {X, Wq, Wk, Wv, Wo, relPosH, relPosW, O})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_decomposed_rel_pos_forward(
            *toTensor(X),
            *toTensor(Wq), tensorFromValue(bq_bits),
            *toTensor(Wk), tensorFromValue(bk_bits),
            *toTensor(Wv), tensorFromValue(bv_bits),
            *toTensor(Wo), tensorFromValue(bo_bits),
            *toTensor(relPosH), *toTensor(relPosW),
            numHeads, gridH, gridW, static_cast<float>(scale), *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionDecomposedRelPosForward")
}

void bro_tensor_selfAttentionDecomposedRelPosWindowedForward(void* X, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits,
                                                             void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits,
                                                             void* relPosH, void* relPosW,
                                                             int32_t numHeads, int32_t gridH, int32_t gridW,
                                                             int32_t window, double scale, void* O) {
    if (!need("selfAttentionDecomposedRelPosWindowedForward", {X, Wq, Wk, Wv, Wo, relPosH, relPosW, O})) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_decomposed_rel_pos_windowed_forward(
            *toTensor(X),
            *toTensor(Wq), tensorFromValue(bq_bits),
            *toTensor(Wk), tensorFromValue(bk_bits),
            *toTensor(Wv), tensorFromValue(bv_bits),
            *toTensor(Wo), tensorFromValue(bo_bits),
            *toTensor(relPosH), *toTensor(relPosW),
            numHeads, gridH, gridW, window, static_cast<float>(scale), *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionDecomposedRelPosWindowedForward")
}

// ---- packed variable-length flash attention (Qwen-VL window attention) -----

void bro_tensor_flashAttentionVarlenForward(void* Q, void* K, void* V, uint64_t cuSeqQ_bits, uint64_t cuSeqK_bits,
                                            int32_t batch, int32_t maxQ, int32_t maxK,
                                            int32_t numHeads, int32_t headDim, bool causal, void* O) {
    if (!need("flashAttentionVarlenForward", {Q, K, V, O})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_varlen_forward(*toTensor(Q), *toTensor(K), *toTensor(V),
                                                  idxPtr(cuSeqQ_bits), idxPtr(cuSeqK_bits),
                                                  batch, maxQ, maxK, numHeads, headDim, causal,
                                                  *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionVarlenForward")
}

void bro_tensor_flashAttentionVarlenBackward(void* Q, void* K, void* V, void* O, void* dO,
                                             uint64_t cuSeqQ_bits, uint64_t cuSeqK_bits,
                                             int32_t batch, int32_t maxQ, int32_t maxK,
                                             int32_t numHeads, int32_t headDim, bool causal,
                                             void* dQ, void* dK, void* dV) {
    if (!need("flashAttentionVarlenBackward", {Q, K, V, O, dO, dQ, dK, dV})) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_varlen_backward(*toTensor(Q), *toTensor(K), *toTensor(V),
                                                   *toTensor(O), *toTensor(dO),
                                                   idxPtr(cuSeqQ_bits), idxPtr(cuSeqK_bits),
                                                   batch, maxQ, maxK, numHeads, headDim, causal,
                                                   *toTensor(dQ), *toTensor(dK), *toTensor(dV));
    BROTENSOR_API_CATCH("flashAttentionVarlenBackward")
}

// ---- gated delta rule (linear attention — Qwen3-Next) ---------------------

void bro_tensor_gatedDeltaRuleChunked(void* Q, void* K, void* V, void* aRaw, void* beta, void* logA,
                                      int32_t numHeads, int32_t d_k, int32_t d_v, void* state, void* O) {
    if (!need("gatedDeltaRuleChunked", {Q, K, V, aRaw, beta, logA, state, O})) return;
    BROTENSOR_API_TRY
        brotensor::gated_delta_rule_chunked(*toTensor(Q), *toTensor(K), *toTensor(V),
                                            *toTensor(aRaw), *toTensor(beta), *toTensor(logA),
                                            numHeads, d_k, d_v, *toTensor(state), *toTensor(O));
    BROTENSOR_API_CATCH("gatedDeltaRuleChunked")
}

void bro_tensor_gatedDeltaRuleStep(void* Q, void* K, void* V, void* aRaw, void* beta, void* logA,
                                   int32_t numHeads, int32_t d_k, int32_t d_v, void* state, void* O) {
    if (!need("gatedDeltaRuleStep", {Q, K, V, aRaw, beta, logA, state, O})) return;
    BROTENSOR_API_TRY
        brotensor::gated_delta_rule_step(*toTensor(Q), *toTensor(K), *toTensor(V),
                                         *toTensor(aRaw), *toTensor(beta), *toTensor(logA),
                                         numHeads, d_k, d_v, *toTensor(state), *toTensor(O));
    BROTENSOR_API_CATCH("gatedDeltaRuleStep")
}

// ---- M-RoPE (Qwen-VL multimodal rotary) -----------------------------------

void bro_tensor_ropeApplyMrope(void* X, void* cosT, void* sinT, void* cosH, void* sinH, void* cosW, void* sinW,
                               uint64_t posT_bits, uint64_t posH_bits, uint64_t posW_bits,
                               int32_t headDim, int32_t numHeads, int32_t d_t, int32_t d_h, int32_t d_w,
                               void* Y) {
    if (!need("ropeApplyMrope", {X, cosT, sinT, cosH, sinH, cosW, sinW, Y})) return;
    BROTENSOR_API_TRY
        brotensor::rope_apply_mrope(*toTensor(X),
                                    *toTensor(cosT), *toTensor(sinT),
                                    *toTensor(cosH), *toTensor(sinH),
                                    *toTensor(cosW), *toTensor(sinW),
                                    idxPtr(posT_bits), idxPtr(posH_bits), idxPtr(posW_bits),
                                    headDim, numHeads, d_t, d_h, d_w, *toTensor(Y));
    BROTENSOR_API_CATCH("ropeApplyMrope")
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_attn2(std::string* error) {
    using namespace brotensor::api::reg;
    return
        fn("__bro_native.tensor.selfAttentionDecomposedRelPosForward", p(&bro_tensor_selfAttentionDecomposedRelPosForward), "void",
           {kTensorCls, kTensorCls, kDyn, kTensorCls, kDyn, kTensorCls, kDyn, kTensorCls, kDyn,
            kTensorCls, kTensorCls, "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionDecomposedRelPosWindowedForward", p(&bro_tensor_selfAttentionDecomposedRelPosWindowedForward), "void",
           {kTensorCls, kTensorCls, kDyn, kTensorCls, kDyn, kTensorCls, kDyn, kTensorCls, kDyn,
            kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionVarlenForward", p(&bro_tensor_flashAttentionVarlenForward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, "i32", "i32", "i32", "i32", "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionVarlenBackward", p(&bro_tensor_flashAttentionVarlenBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn,
            "i32", "i32", "i32", "i32", "i32", "bool", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gatedDeltaRuleChunked", p(&bro_tensor_gatedDeltaRuleChunked), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls,
            "i32", "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gatedDeltaRuleStep", p(&bro_tensor_gatedDeltaRuleStep), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls,
            "i32", "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeApplyMrope", p(&bro_tensor_ropeApplyMrope), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls,
            kDyn, kDyn, kDyn, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error);
}

} // namespace brotensor::api
