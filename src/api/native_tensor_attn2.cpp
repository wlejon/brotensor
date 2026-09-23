// Native bodies for the second attention family (see
// native_tensor_attn2_decl.h): decomposed 2D rel-pos self-attention (global +
// windowed), packed variable-length flash attention, the gated delta rule and
// M-RoPE. js/tensor_attn2.js checks argument types; a body checks the index
// streams' values (see IdxStream), calls, and records any thrown violation.

#include "native_tensor_attn2_decl.h"
#include "api_internal.h"
#include "native_register.h"
#include "native_tensor_extra_util.h"

#include <string>
#include <vector>

using namespace brotensor::api;
using brotensor::Tensor;

namespace {

Tensor* T(void* p) { return toTensor(p); }

// The varlen / M-RoPE ops take `const int32_t*` streams (device pointers on
// CUDA/Metal, host on CPU) whose VALUES address rows of the other operands,
// and the kernels trust them. A stream is an optional GpuTensor holding INT32
// (uploadInt32) or whole-number FP32 (upload); the native reads it back,
// checks every value, and hands the op INT32 storage — the tensor's own, or
// an INT32 copy held in `scratch` (reading FP32 storage as INT32 would turn
// 1.0f into 1065353216).
struct IdxStream {
    std::vector<int32_t> vals;
    Tensor scratch;
    const int32_t* ptr = nullptr;
};

// Reads `n` values of the stream in `bits` into `s`; a null stream leaves
// s.ptr null (the op decides whether that is allowed).
bool readStream(const char* L, const char* name, uint64_t bits, int64_t n, IdxStream& s) {
    const Tensor* t = tensorFromValue(bits);
    if (!t) return true;
    if (!extra::hostInt32(L, name, t, n, s.vals)) return false;
    s.ptr = static_cast<const int32_t*>(extra::int32Operand(*t, s.vals, s.scratch).data);
    return true;
}

// cu_seqlens: batch+1 non-decreasing prefix sums inside [0, total], each
// sequence at most maxLen long (the GPU kernels size their tiles from it).
bool needCuSeq(const char* L, const char* name, uint64_t bits, int32_t batch, int64_t total, int32_t maxLen,
               IdxStream& s) {
    if (batch <= 0) return true;
    if (!tensorFromValue(bits)) {
        setError(std::string(L) + ": " + name + " is required when batch > 0");
        return false;
    }
    if (!readStream(L, name, bits, static_cast<int64_t>(batch) + 1, s)) return false;
    if (!extra::needIndicesIn(L, name, s.vals, static_cast<int64_t>(batch) + 1, 0, total + 1)) return false;
    for (int32_t b = 0; b < batch; ++b) {
        const int64_t len = static_cast<int64_t>(s.vals[b + 1]) - s.vals[b];
        if (len < 0 || len > maxLen) {
            setError(std::string(L) + ": " + name + " sequence " + std::to_string(b) + " has length " +
                     std::to_string(len) + " (must be in [0, " + std::to_string(maxLen) + "])");
            return false;
        }
    }
    return true;
}

// An M-RoPE position stream for an axis of width d: length L, every value a
// row of that axis's cos/sin tables.
bool needPosStream(const char* L, const char* name, uint64_t bits, int32_t d, int64_t rows, const Tensor* cosT,
                   const Tensor* sinT, IdxStream& s) {
    if (d <= 0) return true;
    if (!tensorFromValue(bits)) {
        setError(std::string(L) + ": " + name + " is required when its axis width is > 0");
        return false;
    }
    if (!readStream(L, name, bits, rows, s)) return false;
    const int64_t maxPos = cosT->rows < sinT->rows ? cosT->rows : sinT->rows;
    return extra::needIndicesIn(L, name, s.vals, rows, 0, maxPos);
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
    const char* L = "flashAttentionVarlenForward";
    if (!need(L, {Q, K, V, O}) || !needNonNegative(L, {batch, maxQ, maxK})) return;
    BROTENSOR_API_TRY
        IdxStream cq, ck;
        if (!needCuSeq(L, "cuSeqQ", cuSeqQ_bits, batch, T(Q)->rows, maxQ, cq) ||
            !needCuSeq(L, "cuSeqK", cuSeqK_bits, batch, T(K)->rows, maxK, ck)) return;
        brotensor::flash_attention_varlen_forward(*toTensor(Q), *toTensor(K), *toTensor(V),
                                                  cq.ptr, ck.ptr,
                                                  batch, maxQ, maxK, numHeads, headDim, causal,
                                                  *toTensor(O));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionVarlenBackward(void* Q, void* K, void* V, void* O, void* dO,
                                             uint64_t cuSeqQ_bits, uint64_t cuSeqK_bits,
                                             int32_t batch, int32_t maxQ, int32_t maxK,
                                             int32_t numHeads, int32_t headDim, bool causal,
                                             void* dQ, void* dK, void* dV) {
    const char* L = "flashAttentionVarlenBackward";
    if (!need(L, {Q, K, V, O, dO, dQ, dK, dV}) || !needNonNegative(L, {batch, maxQ, maxK})) return;
    if (!needPair(L, "O", T(O), T(Q)) || !needPair(L, "dO", T(dO), T(Q))) return;
    BROTENSOR_API_TRY
        IdxStream cq, ck;
        if (!needCuSeq(L, "cuSeqQ", cuSeqQ_bits, batch, T(Q)->rows, maxQ, cq) ||
            !needCuSeq(L, "cuSeqK", cuSeqK_bits, batch, T(K)->rows, maxK, ck)) return;
        brotensor::flash_attention_varlen_backward(*toTensor(Q), *toTensor(K), *toTensor(V),
                                                   *toTensor(O), *toTensor(dO),
                                                   cq.ptr, ck.ptr,
                                                   batch, maxQ, maxK, numHeads, headDim, causal,
                                                   *toTensor(dQ), *toTensor(dK), *toTensor(dV));
    BROTENSOR_API_CATCH(L)
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
    const char* L = "ropeApplyMrope";
    if (!need(L, {X, cosT, sinT, cosH, sinH, cosW, sinW, Y})) return;
    const int64_t rows = T(X)->rows;
    BROTENSOR_API_TRY
        IdxStream pt, ph, pw;
        if (!needPosStream(L, "posT", posT_bits, d_t, rows, T(cosT), T(sinT), pt) ||
            !needPosStream(L, "posH", posH_bits, d_h, rows, T(cosH), T(sinH), ph) ||
            !needPosStream(L, "posW", posW_bits, d_w, rows, T(cosW), T(sinW), pw)) return;
        brotensor::rope_apply_mrope(*toTensor(X),
                                    *toTensor(cosT), *toTensor(sinT),
                                    *toTensor(cosH), *toTensor(sinH),
                                    *toTensor(cosW), *toTensor(sinW),
                                    pt.ptr, ph.ptr, pw.ptr,
                                    headDim, numHeads, d_t, d_h, d_w, *toTensor(Y));
    BROTENSOR_API_CATCH(L)
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
