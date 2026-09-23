// native_tensor_int8.cpp — bodies for the INT8 / GGUF-quant group, restored
// from the QuickJS binding (src/js/tensor_bindings_int8.cpp).
//
// Every op except quantizeInt8PerRowHost is GPU-only (see the header): on a
// CPU-only build the dispatcher throws "not implemented on CPU", the
// BROTENSOR_API_CATCH below records it, and the JS wrapper rethrows it as an
// Error. Nothing here has a CPU fallback.

#include "native_tensor_int8_decl.h"
#include "api_internal.h"
#include "native_register.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace brotensor::api;

namespace {

// The last quantizeInt8PerRowHost result on this thread. The weights cross as
// the native's `i8[]` return, the scales through the companion native — one
// quantise pass, two typed arrays, as the old `{ weights, scales }` object.
// Both are copied out by the runtime (release == nullptr), so the vectors may
// be reused by the next call.
thread_local std::vector<int8_t> tl_quant_weights;
thread_local std::vector<float> tl_quant_scales;

void clearBuffer(bronze_native_buffer* out) {
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
}

using brotensor::Tensor;

Tensor* T(void* p) { return toTensor(p); }

// The device ops check weight dtypes and (out,in) shapes themselves; what
// they take on trust is the activation extent the caller's dims imply, the
// optional bias lengths and the raw `const float*` masks. Those are checked
// here.

// An optional per-output bias of `out` elements carried in a kDyn slot.
bool needBias(const char* L, const char* name, uint64_t bits, int64_t out) {
    const Tensor* b = tensorFromValue(bits);
    return !b || needElems(L, name, b, out);
}

} // namespace

extern "C" {

// ---- GGUF k-quant dequant --------------------------------------------------

void bro_tensor_dequantQ4kToFp16(void* W_q, void* W_fp16) {
    if (!need("dequantQ4kToFp16", {W_q, W_fp16})) return;
    BROTENSOR_API_TRY
        brotensor::dequant_q4k_to_fp16(*toTensor(W_q), *toTensor(W_fp16));
    BROTENSOR_API_CATCH("dequantQ4kToFp16")
}

void bro_tensor_dequantQ6kToFp16(void* W_q, void* W_fp16) {
    if (!need("dequantQ6kToFp16", {W_q, W_fp16})) return;
    BROTENSOR_API_TRY
        brotensor::dequant_q6k_to_fp16(*toTensor(W_q), *toTensor(W_fp16));
    BROTENSOR_API_CATCH("dequantQ6kToFp16")
}

void bro_tensor_dequantQ8_0ToFp16(void* W_q, void* W_fp16) {
    if (!need("dequantQ8_0ToFp16", {W_q, W_fp16})) return;
    BROTENSOR_API_TRY
        brotensor::dequant_q8_0_to_fp16(*toTensor(W_q), *toTensor(W_fp16));
    BROTENSOR_API_CATCH("dequantQ8_0ToFp16")
}

// ---- GGUF k-quant weight-only linear ---------------------------------------

void bro_tensor_linearForwardQ4kFp16(void* W_q, uint64_t bias_bits, void* x, void* y) {
    if (!need("linearForwardQ4kFp16", {W_q, x, y})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_q4k_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("linearForwardQ4kFp16")
}

void bro_tensor_linearForwardQ6kFp16(void* W_q, uint64_t bias_bits, void* x, void* y) {
    if (!need("linearForwardQ6kFp16", {W_q, x, y})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_q6k_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("linearForwardQ6kFp16")
}

void bro_tensor_linearForwardQ8_0Fp16(void* W_q, uint64_t bias_bits, void* x, void* y) {
    if (!need("linearForwardQ8_0Fp16", {W_q, x, y})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_q8_0_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("linearForwardQ8_0Fp16")
}

void bro_tensor_linearForwardBatchedQ4kFp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatchedQ4kFp16", {W_q, X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_q4k_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatchedQ4kFp16")
}

void bro_tensor_linearForwardBatchedQ6kFp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatchedQ6kFp16", {W_q, X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_q6k_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatchedQ6kFp16")
}

void bro_tensor_linearForwardBatchedQ8_0Fp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatchedQ8_0Fp16", {W_q, X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_q8_0_fp16(*toTensor(W_q), tensorFromValue(bias_bits), *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatchedQ8_0Fp16")
}

// ---- W8A16 host quantiser --------------------------------------------------

void bro_tensor_quantizeInt8PerRowHost(const uint16_t* W_fp16, uint32_t W_fp16_len,
                                       int32_t out, int32_t in, bronze_native_buffer* weights_out) {
    if (!weights_out) return;
    clearBuffer(weights_out);
    tl_quant_weights.clear();
    tl_quant_scales.clear();
    if (!W_fp16) {
        setError("quantizeInt8PerRowHost: W_fp16 must be a Uint16Array (binary16 bit pattern)");
        return;
    }
    if (out <= 0 || in <= 0) {
        setError("quantizeInt8PerRowHost: out, in must be > 0");
        return;
    }
    const size_t count = static_cast<size_t>(out) * static_cast<size_t>(in);
    if (static_cast<size_t>(W_fp16_len) < count) {
        setError("quantizeInt8PerRowHost: W_fp16 view too small for (out, in)");
        return;
    }
    BROTENSOR_API_TRY
        tl_quant_weights.resize(count);
        tl_quant_scales.resize(static_cast<size_t>(out));
        brotensor::quantize_int8_per_row_host(W_fp16, out, in,
                                              tl_quant_weights.data(), tl_quant_scales.data());
        weights_out->data = tl_quant_weights.data();
        weights_out->length = static_cast<uint32_t>(tl_quant_weights.size());
    BROTENSOR_API_CATCH("quantizeInt8PerRowHost")
}

void bro_tensor_quantizeInt8PerRowHostScales(bronze_native_buffer* scales_out) {
    if (!scales_out) return;
    clearBuffer(scales_out);
    if (tl_quant_scales.empty()) return;
    scales_out->data = tl_quant_scales.data();
    scales_out->length = static_cast<uint32_t>(tl_quant_scales.size());
}

// ---- W8A16 dense -----------------------------------------------------------

void bro_tensor_matmulInt8wFp16(void* W_int8, void* scales, void* X, void* Y) {
    if (!need("matmulInt8wFp16", {W_int8, scales, X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::matmul_int8w_fp16(*toTensor(W_int8), *toTensor(scales), *toTensor(X), *toTensor(Y));
    BROTENSOR_API_CATCH("matmulInt8wFp16")
}

void bro_tensor_linearForwardBatchedInt8wFp16(void* W_int8, void* scales, uint64_t bias_bits,
                                              void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatchedInt8wFp16", {W_int8, scales, X_BD, Y_BD}) ||
        !needBias("linearForwardBatchedInt8wFp16", "bias", bias_bits, T(W_int8)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_int8w_fp16(*toTensor(W_int8), *toTensor(scales),
                                                     tensorFromValue(bias_bits),
                                                     *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatchedInt8wFp16")
}

// ---- W8A16 convolution -----------------------------------------------------

void bro_tensor_conv2dInt8wFp16Forward(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t H, int32_t Win,
                                       int32_t C_out, int32_t kH, int32_t kW,
                                       int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                       int32_t dH, int32_t dW, int32_t groups, void* Y) {
    const char* L = "conv2dInt8wFp16Forward";
    if (!need(L, {X, W_int8, scales, Y})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, Win, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N}) || !needElems(L, "X", T(X), elems({N, C_in, H, Win})) ||
        !needBias(L, "bias", bias_bits, C_out)) return;
    BROTENSOR_API_TRY
        brotensor::conv2d_int8w_fp16_forward(*toTensor(X), *toTensor(W_int8), *toTensor(scales),
                                             tensorFromValue(bias_bits),
                                             N, C_in, H, Win, C_out, kH, kW,
                                             sH, sW, pH, pW, dH, dW, groups, *toTensor(Y));
    BROTENSOR_API_CATCH("conv2dInt8wFp16Forward")
}

void bro_tensor_conv3dInt8wFp16Forward(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t T, int32_t H, int32_t Win,
                                       int32_t C_out, int32_t kT, int32_t kH, int32_t kW,
                                       int32_t sT, int32_t sH, int32_t sW,
                                       int32_t pT, int32_t pH, int32_t pW,
                                       int32_t dT, int32_t dH, int32_t dW,
                                       int32_t groups, void* Y) {
    const char* L = "conv3dInt8wFp16Forward";
    if (!need(L, {X, W_int8, scales, Y})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, Win, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needPositive(L, {kT, sT, dT}) || !needNonNegative(L, {N, T, pT})) return;
    if (!needElems(L, "X", toTensor(X), elems({N, C_in, T, H, Win})) || !needBias(L, "bias", bias_bits, C_out)) return;
    BROTENSOR_API_TRY
        brotensor::conv3d_int8w_fp16_forward(*toTensor(X), *toTensor(W_int8), *toTensor(scales),
                                             tensorFromValue(bias_bits),
                                             N, C_in, T, H, Win, C_out, kT, kH, kW,
                                             sT, sH, sW, pT, pH, pW, dT, dH, dW,
                                             groups, *toTensor(Y));
    BROTENSOR_API_CATCH("conv3dInt8wFp16Forward")
}

// ---- W8A16 diffusion resblock ----------------------------------------------

void bro_tensor_resblockForwardInt8wFp16(void* X, void* gamma1, void* beta1,
                                         void* W1_int8, void* s1, uint64_t b1_bits,
                                         uint64_t t_emb_shift_bits,
                                         void* gamma2, void* beta2,
                                         void* W2_int8, void* s2, uint64_t b2_bits,
                                         uint64_t Wskip_bits, uint64_t sskip_bits, uint64_t bskip_bits,
                                         int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t Win,
                                         int32_t numGroups, double eps, void* Y) {
    const char* L = "resblockForwardInt8wFp16";
    if (!need(L, {X, gamma1, beta1, W1_int8, s1, gamma2, beta2, W2_int8, s2, Y})) return;
    if (!needPositive(L, {numGroups}) || !needNonNegative(L, {N, C_in, C_out, H, Win})) return;
    if (C_in != C_out && !tensorFromValue(Wskip_bits)) {
        setError(std::string(L) + ": C_in != C_out needs a Wskip projection");
        return;
    }
    // The op checks dtypes, numGroups and W1/W2 shapes; the FP16 operands'
    // extents and the optional tensors are checked here.
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C_in, H, Win})},
                                {"gamma1", T(gamma1), C_in}, {"beta1", T(beta1), C_in},
                                {"b1", tensorFromValue(b1_bits), C_out},
                                {"t_emb_shift", tensorFromValue(t_emb_shift_bits), C_out},
                                {"gamma2", T(gamma2), C_out}, {"beta2", T(beta2), C_out},
                                {"b2", tensorFromValue(b2_bits), C_out},
                                {"bskip", tensorFromValue(bskip_bits), C_out}})) return;
    if (!needMask(L, T(s1), C_out, "s1") || !needMask(L, T(s2), C_out, "s2")) return;
    if (const Tensor* ws = tensorFromValue(Wskip_bits)) {
        const Tensor* ss = tensorFromValue(sskip_bits);
        if (!ss) { setError(std::string(L) + ": Wskip_int8 needs sskip scales"); return; }
        if (!needElems(L, "Wskip_int8", ws, elems({C_out, C_in})) || !needMask(L, ss, C_out, "sskip")) return;
    }
    BROTENSOR_API_TRY
        brotensor::resblock_forward_int8w_fp16(*toTensor(X), *toTensor(gamma1), *toTensor(beta1),
                                               *toTensor(W1_int8), *toTensor(s1),
                                               tensorFromValue(b1_bits), tensorFromValue(t_emb_shift_bits),
                                               *toTensor(gamma2), *toTensor(beta2),
                                               *toTensor(W2_int8), *toTensor(s2),
                                               tensorFromValue(b2_bits),
                                               tensorFromValue(Wskip_bits), tensorFromValue(sskip_bits),
                                               tensorFromValue(bskip_bits),
                                               N, C_in, C_out, H, Win, numGroups,
                                               static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("resblockForwardInt8wFp16")
}

// ---- W8A16 flash-attention triplet -----------------------------------------

void bro_tensor_flashAttentionProjectKvInt8wFp16(void* ctx, void* Wk_int8, void* sk, uint64_t bk_bits,
                                                 void* Wv_int8, void* sv, uint64_t bv_bits,
                                                 void* K_out, void* V_out) {
    const char* L = "flashAttentionProjectKvInt8wFp16";
    if (!need(L, {ctx, Wk_int8, sk, Wv_int8, sv, K_out, V_out})) return;
    if (!needBias(L, "bk", bk_bits, T(Wk_int8)->rows) || !needBias(L, "bv", bv_bits, T(Wv_int8)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_project_kv_int8w_fp16(*toTensor(ctx), *toTensor(Wk_int8), *toTensor(sk),
                                                         tensorFromValue(bk_bits),
                                                         *toTensor(Wv_int8), *toTensor(sv),
                                                         tensorFromValue(bv_bits),
                                                         *toTensor(K_out), *toTensor(V_out));
    BROTENSOR_API_CATCH("flashAttentionProjectKvInt8wFp16")
}

void bro_tensor_flashAttentionQWithKvCachedInt8wFp16(void* X, void* K, void* V,
                                                     void* Wq_int8, void* sq, uint64_t bq_bits,
                                                     void* Wo_int8, void* so, uint64_t bo_bits,
                                                     uint64_t mask_bits, int32_t numHeads,
                                                     bool causal, void* O) {
    const char* L = "flashAttentionQWithKvCachedInt8wFp16";
    if (!need(L, {X, K, V, Wq_int8, sq, Wo_int8, so, O})) return;
    if (!needBias(L, "bq", bq_bits, T(Wq_int8)->rows) || !needBias(L, "bo", bo_bits, T(Wo_int8)->rows) ||
        !needMask(L, tensorFromValue(mask_bits), T(K)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_q_with_kv_cached_int8w_fp16(*toTensor(X), *toTensor(K), *toTensor(V),
                                                               *toTensor(Wq_int8), *toTensor(sq),
                                                               tensorFromValue(bq_bits),
                                                               *toTensor(Wo_int8), *toTensor(so),
                                                               tensorFromValue(bo_bits),
                                                               maskPtr(mask_bits), numHeads, causal,
                                                               *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionQWithKvCachedInt8wFp16")
}

void bro_tensor_flashAttentionQkvoInt8wFp16(void* X, uint64_t Ctx_bits,
                                            void* Wq_int8, void* sq, uint64_t bq_bits,
                                            void* Wk_int8, void* sk, uint64_t bk_bits,
                                            void* Wv_int8, void* sv, uint64_t bv_bits,
                                            void* Wo_int8, void* so, uint64_t bo_bits,
                                            uint64_t mask_bits, int32_t numHeads,
                                            bool causal, void* O) {
    const char* L = "flashAttentionQkvoInt8wFp16";
    if (!need(L, {X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv, Wo_int8, so, O})) return;
    const Tensor* kvSrc = tensorFromValue(Ctx_bits) ? tensorFromValue(Ctx_bits) : T(X);
    if (!needBias(L, "bq", bq_bits, T(Wq_int8)->rows) || !needBias(L, "bk", bk_bits, T(Wk_int8)->rows) ||
        !needBias(L, "bv", bv_bits, T(Wv_int8)->rows) || !needBias(L, "bo", bo_bits, T(Wo_int8)->rows) ||
        !needMask(L, tensorFromValue(mask_bits), kvSrc->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_qkvo_int8w_fp16(*toTensor(X), tensorFromValue(Ctx_bits),
                                                   *toTensor(Wq_int8), *toTensor(sq), tensorFromValue(bq_bits),
                                                   *toTensor(Wk_int8), *toTensor(sk), tensorFromValue(bk_bits),
                                                   *toTensor(Wv_int8), *toTensor(sv), tensorFromValue(bv_bits),
                                                   *toTensor(Wo_int8), *toTensor(so), tensorFromValue(bo_bits),
                                                   maskPtr(mask_bits), numHeads, causal, *toTensor(O));
    BROTENSOR_API_CATCH("flashAttentionQkvoInt8wFp16")
}

// ---- W8A16 T5-style bias self-attention ------------------------------------

void bro_tensor_selfAttentionBiasInt8wFp16(void* X,
                                           void* Wq_int8, void* sq,
                                           void* Wk_int8, void* sk,
                                           void* Wv_int8, void* sv,
                                           void* Wo_int8, void* so,
                                           uint64_t mask_bits, uint64_t attnBias_bits,
                                           int32_t numHeads, double scale, void* O) {
    if (!need("selfAttentionBiasInt8wFp16", {X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv, Wo_int8, so, O}) ||
        !needMask("selfAttentionBiasInt8wFp16", tensorFromValue(mask_bits), T(X)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::self_attention_bias_int8w_fp16(*toTensor(X),
                                                  *toTensor(Wq_int8), *toTensor(sq),
                                                  *toTensor(Wk_int8), *toTensor(sk),
                                                  *toTensor(Wv_int8), *toTensor(sv),
                                                  *toTensor(Wo_int8), *toTensor(so),
                                                  maskPtr(mask_bits), tensorFromValue(attnBias_bits),
                                                  numHeads, static_cast<float>(scale), *toTensor(O));
    BROTENSOR_API_CATCH("selfAttentionBiasInt8wFp16")
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_int8(std::string* error) {
    using namespace brotensor::api::reg;
    bool ok =
        fn("__bro_native.tensor.dequantQ4kToFp16", p(&bro_tensor_dequantQ4kToFp16), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.dequantQ6kToFp16", p(&bro_tensor_dequantQ6kToFp16), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.dequantQ8_0ToFp16", p(&bro_tensor_dequantQ8_0ToFp16), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardQ4kFp16", p(&bro_tensor_linearForwardQ4kFp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardQ6kFp16", p(&bro_tensor_linearForwardQ6kFp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardQ8_0Fp16", p(&bro_tensor_linearForwardQ8_0Fp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedQ4kFp16", p(&bro_tensor_linearForwardBatchedQ4kFp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedQ6kFp16", p(&bro_tensor_linearForwardBatchedQ6kFp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedQ8_0Fp16", p(&bro_tensor_linearForwardBatchedQ8_0Fp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.quantizeInt8PerRowHost", p(&bro_tensor_quantizeInt8PerRowHost), "i8[]", {"u16[]", "i32", "i32"}, error) &&
        fn("__bro_native.tensor.quantizeInt8PerRowHostScales", p(&bro_tensor_quantizeInt8PerRowHostScales), "f32[]", {}, error) &&
        fn("__bro_native.tensor.matmulInt8wFp16", p(&bro_tensor_matmulInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedInt8wFp16", p(&bro_tensor_linearForwardBatchedInt8wFp16), "void", {kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.conv2dInt8wFp16Forward", p(&bro_tensor_conv2dInt8wFp16Forward), "void", {kTensorCls, kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv3dInt8wFp16Forward", p(&bro_tensor_conv3dInt8wFp16Forward), "void", {kTensorCls, kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.resblockForwardInt8wFp16", p(&bro_tensor_resblockForwardInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, kDyn, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionProjectKvInt8wFp16", p(&bro_tensor_flashAttentionProjectKvInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionQWithKvCachedInt8wFp16", p(&bro_tensor_flashAttentionQWithKvCachedInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kDyn, "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionQkvoInt8wFp16", p(&bro_tensor_flashAttentionQkvoInt8wFp16), "void", {kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kTensorCls, kTensorCls, kDyn, kDyn, "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionBiasInt8wFp16", p(&bro_tensor_selfAttentionBiasInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, "i32", "f64", kTensorCls}, error);
    return ok;
}

} // namespace brotensor::api
