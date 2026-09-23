// native_tensor_batched.cpp — bodies for the restored "batched" group.
//
// Ported from the QuickJS binding (bro-quickjs-oracle
// src/js/tensor_bindings.cpp, `nngpu::` namespace) onto the op signatures that
// exist today under include/brotensor/ops/. A native is called straight from
// generated code, so nothing may unwind out of one: every body that reaches
// into brotensor is wrapped in BROTENSOR_API_TRY / BROTENSOR_API_CATCH, which
// records the message for js/tensor_batched.js to throw after the call.

#include "native_tensor_batched_decl.h"
#include "api_internal.h"
#include "native_register.h"
#include "native_tensor_extra_util.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace brotensor::api;
using brotensor::Tensor;
using brotensor::Dtype;

// gather_rows / scatter_rows_add take Idx as an (M,1) INT32 tensor whose
// VALUES address rows on the device, unchecked by the kernels. The native
// reads the indices back, range-checks them, and hands the op an INT32
// operand (the tensor itself, or an INT32 copy of a whole-number FP32 one —
// GpuTensor.prototype.upload always lands FP32). See native_tensor_extra_util.h.

namespace {

Tensor* T(void* p) { return toTensor(p); }

// X_BD is (B, W.cols).
bool needBatchIn(const char* L, const Tensor* W, const Tensor* X) {
    if (X->cols != W->cols) {
        setError(std::string(L) + ": X_BD must be (B, " + std::to_string(W->cols) + ") to match W's in-dim");
        return false;
    }
    return true;
}

} // namespace

extern "C" {

// ---- batched forwards ------------------------------------------------------

// W may be FP32/FP16/BF16; bias, X and Y are FP32.
void bro_tensor_linearForwardBatched(void* W, void* bias, void* X_BD, void* Y_BD) {
    const char* L = "linearForwardBatched";
    if (!need(L, {W, bias, X_BD, Y_BD}) || !needBatchIn(L, T(W), T(X_BD))) return;
    if (!needMask(L, T(bias), T(W)->rows, "bias") || !needMask(L, T(X_BD), 0, "X_BD")) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched(*toTensor(W), *toTensor(bias), *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_linearForwardBatchedFp16(void* W, uint64_t bias_bits, void* X_BD, void* Y_BD) {
    const char* L = "linearForwardBatchedFp16";
    if (!need(L, {W, X_BD, Y_BD}) || !needBatchIn(L, T(W), T(X_BD))) return;
    if (!needOperands(L, T(W), {{"X_BD", T(X_BD), 0}, {"bias", tensorFromValue(bias_bits), T(W)->rows}})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_fp16(*toTensor(W), tensorFromValue(bias_bits),
                                               *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_reluForwardBatched(void* X_BD, void* Y_BD) {
    if (!need("reluForwardBatched", {X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::relu_forward_batched(*toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("reluForwardBatched")
}

void bro_tensor_tanhForwardBatched(void* X_BD, void* Y_BD) {
    if (!need("tanhForwardBatched", {X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::tanh_forward_batched(*toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("tanhForwardBatched")
}

void bro_tensor_addInplaceBatched(void* Y_BD, void* X_BD) {
    if (!need("addInplaceBatched", {Y_BD, X_BD}) || !needPair("addInplaceBatched", "X_BD", T(X_BD), T(Y_BD))) return;
    BROTENSOR_API_TRY
        brotensor::add_inplace_batched(*toTensor(Y_BD), *toTensor(X_BD));
    BROTENSOR_API_CATCH("addInplaceBatched")
}

// ---- batched-training backwards --------------------------------------------

void bro_tensor_linearBackwardBatched(void* W, void* X_BD, void* dY_BD, void* dX_BD, void* dW, void* dB) {
    const char* L = "linearBackwardBatched";
    if (!need(L, {W, X_BD, dY_BD, dX_BD, dW, dB}) || !needBatchIn(L, T(W), T(X_BD))) return;
    const Tensor* w = T(W);
    // dW / dB accumulate — "caller zeros".
    if (!needOperands(L, w, {{"X_BD", T(X_BD), 0}, {"dY_BD", T(dY_BD), elems({T(X_BD)->rows, w->rows})},
                             {"dW", T(dW), elems({w->rows, w->cols})}, {"dB", T(dB), w->rows}})) return;
    BROTENSOR_API_TRY
        brotensor::linear_backward_batched(*toTensor(W), *toTensor(X_BD), *toTensor(dY_BD),
                                           *toTensor(dX_BD), *toTensor(dW), *toTensor(dB));
    BROTENSOR_API_CATCH("linearBackwardBatched")
}

void bro_tensor_reluBackwardBatched(void* X_BD, void* dY_BD, void* dX_BD) {
    if (!need("reluBackwardBatched", {X_BD, dY_BD, dX_BD}) ||
        !needPair("reluBackwardBatched", "dY_BD", T(dY_BD), T(X_BD))) return;
    BROTENSOR_API_TRY
        brotensor::relu_backward_batched(*toTensor(X_BD), *toTensor(dY_BD), *toTensor(dX_BD));
    BROTENSOR_API_CATCH("reluBackwardBatched")
}

void bro_tensor_tanhBackwardBatched(void* Y_BD, void* dY_BD, void* dX_BD) {
    if (!need("tanhBackwardBatched", {Y_BD, dY_BD, dX_BD}) ||
        !needPair("tanhBackwardBatched", "dY_BD", T(dY_BD), T(Y_BD))) return;
    BROTENSOR_API_TRY
        brotensor::tanh_backward_batched(*toTensor(Y_BD), *toTensor(dY_BD), *toTensor(dX_BD));
    BROTENSOR_API_CATCH("tanhBackwardBatched")
}

// ---- row gather / scatter / top-k ------------------------------------------

void bro_tensor_gatherRows(void* X, void* Idx, void* Y) {
    const char* L = "gatherRows";
    if (!need(L, {X, Idx, Y})) return;
    const Tensor* idx = T(Idx);
    BROTENSOR_API_TRY
        std::vector<int32_t> vals;
        if (!extra::hostInt32(L, "Idx", idx, idx->size(), vals)) return;
        if (!extra::needIndicesIn(L, "Idx", vals, idx->size(), 0, T(X)->rows)) return;
        Tensor scratch;
        brotensor::gather_rows(*toTensor(X), extra::int32Operand(*idx, vals, scratch), *toTensor(Y));
    BROTENSOR_API_CATCH(L)
}

// dY is (M, C) with M = Idx's length; every index names one of dX's R rows.
void bro_tensor_scatterRowsAdd(void* dY, void* Idx, int32_t R, void* dX) {
    const char* L = "scatterRowsAdd";
    if (!need(L, {dY, Idx, dX}) || !needNonNegative(L, {R})) return;
    const Tensor* idx = T(Idx);
    const Tensor* dy = T(dY);
    BROTENSOR_API_TRY
        std::vector<int32_t> vals;
        if (!extra::hostInt32(L, "Idx", idx, dy->rows, vals)) return;
        if (!extra::needIndicesIn(L, "Idx", vals, dy->rows, 0, R)) return;
        Tensor scratch;
        brotensor::scatter_rows_add(*toTensor(dY), extra::int32Operand(*idx, vals, scratch), R, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_topKRows(void* X, int32_t k, void* Vals, void* Idx) {
    if (!need("topKRows", {X, Vals, Idx})) return;
    BROTENSOR_API_TRY
        brotensor::top_k_rows(*toTensor(X), k, *toTensor(Vals), *toTensor(Idx));
    BROTENSOR_API_CATCH("topKRows")
}

// ---- batched LayerNorm with training caches --------------------------------

void bro_tensor_layernormForwardBatchedWithCaches(void* X, void* gamma, void* beta, void* Y, void* Xhat, void* Mean, void* Rstd, double eps) {
    const char* L = "layernormForwardBatchedWithCaches";
    if (!need(L, {X, gamma, beta, Y, Xhat, Mean, Rstd})) return;
    if (!needOperands(L, T(X), {{"gamma", T(gamma), T(X)->cols}, {"beta", T(beta), T(X)->cols}})) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_forward_batched_with_caches(*toTensor(X), *toTensor(gamma), *toTensor(beta),
                                                         *toTensor(Y), *toTensor(Xhat),
                                                         *toTensor(Mean), *toTensor(Rstd),
                                                         static_cast<float>(eps));
    BROTENSOR_API_CATCH("layernormForwardBatchedWithCaches")
}

void bro_tensor_layernormBackwardBatchedWithCaches(void* dY, void* Xhat, void* gamma, void* Rstd, void* dX, void* dGamma, void* dBeta) {
    const char* L = "layernormBackwardBatchedWithCaches";
    if (!need(L, {dY, Xhat, gamma, Rstd, dX, dGamma, dBeta})) return;
    // dY/Xhat (R,D); gamma, dGamma, dBeta (D,) — the last two accumulate; Rstd (R,1) FP32.
    const Tensor* dy = T(dY);
    if (!needPair(L, "Xhat", T(Xhat), dy)) return;
    if (!needOperands(L, dy, {{"gamma", T(gamma), dy->cols}, {"dGamma", T(dGamma), dy->cols},
                              {"dBeta", T(dBeta), dy->cols}})) return;
    if (!needMask(L, T(Rstd), dy->rows, "Rstd")) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_backward_batched_with_caches(*toTensor(dY), *toTensor(Xhat), *toTensor(gamma),
                                                          *toTensor(Rstd), *toTensor(dX),
                                                          *toTensor(dGamma), *toTensor(dBeta));
    BROTENSOR_API_CATCH("layernormBackwardBatchedWithCaches")
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_batched(std::string* error) {
    using namespace brotensor::api::reg;
    return
        fn("__bro_native.tensor.linearForwardBatched", p(&bro_tensor_linearForwardBatched), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedFp16", p(&bro_tensor_linearForwardBatchedFp16), "void",
           {kTensorCls, kDyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.reluForwardBatched", p(&bro_tensor_reluForwardBatched), "void",
           {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.tanhForwardBatched", p(&bro_tensor_tanhForwardBatched), "void",
           {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.addInplaceBatched", p(&bro_tensor_addInplaceBatched), "void",
           {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearBackwardBatched", p(&bro_tensor_linearBackwardBatched), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.reluBackwardBatched", p(&bro_tensor_reluBackwardBatched), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.tanhBackwardBatched", p(&bro_tensor_tanhBackwardBatched), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gatherRows", p(&bro_tensor_gatherRows), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.scatterRowsAdd", p(&bro_tensor_scatterRowsAdd), "void",
           {kTensorCls, kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.topKRows", p(&bro_tensor_topKRows), "void",
           {kTensorCls, "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.layernormForwardBatchedWithCaches", p(&bro_tensor_layernormForwardBatchedWithCaches), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.layernormBackwardBatchedWithCaches", p(&bro_tensor_layernormBackwardBatchedWithCaches), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error);
}

} // namespace brotensor::api
