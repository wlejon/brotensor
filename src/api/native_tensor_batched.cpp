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

#include <cstdint>
#include <vector>

using namespace brotensor::api;

namespace {

// gather_rows / scatter_rows_add take Idx as an (M,1) INT32 tensor — the
// dtype top_k_rows writes, so an index tensor produced on the device flows
// straight through. GpuTensor.prototype.upload() always lands FP32 (it did in
// the QuickJS binding too), so an index list built in JS arrives as FP32:
// materialize an INT32 copy on the same device, exactly as
// bro_tensor_embeddingLookupForward does for its index argument. An INT32 Idx
// is used as-is — no host round trip, and the op still runs on the device.
const brotensor::Tensor& int32Idx(const brotensor::Tensor& Idx, brotensor::Tensor& scratch) {
    if (Idx.dtype == brotensor::Dtype::INT32) return Idx;
    const brotensor::Tensor host = Idx.to(brotensor::Device::CPU);
    const int n = host.size();
    std::vector<int32_t> buf(static_cast<size_t>(n < 0 ? 0 : n));
    const float* f = host.host_f32();
    for (int i = 0; i < n; ++i) buf[i] = static_cast<int32_t>(f[i]);
    scratch = brotensor::Tensor::from_raw_bytes_on(Idx.device, buf.data(), host.rows, host.cols,
                                                   brotensor::Dtype::INT32,
                                                   buf.size() * sizeof(int32_t));
    return scratch;
}

} // namespace

extern "C" {

// ---- batched forwards ------------------------------------------------------

void bro_tensor_linearForwardBatched(void* W, void* bias, void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatched", {W, bias, X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched(*toTensor(W), *toTensor(bias), *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatched")
}

void bro_tensor_linearForwardBatchedFp16(void* W, uint64_t bias_bits, void* X_BD, void* Y_BD) {
    if (!need("linearForwardBatchedFp16", {W, X_BD, Y_BD})) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_fp16(*toTensor(W), tensorFromValue(bias_bits),
                                               *toTensor(X_BD), *toTensor(Y_BD));
    BROTENSOR_API_CATCH("linearForwardBatchedFp16")
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
    if (!need("addInplaceBatched", {Y_BD, X_BD})) return;
    BROTENSOR_API_TRY
        brotensor::add_inplace_batched(*toTensor(Y_BD), *toTensor(X_BD));
    BROTENSOR_API_CATCH("addInplaceBatched")
}

// ---- batched-training backwards --------------------------------------------

void bro_tensor_linearBackwardBatched(void* W, void* X_BD, void* dY_BD, void* dX_BD, void* dW, void* dB) {
    if (!need("linearBackwardBatched", {W, X_BD, dY_BD, dX_BD, dW, dB})) return;
    BROTENSOR_API_TRY
        brotensor::linear_backward_batched(*toTensor(W), *toTensor(X_BD), *toTensor(dY_BD),
                                           *toTensor(dX_BD), *toTensor(dW), *toTensor(dB));
    BROTENSOR_API_CATCH("linearBackwardBatched")
}

void bro_tensor_reluBackwardBatched(void* X_BD, void* dY_BD, void* dX_BD) {
    if (!need("reluBackwardBatched", {X_BD, dY_BD, dX_BD})) return;
    BROTENSOR_API_TRY
        brotensor::relu_backward_batched(*toTensor(X_BD), *toTensor(dY_BD), *toTensor(dX_BD));
    BROTENSOR_API_CATCH("reluBackwardBatched")
}

void bro_tensor_tanhBackwardBatched(void* Y_BD, void* dY_BD, void* dX_BD) {
    if (!need("tanhBackwardBatched", {Y_BD, dY_BD, dX_BD})) return;
    BROTENSOR_API_TRY
        brotensor::tanh_backward_batched(*toTensor(Y_BD), *toTensor(dY_BD), *toTensor(dX_BD));
    BROTENSOR_API_CATCH("tanhBackwardBatched")
}

// ---- row gather / scatter / top-k ------------------------------------------

void bro_tensor_gatherRows(void* X, void* Idx, void* Y) {
    if (!need("gatherRows", {X, Idx, Y})) return;
    BROTENSOR_API_TRY
        brotensor::Tensor scratch;
        brotensor::gather_rows(*toTensor(X), int32Idx(*toTensor(Idx), scratch), *toTensor(Y));
    BROTENSOR_API_CATCH("gatherRows")
}

void bro_tensor_scatterRowsAdd(void* dY, void* Idx, int32_t R, void* dX) {
    if (!need("scatterRowsAdd", {dY, Idx, dX})) return;
    BROTENSOR_API_TRY
        brotensor::Tensor scratch;
        brotensor::scatter_rows_add(*toTensor(dY), int32Idx(*toTensor(Idx), scratch), R, *toTensor(dX));
    BROTENSOR_API_CATCH("scatterRowsAdd")
}

void bro_tensor_topKRows(void* X, int32_t k, void* Vals, void* Idx) {
    if (!need("topKRows", {X, Vals, Idx})) return;
    BROTENSOR_API_TRY
        brotensor::top_k_rows(*toTensor(X), k, *toTensor(Vals), *toTensor(Idx));
    BROTENSOR_API_CATCH("topKRows")
}

// ---- batched LayerNorm with training caches --------------------------------

void bro_tensor_layernormForwardBatchedWithCaches(void* X, void* gamma, void* beta, void* Y, void* Xhat, void* Mean, void* Rstd, double eps) {
    if (!need("layernormForwardBatchedWithCaches", {X, gamma, beta, Y, Xhat, Mean, Rstd})) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_forward_batched_with_caches(*toTensor(X), *toTensor(gamma), *toTensor(beta),
                                                         *toTensor(Y), *toTensor(Xhat),
                                                         *toTensor(Mean), *toTensor(Rstd),
                                                         static_cast<float>(eps));
    BROTENSOR_API_CATCH("layernormForwardBatchedWithCaches")
}

void bro_tensor_layernormBackwardBatchedWithCaches(void* dY, void* Xhat, void* gamma, void* Rstd, void* dX, void* dGamma, void* dBeta) {
    if (!need("layernormBackwardBatchedWithCaches", {dY, Xhat, gamma, Rstd, dX, dGamma, dBeta})) return;
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
