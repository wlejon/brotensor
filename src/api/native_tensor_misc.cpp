// native_tensor_misc.cpp — bodies for the "misc" restored group: the
// gated-deltanet L2 norm pair, three remaining loss entry points and the
// diffusion sampler steps + timestep embedding. See native_tensor_misc_decl.h
// for the provenance of each function and native_tensor_nn.cpp for the
// mechanism these bodies copy (need() / BROTENSOR_API_TRY / the error slot).

#include "native_tensor_misc_decl.h"
#include "api_internal.h"
#include "native_register.h"

#include <string>

using namespace brotensor::api;

extern "C" {

// ---- L2 norm ---------------------------------------------------------------

void bro_tensor_l2NormForward(void* X, int32_t headDim, int32_t numHeads, double eps, void* Y) {
    if (!need("l2NormForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::l2_norm_forward(*toTensor(X), headDim, numHeads, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("l2NormForward")
}

void bro_tensor_l2NormBackward(void* X, int32_t headDim, int32_t numHeads, double eps, void* dY, void* dX) {
    if (!need("l2NormBackward", {X, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::l2_norm_backward(*toTensor(X), headDim, numHeads, static_cast<float>(eps),
                                    *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("l2NormBackward")
}

// ---- losses ----------------------------------------------------------------

void bro_tensor_bceWithLogitsFusedBatched(void* logits_BL, void* target_BL, uint64_t mask_bits,
                                          double posWeight, void* probs_BL, void* dLogits_BL,
                                          void* lossPerSample) {
    if (!need("bceWithLogitsFusedBatched", {logits_BL, target_BL, probs_BL, dLogits_BL, lossPerSample})) return;
    BROTENSOR_API_TRY
        brotensor::bce_with_logits_fused_batched(*toTensor(logits_BL), *toTensor(target_BL),
                                                 maskPtr(mask_bits), static_cast<float>(posWeight),
                                                 *toTensor(probs_BL), *toTensor(dLogits_BL),
                                                 *toTensor(lossPerSample));
    BROTENSOR_API_CATCH("bceWithLogitsFusedBatched")
}

// Host buffers: the op takes raw `const float*` / `float*` (CPU-only), so the
// four arrays cross as f32[] pairs and probs / dLogits are written in place.
// An empty mask array is "no mask" — the old binding's `mask|null`.
double bro_tensor_softmaxXentSegment(const float* logits, uint32_t logits_len,
                                     const float* target, uint32_t target_len,
                                     float* probs, uint32_t probs_len,
                                     float* dLogits, uint32_t dLogits_len,
                                     int32_t n,
                                     const float* mask, uint32_t mask_len) {
    if (!logits || !target || !probs || !dLogits) {
        setError("softmaxXentSegment: logits/target/probs/dLogits must be Float32Array");
        return 0.0;
    }
    if (n <= 0) {
        setError("softmaxXentSegment: n must be positive");
        return 0.0;
    }
    const auto un = static_cast<uint32_t>(n);
    if (logits_len < un || target_len < un || probs_len < un || dLogits_len < un) {
        setError("softmaxXentSegment: a buffer holds fewer than n=" + std::to_string(n) + " elements");
        return 0.0;
    }
    const float* mask_ptr = (mask && mask_len > 0) ? mask : nullptr;
    if (mask_ptr && mask_len < un) {
        setError("softmaxXentSegment: mask holds fewer than n=" + std::to_string(n) + " elements");
        return 0.0;
    }
    BROTENSOR_API_TRY
        return brotensor::softmax_xent_segment(logits, target, probs, dLogits, n, mask_ptr);
    BROTENSOR_API_CATCH("softmaxXentSegment")
    return 0.0;
}

// mseScalar(pred, target) -> [loss, dPred]: the pair as a length-2 f64[].
void bro_tensor_mseScalar(double pred, double target, bronze_native_buffer* out) {
    static thread_local double pair[2];
    if (!out) return;
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
    BROTENSOR_API_TRY
        float dPred = 0.0f;
        const float loss = brotensor::mse_scalar(static_cast<float>(pred), static_cast<float>(target), dPred);
        pair[0] = loss;
        pair[1] = dPred;
        out->data = pair;
        out->length = 2;
    BROTENSOR_API_CATCH("mseScalar")
}

// ---- diffusion sampler steps + timestep embedding --------------------------

void bro_tensor_ddimStep(void* x_t, void* eps_pred, double alphaT, double alphaPrev,
                         double sigmaT, void* x_prev) {
    if (!need("ddimStep", {x_t, eps_pred, x_prev})) return;
    BROTENSOR_API_TRY
        brotensor::ddim_step(*toTensor(x_t), *toTensor(eps_pred), static_cast<float>(alphaT),
                             static_cast<float>(alphaPrev), static_cast<float>(sigmaT), *toTensor(x_prev));
    BROTENSOR_API_CATCH("ddimStep")
}

void bro_tensor_eulerStep(void* x_t, void* eps_pred, double sigmaT, double sigmaPrev, void* x_prev) {
    if (!need("eulerStep", {x_t, eps_pred, x_prev})) return;
    BROTENSOR_API_TRY
        brotensor::euler_step(*toTensor(x_t), *toTensor(eps_pred), static_cast<float>(sigmaT),
                              static_cast<float>(sigmaPrev), *toTensor(x_prev));
    BROTENSOR_API_CATCH("eulerStep")
}

void bro_tensor_dpmpp2mStep(void* x_t, void* eps_pred, void* x0_prev, double sigmaT,
                            double c_xt, double c_x0t, double c_x0prev,
                            void* x_prev, void* x0_out) {
    if (!need("dpmpp2mStep", {x_t, eps_pred, x0_prev, x_prev, x0_out})) return;
    BROTENSOR_API_TRY
        brotensor::dpmpp_2m_step(*toTensor(x_t), *toTensor(eps_pred), *toTensor(x0_prev),
                                 static_cast<float>(sigmaT), static_cast<float>(c_xt),
                                 static_cast<float>(c_x0t), static_cast<float>(c_x0prev),
                                 *toTensor(x_prev), *toTensor(x0_out));
    BROTENSOR_API_CATCH("dpmpp2mStep")
}

void bro_tensor_timestepEmbedding(void* timesteps, int32_t dim, double maxPeriod, void* Y) {
    if (!need("timestepEmbedding", {timesteps, Y})) return;
    BROTENSOR_API_TRY
        brotensor::timestep_embedding(*toTensor(timesteps), dim, static_cast<float>(maxPeriod), *toTensor(Y));
    BROTENSOR_API_CATCH("timestepEmbedding")
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_misc(std::string* error) {
    using namespace brotensor::api::reg;
    return
        fn("__bro_native.tensor.l2NormForward", p(&bro_tensor_l2NormForward), "void",
           {kTensorCls, "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.l2NormBackward", p(&bro_tensor_l2NormBackward), "void",
           {kTensorCls, "i32", "i32", "f64", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.bceWithLogitsFusedBatched", p(&bro_tensor_bceWithLogitsFusedBatched), "void",
           {kTensorCls, kTensorCls, kDyn, "f64", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.softmaxXentSegment", p(&bro_tensor_softmaxXentSegment), "f64",
           {"f32[]", "f32[]", "f32[]", "f32[]", "i32", "f32[]"}, error) &&
        fn("__bro_native.tensor.mseScalar", p(&bro_tensor_mseScalar), "f64[]",
           {"f64", "f64"}, error) &&
        fn("__bro_native.tensor.ddimStep", p(&bro_tensor_ddimStep), "void",
           {kTensorCls, kTensorCls, "f64", "f64", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.eulerStep", p(&bro_tensor_eulerStep), "void",
           {kTensorCls, kTensorCls, "f64", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.dpmpp2mStep", p(&bro_tensor_dpmpp2mStep), "void",
           {kTensorCls, kTensorCls, kTensorCls, "f64", "f64", "f64", "f64", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.timestepEmbedding", p(&bro_tensor_timestepEmbedding), "void",
           {kTensorCls, "i32", "f64", kTensorCls}, error);
}

} // namespace brotensor::api
