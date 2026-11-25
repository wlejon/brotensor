// native_tensor_misc.cpp — bodies for the "misc" group: the
// gated-deltanet L2 norm pair, three remaining loss entry points and the
// diffusion sampler steps + timestep embedding. See native_tensor_misc_decl.h
// for the ops each function calls and native_tensor_nn.cpp for the
// mechanism these bodies copy (need() / BROTENSOR_API_TRY / the error slot).

#include "native_tensor_misc_decl.h"
#include "api_internal.h"
#include "native_register.h"
#include <brotensor/kernel_cache.h>

#include <string>

using namespace brotensor::api;

namespace {

// X is (L, numHeads*headDim), both positive.
bool needHeadCols(const char* L, const brotensor::Tensor* x, int32_t headDim, int32_t numHeads) {
    if (!needPositive(L, {headDim, numHeads})) return false;
    if (static_cast<int64_t>(x->cols) != static_cast<int64_t>(numHeads) * headDim) {
        setError(std::string(L) + ": X must have numHeads*headDim columns");
        return false;
    }
    return true;
}

} // namespace

extern "C" {

// ---- L2 norm ---------------------------------------------------------------

void bro_tensor_l2NormForward(void* X, int32_t headDim, int32_t numHeads, double eps, void* Y) {
    if (!need("l2NormForward", {X, Y}) || !needHeadCols("l2NormForward", toTensor(X), headDim, numHeads)) return;
    BROTENSOR_API_TRY
        brotensor::l2_norm_forward(*toTensor(X), headDim, numHeads, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("l2NormForward")
}

void bro_tensor_l2NormBackward(void* X, int32_t headDim, int32_t numHeads, double eps, void* dY, void* dX) {
    if (!need("l2NormBackward", {X, dY, dX}) || !needHeadCols("l2NormBackward", toTensor(X), headDim, numHeads)) return;
    if (!needPair("l2NormBackward", "dY", toTensor(dY), toTensor(X))) return;
    BROTENSOR_API_TRY
        brotensor::l2_norm_backward(*toTensor(X), headDim, numHeads, static_cast<float>(eps),
                                    *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("l2NormBackward")
}

// ---- losses ----------------------------------------------------------------

void bro_tensor_bceWithLogitsFusedBatched(void* logits_BL, void* target_BL, uint64_t mask_bits,
                                          double posWeight, void* probs_BL, void* dLogits_BL,
                                          void* lossPerSample) {
    const char* L = "bceWithLogitsFusedBatched";
    if (!need(L, {logits_BL, target_BL, probs_BL, dLogits_BL, lossPerSample})) return;
    if (!needPair(L, "target_BL", toTensor(target_BL), toTensor(logits_BL))) return;
    if (!needMask(L, tensorFromValue(mask_bits), toTensor(logits_BL)->size())) return;
    BROTENSOR_API_TRY
        brotensor::bce_with_logits_fused_batched(*toTensor(logits_BL), *toTensor(target_BL),
                                                 maskPtr(mask_bits), static_cast<float>(posWeight),
                                                 *toTensor(probs_BL), *toTensor(dLogits_BL),
                                                 *toTensor(lossPerSample));
    BROTENSOR_API_CATCH("bceWithLogitsFusedBatched")
}

// Host buffers: the op takes raw `const float*` / `float*` (CPU-only), so the
// four arrays cross as f32[] pairs and probs / dLogits are written in place.
// An empty mask array is "no mask" (the wrapper's `mask|null`).
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
    if (!needPair("ddimStep", "eps_pred", toTensor(eps_pred), toTensor(x_t))) return;
    BROTENSOR_API_TRY
        brotensor::ddim_step(*toTensor(x_t), *toTensor(eps_pred), static_cast<float>(alphaT),
                             static_cast<float>(alphaPrev), static_cast<float>(sigmaT), *toTensor(x_prev));
    BROTENSOR_API_CATCH("ddimStep")
}

void bro_tensor_eulerStep(void* x_t, void* eps_pred, double sigmaT, double sigmaPrev, void* x_prev) {
    if (!need("eulerStep", {x_t, eps_pred, x_prev})) return;
    if (!needPair("eulerStep", "eps_pred", toTensor(eps_pred), toTensor(x_t))) return;
    BROTENSOR_API_TRY
        brotensor::euler_step(*toTensor(x_t), *toTensor(eps_pred), static_cast<float>(sigmaT),
                              static_cast<float>(sigmaPrev), *toTensor(x_prev));
    BROTENSOR_API_CATCH("eulerStep")
}

void bro_tensor_dpmpp2mStep(void* x_t, void* eps_pred, void* x0_prev, double sigmaT,
                            double c_xt, double c_x0t, double c_x0prev,
                            void* x_prev, void* x0_out) {
    if (!need("dpmpp2mStep", {x_t, eps_pred, x0_prev, x_prev, x0_out})) return;
    if (!needPair("dpmpp2mStep", "eps_pred", toTensor(eps_pred), toTensor(x_t)) ||
        !needPair("dpmpp2mStep", "x0_prev", toTensor(x0_prev), toTensor(x_t))) return;
    BROTENSOR_API_TRY
        brotensor::dpmpp_2m_step(*toTensor(x_t), *toTensor(eps_pred), *toTensor(x0_prev),
                                 static_cast<float>(sigmaT), static_cast<float>(c_xt),
                                 static_cast<float>(c_x0t), static_cast<float>(c_x0prev),
                                 *toTensor(x_prev), *toTensor(x0_out));
    BROTENSOR_API_CATCH("dpmpp2mStep")
}

void bro_tensor_timestepEmbedding(void* timesteps, int32_t dim, double maxPeriod, void* Y) {
    if (!need("timestepEmbedding", {timesteps, Y}) || !needNonNegative("timestepEmbedding", {dim})) return;
    BROTENSOR_API_TRY
        brotensor::timestep_embedding(*toTensor(timesteps), dim, static_cast<float>(maxPeriod), *toTensor(Y));
    BROTENSOR_API_CATCH("timestepEmbedding")
}

const char* bro_tensor_cacheDir(void) {
    static thread_local std::string dir;
    BROTENSOR_API_TRY
        dir = brotensor::kernel_cache_dir();
        return dir.c_str();
    BROTENSOR_API_CATCH("cacheDir")
    return "";
}

void bro_tensor_cacheStats(bronze_native_buffer* out) {
    static thread_local double vals[5];
    if (!out) return;
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
    BROTENSOR_API_TRY
        const brotensor::KernelCacheStats s = brotensor::kernel_cache_stats();
        vals[0] = static_cast<double>(s.hits);
        vals[1] = static_cast<double>(s.misses);
        vals[2] = static_cast<double>(s.writes);
        vals[3] = static_cast<double>(s.corruptions);
        vals[4] = static_cast<double>(s.entries);
        out->data = vals;
        out->length = 5;
    BROTENSOR_API_CATCH("cacheStats")
}

void bro_tensor_setAutotune(bool enabled) {
    brotensor::set_autotune_enabled(enabled);
}

bool bro_tensor_autotuneEnabled(void) {
    return brotensor::autotune_enabled();
}

void bro_tensor_warmupKernels(const char* profile, bool autotune, bronze_native_buffer* out) {
    static thread_local double vals[3];
    if (!out) return;
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
    BROTENSOR_API_TRY
        brotensor::WarmupOptions opts;
        if (profile && profile[0] != '\0') opts.profile = profile;
        opts.autotune = autotune;
        const brotensor::WarmupStats w = brotensor::warmup_kernels(brotensor::default_device(), opts);
        vals[0] = static_cast<double>(w.kernels_compiled);
        vals[1] = static_cast<double>(w.autotune_entries);
        vals[2] = w.elapsed_ms;
        out->data = vals;
        out->length = 3;
    BROTENSOR_API_CATCH("warmupKernels")
}

bool bro_tensor_exportKernelBundle(const char* path) {
    std::string resolved;
    if (!gatedPath(path, /*write=*/true, resolved)) return false;
    BROTENSOR_API_TRY
        std::string err;
        if (!brotensor::export_kernel_bundle(resolved, brotensor::default_device(), &err)) {
            setError("exportKernelBundle: " + (err.empty() ? "failed" : err));
            return false;
        }
        return true;
    BROTENSOR_API_CATCH("exportKernelBundle")
    return false;
}

void bro_tensor_importKernelBundle(const char* path, bronze_native_buffer* out) {
    static thread_local double vals[3];
    if (out) {
        out->data = nullptr;
        out->length = 0;
        out->release = nullptr;
        out->ctx = nullptr;
    }
    std::string resolved;
    if (!gatedPath(path, /*write=*/false, resolved)) return;
    BROTENSOR_API_TRY
        brotensor::BundleImportReport rep;
        std::string err;
        if (!brotensor::import_kernel_bundle(resolved, &rep, &err)) {
            setError("importKernelBundle: " + (err.empty() ? "failed" : err));
            return;
        }
        if (out) {
            vals[0] = static_cast<double>(rep.entries_imported);
            vals[1] = static_cast<double>(rep.entries_skipped);
            vals[2] = static_cast<double>(rep.bytes);
            out->data = vals;
            out->length = 3;
        }
    BROTENSOR_API_CATCH("importKernelBundle")
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
           {kTensorCls, "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.cacheDir", p(&bro_tensor_cacheDir), "str", {}, error) &&
        fn("__bro_native.tensor.cacheStats", p(&bro_tensor_cacheStats), "f64[]", {}, error) &&
        fn("__bro_native.tensor.setAutotune", p(&bro_tensor_setAutotune), "void", {"bool"}, error) &&
        fn("__bro_native.tensor.autotuneEnabled", p(&bro_tensor_autotuneEnabled), "bool", {}, error) &&
        fn("__bro_native.tensor.warmupKernels", p(&bro_tensor_warmupKernels), "f64[]", {"str", "bool"}, error) &&
        fn("__bro_native.tensor.exportKernelBundle", p(&bro_tensor_exportKernelBundle), "bool", {"str"}, error) &&
        fn("__bro_native.tensor.importKernelBundle", p(&bro_tensor_importKernelBundle), "f64[]", {"str"}, error);
}

} // namespace brotensor::api
