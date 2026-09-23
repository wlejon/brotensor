#include "native_tensor_decl.h"
#include "api_internal.h"
#include <vector>

using namespace brotensor::api;

extern "C" {

void bro_tensor_layernormBackward(void* dY, void* xhat, void* gamma, double rstd, void* dX, void* dGamma, void* dBeta) {
    auto* dyt = toTensor(dY);
    auto* xht = toTensor(xhat);
    auto* gt = toTensor(gamma);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    auto* dbt = toTensor(dBeta);
    if (!dyt || !xht || !gt || !dxt || !dgt || !dbt) return;
    brotensor::layernorm_backward(*dyt, *xht, *gt, static_cast<float>(rstd), *dxt, *dgt, *dbt);
}

void bro_tensor_layernormForwardInferenceBatched(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps) {
    auto* xt = toTensor(X_RD);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y_RD);
    if (!xt || !gt || !bt || !yt) return;
    brotensor::layernorm_forward_inference_batched(*xt, *gt, *bt, *yt, static_cast<float>(eps));
}

void bro_tensor_layernormForwardInferenceBatchedFp16(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps) {
    auto* xt = toTensor(X_RD);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y_RD);
    if (!xt || !gt || !bt || !yt) return;
    brotensor::layernorm_forward_inference_batched_fp16(*xt, *gt, *bt, *yt, static_cast<float>(eps));
}

void bro_tensor_rmsNormForward(void* X, void* gamma, double eps, void* Y) {
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* yt = toTensor(Y);
    if (!xt || !gt || !yt) return;
    brotensor::rms_norm_forward(*xt, *gt, static_cast<float>(eps), *yt);
}

void bro_tensor_rmsNormBackward(void* X, void* gamma, void* dY, double eps, void* dX, void* dGamma) {
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    if (!xt || !gt || !dyt || !dxt || !dgt) return;
    brotensor::rms_norm_backward(*xt, *gt, *dyt, static_cast<float>(eps), *dxt, *dgt);
}

void bro_tensor_groupNormForward(void* X, void* gamma, void* beta, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* Y) {
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y);
    if (!xt || !gt || !bt || !yt) return;
    brotensor::group_norm_forward(*xt, *gt, *bt, N, C, H, W, numGroups, static_cast<float>(eps), *yt);
}

void bro_tensor_groupNormBackward(void* X, void* gamma, void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* dX, void* dGamma, void* dBeta) {
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    auto* dbt = toTensor(dBeta);
    if (!xt || !gt || !dyt || !dxt || !dgt || !dbt) return;
    brotensor::group_norm_backward(*xt, *gt, *dyt, N, C, H, W, numGroups, static_cast<float>(eps), *dxt, *dgt, *dbt);
}

void bro_tensor_ropeForward(void* X, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::rope_forward(*xt, headDim, numHeads, seqOffset, static_cast<float>(thetaBase), *yt);
}

void bro_tensor_ropeBackward(void* dY, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* dX) {
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!dyt || !dxt) return;
    brotensor::rope_backward(*dyt, headDim, numHeads, seqOffset, static_cast<float>(thetaBase), *dxt);
}

void bro_tensor_ropeApply(void* X, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* Y) {
    auto* xt = toTensor(X);
    auto* ct = toTensor(cosTbl);
    auto* st = toTensor(sinTbl);
    auto* yt = toTensor(Y);
    if (!xt || !ct || !st || !yt) return;
    brotensor::rope_apply(*xt, *ct, *st, headDim, numHeads, *yt);
}

void bro_tensor_ropeApplyBackward(void* dY, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* dX) {
    if (!need("ropeApplyBackward", {dY, cosTbl, sinTbl, dX})) return;
    BROTENSOR_API_TRY
        brotensor::rope_apply_backward(*toTensor(dY), *toTensor(cosTbl), *toTensor(sinTbl),
                                       headDim, numHeads, *toTensor(dX));
    BROTENSOR_API_CATCH("ropeApplyBackward")
}

void bro_tensor_modulate(void* X, void* scale, void* shift, void* Y) {
    auto* xt = toTensor(X);
    auto* st = toTensor(scale);
    auto* sht = toTensor(shift);
    auto* yt = toTensor(Y);
    if (!xt || !st || !sht || !yt) return;
    brotensor::modulate(*xt, *st, *sht, *yt);
}

void bro_tensor_broadcastMul(void* X, void* v, void* Y) {
    auto* xt = toTensor(X);
    auto* vt = toTensor(v);
    auto* yt = toTensor(Y);
    if (!xt || !vt || !yt) return;
    brotensor::broadcast_mul(*xt, *vt, *yt);
}

void bro_tensor_attentionTokenMoments(void* Attn, int32_t h_lat, int32_t w_lat, void* mass, void* centroid) {
    auto* at = toTensor(Attn);
    auto* mt = toTensor(mass);
    auto* ct = toTensor(centroid);
    if (!at || !mt || !ct) return;
    brotensor::attention_token_moments(*at, h_lat, w_lat, *mt, *ct);
}

void bro_tensor_buildSlotMask(void* x, int32_t offset, int32_t K, int32_t stride, void* mask) {
    auto* xt = toTensor(x);
    auto* mt = toTensor(mask);
    if (!xt || !mt) return;
    brotensor::build_slot_mask(*xt, offset, K, stride, *mt);
}

void bro_tensor_buildCausalMaskRow(int32_t L, int32_t q, void* mask) {
    auto* mt = toTensor(mask);
    if (!mt) return;
    brotensor::build_causal_mask_row(L, q, *mt);
}

void bro_tensor_flashAttentionDecode(void* Q, void* K_cache, void* V_cache, int32_t validLen, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window) {
    auto* qt = toTensor(Q);
    auto* kt = toTensor(K_cache);
    auto* vt = toTensor(V_cache);
    auto* ot = toTensor(O);
    if (!qt || !kt || !vt || !ot) return;
    int kvHeads = (numKvHeads_given && numKvHeads > 0) ? numKvHeads : numHeads;
    brotensor::flash_attention_decode(*qt, *kt, *vt, validLen, numHeads, kvHeads, *ot, static_cast<float>(attnSoftcap), window);
}

void bro_tensor_flashAttentionDecodeMasked(void* Q, void* K_cache, void* V_cache, void* dMask, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window) {
    auto* qt = toTensor(Q);
    auto* kt = toTensor(K_cache);
    auto* vt = toTensor(V_cache);
    auto* mt = toTensor(dMask);
    auto* ot = toTensor(O);
    if (!qt || !kt || !vt || !ot) return;
    const float* mask_ptr = mt ? static_cast<const float*>(mt->data) : nullptr;
    int kvHeads = (numKvHeads_given && numKvHeads > 0) ? numKvHeads : numHeads;
    brotensor::flash_attention_decode_masked(*qt, *kt, *vt, mask_ptr, numHeads, kvHeads, *ot, static_cast<float>(attnSoftcap), window);
}

void bro_tensor_kvCacheAppend(void* K_new, void* V_new, int32_t curLen, void* K_cache, void* V_cache) {
    auto* knt = toTensor(K_new);
    auto* vnt = toTensor(V_new);
    auto* kct = toTensor(K_cache);
    auto* vct = toTensor(V_cache);
    if (!knt || !vnt || !kct || !vct) return;
    brotensor::kv_cache_append(*knt, *vnt, curLen, *kct, *vct);
}

void bro_tensor_embeddingLookupForward(void* table, void* idxAsInt32, int32_t B, void* out) {
    auto* tt = toTensor(table);
    auto* it = toTensor(idxAsInt32);
    auto* ot = toTensor(out);
    if (!tt || !it || !ot || B <= 0) return;
    if (it->dtype == brotensor::Dtype::INT32) {
        brotensor::embedding_lookup_forward(*tt, static_cast<const int32_t*>(it->data), B, *ot);
    } else {
        auto host_i = it->to(brotensor::Device::CPU);
        std::vector<int32_t> idx_buf(static_cast<size_t>(B));
        const float* f = host_i.host_f32();
        for (int i = 0; i < B; ++i) idx_buf[i] = static_cast<int32_t>(f[i]);
        brotensor::embedding_lookup_forward(*tt, idx_buf.data(), B, *ot);
    }
}

void bro_tensor_embeddingLookupBackward(void* dOut, void* idxAsInt32, int32_t B, void* dTable) {
    auto* dot = toTensor(dOut);
    auto* it = toTensor(idxAsInt32);
    auto* dtt = toTensor(dTable);
    if (!dot || !it || !dtt || B <= 0) return;
    if (it->dtype == brotensor::Dtype::INT32) {
        brotensor::embedding_lookup_backward(*dot, static_cast<const int32_t*>(it->data), B, *dtt);
    } else {
        auto host_i = it->to(brotensor::Device::CPU);
        std::vector<int32_t> idx_buf(static_cast<size_t>(B));
        const float* f = host_i.host_f32();
        for (int i = 0; i < B; ++i) idx_buf[i] = static_cast<int32_t>(f[i]);
        brotensor::embedding_lookup_backward(*dot, idx_buf.data(), B, *dtt);
    }
}

void bro_tensor_sgdStep(void* param, void* grad, void* velocity, double lr, double momentum) {
    auto* pt = toTensor(param);
    auto* gt = toTensor(grad);
    auto* vt = toTensor(velocity);
    if (!pt || !gt || !vt) return;
    brotensor::sgd_step(*pt, *gt, *vt, static_cast<float>(lr), static_cast<float>(momentum));
}

void bro_tensor_adamStep(void* param, void* grad, void* m, void* v, double lr, double beta1, double beta2, double eps, int32_t step) {
    auto* pt = toTensor(param);
    auto* gt = toTensor(grad);
    auto* mt = toTensor(m);
    auto* vt = toTensor(v);
    if (!pt || !gt || !mt || !vt) return;
    brotensor::adam_step(*pt, *gt, *mt, *vt, static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps), step);
}

// ---- restored from the QuickJS binding (tensor_bindings.cpp) ---------------

void bro_tensor_cast(void* src, void* dst, const char* outDtype) {
    if (!need("cast", {src, dst})) return;
    BROTENSOR_API_TRY
        brotensor::cast(*toTensor(src), *toTensor(dst), parseDtype(outDtype));
    BROTENSOR_API_CATCH("cast")
}

// softmaxForward(logits, probs, mask|null): the old third argument.
void bro_tensor_softmaxForwardMasked(void* logits, void* probs, uint64_t mask_bits) {
    if (!need("softmaxForward", {logits, probs})) return;
    BROTENSOR_API_TRY
        brotensor::softmax_forward(*toTensor(logits), *toTensor(probs), maskPtr(mask_bits));
    BROTENSOR_API_CATCH("softmaxForward")
}

// layernormForward(x, gamma, beta, y, xhat, eps) -> {mean, rstd}: the two
// scalar caches come back as a length-2 f64[] the wrapper unpacks.
void bro_tensor_layernormForward(void* x, void* gamma, void* beta, void* y, void* xhat, double eps, bronze_native_buffer* out) {
    static thread_local double stats[2];
    if (!out) return;
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
    if (!need("layernormForward", {x, gamma, beta, y, xhat})) return;
    BROTENSOR_API_TRY
        float mean = 0.f, rstd = 0.f;
        brotensor::layernorm_forward(*toTensor(x), *toTensor(gamma), *toTensor(beta), *toTensor(y), *toTensor(xhat),
                                     mean, rstd, static_cast<float>(eps));
        stats[0] = mean;
        stats[1] = rstd;
        out->data = stats;
        out->length = 2;
    BROTENSOR_API_CATCH("layernormForward")
}

void bro_tensor_maskedMeanPoolForward(void* X, uint64_t mask_bits, void* y) {
    if (!need("maskedMeanPoolForward", {X, y})) return;
    BROTENSOR_API_TRY
        brotensor::masked_mean_pool_forward(*toTensor(X), maskPtr(mask_bits), *toTensor(y));
    BROTENSOR_API_CATCH("maskedMeanPoolForward")
}

void bro_tensor_maskedMeanPoolBackward(void* dY, uint64_t mask_bits, int32_t K, void* dX) {
    if (!need("maskedMeanPoolBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::masked_mean_pool_backward(*toTensor(dY), maskPtr(mask_bits), K, *toTensor(dX));
    BROTENSOR_API_CATCH("maskedMeanPoolBackward")
}

double bro_tensor_softmaxXentFused(void* logits, void* target, uint64_t mask_bits, void* probs, void* dLogits) {
    if (!need("softmaxXentFused", {logits, target, probs, dLogits})) return 0.0;
    BROTENSOR_API_TRY
        return brotensor::softmax_xent_fused(*toTensor(logits), *toTensor(target), maskPtr(mask_bits),
                                             *toTensor(probs), *toTensor(dLogits));
    BROTENSOR_API_CATCH("softmaxXentFused")
    return 0.0;
}

// The head_offsets GpuTensor is a device INT32 buffer (cumulative, n_heads+1),
// the old binding's convention.
void bro_tensor_softmaxXentFusedBatched(void* logits_BL, void* target_BL, uint64_t mask_bits, void* headOffsets, int32_t nHeads, void* probs_BL, void* dLogits_BL, void* lossPerSample) {
    if (!need("softmaxXentFusedBatched", {logits_BL, target_BL, headOffsets, probs_BL, dLogits_BL, lossPerSample})) return;
    BROTENSOR_API_TRY
        brotensor::softmax_xent_fused_batched(*toTensor(logits_BL), *toTensor(target_BL), maskPtr(mask_bits),
                                              static_cast<const int*>(toTensor(headOffsets)->data), nHeads,
                                              *toTensor(probs_BL), *toTensor(dLogits_BL), *toTensor(lossPerSample));
    BROTENSOR_API_CATCH("softmaxXentFusedBatched")
}

} // extern "C"

namespace {

std::vector<const brotensor::Tensor*> asConst(const std::vector<brotensor::Tensor*>& v) {
    return std::vector<const brotensor::Tensor*>(v.begin(), v.end());
}

std::vector<int> asInts(const int32_t* p, uint32_t n) {
    return p ? std::vector<int>(p, p + n) : std::vector<int>();
}

} // namespace

extern "C" {

void bro_tensor_concatRows(uint64_t parts_bits, void* out) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatRows", {out}) || !readTensorArray(parts_bits, parts, "concatRows")) return;
    BROTENSOR_API_TRY
        brotensor::concat_rows(asConst(parts), *toTensor(out));
    BROTENSOR_API_CATCH("concatRows")
}

void bro_tensor_splitRows(void* in, uint64_t parts_bits) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("splitRows", {in}) || !readTensorArray(parts_bits, parts, "splitRows")) return;
    BROTENSOR_API_TRY
        brotensor::split_rows(*toTensor(in), parts);
    BROTENSOR_API_CATCH("splitRows")
}

void bro_tensor_concatBatchedRows(uint64_t parts_bits, void* out) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatBatchedRows", {out}) || !readTensorArray(parts_bits, parts, "concatBatchedRows")) return;
    BROTENSOR_API_TRY
        brotensor::concat_batched_rows(asConst(parts), *toTensor(out));
    BROTENSOR_API_CATCH("concatBatchedRows")
}

void bro_tensor_concatNchwChannels(uint64_t parts_bits, int32_t N, int32_t H, int32_t W, const int32_t* C_per_part, uint32_t C_len, void* out) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatNchwChannels", {out}) || !readTensorArray(parts_bits, parts, "concatNchwChannels")) return;
    BROTENSOR_API_TRY
        brotensor::concat_nchw_channels(asConst(parts), N, H, W, asInts(C_per_part, C_len), *toTensor(out));
    BROTENSOR_API_CATCH("concatNchwChannels")
}

void bro_tensor_concatNchwChannelsBackward(void* dY, int32_t N, int32_t H, int32_t W, const int32_t* C_per_part, uint32_t C_len, uint64_t parts_bits) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatNchwChannelsBackward", {dY}) || !readTensorArray(parts_bits, parts, "concatNchwChannelsBackward")) return;
    BROTENSOR_API_TRY
        brotensor::concat_nchw_channels_backward(*toTensor(dY), N, H, W, asInts(C_per_part, C_len), parts);
    BROTENSOR_API_CATCH("concatNchwChannelsBackward")
}

// ---- counter-based RNG + init (key / counter / state: BigInt or Number) ----

void bro_tensor_randUniform(uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randUniform", {Y})) return;
    BROTENSOR_API_TRY
        brotensor::rand_uniform(bronze::embed::toUint64(bronze::Value{key_bits}),
                                bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randUniform")
}

void bro_tensor_randn(uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randn", {Y})) return;
    BROTENSOR_API_TRY
        brotensor::randn(bronze::embed::toUint64(bronze::Value{key_bits}),
                         bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randn")
}

void bro_tensor_randBernoulli(double p, uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randBernoulli", {Y})) return;
    BROTENSOR_API_TRY
        brotensor::rand_bernoulli(static_cast<float>(p), bronze::embed::toUint64(bronze::Value{key_bits}),
                                  bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randBernoulli")
}

void bro_tensor_randnTruncated(double lo, double hi, uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randnTruncated", {Y})) return;
    BROTENSOR_API_TRY
        brotensor::randn_truncated(static_cast<float>(lo), static_cast<float>(hi),
                                   bronze::embed::toUint64(bronze::Value{key_bits}),
                                   bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randnTruncated")
}

// xavierInit(W, rngState) -> advanced state. The old binding returned a
// BigInt; the embed API mints no BigInt, so the state comes back as a Number
// (exact below 2^53 — thread it through successive inits as before).
double bro_tensor_xavierInit(void* W, uint64_t state_bits) {
    uint64_t state = bronze::embed::toUint64(bronze::Value{state_bits});
    if (!need("xavierInit", {W})) return static_cast<double>(state);
    BROTENSOR_API_TRY
        brotensor::xavier_init(*toTensor(W), state);
    BROTENSOR_API_CATCH("xavierInit")
    return static_cast<double>(state);
}

} // extern "C"

namespace brotensor::api {

bool readTensorArray(uint64_t bits, std::vector<brotensor::Tensor*>& out, const char* label) {
    namespace ev = bronze::embed;
    ev::Persistent arr(bronze::Value{bits});
    out.clear();
    if (!ev::isObject(arr.get())) {
        setError(std::string(label) + ": expected an array of GpuTensors");
        return false;
    }
    const auto n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arr.get(), "length")));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        void* d = ev::handleData(ev::getElement(arr.get(), i));
        auto* t = d ? toTensor(d) : nullptr;
        if (!t) {
            setError(std::string(label) + ": element " + std::to_string(i) + " is not a GpuTensor");
            return false;
        }
        out.push_back(t);
    }
    return true;
}

} // namespace brotensor::api
