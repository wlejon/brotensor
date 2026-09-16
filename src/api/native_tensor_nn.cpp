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

void bro_tensor_ropeApplyBackward(void* dY, int32_t headDim, int32_t numHeads, void* dX) {
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!dyt || !dxt) return;
    brotensor::rope_backward(*dyt, headDim, numHeads, 0, 10000.0f, *dxt);
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

} // extern "C"
