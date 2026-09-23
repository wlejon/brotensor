#include "native_tensor_decl.h"
#include "api_internal.h"
#include <vector>

using namespace brotensor::api;
using brotensor::Tensor;
using brotensor::Dtype;

// Operand checks come first in every body (api_internal.h "input-size
// contract"); accumulated outputs ("caller zeros") are checked like inputs.

namespace {

// X is (L, numHeads*headDim) with an even, positive headDim.
bool needRopeInput(const char* L, const Tensor* xt, int32_t headDim, int32_t numHeads) {
    if (!needPositive(L, {headDim, numHeads})) return false;
    if (headDim % 2 != 0) {
        setError(std::string(L) + ": headDim must be even");
        return false;
    }
    if (static_cast<int64_t>(xt->cols) != static_cast<int64_t>(numHeads) * headDim) {
        setError(std::string(L) + ": X has " + std::to_string(xt->cols) + " columns, numHeads*headDim is " +
                 std::to_string(static_cast<int64_t>(numHeads) * headDim));
        return false;
    }
    return true;
}

// cos/sin tables: (L, headDim/2) FP32.
bool needRopeTables(const char* L, const Tensor* xt, const Tensor* ct, const Tensor* st, int32_t headDim) {
    const int64_t n = elems({xt->rows, headDim / 2});
    if (!needElems(L, "cosTbl", ct, n) || !needElems(L, "sinTbl", st, n)) return false;
    if (ct->dtype != Dtype::FP32 || st->dtype != Dtype::FP32) {
        setError(std::string(L) + ": cosTbl and sinTbl must be FP32");
        return false;
    }
    return true;
}

// Decode attention: Q (L_q, H*hd), caches (L_max, kvH*hd), kvH | H.
bool needDecodeShapes(const char* L, const Tensor* qt, const Tensor* kt, const Tensor* vt,
                      int32_t numHeads, int32_t kvHeads) {
    if (!needPositive(L, {numHeads, kvHeads})) return false;
    if (numHeads % kvHeads != 0 || qt->cols % numHeads != 0) {
        setError(std::string(L) + ": numKvHeads must divide numHeads, and numHeads Q.cols");
        return false;
    }
    const int64_t headDim = qt->cols / numHeads;
    if (kt->cols != kvHeads * headDim || vt->cols != kvHeads * headDim || vt->rows != kt->rows) {
        setError(std::string(L) + ": K_cache/V_cache must both be (L_max, numKvHeads*headDim) with headDim = Q.cols/numHeads");
        return false;
    }
    return needSameDtype(L, "K_cache", kt, qt) && needSameDtype(L, "V_cache", vt, qt);
}

// Embedding indices: B values in [0, V), read back and range-checked (an
// index out of range is an out-of-bounds read of the table on the device),
// then handed to the op as an INT32 buffer on the table's device. The index
// tensor is INT32 storage or FP32 values.
bool embeddingIndices(const char* L, const Tensor* it, int32_t B, int32_t V, brotensor::Device dev,
                      Tensor& deviceIdx) {
    if (!needElems(L, "indices", it, B)) return false;
    std::vector<int32_t> idx(static_cast<size_t>(B));
    auto host = it->to(brotensor::Device::CPU);
    if (it->dtype == Dtype::INT32) {
        std::memcpy(idx.data(), host.host_raw(), idx.size() * sizeof(int32_t));
    } else if (it->dtype == Dtype::FP32) {
        const float* f = host.host_f32();
        for (int32_t i = 0; i < B; ++i) idx[i] = static_cast<int32_t>(f[i]);
    } else {
        setError(std::string(L) + ": indices must be INT32 storage or FP32 values");
        return false;
    }
    for (int32_t i = 0; i < B; ++i) {
        if (idx[i] < 0 || idx[i] >= V) {
            setError(std::string(L) + ": index " + std::to_string(idx[i]) + " at " + std::to_string(i) +
                     " is outside the table's " + std::to_string(V) + " rows");
            return false;
        }
    }
    deviceIdx = Tensor::from_raw_bytes_on(dev, idx.data(), B, 1, Dtype::INT32, idx.size() * sizeof(int32_t));
    return true;
}

} // namespace

extern "C" {

void bro_tensor_layernormBackward(void* dY, void* xhat, void* gamma, double rstd, void* dX, void* dGamma, void* dBeta) {
    const char* L = "layernormBackward";
    auto* dyt = toTensor(dY);
    auto* xht = toTensor(xhat);
    auto* gt = toTensor(gamma);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    auto* dbt = toTensor(dBeta);
    if (!need(L, {dyt, xht, gt, dxt, dgt, dbt})) return;
    if (!needPair(L, "xhat", xht, dyt) || !needPair(L, "gamma", gt, dyt)) return;
    if (!needElems(L, "dGamma", dgt, dyt->size()) || !needElems(L, "dBeta", dbt, dyt->size())) return;
    if (!needSameDtype(L, "dGamma", dgt, dyt) || !needSameDtype(L, "dBeta", dbt, dyt)) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_backward(*dyt, *xht, *gt, static_cast<float>(rstd), *dxt, *dgt, *dbt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_layernormForwardInferenceBatched(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps) {
    const char* L = "layernormForwardInferenceBatched";
    auto* xt = toTensor(X_RD);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y_RD);
    if (!need(L, {xt, gt, bt, yt})) return;
    if (!needElems(L, "gamma", gt, xt->cols) || !needElems(L, "beta", bt, xt->cols)) return;
    if (!needSameDtype(L, "gamma", gt, xt) || !needSameDtype(L, "beta", bt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_forward_inference_batched(*xt, *gt, *bt, *yt, static_cast<float>(eps));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_layernormForwardInferenceBatchedFp16(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps) {
    const char* L = "layernormForwardInferenceBatchedFp16";
    auto* xt = toTensor(X_RD);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y_RD);
    if (!need(L, {xt, gt, bt, yt})) return;
    if (!needElems(L, "gamma", gt, xt->cols) || !needElems(L, "beta", bt, xt->cols)) return;
    if (!needSameDtype(L, "gamma", gt, xt) || !needSameDtype(L, "beta", bt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::layernorm_forward_inference_batched_fp16(*xt, *gt, *bt, *yt, static_cast<float>(eps));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_rmsNormForward(void* X, void* gamma, double eps, void* Y) {
    const char* L = "rmsNormForward";
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, gt, yt})) return;
    // gamma matches X or is FP32 against 16-bit activations.
    if (!needElems(L, "gamma", gt, xt->cols)) return;
    if (gt->dtype != Dtype::FP32 && !needSameDtype(L, "gamma", gt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::rms_norm_forward(*xt, *gt, static_cast<float>(eps), *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_rmsNormBackward(void* X, void* gamma, void* dY, double eps, void* dX, void* dGamma) {
    const char* L = "rmsNormBackward";
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    if (!need(L, {xt, gt, dyt, dxt, dgt})) return;
    if (!needElems(L, "gamma", gt, xt->cols) || !needSameDtype(L, "gamma", gt, xt)) return;
    if (!needPair(L, "dY", dyt, xt)) return;
    if (!needElems(L, "dGamma", dgt, xt->cols) || !needSameDtype(L, "dGamma", dgt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::rms_norm_backward(*xt, *gt, *dyt, static_cast<float>(eps), *dxt, *dgt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_groupNormForward(void* X, void* gamma, void* beta, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* Y) {
    const char* L = "groupNormForward";
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* bt = toTensor(beta);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, gt, bt, yt})) return;
    if (!needPositive(L, {numGroups}) || !needNonNegative(L, {N, C, H, W})) return;
    if (C % numGroups != 0) { setError("groupNormForward: numGroups must divide C"); return; }
    if (!needElems(L, "X", xt, elems({N, C, H, W}))) return;
    if (!needElems(L, "gamma", gt, C) || !needElems(L, "beta", bt, C)) return;
    if (!needSameDtype(L, "gamma", gt, xt) || !needSameDtype(L, "beta", bt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::group_norm_forward(*xt, *gt, *bt, N, C, H, W, numGroups, static_cast<float>(eps), *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_groupNormBackward(void* X, void* gamma, void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* dX, void* dGamma, void* dBeta) {
    const char* L = "groupNormBackward";
    auto* xt = toTensor(X);
    auto* gt = toTensor(gamma);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dgt = toTensor(dGamma);
    auto* dbt = toTensor(dBeta);
    if (!need(L, {xt, gt, dyt, dxt, dgt, dbt})) return;
    if (!needPositive(L, {numGroups}) || !needNonNegative(L, {N, C, H, W})) return;
    if (C % numGroups != 0) { setError("groupNormBackward: numGroups must divide C"); return; }
    const int64_t n = elems({N, C, H, W});
    if (!needElems(L, "X", xt, n) || !needElems(L, "dY", dyt, n) || !needSameDtype(L, "dY", dyt, xt)) return;
    for (const Tensor* t : {gt, dgt, dbt}) {
        if (!needElems(L, "gamma/dGamma/dBeta", t, C) || !needSameDtype(L, "gamma/dGamma/dBeta", t, xt)) return;
    }
    BROTENSOR_API_TRY
        brotensor::group_norm_backward(*xt, *gt, *dyt, N, C, H, W, numGroups, static_cast<float>(eps), *dxt, *dgt, *dbt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_ropeForward(void* X, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* Y) {
    const char* L = "ropeForward";
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, yt}) || !needRopeInput(L, xt, headDim, numHeads)) return;
    BROTENSOR_API_TRY
        brotensor::rope_forward(*xt, headDim, numHeads, seqOffset, static_cast<float>(thetaBase), *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_ropeBackward(void* dY, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* dX) {
    const char* L = "ropeBackward";
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {dyt, dxt}) || !needRopeInput(L, dyt, headDim, numHeads)) return;
    BROTENSOR_API_TRY
        brotensor::rope_backward(*dyt, headDim, numHeads, seqOffset, static_cast<float>(thetaBase), *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_ropeApply(void* X, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* Y) {
    const char* L = "ropeApply";
    auto* xt = toTensor(X);
    auto* ct = toTensor(cosTbl);
    auto* st = toTensor(sinTbl);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, ct, st, yt})) return;
    if (!needRopeInput(L, xt, headDim, numHeads) || !needRopeTables(L, xt, ct, st, headDim)) return;
    BROTENSOR_API_TRY
        brotensor::rope_apply(*xt, *ct, *st, headDim, numHeads, *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_ropeApplyBackward(void* dY, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* dX) {
    const char* L = "ropeApplyBackward";
    auto* dyt = toTensor(dY);
    auto* ct = toTensor(cosTbl);
    auto* st = toTensor(sinTbl);
    auto* dxt = toTensor(dX);
    if (!need(L, {dyt, ct, st, dxt})) return;
    if (!needRopeInput(L, dyt, headDim, numHeads) || !needRopeTables(L, dyt, ct, st, headDim)) return;
    BROTENSOR_API_TRY
        brotensor::rope_apply_backward(*dyt, *ct, *st, headDim, numHeads, *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_modulate(void* X, void* scale, void* shift, void* Y) {
    const char* L = "modulate";
    auto* xt = toTensor(X);
    auto* st = toTensor(scale);
    auto* sht = toTensor(shift);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, st, sht, yt})) return;
    if (!needElems(L, "scale", st, xt->cols) || !needElems(L, "shift", sht, xt->cols)) return;
    if (!needSameDtype(L, "scale", st, xt) || !needSameDtype(L, "shift", sht, xt)) return;
    BROTENSOR_API_TRY
        brotensor::modulate(*xt, *st, *sht, *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_broadcastMul(void* X, void* v, void* Y) {
    const char* L = "broadcastMul";
    auto* xt = toTensor(X);
    auto* vt = toTensor(v);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, vt, yt})) return;
    if (!needElems(L, "v", vt, xt->cols) || !needSameDtype(L, "v", vt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::broadcast_mul(*xt, *vt, *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_attentionTokenMoments(void* Attn, int32_t h_lat, int32_t w_lat, void* mass, void* centroid) {
    const char* L = "attentionTokenMoments";
    auto* at = toTensor(Attn);
    auto* mt = toTensor(mass);
    auto* ct = toTensor(centroid);
    if (!need(L, {at, mt, ct}) || !needPositive(L, {h_lat, w_lat})) return;
    if (static_cast<int64_t>(at->rows) != static_cast<int64_t>(h_lat) * w_lat) {
        setError("attentionTokenMoments: Attn.rows must equal h_lat*w_lat");
        return;
    }
    BROTENSOR_API_TRY
        brotensor::attention_token_moments(*at, h_lat, w_lat, *mt, *ct);
    BROTENSOR_API_CATCH(L)
}

// mask[k] = x[offset + k*stride] > 0.5: the last read is offset+(K-1)*stride.
void bro_tensor_buildSlotMask(void* x, int32_t offset, int32_t K, int32_t stride, void* mask) {
    const char* L = "buildSlotMask";
    auto* xt = toTensor(x);
    auto* mt = toTensor(mask);
    if (!need(L, {xt, mt}) || !needNonNegative(L, {offset, K, stride})) return;
    if (K > 0 && !needElems(L, "x", xt, static_cast<int64_t>(offset) + static_cast<int64_t>(K - 1) * stride + 1)) return;
    if (xt->dtype != Dtype::FP32) { setError("buildSlotMask: x must be FP32"); return; }
    BROTENSOR_API_TRY
        brotensor::build_slot_mask(*xt, offset, K, stride, *mt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_buildCausalMaskRow(int32_t L_, int32_t q, void* mask) {
    const char* L = "buildCausalMaskRow";
    auto* mt = toTensor(mask);
    if (!need(L, {mt}) || !needNonNegative(L, {L_})) return;
    BROTENSOR_API_TRY
        brotensor::build_causal_mask_row(L_, q, *mt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionDecode(void* Q, void* K_cache, void* V_cache, int32_t validLen, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window) {
    const char* L = "flashAttentionDecode";
    auto* qt = toTensor(Q);
    auto* kt = toTensor(K_cache);
    auto* vt = toTensor(V_cache);
    auto* ot = toTensor(O);
    if (!need(L, {qt, kt, vt, ot})) return;
    const int kvHeads = (numKvHeads_given && numKvHeads > 0) ? numKvHeads : numHeads;
    if (!needDecodeShapes(L, qt, kt, vt, numHeads, kvHeads)) return;
    if (validLen < qt->rows || validLen > kt->rows) {
        setError("flashAttentionDecode: validLen must lie in [Q.rows, K_cache.rows]");
        return;
    }
    BROTENSOR_API_TRY
        brotensor::flash_attention_decode(*qt, *kt, *vt, validLen, numHeads, kvHeads, *ot, static_cast<float>(attnSoftcap), window);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_flashAttentionDecodeMasked(void* Q, void* K_cache, void* V_cache, void* dMask, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window) {
    const char* L = "flashAttentionDecodeMasked";
    auto* qt = toTensor(Q);
    auto* kt = toTensor(K_cache);
    auto* vt = toTensor(V_cache);
    auto* mt = toTensor(dMask);
    auto* ot = toTensor(O);
    // The op reads a length-L_max mask and must not be handed null.
    if (!need(L, {qt, kt, vt, mt, ot})) return;
    const int kvHeads = (numKvHeads_given && numKvHeads > 0) ? numKvHeads : numHeads;
    if (!needDecodeShapes(L, qt, kt, vt, numHeads, kvHeads)) return;
    if (qt->rows != 1) { setError("flashAttentionDecodeMasked: Q must be a single row"); return; }
    if (!needMask(L, mt, kt->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_decode_masked(*qt, *kt, *vt, static_cast<const float*>(mt->data), numHeads, kvHeads,
                                                 *ot, static_cast<float>(attnSoftcap), window);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_kvCacheAppend(void* K_new, void* V_new, int32_t curLen, void* K_cache, void* V_cache) {
    const char* L = "kvCacheAppend";
    auto* knt = toTensor(K_new);
    auto* vnt = toTensor(V_new);
    auto* kct = toTensor(K_cache);
    auto* vct = toTensor(V_cache);
    if (!need(L, {knt, vnt, kct, vct}) || !needNonNegative(L, {curLen})) return;
    if (vnt->rows != knt->rows || vnt->cols != knt->cols || kct->cols != knt->cols || vct->cols != knt->cols) {
        setError("kvCacheAppend: K_new/V_new/K_cache/V_cache must share a column count, K_new and V_new a shape");
        return;
    }
    const int64_t end = static_cast<int64_t>(curLen) + knt->rows;
    if (end > kct->rows || end > vct->rows) {
        setError("kvCacheAppend: curLen + K_new.rows exceeds the cache's rows");
        return;
    }
    for (const Tensor* t : {vnt, kct, vct}) {
        if (!needSameDtype(L, "every operand", t, knt)) return;
    }
    BROTENSOR_API_TRY
        brotensor::kv_cache_append(*knt, *vnt, curLen, *kct, *vct);
    BROTENSOR_API_CATCH(L)
}

// The indices are range-checked on the host and reach the op as a device
// INT32 buffer. (The FP32-index path used to hand a host pointer to the
// device kernel.)
void bro_tensor_embeddingLookupForward(void* table, void* idxAsInt32, int32_t B, void* out) {
    const char* L = "embeddingLookupForward";
    auto* tt = toTensor(table);
    auto* it = toTensor(idxAsInt32);
    auto* ot = toTensor(out);
    if (!need(L, {tt, it, ot})) return;
    if (B <= 0) { setError("embeddingLookupForward: B must be positive"); return; }
    BROTENSOR_API_TRY
        Tensor idx;
        if (!embeddingIndices(L, it, B, tt->rows, tt->device, idx)) return;
        brotensor::embedding_lookup_forward(*tt, static_cast<const int32_t*>(idx.data), B, *ot);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_embeddingLookupBackward(void* dOut, void* idxAsInt32, int32_t B, void* dTable) {
    const char* L = "embeddingLookupBackward";
    auto* dot = toTensor(dOut);
    auto* it = toTensor(idxAsInt32);
    auto* dtt = toTensor(dTable);
    if (!need(L, {dot, it, dtt})) return;
    if (B <= 0) { setError("embeddingLookupBackward: B must be positive"); return; }
    if (!needElems(L, "dOut", dot, elems({B, dtt->cols})) || !needSameDtype(L, "dOut", dot, dtt)) return;
    BROTENSOR_API_TRY
        Tensor idx;
        if (!embeddingIndices(L, it, B, dtt->rows, dtt->device, idx)) return;
        brotensor::embedding_lookup_backward(*dot, static_cast<const int32_t*>(idx.data), B, *dtt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_sgdStep(void* param, void* grad, void* velocity, double lr, double momentum) {
    const char* L = "sgdStep";
    auto* pt = toTensor(param);
    auto* gt = toTensor(grad);
    auto* vt = toTensor(velocity);
    if (!need(L, {pt, gt, vt})) return;
    if (!needPair(L, "grad", gt, pt) || !needPair(L, "velocity", vt, pt)) return;
    // The kernels read every operand as float.
    if (pt->dtype != Dtype::FP32) { setError("sgdStep: tensors must be FP32"); return; }
    BROTENSOR_API_TRY
        brotensor::sgd_step(*pt, *gt, *vt, static_cast<float>(lr), static_cast<float>(momentum));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_adamStep(void* param, void* grad, void* m, void* v, double lr, double beta1, double beta2, double eps, int32_t step) {
    const char* L = "adamStep";
    auto* pt = toTensor(param);
    auto* gt = toTensor(grad);
    auto* mt = toTensor(m);
    auto* vt = toTensor(v);
    if (!need(L, {pt, gt, mt, vt})) return;
    if (!needPair(L, "grad", gt, pt) || !needPair(L, "m", mt, pt) || !needPair(L, "v", vt, pt)) return;
    if (step < 1) { setError("adamStep: step is the 1-based bias-correction counter"); return; }
    if (pt->dtype != Dtype::FP32) { setError("adamStep: tensors must be FP32"); return; }
    BROTENSOR_API_TRY
        brotensor::adam_step(*pt, *gt, *mt, *vt, static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps), step);
    BROTENSOR_API_CATCH(L)
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
    if (!needMask("softmaxForward", tensorFromValue(mask_bits), toTensor(logits)->size())) return;
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
    if (!needPair("layernormForward", "gamma", toTensor(gamma), toTensor(x)) ||
        !needPair("layernormForward", "beta", toTensor(beta), toTensor(x))) return;
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
    if (!needMask("maskedMeanPoolForward", tensorFromValue(mask_bits), toTensor(X)->rows)) return;
    BROTENSOR_API_TRY
        brotensor::masked_mean_pool_forward(*toTensor(X), maskPtr(mask_bits), *toTensor(y));
    BROTENSOR_API_CATCH("maskedMeanPoolForward")
}

void bro_tensor_maskedMeanPoolBackward(void* dY, uint64_t mask_bits, int32_t K, void* dX) {
    if (!need("maskedMeanPoolBackward", {dY, dX})) return;
    if (!needNonNegative("maskedMeanPoolBackward", {K})) return;
    if (!needMask("maskedMeanPoolBackward", tensorFromValue(mask_bits), K)) return;
    BROTENSOR_API_TRY
        brotensor::masked_mean_pool_backward(*toTensor(dY), maskPtr(mask_bits), K, *toTensor(dX));
    BROTENSOR_API_CATCH("maskedMeanPoolBackward")
}

double bro_tensor_softmaxXentFused(void* logits, void* target, uint64_t mask_bits, void* probs, void* dLogits) {
    if (!need("softmaxXentFused", {logits, target, probs, dLogits})) return 0.0;
    if (!needPair("softmaxXentFused", "target", toTensor(target), toTensor(logits))) return 0.0;
    if (!needMask("softmaxXentFused", tensorFromValue(mask_bits), toTensor(logits)->size())) return 0.0;
    BROTENSOR_API_TRY
        return brotensor::softmax_xent_fused(*toTensor(logits), *toTensor(target), maskPtr(mask_bits),
                                             *toTensor(probs), *toTensor(dLogits));
    BROTENSOR_API_CATCH("softmaxXentFused")
    return 0.0;
}

// The head_offsets GpuTensor is a device INT32 buffer (cumulative, n_heads+1),
// the old binding's convention.
void bro_tensor_softmaxXentFusedBatched(void* logits_BL, void* target_BL, uint64_t mask_bits, void* headOffsets, int32_t nHeads, void* probs_BL, void* dLogits_BL, void* lossPerSample) {
    const char* L = "softmaxXentFusedBatched";
    if (!need(L, {logits_BL, target_BL, headOffsets, probs_BL, dLogits_BL, lossPerSample})) return;
    const Tensor* lt = toTensor(logits_BL);
    if (!needPair(L, "target_BL", toTensor(target_BL), lt)) return;
    if (!needMask(L, tensorFromValue(mask_bits), lt->size())) return;
    if (!needPositive(L, {nHeads})) return;
    // The offsets are read on the device as slice bounds into every row: read
    // them back and check them (n_heads+1 cumulative ints within a row).
    const Tensor* ht = toTensor(headOffsets);
    if (ht->dtype != Dtype::INT32 || !needElems(L, "headOffsets", ht, static_cast<int64_t>(nHeads) + 1)) {
        if (ht->dtype != Dtype::INT32) setError("softmaxXentFusedBatched: headOffsets must be INT32 storage");
        return;
    }
    BROTENSOR_API_TRY
        auto host = ht->to(brotensor::Device::CPU);
        const auto* off = static_cast<const int32_t*>(host.host_raw());
        for (int32_t h = 0; h <= nHeads; ++h) {
            if (off[h] < 0 || off[h] > lt->cols || (h > 0 && off[h] < off[h - 1])) {
                setError("softmaxXentFusedBatched: headOffsets must be non-decreasing within [0, logits.cols]");
                return;
            }
        }
        brotensor::softmax_xent_fused_batched(*toTensor(logits_BL), *toTensor(target_BL), maskPtr(mask_bits),
                                              static_cast<const int*>(ht->data), nHeads,
                                              *toTensor(probs_BL), *toTensor(dLogits_BL), *toTensor(lossPerSample));
    BROTENSOR_API_CATCH(L)
}

} // extern "C"

namespace {

std::vector<const brotensor::Tensor*> asConst(const std::vector<brotensor::Tensor*>& v) {
    return std::vector<const brotensor::Tensor*>(v.begin(), v.end());
}

std::vector<int> asInts(const int32_t* p, uint32_t n) {
    return p ? std::vector<int>(p, p + n) : std::vector<int>();
}

// The concat/split kernels copy every part with the FIRST part's element size.
bool needOneDtype(const char* L, const std::vector<brotensor::Tensor*>& parts, const Tensor* ref) {
    for (const Tensor* p : parts) {
        if (!needSameDtype(L, "every part", p, ref)) return false;
    }
    return true;
}

// The RNG fills write Y as float.
bool needFp32Out(const char* L, const Tensor* y) {
    if (y->dtype != Dtype::FP32) {
        setError(std::string(L) + ": Y must be FP32");
        return false;
    }
    return true;
}

} // namespace

extern "C" {

void bro_tensor_concatRows(uint64_t parts_bits, void* out) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatRows", {out}) || !readTensorArray(parts_bits, parts, "concatRows")) return;
    if (!parts.empty() && !needOneDtype("concatRows", parts, parts[0])) return;
    BROTENSOR_API_TRY
        brotensor::concat_rows(asConst(parts), *toTensor(out));
    BROTENSOR_API_CATCH("concatRows")
}

// split_rows copies consecutive runs of `in` into each part at the part's
// size: the parts must fit in `in` and share its dtype.
void bro_tensor_splitRows(void* in, uint64_t parts_bits) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("splitRows", {in}) || !readTensorArray(parts_bits, parts, "splitRows")) return;
    const Tensor* it = toTensor(in);
    if (!needOneDtype("splitRows", parts, it)) return;
    int64_t total = 0;
    for (const Tensor* p : parts) total += p->size();
    if (total > static_cast<int64_t>(it->size())) {
        setError("splitRows: the parts hold " + std::to_string(total) + " elements, in has " + std::to_string(it->size()));
        return;
    }
    BROTENSOR_API_TRY
        brotensor::split_rows(*toTensor(in), parts);
    BROTENSOR_API_CATCH("splitRows")
}

// Every part is (B, d_i) with one B.
void bro_tensor_concatBatchedRows(uint64_t parts_bits, void* out) {
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatBatchedRows", {out}) || !readTensorArray(parts_bits, parts, "concatBatchedRows")) return;
    if (!parts.empty()) {
        if (!needOneDtype("concatBatchedRows", parts, parts[0])) return;
        for (const Tensor* p : parts) {
            if (p->rows != parts[0]->rows) {
                setError("concatBatchedRows: every part must have the same row count");
                return;
            }
        }
    }
    BROTENSOR_API_TRY
        brotensor::concat_batched_rows(asConst(parts), *toTensor(out));
    BROTENSOR_API_CATCH("concatBatchedRows")
}

// C_per_part is an i32[] view INTO THE MOVING HEAP, and readTensorArray walks
// the parts array with allocating getProperty/getElement calls: copy the ints
// out first, or the second read sees a buffer the collector has moved (the
// gcstress run caught exactly that).
void bro_tensor_concatNchwChannels(uint64_t parts_bits, int32_t N, int32_t H, int32_t W, const int32_t* C_per_part, uint32_t C_len, void* out) {
    const std::vector<int> channels = asInts(C_per_part, C_len);
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatNchwChannels", {out}) || !readTensorArray(parts_bits, parts, "concatNchwChannels")) return;
    BROTENSOR_API_TRY
        brotensor::concat_nchw_channels(asConst(parts), N, H, W, channels, *toTensor(out));
    BROTENSOR_API_CATCH("concatNchwChannels")
}

void bro_tensor_concatNchwChannelsBackward(void* dY, int32_t N, int32_t H, int32_t W, const int32_t* C_per_part, uint32_t C_len, uint64_t parts_bits) {
    const std::vector<int> channels = asInts(C_per_part, C_len);
    std::vector<brotensor::Tensor*> parts;
    if (!need("concatNchwChannelsBackward", {dY}) || !readTensorArray(parts_bits, parts, "concatNchwChannelsBackward")) return;
    BROTENSOR_API_TRY
        brotensor::concat_nchw_channels_backward(*toTensor(dY), N, H, W, channels, parts);
    BROTENSOR_API_CATCH("concatNchwChannelsBackward")
}

// ---- counter-based RNG + init (key / counter / state: BigInt or Number) ----

void bro_tensor_randUniform(uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randUniform", {Y}) || !needFp32Out("randUniform", toTensor(Y))) return;
    BROTENSOR_API_TRY
        brotensor::rand_uniform(bronze::embed::toUint64(bronze::Value{key_bits}),
                                bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randUniform")
}

void bro_tensor_randn(uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randn", {Y}) || !needFp32Out("randn", toTensor(Y))) return;
    BROTENSOR_API_TRY
        brotensor::randn(bronze::embed::toUint64(bronze::Value{key_bits}),
                         bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randn")
}

void bro_tensor_randBernoulli(double p, uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randBernoulli", {Y}) || !needFp32Out("randBernoulli", toTensor(Y))) return;
    BROTENSOR_API_TRY
        brotensor::rand_bernoulli(static_cast<float>(p), bronze::embed::toUint64(bronze::Value{key_bits}),
                                  bronze::embed::toUint64(bronze::Value{counter_bits}), *toTensor(Y));
    BROTENSOR_API_CATCH("randBernoulli")
}

void bro_tensor_randnTruncated(double lo, double hi, uint64_t key_bits, uint64_t counter_bits, void* Y) {
    if (!need("randnTruncated", {Y}) || !needFp32Out("randnTruncated", toTensor(Y))) return;
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
