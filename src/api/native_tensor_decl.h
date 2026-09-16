#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Core / GpuTensor Lifecycle ---
void* bro_tensor_GpuTensor_ctor(void);
void bro_tensor_GpuTensor_dtor(void* self);
void* bro_tensor_createTensor(int32_t rows, int32_t cols, const char* dtype);
int32_t bro_tensor_GpuTensor_rows_get(void* self);
int32_t bro_tensor_GpuTensor_cols_get(void* self);
int32_t bro_tensor_GpuTensor_size_get(void* self);
int32_t bro_tensor_GpuTensor_bytes_get(void* self);
void bro_tensor_GpuTensor_zero(void* self);
void bro_tensor_GpuTensor_resize(void* self, int32_t rows, int32_t cols, const char* dtype);
const char* bro_tensor_GpuTensor_dtype(void* self);
void* bro_tensor_GpuTensor_clone(void* self);
void bro_tensor_GpuTensor_upload(void* self, const float* data, uint32_t data_len);
void bro_tensor_GpuTensor_download(void* self, bronze_native_buffer* out);
void bro_tensor_GpuTensor_uploadFp16(void* self, const uint16_t* data, uint32_t data_len);
void bro_tensor_GpuTensor_downloadFp16(void* self, bronze_native_buffer* out);
void bro_tensor_GpuTensor_uploadInt8(void* self, const int8_t* data, uint32_t data_len);
void bro_tensor_GpuTensor_downloadInt8(void* self, bronze_native_buffer* out);

// --- Backend Control ---
bool bro_tensor_available_get(void);
const char* bro_tensor_backend_get(void);
void bro_tensor_init(void);
void bro_tensor_sync(void);

// --- Math & Activation Ops ---
void bro_tensor_linearForward(void* W, void* b, void* x, void* y);
void bro_tensor_linearBackward(void* W, void* x, void* dY, void* dX, void* dW, void* dB);
void bro_tensor_reluForward(void* x, void* y);
void bro_tensor_reluBackward(void* x, void* dY, void* dX);
void bro_tensor_tanhForward(void* x, void* y);
void bro_tensor_tanhBackward(void* y, void* dY, void* dX);
void bro_tensor_sigmoidForward(void* x, void* y);
void bro_tensor_sigmoidBackward(void* y, void* dY, void* dX);
void bro_tensor_addInplace(void* y, void* x);
void bro_tensor_addScalarInplace(void* y, double s);
void bro_tensor_scaleInplace(void* y, double s);
void bro_tensor_mulInplace(void* y, void* x);
void bro_tensor_clamp(void* y, double lo, double hi);
void bro_tensor_siluForward(void* x, void* y);
void bro_tensor_siluBackward(void* x, void* dY, void* dX);
void bro_tensor_geluForward(void* x, void* y);
void bro_tensor_geluBackward(void* x, void* dY, void* dX);
void bro_tensor_geluExactForward(void* x, void* y);
void bro_tensor_geluExactBackward(void* x, void* dY, void* dX);
void bro_tensor_quickGeluForward(void* x, void* y);
void bro_tensor_quickGeluBackward(void* x, void* dY, void* dX);
void bro_tensor_swigluForward(void* X, void* Y);
void bro_tensor_swigluBackward(void* X, void* dY, void* dX);
void bro_tensor_gegluForward(void* X, void* Y);
void bro_tensor_gegluBackward(void* X, void* dY, void* dX);
void bro_tensor_gegluExactForward(void* X, void* Y);
void bro_tensor_gegluExactBackward(void* X, void* dY, void* dX);
void bro_tensor_softmaxForward(void* logits, void* probs, double temp);
void bro_tensor_softmaxBackward(void* probs, void* dProbs, void* dLogits);
void bro_tensor_matmul(void* A, void* B, void* C);
void bro_tensor_matmulBackward(void* A, void* B, void* dC, void* dA, void* dB);
void bro_tensor_conv2dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* Y);
void bro_tensor_conv2dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dX);
void bro_tensor_conv2dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dWt);
void bro_tensor_conv2dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t H_out, int32_t W_out, void* dB);
void bro_tensor_upsampleNearest2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
void bro_tensor_upsampleNearest2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX);
void bro_tensor_upsampleBilinear2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
void bro_tensor_upsampleBilinear2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX);
void bro_tensor_downsampleAvg2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
void bro_tensor_downsampleAvg2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX);
void bro_tensor_sumRows(void* X, void* Y);
void bro_tensor_sumCols(void* X, void* Y);
void bro_tensor_argmaxRows(void* X, void* Idx);
void bro_tensor_copyD2D(void* src, int32_t srcOff, void* dst, int32_t dstOff, int32_t n);
void bro_tensor_nchwToSequence(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
void bro_tensor_sequenceToNchw(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
void bro_tensor_interp2dForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y);
void bro_tensor_interp2dAlignCornersForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y);
void bro_tensor_unfold2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padT, int32_t padB, int32_t padL, int32_t padR, int32_t mode, void* Y);
void bro_tensor_l2NormalizeNchwForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y);
void bro_tensor_convexUpsampleForward(void* X, void* Mask, int32_t N, int32_t C, int32_t H, int32_t W, int32_t scale, void* Y);
double bro_tensor_mseVecForward(void* pred, void* target);
void bro_tensor_mseVecBackward(void* pred, void* target, void* dPred);
void bro_tensor_mseVecPerSample(void* pred, void* target, void* dPred, void* lossPerSample);

// --- NN / Normalization / Attention / Optim Ops ---
void bro_tensor_layernormBackward(void* dY, void* xhat, void* gamma, double rstd, void* dX, void* dGamma, void* dBeta);
void bro_tensor_layernormForwardInferenceBatched(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps);
void bro_tensor_layernormForwardInferenceBatchedFp16(void* X_RD, void* gamma, void* beta, void* Y_RD, double eps);
void bro_tensor_rmsNormForward(void* X, void* gamma, double eps, void* Y);
void bro_tensor_rmsNormBackward(void* X, void* gamma, void* dY, double eps, void* dX, void* dGamma);
void bro_tensor_groupNormForward(void* X, void* gamma, void* beta, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* Y);
void bro_tensor_groupNormBackward(void* X, void* gamma, void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t numGroups, double eps, void* dX, void* dGamma, void* dBeta);
void bro_tensor_ropeForward(void* X, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* Y);
void bro_tensor_ropeBackward(void* dY, int32_t headDim, int32_t numHeads, int32_t seqOffset, double thetaBase, void* dX);
void bro_tensor_ropeApply(void* X, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* Y);
void bro_tensor_ropeApplyBackward(void* dY, int32_t headDim, int32_t numHeads, void* dX);
void bro_tensor_modulate(void* X, void* scale, void* shift, void* Y);
void bro_tensor_broadcastMul(void* X, void* v, void* Y);
void bro_tensor_attentionTokenMoments(void* Attn, int32_t h_lat, int32_t w_lat, void* mass, void* centroid);
void bro_tensor_buildSlotMask(void* x, int32_t offset, int32_t K, int32_t stride, void* mask);
void bro_tensor_buildCausalMaskRow(int32_t L, int32_t q, void* mask);
void bro_tensor_flashAttentionDecode(void* Q, void* K_cache, void* V_cache, int32_t validLen, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window);
void bro_tensor_flashAttentionDecodeMasked(void* Q, void* K_cache, void* V_cache, void* dMask, int32_t numHeads, void* O, bool numKvHeads_given, int32_t numKvHeads, double attnSoftcap, int32_t window);
void bro_tensor_kvCacheAppend(void* K_new, void* V_new, int32_t curLen, void* K_cache, void* V_cache);
void bro_tensor_embeddingLookupForward(void* table, void* idxAsInt32, int32_t B, void* out);
void bro_tensor_embeddingLookupBackward(void* dOut, void* idxAsInt32, int32_t B, void* dTable);
void bro_tensor_sgdStep(void* param, void* grad, void* velocity, double lr, double momentum);
void bro_tensor_adamStep(void* param, void* grad, void* m, void* v, double lr, double beta1, double beta2, double eps, int32_t step);

#ifdef __cplusplus
}
#endif
