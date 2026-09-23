#pragma once

// native_tensor_extra_decl.h — the brotensor ops that had no JS binding at all
// (neither in the QuickJS binding nor in the bronze port). Bodies in
// native_tensor_extra.cpp (elementwise / reduction / index + the INT32
// upload/download pair), native_tensor_extra_nn.cpp (dense, LSTM, attention,
// RoPE) and native_tensor_extra_conv.cpp (StyleGAN3-R, deformable conv,
// pixel shuffle, unpatchify); wrappers in js/tensor_extra.js.
//
// `uint64_t *_bits` parameters are `dynamic` slots: an optional GpuTensor
// (null / undefined = absent) or, for `seed_bits`, a Number / BigInt.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- INT32 upload / download (native_tensor_extra.cpp) ----------------------
void bro_tensor_uploadInt32(void* self, const int32_t* data, uint32_t data_len);
void bro_tensor_downloadInt32(void* self, bronze_native_buffer* out);

// --- elementwise ------------------------------------------------------------
void bro_tensor_addChannelBiasInplace(void* y, void* bias, int32_t C, int32_t L);
void bro_tensor_addRowBiasInplace(void* y, void* bias);
void bro_tensor_axpbyInplace(void* y, void* x, double a, double b);
void bro_tensor_sinForward(void* x, void* y);
void bro_tensor_sinBackward(void* x, void* dY, void* dX);
void bro_tensor_cosForward(void* x, void* y);
void bro_tensor_cosBackward(void* x, void* dY, void* dX);
void bro_tensor_rsqrtForward(void* x, void* y);
void bro_tensor_rsqrtBackward(void* y, void* dY, void* dX);
void bro_tensor_pixelNormForward(void* X, double eps, void* Y);
void bro_tensor_pixelNormBackward(void* X, void* dY, double eps, void* dX);
void bro_tensor_thresholdU8(void* X, double t, void* Y);
void bro_tensor_rowsCountAbove(void* X, double tLo, double tHi, void* counts);

// --- softmax / loss ---------------------------------------------------------
void bro_tensor_softmaxRowsForward(void* X, void* Y, int32_t rows, int32_t cols);
double bro_tensor_softmaxXent(void* logits, void* target, void* probs, void* dLogits, uint64_t mask_bits);

// --- copies, scatters, segments ---------------------------------------------
void bro_tensor_copyD2DStrided(void* src, int32_t srcOff, int32_t srcPitch, void* dst, int32_t dstOff,
                               int32_t dstPitch, int32_t width, int32_t height);
void bro_tensor_scatterRows(void* Y, void* Idx, void* X);
void bro_tensor_segmentSoftmaxStats(void* logits, void* segOffsets, void* out);

// --- masked-diffusion token selection ---------------------------------------
void bro_tensor_maskedDiffusionScores(void* logits, void* tokens, int32_t T, int32_t C, int32_t V,
                                      int32_t maskId, double guidanceScale, double layerPenalty,
                                      double positionTemperature, double classTemperature,
                                      double classTopFrac, uint64_t seed_bits,
                                      void* pred, void* scores, void* confidence);
void bro_tensor_maskedDiffusionCommit(void* pred, void* idx, int32_t k, int32_t step, void* tokens,
                                      void* unmaskStep);

// --- dense (native_tensor_extra_nn.cpp) -------------------------------------
void bro_tensor_linearForwardBatchedEx(void* W, uint64_t bias_bits, void* X, int32_t act, int32_t epilogue,
                                       uint64_t workspace_bits, void* Y);
void bro_tensor_linearForwardBatchedFp16Act(void* W, uint64_t bias_bits, void* X, int32_t act, void* Y);
void bro_tensor_matmulAbt(void* A, void* B, void* C, int32_t batch, int32_t M, int32_t N, int32_t K,
                          double strideA, double strideB, double strideC, uint64_t bias_bits, int32_t act);

// --- LSTM -------------------------------------------------------------------
void bro_tensor_lstmForwardTrain(void* X, void* W_ih, void* W_hh, uint64_t b_ih_bits, uint64_t b_hh_bits,
                                 uint64_t h0_bits, uint64_t c0_bits, int32_t T, int32_t B,
                                 void* Y, void* gates, void* C, uint64_t hT_bits, uint64_t cT_bits);
void bro_tensor_lstmBackward(void* X, void* W_ih, void* W_hh, uint64_t h0_bits, uint64_t c0_bits,
                             void* Y, void* gates, void* C, void* dY, int32_t T, int32_t B,
                             void* dX, void* dW_ih, void* dW_hh, uint64_t db_ih_bits, uint64_t db_hh_bits,
                             uint64_t dh0_bits, uint64_t dc0_bits);

// --- attention + RoPE -------------------------------------------------------
void bro_tensor_flashAttentionGqaForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numQHeads,
                                         int32_t numKvHeads, bool causal, void* O);
void bro_tensor_flashAttentionPackedQkvForward(void* QKV, void* seqBounds, int32_t numHeads, int32_t window,
                                               void* O);
void bro_tensor_flashAttentionPackedQkvBackward(void* QKV, void* dO, void* seqBounds, int32_t numHeads,
                                                int32_t window, void* dQKV);
void bro_tensor_relPosBiasXlForward(void* Qv, void* Pk, int32_t numHeads, int32_t headDim, void* Bias);
void bro_tensor_ropeApplyPerhead(void* X, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* Y);
void bro_tensor_ropeQkvPackedInplace(void* QKV, void* cosTbl, void* sinTbl, void* pos, int32_t numHeads,
                                     int32_t headDim);

// --- StyleGAN3-R (native_tensor_extra_conv.cpp) -----------------------------
void bro_tensor_biasActForward(void* X, uint64_t b_bits, int32_t N, int32_t C, int32_t HW, int32_t act,
                               double alpha, double gain, double clamp, void* Y);
void bro_tensor_biasActBackward(void* dY, void* X, uint64_t b_bits, int32_t N, int32_t C, int32_t HW, int32_t act,
                                double alpha, double gain, double clamp, void* dX, uint64_t dB_bits);
void bro_tensor_upfirdn2dForward(void* X, void* f, int32_t N, int32_t C, int32_t H, int32_t W, int32_t fH,
                                 int32_t fW, int32_t upX, int32_t upY, int32_t downX, int32_t downY,
                                 int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1, bool flipFilter,
                                 double gain, void* Y);
void bro_tensor_upfirdn2dBackward(void* dY, void* f, int32_t N, int32_t C, int32_t H, int32_t W, int32_t fH,
                                  int32_t fW, int32_t upX, int32_t upY, int32_t downX, int32_t downY,
                                  int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1, bool flipFilter,
                                  double gain, void* dX);
void bro_tensor_modulatedConv2dForward(void* X, void* W, void* s, int32_t N, int32_t Cin, int32_t H, int32_t Wd,
                                       int32_t Cout, int32_t kH, int32_t kW, int32_t padH, int32_t padW,
                                       bool demodulate, double eps, void* dcoef, void* Y);
void bro_tensor_modulatedConv2dBackward(void* X, void* W, void* s, void* dcoef, void* dY, int32_t N, int32_t Cin,
                                        int32_t H, int32_t Wd, int32_t Cout, int32_t kH, int32_t kW, int32_t padH,
                                        int32_t padW, bool demodulate, double eps, void* dX, uint64_t dW_bits,
                                        void* ds);
void bro_tensor_filteredLreluForward(void* X, void* fu, void* fd, uint64_t b_bits, int32_t N, int32_t C, int32_t H,
                                     int32_t W, int32_t up, int32_t down, int32_t padX0, int32_t padX1,
                                     int32_t padY0, int32_t padY1, double gain, double slope, double clamp,
                                     void* upBuf, void* actBuf, void* Y);
void bro_tensor_filteredLreluBackward(void* dY, void* X, void* fu, void* fd, uint64_t b_bits, int32_t N, int32_t C,
                                      int32_t H, int32_t W, int32_t up, int32_t down, int32_t padX0, int32_t padX1,
                                      int32_t padY0, int32_t padY1, double gain, double slope, double clamp,
                                      uint64_t upBuf_bits, void* dX, uint64_t dB_bits);

// --- deformable conv, pixel shuffle, unpatchify ------------------------------
void bro_tensor_deformConv2dForward(void* X, void* offset, uint64_t mask_bits, void* Wt, uint64_t bias_bits,
                                    int32_t N, int32_t Cin, int32_t H, int32_t W, int32_t Cout, int32_t kH,
                                    int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH,
                                    int32_t dW, int32_t groups, int32_t deformGroups, void* Y);
void bro_tensor_pixelShuffleUpsample2xForward(void* X, int32_t N, int32_t Cin, int32_t H, int32_t W, int32_t Cout,
                                              void* Y);
void bro_tensor_patchUnpackForward(void* tokens, int32_t hp, int32_t wp, int32_t P, int32_t Ctotal, int32_t Ckeep,
                                   bool channelMajor, void* Y);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api {
bool registerTensorNatives_extra(std::string* error);
bool registerTensorNatives_extra_nn(std::string* error);
bool registerTensorNatives_extra_conv(std::string* error);
}
#endif
