#pragma once

// native_tensor_int8_decl.h — the INT8 / GGUF-quant group of the restored
// `bro.tensor.*` free-function surface (the QuickJS binding's
// tensor_bindings_int8.cpp).
//
// Two families live here:
//   * W8A16 — an INT8 weight matrix paired with a per-output-row FP32 dequant
//     scale (the quantize_int8_per_row_host convention), FP16 activations:
//     matmul, conv2d, conv3d, batched linear, the diffusion resblock, the
//     flash-attention triplet and the T5-bias self-attention.
//   * GGUF k-quant — Q4_K / Q6_K / Q8_0 block-quantised weights: a standalone
//     dequant to FP16 plus the fused single-token / batched linears.
//
// Every op here is GPU-only: the CPU backend leaves the FP16 / INT8-W8A16 /
// GGUF-quant vtable slots null and the dispatcher throws "not implemented on
// CPU". The sole exception is quantizeInt8PerRowHost, a pure host helper over
// raw buffers that is not device-dispatched at all.
//
// Nullable GpuTensor slots (bias|null, mask|null, ...) cross as `dynamic` —
// the raw Value bits — and resolve with tensorFromValue() / maskPtr().

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- GGUF k-quant dequant (Q4_K / Q6_K / Q8_0 -> FP16) ---
// dequantQ4kToFp16(W_q4k, W_fp16)
void bro_tensor_dequantQ4kToFp16(void* W_q, void* W_fp16);
// dequantQ6kToFp16(W_q6k, W_fp16)
void bro_tensor_dequantQ6kToFp16(void* W_q, void* W_fp16);
// dequantQ8_0ToFp16(W_q8_0, W_fp16)
void bro_tensor_dequantQ8_0ToFp16(void* W_q, void* W_fp16);

// --- GGUF k-quant weight-only linear (W = k-quant, x / y FP16) ---
// linearForwardQ4kFp16(W_q4k, bias|null, x, y) — single token, x:(in,1), y:(out,1)
void bro_tensor_linearForwardQ4kFp16(void* W_q, uint64_t bias_bits, void* x, void* y);
// linearForwardQ6kFp16(W_q6k, bias|null, x, y)
void bro_tensor_linearForwardQ6kFp16(void* W_q, uint64_t bias_bits, void* x, void* y);
// linearForwardQ8_0Fp16(W_q8_0, bias|null, x, y)
void bro_tensor_linearForwardQ8_0Fp16(void* W_q, uint64_t bias_bits, void* x, void* y);
// linearForwardBatchedQ4kFp16(W_q4k, bias|null, X_BD, Y_BD) — (B,in) -> (B,out)
void bro_tensor_linearForwardBatchedQ4kFp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD);
// linearForwardBatchedQ6kFp16(W_q6k, bias|null, X_BD, Y_BD)
void bro_tensor_linearForwardBatchedQ6kFp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD);
// linearForwardBatchedQ8_0Fp16(W_q8_0, bias|null, X_BD, Y_BD)
void bro_tensor_linearForwardBatchedQ8_0Fp16(void* W_q, uint64_t bias_bits, void* X_BD, void* Y_BD);

// --- W8A16 host quantiser ---
// quantizeInt8PerRowHost(W_fp16:Uint16Array, out, in) -> Int8Array weights.
// The matching FP32 per-row scales stay in a thread-local slot the JS wrapper
// reads back with the call below, so one quantise pass answers both halves of
// the old `{ weights, scales }` object.
void bro_tensor_quantizeInt8PerRowHost(const uint16_t* W_fp16, uint32_t W_fp16_len,
                                       int32_t out, int32_t in, bronze_native_buffer* weights_out);
// The scales of the last quantizeInt8PerRowHost on this thread, as Float32Array.
void bro_tensor_quantizeInt8PerRowHostScales(bronze_native_buffer* scales_out);

// --- W8A16 dense ---
// matmulInt8wFp16(W_int8, scales, X, Y) — Y(out,B) = dequant(W)(out,in) @ X(in,B)
void bro_tensor_matmulInt8wFp16(void* W_int8, void* scales, void* X, void* Y);
// linearForwardBatchedInt8wFp16(W_int8, scales, bias|null, X_BD, Y_BD)
void bro_tensor_linearForwardBatchedInt8wFp16(void* W_int8, void* scales, uint64_t bias_bits,
                                              void* X_BD, void* Y_BD);

// --- W8A16 convolution ---
// conv2dInt8wFp16Forward(X, W_int8, scales, bias|null,
//                        N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, Y)
void bro_tensor_conv2dInt8wFp16Forward(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t H, int32_t Win,
                                       int32_t C_out, int32_t kH, int32_t kW,
                                       int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                       int32_t dH, int32_t dW, int32_t groups, void* Y);
// conv3dInt8wFp16Forward(X, W_int8, scales, bias|null,
//                        N, C_in, T, H, W, C_out, kT, kH, kW,
//                        sT, sH, sW, pT, pH, pW, dT, dH, dW, groups, Y)
void bro_tensor_conv3dInt8wFp16Forward(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t T, int32_t H, int32_t Win,
                                       int32_t C_out, int32_t kT, int32_t kH, int32_t kW,
                                       int32_t sT, int32_t sH, int32_t sW,
                                       int32_t pT, int32_t pH, int32_t pW,
                                       int32_t dT, int32_t dH, int32_t dW,
                                       int32_t groups, void* Y);

// --- W8A16 diffusion resblock (options object on the JS side) ---
// resblockForwardInt8wFp16(opts): X, gamma1, beta1, W1_int8, s1, b1|null,
//   t_emb_shift|null, gamma2, beta2, W2_int8, s2, b2|null, Wskip_int8|null,
//   sskip|null, bskip|null, N, C_in, C_out, H, W, numGroups, eps, Y
void bro_tensor_resblockForwardInt8wFp16(void* X, void* gamma1, void* beta1,
                                         void* W1_int8, void* s1, uint64_t b1_bits,
                                         uint64_t t_emb_shift_bits,
                                         void* gamma2, void* beta2,
                                         void* W2_int8, void* s2, uint64_t b2_bits,
                                         uint64_t Wskip_bits, uint64_t sskip_bits, uint64_t bskip_bits,
                                         int32_t N, int32_t C_in, int32_t C_out, int32_t H, int32_t Win,
                                         int32_t numGroups, double eps, void* Y);

// --- W8A16 flash-attention triplet ---
// flashAttentionProjectKvInt8wFp16(ctx, Wk_int8, sk, bk|null, Wv_int8, sv, bv|null, K_out, V_out)
void bro_tensor_flashAttentionProjectKvInt8wFp16(void* ctx, void* Wk_int8, void* sk, uint64_t bk_bits,
                                                 void* Wv_int8, void* sv, uint64_t bv_bits,
                                                 void* K_out, void* V_out);
// flashAttentionQWithKvCachedInt8wFp16(X, K, V, Wq_int8, sq, bq|null,
//                                      Wo_int8, so, bo|null, mask|null, numHeads, causal, O)
void bro_tensor_flashAttentionQWithKvCachedInt8wFp16(void* X, void* K, void* V,
                                                     void* Wq_int8, void* sq, uint64_t bq_bits,
                                                     void* Wo_int8, void* so, uint64_t bo_bits,
                                                     uint64_t mask_bits, int32_t numHeads,
                                                     bool causal, void* O);
// flashAttentionQkvoInt8wFp16(opts): X, Ctx|null, Wq_int8, sq, bq|null,
//   Wk_int8, sk, bk|null, Wv_int8, sv, bv|null, Wo_int8, so, bo|null,
//   mask|null, numHeads, causal, O
void bro_tensor_flashAttentionQkvoInt8wFp16(void* X, uint64_t Ctx_bits,
                                            void* Wq_int8, void* sq, uint64_t bq_bits,
                                            void* Wk_int8, void* sk, uint64_t bk_bits,
                                            void* Wv_int8, void* sv, uint64_t bv_bits,
                                            void* Wo_int8, void* so, uint64_t bo_bits,
                                            uint64_t mask_bits, int32_t numHeads,
                                            bool causal, void* O);

// --- W8A16 T5-style bias self-attention ---
// selfAttentionBiasInt8wFp16(X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv,
//                            Wo_int8, so, mask|null, attnBias|null, numHeads, scale, O)
void bro_tensor_selfAttentionBiasInt8wFp16(void* X,
                                           void* Wq_int8, void* sq,
                                           void* Wk_int8, void* sk,
                                           void* Wv_int8, void* sv,
                                           void* Wo_int8, void* so,
                                           uint64_t mask_bits, uint64_t attnBias_bits,
                                           int32_t numHeads, double scale, void* O);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_int8(std::string* error); }
#endif
