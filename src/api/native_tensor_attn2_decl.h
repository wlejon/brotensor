#pragma once

// The second attention family of bro.tensor, restored from the QuickJS
// binding (tensor_bindings_attention.cpp): the SAM / ViTDet decomposed 2D
// rel-pos self-attention (global + windowed), packed variable-length flash
// attention (forward + backward), the Qwen3-Next gated delta rule (chunked +
// step) and Qwen-VL M-RoPE.
//
// Conventions, all inherited from the group's siblings:
//   - a required GpuTensor crosses as `void*` (registered kTensorCls),
//   - a nullable GpuTensor crosses as `uint64_t <name>_bits` (registered
//     "dynamic") and resolves with tensorFromValue() / the file-local
//     idxPtr() for the INT32 index buffers,
//   - the cu_seqlens / M-RoPE position streams keep the old binding's
//     "GpuTensor viewed as INT32 device storage" convention (the ops take a
//     `const int32_t*` that is a device pointer on CUDA/Metal and a host
//     pointer on CPU), exactly like embeddingLookupForward's idx.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// selfAttentionDecomposedRelPosForward(X,Wq,bq|null,Wk,bk|null,Wv,bv|null,Wo,bo|null,relPosH,relPosW,numHeads,gridH,gridW,scale,O)
void bro_tensor_selfAttentionDecomposedRelPosForward(void* X, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits,
                                                     void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits,
                                                     void* relPosH, void* relPosW,
                                                     int32_t numHeads, int32_t gridH, int32_t gridW, double scale,
                                                     void* O);
// selfAttentionDecomposedRelPosWindowedForward(X,Wq,bq|null,Wk,bk|null,Wv,bv|null,Wo,bo|null,relPosH,relPosW,numHeads,gridH,gridW,window,scale,O)
void bro_tensor_selfAttentionDecomposedRelPosWindowedForward(void* X, void* Wq, uint64_t bq_bits, void* Wk, uint64_t bk_bits,
                                                             void* Wv, uint64_t bv_bits, void* Wo, uint64_t bo_bits,
                                                             void* relPosH, void* relPosW,
                                                             int32_t numHeads, int32_t gridH, int32_t gridW,
                                                             int32_t window, double scale, void* O);
// flashAttentionVarlenForward(Q,K,V,cuSeqQ|null,cuSeqK|null,batch,maxQ,maxK,numHeads,headDim,causal,O)
void bro_tensor_flashAttentionVarlenForward(void* Q, void* K, void* V, uint64_t cuSeqQ_bits, uint64_t cuSeqK_bits,
                                            int32_t batch, int32_t maxQ, int32_t maxK,
                                            int32_t numHeads, int32_t headDim, bool causal, void* O);
// flashAttentionVarlenBackward(Q,K,V,O,dO,cuSeqQ|null,cuSeqK|null,batch,maxQ,maxK,numHeads,headDim,causal,dQ,dK,dV)
void bro_tensor_flashAttentionVarlenBackward(void* Q, void* K, void* V, void* O, void* dO,
                                             uint64_t cuSeqQ_bits, uint64_t cuSeqK_bits,
                                             int32_t batch, int32_t maxQ, int32_t maxK,
                                             int32_t numHeads, int32_t headDim, bool causal,
                                             void* dQ, void* dK, void* dV);
// gatedDeltaRuleChunked(Q,K,V,aRaw,beta,logA,numHeads,d_k,d_v,state,O)
void bro_tensor_gatedDeltaRuleChunked(void* Q, void* K, void* V, void* aRaw, void* beta, void* logA,
                                      int32_t numHeads, int32_t d_k, int32_t d_v, void* state, void* O);
// gatedDeltaRuleStep(Q,K,V,aRaw,beta,logA,numHeads,d_k,d_v,state,O)
void bro_tensor_gatedDeltaRuleStep(void* Q, void* K, void* V, void* aRaw, void* beta, void* logA,
                                   int32_t numHeads, int32_t d_k, int32_t d_v, void* state, void* O);
// ropeApplyMrope(X,cosT,sinT,cosH,sinH,cosW,sinW,posT|null,posH|null,posW|null,headDim,numHeads,d_t,d_h,d_w,Y)
void bro_tensor_ropeApplyMrope(void* X, void* cosT, void* sinT, void* cosH, void* sinH, void* cosW, void* sinW,
                               uint64_t posT_bits, uint64_t posH_bits, uint64_t posW_bits,
                               int32_t headDim, int32_t numHeads, int32_t d_t, int32_t d_h, int32_t d_w,
                               void* Y);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_attn2(std::string* error); }
#endif
