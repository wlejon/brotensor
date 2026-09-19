#pragma once

// native_tensor_conv_decl.h — the restored `conv` group: the 2-D spatial
// family the QuickJS binding exposed under bro.tensor (tensor_bindings_conv.cpp)
// and the bronze port had not carried over yet — interp2d backward, pad2d /
// slice2d, max-pool / adaptive-avg-pool, transposed conv, conv3d, the
// Swin/SAM/Qwen-VL window + patch-merge gathers, batch-norm (train / backward /
// inference) and the two image-preprocessing helpers.
//
// conv2d* / upsample* / interp2d forward / unfold2d / l2Normalize /
// convexUpsample already live in native_tensor_ops.cpp + api.cpp; they are not
// repeated here.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// interp2dBackward(dY,N,C,H_in,W_in,H_out,W_out,mode,dX)
void bro_tensor_interp2dBackward(void* dY, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* dX);

// pad2dForward(X,N,C,H,W,padT,padB,padL,padR,mode,Y)   — mode 0 zero / 1 reflect / 2 replicate
void bro_tensor_pad2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t padT, int32_t padB, int32_t padL, int32_t padR, int32_t mode, void* Y);
// pad2dBackward(dY,N,C,H,W,padT,padB,padL,padR,mode,dX)
void bro_tensor_pad2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t padT, int32_t padB, int32_t padL, int32_t padR, int32_t mode, void* dX);

// slice2dForward(X,N,C,H,W,h0,w0,H_out,W_out,Y)
void bro_tensor_slice2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* Y);
// slice2dBackward(dY,N,C,H,W,h0,w0,H_out,W_out,dX)
void bro_tensor_slice2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* dX);

// maxPool2dForward(X,N,C,H,W,kH,kW,sH,sW,padH,padW,Y,Idx)  — Idx is INT32, feeds the backward
void bro_tensor_maxPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padH, int32_t padW, void* Y, void* Idx);
// maxPool2dBackward(dY,Idx,N,C,H,W,H_out,W_out,dX)
void bro_tensor_maxPool2dBackward(void* dY, void* Idx, int32_t N, int32_t C, int32_t H, int32_t W, int32_t H_out, int32_t W_out, void* dX);

// adaptiveAvgPool2dForward(X,N,C,H,W,H_out,W_out,Y)
void bro_tensor_adaptiveAvgPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t H_out, int32_t W_out, void* Y);
// adaptiveAvgPool2dBackward(dY,N,C,H,W,H_out,W_out,dX)
void bro_tensor_adaptiveAvgPool2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, int32_t H_out, int32_t W_out, void* dX);

// convTranspose2dForward(X,Wt,bias|null,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,Y)
void bro_tensor_convTranspose2dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t opH, int32_t opW, int32_t dH, int32_t dW, int32_t groups, void* Y);
// convTranspose2dBackwardInput(Wt,dY,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,dX)
void bro_tensor_convTranspose2dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t opH, int32_t opW, int32_t dH, int32_t dW, int32_t groups, void* dX);
// convTranspose2dBackwardWeight(X,dY,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,dWt)
void bro_tensor_convTranspose2dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t opH, int32_t opW, int32_t dH, int32_t dW, int32_t groups, void* dWt);
// convTranspose2dBackwardBias(dY,N,C_out,H_out,W_out,dB)
void bro_tensor_convTranspose2dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t H_out, int32_t W_out, void* dB);

// conv3dForward(X,Wt,bias|null,N,C_in,T,H,W,C_out,kT,kH,kW,sT,sH,sW,pT,pH,pW,dT,dH,dW,groups,Y)
void bro_tensor_conv3dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in, int32_t T, int32_t H, int32_t W, int32_t C_out, int32_t kT, int32_t kH, int32_t kW, int32_t sT, int32_t sH, int32_t sW, int32_t pT, int32_t pH, int32_t pW, int32_t dT, int32_t dH, int32_t dW, int32_t groups, void* Y);

// windowPartitionForward(X,N,C,H,W,window,Y)
void bro_tensor_windowPartitionForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t window, void* Y);
// windowReverseForward(X,N,C,H,W,window,Y)
void bro_tensor_windowReverseForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t window, void* Y);
// spatialMerge2x2Forward(X,N,C,H,W,Y,channelMajor?)  — channelMajor defaults to false
void bro_tensor_spatialMerge2x2Forward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y, bool channelMajor);

// batchNormForward(X,gamma,beta,runningMean,runningVar,N,C,H,W,eps,momentum,Y,savedMean,savedRstd)
void bro_tensor_batchNormForward(void* X, void* gamma, void* beta, void* runningMean, void* runningVar, int32_t N, int32_t C, int32_t H, int32_t W, double eps, double momentum, void* Y, void* savedMean, void* savedRstd);
// batchNormBackward(X,gamma,savedMean,savedRstd,dY,N,C,H,W,dX,dGamma,dBeta)
void bro_tensor_batchNormBackward(void* X, void* gamma, void* savedMean, void* savedRstd, void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX, void* dGamma, void* dBeta);
// batchNormInference(X,gamma,beta,runningMean,runningVar,N,C,H,W,eps,Y)
void bro_tensor_batchNormInference(void* X, void* gamma, void* beta, void* runningMean, void* runningVar, int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y);

// imageNormalize(X,mean,std,N,C,H,W,Y)
void bro_tensor_imageNormalize(void* X, void* mean, void* std_, int32_t N, int32_t C, int32_t H, int32_t W, void* Y);
// imageU8ToF32NhwcToNchw(srcUint8,N,H,W,C,scale,bias,Y) — src is a host Uint8Array
void bro_tensor_imageU8ToF32NhwcToNchw(const uint8_t* src, uint32_t src_len, int32_t N, int32_t H, int32_t W, int32_t C, double scale, double bias, void* Y);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_conv(std::string* error); }
#endif
