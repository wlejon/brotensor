#pragma once

// native_tensor_misc_decl.h — the "misc" group of the restored QuickJS
// `bro.tensor.*` free-function surface: the gated-deltanet L2 norm pair, three
// loss entry points that the fused-loss family in native_tensor_nn.cpp left
// behind, and the diffusion sampler steps + sinusoidal timestep embedding.
//
// Ported from the old bindings in
//   src/js/tensor_bindings_activations.cpp  (l2Norm*, bceWithLogitsFusedBatched,
//                                             softmaxXentSegment, mseScalar)
//   src/js/tensor_bindings_diffusion.cpp    (ddimStep, eulerStep, dpmpp2mStep,
//                                             timestepEmbedding)
// keeping the old argument order, optional arguments and defaults. The old
// `nngpu::` ops are today's `brotensor::` ops in ops/norm.h, ops/loss.h and
// ops/diffusion.h.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- L2 norm (gated-deltanet per-head, last dim of an (L, H*D) tensor) ------

// l2NormForward(X, headDim, numHeads, eps, Y)
void bro_tensor_l2NormForward(void* X, int32_t headDim, int32_t numHeads, double eps, void* Y);
// l2NormBackward(X, headDim, numHeads, eps, dY, dX)
void bro_tensor_l2NormBackward(void* X, int32_t headDim, int32_t numHeads, double eps, void* dY, void* dX);

// --- losses ----------------------------------------------------------------

// bceWithLogitsFusedBatched(logits, target, mask|null, posWeight, probs, dLogits, lossPerSample)
// `mask` is an optional FP32 GpuTensor over the (B,L) grid; it crosses as the
// raw value and resolves to the ops' `const float* d_mask_BL`.
void bro_tensor_bceWithLogitsFusedBatched(void* logits_BL, void* target_BL, uint64_t mask_bits,
                                          double posWeight, void* probs_BL, void* dLogits_BL,
                                          void* lossPerSample);

// softmaxXentSegment(logits, target, probs, dLogits, n, mask|null) -> loss
// Host Float32Array buffers (the op takes raw host pointers and is CPU-only);
// probs and dLogits are written in place. An empty mask array means "no mask".
double bro_tensor_softmaxXentSegment(const float* logits, uint32_t logits_len,
                                     const float* target, uint32_t target_len,
                                     float* probs, uint32_t probs_len,
                                     float* dLogits, uint32_t dLogits_len,
                                     int32_t n,
                                     const float* mask, uint32_t mask_len);

// mseScalar(pred, target) -> [loss, dPred]
// The pair comes back as a length-2 f64[] the JS wrapper unpacks.
void bro_tensor_mseScalar(double pred, double target, bronze_native_buffer* out);

// --- diffusion sampler steps + timestep embedding --------------------------

// ddimStep(x_t, eps_pred, alpha_t, alpha_prev, sigma_t, x_prev)
void bro_tensor_ddimStep(void* x_t, void* eps_pred, double alphaT, double alphaPrev,
                         double sigmaT, void* x_prev);
// eulerStep(x_t, eps_pred, sigma_t, sigma_prev, x_prev)
void bro_tensor_eulerStep(void* x_t, void* eps_pred, double sigmaT, double sigmaPrev, void* x_prev);
// dpmpp2mStep(x_t, eps_pred, x0_prev, sigma_t, c_xt, c_x0t, c_x0prev, x_prev, x0_out)
void bro_tensor_dpmpp2mStep(void* x_t, void* eps_pred, void* x0_prev, double sigmaT,
                            double c_xt, double c_x0t, double c_x0prev,
                            void* x_prev, void* x0_out);
// timestepEmbedding(timesteps, dim, maxPeriod, Y) — maxPeriod defaults to 10000
void bro_tensor_timestepEmbedding(void* timesteps, int32_t dim, double maxPeriod, void* Y);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_misc(std::string* error); }
#endif
