#pragma once

// ─── Brass JIT CPU acceleration interface for brotensor ──────────────────────
//
// Provides high-performance AVX2/FMA compiled numerical kernels with zero GC
// overhead and multi-threaded parallel execution across rows and token chunks,
// resolving the N=1 single-thread bottleneck for LLM decode and DiT inference.

namespace brotensor::detail::cpu::jit {

// Checks whether Brass JIT compiler and runtime are initialized and available.
bool is_jit_available();

// RMSNorm forward: Y = X * gamma / sqrt(mean(X^2) + eps)
void rms_norm_forward(const float* X, const float* gamma, float eps, float* Y, int B, int D);

// SwiGLU forward: Y = silu(X[:, :D]) * X[:, D:]
void swiglu_forward(const float* X, float* Y, int B, int D);

// AdaLN modulate: Y = X * (1 + scale) + shift
void modulate(const float* X, const float* scale, const float* shift, float* Y, int L, int D);

// Broadcast vector multiply: Y = X * v
void broadcast_mul(const float* X, const float* v, float* Y, int L, int D);

// Batched LayerNorm forward inference: Y = gamma * (X - mean) / sqrt(var + eps) + beta
void layernorm_forward_inference_batched(const float* X, const float* gamma, const float* beta, float eps, float* Y, int R, int D);

// Fused Residual RMSNorm: X += res in-place, accumulates sum_sq in same pass, Y = X * gamma * rrms
void fused_residual_rmsnorm(float* X, const float* res, const float* gamma, float eps, float* Y, int B, int D);

// Fused Residual LayerNorm: X += res in-place, cooperatively computes mean/var, Y = gamma * (X - mean) * rstd + beta
void fused_residual_layernorm(float* X, const float* res, const float* gamma, const float* beta, float eps, float* Y, int B, int D);

// Fused LayerNorm + AdaLN Modulate: computes LayerNorm directly in registers and applies modulate into Y (0 intermediate writes)
void fused_layernorm_modulate(const float* X, const float* gamma, const float* beta, const float* scale, const float* shift, float eps, float* Y, int R, int D);

// Quantized GEMV Q8_0: Y = W_q8_0 @ X
void gemv_q8_0(const void* W, const float* X, float* Y, int N, int K);

// Quantized GEMV Q4_K: Y = W_q4_k @ X
void gemv_q4_k(const void* W, const float* X, float* Y, int N, int K);

} // namespace brotensor::detail::cpu::jit
