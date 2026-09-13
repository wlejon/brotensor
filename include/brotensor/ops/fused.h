#pragma once

// ─── brotensor ops/fused.h — High-performance fused operators ────────────────
//
// First-class public API for fused kernel execution (in-register fused dequant,
// residual RMSNorm, LayerNorm modulate, and GEMV SwiGLU / residual).
// Automatically dispatches to high-performance CUDA JIT PTX kernels on sm_89,
// CPU JIT AVX2/FMA kernels on host, or transparent fallback to separate ops.

#include "../tensor.h"

namespace brotensor {

// In-place residual add (h += proj) and RMSNorm (out = rms_norm(h, gamma, eps))
// fused into a single register pass without intermediate DRAM round-trips.
void fused_residual_rmsnorm(Tensor& h, const Tensor& proj, const Tensor& gamma,
                            float eps, Tensor& out);

// In-place residual add (x += res) and LayerNorm (out = layernorm(x, gamma, beta, eps))
// fused into a single register pass without intermediate DRAM round-trips.
void fused_residual_layernorm(Tensor& x, const Tensor& res, const Tensor& gamma,
                             const Tensor& beta, float eps, Tensor& out);

// LayerNorm over x with gamma, beta, followed by AdaLN modulate with (1 + scale) + shift
// computed directly in registers without intermediate writes.
void fused_layernorm_modulate(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                              const Tensor& scale, const Tensor& shift, float eps,
                              Tensor& out);

// Single-token SwiGLU GEMV: computes silu(W_gate * x) * (W_up * x) directly in registers.
void fused_gemv_swiglu(const Tensor& x, const Tensor& w_gate, const Tensor& w_up,
                       Tensor& out);

// Single-token GEMV with residual addition: out = W_down * x + res directly in registers.
void fused_gemv_residual(const Tensor& x, const Tensor& w_down, const Tensor& res,
                         Tensor& out);

} // namespace brotensor
