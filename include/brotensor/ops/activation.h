#pragma once

// brotensor ops/activation.h — Pointwise activations + GLU gates (relu/tanh/sigmoid/silu/gelu/snake/elu/geglu/swiglu).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// y = max(x, 0). Shapes match; y resized if mis-shaped. x and y may alias.
void relu_forward(const Tensor& x, Tensor& y);


// dX = dY * (x > 0). dX resized to match x; may alias dY.
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);


// y = tanh(x). y resized to match x.
void tanh_forward(const Tensor& x, Tensor& y);


// dX = dY * (1 - y*y). `y` is the cached forward output (not raw x).
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX);


// y = 1 / (1 + exp(-x)).
void sigmoid_forward(const Tensor& x, Tensor& y);


// dX = dY * y * (1 - y). `y` is the cached forward output.
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX);


// Elementwise ReLU / Tanh over (B,D). Y resized to match X; X and Y may alias.
void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD);

void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD);


// Elementwise activation backward over (B,D), same shapes throughout.
//   relu: dX = dY*(X>0), reads X_BD (forward input).
//   tanh: dX = dY*(1-Y*Y), reads Y_BD (forward output).
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);

void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);


// SiLU / Swish: y = x*sigmoid(x). Dispatched FP32/FP16 on x.dtype; y resized +
// dtype-set to match x. x and y may alias.
void silu_forward(const Tensor& x, Tensor& y);


// SiLU backward, reads the raw forward input x:
//   dX = dY * sigmoid(x) * (1 + x*(1-sigmoid(x))).
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);


// GELU, tanh approximation (PyTorch approximate="tanh"):
//   y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// Dispatched FP32/FP16 on x.dtype.
void gelu_forward(const Tensor& x, Tensor& y);


// GELU (tanh-approx) backward, reads x. With k=sqrt(2/pi),
// u=k*(x+0.044715*x^3), t=tanh(u):
//   dX = dY * [0.5*(1+t) + 0.5*x*(1-t^2)*k*(1+3*0.044715*x^2)].
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);


// Exact GELU (erf form, PyTorch approximate="none", diffusers default):
//   y = 0.5*x*(1 + erf(x/sqrt(2))).
// Distinct from the tanh-approx gelu_forward. Dispatched FP32/FP16 on x.dtype;
// y resized + dtype-set to match x. x and y may alias.
void gelu_exact_forward(const Tensor& x, Tensor& y);


// Exact-GELU backward, reads x:
//   dX = dY * [0.5*(1+erf(x/sqrt(2))) + x*phi(x)],  phi = standard normal pdf.
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);


// QuickGELU: y = x*sigmoid(1.702*x). OpenAI CLIP's activation (SD1.5 text
// encoder). Dispatched FP32/FP16 on x.dtype; y resized + dtype-set to match x.
// x and y may alias.
void quick_gelu_forward(const Tensor& x, Tensor& y);


// QuickGELU backward, reads x. With s = sigmoid(1.702*x):
//   dX = dY * (s + x*1.702*s*(1-s)).
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);


// GEGLU: input (B,2*D) split along the last dim into A=(B,D) and B_half=(B,D);
// output (B,D) = A * gelu(B_half) (tanh-approx). Dispatched FP32/FP16 on
// X.dtype; Y resized + dtype-set to match X.
void geglu_forward(const Tensor& X, Tensor& Y);


// GEGLU backward. With g = gelu(B_half) (tanh-approx):
//   dA = dY*g;   dB_half = dY*A*gelu'(B_half).
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched
// FP32/FP16 on X.dtype; dX resized + dtype-set to match X.
void geglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);


// Exact-GELU GEGLU: same split as geglu_forward but output = A*gelu_exact(B_half),
// using the exact erf-based GELU. Matches diffusers' default GEGLU. Dispatched
// FP32/FP16 on X.dtype; Y resized + dtype-set to match X.
void geglu_exact_forward(const Tensor& X, Tensor& Y);


// Exact-GELU GEGLU backward. With g = gelu_exact(B_half):
//   dA = dY*g;   dB_half = dY*A*gelu_exact'(B_half).
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched
// FP32/FP16 on X.dtype; dX resized + dtype-set to match X.
void geglu_exact_backward(const Tensor& X, const Tensor& dY,
                          Tensor& dX);


// SwiGLU (Llama FFN gate): input (B,2*D) split along the last dim into A=(B,D)
// and B_half=(B,D); output (B,D) = silu(A) * B_half. Dispatched FP32/FP16 on
// X.dtype; Y resized + dtype-set to match X.
void swiglu_forward(const Tensor& X, Tensor& Y);


// SwiGLU backward. With s = silu(A):
//   dA = dY*B_half*silu'(A);   dB_half = dY*s.
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched on
// X.dtype; dX resized + dtype-set to match X.
void swiglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);


// ─── Vocoder / codec activations (audio) ───────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal). NCL
// layout — the (N,C,L) dims are passed as int args; element (n,c,l) is at flat
// index (n*C+c)*L + l.

// Snake activation (BigVGAN / DAC vocoder), per-channel learnable alpha (and
// optional beta):
//   plain snake (beta == null): y = x + (1/alpha_c)*sin^2(alpha_c*x)
//   snakebeta   (beta != null): y = x + (1/beta_c) *sin^2(alpha_c*x)
// alpha/beta are per-channel (broadcast across the (n,l) plane). The reciprocal
// denominator is sign-preserved-floored at magnitude 1e-9 to avoid NaN/Inf.
//   X, Y: (N,C*L).  alpha: (C,1) or (1,C).  beta: (C,1)/(1,C) or null.
//   Y resized + dtype-set to match X; X and Y may alias. FP32, CPU-resident.
void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y);


// Snake backward, reads the raw forward input X. With s=sin(a*x), c=cos(a*x),
// a=alpha_c, denom=(beta?beta_c:a), r=1/denom (sign-guarded as in the forward):
//   dy/dx     = 1 + 2*a*r*s*c
//   dy/dalpha = 2*r*x*s*c          (plain snake also adds the -r^2*s^2 term,
//                                   since denom==alpha there)
//   dy/dbeta  = -r^2*s^2           (snakebeta only)
//   dX: (N,C*L) overwritten (resized + dtype-set to X).
//   dAlpha: (C,1) accumulated — caller zeros.
//   dBeta:  (C,1) accumulated — caller zeros; non-null exactly when beta is.
void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta);


// ELU (EnCodec activation), elementwise:
//   y = x                    if x > 0
//   y = alpha*(exp(x) - 1)   otherwise
// y resized to match x; x and y may alias. CPU FP32-only.
void elu_forward(const Tensor& x, float alpha, Tensor& y);

inline void elu_forward(const Tensor& x, Tensor& y) {
    elu_forward(x, /*alpha=*/1.0f, y);
}


// ELU backward, reads the raw forward input x:
//   dX = dY * (x > 0 ? 1 : alpha*exp(x)).
// dX overwritten (resized to match x); may alias dY. CPU FP32-only.
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX);

inline void elu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    elu_backward(x, dY, /*alpha=*/1.0f, dX);
}


// Leaky ReLU (HiFi-GAN activation), elementwise:
//   y = x > 0 ? x : negative_slope*x.
// y resized to match x; x and y may alias. CPU FP32-only.
void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y);


// Leaky ReLU backward, reads the raw forward input x:
//   dX = dY * (x > 0 ? 1 : negative_slope).
// dX overwritten (resized to match x); may alias dY. CPU FP32-only.
void leaky_relu_backward(const Tensor& x, const Tensor& dY,
                         float negative_slope, Tensor& dX);

}  // namespace brotensor
