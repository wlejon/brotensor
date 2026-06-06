#pragma once

// brotensor ops/lora.h — Low-rank (LoRA) adapter delta on a frozen base
// linear, optionally gated on the bottleneck.
//
// Header-only: pure orchestration of the device-neutral linear + elementwise
// free functions, so it runs on whichever device the input tensors live on.
// There is no new vtable op and no backend code — a LoRA "layer" is just a
// composition of linear_forward / linear_backward with an elementwise gate.
//
// For a frozen base weight W:(out,in) and bias b:(out,1), input x:(in,1),
// down-projection A:(r,in), up-projection B:(out,r), scalar `scale`, and an
// optional per-rank gate g:(r,1):
//
//     h   = A x                  (bottleneck, r-dim)
//     hg  = g (.) h              (h unchanged if g is null)
//     y   = W x + b + scale * (B hg)
//
// The gate is how conditioning rides on top of a single shared adapter: with
// g = g(condition) and g(0) = 0 the delta vanishes and y reproduces the base
// model exactly. The base weight/bias never receive a gradient (frozen); the
// trainable parameters are A, B and whatever produces the gate.

#include "../tensor.h"
#include "linear.h"
#include "elementwise.h"

namespace brotensor {

// LoRA forward. `y` is resized to (out,1). `h_out` / `hg_out` are the
// bottleneck activations cached for lora_backward — pass them straight back
// in. `g` may be null (plain ungated LoRA), else shape (r,1).
inline void lora_forward(const Tensor& W, const Tensor& b, const Tensor& x,
                         const Tensor& A, const Tensor& B, float scale,
                         const Tensor* g,
                         Tensor& y, Tensor& h_out, Tensor& hg_out) {
    const Device dev = x.device;
    // base: y = W x + b
    linear_forward(W, b, x, y);
    // bottleneck h = A x  (no bias)
    Tensor zr = Tensor::zeros_on(dev, A.rows, 1);
    linear_forward(A, zr, x, h_out);
    // gate: hg = g (.) h
    hg_out = h_out.clone();
    if (g) mul_inplace(hg_out, *g);
    // delta = scale * (B hg);  y += delta
    Tensor zo = Tensor::zeros_on(dev, B.rows, 1);
    Tensor delta;
    linear_forward(B, zo, hg_out, delta);
    scale_inplace(delta, scale);
    add_inplace(y, delta);
}

// LoRA backward. `dY`:(out,1) upstream. Base W,b are frozen (no grad).
//   dA:(r,in), dB:(out,r) are ACCUMULATED — caller zeros before the first call.
//   dG:(r,1) accumulated if non-null AND `g` was non-null in the forward.
//   dX:(in,1) overwritten if non-null (base + lora input grad). Pass null when
//   the input is frozen (e.g. the AdaIN style vector) to skip the base
//   backward entirely.
// `h` / `hg` are the cached bottleneck activations from lora_forward.
inline void lora_backward(const Tensor& W, const Tensor& x,
                          const Tensor& A, const Tensor& B, float scale,
                          const Tensor* g,
                          const Tensor& h, const Tensor& hg,
                          const Tensor& dY,
                          Tensor& dA, Tensor& dB,
                          Tensor* dG, Tensor* dX) {
    const Device dev = x.device;
    const int out = B.rows, r = A.rows, in = A.cols;
    // dp = scale * dY   (grad into p = B hg, since y = base + scale*p)
    Tensor dp = dY.clone();
    scale_inplace(dp, scale);
    // through B:  dB += dp hg^T ;  dHg = B^T dp
    Tensor dHg, dBbias = Tensor::zeros_on(dev, out, 1);
    linear_backward(B, hg, dp, dHg, dB, dBbias);
    // gate split:  dH = dHg (.) g ;  dG += dHg (.) h   (hg = g (.) h)
    Tensor dH;
    if (g) {
        dH = dHg.clone();
        mul_inplace(dH, *g);
        if (dG) {
            Tensor t = dHg.clone();
            mul_inplace(t, h);
            add_inplace(*dG, t);
        }
    } else {
        dH = dHg;
    }
    // through A:  dA += dH x^T ;  dX_lora = A^T dH
    Tensor dXl, dAbias = Tensor::zeros_on(dev, r, 1);
    linear_backward(A, x, dH, dXl, dA, dAbias);
    if (dX) {
        // base contributes W^T dY to dX; W is frozen so dW/dB are discarded.
        Tensor dWbase = Tensor::zeros_on(dev, out, in);
        Tensor dBbase = Tensor::zeros_on(dev, out, 1);
        linear_backward(W, x, dY, *dX, dWbase, dBbase);
        add_inplace(*dX, dXl);
    }
}

}  // namespace brotensor
