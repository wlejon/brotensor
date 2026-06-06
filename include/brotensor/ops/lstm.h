#pragma once

// brotensor ops/lstm.h — Trainable LSTM cell over a sequence (forward + full BPTT).
//
// A single-layer, single-direction LSTM run over a length-T sequence (batch B),
// using the PyTorch `nn.LSTM` weight layout and gate order [input, forget,
// cell, output]. The forward is a *training* forward: besides the hidden-state
// sequence it caches the per-step gate activations and cell states that
// lstm_backward needs, so back-propagation-through-time (BPTT) is exact.
//
// This is the recurrent building block brotensor was missing — every primitive
// it composes (the gate GEMMs, sigmoid/tanh and their derivatives, the
// elementwise gate products) already had a backward; this packages the
// time-unrolled chain. Bidirectional / multi-layer stacks are built by the
// caller wrapping this cell (reverse the sequence for the backward direction;
// feed one layer's Y as the next layer's X).
//
// CPU FP32. The cell math, per timestep t and batch b (h_prev, c_prev carried):
//   z = W_ih·x + b_ih + W_hh·h_prev + b_hh           (4H, split i|f|g|o)
//   i = σ(z_i)  f = σ(z_f)  g = tanh(z_g)  o = σ(z_o)
//   c = f⊙c_prev + i⊙g
//   h = o⊙tanh(c)

#include "../tensor.h"

namespace brotensor {

// Training forward. I (input size) and H (hidden size) are inferred from the
// weight shapes; T and B are passed explicitly since X folds them into rows.
//
// Shapes (row-major; row (t*B + b) is the (t,b) slot):
//   X      : (T*B, I)    inputs
//   W_ih   : (4H, I)     input-hidden weights, gate-row blocks i|f|g|o
//   W_hh   : (4H, H)     hidden-hidden weights, same gate-row blocks
//   b_ih   : (4H, 1)     input-hidden bias,  or null/empty -> 0
//   b_hh   : (4H, 1)     hidden-hidden bias, or null/empty -> 0
//   h0,c0  : (B, H)      initial states,     or null/empty -> 0
//   Y      : (T*B, H)    OUT hidden states h_{t,b}            (resized)
//   gates  : (T*B, 4H)   OUT cache: post-activation [i|f|g|o] (resized)
//   C      : (T*B, H)    OUT cache: cell states c_{t,b}       (resized)
//   hT,cT  : (B, H)      OUT final states, or null -> not written
//
// gates and C are the backward cache — pass the same tensors back to
// lstm_backward unmodified.
void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                        const Tensor* b_ih, const Tensor* b_hh,
                        const Tensor* h0, const Tensor* c0,
                        int T, int B,
                        Tensor& Y, Tensor& gates, Tensor& C,
                        Tensor* hT, Tensor* cT);

// Backward (full BPTT). Consumes the forward cache (Y, gates, C) and the same
// inputs/weights/initial-states.
//   dY      : (T*B, H)   upstream grad on the output hidden states. For a
//                        last-step-only objective, zero all rows but the final.
// Parameter gradients ACCUMULATE — the caller pre-sizes and zeros them, exactly
// like linear_backward / matmul_backward:
//   dW_ih:(4H,I)  dW_hh:(4H,H)  db_ih:(4H,1)  db_hh:(4H,1)
// (db_ih and db_hh receive the same gradient, mirroring nn.LSTM's two biases.)
//   dX      : (T*B, I)   OVERWRITTEN (resized)
//   dh0,dc0 : (B, H)     OVERWRITTEN if non-null — grad w.r.t. the initial states
// db_ih, db_hh, dh0, dc0 may be null to skip; dX, dW_ih, dW_hh are required.
void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                   const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C,
                   const Tensor& dY, int T, int B,
                   Tensor& dX, Tensor& dW_ih, Tensor& dW_hh,
                   Tensor* db_ih, Tensor* db_hh,
                   Tensor* dh0, Tensor* dc0);

}  // namespace brotensor
