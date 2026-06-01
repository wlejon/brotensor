#pragma once

// brotensor ops/optim.h — Optimizers + init: sgd_step, adam_step, xavier_init.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// SGD with momentum, in place:
//   velocity = momentum*velocity + grad;  param -= lr*velocity.
// All three tensors share shape; caller zeros grad between batches.
void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity,
              float lr, float momentum);


// Adam step, in place. `step` is the 1-based bias-correction counter.
//   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g^2
//   param -= lr * (m/(1-b1^step)) / (sqrt(v/(1-b2^step)) + eps)
// All four tensors share shape.
void adam_step(Tensor& param, const Tensor& grad,
               Tensor& m, Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);


// Deterministic xavier-uniform init of a Linear weight; rng_state is a
// splitmix64 state advanced in place. CPU-only (weights init on the host).
void xavier_init(Tensor& W, uint64_t& rng_state);

}  // namespace brotensor
