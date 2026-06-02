#pragma once

// Shared FP32-domain activation scalars for CUDA kernels that apply an
// activation in-register — notably the GEMM epilogue (linear forward fuses
// bias + activation into the matmul's output write, killing the separate
// bias-add and activation launches and their HBM round-trips).
//
// The formulas mirror the standalone activation kernels in elementwise.cu
// (same __expf / tanhf / erff intrinsics) so a fused result matches the
// unfused linear→activation sequence within FP16 tolerance.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace brotensor::detail::cuda {

// Activation selector for the linear/GEMM epilogue. Kept as plain int values
// (mirrored by brotensor::LinearActivation in <brotensor/ops/linear.h>) so the
// vtable signature stays int-typed, matching the `int mode` convention used by
// interp2d / pad2d.
//   0 none · 1 relu · 2 gelu(tanh) · 3 gelu(exact) · 4 silu · 5 quick_gelu

__device__ inline float act_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ inline float act_gelu_tanh(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

__device__ inline float act_gelu_exact(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

__device__ inline float act_quick_gelu(float v) {
    return v / (1.0f + __expf(-1.702f * v));
}

// Apply the selected epilogue activation. `act == 0` is identity, so callers
// can pass it unconditionally.
__device__ inline float apply_linear_act(int act, float v) {
    switch (act) {
        case 1:  return v > 0.0f ? v : 0.0f;
        case 2:  return act_gelu_tanh(v);
        case 3:  return act_gelu_exact(v);
        case 4:  return act_silu(v);
        case 5:  return act_quick_gelu(v);
        default: return v;
    }
}

} // namespace brotensor::detail::cuda
