// The linear epilogue's activation (LACT_*) and the two GLU gates, shared by
// gemm_common.glsl's kernels and gemv_q.comp. Include after common.glsl.

#include "op_codes.h"

float gemm_act(const uint act, float v) {
    if (act == LACT_RELU) return max(v, 0.0);
    if (act == LACT_GELU_TANH) return bt_gelu_tanh(v);
    if (act == LACT_GELU_EXACT) return 0.5 * v * (1.0 + bt_erf(v * 0.70710678118654752440));
    if (act == LACT_SILU) return v / (1.0 + exp(-v));
    if (act == LACT_QUICK_GELU) return v / (1.0 + exp(-1.702 * v));
    return v;
}

float gelu_exact(float v) { return 0.5 * v * (1.0 + bt_erf(v * 0.70710678118654752440)); }
float silu(float v) { return v / (1.0 + exp(-v)); }
