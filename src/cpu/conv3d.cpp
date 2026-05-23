// ─── CPU conv3d ops (Qwen3-VL patch embed) ────────────────────────────────
//
// FP32 scalar host implementation of conv3d_forward. Mirrors src/cpu/conv2d.cpp
// — direct convolution loop, FP32 accumulator. CPU is FP32-only per the
// brotensor convention (CLAUDE.md); conv3d_int8w_fp16_forward is GPU-only and
// has no CPU registration (its vtable slot stays null and the dispatcher
// throws "not implemented on CPU").
//
// Memory layout (matches the GPU exactly):
//   X   : NCTHW              — (((n*C_in + c_in) * T + t) * H + h) * W + w
//   Y   : NCTHW              — (((n*C_out + c_out) * T_out + ot) * H_out + oh) * W_out + ow
//   Wt  : OICTHW (grouped)   — (((c_out*Cg_in + c_in_local) * kT + kt) * kH + kh) * kW + kw
//         where Cg_in = C_in/groups is the per-group input-channel count.
//   bias: (C_out, 1), optional (may be null)
//
// Groups convention: output channel c_out belongs to group g = c_out/Cg_out;
// that group's absolute input channels start at g*Cg_in (Cg_in channels wide).
//
// Output-size formula (identical to conv2d, applied per-axis):
//   T_out = (T + 2*pad_t - dil_t*(kT-1) - 1) / stride_t + 1
//   H_out = (H + 2*pad_h - dil_h*(kH-1) - 1) / stride_h + 1
//   W_out = (W + 2*pad_w - dil_w*(kW-1) - 1) / stride_w + 1
//
// ACCUMULATION: conv3d_forward — Y OVERWRITTEN (kernel stores acc directly).

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            std::string(op) +
            ": groups must be >=1 and divide both C_in and C_out");
    }
}

inline int out_dim(int in, int pad, int dil, int k, int stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

} // namespace

void conv3d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: bias dtype must match X");
    }
    check_groups("conv3d_forward", C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int T_out = out_dim(T, pad_t, dil_t, kT, stride_t);
    const int H_out = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int W_out = out_dim(W, pad_w, dil_w, kW, stride_w);
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            const int g = oc / Cg_out;
            const int ic_base = g * Cg_in;
            const int w_oc_base = oc * Cg_in * kT * kH * kW;
            const float bias_v = Bp ? Bp[oc] : 0.0f;
            for (int ot = 0; ot < T_out; ++ot) {
                const int in_t_origin = ot * stride_t - pad_t;
                for (int oh = 0; oh < H_out; ++oh) {
                    const int in_h_origin = oh * stride_h - pad_h;
                    for (int ow = 0; ow < W_out; ++ow) {
                        const int in_w_origin = ow * stride_w - pad_w;
                        float acc = 0.0f;
                        for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                            const int ic = ic_base + ic_local;
                            const int w_ic_base =
                                w_oc_base + ic_local * kT * kH * kW;
                            const int x_ic_base =
                                (n * C_in + ic) * T * H * W;
                            for (int kt = 0; kt < kT; ++kt) {
                                const int in_t = in_t_origin + kt * dil_t;
                                if (in_t < 0 || in_t >= T) continue;
                                for (int kh = 0; kh < kH; ++kh) {
                                    const int in_h = in_h_origin + kh * dil_h;
                                    if (in_h < 0 || in_h >= H) continue;
                                    for (int kw = 0; kw < kW; ++kw) {
                                        const int in_w = in_w_origin + kw * dil_w;
                                        if (in_w < 0 || in_w >= W) continue;
                                        acc +=
                                            Xp[x_ic_base +
                                               (in_t * H + in_h) * W + in_w] *
                                            Wp[w_ic_base +
                                               (kt * kH + kh) * kW + kw];
                                    }
                                }
                            }
                        }
                        acc += bias_v;
                        const int y_idx =
                            (((n * C_out + oc) * T_out + ot) * H_out + oh) *
                                W_out + ow;
                        Yp[y_idx] = acc;   // overwrite
                    }
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
