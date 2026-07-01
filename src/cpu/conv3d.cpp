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
#include <brotensor/detail/cpu/thread_pool.h>

#include <cstddef>
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

    // Interior region: the (ot, oh, ow) box for which every kernel tap is
    // guaranteed in-bounds on every axis (mirrors conv2d.cpp's split,
    // extended to 3 axes), computed once — independent of n/oc. Only the
    // thin border shell outside that box needs the per-tap bounds check;
    // the interior runs a branch-free kt/kh/kw loop.
    int ot_lo = (pad_t + stride_t - 1) / stride_t;
    int ot_hi = T - 1 + pad_t - (kT - 1) * dil_t;
    ot_hi = (ot_hi >= 0) ? (ot_hi / stride_t) : -1;
    if (ot_lo < 0) ot_lo = 0;
    if (ot_hi >= T_out) ot_hi = T_out - 1;

    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = H - 1 + pad_h - (kH - 1) * dil_h;
    oh_hi = (oh_hi >= 0) ? (oh_hi / stride_h) : -1;
    if (oh_lo < 0) oh_lo = 0;
    if (oh_hi >= H_out) oh_hi = H_out - 1;

    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = W - 1 + pad_w - (kW - 1) * dil_w;
    ow_hi = (ow_hi >= 0) ? (ow_hi / stride_w) : -1;
    if (ow_lo < 0) ow_lo = 0;
    if (ow_hi >= W_out) ow_hi = W_out - 1;

    const bool has_interior =
        (ot_lo <= ot_hi) && (oh_lo <= oh_hi) && (ow_lo <= ow_hi);

    // Each n exclusively owns Y's batch slice n (X/Wt/bias are read-only), so
    // this parallelizes across n with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        for (int oc = 0; oc < C_out; ++oc) {
            const int g = oc / Cg_out;
            const int ic_base = g * Cg_in;
            const int w_oc_base = oc * Cg_in * kT * kH * kW;
            const float bias_v = Bp ? Bp[oc] : 0.0f;

            // Border voxel: same bounds-checked accumulation as before.
            auto compute_bordered = [&](int ot, int oh, int ow) -> float {
                const int in_t_origin = ot * stride_t - pad_t;
                const int in_h_origin = oh * stride_h - pad_h;
                const int in_w_origin = ow * stride_w - pad_w;
                float acc = 0.0f;
                for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                    const int ic = ic_base + ic_local;
                    const int w_ic_base = w_oc_base + ic_local * kT * kH * kW;
                    const int x_ic_base = (n * C_in + ic) * T * H * W;
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
                return acc;
            };

            // Interior voxel: every tap is guaranteed in-bounds — no checks.
            auto compute_interior = [&](int ot, int oh, int ow) -> float {
                const int in_t_origin = ot * stride_t - pad_t;
                const int in_h_origin = oh * stride_h - pad_h;
                const int in_w_origin = ow * stride_w - pad_w;
                float acc = 0.0f;
                for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                    const int ic = ic_base + ic_local;
                    const int w_ic_base = w_oc_base + ic_local * kT * kH * kW;
                    const int x_ic_base = (n * C_in + ic) * T * H * W;
                    for (int kt = 0; kt < kT; ++kt) {
                        const int in_t = in_t_origin + kt * dil_t;
                        const int x_t_base = x_ic_base + in_t * H * W;
                        const int w_t_base = w_ic_base + kt * kH * kW;
                        for (int kh = 0; kh < kH; ++kh) {
                            const int in_h = in_h_origin + kh * dil_h;
                            const int x_row_base = x_t_base + in_h * W;
                            const int w_row_base = w_t_base + kh * kW;
                            for (int kw = 0; kw < kW; ++kw) {
                                const int in_w = in_w_origin + kw * dil_w;
                                acc += Xp[x_row_base + in_w] * Wp[w_row_base + kw];
                            }
                        }
                    }
                }
                return acc;
            };

            for (int ot = 0; ot < T_out; ++ot) {
                const bool ot_interior = has_interior && ot >= ot_lo && ot <= ot_hi;
                const int t_base = ((n * C_out + oc) * T_out + ot) * H_out;
                if (!ot_interior) {
                    for (int oh = 0; oh < H_out; ++oh) {
                        const int y_row_base = (t_base + oh) * W_out;
                        for (int ow = 0; ow < W_out; ++ow) {
                            Yp[y_row_base + ow] = compute_bordered(ot, oh, ow) + bias_v;
                        }
                    }
                    continue;
                }
                for (int oh = 0; oh < H_out; ++oh) {
                    const int y_row_base = (t_base + oh) * W_out;
                    const bool oh_interior = oh >= oh_lo && oh <= oh_hi;
                    if (!oh_interior) {
                        for (int ow = 0; ow < W_out; ++ow) {
                            Yp[y_row_base + ow] = compute_bordered(ot, oh, ow) + bias_v;
                        }
                        continue;
                    }
                    for (int ow = 0; ow < ow_lo; ++ow) {
                        Yp[y_row_base + ow] = compute_bordered(ot, oh, ow) + bias_v;
                    }
                    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
                        Yp[y_row_base + ow] = compute_interior(ot, oh, ow) + bias_v;
                    }
                    for (int ow = ow_hi + 1; ow < W_out; ++ow) {
                        Yp[y_row_base + ow] = compute_bordered(ot, oh, ow) + bias_v;
                    }
                }
            }
        }
    });
}

} // namespace brotensor::detail::cpu
