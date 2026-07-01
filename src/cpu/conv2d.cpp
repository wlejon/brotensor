// ─── CPU conv2d ops (CHUNK 3) ──────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/conv2d.cu — the plain
// direct-conv kernels (NOT the WMMA implicit-GEMM fast path), FP32 path only.
//
// Memory layout (matches the GPU exactly):
//   X   : NCHW              — ((n*C_in  + c_in)  * H     + h)     * W     + w
//   Y   : NCHW              — ((n*C_out + c_out) * H_out + h_out) * W_out + w_out
//   Wt  : OIHW (grouped)    — ((c_out*Cg_in + c_in_local) * kH + kh) * kW + kw
//         where Cg_in = C_in/groups is the per-group input-channel count.
//   bias: (C_out, 1), optional (may be null)
//
// Groups convention: output channel c_out belongs to group g = c_out/Cg_out;
// that group's absolute input channels start at g*Cg_in (Cg_in channels wide).
//
// Output-size formula (identical to the GPU):
//   H_out = (H + 2*pad_h - dil_h*(kH-1) - 1) / stride_h + 1
//   W_out = (W + 2*pad_w - dil_w*(kW-1) - 1) / stride_w + 1
//
// ACCUMULATION (matches the GPU kernels):
//   conv2d_forward         — Y  OVERWRITTEN (kernel stores acc directly).
//   conv2d_backward_input  — dX OVERWRITTEN.
//   conv2d_backward_weight — dWt ACCUMULATES (+=); GPU folds an FP32 scratch
//                            into the caller's dWt. Caller zeros dWt first.
//   conv2d_backward_bias   — dB ACCUMULATES (+=); GPU folds an FP32 scratch
//                            into the caller's dB. Caller zeros dB first.

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

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: bias dtype must match X");
    }
    check_groups("conv2d_forward", C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int W_out = out_dim(W, pad_w, dil_w, kW, stride_w);
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    // Interior region: the output rows/cols for which every kernel tap is
    // guaranteed in-bounds (the whole kH*kW*dil window sits inside the padded
    // input), computed once — independent of n/oc. In-bounds is monotonic in
    // (oh, ow), so the intersection across all kh (resp. kw) is itself a
    // single contiguous range. Only the thin border ring outside that range
    // needs the per-tap bounds check; the interior runs a branch-free loop.
    int oh_lo = (pad_h + stride_h - 1) / stride_h;              // ceil(pad_h/stride_h)
    int oh_hi = H - 1 + pad_h - (kH - 1) * dil_h;
    oh_hi = (oh_hi >= 0) ? (oh_hi / stride_h) : -1;             // floor(.../stride_h)
    if (oh_lo < 0) oh_lo = 0;
    if (oh_hi >= H_out) oh_hi = H_out - 1;

    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = W - 1 + pad_w - (kW - 1) * dil_w;
    ow_hi = (ow_hi >= 0) ? (ow_hi / stride_w) : -1;
    if (ow_lo < 0) ow_lo = 0;
    if (ow_hi >= W_out) ow_hi = W_out - 1;

    const bool has_interior = (oh_lo <= oh_hi) && (ow_lo <= ow_hi);

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            const int g = oc / Cg_out;
            const int ic_base = g * Cg_in;
            const int w_oc_base = oc * Cg_in * kH * kW;
            const float bias_v = Bp ? Bp[oc] : 0.0f;

            // Border pixel: same bounds-checked accumulation as before.
            auto compute_bordered = [&](int oh, int ow) -> float {
                const int in_h_origin = oh * stride_h - pad_h;
                const int in_w_origin = ow * stride_w - pad_w;
                float acc = 0.0f;
                for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                    const int ic = ic_base + ic_local;
                    const int w_ic_base = w_oc_base + ic_local * kH * kW;
                    const int x_ic_base = (n * C_in + ic) * H * W;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int in_h = in_h_origin + kh * dil_h;
                        if (in_h < 0 || in_h >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int in_w = in_w_origin + kw * dil_w;
                            if (in_w < 0 || in_w >= W) continue;
                            acc += Xp[x_ic_base + in_h * W + in_w] *
                                   Wp[w_ic_base + kh * kW + kw];
                        }
                    }
                }
                return acc;
            };

            // Interior pixel: every tap is guaranteed in-bounds — no checks.
            auto compute_interior = [&](int oh, int ow) -> float {
                const int in_h_origin = oh * stride_h - pad_h;
                const int in_w_origin = ow * stride_w - pad_w;
                float acc = 0.0f;
                for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                    const int ic = ic_base + ic_local;
                    const int w_ic_base = w_oc_base + ic_local * kH * kW;
                    const int x_ic_base = (n * C_in + ic) * H * W;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int in_h = in_h_origin + kh * dil_h;
                        const int x_row_base = x_ic_base + in_h * W;
                        const int w_row_base = w_ic_base + kh * kW;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int in_w = in_w_origin + kw * dil_w;
                            acc += Xp[x_row_base + in_w] * Wp[w_row_base + kw];
                        }
                    }
                }
                return acc;
            };

            for (int oh = 0; oh < H_out; ++oh) {
                const int y_row_base = ((n * C_out + oc) * H_out + oh) * W_out;
                const bool oh_interior = has_interior && oh >= oh_lo && oh <= oh_hi;
                if (!oh_interior) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        Yp[y_row_base + ow] = compute_bordered(oh, ow) + bias_v;
                    }
                    continue;
                }
                for (int ow = 0; ow < ow_lo; ++ow) {
                    Yp[y_row_base + ow] = compute_bordered(oh, ow) + bias_v;
                }
                for (int ow = ow_lo; ow <= ow_hi; ++ow) {
                    Yp[y_row_base + ow] = compute_interior(oh, ow) + bias_v;
                }
                for (int ow = ow_hi + 1; ow < W_out; ++ow) {
                    Yp[y_row_base + ow] = compute_bordered(oh, ow) + bias_v;
                }
            }
        }
    }
}

void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           ::brotensor::Tensor& dX) {
    if (dY.dtype != Wt.dtype) {
        throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    }
    check_groups("conv2d_backward_input", C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int W_out = out_dim(W, pad_w, dil_w, kW, stride_w);
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    const float* Wp  = Wt.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Gather form: one accumulation per input pixel, inverting the forward
    // index relation in_h = stride_h*i_out - pad_h + dil_h*kh.
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int c_in_local = c_in - g * Cg_in;
            const int oc_lo = g * Cg_out;
            const int oc_hi = oc_lo + Cg_out;
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int num_h = i + pad_h - dil_h * kh;
                        if (num_h < 0 || num_h % stride_h != 0) continue;
                        const int i_out = num_h / stride_h;
                        if (i_out < 0 || i_out >= H_out) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int num_w = j + pad_w - dil_w * kw;
                            if (num_w < 0 || num_w % stride_w != 0) continue;
                            const int j_out = num_w / stride_w;
                            if (j_out < 0 || j_out >= W_out) continue;
                            for (int c_out = oc_lo; c_out < oc_hi; ++c_out) {
                                const int dy_idx =
                                    ((n * C_out + c_out) * H_out + i_out) *
                                        W_out + j_out;
                                const int w_idx =
                                    ((c_out * Cg_in + c_in_local) * kH + kh) *
                                        kW + kw;
                                acc += dYp[dy_idx] * Wp[w_idx];
                            }
                        }
                    }
                    const int dx_idx = ((n * C_in + c_in) * H + i) * W + j;
                    dXp[dx_idx] = acc;   // overwrite
                }
            }
        }
    }
}

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt) {
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error(
            "conv2d_backward_weight: X, dY, dWt dtype must match");
    }
    check_groups("conv2d_backward_weight", C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int W_out = out_dim(W, pad_w, dil_w, kW, stride_w);
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    const int total = C_out * Cg_in * kH * kW;
    if (total == 0) return;

    const float* Xp  = X.host_f32();
    const float* dYp = dY.host_f32();
    float* dWp = dWt.host_f32_mut();

    // One accumulation per weight element; accumulate (+=) into dWt to match
    // the GPU's FP32-scratch-fold-into-dWt contract.
    for (int c_out = 0; c_out < C_out; ++c_out) {
        const int g = c_out / Cg_out;
        for (int c_in_local = 0; c_in_local < Cg_in; ++c_in_local) {
            const int c_in = g * Cg_in + c_in_local;
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    float acc = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        for (int i_out = 0; i_out < H_out; ++i_out) {
                            const int in_h =
                                i_out * stride_h - pad_h + kh * dil_h;
                            if (in_h < 0 || in_h >= H) continue;
                            for (int j_out = 0; j_out < W_out; ++j_out) {
                                const int in_w =
                                    j_out * stride_w - pad_w + kw * dil_w;
                                if (in_w < 0 || in_w >= W) continue;
                                const int x_idx =
                                    ((n * C_in + c_in) * H + in_h) * W + in_w;
                                const int dy_idx =
                                    ((n * C_out + c_out) * H_out + i_out) *
                                        W_out + j_out;
                                acc += dYp[dy_idx] * Xp[x_idx];
                            }
                        }
                    }
                    const int w_idx =
                        ((c_out * Cg_in + c_in_local) * kH + kh) * kW + kw;
                    dWp[w_idx] += acc;   // accumulate
                }
            }
        }
    }
}

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB) {
    if (dB.dtype != dY.dtype) {
        throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    const float* dYp = dY.host_f32();
    float* dBp = dB.host_f32_mut();

    const int spatial = H_out * W_out;
    // Per-channel sum over (N, H_out, W_out); accumulate (+=) into dB to match
    // the GPU's FP32-scratch-fold-into-dB contract.
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const int base = (n * C_out + c_out) * spatial;
            for (int sp = 0; sp < spatial; ++sp) {
                acc += dYp[base + sp];
            }
        }
        dBp[c_out] += acc;   // accumulate
    }
}

} // namespace brotensor::detail::cpu
