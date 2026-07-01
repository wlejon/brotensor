// ─── CPU 1D-convolution family (brosoundml CHUNK 3) ────────────────────────
//
// FP32 scalar host implementations of the genuinely-new 1D-conv ops:
//   conv_transpose1d_forward / _backward_input / _backward_weight / _backward_bias
//   causal_conv1d_update
//   pad1d_forward / pad1d_backward
//
// Plain conv1d, its three backward halves, conv1d_int8w_fp16, and causal_conv1d
// are header-only inline wrappers in <brotensor/ops.h> that forward to the
// conv2d ops — they do not appear here.
//
// ── Layout (NCL) ────────────────────────────────────────────────────────────
//   X / Y : NCL — ((n*C + c) * L + l). N batched signals folded into rows.
//   conv_transpose1d weights: OIL, input-channel-major (transposed-conv
//     convention): Wt[(c_in*Cg_out + c_out_local) * kL + kl], Cg_out = C_out/groups.
//   causal_conv1d_update weights: depthwise, one row per channel: Wt[c*kL + kl].
//
// ── Accumulation (matches the conv2d contract) ──────────────────────────────
//   *_forward / *_backward_input / pad1d_*   — output OVERWRITTEN.
//   conv_transpose1d_backward_weight / _bias — dWt / dB ACCUMULATE (+=);
//                                              caller zeros them first.

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

void require_fp32(const char* op, const ::brotensor::Tensor& t,
                  const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
    }
}

// L_out of a 1D transposed convolution (torch ConvTranspose1d formula).
int convt1d_out_len(int L, int stride, int padding, int output_padding,
                    int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

} // namespace

// ─── conv_transpose1d_forward ──────────────────────────────────────────────
void conv_transpose1d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups,
                              ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose1d_forward";
    require_fp32(op, X, "X");
    require_fp32(op, Wt, "Wt");
    if (bias) require_fp32(op, *bias, "bias");
    check_groups(op, C_in, C_out, groups);
    if (kL < 1 || stride < 1 || dilation < 1 || padding < 0
        || output_padding < 0) {
        fail(op, "kL/stride/dilation must be >=1 and padding/output_padding >=0");
    }
    if (output_padding >= stride && output_padding >= dilation) {
        fail(op, "output_padding must be < stride or < dilation");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * L_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, out_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    // Seed every output sample with its channel bias, then scatter-add. Each
    // n exclusively owns Y's batch slice n, so this parallelizes across n
    // with no cross-thread writes; it fully completes (parallel_for blocks)
    // before the scatter-add pass below starts touching the same Y buffer.
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        for (int oc = 0; oc < C_out; ++oc) {
            const float bv = Bp ? Bp[oc] : 0.0f;
            float* y_row = Yp + (static_cast<long>(n) * C_out + oc) * L_out;
            for (int lo = 0; lo < L_out; ++lo) y_row[lo] = bv;
        }
    });
    // Interior region: input samples for which every kernel tap scatters
    // into a valid output position (1D mirror of conv_transpose2d.cpp's
    // split), computed once — independent of n/c_in. Only the thin border
    // of input samples needs the per-tap bounds check; the interior runs a
    // branch-free kl/oc_local loop.
    int l_lo = (padding + stride - 1) / stride;
    int l_hi = L_out - 1 + padding - (kL - 1) * dilation;
    l_hi = (l_hi >= 0) ? (l_hi / stride) : -1;
    if (l_lo < 0) l_lo = 0;
    if (l_hi >= L) l_hi = L - 1;
    const bool has_interior = l_lo <= l_hi;

    // Scatter: input sample (n, c_in, l) reaches l_out = l*stride - padding +
    // kl*dilation in each output channel of c_in's group. Each n only ever
    // scatters into Y's own batch slice n, so this parallelizes across n with
    // no cross-thread writes — the c_in loop stays sequential per-n, so the
    // += accumulation across c_in into a shared oc is untouched by another
    // thread.
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int oc_base = g * Cg_out;
            const float* x_row =
                Xp + (static_cast<long>(n) * C_in + c_in) * L;

            // Border sample: same bounds-checked scatter as before.
            auto scatter_bordered = [&](int l) {
                const float xv = x_row[l];
                if (xv == 0.0f) return;
                const int lo_origin = l * stride - padding;
                for (int kl = 0; kl < kL; ++kl) {
                    const int lo = lo_origin + kl * dilation;
                    if (lo < 0 || lo >= L_out) continue;
                    for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                        const int oc = oc_base + oc_local;
                        const int w_idx =
                            (c_in * Cg_out + oc_local) * kL + kl;
                        Yp[(static_cast<long>(n) * C_out + oc) * L_out + lo]
                            += xv * Wp[w_idx];
                    }
                }
            };

            // Interior sample: every tap guaranteed in-bounds — no checks.
            auto scatter_interior = [&](int l) {
                const float xv = x_row[l];
                if (xv == 0.0f) return;
                const int lo_origin = l * stride - padding;
                for (int kl = 0; kl < kL; ++kl) {
                    const int lo = lo_origin + kl * dilation;
                    for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                        const int oc = oc_base + oc_local;
                        const int w_idx =
                            (c_in * Cg_out + oc_local) * kL + kl;
                        Yp[(static_cast<long>(n) * C_out + oc) * L_out + lo]
                            += xv * Wp[w_idx];
                    }
                }
            };

            if (has_interior) {
                for (int l = 0; l < l_lo; ++l) scatter_bordered(l);
                for (int l = l_lo; l <= l_hi; ++l) scatter_interior(l);
                for (int l = l_hi + 1; l < L; ++l) scatter_bordered(l);
            } else {
                for (int l = 0; l < L; ++l) scatter_bordered(l);
            }
        }
    });
}

// ─── conv_transpose1d_backward_input ───────────────────────────────────────
void conv_transpose1d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, ::brotensor::Tensor& dX) {
    const char* op = "conv_transpose1d_backward_input";
    require_fp32(op, Wt, "Wt");
    require_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    const int in_cols = C_in * L;
    if (dX.rows != N || dX.cols != in_cols
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    const float* Wp  = Wt.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Adjoint of the transposed-conv scatter is a plain gather conv. Each n
    // exclusively owns dX's batch slice n (dY/Wt are read-only), so this
    // parallelizes across n with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int oc_base = g * Cg_out;
            for (int l = 0; l < L; ++l) {
                const int lo_origin = l * stride - padding;
                float acc = 0.0f;
                for (int kl = 0; kl < kL; ++kl) {
                    const int lo = lo_origin + kl * dilation;
                    if (lo < 0 || lo >= L_out) continue;
                    for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                        const int oc = oc_base + oc_local;
                        const int w_idx =
                            (c_in * Cg_out + oc_local) * kL + kl;
                        const int dy_idx =
                            (static_cast<long>(n) * C_out + oc) * L_out + lo;
                        acc += dYp[dy_idx] * Wp[w_idx];
                    }
                }
                dXp[(static_cast<long>(n) * C_in + c_in) * L + l] = acc;
            }
        }
    });
}

// ─── conv_transpose1d_backward_weight ──────────────────────────────────────
void conv_transpose1d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, ::brotensor::Tensor& dWt) {
    const char* op = "conv_transpose1d_backward_weight";
    require_fp32(op, X, "X");
    require_fp32(op, dY, "dY");
    require_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (dWt.rows != C_in || dWt.cols != Cg_out * kL) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (X.rows != N || X.cols != C_in * L) {
        fail(op, "X shape must be (N, C_in*L)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kL == 0) return;

    const float* Xp  = X.host_f32();
    const float* dYp = dY.host_f32();
    float* dWp = dWt.host_f32_mut();

    // One accumulation per weight element; += into dWt (caller zeroed it).
    //
    // NOT parallelized over n: n is the innermost reduction axis (every
    // batch item's contribution sums into the same dWp element), not an
    // outer axis — parallelizing it would race every thread on the same
    // dWt element. Left single-threaded per this task's scope.
    for (int c_in = 0; c_in < C_in; ++c_in) {
        const int g = c_in / Cg_in;
        const int oc_base = g * Cg_out;
        for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
            const int oc = oc_base + oc_local;
            for (int kl = 0; kl < kL; ++kl) {
                float acc = 0.0f;
                for (int n = 0; n < N; ++n) {
                    const float* x_row =
                        Xp + (static_cast<long>(n) * C_in + c_in) * L;
                    const float* dy_row =
                        dYp + (static_cast<long>(n) * C_out + oc) * L_out;
                    for (int l = 0; l < L; ++l) {
                        const int lo = l * stride - padding + kl * dilation;
                        if (lo < 0 || lo >= L_out) continue;
                        acc += x_row[l] * dy_row[lo];
                    }
                }
                dWp[(c_in * Cg_out + oc_local) * kL + kl] += acc;
            }
        }
    }
}

// ─── conv_transpose1d_backward_bias ────────────────────────────────────────
void conv_transpose1d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int L_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose1d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_out == 0 || N == 0 || L_out == 0) return;

    const float* dYp = dY.host_f32();
    float* dBp = dB.host_f32_mut();

    // Per-channel sum over (N, L_out); += into dB (caller zeroed it).
    //
    // NOT parallelized over n: same reason as backward_weight above — n is
    // the reduction axis here, not the outer axis.
    for (int oc = 0; oc < C_out; ++oc) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* dy_row =
                dYp + (static_cast<long>(n) * C_out + oc) * L_out;
            for (int lo = 0; lo < L_out; ++lo) acc += dy_row[lo];
        }
        dBp[oc] += acc;
    }
}

// ─── causal_conv1d_update ──────────────────────────────────────────────────
void causal_conv1d_update(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& Wt,
                          const ::brotensor::Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          ::brotensor::Tensor& state, ::brotensor::Tensor& Y) {
    const char* op = "causal_conv1d_update";
    require_fp32(op, X, "X");
    require_fp32(op, Wt, "Wt");
    if (bias) require_fp32(op, *bias, "bias");
    require_fp32(op, state, "state");
    if (kL < 1 || dilation < 1 || L_step < 1 || N < 0 || C < 1) {
        fail(op, "kL/dilation/L_step/C must be >=1 and N >=0");
    }
    if (Wt.rows != C || Wt.cols != kL) {
        fail(op, "Wt shape must be (C, kL) — one depthwise filter per channel");
    }
    if (bias && (bias->rows != C || bias->cols != 1)) {
        fail(op, "bias shape must be (C, 1)");
    }
    const int hist = (kL - 1) * dilation;   // state samples per channel
    if (state.rows != N || state.cols != C * hist) {
        fail(op, "state shape must be (N, C*(kL-1)*dilation)");
    }
    if (Y.rows != N || Y.cols != C * L_step
        || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, C * L_step, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0 || L_step == 0) return;

    const float* Xp = X.host_f32();
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Sp = state.host_f32_mut();
    float* Yp = Y.host_f32_mut();

    const int buf_len = hist + L_step;   // [state ++ new] window per channel

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* s_row =
                Sp + (static_cast<long>(n) * C + c) * hist;
            const float* x_row =
                Xp + (static_cast<long>(n) * C + c) * L_step;
            const float* w_row = Wp + static_cast<long>(c) * kL;
            const float bv = Bp ? Bp[c] : 0.0f;
            // buf index helper: [0, hist) -> state, [hist, buf_len) -> new.
            auto buf_at = [&](int idx) -> float {
                return idx < hist ? s_row[idx] : x_row[idx - hist];
            };
            // Output sample t convolves buf[t .. t + hist] (causal).
            for (int t = 0; t < L_step; ++t) {
                float acc = bv;
                for (int kl = 0; kl < kL; ++kl) {
                    acc += w_row[kl] * buf_at(t + kl * dilation);
                }
                Yp[(static_cast<long>(n) * C + c) * L_step + t] = acc;
            }
        }
    }
    // Roll the state forward: new state = last `hist` samples of the window.
    // Done after all reads above so a tiny scratch suffices per (n,c).
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float* s_row = Sp + (static_cast<long>(n) * C + c) * hist;
            const float* x_row =
                Xp + (static_cast<long>(n) * C + c) * L_step;
            for (int i = 0; i < hist; ++i) {
                const int idx = buf_len - hist + i;   // tail of the window
                s_row[i] = idx < hist
                               ? s_row[idx]            // still inside old state
                               : x_row[idx - hist];    // inside new input
            }
        }
    }
}

// ─── pad1d_forward ─────────────────────────────────────────────────────────
namespace {

// Map an output position p in [0, L_pad) to a source index in [0, L) for the
// given mode, or return -1 for a zero-padded position. p covers
// [pad_left, pad_left + L) as the copied interior.
int pad1d_src(const char* op, int p, int L, int pad_left, int mode) {
    const int rel = p - pad_left;          // index into the original [0, L)
    if (rel >= 0 && rel < L) return rel;    // interior — straight copy
    if (mode == 0) return -1;               // zero
    if (mode == 2) {                        // replicate (clamp to edge)
        return rel < 0 ? 0 : L - 1;
    }
    // mode == 1: reflect without repeating the edge sample (numpy 'reflect').
    if (L == 1) return 0;
    int q = rel;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

} // namespace

void pad1d_forward(const ::brotensor::Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode,
                   ::brotensor::Tensor& Y) {
    const char* op = "pad1d_forward";
    require_fp32(op, X, "X");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    if (X.rows != N || X.cols != C * L) {
        fail(op, "X shape must be (N, C*L)");
    }
    const int L_pad = L + pad_left + pad_right;
    if (Y.rows != N || Y.cols != C * L_pad
        || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, C * L_pad, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_row = Xp + (static_cast<long>(n) * C + c) * L;
            float* y_row = Yp + (static_cast<long>(n) * C + c) * L_pad;
            for (int p = 0; p < L_pad; ++p) {
                const int src = pad1d_src(op, p, L, pad_left, mode);
                y_row[p] = src < 0 ? 0.0f : x_row[src];
            }
        }
    }
}

// ─── pad1d_backward ────────────────────────────────────────────────────────
void pad1d_backward(const ::brotensor::Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode,
                    ::brotensor::Tensor& dX) {
    const char* op = "pad1d_backward";
    require_fp32(op, dY, "dY");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    const int L_pad = L + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * L_pad) {
        fail(op, "dY shape must be (N, C*(L+pad_left+pad_right))");
    }
    if (dX.rows != N || dX.cols != C * L
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, C * L, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Adjoint: scatter each output gradient onto the input sample it read.
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* dy_row = dYp + (static_cast<long>(n) * C + c) * L_pad;
            float* dx_row = dXp + (static_cast<long>(n) * C + c) * L;
            for (int l = 0; l < L; ++l) dx_row[l] = 0.0f;
            for (int p = 0; p < L_pad; ++p) {
                const int src = pad1d_src(op, p, L, pad_left, mode);
                if (src >= 0) dx_row[src] += dy_row[p];
            }
        }
    }
}

} // namespace brotensor::detail::cpu
