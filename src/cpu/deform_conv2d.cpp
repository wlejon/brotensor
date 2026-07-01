// ─── CPU modulated deformable conv2d (torchvision deform_conv2d v2) ─────────
//
// FP32 scalar host implementation, forward/inference only. Mirrors
// torchvision's deformable_im2col + GEMM, fused into a single direct loop:
// for each output pixel and conv group, each kH×kW tap is bilinearly sampled
// from X at a per-tap, per-pixel-shifted location (offset field) with ZERO
// padding outside the input, optionally reweighted by the mask modulator, then
// reduced against the OIHW weight.
//
// Layouts match conv2d_forward (X/Y/Wt NCHW/OIHW). offset/mask are dense NCHW
// fields produced by the model's offset_conv / modulator_conv:
//   offset: (N, deform_groups*2*kH*kW, H_out, W_out), channel
//           grp*(2*kH*kW) + 2*(kh*kW+kw) [+1 for the col axis].
//   mask:   (N, deform_groups*kH*kW,   H_out, W_out), channel
//           grp*(kH*kW) + (kh*kW+kw); null == all modulators 1.
//
// bilinear_interpolate matches torchvision exactly (the (h<=-1 || h>=H ...)
// early-out plus per-corner in-bounds guards), so taps that fall on or past the
// border contribute zero — NOT clamped.

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

inline int out_dim(int in, int pad, int dil, int k, int stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// torchvision bilinear_interpolate: zero outside [0,H)×[0,W), per-corner guard.
inline float bilinear(const float* in, int H, int W, float h, float w) {
    if (h <= -1.0f || static_cast<float>(H) <= h ||
        w <= -1.0f || static_cast<float>(W) <= w) {
        return 0.0f;
    }
    int h_low = static_cast<int>(std::floor(h));
    int w_low = static_cast<int>(std::floor(w));
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    float lh = h - h_low, lw = w - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;
    float v1 = (h_low >= 0 && w_low >= 0) ? in[h_low * W + w_low] : 0.0f;
    float v2 = (h_low >= 0 && w_high <= W - 1) ? in[h_low * W + w_high] : 0.0f;
    float v3 = (h_high <= H - 1 && w_low >= 0) ? in[h_high * W + w_low] : 0.0f;
    float v4 = (h_high <= H - 1 && w_high <= W - 1) ? in[h_high * W + w_high] : 0.0f;
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

} // namespace

void deform_conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& offset,
                           const ::brotensor::Tensor* mask,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, int deform_groups,
                           ::brotensor::Tensor& Y) {
    if (Wt.dtype != X.dtype || offset.dtype != X.dtype ||
        (mask && mask->dtype != X.dtype) || (bias && bias->dtype != X.dtype)) {
        throw std::runtime_error(
            "deform_conv2d_forward: X, offset, mask, Wt, bias dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "deform_conv2d_forward: groups must divide C_in and C_out");
    }
    if (deform_groups < 1 || C_in % deform_groups != 0) {
        throw std::runtime_error(
            "deform_conv2d_forward: deform_groups must divide C_in");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int c_per_off_grp = C_in / deform_groups;
    const int H_out = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int W_out = out_dim(W, pad_w, dil_w, kW, stride_w);
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("deform_conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Op = offset.host_f32();
    const float* Mp = mask ? mask->host_f32() : nullptr;
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    const int ksz = kH * kW;
    const int off_row_stride  = deform_groups * 2 * ksz * H_out * W_out;  // per-n
    const int mask_row_stride = deform_groups * ksz * H_out * W_out;      // per-n

    // The modulated bilinear sample depends only on (n, oh, ow, ic, tap) — it
    // never depends on oc — but the original oc -> ic_local -> kh,kw loop
    // order resampled it once per output channel (O(C_out) redundant work).
    // Reorder to n -> g -> oh -> ow: sample each group's (Cg_in * ksz) taps
    // once into a scratch column (im2col-style), then dot that column against
    // every oc's weight row within the group. The inner accumulation order
    // (ic_local-major, tap-minor) is preserved exactly, so results match the
    // original bit-for-bit.
    // Each n exclusively owns Y's batch slice n (X/offset/mask/Wt/bias are
    // read-only), so this parallelizes across n with no cross-thread writes.
    // The im2col scratch `col` MUST be declared inside this lambda (fresh
    // per invocation, even across threads) rather than hoisted above the
    // loop — a single shared vector reused across n/g would race under
    // parallel_for (every thread scribbling into the same buffer).
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        std::vector<float> col(static_cast<size_t>(Cg_in) * ksz);
        const float* off_n  = Op + static_cast<size_t>(n) * off_row_stride;
        const float* mask_n = Mp ? Mp + static_cast<size_t>(n) * mask_row_stride : nullptr;
        for (int g = 0; g < groups; ++g) {
            const int ic_base = g * Cg_in;
            for (int oh = 0; oh < H_out; ++oh) {
                const int in_h_origin = oh * stride_h - pad_h;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int in_w_origin = ow * stride_w - pad_w;
                    const int sp = oh * W_out + ow;

                    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                        const int ic = ic_base + ic_local;
                        const int off_grp = ic / c_per_off_grp;
                        const float* in_ch = Xp + (static_cast<size_t>(n) * C_in + ic) * H * W;
                        const float* off_grp_base =
                            off_n + static_cast<size_t>(off_grp) * 2 * ksz * H_out * W_out;
                        const float* mask_grp_base =
                            mask_n ? mask_n + static_cast<size_t>(off_grp) * ksz * H_out * W_out
                                   : nullptr;
                        float* col_ic = col.data() + ic_local * ksz;
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int tap = kh * kW + kw;
                                const float off_y = off_grp_base[((2 * tap) * H_out + oh) * W_out + ow];
                                const float off_x = off_grp_base[((2 * tap + 1) * H_out + oh) * W_out + ow];
                                const float m = mask_grp_base
                                    ? mask_grp_base[(tap * H_out + oh) * W_out + ow] : 1.0f;
                                const float y = in_h_origin + kh * dil_h + off_y;
                                const float x = in_w_origin + kw * dil_w + off_x;
                                const float val = bilinear(in_ch, H, W, y, x);
                                col_ic[tap] = m * val;
                            }
                        }
                    }

                    const int col_len = Cg_in * ksz;
                    for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                        const int oc = g * Cg_out + oc_local;
                        const int w_oc_base = oc * Cg_in * ksz;
                        const float bias_v = Bp ? Bp[oc] : 0.0f;
                        float acc = bias_v;
                        for (int k = 0; k < col_len; ++k) {
                            acc += Wp[w_oc_base + k] * col[k];
                        }
                        Yp[(static_cast<size_t>(n) * C_out + oc) * H_out * W_out + sp] = acc;
                    }
                }
            }
        }
    });
}

} // namespace brotensor::detail::cpu
