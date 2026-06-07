// ─── CPU upfirdn2d (StyleGAN3-R) ────────────────────────────────────────────
//
// Upsample (zero-insert) → pad/crop → 2D FIR correlation → downsample → gain.
// FP32 reference mirroring NVlabs `_upfirdn2d_ref` (general non-separable 2D
// path — the one config-R's radial filters need). The filter is a constant
// shared across channels (depthwise); there is no gradient to it.
//
// All of forward and backward funnel through `upfirdn2d_run`: the op is linear
// in X, so the backward is itself a forward call with up/down swapped, the
// filter flip inverted, and padding recomputed (mirrors `_upfirdn2d_cuda`).
//
// Index relation (per output pixel oh,ow and filter tap kh,kw), where the
// downsample picks conv row oh*down_y:
//   padded row py = oh*down_y + kh ;  upsampled row uy = py - pad_y0
//   contributes iff 0<=uy<H*up_y and uy % up_y == 0, reading X[iy=uy/up_y].
// Effective filter weight = flip_filter ? f[kh,kw] : f[fH-1-kh, fW-1-kw]
// (the flip makes the default flip_filter=false a true convolution).
//
// Output dims: H_out = (H*up_y + pad_y0 + pad_y1 - fH)/down_y + 1, i.e.
// ceil(conv_valid_len / down_y) — equal to the slicing x[::down_y] count.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " +
                                 name + " must be FP32 (CPU backend is "
                                 "FP32-only)");
    }
}

// Shared engine. `In` is (N, C*Hin*Win); resizes `Out` to (N, C*Hout*Wout).
void upfirdn2d_run(const ::brotensor::Tensor& In, int N, int C, int Hin, int Win,
                   const ::brotensor::Tensor& f, int fH, int fW,
                   int up_x, int up_y, int down_x, int down_y,
                   int px0, int px1, int py0, int py1,
                   bool flip_filter, float gain,
                   ::brotensor::Tensor& Out, const char* op) {
    check_fp32(In, op, "input");
    check_fp32(f, op, "f");
    if (up_x < 1 || up_y < 1 || down_x < 1 || down_y < 1) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": up/down factors must be >= 1");
    }
    if (In.rows != N || In.cols != C * Hin * Win) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": input shape mismatch");
    }
    if (f.rows != fH || f.cols != fW) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": filter shape mismatch");
    }
    const int Hu = Hin * up_y, Wu = Win * up_x;
    const int Hp = Hu + py0 + py1, Wp = Wu + px0 + px1;
    if (Hp < fH || Wp < fW) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": padded input smaller than filter");
    }
    const int Hc = Hp - fH + 1, Wc = Wp - fW + 1;
    const int Hout = (Hc - 1) / down_y + 1;
    const int Wout = (Wc - 1) / down_x + 1;
    const int out_cols = C * Hout * Wout;
    if (Out.rows != N || Out.cols != out_cols || Out.dtype != Dtype::FP32) {
        Out.resize(N, out_cols, Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Ip = In.host_f32();
    const float* Fp = f.host_f32();
    float* Op = Out.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const size_t in_base  = (static_cast<size_t>(n) * C + c) * Hin * Win;
            const size_t out_base = (static_cast<size_t>(n) * C + c) * Hout * Wout;
            for (int oh = 0; oh < Hout; ++oh) {
                const int py_base = oh * down_y;
                for (int ow = 0; ow < Wout; ++ow) {
                    const int px_base = ow * down_x;
                    float acc = 0.0f;
                    for (int kh = 0; kh < fH; ++kh) {
                        const int uy = py_base + kh - py0;
                        if (uy < 0 || uy >= Hu || (uy % up_y) != 0) continue;
                        const int iy = uy / up_y;
                        const int frow = flip_filter ? kh : (fH - 1 - kh);
                        for (int kw = 0; kw < fW; ++kw) {
                            const int ux = px_base + kw - px0;
                            if (ux < 0 || ux >= Wu || (ux % up_x) != 0) continue;
                            const int ix = ux / up_x;
                            const int fcol = flip_filter ? kw : (fW - 1 - kw);
                            acc += Ip[in_base + static_cast<size_t>(iy) * Win + ix] *
                                   Fp[static_cast<size_t>(frow) * fW + fcol];
                        }
                    }
                    Op[out_base + static_cast<size_t>(oh) * Wout + ow] = acc * gain;
                }
            }
        }
    }
}

// Forward output height/width for the given params (shared by the public
// forward and the backward's padding recompute).
inline int out_dim(int in, int up, int down, int pad0, int pad1, int fdim) {
    return (in * up + pad0 + pad1 - fdim) / down + 1;
}

} // namespace

void upfirdn2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, ::brotensor::Tensor& Y) {
    upfirdn2d_run(X, N, C, H, Wd, f, fH, fW,
                  up_x, up_y, down_x, down_y,
                  pad_x0, pad_x1, pad_y0, pad_y1,
                  flip_filter, gain, Y, "upfirdn2d_forward");
}

void upfirdn2d_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, ::brotensor::Tensor& dX) {
    // Forward output dims (dY's spatial extent).
    const int Hout = out_dim(H,  up_y, down_y, pad_y0, pad_y1, fH);
    const int Wout = out_dim(Wd, up_x, down_x, pad_x0, pad_x1, fW);
    // Backward padding (NVlabs _upfirdn2d_cuda): swap up<->down, flip the flip.
    const int p_x0 = fW - pad_x0 - 1;
    const int p_x1 = Wd * up_x - Wout * down_x + pad_x0 - up_x + 1;
    const int p_y0 = fH - pad_y0 - 1;
    const int p_y1 = H * up_y - Hout * down_y + pad_y0 - up_y + 1;
    upfirdn2d_run(dY, N, C, Hout, Wout, f, fH, fW,
                  /*up=*/down_x, down_y, /*down=*/up_x, up_y,
                  p_x0, p_x1, p_y0, p_y1,
                  !flip_filter, gain, dX, "upfirdn2d_backward");
    if (dX.rows != N || dX.cols != C * H * Wd) {
        throw std::runtime_error("upfirdn2d_backward: internal dX shape "
                                 "mismatch (param inconsistency)");
    }
}

} // namespace brotensor::detail::cpu
