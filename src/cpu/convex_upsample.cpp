// ─── CPU convex (mask-based) upsample, NCHW ─────────────────────────────────
//
// FP32 scalar host implementation of convex_upsample_forward — the RAFT-style
// "learned convex combination" upsampler used by optical-flow, stereo, and
// surface-normal refinement (DSINE up_prob_head). Each low-res pixel is
// expanded to a scale×scale block; every fine pixel in the block is a softmax-
// weighted blend of the 3×3 low-res neighborhood around its source pixel:
//
//   Y[n, c, k*y+sy, k*x+sx] = sum_{m=0..8} W[n, m, sy, sx, y, x] * X[n, c, ny, nx]
//   W = softmax over the 9 neighbors m of Mask[n, m, sy, sx, y, x]
//   neighbor m: ny = clamp(y - 1 + m/3), nx = clamp(x - 1 + m%3)   (replicate pad)
//
// Mask layout (matches torch view (N, 9, k, k, H, W)):
//   flat channel = ((m*k + sy)*k + sx), so
//   Mask[n, m, sy, sx, y, x] = mask[n, (m*k*k + sy*k + sx)*HW + (y*W + x)].
//   k = scale; the 9 axis is the 3×3 spatial neighborhood (m = my*3 + mx).
//
//   X:    (N, C*H*W).
//   Mask: (N, 9*k*k*H*W).
//   Y:    (N, C*(k*H)*(k*W)), resized + dtype-set to X.
//
// Softmax accumulates in double. NOTE: the softmax over the 9 neighbors is the
// same for every channel, but we recompute it per (channel, fine-pixel) for
// implementation simplicity — fine at the C/scale this targets (DSINE C=3,
// k=8). Y OVERWRITTEN. Inference-only: no backward.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32)
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
}

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

} // namespace

void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y) {
    const char* op = "convex_upsample_forward";
    check_fp32(X, op, "X");
    check_fp32(Mask, op, "Mask");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (scale < 1) fail(op, "scale must be >=1");
    const int HW = H * W;
    const int kk = scale * scale;
    if (X.rows != N || X.cols != C * HW) fail(op, "X shape must be (N, C*H*W)");
    if (Mask.rows != N || Mask.cols != 9 * kk * HW)
        fail(op, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int oH = scale * H, oW = scale * W, oHW = oH * oW;
    if (Y.rows != N || Y.cols != C * oHW || Y.dtype != Dtype::FP32)
        Y.resize(N, C * oHW, Dtype::FP32);
    if (N == 0) return;

    const float* Xp = X.host_f32();
    const float* Mp = Mask.host_f32();
    float* Yp = Y.host_f32_mut();

    double w[9];
    for (int n = 0; n < N; ++n) {
        const float* x_img = Xp + static_cast<long>(n) * C * HW;
        const float* m_img = Mp + static_cast<long>(n) * 9 * kk * HW;
        float* y_img = Yp + static_cast<long>(n) * C * oHW;
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            for (int sy = 0; sy < scale; ++sy) {
              for (int sx = 0; sx < scale; ++sx) {
                const int sub = sy * scale + sx;
                // softmax over the 9 neighbors m
                double mx = -1e300;
                for (int m = 0; m < 9; ++m) {
                    const double v = m_img[(static_cast<long>(m) * kk + sub) * HW + pix];
                    if (v > mx) mx = v;
                }
                double sum = 0.0;
                for (int m = 0; m < 9; ++m) {
                    const double e = std::exp(
                        m_img[(static_cast<long>(m) * kk + sub) * HW + pix] - mx);
                    w[m] = e; sum += e;
                }
                const double invs = 1.0 / sum;
                for (int m = 0; m < 9; ++m) w[m] *= invs;

                const int oy = scale * y + sy, ox = scale * x + sx;
                const long opix = static_cast<long>(oy) * oW + ox;
                for (int c = 0; c < C; ++c) {
                    const float* xc = x_img + static_cast<long>(c) * HW;
                    double acc = 0.0;
                    for (int m = 0; m < 9; ++m) {
                        const int ny = clampi(y - 1 + m / 3, 0, H - 1);
                        const int nx = clampi(x - 1 + m % 3, 0, W - 1);
                        acc += w[m] * xc[static_cast<long>(ny) * W + nx];
                    }
                    y_img[static_cast<long>(c) * oHW + opix] = static_cast<float>(acc);
                }
              }
            }
          }
        }
    }
}

} // namespace brotensor::detail::cpu
