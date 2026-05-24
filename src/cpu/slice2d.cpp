// ─── CPU 2D spatial slice / crop ────────────────────────────────────────────
//
// FP32 scalar host implementation. Extracts the (H_out, W_out) sub-region
// of an NCHW tensor starting at (h0, w0); N and C pass through unchanged.
// This is the "crop" half of the pad2d / slice2d pair — together they are
// inverse operations when pad2d uses zero mode.
//
// Memory layout (NCHW flat):
//   X / dX : ((n*C + c)*H     + h)*W     + w
//   Y / dY : ((n*C + c)*H_out + h)*W_out + w
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   slice2d_forward  — Y  OVERWRITTEN (straight copy).
//   slice2d_backward — dX OVERWRITTEN (zeroed, then dY copied into the slice
//                       region). The forward is a pure read so the adjoint
//                       has no aliasing; this is just a memset+memcpy in
//                       disguise.

#include <brotensor/tensor.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) +
                 " must be FP32 (CPU backend is FP32-only)");
    }
}

inline void check_args(const char* op,
                       int N, int C, int H, int W,
                       int h0, int w0, int H_out, int W_out) {
    if (N < 0 || C < 1 || H < 0 || W < 0) {
        fail(op, "C must be >=1; N, H, W must be >=0");
    }
    if (H_out < 0 || W_out < 0) {
        fail(op, "H_out and W_out must be >=0");
    }
    if (h0 < 0 || w0 < 0) {
        fail(op, "h0 and w0 must be >=0");
    }
    if (h0 + H_out > H) {
        fail(op, "h0 + H_out must be <= H");
    }
    if (w0 + W_out > W) {
        fail(op, "w0 + W_out must be <= W");
    }
}

} // namespace

// ─── slice2d_forward ───────────────────────────────────────────────────────

void slice2d_forward(const ::brotensor::Tensor& X,
                     int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out,
                     ::brotensor::Tensor& Y) {
    const char* op = "slice2d_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan =
                Xp + (static_cast<long>(n) * C + c) * H * W;
            float* y_chan =
                Yp + (static_cast<long>(n) * C + c) * H_out * W_out;
            for (int h = 0; h < H_out; ++h) {
                const float* x_row =
                    x_chan + static_cast<long>(h0 + h) * W + w0;
                float* y_row = y_chan + static_cast<long>(h) * W_out;
                std::memcpy(y_row, x_row,
                            static_cast<size_t>(W_out) * sizeof(float));
            }
        }
    }
}

// ─── slice2d_backward ──────────────────────────────────────────────────────

void slice2d_backward(const ::brotensor::Tensor& dY,
                      int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out,
                      ::brotensor::Tensor& dX) {
    const char* op = "slice2d_backward";
    check_fp32(dY, op, "dY");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (dY.rows != N || dY.cols != C * H_out * W_out) {
        fail(op, "dY shape must be (N, C*H_out*W_out)");
    }
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Zero dX, then scatter dY into the slice region.
    const long total_in = static_cast<long>(N) * cols_in;
    for (long i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    if (H_out == 0 || W_out == 0) return;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* dy_chan =
                dYp + (static_cast<long>(n) * C + c) * H_out * W_out;
            float* dx_chan =
                dXp + (static_cast<long>(n) * C + c) * H * W;
            for (int h = 0; h < H_out; ++h) {
                const float* dy_row =
                    dy_chan + static_cast<long>(h) * W_out;
                float* dx_row =
                    dx_chan + static_cast<long>(h0 + h) * W + w0;
                std::memcpy(dx_row, dy_row,
                            static_cast<size_t>(W_out) * sizeof(float));
            }
        }
    }
}

} // namespace brotensor::detail::cpu
