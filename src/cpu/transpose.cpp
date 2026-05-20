// ─── CPU NCHW <-> sequence transposes (CHUNK 4) ────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/transpose.cu — FP32 path
// only (CPU is FP32-only). Pure gathers — no arithmetic, no rounding.
//
// Layout convention (matches the GPU verbatim):
//   NCHW form     : X[((n*C + c) * H + h) * W + w], shape (N, C*H*W).
//   sequence form : Y[(n*HW + p) * C + c],          shape (N*HW, C),
//                   where HW = H*W and p = h*W + w.
//
//   nchw_to_sequence : (N, C*H*W) -> (N*HW, C)
//   sequence_to_nchw : (N*HW, C)  -> (N, C*H*W)   (exact inverse)
//
// Both ops simply OVERWRITE the output tensor.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

inline void check_dims(const char* op, int N, int C, int H, int W) {
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error(std::string(op) + ": negative dimension");
    }
}

} // namespace

void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      ::brotensor::Tensor& Y) {
    check_fp32(X, "nchw_to_sequence", "X");
    check_dims("nchw_to_sequence", N, C, H, W);
    const int HW = H * W;
    const int rows = N * HW;
    if (Y.rows != rows || Y.cols != C || Y.dtype != Dtype::FP32) {
        Y.resize(rows, C, Dtype::FP32);
    }
    if (rows == 0 || C == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < HW; ++p) {
                Yp[(n * HW + p) * C + c] = Xp[(n * C + c) * HW + p];
            }
        }
    }
}

void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      ::brotensor::Tensor& Y) {
    check_fp32(X, "sequence_to_nchw", "X");
    check_dims("sequence_to_nchw", N, C, H, W);
    const int HW = H * W;
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < HW; ++p) {
                Yp[(n * C + c) * HW + p] = Xp[(n * HW + p) * C + c];
            }
        }
    }
}

} // namespace brotensor::detail::cpu
