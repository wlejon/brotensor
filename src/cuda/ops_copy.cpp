// CUDA batched NCHW <-> sequence transpose backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

void check_dims(const char* op, int N, int C, int H, int W) {
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error(std::string(op) + ": negative dimension");
    }
}

void dispatch_transpose(const Tensor& X, Tensor& Y, int batch, int rows, int cols, const char* op) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": unsupported dtype (expected FP32/FP16/BF16)");
    }
    const std::uint32_t esz = static_cast<std::uint32_t>(::brotensor::dtype_size_bytes(X.dtype));
    const std::uint32_t b = static_cast<std::uint32_t>(batch);
    const std::uint32_t r = static_cast<std::uint32_t>(rows);
    const std::uint32_t c = static_cast<std::uint32_t>(cols);
    const std::uint32_t tiles_c = (c + 31u) / 32u;
    const std::uint32_t tiles_r = (r + 31u) / 32u;
    const std::uint32_t gx = (std::max)(tiles_c, 1u);
    const std::uint32_t gy = std::clamp(tiles_r, 1u, 65535u);
    const std::uint32_t gz = std::clamp(b, 1u, 65535u);
    CUfunction fn = mir_function({MirKind::Transpose, esz});
    mir_dispatch_transpose(fn, gx, gy, gz, X.data, Y.data, b, r, c);
}

}  // namespace

void nchw_to_sequence(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    check_dims("nchw_to_sequence", N, C, H, W);
    const int HW = H * W;
    const int rows = N * HW;
    if (Y.rows != rows || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(rows, C, X.dtype);
    }
    if (rows == 0 || C == 0) return;
    dispatch_transpose(X, Y, N, C, HW, "nchw_to_sequence");
}

void sequence_to_nchw(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    check_dims("sequence_to_nchw", N, C, H, W);
    const int HW = H * W;
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;
    dispatch_transpose(X, Y, N, HW, C, "sequence_to_nchw");
}

}  // namespace brotensor::detail::cuda
