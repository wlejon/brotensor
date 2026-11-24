// CUDA public reductions (sum_rows, sum_cols, argmax_rows) backed by Brass MIR -> PTX.

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

inline std::uintptr_t addr(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}

inline void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": X must be FP16, BF16, or FP32");
    }
}

}  // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    require_fp(X.dtype, "sum_rows");
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) {
        Y.resize(M, 1, X.dtype);
    }
    if (M == 0) return;
    if (N == 0) {
        Y.zero();
        return;
    }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, addr(X.data));
    CUfunction fn = mir_function({MirKind::SumRows, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(fn, (std::min)(m, 65535u), X.data, Y.data, m, n);
}

void sum_cols(const Tensor& X, Tensor& Y) {
    require_fp(X.dtype, "sum_cols");
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) {
        Y.resize(1, N, X.dtype);
    }
    if (N == 0) return;
    if (M == 0) {
        Y.zero();
        return;
    }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, addr(X.data), addr(Y.data));
    CUfunction fn = mir_function({MirKind::SumCols, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(fn, mir_groups_1d(n, lanes), X.data, Y.data, m, n);
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    require_fp(X.dtype, "argmax_rows");
    const int M = X.rows;
    const int N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) {
        Idx.resize(M, 1, out_dt);
    }
    if (M == 0) return;
    if (N == 0) {
        Idx.zero();
        return;
    }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, addr(X.data));
    const std::uint8_t flags = (out_dt == Dtype::INT32) ? 1u : 0u;
    CUfunction fn = mir_function(
        {MirKind::ArgmaxRows, 0, X.dtype, out_dt, static_cast<std::uint8_t>(lanes), Dtype::FP32, flags});
    mir_dispatch_reduce(fn, (std::min)(m, 65535u), X.data, Idx.data, m, n);
}

}  // namespace brotensor::detail::cuda
