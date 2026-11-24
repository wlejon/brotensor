// Metal public reductions: sum_rows, sum_cols, argmax_rows via single-source
// brass MIR kernels (mtl_mir_cache.mm).

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

std::uint64_t dev_addr(const Tensor& t) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.data));
}

void require_fp(const Tensor& t, const char* op) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": X must be FP32, FP16, or BF16");
    }
}

}  // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    require_fp(X, "sum_rows");
    const int M = X.rows, N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) Y.resize(M, 1, X.dtype);
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X));
    const MirKernel mk = mir_kernel(
        {MirKind::SumRows, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(mk, std::min<std::uint32_t>(m, 65535u), X, Y, m, n);
}

void sum_cols(const Tensor& X, Tensor& Y) {
    require_fp(X, "sum_cols");
    const int M = X.rows, N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) Y.resize(1, N, X.dtype);
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X), dev_addr(Y));
    const MirKernel mk = mir_kernel(
        {MirKind::SumCols, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(mk, mir_groups_1d(n, lanes), X, Y, m, n);
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    require_fp(X, "argmax_rows");
    const int M = X.rows, N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) Idx.resize(M, 1, out_dt);
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X));
    const std::uint8_t flags = (out_dt == Dtype::INT32) ? 1u : 0u;
    const MirKernel mk = mir_kernel(
        {MirKind::ArgmaxRows, 0, X.dtype, out_dt,
         static_cast<std::uint8_t>(lanes), Dtype::FP32, flags});
    mir_dispatch_reduce(mk, std::min<std::uint32_t>(m, 65535u), X, Idx, m, n);
}

}  // namespace brotensor::detail::metal
