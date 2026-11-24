// Vulkan public reductions: sum_rows, sum_cols, argmax_rows. GPU contract
// (src/cuda/public_reductions.cu): sums are written in X's dtype with FP32
// accumulation; argmax writes INT32 when Idx is INT32-typed, FP32 otherwise,
// ties keep the lowest index. Kernels: src/kernels/mir_reduce.{h,cpp}.

#include "detail/kernels.h"
#include "vk_mir_cache.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cstdint>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void sum_rows(const Tensor& X, Tensor& Y) {
    dt_code(X.dtype, "sum_rows");
    const int M = X.rows, N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) Y.resize(M, 1, X.dtype);
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    DeviceCtx& d = device_of(X);
    const std::uint64_t ax = addr(X.data), ay = addr(Y.data);
    const std::uint32_t m = static_cast<std::uint32_t>(M), n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, ax);
    const MirKernel mk = mir_kernel(d, {MirKind::SumRows, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(d, mk, std::min<std::uint32_t>(m, 65535u), ax, ay, m, n);
}

void sum_cols(const Tensor& X, Tensor& Y) {
    dt_code(X.dtype, "sum_cols");
    const int M = X.rows, N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) Y.resize(1, N, X.dtype);
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    DeviceCtx& d = device_of(X);
    const std::uint64_t ax = addr(X.data), ay = addr(Y.data);
    const std::uint32_t m = static_cast<std::uint32_t>(M), n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, ax, ay);
    const MirKernel mk = mir_kernel(d, {MirKind::SumCols, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(d, mk, mir_groups_1d(n, lanes), ax, ay, m, n);
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    dt_code(X.dtype, "argmax_rows");
    const int M = X.rows, N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) Idx.resize(M, 1, out_dt);
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    DeviceCtx& d = device_of(X);
    const std::uint64_t ax = addr(X.data), ay = addr(Idx.data);
    const std::uint32_t m = static_cast<std::uint32_t>(M), n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, ax);
    const std::uint8_t flags = (out_dt == Dtype::INT32) ? 1u : 0u;
    const MirKernel mk = mir_kernel(
        d, {MirKind::ArgmaxRows, 0, X.dtype, out_dt, static_cast<std::uint8_t>(lanes), Dtype::FP32, flags});
    mir_dispatch_reduce(d, mk, std::min<std::uint32_t>(m, 65535u), ax, ay, m, n);
}

void fill_vulkan_vtable_reduce(::brotensor::detail::OpsVTable& v) {
    v.sum_rows = &sum_rows;
    v.sum_cols = &sum_cols;
    v.argmax_rows = &argmax_rows;
}

}  // namespace brotensor::detail::vulkan
