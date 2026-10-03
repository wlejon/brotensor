// Vulkan public reductions: sum_rows, sum_cols, argmax_rows. GPU contract
// (src/cuda/public_reductions.cu): sums are written in X's dtype with FP32
// accumulation; argmax writes INT32 when Idx is INT32-typed, FP32 otherwise,
// ties keep the lowest index. Kernels: shaders/reduce_rows.comp, sum_cols.comp.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {
struct ReducePush { std::uint64_t x, y; std::uint32_t m, n; };
}  // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    dt_code(X.dtype, "sum_rows");
    const int M = X.rows, N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) Y.resize(M, 1, X.dtype);
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::sum_rows_f32, X.dtype));
    const ReducePush pc{addr(X.data), addr(Y.data), static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N)};
    launch(d, k, pc, std::min<std::uint32_t>(pc.m, 65535));
}

void sum_cols(const Tensor& X, Tensor& Y) {
    dt_code(X.dtype, "sum_cols");
    const int M = X.rows, N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) Y.resize(1, N, X.dtype);
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::sum_cols_f32, X.dtype));
    const ReducePush pc{addr(X.data), addr(Y.data), static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N)};
    launch(d, k, pc, groups_1d(pc.n, k));
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    dt_code(X.dtype, "argmax_rows");
    const int M = X.rows, N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) Idx.resize(M, 1, out_dt);
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::argmax_rows_f32, X.dtype),
                                        {out_dt == Dtype::INT32 ? 1u : 0u});
    const ReducePush pc{addr(X.data), addr(Idx.data), static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N)};
    launch(d, k, pc, std::min<std::uint32_t>(pc.m, 65535));
}

void fill_vulkan_vtable_reduce(::brotensor::detail::OpsVTable& v) {
    v.sum_rows = &sum_rows;
    v.sum_cols = &sum_cols;
    v.argmax_rows = &argmax_rows;
}

}  // namespace brotensor::detail::vulkan
