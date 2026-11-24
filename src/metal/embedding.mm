// Metal embedding lookup (forward via MirKind::Remap gather_rows; backward
// and scatter_rows_add via deterministic bitonic-sort MirKind::ScatterAdd).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace brotensor::detail::metal {

// Defined in resample.mm (MirKind::Remap RM_GATHER_ROWS).
void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need_float(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        fail(op, std::string(what) + " must be FP32, FP16 or BF16");
    }
}

void scatter_add(const void* idx, ::brotensor::Device dev, int M, int R, int C,
                 const void* src, const void* dst, Dtype dt, const char* op) {
    if (M <= 0 || R <= 0 || C <= 0) return;
    if (static_cast<long long>(R) * C > 0xffffffffLL ||
        static_cast<long long>(M) * C > 0xffffffffLL) {
        fail(op, "tensor too large");
    }
    std::uint32_t P = 1;
    while (P < static_cast<std::uint32_t>(M)) P <<= 1;
    Tensor keys = Tensor::empty_on(dev, static_cast<int>(P), 2, Dtype::FP32);
    Tensor keys2;
    const void* keys_a = keys.data;
    const void* keys_b = nullptr;
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t r = static_cast<std::uint32_t>(R);
    const std::uint32_t c = static_cast<std::uint32_t>(C);

    const MirKernel mk_keys = mir_kernel({MirKind::ScatterAdd, static_cast<std::uint32_t>(SA_KEYS), dt});
    mir_dispatch_scatter(mk_keys, mir_groups_1d(P, 1), 1, op,
                         idx, keys_a, keys_b, src, dst, m, r, c, P, 0, 0);

    if (P <= 512) {
        const MirKernel mk_local = mir_kernel({MirKind::ScatterAdd, static_cast<std::uint32_t>(SA_LOCAL), dt});
        mir_dispatch_scatter(mk_local, 1, 1, op,
                             idx, keys_a, keys_b, src, dst, m, r, c, P, 0, 0);
    } else {
        keys2 = Tensor::empty_on(dev, static_cast<int>(P), 2, Dtype::FP32);
        keys_b = keys2.data;
        const MirKernel mk_step = mir_kernel({MirKind::ScatterAdd, static_cast<std::uint32_t>(SA_STEP), dt});
        const std::uint32_t gs = mir_groups_1d(P, 1);
        for (std::uint32_t k = 2; k <= P; k <<= 1) {
            for (std::uint32_t j = k >> 1; j > 0; j >>= 1) {
                mir_dispatch_scatter(mk_step, gs, 1, op,
                                     idx, keys_a, keys_b, src, dst, m, r, c, P, k, j);
                std::swap(keys_a, keys_b);
            }
        }
    }
    const std::uint32_t gx = static_cast<std::uint32_t>((C + 255) / 256);
    const std::uint32_t gy = static_cast<std::uint32_t>(std::min(M, 65535));
    const MirKernel mk_sum = mir_kernel({MirKind::ScatterAdd, static_cast<std::uint32_t>(SA_SUM), dt});
    mir_dispatch_scatter(mk_sum, gx, gy, op,
                         idx, keys_a, keys_b, src, dst, m, r, c, P, 0, 0);
}

}  // namespace

void embedding_lookup_forward(const Tensor& table, const int32_t* d_idx, int B, Tensor& out) {
    constexpr const char* op = "embedding_lookup_forward";
    need_float(table, op, "table");
    if (B < 0) fail(op, "B must be >= 0");
    if (out.rows != B || out.cols != table.cols || out.dtype != table.dtype) {
        out.resize(B, table.cols, table.dtype);
    }
    if (B == 0 || table.cols == 0) return;
    if (!d_idx) fail(op, "null index pointer");
    NSUInteger ofs = 0;
    metal_impl::pool_require(d_idx, ofs, op, "indices");
    const Tensor idx = Tensor::view(table.device, const_cast<int32_t*>(d_idx), B, 1, Dtype::INT32);
    metal::gather_rows(table, idx, out);
}

void embedding_lookup_backward(const Tensor& dOut, const int32_t* d_idx, int B, Tensor& dTable) {
    constexpr const char* op = "embedding_lookup_backward";
    need_float(dTable, op, "dTable");
    if (dOut.dtype != dTable.dtype) fail(op, "dOut/dTable dtype must match");
    const int D = dTable.cols;
    if (B <= 0 || D == 0) return;
    if (dOut.size() < static_cast<long long>(B) * D) fail(op, "dOut is smaller than B x D");
    NSUInteger ofs = 0;
    metal_impl::pool_require(d_idx, ofs, op, "indices");
    scatter_add(d_idx, dTable.device, B, dTable.rows, D, dOut.data, dTable.data, dTable.dtype, op);
}

void scatter_rows_add(const Tensor& dY, const Tensor& Idx, int R, Tensor& dX) {
    constexpr const char* op = "scatter_rows_add";
    need_float(dY, op, "dY");
    if (Idx.dtype != Dtype::INT32) fail(op, "Idx must be INT32");
    if (Idx.cols != 1) fail(op, "Idx must be shaped (M, 1)");
    if (R < 0) fail(op, "R must be >= 0");
    const int M = Idx.rows;
    if (dY.rows != M) fail(op, "dY.rows must equal Idx.rows");
    const int C = dY.cols;
    if (dX.data == nullptr) dX.device = dY.device;
    if (dX.rows != R || dX.cols != C || dX.dtype != dY.dtype) dX.resize(R, C, dY.dtype);
    if (R == 0 || C == 0) return;
    dX.zero();
    scatter_add(Idx.data, dY.device, M, R, C, dY.data, dX.data, dY.dtype, op);
}

}  // namespace brotensor::detail::metal
