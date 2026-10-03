// Vulkan scatter-adds: scatter_rows_add (dX = 0, dX[Idx[m], :] += dY[m, :])
// and embedding_lookup_backward (dTable[idx[b], :] += dOut[b, :]),
// deterministic (shaders/scatter_add.comp): the (row, m) pairs are bitonic-
// sorted on the device, then each destination row sums its run of source
// rows in increasing m, in FP32 from the row's current value, with one
// rounding into the dtype. No float atomics: VK_EXT_shader_atomic_float would
// make the sum order (and the 16-bit results) vary from run to run, and
// FP32 now matches the CPU reference bit for bit. Contracts follow CUDA
// (src/cuda/gather_scatter.cu, embedding.cu): FP32 / FP16 / BF16, the
// destination in the source's dtype; an index outside [0, R) is skipped
// (a device fault would lose the device).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct SaPush {
    std::uint64_t idx, keys, src, dst;
    std::uint32_t m, r, c, p, k, j;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

// dst(R, C)[idx[m], :] += src(M, C)[m, :] in `dt`.
void scatter_add(DeviceCtx& d, std::uint64_t idx, int M, int R, int C, std::uint64_t src, std::uint64_t dst, Dtype dt,
                 const char* op) {
    if (M <= 0 || R <= 0 || C <= 0) return;
    if (static_cast<long long>(R) * C > 0xffffffffLL || static_cast<long long>(M) * C > 0xffffffffLL) {
        fail(op, "tensor too large");
    }
    std::uint32_t P = 1;
    while (P < static_cast<std::uint32_t>(M)) P <<= 1;
    Tensor keys = Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(P), 2, Dtype::FP32);
    const ShaderId id = dt_variant(ShaderId::scatter_add_f32, dt, op);
    SaPush pc{idx, addr(keys.data), src, dst, static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(R),
              static_cast<std::uint32_t>(C), P, 0, 0};
    const Kernel& kk = d.pipelines().get(id, {std::uint32_t(SA_KEYS)});
    launch(d, kk, pc, groups_1d(P, kk));
    if (P <= 512) {
        launch(d, d.pipelines().get(id, {std::uint32_t(SA_LOCAL)}), pc, 1);
    } else {
        const Kernel& ks = d.pipelines().get(id, {std::uint32_t(SA_STEP)});
        for (std::uint32_t k = 2; k <= P; k <<= 1) {
            for (std::uint32_t j = k >> 1; j > 0; j >>= 1) {
                pc.k = k;
                pc.j = j;
                launch(d, ks, pc, groups_1d(P, ks));
            }
        }
    }
    const std::uint32_t gx = static_cast<std::uint32_t>((C + 255) / 256);
    launch(d, d.pipelines().get(id, {std::uint32_t(SA_SUM)}), pc, gx, static_cast<std::uint32_t>(std::min(M, 65535)));
}

void need_float(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        fail(op, std::string(what) + " must be FP32, FP16 or BF16");
    }
}

}  // namespace

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
    scatter_add(device_of(dY), addr(Idx.data), M, R, C, addr(dY.data), addr(dX.data), dY.dtype, op);
}

void embedding_lookup_backward(const Tensor& dOut, const int32_t* d_idx, int B, Tensor& dTable) {
    constexpr const char* op = "embedding_lookup_backward";
    need_float(dTable, op, "dTable");
    if (dOut.dtype != dTable.dtype) fail(op, "dOut/dTable dtype must match");
    const int D = dTable.cols;
    if (B <= 0 || D == 0) return;
    if (dOut.size() < static_cast<long long>(B) * D) fail(op, "dOut is smaller than B x D");
    scatter_add(device_of(dTable), addr(d_idx), B, dTable.rows, D, addr(dOut.data), addr(dTable.data), dTable.dtype,
                op);
}

void fill_vulkan_vtable_scatter(::brotensor::detail::OpsVTable& v) {
    v.scatter_rows_add = &vulkan::scatter_rows_add;
    v.embedding_lookup_backward = &vulkan::embedding_lookup_backward;
}

}  // namespace brotensor::detail::vulkan
