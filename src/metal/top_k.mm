// Metal per-row exact top-k selection and segment softmax statistics via
// single-source brass MIR kernel (src/kernels/mir_topk.cpp -> mtl_mir_cache.mm).

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "tensor must be FP32, FP16 or BF16");
    }
}

const void* byte_offset(const void* p, std::size_t off) {
    return static_cast<const char*>(p) + off;
}

}  // namespace

void top_k_rows(const Tensor& X, int k, Tensor& Vals, Tensor& Idx) {
    constexpr const char* op = "top_k_rows";
    require_fp(X.dtype, op);
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");
    if (Vals.rows != R || Vals.cols != k || Vals.dtype != Dtype::FP32) {
        Vals.resize(R, k, Dtype::FP32);
    }
    if (Idx.rows != R || Idx.cols != k || Idx.dtype != Dtype::INT32) {
        Idx.resize(R, k, Dtype::INT32);
    }
    if (R == 0) return;

    const std::uint64_t groups = (static_cast<std::uint64_t>(C) + 255u) / 256u;
    if (groups > 65535u) fail(op, "rows longer than 16M elements");
    const MirKernel mk = mir_kernel({MirKind::Topk, 0u, X.dtype});
    const std::size_t es = (X.dtype == Dtype::FP32) ? 4u : 2u;
    for (int r0 = 0; r0 < R; r0 += 65535) {
        const int n = std::min(65535, R - r0);
        mir_dispatch_topk(
            mk, static_cast<std::uint32_t>(groups), static_cast<std::uint32_t>(n), op,
            byte_offset(X.data, static_cast<std::size_t>(r0) * C * es),
            byte_offset(Vals.data, static_cast<std::size_t>(r0) * k * 4u),
            byte_offset(Idx.data, static_cast<std::size_t>(r0) * k * 4u),
            nullptr, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(C),
            static_cast<std::uint32_t>(k), 0u);
    }
}

void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets, Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    require_fp(logits.dtype, op);
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) {
        fail(op, "seg_offsets must be (S+1, 1) INT32");
    }
    const int S = seg_offsets.rows - 1;
    if (out.rows != S || out.cols != 4 || out.dtype != logits.dtype) {
        out.resize(S, 4, logits.dtype);
    }
    if (S == 0) return;
    const long long total = logits.size();
    if (total > 0x7fffffffLL) fail(op, "logits too large");
    const MirKernel mk = mir_kernel({MirKind::Topk, 1u, logits.dtype});
    mir_dispatch_topk(mk, static_cast<std::uint32_t>(std::min(S, 65535)), 1u, op,
                      logits.data, out.data, nullptr, seg_offsets.data,
                      0u, static_cast<std::uint32_t>(total), 0u,
                      static_cast<std::uint32_t>(S));
}

}  // namespace brotensor::detail::metal
