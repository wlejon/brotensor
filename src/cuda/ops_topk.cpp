// CUDA per-row exact top-k selection and packed-encoder segment softmax stats
// backed by Brass MIR -> PTX (MirKind::Topk).

#include "cu_mir_cache.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in flash_attention_packed.cu / flash_attention_packed_backward.cu / ops_rope.cpp.
void flash_attention_packed_qkv_forward(const Tensor& QKV, const Tensor& seq_bounds,
                                        int num_heads, int window, Tensor& O);
void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV);
void rope_qkv_packed_inplace(Tensor& QKV, const Tensor& cos_tbl, const Tensor& sin_tbl,
                             const Tensor& pos, int num_heads, int head_dim);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

}  // namespace

void top_k_rows(const Tensor& X, int k, Tensor& Vals, Tensor& Idx) {
    constexpr const char* op = "top_k_rows";
    require_fp(X.dtype, op);
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");
    if (Vals.rows != R || Vals.cols != k || Vals.dtype != Dtype::FP32) Vals.resize(R, k, Dtype::FP32);
    if (Idx.rows != R || Idx.cols != k || Idx.dtype != Dtype::INT32) Idx.resize(R, k, Dtype::INT32);
    if (R == 0) return;

    const std::uint64_t groups = (static_cast<std::uint64_t>(C) + 255) / 256;
    if (groups > 65535) fail(op, "rows longer than 16M elements");
    CUfunction fn = mir_function({MirKind::Topk, 0u, X.dtype});
    const std::size_t es = static_cast<std::size_t>(::brotensor::dtype_size_bytes(X.dtype));
    for (int r0 = 0; r0 < R; r0 += 65535) {
        const int n = std::min(65535, R - r0);
        mir_dispatch_topk(
            fn, static_cast<std::uint32_t>(groups), static_cast<std::uint32_t>(n),
            static_cast<const char*>(X.data) + static_cast<std::size_t>(r0) * C * es,
            static_cast<float*>(Vals.data) + static_cast<std::size_t>(r0) * k,
            static_cast<std::int32_t*>(Idx.data) + static_cast<std::size_t>(r0) * k,
            nullptr, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(C),
            static_cast<std::uint32_t>(k), 0);
    }
}

void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets, Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    require_fp(logits.dtype, op);
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) {
        fail(op, "seg_offsets must be (S+1, 1) INT32");
    }
    const int S = seg_offsets.rows - 1;
    if (out.rows != S || out.cols != 4 || out.dtype != logits.dtype) out.resize(S, 4, logits.dtype);
    if (S == 0) return;
    const long long total = logits.size();
    if (total > 0x7fffffffLL) fail(op, "logits too large");
    CUfunction fn = mir_function({MirKind::Topk, 1u, logits.dtype});
    mir_dispatch_topk(
        fn, static_cast<std::uint32_t>(std::min(S, 65535)), 1u,
        logits.data, out.data, nullptr, seg_offsets.data,
        0, static_cast<std::uint32_t>(total), 0, static_cast<std::uint32_t>(S));
}

void fill_cuda_vtable_top_k(::brotensor::detail::OpsVTable& v) {
    v.top_k_rows = &top_k_rows;
}

void fill_cuda_vtable_packed_encoder(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_packed_qkv_forward = &flash_attention_packed_qkv_forward;
    v.flash_attention_packed_qkv_backward = &flash_attention_packed_qkv_backward;
    v.rope_qkv_packed_inplace = &rope_qkv_packed_inplace;
    v.segment_softmax_stats = &segment_softmax_stats;
}

}  // namespace brotensor::detail::cuda
