// Backward of the packed, windowed bidirectional attention
// (flash_attention_packed_qkv_forward): the gradient of O w.r.t. the fused
// (L, 3*H*hd) QKV buffer, under the same per-row sequence bounds and window.
//
// Recompute-based, two launches, one warp per (row, head), FP32 math:
//
//   1. stats: each query row re-runs its online softmax over the keys it
//      attends, keeping the row max m_r and normaliser l_r, and the output
//      O_r it produced, then D_r = dO_r . O_r (= sum_j P_rj dP_rj).
//   2. grads: each row plays both roles.
//        as a query:  dQ_r = scale * sum_j dS_rj k_j
//        as a key:    dK_j = scale * sum_r dS_rj q_r,  dV_j = sum_r P_rj dO_r
//      with P_rj = exp(s_rj - m_r) / l_r and dS_rj = P_rj (dO_r . v_j - D_r).
//      The window is symmetric (|r - j| <= window/2 inside one sequence), so
//      the queries attending key j are found by the same bounds as the keys
//      row j attends.
//
// No P / dS matrix is stored and nothing is accumulated atomically, so the
// result is deterministic and the scratch is 3 floats per (row, head). Each
// lane holds head_dim/32 (rounded up) dimensions of every vector; dot
// products reduce across the warp.
//
// Built for the short sequences of a packed encoder batch (tens to a few
// hundred rows); cost is O(L * span * hd) with span the keys per row.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include "ops_fa_mir.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr const char* kOp = "flash_attention_packed_qkv_backward";

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + kOp + ": " + reason);
}

}  // namespace

void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV) {
    const Dtype dt = QKV.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail("QKV must be FP16, BF16 or FP32");
    if (dO.dtype != dt) fail("dO must share QKV's dtype");
    if (seq_bounds.dtype != Dtype::INT32) fail("seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail("QKV.cols must be 3 * num_heads * head_dim");
    const int L = QKV.rows;
    const int D = QKV.cols / 3;
    const int hd = D / num_heads;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail("seq_bounds must be (L, 2)");
    if (dO.rows != L || dO.cols != D) fail("dO must be (L, num_heads*head_dim)");
    if (dQKV.rows != L || dQKV.cols != 3 * D || dQKV.dtype != dt) dQKV.resize(L, 3 * D, dt);
    if (L == 0) return;

    const std::size_t esz = (dt == Dtype::FP32) ? 4u : 2u;
    const char* qkv_b = static_cast<const char*>(QKV.data);
    char* dqkv_b = static_cast<char*>(dQKV.data);
    dispatch_fa_bwd(kFaModePacked, dt, QKV.device,
                    qkv_b, qkv_b + static_cast<std::size_t>(D) * esz,
                    qkv_b + static_cast<std::size_t>(2 * D) * esz,
                    dO.data, nullptr, static_cast<const std::int32_t*>(seq_bounds.data), nullptr,
                    dqkv_b, dqkv_b + static_cast<std::size_t>(D) * esz,
                    dqkv_b + static_cast<std::size_t>(2 * D) * esz,
                    L, L, 3 * D, 3 * D, D, 3 * D,
                    num_heads, hd, window > 0 ? window : 0, false, 0);
}

}  // namespace brotensor::detail::cuda
