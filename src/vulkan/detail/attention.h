#pragma once

// The attention dispatcher every Vulkan attention op goes through
// (ops_attention.cpp), and the dense path (ops_attention_dense.cpp). One
// AttnProblem describes scaled dot-product attention of lq query rows over
// lk key rows with hq query heads and hkv KV heads (GQA), each row's keys
// chosen by a mode (fa_common.glsl):
//   FA_MODE_ROWS    q_offset + causal / window, keys [0, lk), optional mask
//   FA_MODE_VARLEN  cu_seqlens_q / cu_seqlens_k (device INT32), causal per sequence
//   FA_MODE_PACKED  per-row seq_bounds (device INT32, lq x 2), window
// attention() picks
//   * fa_cm.comp    FP16 / BF16, cooperative matrix, more than a few query rows;
//   * fa_rows.comp  one query row per workgroup with split keys: decode, FP32,
//                   no cooperative matrix, soft-capping, masked-decode windows;
//   * the dense path  S = Q K^T (FP32) -> softmax -> P V through gemm(), for
//                   unmasked-window, non-causal FP32 (or very wide heads).

#include "device.h"

#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>

namespace brotensor::detail::vulkan {

struct AttnProblem {
    ::brotensor::Dtype dt = ::brotensor::Dtype::FP16;   // Q, K, V and O
    std::uint64_t q = 0, k = 0, v = 0, o = 0;           // device addresses (head 0, row 0)
    std::uint64_t mask = 0;                              // FP32 key mask (lk), 0 = none
    std::uint64_t aux = 0, aux2 = 0;                     // VARLEN / PACKED tables
    int lq = 0, lk = 0;
    int ldq = 0, ldk = 0, ldo = 0;                       // row strides in elements (V shares ldk)
    int hd = 0, hq = 0, hkv = 0;
    int mode = 0;                                        // FA_MODE_*
    bool causal = false;
    int window = 0;
    int q_offset = 0;
    int nseq = 0;                                        // VARLEN
    float softcap = 0.0f;                                // > 0: tanh soft-capping (fa_rows only)
    bool maskpos = false;                                // masked decode with a window
    bool nan_safe = false;                               // masked V rows may hold NaN: fa_rows only
    const char* op = "attention";
};

// Records the attention on `d`'s stream.
void attention(DeviceCtx& d, const AttnProblem& p);

// Which path attention() would take: "coopmat", "rows" or "dense".
const char* attention_path(DeviceCtx& d, const AttnProblem& p);

// Test / benchmark hooks, process-wide. Override: 0 = automatic, 1 = rows,
// 2 = coopmat when eligible, 3 = dense when eligible. Config: force the
// fa_cm tile (bc, nsg), (0, 0) = automatic.
void set_attention_override(int mode);
void set_attention_cm_config(int bc, int nsg);

// The dense (materialised) path, also the engine of the projection-fused
// ops: per head, S = scale * Q K^T (FP32) [+ bias], P = softmax over the
// valid keys (rows with a masked query or no valid key are zero), O = P V.
// Keys only by `mask` (no causal / window / modes). `probs`, when set,
// receives P as FP32 (hq * lq, lk) and serves as the PV operand (FP32 runs
// only). Score memory is bounded: heads, then query rows, go in passes.
struct DenseExtras {
    float scale = NAN;               // NaN = 1 / sqrt(hd)
    std::uint64_t bias = 0;          // FP32 (hq * lq, lk), added after scaling
    long long bias_hs = -1;          // the bias's head stride in elements (-1 = lq * lk, 0 = shared)
    std::uint64_t qmask = 0;         // FP32 (lq): rows with qmask < 0.5 are zero
    bool mask_ge = false;            // key valid when mask >= 0.5 (else > 0.5)
    std::uint64_t probs = 0;
};
void dense_attention(DeviceCtx& d, const AttnProblem& p, const DenseExtras& x = {});

}  // namespace brotensor::detail::vulkan
