// CUDA rotary position embeddings (theta, table, per-head, packed QKV, M-RoPE) backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"
#include "../kernels/mir_rope.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct RopeLaunch {
    const void* x = nullptr;
    const void* y = nullptr;
    const void* ct = nullptr;
    const void* st = nullptr;
    const void* ch = nullptr;
    const void* sh = nullptr;
    const void* cw = nullptr;
    const void* sw = nullptr;
    const void* pt = nullptr;
    const void* ph = nullptr;
    const void* pw = nullptr;
    std::uint32_t dt = 0;
    std::uint32_t dh = 0;
    std::uint32_t dw = 0;
    std::int32_t offset = 0;
    float ln_base = 0.0f;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string(op) + ": " + why);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "unsupported dtype (expected FP32/FP16/BF16)");
    }
}

void check_heads(const Tensor& X, int head_dim, int num_heads, const char* op, int sections = 1) {
    require_fp(X.dtype, op);
    if (head_dim <= 0 || (head_dim & 1) != 0) fail(op, "head_dim must be a positive even integer");
    if (num_heads <= 0) fail(op, "num_heads must be positive");
    if (X.cols != sections * num_heads * head_dim) {
        fail(op, sections == 1 ? "X.cols != num_heads * head_dim" : "QKV.cols != 3 * num_heads * head_dim");
    }
}

void check_table(const Tensor& t, long long n, const char* op) {
    if (t.dtype != Dtype::FP32) fail(op, "cos / sin tables must be FP32");
    if (t.size() != n) fail(op, "cos / sin table element count mismatch");
}

void ensure_like(const Tensor& X, Tensor& Y) {
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
}

void run_rope(const Tensor& X, std::uint32_t mode, bool bwd, const RopeLaunch& rl,
              int head_dim, int num_heads) {
    const long long total = static_cast<long long>(X.rows) * num_heads * (head_dim / 2);
    if (total == 0) return;
    if (total > 0x7fffffffLL) fail("rope", "tensor too large");
    CUfunction fn = mir_function(
        {MirKind::Rope, mode, X.dtype, X.dtype, 1, Dtype::FP32, static_cast<std::uint8_t>(bwd ? 1u : 0u)});
    mir_dispatch_rope(
        fn, mir_groups_1d(static_cast<std::uint32_t>(total), 1, kernels::kRopeBlock),
        rl.x, rl.y, rl.ct, rl.st, rl.ch, rl.sh, rl.cw, rl.sw, rl.pt, rl.ph, rl.pw,
        static_cast<std::uint32_t>(X.rows), static_cast<std::uint32_t>(num_heads),
        static_cast<std::uint32_t>(head_dim), rl.dt, rl.dh, rl.dw, rl.offset, rl.ln_base);
}

void rope_theta(const char* op, const Tensor& X, int head_dim, int num_heads,
                int seq_offset, float theta_base, Tensor& Y, bool bwd) {
    check_heads(X, head_dim, num_heads, op);
    ensure_like(X, Y);
    RopeLaunch rl{};
    rl.x = X.data;
    rl.y = Y.data;
    rl.offset = seq_offset;
    rl.ln_base = std::log(theta_base);
    run_rope(X, ROPE_THETA, bwd, rl, head_dim, num_heads);
}

void rope_table(const char* op, const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y, std::uint32_t mode, bool bwd) {
    check_heads(X, head_dim, num_heads, op);
    const long long rows = (mode == ROPE_PERHEAD) ? static_cast<long long>(X.rows) * num_heads : X.rows;
    check_table(cos_tbl, rows * (head_dim / 2), op);
    check_table(sin_tbl, rows * (head_dim / 2), op);
    ensure_like(X, Y);
    RopeLaunch rl{};
    rl.x = X.data;
    rl.y = Y.data;
    rl.ct = cos_tbl.data;
    rl.st = sin_tbl.data;
    run_rope(X, mode, bwd, rl, head_dim, num_heads);
}

}  // namespace

void rope_forward(const Tensor& X, int head_dim, int num_heads, int seq_offset,
                  float theta_base, Tensor& Y) {
    rope_theta("rope_forward", X, head_dim, num_heads, seq_offset, theta_base, Y, false);
}

void rope_backward(const Tensor& dY, int head_dim, int num_heads, int seq_offset,
                   float theta_base, Tensor& dX) {
    rope_theta("rope_backward", dY, head_dim, num_heads, seq_offset, theta_base, dX, true);
}

void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y) {
    rope_table("rope_apply", X, cos_tbl, sin_tbl, head_dim, num_heads, Y, ROPE_TABLE, false);
}

void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl, const Tensor& sin_tbl,
                         int head_dim, int num_heads, Tensor& dX) {
    rope_table("rope_apply_backward", dY, cos_tbl, sin_tbl, head_dim, num_heads, dX, ROPE_TABLE, true);
}

void rope_apply_perhead(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                        int head_dim, int num_heads, Tensor& Y) {
    rope_table("rope_apply_perhead", X, cos_tbl, sin_tbl, head_dim, num_heads, Y, ROPE_PERHEAD, false);
}

void rope_qkv_packed_inplace(Tensor& QKV, const Tensor& cos_tbl, const Tensor& sin_tbl,
                             const Tensor& pos, int num_heads, int head_dim) {
    constexpr const char* op = "rope_qkv_packed_inplace";
    check_heads(QKV, head_dim, num_heads, op, 3);
    const int half = head_dim / 2;
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) fail(op, "cos/sin tables must be FP32");
    if (cos_tbl.cols != half || sin_tbl.cols != half || sin_tbl.rows != cos_tbl.rows) {
        fail(op, "cos_tbl / sin_tbl must be (P, head_dim/2)");
    }
    if (pos.dtype != Dtype::INT32 || pos.size() != QKV.rows) fail(op, "pos must be (L, 1) INT32");
    RopeLaunch rl{};
    rl.x = QKV.data;
    rl.ct = cos_tbl.data;
    rl.st = sin_tbl.data;
    rl.pt = pos.data;
    rl.dh = static_cast<std::uint32_t>(cos_tbl.rows);
    run_rope(QKV, ROPE_PACKED, false, rl, head_dim, num_heads);
}

void rope_apply_mrope(const Tensor& X, const Tensor& cos_t, const Tensor& sin_t,
                      const Tensor& cos_h, const Tensor& sin_h,
                      const Tensor& cos_w, const Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,
                      int head_dim, int num_heads, int d_t, int d_h, int d_w, Tensor& Y) {
    constexpr const char* op = "rope_apply_mrope";
    check_heads(X, head_dim, num_heads, op);
    if (d_t < 0 || d_h < 0 || d_w < 0 || 2 * (d_t + d_h + d_w) != head_dim) {
        fail(op, "2*(d_t + d_h + d_w) must equal head_dim");
    }
    auto check_axis = [&](const Tensor& c, const Tensor& s, int da, const int32_t* p, const char* axis) {
        if (da == 0) return;
        if (c.dtype != Dtype::FP32 || s.dtype != Dtype::FP32) {
            fail(op, std::string("cos_") + axis + " / sin_" + axis + " must be FP32");
        }
        if (c.cols != da || s.cols != da) {
            fail(op, std::string("cos_") + axis + " / sin_" + axis + " must have d_" + axis + " columns");
        }
        if (c.rows != s.rows) {
            fail(op, std::string("cos_") + axis + " / sin_" + axis + " row count mismatch");
        }
        if (!p) fail(op, std::string("pos_") + axis + " is null");
    };
    check_axis(cos_t, sin_t, d_t, pos_t, "t");
    check_axis(cos_h, sin_h, d_h, pos_h, "h");
    check_axis(cos_w, sin_w, d_w, pos_w, "w");
    ensure_like(X, Y);
    RopeLaunch rl{};
    rl.x = X.data;
    rl.y = Y.data;
    rl.ct = d_t ? cos_t.data : nullptr; rl.st = d_t ? sin_t.data : nullptr; rl.pt = pos_t;
    rl.ch = d_h ? cos_h.data : nullptr; rl.sh = d_h ? sin_h.data : nullptr; rl.ph = pos_h;
    rl.cw = d_w ? cos_w.data : nullptr; rl.sw = d_w ? sin_w.data : nullptr; rl.pw = pos_w;
    rl.dt = static_cast<std::uint32_t>(d_t);
    rl.dh = static_cast<std::uint32_t>(d_h);
    rl.dw = static_cast<std::uint32_t>(d_w);
    run_rope(X, ROPE_MROPE, false, rl, head_dim, num_heads);
}

// ─── vtable registration (Qwen3-VL polish) ─────────────────────────────────

void spatial_merge_2x2_forward(const Tensor& X, int N, int C, int H, int W,
                               bool channel_major, Tensor& Y);
void pixel_shuffle_upsample_2x_forward(const Tensor& X, int N, int C_in, int H, int W,
                                       int C_out, Tensor& Y);
void patch_unpack_forward(const Tensor& tokens, int hp, int wp, int P, int C_total, int C_keep,
                          bool channel_major, Tensor& Y);

void fill_cuda_vtable_qwen3_vl_polish(::brotensor::detail::OpsVTable& v) {
    v.spatial_merge_2x2_forward         = &spatial_merge_2x2_forward;
    v.pixel_shuffle_upsample_2x_forward = &pixel_shuffle_upsample_2x_forward;
    v.patch_unpack_forward              = &patch_unpack_forward;
    v.rope_apply_mrope                  = &rope_apply_mrope;
}

}  // namespace brotensor::detail::cuda
