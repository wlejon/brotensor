// Vulkan rotary position embedding: rope_forward / rope_backward (angles from
// theta_base), rope_apply / rope_apply_backward (row tables),
// rope_apply_perhead, rope_qkv_packed_inplace (Q and K of a fused QKV row,
// tables indexed by a per-row position) and rope_apply_mrope (Qwen-VL's three
// position streams). All in shaders/rope.comp, interleaved-pair convention,
// FP32 / FP16 / BF16 storage with FP32 math, FP32 tables. Contracts follow
// src/cuda/rope.cu (output resized to X's shape and dtype).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct RopePush {
    std::uint64_t x, y, ct, st, ch, sh, cw, sw, pt, ph, pw;
    std::uint32_t rows, heads, hd, dt, dh, dw;
    std::int32_t offset;
    float ln_base;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void check_heads(const Tensor& X, int head_dim, int num_heads, const char* op, int sections = 1) {
    dt_code(X.dtype, op);
    if (head_dim <= 0 || (head_dim & 1) != 0) fail(op, "head_dim must be a positive even integer");
    if (num_heads <= 0) fail(op, "num_heads must be positive");
    if (X.cols != sections * num_heads * head_dim) {
        fail(op, sections == 1 ? "X.cols != num_heads * head_dim" : "QKV.cols != 3 * num_heads * head_dim");
    }
}

void check_table(const Tensor& t, long long n, const char* op) {
    if (t.dtype != Dtype::FP32) fail(op, "cos / sin tables must be FP32");
    if (t.size() != n) fail(op, "cos / sin tables must have " + std::to_string(n) + " elements");
}

void ensure_like(const Tensor& X, Tensor& Y) {
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) Y.resize(X.rows, X.cols, X.dtype);
}

void run(const Tensor& X, std::uint32_t mode, bool bwd, RopePush pc, int head_dim, int num_heads) {
    pc.rows = static_cast<std::uint32_t>(X.rows);
    pc.heads = static_cast<std::uint32_t>(num_heads);
    pc.hd = static_cast<std::uint32_t>(head_dim);
    const long long total = static_cast<long long>(X.rows) * num_heads * (head_dim / 2);
    if (total == 0) return;
    if (total > 0x7fffffffLL) fail("rope", "tensor too large");
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::rope_f32, X.dtype), {mode, bwd ? 1u : 0u});
    launch(d, k, pc, groups_1d(static_cast<std::uint64_t>(total), k));
}

void rope_theta(const char* op, const Tensor& X, int head_dim, int num_heads, int seq_offset, float theta_base,
                Tensor& Y, bool bwd) {
    check_heads(X, head_dim, num_heads, op);
    ensure_like(X, Y);
    RopePush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.offset = seq_offset;
    pc.ln_base = std::log(theta_base);   // logf, as the CPU reference computes it
    run(X, ROPE_THETA, bwd, pc, head_dim, num_heads);
}

void rope_table(const char* op, const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl, int head_dim,
                int num_heads, Tensor& Y, std::uint32_t mode, bool bwd) {
    check_heads(X, head_dim, num_heads, op);
    const long long rows = mode == ROPE_PERHEAD ? static_cast<long long>(X.rows) * num_heads : X.rows;
    check_table(cos_tbl, rows * (head_dim / 2), op);
    check_table(sin_tbl, rows * (head_dim / 2), op);
    ensure_like(X, Y);
    RopePush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.ct = addr(cos_tbl.data);
    pc.st = addr(sin_tbl.data);
    run(X, mode, bwd, pc, head_dim, num_heads);
}

}  // namespace

void rope_forward(const Tensor& X, int head_dim, int num_heads, int seq_offset, float theta_base, Tensor& Y) {
    rope_theta("rope_forward", X, head_dim, num_heads, seq_offset, theta_base, Y, false);
}

void rope_backward(const Tensor& dY, int head_dim, int num_heads, int seq_offset, float theta_base, Tensor& dX) {
    rope_theta("rope_backward", dY, head_dim, num_heads, seq_offset, theta_base, dX, true);
}

void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl, int head_dim, int num_heads,
                Tensor& Y) {
    rope_table("rope_apply", X, cos_tbl, sin_tbl, head_dim, num_heads, Y, ROPE_TABLE, false);
}

void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl, const Tensor& sin_tbl, int head_dim,
                         int num_heads, Tensor& dX) {
    rope_table("rope_apply_backward", dY, cos_tbl, sin_tbl, head_dim, num_heads, dX, ROPE_TABLE, true);
}

void rope_apply_perhead(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl, int head_dim,
                        int num_heads, Tensor& Y) {
    rope_table("rope_apply_perhead", X, cos_tbl, sin_tbl, head_dim, num_heads, Y, ROPE_PERHEAD, false);
}

void rope_qkv_packed_inplace(Tensor& QKV, const Tensor& cos_tbl, const Tensor& sin_tbl, const Tensor& pos,
                             int num_heads, int head_dim) {
    constexpr const char* op = "rope_qkv_packed_inplace";
    check_heads(QKV, head_dim, num_heads, op, 3);
    const int half = head_dim / 2;
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) fail(op, "cos_tbl / sin_tbl must be FP32");
    if (cos_tbl.cols != half || sin_tbl.cols != half || sin_tbl.rows != cos_tbl.rows) {
        fail(op, "cos_tbl / sin_tbl must be (P, head_dim/2)");
    }
    if (pos.dtype != Dtype::INT32 || pos.size() != QKV.rows) fail(op, "pos must be (L, 1) INT32");
    RopePush pc{};
    pc.x = addr(QKV.data);
    pc.ct = addr(cos_tbl.data);
    pc.st = addr(sin_tbl.data);
    pc.pt = addr(pos.data);
    pc.dh = static_cast<std::uint32_t>(cos_tbl.rows);   // positions >= P leave the row untouched
    run(QKV, ROPE_PACKED, false, pc, head_dim, num_heads);
}

void rope_apply_mrope(const Tensor& X, const Tensor& cos_t, const Tensor& sin_t, const Tensor& cos_h,
                      const Tensor& sin_h, const Tensor& cos_w, const Tensor& sin_w, const int32_t* pos_t,
                      const int32_t* pos_h, const int32_t* pos_w, int head_dim, int num_heads, int d_t, int d_h,
                      int d_w, Tensor& Y) {
    constexpr const char* op = "rope_apply_mrope";
    check_heads(X, head_dim, num_heads, op);
    if (d_t < 0 || d_h < 0 || d_w < 0 || 2 * (d_t + d_h + d_w) != head_dim) fail(op, "2*(d_t + d_h + d_w) != head_dim");
    auto tables = [&](const Tensor& c, const Tensor& s, int da, const int32_t* p, const char* axis) {
        if (da == 0) return;
        if (c.dtype != Dtype::FP32 || s.dtype != Dtype::FP32) fail(op, std::string("cos_") + axis + " / sin_" + axis + " must be FP32");
        if (c.cols != da || s.cols != da) fail(op, std::string("cos_") + axis + " / sin_" + axis + " must have d_" + axis + " columns");
        if (!p) fail(op, std::string("pos_") + axis + " is null");
    };
    tables(cos_t, sin_t, d_t, pos_t, "t");
    tables(cos_h, sin_h, d_h, pos_h, "h");
    tables(cos_w, sin_w, d_w, pos_w, "w");
    ensure_like(X, Y);
    RopePush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.ct = d_t ? addr(cos_t.data) : 0; pc.st = d_t ? addr(sin_t.data) : 0; pc.pt = addr(pos_t);
    pc.ch = d_h ? addr(cos_h.data) : 0; pc.sh = d_h ? addr(sin_h.data) : 0; pc.ph = addr(pos_h);
    pc.cw = d_w ? addr(cos_w.data) : 0; pc.sw = d_w ? addr(sin_w.data) : 0; pc.pw = addr(pos_w);
    pc.dt = static_cast<std::uint32_t>(d_t);
    pc.dh = static_cast<std::uint32_t>(d_h);
    pc.dw = static_cast<std::uint32_t>(d_w);
    run(X, ROPE_MROPE, false, pc, head_dim, num_heads);
}

void fill_vulkan_vtable_rope(::brotensor::detail::OpsVTable& v) {
    v.rope_forward = &rope_forward;
    v.rope_backward = &rope_backward;
    v.rope_apply = &rope_apply;
    v.rope_apply_backward = &rope_apply_backward;
    v.rope_apply_perhead = &rope_apply_perhead;
    v.rope_qkv_packed_inplace = &rope_qkv_packed_inplace;
    v.rope_apply_mrope = &rope_apply_mrope;
}

}  // namespace brotensor::detail::vulkan
