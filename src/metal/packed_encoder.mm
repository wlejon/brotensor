// Metal packed-encoder flash attention (forward and backward) — routed onto
// single-source brass MIR (MirKind::FaRows and MirKind::FaBwd, FA_MODE_PACKED).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/tensor.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void check_dt(Dtype dt, const char* op) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        fail(op, "tensor must be FP32, FP16 or BF16");
}

struct PackedShape {
    int L, D, hd;
};

PackedShape check_packed(const char* op, const Tensor& QKV, const Tensor& seq_bounds, int num_heads) {
    check_dt(QKV.dtype, op);
    if (seq_bounds.dtype != Dtype::INT32) fail(op, "seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0)
        fail(op, "QKV.cols must be 3 * num_heads * head_dim");
    PackedShape s{};
    s.L = QKV.rows;
    s.D = QKV.cols / 3;
    s.hd = s.D / num_heads;
    if (seq_bounds.rows != s.L || seq_bounds.cols != 2) fail(op, "seq_bounds must be (L, 2)");
    return s;
}

}  // namespace

void flash_attention_packed_qkv_forward(const Tensor& QKV, const Tensor& seq_bounds,
                                        int num_heads, int window, Tensor& O) {
    constexpr const char* op = "flash_attention_packed_qkv_forward";
    const PackedShape s = check_packed(op, QKV, seq_bounds, num_heads);
    const Dtype dt = QKV.dtype;
    if (O.rows != s.L || O.cols != s.D || O.dtype != dt) O.resize(s.L, s.D, dt);
    if (s.L == 0 || s.D == 0) return;

    const std::size_t es = dtype_size_bytes(dt);
    const auto* base = static_cast<const char*>(QKV.data);
    FaRowsProblem p;
    p.op = op;
    p.dt = dt;
    p.q = base;
    p.k = base + static_cast<std::size_t>(s.D) * es;
    p.v = base + static_cast<std::size_t>(2) * s.D * es;
    p.o = O.data;
    p.aux = seq_bounds.data;
    p.lq = p.lk = s.L;
    p.ldq = p.ldk = 3 * s.D;
    p.ldo = s.D;
    p.hq = p.hkv = num_heads;
    p.hd = s.hd;
    p.mode = FA_MODE_PACKED;
    p.window = window;
    fa_rows_metal(p);
}

void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO,
                                         const Tensor& seq_bounds, int num_heads,
                                         int window, Tensor& dQKV) {
    constexpr const char* op = "flash_attention_packed_qkv_backward";
    const PackedShape s = check_packed(op, QKV, seq_bounds, num_heads);
    const Dtype dt = QKV.dtype;
    if (dO.dtype != dt) fail(op, "dO must share QKV's dtype");
    if (dO.rows != s.L || dO.cols != s.D) fail(op, "dO must be (L, num_heads*head_dim)");
    if (dQKV.rows != s.L || dQKV.cols != 3 * s.D || dQKV.dtype != dt) dQKV.resize(s.L, 3 * s.D, dt);
    if (s.L == 0 || s.D == 0) return;

    const std::size_t es = dtype_size_bytes(dt);
    const auto* base = static_cast<const char*>(QKV.data);
    auto* gbase = static_cast<char*>(dQKV.data);
    FaBwdProblem p;
    p.op = op;
    p.dt = dt;
    p.q = base;
    p.k = base + static_cast<std::size_t>(s.D) * es;
    p.v = base + static_cast<std::size_t>(2) * s.D * es;
    p.g = dO.data;
    p.dq = gbase;
    p.dk = gbase + static_cast<std::size_t>(s.D) * es;
    p.dv = gbase + static_cast<std::size_t>(2) * s.D * es;
    p.aux = seq_bounds.data;
    p.mode = FA_MODE_PACKED;
    p.window = window;
    p.lq = p.lk = s.L;
    p.ldq = p.ldk = p.ldq_out = p.ldk_out = 3 * s.D;
    p.ldg = s.D;
    p.heads = num_heads;
    p.hd = s.hd;
    fa_bwd_metal(p);
}

}  // namespace brotensor::detail::metal
