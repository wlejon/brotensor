// Vulkan flash-attention backwards: flash_attention_backward,
// flash_attention_varlen_backward, flash_attention_packed_qkv_backward and
// the projection-fused flash_attention_qkvo_backward. One kernel pair
// (shaders/fa_bwd.comp, docs/vulkan-training.md) does the attention
// core of all four: recompute-based (O is not read), FP32 scores, softmax
// statistics, dP and dS, no atomics.
//
// Contracts follow the CUDA / HIP backend (src/cuda/flash_attention_backward.cu,
// flash_attention_packed_backward.cu, flash_attention.cu), widened to every
// float dtype: Q, K, V and dO share one dtype (FP32 / FP16 / BF16; the HIP
// bare backward takes 16-bit only), dQ / dK / dV are resized to it and
// OVERWRITTEN, scale 1 / sqrt(head_dim), a key mask is valid where > 0.5,
// causal needs Lq == Lk. The device-resident cu_seqlens / seq_bounds tables
// are not read back (as the Vulkan forwards): they are clamped to [0, lk] in
// the kernel, so a malformed table gives wrong rows, never a fault.
// flash_attention_qkvo_backward composes the projections (dX / dCtx
// overwritten, dW* / db* accumulated) around the core.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const float* d_mask, int num_heads,
                             bool causal, Tensor& O);                                    // ops_attention.cpp
void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X, int act, int epilogue,
                               Tensor* workspace, Tensor& Y);                            // ops_linear.cpp
void linear_backward_batched(const Tensor& W, const Tensor& X, const Tensor& dY, Tensor& dX, Tensor& dW,
                             Tensor& dB);                                                // ops_linear.cpp
void add_inplace(Tensor& a, const Tensor& b);                                            // ops_elementwise.cpp

namespace {

struct BwdPush {
    std::uint64_t q, k, v, g, mask, aux, aux2, stats, o0, o1;
    std::uint32_t lq, lk, ldq, ldk, ldg, ldo, window, causal, nseq;
    float scale, oscale;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

std::uint32_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }

void ensure_out(Tensor& t, int rows, int cols, Dtype dt, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

void zero_if(Tensor& t) {
    if (t.size() > 0) t.zero();
}

// The attention-core backward over device addresses: lq query rows, lk key
// rows, `heads` heads of width hd at column h * hd of each operand.
struct BwdProblem {
    const char* op = "flash_attention_backward";
    Dtype dt = Dtype::FP16;
    std::uint64_t q = 0, k = 0, v = 0, g = 0;   // Q, K, V, dO
    std::uint64_t dq = 0, dk = 0, dv = 0;
    std::uint64_t mask = 0, aux = 0, aux2 = 0;
    int lq = 0, lk = 0, ldq = 0, ldk = 0, ldg = 0, ldq_out = 0, ldk_out = 0;
    int hd = 0, heads = 0, mode = FA_MODE_ROWS, window = 0, nseq = 0;
    bool causal = false;
};

struct Cfg { std::uint32_t nt, br, bc; };

// Tiles that keep Q, dO (br rows), K, V (bc rows), P and dS under the 64 KiB
// of shared memory (the pipeline guard checks): the inputs are staged in
// their own encoding, so 16-bit tiles take half the room of FP32 ones.
Cfg pick(Dtype dt, int hdp) {
    if (dt != Dtype::FP32) return hdp <= 128 ? Cfg{128, 32, 32} : Cfg{128, 32, 16};
    if (hdp <= 64) return {128, 32, 32};
    if (hdp <= 128) return {128, 32, 16};
    return {32, 16, 8};
}

constexpr int kMaxHd = 256;

void core(DeviceCtx& d, const BwdProblem& p) {
    if (p.lq == 0 || p.lk == 0 || p.heads == 0 || p.hd == 0) return;
    if (p.hd > kMaxHd) fail(p.op, "head_dim above 256 is not supported on Vulkan");
    if (p.heads > 65535) fail(p.op, "too many heads");
    const int hdp = (p.hd + 31) / 32 * 32;
    const Cfg c = pick(p.dt, hdp);
    const long long nstat = static_cast<long long>(p.heads) * p.lq * 4;
    if (nstat > 0x7fffffffLL) fail(p.op, "problem too large");
    Tensor stats = Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(nstat), 1, Dtype::FP32);
    const ShaderId id = dt_variant(ShaderId::fa_bwd_f32, p.dt, p.op);
    auto kernel = [&](std::uint32_t pass) -> const Kernel& {
        return d.pipelines().get(id, {c.nt, static_cast<std::uint32_t>(p.hd), static_cast<std::uint32_t>(hdp), c.br,
                                      c.bc, static_cast<std::uint32_t>(p.mode), p.mask ? 1u : 0u, pass});
    };
    BwdPush pc{};
    pc.q = p.q; pc.k = p.k; pc.v = p.v; pc.g = p.g;
    pc.mask = p.mask; pc.aux = p.aux; pc.aux2 = p.aux2;
    pc.stats = addr(stats.data);
    pc.lq = static_cast<std::uint32_t>(p.lq);
    pc.lk = static_cast<std::uint32_t>(p.lk);
    pc.ldq = static_cast<std::uint32_t>(p.ldq);
    pc.ldk = static_cast<std::uint32_t>(p.ldk);
    pc.ldg = static_cast<std::uint32_t>(p.ldg);
    pc.window = static_cast<std::uint32_t>(std::max(0, p.window));
    pc.causal = p.causal ? 1u : 0u;
    pc.nseq = static_cast<std::uint32_t>(p.nseq);
    pc.oscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.hd)));
    pc.scale = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(p.hd)));
    const std::uint64_t nqb = (static_cast<std::uint64_t>(p.lq) + c.br - 1) / c.br;
    const std::uint64_t nkb = (static_cast<std::uint64_t>(p.lk) + c.bc - 1) / c.bc;
    if (nqb > 65535 || nkb > 65535) fail(p.op, "sequence too long for one dispatch");
    // Pass 0: statistics + dQ, query-major.
    pc.o0 = p.dq;
    pc.ldo = static_cast<std::uint32_t>(p.ldq_out);
    launch(d, kernel(0), pc, static_cast<std::uint32_t>(nqb), static_cast<std::uint32_t>(p.heads));
    // Pass 1: dK, dV, key-major.
    pc.o0 = p.dk;
    pc.o1 = p.dv;
    pc.ldo = static_cast<std::uint32_t>(p.ldk_out);
    launch(d, kernel(1), pc, static_cast<std::uint32_t>(nkb), static_cast<std::uint32_t>(p.heads));
}

void need_same(const char* op, Dtype dt, std::initializer_list<const Tensor*> ts) {
    dt_code(dt, op);
    for (const Tensor* t : ts)
        if (t->dtype != dt) fail(op, "Q, K, V and dO must share one dtype");
}

}  // namespace

void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& /*O*/,
                              const Tensor& dO, const float* d_mask, int num_heads, bool causal, Tensor& dQ,
                              Tensor& dK, Tensor& dV) {
    constexpr const char* op = "flash_attention_backward";
    const Dtype dt = Q.dtype;
    need_same(op, dt, {&K, &V, &dO});
    const int Lq = Q.rows, Lk = K.rows, D = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) fail(op, "Q/K/V shape mismatch");
    if (dO.rows != Lq || dO.cols != D) fail(op, "dO shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0) fail(op, "num_heads must divide D");
    if (causal && Lq != Lk) fail(op, "causal requires Lq == Lk");
    ensure_out(dQ, Lq, D, dt, Q.device);
    ensure_out(dK, Lk, D, dt, Q.device);
    ensure_out(dV, Lk, D, dt, Q.device);
    if (Lq == 0 || Lk == 0 || D == 0) {
        zero_if(dQ);
        zero_if(dK);
        zero_if(dV);
        return;
    }
    BwdProblem p;
    p.op = op;
    p.dt = dt;
    p.q = addr(Q.data); p.k = addr(K.data); p.v = addr(V.data); p.g = addr(dO.data);
    p.dq = addr(dQ.data); p.dk = addr(dK.data); p.dv = addr(dV.data);
    p.mask = addr(d_mask);
    p.lq = Lq; p.lk = Lk;
    p.ldq = p.ldk = p.ldg = p.ldq_out = p.ldk_out = D;
    p.heads = num_heads;
    p.hd = D / num_heads;
    p.causal = causal;
    core(device_of(Q), p);
}

void flash_attention_varlen_backward(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& /*O*/,
                                     const Tensor& dO, const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads, int head_dim,
                                     bool causal, Tensor& dQ, Tensor& dK, Tensor& dV) {
    constexpr const char* op = "flash_attention_varlen_backward";
    const Dtype dt = Q.dtype;
    need_same(op, dt, {&K, &V, &dO});
    const int D = num_heads * head_dim;
    if (num_heads <= 0 || head_dim <= 0) fail(op, "num_heads/head_dim must be positive");
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != K.rows) fail(op, "shape mismatch");
    if (dO.rows != Q.rows || dO.cols != D) fail(op, "dO shape mismatch");
    if (batch_size < 0) fail(op, "batch_size must be non-negative");
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) fail(op, "cu_seqlens_q/k required when batch_size > 0");
    if (max_seqlen_q < 0 || max_seqlen_k < 0) fail(op, "max_seqlen_q/k must be non-negative");
    ensure_out(dQ, Q.rows, D, dt, Q.device);
    ensure_out(dK, K.rows, D, dt, Q.device);
    ensure_out(dV, K.rows, D, dt, Q.device);
    if (batch_size == 0 || Q.rows == 0 || K.rows == 0) {
        zero_if(dQ);
        zero_if(dK);
        zero_if(dV);
        return;
    }
    // Every row is written: rows outside any sequence (or of a sequence
    // without keys / queries) come out zero.
    BwdProblem p;
    p.op = op;
    p.dt = dt;
    p.q = addr(Q.data); p.k = addr(K.data); p.v = addr(V.data); p.g = addr(dO.data);
    p.dq = addr(dQ.data); p.dk = addr(dK.data); p.dv = addr(dV.data);
    p.aux = addr(cu_seqlens_q);
    p.aux2 = addr(cu_seqlens_k);
    p.nseq = batch_size;
    p.mode = FA_MODE_VARLEN;
    p.lq = Q.rows; p.lk = K.rows;
    p.ldq = p.ldk = p.ldg = p.ldq_out = p.ldk_out = D;
    p.heads = num_heads;
    p.hd = head_dim;
    p.causal = causal;
    core(device_of(Q), p);
}

void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV) {
    constexpr const char* op = "flash_attention_packed_qkv_backward";
    const Dtype dt = QKV.dtype;
    dt_code(dt, op);
    if (dO.dtype != dt) fail(op, "dO must share QKV's dtype");
    if (seq_bounds.dtype != Dtype::INT32) fail(op, "seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail(op, "QKV.cols must be 3 * num_heads * head_dim");
    const int L = QKV.rows, D = QKV.cols / 3;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail(op, "seq_bounds must be (L, 2)");
    if (dO.rows != L || dO.cols != D) fail(op, "dO must be (L, num_heads*head_dim)");
    ensure_out(dQKV, L, 3 * D, dt, QKV.device);
    if (L == 0 || D == 0) return;
    const std::uint64_t es = esize(dt);
    const std::uint64_t base = addr(QKV.data), gbase = addr(dQKV.data);
    BwdProblem p;
    p.op = op;
    p.dt = dt;
    p.q = base; p.k = base + D * es; p.v = base + 2ull * D * es; p.g = addr(dO.data);
    p.dq = gbase; p.dk = gbase + D * es; p.dv = gbase + 2ull * D * es;
    p.aux = addr(seq_bounds.data);
    p.mode = FA_MODE_PACKED;
    p.window = window;
    p.lq = p.lk = L;
    p.ldq = p.ldk = p.ldq_out = p.ldk_out = 3 * D;
    p.ldg = D;
    p.heads = num_heads;
    p.hd = D / num_heads;
    core(device_of(QKV), p);
}

void flash_attention_qkvo_backward(const Tensor& X, const Tensor* Ctx, const Tensor& Wq, const Tensor* bq,
                                   const Tensor& Wk, const Tensor* bk, const Tensor& Wv, const Tensor* bv,
                                   const Tensor& Wo, const Tensor* bo, const float* d_mask, int num_heads, bool causal,
                                   const Tensor& dO, Tensor& dX, Tensor* dCtx, Tensor& dWq, Tensor* dbq, Tensor& dWk,
                                   Tensor* dbk, Tensor& dWv, Tensor* dbv, Tensor& dWo, Tensor* dbo) {
    constexpr const char* op = "flash_attention_qkvo_backward";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    const bool self_attn = Ctx == nullptr;
    if (self_attn && dCtx) fail(op, "dCtx must be null when Ctx is null");
    if (!self_attn && !dCtx) fail(op, "dCtx must be non-null when Ctx is non-null");
    if (!bq != !dbq || !bk != !dbk || !bv != !dbv || !bo != !dbo) fail(op, "bias/grad-bias presence mismatch");
    for (const Tensor* t : {&dO, &Wq, &Wk, &Wv, &Wo})
        if (t->dtype != dt) fail(op, "all tensors must share dtype");
    if (Ctx && Ctx->dtype != dt) fail(op, "Ctx dtype must match X");
    const Tensor& kv = self_attn ? X : *Ctx;
    const int Lq = X.rows, D = X.cols, Lk = kv.rows, Dc = kv.cols;
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != Dc || Wv.rows != D || Wv.cols != Dc ||
        Wo.rows != D || Wo.cols != D) {
        fail(op, "shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) fail(op, "dO shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0) fail(op, "num_heads must divide D");
    if (causal && Lq != Lk) fail(op, "causal requires Lq == Lk");
    ensure_out(dX, Lq, D, dt, X.device);
    if (!self_attn) {
        ensure_out(*dCtx, Lk, Dc, dt, X.device);
        zero_if(*dCtx);
    }
    zero_if(dX);
    if (Lq == 0 || Lk == 0 || D == 0) return;
    const ::brotensor::Device dev = X.device;
    auto tmp = [&](int r, int c) { return Tensor::empty_on(dev, r, c, dt); };

    // 1-2. Recompute the projections and the attention output.
    Tensor Q = tmp(Lq, D), K = tmp(Lk, D), V = tmp(Lk, D), A = tmp(Lq, D);
    linear_forward_batched_ex(Wq, bq, X, 0, 0, nullptr, Q);
    linear_forward_batched_ex(Wk, bk, kv, 0, 0, nullptr, K);
    linear_forward_batched_ex(Wv, bv, kv, 0, 0, nullptr, V);
    flash_attention_forward(Q, K, V, d_mask, num_heads, causal, A);

    // 3. Wo: dA = dO Wo, dWo += dO^T A, dbo += colsum(dO).
    auto lin_back = [&](const Tensor& W, const Tensor& In, const Tensor& dOut, Tensor& dIn, Tensor& dW, Tensor* db) {
        Tensor scratch;
        if (!db) {
            scratch = Tensor::zeros_on(dev, W.rows, 1, dt);
            db = &scratch;
        }
        linear_backward_batched(W, In, dOut, dIn, dW, *db);
    };
    Tensor dA = tmp(Lq, D);
    lin_back(Wo, A, dO, dA, dWo, dbo);

    // 4. The attention core.
    Tensor dQ = tmp(Lq, D), dK = tmp(Lk, D), dV = tmp(Lk, D);
    flash_attention_backward(Q, K, V, A, dA, d_mask, num_heads, causal, dQ, dK, dV);

    // 5. Q / K / V projections; dX (and dCtx) collect the input gradients.
    Tensor gq = tmp(Lq, D), gk = tmp(Lk, Dc), gv = tmp(Lk, Dc);
    lin_back(Wq, X, dQ, gq, dWq, dbq);
    lin_back(Wk, kv, dK, gk, dWk, dbk);
    lin_back(Wv, kv, dV, gv, dWv, dbv);
    add_inplace(dX, gq);
    Tensor& dkv = self_attn ? dX : *dCtx;
    add_inplace(dkv, gk);
    add_inplace(dkv, gv);
}

void fill_vulkan_vtable_fa_bwd(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_backward = &vulkan::flash_attention_backward;
    v.flash_attention_varlen_backward = &vulkan::flash_attention_varlen_backward;
    v.flash_attention_packed_qkv_backward = &vulkan::flash_attention_packed_qkv_backward;
    v.flash_attention_qkvo_backward = &vulkan::flash_attention_qkvo_backward;
}

}  // namespace brotensor::detail::vulkan
