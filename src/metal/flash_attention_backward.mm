// Metal backend for flash_attention_backward and flash_attention_varlen_backward.
// Recompute-based bare-core attention backward, mirroring
// src/cuda/flash_attention_backward.cu.
//
// All of the math is FP32. A 16-bit call widens Q / K / V / dO to FP32 once,
// runs the FP32 core, and narrows dQ / dK / dV once on the way out: held in
// 16 bits, the scores, P, dP or dS put 5e-3 - 2e-2 (FP16) and 4e-2 - 2 (BF16)
// of the row max on every gradient, where exact intermediates leave only the
// output rounding (tests/test_vit_block_ops.cpp measures both).
//
// The core, per head, over one attention block (Lq queries x Lk keys) whose
// rows live in (L, D = heads * hd) tensors, each GEMM reading and writing the
// head's column slice in place (row stride D, no per-head copies):
//   1) P  = Q_h K_h^T                                   (Lq, Lk)
//   2) P  = softmax(scale * P, mask, causal)            in place
//   3) dP = dO_h V_h^T                                  (Lq, Lk)
//   4) dV_h = P^T dO_h                                  (Lk, hd)
//   5) dS = P * (dP - rowsum(P * dP)) * scale           in place over P
//   6) dQ_h = dS K_h                                    (Lq, hd)
//   7) dK_h = dS^T Q_h                                  (Lk, hd)
// The query axis is chunked to hold P + dP under 256 MB; dK / dV accumulate
// across chunks through the GEMM's accumulate epilogue.

#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

NSString* const kBwdSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// Threadgroup-wide max / sum; `red` holds one float per simdgroup.
inline float tg_max(float v, threadgroup float* red, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_max(lane < nsg ? red[lane] : -1e30f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}
inline float tg_sum(float v, threadgroup float* red, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_sum(lane < nsg ? red[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

// In-place row softmax over FP32 scores. Row r of the block is query q0 + r;
// key k is dropped when masked (mask[k] <= 0.5) or, causal, when k > q0 + r.
kernel void k_fab_softmax_f32(device float*       S        [[buffer(0)]],
                              device const float* mask     [[buffer(1)]],
                              constant uint&      Lk       [[buffer(2)]],
                              constant float&     scale    [[buffer(3)]],
                              constant uint&      has_mask [[buffer(4)]],
                              constant uint&      causal   [[buffer(5)]],
                              constant uint&      q0       [[buffer(6)]],
                              threadgroup float*  red      [[threadgroup(0)]],
                              uint row  [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint tg   [[threads_per_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    device float* s = S + (ulong)row * Lk;
    const uint q = q0 + row, nsg = (tg + 31) / 32;
    float mx = -1e30f;
    for (uint k = tid; k < Lk; k += tg) {
        const bool drop = (has_mask != 0u && mask[k] <= 0.5f) || (causal != 0u && k > q);
        if (!drop) mx = max(mx, s[k] * scale);
    }
    mx = tg_max(mx, red, sg, lane, nsg);
    const bool empty = mx <= -1e29f;
    float sum = 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        const bool drop = empty || (has_mask != 0u && mask[k] <= 0.5f) || (causal != 0u && k > q);
        const float e = drop ? 0.0f : exp(s[k] * scale - mx);
        s[k] = e;
        sum += e;
    }
    sum = tg_sum(sum, red, sg, lane, nsg);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint k = tid; k < Lk; k += tg) s[k] *= inv;
}

// dS = P * (dP - D_q) * scale in place over P, D_q = sum_k P[q,k] dP[q,k].
kernel void k_fab_dS_f32(device float*       P     [[buffer(0)]],
                         device const float* dP    [[buffer(1)]],
                         constant uint&      Lk    [[buffer(2)]],
                         constant float&     scale [[buffer(3)]],
                         threadgroup float*  red   [[threadgroup(0)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tg   [[threads_per_threadgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device float* p = P + (ulong)row * Lk;
    device const float* dp = dP + (ulong)row * Lk;
    float acc = 0.0f;
    for (uint k = tid; k < Lk; k += tg) acc += p[k] * dp[k];
    const float Dq = tg_sum(acc, red, sg, lane, (tg + 31) / 32);
    for (uint k = tid; k < Lk; k += tg) p[k] = p[k] * (dp[k] - Dq) * scale;
}
)msl";

id<MTLComputePipelineState> pso_softmax() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_softmax_f32"); });
    return pso;
}
id<MTLComputePipelineState> pso_dS() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dS_f32"); });
    return pso;
}

// A (buffer, byte offset) view of FP32 (L, D) rows.
struct Rows {
    id<MTLBuffer> buf;
    NSUInteger ofs;
};
Rows rows_of(const Tensor& t) { return {buffer_for(t), buffer_offset_for(t)}; }
Rows advance(Rows r, size_t elems) { return {r.buf, r.ofs + static_cast<NSUInteger>(elems * sizeof(float))}; }

// One FP32 GEMM C (op)= A @ B^T through the mixed-precision kernel.
void gemm(Rows A, uint64_t lda, bool ta, Rows B, uint64_t ldb, bool tb, Rows C, uint64_t ldc,
          int M, int N, int K, bool accumulate) {
    metal_impl::AbtMixed g;
    g.A = A.buf; g.ofs_A = A.ofs; g.lda = lda; g.transA = ta;
    g.B = B.buf; g.ofs_B = B.ofs; g.ldb = ldb; g.transB = tb;
    g.C = C.buf; g.ofs_C = C.ofs; g.ldc = ldc;
    g.M = M; g.N = N; g.K = K;
    g.in = g.out = metal_impl::kAbtF32;
    g.epilogue = accumulate ? metal_impl::kAbtAccumulate : metal_impl::kAbtStore;
    metal_impl::launch_matmul_abt_mixed(g);
}

void run_rows(id<MTLComputePipelineState> pso, int rows, int Lk,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    NSUInteger tg = 32;
    while (static_cast<int>(tg) < Lk && tg < 1024) tg *= 2;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// FP32 attention backward over one block; see the file comment. dQ / dK / dV
// are overwritten on every head's columns.
void backward_block_f32(Rows Q, Rows K, Rows V, Rows dO, Rows dQ, Rows dK, Rows dV, int Lq, int Lk, int D,
                        int num_heads, int hd, id<MTLBuffer> bMask, NSUInteger oMask, bool causal) {
    const size_t kMaxScratchBytes = size_t(256) << 20;   // P + dP
    const int chunk = std::max(1, std::min(Lq, static_cast<int>(kMaxScratchBytes / (2 * sizeof(float) * size_t(Lk)))));
    thread_local static Tensor P = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dP = Tensor::empty_on(Device::Metal, 0, 0);
    if (P.rows != chunk || P.cols != Lk || P.dtype != Dtype::FP32) P.resize(chunk, Lk, Dtype::FP32);
    if (dP.rows != chunk || dP.cols != Lk || dP.dtype != Dtype::FP32) dP.resize(chunk, Lk, Dtype::FP32);
    const Rows rP = rows_of(P), rdP = rows_of(dP);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t Lku = static_cast<uint32_t>(Lk), has_mask = bMask ? 1u : 0u, causal_u = causal ? 1u : 0u;
    id<MTLBuffer> mbuf = bMask ? bMask : rP.buf;
    const NSUInteger mofs = bMask ? oMask : rP.ofs;
    id<MTLComputePipelineState> p_sm = pso_softmax(), p_ds = pso_dS();

    for (int h = 0; h < num_heads; ++h) {
        const size_t c0 = static_cast<size_t>(h) * hd;
        for (int q0 = 0; q0 < Lq; q0 += chunk) {
            const int rows = std::min(chunk, Lq - q0);
            const size_t qe = static_cast<size_t>(q0) * D + c0;
            const bool first = q0 == 0;
            gemm(advance(Q, qe), D, false, advance(K, c0), D, false, rP, Lk, rows, Lk, hd, false);
            const uint32_t q0u = static_cast<uint32_t>(q0);
            run_rows(p_sm, rows, Lk, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:rP.buf offset:rP.ofs atIndex:0];
                [enc setBuffer:mbuf offset:mofs atIndex:1];
                [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&scale length:sizeof(float) atIndex:3];
                [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&causal_u length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&q0u length:sizeof(uint32_t) atIndex:6];
            });
            gemm(advance(dO, qe), D, false, advance(V, c0), D, false, rdP, Lk, rows, Lk, hd, false);
            // dV_h (+)= P^T dO_h: A = P read transposed, B = dO_h read transposed.
            gemm(rP, Lk, true, advance(dO, qe), D, true, advance(dV, c0), D, Lk, hd, rows, !first);
            run_rows(p_ds, rows, Lk, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:rP.buf offset:rP.ofs atIndex:0];
                [enc setBuffer:rdP.buf offset:rdP.ofs atIndex:1];
                [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&scale length:sizeof(float) atIndex:3];
            });
            gemm(rP, Lk, false, advance(K, c0), D, true, advance(dQ, qe), D, rows, hd, Lk, false);
            gemm(rP, Lk, true, advance(Q, qe), D, true, advance(dK, c0), D, Lk, hd, rows, !first);
        }
    }
}

// FP32 views of the inputs and outputs: the tensors themselves for an FP32
// call, widened copies (and narrowed-on-exit results) for a 16-bit one.
struct Fp32Io {
    Tensor q, k, v, g, dq, dk, dv;
    const Tensor* Q;
    const Tensor* K;
    const Tensor* V;
    const Tensor* dO;
    Tensor* dQ;
    Tensor* dK;
    Tensor* dV;

    Fp32Io(const Tensor& Q_, const Tensor& K_, const Tensor& V_, const Tensor& dO_, Tensor& dQ_, Tensor& dK_,
           Tensor& dV_)
        : Q(&Q_), K(&K_), V(&V_), dO(&dO_), dQ(&dQ_), dK(&dK_), dV(&dV_) {
        if (Q_.dtype == Dtype::FP32) return;
        for (Tensor* t : {&q, &k, &v, &g}) *t = Tensor::empty_on(Device::Metal, 0, 0);   // cast() keeps dst's device
        cast(Q_, q, Dtype::FP32);
        cast(K_, k, Dtype::FP32);
        cast(V_, v, Dtype::FP32);
        cast(dO_, g, Dtype::FP32);
        dq = Tensor::empty_on(Device::Metal, Q_.rows, Q_.cols, Dtype::FP32);
        dk = Tensor::empty_on(Device::Metal, K_.rows, K_.cols, Dtype::FP32);
        dv = Tensor::empty_on(Device::Metal, V_.rows, V_.cols, Dtype::FP32);
        Q = &q; K = &k; V = &v; dO = &g; dQ = &dq; dK = &dk; dV = &dv;
    }
    // Round the FP32 gradients into the caller's tensors (no-op for FP32).
    void finish(Tensor& dQ_, Tensor& dK_, Tensor& dV_) {
        if (dQ == &dQ_) return;
        cast(dq, dQ_, dQ_.dtype);
        cast(dk, dK_, dK_.dtype);
        cast(dv, dV_, dV_.dtype);
    }
};

} // namespace

void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry with CUDA.

    const Dtype dt = Q.dtype;
    if ((dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) ||
        K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must share one of FP16, BF16, FP32");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != dt) dQ.resize(Lq, D, dt);
    if (dK.rows != Lk || dK.cols != D || dK.dtype != dt) dK.resize(Lk, D, dt);
    if (dV.rows != Lk || dV.cols != D || dV.dtype != dt) dV.resize(Lk, D, dt);
    if (Lq == 0 || Lk == 0 || D == 0) {
        dQ.zero();
        dK.zero();
        dV.zero();
        return;
    }

    id<MTLBuffer> bMask = d_mask ? pool_lookup(d_mask) : nil;
    const NSUInteger oMask = d_mask ? pool_lookup_offset(d_mask) : 0;
    Fp32Io io(Q, K, V, dO, dQ, dK, dV);
    backward_block_f32(rows_of(*io.Q), rows_of(*io.K), rows_of(*io.V), rows_of(*io.dO), rows_of(*io.dQ),
                       rows_of(*io.dK), rows_of(*io.dV), Lq, Lk, D, num_heads, hd, bMask, oMask, causal);
    io.finish(dQ, dK, dV);
}

// ─── flash_attention_varlen_backward ────────────────────────────────────────
//
// Packed variable-length backward: the same FP32 block per sequence, at the
// sequence's row offsets into the (total_tokens, D) tensors. cu_seqlens_q/k
// are DEVICE pointers (shared storage on Metal), read host-side once to drive
// the per-sequence loop.
void flash_attention_varlen_backward(const Tensor& Q,
                                     const Tensor& K,
                                     const Tensor& V,
                                     const Tensor& O,
                                     const Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     Tensor& dQ,
                                     Tensor& dK,
                                     Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry with CUDA.
    (void)max_seqlen_q;
    (void)max_seqlen_k;

    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO dtype must match");
    }
    const int total_q = Q.rows;
    const int total_k = K.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_backward: shape mismatch");
    }
    if (dO.rows != total_q || dO.cols != D) {
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: max_seqlen_q/k must be non-negative");
    }

    if (dQ.rows != total_q || dQ.cols != D || dQ.dtype != dt) dQ.resize(total_q, D, dt);
    if (dK.rows != total_k || dK.cols != D || dK.dtype != dt) dK.resize(total_k, D, dt);
    if (dV.rows != total_k || dV.cols != D || dV.dtype != dt) dV.resize(total_k, D, dt);
    if (D == 0 || batch_size == 0 || (total_q == 0 && total_k == 0)) {
        dQ.zero();
        dK.zero();
        dV.zero();
        return;
    }

    ::brotensor::sync(Device::Metal);   // the offsets may come from a GPU op still in flight
    id<MTLBuffer> bCQ = pool_lookup(cu_seqlens_q);
    id<MTLBuffer> bCK = pool_lookup(cu_seqlens_k);
    const int32_t* cq = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const char*>([bCQ contents]) + pool_lookup_offset(cu_seqlens_q));
    const int32_t* ck = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const char*>([bCK contents]) + pool_lookup_offset(cu_seqlens_k));

    if (cq[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[0] must be 0");
    if (ck[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[0] must be 0");
    if (cq[batch_size] != total_q)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[B] != total_tokens_q");
    if (ck[batch_size] != total_k)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[B] != total_tokens_k");
    for (int b = 0; b < batch_size; ++b) {
        const int Lq_b = cq[b + 1] - cq[b];
        const int Lk_b = ck[b + 1] - ck[b];
        if (Lq_b < 0 || Lk_b < 0)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens must be non-decreasing");
        if (causal && Lq_b != Lk_b)
            throw std::runtime_error("flash_attention_varlen_backward: causal requires per-sequence Lq == Lk");
    }

    Fp32Io io(Q, K, V, dO, dQ, dK, dV);
    // A sequence with no keys (or no queries) contributes nothing; its rows
    // of the gradients must still read zero.
    io.dQ->zero();
    io.dK->zero();
    io.dV->zero();
    const Rows rQ = rows_of(*io.Q), rK = rows_of(*io.K), rV = rows_of(*io.V), rdO = rows_of(*io.dO);
    const Rows rdQ = rows_of(*io.dQ), rdK = rows_of(*io.dK), rdV = rows_of(*io.dV);
    for (int b = 0; b < batch_size; ++b) {
        const int Lq = cq[b + 1] - cq[b];
        const int Lk = ck[b + 1] - ck[b];
        if (Lq == 0 || Lk == 0) continue;
        const size_t qe = static_cast<size_t>(cq[b]) * D, ke = static_cast<size_t>(ck[b]) * D;
        backward_block_f32(advance(rQ, qe), advance(rK, ke), advance(rV, ke), advance(rdO, qe), advance(rdQ, qe),
                           advance(rdK, ke), advance(rdV, ke), Lq, Lk, D, num_heads, head_dim, nil, 0, causal);
    }
    io.finish(dQ, dK, dV);
}

} // namespace brotensor::detail::metal
