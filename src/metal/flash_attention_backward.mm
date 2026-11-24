// Metal backend for flash_attention_backward, flash_attention_varlen_backward,
// and the shared fa_bwd_metal dispatcher backed by single-source brass MIR
// (MirKind::FaBwd, with non-causal hd > 256 fallback on simdgroup GEMM +
// MirKind::{AttnSoftmax, AttnBwd}). Mirrors src/vulkan/ops_fa_bwd.cpp.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

namespace {

constexpr int kFaBwdMaxHd = 256;

struct FaBwdCfg {
    std::uint32_t nt, br, bc;
};

std::uint32_t fa_bwd_shared_bytes(Dtype dt, int hdp, FaBwdCfg c) {
    const std::uint32_t st4 = dt == Dtype::FP32 ? 16u : 8u;
    const std::uint32_t hs4 = static_cast<std::uint32_t>(hdp / 4 + 1);
    return 2u * (c.br + c.bc) * hs4 * st4 + 8u * c.br * c.bc + 4u * c.bc +
           4u * (c.nt * 2u + c.br * 4u) + 64u;
}

FaBwdCfg pick_fa_bwd_cfg(Dtype dt, int hdp, std::uint32_t max_shared) {
    static constexpr FaBwdCfg kConfigs[] = {
        {128, 32, 32},
        {128, 32, 16},
        {128, 16, 16},
        {128, 16,  8},
        { 32, 16,  8},
        { 32,  8,  8},
        {  8,  8,  4},
    };
    for (const FaBwdCfg& c : kConfigs) {
        if (fa_bwd_shared_bytes(dt, hdp, c) <= max_shared) return c;
    }
    return kConfigs[std::size(kConfigs) - 1];
}

std::uint64_t cdiv_u64(std::uint64_t a, std::uint64_t b) { return (a + b - 1u) / b; }

struct Rows {
    id<MTLBuffer> buf;
    NSUInteger ofs;
};
Rows rows_of(const Tensor& t) { return {buffer_for(t), buffer_offset_for(t)}; }
Rows advance(Rows r, size_t elems) { return {r.buf, r.ofs + static_cast<NSUInteger>(elems * sizeof(float))}; }

void gemm_f32(Rows A, uint64_t lda, bool ta, Rows B, uint64_t ldb, bool tb, Rows C, uint64_t ldc,
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

// Zero-MSL fallback for non-causal wide heads (hd > 256).
void backward_noncausal_wide(const Tensor& Q_, const Tensor& K_, const Tensor& V_, const Tensor& dO_,
                             const float* d_mask, int num_heads, int hd,
                             Tensor& dQ_, Tensor& dK_, Tensor& dV_) {
    Tensor q, k, v, g, dq, dk, dv;
    const Tensor *Q = &Q_, *K = &K_, *V = &V_, *dO = &dO_;
    Tensor *dQ = &dQ_, *dK = &dK_, *dV = &dV_;
    if (Q_.dtype != Dtype::FP32) {
        for (Tensor* t : {&q, &k, &v, &g}) *t = Tensor::empty_on(Device::Metal, 0, 0);
        cast(Q_, q, Dtype::FP32);
        cast(K_, k, Dtype::FP32);
        cast(V_, v, Dtype::FP32);
        cast(dO_, g, Dtype::FP32);
        dq = Tensor::empty_on(Device::Metal, Q_.rows, Q_.cols, Dtype::FP32);
        dk = Tensor::empty_on(Device::Metal, K_.rows, K_.cols, Dtype::FP32);
        dv = Tensor::empty_on(Device::Metal, V_.rows, V_.cols, Dtype::FP32);
        Q = &q; K = &k; V = &v; dO = &g; dQ = &dq; dK = &dk; dV = &dv;
    }
    const int Lq = Q->rows, Lk = K->rows, D = Q->cols;
    const size_t kMaxScratchBytes = size_t(256) << 20;
    const int chunk = std::max(1, std::min(Lq, static_cast<int>(kMaxScratchBytes / (2 * sizeof(float) * size_t(Lk)))));
    Tensor P = Tensor::empty_on(Device::Metal, chunk, Lk, Dtype::FP32);
    Tensor dP = Tensor::empty_on(Device::Metal, chunk, Lk, Dtype::FP32);
    const Rows rQ = rows_of(*Q), rK = rows_of(*K), rV = rows_of(*V), rdO = rows_of(*dO);
    const Rows rdQ = rows_of(*dQ), rdK = rows_of(*dK), rdV = rows_of(*dV), rP = rows_of(P), rdP = rows_of(dP);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const MirKernel mk_sm = mir_kernel({MirKind::AttnSoftmax, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32,
                                        static_cast<std::uint8_t>(d_mask ? 1u : 0u)});
    const MirKernel mk_bwd = mir_kernel({MirKind::AttnBwd, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 0u});
    for (int h = 0; h < num_heads; ++h) {
        const size_t c0 = static_cast<size_t>(h) * hd;
        for (int q0 = 0; q0 < Lq; q0 += chunk) {
            const int rows = std::min(chunk, Lq - q0);
            const size_t qe = static_cast<size_t>(q0) * D + c0;
            const bool first = (q0 == 0);
            gemm_f32(advance(rQ, qe), D, false, advance(rK, c0), D, false, rP, Lk, rows, Lk, hd, false);
            const void* sm_ptrs[] = {P.data, P.data, nullptr, d_mask, nullptr};
            const std::int32_t sm_ints[] = {rows, Lk, rows, 0, Lk, Lk, 0, 0};
            const float sm_flts[] = {scale};
            mir_dispatch_args(mk_sm, static_cast<std::uint32_t>(std::clamp(rows, 1, 65535)),
                              "flash_attention_backward", sm_ptrs, sm_ints, sm_flts);
            gemm_f32(advance(rdO, qe), D, false, advance(rV, c0), D, false, rdP, Lk, rows, Lk, hd, false);
            gemm_f32(rP, Lk, true, advance(rdO, qe), D, true, advance(rdV, c0), D, Lk, hd, rows, !first);
            const void* bwd_ptrs[] = {P.data, dP.data, nullptr, nullptr};
            const std::int32_t bwd_ints[] = {rows, rows, Lk};
            const float bwd_flts[] = {scale};
            mir_dispatch_args(mk_bwd, static_cast<std::uint32_t>(std::clamp(rows, 1, 65535)),
                              "flash_attention_backward", bwd_ptrs, bwd_ints, bwd_flts);
            gemm_f32(rdP, Lk, false, advance(rK, c0), D, true, advance(rdQ, qe), D, rows, hd, Lk, false);
            gemm_f32(rdP, Lk, true, advance(rQ, qe), D, true, advance(rdK, c0), D, Lk, hd, rows, !first);
        }
    }
    if (dQ != &dQ_) {
        cast(dq, dQ_, dQ_.dtype);
        cast(dk, dK_, dK_.dtype);
        cast(dv, dV_, dV_.dtype);
    }
}

}  // namespace

void fa_bwd_metal(const FaBwdProblem& p) {
    if (p.lq == 0 || p.lk == 0 || p.heads == 0 || p.hd == 0) return;
    if (p.hd > kFaBwdMaxHd) {
        throw std::runtime_error(std::string("brotensor: ") + p.op +
                                 ": head_dim above 256 is not supported by MirKind::FaBwd");
    }
    if (p.heads > 65535) {
        throw std::runtime_error(std::string("brotensor: ") + p.op + ": too many heads");
    }
    const long long nstat = static_cast<long long>(p.heads) * p.lq * 4;
    if (nstat > 0x7fffffffLL) {
        throw std::runtime_error(std::string("brotensor: ") + p.op + ": problem too large");
    }
    Tensor stats = Tensor::empty_on(Device::Metal, static_cast<int>(nstat), 1, Dtype::FP32);

    const std::uint32_t max_shared = std::min<std::uint32_t>(
        32768u, static_cast<std::uint32_t>([metal_impl::device() maxThreadgroupMemoryLength]));
    const int hdp = (p.hd + 31) / 32 * 32;
    const FaBwdCfg c = pick_fa_bwd_cfg(p.dt, hdp, max_shared);
    if (fa_bwd_shared_bytes(p.dt, hdp, c) > max_shared) {
        throw std::runtime_error(std::string("brotensor: ") + p.op +
                                 ": tile exceeds Metal 32 KiB threadgroup memory limit");
    }

    const std::uint32_t op = (static_cast<std::uint32_t>(p.hd) & 0xffffu) |
                             ((static_cast<std::uint32_t>(hdp) & 0xffffu) << 16);
    const std::uint64_t aux_geom = static_cast<std::uint64_t>(c.br & 0xffffu) |
                                   (static_cast<std::uint64_t>(c.bc & 0xffffu) << 16);
    auto kernel = [&](std::uint32_t pass) -> MirKernel {
        const std::uint8_t flags = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(p.mode) & 0x0fu) |
            (p.mask != nullptr ? 0x10u : 0u) |
            ((pass & 1u) << 5));
        return mir_kernel({MirKind::FaBwd, op, p.dt, Dtype::FP32,
                           static_cast<std::uint8_t>(c.nt), Dtype::FP32, flags, aux_geom});
    };

    const std::uint64_t nqb = cdiv_u64(static_cast<std::uint64_t>(p.lq), c.br);
    const std::uint64_t nkb = cdiv_u64(static_cast<std::uint64_t>(p.lk), c.bc);
    if (nqb > 65535u || nkb > 65535u) {
        throw std::runtime_error(std::string("brotensor: ") + p.op + ": sequence too long for one dispatch");
    }

    const float flts[] = {
        static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(p.hd))),
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.hd))),
    };
    {
        const void* ptrs0[] = {p.q, p.k, p.v, p.g, p.mask, p.aux, p.aux2, stats.data, p.dq, nullptr};
        const std::int32_t ints0[] = {
            p.lq, p.lk, p.ldq, p.ldk, p.ldg, p.ldq_out, std::max(0, p.window), p.causal ? 1 : 0, p.nseq,
        };
        mir_dispatch_args_3d(kernel(0), static_cast<std::uint32_t>(nqb),
                             static_cast<std::uint32_t>(p.heads), 1u, p.op, ptrs0, ints0, flts);
    }
    {
        const void* ptrs1[] = {p.q, p.k, p.v, p.g, p.mask, p.aux, p.aux2, stats.data, p.dk, p.dv};
        const std::int32_t ints1[] = {
            p.lq, p.lk, p.ldq, p.ldk, p.ldg, p.ldk_out, std::max(0, p.window), p.causal ? 1 : 0, p.nseq,
        };
        mir_dispatch_args_3d(kernel(1), static_cast<std::uint32_t>(nkb),
                             static_cast<std::uint32_t>(p.heads), 1u, p.op, ptrs1, ints1, flts);
    }
}

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
    (void)O;
    const Dtype dt = Q.dtype;
    if ((dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) ||
        K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must share one of FP16, BF16, FP32");
    }
    const int Lq = Q.rows, Lk = K.rows, D = Q.cols;
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
        if (dQ.size() > 0) dQ.zero();
        if (dK.size() > 0) dK.zero();
        if (dV.size() > 0) dV.zero();
        return;
    }

    if (!causal && hd > kFaBwdMaxHd) {
        backward_noncausal_wide(Q, K, V, dO, d_mask, num_heads, hd, dQ, dK, dV);
        return;
    }

    FaBwdProblem p;
    p.op = "flash_attention_backward";
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.g = dO.data;
    p.dq = dQ.data; p.dk = dK.data; p.dv = dV.data;
    p.mask = d_mask;
    p.lq = Lq; p.lk = Lk;
    p.ldq = p.ldk = p.ldg = p.ldq_out = p.ldk_out = D;
    p.heads = num_heads;
    p.hd = hd;
    p.mode = FA_MODE_ROWS;
    p.causal = causal;
    fa_bwd_metal(p);
}

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
    (void)O;
    (void)max_seqlen_q;
    (void)max_seqlen_k;

    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO dtype must match");
    }
    const int total_q = Q.rows, total_k = K.rows;
    const int D = num_heads * head_dim;
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    }
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_backward: shape mismatch");
    }
    if (dO.rows != total_q || dO.cols != D) {
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
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
    if (D == 0 || batch_size == 0 || total_q == 0 || total_k == 0) {
        if (dQ.size() > 0) dQ.zero();
        if (dK.size() > 0) dK.zero();
        if (dV.size() > 0) dV.zero();
        return;
    }

    FaBwdProblem p;
    p.op = "flash_attention_varlen_backward";
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.g = dO.data;
    p.dq = dQ.data; p.dk = dK.data; p.dv = dV.data;
    p.aux = cu_seqlens_q; p.aux2 = cu_seqlens_k;
    p.lq = total_q; p.lk = total_k;
    p.ldq = p.ldk = p.ldg = p.ldq_out = p.ldk_out = D;
    p.heads = num_heads;
    p.hd = head_dim;
    p.mode = FA_MODE_VARLEN;
    p.nseq = batch_size;
    p.causal = causal;
    fa_bwd_metal(p);
}

}  // namespace brotensor::detail::metal
