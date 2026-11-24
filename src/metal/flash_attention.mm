#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

// Forward declarations of sibling Metal ops implemented in other TUs (and of
// ops defined later in this file). Generated from the canonical op table.
#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

namespace {

constexpr std::uint64_t kMetalMaxSharedBytes = 32768u;

std::uint64_t cdiv_u64(std::uint64_t a, std::uint64_t b) { return (a + b - 1u) / b; }

void fa_combine_metal(const FaRowsProblem& p, const void* part, std::uint64_t nsplit) {
    const MirKernel mk = mir_kernel({MirKind::FaCombine, 0u, p.dt});
    const std::uint32_t entries = static_cast<std::uint32_t>(static_cast<long long>(p.lq) * p.hq);
    const void* ptrs[] = {part, p.o};
    const std::int32_t ints[] = {
        static_cast<std::int32_t>(entries), static_cast<std::int32_t>(nsplit),
        p.hd, p.hq, p.ldo,
    };
    mir_dispatch_args(mk, std::max<std::uint32_t>(1u, std::min<std::uint32_t>(65535u, entries)),
                      p.op, ptrs, ints, {});
}

// The BF16 projections' name for linear_forward_batched_fp16 (gemm.mm), which
// takes BF16 too, with the bias fused and one rounding.
void fa_linear_forward_batched_bf16(const Tensor& W, const Tensor* bias,
                                    const Tensor& X_BD, Tensor& Y_BD) {
    linear_forward_batched_fp16(W, bias, X_BD, Y_BD);
}

}  // namespace

void fa_rows_metal(const FaRowsProblem& p) {
    if (p.lq <= 0 || p.hq <= 0 || p.hd <= 0) return;
    const int group = p.hq / p.hkv;
    const int slots = (p.hd + 127) / 128;
    // Cap threadgroup memory strictly <= 32 KiB on Apple Metal:
    // (g * hd + g * 128 + 136) * 4 <= 32768 (and g * slots <= 16, g * hd <= 6144).
    int G = 1;
    for (int g = std::min(group, 16); g >= 1; --g) {
        const std::uint64_t sh_bytes =
            (static_cast<std::uint64_t>(g) * static_cast<std::uint64_t>(p.hd) +
             static_cast<std::uint64_t>(g) * 128u + 136u) * 4u;
        if (group % g == 0 && g * slots <= 16 && g * p.hd <= 6144 && sh_bytes <= kMetalMaxSharedBytes) {
            G = g;
            break;
        }
    }
    const std::uint64_t sh_bytes =
        (static_cast<std::uint64_t>(G) * static_cast<std::uint64_t>(p.hd) +
         static_cast<std::uint64_t>(G) * 128u + 136u) * 4u;
    if (slots > 16 || sh_bytes > kMetalMaxSharedBytes) {
        throw std::runtime_error(std::string("brotensor: ") + p.op +
                                 ": head_dim exceeds Metal 32 KiB threadgroup memory limit");
    }

    const std::uint64_t entries = static_cast<std::uint64_t>(p.lq) * static_cast<std::uint64_t>(p.hq / G);
    constexpr std::uint64_t kTarget = 80u;
    std::uint64_t nsplit = 1u;
    if (entries < kTarget && p.lk > 128) {
        nsplit = std::min<std::uint64_t>({cdiv_u64(kTarget, entries), cdiv_u64(p.lk, 128u), 64u});
    }
    std::uint64_t chunk = cdiv_u64(cdiv_u64(static_cast<std::uint64_t>(std::max(0, p.lk)), nsplit), 128u) * 128u;
    nsplit = std::max<std::uint64_t>(1u, cdiv_u64(static_cast<std::uint64_t>(std::max(0, p.lk)), std::max<std::uint64_t>(1u, chunk)));
    if (p.lk == 0) chunk = 128u;

    const bool vec = (p.hd % 8 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(p.k) % 16u == 0u) &&
                     (p.ldk % 8 == 0);
    const bool split = nsplit > 1u;
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (p.mask != nullptr ? 1u : 0u) | (split ? 2u : 0u) | (p.softcap > 0.0f ? 4u : 0u) |
        (vec ? 8u : 0u) | (p.maskpos ? 16u : 0u));
    const MirKernel mk = mir_kernel(
        {MirKind::FaRows, static_cast<std::uint32_t>(p.hd), p.dt, Dtype::FP32,
         static_cast<std::uint8_t>(G), Dtype::FP32, flags, static_cast<std::uint64_t>(p.mode)});
    Tensor part;
    const void* part_ptr = nullptr;
    if (split) {
        const long long pn = static_cast<long long>(p.lq) * p.hq * static_cast<long long>(nsplit) * (p.hd + 2);
        if (pn > 0x7fffffffLL) {
            throw std::runtime_error(std::string("brotensor: ") + p.op + ": attention partials too large");
        }
        part = Tensor::empty_on(Device::Metal, static_cast<int>(pn), 1, Dtype::FP32);
        part_ptr = part.data;
    }
    const void* ptrs[] = {p.q, p.k, p.v, p.o, p.mask, p.aux, p.aux2, part_ptr};
    const float flts[] = {static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.hd))), p.softcap};
    for (int r0 = 0; r0 < p.lq; r0 += 65535) {
        const std::int32_t ints[] = {
            p.lq, p.lk, p.ldq, p.ldk, p.ldo, group, std::max(0, p.window),
            p.causal ? 1 : 0, p.nseq, static_cast<std::int32_t>(chunk), p.hq, r0, p.q_offset,
        };
        mir_dispatch_args_3d(
            mk, static_cast<std::uint32_t>(nsplit),
            static_cast<std::uint32_t>(std::min(65535, p.lq - r0)),
            static_cast<std::uint32_t>(p.hq / G), p.op, ptrs, ints, flts);
    }
    if (split) fa_combine_metal(p, part_ptr, nsplit);
}

void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_forward: Q, K, V must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_forward: Q, K, V dtype must match");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    }
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || D == 0) return;
    if (Lk == 0) { O.zero(); return; }

    if (causal) {
        FaRowsProblem p;
        p.op = "flash_attention_forward";
        p.dt = dt;
        p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
        p.mask = d_mask;
        p.lq = Lq; p.lk = Lk;
        p.ldq = p.ldk = p.ldo = D;
        p.hd = head_dim; p.hq = p.hkv = num_heads;
        p.mode = FA_MODE_ROWS;
        p.causal = true;
        fa_rows_metal(p);
        return;
    }

    DenseAttnProblem p;
    p.op = "flash_attention_forward";
    p.dev = Q.device;
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
    p.mask = d_mask;
    p.lq = Lq; p.lk = Lk;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = head_dim; p.hq = p.hkv = num_heads;
    DenseAttnExtras x;
    x.auto_scale = true;
    x.mask_ge = false;
    dense_attention_metal(p, x);
}

void flash_attention_varlen_forward(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size,
                                    int max_seqlen_q,
                                    int max_seqlen_k,
                                    int num_heads,
                                    int head_dim,
                                    bool causal,
                                    Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_forward: Q, K, V must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_forward: Q, K, V dtype must match");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_forward: num_heads/head_dim must be positive");
    }
    const int D = num_heads * head_dim;
    const int total_q = Q.rows;
    const int total_k = K.rows;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_forward: shape mismatch");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_forward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_forward: max_seqlen_q/k must be non-negative");
    }
    if (O.rows != total_q || O.cols != D || O.dtype != dt) {
        O.resize(total_q, D, dt);
    }
    if (total_q == 0 || D == 0) return;
    if (batch_size == 0 || total_k == 0) { O.zero(); return; }
    (void)max_seqlen_q; (void)max_seqlen_k;

    FaRowsProblem p;
    p.op = "flash_attention_varlen_forward";
    p.dt = dt;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = O.data;
    p.aux = cu_seqlens_q; p.aux2 = cu_seqlens_k;
    p.lq = total_q; p.lk = total_k;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = head_dim; p.hq = p.hkv = num_heads;
    p.mode = FA_MODE_VARLEN;
    p.nseq = batch_size;
    p.causal = causal;
    fa_rows_metal(p);
}

void flash_attention_qkvo_forward(const Tensor& X,
                                  const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask,
                                  int num_heads,
                                  bool causal,
                                  Tensor& O) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_qkvo_forward: tensors must be FP16 or BF16");
    }
    if (Wq.dtype != dt || Wk.dtype != dt || Wv.dtype != dt || Wo.dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_forward: weight dtype must match X");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    if (Ctx && Ctx->dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_forward: Ctx dtype must match X");
    }
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_forward: num_heads must divide D");
    }

    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Kp = Tensor::empty_on(Device::Metal, Lk, D, dt);
    Tensor Vp = Tensor::empty_on(Device::Metal, Lk, D, dt);

    flash_attention_project_kv(kv_src, Wk, bk, Wv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_forward(
        X, Kp, Vp, Wq, bq, Wo, bo, d_mask, num_heads, causal, O);
}

void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out,
                                Tensor& V_out) {
    const Dtype dt = ctx.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_project_kv: tensors must be FP16 or BF16");
    }
    if (Wk.dtype != dt || Wv.dtype != dt ||
        (bk && bk->dtype != dt) || (bv && bv->dtype != dt)) {
        throw std::runtime_error("flash_attention_project_kv: dtype mismatch");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk.rows;
    if (Wk.cols != D_ctx || Wv.rows != D || Wv.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != dt) {
        K_out.resize(Lk, D, dt);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != dt) {
        V_out.resize(Lk, D, dt);
    }
    if (Lk == 0 || D == 0) return;
    if (dt == Dtype::BF16) {
        fa_linear_forward_batched_bf16(Wk, bk, ctx, K_out);
        fa_linear_forward_batched_bf16(Wv, bv, ctx, V_out);
    } else {
        linear_forward_batched_fp16(Wk, bk, ctx, K_out);
        linear_forward_batched_fp16(Wv, bv, ctx, V_out);
    }
}

void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K,
                                              const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask,
                                              int num_heads,
                                              bool causal,
                                              Tensor& O) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: tensors must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt || Wq.dtype != dt || Wo.dtype != dt ||
        (bq && bq->dtype != dt) || (bo && bo->dtype != dt)) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: dtype mismatch");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: K/V shape mismatch");
    }
    if (Wq.rows != D || Wq.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::Metal, Lq, D, dt);
    Tensor Op = Tensor::empty_on(Device::Metal, Lq, D, dt);

    if (dt == Dtype::BF16) {
        fa_linear_forward_batched_bf16(Wq, bq, X, Qp);
        flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
        fa_linear_forward_batched_bf16(Wo, bo, Op, O);
    } else {
        linear_forward_batched_fp16(Wq, bq, X, Qp);
        flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
        linear_forward_batched_fp16(Wo, bo, Op, O);
    }
}

// ─── W8A16 variants of the three fused flash-attention ops ─────────────────
// See src/cuda/flash_attention.cu for documentation.

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8,
                                           const Tensor& sk,
                                           const Tensor* bk,
                                           const Tensor& Wv_int8,
                                           const Tensor& sv,
                                           const Tensor* bv,
                                           Tensor& K_out,
                                           Tensor& V_out) {
    if (ctx.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: ctx must be FP16");
    }
    if (Wk_int8.dtype != Dtype::INT8 || Wv_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv must be INT8");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk_int8.rows;
    if (Wk_int8.cols != D_ctx || Wv_int8.rows != D || Wv_int8.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != Dtype::FP16) {
        K_out.resize(Lk, D, Dtype::FP16);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != Dtype::FP16) {
        V_out.resize(Lk, D, Dtype::FP16);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_int8w_fp16(Wk_int8, sk, bk, ctx, K_out);
    linear_forward_batched_int8w_fp16(Wv_int8, sv, bv, ctx, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                 const Tensor& K,
                                                 const Tensor& V,
                                                 const Tensor& Wq_int8,
                                                 const Tensor& sq,
                                                 const Tensor* bq,
                                                 const Tensor& Wo_int8,
                                                 const Tensor& so,
                                                 const Tensor* bo,
                                                 const float* d_mask,
                                                 int num_heads,
                                                 bool causal,
                                                 Tensor& O) {
    if (X.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: X/K/V must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: K/V shape mismatch");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);
    Tensor Op = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);

    linear_forward_batched_int8w_fp16(Wq_int8, sq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_int8w_fp16(Wo_int8, so, bo, Op, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X,
                                     const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask,
                                     int num_heads,
                                     bool causal,
                                     Tensor& O) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: X must be FP16");
    }
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: Ctx must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: all weights must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D_ctx ||
        Wv_int8.rows != D || Wv_int8.cols != D_ctx ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Kp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);
    Tensor Vp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);
    flash_attention_project_kv_int8w_fp16(kv_src,
                                          Wk_int8, sk, bk,
                                          Wv_int8, sv, bv,
                                          Kp, Vp);
    flash_attention_q_with_kv_cached_int8w_fp16(
        X, Kp, Vp,
        Wq_int8, sq, bq,
        Wo_int8, so, bo,
        d_mask, num_heads, causal, O);
}

// Recompute-style FP16 backward. Mirrors the CUDA implementation in
// src/cuda/flash_attention.cu (flash_attention_qkvo_backward_gpu, lines 724+):
//   1. Re-project X→Q, kv_src→K, kv_src→V.
//   2. Per-head: extract, S = Qh·Kh^T, softmax → P, Oh = P·Vh; pack into O_attn.
//   3. Wo backward via linear_backward_batched_gpu → dO_attn.
//   4. Per-head: extract dOh; recompute P; dVh = P^T·dOh; dP = dOh·Vh^T;
//                dS = P*(dP-D_q)*inv_sqrt (in-place over P); dQh = dS·Kh;
//                dKh = dS^T·Qh; pack dQ/dK/dV.
//   5. Linear backward for Q/K/V projections, accumulate into dX (and dCtx).
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask,
    int num_heads,
    bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo) {

    // ── Argument validation ──────────────────────────────────────────────
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_qkvo_backward: tensors must be FP16 or BF16");
    }
    if (dO.dtype != dt || Wq.dtype != dt || Wk.dtype != dt ||
        Wv.dtype != dt || Wo.dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_backward: all tensors must share dtype");
    }
    if (Ctx && Ctx->dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_backward: Ctx dtype must match X");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const bool self_attn = (Ctx == nullptr);
    if (self_attn) {
        if (dCtx != nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be null when Ctx is null");
        }
    } else {
        if (dCtx == nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be non-null when Ctx is non-null");
        }
    }
    auto bias_pair_ok = [](const Tensor* b, const Tensor* db) {
        return static_cast<bool>(b) == static_cast<bool>(db);
    };
    if (!bias_pair_ok(bq, dbq) || !bias_pair_ok(bk, dbk) ||
        !bias_pair_ok(bv, dbv) || !bias_pair_ok(bo, dbo)) {
        throw std::runtime_error("flash_attention_qkvo_backward: bias/grad-bias presence mismatch");
    }

    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_qkvo_backward: causal requires Lq == Lk");
    }

    if (dX.rows != Lq || dX.cols != D || dX.dtype != dt) {
        dX.resize(Lq, D, dt);
    }
    if (!self_attn) {
        if (dCtx->rows != Lk || dCtx->cols != D_ctx || dCtx->dtype != dt) {
            dCtx->resize(Lk, D_ctx, dt);
        }
        dCtx->zero();
    }
    dX.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    // ── Scratch tensors (reused across calls). ────────────────────────────
    thread_local static Tensor Qp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Kp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Vp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor O_attn = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dO_attn = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dQ = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dK = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dV = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_Q = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_K = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_V = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor scratch_db = Tensor::empty_on(Device::Metal, 0, 0);

    auto ensure = [dt](Tensor& t, int r, int c) {
        if (t.rows != r || t.cols != c || t.dtype != dt) {
            t.resize(r, c, dt);
        }
    };
    ensure(dX_from_Q, Lq, D);
    ensure(dX_from_K, Lk, D_ctx);
    ensure(dX_from_V, Lk, D_ctx);

    // ── 1. Recompute forward projections. ─────────────────────────────────
    if (bf16) {
        fa_linear_forward_batched_bf16(Wq, bq, X,      Qp);
        fa_linear_forward_batched_bf16(Wk, bk, kv_src, Kp);
        fa_linear_forward_batched_bf16(Wv, bv, kv_src, Vp);
    } else {
        linear_forward_batched_fp16(Wq, bq, X,      Qp);
        linear_forward_batched_fp16(Wk, bk, kv_src, Kp);
        linear_forward_batched_fp16(Wv, bv, kv_src, Vp);
    }

    // ── 2. Recompute the attention output O_attn (Lq, D). ─────────────────
    flash_attention_forward(Qp, Kp, Vp, d_mask, num_heads, causal, O_attn);

    // ── 3. Wo + bo backward. dO_attn(Lq, D) = dO @ Wo. ────────────────────
    {
        const bool has_bo = (bo != nullptr);
        if (!has_bo) {
            if (scratch_db.rows != D || scratch_db.cols != 1 || scratch_db.dtype != dt) {
                scratch_db.resize(D, 1, dt);
            }
            scratch_db.zero();
        }
        linear_backward_batched(Wo, O_attn, dO, dO_attn, dWo,
                                has_bo ? *dbo : scratch_db);
    }

    // ── 4. Attention core backward (FP32 inside, dQ / dK / dV rounded once).
    flash_attention_backward(Qp, Kp, Vp, O_attn, dO_attn, d_mask, num_heads, causal, dQ, dK, dV);

    // ── 5. Q / K / V projection backward. ─────────────────────────────────
    auto run_proj_back = [&](const Tensor& W, const Tensor& In,
                             const Tensor& dOut, Tensor& dIn_out,
                             Tensor& dW_acc, const Tensor* b_fwd,
                             Tensor* db_acc) {
        const bool has_b = (b_fwd != nullptr);
        if (!has_b) {
            if (scratch_db.rows != W.rows || scratch_db.cols != 1 ||
                scratch_db.dtype != dt) {
                scratch_db.resize(W.rows, 1, dt);
            }
            scratch_db.zero();
        }
        linear_backward_batched(W, In, dOut, dIn_out, dW_acc,
                                has_b ? *db_acc : scratch_db);
    };

    run_proj_back(Wq, X,      dQ, dX_from_Q, dWq, bq, dbq);
    run_proj_back(Wk, kv_src, dK, dX_from_K, dWk, bk, dbk);
    run_proj_back(Wv, kv_src, dV, dX_from_V, dWv, bv, dbv);

    // ── 6. Accumulate into dX / dCtx. ─────────────────────────────────────
    add_inplace(dX, dX_from_Q);
    if (self_attn) {
        add_inplace(dX, dX_from_K);
        add_inplace(dX, dX_from_V);
    } else {
        add_inplace(*dCtx, dX_from_K);
        add_inplace(*dCtx, dX_from_V);
    }
}

} // namespace brotensor::detail::metal
