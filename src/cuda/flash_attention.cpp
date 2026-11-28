#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "flash_fused_internal.h"
#include "fp16_internal.h"
#include "gemm_mma.h"
#include "ops_fa_mir.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr std::size_t kFaScoreBudget = std::size_t(32) << 20;

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::uint32_t grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return static_cast<std::uint32_t>(b);
}

bool fa_per_head_forward(const void* Q, const void* K, const void* V, const float* mask, void* O,
                         Dtype dt, int Lq, int Lk, int D, int num_heads, int hd,
                         CUstream stream) {
    const int hdp = (hd + 7) & ~7;
    const int Lk_pad = (Lk + 7) & ~7;
    const std::size_t esz = 2u;
    int rows = Lq;
    if (static_cast<std::size_t>(Lq) * Lk_pad > kFaScoreBudget) {
        rows = static_cast<int>(kFaScoreBudget / Lk_pad) / 128 * 128;
        if (rows < 128) rows = 128;
        if (rows > Lq) rows = Lq;
    }

    const Device dev = Device::cuda(current_device());
    Tensor Qh  = Tensor::empty_on(dev, Lq, hdp, dt);
    Tensor Kh  = Tensor::empty_on(dev, Lk_pad, hdp, dt);
    Tensor Vth = Tensor::empty_on(dev, hdp, Lk_pad, dt);
    Tensor Oh  = Tensor::empty_on(dev, Lq, hdp, dt);
    Tensor S   = Tensor::empty_on(dev, rows, Lk_pad, Dtype::FP32);
    Tensor P   = Tensor::empty_on(dev, rows, Lk_pad, dt);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    constexpr int CP = 256;
    CUfunction g_fn = mir_function({MirKind::FaAux, 1u, dt, dt, 1u});
    CUfunction gt_fn = mir_function({MirKind::FaAux, 2u, dt, dt, 1u, Dtype::FP32, 0u});
    CUfunction sc_fn = mir_function({MirKind::FaAux, 3u, dt, dt, 1u});
    CUfunction sm_fn = mir_function({MirKind::FaAux, 4u, dt, dt, 1u, Dtype::FP32, mask ? 1u : 0u});

    for (int h = 0; h < num_heads; ++h) {
        const int off = h * hd;
        mir_dispatch_args(g_fn, grid_for(Lq * hdp, CP), CP,
                          std::array<std::uintptr_t, 2>{addr(Q), addr(Qh.data)},
                          std::array<std::int32_t, 6>{D, off, Lq, hd, Lq, hdp}, {});
        mir_dispatch_args(g_fn, grid_for(Lk_pad * hdp, CP), CP,
                          std::array<std::uintptr_t, 2>{addr(K), addr(Kh.data)},
                          std::array<std::int32_t, 6>{D, off, Lk, hd, Lk_pad, hdp}, {});
        mir_dispatch_args(gt_fn, grid_for(hdp * Lk_pad, CP), CP,
                          std::array<std::uintptr_t, 2>{addr(V), addr(Vth.data)},
                          std::array<std::int32_t, 7>{D, off, Lk, hd, Lk_pad, hdp, Lk_pad}, {});

        for (int q0 = 0; q0 < Lq; q0 += rows) {
            const int m = Lq - q0 < rows ? Lq - q0 : rows;
            const void* q_slice = static_cast<const char*>(Qh.data) + static_cast<std::size_t>(q0) * hdp * esz;
            void* o_slice = static_cast<char*>(Oh.data) + static_cast<std::size_t>(q0) * hdp * esz;

            if (dt == Dtype::BF16) {
                if (!mma_gemm::launch_f32out(static_cast<const Bf16*>(q_slice),
                                             static_cast<const Bf16*>(Kh.data),
                                             static_cast<float*>(S.data), m, Lk_pad, hdp, stream)) {
                    return false;
                }
                mir_dispatch_args(sm_fn, static_cast<std::uint32_t>(m), 256u,
                                  std::array<std::uintptr_t, 3>{addr(S.data), addr(P.data), addr(mask)},
                                  std::array<std::int32_t, 2>{Lk, Lk_pad}, std::array<float, 1>{scale});
                fp16_internal::launch_matmul_ABT(static_cast<const Bf16*>(P.data),
                                                 static_cast<const Bf16*>(Vth.data),
                                                 static_cast<Bf16*>(o_slice), m, hdp, Lk_pad);
            } else {
                if (!mma_gemm::launch_f32out(static_cast<const Fp16*>(q_slice),
                                             static_cast<const Fp16*>(Kh.data),
                                             static_cast<float*>(S.data), m, Lk_pad, hdp, stream)) {
                    return false;
                }
                mir_dispatch_args(sm_fn, static_cast<std::uint32_t>(m), 256u,
                                  std::array<std::uintptr_t, 3>{addr(S.data), addr(P.data), addr(mask)},
                                  std::array<std::int32_t, 2>{Lk, Lk_pad}, std::array<float, 1>{scale});
                fp16_internal::launch_matmul_ABT(static_cast<const Fp16*>(P.data),
                                                 static_cast<const Fp16*>(Vth.data),
                                                 static_cast<Fp16*>(o_slice), m, hdp, Lk_pad);
            }
        }
        mir_dispatch_args(sc_fn, grid_for(Lq * hd, CP), CP,
                          std::array<std::uintptr_t, 2>{addr(Oh.data), addr(O)},
                          std::array<std::int32_t, 5>{Lq, D, off, hd, hdp}, {});
    }
    return true;
}

}  // namespace

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                             const float* d_mask, int num_heads, bool causal, Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_forward: Q, K, V must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_forward: Q, K, V dtype must match");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    if (flash_fused::supported(head_dim)) {
        CUstream stream = reinterpret_cast<CUstream>(cuda_current_stream());
        if (bf16) {
            flash_fused::launch(
                reinterpret_cast<const Bf16*>(Q.data),
                reinterpret_cast<const Bf16*>(K.data),
                reinterpret_cast<const Bf16*>(V.data),
                d_mask, reinterpret_cast<Bf16*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        } else {
            flash_fused::launch(
                reinterpret_cast<const Fp16*>(Q.data),
                reinterpret_cast<const Fp16*>(K.data),
                reinterpret_cast<const Fp16*>(V.data),
                d_mask, reinterpret_cast<Fp16*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        }
        return;
    }

    if (!causal) {
        CUstream stream = reinterpret_cast<CUstream>(cuda_current_stream());
        if (fa_per_head_forward(Q.data, K.data, V.data, d_mask, O.data, dt, Lq, Lk, D, num_heads, head_dim, stream)) {
            return;
        }
    }

    dispatch_fa_rows(
        kFaModeRows, dt, Q.device,
        Q.data, K.data, V.data, O.data,
        d_mask, nullptr, nullptr,
        Lq, Lk, D, D, D,
        num_heads, num_heads, head_dim,
        0, causal, false, 0, 0, 0.0f);
}

void flash_attention_gqa_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 const float* d_mask, int num_q_heads, int num_kv_heads,
                                 bool causal, Tensor& O);
void flash_attention_windowed_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                      const float* d_mask, int num_heads, int window,
                                      Tensor& O, bool causal);
void flash_attention_varlen_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                    int batch_size, int max_seqlen_q, int max_seqlen_k,
                                    int num_heads, int head_dim, bool causal, Tensor& O);
void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask, int num_heads, bool causal, Tensor& O);
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask, int num_heads, bool causal, const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo);
void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& O, const Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              Tensor& dQ, Tensor& dK, Tensor& dV);
void flash_attention_varlen_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                     const Tensor& O, const Tensor& dO,
                                     const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     Tensor& dQ, Tensor& dK, Tensor& dV);
void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out, Tensor& V_out);
void flash_attention_q_with_kv_cached_forward(const Tensor& X, const Tensor& K, const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask, int num_heads,
                                              bool causal, Tensor& O);
void flash_attention_qkvo_int8w_fp16(const Tensor& X, const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask, int num_heads, bool causal, Tensor& O);
void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X, const Tensor& K, const Tensor& V,
                                                 const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                                 const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                                 const float* d_mask, int num_heads,
                                                 bool causal, Tensor& O);
void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                           const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                           Tensor& K_out, Tensor& V_out);

void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward                     = &flash_attention_forward;
    v.flash_attention_gqa_forward                 = &flash_attention_gqa_forward;
    v.flash_attention_windowed_forward            = &flash_attention_windowed_forward;
    v.flash_attention_varlen_forward              = &flash_attention_varlen_forward;
    v.flash_attention_qkvo_forward                = &flash_attention_qkvo_forward;
    v.flash_attention_qkvo_backward               = &flash_attention_qkvo_backward;
    v.flash_attention_backward                    = &flash_attention_backward;
    v.flash_attention_varlen_backward             = &flash_attention_varlen_backward;
    v.flash_attention_project_kv                  = &flash_attention_project_kv;
    v.flash_attention_q_with_kv_cached_forward    = &flash_attention_q_with_kv_cached_forward;
    v.flash_attention_qkvo_int8w_fp16             = &flash_attention_qkvo_int8w_fp16;
    v.flash_attention_q_with_kv_cached_int8w_fp16 = &flash_attention_q_with_kv_cached_int8w_fp16;
    v.flash_attention_project_kv_int8w_fp16       = &flash_attention_project_kv_int8w_fp16;
}

}  // namespace detail::cuda
}  // namespace brotensor
