#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD);
void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB);
void linear_forward_batched_int8w_fp16(const ::brotensor::Tensor& W_int8,
                                       const ::brotensor::Tensor& scales,
                                       const ::brotensor::Tensor* bias,
                                       const ::brotensor::Tensor& X_BD,
                                       ::brotensor::Tensor& Y_BD);
void flash_attention_forward(const ::brotensor::Tensor& Q,
                             const ::brotensor::Tensor& K,
                             const ::brotensor::Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             ::brotensor::Tensor& O);
void flash_attention_backward(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O,
                              const ::brotensor::Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              ::brotensor::Tensor& dQ,
                              ::brotensor::Tensor& dK,
                              ::brotensor::Tensor& dV);

namespace {

__global__ void fa_fp16_add_inplace_kernel(__half* __restrict__ dst,
                                           const __half* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

__global__ void fa_bf16_add_inplace_kernel(__nv_bfloat16* __restrict__ dst,
                                           const __nv_bfloat16* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

__global__ void fa_lbb_dx_bf16_kernel(const __nv_bfloat16* __restrict__ W,
                                      const __nv_bfloat16* __restrict__ dY,
                                      __nv_bfloat16* __restrict__ dX,
                                      int B, int out_dim, int in_dim) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= in_dim) return;
    const __nv_bfloat16* dY_row = dY + static_cast<size_t>(b) * out_dim;
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        acc += __bfloat162float(W[static_cast<size_t>(i) * in_dim + j]) *
               __bfloat162float(dY_row[i]);
    }
    dX[static_cast<size_t>(b) * in_dim + j] = __float2bfloat16(acc);
}

__global__ void fa_lbb_dw_bf16_kernel(const __nv_bfloat16* __restrict__ dY,
                                      const __nv_bfloat16* __restrict__ X,
                                      float* __restrict__ dW_scratch,
                                      int B, int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += __bfloat162float(dY[static_cast<size_t>(b) * out_dim + i]) *
               __bfloat162float(X [static_cast<size_t>(b) * in_dim  + j]);
    }
    dW_scratch[static_cast<size_t>(i) * in_dim + j] = acc;
}

__global__ void fa_lbb_db_bf16_kernel(const __nv_bfloat16* __restrict__ dY,
                                      float* __restrict__ dB_scratch,
                                      int B, int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += __bfloat162float(dY[static_cast<size_t>(b) * out_dim + i]);
    }
    dB_scratch[i] = acc;
}

__global__ void fa_add_fp32_into_bf16_kernel(const float* __restrict__ src,
                                             __nv_bfloat16* __restrict__ dst,
                                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

void fa_linear_backward_batched_bf16(const Tensor& W, const Tensor& X_BD,
                                     const Tensor& dY_BD,
                                     Tensor& dX_BD, Tensor& dW, Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != Dtype::BF16) {
        dX_BD.resize(B, in_dim, Dtype::BF16);
    }
    if (B == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    if (in_dim > 0 && out_dim > 0) {
        dim3 block(64, 1);
        constexpr int kMaxGridY = 65535;
        const int grid_x = (in_dim + 63) / 64;
        const auto* dY_p = static_cast<const __nv_bfloat16*>(dY_BD.data);
        auto*       dX_p = static_cast<__nv_bfloat16*>(dX_BD.data);
        for (int b0 = 0; b0 < B; b0 += kMaxGridY) {
            const int b_chunk = (B - b0) < kMaxGridY ? (B - b0) : kMaxGridY;
            dim3 grid(grid_x, b_chunk);
            fa_lbb_dx_bf16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(W.data),
                dY_p + static_cast<size_t>(b0) * out_dim,
                dX_p + static_cast<size_t>(b0) * in_dim,
                b_chunk, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    if (out_dim > 0 && in_dim > 0) {
        const int dw_n = out_dim * in_dim;
        float* d_dw_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw_scratch),
                                        dw_n * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        fa_lbb_dw_bf16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            static_cast<const __nv_bfloat16*>(X_BD.data),
            d_dw_scratch, B, out_dim, in_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (dw_n + 255) / 256;
        fa_add_fp32_into_bf16_kernel<<<blocks_fold, 256, 0, stream>>>(
            d_dw_scratch, static_cast<__nv_bfloat16*>(dW.data), dw_n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dw_scratch);
    }
    if (out_dim > 0) {
        float* d_db_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db_scratch),
                                        out_dim * sizeof(float)));
        const int blocks = (out_dim + 255) / 256;
        fa_lbb_db_bf16_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            d_db_scratch, B, out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (out_dim + 255) / 256;
        fa_add_fp32_into_bf16_kernel<<<blocks_fold, 256, 0, stream>>>(
            d_db_scratch, static_cast<__nv_bfloat16*>(dB.data), out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_db_scratch);
    }
}

} // namespace

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
    linear_forward_batched_fp16(Wk, bk, ctx, K_out);
    linear_forward_batched_fp16(Wv, bv, ctx, V_out);
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

    Tensor Qp = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor Op = Tensor::empty_on(Device::CUDA, Lq, D, dt);

    linear_forward_batched_fp16(Wq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_fp16(Wo, bo, Op, O);
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

    Tensor Kp = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor Vp = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    flash_attention_project_kv(kv_src, Wk, bk, Wv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_forward(
        X, Kp, Vp, Wq, bq, Wo, bo, d_mask, num_heads, causal, O);
}

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

    Tensor Qp = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor Op = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);

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

    Tensor Kp = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
    Tensor Vp = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    Tensor Q = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor K = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor V = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    linear_forward_batched_fp16(Wq, bq, X,      Q);
    linear_forward_batched_fp16(Wk, bk, kv_src, K);
    linear_forward_batched_fp16(Wv, bv, kv_src, V);

    Tensor O_attn = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    flash_attention_forward(Q, K, V, d_mask, num_heads, causal, O_attn);

    Tensor dO_attn = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    {
        Tensor scratch_db = Tensor::empty_on(Device::CUDA, 0, 0, dt);
        const bool has_bo = (bo != nullptr);
        if (!has_bo) {
            scratch_db.resize(D, 1, dt);
            scratch_db.zero();
        }
        if (bf16) {
            fa_linear_backward_batched_bf16(Wo, O_attn, dO, dO_attn, dWo,
                                            has_bo ? *dbo : scratch_db);
        } else {
            linear_backward_batched(Wo, O_attn, dO, dO_attn, dWo,
                                    has_bo ? *dbo : scratch_db);
        }
    }

    Tensor dQ = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor dK = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor dV = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    flash_attention_backward(Q, K, V, O_attn, dO_attn, d_mask, num_heads, causal, dQ, dK, dV);

    auto run_proj_back = [&](const Tensor& W, const Tensor& In,
                             const Tensor& dOut, Tensor& dIn_out,
                             Tensor& dW_acc, const Tensor* b_fwd,
                             Tensor* db_acc) {
        Tensor scratch_db = Tensor::empty_on(Device::CUDA, 0, 0, dt);
        bool has_b = (b_fwd != nullptr);
        if (!has_b) {
            scratch_db.resize(W.rows, 1, dt);
            scratch_db.zero();
        }
        if (bf16) {
            fa_linear_backward_batched_bf16(W, In, dOut, dIn_out, dW_acc,
                                            has_b ? *db_acc : scratch_db);
        } else {
            linear_backward_batched(W, In, dOut, dIn_out, dW_acc,
                                    has_b ? *db_acc : scratch_db);
        }
    };

    Tensor dX_from_Q = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor dX_from_K = Tensor::empty_on(Device::CUDA, Lk, D_ctx, dt);
    Tensor dX_from_V = Tensor::empty_on(Device::CUDA, Lk, D_ctx, dt);

    run_proj_back(Wq, X,      dQ, dX_from_Q, dWq, bq, dbq);
    run_proj_back(Wk, kv_src, dK, dX_from_K, dWk, bk, dbk);
    run_proj_back(Wv, kv_src, dV, dX_from_V, dWv, bv, dbv);

    auto add_into = [&](Tensor& dst, const Tensor& src, int n) {
        const int blocks = (n + 255) / 256;
        if (bf16) {
            fa_bf16_add_inplace_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(dst.data),
                reinterpret_cast<const __nv_bfloat16*>(src.data), n);
        } else {
            fa_fp16_add_inplace_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<__half*>(dst.data),
                reinterpret_cast<const __half*>(src.data), n);
        }
    };
    add_into(dX, dX_from_Q, Lq * D);
    if (self_attn) {
        add_into(dX, dX_from_K, Lq * D);
        add_into(dX, dX_from_V, Lq * D);
    } else {
        add_into(*dCtx, dX_from_K, Lk * D_ctx);
        add_into(*dCtx, dX_from_V, Lk * D_ctx);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda

} // namespace brotensor
