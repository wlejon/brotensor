// FP16 cross-attention forward that exposes a head-averaged attention map
// and accepts an optional pre-softmax logit bias. Non-flash, materialised
// per-head FP32 score matrix in global memory — the cross-attn site is not
// hot in the diffusion loop, and this lets us emit AttnAvg and fuse the
// bias add cheaply. Structurally mirrors the FP32 train path kernels in
// cross_attention.cu but in FP16 with FP32 accumulation, head-averaging
// the softmax probs into AttnAvg instead of returning per-head, and an
// inlined bias add inside the score kernel.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

namespace {

constexpr int ROW_SM_BLOCK = 256;

// Per-head projection from FP16 input + FP16 weight to FP32 per-head buffer.
// In:  (L, Din) FP16. W: (D, Din) FP16. Out: (h*L, dh) FP32.
__global__ void cxa_proj_kernel(const __half* __restrict__ In,
                                const __half* __restrict__ W,
                                float* __restrict__ Out,
                                int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int row_off = hh * dh;
    const __half* xr = In + static_cast<size_t>(i) * Din;
    const __half* wr = W  + static_cast<size_t>(row_off + j) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) {
        acc += __half2float(xr[k]) * __half2float(wr[k]);
    }
    const size_t out_row = static_cast<size_t>(hh) * L + i;
    Out[out_row * dh + j] = acc;
}

// Per-head scores S(i, j) = (Q_h(i) . K_h(j)) / sqrt(dh) + bias(i, j) (if any).
// Qh: (h*Lq, dh) FP32. Kh: (h*Lk, dh) FP32. S: (h*Lq, Lk) FP32. bias: (Lq, Lk) FP32 or null.
__global__ void cxa_scores_kernel(const float* __restrict__ Qh,
                                  const float* __restrict__ Kh,
                                  const float* __restrict__ bias,
                                  float* __restrict__ S,
                                  int Lq, int Lk, int dh, float inv_sqrtdh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || j >= Lk) return;
    const size_t qrow = (static_cast<size_t>(hh) * Lq + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * Lk + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    s *= inv_sqrtdh;
    if (bias) s += bias[static_cast<size_t>(i) * Lk + j];
    const size_t srow = (static_cast<size_t>(hh) * Lq + i) * Lk;
    S[srow + j] = s;
}

// Per-row masked softmax over (h*Lq, Lk). One block per (head, row). Writes
// FP32 attn probs. Mask is length-Lk key validity (FP32, 1=valid, 0=invalid).
__global__ void cxa_row_softmax_kernel(const float* __restrict__ scores,
                                       float* __restrict__ Attn,
                                       const float* __restrict__ mask,
                                       int Lk) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * Lk;
    float* arow = Attn + static_cast<size_t>(row) * Lk;

    float local_max = -1e30f;
    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        const float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
        const float e = expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < Lk; j += blockDim.x) arow[j] = arow[j] * inv;
}

// Y_h(i, k) = sum_j Attn(hh, i, j) * Vh(hh, j, k). Writes Yconcat(i, hh*dh+k).
// Attnh: (h*Lq, Lk) FP32. Vh: (h*Lk, dh) FP32. Y: (Lq, D) FP32.
__global__ void cxa_attn_apply_v_kernel(const float* __restrict__ Attnh,
                                        const float* __restrict__ Vh,
                                        float* __restrict__ Yconcat,
                                        int Lq, int Lk, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * Lq + i) * Lk;
    float acc = 0.0f;
    for (int j = 0; j < Lk; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * Lk + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// O = Yconcat @ Wo^T. Wo is (D, D) FP16. Y is (Lq, D) FP32. O is (Lq, D) FP16.
__global__ void cxa_output_proj_kernel(const float* __restrict__ Y,
                                       const __half* __restrict__ Wo,
                                       __half* __restrict__ O,
                                       int Lq, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Lq || c >= D) return;
    const float* yr = Y + static_cast<size_t>(i) * D;
    const __half* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * __half2float(wr[k]);
    O[static_cast<size_t>(i) * D + c] = __float2half(acc);
}

// AttnAvg(i, j) = (1/H) sum_h Attnh(h, i, j). FP32 -> FP16.
__global__ void cxa_head_average_kernel(const float* __restrict__ Attnh,
                                        __half* __restrict__ AttnAvg,
                                        int Lq, int Lk, int H) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Lq || j >= Lk) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        acc += Attnh[(static_cast<size_t>(hh) * Lq + i) * Lk + j];
    }
    AttnAvg[static_cast<size_t>(i) * Lk + j] = __float2half(acc / static_cast<float>(H));
}

inline void check_fp16(const GpuTensor& t, const char* name) {
    if (t.dtype != Dtype::FP16) {
        throw std::runtime_error(
            std::string("cross_attention_forward_with_attn_gpu requires FP16 ") + name);
    }
}

} // namespace

void cross_attention_forward_with_attn_gpu(const GpuTensor& X,
                                           const GpuTensor& Ctx,
                                           const GpuTensor& Wq, const GpuTensor& Wk,
                                           const GpuTensor& Wv, const GpuTensor& Wo,
                                           const float* d_mask,
                                           const GpuTensor* attn_logit_bias,
                                           int num_heads,
                                           GpuTensor& O,
                                           GpuTensor& AttnAvg) {
    check_fp16(X, "X");
    check_fp16(Ctx, "Ctx");
    check_fp16(Wq, "Wq"); check_fp16(Wk, "Wk");
    check_fp16(Wv, "Wv"); check_fp16(Wo, "Wo");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (H <= 0 || dh * H != D) {
        throw std::runtime_error("cross_attention_forward_with_attn_gpu: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (AttnAvg.rows != Lq || AttnAvg.cols != Lk || AttnAvg.dtype != Dtype::FP16) {
        AttnAvg.resize(Lq, Lk, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    if (attn_logit_bias) {
        if (attn_logit_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "cross_attention_forward_with_attn_gpu: attn_logit_bias must be FP32");
        }
        if (attn_logit_bias->rows != Lq || attn_logit_bias->cols != Lk) {
            throw std::runtime_error(
                "cross_attention_forward_with_attn_gpu: attn_logit_bias must be (Lq, Lk)");
        }
    }

    const __half* X_h   = reinterpret_cast<const __half*>(X.data);
    const __half* Ctx_h = reinterpret_cast<const __half*>(Ctx.data);
    const __half* Wq_h  = reinterpret_cast<const __half*>(Wq.data);
    const __half* Wk_h  = reinterpret_cast<const __half*>(Wk.data);
    const __half* Wv_h  = reinterpret_cast<const __half*>(Wv.data);
    const __half* Wo_h  = reinterpret_cast<const __half*>(Wo.data);
    __half* O_h        = reinterpret_cast<__half*>(O.data);
    __half* AttnAvg_h  = reinterpret_cast<__half*>(AttnAvg.data);

    GpuTensor Qh(H * Lq, dh, Dtype::FP32);
    GpuTensor Kh(H * Lk, dh, Dtype::FP32);
    GpuTensor Vh(H * Lk, dh, Dtype::FP32);
    GpuTensor scores(H * Lq, Lk, Dtype::FP32);
    GpuTensor Attnh(H * Lq, Lk, Dtype::FP32);
    GpuTensor Yconcat(Lq, D, Dtype::FP32);

    const dim3 block(16, 16);

    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cxa_proj_kernel<<<grid, block>>>(X_h, Wq_h, Qh.data, Lq, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lk + block.y - 1) / block.y, H);
        cxa_proj_kernel<<<grid, block>>>(Ctx_h, Wk_h, Kh.data, Lk, Dctx, dh);
        cxa_proj_kernel<<<grid, block>>>(Ctx_h, Wv_h, Vh.data, Lk, Dctx, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    {
        const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
        const float* bias_ptr = attn_logit_bias ? attn_logit_bias->data : nullptr;
        dim3 grid((Lk + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cxa_scores_kernel<<<grid, block>>>(Qh.data, Kh.data, bias_ptr,
                                           scores.data, Lq, Lk, dh, inv_sqrtdh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    cxa_row_softmax_kernel<<<H * Lq, ROW_SM_BLOCK>>>(scores.data, Attnh.data,
                                                     d_mask, Lk);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    {
        dim3 grid((Lk + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y);
        cxa_head_average_kernel<<<grid, block>>>(Attnh.data, AttnAvg_h, Lq, Lk, H);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cxa_attn_apply_v_kernel<<<grid, block>>>(Attnh.data, Vh.data,
                                                 Yconcat.data, Lq, Lk, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid((D  + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y);
        cxa_output_proj_kernel<<<grid, block>>>(Yconcat.data, Wo_h, O_h, Lq, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace brotensor
