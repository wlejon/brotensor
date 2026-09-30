// Native AMD HIP attention and cross-attention core for RDNA 3.5 (Wave32).
//
// Implements:
// - attention_forward / attention_backward (standard FP32 attention)
// - cross_attention_forward / cross_attention_backward
// - cross_attention_forward_train / cross_attention_forward_with_attn
// - self_attention_forward / self_attention_forward_train / self_attention_backward
// - self_attention_bias_forward / rel_pos_bias_xl_forward
// - attention_token_moments (mean and variance across attention tokens)

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/hip_check.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brotensor::detail::cuda {
void attention_backward(const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                        const ::brotensor::Tensor& V, const ::brotensor::Tensor& Attn,
                        const ::brotensor::Tensor& Y_pre_Wo, const ::brotensor::Tensor& Wq,
                        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor& Wv,
                        const ::brotensor::Tensor& Wo, const float* d_mask,
                        ::brotensor::Tensor& dX, ::brotensor::Tensor& dWq,
                        ::brotensor::Tensor& dWk, ::brotensor::Tensor& dWv,
                        ::brotensor::Tensor& dWo);
void cross_attention_backward(const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Ctx, const ::brotensor::Tensor& Qh,
                              const ::brotensor::Tensor& Kh, const ::brotensor::Tensor& Vh,
                              const ::brotensor::Tensor& Attnh, const ::brotensor::Tensor& Yconcat,
                              const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                              const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                              const float* d_mask, int num_heads,
                              ::brotensor::Tensor& dX, ::brotensor::Tensor& dCtx,
                              ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                              ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
void cross_attention_forward_with_attn(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Ctx,
                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                       const float* d_mask, const ::brotensor::Tensor* attn_logit_bias,
                                       int num_heads,
                                       ::brotensor::Tensor& O, ::brotensor::Tensor& AttnAvg);
void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                 const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                                 const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O);
void rel_pos_bias_xl_forward(const ::brotensor::Tensor& Qv, const ::brotensor::Tensor& Pk,
                             int num_heads, int head_dim, ::brotensor::Tensor& Bias);
void self_attention_bias_int8w_fp16(const ::brotensor::Tensor& X,
                                    const ::brotensor::Tensor& Wq_int8, const ::brotensor::Tensor& sq,
                                    const ::brotensor::Tensor& Wk_int8, const ::brotensor::Tensor& sk,
                                    const ::brotensor::Tensor& Wv_int8, const ::brotensor::Tensor& sv,
                                    const ::brotensor::Tensor& Wo_int8, const ::brotensor::Tensor& so,
                                    const float* d_mask, const ::brotensor::Tensor* attn_bias,
                                    int num_heads, float scale, ::brotensor::Tensor& O);
void self_attention_decomposed_rel_pos_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h, const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        ::brotensor::Tensor& O);
void self_attention_decomposed_rel_pos_windowed_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h, const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        ::brotensor::Tensor& O);
} // namespace brotensor::detail::cuda

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

// Forward declarations from flash_attention.hip and mha.hip
void hip_flash_attention_qkvo_forward(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor* Ctx,
                                      const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
                                      const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
                                      const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
                                      const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
                                      const float* d_mask,
                                      int num_heads,
                                      bool causal,
                                      ::brotensor::Tensor& O);

void hip_mha_forward(const ::brotensor::Tensor& X,
                     const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                     const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                     const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                     const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                     const float* d_mask,
                     int num_heads,
                     ::brotensor::Tensor& Qh,
                     ::brotensor::Tensor& Kh,
                     ::brotensor::Tensor& Vh,
                     ::brotensor::Tensor& Attnh,
                     ::brotensor::Tensor& Yconcat,
                     ::brotensor::Tensor& Output);

void hip_mha_backward(const ::brotensor::Tensor& dOutput,
                      const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& Qh,
                      const ::brotensor::Tensor& Kh,
                      const ::brotensor::Tensor& Vh,
                      const ::brotensor::Tensor& Attnh,
                      const ::brotensor::Tensor& Yconcat,
                      const ::brotensor::Tensor& Wq,
                      const ::brotensor::Tensor& Wk,
                      const ::brotensor::Tensor& Wv,
                      const ::brotensor::Tensor& Wo,
                      const float* d_mask,
                      int num_heads,
                      ::brotensor::Tensor& dX,
                      ::brotensor::Tensor& dWq,
                      ::brotensor::Tensor& dWk,
                      ::brotensor::Tensor& dWv,
                      ::brotensor::Tensor& dWo,
                      ::brotensor::Tensor* dbq,
                      ::brotensor::Tensor* dbk,
                      ::brotensor::Tensor* dbv,
                      ::brotensor::Tensor* dbo);

namespace {

constexpr int kWave32 = 32;
constexpr int kRowSmBlock = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

__device__ __forceinline__ float wave32_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, kWave32));
    }
    return val;
}

__device__ __forceinline__ float wave32_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, kWave32);
    }
    return val;
}

// ─── Standard FP32 Attention Kernels ─────────────────────────────────────────

__global__ void matmul_xwT_kernel(const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ Y,
                                  int N, int D_in, int D_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= D_out) return;
    const float* xr = X + static_cast<size_t>(i) * D_in;
    const float* wr = W + static_cast<size_t>(j) * D_in;
    float acc = 0.0f;
    for (int k = 0; k < D_in; ++k) acc += xr[k] * wr[k];
    Y[static_cast<size_t>(i) * D_out + j] = acc;
}

__global__ void scores_kernel(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ S,
                              int N, int D, float inv_sqrtd) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    const float* qr = Q + static_cast<size_t>(i) * D;
    const float* kr = K + static_cast<size_t>(j) * D;
    float s = 0.0f;
    for (int k = 0; k < D; ++k) s += qr[k] * kr[k];
    S[static_cast<size_t>(i) * N + j] = s * inv_sqrtd;
}

__global__ void row_masked_softmax_kernel(const float* __restrict__ scores,
                                          float* __restrict__ Attn,
                                          const float* __restrict__ mask,
                                          int N) {
    __shared__ float s_wave[8];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const float* srow = scores + static_cast<size_t>(i) * N;
    float* arow = Attn + static_cast<size_t>(i) * N;

    if (mask && mask[i] < 0.5f) {
        for (int j = tid; j < N; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        local_max = fmaxf(local_max, srow[j]);
    }
    local_max = wave32_max(local_max);
    if (lane == 0) s_wave[warp_id] = local_max;
    __syncthreads();

    float block_max = -1e30f;
    if (tid < num_warps) block_max = s_wave[tid];
    block_max = wave32_max(block_max);
    if (warp_id == 0) s_wave[lane] = block_max;
    __syncthreads();
    const float m = s_wave[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            arow[j] = 0.0f;
            continue;
        }
        const float e = __expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    local_sum = wave32_sum(local_sum);
    if (lane == 0) s_wave[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = s_wave[tid];
    block_sum = wave32_sum(block_sum);
    if (warp_id == 0) s_wave[lane] = block_sum;
    __syncthreads();
    const float sum = s_wave[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    const float* ar = A + static_cast<size_t>(i) * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += ar[k] * B[static_cast<size_t>(k) * N + j];
    C[static_cast<size_t>(i) * N + j] = acc;
}

// ─── Cross-Attention Projections & Softmax ───────────────────────────────────

__global__ void cx_proj_bias_kernel(float* __restrict__ Out, const float* __restrict__ bias,
                                    int L, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] += bias[hh * dh + j];
}

__global__ void cx_proj_f32_kernel(const float* __restrict__ In,
                                   const float* __restrict__ W,
                                   float* __restrict__ Out,
                                   int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int row_off = hh * dh;
    const float* xr = In + static_cast<size_t>(i) * Din;
    const float* wr = W  + static_cast<size_t>(row_off + j) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) acc += xr[k] * wr[k];
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
}

__global__ void cx_scores_f32_kernel(const float* __restrict__ Qh,
                                     const float* __restrict__ Kh,
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
    S[(static_cast<size_t>(hh) * Lq + i) * Lk + j] = s * inv_sqrtdh;
}

__global__ void cx_row_softmax_f32_kernel(const float* __restrict__ scores,
                                          float* __restrict__ Attn,
                                          const float* __restrict__ mask,
                                          int Lq, int Lk) {
    __shared__ float s_wave[8];
    const int row = blockIdx.x;
    const int i_within = row % Lq;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const float* srow = scores + static_cast<size_t>(row) * Lk;
    float* arow = Attn + static_cast<size_t>(row) * Lk;

    if (mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < Lk; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        local_max = fmaxf(local_max, srow[j]);
    }
    local_max = wave32_max(local_max);
    if (lane == 0) s_wave[warp_id] = local_max;
    __syncthreads();

    float block_max = -1e30f;
    if (tid < num_warps) block_max = s_wave[tid];
    block_max = wave32_max(block_max);
    if (warp_id == 0) s_wave[lane] = block_max;
    __syncthreads();
    const float m = s_wave[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            arow[j] = 0.0f;
            continue;
        }
        const float e = __expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    local_sum = wave32_sum(local_sum);
    if (lane == 0) s_wave[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = s_wave[tid];
    block_sum = wave32_sum(block_sum);
    if (warp_id == 0) s_wave[lane] = block_sum;
    __syncthreads();
    const float sum = s_wave[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < Lk; j += blockDim.x) arow[j] = arow[j] * inv;
}

__global__ void cx_attn_apply_v_f32_kernel(const float* __restrict__ Attnh,
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

// ─── Attention Token Moments Kernel ──────────────────────────────────────────

__global__ void attention_token_moments_kernel(
        const __half* __restrict__ Attn,
        int Lq, int Lk,
        int h_lat, int w_lat,
        float* __restrict__ mass,
        float* __restrict__ centroid) {
    const int k = blockIdx.x;
    if (k >= Lk) return;
    const int tid = threadIdx.x;

    float am = 0.0f, ay = 0.0f, ax = 0.0f;
    for (int q = tid; q < Lq; q += blockDim.x) {
        const float a = __half2float(Attn[q * Lk + k]);
        const int y = q / w_lat;
        const int x = q - y * w_lat;
        am += a;
        ay += static_cast<float>(y) * a;
        ax += static_cast<float>(x) * a;
    }
    am = wave32_sum(am);
    ay = wave32_sum(ay);
    ax = wave32_sum(ax);

    __shared__ float s_m[8], s_y[8], s_x[8];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    if (lane == 0) {
        s_m[warp_id] = am;
        s_y[warp_id] = ay;
        s_x[warp_id] = ax;
    }
    __syncthreads();

    float bm = 0.0f, by = 0.0f, bx = 0.0f;
    if (tid < num_warps) {
        bm = s_m[tid];
        by = s_y[tid];
        bx = s_x[tid];
    }
    bm = wave32_sum(bm);
    by = wave32_sum(by);
    bx = wave32_sum(bx);

    if (tid == 0) {
        mass[k] = bm;
        if (bm > 1e-8f) {
            const float inv = 1.0f / bm;
            centroid[k * 2 + 0] = by * inv;
            centroid[k * 2 + 1] = bx * inv;
        } else {
            centroid[k * 2 + 0] = 0.0f;
            centroid[k * 2 + 1] = 0.0f;
        }
    }
}

} // namespace

// ─── Public Functions ────────────────────────────────────────────────────────

void hip_attention_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                           const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                           const float* d_mask,
                           ::brotensor::Tensor& Q, ::brotensor::Tensor& K,
                           ::brotensor::Tensor& V, ::brotensor::Tensor& Attn,
                           ::brotensor::Tensor& Y_pre_Wo, ::brotensor::Tensor& O) {
    const int N = X.rows;
    const int D = X.cols;
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("hip_attention_forward: weight shape mismatch");
    }

    Q.resize(N, D);
    K.resize(N, D);
    V.resize(N, D);
    Attn.resize(N, N);
    Y_pre_Wo.resize(N, D);
    O.resize(N, D);

    hipStream_t stream = cur_stream();
    dim3 b16(16, 16);
    dim3 g_ND = grid2d(D, N, b16);
    dim3 g_NN = grid2d(N, N, b16);

    matmul_xwT_kernel<<<g_ND, b16, 0, stream>>>(static_cast<const float*>(X.data), static_cast<const float*>(Wq.data), static_cast<float*>(Q.data), N, D, D);
    matmul_xwT_kernel<<<g_ND, b16, 0, stream>>>(static_cast<const float*>(X.data), static_cast<const float*>(Wk.data), static_cast<float*>(K.data), N, D, D);
    matmul_xwT_kernel<<<g_ND, b16, 0, stream>>>(static_cast<const float*>(X.data), static_cast<const float*>(Wv.data), static_cast<float*>(V.data), N, D, D);

    ::brotensor::Tensor S = ::brotensor::Tensor::empty_on(X.device, N, N, X.dtype);
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));
    scores_kernel<<<g_NN, b16, 0, stream>>>(static_cast<const float*>(Q.data), static_cast<const float*>(K.data), static_cast<float*>(S.data), N, D, inv_sqrtd);

    row_masked_softmax_kernel<<<N, kRowSmBlock, 0, stream>>>(static_cast<const float*>(S.data), static_cast<float*>(Attn.data), d_mask, N);

    matmul_kernel<<<g_ND, b16, 0, stream>>>(static_cast<const float*>(Attn.data), static_cast<const float*>(V.data), static_cast<float*>(Y_pre_Wo.data), N, N, D);

    matmul_xwT_kernel<<<g_ND, b16, 0, stream>>>(static_cast<const float*>(Y_pre_Wo.data), static_cast<const float*>(Wo.data), static_cast<float*>(O.data), N, D, D);

    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_cross_attention_forward_train(const ::brotensor::Tensor& X,
                                       const ::brotensor::Tensor& Ctx,
                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                       const float* d_mask,
                                       int num_heads,
                                       ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                                       ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                                       ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O) {
    const int Lq = X.rows, Lk = Ctx.rows, Din_q = X.cols, Din_ctx = Ctx.cols, D = Wq.rows;
    const int dh = D / num_heads;
    const int H = num_heads;

    Qh.resize(H * Lq, dh); Kh.resize(H * Lk, dh); Vh.resize(H * Lk, dh);
    Attnh.resize(H * Lq, Lk); Yconcat.resize(Lq, D); O.resize(Lq, D);

    hipStream_t stream = cur_stream();
    dim3 b16(16, 16);

    dim3 g_Q = grid2d(dh, Lq, b16); g_Q.z = H;
    cx_proj_f32_kernel<<<g_Q, b16, 0, stream>>>(static_cast<const float*>(X.data), static_cast<const float*>(Wq.data), static_cast<float*>(Qh.data), Lq, Din_q, dh);

    dim3 g_KV = grid2d(dh, Lk, b16); g_KV.z = H;
    cx_proj_f32_kernel<<<g_KV, b16, 0, stream>>>(static_cast<const float*>(Ctx.data), static_cast<const float*>(Wk.data), static_cast<float*>(Kh.data), Lk, Din_ctx, dh);
    cx_proj_f32_kernel<<<g_KV, b16, 0, stream>>>(static_cast<const float*>(Ctx.data), static_cast<const float*>(Wv.data), static_cast<float*>(Vh.data), Lk, Din_ctx, dh);

    ::brotensor::Tensor S = ::brotensor::Tensor::empty_on(X.device, H * Lq, Lk, X.dtype);
    dim3 g_S = grid2d(Lk, Lq, b16); g_S.z = H;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
    cx_scores_f32_kernel<<<g_S, b16, 0, stream>>>(static_cast<const float*>(Qh.data), static_cast<const float*>(Kh.data), static_cast<float*>(S.data), Lq, Lk, dh, inv_sqrtdh);

    cx_row_softmax_f32_kernel<<<H * Lq, kRowSmBlock, 0, stream>>>(static_cast<const float*>(S.data), static_cast<float*>(Attnh.data), d_mask, Lq, Lk);

    dim3 g_apply = grid2d(dh, Lq, b16); g_apply.z = H;
    cx_attn_apply_v_f32_kernel<<<g_apply, b16, 0, stream>>>(static_cast<const float*>(Attnh.data), static_cast<const float*>(Vh.data), static_cast<float*>(Yconcat.data), Lq, Lk, dh, D);

    dim3 g_out = grid2d(D, Lq, b16);
    matmul_xwT_kernel<<<g_out, b16, 0, stream>>>(static_cast<const float*>(Yconcat.data), static_cast<const float*>(Wo.data), static_cast<float*>(O.data), Lq, D, D);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_cross_attention_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Ctx,
                                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        if (Ctx.dtype != X.dtype) {
            throw std::runtime_error("hip_cross_attention_forward: Ctx dtype must match X dtype");
        }
        hip_flash_attention_qkvo_forward(X, &Ctx, Wq, nullptr, Wk, nullptr, Wv, nullptr, Wo, nullptr, d_mask, num_heads, false, O);
        return;
    }
    if (Ctx.dtype != Dtype::FP32) {
        throw std::runtime_error("hip_cross_attention_forward: Ctx dtype must match X dtype");
    }
    ::brotensor::Tensor Qh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Kh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Vh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Attnh   = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Yconcat = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    hip_cross_attention_forward_train(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                      Qh, Kh, Vh, Attnh, Yconcat, O);
}

void hip_self_attention_forward(const ::brotensor::Tensor& X,
                                const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                const float* d_mask,
                                int num_heads,
                                ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        hip_flash_attention_qkvo_forward(X, nullptr, Wq, nullptr, Wk, nullptr, Wv, nullptr, Wo, nullptr, d_mask, num_heads, false, O);
        return;
    }
    ::brotensor::Tensor Qh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Kh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Vh      = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Attnh   = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    ::brotensor::Tensor Yconcat = ::brotensor::Tensor::empty_on(X.device, 0, 0, Dtype::FP32);
    hip_mha_forward(X, Wq, Wk, Wv, Wo, nullptr, nullptr, nullptr, nullptr, d_mask, num_heads,
                    Qh, Kh, Vh, Attnh, Yconcat, O);
}

void hip_self_attention_forward_train(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                      const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                      const float* d_mask,
                                      int num_heads,
                                      ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                                      ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                                      ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O) {
    hip_mha_forward(X, Wq, Wk, Wv, Wo, nullptr, nullptr, nullptr, nullptr, d_mask, num_heads, Qh, Kh, Vh, Attnh, Yconcat, O);
}

void hip_self_attention_backward(const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                                 const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                                 const ::brotensor::Tensor& Yconcat,
                                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 ::brotensor::Tensor& dX,
                                 ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                                 ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo) {
    hip_mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, d_mask, num_heads,
                     dX, dWq, dWk, dWv, dWo, nullptr, nullptr, nullptr, nullptr);
}

void hip_attention_token_moments(const ::brotensor::Tensor& Attn, int h_lat, int w_lat,
                                 ::brotensor::Tensor& mass, ::brotensor::Tensor& centroid) {
    using ::brotensor::Dtype;
    if (Attn.dtype != Dtype::FP16) {
        throw std::runtime_error("hip_attention_token_moments: Attn must be FP16");
    }
    if (h_lat <= 0 || w_lat <= 0) {
        throw std::runtime_error("hip_attention_token_moments: h_lat and w_lat must be positive");
    }
    const int Lq = h_lat * w_lat;
    const int Lk = Attn.cols;
    if (Attn.rows != Lq) {
        throw std::runtime_error("hip_attention_token_moments: Attn.rows must equal h_lat * w_lat");
    }
    if (mass.rows != Lk || mass.cols != 1 || mass.dtype != Dtype::FP32) {
        mass.resize(Lk, 1, Dtype::FP32);
    }
    if (centroid.rows != Lk || centroid.cols != 2 || centroid.dtype != Dtype::FP32) {
        centroid.resize(Lk, 2, Dtype::FP32);
    }
    if (Lk == 0) return;

    hipStream_t stream = cur_stream();
    attention_token_moments_kernel<<<Lk, 256, 0, stream>>>(
        static_cast<const __half*>(Attn.data),
        Lq, Lk, h_lat, w_lat,
        static_cast<float*>(mass.data),
        static_cast<float*>(centroid.data));
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_attention(::brotensor::detail::OpsVTable& v) {
    v.attention_forward                      = &hip_attention_forward;
    v.attention_backward                     = &::brotensor::detail::cuda::attention_backward;
    v.cross_attention_forward                = &hip_cross_attention_forward;
    v.cross_attention_forward_train          = &hip_cross_attention_forward_train;
    v.cross_attention_backward               = &::brotensor::detail::cuda::cross_attention_backward;
    v.cross_attention_forward_with_attn      = &::brotensor::detail::cuda::cross_attention_forward_with_attn;
    v.self_attention_forward                 = &hip_self_attention_forward;
    v.self_attention_forward_train           = &hip_self_attention_forward_train;
    v.self_attention_backward                = &hip_self_attention_backward;
    v.attention_token_moments                = &hip_attention_token_moments;

    v.self_attention_bias_forward            = &::brotensor::detail::cuda::self_attention_bias_forward;
    v.rel_pos_bias_xl_forward                = &::brotensor::detail::cuda::rel_pos_bias_xl_forward;
    v.self_attention_bias_int8w_fp16         = &::brotensor::detail::cuda::self_attention_bias_int8w_fp16;
    v.self_attention_decomposed_rel_pos_forward = &::brotensor::detail::cuda::self_attention_decomposed_rel_pos_forward;
    v.self_attention_decomposed_rel_pos_windowed_forward = &::brotensor::detail::cuda::self_attention_decomposed_rel_pos_windowed_forward;
}

} // namespace brotensor::detail::hip
