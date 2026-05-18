#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int RB_GN_BLOCK = 256;
constexpr int RB_CONV_BLOCK = 256;

// Fused GroupNorm + SiLU (optional in-place into Y). One block per
// (sample, group); same partitioning as group_norm. Writes
// y = silu(gn(x, gamma, beta)) to Y. FP32 accumulation, FP16 storage.
__global__ void gn_silu_fused_kernel(const __half* __restrict__ X,
                                     const __half* __restrict__ gamma,
                                     const __half* __restrict__ beta,
                                     __half* __restrict__ Y,
                                     int C, int spatial,
                                     int channels_per_group,
                                     float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int tile_size = channels_per_group * spatial;
    const int chan_base = g * channels_per_group;
    const int sample_stride = C * spatial;
    const __half* x_tile = X + n * sample_stride + chan_base * spatial;
    __half*       y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = __half2float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }
    __shared__ float s_sum[RB_GN_BLOCK];
    __shared__ float s_sumsq[RB_GN_BLOCK];
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = __half2float(gamma[channel]);
        const float bv = __half2float(beta[channel]);
        const float v  = __half2float(x_tile[i]);
        const float yn = (v - mean) * rstd * gv + bv;
        const float silu = yn / (1.0f + __expf(-yn));
        y_tile[i] = __float2half(silu);
    }
}

// Add a per-(N, C) or per-(C) shift to an NCHW activation in place.
//   shift: (N, C) row-major if has_N == true, else (C,) broadcast across N.
//   spatial: H*W.
__global__ void add_NC_shift_kernel(__half* __restrict__ Y,
                                    const __half* __restrict__ shift,
                                    int N, int C, int spatial, int has_N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * spatial;
    if (idx >= total) return;
    int t = idx / spatial;
    const int c = t % C;
    const int n = t / C;
    const int sidx = has_N ? (n * C + c) : c;
    const float yv = __half2float(Y[idx]);
    const float sv = __half2float(shift[sidx]);
    Y[idx] = __float2half(yv + sv);
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

} // namespace

void resblock_forward_gpu(const GpuTensor& X,
                          const GpuTensor& gamma1, const GpuTensor& beta1,
                          const GpuTensor& W1, const GpuTensor* b1,
                          const GpuTensor* t_emb_shift,
                          const GpuTensor& gamma2, const GpuTensor& beta2,
                          const GpuTensor& W2, const GpuTensor* b2,
                          const GpuTensor* Wskip, const GpuTensor* bskip,
                          int N, int C_in, int C_out, int H, int Wd,
                          int num_groups, float eps,
                          GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || W1.dtype != Dtype::FP16 ||
        gamma2.dtype != Dtype::FP16 || beta2.dtype != Dtype::FP16 ||
        W2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_gpu: all required tensors must be FP16");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_gpu: num_groups must divide C_in and C_out");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_forward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    // Scratch buffers. thread_local static so that repeated calls at the
    // same shapes reuse the underlying cudaMalloc'd storage (GpuTensor::resize
    // is a no-op when shape+dtype match). Destruction at thread exit frees
    // via cudaFree, which is fine.
    thread_local static GpuTensor h1;  // post-GN1+SiLU
    thread_local static GpuTensor h2;  // post-conv1 (+t_shift)
    thread_local static GpuTensor h3;  // post-GN2+SiLU
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    // Leg 1: GN1 → SiLU, fused.
    {
        dim3 grid(num_groups, N, 1);
        gn_silu_fused_kernel<<<grid, RB_GN_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<const __half*>(gamma1.data_fp16()),
            reinterpret_cast<const __half*>(beta1.data_fp16()),
            reinterpret_cast<__half*>(h1.data_fp16()),
            C_in, spatial, C_in / num_groups, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Conv1: 3x3 same. Dispatch through the public conv2d (WMMA path).
    conv2d_forward_gpu(h1, W1, b1,
                       N, C_in, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       h2);

    // Optional t_emb shift: (N, C_out) or (C_out,) added channelwise.
    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        const int total = N * C_out * spatial;
        add_NC_shift_kernel<<<grid_for(total, RB_CONV_BLOCK), RB_CONV_BLOCK>>>(
            reinterpret_cast<__half*>(h2.data_fp16()),
            reinterpret_cast<const __half*>(t_emb_shift->data_fp16()),
            N, C_out, spatial, has_N);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Leg 2: GN2 → SiLU, fused.
    {
        dim3 grid(num_groups, N, 1);
        gn_silu_fused_kernel<<<grid, RB_GN_BLOCK>>>(
            reinterpret_cast<const __half*>(h2.data_fp16()),
            reinterpret_cast<const __half*>(gamma2.data_fp16()),
            reinterpret_cast<const __half*>(beta2.data_fp16()),
            reinterpret_cast<__half*>(h3.data_fp16()),
            C_out, spatial, C_out / num_groups, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Prepare the skip tensor for the conv2 epilogue (post-conv2 add).
    // Reused across calls; conv2d_forward_gpu will resize as needed.
    thread_local static GpuTensor skip_scratch;
    if (Wskip != nullptr) {
        if (Wskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_gpu: Wskip must be FP16");
        }
        // 1x1 conv through the public path (WMMA implicit-GEMM).
        conv2d_forward_gpu(X, *Wskip, bskip,
                           N, C_in, H, Wd,
                           C_out, 1, 1,
                           /*stride*/1, 1,
                           /*pad*/0, 0,
                           /*dil*/1, 1,
                           skip_scratch);
    }

    // Conv2 (3x3 same) → Y, then fuse-in the skip via add_inplace_gpu.
    conv2d_forward_gpu(h3, W2, b2,
                       N, C_out, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       Y);
    if (Wskip == nullptr) {
        // skip is X itself (C_in == C_out so shapes match).
        add_inplace_gpu(Y, X);
    } else {
        add_inplace_gpu(Y, skip_scratch);
    }
}

void resblock_forward_int8w_fp16_gpu(const GpuTensor& X,
                                     const GpuTensor& gamma1, const GpuTensor& beta1,
                                     const GpuTensor& W1_int8, const GpuTensor& s1,
                                     const GpuTensor* b1,
                                     const GpuTensor* t_emb_shift,
                                     const GpuTensor& gamma2, const GpuTensor& beta2,
                                     const GpuTensor& W2_int8, const GpuTensor& s2,
                                     const GpuTensor* b2,
                                     const GpuTensor* Wskip_int8, const GpuTensor* sskip,
                                     const GpuTensor* bskip,
                                     int N, int C_in, int C_out, int H, int Wd,
                                     int num_groups, float eps,
                                     GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || gamma2.dtype != Dtype::FP16 ||
        beta2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: activation/norm tensors must be FP16");
    }
    if (W1_int8.dtype != Dtype::INT8 || W2_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1/W2 must be INT8");
    }
    if (s1.dtype != Dtype::FP32 || s2.dtype != Dtype::FP32) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: scales s1/s2 must be FP32");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: num_groups must divide C_in and C_out");
    }
    if (W1_int8.rows != C_out || W1_int8.cols != C_in * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1_int8 shape mismatch");
    }
    if (W2_int8.rows != C_out || W2_int8.cols != C_out * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W2_int8 shape mismatch");
    }
    if (s1.rows != C_out || s1.cols != 1 || s2.rows != C_out || s2.cols != 1) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: s1/s2 must be (C_out, 1)");
    }
    if (b1 && b1->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b1 must be FP16");
    }
    if (b2 && b2->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b2 must be FP16");
    }
    if (Wskip_int8 == nullptr) {
        if (C_in != C_out) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip required when C_in != C_out");
        }
    } else {
        if (Wskip_int8->dtype != Dtype::INT8) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 must be INT8");
        }
        if (Wskip_int8->rows != C_out || Wskip_int8->cols != C_in) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 shape mismatch");
        }
        if (sskip == nullptr || sskip->dtype != Dtype::FP32 ||
            sskip->rows != C_out || sskip->cols != 1) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: sskip must be FP32 (C_out, 1)");
        }
        if (bskip && bskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: bskip must be FP16");
        }
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    thread_local static GpuTensor h1;
    thread_local static GpuTensor h2;
    thread_local static GpuTensor h3;
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    // Leg 1: GN1 → SiLU.
    {
        dim3 grid(num_groups, N, 1);
        gn_silu_fused_kernel<<<grid, RB_GN_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<const __half*>(gamma1.data_fp16()),
            reinterpret_cast<const __half*>(beta1.data_fp16()),
            reinterpret_cast<__half*>(h1.data_fp16()),
            C_in, spatial, C_in / num_groups, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Conv1: 3x3 same, INT8 weights.
    conv2d_int8w_fp16_forward_gpu(h1, W1_int8, s1, b1,
                                  N, C_in, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  h2);

    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        const int total = N * C_out * spatial;
        add_NC_shift_kernel<<<grid_for(total, RB_CONV_BLOCK), RB_CONV_BLOCK>>>(
            reinterpret_cast<__half*>(h2.data_fp16()),
            reinterpret_cast<const __half*>(t_emb_shift->data_fp16()),
            N, C_out, spatial, has_N);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Leg 2: GN2 → SiLU.
    {
        dim3 grid(num_groups, N, 1);
        gn_silu_fused_kernel<<<grid, RB_GN_BLOCK>>>(
            reinterpret_cast<const __half*>(h2.data_fp16()),
            reinterpret_cast<const __half*>(gamma2.data_fp16()),
            reinterpret_cast<const __half*>(beta2.data_fp16()),
            reinterpret_cast<__half*>(h3.data_fp16()),
            C_out, spatial, C_out / num_groups, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    thread_local static GpuTensor skip_scratch;
    if (Wskip_int8 != nullptr) {
        conv2d_int8w_fp16_forward_gpu(X, *Wskip_int8, *sskip, bskip,
                                      N, C_in, H, Wd,
                                      C_out, 1, 1,
                                      1, 1, 0, 0, 1, 1, /*groups*/1,
                                      skip_scratch);
    }

    conv2d_int8w_fp16_forward_gpu(h3, W2_int8, s2, b2,
                                  N, C_out, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  Y);
    if (Wskip_int8 == nullptr) {
        add_inplace_gpu(Y, X);
    } else {
        add_inplace_gpu(Y, skip_scratch);
    }
}

namespace {

// Accumulate per-(n, c) HW sum of an NCHW FP16 tensor into a (N, C_out) FP16
// shift-gradient. Each block owns one (n, c); threads reduce HW elements.
// FP32 accumulation; result is added (folded) into d_shift[n, c].
__global__ void rb_sum_hw_per_NC_fp16(const __half* __restrict__ dh2,
                                      __half* __restrict__ d_shift,
                                      int N, int C, int spatial) {
    const int nc = blockIdx.x;
    if (nc >= N * C) return;
    const int n = nc / C;
    const int c = nc - n * C;
    const __half* row = dh2 + (n * C + c) * spatial;

    __shared__ float s_buf[RB_GN_BLOCK];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < spatial; i += blockDim.x) {
        acc += __half2float(row[i]);
    }
    s_buf[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float prev = __half2float(d_shift[nc]);
        d_shift[nc] = __float2half(prev + s_buf[0]);
    }
}

} // namespace

void resblock_backward_gpu(const GpuTensor& X,
                           const GpuTensor& gamma1, const GpuTensor& beta1,
                           const GpuTensor& W1, const GpuTensor* b1,
                           const GpuTensor* t_emb_shift,
                           const GpuTensor& gamma2, const GpuTensor& beta2,
                           const GpuTensor& W2, const GpuTensor* b2,
                           const GpuTensor* Wskip, const GpuTensor* bskip,
                           int N, int C_in, int C_out, int H, int Wd,
                           int num_groups, float eps,
                           const GpuTensor& dY,
                           GpuTensor& dX,
                           GpuTensor& dGamma1, GpuTensor& dBeta1,
                           GpuTensor& dW1, GpuTensor* db1,
                           GpuTensor* dt_emb_shift,
                           GpuTensor& dGamma2, GpuTensor& dBeta2,
                           GpuTensor& dW2, GpuTensor* db2,
                           GpuTensor* dWskip, GpuTensor* dbskip) {
    if (X.dtype != Dtype::FP16 || dY.dtype != Dtype::FP16 ||
        gamma1.dtype != Dtype::FP16 || beta1.dtype != Dtype::FP16 ||
        W1.dtype != Dtype::FP16 ||
        gamma2.dtype != Dtype::FP16 || beta2.dtype != Dtype::FP16 ||
        W2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_backward_gpu: all required tensors must be FP16");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_backward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    if (dY.rows != N || dY.cols != C_out * spatial) {
        throw std::runtime_error("resblock_backward_gpu: dY shape mismatch");
    }
    if (dX.rows != N || dX.cols != C_in * spatial || dX.dtype != Dtype::FP16) {
        dX.resize(N, C_in * spatial, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    // ── Recompute forward intermediates we need:
    //   h1_pre_silu = GN1(X);  h1 = SiLU(h1_pre_silu)
    //   h2_pre_t    = conv1(h1, W1, b1)
    //   h2          = h2_pre_t + broadcast(t_emb_shift)   (if t_emb_shift)
    //   h3_pre_silu = GN2(h2); h3 = SiLU(h3_pre_silu)
    GpuTensor h1_pre_silu, h1;
    group_norm_forward_gpu(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps,
                           h1_pre_silu);
    silu_forward_gpu(h1_pre_silu, h1);

    GpuTensor h2;
    conv2d_forward_gpu(h1, W1, b1, N, C_in, H, Wd,
                       C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        const int total = N * C_out * spatial;
        add_NC_shift_kernel<<<grid_for(total, RB_CONV_BLOCK), RB_CONV_BLOCK>>>(
            reinterpret_cast<__half*>(h2.data_fp16()),
            reinterpret_cast<const __half*>(t_emb_shift->data_fp16()),
            N, C_out, spatial, has_N);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    GpuTensor h3_pre_silu, h3;
    group_norm_forward_gpu(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps,
                           h3_pre_silu);
    silu_forward_gpu(h3_pre_silu, h3);

    // ── Conv2 backward: dh3 (input grad), dW2 += grad, db2 += grad.
    GpuTensor dh3(N, C_out * spatial, Dtype::FP16);
    conv2d_backward_input_gpu(W2, dY, N, C_out, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh3);
    conv2d_backward_weight_gpu(h3, dY, N, C_out, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) {
        conv2d_backward_bias_gpu(dY, N, C_out, H, Wd, *db2);
    }

    // ── SiLU2 backward over h3_pre_silu.
    GpuTensor dh3_pre_silu;
    silu_backward_gpu(h3_pre_silu, dh3, dh3_pre_silu);

    // ── GN2 backward: writes dh2 (overwritten), accumulates dGamma2, dBeta2.
    GpuTensor dh2;
    group_norm_backward_gpu(h2, gamma2, dh3_pre_silu, N, C_out, H, Wd,
                            num_groups, eps, dh2, dGamma2, dBeta2);

    // ── t_emb_shift backward: channel-axis reduction of dh2.
    if (t_emb_shift && dt_emb_shift) {
        if (dt_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: dt_emb_shift must be FP16");
        }
        const bool has_N = (t_emb_shift->rows == N && t_emb_shift->cols == C_out);
        if (has_N) {
            // (N, C_out) — sum over HW per (n, c), accumulate into dt_emb_shift.
            const int blocks = N * C_out;
            rb_sum_hw_per_NC_fp16<<<blocks, RB_GN_BLOCK>>>(
                reinterpret_cast<const __half*>(dh2.data_fp16()),
                reinterpret_cast<__half*>(dt_emb_shift->data_fp16()),
                N, C_out, spatial);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
        } else {
            // (C_out,) — sum over (N, H, W) per channel. Reuse conv bias bwd.
            conv2d_backward_bias_gpu(dh2, N, C_out, H, Wd, *dt_emb_shift);
        }
    }

    // ── Conv1 backward: dh1, dW1 +=, db1 +=.
    GpuTensor dh1(N, C_in * spatial, Dtype::FP16);
    conv2d_backward_input_gpu(W1, dh2, N, C_in, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh1);
    conv2d_backward_weight_gpu(h1, dh2, N, C_in, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) {
        conv2d_backward_bias_gpu(dh2, N, C_out, H, Wd, *db1);
    }

    // ── SiLU1 backward over h1_pre_silu.
    GpuTensor dh1_pre_silu;
    silu_backward_gpu(h1_pre_silu, dh1, dh1_pre_silu);

    // ── GN1 backward: writes dX (overwritten = dX_from_main_path),
    //                  accumulates dGamma1, dBeta1.
    group_norm_backward_gpu(X, gamma1, dh1_pre_silu, N, C_in, H, Wd,
                            num_groups, eps, dX, dGamma1, dBeta1);

    // ── Skip path backward, then sum into dX.
    if (Wskip == nullptr) {
        // identity: dX += dY (shapes already match since C_in == C_out).
        add_inplace_gpu(dX, dY);
    } else {
        if (Wskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_backward_gpu: Wskip must be FP16");
        }
        GpuTensor dX_skip(N, C_in * spatial, Dtype::FP16);
        conv2d_backward_input_gpu(*Wskip, dY, N, C_in, H, Wd,
                                  C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX_skip);
        if (dWskip) {
            conv2d_backward_weight_gpu(X, dY, N, C_in, H, Wd,
                                       C_out, 1, 1, 1, 1, 0, 0, 1, 1, *dWskip);
        }
        if (dbskip) {
            conv2d_backward_bias_gpu(dY, N, C_out, H, Wd, *dbskip);
        }
        add_inplace_gpu(dX, dX_skip);
    }
}

} // namespace brotensor
