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

    // Scratch buffers.
    GpuTensor h1(N, C_in  * spatial, Dtype::FP16);   // post-GN1+SiLU
    GpuTensor h2(N, C_out * spatial, Dtype::FP16);   // post-conv1 (+t_shift)
    GpuTensor h3(N, C_out * spatial, Dtype::FP16);   // post-GN2+SiLU

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
    GpuTensor skip_scratch;
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

} // namespace brotensor
