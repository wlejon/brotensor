#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int CONV_BLOCK = 256;

template <typename T>
__device__ inline float load_f32(const T* p);
template <> __device__ inline float load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float load_f32<__half>(const __half* p) { return __half2float(*p); }

template <typename T>
__device__ inline void store_f32(T* p, float v);
template <> __device__ inline void store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void store_f32<__half>(__half* p, float v) { *p = __float2half(v); }

// One thread per output element. Naive direct-conv reduction over
// (C_in, kH, kW). FP32 accumulator; storage dtype T (FP16 or FP32).
template <typename T>
__global__ void conv2d_forward_kernel(
        const T* __restrict__ X,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,   // may be null
        T* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Unflatten idx → (n, oc, oh, ow).
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int oc = t % C_out;
    const int n  = t / C_out;

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    float acc = 0.0f;
    // Weight base for this output channel: (oc, 0, 0, 0) in OIHW.
    const int w_oc_base = oc * C_in * kH * kW;
    // Input base for this sample.
    const int x_n_base = n * C_in * H * W;

    for (int ic = 0; ic < C_in; ++ic) {
        const int w_ic_base = w_oc_base + ic * kH * kW;
        const int x_ic_base = x_n_base + ic * H * W;
        for (int kh = 0; kh < kH; ++kh) {
            const int in_h = in_h_origin + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int in_w = in_w_origin + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                const float x_v = load_f32<T>(&X[x_ic_base + in_h * W + in_w]);
                const float w_v = load_f32<T>(&Wt[w_ic_base + kh * kW + kw]);
                acc += x_v * w_v;
            }
        }
    }
    if (bias) {
        acc += load_f32<T>(&bias[oc]);
    }
    store_f32<T>(&Y[idx], acc);
}

// One thread per input pixel. Gather form of backward-w.r.t.-input.
// FP32 only. No atomics.
__global__ void conv2d_backward_input_kernel(
        const float* __restrict__ Wt,
        const float* __restrict__ dY,
        float* __restrict__ dX,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Unflatten idx → (n, c_in, i, j).
    const int j = idx % W;
    int t = idx / W;
    const int i = t % H;
    t /= H;
    const int c_in = t % C_in;
    const int n    = t / C_in;

    float acc = 0.0f;

    // For each kernel tap (kh, kw), invert the forward index relation:
    //   in_h = stride_h * i_out - pad_h + dil_h * kh = i
    //   ⇒ i_out = (i + pad_h - dil_h * kh) / stride_h, must be exact + in [0, H_out)
    for (int kh = 0; kh < kH; ++kh) {
        const int num_h = i + pad_h - dil_h * kh;
        if (num_h < 0) continue;
        if (num_h % stride_h != 0) continue;
        const int i_out = num_h / stride_h;
        if (i_out < 0 || i_out >= H_out) continue;
        for (int kw = 0; kw < kW; ++kw) {
            const int num_w = j + pad_w - dil_w * kw;
            if (num_w < 0) continue;
            if (num_w % stride_w != 0) continue;
            const int j_out = num_w / stride_w;
            if (j_out < 0 || j_out >= W_out) continue;

            // Sum over c_out of dY[n, c_out, i_out, j_out] * Wt[c_out, c_in, kh, kw].
            for (int c_out = 0; c_out < C_out; ++c_out) {
                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                const int w_idx  = ((c_out * C_in + c_in) * kH + kh) * kW + kw;
                acc += dY[dy_idx] * Wt[w_idx];
            }
        }
    }
    dX[idx] = acc;
}

// One thread per (c_out, c_in, kh, kw) element of dWt. Iterates (n, i_out,
// j_out) and accumulates into a single dWt slot. FP32 only. No atomics.
__global__ void conv2d_backward_weight_kernel(
        const float* __restrict__ X,
        const float* __restrict__ dY,
        float* __restrict__ dWt,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Unflatten idx → (c_out, c_in, kh, kw) in OIHW layout.
    const int kw = idx % kW;
    int t = idx / kW;
    const int kh = t % kH;
    t /= kH;
    const int c_in  = t % C_in;
    const int c_out = t / C_in;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            const int in_h = i_out * stride_h - pad_h + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int j_out = 0; j_out < W_out; ++j_out) {
                const int in_w = j_out * stride_w - pad_w + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                const int x_idx  = ((n * C_in  + c_in)  * H     + in_h)  * W     + in_w;
                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                acc += dY[dy_idx] * X[x_idx];
            }
        }
    }
    // Accumulate into caller's dWt (caller zeros).
    dWt[idx] += acc;
}

constexpr int BIAS_BLOCK = 256;

// One block per c_out; threads stride-loop over (n, i_out, j_out) summing
// dY[n, c_out, i_out, j_out] in FP32, then a shared-mem reduction folds the
// per-thread partials and thread 0 adds into dB[c_out].
__global__ void conv2d_backward_bias_kernel(
        const float* __restrict__ dY,
        float* __restrict__ dB,
        int N, int C_out, int H_out, int W_out) {
    const int c_out = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial = H_out * W_out;
    const int total_per_chan = N * spatial;

    float acc = 0.0f;
    for (int idx = tid; idx < total_per_chan; idx += blockDim.x) {
        const int n  = idx / spatial;
        const int sp = idx - n * spatial;
        const int dy_idx = (n * C_out + c_out) * spatial + sp;
        acc += dY[dy_idx];
    }

    __shared__ float s_acc[BIAS_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) dB[c_out] += s_acc[0];
}

} // namespace

void conv2d_forward_gpu(const GpuTensor& X,
                        const GpuTensor& Wt,
                        const GpuTensor* bias,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dil_h, int dil_w,
                        GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_forward_gpu: X must be FP16 or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward_gpu: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward_gpu: bias dtype must match X");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    if (X.dtype == Dtype::FP16) {
        const __half* x_p  = reinterpret_cast<const __half*>(X.data_fp16());
        const __half* w_p  = reinterpret_cast<const __half*>(Wt.data_fp16());
        const __half* b_p  = bias ? reinterpret_cast<const __half*>(bias->data_fp16())
                                  : nullptr;
        __half* y_p        = reinterpret_cast<__half*>(Y.data_fp16());
        conv2d_forward_kernel<__half><<<blocks, CONV_BLOCK>>>(
            x_p, w_p, b_p, y_p,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else {
        const float* b_p = bias ? bias->data : nullptr;
        conv2d_forward_kernel<float><<<blocks, CONV_BLOCK>>>(
            X.data, Wt.data, b_p, Y.data,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_input_gpu(const GpuTensor& Wt,
                               const GpuTensor& dY,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w,
                               GpuTensor& dX) {
    if (Wt.dtype != Dtype::FP32 || dY.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_input_gpu: Wt and dY must be FP32");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input_gpu: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, in_cols, Dtype::FP32);
    }
    const int total = N * in_cols;
    if (total == 0) return;

    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    conv2d_backward_input_kernel<<<blocks, CONV_BLOCK>>>(
        Wt.data, dY.data, dX.data,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_weight_gpu(const GpuTensor& X,
                                const GpuTensor& dY,
                                int N, int C_in, int H, int W,
                                int C_out, int kH, int kW,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dil_h, int dil_w,
                                GpuTensor& dWt) {
    if (X.dtype != Dtype::FP32 || dY.dtype != Dtype::FP32 || dWt.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_weight_gpu: X, dY, dWt must be FP32");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight_gpu: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != C_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight_gpu: dWt shape mismatch");
    }
    const int total = C_out * C_in * kH * kW;
    if (total == 0) return;

    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    conv2d_backward_weight_kernel<<<blocks, CONV_BLOCK>>>(
        X.data, dY.data, dWt.data,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_bias_gpu(const GpuTensor& dY,
                              int N, int C_out, int H_out, int W_out,
                              GpuTensor& dB) {
    if (dY.dtype != Dtype::FP32 || dB.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_bias_gpu: dY and dB must be FP32");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias_gpu: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    conv2d_backward_bias_kernel<<<C_out, BIAS_BLOCK>>>(
        dY.data, dB.data, N, C_out, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
