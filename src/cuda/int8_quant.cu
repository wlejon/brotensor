// INT8 weight-only quantisation (W8A16): per-row symmetric INT8 weights with
// per-row FP32 scales, applied via tiled GEMM that dequantises on-the-fly.
//
// Host helper:  quantize_int8_per_row_host
// Device ops:   matmul_int8w_fp16_gpu  (Y(out,B) = dequant(W) @ X)
//               conv2d_int8w_fp16_forward_gpu (implicit-GEMM over OIHW filter)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace brotensor {

// ─── Host quantiser ────────────────────────────────────────────────────────

void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    for (int r = 0; r < out; ++r) {
        const uint16_t* row = W_fp16 + static_cast<size_t>(r) * static_cast<size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + static_cast<size_t>(r) * static_cast<size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    }
}

// ─── Device kernels ────────────────────────────────────────────────────────

namespace {

constexpr int MM_TILE = 16;

// Y(out, B) = (W_int8(out, K) * scale[out]) @ X_fp16(K, B). Tiled in MM_TILE.
__global__ void matmul_int8w_fp16_kernel(const int8_t* __restrict__ W,
                                         const float*  __restrict__ scales,
                                         const __half* __restrict__ X,
                                         __half* __restrict__ Y,
                                         int M, int N, int K) {
    __shared__ float Ws[MM_TILE][MM_TILE];   // dequantised tile of W (row m, col k)
    __shared__ float Xs[MM_TILE][MM_TILE];   // X tile (row k, col n)

    const int row = blockIdx.y * MM_TILE + threadIdx.y;  // out row m
    const int col = blockIdx.x * MM_TILE + threadIdx.x;  // batch col n

    // Cache scale for this row in a register (only valid when row<M).
    const float row_scale = (row < M) ? scales[row] : 0.0f;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int w_col = t * MM_TILE + threadIdx.x;
        const int x_row = t * MM_TILE + threadIdx.y;

        // Load and dequantise one W element per thread.
        if (row < M && w_col < K) {
            Ws[threadIdx.y][threadIdx.x] =
                static_cast<float>(W[row * K + w_col]) * row_scale;
        } else {
            Ws[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (x_row < K && col < N) {
            Xs[threadIdx.y][threadIdx.x] = __half2float(X[x_row * N + col]);
        } else {
            Xs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += Ws[threadIdx.y][k] * Xs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}

// One thread per output element of conv2d. Mirrors conv2d.cu's naive direct
// path; only difference is the W load goes through int8 + per-row scale.
__global__ void conv2d_int8w_fp16_forward_kernel(
        const __half* __restrict__ X,
        const int8_t* __restrict__ W,
        const float*  __restrict__ scales,
        const __half* __restrict__ bias,   // may be null
        __half* __restrict__ Y,
        int N, int C_in, int H, int W_in_,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int /*groups*/, int Cg_in, int Cg_out,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int oc = t % C_out;
    const int n  = t / C_out;

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    const float scale = scales[oc];

    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const int w_oc_base = oc * Cg_in * kH * kW;
    const int x_n_base = n * C_in * H * W_in_;

    float acc = 0.0f;
    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const int w_ic_base = w_oc_base + ic_local * kH * kW;
        const int x_ic_base = x_n_base + ic * H * W_in_;
        for (int kh = 0; kh < kH; ++kh) {
            const int in_h = in_h_origin + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int in_w = in_w_origin + kw * dil_w;
                if (in_w < 0 || in_w >= W_in_) continue;
                const float xv = __half2float(X[x_ic_base + in_h * W_in_ + in_w]);
                const float wv = static_cast<float>(W[w_ic_base + kh * kW + kw]) * scale;
                acc += xv * wv;
            }
        }
    }
    if (bias) acc += __half2float(bias[oc]);
    Y[idx] = __float2half(acc);
}

} // namespace

// ─── Device ops ────────────────────────────────────────────────────────────

void matmul_int8w_fp16_gpu(const GpuTensor& W_int8,
                           const GpuTensor& scales,
                           const GpuTensor& X,
                           GpuTensor& Y) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: W_int8 must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: scales must be FP32");
    }
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: X must be FP16");
    }
    const int M = W_int8.rows;   // out
    const int K = W_int8.cols;   // in
    if (X.rows != K) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: K mismatch (W.cols != X.rows)");
    }
    if (scales.rows != M || scales.cols != 1) {
        throw std::runtime_error("matmul_int8w_fp16_gpu: scales shape must be (out, 1)");
    }
    const int Nb = X.cols;       // batch
    if (Y.rows != M || Y.cols != Nb || Y.dtype != Dtype::FP16) {
        Y.resize(M, Nb, Dtype::FP16);
    }
    if (M == 0 || Nb == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (K == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(Y.data, 0, Y.bytes(), stream));
        return;
    }

    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((Nb + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
    matmul_int8w_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const int8_t*>(W_int8.data),
        scales.data,
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        M, Nb, K);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv2d_int8w_fp16_forward_gpu(const GpuTensor& X,
                                   const GpuTensor& W_int8,
                                   const GpuTensor& scales,
                                   const GpuTensor* bias,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dil_h, int dil_w, int groups,
                                   GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_int8w_fp16_forward_gpu: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: scales shape mismatch");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    constexpr int CONV_BLOCK = 256;
    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = bias ? reinterpret_cast<const __half*>(bias->data_fp16())
                             : nullptr;
    conv2d_int8w_fp16_forward_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<const int8_t*>(W_int8.data),
        scales.data,
        b_p,
        reinterpret_cast<__half*>(Y.data_fp16()),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        groups, Cg_in, Cg_out, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
