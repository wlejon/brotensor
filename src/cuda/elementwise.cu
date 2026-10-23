#include "elementwise_common.cuh"

namespace brotensor::detail::cuda {

namespace {

struct ScaleOp {
    float s;
    __device__ float operator()(float v) const { return v * s; }
};
struct AddScalarOp {
    float s;
    __device__ float operator()(float v) const { return v + s; }
};
struct ClampOp {
    float lo, hi;
    __device__ float operator()(float v) const {
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return v;
    }
};
struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };

__global__ void add_inplace_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

__global__ void add_scalar_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += s;
    }
}

__global__ void scale_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= s;
    }
}

__global__ void build_slot_mask_kernel(const float* __restrict__ x,
                                       float* __restrict__ mask,
                                       int offset, int K, int stride) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float v = x[offset + k * stride];
    mask[k] = v > 0.5f ? 1.0f : 0.0f;
}

__device__ inline float axpby_ld(const float* p, int i)          { return p[i]; }
__device__ inline float axpby_ld(const __half* p, int i)         { return __half2float(p[i]); }
__device__ inline float axpby_ld(const __nv_bfloat16* p, int i)  { return __bfloat162float(p[i]); }
__device__ inline void axpby_st(float* p, int i, float v)         { p[i] = v; }
__device__ inline void axpby_st(__half* p, int i, float v)        { p[i] = __float2half(v); }
__device__ inline void axpby_st(__nv_bfloat16* p, int i, float v) { p[i] = __float2bfloat16(v); }

template <typename T>
__global__ void axpby_inplace_kernel(T* __restrict__ y,
                                     const T* __restrict__ x,
                                     float a, float b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        axpby_st(y, i, a * axpby_ld(y, i) + b * axpby_ld(x, i));
    }
}

__global__ void add_inplace_f32_f16_kernel(float* __restrict__ y,
                                           const __half* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += __half2float(x[i]);
    }
}

__global__ void add_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    vec_apply_inplace2<__half, HalfCvt>(y, x, n, AddOp{});
}

__global__ void scale_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    vec_apply_inplace<__half, HalfCvt>(y, n, ScaleOp{s});
}

__global__ void add_scalar_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    vec_apply_inplace<__half, HalfCvt>(y, n, AddScalarOp{s});
}

__global__ void clamp_fp32_kernel(float* __restrict__ y, float lo, float hi, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = y[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        y[i] = v;
    }
}

__global__ void clamp_fp16_kernel(__half* __restrict__ y, float lo, float hi, int n) {
    vec_apply_inplace<__half, HalfCvt>(y, n, ClampOp{lo, hi});
}

__global__ void threshold_u8_fp32_kernel(const float* __restrict__ x, float t,
                                         int8_t* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = (x[i] > t) ? int8_t{1} : int8_t{0};
    }
}

__global__ void threshold_u8_fp16_kernel(const __half* __restrict__ x, float t,
                                         int8_t* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = (__half2float(x[i]) > t) ? int8_t{1} : int8_t{0};
    }
}

__global__ void mul_inplace_fp32_kernel(float* __restrict__ y,
                                        const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= x[i];
    }
}

__global__ void mul_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    vec_apply_inplace2<__half, HalfCvt>(y, x, n, MulOp{});
}

__global__ void add_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                        const __nv_bfloat16* __restrict__ x, int n) {
    vec_apply_inplace2<__nv_bfloat16, Bf16Cvt>(y, x, n, AddOp{});
}

__global__ void scale_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y, float s, int n) {
    vec_apply_inplace<__nv_bfloat16, Bf16Cvt>(y, n, ScaleOp{s});
}

__global__ void add_scalar_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y, float s, int n) {
    vec_apply_inplace<__nv_bfloat16, Bf16Cvt>(y, n, AddScalarOp{s});
}

__global__ void clamp_bf16_kernel(__nv_bfloat16* __restrict__ y, float lo, float hi, int n) {
    vec_apply_inplace<__nv_bfloat16, Bf16Cvt>(y, n, ClampOp{lo, hi});
}

__global__ void mul_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                        const __nv_bfloat16* __restrict__ x, int n) {
    vec_apply_inplace2<__nv_bfloat16, Bf16Cvt>(y, x, n, MulOp{});
}

__global__ void cast_f2bf_kernel(const float* __restrict__ s,
                                 __nv_bfloat16* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __float2bfloat16(s[i]);
    }
}

__global__ void cast_bf2f_kernel(const __nv_bfloat16* __restrict__ s,
                                 float* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __bfloat162float(s[i]);
    }
}

__global__ void causal_mask_row_kernel(float* __restrict__ mask, int L, int q) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
}

__global__ void cast_f2h_kernel(const float* __restrict__ s,
                                __half* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __float2half(s[i]);
    }
}

__global__ void cast_h2f_kernel(const __half* __restrict__ s,
                                float* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __half2float(s[i]);
    }
}

__global__ void cast_h2bf_kernel(const __half* __restrict__ s,
                                 __nv_bfloat16* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __float2bfloat16(__half2float(s[i]));
    }
}

__global__ void cast_bf2h_kernel(const __nv_bfloat16* __restrict__ s,
                                 __half* __restrict__ d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = __float2half(__bfloat162float(s[i]));
    }
}

// Channel-broadcast bias add: y[c*L+i] += bias[c], y channel-major (C,L).
__global__ void add_channel_bias_inplace_kernel(float* __restrict__ y,
                                                const float* __restrict__ bias,
                                                int C, int L, int n,
                                                FastDivmodU32 div_l) {
    (void)C; (void)L;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] += bias[div_l.div(i)];
    }
}
__global__ void add_channel_bias_inplace_fp16_kernel(__half* __restrict__ y,
                                                     const __half* __restrict__ bias,
                                                     int C, int L, int n,
                                                     FastDivmodU32 div_l) {
    (void)C; (void)L;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) + __half2float(bias[div_l.div(i)]));
    }
}
__global__ void add_channel_bias_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                                     const __nv_bfloat16* __restrict__ bias,
                                                     int C, int L, int n,
                                                     FastDivmodU32 div_l) {
    (void)C; (void)L;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(__bfloat162float(y[i]) + __bfloat162float(bias[div_l.div(i)]));
    }
}

// Row-broadcast bias add: Y[r*D + d] += bias[d], Y row-major (R,D), bias (D).
__global__ void add_row_bias_inplace_kernel(float* __restrict__ y,
                                            const float* __restrict__ bias,
                                            int D, int n, FastDivmodU32 div_d) {
    (void)D;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] += bias[div_d.mod(i)];
    }
}
__global__ void add_row_bias_inplace_fp16_kernel(__half* __restrict__ y,
                                                 const __half* __restrict__ bias,
                                                 int D, int n, FastDivmodU32 div_d) {
    (void)D;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) + __half2float(bias[div_d.mod(i)]));
    }
}
__global__ void add_row_bias_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                                 const __nv_bfloat16* __restrict__ bias,
                                                 int D, int n, FastDivmodU32 div_d) {
    (void)D;
    const std::uint32_t un = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < un;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(__bfloat162float(y[i]) + __bfloat162float(bias[div_d.mod(i)]));
    }
}

} // anonymous namespace

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void add_inplace(Tensor& y, const Tensor& x) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        if (x.dtype != Dtype::FP16) {
            throw std::runtime_error("add_inplace: dtype mismatch");
        }
        add_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data),
            static_cast<const __half*>(x.data), n);
    } else if (y.dtype == Dtype::BF16) {
        if (x.dtype != Dtype::BF16) {
            throw std::runtime_error("add_inplace: dtype mismatch");
        }
        add_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data),
            static_cast<const __nv_bfloat16*>(x.data), n);
    } else if (x.dtype == Dtype::FP16) {
        add_inplace_f32_f16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data),
            static_cast<const __half*>(x.data), n);
    } else {
        if (x.dtype != Dtype::FP32) {
            throw std::runtime_error("add_inplace: dtype mismatch");
        }
        add_inplace_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data),
            static_cast<const float*>(x.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("axpby_inplace: shape/dtype mismatch");
    }
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        axpby_inplace_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data),
            static_cast<const __half*>(x.data), a, b, n);
    } else if (y.dtype == Dtype::BF16) {
        axpby_inplace_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data),
            static_cast<const __nv_bfloat16*>(x.data), a, b, n);
    } else {
        axpby_inplace_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data),
            static_cast<const float*>(x.data), a, b, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const int n = src.size();
    if (n == 0) return;
    if (src.dtype == out_dtype) {
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(dst.data, src.data, src.bytes(),
                                             cudaMemcpyDeviceToDevice,
                                             cur_stream()));
        return;
    }
    if (src.dtype == Dtype::FP32 && out_dtype == Dtype::FP16) {
        cast_f2h_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(src.data),
            static_cast<__half*>(dst.data), n);
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::FP32) {
        cast_h2f_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(src.data),
            static_cast<float*>(dst.data), n);
    } else if (src.dtype == Dtype::FP32 && out_dtype == Dtype::BF16) {
        cast_f2bf_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(src.data),
            static_cast<__nv_bfloat16*>(dst.data), n);
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP32) {
        cast_bf2f_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<float*>(dst.data), n);
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::BF16) {
        cast_h2bf_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(src.data),
            static_cast<__nv_bfloat16*>(dst.data), n);
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP16) {
        cast_bf2h_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<__half*>(dst.data), n);
    } else {
        throw std::runtime_error(
            "cast: unsupported dtype pair (CUDA supports FP32<->FP16, "
            "FP32<->BF16, and FP16<->BF16)");
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_scalar_inplace(Tensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        add_scalar_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data), s, n);
    } else if (y.dtype == Dtype::BF16) {
        add_scalar_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data), s, n);
    } else {
        add_scalar_inplace_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data), s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    const int n = C * L;
    if (n == 0) return;
    if (y.dtype != bias.dtype)
        throw std::runtime_error("add_channel_bias_inplace: dtype mismatch");
    if (y.size() != n)
        throw std::runtime_error("add_channel_bias_inplace: y size != C*L");
    const FastDivmodU32 div_l(L);
    if (y.dtype == Dtype::FP16) {
        add_channel_bias_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data), static_cast<const __half*>(bias.data), C, L, n, div_l);
    } else if (y.dtype == Dtype::BF16) {
        add_channel_bias_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data),
            static_cast<const __nv_bfloat16*>(bias.data), C, L, n, div_l);
    } else {
        add_channel_bias_inplace_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data), static_cast<const float*>(bias.data), C, L, n, div_l);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_row_bias_inplace(Tensor& Y, const Tensor& bias) {
    if (Y.dtype != bias.dtype)
        throw std::runtime_error("add_row_bias_inplace: dtype mismatch");
    if (bias.size() != Y.cols)
        throw std::runtime_error("add_row_bias_inplace: bias size != Y.cols");
    const int n = Y.size();
    if (n == 0) return;
    const int D = Y.cols;
    const FastDivmodU32 div_d(D);
    if (Y.dtype == Dtype::FP16) {
        add_row_bias_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(Y.data), static_cast<const __half*>(bias.data), D, n, div_d);
    } else if (Y.dtype == Dtype::BF16) {
        add_row_bias_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(Y.data),
            static_cast<const __nv_bfloat16*>(bias.data), D, n, div_d);
    } else if (Y.dtype == Dtype::FP32) {
        add_row_bias_inplace_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(Y.data), static_cast<const float*>(bias.data), D, n, div_d);
    } else {
        throw std::runtime_error("add_row_bias_inplace: unsupported dtype");
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void clamp(Tensor& y, float lo, float hi) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        clamp_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data), lo, hi, n);
    } else if (y.dtype == Dtype::BF16) {
        clamp_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data), lo, hi, n);
    } else {
        clamp_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data), lo, hi, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scale_inplace(Tensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        scale_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data), s, n);
    } else if (y.dtype == Dtype::BF16) {
        scale_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data), s, n);
    } else {
        scale_inplace_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data), s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void mul_inplace(Tensor& y, const Tensor& x) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace: shape/dtype mismatch");
    }
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        mul_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(y.data),
            static_cast<const __half*>(x.data), n);
    } else if (y.dtype == Dtype::BF16) {
        mul_inplace_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(y.data),
            static_cast<const __nv_bfloat16*>(x.data), n);
    } else {
        mul_inplace_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(y.data),
            static_cast<const float*>(x.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("threshold_u8: X must be FP32 or FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const int n = X.size();
    if (n == 0) return;
    if (X.dtype == Dtype::FP16) {
        threshold_u8_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), t,
            static_cast<int8_t*>(Y.data), n);
    } else {
        threshold_u8_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), t,
            static_cast<int8_t*>(Y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    const int blocks = (L + EW_BLOCK - 1) / EW_BLOCK;
    causal_mask_row_kernel<<<blocks, EW_BLOCK, 0, cur_stream()>>>(
        static_cast<float*>(mask.data), L, q);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask) {
    if (mask.rows != K || mask.cols != 1) mask.resize(K, 1);
    if (K <= 0) return;
    const int blocks = (K + EW_BLOCK - 1) / EW_BLOCK;
    build_slot_mask_kernel<<<blocks, EW_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(mask.data),
        offset, K, stride);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── Vtable contribution ───────────────────────────────────────────────────
namespace {

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ inline float sigmoid_scalar(float v) {
    return 1.0f / (1.0f + expf(-v));
}

template <typename T>
__global__ void relu_forward_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = to_f32(x[i]);
        y[i] = from_f32<T>(v > 0.0f ? v : 0.0f);
    }
}

template <typename T>
__global__ void relu_backward_kernel(const T* __restrict__ x, const T* __restrict__ dY, T* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dX[i] = from_f32<T>(to_f32(x[i]) > 0.0f ? to_f32(dY[i]) : 0.0f);
    }
}

template <typename T>
__global__ void tanh_forward_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = from_f32<T>(tanhf(to_f32(x[i])));
    }
}

template <typename T>
__global__ void tanh_backward_kernel(const T* __restrict__ y, const T* __restrict__ dY, T* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float yv = to_f32(y[i]);
        dX[i] = from_f32<T>(to_f32(dY[i]) * (1.0f - yv * yv));
    }
}

template <typename T>
__global__ void sigmoid_forward_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = from_f32<T>(sigmoid_scalar(to_f32(x[i])));
    }
}

template <typename T>
__global__ void sigmoid_backward_kernel(const T* __restrict__ y, const T* __restrict__ dY, T* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float yv = to_f32(y[i]);
        dX[i] = from_f32<T>(to_f32(dY[i]) * yv * (1.0f - yv));
    }
}

} // namespace

void relu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) y.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) relu_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    else if (x.dtype == Dtype::BF16) relu_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    else relu_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) dX.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) relu_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data), n);
    else if (x.dtype == Dtype::BF16) relu_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data), static_cast<__nv_bfloat16*>(dX.data), n);
    else relu_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(x.data), static_cast<const float*>(dY.data), static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) y.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) tanh_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    else if (x.dtype == Dtype::BF16) tanh_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    else tanh_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols || dX.dtype != y.dtype) dX.resize(y.rows, y.cols, y.dtype);
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) tanh_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(y.data), static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data), n);
    else if (y.dtype == Dtype::BF16) tanh_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(y.data), static_cast<const __nv_bfloat16*>(dY.data), static_cast<__nv_bfloat16*>(dX.data), n);
    else tanh_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(y.data), static_cast<const float*>(dY.data), static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) y.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    else if (x.dtype == Dtype::BF16) sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    else sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols || dX.dtype != y.dtype) dX.resize(y.rows, y.cols, y.dtype);
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __half*>(y.data), static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data), n);
    else if (y.dtype == Dtype::BF16) sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const __nv_bfloat16*>(y.data), static_cast<const __nv_bfloat16*>(dY.data), static_cast<__nv_bfloat16*>(dX.data), n);
    else sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(static_cast<const float*>(y.data), static_cast<const float*>(dY.data), static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void silu_forward(const Tensor& x, Tensor& y);
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);
void gelu_forward(const Tensor& x, Tensor& y);
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);
void gelu_exact_forward(const Tensor& x, Tensor& y);
void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX);
void quick_gelu_forward(const Tensor& x, Tensor& y);
void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);
void geglu_forward(const Tensor& X, Tensor& Y);
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX);
void geglu_exact_forward(const Tensor& X, Tensor& Y);
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX);

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y);
void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma);
void swiglu_forward(const Tensor& X, Tensor& Y);
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX);
void modulate(const ::brotensor::Tensor& X, const ::brotensor::Tensor& scale,
              const ::brotensor::Tensor& shift, ::brotensor::Tensor& Y);
void broadcast_mul(const ::brotensor::Tensor& X, const ::brotensor::Tensor& v,
                   ::brotensor::Tensor& Y);

void fill_cuda_vtable_elementwise(::brotensor::detail::OpsVTable& v) {
    v.relu_forward             = &relu_forward;
    v.modulate                 = &modulate;
    v.broadcast_mul            = &broadcast_mul;
    v.relu_backward            = &relu_backward;
    v.tanh_forward             = &tanh_forward;
    v.tanh_backward            = &tanh_backward;
    v.sigmoid_forward          = &sigmoid_forward;
    v.sigmoid_backward         = &sigmoid_backward;
    v.add_inplace              = &add_inplace;
    v.axpby_inplace            = &axpby_inplace;
    v.add_scalar_inplace       = &add_scalar_inplace;
    v.add_channel_bias_inplace = &add_channel_bias_inplace;
    v.add_row_bias_inplace     = &add_row_bias_inplace;
    v.cast                     = &cast;
    v.scale_inplace            = &scale_inplace;
    v.mul_inplace              = &mul_inplace;
    v.threshold_u8             = &threshold_u8;
    v.clamp                    = &clamp;
    v.silu_forward             = &silu_forward;
    v.silu_backward            = &silu_backward;
    v.gelu_forward             = &gelu_forward;
    v.gelu_backward            = &gelu_backward;
    v.gelu_exact_forward       = &gelu_exact_forward;
    v.gelu_exact_backward      = &gelu_exact_backward;
    v.quick_gelu_forward       = &quick_gelu_forward;
    v.quick_gelu_backward      = &quick_gelu_backward;
    v.geglu_forward            = &geglu_forward;
    v.geglu_backward           = &geglu_backward;
    v.geglu_exact_forward      = &geglu_exact_forward;
    v.geglu_exact_backward     = &geglu_exact_backward;
    v.build_slot_mask          = &build_slot_mask;
    v.build_causal_mask_row    = &build_causal_mask_row;
    v.rms_norm_forward         = &rms_norm_forward;
    v.rms_norm_backward        = &rms_norm_backward;
    v.swiglu_forward           = &swiglu_forward;
    v.swiglu_backward          = &swiglu_backward;
}

} // namespace brotensor::detail::cuda
