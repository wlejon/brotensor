// ─── CUDA bias_act (StyleGAN3) ──────────────────────────────────────────────
//
// CUDA port of src/cpu/bias_act.cpp. Fused per-channel bias + activation +
// gain + clamp, mirroring NVlabs `_bias_act_ref`.
//
//   X: (N, C*HW) — channel c owns the contiguous block [c*HW, (c+1)*HW)
//                  within each row.
//   b: (C,1) or null.   act: 0 = linear, 1 = lrelu.   clamp < 0 ⇒ no clamp.
//
// Forward:  t = X + b[c];  y = act(t);  y *= gain;  if clamp>=0: clip(±clamp).
// Backward: dt = dY*gain*act'(t)*(clamp active ? 0 : 1);
//           dX = dt (overwrite);  dB[c] += Σ dt (accumulate — caller zeros).
// The clamp gradient mask uses the PRE-clamp value y_pre = gain*act(t): where
// |y_pre| > clamp the output saturated, so the gradient is zero there.
//
// CPU is FP32-only; CUDA additionally supports FP16/BF16 (math in FP32). dB is
// accumulated through an FP32 scratch buffer (atomicAdd) then merged back into
// the caller's dB so the FP16/BF16 path keeps full-precision reduction.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int BA_BLOCK = 256;
constexpr int ACT_LINEAR = 0;
constexpr int ACT_LRELU  = 1;

inline int ba_grid(long long n) {
    long long blocks = (n + BA_BLOCK - 1) / BA_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

inline void require_fp(const char* op, const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

inline void check_act(int act, const char* op) {
    if (act != ACT_LINEAR && act != ACT_LRELU) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": act must be 0 (linear) or 1 (lrelu)");
    }
}

template <typename T> __device__ inline float ba_load(const T* p);
template <> __device__ inline float ba_load<float>(const float* p) { return *p; }
template <> __device__ inline float ba_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float ba_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void ba_store(T* p, float v);
template <> __device__ inline void ba_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void ba_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void ba_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ inline float apply_act(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? t : alpha * t;
    return t;
}
__device__ inline float act_grad(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? 1.0f : alpha;
    return 1.0f;
}

template <typename T>
__global__ void bias_act_forward_kernel(const T* __restrict__ X,
                                        const T* __restrict__ b,
                                        long long total, int C, int HW,
                                        int act, float alpha, float gain,
                                        float clamp, T* __restrict__ Y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>((i / HW) % C);
        const float bias_v = b ? ba_load<T>(&b[c]) : 0.0f;
        float y = apply_act(ba_load<T>(&X[i]) + bias_v, act, alpha) * gain;
        if (clamp >= 0.0f) {
            if (y < -clamp) y = -clamp;
            else if (y > clamp) y = clamp;
        }
        ba_store<T>(&Y[i], y);
    }
}

template <typename T>
__global__ void bias_act_backward_kernel(const T* __restrict__ dY,
                                         const T* __restrict__ X,
                                         const T* __restrict__ b,
                                         long long total, int C, int HW,
                                         int act, float alpha, float gain,
                                         float clamp, T* __restrict__ dX,
                                         float* __restrict__ dB_scratch) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>((i / HW) % C);
        const float bias_v = b ? ba_load<T>(&b[c]) : 0.0f;
        const float t = ba_load<T>(&X[i]) + bias_v;
        float dt = ba_load<T>(&dY[i]) * gain * act_grad(t, act, alpha);
        if (clamp >= 0.0f) {
            const float y_pre = gain * apply_act(t, act, alpha);
            if (y_pre < -clamp || y_pre > clamp) dt = 0.0f;
        }
        ba_store<T>(&dX[i], dt);            // overwrite
        if (dB_scratch) atomicAdd(&dB_scratch[c], dt);
    }
}

// Merge the FP32 dB scratch into the caller's dB (accumulate — caller zeroed).
template <typename T>
__global__ void ba_merge_dB_kernel(const float* __restrict__ src,
                                   T* __restrict__ dst, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    ba_store<T>(&dst[c], ba_load<T>(&dst[c]) + src[c]);
}

inline void check_shapes(const char* op, const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor* b, int N, int C, int HW) {
    const int cols = C * HW;
    if (X.rows != N || X.cols != cols)
        throw std::runtime_error(std::string("brotensor: ") + op + ": X shape mismatch");
    if (b && b->size() != C)
        throw std::runtime_error(std::string("brotensor: ") + op + ": b must have C elements");
}

template <typename T>
void launch_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor* b,
                    long long total, int C, int HW, int act, float alpha,
                    float gain, float clamp, ::brotensor::Tensor& Y) {
    bias_act_forward_kernel<T><<<ba_grid(total), BA_BLOCK>>>(
        static_cast<const T*>(X.data),
        b ? static_cast<const T*>(b->data) : nullptr,
        total, C, HW, act, alpha, gain, clamp, static_cast<T*>(Y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bias_act_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, ::brotensor::Tensor& Y) {
    require_fp("bias_act_forward", X, "X");
    if (b) {
        require_fp("bias_act_forward", *b, "b");
        if (b->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_forward: b.dtype must match X.dtype");
    }
    check_act(act, "bias_act_forward");
    check_shapes("bias_act_forward", X, b, N, C, HW);
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype)
        Y.resize(N, cols, X.dtype);
    const long long total = static_cast<long long>(N) * cols;
    if (total == 0) return;
    if (X.dtype == ::brotensor::Dtype::FP16)
        launch_forward<__half>(X, b, total, C, HW, act, alpha, gain, clamp, Y);
    else if (X.dtype == ::brotensor::Dtype::BF16)
        launch_forward<__nv_bfloat16>(X, b, total, C, HW, act, alpha, gain, clamp, Y);
    else
        launch_forward<float>(X, b, total, C, HW, act, alpha, gain, clamp, Y);
}

void bias_act_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp,
                       ::brotensor::Tensor& dX, ::brotensor::Tensor* dB) {
    require_fp("bias_act_backward", dY, "dY");
    require_fp("bias_act_backward", X, "X");
    if (dY.dtype != X.dtype)
        throw std::runtime_error("brotensor: bias_act_backward: dY.dtype must match X.dtype");
    if (b) {
        require_fp("bias_act_backward", *b, "b");
        if (b->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_backward: b.dtype must match X.dtype");
    }
    if (dB) {
        require_fp("bias_act_backward", *dB, "dB");
        if (dB->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_backward: dB.dtype must match X.dtype");
        if (dB->size() != C)
            throw std::runtime_error("brotensor: bias_act_backward: dB must have C elements");
    }
    check_act(act, "bias_act_backward");
    check_shapes("bias_act_backward", X, b, N, C, HW);
    const int cols = C * HW;
    if (dY.rows != N || dY.cols != cols)
        throw std::runtime_error("brotensor: bias_act_backward: dY shape mismatch");
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype)
        dX.resize(N, cols, X.dtype);
    const long long total = static_cast<long long>(N) * cols;
    if (total == 0) return;

    // dB reduction goes through an FP32 scratch (atomicAdd needs FP32 for the
    // FP16/BF16 paths, and gives a stable full-precision reduction in FP32).
    float* d_scratch = nullptr;
    if (dB) {
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        static_cast<size_t>(C) * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_scratch, 0,
                                             static_cast<size_t>(C) * sizeof(float)));
    }

    auto run = [&](auto tag) {
        using T = decltype(tag);
        bias_act_backward_kernel<T><<<ba_grid(total), BA_BLOCK>>>(
            static_cast<const T*>(dY.data), static_cast<const T*>(X.data),
            b ? static_cast<const T*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp,
            static_cast<T*>(dX.data), d_scratch);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        if (dB) {
            const int blocks = (C + BA_BLOCK - 1) / BA_BLOCK;
            ba_merge_dB_kernel<T><<<blocks, BA_BLOCK>>>(
                d_scratch, static_cast<T*>(dB->data), C);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
        }
    };

    if (X.dtype == ::brotensor::Dtype::FP16)       run(__half{});
    else if (X.dtype == ::brotensor::Dtype::BF16)  run(__nv_bfloat16{});
    else                                           run(float{});

    if (d_scratch) {
        // The merge kernel has consumed the scratch; sync before freeing so we
        // don't reclaim memory a still-running kernel reads.
        BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(0));
        cudaFree(d_scratch);
    }
}

void fill_cuda_vtable_bias_act(::brotensor::detail::OpsVTable& v) {
    v.bias_act_forward  = &bias_act_forward;
    v.bias_act_backward = &bias_act_backward;
}

} // namespace brotensor::detail::cuda
