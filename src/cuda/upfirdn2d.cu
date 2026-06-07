// ─── CUDA upfirdn2d (StyleGAN3-R) ───────────────────────────────────────────
//
// CUDA port of src/cpu/upfirdn2d.cpp. Upsample (zero-insert) → pad/crop → 2D
// FIR correlation → downsample → gain. General non-separable 2D path mirroring
// NVlabs `_upfirdn2d_ref`. The filter is a constant shared across channels
// (depthwise); there is no gradient to it.
//
// Both forward and backward funnel through `upfirdn2d_run`: the op is linear in
// X, so the backward is itself a forward with up/down swapped, the filter flip
// inverted, and padding recomputed (mirrors `_upfirdn2d_cuda`).
//
// One CUDA thread per output element (n,c,oh,ow); the per-tap accumulation runs
// in FP32 regardless of storage dtype (FP32/FP16/BF16 supported; CPU is
// FP32-only).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

// Defined in runtime.cu. Launch on the current stream so this op composes
// correctly with the stream-ordered modulated_conv2d / bias_act it chains with
// in filtered_lrelu (rather than always serializing on the default stream).
void* cuda_current_stream();

namespace {

constexpr int UF_BLOCK = 256;

inline void require_fp(const char* op, const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

template <typename T> __device__ inline float uf_load(const T* p);
template <> __device__ inline float uf_load<float>(const float* p) { return *p; }
template <> __device__ inline float uf_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float uf_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void uf_store(T* p, float v);
template <> __device__ inline void uf_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void uf_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void uf_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__global__ void upfirdn2d_kernel(const T* __restrict__ In, const T* __restrict__ Fp,
                                 int N, int C, int Hin, int Win, int Hout, int Wout,
                                 int fH, int fW, int up_x, int up_y,
                                 int down_x, int down_y, int px0, int py0,
                                 int Hu, int Wu, int flip_filter, float gain,
                                 T* __restrict__ Out) {
    const long long total = static_cast<long long>(N) * C * Hout * Wout;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(i % Wout);
        long long t = i / Wout;
        const int oh = static_cast<int>(t % Hout);
        t /= Hout;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);

        const size_t in_base = (static_cast<size_t>(n) * C + c) * Hin * Win;
        const int py_base = oh * down_y;
        const int px_base = ow * down_x;

        float acc = 0.0f;
        for (int kh = 0; kh < fH; ++kh) {
            const int uy = py_base + kh - py0;
            if (uy < 0 || uy >= Hu || (uy % up_y) != 0) continue;
            const int iy = uy / up_y;
            const int frow = flip_filter ? kh : (fH - 1 - kh);
            for (int kw = 0; kw < fW; ++kw) {
                const int ux = px_base + kw - px0;
                if (ux < 0 || ux >= Wu || (ux % up_x) != 0) continue;
                const int ix = ux / up_x;
                const int fcol = flip_filter ? kw : (fW - 1 - kw);
                acc += uf_load<T>(&In[in_base + static_cast<size_t>(iy) * Win + ix]) *
                       uf_load<T>(&Fp[static_cast<size_t>(frow) * fW + fcol]);
            }
        }
        const size_t out_base = (static_cast<size_t>(n) * C + c) * Hout * Wout;
        uf_store<T>(&Out[out_base + static_cast<size_t>(oh) * Wout + ow], acc * gain);
    }
}

// Shared engine. `In` is (N, C*Hin*Win); resizes `Out` to (N, C*Hout*Wout).
void upfirdn2d_run(const ::brotensor::Tensor& In, int N, int C, int Hin, int Win,
                   const ::brotensor::Tensor& f, int fH, int fW,
                   int up_x, int up_y, int down_x, int down_y,
                   int px0, int px1, int py0, int py1,
                   bool flip_filter, float gain,
                   ::brotensor::Tensor& Out, const char* op) {
    require_fp(op, In, "input");
    require_fp(op, f, "f");
    if (f.dtype != In.dtype)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": f.dtype must match input.dtype");
    if (up_x < 1 || up_y < 1 || down_x < 1 || down_y < 1)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": up/down factors must be >= 1");
    if (In.rows != N || In.cols != C * Hin * Win)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": input shape mismatch");
    if (f.rows != fH || f.cols != fW)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": filter shape mismatch");
    const int Hu = Hin * up_y, Wu = Win * up_x;
    const int Hp = Hu + py0 + py1, Wp = Wu + px0 + px1;
    if (Hp < fH || Wp < fW)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": padded input smaller than filter");
    const int Hc = Hp - fH + 1, Wc = Wp - fW + 1;
    const int Hout = (Hc - 1) / down_y + 1;
    const int Wout = (Wc - 1) / down_x + 1;
    const int out_cols = C * Hout * Wout;
    if (Out.rows != N || Out.cols != out_cols || Out.dtype != In.dtype)
        Out.resize(N, out_cols, In.dtype);
    if (N == 0 || out_cols == 0) return;

    const long long total = static_cast<long long>(N) * out_cols;
    long long blocks = (total + UF_BLOCK - 1) / UF_BLOCK;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    const int grid = static_cast<int>(blocks);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    if (In.dtype == ::brotensor::Dtype::FP16)
        upfirdn2d_kernel<__half><<<grid, UF_BLOCK, 0, stream>>>(
            static_cast<const __half*>(In.data), static_cast<const __half*>(f.data),
            N, C, Hin, Win, Hout, Wout, fH, fW, up_x, up_y, down_x, down_y,
            px0, py0, Hu, Wu, flip_filter ? 1 : 0, gain, static_cast<__half*>(Out.data));
    else if (In.dtype == ::brotensor::Dtype::BF16)
        upfirdn2d_kernel<__nv_bfloat16><<<grid, UF_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(In.data), static_cast<const __nv_bfloat16*>(f.data),
            N, C, Hin, Win, Hout, Wout, fH, fW, up_x, up_y, down_x, down_y,
            px0, py0, Hu, Wu, flip_filter ? 1 : 0, gain, static_cast<__nv_bfloat16*>(Out.data));
    else
        upfirdn2d_kernel<float><<<grid, UF_BLOCK, 0, stream>>>(
            static_cast<const float*>(In.data), static_cast<const float*>(f.data),
            N, C, Hin, Win, Hout, Wout, fH, fW, up_x, up_y, down_x, down_y,
            px0, py0, Hu, Wu, flip_filter ? 1 : 0, gain, static_cast<float*>(Out.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

inline int out_dim(int in, int up, int down, int pad0, int pad1, int fdim) {
    return (in * up + pad0 + pad1 - fdim) / down + 1;
}

} // namespace

void upfirdn2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, ::brotensor::Tensor& Y) {
    upfirdn2d_run(X, N, C, H, Wd, f, fH, fW,
                  up_x, up_y, down_x, down_y,
                  pad_x0, pad_x1, pad_y0, pad_y1,
                  flip_filter, gain, Y, "upfirdn2d_forward");
}

void upfirdn2d_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, ::brotensor::Tensor& dX) {
    const int Hout = out_dim(H,  up_y, down_y, pad_y0, pad_y1, fH);
    const int Wout = out_dim(Wd, up_x, down_x, pad_x0, pad_x1, fW);
    // Backward padding (NVlabs _upfirdn2d_cuda): swap up<->down, flip the flip.
    const int p_x0 = fW - pad_x0 - 1;
    const int p_x1 = Wd * up_x - Wout * down_x + pad_x0 - up_x + 1;
    const int p_y0 = fH - pad_y0 - 1;
    const int p_y1 = H * up_y - Hout * down_y + pad_y0 - up_y + 1;
    upfirdn2d_run(dY, N, C, Hout, Wout, f, fH, fW,
                  /*up=*/down_x, down_y, /*down=*/up_x, up_y,
                  p_x0, p_x1, p_y0, p_y1,
                  !flip_filter, gain, dX, "upfirdn2d_backward");
    if (dX.rows != N || dX.cols != C * H * Wd)
        throw std::runtime_error("upfirdn2d_backward: internal dX shape "
                                 "mismatch (param inconsistency)");
}

void fill_cuda_vtable_upfirdn2d(::brotensor::detail::OpsVTable& v) {
    v.upfirdn2d_forward  = &upfirdn2d_forward;
    v.upfirdn2d_backward = &upfirdn2d_backward;
}

} // namespace brotensor::detail::cuda
