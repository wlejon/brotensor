// ─── CUDA convex (mask-based) upsample, NCHW ────────────────────────────────
//
// CUDA port of src/cpu/convex_upsample.cpp. One block per OUTPUT spatial
// location (n, oy, ox), looping internally over channel c; the block derives
// the source low-res pixel (y, x) and sub-position (sy, sx), softmaxes the 9
// mask logits for that (sy, sx, y, x) once, and blends the 3×3 low-res
// neighborhood of every channel c:
//   Y[n,c,k*y+sy,k*x+sx] = sum_m softmax_m(Mask[n,m,sy,sx,y,x]) * X[n,c,ny,nx]
//   neighbor m: ny=clamp(y-1+m/3), nx=clamp(x-1+m%3)  (replicate pad)
// Mask flat channel = (m*k*k + sy*k + sx). Softmax in double (matches CPU).
//
// The 9-way softmax depends only on (n, sy, sx, y, x) — not on channel c — so
// it is computed ONCE per spatial location by thread 0 of the owning block and
// cached in shared memory; all C channel-threads for that spatial location
// then read the cached weights instead of redoing the FP64 softmax. Each block
// owns a grid-strided set of spatial locations (never split across blocks) so
// the __shared__ cache is always valid for every thread that reads it. CPU is
// FP32-only; CUDA adds FP16/BF16 (cast at load/store). Y OVERWRITTEN.
// Inference-only: no backward.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int CU_BLOCK = 256;

inline int cu_grid(long long n) {
    long long blocks = (n + CU_BLOCK - 1) / CU_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp(const ::brotensor::Tensor& t,
                     const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16)
        fail(op, std::string(name) + " must be FP32/FP16/BF16");
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T> __device__ inline float cu_load(const T* p);
template <> __device__ inline float cu_load<float>(const float* p) { return *p; }
template <> __device__ inline float cu_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float cu_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void cu_store(T* p, float v);
template <> __device__ inline void cu_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void cu_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void cu_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// One block handles a grid-strided set of spatial locations (n, oy, ox); all
// C channels for the CURRENT spatial location are processed by that same
// block before it moves to the next one, so the shared-memory softmax cache
// is always populated by a thread in the same block that reads it.
template <typename T>
__global__ void convex_upsample_kernel(const T* __restrict__ X,
                                       const T* __restrict__ Mask,
                                       T* __restrict__ Y,
                                       int N, int C, int H, int W, int scale) {
    const int HW = H * W;
    const int kk = scale * scale;
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    const long long spatial_total = (long long)N * oHW;

    __shared__ double w[9];

    for (long long sidx = blockIdx.x; sidx < spatial_total; sidx += gridDim.x) {
        const int ox = static_cast<int>(sidx % oW);
        long long t = sidx / oW;
        const int oy = static_cast<int>(t % oH);
        const int n = static_cast<int>(t / oH);

        const int y = oy / scale, sy = oy % scale;
        const int x = ox / scale, sx = ox % scale;
        const int sub = sy * scale + sx;
        const int pix = y * W + x;
        const T* m_img = Mask + (long long)n * 9 * kk * HW;

        // Softmax over the 9 mask logits depends only on (n, sy, sx, y, x) —
        // computed once by thread 0, cached in shared memory for all C
        // channel-threads of this block.
        if (threadIdx.x == 0) {
            double mx = -1e300;
            for (int m = 0; m < 9; ++m) {
                const double v = cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]);
                if (v > mx) mx = v;
            }
            double sum = 0.0;
            for (int m = 0; m < 9; ++m) {
                const double e = exp(cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]) - mx);
                w[m] = e; sum += e;
            }
            const double invs = 1.0 / sum;
            for (int m = 0; m < 9; ++m) w[m] *= invs;
        }
        __syncthreads();

        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const T* xc = X + ((long long)n * C + c) * HW;
            double acc = 0.0;
            for (int m = 0; m < 9; ++m) {
                const int ny = clampi(y - 1 + m / 3, 0, H - 1);
                const int nx = clampi(x - 1 + m % 3, 0, W - 1);
                acc += w[m] * cu_load<T>(&xc[(long long)ny * W + nx]);
            }
            const long long oidx = ((long long)n * C + c) * oHW + (long long)oy * oW + ox;
            cu_store<T>(&Y[oidx], static_cast<float>(acc));
        }
        __syncthreads();   // guard w[] before the next sidx iteration overwrites it
    }
}

} // namespace

void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y) {
    const char* op = "convex_upsample_forward";
    check_fp(X, op, "X");
    check_fp(Mask, op, "Mask");
    if (X.dtype != Mask.dtype) fail(op, "X and Mask must share dtype");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (scale < 1) fail(op, "scale must be >=1");
    const int HW = H * W;
    const int kk = scale * scale;
    if (X.rows != N || X.cols != C * HW) fail(op, "X shape must be (N, C*H*W)");
    if (Mask.rows != N || Mask.cols != 9 * kk * HW)
        fail(op, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    if (Y.rows != N || Y.cols != C * oHW || Y.dtype != X.dtype)
        Y.resize(N, static_cast<int>(C * oHW), X.dtype);
    if (N == 0) return;

    // Grid is sized over spatial locations (n, oy, ox) — the softmax is cached
    // once per block per location and shared across all C channel-threads.
    const long long spatial_total = (long long)N * oHW;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        convex_upsample_kernel<__half><<<cu_grid(spatial_total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(Mask.data),
            static_cast<__half*>(Y.data), N, C, H, W, scale);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        convex_upsample_kernel<__nv_bfloat16><<<cu_grid(spatial_total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(Mask.data),
            static_cast<__nv_bfloat16*>(Y.data), N, C, H, W, scale);
    } else {
        convex_upsample_kernel<float><<<cu_grid(spatial_total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(Mask.data),
            static_cast<float*>(Y.data), N, C, H, W, scale);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_convex_upsample(::brotensor::detail::OpsVTable& v) {
    v.convex_upsample_forward = &convex_upsample_forward;
}

} // namespace brotensor::detail::cuda
