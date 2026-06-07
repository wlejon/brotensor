// ─── CUDA fused filtered_lrelu (StyleGAN3 alias-free nonlinearity) ──────────
//
// The composite filtered_lrelu is bias_act → upfirdn2d(up, gain=up²) →
// bias_act(lrelu) → upfirdn2d(down): 3–4 full global-memory passes at the
// up-sampled (2×) resolution per layer. This fuses the whole chain into one
// kernel that reads X (+bias) directly and writes Y directly — no global
// up_buf/act_buf intermediates — so the up-FIR, the leaky-ReLU/clamp, and the
// down-FIR all happen in registers. That removes the memory passes the
// composite spends at 2× resolution, and it makes FP16/BF16 a real win (the
// input reads halve their bandwidth and the FIR runs on the half-rate ALUs)
// instead of pure conversion overhead. Supports the general non-separable 2D
// filters config-R uses (radial down-filter); FP32/FP16/BF16 storage, FP32
// math, mirroring `_filtered_lrelu_ref` exactly.
//
// Cache contract: the fully-fused path produces NO up_buf/act_buf. The
// differentiable (inversion) path still needs up_buf for the composite
// backward, so when the caller commits up_buf/act_buf (signaling it wants the
// caches) this falls back to the composite, keeping that FP32 path bit-exact.
// Pure synthesis passes uncommitted caches and takes the fused path.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor {

// Composite fallback (src/filtered_lrelu.cpp) — reused for the cache-wanted /
// unsupported configs. Runs over the public bias_act + upfirdn2d sub-ops, so it
// dispatches back to this same CUDA backend.
void filtered_lrelu_forward_composite(const Tensor& X, const Tensor& fu,
                                      const Tensor& fd, const Tensor* b,
                                      int N, int C, int H, int W, int up, int down,
                                      int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                                      float gain, float slope, float clamp,
                                      Tensor& up_buf, Tensor& act_buf, Tensor& Y);

namespace detail::cuda {

void* cuda_current_stream();

namespace {

// Output tile per CTA. TH*TW threads; each CTA computes the AH×AW activation
// region once into shared memory, then downsamples it. 16×16 output tile (256
// threads) measured best across config-R sizes — enough warps to hide latency
// while keeping the shared act tile (≈36×36 floats for 6-tap filters) small
// enough for high occupancy; 8×8 under-occupies and 32×32 spills occupancy.
constexpr int FL_TH = 16;
constexpr int FL_TW = 16;
constexpr int FL_THREADS = FL_TH * FL_TW;

template <typename T> __device__ inline float fl_load(const T* p);
template <> __device__ inline float fl_load<float>(const float* p) { return *p; }
template <> __device__ inline float fl_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float fl_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void fl_store(T* p, float v);
template <> __device__ inline void fl_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void fl_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void fl_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// One CTA per (n, c, output-tile). The tile covers FL_TH×FL_TW output pixels.
// Phase A computes the activation region the tile needs — exactly once each —
// into shared memory:
//   act[ay,ax] = clamp( lrelu( up² · Σ fu[flip]·(X[iy,ix]+b[c]) ) · gain )
// where the up-FIR reads input rows iy=(ay+fuy−py0)/up only where that lands on
// the up grid (zero-insertion). Phase B downsamples from shared memory:
//   Y[oh,ow] = Σ fd[flip] · act[(oh−oh0)·down+fdy, (ow−ow0)·down+fdx]
// Filters are correlated flipped (flip_filter=false ⇒ true convolution), so the
// tap index is mirrored, matching upfirdn2d's `frow = fH−1−kh`. The act region
// origin is (oh0·down, ow0·down); its extent AH×AW is passed in. Computing each
// activation once (vs recomputing per output tap) is the whole point — the
// down-FIR footprints of neighbouring outputs overlap heavily.
template <typename T>
__global__ void filtered_lrelu_kernel(
        const T* __restrict__ X, const T* __restrict__ fu, const T* __restrict__ fd,
        const T* __restrict__ b, int N, int C, int H, int W,
        int Hu, int Wu, int Hout, int Wout,
        int fuH, int fuW, int fdH, int fdW,
        int up, int down, int px0, int py0,
        int AH, int AW, float up_gain, float gain, float slope, float clamp,
        T* __restrict__ Y) {
    extern __shared__ float act[];   // AH * AW activations

    const int nc = blockIdx.z;
    const int c  = nc % C;
    const int n  = nc / C;
    const int oh0 = blockIdx.y * FL_TH;   // tile's first output row
    const int ow0 = blockIdx.x * FL_TW;   // tile's first output col
    const int ay0 = oh0 * down;           // act region origin
    const int ax0 = ow0 * down;
    const int tid = threadIdx.x;

    const size_t in_base = (static_cast<size_t>(n) * C + c) * H * W;
    const float bias_v = b ? fl_load<T>(&b[c]) : 0.0f;

    // Phase A: fill the shared activation tile (each cell computed once).
    const int acells = AH * AW;
    for (int a = tid; a < acells; a += FL_THREADS) {
        const int ly = a / AW, lx = a % AW;
        const int ay = ay0 + ly, ax = ax0 + lx;
        // Only taps where (ay+fuy−py0) and (ax+fux−px0) land on the up grid
        // (≡0 mod up) contribute — the rest hit zero-inserted samples. Step the
        // tap loops by `up` from the first valid residue instead of testing
        // every tap (≈up² fewer iterations).
        const int ry = (((py0 - ay) % up) + up) % up;
        const int rx = (((px0 - ax) % up) + up) % up;
        float u = 0.0f;
        for (int fuy = ry; fuy < fuH; fuy += up) {
            const int uy = ay + fuy - py0;            // divisible by up
            if (uy < 0 || uy >= Hu) continue;
            const int iy = uy / up;
            const int frow_u = fuH - 1 - fuy;
            for (int fux = rx; fux < fuW; fux += up) {
                const int ux = ax + fux - px0;        // divisible by up
                if (ux < 0 || ux >= Wu) continue;
                const int ix = ux / up;
                const float xv = fl_load<T>(&X[in_base + static_cast<size_t>(iy) * W + ix]) + bias_v;
                u += fl_load<T>(&fu[frow_u * fuW + (fuW - 1 - fux)]) * xv;
            }
        }
        u *= up_gain;
        float av = (u > 0.0f ? u : slope * u) * gain;
        if (clamp >= 0.0f) {
            if (av < -clamp) av = -clamp;
            else if (av > clamp) av = clamp;
        }
        act[a] = av;
    }
    __syncthreads();

    // Phase B: one thread per output pixel in the tile; downsample from shared.
    const int oh = oh0 + tid / FL_TW;
    const int ow = ow0 + tid % FL_TW;
    if (oh < Hout && ow < Wout) {
        const int ly0 = (oh - oh0) * down;   // local act row base for this output
        const int lx0 = (ow - ow0) * down;
        float acc = 0.0f;
        for (int fdy = 0; fdy < fdH; ++fdy) {
            const int frow_d = fdH - 1 - fdy;
            const int ly = ly0 + fdy;
            for (int fdx = 0; fdx < fdW; ++fdx) {
                acc += fl_load<T>(&fd[frow_d * fdW + (fdW - 1 - fdx)]) * act[ly * AW + (lx0 + fdx)];
            }
        }
        const size_t out_base = (static_cast<size_t>(n) * C + c) * Hout * Wout;
        fl_store<T>(&Y[out_base + static_cast<size_t>(oh) * Wout + ow], acc);
    }
}

inline bool is_fp(::brotensor::Dtype d) {
    return d == ::brotensor::Dtype::FP32 || d == ::brotensor::Dtype::FP16 ||
           d == ::brotensor::Dtype::BF16;
}

} // namespace

void filtered_lrelu_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& fu,
                            const ::brotensor::Tensor& fd, const ::brotensor::Tensor* b,
                            int N, int C, int H, int W, int up, int down,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                            float gain, float slope, float clamp,
                            ::brotensor::Tensor& up_buf, ::brotensor::Tensor& act_buf,
                            ::brotensor::Tensor& Y) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;
    const int Hu = H * up, Wu = W * up;
    const int Huo = Hu + pad_y0 + pad_y1 - fuH + 1;
    const int Wuo = Wu + pad_x0 + pad_x1 - fuW + 1;
    const int Hout = (Huo - fdH) / down + 1;
    const int Wout = (Wuo - fdW) / down + 1;

    // Fall back to the composite when the caller wants the up_buf/act_buf caches
    // (the differentiable inversion path), for an unsupported dtype/config, or a
    // degenerate shape. The composite keeps that FP32 path bit-exact and covers
    // every config this fused kernel doesn't.
    const int AH = (FL_TH - 1) * down + fdH;   // shared activation tile extent
    const int AW = (FL_TW - 1) * down + fdW;
    const size_t shmem = static_cast<size_t>(AH) * AW * sizeof(float);

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool supported =
        is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) && up >= 1 && down >= 1 &&
        Huo > 0 && Wuo > 0 && Hout > 0 && Wout > 0 &&
        shmem <= 48u * 1024u &&            // larger tiles fall back to composite
        static_cast<long long>(N) * C <= 65535;  // gridDim.z limit
    if (want_cache || !supported) {
        ::brotensor::filtered_lrelu_forward_composite(
            X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, act_buf, Y);
        return;
    }

    if (X.rows != N || X.cols != C * H * W)
        throw std::runtime_error("brotensor: filtered_lrelu_forward: X shape mismatch");
    const int out_cols = C * Hout * Wout;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype)
        Y.resize(N, out_cols, X.dtype);
    const long long total = static_cast<long long>(N) * out_cols;
    if (total == 0) return;

    const float up_gain = static_cast<float>(up) * static_cast<float>(up);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    dim3 grid((Wout + FL_TW - 1) / FL_TW, (Hout + FL_TH - 1) / FL_TH,
              static_cast<unsigned>(N) * static_cast<unsigned>(C));

    auto launch = [&](auto tag) {
        using T = decltype(tag);
        filtered_lrelu_kernel<T><<<grid, FL_THREADS, shmem, stream>>>(
            static_cast<const T*>(X.data), static_cast<const T*>(fu.data),
            static_cast<const T*>(fd.data), b ? static_cast<const T*>(b->data) : nullptr,
            N, C, H, W, Hu, Wu, Hout, Wout, fuH, fuW, fdH, fdW,
            up, down, pad_x0, pad_y0, AH, AW, up_gain, gain, slope, clamp,
            static_cast<T*>(Y.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    };
    if (X.dtype == ::brotensor::Dtype::FP16)      launch(__half{});
    else if (X.dtype == ::brotensor::Dtype::BF16) launch(__nv_bfloat16{});
    else                                          launch(float{});
}

void fill_cuda_vtable_filtered_lrelu(::brotensor::detail::OpsVTable& v) {
    v.filtered_lrelu_forward = &filtered_lrelu_forward;
    // filtered_lrelu_backward stays null for now → the public dispatcher uses
    // the composite backward (FP32, correct). The fused backward is a separate
    // change that also shifts the cache contract.
}

} // namespace detail::cuda
} // namespace brotensor
