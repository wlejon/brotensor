// ─── CUDA fused filtered_lrelu (StyleGAN3 alias-free nonlinearity) ──────────
//
// The composite filtered_lrelu is bias_act → upfirdn2d(up, gain=up²) →
// bias_act(lrelu) → upfirdn2d(down): 3–4 full global-memory passes at the
// up-sampled (2×) resolution per layer. This fuses the whole chain into one
// kernel that reads X (+bias) directly and writes Y directly — no global
// up_buf/act_buf intermediates — so the up-FIR, the leaky-ReLU/clamp, and the
// down-FIR all happen in registers/shared. That removes the memory passes the
// composite spends at 2× resolution.
//
// Polyphase. The up-FIR has zero-insertion (stride `up`) and a flipped filter,
// so a naïve loop steps the filter at stride −up and recomputes a residue
// modulo per cell — strided, branchy, and a poor fit for the compiler. We
// instead precompute, on-device, the `up²` dense polyphase banks of the up
// filter (and a pre-flipped dense down filter) so the per-cell inner loops read
// BOTH the input and the filter contiguously, as a tight dense double-loop the
// compiler pipelines well. That alone ~1.8×'d the fused kernel over the strided
// version, for every dtype. (half2 FMAs were tried for the FP16/BF16 FIRs and
// measured *slower* — the kernel is conversion/issue-bound on these short
// inner loops, not FMA-rate-bound, so the half↔float packing cost the half2
// throughput can't repay; the scalar template wins for all three dtypes.) FP32
// stays bit-identical to the reference (same tap order). Supports the general
// non-separable 2D filters config-R uses (radial down-filter); FP32/FP16/BF16
// storage, mirroring `_filtered_lrelu_ref`.
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
void* cuda_alloc(std::size_t);
void  cuda_free(void*);

namespace {

// Output tile per CTA. TH*TW threads; each CTA computes the AH×AW activation
// region once into shared memory, then downsamples it. 16×16 output tile (256
// threads) measured best across config-R sizes — enough warps to hide latency
// while keeping the shared act tile (≈36×36 elems for 6-tap filters) small
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

// ── Polyphase filter builders (run once per call, on-device) ────────────────
// poly_u: the up filter reorganized into `up*up` dense banks. For activation
// residue (ry,rx) the contributing up-taps are fuy=ry+j·up, fux=rx+i·up — a
// dense JyMax×JxMax block with no zero-inserted gaps — stored already flipped
// (frow = fuH−1−fuy) so the kernel reads weights in input order. Short residues
// (fuy/fux ≥ filter extent) are zero-padded; those taps contribute nothing.
template <typename T>
__global__ void build_polyu(const T* __restrict__ fu, int fuH, int fuW, int up,
                            int JyMax, int JxMax, T* __restrict__ poly) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = up * up * JyMax * JxMax;
    if (idx >= total) return;
    int t = idx;
    const int i = t % JxMax; t /= JxMax;
    const int j = t % JyMax; t /= JyMax;
    const int rx = t % up;   t /= up;
    const int ry = t;
    const int fuy = ry + j * up, fux = rx + i * up;
    const float v = (fuy < fuH && fux < fuW)
                  ? fl_load<T>(&fu[(fuH - 1 - fuy) * fuW + (fuW - 1 - fux)]) : 0.0f;
    fl_store<T>(&poly[idx], v);
}

// poly_d: the down filter pre-flipped into a dense [fdH][fdW] block so Phase B
// reads it contiguously alongside the contiguous shared activations.
template <typename T>
__global__ void build_polyd(const T* __restrict__ fd, int fdH, int fdW,
                            T* __restrict__ poly) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fdH * fdW) return;
    const int fdx = idx % fdW, fdy = idx / fdW;
    fl_store<T>(&poly[idx], fl_load<T>(&fd[(fdH - 1 - fdy) * fdW + (fdW - 1 - fdx)]));
}

// ── FIR row dot-products ────────────────────────────────────────────────────
// Both phases reduce a contiguous filter×data row in FP32 (the bit-exact
// reference order for FP32; the dense polyphase layout makes both operands
// contiguous so the compiler unrolls/pipelines these tight loops).
//
// Phase A row: Σ_{i<n} pol[i]·(x[i]+bias).
template <typename T>
__device__ inline float pa_rowdot(const T* pol, const T* x, int n, float bias) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += fl_load<T>(&pol[i]) * (fl_load<T>(&x[i]) + bias);
    return s;
}
// Phase B row: Σ_{i<n} pol[i]·act[i] (no bias).
template <typename T>
__device__ inline float pb_rowdot(const T* pol, const T* act, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += fl_load<T>(&pol[i]) * fl_load<T>(&act[i]);
    return s;
}

// One CTA per (n, c, output-tile). Phase A computes the activation region the
// tile needs — exactly once each — into a shared tile of type T:
//   act[ay,ax] = clamp( lrelu( up² · Σ_polyphase fu·(X[iy,ix]+b[c]) ) · gain )
// using the dense polyphase banks (iy=iyb+j, ix=ixb+i contiguous). Phase B
// downsamples from shared with the pre-flipped dense down filter:
//   Y[oh,ow] = Σ poly_d[fdy,fdx] · act[(oh−oh0)·down+fdy, (ow−ow0)·down+fdx]
// Computing each activation once (vs per output tap) is the whole point — the
// down-FIR footprints of neighbouring outputs overlap heavily.
template <typename T>
__global__ void filtered_lrelu_kernel(
        const T* __restrict__ X, const T* __restrict__ poly_u, const T* __restrict__ poly_d,
        const T* __restrict__ b, int N, int C, int H, int W,
        int Hout, int Wout, int JyMax, int JxMax, int fdH, int fdW,
        int up, int down, int px0, int py0,
        int AH, int AW, float up_gain, float gain, float slope, float clamp,
        T* __restrict__ Y) {
    extern __shared__ char smem[];
    T* act = reinterpret_cast<T*>(smem);   // AH * AW activations, storage dtype

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
    const int bank = JyMax * JxMax;
    for (int a = tid; a < acells; a += FL_THREADS) {
        const int ly = a / AW, lx = a % AW;
        const int ay = ay0 + ly, ax = ax0 + lx;
        const int ry = (((py0 - ay) % up) + up) % up;   // first contributing residue
        const int rx = (((px0 - ax) % up) + up) % up;
        const int iyb = (ay + ry - py0) / up;           // input row at j=0 (exact)
        const int ixb = (ax + rx - px0) / up;
        const int jlo = iyb < 0 ? -iyb : 0;
        const int jhi = (H - iyb) < JyMax ? (H - iyb) : JyMax;
        const int ilo = ixb < 0 ? -ixb : 0;
        const int ihi = (W - ixb) < JxMax ? (W - ixb) : JxMax;
        const T* polbase = poly_u + (ry * up + rx) * bank;
        float u = 0.0f;
        for (int j = jlo; j < jhi; ++j) {
            const T* xr  = &X[in_base + static_cast<size_t>(iyb + j) * W + (ixb + ilo)];
            const T* pol = polbase + j * JxMax + ilo;
            u += pa_rowdot<T>(pol, xr, ihi - ilo, bias_v);
        }
        u *= up_gain;
        float av = (u > 0.0f ? u : slope * u) * gain;
        if (clamp >= 0.0f) {
            if (av < -clamp) av = -clamp;
            else if (av > clamp) av = clamp;
        }
        fl_store<T>(&act[a], av);
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
            const T* ar  = &act[(ly0 + fdy) * AW + lx0];
            const T* pol = poly_d + fdy * fdW;
            acc += pb_rowdot<T>(pol, ar, fdW);
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
    // every config this fused kernel doesn't. shmem checked at the FP32 worst
    // case (the reduced-precision tile is half that).
    const int AH = (FL_TH - 1) * down + fdH;   // shared activation tile extent
    const int AW = (FL_TW - 1) * down + fdW;
    const size_t shmem_max = static_cast<size_t>(AH) * AW * sizeof(float);

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool supported =
        is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) && up >= 1 && down >= 1 &&
        Huo > 0 && Wuo > 0 && Hout > 0 && Wout > 0 &&
        shmem_max <= 48u * 1024u &&        // larger tiles fall back to composite
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
        // Dense polyphase banks for the up filter + pre-flipped down filter,
        // built on the current stream into pooled scratch (no host round-trip,
        // no device sync). Both are tiny (≤ a few hundred elems).
        const int JyMax = (fuH + up - 1) / up;
        const int JxMax = (fuW + up - 1) / up;
        const size_t polyu_n = static_cast<size_t>(up) * up * JyMax * JxMax;
        const size_t polyd_n = static_cast<size_t>(fdH) * fdW;
        T* poly_u = static_cast<T*>(cuda_alloc(polyu_n * sizeof(T)));
        T* poly_d = static_cast<T*>(cuda_alloc(polyd_n * sizeof(T)));
        build_polyu<T><<<static_cast<unsigned>((polyu_n + 127) / 128), 128, 0, stream>>>(
            static_cast<const T*>(fu.data), fuH, fuW, up, JyMax, JxMax, poly_u);
        build_polyd<T><<<static_cast<unsigned>((polyd_n + 127) / 128), 128, 0, stream>>>(
            static_cast<const T*>(fd.data), fdH, fdW, poly_d);

        const size_t shmem = static_cast<size_t>(AH) * AW * sizeof(T);
        filtered_lrelu_kernel<T><<<grid, FL_THREADS, shmem, stream>>>(
            static_cast<const T*>(X.data), poly_u, poly_d,
            b ? static_cast<const T*>(b->data) : nullptr,
            N, C, H, W, Hout, Wout, JyMax, JxMax, fdH, fdW,
            up, down, pad_x0, pad_y0, AH, AW, up_gain, gain, slope, clamp,
            static_cast<T*>(Y.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(poly_u);
        cuda_free(poly_d);
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
