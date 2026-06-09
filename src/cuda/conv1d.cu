// ─── CUDA 1D-convolution family (brosoundml CHUNK 3) ───────────────────────
//
// CUDA port of src/cpu/conv1d.cpp. FP32-only, mirroring the CPU contracts:
//   conv_transpose1d_forward / _backward_input / _backward_weight / _backward_bias
//   causal_conv1d_update
//   pad1d_forward / pad1d_backward
//
// Layout (NCL): X / Y row n holds C channels of L samples — index
// ((n*C + c)*L + l). conv_transpose1d weights are OIL input-channel-major:
// Wt[(c_in*Cg_out + oc_local)*kL + kl], Cg_out = C_out/groups.
// causal_conv1d_update weights are depthwise: Wt[c*kL + kl].
//
// Accumulation (matches the conv2d contract):
//   *_forward / *_backward_input / pad1d_*   — output OVERWRITTEN.
//   conv_transpose1d_backward_weight / _bias — dWt / dB ACCUMULATE (+=).
//
// Where the CPU forward op scatters, the CUDA forward op gathers (one thread
// per output element, no atomics) — the same linear map with a parallel-safe
// memory pattern.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int C1D_BLOCK = 256;

inline int c1d_grid(long long n) {
    long long blocks = (n + C1D_BLOCK - 1) / C1D_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

// L_out of a 1D transposed convolution (torch ConvTranspose1d formula).
inline int convt1d_out_len(int L, int stride, int padding, int output_padding,
                           int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

// ─── conv_transpose1d_forward — gather form ─────────────────────────────────
// One thread per (n, oc, lo). The CPU scatters input (n,c_in,l) to
// lo = l*stride - padding + kl*dilation; the gather inverts that: given lo,
// l = (lo + padding - kl*dilation) / stride must be an exact integer in range.
__global__ void convt1d_forward_kernel(const float* __restrict__ X,
                                       const float* __restrict__ Wt,
                                       const float* __restrict__ bias,
                                       float* __restrict__ Y,
                                       int N, int C_in, int L, int C_out,
                                       int kL, int stride, int padding,
                                       int dilation, int Cg_in, int Cg_out,
                                       int L_out) {
    const long long total = (long long)N * C_out * L_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int lo = static_cast<int>(idx % L_out);
        const long long t = idx / L_out;
        const int oc = static_cast<int>(t % C_out);
        const int n = static_cast<int>(t / C_out);
        const int g = oc / Cg_out;
        const int oc_local = oc - g * Cg_out;
        const int c_in_base = g * Cg_in;

        float acc = bias ? bias[oc] : 0.0f;
        for (int kl = 0; kl < kL; ++kl) {
            const int num = lo + padding - kl * dilation;
            if (num < 0 || num % stride != 0) continue;
            const int l = num / stride;
            if (l < 0 || l >= L) continue;
            for (int ci = 0; ci < Cg_in; ++ci) {
                const int c_in = c_in_base + ci;
                const float xv =
                    X[((long long)n * C_in + c_in) * L + l];
                const float wv =
                    Wt[(long long)(c_in * Cg_out + oc_local) * kL + kl];
                acc += xv * wv;
            }
        }
        Y[idx] = acc;
    }
}

// ─── conv_transpose1d_backward_input ────────────────────────────────────────
// One thread per (n, c_in, l). Adjoint of the scatter is a plain gather conv.
__global__ void convt1d_bwd_input_kernel(const float* __restrict__ Wt,
                                         const float* __restrict__ dY,
                                         float* __restrict__ dX,
                                         int N, int C_in, int L, int C_out,
                                         int kL, int stride, int padding,
                                         int dilation, int Cg_in, int Cg_out,
                                         int L_out) {
    const long long total = (long long)N * C_in * L;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int l = static_cast<int>(idx % L);
        const long long t = idx / L;
        const int c_in = static_cast<int>(t % C_in);
        const int n = static_cast<int>(t / C_in);
        const int g = c_in / Cg_in;
        const int oc_base = g * Cg_out;
        const int lo_origin = l * stride - padding;

        float acc = 0.0f;
        for (int kl = 0; kl < kL; ++kl) {
            const int lo = lo_origin + kl * dilation;
            if (lo < 0 || lo >= L_out) continue;
            for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                const int oc = oc_base + oc_local;
                const float wv =
                    Wt[(long long)(c_in * Cg_out + oc_local) * kL + kl];
                const float gv =
                    dY[((long long)n * C_out + oc) * L_out + lo];
                acc += gv * wv;
            }
        }
        dX[idx] = acc;
    }
}

// ─── conv_transpose1d_backward_weight ───────────────────────────────────────
// One thread per weight element (c_in, oc_local, kl); accumulates over (n, l)
// then += into dWt (caller zeroed it). Distinct weight per thread — no atomics.
__global__ void convt1d_bwd_weight_kernel(const float* __restrict__ X,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dWt,
                                          int N, int C_in, int L, int C_out,
                                          int kL, int stride, int padding,
                                          int dilation, int Cg_in, int Cg_out,
                                          int L_out) {
    const long long total = (long long)C_in * Cg_out * kL;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int kl = static_cast<int>(idx % kL);
        const long long t = idx / kL;
        const int oc_local = static_cast<int>(t % Cg_out);
        const int c_in = static_cast<int>(t / Cg_out);
        const int g = c_in / Cg_in;
        const int oc = g * Cg_out + oc_local;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* x_row = X + ((long long)n * C_in + c_in) * L;
            const float* dy_row = dY + ((long long)n * C_out + oc) * L_out;
            for (int l = 0; l < L; ++l) {
                const int lo = l * stride - padding + kl * dilation;
                if (lo < 0 || lo >= L_out) continue;
                acc += x_row[l] * dy_row[lo];
            }
        }
        dWt[idx] += acc;
    }
}

// ─── conv_transpose1d_backward_bias ─────────────────────────────────────────
// One thread per output channel; accumulates over (N, L_out), += into dB.
__global__ void convt1d_bwd_bias_kernel(const float* __restrict__ dY,
                                        float* __restrict__ dB,
                                        int N, int C_out, int L_out) {
    for (int oc = blockIdx.x * blockDim.x + threadIdx.x; oc < C_out;
         oc += blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* dy_row = dY + ((long long)n * C_out + oc) * L_out;
            for (int lo = 0; lo < L_out; ++lo) acc += dy_row[lo];
        }
        dB[oc] += acc;
    }
}

// ─── causal_conv1d_update ───────────────────────────────────────────────────
// One thread per (n, c, t) output sample. buf is [state ++ new input];
// output t convolves buf[t .. t + (kL-1)*dilation] (causal).
__global__ void causal_conv1d_y_kernel(const float* __restrict__ X,
                                       const float* __restrict__ Wt,
                                       const float* __restrict__ bias,
                                       const float* __restrict__ state,
                                       float* __restrict__ Y,
                                       int N, int C, int L_step, int kL,
                                       int dilation, int hist) {
    const long long total = (long long)N * C * L_step;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int tt = static_cast<int>(idx % L_step);
        const long long q = idx / L_step;
        const int c = static_cast<int>(q % C);
        const int n = static_cast<int>(q / C);
        const float* s_row = state + ((long long)n * C + c) * hist;
        const float* x_row = X + ((long long)n * C + c) * L_step;
        const float* w_row = Wt + (long long)c * kL;
        float acc = bias ? bias[c] : 0.0f;
        for (int kl = 0; kl < kL; ++kl) {
            const int bidx = tt + kl * dilation;
            const float bv = (bidx < hist) ? s_row[bidx]
                                           : x_row[bidx - hist];
            acc += w_row[kl] * bv;
        }
        Y[idx] = acc;
    }
}

// State roll: new state = last `hist` samples of the [state ++ new] window.
// One thread per (n, c). Runs after the Y kernel so all state reads are done;
// the in-place roll reads index L_step+i > i, never a slot already written.
__global__ void causal_conv1d_roll_kernel(const float* __restrict__ X,
                                          float* __restrict__ state,
                                          int N, int C, int L_step, int hist) {
    const long long total = (long long)N * C;
    const int buf_len = hist + L_step;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         q < total; q += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>(q % C);
        const int n = static_cast<int>(q / C);
        float* s_row = state + ((long long)n * C + c) * hist;
        const float* x_row = X + ((long long)n * C + c) * L_step;
        for (int i = 0; i < hist; ++i) {
            const int idx = buf_len - hist + i;   // tail of the window
            s_row[i] = (idx < hist) ? s_row[idx] : x_row[idx - hist];
        }
    }
}

// ─── pad1d ──────────────────────────────────────────────────────────────────
// Map an output position p to a source index in [0, L), or -1 for a
// zero-padded slot. mode: 0 zero, 1 reflect (numpy, edge not repeated),
// 2 replicate.
__device__ inline int pad1d_src(int p, int L, int pad_left, int mode) {
    const int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;
    if (mode == 0) return -1;
    if (mode == 2) return rel < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int q = rel % period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

// pad1d_forward: one thread per (n, c, p) output sample.
__global__ void pad1d_forward_kernel(const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     int N, int C, int L, int L_pad,
                                     int pad_left, int mode) {
    const long long total = (long long)N * C * L_pad;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int p = static_cast<int>(idx % L_pad);
        const long long t = idx / L_pad;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const int src = pad1d_src(p, L, pad_left, mode);
        Y[idx] = (src < 0) ? 0.0f
                           : X[((long long)n * C + c) * L + src];
    }
}

// pad1d_backward: one thread per (n, c, l) input sample — gathers every
// output position that read it (adjoint of the copy/reflect/replicate map).
__global__ void pad1d_backward_kernel(const float* __restrict__ dY,
                                      float* __restrict__ dX,
                                      int N, int C, int L, int L_pad,
                                      int pad_left, int mode) {
    const long long total = (long long)N * C * L;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int l = static_cast<int>(idx % L);
        const long long t = idx / L;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const float* dy_row = dY + ((long long)n * C + c) * L_pad;
        float acc = 0.0f;
        for (int p = 0; p < L_pad; ++p) {
            if (pad1d_src(p, L, pad_left, mode) == l) acc += dy_row[p];
        }
        dX[idx] = acc;
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

void conv_transpose1d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose1d_forward";
    require_fp32(op, X, "X");
    require_fp32(op, Wt, "Wt");
    if (bias) require_fp32(op, *bias, "bias");
    check_groups(op, C_in, C_out, groups);
    if (kL < 1 || stride < 1 || dilation < 1 || padding < 0
        || output_padding < 0) {
        fail(op, "kL/stride/dilation must be >=1 and padding/output_padding >=0");
    }
    if (output_padding >= stride && output_padding >= dilation) {
        fail(op, "output_padding must be < stride or < dilation");
    }
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * L_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, out_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;
    const long long total = (long long)N * C_out * L_out;
    convt1d_forward_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(Wt.data),
        bias ? static_cast<const float*>(bias->data) : nullptr,
        static_cast<float*>(Y.data),
        N, C_in, L, C_out, kL, stride, padding, dilation, Cg_in, Cg_out, L_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose1d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, ::brotensor::Tensor& dX) {
    const char* op = "conv_transpose1d_backward_input";
    require_fp32(op, Wt, "Wt");
    require_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (Wt.rows != C_in || Wt.cols != Cg_out * kL) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    const int in_cols = C_in * L;
    if (dX.rows != N || dX.cols != in_cols
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;
    const long long total = (long long)N * C_in * L;
    convt1d_bwd_input_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(Wt.data), static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data),
        N, C_in, L, C_out, kL, stride, padding, dilation, Cg_in, Cg_out, L_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose1d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, ::brotensor::Tensor& dWt) {
    const char* op = "conv_transpose1d_backward_weight";
    require_fp32(op, X, "X");
    require_fp32(op, dY, "dY");
    require_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int L_out = convt1d_out_len(L, stride, padding, output_padding,
                                      dilation, kL);
    if (L_out <= 0) fail(op, "non-positive output length");
    if (dWt.rows != C_in || dWt.cols != Cg_out * kL) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kL)");
    }
    if (X.rows != N || X.cols != C_in * L) {
        fail(op, "X shape must be (N, C_in*L)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kL == 0) return;
    const long long total = (long long)C_in * Cg_out * kL;
    convt1d_bwd_weight_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
        static_cast<float*>(dWt.data),
        N, C_in, L, C_out, kL, stride, padding, dilation, Cg_in, Cg_out, L_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose1d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int L_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose1d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * L_out) {
        fail(op, "dY shape must be (N, C_out*L_out)");
    }
    if (C_out == 0 || N == 0 || L_out == 0) return;
    convt1d_bwd_bias_kernel<<<c1d_grid(C_out), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dB.data),
        N, C_out, L_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_update(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& Wt,
                          const ::brotensor::Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          ::brotensor::Tensor& state, ::brotensor::Tensor& Y) {
    const char* op = "causal_conv1d_update";
    require_fp32(op, X, "X");
    require_fp32(op, Wt, "Wt");
    if (bias) require_fp32(op, *bias, "bias");
    require_fp32(op, state, "state");
    if (kL < 1 || dilation < 1 || L_step < 1 || N < 0 || C < 1) {
        fail(op, "kL/dilation/L_step/C must be >=1 and N >=0");
    }
    if (Wt.rows != C || Wt.cols != kL) {
        fail(op, "Wt shape must be (C, kL) — one depthwise filter per channel");
    }
    if (bias && (bias->rows != C || bias->cols != 1)) {
        fail(op, "bias shape must be (C, 1)");
    }
    const int hist = (kL - 1) * dilation;
    if (state.rows != N || state.cols != C * hist) {
        fail(op, "state shape must be (N, C*(kL-1)*dilation)");
    }
    if (Y.rows != N || Y.cols != C * L_step
        || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, C * L_step, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0 || L_step == 0) return;
    const long long total = (long long)N * C * L_step;
    causal_conv1d_y_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(Wt.data),
        bias ? static_cast<const float*>(bias->data) : nullptr,
        static_cast<const float*>(state.data), static_cast<float*>(Y.data),
        N, C, L_step, kL, dilation, hist);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    // Roll the state forward — strictly after the Y kernel (same stream).
    if (hist > 0) {
        causal_conv1d_roll_kernel<<<c1d_grid((long long)N * C), C1D_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(state.data), N, C, L_step, hist);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

void pad1d_forward(const ::brotensor::Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode,
                   ::brotensor::Tensor& Y) {
    const char* op = "pad1d_forward";
    require_fp32(op, X, "X");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    if (X.rows != N || X.cols != C * L) fail(op, "X shape must be (N, C*L)");
    const int L_pad = L + pad_left + pad_right;
    if (Y.rows != N || Y.cols != C * L_pad
        || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, C * L_pad, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;
    const long long total = (long long)N * C * L_pad;
    pad1d_forward_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data),
        N, C, L, L_pad, pad_left, mode);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void pad1d_backward(const ::brotensor::Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode,
                    ::brotensor::Tensor& dX) {
    const char* op = "pad1d_backward";
    require_fp32(op, dY, "dY");
    if (N < 0 || C < 1 || L < 1) fail(op, "C/L must be >=1 and N >=0");
    if (pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1 && (pad_left >= L || pad_right >= L)) {
        fail(op, "reflect padding requires pad_left and pad_right < L");
    }
    const int L_pad = L + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * L_pad) {
        fail(op, "dY shape must be (N, C*(L+pad_left+pad_right))");
    }
    if (dX.rows != N || dX.cols != C * L
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, C * L, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;
    const long long total = (long long)N * C * L;
    pad1d_backward_kernel<<<c1d_grid(total), C1D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        N, C, L, L_pad, pad_left, mode);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_conv1d(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose1d_forward         = &conv_transpose1d_forward;
    v.conv_transpose1d_backward_input  = &conv_transpose1d_backward_input;
    v.conv_transpose1d_backward_weight = &conv_transpose1d_backward_weight;
    v.conv_transpose1d_backward_bias   = &conv_transpose1d_backward_bias;
    v.causal_conv1d_update             = &causal_conv1d_update;
    v.pad1d_forward                    = &pad1d_forward;
    v.pad1d_backward                   = &pad1d_backward;
}

} // namespace brotensor::detail::cuda
