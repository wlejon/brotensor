// Native AMD HIP Audio DSP: Vocoder Activations, Codec Quantization, 1D Resampling, and Sample Logits.
//
// Supports RDNA 3.5 (gfx1151).

#include "detail/audio_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int DSP_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int grid_for(long long n) {
    long long b = (n + DSP_BLOCK - 1) / DSP_BLOCK;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

// ─── Vocoder Activations ───────────────────────────────────────────────────

__device__ inline float guard_denom(float d) {
    constexpr float kMin = 1e-9f;
    if (d >= 0.0f) return d < kMin ? kMin : d;
    return d > -kMin ? -kMin : d;
}

__global__ void snake_forward_kernel(const float* __restrict__ X,
                                     const float* __restrict__ alpha,
                                     const float* __restrict__ beta,
                                     int C, int L, long long total,
                                     float* __restrict__ Y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>((i / L) % C);
        const float a = alpha[c];
        const float r = 1.0f / guard_denom(beta ? beta[c] : a);
        const float x = X[i];
        const float s = sinf(a * x);
        Y[i] = x + r * s * s;
    }
}

__global__ void snake_backward_kernel(const float* __restrict__ X,
                                      const float* __restrict__ alpha,
                                      const float* __restrict__ beta,
                                      const float* __restrict__ dY,
                                      int C, int L, long long total,
                                      float* __restrict__ dX,
                                      float* __restrict__ dAlpha,
                                      float* __restrict__ dBeta) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>((i / L) % C);
        const float a = alpha[c];
        const float r = 1.0f / guard_denom(beta ? beta[c] : a);
        const float x  = X[i];
        const float dy = dY[i];
        const float s  = sinf(a * x);
        const float co = cosf(a * x);
        const float sc = s * co;

        // dy/dx = 1 + 2*a*r*s*c
        dX[i] = dy * (1.0f + 2.0f * a * r * sc);

        // dy/dalpha frequency term = 2*r*x*s*c
        float dalpha = dy * (2.0f * r * x * sc);
        if (beta) {
            // snakebeta: dy/dbeta = -r^2 * s^2.
            atomicAdd(&dBeta[c], dy * (-r * r * s * s));
        } else {
            // plain snake: denom == alpha — fold the -r^2*s^2 term into dAlpha.
            dalpha += dy * (-r * r * s * s);
        }
        atomicAdd(&dAlpha[c], dalpha);
    }
}

__global__ void elu_forward_kernel(const float* __restrict__ x, float alpha, float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    }
}

__global__ void elu_backward_kernel(const float* __restrict__ x, const float* __restrict__ dY,
                                    float alpha, float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = x[i];
        dX[i] = dY[i] * (v > 0.0f ? 1.0f : alpha * expf(v));
    }
}

template <typename T>
__global__ void leaky_relu_forward_kernel(const T* __restrict__ x, float negative_slope, T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v;
        if constexpr (std::is_same_v<T, float>) v = x[i];
        else if constexpr (std::is_same_v<T, __half>) v = __half2float(x[i]);
        else v = float(x[i]);

        const float out = v > 0.0f ? v : negative_slope * v;

        if constexpr (std::is_same_v<T, float>) y[i] = out;
        else if constexpr (std::is_same_v<T, __half>) y[i] = __float2half(out);
        else y[i] = hip_bfloat16(out);
    }
}

__global__ void leaky_relu_backward_kernel(const float* __restrict__ x, const float* __restrict__ dY,
                                           float negative_slope, float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dX[i] = dY[i] * (x[i] > 0.0f ? 1.0f : negative_slope);
    }
}

// ─── Codec Quantization ────────────────────────────────────────────────────

__global__ void vq_encode_kernel(const float* __restrict__ x, const float* __restrict__ codebook,
                                 int32_t* __restrict__ indices, float* __restrict__ quantized,
                                 int N, int D, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* x_n = x + static_cast<size_t>(n) * D;

    float best_dist = 1e38f;
    int best_k = 0;
    for (int k = 0; k < K; ++k) {
        const float* cb_k = codebook + static_cast<size_t>(k) * D;
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float diff = x_n[d] - cb_k[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }
    indices[n] = best_k;
    const float* cb_best = codebook + static_cast<size_t>(best_k) * D;
    float* q_n = quantized + static_cast<size_t>(n) * D;
    for (int d = 0; d < D; ++d) {
        q_n[d] = cb_best[d];
    }
}

__global__ void fsq_quantize_kernel(const float* __restrict__ x,
                                    const int32_t* __restrict__ levels,
                                    float* __restrict__ quantized,
                                    int32_t* __restrict__ packed,
                                    int N, int D) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.x) {
        const float* x_row = x + static_cast<size_t>(n) * D;
        float* q_row = quantized + static_cast<size_t>(n) * D;
        long long code = 0;
        for (int d = D - 1; d >= 0; --d) {
            const int L = levels[d];
            const float h = static_cast<float>(L - 1) * 0.5f;
            float v = x_row[d];
            if (v < -1.0f) v = -1.0f;
            else if (v > 1.0f) v = 1.0f;
            float idx_f = roundf((v + 1.0f) * 0.5f * static_cast<float>(L - 1));
            int idx = static_cast<int>(idx_f);
            if (idx < 0) idx = 0;
            else if (idx > L - 1) idx = L - 1;
            q_row[d] = static_cast<float>(idx) / h - 1.0f;
            code = code * static_cast<long long>(L) + static_cast<long long>(idx);
        }
        packed[n] = static_cast<int32_t>(code);
    }
}

// ─── 1D Resampling ─────────────────────────────────────────────────────────

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void resample1d_forward_kernel(const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int N, int C, int L_in, int L_out,
                                          int mode, double scale) {
    const long long total = (long long)N * C * L_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int dst = static_cast<int>(idx % L_out);
        const long long t = idx / L_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * L_in;
        const double src = (dst + 0.5) * scale - 0.5;
        if (mode == 0) {
            const int i = clampi(static_cast<int>(nearbyint(src)), 0, L_in - 1);
            Y[idx] = X[xbase + i];
        } else {
            double s = src;
            if (s < 0.0) s = 0.0;
            if (s > L_in - 1) s = L_in - 1;
            const int x0 = static_cast<int>(floor(s));
            const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
            const float f = static_cast<float>(s - x0);
            Y[idx] = (1.0f - f) * X[xbase + x0] + f * X[xbase + x1];
        }
    }
}

__global__ void resample1d_backward_kernel(const float* __restrict__ dY,
                                           float* __restrict__ dX,
                                           int N, int C, int L_in, int L_out,
                                           int mode, double scale) {
    const long long total = (long long)N * C * L_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int dst = static_cast<int>(idx % L_out);
        const long long t = idx / L_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * L_in;
        const double src = (dst + 0.5) * scale - 0.5;
        const float g = dY[idx];
        if (mode == 0) {
            const int i = clampi(static_cast<int>(nearbyint(src)), 0, L_in - 1);
            atomicAdd(&dX[xbase + i], g);
        } else {
            double s = src;
            if (s < 0.0) s = 0.0;
            if (s > L_in - 1) s = L_in - 1;
            const int x0 = static_cast<int>(floor(s));
            const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
            const float f = static_cast<float>(s - x0);
            atomicAdd(&dX[xbase + x0], (1.0f - f) * g);
            atomicAdd(&dX[xbase + x1], f * g);
        }
    }
}

// ─── Autoregressive Logit Sampling ─────────────────────────────────────────

__device__ inline void mulhilo32(uint32_t a, uint32_t b,
                                 uint32_t& hi, uint32_t& lo) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product);
}

__device__ inline void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr[0], hi0, lo0);
    mulhilo32(0xCD9E8D57u, ctr[2], hi1, lo1);
    const uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    const uint32_t new1 = lo1;
    const uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    const uint32_t new3 = lo0;
    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

__device__ inline float philox_uniform(uint64_t key64, uint64_t substream) {
    uint32_t key[2] = {
        static_cast<uint32_t>(key64 & 0xFFFFFFFFull),
        static_cast<uint32_t>(key64 >> 32),
    };
    uint32_t ctr[4] = {
        static_cast<uint32_t>(substream & 0xFFFFFFFFull),
        static_cast<uint32_t>(substream >> 32),
        0u,
        0u,
    };
    for (int round = 0; round < 10; ++round) {
        philox_round(ctr, key);
        if (round < 9) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
    }
    return static_cast<float>(ctr[0] >> 8) / 16777216.0f;
}

constexpr float kNeg = -3.4028235e38f;
constexpr int SLI_THREADS = 256;

__device__ inline void sample_logits_row(
        const float* __restrict__ rowL, int* __restrict__ indices, int row,
        float* __restrict__ prob_n, float* __restrict__ work_n,
        int* __restrict__ ord_n, double* __restrict__ sdv,
        int* __restrict__ sidx, int V, float temperature, int top_k,
        float top_p, uint64_t key, uint64_t base_counter, int tid, int nth) {
    if (temperature == 0.0f) {
        double bv = -1e300; int bi = 0;
        for (int v = tid; v < V; v += nth) {
            double x = rowL[v];
            if (x > bv) { bv = x; bi = v; }
        }
        sdv[tid] = bv; sidx[tid] = bi; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (sdv[tid + s] > sdv[tid] ||
                    (sdv[tid + s] == sdv[tid] && sidx[tid + s] < sidx[tid])) {
                    sdv[tid] = sdv[tid + s]; sidx[tid] = sidx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) indices[row] = sidx[0];
        return;
    }

    double lmax = -1e300;
    for (int v = tid; v < V; v += nth) {
        double s = static_cast<double>(rowL[v]) / temperature;
        if (s > lmax) lmax = s;
    }
    sdv[tid] = lmax; __syncthreads();
    for (int s = nth >> 1; s > 0; s >>= 1) {
        if (tid < s) sdv[tid] = (sdv[tid + s] > sdv[tid]) ? sdv[tid + s] : sdv[tid];
        __syncthreads();
    }
    const float maxs = static_cast<float>(sdv[0]);
    __syncthreads();

    double lsum = 0.0;
    for (int v = tid; v < V; v += nth) {
        const float e = expf(static_cast<float>(rowL[v]) / temperature - maxs);
        prob_n[v] = e;
        lsum += e;
    }
    sdv[tid] = lsum; __syncthreads();
    for (int s = nth >> 1; s > 0; s >>= 1) {
        if (tid < s) sdv[tid] += sdv[tid + s];
        __syncthreads();
    }
    const float inv_sum = (sdv[0] > 0.0) ? static_cast<float>(1.0 / sdv[0]) : 0.0f;
    __syncthreads();

    for (int v = tid; v < V; v += nth) {
        const float p = prob_n[v] * inv_sum;
        prob_n[v] = p;
        work_n[v] = p;
    }

    int keep_max = V;
    if (top_k > 0 && top_k < keep_max) keep_max = top_k;

    __shared__ double s_cum;
    __shared__ int    s_keep;
    __shared__ int    s_stop;
    if (tid == 0) { s_cum = 0.0; s_keep = 0; s_stop = 0; }

    for (int r = 0; r < keep_max; ++r) {
        __syncthreads();
        double bv = -1e300; int bi = 0;
        for (int v = tid; v < V; v += nth) {
            double pv = work_n[v];
            if (pv > bv) { bv = pv; bi = v; }
        }
        sdv[tid] = bv; sidx[tid] = bi; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (sdv[tid + s] > sdv[tid] ||
                    (sdv[tid + s] == sdv[tid] && sidx[tid + s] < sidx[tid])) {
                    sdv[tid] = sdv[tid + s]; sidx[tid] = sidx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int idx = sidx[0];
            ord_n[r] = idx;
            work_n[idx] = kNeg;
            s_cum += prob_n[idx];
            if ((top_p < 1.0f && s_cum >= static_cast<double>(top_p)) ||
                r + 1 == keep_max) {
                s_keep = r + 1;
                s_stop = 1;
            }
        }
        __syncthreads();
        if (s_stop) break;
    }
    int keep = s_keep;
    if (keep < 1) keep = 1;

    if (tid == 0) {
        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += prob_n[ord_n[r]];
        const float u = philox_uniform(key, base_counter + static_cast<uint64_t>(row));
        int chosen = ord_n[0];
        if (kept_sum > 0.0) {
            const double target = static_cast<double>(u) * kept_sum;
            double acc = 0.0;
            chosen = ord_n[keep - 1];
            for (int r = 0; r < keep; ++r) {
                acc += prob_n[ord_n[r]];
                if (target < acc) { chosen = ord_n[r]; break; }
            }
        }
        indices[row] = chosen;
    }
}

__global__ void sample_logits_block_kernel(const float* __restrict__ logits,
                                           int* __restrict__ indices,
                                           float* __restrict__ prob,
                                           float* __restrict__ work,
                                           int* __restrict__ order,
                                           int N, int V, float temperature,
                                           int top_k, float top_p,
                                           uint64_t key, uint64_t counter) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ char smem[];
    double* sdv  = reinterpret_cast<double*>(smem);
    int*    sidx = reinterpret_cast<int*>(sdv + nth);

    const float* rowL = logits + static_cast<long long>(row) * V;
    float* prob_n = prob + static_cast<long long>(row) * V;
    float* work_n = work + static_cast<long long>(row) * V;
    int*   ord_n  = order + static_cast<long long>(row) * V;

    sample_logits_row(rowL, indices, row, prob_n, work_n, ord_n, sdv, sidx,
                      V, temperature, top_k, top_p, key, counter, tid, nth);
}

__global__ void sample_logits_into_kernel(const float* __restrict__ logits,
                                          int* __restrict__ indices,
                                          float* __restrict__ prob,
                                          float* __restrict__ work,
                                          int* __restrict__ order,
                                          const int* __restrict__ counter,
                                          int N, int V, float temperature,
                                          int top_k, float top_p, uint64_t key) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ char smem[];
    double* sdv  = reinterpret_cast<double*>(smem);
    int*    sidx = reinterpret_cast<int*>(sdv + nth);

    const float* rowL = logits + static_cast<long long>(row) * V;
    float* prob_n = prob + static_cast<long long>(row) * V;
    float* work_n = work + static_cast<long long>(row) * V;
    int*   ord_n  = order + static_cast<long long>(row) * V;

    const uint64_t base_counter = static_cast<uint64_t>(
        static_cast<uint32_t>(counter[0]));
    sample_logits_row(rowL, indices, row, prob_n, work_n, ord_n, sdv, sidx,
                      V, temperature, top_k, top_p, key, base_counter, tid, nth);
}

__global__ void advance_counter_kernel(int* counter, int N) {
    counter[0] = static_cast<int>(
        static_cast<uint32_t>(counter[0]) + static_cast<uint32_t>(N));
}

} // namespace

void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 || alpha.dtype != Dtype::FP32 || (beta && beta->dtype != Dtype::FP32))
        throw std::runtime_error("snake_forward: tensors must be FP32");
    const long long total = static_cast<long long>(N) * C * L;
    if (Y.size() != total || Y.dtype != Dtype::FP32) Y.resize(N, C * L, Dtype::FP32);
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    snake_forward_kernel<<<grid_for(total), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(alpha.data),
        beta ? static_cast<const float*>(beta->data) : nullptr,
        C, L, total, static_cast<float*>(Y.data));
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 || alpha.dtype != Dtype::FP32 || dY.dtype != Dtype::FP32 ||
        (beta && beta->dtype != Dtype::FP32))
        throw std::runtime_error("snake_backward: tensors must be FP32");
    const long long total = static_cast<long long>(N) * C * L;
    if (dX.size() != total || dX.dtype != Dtype::FP32) dX.resize(N, C * L, Dtype::FP32);
    if (dAlpha.size() != C || dAlpha.dtype != Dtype::FP32) dAlpha.resize(C, 1, Dtype::FP32);
    if (dBeta && (dBeta->size() != C || dBeta->dtype != Dtype::FP32)) dBeta->resize(C, 1, Dtype::FP32);
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    snake_backward_kernel<<<grid_for(total), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(alpha.data),
        beta ? static_cast<const float*>(beta->data) : nullptr,
        static_cast<const float*>(dY.data), C, L, total,
        static_cast<float*>(dX.data),
        static_cast<float*>(dAlpha.data),
        dBeta ? static_cast<float*>(dBeta->data) : nullptr);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void elu_forward(const Tensor& x, float alpha, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) y.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    hipStream_t stream = cur_stream();
    elu_forward_kernel<<<grid_for(n), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(x.data), alpha, static_cast<float*>(y.data), n);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX) {
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) dX.resize(dY.rows, dY.cols, dY.dtype);
    const int n = dY.size();
    if (n == 0) return;
    hipStream_t stream = cur_stream();
    elu_backward_kernel<<<grid_for(n), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(dY.data), alpha, static_cast<float*>(dX.data), n);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y) {
    using ::brotensor::Dtype;
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) y.resize(x.rows, x.cols, x.dtype);
    const int n = x.size();
    if (n == 0) return;
    hipStream_t stream = cur_stream();
    if (x.dtype == Dtype::FP32) {
        leaky_relu_forward_kernel<float><<<grid_for(n), DSP_BLOCK, 0, stream>>>(
            static_cast<const float*>(x.data), negative_slope, static_cast<float*>(y.data), n);
    } else if (x.dtype == Dtype::FP16) {
        leaky_relu_forward_kernel<__half><<<grid_for(n), DSP_BLOCK, 0, stream>>>(
            static_cast<const __half*>(x.data), negative_slope, static_cast<__half*>(y.data), n);
    } else {
        leaky_relu_forward_kernel<hip_bfloat16><<<grid_for(n), DSP_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(x.data), negative_slope, static_cast<hip_bfloat16*>(y.data), n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void leaky_relu_backward(const Tensor& x, const Tensor& dY, float negative_slope, Tensor& dX) {
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) dX.resize(dY.rows, dY.cols, dY.dtype);
    const int n = dY.size();
    if (n == 0) return;
    hipStream_t stream = cur_stream();
    leaky_relu_backward_kernel<<<grid_for(n), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(dY.data), negative_slope, static_cast<float*>(dX.data), n);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void vq_encode_forward(const Tensor& x, const Tensor& codebook, Tensor& indices, Tensor& quantized) {
    using ::brotensor::Dtype;
    const int N = x.rows;
    const int D = x.cols;
    const int K = codebook.rows;
    if (indices.rows != N || indices.cols != 1 || indices.dtype != Dtype::INT32) {
        indices.resize(N, 1, Dtype::INT32);
    }
    if (quantized.rows != N || quantized.cols != D || quantized.dtype != Dtype::FP32) {
        quantized.resize(N, D, Dtype::FP32);
    }
    if (N == 0 || D == 0 || K == 0) return;
    hipStream_t stream = cur_stream();
    vq_encode_kernel<<<grid_for(N), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(codebook.data),
        static_cast<int32_t*>(indices.data), static_cast<float*>(quantized.data),
        N, D, K);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void vq_encode_backward(const Tensor& dQuantized, Tensor& dX) {
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols || dX.dtype != dQuantized.dtype) {
        dX.resize(dQuantized.rows, dQuantized.cols, dQuantized.dtype);
    }
    if (dX.size() == 0) return;
    BROTENSOR_HIP_CHECK(hipMemcpyAsync(dX.data, dQuantized.data, dQuantized.bytes(), hipMemcpyDeviceToDevice, cur_stream()));
}

void fsq_quantize_forward(const Tensor& x, const Tensor& levels, Tensor& quantized, Tensor& packed_indices) {
    using ::brotensor::Dtype;
    const int N = x.rows;
    const int D = x.cols;
    if (quantized.rows != N || quantized.cols != D || quantized.dtype != Dtype::FP32) {
        quantized.resize(N, D, Dtype::FP32);
    }
    if (packed_indices.rows != N || packed_indices.cols != 1 || packed_indices.dtype != Dtype::INT32) {
        packed_indices.resize(N, 1, Dtype::INT32);
    }
    if (N == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    std::vector<int32_t> host_levels(static_cast<size_t>(D));
    BROTENSOR_HIP_CHECK(hipMemcpyAsync(
        host_levels.data(), levels.data,
        static_cast<size_t>(D) * sizeof(int32_t), hipMemcpyDeviceToHost,
        stream));
    BROTENSOR_HIP_CHECK(hipStreamSynchronize(stream));
    for (int d = 0; d < D; ++d) {
        if (host_levels[d] < 2) {
            throw std::runtime_error("fsq_quantize_forward: every level count must be >= 2");
        }
    }

    fsq_quantize_kernel<<<grid_for(N), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(x.data), static_cast<const int32_t*>(levels.data),
        static_cast<float*>(quantized.data), static_cast<int32_t*>(packed_indices.data), N, D);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fsq_quantize_backward(const Tensor& dQuantized, Tensor& dX) {
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols || dX.dtype != dQuantized.dtype) {
        dX.resize(dQuantized.rows, dQuantized.cols, dQuantized.dtype);
    }
    if (dX.size() == 0) return;
    BROTENSOR_HIP_CHECK(hipMemcpyAsync(dX.data, dQuantized.data, dQuantized.bytes(), hipMemcpyDeviceToDevice, cur_stream()));
}

void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out, int mode, Tensor& Y) {
    using ::brotensor::Dtype;
    const int cols = C * L_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;
    const double scale = static_cast<double>(L_in) / static_cast<double>(L_out);
    const long long total = (long long)N * C * L_out;
    hipStream_t stream = cur_stream();
    resample1d_forward_kernel<<<grid_for(total), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data),
        N, C, L_in, L_out, mode, scale);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out, int mode, Tensor& dX) {
    using ::brotensor::Dtype;
    const int cols_in = C * L_in;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;
    hipStream_t stream = cur_stream();
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, static_cast<size_t>(N) * cols_in * sizeof(float), stream));
    if (L_out == 0) return;
    const double scale = static_cast<double>(L_in) / static_cast<double>(L_out);
    const long long total = (long long)N * C * L_out;
    resample1d_backward_kernel<<<grid_for(total), DSP_BLOCK, 0, stream>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        N, C, L_in, L_out, mode, scale);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void sample_logits(const Tensor& logits, float temperature, int top_k, float top_p,
                   uint64_t key, uint64_t counter, Tensor& indices) {
    using ::brotensor::Dtype;
    if (logits.dtype != Dtype::FP32) throw std::runtime_error("sample_logits: logits must be FP32");
    if (temperature < 0.0f) throw std::runtime_error("sample_logits: temperature must be >= 0");
    if (top_k < 0) throw std::runtime_error("sample_logits: top_k must be >= 0");
    if (top_p < 0.0f) throw std::runtime_error("sample_logits: top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) throw std::runtime_error("sample_logits: vocabulary size must be > 0");

    if (indices.rows != N || indices.cols != 1 || indices.dtype != Dtype::INT32) {
        indices.resize(N, 1, Dtype::INT32);
    }
    if (N == 0) return;

    const size_t nv = static_cast<size_t>(N) * static_cast<size_t>(V);
    float* prob = static_cast<float*>(hip_alloc(3 * nv * sizeof(float)));
    float* work = prob + nv;
    int* order = reinterpret_cast<int*>(prob + 2 * nv);

    int threads = SLI_THREADS;
    while (threads > 32 && threads > V) threads >>= 1;
    const size_t shmem = static_cast<size_t>(threads) * (sizeof(double) + sizeof(int));
    hipStream_t stream = cur_stream();
    sample_logits_block_kernel<<<N, threads, shmem, stream>>>(
        static_cast<const float*>(logits.data),
        static_cast<int*>(indices.data),
        prob, work, order,
        N, V, temperature, top_k, top_p, key, counter);
    const hipError_t launch_err = hipGetLastError();
    hip_free(prob);
    BROTENSOR_HIP_CHECK(launch_err);
}

void sample_logits_into(const Tensor& logits, float temperature, int top_k, float top_p,
                        uint64_t key, Tensor& counter, Tensor& scratch, Tensor& indices) {
    using ::brotensor::Dtype;
    if (logits.dtype != Dtype::FP32) throw std::runtime_error("sample_logits_into: logits must be FP32");
    if (temperature < 0.0f) throw std::runtime_error("sample_logits_into: temperature must be >= 0");
    if (top_k < 0) throw std::runtime_error("sample_logits_into: top_k must be >= 0");
    if (top_p < 0.0f) throw std::runtime_error("sample_logits_into: top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) throw std::runtime_error("sample_logits_into: vocabulary size must be > 0");

    if (counter.dtype != Dtype::INT32 || static_cast<size_t>(counter.rows) * counter.cols < 1) {
        throw std::runtime_error("sample_logits_into: counter must be INT32 tensor with >= 1 element");
    }
    const size_t nv = static_cast<size_t>(N) * static_cast<size_t>(V);
    if (scratch.dtype != Dtype::FP32 || static_cast<size_t>(scratch.rows) * scratch.cols < 3 * nv) {
        throw std::runtime_error("sample_logits_into: scratch must be FP32 with at least 3*N*V elements");
    }
    if (indices.rows != N || indices.cols != 1 || indices.dtype != Dtype::INT32) {
        throw std::runtime_error("sample_logits_into: indices must be pre-sized (N,1) INT32 tensor");
    }
    if (N == 0) return;

    float* prob = static_cast<float*>(scratch.data);
    float* work = prob + nv;
    int* order = reinterpret_cast<int*>(prob + 2 * nv);

    int threads = SLI_THREADS;
    while (threads > 32 && threads > V) threads >>= 1;
    const size_t shmem = static_cast<size_t>(threads) * (sizeof(double) + sizeof(int));
    hipStream_t stream = cur_stream();
    sample_logits_into_kernel<<<N, threads, shmem, stream>>>(
        static_cast<const float*>(logits.data),
        static_cast<int*>(indices.data),
        prob, work, order,
        static_cast<const int*>(counter.data),
        N, V, temperature, top_k, top_p, key);
    BROTENSOR_HIP_CHECK(hipGetLastError());

    if (temperature != 0.0f) {
        advance_counter_kernel<<<1, 1, 0, stream>>>(static_cast<int*>(counter.data), N);
        BROTENSOR_HIP_CHECK(hipGetLastError());
    }
}

} // namespace brotensor::detail::hip
