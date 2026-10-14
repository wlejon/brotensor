// ─── block_fft.cuh — one power-of-two complex FFT per thread block ─────────
//
// An in-place radix-2 Cooley-Tukey transform over n complex values held in
// shared memory, all threads of the block cooperating. Callers load their
// signal with block_fft_store (which scatters to bit-reversed positions),
// call block_fft, and read the result in natural order.
//
//   sign = -1: forward,  X[k] = sum_i x[i] exp(-2 pi i k i / n)
//   sign = +1: inverse (unscaled), x[i] = sum_k X[k] exp(+2 pi i k i / n)
//
// Twiddles come from sincospif on the exact float ratio 2j/len (len a power
// of two), so each one is within an ulp; the transform's error is the usual
// FP32 radix-2 O(eps log n).
#pragma once

#include <cuda_runtime.h>

namespace brotensor::detail::cuda::blockfft {

__host__ __device__ inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

__host__ __device__ inline int log2_pow2(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

// Position i of an n = 2^logn point transform lands at bitrev(i).
__device__ inline int bitrev(int i, int logn) {
    return static_cast<int>(__brev(static_cast<unsigned>(i)) >> (32 - logn));
}

__device__ inline void block_fft_store(float2* s, int i, int logn, float2 v) {
    s[logn == 0 ? 0 : bitrev(i, logn)] = v;
}

// s holds the bit-reversed input; on return it holds the transform in natural
// order. Ends with a __syncthreads, so s may be read straight after.
__device__ inline void block_fft(float2* s, int n, int sign) {
    __syncthreads();
    const int half_n = n >> 1;
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const float step = 2.0f * static_cast<float>(sign) / static_cast<float>(len);
        for (int t = threadIdx.x; t < half_n; t += blockDim.x) {
            const int j = t & (half - 1);
            const int i0 = (t - j) * 2 + j;
            const int i1 = i0 + half;
            float sn, cs;
            sincospif(step * static_cast<float>(j), &sn, &cs);
            const float2 a = s[i0];
            const float2 b = s[i1];
            const float2 bw = make_float2(b.x * cs - b.y * sn, b.x * sn + b.y * cs);
            s[i0] = make_float2(a.x + bw.x, a.y + bw.y);
            s[i1] = make_float2(a.x - bw.x, a.y - bw.y);
        }
        __syncthreads();
    }
}

}  // namespace brotensor::detail::cuda::blockfft
