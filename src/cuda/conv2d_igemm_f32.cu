// FP32 implicit-GEMM conv2d forward, and conv_transpose2d built on it.
//
// conv2d.cu's FP32 path is a direct convolution, one thread per output
// value; FP16/BF16 have tensor-core implicit GEMMs, FP32 had nothing. This is
// the FP32 counterpart on plain FMAs, for groups == 1 and any kernel, stride,
// padding and dilation:
//
//   C(M, Cout) = A(M, K) . W(Cout, K)^T + bias,  M = N*Ho*Wo,  K = Cin*kH*kW
//   A[m, k] = X[n, ic, oh*sh - ph + kh*dh, ow*sw - pw + kw*dw]  (0 outside)
//
// A is never materialised: each 16-deep K step gathers a 16 x 128 tile of it
// straight from X into shared memory, beside a 16 x 64 tile of W. 256 threads
// each hold an 8 x 4 accumulator tile; the next K step's tiles are loaded into
// registers while the current one is multiplied (double-buffered shared).
// The store scatters the (pixel, channel) tile back to NCHW.
//
// conv_transpose2d (FP32, groups == 1) runs as that GEMM with a 1x1 kernel —
// cols(N, Cout*kH*kW, H, W) = Wt^T . X — and a col2im gather that sums, for
// every output pixel, the taps that land on it, plus the bias. Every product
// is a real one (the gather formulation would multiply stride^2 zeros).

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace brotensor::detail::cuda {

void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

namespace {

constexpr int BM = 128, BN = 64, BK = 16, NT = 256;
constexpr int BNP = BN + 4;   // W tile row stride

struct ConvGeom {
    int N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, Ho, Wo;
    int M, K, KHW, HWo;
    int vec_store;   // HWo % 4 == 0 and Y 16-byte aligned: float4 stores
};

__global__ void __launch_bounds__(NT, 2)
conv2d_igemm_f32_kernel(const float* __restrict__ X, const float* __restrict__ Wt,
                        const float* __restrict__ bias, float* __restrict__ Y, ConvGeom g) {
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BNP];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    // ── A loader: one pixel (column of the A tile), 8 of the 16 K rows ──
    const int a_m = tid & (BM - 1);
    const int a_k = tid >> 7;                 // 0 or 1; rows a_k + 2*i
    const int m_glob = m0 + a_m;
    const bool m_ok = m_glob < g.M;
    int ih0 = 0, iw0 = 0;
    long long xbase = 0;
    if (m_ok) {
        const int n_img = m_glob / g.HWo;
        const int p = m_glob - n_img * g.HWo;
        const int oh = p / g.Wo;
        const int ow = p - oh * g.Wo;
        ih0 = oh * g.sh - g.ph;
        iw0 = ow * g.sw - g.pw;
        xbase = (long long)n_img * g.Cin * g.H * g.W;
    }
    // ── W loader: one output channel, 4 consecutive K ──
    const int b_n = tid >> 2;
    const int b_k = (tid & 3) * 4;
    const bool n_ok = n0 + b_n < g.Cout;
    const float* wrow = Wt + (long long)(n0 + b_n) * g.K;

    float ra[8], rb[4];
    auto load_tiles = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + a_k + 2 * i;
            float v = 0.f;
            if (m_ok && k < g.K) {
                const int ic = k / g.KHW;
                const int r = k - ic * g.KHW;
                const int kh = r / g.kW;
                const int kw = r - kh * g.kW;
                const int ih = ih0 + kh * g.dh;
                const int iw = iw0 + kw * g.dw;
                if (ih >= 0 && ih < g.H && iw >= 0 && iw < g.W)
                    v = X[xbase + ((long long)ic * g.H + ih) * g.W + iw];
            }
            ra[i] = v;
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int k = k0 + b_k + e;
            rb[e] = (n_ok && k < g.K) ? wrow[k] : 0.f;
        }
    };
    auto store_tiles = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 8; ++i) As[buf][a_k + 2 * i][a_m] = ra[i];
#pragma unroll
        for (int e = 0; e < 4; ++e) Bs[buf][b_k + e][b_n] = rb[e];
    };

    // ── compute layout: rows tm*4..+3 and 64+tm*4..+3, cols tn*4..+3 ──
    const int tm = tid & 15;
    const int tn = tid >> 4;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    load_tiles(0);
    store_tiles(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < g.K; k0 += BK) {
        const bool more = k0 + BK < g.K;
        if (more) load_tiles(k0 + BK);
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[buf][k][tm * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[buf][k][64 + tm * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Bs[buf][k][tn * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        if (more) {
            store_tiles(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    // ── store: Y[n][oc][p] ──
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int oc = n0 + tn * 4 + j;
        if (oc >= g.Cout) continue;
        const float bv = bias ? bias[oc] : 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int mb = m0 + h * 64 + tm * 4;
            if (mb >= g.M) continue;
            const int n_img = mb / g.HWo;
            const int p = mb - n_img * g.HWo;
            float* yrow = Y + ((long long)n_img * g.Cout + oc) * g.HWo;
            if (g.vec_store && mb + 3 < g.M) {   // 4 pixels in one image, aligned
                *reinterpret_cast<float4*>(yrow + p) =
                    make_float4(acc[h * 4][j] + bv, acc[h * 4 + 1][j] + bv,
                                acc[h * 4 + 2][j] + bv, acc[h * 4 + 3][j] + bv);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int m = mb + i;
                    if (m >= g.M) break;
                    const int ni = m / g.HWo;
                    const int pi = m - ni * g.HWo;
                    Y[((long long)ni * g.Cout + oc) * g.HWo + pi] = acc[h * 4 + i][j] + bv;
                }
            }
        }
    }
}

// (C_in, Cout*kHW) -> (Cout*kHW, C_in)
__global__ void transpose_f32_kernel(const float* __restrict__ in, float* __restrict__ out,
                                     int rows, int cols) {
    __shared__ float tile[32][33];
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int r0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int r = r0 + i;
        if (r < rows && c < cols) tile[i][threadIdx.x] = in[(long long)r * cols + c];
    }
    __syncthreads();
    const int oc = r0 + threadIdx.x;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int orow = blockIdx.x * 32 + i;
        if (orow < cols && oc < rows) out[(long long)orow * rows + oc] = tile[threadIdx.x][i];
    }
}

// Y[n, oc, oh, ow] = bias[oc] + sum over taps (kh, kw) landing on (oh, ow) of
// cols[n, (oc*kH + kh)*kW + kw, ih, iw].
__global__ void col2im_f32_kernel(const float* __restrict__ cols, const float* __restrict__ bias,
                                  float* __restrict__ Y, int N, int Cout, int H, int W, int kH,
                                  int kW, int sh, int sw, int ph, int pw, int dh, int dw, int Ho,
                                  int Wo) {
    const long long total = (long long)N * Cout * Ho * Wo;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
         idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % Wo);
        long long t = idx / Wo;
        const int oh = static_cast<int>(t % Ho);
        t /= Ho;
        const int oc = static_cast<int>(t % Cout);
        const int n = static_cast<int>(t / Cout);
        float acc = bias ? bias[oc] : 0.f;
        const float* cbase = cols + ((long long)n * Cout + oc) * kH * kW * H * W;
        for (int kh = 0; kh < kH; ++kh) {
            const int th = oh + ph - kh * dh;
            if (th < 0 || th % sh) continue;
            const int ih = th / sh;
            if (ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int tw = ow + pw - kw * dw;
                if (tw < 0 || tw % sw) continue;
                const int iw = tw / sw;
                if (iw >= W) continue;
                acc += cbase[((long long)(kh * kW + kw) * H + ih) * W + iw];
            }
        }
        Y[idx] = acc;
    }
}

int grid1d(long long n, int block) {
    long long b = (n + block - 1) / block;
    if (b > 65535LL * 32) b = 65535LL * 32;
    return static_cast<int>(b < 1 ? 1 : b);
}

}  // namespace

bool conv2d_igemm_f32(const float* X, const float* Wt, const float* bias, float* Y, int N,
                      int Cin, int H, int W, int Cout, int kH, int kW, int sh, int sw, int ph,
                      int pw, int dh, int dw, int Ho, int Wo, cudaStream_t stream) {
    ConvGeom g{N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, Ho, Wo, 0, 0, 0, 0, 0};
    const long long M = (long long)N * Ho * Wo;
    const long long K = (long long)Cin * kH * kW;
    if (M <= 0 || K <= 0 || M > 0x7fffffffLL || K > 0x7fffffffLL) return false;
    if ((long long)N * Cin * H * W > 0x7fffffffLL * 4LL) return false;
    g.M = static_cast<int>(M);
    g.K = static_cast<int>(K);
    g.KHW = kH * kW;
    g.HWo = Ho * Wo;
    g.vec_store = (g.HWo % 4 == 0) && (reinterpret_cast<std::uintptr_t>(Y) % 16 == 0);
    const long long gx = (M + BM - 1) / BM, gy = (Cout + BN - 1) / BN;
    if (gx > 0x7fffffffLL || gy > 65535) return false;
    conv2d_igemm_f32_kernel<<<dim3(static_cast<unsigned>(gx), static_cast<unsigned>(gy)), NT, 0,
                              stream>>>(X, Wt, bias, Y, g);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

bool conv_transpose2d_f32_gemm(const float* X, const float* Wt, const float* bias, float* Y,
                               int N, int Cin, int H, int W, int Cout, int kH, int kW, int sh,
                               int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                               cudaStream_t stream) {
    const long long R = (long long)Cout * kH * kW;   // GEMM output channels
    if (R > 0x7fffffffLL || (long long)N * R * H * W > 0x7fffffffLL * 4LL) return false;
    float* wT = static_cast<float*>(cuda_alloc(static_cast<size_t>(R) * Cin * sizeof(float)));
    float* cols = static_cast<float*>(cuda_alloc(static_cast<size_t>(N) * R * H * W * sizeof(float)));
    transpose_f32_kernel<<<dim3(static_cast<unsigned>((R + 31) / 32), static_cast<unsigned>((Cin + 31) / 32)),
                           dim3(32, 8), 0, stream>>>(Wt, wT, Cin, static_cast<int>(R));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    const bool ok = conv2d_igemm_f32(X, wT, nullptr, cols, N, Cin, H, W, static_cast<int>(R), 1, 1,
                                     1, 1, 0, 0, 1, 1, H, W, stream);
    if (ok) {
        const long long total = (long long)N * Cout * Ho * Wo;
        col2im_f32_kernel<<<grid1d(total, 256), 256, 0, stream>>>(
            cols, bias, Y, N, Cout, H, W, kH, kW, sh, sw, ph, pw, dh, dw, Ho, Wo);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    cuda_free(cols);
    cuda_free(wT);
    return ok;
}

}  // namespace brotensor::detail::cuda
