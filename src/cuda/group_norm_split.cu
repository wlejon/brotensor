// FP32 group_norm_forward for few, large tiles.
//
// group_norm.cu's kernel runs one block per (sample, group) tile, which is
// right for the usual many-group image norms and starves the GPU when there
// are only a handful of tiles of millions of elements each (GroupNorm(1, C)
// over a long 1D signal: two tiles leave all but two SMs idle). Here each tile
// is cut into S chunks and three passes run over (S, tiles) grids:
//
//   1. per-chunk sums
//   2. per-chunk sums of squared deviations from the mean (combined from 1)
//   3. y = (x - mean) * rstd * gamma[c] + beta[c]
//
// The same two-pass statistics as the one-block kernel (never E[x^2] -
// E[x]^2); chunk partials combine in a fixed order in double, so the result is
// deterministic.

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace brotensor::detail::cuda {

void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

namespace {

constexpr int kBlock = 256;
constexpr int kTargetBlocks = 1024;   // a few waves on any current GPU
constexpr long long kMinChunk = 8192;

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0) {
        r = lane < (blockDim.x >> 5) ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    }
    return r;   // valid in thread 0
}

// Mean of tile t from its S chunk sums (thread 0 computes, all read).
__device__ float tile_mean(const float* part, int S, long long tile, float* sh) {
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < S; ++i) s += part[i];
        sh[0] = static_cast<float>(s / static_cast<double>(tile));
    }
    __syncthreads();
    return sh[0];
}

__global__ void gn_partial_sum(const float* __restrict__ X, float* __restrict__ part,
                               long long tile, long long chunk, int S) {
    __shared__ float red[32];
    const int s = blockIdx.x, t = blockIdx.y;
    const float* x = X + t * tile;
    const long long lo = s * chunk, hi = min(tile, lo + chunk);
    float acc = 0.0f;
    for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) acc += x[i];
    const float r = block_sum(acc, red);
    if (threadIdx.x == 0) part[(long long)t * S + s] = r;
}

__global__ void gn_partial_sqdev(const float* __restrict__ X, const float* __restrict__ psum,
                                 float* __restrict__ psq, long long tile, long long chunk, int S) {
    __shared__ float red[32];
    __shared__ float sh[1];
    const int s = blockIdx.x, t = blockIdx.y;
    const float mean = tile_mean(psum + (long long)t * S, S, tile, sh);
    const float* x = X + t * tile;
    const long long lo = s * chunk, hi = min(tile, lo + chunk);
    float acc = 0.0f;
    for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float d = x[i] - mean;
        acc += d * d;
    }
    const float r = block_sum(acc, red);
    if (threadIdx.x == 0) psq[(long long)t * S + s] = r;
}

__global__ void gn_apply(const float* __restrict__ X, const float* __restrict__ gamma,
                         const float* __restrict__ beta, const float* __restrict__ psum,
                         const float* __restrict__ psq, float* __restrict__ Y,
                         long long tile, long long chunk, int S, int groups, int cpg,
                         int spatial, float eps) {
    __shared__ float sh[2];
    const int s = blockIdx.x, t = blockIdx.y;
    if (threadIdx.x == 0) {
        double a = 0.0, b = 0.0;
        for (int i = 0; i < S; ++i) {
            a += psum[(long long)t * S + i];
            b += psq[(long long)t * S + i];
        }
        sh[0] = static_cast<float>(a / static_cast<double>(tile));
        sh[1] = rsqrtf(static_cast<float>(b / static_cast<double>(tile)) + eps);
    }
    __syncthreads();
    const float mean = sh[0], rstd = sh[1];
    const int chan_base = (t % groups) * cpg;
    const float* x = X + t * tile;
    float* y = Y + t * tile;
    const long long lo = s * chunk, hi = min(tile, lo + chunk);
    for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const int c = chan_base + static_cast<int>(i / spatial);
        y[i] = (x[i] - mean) * rstd * gamma[c] + beta[c];
    }
}

}  // namespace

// Returns false (nothing issued) when the one-block-per-tile kernel already
// fills the GPU or the tiles are small.
bool group_norm_forward_fp32_split(const float* X, const float* gamma, const float* beta,
                                   float* Y, int N, int C, int spatial, int groups,
                                   float eps, cudaStream_t stream) {
    const int tiles = N * groups;
    const int cpg = C / groups;
    const long long tile = static_cast<long long>(cpg) * spatial;
    if (tiles >= kTargetBlocks / 4 || tile < 2 * kMinChunk) return false;
    long long S = (kTargetBlocks + tiles - 1) / tiles;
    long long chunk = (tile + S - 1) / S;
    if (chunk < kMinChunk) chunk = kMinChunk;
    S = (tile + chunk - 1) / chunk;
    if (tiles > 65535) return false;

    float* psum = static_cast<float*>(cuda_alloc(static_cast<size_t>(tiles) * S * 2 * sizeof(float)));
    float* psq = psum + static_cast<size_t>(tiles) * S;
    const dim3 grid(static_cast<unsigned>(S), static_cast<unsigned>(tiles));
    gn_partial_sum<<<grid, kBlock, 0, stream>>>(X, psum, tile, chunk, static_cast<int>(S));
    gn_partial_sqdev<<<grid, kBlock, 0, stream>>>(X, psum, psq, tile, chunk, static_cast<int>(S));
    gn_apply<<<grid, kBlock, 0, stream>>>(X, gamma, beta, psum, psq, Y, tile, chunk,
                                          static_cast<int>(S), groups, cpg, spatial, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    cuda_free(psum);
    return true;
}

}  // namespace brotensor::detail::cuda
