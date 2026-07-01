#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor {
namespace detail::cuda {

namespace {

// Both nchw_to_sequence and sequence_to_nchw are, per batch item n, a plain
// 2D matrix transpose (nchw_to_sequence: (C, HW) -> (HW, C); sequence_to_nchw:
// (HW, C) -> (C, HW)). The old element-at-a-time kernels indexed input and
// output independently, which coalesces one side but strides the other by C
// or HW elements per consecutive thread — every warp issued up to 32 separate
// memory transactions on that side. This is the textbook fix: a 32x32
// shared-memory tile. Threads read a contiguous tile from the input with
// coalesced reads (consecutive threadIdx.x -> consecutive input addresses),
// then write it out transposed with coalesced writes (consecutive
// threadIdx.x -> consecutive output addresses); the tile's +1 column padding
// avoids shared-memory bank conflicts on the transposed read. blockIdx.z
// batches over N independent (rows, cols) matrices.
constexpr int TR_TILE = 32;

// `rows_on_x` picks which logical tile axis (row-tiles vs col-tiles) rides
// gridDim.x vs gridDim.y: CUDA caps gridDim.y/z at 65535 but allows gridDim.x
// up to ~2^31-1, so the launcher below always puts whichever axis needs more
// tiles onto x. rows/cols themselves are the actual memory row-stride/extent
// (unchanged) — only the blockIdx<->tile mapping is swapped.
template <typename T>
__global__ void transpose_tiled_kernel(const T* __restrict__ in,
                                       T* __restrict__ out,
                                       int rows, int cols, bool rows_on_x) {
    __shared__ T tile[TR_TILE][TR_TILE + 1];

    const size_t batch_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const T* in_b  = in  + static_cast<size_t>(blockIdx.z) * batch_elems;
    T*       out_b = out + static_cast<size_t>(blockIdx.z) * batch_elems;

    const int row_tile = rows_on_x ? blockIdx.x : blockIdx.y;
    const int col_tile = rows_on_x ? blockIdx.y : blockIdx.x;

    int x = col_tile * TR_TILE + threadIdx.x;   // input column, 0..cols
    int y = row_tile * TR_TILE + threadIdx.y;   // input row,    0..rows
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x] = in_b[static_cast<size_t>(y) * cols + x];
    }
    __syncthreads();

    x = row_tile * TR_TILE + threadIdx.x;       // output column = input row, 0..rows
    y = col_tile * TR_TILE + threadIdx.y;       // output row    = input col, 0..cols
    if (y < cols && x < rows) {
        out_b[static_cast<size_t>(y) * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

inline dim3 transpose_grid(int rows, int cols, int batch, bool rows_on_x) {
    const int row_tiles = (rows + TR_TILE - 1) / TR_TILE;
    const int col_tiles = (cols + TR_TILE - 1) / TR_TILE;
    return rows_on_x ? dim3(row_tiles, col_tiles, batch)
                      : dim3(col_tiles, row_tiles, batch);
}

void check_dims(const char* op, int N, int C, int H, int W) {
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error(std::string(op) + ": negative dimension");
    }
}

} // namespace

void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      ::brotensor::Tensor& Y) {
    check_dims("nchw_to_sequence", N, C, H, W);
    const int HW = H * W;
    const int rows = N * HW;
    if (Y.rows != rows || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(rows, C, X.dtype);
    }
    const int total = rows * C;
    if (total == 0) return;
    // Per batch item, X_n is (C, HW) row-major and Y_n is its (HW, C)
    // transpose. Put whichever axis needs more tiles on gridDim.x (its
    // ~2^31-1 headroom vs. gridDim.y/z's 65535 cap).
    const bool rows_on_x = C >= HW;
    const dim3 block(TR_TILE, TR_TILE);
    const dim3 grid = transpose_grid(/*rows=*/C, /*cols=*/HW, /*batch=*/N, rows_on_x);
    if (X.dtype == Dtype::FP16) {
        transpose_tiled_kernel<__half><<<grid, block, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            C, HW, rows_on_x);
    } else if (X.dtype == Dtype::BF16) {
        transpose_tiled_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            C, HW, rows_on_x);
    } else {
        transpose_tiled_kernel<float><<<grid, block, 0, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            C, HW, rows_on_x);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      ::brotensor::Tensor& Y) {
    check_dims("sequence_to_nchw", N, C, H, W);
    const int HW = H * W;
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    // Per batch item, X_n is (HW, C) row-major and Y_n is its (C, HW)
    // transpose. Put whichever axis needs more tiles on gridDim.x (its
    // ~2^31-1 headroom vs. gridDim.y/z's 65535 cap).
    const bool rows_on_x = HW >= C;
    const dim3 block(TR_TILE, TR_TILE);
    const dim3 grid = transpose_grid(/*rows=*/HW, /*cols=*/C, /*batch=*/N, rows_on_x);
    if (X.dtype == Dtype::FP16) {
        transpose_tiled_kernel<__half><<<grid, block, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            HW, C, rows_on_x);
    } else if (X.dtype == Dtype::BF16) {
        transpose_tiled_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            HW, C, rows_on_x);
    } else {
        transpose_tiled_kernel<float><<<grid, block, 0, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            HW, C, rows_on_x);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
