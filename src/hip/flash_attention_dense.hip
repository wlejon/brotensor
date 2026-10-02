// Dense (GEMM) attention for the HIP backend: non-causal, unwindowed SDPA of
// any head_dim as two hipBLAS GEMMs around a row softmax.
//
//   1. Transpose Q (Lq, H*d) -> Qt (H, Lq, d) and K/V (Lk, Hkv*d) -> (Hkv, Lk, d).
//      Under GQA the group*Lq query rows of one KV head are contiguous in Qt,
//      so each KV head is one (group*Lq, d) x (d, Lk) GEMM.
//   2. S = Q K^T / sqrt(d), written in FP32 (hipBLAS 16-bit-in / FP32-out).
//   3. P = softmax_row(S, mask), FP32 math, P narrowed to the input dtype
//      (FP32 inputs keep P in S's storage).
//   4. Ot = P V, transposed back into O's (Lq, H*d) layout.
//
// The scores never exist at 16-bit precision: holding Q.K^T in FP16/BF16
// ahead of the max subtraction rounds every exp() argument (~1-2% per
// probability of a peaked FP16 softmax, ~8x that in BF16) — the CUDA backend's
// fa_per_head_forward makes the same choice for the same reason. S is bounded
// by score_budget() floats: past that, KV heads run a few at a time, then one
// at a time with their query rows in chunks, so a one-head VAE mid-block at
// 1024 px (Lq = Lk = 16384) streams through the score buffer instead of
// materialising 1 GB at once.
//
// A query row whose keys are all masked comes out all zero, matching the CPU
// reference and the online-softmax kernel.

#include <brotensor/tensor.h>
#include "detail/flash_internal.h"
#include "detail/hip_check.h"
#include "detail/hipblas_handle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifndef BROTENSOR_HIPBLAS_CHECK
#define BROTENSOR_HIPBLAS_CHECK(expr)                                                 \
    do {                                                                               \
        hipblasStatus_t _bga_err = (expr);                                             \
        if (_bga_err != HIPBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("hipBLAS error: ") + #expr);          \
        }                                                                              \
    } while (0)
#endif

namespace brotensor::detail::hip {

namespace {

using fa::FaScalar;
using fa::wave32_max;
using fa::wave32_sum;

// FP32 scores per pass: 1/16 of the device's memory, between 128 MB and 2 GB.
// Fewer passes mean fewer kernels per call, which matters under graph capture
// (TripoSplat's 8192-token flow step is 16 heads of 256 MB scores apiece; at
// 128 MB it took 32 passes, 100 nodes per attention call, and the captured
// step overran the AQL queue budget in graph.hip).
size_t score_budget(int dev) {
    constexpr int kMaxDevices = 16;
    static size_t cached[kMaxDevices] = {};
    const int slot = (dev >= 0 && dev < kMaxDevices) ? dev : 0;
    if (cached[slot] == 0) {
        size_t free_b = 0, total_b = 0;
        size_t bytes = size_t(128) << 20;
        if (hipMemGetInfo(&free_b, &total_b) == hipSuccess && total_b > 0) {
            bytes = std::min(std::max(total_b / 16, size_t(128) << 20), size_t(2) << 30);
        }
        (void)hipGetLastError();
        cached[slot] = bytes / sizeof(float);
    }
    return cached[slot];
}
constexpr int kSoftmaxBlock = 256;

template <typename T>
__global__ void transpose_lhd_to_hld_kernel(const T* __restrict__ In,
                                            T* __restrict__ Out,
                                            int L, int H, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    const int h = blockIdx.z;
    if (c < d && l < L && h < H) {
        Out[(static_cast<size_t>(h) * L + l) * d + c] = In[(static_cast<size_t>(l) * H + h) * d + c];
    }
}

template <typename T>
__global__ void transpose_hld_to_lhd_kernel(const T* __restrict__ In,
                                            T* __restrict__ Out,
                                            int L, int H, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    const int h = blockIdx.z;
    if (c < d && l < L && h < H) {
        Out[(static_cast<size_t>(l) * H + h) * d + c] = In[(static_cast<size_t>(h) * L + l) * d + c];
    }
}

// Block-wide reduction over kSoftmaxBlock threads; every thread gets the result.
__device__ __forceinline__ float block_max(float v, float* red) {
    v = wave32_max(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : -1e30f;
    v = wave32_max(v);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = wave32_sum(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    v = wave32_sum(v);
    __syncthreads();
    return v;
}

// P[row, :] = softmax(S[row, :Lk]) over the keys with mask[k] > 0.5 (all keys
// when mask is null); masked keys get exact zeros. One row per block. P may
// alias S (FP32): every element is read and then written by the same thread.
template <typename T>
__global__ void dense_softmax_rows_kernel(const float* S, T* P, int Lk,
                                          const float* __restrict__ mask) {
    __shared__ float red[32];
    const float* srow = S + static_cast<size_t>(blockIdx.x) * Lk;
    T* prow = P + static_cast<size_t>(blockIdx.x) * Lk;

    float mx = -1e30f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) mx = fmaxf(mx, srow[k]);
    }
    mx = block_max(mx, red);
    const bool empty = mx <= -1e29f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) sum += __expf(srow[k] - mx);
    }
    sum = block_sum(sum, red);
    const float inv = (!empty && sum > 0.0f) ? 1.0f / sum : 0.0f;

    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        const bool valid = !mask || mask[k] > 0.5f;
        prow[k] = FaScalar<T>::from_f32(valid ? __expf(srow[k] - mx) * inv : 0.0f);
    }
}

template <typename T>
void launch_transpose_in(const void* in, void* out, int L, int H, int d, hipStream_t stream) {
    dim3 blk(std::min(d, 256));
    dim3 grid((d + blk.x - 1) / blk.x, L, H);
    transpose_lhd_to_hld_kernel<T><<<grid, blk, 0, stream>>>(
        static_cast<const T*>(in), static_cast<T*>(out), L, H, d);
}

template <typename T>
void launch_transpose_out(const void* in, void* out, int L, int H, int d, hipStream_t stream) {
    dim3 blk(std::min(d, 256));
    dim3 grid((d + blk.x - 1) / blk.x, L, H);
    transpose_hld_to_lhd_kernel<T><<<grid, blk, 0, stream>>>(
        static_cast<const T*>(in), static_cast<T*>(out), L, H, d);
}

template <typename T>
void dense_attention_typed(const ::brotensor::Tensor& Q,
                           const ::brotensor::Tensor& K,
                           const ::brotensor::Tensor& V,
                           const float* d_mask,
                           int H, int Hkv, int d,
                           ::brotensor::Tensor& O,
                           hipDataType hip_dt) {
    using ::brotensor::Dtype;
    using ::brotensor::Tensor;
    const Dtype dt = Q.dtype;
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int group = H / Hkv;
    const int m = group * Lq;  // query rows per KV head
    const size_t esz = sizeof(T);
    constexpr bool kF32 = std::is_same_v<T, float>;

    // Whole problem in one batched pass when the scores fit the budget; else
    // as many whole KV heads per pass as fit; else one head at a time, query
    // rows in chunks.
    const size_t budget = score_budget(Q.device.index);
    const size_t per_head = static_cast<size_t>(m) * Lk;
    const bool batched = static_cast<size_t>(Hkv) * per_head <= budget;
    const int heads_per_pass =
        batched ? Hkv : static_cast<int>(std::min<size_t>(Hkv, std::max<size_t>(1, budget / per_head)));
    int rows = m;
    if (per_head > budget) {
        rows = static_cast<int>(std::min<size_t>(m, std::max<size_t>(1, budget / Lk)));
        if (rows >= 128) rows = rows / 128 * 128;
    }
    const int s_rows = per_head > budget ? rows : heads_per_pass * m;

    struct Scratch {
        Tensor Qt, Kt, Vt, S, P, Ot;
    };
    static thread_local Scratch scratch;
    if (scratch.Qt.device != Q.device) {
        scratch.Qt = Tensor::empty_on(Q.device, 1, 1, dt);
        scratch.Kt = Tensor::empty_on(Q.device, 1, 1, dt);
        scratch.Vt = Tensor::empty_on(Q.device, 1, 1, dt);
        scratch.S  = Tensor::empty_on(Q.device, 1, 1, Dtype::FP32);
        scratch.P  = Tensor::empty_on(Q.device, 1, 1, dt);
        scratch.Ot = Tensor::empty_on(Q.device, 1, 1, dt);
    }
    scratch.Qt.resize(H * Lq, d, dt);
    scratch.Kt.resize(Hkv * Lk, d, dt);
    scratch.Vt.resize(Hkv * Lk, d, dt);
    scratch.S.resize(s_rows, Lk, Dtype::FP32);
    if (!kF32) scratch.P.resize(s_rows, Lk, dt);
    scratch.Ot.resize(H * Lq, d, dt);
    char* qt = static_cast<char*>(scratch.Qt.data);
    char* kt = static_cast<char*>(scratch.Kt.data);
    char* vt = static_cast<char*>(scratch.Vt.data);
    char* ot = static_cast<char*>(scratch.Ot.data);
    float* s = static_cast<float*>(scratch.S.data);
    T* p = kF32 ? reinterpret_cast<T*>(s) : static_cast<T*>(scratch.P.data);

    hipStream_t stream = fa::cur_stream();
    hipblasHandle_t handle = hipblas_handle(Q.device.index);

    launch_transpose_in<T>(Q.data, qt, Lq, H, d, stream);
    launch_transpose_in<T>(K.data, kt, Lk, Hkv, d, stream);
    launch_transpose_in<T>(V.data, vt, Lk, Hkv, d, stream);

    const float alpha_s = 1.0f / std::sqrt(static_cast<float>(d));
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const long long strideKV = static_cast<long long>(Lk) * d;

    // One pass: `batch` KV heads from `g0`, query rows [r0, r0 + n) of each.
    auto pass = [&](int g0, int batch, int r0, int n) {
        const size_t q_off = (static_cast<size_t>(g0) * m + r0) * d;
        const size_t kv_off = static_cast<size_t>(g0) * Lk * d;
        // S^T (Lk x n, column-major) = K_g (Lk x d) . Q_rows^T (d x n)
        BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
            Lk, n, d,
            &alpha_s,
            kt + kv_off * esz, hip_dt, d, strideKV,
            qt + q_off * esz, hip_dt, d, static_cast<long long>(m) * d,
            &beta_zero,
            s, HIP_R_32F, Lk, static_cast<long long>(n) * Lk,
            batch,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));

        dense_softmax_rows_kernel<T><<<batch * n, kSoftmaxBlock, 0, stream>>>(s, p, Lk, d_mask);

        // Ot_rows^T (d x n) = V_g^T (d x Lk) . P^T (Lk x n)
        BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            d, n, Lk,
            &alpha_one,
            vt + kv_off * esz, hip_dt, d, strideKV,
            p, hip_dt, Lk, static_cast<long long>(n) * Lk,
            &beta_zero,
            ot + q_off * esz, hip_dt, d, static_cast<long long>(m) * d,
            batch,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    };

    if (per_head <= budget) {
        for (int g0 = 0; g0 < Hkv; g0 += heads_per_pass) pass(g0, std::min(heads_per_pass, Hkv - g0), 0, m);
    } else {
        for (int g = 0; g < Hkv; ++g) {
            for (int r0 = 0; r0 < m; r0 += rows) pass(g, 1, r0, std::min(rows, m - r0));
        }
    }

    launch_transpose_out<T>(ot, O.data, Lq, H, d, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace

void hip_dense_attention(const ::brotensor::Tensor& Q,
                         const ::brotensor::Tensor& K,
                         const ::brotensor::Tensor& V,
                         const float* d_mask,
                         int num_heads, int n_kv, int head_dim,
                         ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (num_heads <= 0 || n_kv <= 0 || num_heads % n_kv != 0 || head_dim <= 0) {
        throw std::runtime_error("hip_dense_attention: bad head configuration");
    }
    if (Q.rows == 0 || K.rows == 0) return;
    switch (Q.dtype) {
        case Dtype::FP16:
            dense_attention_typed<__half>(Q, K, V, d_mask, num_heads, n_kv, head_dim, O, HIP_R_16F);
            break;
        case Dtype::BF16:
            dense_attention_typed<hip_bfloat16>(Q, K, V, d_mask, num_heads, n_kv, head_dim, O, HIP_R_16BF);
            break;
        case Dtype::FP32:
            dense_attention_typed<float>(Q, K, V, d_mask, num_heads, n_kv, head_dim, O, HIP_R_32F);
            break;
        default:
            throw std::runtime_error("hip_dense_attention: Q, K, V must be FP16, BF16, or FP32");
    }
}

} // namespace brotensor::detail::hip
