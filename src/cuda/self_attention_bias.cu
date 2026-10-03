// ─── CUDA self-attention with additive pre-softmax bias ────────────────────
//
// Multi-head self-attention that adds an optional per-head (L, L) bias to the
// attention logits before softmax — the general primitive behind T5's
// relative-position bias and ALiBi-style biases.
//
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Scores are materialised (L, L) per head — intended for encoder-length
// sequences (T5 ≤ 512), not long-context decoding. Dispatched on X.dtype
// (FP32 / FP16 / BF16): the projection inputs/outputs are typed, every
// intermediate (Q/K/V/scores/softmax) is FP32 scratch, math is FP32.
// attn_bias is FP32 on every backend.

#include "self_attention_internal.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

// S[(hh*L+i), j] = scale * (Q_h[i] . K_h[j]) + bias[(hh*L+i), j].
// Qh, Kh: (H*L, dh) FP32; bias: (H*L, L) FP32 or null; S: (H*L, L) FP32.
__global__ void sab_scores_kernel(const float* __restrict__ Qh,
                                  const float* __restrict__ Kh,
                                  const float* __restrict__ bias,
                                  float* __restrict__ S,
                                  int L, int dh, float scale) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= L) return;
    const size_t qrow = (static_cast<size_t>(hh) * L + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * L + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    const size_t srow = (static_cast<size_t>(hh) * L + i) * L;
    s *= scale;
    if (bias) s += bias[srow + j];
    S[srow + j] = s;
}

// Per-row masked softmax over (H*L, L). One block per (head, query row).
// mask is length L (key validity); a padded query row produces a zero row.
__global__ void sab_softmax_kernel(const float* __restrict__ scores,
                                   float* __restrict__ Attn,
                                   const float* __restrict__ mask,
                                   int L) {
    __shared__ float sdata[SAB_SM_BLOCK];
    const int row = blockIdx.x;            // hh*L + i
    const int i_within = row % L;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * L;
    float* arow = Attn + static_cast<size_t>(row) * L;

    if (mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < L; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < L; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        const float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < L; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
        const float e = expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = tid; j < L; j += blockDim.x) arow[j] *= inv;
}

// Yconcat[i, hh*dh+k] = sum_j Attn[(hh*L+i), j] * Vh[(hh*L+j), k].
__global__ void sab_apply_v_kernel(const float* __restrict__ Attn,
                                   const float* __restrict__ Vh,
                                   float* __restrict__ Yconcat,
                                   int L, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * L + i) * L;
    float acc = 0.0f;
    for (int j = 0; j < L; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * L + j) * dh;
        acc += Attn[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// ── INT8 weight-only (W8A16) projection / output kernels ───────────────────
//
// Same dot products as sab_proj_kernel / sab_output_kernel, but the weight is
// an INT8 (D, Din) matrix paired with an FP32 per-output-row dequant scale.
// The row `wrow` accumulates against int8 weights, then the whole sum is
// multiplied by scales[wrow] — equivalent to dequantising the row first,
// since one scale covers the entire row.
template <typename T>
__global__ void sab_proj_kernel_int8(const T* __restrict__ In,
                                     const int8_t* __restrict__ W,
                                     const float* __restrict__ scales,
                                     float* __restrict__ Out,
                                     int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int wrow = hh * dh + j;
    const T* xr = In + static_cast<size_t>(i) * Din;
    const int8_t* wr = W + static_cast<size_t>(wrow) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) acc += sab_ld(xr[k]) * static_cast<float>(wr[k]);
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc * scales[wrow];
}

template <typename T>
__global__ void sab_output_kernel_int8(const float* __restrict__ Y,
                                       const int8_t* __restrict__ Wo,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ mask,
                                       T* __restrict__ O, int L, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= L || c >= D) return;
    if (mask && mask[i] < 0.5f) { sab_st(O[static_cast<size_t>(i) * D + c], 0.0f); return; }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const int8_t* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * static_cast<float>(wr[k]);
    sab_st(O[static_cast<size_t>(i) * D + c], acc * scales[c]);
}

// INT8-weight variant of run_sab: the four projection matmuls consume INT8
// weights + FP32 per-row scales; the attention core (scores / softmax / PV)
// is byte-identical to the FP16 path.
template <typename T>
void run_sab_int8(const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& sq,
                  const ::brotensor::Tensor& Wk, const ::brotensor::Tensor& sk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& sv,
                  const ::brotensor::Tensor& Wo, const ::brotensor::Tensor& so,
                  const float* d_mask, const float* bias_p,
                  int num_heads, float scale, ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int L  = X.rows;
    const int D  = X.cols;
    const int H  = num_heads;
    const int dh = D / H;

    Tensor Qh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::CUDA, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::CUDA, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::CUDA, L, D, Dtype::FP32);

    const T* X_p  = static_cast<const T*>(X.data);
    const int8_t* Wq_p = static_cast<const int8_t*>(Wq.data);
    const int8_t* Wk_p = static_cast<const int8_t*>(Wk.data);
    const int8_t* Wv_p = static_cast<const int8_t*>(Wv.data);
    const int8_t* Wo_p = static_cast<const int8_t*>(Wo.data);
    const float* sq_p = static_cast<const float*>(sq.data);
    const float* sk_p = static_cast<const float*>(sk.data);
    const float* sv_p = static_cast<const float*>(sv.data);
    const float* so_p = static_cast<const float*>(so.data);
    float* Qh_p = static_cast<float*>(Qh.data);
    float* Kh_p = static_cast<float*>(Kh.data);
    float* Vh_p = static_cast<float*>(Vh.data);
    float* S_p  = static_cast<float*>(S.data);
    float* A_p  = static_cast<float*>(A.data);
    float* Yc_p = static_cast<float*>(Yc.data);

    const dim3 block(16, 16);
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (L  + block.y - 1) / block.y, H);
        sab_proj_kernel_int8<T><<<grid, block, 0, cur_stream()>>>(X_p, Wq_p, sq_p, Qh_p, L, D, dh);
        sab_proj_kernel_int8<T><<<grid, block, 0, cur_stream()>>>(X_p, Wk_p, sk_p, Kh_p, L, D, dh);
        sab_proj_kernel_int8<T><<<grid, block, 0, cur_stream()>>>(X_p, Wv_p, sv_p, Vh_p, L, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((L + block.x - 1) / block.x,
                  (L + block.y - 1) / block.y, H);
        sab_scores_kernel<<<grid, block, 0, cur_stream()>>>(Qh_p, Kh_p, bias_p, S_p, L, dh, scale);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    sab_softmax_kernel<<<H * L, SAB_SM_BLOCK, 0, cur_stream()>>>(S_p, A_p, d_mask, L);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (L  + block.y - 1) / block.y, H);
        sab_apply_v_kernel<<<grid, block, 0, cur_stream()>>>(A_p, Vh_p, Yc_p, L, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((D + block.x - 1) / block.x,
                  (L + block.y - 1) / block.y);
        sab_output_kernel_int8<T><<<grid, block, 0, cur_stream()>>>(Yc_p, Wo_p, so_p, d_mask,
                                                   static_cast<T*>(O.data), L, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ── Fast (tensor-core) general-bias self-attention ─────────────────────────
//
// Same math as the (now-removed) naive run_sab, but every GEMM-shaped stage
// runs through sardp_gemm — WMMA tensor cores for FP16/BF16, the register-tiled
// kernel for FP32 — instead of one-thread-per-output kernels. This is the path
// T5's encoder self-attention (24 BF16 blocks, D=4096) actually hits, where the
// four (L,D)x(D,D) projections dominate; routing them onto tensor cores is the
// bulk of the speed-up. Intermediates are stored in the model dtype (WMMA still
// accumulates each dot product in FP32); the softmax runs in FP32.
//
// Structure mirrors run_sardp_fast (single GEMM per projection then split into
// head panels, batched scores / A@V over heads, deferred-sum softmax, head
// merge, output GEMM) but with a precomputed full (H*L, L) additive bias added
// in the softmax rather than the 2D-decomposed rel-pos factors. nWin == 1.

// In-place row softmax over the typed (H*Lp, Lp) scores panel, folding the qk
// scale and an optional precomputed (H*L, L) FP32 bias into the load, honouring
// an optional length-L key/query mask. One block per (head, query) row. Writes
// UNNORMALISED exp(s - max) and the row sum to `sums`; the 1/sum scaling is
// deferred to sardp_merge_heads_kernel. Padded query rows (i >= L) are left
// untouched (their Q is zero, so the merge never reads them); masked query rows
// get a zero score row and sums = 1 so the deferred divide stays finite (their
// Y is zero, hence a zero attention output — exactly the old kernel's result
// for the bias-free callers, which is every current caller).
template <typename T>
__global__ void sab_fast_softmax_kernel(T* __restrict__ S,
                                        const float* __restrict__ bias, // (H*L, L) or null
                                        const float* __restrict__ mask, // length L or null
                                        float* __restrict__ sums,
                                        int L, int Lp, float scale) {
    __shared__ float sdata[SAB_SM_BLOCK];
    const long long row = blockIdx.x;            // h*Lp + i
    const int i = static_cast<int>(row % Lp);
    if (i >= L) return;                          // alignment-pad row
    const int h = static_cast<int>(row / Lp);
    const int tid = threadIdx.x;
    T* srow = S + row * Lp;
    const float* brow = bias ? bias + (static_cast<size_t>(h) * L + i) * L : nullptr;

    if (mask && mask[i] < 0.5f) {                // masked (pad) query row
        for (int j = tid; j < Lp; j += blockDim.x) sab_st(srow[j], 0.0f);
        if (tid == 0) sums[row] = 1.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < L; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        const float s = scale * sab_ld(srow[j]) + (brow ? brow[j] : 0.0f);
        if (s > local_max) local_max = s;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < Lp; j += blockDim.x) {
        if (j >= L || (mask && mask[j] < 0.5f)) { sab_st(srow[j], 0.0f); continue; }
        const float e = expf(scale * sab_ld(srow[j]) + (brow ? brow[j] : 0.0f) - m);
        sab_st(srow[j], e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sums[row] = sdata[0] > 0.0f ? sdata[0] : 1.0f;
}

template <typename T>
void run_sab_fast(const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                  const T* bq, const T* bk, const T* bv, const T* bo,
                  const float* d_mask, const float* bias_p,
                  int num_heads, float scale, ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    constexpr Dtype dt = sardp_dtype<T>::value;
    const int L  = X.rows;
    const int D  = X.cols;
    const int H  = num_heads;
    const int dh = D / H;
    const int Lp = (L + 7) & ~7;
    const size_t panel_q = static_cast<size_t>(Lp) * dh;
    const size_t panel_s = static_cast<size_t>(Lp) * Lp;

    const T* X_p  = static_cast<const T*>(X.data);
    const T* Wq_p = static_cast<const T*>(Wq.data);
    const T* Wk_p = static_cast<const T*>(Wk.data);
    const T* Wv_p = static_cast<const T*>(Wv.data);
    const T* Wo_p = static_cast<const T*>(Wo.data);

    // Per-head Q/K panels, V transposed to (dh, Lp) so A@V is an A@B^T GEMM.
    Tensor Qh = Tensor::empty_on(Device::CUDA, H * Lp, dh, dt);
    Tensor Kh = Tensor::empty_on(Device::CUDA, H * Lp, dh, dt);
    Tensor Vt = Tensor::empty_on(Device::CUDA, H * dh, Lp, dt);
    T* Qh_p = static_cast<T*>(Qh.data);
    T* Kh_p = static_cast<T*>(Kh.data);
    T* Vt_p = static_cast<T*>(Vt.data);
    {
        Tensor Qrow = Tensor::empty_on(Device::CUDA, L, D, dt);
        Tensor Krow = Tensor::empty_on(Device::CUDA, L, D, dt);
        Tensor Vrow = Tensor::empty_on(Device::CUDA, L, D, dt);
        sardp_gemm(X_p, Wq_p, static_cast<T*>(Qrow.data), 1, L, D, D, 0, 0, 0, bq);
        sardp_gemm(X_p, Wk_p, static_cast<T*>(Krow.data), 1, L, D, D, 0, 0, 0, bk);
        sardp_gemm(X_p, Wv_p, static_cast<T*>(Vrow.data), 1, L, D, D, 0, 0, 0, bv);

        const long long total = static_cast<long long>(H) * Lp * dh;
        const int grid = sardp_flat_grid(total);
        sardp_split_heads_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Qrow.data), Qh_p, H, L, Lp, dh, total);
        sardp_split_heads_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Krow.data), Kh_p, H, L, Lp, dh, total);
        sardp_split_heads_t_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Vrow.data), Vt_p, H, L, Lp, dh, total);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Scores = Q @ K^T (batched over heads), then in-place fused softmax.
    Tensor S    = Tensor::empty_on(Device::CUDA, H * Lp, Lp, dt);
    Tensor sums = Tensor::empty_on(Device::CUDA, H * Lp, 1,  Dtype::FP32);
    T* S_p = static_cast<T*>(S.data);
    sardp_gemm(Qh_p, Kh_p, S_p, H, Lp, Lp, dh, panel_q, panel_q, panel_s, nullptr);
    sab_fast_softmax_kernel<T><<<H * Lp, SAB_SM_BLOCK, 0, cur_stream()>>>(
        S_p, bias_p, d_mask, static_cast<float*>(sums.data), L, Lp, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // A @ V (batched against the transposed V panels), merge heads with the
    // deferred 1/sum, output projection.
    Tensor Y = Tensor::empty_on(Device::CUDA, H * Lp, dh, dt);
    sardp_gemm(S_p, Vt_p, static_cast<T*>(Y.data), H, Lp, dh, Lp,
               panel_s, panel_q, panel_q, nullptr);
    Tensor Yrow = Tensor::empty_on(Device::CUDA, L, D, dt);
    {
        const long long total = static_cast<long long>(L) * D;
        sardp_merge_heads_kernel<T><<<sardp_flat_grid(total), 256, 0, cur_stream()>>>(
            static_cast<const T*>(Y.data), static_cast<const float*>(sums.data),
            static_cast<T*>(Yrow.data), H, L, Lp, dh, total);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    sardp_gemm(static_cast<const T*>(Yrow.data), Wo_p, static_cast<T*>(O.data),
               1, L, D, D, 0, 0, 0, bo);
}

} // namespace

void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const ::brotensor::Tensor* bq,
                                 const ::brotensor::Tensor* bk,
                                 const ::brotensor::Tensor* bv,
                                 const ::brotensor::Tensor* bo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("self_attention_bias_forward: X must be FP32, FP16, or BF16");
    }
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo dtype must match X");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const float* bias_p = nullptr;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be FP32");
        }
        if (attn_bias->size() != num_heads * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        bias_p = static_cast<const float*>(attn_bias->data);
    }
    auto check_proj_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string("self_attention_bias_forward: ") +
                                         name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string("self_attention_bias_forward: ") +
                                         name + " must have D entries");
        }
    };
    check_proj_bias(bq, "bq"); check_proj_bias(bk, "bk");
    check_proj_bias(bv, "bv"); check_proj_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_sab_fast<float>(X, Wq, Wk, Wv, Wo,
                            static_cast<const float*>(bp(bq)), static_cast<const float*>(bp(bk)),
                            static_cast<const float*>(bp(bv)), static_cast<const float*>(bp(bo)),
                            d_mask, bias_p, num_heads, scale, O);
        break;
    case Dtype::FP16:
        run_sab_fast<__half>(X, Wq, Wk, Wv, Wo,
                             static_cast<const __half*>(bp(bq)), static_cast<const __half*>(bp(bk)),
                             static_cast<const __half*>(bp(bv)), static_cast<const __half*>(bp(bo)),
                             d_mask, bias_p, num_heads, scale, O);
        break;
    default:  // BF16
        run_sab_fast<__nv_bfloat16>(X, Wq, Wk, Wv, Wo,
                                    static_cast<const __nv_bfloat16*>(bp(bq)), static_cast<const __nv_bfloat16*>(bp(bk)),
                                    static_cast<const __nv_bfloat16*>(bp(bv)), static_cast<const __nv_bfloat16*>(bp(bo)),
                                    d_mask, bias_p, num_heads, scale, O);
        break;
    }
}

void self_attention_bias_int8w_fp16(const ::brotensor::Tensor& X,
                                    const ::brotensor::Tensor& Wq_int8,
                                    const ::brotensor::Tensor& sq,
                                    const ::brotensor::Tensor& Wk_int8,
                                    const ::brotensor::Tensor& sk,
                                    const ::brotensor::Tensor& Wv_int8,
                                    const ::brotensor::Tensor& sv,
                                    const ::brotensor::Tensor& Wo_int8,
                                    const ::brotensor::Tensor& so,
                                    const float* d_mask,
                                    const ::brotensor::Tensor* attn_bias,
                                    int num_heads, float scale,
                                    ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("self_attention_bias_int8w_fp16: X must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be INT8");
    }
    if (sq.dtype != Dtype::FP32 || sk.dtype != Dtype::FP32 ||
        sv.dtype != Dtype::FP32 || so.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: scales must be FP32");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: num_heads must divide D");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D ||
        Wv_int8.rows != D || Wv_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be (D, D)");
    }
    if (sq.size() != D || sk.size() != D || sv.size() != D || so.size() != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: each scale tensor must have D entries");
    }
    const float* bias_p = nullptr;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be FP32");
        }
        if (attn_bias->size() != num_heads * L * L) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be (num_heads*L, L)");
        }
        bias_p = static_cast<const float*>(attn_bias->data);
    }
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(L, D, Dtype::FP16);
    }
    if (L == 0 || D == 0) return;

    run_sab_int8<__half>(X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv, Wo_int8, so,
                         d_mask, bias_p, num_heads, scale, O);
}

} // namespace detail::cuda
} // namespace brotensor
