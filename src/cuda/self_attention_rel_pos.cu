// ─── CUDA decomposed relative-position self-attention ──────────────────────
//
// self_attention_decomposed_rel_pos_forward (SAM / ViTDet global blocks), its
// windowed variant, and rel_pos_bias_xl_forward (Transformer-XL). Shares the
// tiled GEMM and the head split / merge kernels with self_attention_bias.cu
// through self_attention_internal.cuh.

#include "self_attention_internal.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

// ── Decomposed 2D relative-position attention (SAM / ViTDet) ───────────────
//
// Same math as run_sab plus the decomposed rel-pos bias, but every GEMM-shaped
// stage runs as a real tiled GEMM instead of a one-thread-per-output kernel:
// FP16/BF16 go through fp16_internal's WMMA tensor-core matmul (FP32
// accumulate), FP32 through the register-tiled kernel below. The rel-pos bias
// is factored as in segment_anything's add_decomposed_rel_pos:
//   Bh[i, r] = q_i . rel_pos_h[r]   and   Bw[i, r] = q_i . rel_pos_w[r]
// are two skinny GEMMs against the rel tables; the (qh-kh)/(qw-kw) lookups and
// the qk scale fold into the softmax load, so the bias never costs a pass over
// the (L, L) scores. Softmax runs in place on the scores buffer (the only
// (H*L, L)-sized scratch), with row normalisation deferred to the head-merge.
//
// One pipeline serves both the global op (one unit = the whole grid) and the
// windowed op (one unit per window): all units share weights and rel tables,
// so the projections are single GEMMs over the concatenated unit rows and
// scores / A@V are strided-batched GEMMs with batch = nWin * num_heads.
//
// Intermediates are stored in the model dtype; all accumulation is FP32. Token
// panels are padded to Lp = ceil8(Lw) rows so the 16-bit WMMA path keeps its
// int4 alignment; padded rows project from zero Q rows (zero scores, skipped
// by softmax) and padded key columns get probability 0 before the A@V GEMM.
//
// sardp_dtype, the GEMM and the head split / merge kernels are in
// self_attention_internal.cuh.

// In-place row softmax over the typed (batch*Lp, Lp) scores panel, folding the
// qk scale and the decomposed rel-pos lookups into the load:
//   s_j = scale * S[row, j] + Bh[row, (qh-kh)+gh-1] + Bw[row, (qw-kw)+gw-1]
// One block per query row. Writes UNNORMALISED exp(s - max) and the row sum to
// `sums` — the 1/sum scaling is deferred to the head-merge kernel, saving a
// third pass over the (Lp, Lp) panel. Alignment-padding rows (i >= Lw) keep
// their zero scores; padded key columns get probability 0 so the A@V GEMM can
// run over the full Lp width.
template <typename T>
__global__ void sardp_softmax_kernel(T* __restrict__ S,
                                     const T* __restrict__ Bh,
                                     const T* __restrict__ Bw,
                                     float* __restrict__ sums,
                                     int Lw, int Lp, int gh, int gw,
                                     float scale) {
    __shared__ float sdata[SAB_SM_BLOCK];
    const long long row = blockIdx.x;           // b*Lp + i
    const int i = static_cast<int>(row % Lp);
    if (i >= Lw) return;
    const int tid = threadIdx.x;
    T* srow = S + row * Lp;
    const T* bhrow = Bh + row * (2 * gh - 1);
    const T* bwrow = Bw + row * (2 * gw - 1);
    const int qh = i / gw, qw = i % gw;

    float local_max = -1e30f;
    for (int j = tid; j < Lw; j += blockDim.x) {
        const float s = scale * sab_ld(srow[j])
                      + sab_ld(bhrow[qh - j / gw + gh - 1])
                      + sab_ld(bwrow[qw - j % gw + gw - 1]);
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
        if (j >= Lw) { sab_st(srow[j], 0.0f); continue; }
        const float s = scale * sab_ld(srow[j])
                      + sab_ld(bhrow[qh - j / gw + gh - 1])
                      + sab_ld(bwrow[qw - j % gw + gw - 1]);
        const float e = expf(s - m);
        sab_st(srow[j], e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sums[row] = sdata[0];  // >= exp(0) — never zero
}

// Run the pipeline over nWin units of Lw tokens each. Xr is the concatenated
// (nWin*Lw, D) unit rows (the grid itself for the global op, the gathered
// window partition for the windowed op); Or receives the same layout.
template <typename T>
void run_sardp_fast(const T* Xr, T* Or, int nWin, int Lw, int D, int H,
                    int gh, int gw, float scale,
                    const ::brotensor::Tensor& Wq, const T* bq,
                    const ::brotensor::Tensor& Wk, const T* bk,
                    const ::brotensor::Tensor& Wv, const T* bv,
                    const ::brotensor::Tensor& Wo, const T* bo,
                    const ::brotensor::Tensor& rel_h,
                    const ::brotensor::Tensor& rel_w) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    constexpr Dtype dt = sardp_dtype<T>::value;
    const int dh    = D / H;
    const int Lp    = (Lw + 7) & ~7;
    const int batch = nWin * H;
    const int R     = nWin * Lw;
    const size_t panel_q = static_cast<size_t>(Lp) * dh;
    const size_t panel_s = static_cast<size_t>(Lp) * Lp;

    const T* Wq_p = static_cast<const T*>(Wq.data);
    const T* Wk_p = static_cast<const T*>(Wk.data);
    const T* Wv_p = static_cast<const T*>(Wv.data);
    const T* Wo_p = static_cast<const T*>(Wo.data);
    const T* rh_p = static_cast<const T*>(rel_h.data);
    const T* rw_p = static_cast<const T*>(rel_w.data);

    // Per-(unit, head) Q/K panels, V transposed to (dh, Lp).
    Tensor Qh = Tensor::empty_on(Device::CUDA, batch * Lp, dh, dt);
    Tensor Kh = Tensor::empty_on(Device::CUDA, batch * Lp, dh, dt);
    Tensor Vt = Tensor::empty_on(Device::CUDA, batch * dh, Lp, dt);
    T* Qh_p = static_cast<T*>(Qh.data);
    T* Kh_p = static_cast<T*>(Kh.data);
    T* Vt_p = static_cast<T*>(Vt.data);
    {
        // Q/K/V projections: one GEMM each over all units' rows, then split
        // into head panels (the row buffers die at scope exit).
        Tensor Qrow = Tensor::empty_on(Device::CUDA, R, D, dt);
        Tensor Krow = Tensor::empty_on(Device::CUDA, R, D, dt);
        Tensor Vrow = Tensor::empty_on(Device::CUDA, R, D, dt);
        sardp_gemm(Xr, Wq_p, static_cast<T*>(Qrow.data), 1, R, D, D, 0, 0, 0, bq);
        sardp_gemm(Xr, Wk_p, static_cast<T*>(Krow.data), 1, R, D, D, 0, 0, 0, bk);
        sardp_gemm(Xr, Wv_p, static_cast<T*>(Vrow.data), 1, R, D, D, 0, 0, 0, bv);

        const long long total = static_cast<long long>(batch) * Lp * dh;
        const int grid = sardp_flat_grid(total);
        sardp_split_heads_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Qrow.data), Qh_p, H, Lw, Lp, dh, total);
        sardp_split_heads_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Krow.data), Kh_p, H, Lw, Lp, dh, total);
        sardp_split_heads_t_kernel<T><<<grid, 256, 0, cur_stream()>>>(
            static_cast<const T*>(Vrow.data), Vt_p, H, Lw, Lp, dh, total);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Rel-pos factor panels (skinny GEMMs; the tables are shared by every
    // unit and head, so one launch covers the whole batch).
    Tensor Bh = Tensor::empty_on(Device::CUDA, batch * Lp, 2 * gh - 1, dt);
    Tensor Bw = Tensor::empty_on(Device::CUDA, batch * Lp, 2 * gw - 1, dt);
    sardp_gemm(Qh_p, rh_p, static_cast<T*>(Bh.data),
               1, batch * Lp, 2 * gh - 1, dh, 0, 0, 0, nullptr);
    sardp_gemm(Qh_p, rw_p, static_cast<T*>(Bw.data),
               1, batch * Lp, 2 * gw - 1, dh, 0, 0, 0, nullptr);

    // Scores = Q @ K^T (batched), then in-place fused softmax.
    Tensor S = Tensor::empty_on(Device::CUDA, batch * Lp, Lp, dt);
    Tensor sums = Tensor::empty_on(Device::CUDA, batch * Lp, 1, Dtype::FP32);
    T* S_p = static_cast<T*>(S.data);
    sardp_gemm(Qh_p, Kh_p, S_p, batch, Lp, Lp, dh,
               panel_q, panel_q, panel_s, nullptr);
    sardp_softmax_kernel<T><<<batch * Lp, SAB_SM_BLOCK, 0, cur_stream()>>>(
        S_p, static_cast<const T*>(Bh.data), static_cast<const T*>(Bw.data),
        static_cast<float*>(sums.data), Lw, Lp, gh, gw, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // A @ V (batched against the transposed V panels), merge heads with the
    // deferred 1/sum, output projection.
    Tensor Y = Tensor::empty_on(Device::CUDA, batch * Lp, dh, dt);
    sardp_gemm(S_p, Vt_p, static_cast<T*>(Y.data), batch, Lp, dh, Lp,
               panel_s, panel_q, panel_q, nullptr);
    Tensor Yrow = Tensor::empty_on(Device::CUDA, R, D, dt);
    {
        const long long total = static_cast<long long>(R) * D;
        sardp_merge_heads_kernel<T><<<sardp_flat_grid(total), 256, 0, cur_stream()>>>(
            static_cast<const T*>(Y.data), static_cast<const float*>(sums.data),
            static_cast<T*>(Yrow.data), H, Lw, Lp, dh, total);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    sardp_gemm(static_cast<const T*>(Yrow.data), Wo_p, Or, 1, R, D, D, 0, 0, 0, bo);
}

template <typename T>
void run_sardp(const ::brotensor::Tensor& X,
               const ::brotensor::Tensor& Wq, const T* bq,
               const ::brotensor::Tensor& Wk, const T* bk,
               const ::brotensor::Tensor& Wv, const T* bv,
               const ::brotensor::Tensor& Wo, const T* bo,
               const ::brotensor::Tensor& rel_h, const ::brotensor::Tensor& rel_w,
               int num_heads, int gh, int gw, float scale,
               ::brotensor::Tensor& O) {
    run_sardp_fast<T>(static_cast<const T*>(X.data), static_cast<T*>(O.data),
                      /*nWin=*/1, /*Lw=*/X.rows, X.cols, num_heads, gh, gw,
                      scale, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_h, rel_w);
}

// ─── Windowed decomposed-rel-pos attention (SAM windowed encoder block) ──────
//
// Gathers the (grid_h, grid_w) token grid into a contiguous per-window batch
// (zero-padding the bottom/right up to a multiple of `window`), runs ONE
// run_sardp_fast over all windows at once — projections as single GEMMs over
// the gathered rows, scores/A@V batched across windows x heads — then scatters
// the result back, dropping the padded tokens.
//
// Partition row r = w_idx*window*window + lh*window + lw maps to grid token
// (h, w) = (nh*window+lh, nw*window+lw) with (nh, nw) = (w_idx/nw_w, w_idx%nw_w).

template <typename T>
__global__ void win_gather_kernel(const T* __restrict__ X, T* __restrict__ P,
                                  int grid_h, int grid_w, int window,
                                  int nw_w, int D, int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= D || row >= nrows) return;
    const int ww  = window * window;
    const int loc = row % ww;
    const int wi  = row / ww;
    const int h   = (wi / nw_w) * window + loc / window;
    const int w   = (wi % nw_w) * window + loc % window;
    T* dst = P + static_cast<size_t>(row) * D + col;
    if (h < grid_h && w < grid_w)
        *dst = X[static_cast<size_t>(h * grid_w + w) * D + col];
    else
        sab_st(*dst, 0.0f);
}

template <typename T>
__global__ void win_scatter_kernel(const T* __restrict__ P, T* __restrict__ O,
                                   int grid_h, int grid_w, int window,
                                   int nw_w, int D, int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= D || row >= nrows) return;
    const int ww  = window * window;
    const int loc = row % ww;
    const int wi  = row / ww;
    const int h   = (wi / nw_w) * window + loc / window;
    const int w   = (wi % nw_w) * window + loc % window;
    if (h < grid_h && w < grid_w)
        O[static_cast<size_t>(h * grid_w + w) * D + col] =
            P[static_cast<size_t>(row) * D + col];
}

template <typename T>
void run_windowed_sardp(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Wq, const T* bq,
                        const ::brotensor::Tensor& Wk, const T* bk,
                        const ::brotensor::Tensor& Wv, const T* bv,
                        const ::brotensor::Tensor& Wo, const T* bo,
                        const ::brotensor::Tensor& rel_h,
                        const ::brotensor::Tensor& rel_w,
                        int num_heads, int grid_h, int grid_w, int window,
                        float scale, ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    const int D     = X.cols;
    const auto dt   = X.dtype;
    const int pad_h = (window - grid_h % window) % window;
    const int pad_w = (window - grid_w % window) % window;
    const int nw_h  = (grid_h + pad_h) / window;
    const int nw_w  = (grid_w + pad_w) / window;
    const int nW    = nw_h * nw_w;
    const int ww    = window * window;
    const int nrows = nW * ww;

    Tensor Pin  = Tensor::empty_on(Device::CUDA, nrows, D, dt);
    Tensor Pout = Tensor::empty_on(Device::CUDA, nrows, D, dt);

    const dim3 block(16, 16);
    const dim3 grid((D + block.x - 1) / block.x,
                    (nrows + block.y - 1) / block.y);
    win_gather_kernel<T><<<grid, block, 0, cur_stream()>>>(static_cast<const T*>(X.data),
                                          static_cast<T*>(Pin.data),
                                          grid_h, grid_w, window, nw_w, D, nrows);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // All windows share weights and rel-pos tables, so one batched pipeline
    // covers the whole partition (padded windows are ordinary zero tokens,
    // exactly as in SAM's window_partition).
    run_sardp_fast<T>(static_cast<const T*>(Pin.data), static_cast<T*>(Pout.data),
                      nW, ww, D, num_heads, window, window, scale,
                      Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_h, rel_w);

    win_scatter_kernel<T><<<grid, block, 0, cur_stream()>>>(static_cast<const T*>(Pout.data),
                                           static_cast<T*>(O.data),
                                           grid_h, grid_w, window, nw_w, D, nrows);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void self_attention_decomposed_rel_pos_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h,
        const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const char* fn = "self_attention_decomposed_rel_pos_forward";
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_pos_h.dtype != X.dtype || rel_pos_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * grid_h - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*grid_h-1, head_dim)");
    if (rel_pos_w.rows != 2 * grid_w - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*grid_w-1, head_dim)");
    auto check_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_sardp<float>(X, Wq, static_cast<const float*>(bp(bq)),
                         Wk, static_cast<const float*>(bp(bk)),
                         Wv, static_cast<const float*>(bp(bv)),
                         Wo, static_cast<const float*>(bp(bo)),
                         rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    case Dtype::FP16:
        run_sardp<__half>(X, Wq, static_cast<const __half*>(bp(bq)),
                          Wk, static_cast<const __half*>(bp(bk)),
                          Wv, static_cast<const __half*>(bp(bv)),
                          Wo, static_cast<const __half*>(bp(bo)),
                          rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    default:  // BF16
        run_sardp<__nv_bfloat16>(X, Wq, static_cast<const __nv_bfloat16*>(bp(bq)),
                                 Wk, static_cast<const __nv_bfloat16*>(bp(bk)),
                                 Wv, static_cast<const __nv_bfloat16*>(bp(bv)),
                                 Wo, static_cast<const __nv_bfloat16*>(bp(bo)),
                                 rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    }
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h,
        const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const char* fn = "self_attention_decomposed_rel_pos_windowed_forward";
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_pos_h.dtype != X.dtype || rel_pos_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    const int L = X.rows;
    const int D = X.cols;
    if (window <= 0)
        throw std::runtime_error(std::string(fn) + ": window must be >= 1");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * window - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*window-1, head_dim)");
    if (rel_pos_w.rows != 2 * window - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*window-1, head_dim)");
    auto check_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_windowed_sardp<float>(X, Wq, static_cast<const float*>(bp(bq)),
                                  Wk, static_cast<const float*>(bp(bk)),
                                  Wv, static_cast<const float*>(bp(bv)),
                                  Wo, static_cast<const float*>(bp(bo)),
                                  rel_pos_h, rel_pos_w, num_heads,
                                  grid_h, grid_w, window, scale, O);
        break;
    case Dtype::FP16:
        run_windowed_sardp<__half>(X, Wq, static_cast<const __half*>(bp(bq)),
                                   Wk, static_cast<const __half*>(bp(bk)),
                                   Wv, static_cast<const __half*>(bp(bv)),
                                   Wo, static_cast<const __half*>(bp(bo)),
                                   rel_pos_h, rel_pos_w, num_heads,
                                   grid_h, grid_w, window, scale, O);
        break;
    default:  // BF16
        run_windowed_sardp<__nv_bfloat16>(X, Wq, static_cast<const __nv_bfloat16*>(bp(bq)),
                                          Wk, static_cast<const __nv_bfloat16*>(bp(bk)),
                                          Wv, static_cast<const __nv_bfloat16*>(bp(bv)),
                                          Wo, static_cast<const __nv_bfloat16*>(bp(bo)),
                                          rel_pos_h, rel_pos_w, num_heads,
                                          grid_h, grid_w, window, scale, O);
        break;
    }
}

// ─── Transformer-XL relative-position bias ─────────────────────────────────
//
// Bias[h*T + q, k] = sum_d Qv[q, h*dk + d] * Pk[(T-1-q) + k, h*dk + d].
//
// One block per (head, query) row and one thread per key, so the whole
// num_heads*T*T*head_dim reduction is num_heads*T blocks deep — hundreds of
// blocks of T threads, which is the shape that fills a card. The query's
// head_dim slice is read once into shared memory because every thread in the
// block multiplies against it; `Pk` is read straight from global, where the
// L2 carries it: consecutive `k` are consecutive *rows* of Pk, so the reads are
// strided, but each row is read by T of the blocks and the whole (2T-1, D)
// encoding is a couple of megabytes.
//
// The rel_shift is the `base + k` index and costs nothing — there is no
// (T, 2T-1) matrix_bd here to build and then shift.
namespace {

constexpr int kRelPosMaxHeadDim = 512;   // 4 kB shared; Conformer's is 128

__global__ void rel_pos_bias_xl_kernel(const float* __restrict__ qv,
                                       const float* __restrict__ pk,
                                       float* __restrict__ out,
                                       int T, int D, int head_dim) {
    extern __shared__ float qs[];        // head_dim floats: this query's slice

    const int row = blockIdx.x;          // h*T + q
    const int h   = row / T;
    const int q   = row - h * T;
    const int co  = h * head_dim;

    const float* qrow = qv + static_cast<long long>(q) * D + co;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) qs[d] = qrow[d];
    __syncthreads();

    const int base = T - 1 - q;
    for (int k = threadIdx.x; k < T; k += blockDim.x) {
        const float* prow = pk + static_cast<long long>(base + k) * D + co;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += qs[d] * prow[d];
        out[static_cast<long long>(row) * T + k] = s;
    }
}

} // namespace

void rel_pos_bias_xl_forward(const ::brotensor::Tensor& Qv,
                             const ::brotensor::Tensor& Pk,
                             int num_heads, int head_dim,
                             ::brotensor::Tensor& Bias) {
    if (Qv.dtype != Dtype::FP32 || Pk.dtype != Dtype::FP32)
        throw std::runtime_error("rel_pos_bias_xl_forward: Qv and Pk must be FP32");
    const int T = Qv.rows;
    const int D = Qv.cols;
    if (num_heads <= 0 || head_dim <= 0 || num_heads * head_dim != D)
        throw std::runtime_error("rel_pos_bias_xl_forward: num_heads*head_dim "
                                 "must equal Qv.cols");
    if (Pk.cols != D || Pk.rows != 2 * T - 1)
        throw std::runtime_error("rel_pos_bias_xl_forward: Pk must be "
                                 "(2*Qv.rows - 1, Qv.cols)");
    if (head_dim > kRelPosMaxHeadDim)
        throw std::runtime_error("rel_pos_bias_xl_forward: head_dim above 512 "
                                 "is not implemented on CUDA");
    if (Bias.rows != num_heads * T || Bias.cols != T ||
        Bias.dtype != Dtype::FP32) {
        Bias = ::brotensor::Tensor::empty_on(Qv.device, num_heads * T, T,
                                             Dtype::FP32);
    }
    if (T <= 0) return;

    const int threads = T < 256 ? ((T + 31) / 32) * 32 : 256;
    const dim3 grid(static_cast<unsigned>(num_heads * T));
    const size_t shmem = static_cast<size_t>(head_dim) * sizeof(float);
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    rel_pos_bias_xl_kernel<<<grid, threads, shmem, stream>>>(
        static_cast<const float*>(Qv.data),
        static_cast<const float*>(Pk.data),
        static_cast<float*>(Bias.data), T, D, head_dim);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
