// ─── CUDA masked-diffusion token selection ─────────────────────────────────
//
// CUDA port of src/cpu/masked_diffusion.cpp — `masked_diffusion_scores` and
// `masked_diffusion_commit`, the per-step token selection of a masked-
// diffusion language model over a (C codebooks, T frames) grid. See
// ops/sampling.h for the contract.
//
// ONE THREAD BLOCK PER CELL (c, t): the block cooperates over the vocabulary
// row (V ~ 1k) with warp-shuffle + shared-memory reductions — block max,
// FP64 block sum, block count, and a (value, index) block argmax with the
// CPU's tie rule (lowest index). The row's log-probs are staged once in a
// per-call device scratch (rows x V FP32, drawn from the stream-ordered pool;
// never host memory) and re-read from L2 by the later passes.
//
// The top-k class filter needs the k-th largest log-prob of the row. Instead
// of sorting, the kernel bisects the 32-bit order-preserving key of the
// floats: 32 rounds of "how many entries have key >= candidate" (one block
// count each) converge on exactly the k-th largest key, for any V, with no
// shared-memory staging of the row. Entries above that value are kept; among
// entries equal to it the lowest indices fill the remaining slots — the same
// rule the CPU applies after nth_element.
//
// Arithmetic mirrors the CPU op operation for operation: LSE = max +
// (float)log(FP64 sum of FP32 expf terms); the guided logit a + gs*(a - b)
// is written with __fadd_rn / __fmul_rn / __fsub_rn so nvcc cannot contract
// it into an FMA the CPU does not perform; the noise is the shared
// detail/hash_rng.h hash. CPU and CUDA therefore agree to the ulp except
// where their libm expf / logf differ.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   masked_diffusion_scores — pred, scores and confidence OVERWRITTEN.
//   masked_diffusion_commit — tokens / unmask_step written only at idx[0..k).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/hash_rng.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Defined in tensor.cu: stream-ordered pooled allocation (cudaMallocAsync).
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr unsigned kFullMask = 0xffffffffu;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// (a, ia) precedes (b, ib) in descending-value / ascending-index order.
__device__ __forceinline__ bool prefers(float a, int ia, float b, int ib) {
    if (a != b) return a > b;
    return ia < ib;
}

// Order-preserving 32-bit key: key(x) < key(y) iff x < y for non-NaN floats,
// with -0.0 folded onto +0.0 so the key order matches float comparison.
__device__ __forceinline__ unsigned sortable_key(float x) {
    unsigned b = __float_as_uint(x);
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float gumbel_from_uniform(float u) {
    return -logf(-logf(u + 1e-10f) + 1e-10f);
}

// ── block reductions (blockDim.x == kThreads; each ends in __syncthreads so
//    the shared scratch can be reused immediately) ──

__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(kFullMask, v, o));
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < kWarps) ? sh[lane] : neg_inf();
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(kFullMask, v, o));
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ double block_sum(double v, double* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFullMask, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < kWarps) ? sh[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFullMask, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    const double r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ int block_count(int v, int* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFullMask, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < kWarps) ? sh[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFullMask, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    const int r = sh[0];
    __syncthreads();
    return r;
}

// Block argmax under `prefers`: returns the index of the best (value, index).
__device__ __forceinline__ int block_argmax(float v, int i, float* shv, int* shi) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(kFullMask, v, o);
        const int oi = __shfl_down_sync(kFullMask, i, o);
        if (prefers(ov, oi, v, i)) { v = ov; i = oi; }
    }
    if (lane == 0) { shv[wid] = v; shi[wid] = i; }
    __syncthreads();
    if (wid == 0) {
        v = (lane < kWarps) ? shv[lane] : neg_inf();
        i = (lane < kWarps) ? shi[lane] : INT_MAX;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(kFullMask, v, o);
            const int oi = __shfl_down_sync(kFullMask, i, o);
            if (prefers(ov, oi, v, i)) { v = ov; i = oi; }
        }
        if (lane == 0) shi[0] = i;
    }
    __syncthreads();
    const int r = shi[0];
    __syncthreads();
    return r;
}

// One block per cell p = c*T + t.
__global__ void __launch_bounds__(kThreads)
masked_diffusion_scores_kernel(const float* __restrict__ logits,
                               const int32_t* __restrict__ tokens,
                               int T, int C, int V, int mask_id,
                               float gs, float layer_penalty,
                               float pos_temp, float class_temp, int k_keep,
                               uint64_t seed,
                               float* __restrict__ scratch,
                               int32_t* __restrict__ pred,
                               float* __restrict__ scores,
                               float* __restrict__ confidence) {
    __shared__ float sh_f[kWarps];
    __shared__ double sh_d[kWarps];
    __shared__ int sh_i[kWarps];
    __shared__ float sh_av[kWarps];
    __shared__ int sh_ai[kWarps];

    const int p = blockIdx.x;
    const int c = p / T;
    const int t = p - c * T;
    const int tid = threadIdx.x;
    const std::size_t row_stride = static_cast<std::size_t>(C) * V;
    const float* cond = logits + static_cast<std::size_t>(t) * row_stride +
                        static_cast<std::size_t>(c) * V;
    float* lp = scratch + static_cast<std::size_t>(p) * V;

    // ── log_probs into lp ──
    if (gs != 0.0f) {
        const float* unc = logits + static_cast<std::size_t>(T + t) * row_stride +
                           static_cast<std::size_t>(c) * V;
        float mc = neg_inf(), mu = neg_inf();
        for (int v = tid; v < V; v += kThreads) {
            mc = fmaxf(mc, cond[v]);
            mu = fmaxf(mu, unc[v]);
        }
        mc = block_max(mc, sh_f);
        mu = block_max(mu, sh_f);
        double sc = 0.0, su = 0.0;
        for (int v = tid; v < V; v += kThreads) {
            sc += static_cast<double>(expf(cond[v] - mc));
            su += static_cast<double>(expf(unc[v] - mu));
        }
        sc = block_sum(sc, sh_d);
        su = block_sum(su, sh_d);
        const float lse_c = mc + static_cast<float>(log(sc));
        const float lse_u = mu + static_cast<float>(log(su));
        float mg = neg_inf();
        for (int v = tid; v < V; v += kThreads) {
            const float a = __fsub_rn(cond[v], lse_c);
            const float b = __fsub_rn(unc[v], lse_u);
            const float g = __fadd_rn(a, __fmul_rn(gs, __fsub_rn(a, b)));
            lp[v] = g;
            mg = fmaxf(mg, g);
        }
        mg = block_max(mg, sh_f);
        double sg = 0.0;
        for (int v = tid; v < V; v += kThreads) {
            sg += static_cast<double>(expf(lp[v] - mg));
        }
        sg = block_sum(sg, sh_d);
        const float lse_g = mg + static_cast<float>(log(sg));
        for (int v = tid; v < V; v += kThreads) lp[v] = __fsub_rn(lp[v], lse_g);
    } else {
        float m = neg_inf();
        for (int v = tid; v < V; v += kThreads) m = fmaxf(m, cond[v]);
        m = block_max(m, sh_f);
        double s = 0.0;
        for (int v = tid; v < V; v += kThreads) {
            s += static_cast<double>(expf(cond[v] - m));
        }
        s = block_sum(s, sh_d);
        const float lse = m + static_cast<float>(log(s));
        for (int v = tid; v < V; v += kThreads) lp[v] = __fsub_rn(cond[v], lse);
    }
    __syncthreads();
    if (tid == 0) lp[mask_id] = neg_inf();
    __syncthreads();

    // ── confidence: max of the UNfiltered log_probs ──
    float conf = neg_inf();
    for (int v = tid; v < V; v += kThreads) conf = fmaxf(conf, lp[v]);
    conf = block_max(conf, sh_f);

    // ── prediction ──
    int best;
    if (class_temp > 0.0f) {
        // k-th largest key by bisection: the largest key K with
        // count(key >= K) >= k is exactly an element's key, the k-th largest.
        unsigned cur = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned cand = cur | (1u << bit);
            int cnt = 0;
            for (int v = tid; v < V; v += kThreads) cnt += (sortable_key(lp[v]) >= cand) ? 1 : 0;
            cnt = block_count(cnt, sh_i);
            if (cnt >= k_keep) cur = cand;
        }
        const unsigned thr_key = cur;
        int cnt_gt = 0;
        for (int v = tid; v < V; v += kThreads) cnt_gt += (sortable_key(lp[v]) > thr_key) ? 1 : 0;
        cnt_gt = block_count(cnt_gt, sh_i);
        const int m = k_keep - cnt_gt;   // equal-to-threshold slots, by index

        const uint64_t base = static_cast<uint64_t>(p) * static_cast<uint64_t>(V);
        float bs = neg_inf();
        int bi = INT_MAX;
        for (int v = tid; v < V; v += kThreads) {
            const unsigned key = sortable_key(lp[v]);
            bool kept = key > thr_key;
            if (!kept && key == thr_key) {
                int rank = 0;
                for (int w = 0; w < v; ++w) rank += (sortable_key(lp[w]) == thr_key) ? 1 : 0;
                kept = rank < m;
            }
            float s = neg_inf();
            if (kept) {
                const float u = hash_uniform(seed, kHashDomainMaskedDiffusionClass,
                                             base + static_cast<uint64_t>(v));
                s = __fadd_rn(__fdiv_rn(lp[v], class_temp), gumbel_from_uniform(u));
            }
            if (prefers(s, v, bs, bi)) { bs = s; bi = v; }
        }
        best = block_argmax(bs, bi, sh_av, sh_ai);
    } else {
        float bv = neg_inf();
        int bi = INT_MAX;
        for (int v = tid; v < V; v += kThreads) {
            if (prefers(lp[v], v, bv, bi)) { bv = lp[v]; bi = v; }
        }
        best = block_argmax(bv, bi, sh_av, sh_ai);
    }

    // ── score ──
    if (tid == 0) {
        pred[p] = best;
        confidence[p] = conf;   // raw, for every cell — masked or already decided
        float s = neg_inf();
        if (tokens[p] == mask_id) {
            s = __fsub_rn(conf, __fmul_rn(static_cast<float>(c), layer_penalty));
            if (pos_temp > 0.0f) {
                const float u = hash_uniform(seed, kHashDomainMaskedDiffusionPosition,
                                             static_cast<uint64_t>(p));
                s = __fadd_rn(__fdiv_rn(s, pos_temp), gumbel_from_uniform(u));
            }
        }
        scores[p] = s;
    }
}

__global__ void masked_diffusion_commit_kernel(const int32_t* __restrict__ pred,
                                               const int32_t* __restrict__ idx,
                                               int k, int step, int n,
                                               int32_t* __restrict__ tokens,
                                               int32_t* __restrict__ unmask_step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int32_t p = idx[i];
    if (p < 0 || p >= n) return;
    tokens[p] = pred[p];
    unmask_step[p] = step;
}

}  // namespace

// ─── masked_diffusion_scores ────────────────────────────────────────────────

void masked_diffusion_scores(const ::brotensor::Tensor& logits,
                             const ::brotensor::Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, uint64_t seed,
                             ::brotensor::Tensor& pred, ::brotensor::Tensor& scores,
                             ::brotensor::Tensor& confidence) {
    using ::brotensor::Dtype;
    const char* op = "masked_diffusion_scores";
    if (logits.dtype != Dtype::FP32) fail(op, "logits must be FP32");
    if (tokens.dtype != Dtype::INT32) fail(op, "tokens must be INT32");
    if (T < 1 || C < 1 || V < 1) fail(op, "T, C and V must be >= 1");
    if (mask_id < 0 || mask_id >= V) fail(op, "mask_id must be in [0, V)");
    const bool guided = guidance_scale != 0.0f;
    const int R = guided ? 2 * T : T;
    if (logits.rows != R || logits.cols != C * V) {
        fail(op, "logits must be (" + std::to_string(R) + ", C*V=" +
                 std::to_string(C * V) + ") for T=" + std::to_string(T) +
                 (guided ? " with" : " without") + " guidance, got (" +
                 std::to_string(logits.rows) + ", " + std::to_string(logits.cols) + ")");
    }
    if (tokens.rows != C || tokens.cols != T) fail(op, "tokens must be (C, T)");

    const int rows = C * T;
    if (pred.rows != C || pred.cols != T || pred.dtype != Dtype::INT32) {
        pred.resize(C, T, Dtype::INT32);
    }
    if (scores.rows != C || scores.cols != T || scores.dtype != Dtype::FP32) {
        scores.resize(C, T, Dtype::FP32);
    }
    if (confidence.rows != C || confidence.cols != T || confidence.dtype != Dtype::FP32) {
        confidence.resize(C, T, Dtype::FP32);
    }

    int k_keep = V;
    if (class_temperature > 0.0f) {
        const double kd = std::ceil(static_cast<double>(class_top_frac) * V);
        k_keep = (kd < 1.0) ? 1 : (kd > V ? V : static_cast<int>(kd));
    }

    const std::size_t scratch_bytes =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(V) * sizeof(float);
    float* scratch = static_cast<float*>(cuda_alloc(scratch_bytes));

    masked_diffusion_scores_kernel<<<rows, kThreads, 0, cur_stream()>>>(
        static_cast<const float*>(logits.data),
        static_cast<const int32_t*>(tokens.data),
        T, C, V, mask_id, guidance_scale, layer_penalty,
        position_temperature, class_temperature, k_keep, seed,
        scratch,
        static_cast<int32_t*>(pred.data),
        static_cast<float*>(scores.data),
        static_cast<float*>(confidence.data));
    const cudaError_t err = cudaGetLastError();
    cuda_free(scratch);   // stream-ordered: released after the kernel drains
    BROTENSOR_CUDA_CHECK(err);
}

// ─── masked_diffusion_commit ────────────────────────────────────────────────

void masked_diffusion_commit(const ::brotensor::Tensor& pred,
                             const ::brotensor::Tensor& idx, int k, int step,
                             ::brotensor::Tensor& tokens,
                             ::brotensor::Tensor& unmask_step) {
    using ::brotensor::Dtype;
    const char* op = "masked_diffusion_commit";
    if (pred.dtype != Dtype::INT32) fail(op, "pred must be INT32");
    if (idx.dtype != Dtype::INT32) fail(op, "idx must be INT32");
    if (tokens.dtype != Dtype::INT32) fail(op, "tokens must be INT32");
    if (unmask_step.dtype != Dtype::INT32) fail(op, "unmask_step must be INT32");
    if (tokens.rows != pred.rows || tokens.cols != pred.cols ||
        unmask_step.rows != pred.rows || unmask_step.cols != pred.cols) {
        fail(op, "pred, tokens and unmask_step must share one (C, T) shape");
    }
    if (k < 0) fail(op, "k must be >= 0");
    if (k > idx.size()) fail(op, "idx must hold at least k entries");
    if (k == 0) return;

    const int threads = 128;
    const int blocks = (k + threads - 1) / threads;
    masked_diffusion_commit_kernel<<<blocks, threads, 0, cur_stream()>>>(
        static_cast<const int32_t*>(pred.data),
        static_cast<const int32_t*>(idx.data),
        k, step, pred.size(),
        static_cast<int32_t*>(tokens.data),
        static_cast<int32_t*>(unmask_step.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_masked_diffusion(::brotensor::detail::OpsVTable& v) {
    v.masked_diffusion_scores = &masked_diffusion_scores;
    v.masked_diffusion_commit = &masked_diffusion_commit;
}

}  // namespace brotensor::detail::cuda
