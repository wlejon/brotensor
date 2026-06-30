// ─── CUDA autoregressive logit sampling (CHUNK 7, family F) ─────────────────
//
// CUDA port of src/cpu/sample_logits.cpp — the next-token sampler used by
// autoregressive generation loops (brosoundml codec-LM decoding and the brolm
// language-model project).
//
// One thread per row of the (N, V) logit matrix. Each thread runs the full
// per-row pipeline serially:
//   temperature scale -> softmax -> descending-probability sort -> top-k filter
//   -> top-p (nucleus) filter -> renormalise over the kept set -> inverse-CDF
//   draw with a Philox-generated uniform. temperature == 0 short-circuits to a
//   deterministic argmax (no RNG consumed).
//
// ── INT32 output ────────────────────────────────────────────────────────────
//   indices — (N, 1) INT32 sampled token ids. Resized AND dtype-set to INT32.
//
// ── Philox (key, counter) ABI ───────────────────────────────────────────────
//   Standard Philox 4x32-10 counter-based generator. Row n draws its uniform
//   from substream (counter + n); the construction here is byte-identical to
//   the CPU op so a given (key, counter) yields the same draws on both
//   backends. See ops.h / src/cpu/sample_logits.cpp for the full ABI contract.
//
// ── Accumulator precision ───────────────────────────────────────────────────
//   The softmax sum, the nucleus cumulative, the kept-set sum and the
//   inverse-CDF accumulator all use FP64 — matching the CPU reference so a
//   draw landing near a CDF-bucket boundary resolves to the same token.
//
// ── Scratch ─────────────────────────────────────────────────────────────────
//   The per-row probability vector and its sorted index order do not fit in a
//   register file for arbitrary V, so two N*V FP32 buffers (prob + sort work)
//   and one N*V INT32 buffer (order) are cudaMalloc'd per call.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

constexpr int SL_BLOCK = 128;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

inline int sl_grid(long long n) {
    long long blocks = (n + SL_BLOCK - 1) / SL_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// ── Philox 4x32-10 counter-based RNG (PyTorch / JAX compatible) ──────────────
// Byte-identical to the CPU reference in src/cpu/sample_logits.cpp.

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

// Draw one uniform in [0, 1) for `substream`, seeded by `key64`. The first of
// the four uint32 outputs is mapped to [0, 1) via its top 24 bits / 2^24.
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
            key[0] += 0x9E3779B9u;   // golden ratio
            key[1] += 0xBB67AE85u;   // sqrt(3) - 1
        }
    }
    return static_cast<float>(ctr[0] >> 8) / 16777216.0f;   // top 24 bits
}

constexpr float kNeg = -3.4028235e38f;   // -FLT_MAX

// One thread per row of the (N, V) logit matrix.
__global__ void sample_logits_kernel(const float* __restrict__ logits,
                                     int* __restrict__ indices,
                                     float* __restrict__ prob,
                                     float* __restrict__ work,
                                     int* __restrict__ order,
                                     int N, int V, float temperature,
                                     int top_k, float top_p,
                                     uint64_t key, uint64_t counter) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         n < N; n += (long long)blockDim.x * gridDim.x) {
        const float* row = logits + n * V;

        // ── Greedy: temperature == 0 -> deterministic argmax, no RNG. ──
        if (temperature == 0.0f) {
            float best_v = kNeg;
            int   best_i = 0;
            for (int v = 0; v < V; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best_i = v; }
            }
            indices[n] = best_i;
            continue;
        }

        float* prob_n = prob + n * V;
        float* work_n = work + n * V;
        int*   ord_n  = order + n * V;

        // ── 1. temperature scale + 2. softmax (numerically stable). ──
        float max_logit = kNeg;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            if (s > max_logit) max_logit = s;
        }
        double sum = 0.0;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            const float e = expf(s - max_logit);
            prob_n[v] = e;
            sum += e;
        }
        const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum)
                                          : 0.0f;
        for (int v = 0; v < V; ++v) {
            prob_n[v] *= inv_sum;
            work_n[v]  = prob_n[v];     // sort scratch
        }

        // Descending-probability order; ties broken by lower index. Selection
        // sort scanning ascending with strict `>` reproduces std::stable_sort.
        for (int r = 0; r < V; ++r) {
            float best = kNeg;
            int   best_i = 0;
            for (int v = 0; v < V; ++v) {
                if (work_n[v] > best) { best = work_n[v]; best_i = v; }
            }
            ord_n[r] = best_i;
            work_n[best_i] = kNeg;      // remove from the running set
        }

        // ── 3. top-k filter: keep the top_k highest-probability tokens. ──
        int keep = V;
        if (top_k > 0 && top_k < keep) keep = top_k;

        // ── 4. top-p (nucleus): smallest high-prob set with cumprob >= top_p. ──
        if (top_p < 1.0f) {
            double cum = 0.0;
            int nucleus = 0;
            for (int r = 0; r < keep; ++r) {
                cum += prob_n[ord_n[r]];
                ++nucleus;
                if (cum >= static_cast<double>(top_p)) break;
            }
            if (nucleus < 1) nucleus = 1;   // always keep at least one token
            keep = nucleus;
        }

        // ── 5. renormalise over the kept set. ──
        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += prob_n[ord_n[r]];

        // ── 6. inverse-CDF draw with a Philox uniform for substream
        //       (counter + n). ──
        const float u = philox_uniform(key, counter + static_cast<uint64_t>(n));
        int chosen = ord_n[0];
        if (kept_sum > 0.0) {
            const double target = static_cast<double>(u) * kept_sum;
            double acc = 0.0;
            chosen = ord_n[keep - 1];   // fallback: last kept (covers u≈1).
            for (int r = 0; r < keep; ++r) {
                acc += prob_n[ord_n[r]];
                if (target < acc) { chosen = ord_n[r]; break; }
            }
        }
        indices[n] = chosen;
    }
}

// Graph-capturable variant — ONE THREAD BLOCK PER ROW.
//
// The autoregressive decode case is N==1 with a large vocab V, so a thread-per-
// row kernel would leave one thread to do an O(V^2) selection sort while the
// rest of the block idles (the original bottleneck). Here each block cooperates
// on a single row: parallel softmax (block max + sum), then a partial selection
// that extracts the kept set in descending-probability order, stopping as soon
// as top_k / top_p is satisfied (so the cost is O(keep * V / threads), not
// O(V^2)). The Philox base counter is read from device memory so a captured
// replay advances the RNG with no host involvement; scratch is caller-owned (no
// per-call malloc, illegal during capture). Accumulations are FP64 to match the
// CPU reference bit-for-bit on robust draws. The counter is advanced by a
// separate single-thread kernel so no block races the write.
//
// Shared memory (dynamic): blockDim doubles (reduction values) + blockDim ints
// (reduction indices).
constexpr int SLI_THREADS = 256;

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
    double* sdv  = reinterpret_cast<double*>(smem);          // [nth] reduction val
    int*    sidx = reinterpret_cast<int*>(sdv + nth);        // [nth] reduction idx

    const float* rowL = logits + static_cast<long long>(row) * V;

    // ── Greedy: block-parallel argmax (max value, lowest index on ties). ──
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

    float* prob_n = prob + static_cast<long long>(row) * V;
    float* work_n = work + static_cast<long long>(row) * V;
    int*   ord_n  = order + static_cast<long long>(row) * V;

    // ── softmax max (parallel). ──
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

    // ── softmax exp + sum (parallel, FP64 accumulation to match the CPU op). ──
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

    // normalize + seed the destructive work copy (parallel).
    for (int v = tid; v < V; v += nth) {
        const float p = prob_n[v] * inv_sum;
        prob_n[v] = p;
        work_n[v] = p;
    }

    // ── Partial selection: extract the kept set in descending order, stopping
    //    once top_k / top_p is met. O(keep * V / threads) instead of O(V^2). ──
    int keep_max = V;
    if (top_k > 0 && top_k < keep_max) keep_max = top_k;

    __shared__ double s_cum;
    __shared__ int    s_keep;
    __shared__ int    s_stop;
    if (tid == 0) { s_cum = 0.0; s_keep = 0; s_stop = 0; }

    for (int r = 0; r < keep_max; ++r) {
        __syncthreads();                       // isolate from the prior iteration
        double bv = -1e300; int bi = 0;
        for (int v = tid; v < V; v += nth) {
            double pv = work_n[v];
            if (pv > bv) { bv = pv; bi = v; }   // strict '>' -> lowest index wins
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
            work_n[idx] = kNeg;                 // remove from the running set
            s_cum += prob_n[idx];
            if ((top_p < 1.0f && s_cum >= static_cast<double>(top_p)) ||
                r + 1 == keep_max) {
                s_keep = r + 1;
                s_stop = 1;
            }
        }
        __syncthreads();
        if (s_stop) break;                      // uniform across the block
    }
    int keep = s_keep;
    if (keep < 1) keep = 1;

    // ── Renormalise over the kept set + inverse-CDF draw (serial on thread 0;
    //    O(keep) <= O(V), no longer the quadratic term). ──
    if (tid == 0) {
        const uint64_t base = static_cast<uint64_t>(
            static_cast<uint32_t>(counter[0]));
        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += prob_n[ord_n[r]];
        const float u = philox_uniform(key, base + static_cast<uint64_t>(row));
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

// Advance the device base counter by N (single thread). Greedy (temperature==0)
// consumes no RNG, so the caller skips this and the counter is untouched.
__global__ void advance_counter_kernel(int* counter, int N) {
    counter[0] = static_cast<int>(
        static_cast<uint32_t>(counter[0]) + static_cast<uint32_t>(N));
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrapper
// ════════════════════════════════════════════════════════════════════════════

void sample_logits(const ::brotensor::Tensor& logits, float temperature,
                   int top_k, float top_p, uint64_t key, uint64_t counter,
                   ::brotensor::Tensor& indices) {
    const char* op = "sample_logits";
    if (logits.dtype != ::brotensor::Dtype::FP32) {
        fail(op, "logits must be FP32");
    }
    if (temperature < 0.0f) fail(op, "temperature must be >= 0");
    if (top_k < 0)          fail(op, "top_k must be >= 0");
    if (top_p < 0.0f)       fail(op, "top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) {
        fail(op, "vocabulary size (logits.cols) must be > 0");
    }

    // indices: (N, 1) INT32 — resize AND dtype-set.
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }
    if (N == 0) return;

    // Per-call scratch: prob + sort work (FP32) and the sorted order (INT32).
    const size_t nv = static_cast<size_t>(N) * static_cast<size_t>(V);
    float* prob  = nullptr;
    float* work  = nullptr;
    int*   order = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&prob,  nv * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMalloc(&work,  nv * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMalloc(&order, nv * sizeof(int)));

    sample_logits_kernel<<<sl_grid(N), SL_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(logits.data),
        static_cast<int*>(indices.data),
        prob, work, order,
        N, V, temperature, top_k, top_p, key, counter);
    const cudaError_t launch_err = cudaGetLastError();

    BROTENSOR_CUDA_CHECK(cudaFree(prob));
    BROTENSOR_CUDA_CHECK(cudaFree(work));
    BROTENSOR_CUDA_CHECK(cudaFree(order));
    BROTENSOR_CUDA_CHECK(launch_err);
}

void sample_logits_into(const ::brotensor::Tensor& logits, float temperature,
                        int top_k, float top_p, uint64_t key,
                        ::brotensor::Tensor& counter,
                        ::brotensor::Tensor& scratch,
                        ::brotensor::Tensor& indices) {
    const char* op = "sample_logits_into";
    if (logits.dtype != ::brotensor::Dtype::FP32) fail(op, "logits must be FP32");
    if (temperature < 0.0f) fail(op, "temperature must be >= 0");
    if (top_k < 0)          fail(op, "top_k must be >= 0");
    if (top_p < 0.0f)       fail(op, "top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) fail(op, "vocabulary size (logits.cols) must be > 0");

    if (counter.dtype != ::brotensor::Dtype::INT32 ||
        static_cast<size_t>(counter.rows) * counter.cols < 1) {
        fail(op, "counter must be an INT32 tensor with >= 1 element");
    }
    const size_t nv = static_cast<size_t>(N) * static_cast<size_t>(V);
    if (scratch.dtype != ::brotensor::Dtype::FP32 ||
        static_cast<size_t>(scratch.rows) * scratch.cols < 3 * nv) {
        fail(op, "scratch must be FP32 with at least 3*N*V elements");
    }
    // indices must be a pre-sized (N,1) INT32 — never resized (a resize would
    // allocate mid-capture).
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        fail(op, "indices must be a pre-sized (N,1) INT32 tensor");
    }
    if (N == 0) return;

    // Carve the caller's workspace: prob | sort-work (FP32) | order (INT32).
    float* prob  = static_cast<float*>(scratch.data);
    float* work  = prob + nv;
    int*   order = reinterpret_cast<int*>(prob + 2 * nv);

    // One block per row; the block cooperates over the vocab. Cap the thread
    // count at the vocab (no point in more threads than elements).
    int threads = SLI_THREADS;
    while (threads > 32 && threads > V) threads >>= 1;
    const size_t shmem = static_cast<size_t>(threads) *
                         (sizeof(double) + sizeof(int));
    sample_logits_into_kernel<<<N, threads, shmem, cur_stream()>>>(
        static_cast<const float*>(logits.data),
        static_cast<int*>(indices.data),
        prob, work, order,
        static_cast<const int*>(counter.data),
        N, V, temperature, top_k, top_p, key);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Greedy consumes no RNG; only advance the stream when actually sampling.
    if (temperature != 0.0f) {
        advance_counter_kernel<<<1, 1, 0, cur_stream()>>>(
            static_cast<int*>(counter.data), N);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_sample_logits(::brotensor::detail::OpsVTable& v) {
    v.sample_logits = &sample_logits;
    v.sample_logits_into = &sample_logits_into;
}

} // namespace brotensor::detail::cuda
