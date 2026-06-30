// ─── Metal autoregressive logit sampling (CHUNK 7, family F) ────────────────
//
// Metal counterpart of src/cpu/sample_logits.cpp — the next-token sampler used
// by autoregressive generation loops (brosoundml codec-LM decoding and the
// brolm language-model project).
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
// ── Scratch ─────────────────────────────────────────────────────────────────
//   The per-row probability vector and its sorted index order do not fit in a
//   register file for arbitrary V, so two N*V FP32 buffers (prob + sort work)
//   and one N*V INT32 buffer (order) are allocated per call as shared scratch.

#include <brotensor/runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// Parameter block — must match the MSL struct below (ulong fields first so the
// 8-byte alignment is natural).
struct SampleParams {
    uint64_t key;
    uint64_t counter;
    float    temperature;
    int32_t  top_k;
    float    top_p;
    uint32_t N;
    uint32_t V;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct SampleParams {
    ulong key;
    ulong counter;
    float temperature;
    int   top_k;
    float top_p;
    uint  N;
    uint  V;
};

constant float kNeg = -3.4028235e38f;   // -FLT_MAX

// ── Philox 4x32-10 counter-based RNG (PyTorch / JAX compatible) ──────────────
// Byte-identical to the CPU reference in src/cpu/sample_logits.cpp.
inline void mulhilo32(uint a, uint b, thread uint& hi, thread uint& lo) {
    ulong product = (ulong)a * (ulong)b;
    hi = (uint)(product >> 32);
    lo = (uint)product;
}

inline void philox_round(thread uint ctr[4], thread const uint key[2]) {
    uint hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr[0], hi0, lo0);
    mulhilo32(0xCD9E8D57u, ctr[2], hi1, lo1);
    uint new0 = hi1 ^ ctr[1] ^ key[0];
    uint new1 = lo1;
    uint new2 = hi0 ^ ctr[3] ^ key[1];
    uint new3 = lo0;
    ctr[0] = new0; ctr[1] = new1; ctr[2] = new2; ctr[3] = new3;
}

// Draw one uniform in [0, 1) for `substream`, seeded by `key64`.
inline float philox_uniform(ulong key64, ulong substream) {
    uint key[2] = { (uint)(key64 & 0xFFFFFFFFul), (uint)(key64 >> 32) };
    uint ctr[4] = { (uint)(substream & 0xFFFFFFFFul),
                    (uint)(substream >> 32), 0u, 0u };
    for (int round = 0; round < 10; ++round) {
        philox_round(ctr, key);
        if (round < 9) {
            key[0] += 0x9E3779B9u;     // golden ratio
            key[1] += 0xBB67AE85u;     // sqrt(3) - 1
        }
    }
    return (float)(ctr[0] >> 8) / 16777216.0f;   // top 24 bits
}

// One thread per row of the (N, V) logit matrix.
kernel void k_sample_logits(device const float* logits   [[buffer(0)]],
                            device int*         indices  [[buffer(1)]],
                            device float*       prob     [[buffer(2)]],
                            device float*       work     [[buffer(3)]],
                            device int*         order    [[buffer(4)]],
                            constant SampleParams& P      [[buffer(5)]],
                            uint n [[thread_position_in_grid]]) {
    if (n >= P.N) return;
    uint V = P.V;
    device const float* row = logits + (ulong)n * V;

    // ── Greedy: temperature == 0 -> deterministic argmax, no RNG. ──
    if (P.temperature == 0.0f) {
        float best_v = kNeg;
        int   best_i = 0;
        for (uint v = 0u; v < V; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best_i = int(v); }
        }
        indices[n] = best_i;
        return;
    }

    device float* prob_n = prob + (ulong)n * V;
    device float* work_n = work + (ulong)n * V;
    device int*   ord_n  = order + (ulong)n * V;

    // ── 1. temperature scale + 2. softmax (numerically stable). ──
    float max_logit = kNeg;
    for (uint v = 0u; v < V; ++v) {
        float s = row[v] / P.temperature;
        if (s > max_logit) max_logit = s;
    }
    float sum = 0.0f;
    for (uint v = 0u; v < V; ++v) {
        float s = row[v] / P.temperature;
        float e = precise::exp(s - max_logit);
        prob_n[v] = e;
        sum += e;
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (uint v = 0u; v < V; ++v) {
        prob_n[v] *= inv_sum;
        work_n[v]  = prob_n[v];     // sort scratch
    }

    // Descending-probability order; ties broken by lower index. Selection sort
    // scanning ascending with strict `>` reproduces std::stable_sort's order.
    for (uint r = 0u; r < V; ++r) {
        float best = kNeg;
        int   best_i = 0;
        for (uint v = 0u; v < V; ++v) {
            if (work_n[v] > best) { best = work_n[v]; best_i = int(v); }
        }
        ord_n[r] = best_i;
        work_n[best_i] = kNeg;      // remove from the running set
    }

    // ── 3. top-k filter: keep the top_k highest-probability tokens. ──
    int keep = int(V);
    if (P.top_k > 0 && P.top_k < keep) keep = P.top_k;

    // ── 4. top-p (nucleus): smallest high-prob set with cumprob >= top_p. ──
    if (P.top_p < 1.0f) {
        float cum = 0.0f;
        int nucleus = 0;
        for (int r = 0; r < keep; ++r) {
            cum += prob_n[ord_n[r]];
            ++nucleus;
            if (cum >= P.top_p) break;
        }
        if (nucleus < 1) nucleus = 1;
        keep = nucleus;
    }

    // ── 5. renormalise over the kept set. ──
    float kept_sum = 0.0f;
    for (int r = 0; r < keep; ++r) kept_sum += prob_n[ord_n[r]];

    // ── 6. inverse-CDF draw with a Philox uniform for substream (counter+n). ──
    float u = philox_uniform(P.key, P.counter + (ulong)n);
    int chosen = ord_n[0];
    if (kept_sum > 0.0f) {
        float target = u * kept_sum;
        float acc = 0.0f;
        chosen = ord_n[keep - 1];          // fallback: last kept (covers u~1).
        for (int r = 0; r < keep; ++r) {
            acc += prob_n[ord_n[r]];
            if (target < acc) { chosen = ord_n[r]; break; }
        }
    }
    indices[n] = chosen;
}

// ── Graph-capturable variant ────────────────────────────────────────────────
// Philox base counter read from a device buffer (counter[0]); scratch is
// caller-owned. The counter is advanced by a separate single-thread kernel so
// no row races the write. Mirrors the CUDA sample_logits_into path.
struct SampleParamsG {
    ulong key;
    float temperature;
    int   top_k;
    float top_p;
    uint  N;
    uint  V;
};

kernel void k_sample_logits_into(device const float* logits  [[buffer(0)]],
                                 device int*         indices [[buffer(1)]],
                                 device float*       prob    [[buffer(2)]],
                                 device float*       work    [[buffer(3)]],
                                 device int*         order   [[buffer(4)]],
                                 device const int*   counter [[buffer(5)]],
                                 constant SampleParamsG& P    [[buffer(6)]],
                                 uint n [[thread_position_in_grid]]) {
    if (n >= P.N) return;
    ulong base = (ulong)(uint)counter[0];
    uint V = P.V;
    device const float* row = logits + (ulong)n * V;

    if (P.temperature == 0.0f) {
        float best_v = kNeg;
        int   best_i = 0;
        for (uint v = 0u; v < V; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best_i = int(v); }
        }
        indices[n] = best_i;
        return;
    }

    device float* prob_n = prob + (ulong)n * V;
    device float* work_n = work + (ulong)n * V;
    device int*   ord_n  = order + (ulong)n * V;

    float max_logit = kNeg;
    for (uint v = 0u; v < V; ++v) {
        float s = row[v] / P.temperature;
        if (s > max_logit) max_logit = s;
    }
    float sum = 0.0f;
    for (uint v = 0u; v < V; ++v) {
        float s = row[v] / P.temperature;
        float e = precise::exp(s - max_logit);
        prob_n[v] = e;
        sum += e;
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (uint v = 0u; v < V; ++v) {
        prob_n[v] *= inv_sum;
        work_n[v]  = prob_n[v];
    }

    for (uint r = 0u; r < V; ++r) {
        float best = kNeg;
        int   best_i = 0;
        for (uint v = 0u; v < V; ++v) {
            if (work_n[v] > best) { best = work_n[v]; best_i = int(v); }
        }
        ord_n[r] = best_i;
        work_n[best_i] = kNeg;
    }

    int keep = int(V);
    if (P.top_k > 0 && P.top_k < keep) keep = P.top_k;

    if (P.top_p < 1.0f) {
        float cum = 0.0f;
        int nucleus = 0;
        for (int r = 0; r < keep; ++r) {
            cum += prob_n[ord_n[r]];
            ++nucleus;
            if (cum >= P.top_p) break;
        }
        if (nucleus < 1) nucleus = 1;
        keep = nucleus;
    }

    float kept_sum = 0.0f;
    for (int r = 0; r < keep; ++r) kept_sum += prob_n[ord_n[r]];

    float u = philox_uniform(P.key, base + (ulong)n);
    int chosen = ord_n[0];
    if (kept_sum > 0.0f) {
        float target = u * kept_sum;
        float acc = 0.0f;
        chosen = ord_n[keep - 1];
        for (int r = 0; r < keep; ++r) {
            acc += prob_n[ord_n[r]];
            if (target < acc) { chosen = ord_n[r]; break; }
        }
    }
    indices[n] = chosen;
}

kernel void k_advance_counter(device int* counter [[buffer(0)]],
                              constant uint& N     [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    counter[0] = int((uint)counter[0] + N);
}
)msl";

id<MTLComputePipelineState> pso_sample_logits() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_sample_logits"); });
    return pso;
}

id<MTLComputePipelineState> pso_sample_logits_into() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once,
                  ^{ pso = compile_pipeline(kSrc, @"k_sample_logits_into"); });
    return pso;
}

id<MTLComputePipelineState> pso_advance_counter() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_advance_counter"); });
    return pso;
}

} // namespace

// ─── sample_logits ───────────────────────────────────────────────────────────
void sample_logits(const Tensor& logits, float temperature, int top_k,
                   float top_p, uint64_t key, uint64_t counter,
                   Tensor& indices) {
    const char* op = "sample_logits";
    if (logits.dtype != Dtype::FP32) {
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
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != Dtype::INT32) {
        indices.resize(N, 1, Dtype::INT32);
    }
    if (N == 0) return;

    SampleParams p{};
    p.key = key;
    p.counter = counter;
    p.temperature = temperature;
    p.top_k = top_k;
    p.top_p = top_p;
    p.N = static_cast<uint32_t>(N);
    p.V = static_cast<uint32_t>(V);

    @autoreleasepool {
        id<MTLDevice> dev = ::brotensor::metal_impl::device();
        const NSUInteger nv = static_cast<NSUInteger>(N) *
                              static_cast<NSUInteger>(V);
        id<MTLBuffer> probBuf =
            [dev newBufferWithLength:nv * sizeof(float)
                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> workBuf =
            [dev newBufferWithLength:nv * sizeof(float)
                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> orderBuf =
            [dev newBufferWithLength:nv * sizeof(int32_t)
                             options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_sample_logits()];
        [enc setBuffer:buffer_for(logits)
                offset:buffer_offset_for(logits) atIndex:0];
        [enc setBuffer:buffer_for(indices)
                offset:buffer_offset_for(indices) atIndex:1];
        [enc setBuffer:probBuf  offset:0 atIndex:2];
        [enc setBuffer:workBuf  offset:0 atIndex:3];
        [enc setBuffer:orderBuf offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(SampleParams) atIndex:5];
        NSUInteger tpt = [pso_sample_logits() maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// ─── sample_logits_into ──────────────────────────────────────────────────────
void sample_logits_into(const Tensor& logits, float temperature, int top_k,
                        float top_p, uint64_t key, Tensor& counter,
                        Tensor& scratch, Tensor& indices) {
    const char* op = "sample_logits_into";
    if (logits.dtype != Dtype::FP32) fail(op, "logits must be FP32");
    if (temperature < 0.0f) fail(op, "temperature must be >= 0");
    if (top_k < 0)          fail(op, "top_k must be >= 0");
    if (top_p < 0.0f)       fail(op, "top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) fail(op, "vocabulary size (logits.cols) must be > 0");

    if (counter.dtype != Dtype::INT32 ||
        static_cast<size_t>(counter.rows) * counter.cols < 1) {
        fail(op, "counter must be an INT32 tensor with >= 1 element");
    }
    const size_t nv = static_cast<size_t>(N) * static_cast<size_t>(V);
    if (scratch.dtype != Dtype::FP32 ||
        static_cast<size_t>(scratch.rows) * scratch.cols < 3 * nv) {
        fail(op, "scratch must be FP32 with at least 3*N*V elements");
    }
    if (indices.rows != N || indices.cols != 1 || indices.dtype != Dtype::INT32) {
        fail(op, "indices must be a pre-sized (N,1) INT32 tensor");
    }
    if (N == 0) return;

    SampleParamsG p{};
    p.key = key;
    p.temperature = temperature;
    p.top_k = top_k;
    p.top_p = top_p;
    p.N = static_cast<uint32_t>(N);
    p.V = static_cast<uint32_t>(V);

    @autoreleasepool {
        // The caller's scratch buffer carries prob | sort-work | order, bound at
        // 0 / nv / 2*nv floats (INT32 and FP32 are both 4 bytes).
        id<MTLBuffer> scratchBuf = buffer_for(scratch);
        const NSUInteger sbase = buffer_offset_for(scratch);

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_sample_logits_into()];
        [enc setBuffer:buffer_for(logits)
                offset:buffer_offset_for(logits) atIndex:0];
        [enc setBuffer:buffer_for(indices)
                offset:buffer_offset_for(indices) atIndex:1];
        [enc setBuffer:scratchBuf offset:sbase                          atIndex:2];
        [enc setBuffer:scratchBuf offset:sbase + nv * sizeof(float)     atIndex:3];
        [enc setBuffer:scratchBuf offset:sbase + 2 * nv * sizeof(float) atIndex:4];
        [enc setBuffer:buffer_for(counter)
                offset:buffer_offset_for(counter) atIndex:5];
        [enc setBytes:&p length:sizeof(SampleParamsG) atIndex:6];
        NSUInteger tpt = [pso_sample_logits_into() maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];

        // Greedy consumes no RNG; only advance the stream when sampling.
        if (temperature != 0.0f) {
            uint32_t nN = static_cast<uint32_t>(N);
            id<MTLComputeCommandEncoder> aenc = [cmd computeCommandEncoder];
            [aenc setComputePipelineState:pso_advance_counter()];
            [aenc setBuffer:buffer_for(counter)
                     offset:buffer_offset_for(counter) atIndex:0];
            [aenc setBytes:&nN length:sizeof(uint32_t) atIndex:1];
            [aenc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [aenc endEncoding];
        }
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
