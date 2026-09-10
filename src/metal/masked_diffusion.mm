// ─── Metal masked-diffusion token selection ────────────────────────────────
//
// Metal counterpart of src/cpu/masked_diffusion.cpp — `masked_diffusion_
// scores` and `masked_diffusion_commit` (see ops/sampling.h). Written in the
// style of sample_logits.mm: ONE THREAD PER CELL (c, t), serial over the
// vocabulary row, with a per-call device scratch (rows x V FP32) holding the
// row's log-probs. The k-th largest log-prob for the class filter is found by
// bisecting the order-preserving 32-bit float key (32 counting passes over
// the row), exactly as the CUDA kernel does, so the kept set and its tie rule
// (lowest index among entries equal to the k-th value) match the CPU.
//
// Numerics: Metal has no FP64, so the log-sum-exp sums accumulate in FP32
// (CPU / CUDA use FP64) and the exp/log come from precise::. Results agree
// with the CPU to FP32 rounding; `pred` can differ from the CPU only on a
// near-tie. The hash RNG is a verbatim MSL copy of detail/hash_rng.h (MSL
// cannot include the header), so the noise itself is bit-identical.
//
// Not built or run on this port's development host (Windows) — mirrors the
// CPU code and the neighbouring Metal kernels' registration pattern.

#include <brotensor/runtime.h>

#include <cmath>
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

// Layout mirrored in the MSL source below (ulong first: 8-byte alignment).
struct MDParams {
    uint64_t seed;
    uint32_t T, C, V;
    int32_t  mask_id;
    float    gs, layer_penalty, pos_temp, class_temp;
    int32_t  k_keep;
    uint32_t pad;
};

struct CommitParams {
    uint32_t k;
    int32_t  step;
    uint32_t n;
    uint32_t pad;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct MDParams {
    ulong seed;
    uint  T, C, V;
    int   mask_id;
    float gs, layer_penalty, pos_temp, class_temp;
    int   k_keep;
    uint  pad;
};

struct CommitParams { uint k; int step; uint n; uint pad; };

// ── detail/hash_rng.h, verbatim ──
constant constexpr ulong kDomainPosition = 0x5851F42D4C957F2DUL;
constant constexpr ulong kDomainClass    = 0x14057B7EF767814FUL;

static inline ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}
static inline float hash_uniform(ulong seed, ulong domain, ulong index) {
    ulong h = splitmix64(seed ^ (domain + index));
    return float(h >> 40) * (1.0f / 16777216.0f);
}
static inline float gumbel_from_uniform(float u) {
    return -precise::log(-precise::log(u + 1e-10f) + 1e-10f);
}
static inline uint sortable_key(float x) {
    uint b = as_type<uint>(x);
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static inline bool prefers(float a, int ia, float b, int ib) {
    if (a != b) return a > b;
    return ia < ib;
}
static inline float row_lse(device const float* x, uint n) {
    float m = -INFINITY;
    for (uint i = 0; i < n; ++i) m = (x[i] > m) ? x[i] : m;
    float s = 0.0f;
    for (uint i = 0; i < n; ++i) s += precise::exp(x[i] - m);
    return m + precise::log(s);
}

kernel void k_masked_diffusion_scores(device const float* logits  [[buffer(0)]],
                                      device const int*   tokens  [[buffer(1)]],
                                      device float*       scratch [[buffer(2)]],
                                      device int*         pred    [[buffer(3)]],
                                      device float*       scores  [[buffer(4)]],
                                      device float*       confidence [[buffer(5)]],
                                      constant MDParams&  P       [[buffer(6)]],
                                      uint p [[thread_position_in_grid]]) {
    const uint T = P.T, C = P.C, V = P.V;
    if (p >= C * T) return;
    const uint c = p / T;
    const uint t = p - c * T;
    const ulong row_stride = ulong(C) * ulong(V);
    device const float* cond = logits + ulong(t) * row_stride + ulong(c) * ulong(V);
    device float* lp = scratch + ulong(p) * ulong(V);

    // ── log_probs ──
    if (P.gs != 0.0f) {
        device const float* unc = logits + ulong(T + t) * row_stride + ulong(c) * ulong(V);
        const float lse_c = row_lse(cond, V);
        const float lse_u = row_lse(unc, V);
        for (uint v = 0; v < V; ++v) {
            const float a = cond[v] - lse_c;
            const float b = unc[v] - lse_u;
            lp[v] = a + P.gs * (a - b);
        }
        const float lse_g = row_lse(lp, V);
        for (uint v = 0; v < V; ++v) lp[v] = lp[v] - lse_g;
    } else {
        const float lse = row_lse(cond, V);
        for (uint v = 0; v < V; ++v) lp[v] = cond[v] - lse;
    }
    lp[uint(P.mask_id)] = -INFINITY;

    // ── confidence: max of the UNfiltered log_probs ──
    float conf = -INFINITY;
    for (uint v = 0; v < V; ++v) conf = (lp[v] > conf) ? lp[v] : conf;
    confidence[p] = conf;   // raw, for every cell — masked or already decided

    // ── prediction ──
    int best = 0;
    if (P.class_temp > 0.0f) {
        const int k_keep = P.k_keep;
        uint cur = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const uint cand = cur | (1u << uint(bit));
            int cnt = 0;
            for (uint v = 0; v < V; ++v) cnt += (sortable_key(lp[v]) >= cand) ? 1 : 0;
            if (cnt >= k_keep) cur = cand;
        }
        const uint thr_key = cur;
        int cnt_gt = 0;
        for (uint v = 0; v < V; ++v) cnt_gt += (sortable_key(lp[v]) > thr_key) ? 1 : 0;
        const int m = k_keep - cnt_gt;

        const ulong base = ulong(p) * ulong(V);
        float best_s = -INFINITY;
        int seen_eq = 0;
        for (uint v = 0; v < V; ++v) {
            const uint key = sortable_key(lp[v]);
            bool kept = key > thr_key;
            if (!kept && key == thr_key) {
                kept = seen_eq < m;
                ++seen_eq;
            }
            float s = -INFINITY;
            if (kept) {
                const float u = hash_uniform(P.seed, kDomainClass, base + ulong(v));
                s = lp[v] / P.class_temp + gumbel_from_uniform(u);
            }
            if (prefers(s, int(v), best_s, best)) { best_s = s; best = int(v); }
        }
    } else {
        float best_v = -INFINITY;
        for (uint v = 0; v < V; ++v) {
            if (prefers(lp[v], int(v), best_v, best)) { best_v = lp[v]; best = int(v); }
        }
    }
    pred[p] = best;

    // ── score ──
    float s = -INFINITY;
    if (tokens[p] == P.mask_id) {
        s = conf - float(c) * P.layer_penalty;
        if (P.pos_temp > 0.0f) {
            const float u = hash_uniform(P.seed, kDomainPosition, ulong(p));
            s = s / P.pos_temp + gumbel_from_uniform(u);
        }
    }
    scores[p] = s;
}

kernel void k_masked_diffusion_commit(device const int* pred        [[buffer(0)]],
                                      device const int* idx         [[buffer(1)]],
                                      device int*       tokens      [[buffer(2)]],
                                      device int*       unmask_step [[buffer(3)]],
                                      constant CommitParams& P      [[buffer(4)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= P.k) return;
    const int p = idx[i];
    if (p < 0 || uint(p) >= P.n) return;
    tokens[p] = pred[p];
    unmask_step[p] = P.step;
}
)msl";

id<MTLComputePipelineState> pso_scores() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_masked_diffusion_scores"); });
    return pso;
}

id<MTLComputePipelineState> pso_commit() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_masked_diffusion_commit"); });
    return pso;
}

} // namespace

// ─── masked_diffusion_scores ────────────────────────────────────────────────

void masked_diffusion_scores(const Tensor& logits, const Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, uint64_t seed,
                             Tensor& pred, Tensor& scores, Tensor& confidence) {
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

    MDParams p{};
    p.seed = seed;
    p.T = static_cast<uint32_t>(T);
    p.C = static_cast<uint32_t>(C);
    p.V = static_cast<uint32_t>(V);
    p.mask_id = mask_id;
    p.gs = guidance_scale;
    p.layer_penalty = layer_penalty;
    p.pos_temp = position_temperature;
    p.class_temp = class_temperature;
    p.k_keep = k_keep;

    @autoreleasepool {
        id<MTLDevice> dev = ::brotensor::metal_impl::device();
        const NSUInteger nv = static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(V);
        id<MTLBuffer> scratch =
            [dev newBufferWithLength:nv * sizeof(float)
                             options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_scores();
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(logits) offset:buffer_offset_for(logits) atIndex:0];
        [enc setBuffer:buffer_for(tokens) offset:buffer_offset_for(tokens) atIndex:1];
        [enc setBuffer:scratch offset:0 atIndex:2];
        [enc setBuffer:buffer_for(pred)   offset:buffer_offset_for(pred)   atIndex:3];
        [enc setBuffer:buffer_for(scores) offset:buffer_offset_for(scores) atIndex:4];
        [enc setBuffer:buffer_for(confidence) offset:buffer_offset_for(confidence) atIndex:5];
        [enc setBytes:&p length:sizeof(MDParams) atIndex:6];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 64) tpt = 64;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// ─── masked_diffusion_commit ────────────────────────────────────────────────

void masked_diffusion_commit(const Tensor& pred, const Tensor& idx, int k, int step,
                             Tensor& tokens, Tensor& unmask_step) {
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

    CommitParams p{};
    p.k = static_cast<uint32_t>(k);
    p.step = step;
    p.n = static_cast<uint32_t>(pred.size());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_commit();
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(pred)        offset:buffer_offset_for(pred)        atIndex:0];
        [enc setBuffer:buffer_for(idx)         offset:buffer_offset_for(idx)         atIndex:1];
        [enc setBuffer:buffer_for(tokens)      offset:buffer_offset_for(tokens)      atIndex:2];
        [enc setBuffer:buffer_for(unmask_step) offset:buffer_offset_for(unmask_step) atIndex:3];
        [enc setBytes:&p length:sizeof(CommitParams) atIndex:4];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 64) tpt = 64;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(k), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
