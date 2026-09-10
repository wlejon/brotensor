// ─── Metal per-row top-k ────────────────────────────────────────────────────
//
// Metal counterpart of src/cpu/top_k.cpp. FP32 input -> FP32 Vals + INT32 Idx.
// One thread per row. k is small in practice (5, 100, ...) so each thread
// keeps a per-row working set of size k in thread-private memory using the
// same streaming-replacement strategy as the CPU. Tie-break: smaller column
// index wins — matches CPU bit-exactly.
//
// The streaming kernel's stack-allocated working set is bounded at kMaxK
// (=256); k beyond that (the masked-diffusion host loop asks for k up to the
// full row length) goes through k_top_k_rank, a one-thread-per-element rank
// selection with the same order and tie rule, so there is no cap on k.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr int kMaxK = 256;

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

struct TopKParams {
    uint32_t R, C, K;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct TopKParams { uint R, C, K; };

constant constexpr uint kMaxK = 256;

// (val_a, idx_a) is preferred over (val_b, idx_b)?
static inline bool prefers(float a, int ia, float b, int ib) {
    if (a != b) return a > b;
    return ia < ib;
}

kernel void k_top_k_rows(device const float* X    [[buffer(0)]],
                         device float*       Vals [[buffer(1)]],
                         device int*         Idx  [[buffer(2)]],
                         constant TopKParams& P   [[buffer(3)]],
                         uint r [[thread_position_in_grid]]) {
    if (r >= P.R) return;
    uint K = P.K;
    uint C = P.C;
    float vbuf[kMaxK];
    int   ibuf[kMaxK];

    // Seed with first K columns.
    for (uint j = 0; j < K; ++j) {
        vbuf[j] = X[r * C + j];
        ibuf[j] = int(j);
    }
    uint weakest = 0;
    for (uint j = 1; j < K; ++j) {
        if (prefers(vbuf[weakest], ibuf[weakest], vbuf[j], ibuf[j])) {
            weakest = j;
        }
    }
    // Scan remainder.
    for (uint c = K; c < C; ++c) {
        float v = X[r * C + c];
        if (prefers(v, int(c), vbuf[weakest], ibuf[weakest])) {
            vbuf[weakest] = v;
            ibuf[weakest] = int(c);
            weakest = 0;
            for (uint j = 1; j < K; ++j) {
                if (prefers(vbuf[weakest], ibuf[weakest],
                            vbuf[j], ibuf[j])) {
                    weakest = j;
                }
            }
        }
    }
    // Insertion-sort into descending order (smaller index wins tie).
    for (uint i = 1; i < K; ++i) {
        float v = vbuf[i];
        int   id = ibuf[i];
        uint j = i;
        while (j > 0 && prefers(v, id, vbuf[j - 1], ibuf[j - 1])) {
            vbuf[j] = vbuf[j - 1];
            ibuf[j] = ibuf[j - 1];
            --j;
        }
        vbuf[j] = v;
        ibuf[j] = id;
    }
    // Write out.
    for (uint j = 0; j < K; ++j) {
        Vals[r * K + j] = vbuf[j];
        Idx [r * K + j] = ibuf[j];
    }
}

// Large k (> kMaxK): rank selection. `prefers` is a strict total order over
// the row, so rank(c) = #{ w : (X[w], w) precedes (X[c], c) } is a
// permutation of [0, C) and every element with rank < K writes itself into
// output slot `rank`. One thread per (column, row); O(C) per thread. Same
// order and tie rule as the streaming kernel, so the output is identical.
kernel void k_top_k_rank(device const float* X    [[buffer(0)]],
                         device float*       Vals [[buffer(1)]],
                         device int*         Idx  [[buffer(2)]],
                         constant TopKParams& P   [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, r = gid.y;
    if (c >= P.C || r >= P.R) return;
    device const float* row = X + r * P.C;
    const float v = row[c];
    uint rank = 0;
    for (uint w = 0; w < P.C; ++w) {
        rank += prefers(row[w], int(w), v, int(c)) ? 1u : 0u;
    }
    if (rank < P.K) {
        Vals[r * P.K + rank] = v;
        Idx [r * P.K + rank] = int(c);
    }
}
)msl";

id<MTLComputePipelineState> pso_top_k() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_top_k_rows"); });
    return pso;
}

id<MTLComputePipelineState> pso_top_k_rank() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_top_k_rank"); });
    return pso;
}

} // namespace

void top_k_rows(const Tensor& X, int k,
                Tensor& Vals, Tensor& Idx) {
    const char* op = "top_k_rows";
    req_fp32(op, X, "X");
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");
    if (Vals.rows != R || Vals.cols != k || Vals.dtype != Dtype::FP32) {
        Vals.resize(R, k, Dtype::FP32);
    }
    if (Idx.rows != R || Idx.cols != k || Idx.dtype != Dtype::INT32) {
        Idx.resize(R, k, Dtype::INT32);
    }
    if (R == 0 || k == 0) return;

    TopKParams p{
        static_cast<uint32_t>(R),
        static_cast<uint32_t>(C),
        static_cast<uint32_t>(k),
    };

    // k beyond the per-thread working-set cap: rank selection (one thread
    // per element, see k_top_k_rank) instead of the streaming kernel.
    if (k > kMaxK) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            id<MTLComputePipelineState> pso = pso_top_k_rank();
            [enc setComputePipelineState:pso];
            [enc setBuffer:buffer_for(X)    offset:buffer_offset_for(X)    atIndex:0];
            [enc setBuffer:buffer_for(Vals) offset:buffer_offset_for(Vals) atIndex:1];
            [enc setBuffer:buffer_for(Idx)  offset:buffer_offset_for(Idx)  atIndex:2];
            [enc setBytes:&p length:sizeof(TopKParams) atIndex:3];
            NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
            if (tpt > 64) tpt = 64;
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(C),
                                             static_cast<NSUInteger>(R), 1)
                threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_top_k();
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X)    offset:buffer_offset_for(X)    atIndex:0];
        [enc setBuffer:buffer_for(Vals) offset:buffer_offset_for(Vals) atIndex:1];
        [enc setBuffer:buffer_for(Idx)  offset:buffer_offset_for(Idx)  atIndex:2];
        [enc setBytes:&p length:sizeof(TopKParams) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 64) tpt = 64;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(R), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
