// ─── Metal counter-based noise generation ───────────────────────────────────
//
// Metal counterparts of randn / rand_uniform / rand_bernoulli /
// randn_truncated. One thread per output element; substream (counter + i) per
// element. Philox 4x32-10 construction byte-identical to the CPU / CUDA
// references, so a given (key, counter) yields the same draws on every
// backend.
//
// All four ops require Y FP32 and pre-sized; the op fills rows*cols elements
// in row-major linear order. See ops.h for the full ABI contract.

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

struct NoiseParams {
    uint64_t key;
    uint64_t counter;
    uint64_t n;
    float    a;       // bernoulli: p; trunc_normal: lo; else unused
    float    b;       // trunc_normal: hi; else unused
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct NoiseParams {
    ulong key;
    ulong counter;
    ulong n;
    float a;
    float b;
};

inline void mulhilo32(uint a, uint b, thread uint& hi, thread uint& lo) {
    ulong p = (ulong)a * (ulong)b;
    hi = (uint)(p >> 32);
    lo = (uint)p;
}

inline void philox_round(thread uint ctr[4], thread const uint key[2]) {
    uint hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr[0], hi0, lo0);
    mulhilo32(0xCD9E8D57u, ctr[2], hi1, lo1);
    uint n0 = hi1 ^ ctr[1] ^ key[0];
    uint n1 = lo1;
    uint n2 = hi0 ^ ctr[3] ^ key[1];
    uint n3 = lo0;
    ctr[0] = n0; ctr[1] = n1; ctr[2] = n2; ctr[3] = n3;
}

inline void philox4x32(ulong key64, ulong substream, thread uint out[4]) {
    uint key[2] = { (uint)(key64 & 0xFFFFFFFFul), (uint)(key64 >> 32) };
    uint ctr[4] = { (uint)(substream & 0xFFFFFFFFul),
                    (uint)(substream >> 32), 0u, 0u };
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, key);
        if (r < 9) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

inline float u01_from(uint w) {
    return (float)(w >> 8) / 16777216.0f;
}

inline float philox_uniform(ulong key64, ulong substream) {
    uint ctr[4];
    philox4x32(key64, substream, ctr);
    return u01_from(ctr[0]);
}

inline float philox_normal(ulong key64, ulong substream) {
    uint ctr[4];
    philox4x32(key64, substream, ctr);
    float u1 = 1.0f - u01_from(ctr[0]);
    float u2 = u01_from(ctr[1]);
    // 2π in double, downcast to float — matches CPU.
    const float kTwoPi = (float)(2.0 * 3.14159265358979323846);
    float radius = precise::sqrt(-2.0f * precise::log(u1));
    float theta  = kTwoPi * u2;
    return radius * precise::cos(theta);
}

kernel void k_randn(device float* y       [[buffer(0)]],
                    constant NoiseParams& P [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]]) {
    if ((ulong)gid >= P.n) return;
    y[gid] = philox_normal(P.key, P.counter + (ulong)gid);
}

kernel void k_rand_uniform(device float* y       [[buffer(0)]],
                           constant NoiseParams& P [[buffer(1)]],
                           uint gid               [[thread_position_in_grid]]) {
    if ((ulong)gid >= P.n) return;
    y[gid] = philox_uniform(P.key, P.counter + (ulong)gid);
}

kernel void k_rand_bernoulli(device float* y       [[buffer(0)]],
                             constant NoiseParams& P [[buffer(1)]],
                             uint gid               [[thread_position_in_grid]]) {
    if ((ulong)gid >= P.n) return;
    float u = philox_uniform(P.key, P.counter + (ulong)gid);
    y[gid] = (u < P.a) ? 1.0f : 0.0f;
}

kernel void k_randn_truncated(device float* y       [[buffer(0)]],
                              constant NoiseParams& P [[buffer(1)]],
                              uint gid               [[thread_position_in_grid]]) {
    if ((ulong)gid >= P.n) return;
    float lo = P.a;
    float hi = P.b;
    float z = 0.0f;
    for (int r = 0; r < 64; ++r) {
        ulong sub = P.counter + (ulong)gid + (ulong)r * P.n;
        z = philox_normal(P.key, sub);
        if (z >= lo && z <= hi) break;
    }
    if (z < lo) z = lo;
    if (z > hi) z = hi;
    y[gid] = z;
}
)msl";

id<MTLComputePipelineState> pso_randn() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_randn"); });
    return pso;
}
id<MTLComputePipelineState> pso_rand_uniform() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_rand_uniform"); });
    return pso;
}
id<MTLComputePipelineState> pso_rand_bernoulli() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_rand_bernoulli"); });
    return pso;
}
id<MTLComputePipelineState> pso_randn_truncated() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_randn_truncated"); });
    return pso;
}

inline std::size_t check_y(const char* op, const Tensor& Y) {
    if (Y.dtype != Dtype::FP32) fail(op, "Y must be FP32");
    if (Y.rows < 0 || Y.cols < 0) fail(op, "Y has negative dimension");
    const std::size_t n = static_cast<std::size_t>(Y.rows) *
                          static_cast<std::size_t>(Y.cols);
    if (n != 0 && Y.data == nullptr) {
        fail(op, "Y is uncommitted; pre-allocate before calling");
    }
    return n;
}

void dispatch_fill(id<MTLComputePipelineState> pso, Tensor& Y,
                   const NoiseParams& P, std::size_t n) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:0];
        [enc setBytes:&P length:sizeof(NoiseParams) atIndex:1];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

// ─── randn ──────────────────────────────────────────────────────────────────
void randn(uint64_t key, uint64_t counter, Tensor& Y) {
    const std::size_t n = check_y("randn", Y);
    if (n == 0) return;
    NoiseParams P{ key, counter, static_cast<uint64_t>(n), 0.0f, 0.0f };
    dispatch_fill(pso_randn(), Y, P, n);
}

// ─── rand_uniform ───────────────────────────────────────────────────────────
void rand_uniform(uint64_t key, uint64_t counter, Tensor& Y) {
    const std::size_t n = check_y("rand_uniform", Y);
    if (n == 0) return;
    NoiseParams P{ key, counter, static_cast<uint64_t>(n), 0.0f, 0.0f };
    dispatch_fill(pso_rand_uniform(), Y, P, n);
}

// ─── rand_bernoulli ─────────────────────────────────────────────────────────
void rand_bernoulli(float p, uint64_t key, uint64_t counter, Tensor& Y) {
    const char* op = "rand_bernoulli";
    if (!(p >= 0.0f && p <= 1.0f)) fail(op, "p must be in [0, 1]");
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    NoiseParams P{ key, counter, static_cast<uint64_t>(n), p, 0.0f };
    dispatch_fill(pso_rand_bernoulli(), Y, P, n);
}

// ─── randn_truncated ────────────────────────────────────────────────────────
void randn_truncated(float lo, float hi, uint64_t key, uint64_t counter,
                     Tensor& Y) {
    const char* op = "randn_truncated";
    if (!(lo < hi)) fail(op, "lo must be < hi");
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    NoiseParams P{ key, counter, static_cast<uint64_t>(n), lo, hi };
    dispatch_fill(pso_randn_truncated(), Y, P, n);
}

} // namespace brotensor::detail::metal
