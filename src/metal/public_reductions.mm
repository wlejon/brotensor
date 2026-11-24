// Metal public reductions: sum_rows, sum_cols, argmax_rows via single-source
// brass MIR kernels (mtl_mir_cache.mm). Specialized non-Wave-2 op
// rows_count_above remains cleanly at the bottom of this file.

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

std::uint64_t dev_addr(const Tensor& t) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.data));
}

void require_fp(const Tensor& t, const char* op) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": X must be FP32, FP16, or BF16");
    }
}

// ─── Specialized non-Wave-2 Metal kernel (rows_count_above) ────────────────

constexpr NSUInteger kRedBlock = 256;

NSString* const kCountAboveSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint RED_BLOCK = 256;

#define ROWS_COUNT_ABOVE(NAME, T)                                             \
kernel void NAME(device const T* X     [[buffer(0)]],                         \
                 constant float& t_lo  [[buffer(1)]],                         \
                 constant float& t_hi  [[buffer(2)]],                         \
                 device int*     counts[[buffer(3)]],                         \
                 constant uint&  R     [[buffer(4)]],                         \
                 constant uint&  C     [[buffer(5)]],                         \
                 uint r   [[threadgroup_position_in_grid]],                   \
                 uint tid [[thread_position_in_threadgroup]],                 \
                 uint tgs [[threads_per_threadgroup]]) {                      \
    threadgroup int sm_lo[RED_BLOCK];                                         \
    threadgroup int sm_hi[RED_BLOCK];                                         \
    if (r >= R) return;                                                       \
    int n_lo = 0, n_hi = 0;                                                   \
    for (uint c = tid; c < C; c += tgs) {                                     \
        float v = float(X[r * C + c]);                                        \
        n_lo += (v > t_lo) ? 1 : 0;                                           \
        n_hi += (v > t_hi) ? 1 : 0;                                           \
    }                                                                         \
    sm_lo[tid] = n_lo;                                                        \
    sm_hi[tid] = n_hi;                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs / 2; s > 0; s >>= 1) {                                  \
        if (tid < s) {                                                        \
            sm_lo[tid] += sm_lo[tid + s];                                     \
            sm_hi[tid] += sm_hi[tid + s];                                     \
        }                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) {                                                           \
        counts[2 * r + 0] = sm_lo[0];                                         \
        counts[2 * r + 1] = sm_hi[0];                                         \
    }                                                                         \
}

ROWS_COUNT_ABOVE(k_rows_count_above_fp32, float)
ROWS_COUNT_ABOVE(k_rows_count_above_fp16, half)
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kCountAboveSrc, FN); });\
        return pso;                                                            \
    }
DEF_PSO(pso_rows_count_above_fp32, @"k_rows_count_above_fp32")
DEF_PSO(pso_rows_count_above_fp16, @"k_rows_count_above_fp16")
#undef DEF_PSO

}  // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    require_fp(X, "sum_rows");
    const int M = X.rows, N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) Y.resize(M, 1, X.dtype);
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X));
    const MirKernel mk = mir_kernel(
        {MirKind::SumRows, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(mk, std::min<std::uint32_t>(m, 65535u), X, Y, m, n);
}

void sum_cols(const Tensor& X, Tensor& Y) {
    require_fp(X, "sum_cols");
    const int M = X.rows, N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) Y.resize(1, N, X.dtype);
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X), dev_addr(Y));
    const MirKernel mk = mir_kernel(
        {MirKind::SumCols, 0, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_reduce(mk, mir_groups_1d(n, lanes), X, Y, m, n);
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    require_fp(X, "argmax_rows");
    const int M = X.rows, N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) Idx.resize(M, 1, out_dt);
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    const std::uint32_t m = static_cast<std::uint32_t>(M);
    const std::uint32_t n = static_cast<std::uint32_t>(N);
    const int lanes = mir_vec_lanes(X.dtype, n, dev_addr(X));
    const std::uint8_t flags = (out_dt == Dtype::INT32) ? 1u : 0u;
    const MirKernel mk = mir_kernel(
        {MirKind::ArgmaxRows, 0, X.dtype, out_dt,
         static_cast<std::uint8_t>(lanes), Dtype::FP32, flags});
    mir_dispatch_reduce(mk, std::min<std::uint32_t>(m, 65535u), X, Idx, m, n);
}

void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("rows_count_above: X must be FP32 or FP16");
    }
    const std::uint32_t R = static_cast<std::uint32_t>(X.rows);
    const std::uint32_t C = static_cast<std::uint32_t>(X.cols);
    if (counts.rows != static_cast<int>(R) || counts.cols != 2 ||
        counts.dtype != Dtype::INT32) {
        counts.resize(static_cast<int>(R), 2, Dtype::INT32);
    }
    if (R == 0) return;
    if (C == 0) { counts.zero(); return; }
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_rows_count_above_fp16()
                                 : pso_rows_count_above_fp32();
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bC = buffer_for(counts);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oC = buffer_offset_for(counts);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBytes:&t_lo length:sizeof(float) atIndex:1];
        [enc setBytes:&t_hi length:sizeof(float) atIndex:2];
        [enc setBuffer:bC offset:oC atIndex:3];
        [enc setBytes:&R length:sizeof(std::uint32_t) atIndex:4];
        [enc setBytes:&C length:sizeof(std::uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kRedBlock, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace brotensor::detail::metal
