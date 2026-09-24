// Metal port of src/cpu/ops_impl.cpp::xavier_init (and src/cuda/xavier_init.cu).
//
// The CPU walks splitmix64 sequentially, one step per element, and a step
// only adds a constant to the state before mixing, so element i's state is
// rng_state + (i+1)*K independently of the others: one thread per element,
// then the host advances rng_state by n*K. Same values and final state as the
// CPU, bit for bit.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr uint64_t kSplitmixK = 0x9E3779B97F4A7C15ULL;

NSString* const kXavierSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_xavier_init(device float*   W     [[buffer(0)]],
                          constant uint&  n     [[buffer(1)]],
                          constant float& limit [[buffer(2)]],
                          constant ulong& base  [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    ulong z = base + ulong(i + 1) * 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    const float u = float(z >> 40) / 16777216.0f;   // top 24 bits -> [0, 1)
    W[i] = (u * 2.0f - 1.0f) * limit;
}
)msl";

id<MTLComputePipelineState> pso_xavier() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kXavierSrc, @"k_xavier_init"); });
    return pso;
}

}  // namespace

void xavier_init(Tensor& W, uint64_t& rng_state) {
    if (W.dtype != Dtype::FP32) throw std::runtime_error("brotensor: xavier_init: W must be FP32");
    const int n = W.size();
    if (n <= 0) return;
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    const uint64_t base = rng_state;
    const uint32_t nu = static_cast<uint32_t>(n);
    id<MTLComputePipelineState> pso = pso_xavier();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(W) offset:buffer_offset_for(W) atIndex:0];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&limit length:sizeof(float) atIndex:2];
        [enc setBytes:&base length:sizeof(uint64_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nu, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
    rng_state = base + static_cast<uint64_t>(n) * kSplitmixK;
}

}  // namespace brotensor::detail::metal
