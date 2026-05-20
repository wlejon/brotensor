#pragma once
//
// Public Metal custom-kernel surface.
//
// brotensor's Metal backend keeps most of its plumbing in the non-installed
// src/metal/internal.h. This header promotes the minimal subset a *consumer*
// needs to author its own Metal compute kernels against brotensor tensors:
// the shared MTLDevice, a command-buffer factory, the Tensor -> MTLBuffer
// mapping, and an MSL-source pipeline compiler.
//
// The CUDA backend needs no equivalent — a consumer's .cu file just includes
// <cuda_runtime.h> and uses Tensor::data as a raw device pointer. On Metal a
// Tensor's `data` is the `contents` pointer of an MTLBuffer drawn from a
// process-wide pool; a custom kernel must resolve that pointer back to its
// MTLBuffer (+ view offset) to bind it. buffer_for / buffer_offset_for do
// exactly that.
//
// Objective-C++ only — include from a .mm translation unit in a build with
// the Metal backend enabled (BROTENSOR_HAS_METAL). The definitions live in
// the brotensor_metal static library (tensor.mm / runtime.mm / kernels.mm),
// so a consumer that links `brotensor` picks them up with no extra wiring.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <brotensor/tensor.h>

#include <cstddef>

namespace brotensor::metal_impl {

// ─── Shared device + command submission ────────────────────────────────────
//
// Process-singleton accessors. The MTLDevice + command queue are brought up
// lazily by brotensor::init()'s Metal probe; they are valid for any consumer
// running after a Metal-backed tensor has been created.
id<MTLDevice>       device();
id<MTLCommandQueue> queue();

// A fresh MTLCommandBuffer drawn from queue(). The caller encodes, commits,
// and (if synchronous) waits.
id<MTLCommandBuffer> new_command_buffer();

// ─── Tensor -> MTLBuffer resolution ────────────────────────────────────────
//
// brotensor::Tensor stores only an opaque `void* data`. On Metal that is the
// contents pointer of a pool-owned MTLBuffer (MTLResourceStorageModeShared).
// pool_lookup maps it back to the base MTLBuffer; pool_lookup_offset gives
// the byte offset of a viewed sub-region within that base buffer.
id<MTLBuffer> pool_lookup(const void* data_ptr);
NSUInteger    pool_lookup_offset(const void* data_ptr);

// Convenience: the MTLBuffer backing a Tensor (nil if the tensor is empty)
// and the tensor's byte offset within it.
inline id<MTLBuffer> buffer_for(const ::brotensor::Tensor& t) {
    if (t.data == nullptr) return nil;
    return pool_lookup(t.data);
}
inline NSUInteger buffer_offset_for(const ::brotensor::Tensor& t) {
    if (t.data == nullptr) return 0;
    return pool_lookup_offset(t.data);
}

// Bind a Tensor (base buffer + view offset) to a compute encoder at `index`.
inline void set_tensor(id<MTLComputeCommandEncoder> enc,
                       const ::brotensor::Tensor& t,
                       NSUInteger index) {
    [enc setBuffer:buffer_for(t) offset:buffer_offset_for(t) atIndex:index];
}

// ─── MSL pipeline compilation ──────────────────────────────────────────────
//
// Compile an MSL source string and build a compute pipeline for one of its
// kernel functions. Throws std::runtime_error on compile or pipeline-build
// failure. Consumers typically cache the result behind a dispatch_once.
id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name);

} // namespace brotensor::metal_impl
