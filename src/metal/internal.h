#pragma once

// Internal helpers shared across the Metal backend translation units.
// NOT installed; not part of the public API.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <brotensor/tensor.h>

#include <cstddef>

namespace brotensor {

// ─── Metal backend runtime (internal) ──────────────────────────────────────
//
// The public runtime surface (init / sync / DeviceScope) lives in
// <brotensor/runtime.h> and src/init.cpp. These are the Metal backend's own
// private lifecycle hooks, kept under the historical `cuda_*` names for
// source continuity with the op .mm files. Declared here (not in any public
// header) so the op TUs can call cuda_init() lazily.
void  cuda_init();
void  cuda_sync();
void  cuda_set_stream(void* stream);
void* cuda_current_stream();
void  cuda_stream_sync(void* stream);
void  cuda_check_throw(int err, const char* expr_text,
                       const char* file, int line);

} // namespace brotensor

namespace brotensor::metal_impl {

// Process-singleton accessors. Lazily initialized by cuda_init() (on Metal it
// initializes the MTLDevice + command queue + the precompiled MSL pipelines).
id<MTLDevice>       device();
id<MTLCommandQueue> queue();

// Returns a fresh MPSCommandBuffer wrapping a new MTLCommandBuffer drawn
// from queue(). Callers are responsible for committing.
id<MTLCommandBuffer> new_command_buffer();

// Pool of MTLBuffers keyed by (data_ptr → buffer). brotensor::Tensor stores
// only an opaque `void* data`; on Metal that pointer is the contents of an
// MTLBuffer allocated with MTLResourceStorageModeShared (unified memory). The
// pool owns the MTLBuffer reference for the lifetime of the Tensor.
//
// Internal — used by tensor.mm (the Metal AllocVTable) and the buffer_for /
// set_tensor helpers below.
void  pool_register(void* data_ptr, id<MTLBuffer> buf);
id<MTLBuffer> pool_lookup(const void* data_ptr);
NSUInteger pool_lookup_offset(const void* data_ptr);
void  pool_release(void* data_ptr);

// Convenience: get the MTLBuffer backing a Tensor's data pointer.
// Returns nil if the tensor is empty.
inline id<MTLBuffer> buffer_for(const ::brotensor::Tensor& t) {
    if (t.data == nullptr) return nil;
    return pool_lookup(t.data);
}
inline NSUInteger buffer_offset_for(const ::brotensor::Tensor& t) {
    if (t.data == nullptr) return 0;
    return pool_lookup_offset(t.data);
}

// Bind a Tensor (resolves base buffer + view offset) to a compute encoder
// at the given buffer index.
inline void set_tensor(id<MTLComputeCommandEncoder> enc,
                       const ::brotensor::Tensor& t,
                       NSUInteger index) {
    [enc setBuffer:buffer_for(t) offset:buffer_offset_for(t) atIndex:index];
}

// Run a single-output MPSGraph synchronously: feed the named placeholders
// from MTLBuffers, write the result to result_buffer. Synchronous — caller
// does not need a separate cuda_sync.
void run_graph_sync(MPSGraph* g,
                    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                    MPSGraphTensor* result,
                    id<MTLBuffer> result_buffer,
                    NSArray<NSNumber*>* result_shape);
void run_graph_sync_off(MPSGraph* g,
                        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                        MPSGraphTensor* result,
                        id<MTLBuffer> result_buffer,
                        NSUInteger result_offset_bytes,
                        NSArray<NSNumber*>* result_shape);

// MTLBuffer wrapper for an existing host-shared region (no copy). Used to
// build MPSGraphTensorData for tensors pulled from the pool.
MPSGraphTensorData* tensor_data_for(id<MTLBuffer> buf,
                                    NSArray<NSNumber*>* shape,
                                    MPSDataType dt = MPSDataTypeFloat32);
MPSGraphTensorData* tensor_data_for_offset(id<MTLBuffer> buf,
                                           NSUInteger offset_bytes,
                                           NSArray<NSNumber*>* shape,
                                           MPSDataType dt = MPSDataTypeFloat32);

// Return the cached compute pipeline for a named MSL kernel (precompiled in
// kernels.mm at first cuda_init()). Throws if name is unknown.
id<MTLComputePipelineState> pipeline(NSString* name);

// Compile an MSL source string and build a pipeline for a named function.
// Used by per-op-file lazy compile patterns: each op .mm holds its own MSL
// source plus a `static dispatch_once_t once + id<MTLComputePipelineState> pso;`
// — first call compiles, subsequent calls reuse.
//
// Throws std::runtime_error on compile or pipeline-build failure.
id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name);

// Helper to dispatch an in-process MSL kernel against a 1-D thread grid
// covering n elements. Encodes onto a fresh command buffer and commits +
// waits. For best latency, several launches in a row should batch onto one
// command buffer manually rather than going through this helper.
void dispatch1d_sync(NSString* pipeline_name,
                     NSUInteger n,
                     void (^bind)(id<MTLComputeCommandEncoder>));

} // namespace brotensor::metal_impl
