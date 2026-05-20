#pragma once

// Internal helpers shared across the Metal backend translation units.
// NOT installed; not part of the public API.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <brotensor/tensor.h>

// The custom-kernel surface — device(), queue(), new_command_buffer(),
// pool_lookup[_offset](), buffer_for(), buffer_offset_for(), set_tensor(),
// compile_pipeline() — is the public, installed half of this header. The
// declarations below are the internal-only remainder (pool mutation, MPSGraph
// helpers, the precompiled-pipeline cache, the cuda_* lifecycle hooks).
#include <brotensor/metal_interop.h>

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

// Pool mutation. The pool maps (data_ptr → MTLBuffer); device(), queue(),
// new_command_buffer(), pool_lookup[_offset]() and the buffer_for /
// set_tensor helpers are declared in <brotensor/metal_interop.h>. These two
// mutators stay internal — only tensor.mm's AllocVTable touches the pool.
void  pool_register(void* data_ptr, id<MTLBuffer> buf);
void  pool_release(void* data_ptr);

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
// kernels.mm at first cuda_init()). Throws if name is unknown. (The on-demand
// MSL-source compiler compile_pipeline() is in <brotensor/metal_interop.h>.)
id<MTLComputePipelineState> pipeline(NSString* name);

// Helper to dispatch an in-process MSL kernel against a 1-D thread grid
// covering n elements. Encodes onto a fresh command buffer and commits +
// waits. For best latency, several launches in a row should batch onto one
// command buffer manually rather than going through this helper.
void dispatch1d_sync(NSString* pipeline_name,
                     NSUInteger n,
                     void (^bind)(id<MTLComputeCommandEncoder>));

} // namespace brotensor::metal_impl
