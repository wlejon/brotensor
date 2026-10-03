#pragma once

// Run-time compiled kernels on the Vulkan backend: a brass
// `target::SpirvKernel` (MIR lowered by brass's SPIR-V target) turned into a
// pipeline brotensor's stream can dispatch. Used by the trace JIT
// (src/jit/trace_compiler_vulkan.cpp).
//
// brass's kernel ABI is the backend's own (docs/vulkan.md, "Dispatch model";
// brass docs/spirv_backend_design.md, "Runtime contract"), so nothing new is
// needed on the device side:
//
//   * every `ptr` parameter is a u64 buffer device address in the one
//     push-constant block, at SpirvKernel::params[i].offset — and a Vulkan
//     tensor's `data` already IS its device address (allocator.h: every
//     block is a SHADER_DEVICE_ADDRESS buffer on DEVICE_ADDRESS memory), so
//     an interior pointer (a row view) is passed as is, with no lookup;
//   * no descriptor sets: the pipeline uses Pipelines' shared push-constant-
//     only layout, so a JIT dispatch binds and pushes like any built-in one;
//   * the workgroup size is specialisation constants 0..2;
//   * subgroup size 32 is required when the device can require it (the
//     kernels' shuffles are written for 32-lane segments; they are exact at
//     64 too, so a device that cannot require 32 still runs them).
//
// The SPIR-V is interned by content: compiling the same trace twice (a cache
// miss after TraceCache::clear(), or two processes' worth of the same shape)
// yields the same words, the same registry key and therefore the pipeline
// already in Pipelines' cache. The capability check — SpirvKernel::
// capabilities against the features the device was created with — runs once
// per kernel and device, and throws listing everything that is missing.

#include "detail/device.h"
#include "detail/pipelines.h"

#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <string>

namespace brotensor::detail::vulkan {

struct JitPipeline {
    VkPipeline pipe = VK_NULL_HANDLE;
    std::uint32_t block = 0;          // workgroup x size the pipeline was specialised for
    std::uint32_t push_bytes = 0;     // SpirvKernel::push_constant_bytes
    std::uint32_t subgroup = 0;       // required subgroup size, 0 = the driver's choice
};

// Empty when `ctx`'s device can run `k`; otherwise one line per missing
// feature, naming the SPIR-V capability that needs it.
std::string jit_missing_for(DeviceCtx& ctx, const brass::target::SpirvKernel& k);

// The pipeline for `k` at a (block_x, 1, 1) workgroup on `ctx`, created on
// first use. Throws std::runtime_error when the device lacks a capability the
// kernel declares, the push block exceeds the layout, or the driver rejects it.
JitPipeline jit_pipeline(DeviceCtx& ctx, const brass::target::SpirvKernel& k, std::uint32_t block_x);

// Records one dispatch on `ctx`'s stream: ordered after everything recorded
// before it (the stream's compute->compute barrier), and captured into a
// graph like any other op while a capture is recording.
inline void jit_dispatch(DeviceCtx& ctx, const JitPipeline& p, const void* push,
                         std::uint32_t gx, std::uint32_t gy = 1, std::uint32_t gz = 1) {
    ctx.stream().dispatch(p.pipe, push, p.push_bytes, gx, gy, gz);
}

// Number of distinct SPIR-V modules interned so far (tests).
std::size_t jit_module_count();

}  // namespace brotensor::detail::vulkan
