#pragma once

#include "trace_dag.h"
#include "trace_cache.h"
#include <memory>
#include <vector>
#include <functional>

namespace brotensor::jit {

enum class FusionPattern {
    Elementwise,
    ResidualRMSNorm,
    LayerNormModulate
};

class TraceCompiler {
public:
    static TraceHandle compile_and_cache(const TraceDAG& dag);
    static FusionPattern classify_dag(const TraceDAG& dag);
};

namespace cpu {
std::shared_ptr<TraceHandleImpl> compile_cpu(const TraceDAG& dag, FusionPattern pattern);
}

// Op-by-op replay through the dispatched ops, for a Vulkan trace the Vulkan
// compiler has no fusion for (or a build without it). Correct, not fused.
// See trace_eager.cpp.
namespace eager {
std::shared_ptr<TraceHandleImpl> compile_eager(const TraceDAG& dag);
}

using CudaTraceCompilerFn = std::shared_ptr<TraceHandleImpl> (*)(const TraceDAG& dag, FusionPattern pattern);

CudaTraceCompilerFn get_cuda_trace_compiler_hook();
void register_cuda_trace_compiler(CudaTraceCompilerFn fn);

// The Vulkan compiler (trace_compiler_vulkan.cpp, brass's SPIR-V target),
// registered by the Vulkan backend when it is built with brass. It returns
// nullptr for a DAG it has no fusion for, and the caller replays that DAG op
// by op instead.
using VulkanTraceCompilerFn = std::shared_ptr<TraceHandleImpl> (*)(const TraceDAG& dag, FusionPattern pattern);

VulkanTraceCompilerFn get_vulkan_trace_compiler_hook();
void register_vulkan_trace_compiler(VulkanTraceCompilerFn fn);

} // namespace brotensor::jit
