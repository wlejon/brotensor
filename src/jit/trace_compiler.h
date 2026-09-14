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

using CudaTraceCompilerFn = std::shared_ptr<TraceHandleImpl> (*)(const TraceDAG& dag, FusionPattern pattern);

CudaTraceCompilerFn get_cuda_trace_compiler_hook();
void register_cuda_trace_compiler(CudaTraceCompilerFn fn);

} // namespace brotensor::jit
