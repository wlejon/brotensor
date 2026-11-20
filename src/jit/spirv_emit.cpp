// Thin SPIR-V compilation wrapper over the shared Brass GPU MIR builders.

#include "spirv_emit.h"
#include "gpu_mir_emit_detail.h"

#include <string>

namespace brotensor::jit::spirv {

namespace {

brass::target::SpirvKernel compile_ew_entry(const TraceDAG& dag,
                                            const plan::ElementwisePlan& plan,
                                            int lanes,
                                            const char* name) {
    brass::Module mod(std::string("mod_") + name);
    brass::Function* fn = gpu_mir::build_elementwise_fn(mod, dag, plan, lanes, name, true);
    brass::target::SpirvOptions opts;
    opts.local_size_x = kEwBlock;
    return brass::target::SpirvTarget::compile(*fn, opts);
}

brass::target::SpirvKernel compile_rn_entry(const TraceDAG& dag,
                                            const plan::RowNormPlan& plan,
                                            int lanes,
                                            const char* name) {
    brass::Module mod(std::string("mod_") + name);
    brass::Function* fn = gpu_mir::build_row_norm_fn(mod, dag, plan, lanes, name, true);
    int tpr = 0, rpb = 0;
    plan::row_norm_geometry(plan.ew.cols, plan.ew.rows, lanes, tpr, rpb);
    brass::target::SpirvOptions opts;
    opts.local_size_x = static_cast<std::uint32_t>(tpr * rpb);
    return brass::target::SpirvTarget::compile(*fn, opts);
}

}  // namespace

bool ops_supported(const TraceDAG& dag) {
    return gpu_mir::ops_supported(dag);
}

Kernels emit_elementwise(const TraceDAG& dag, const plan::ElementwisePlan& plan) {
    Kernels k;
    if (plan.vec > 1) k.vec = compile_ew_entry(dag, plan, plan.vec, kEntryVec);
    k.scalar = compile_ew_entry(dag, plan, 1, kEntryScalar);
    return k;
}

Kernels emit_row_norm(const TraceDAG& dag, const plan::RowNormPlan& plan) {
    Kernels k;
    if (plan.ew.vec > 1) k.vec = compile_rn_entry(dag, plan, plan.ew.vec, kEntryRowNorm);
    k.scalar = compile_rn_entry(dag, plan, 1, kEntryRowNormScalar);
    return k;
}

}  // namespace brotensor::jit::spirv
