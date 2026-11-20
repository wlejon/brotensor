// Thin PTX compilation wrapper over the shared Brass GPU MIR builders.

#include "ptx_emit.h"
#include "gpu_mir_emit_detail.h"

#include <brass/target/ptx_target.hpp>

namespace brotensor::jit::ptx {

namespace {

brass::target::PtxOptions ptx_opts_for(const std::string& arch) {
    brass::target::PtxOptions opts;
    opts.sm_arch = arch.empty() ? "sm_80" : arch;
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 8;
    opts.cleanup = true;
    return opts;
}

}  // namespace

std::string emit_elementwise(const TraceDAG& dag, const ElementwisePlan& plan,
                             const std::string& arch) {
    brass::Module mod("brotensor_trace_ew_ptx");
    gpu_mir::build_elementwise_fn(mod, dag, plan, plan.vec, kEntryVec, true);
    gpu_mir::build_elementwise_fn(mod, dag, plan, 1, kEntryScalar, true);
    return brass::target::PtxTarget::emit_module(mod, ptx_opts_for(arch));
}

std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch) {
    brass::Module mod("brotensor_trace_rn_ptx");
    gpu_mir::build_row_norm_fn(mod, dag, plan, plan.ew.vec, kEntryRowNorm, true);
    gpu_mir::build_row_norm_fn(mod, dag, plan, 1, kEntryRowNormScalar, true);
    return brass::target::PtxTarget::emit_module(mod, ptx_opts_for(arch));
}

}  // namespace brotensor::jit::ptx
