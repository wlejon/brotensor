#include <brotensor/tensor.h>

#if BROTENSOR_HAS_BRASS_JIT
#include "kernels/mir_conv_cm.h"
#include "kernels/mir_conv_direct.h"
#include "kernels/mir_fa_cm.h"
#include "kernels/mir_gemm_cm.h"
#include "kernels/mir_gemm_simt.h"
#include "vulkan/shaders/op_codes.h"
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>

#include <functional>
#include <string>

namespace brotensor::verify::extra_audit {

void audit_extra_mir_kernels(
    const brass::target::SpirvOptions& spv_opts,
    const std::function<void(const char*, const brass::target::SpirvOptions&,
                             const std::function<brass::Function*(brass::Module&)>&, bool)>& audit_mir_opts) {
    brass::target::SpirvOptions cm_opts = spv_opts;
    cm_opts.local_size_x = 128u;

    // GemmCm swizzle + EPI_RESIDUAL_ADD (direct and shared-scratch epilogue)
    {
        kernels::GemmCmSpec s;
        s.dt_a = Dtype::FP16; s.dt_b = Dtype::FP16; s.dt_c = Dtype::FP16;
        s.cfg_idx = 1u; s.epi = EPI_STORE; s.direct = true; s.swizzle = true; s.has_bias = true;
        audit_mir_opts("gemm_cm_f16_swizzle", cm_opts,
                       [s](brass::Module& m) { return kernels::build_gemm_cm_mir(m, s); }, false);
    }
    {
        kernels::GemmCmSpec s;
        s.dt_a = Dtype::FP16; s.dt_b = Dtype::FP16; s.dt_c = Dtype::FP16;
        s.cfg_idx = 1u; s.epi = EPI_RESIDUAL_ADD; s.direct = true; s.has_bias = true;
        audit_mir_opts("gemm_cm_f16_residual_direct", cm_opts,
                       [s](brass::Module& m) { return kernels::build_gemm_cm_mir(m, s); }, false);
    }
    {
        kernels::GemmCmSpec s;
        s.dt_a = Dtype::BF16; s.dt_b = Dtype::BF16; s.dt_c = Dtype::BF16;
        s.cfg_idx = 1u; s.epi = EPI_RESIDUAL_ADD; s.direct = false; s.has_bias = true; s.scale_a = true;
        audit_mir_opts("gemm_cm_bf16_residual", cm_opts,
                       [s](brass::Module& m) { return kernels::build_gemm_cm_mir(m, s); }, false);
    }

    // FaCm head_swizzle + softcap
    {
        brass::target::SpirvOptions cm64_opts = spv_opts;
        cm64_opts.local_size_x = 64u;
        kernels::FaCmSpec s;
        s.dt = Dtype::FP16; s.hd = 64u; s.bc = 16u; s.nsg = 2u;
        s.has_mask = true; s.head_swizzle = true; s.has_softcap = true;
        audit_mir_opts("fa_cm_f16_swizzle_softcap", cm64_opts,
                       [s](brass::Module& m) { return kernels::build_fa_cm_mir(m, s); }, false);
    }

    // GemmSimt fused EPI_SWIGLU (FP32/FP16/BF16) and EPI_RESIDUAL_ADD
    for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
             {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
        const std::string sw_name = std::string("gemm_simt_swiglu") + sfx;
        audit_mir_opts(sw_name.c_str(), spv_opts, [dt](brass::Module& m) {
            return kernels::build_gemm_simt_mir(m, dt, dt, dt, dt, false, EPI_SWIGLU, 0u, false, false, true);
        }, false);
        const std::string res_name = std::string("gemm_simt_residual") + sfx;
        audit_mir_opts(res_name.c_str(), spv_opts, [dt](brass::Module& m) {
            return kernels::build_gemm_simt_mir(m, dt, dt, dt, dt, false, EPI_RESIDUAL_ADD, LACT_SILU, false, false, true);
        }, false);
    }

    // ConvCm / ConvSimt / ConvDirect fused activations
    for (const auto& [asfx, act] : std::vector<std::pair<const char*, std::uint32_t>>{
             {"_silu", LACT_SILU}, {"_gelu", LACT_GELU_EXACT}}) {
        {
            kernels::ConvCmSpec s;
            s.dt = Dtype::FP16; s.cfg_idx = 2u; s.has_bias = true; s.act = act;
            const std::string n = std::string("conv_cm_f16") + asfx;
            audit_mir_opts(n.c_str(), cm_opts,
                           [s](brass::Module& m) { return kernels::build_conv_cm_mir(m, s); }, false);
        }
        {
            const std::string n = std::string("conv_simt_f32") + asfx;
            audit_mir_opts(n.c_str(), spv_opts, [act](brass::Module& m) {
                return kernels::build_conv_simt_mir(m, Dtype::FP32, false, false, true, 3, 3, 1, 1, 1, 1, 1, 1, act);
            }, false);
        }
    }
    audit_mir_opts("conv_direct_f32_silu", spv_opts, [](brass::Module& m) {
        return kernels::build_conv_direct_mir(m, (LACT_SILU << 4), Dtype::FP32, true);
    }, false);
}

}  // namespace brotensor::verify::extra_audit
#endif
