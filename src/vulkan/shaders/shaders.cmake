# The Vulkan backend's shader list. Each entry is "name|source|glslc defines"
# (defines space-separated); every entry becomes a ShaderId::<name> in the
# generated shader_table.h and an embedded SPIR-V blob. Sources are in this
# directory and may #include anything else here (common.glsl, op_codes.h).
#
# Dtype families are listed as consecutive <base>_f32, <base>_f16, <base>_bf16
# entries and named in kDtypeFamilies (detail/kernels.h), which picks a variant
# by offsetting from the _f32 id; register.cpp checks the order at start-up.

set(BROTENSOR_VK_SHADERS
    "gemm_cm_f16|gemm_cm.comp|-DDT=1"
    "gemm_cm_bf16|gemm_cm.comp|-DDT=2"
    "gemm_cm_f16_c32|gemm_cm.comp|-DDT=1 -DDTC=0 -DDTBI=1"
    "gemm_cm_bf16_c32|gemm_cm.comp|-DDT=2 -DDTC=0 -DDTBI=2"
    "gemm_cm_f32a_w16|gemm_cm.comp|-DDTA=0 -DDT=1 -DDTC=0"
    "gemm_cm_f32a_wbf16|gemm_cm.comp|-DDTA=0 -DDT=2 -DDTC=0"
    "fa_cm_f16|fa_cm.comp|-DDT=1"
    "fa_cm_bf16|fa_cm.comp|-DDT=2"
    "conv_cm_f16|conv_cm.comp|-DDT=1"
    "conv_cm_bf16|conv_cm.comp|-DDT=2"
    "dft64|dft64.comp|"
    "fa_bwd_f32|fa_bwd.comp|-DDT=0"
    "fa_bwd_f16|fa_bwd.comp|-DDT=1"
    "fa_bwd_bf16|fa_bwd.comp|-DDT=2"
    "fa_bwd_cm_f16|fa_bwd_cm.comp|-DDT=1"
    "fa_bwd_cm_bf16|fa_bwd_cm.comp|-DDT=2"
    "test_shared_overflow|test_shared_overflow.comp|"
)
