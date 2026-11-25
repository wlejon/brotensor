# The Vulkan backend's shader list. Each entry is "name|source|glslc defines"
# (defines space-separated); every entry becomes a ShaderId::<name> in the
# generated shader_table.h and an embedded SPIR-V blob. Sources are in this
# directory and may #include anything else here (common.glsl, op_codes.h).

set(BROTENSOR_VK_SHADERS
    "fa_cm_f16|fa_cm.comp|-DDT=1"
    "fa_cm_bf16|fa_cm.comp|-DDT=2"
    "conv_cm_f16|conv_cm.comp|-DDT=1"
    "conv_cm_bf16|conv_cm.comp|-DDT=2"
    "fa_bwd_cm_f16|fa_bwd_cm.comp|-DDT=1"
    "fa_bwd_cm_bf16|fa_bwd_cm.comp|-DDT=2"
    "test_shared_overflow|test_shared_overflow.comp|"
)
