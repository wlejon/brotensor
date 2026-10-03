# The Vulkan backend's shader list. Each entry is "name|source|glslc defines"
# (defines space-separated); every entry becomes a ShaderId::<name> in the
# generated shader_table.h and an embedded SPIR-V blob. Sources are in this
# directory and may #include anything else here (common.glsl, op_codes.h).
#
# Dtype families are listed as consecutive <base>_f32, <base>_f16, <base>_bf16
# entries and named in kDtypeFamilies (detail/kernels.h), which picks a variant
# by offsetting from the _f32 id; register.cpp checks the order at start-up.

set(BROTENSOR_VK_SHADERS
    "unary_f32|unary.comp|-DDT=0"
    "unary_f16|unary.comp|-DDT=1"
    "unary_bf16|unary.comp|-DDT=2"
    "binary_f32|binary.comp|-DDTY=0 -DDTX=0"
    "binary_f16|binary.comp|-DDTY=1 -DDTX=1"
    "binary_bf16|binary.comp|-DDTY=2 -DDTX=2"
    "binary_f32_f16|binary.comp|-DDTY=0 -DDTX=1"
    "act_bwd_f32|act_bwd.comp|-DDT=0"
    "act_bwd_f16|act_bwd.comp|-DDT=1"
    "act_bwd_bf16|act_bwd.comp|-DDT=2"
    "bias_f32|bias.comp|-DDT=0"
    "bias_f16|bias.comp|-DDT=1"
    "bias_bf16|bias.comp|-DDT=2"
    "cast_f32_to_f16|cast.comp|-DSDT=0 -DDDT=1"
    "cast_f32_to_bf16|cast.comp|-DSDT=0 -DDDT=2"
    "cast_f16_to_f32|cast.comp|-DSDT=1 -DDDT=0"
    "cast_f16_to_bf16|cast.comp|-DSDT=1 -DDDT=2"
    "cast_bf16_to_f32|cast.comp|-DSDT=2 -DDDT=0"
    "cast_bf16_to_f16|cast.comp|-DSDT=2 -DDDT=1"
    "sum_rows_f32|reduce_rows.comp|-DDT=0 -DREDUCE_SUM"
    "sum_rows_f16|reduce_rows.comp|-DDT=1 -DREDUCE_SUM"
    "sum_rows_bf16|reduce_rows.comp|-DDT=2 -DREDUCE_SUM"
    "argmax_rows_f32|reduce_rows.comp|-DDT=0 -DREDUCE_ARGMAX"
    "argmax_rows_f16|reduce_rows.comp|-DDT=1 -DREDUCE_ARGMAX"
    "argmax_rows_bf16|reduce_rows.comp|-DDT=2 -DREDUCE_ARGMAX"
    "sum_cols_f32|sum_cols.comp|-DDT=0"
    "sum_cols_f16|sum_cols.comp|-DDT=1"
    "sum_cols_bf16|sum_cols.comp|-DDT=2"
    "copy2d_b1|copy2d.comp|-DES=1"
    "copy2d_b2|copy2d.comp|-DES=2"
    "copy2d_b4|copy2d.comp|-DES=4"
    "transpose_b2|transpose.comp|-DES=2"
    "transpose_b4|transpose.comp|-DES=4"
    "fill_bytes|fill_bytes.comp|"
    "test_shared_overflow|test_shared_overflow.comp|"
)
