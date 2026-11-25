# The Vulkan backend's shader list. Each entry is "name|source|glslc defines"
# (defines space-separated); every entry becomes a ShaderId::<name> in the
# generated shader_table.h and an embedded SPIR-V blob. Sources are in this
# directory and may #include anything else here (common.glsl, op_codes.h).

set(BROTENSOR_VK_SHADERS
    "test_shared_overflow|test_shared_overflow.comp|"
)
