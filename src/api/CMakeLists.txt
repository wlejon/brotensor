# On a target brass has no code generator for (BRASS_HOST_BACKEND, decided in
# brass's CMakeLists — Apple Silicon today) js/tensor.js is not compiled and
# a no-op entry stands in: the natives still register, but nothing mounts
# `bro.tensor`. BROTENSOR_API_COMPILED_JS tells the tests which of the two
# they are looking at.
set(BROTENSOR_API_NATIVE_SOURCES
    api.cpp
    native_tensor_core.cpp
    native_tensor_ops.cpp
    native_tensor_nn.cpp
    native_tensor_attention.cpp
    native_tensor_safetensors.cpp
    native_tensor_batched.cpp
    native_tensor_attn2.cpp
    native_tensor_audio.cpp
    native_tensor_audio_b.cpp
    native_tensor_conv.cpp
    native_tensor_int8.cpp
    native_tensor_misc.cpp
)

if(NOT BRASS_HOST_BACKEND)
    set(BROTENSOR_COMPILED_JS_OBJECTS "${CMAKE_CURRENT_SOURCE_DIR}/bronze_js_stubs_arm64.cpp")
    set(BROTENSOR_API_COMPILED_JS OFF PARENT_SCOPE)
    message(STATUS "brotensor: api built with the no-op JS entry (brass has no code generator for this target)")
else()
    set(BROTENSOR_API_COMPILED_JS ON PARENT_SCOPE)
    set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${GEN_DIR}")

    add_executable(brotensor-native-manifest
        manifest_tool.cpp
        ${BROTENSOR_API_NATIVE_SOURCES}
        bronze_js_stubs_arm64.cpp
    )
    target_include_directories(brotensor-native-manifest PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(brotensor-native-manifest PRIVATE
        brotensor
        bronze::runtime_shared
    )
    # The tool runs at build time and links bronze's shared runtime, which is
    # built into BRONZE_SHARED_RUNTIME_DIR rather than beside this exe. The PE
    # loader only looks beside the .exe, so without this the custom command
    # dies with 0xc0000135 (and a modal "dll was not found" box).
    if(WIN32)
        add_custom_command(TARGET brotensor-native-manifest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brotensor-native-manifest>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brotensor-native-manifest>
        )
    endif()

    set(TENSOR_NATIVE_MANIFEST "${GEN_DIR}/tensor_natives.json")
    add_custom_command(
        OUTPUT "${TENSOR_NATIVE_MANIFEST}"
        COMMAND $<TARGET_FILE:brotensor-native-manifest> "${TENSOR_NATIVE_MANIFEST}.new"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TENSOR_NATIVE_MANIFEST}.new" "${TENSOR_NATIVE_MANIFEST}"
        DEPENDS brotensor-native-manifest
        COMMENT "brotensor-native-manifest -> generated/tensor_natives.json"
        VERBATIM
    )

    set(BROTENSOR_GLOBALS "${CMAKE_CURRENT_SOURCE_DIR}/tensor.globals")

    # One compiled module per file, one entry symbol each, mounted in this
    # order by installTensorJS: js/tensor.js (bro.tensor, GpuTensor, the dense
    # / elementwise / conv surface, safetensors, RNG), js/tensor_ext.js (the
    # attention family, losses, pooling, concat), then the restored groups —
    # batched dense + gather/scatter/topK, the second attention family
    # (decomposed rel-pos, varlen flash, gated delta rule, mrope), the audio /
    # 1-D / spectral family, the 2-D conv + pooling + batch-norm family, the
    # GGUF-quant + INT8-W8A16 family and the l2/bce/diffusion-sampler
    # leftovers. Split so each file stays readable; they all read the same host
    # globals and the same native manifest.
    set(BROTENSOR_COMPILED_JS_OBJECTS "")
    foreach(module tensor tensor_ext tensor_batched tensor_attn2 tensor_audio tensor_conv tensor_int8 tensor_misc)
        set(INPUT "${CMAKE_CURRENT_SOURCE_DIR}/js/${module}.js")
        set(OUTPUT "${GEN_DIR}/${module}.o")
        add_custom_command(
            OUTPUT "${OUTPUT}"
            COMMAND $<TARGET_FILE:bronze> build "${INPUT}" --emit-obj
                    --host-globals "${BROTENSOR_GLOBALS}"
                    --native-manifest "${TENSOR_NATIVE_MANIFEST}"
                    --entry-symbol=bronze_${module}_main -o "${OUTPUT}"
            DEPENDS bronze "${INPUT}" "${BROTENSOR_GLOBALS}" "${TENSOR_NATIVE_MANIFEST}"
            COMMENT "bronze compile js/${module}.js -> ${module}.o (bronze_${module}_main)"
            VERBATIM
        )
        set_source_files_properties("${OUTPUT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND BROTENSOR_COMPILED_JS_OBJECTS "${OUTPUT}")
    endforeach()
endif()

add_library(brotensor_api STATIC
    ${BROTENSOR_API_NATIVE_SOURCES}
    ${BROTENSOR_COMPILED_JS_OBJECTS}
)

target_include_directories(brotensor_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(brotensor_api PUBLIC
    brotensor
    bronze::runtime_shared
)
