if(APPLE AND (CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR (NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")))
    set(BROTENSOR_COMPILED_JS_OBJECTS "${CMAKE_CURRENT_SOURCE_DIR}/bronze_js_stubs_arm64.cpp")
else()
    set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${GEN_DIR}")

    add_executable(brotensor-native-manifest
        manifest_tool.cpp
        api.cpp
        native_tensor_core.cpp
        native_tensor_ops.cpp
        native_tensor_nn.cpp
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
    set(INPUT "${CMAKE_CURRENT_SOURCE_DIR}/js/tensor.js")
    set(OUTPUT "${GEN_DIR}/tensor.o")

    add_custom_command(
        OUTPUT "${OUTPUT}"
        COMMAND $<TARGET_FILE:bronze> build "${INPUT}" --emit-obj
                --host-globals "${BROTENSOR_GLOBALS}"
                --native-manifest "${TENSOR_NATIVE_MANIFEST}"
                --entry-symbol=bronze_tensor_main -o "${OUTPUT}"
        DEPENDS bronze "${INPUT}" "${BROTENSOR_GLOBALS}" "${TENSOR_NATIVE_MANIFEST}"
        COMMENT "bronze compile js/tensor.js -> tensor.o (bronze_tensor_main)"
        VERBATIM
    )
    set_source_files_properties("${OUTPUT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(BROTENSOR_COMPILED_JS_OBJECTS "${OUTPUT}")
endif()

add_library(brotensor_api STATIC
    api.cpp
    native_tensor_core.cpp
    native_tensor_ops.cpp
    native_tensor_nn.cpp
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
