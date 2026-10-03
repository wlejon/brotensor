# BrotensorVulkan.cmake — the Vulkan compute backend's build: headers, the
# GLSL -> SPIR-V step, the embedded shader table and the brotensor_vulkan
# static library. Included by the top-level CMakeLists.txt when
# BROTENSOR_WITH_VULKAN is ON (it coexists with BROTENSOR_WITH_HIP), and by
# siblings that ship their own Vulkan kernels, the way they include
# BrotensorHip.cmake:
#
#   brotensor_vulkan_prepare([REQUIRE_HEADERS])
#                                    finds glslc (BROTENSOR_GLSLC) and the Vulkan
#                                    headers (BROTENSOR_VULKAN_HEADERS_TARGET or
#                                    BROTENSOR_VULKAN_INCLUDE_DIR; required only
#                                    with REQUIRE_HEADERS: a consumer's C++ never
#                                    needs them), and sets
#                                    BROTENSOR_VULKAN_SHADER_INCLUDE_DIR, brotensor's
#                                    shader directory (common.glsl & co.) for
#                                    #include in a consumer's GLSL.
#   brotensor_vulkan_add_shaders(<target> NAMESPACE <c++ ns> HEADER <file.h>
#                                SOURCE_DIR <dir> SHADERS "name|file.comp|-Ddefs" ...)
#                                    compiles a consumer's GLSL (same glslc flags as
#                                    brotensor's, brotensor's shader dir on the include
#                                    path) into SPIR-V embedded in <target>, with a
#                                    generated <file.h>: `enum class Shader` and
#                                    `brotensor::vulkan::ShaderHandle handle(Shader)`
#                                    (registered on first use). Dispatch with
#                                    brotensor::vulkan::dispatch (include/brotensor/vulkan.h).
#   brotensor_vulkan_detect(<out_var>)
#                                    TRUE when this machine can build and run the
#                                    backend: glslc, the Vulkan headers and the
#                                    Vulkan loader (libvulkan) are all found. Never
#                                    fails; for a parent's auto-detection (bro), as
#                                    brotensor_hip_detect_gpus is for HIP.
#   brotensor_vulkan_add_backend()   defines the brotensor_vulkan target and
#                                    the defines brotensor_core needs.
#
# A consumer learns whether the backend is in the build from BROTENSOR_WITH_VULKAN
# (brotensor's option, visible after add_subdirectory) or, in C++, from
# BROTENSOR_HAS_VULKAN, which the brotensor target propagates.
#
# What it needs at build time:
#   * Vulkan headers (find_package(Vulkan) -> Vulkan::Headers; any recent
#     vulkan-headers package or the LunarG SDK). Nothing links libvulkan: the
#     loader is dlopen()ed at run time (src/vulkan/detail/vk_fns.h), so a binary
#     built with the backend still starts where Vulkan is absent.
#   * glslc (shaderc), found on PATH, in $VULKAN_SDK/bin, or via
#     -DBROTENSOR_GLSLC=<path>.
#
# Shaders. src/vulkan/shaders/shaders.cmake lists every kernel as
# "name|source|defines". Each is compiled by glslc (--target-env=vulkan1.2, -O)
# to a C initializer list (-mfmt=c) in <build>/vulkan_shaders/<name>.inc, and a
# generated shader_table.{h,cpp} turns the list into `enum class ShaderId` plus
# the embedded words. No SPIR-V is read from disk at run time. Editing any
# .comp / .glsl / .h in the shader directory recompiles every shader (simple
# and correct; the full set compiles in a few seconds).

include_guard(GLOBAL)

# Cache entries, not directory variables: the GLOBAL guard means only the first
# include() in a configure runs this file, and that can be in a sibling
# directory of the consumer (bro includes it from brotensor's subdirectory,
# then brodiffusion's and brovisionml's includes are skipped), so a plain
# set() here would be invisible where brotensor_vulkan_add_shaders is called.
set(_BROTENSOR_VULKAN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE INTERNAL "brotensor source root")
set(BROTENSOR_VULKAN_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../src/vulkan/shaders"
    CACHE INTERNAL "brotensor's GLSL include directory (common.glsl, math_acc.glsl, ...)")

macro(brotensor_vulkan_prepare)
    cmake_parse_arguments(_btvp "REQUIRE_HEADERS" "" "" ${ARGN})
    if(NOT BROTENSOR_GLSLC)
        find_program(BROTENSOR_GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
    endif()
    if(NOT BROTENSOR_GLSLC)
        message(FATAL_ERROR
            "The Vulkan backend needs glslc (shaderc) to compile GLSL kernels to SPIR-V: "
            "install shaderc, or pass -DBROTENSOR_GLSLC=<path to glslc>.")
    endif()
    set(BROTENSOR_VULKAN_HEADERS_TARGET "")
    find_package(Vulkan QUIET)
    if(TARGET Vulkan::Headers)
        set(BROTENSOR_VULKAN_HEADERS_TARGET Vulkan::Headers)
    else()
        find_path(BROTENSOR_VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/include")
        if(NOT BROTENSOR_VULKAN_INCLUDE_DIR AND _btvp_REQUIRE_HEADERS)
            message(FATAL_ERROR
                "BROTENSOR_WITH_VULKAN needs the Vulkan headers (vulkan/vulkan.h): install "
                "vulkan-headers (or the LunarG SDK and set VULKAN_SDK).")
        endif()
    endif()
endmacro()

function(brotensor_vulkan_detect out_var)
    set(_ok FALSE)
    find_program(_btvd_glslc NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" NO_CACHE)
    find_path(_btvd_inc vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/include" NO_CACHE)
    # The loader is dlopen()ed at run time, never linked: look for it only to
    # tell whether a binary built here would find a driver.
    find_library(_btvd_loader NAMES vulkan vulkan-1 libvulkan.so.1
                 HINTS "$ENV{VULKAN_SDK}/lib" "$ENV{VULKAN_SDK}/Lib" NO_CACHE)
    if(_btvd_glslc AND _btvd_inc AND _btvd_loader)
        set(_ok TRUE)
    endif()
    set(${out_var} ${_ok} PARENT_SCOPE)
endfunction()

function(brotensor_vulkan_add_shaders target)
    cmake_parse_arguments(_a "" "NAMESPACE;HEADER;SOURCE_DIR" "SHADERS;DEPENDS" ${ARGN})
    if(NOT _a_NAMESPACE OR NOT _a_HEADER OR NOT _a_SHADERS)
        message(FATAL_ERROR "brotensor_vulkan_add_shaders(${target}): NAMESPACE, HEADER and SHADERS are required")
    endif()
    if(NOT BROTENSOR_GLSLC)
        brotensor_vulkan_prepare()
    endif()
    if(NOT _a_SOURCE_DIR)
        set(_a_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    set(_gen "${CMAKE_CURRENT_BINARY_DIR}/${target}_vk_shaders")
    file(MAKE_DIRECTORY "${_gen}")
    file(GLOB _deps CONFIGURE_DEPENDS
        "${_a_SOURCE_DIR}/*.comp" "${_a_SOURCE_DIR}/*.glsl" "${_a_SOURCE_DIR}/*.h"
        "${BROTENSOR_VULKAN_SHADER_INCLUDE_DIR}/*.glsl" "${BROTENSOR_VULKAN_SHADER_INCLUDE_DIR}/*.h")
    set(_incs "")
    set(_enum "")
    set(_blobs "")
    set(_cases "")
    foreach(_entry IN LISTS _a_SHADERS)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _name)
        list(GET _parts 1 _file)
        list(LENGTH _parts _np)
        set(_defs "")
        if(_np GREATER 2)
            list(GET _parts 2 _defstr)
            separate_arguments(_defs UNIX_COMMAND "${_defstr}")
        endif()
        set(_out "${_gen}/${_name}.inc")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${BROTENSOR_GLSLC}" --target-env=vulkan1.2 -O -mfmt=c ${_defs}
                    -I "${_a_SOURCE_DIR}" -I "${BROTENSOR_VULKAN_SHADER_INCLUDE_DIR}"
                    "${_a_SOURCE_DIR}/${_file}" -o "${_out}"
            DEPENDS ${_deps} ${_a_DEPENDS}
            COMMENT "glslc ${target}/${_name}"
            VERBATIM)
        list(APPEND _incs "${_out}")
        string(APPEND _enum "    ${_name},\n")
        string(APPEND _blobs "const std::uint32_t k_${_name}[] =\n#include \"${_name}.inc\"\n;\n")
        string(APPEND _cases "        case Shader::${_name}: {\n            static const auto h = reg(\"${target}/${_name}\", k_${_name}, sizeof(k_${_name}) / 4);\n            return h;\n        }\n")
    endforeach()
    file(CONFIGURE OUTPUT "${_gen}/${_a_HEADER}" CONTENT
"// Generated by brotensor_vulkan_add_shaders(${target}) (brotensor's cmake/BrotensorVulkan.cmake).
#pragma once
#include <brotensor/vulkan.h>
#include <cstdint>

namespace @_a_NAMESPACE@ {

enum class Shader : std::uint32_t {
@_enum@};

// The shader's handle for brotensor::vulkan::dispatch / kernel_info,
// registered with brotensor on first use (thread-safe).
brotensor::vulkan::ShaderHandle handle(Shader s);

}  // namespace @_a_NAMESPACE@
" @ONLY)
    file(CONFIGURE OUTPUT "${_gen}/${target}_vk_shaders.cpp" CONTENT
"// Generated by brotensor_vulkan_add_shaders(${target}): the embedded SPIR-V.
#include \"@_a_HEADER@\"

#include <cstddef>

namespace @_a_NAMESPACE@ {

namespace {
@_blobs@
brotensor::vulkan::ShaderHandle reg(const char* name, const std::uint32_t* w, std::size_t n) {
    // register_shader returns the existing handle for the same words.
    return brotensor::vulkan::register_shader(name, w, n);
}
}  // namespace

brotensor::vulkan::ShaderHandle handle(Shader s) {
    switch (s) {
@_cases@    }
    return 0;
}

}  // namespace @_a_NAMESPACE@
" @ONLY)
    target_sources(${target} PRIVATE "${_gen}/${target}_vk_shaders.cpp" ${_incs})
    target_include_directories(${target} PRIVATE "${_gen}")
endfunction()

function(brotensor_vulkan_add_backend)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan")
    set(_shader_dir "${_src}/shaders")
    set(_gen "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders")
    file(MAKE_DIRECTORY "${_gen}")

    brotensor_vulkan_prepare(REQUIRE_HEADERS)
    set(_vk_headers "${BROTENSOR_VULKAN_HEADERS_TARGET}")

    include("${_shader_dir}/shaders.cmake")
    file(GLOB _shader_deps CONFIGURE_DEPENDS
        "${_shader_dir}/*.comp" "${_shader_dir}/*.glsl" "${_shader_dir}/*.h")

    set(_incs "")
    set(_enum "")
    set(_blobs "")
    set(_table "")
    foreach(_entry IN LISTS BROTENSOR_VK_SHADERS)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _name)
        list(GET _parts 1 _file)
        list(LENGTH _parts _np)
        set(_defs "")
        if(_np GREATER 2)
            list(GET _parts 2 _defstr)
            separate_arguments(_defs UNIX_COMMAND "${_defstr}")
        endif()
        set(_out "${_gen}/${_name}.inc")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${BROTENSOR_GLSLC}" --target-env=vulkan1.2 -O -mfmt=c ${_defs}
                    -I "${_shader_dir}" "${_shader_dir}/${_file}" -o "${_out}"
            DEPENDS ${_shader_deps}
            COMMENT "glslc ${_name}"
            VERBATIM)
        list(APPEND _incs "${_out}")
        string(APPEND _enum "    ${_name},\n")
        string(APPEND _blobs "static const std::uint32_t k_${_name}[] =\n#include \"${_name}.inc\"\n;\n")
        string(APPEND _table "    {\"${_name}\", k_${_name}, sizeof(k_${_name}) / sizeof(std::uint32_t)},\n")
    endforeach()

    file(CONFIGURE OUTPUT "${_gen}/shader_table.h" CONTENT
"// Generated by cmake/BrotensorVulkan.cmake from src/vulkan/shaders/shaders.cmake.
#pragma once
#include <cstddef>
#include <cstdint>

namespace brotensor::detail::vulkan {

enum class ShaderId : std::uint32_t {
@_enum@    kCount
};

struct ShaderBlob {
    const char* name;
    const std::uint32_t* words;
    std::size_t nwords;
};

const ShaderBlob& shader_blob(ShaderId id);

}  // namespace brotensor::detail::vulkan
" @ONLY)
    file(CONFIGURE OUTPUT "${_gen}/shader_table.cpp" CONTENT
"// Generated by cmake/BrotensorVulkan.cmake: the embedded SPIR-V of every kernel.
#include \"shader_table.h\"

namespace brotensor::detail::vulkan {

@_blobs@
static const ShaderBlob k_table[] = {
@_table@};

const ShaderBlob& shader_blob(ShaderId id) { return k_table[static_cast<std::uint32_t>(id)]; }

}  // namespace brotensor::detail::vulkan
" @ONLY)

    add_library(brotensor_vulkan STATIC
        ${_src}/loader.cpp
        ${_src}/instance.cpp
        ${_src}/allocator.cpp
        ${_src}/stream.cpp
        ${_src}/pipelines.cpp
        ${_src}/spirv_reflect.cpp
        ${_src}/tensor.cpp
        ${_src}/graph.cpp
        ${_src}/custom.cpp
        ${_src}/register.cpp
        ${_src}/ops_elementwise.cpp
        ${_src}/ops_copy.cpp
        ${_src}/ops_reduce.cpp
        ${_src}/gemm.cpp
        ${_src}/ops_linear.cpp
        ${_src}/ops_norm.cpp
        ${_src}/ops_rope.cpp
        ${_src}/ops_glu.cpp
        ${_src}/ops_attention.cpp
        ${_src}/ops_attention_dense.cpp
        ${_src}/ops_attention_proj.cpp
        ${_src}/ops_attention_bwd.cpp
        ${_src}/ops_topk.cpp
        ${_src}/ops_xent.cpp
        ${_src}/ops_conv.cpp
        ${_src}/ops_gnorm.cpp
        ${_src}/ops_spatial.cpp
        ${_src}/ops_diffusion.cpp
        ${_src}/ops_quant.cpp
        ${_src}/ops_quant_attention.cpp
        ${_src}/ops_audio.cpp
        ${_src}/ops_spectral.cpp
        ${_src}/ops_sampling.cpp
        ${_src}/ops_misc.cpp
        ${_src}/ops_delta.cpp
        ${_src}/ops_vision.cpp
        "${_gen}/shader_table.cpp"
        ${_incs}
    )
    target_include_directories(brotensor_vulkan PRIVATE "${_gen}" "${_src}" "${_shader_dir}")
    target_link_libraries(brotensor_vulkan PUBLIC brotensor_core PRIVATE "$<BUILD_INTERFACE:${_vk_headers}>" ${CMAKE_DL_LIBS})
    if(BROTENSOR_VULKAN_INCLUDE_DIR AND NOT _vk_headers)
        target_include_directories(brotensor_vulkan PRIVATE "${BROTENSOR_VULKAN_INCLUDE_DIR}")
    endif()
    target_compile_definitions(brotensor_vulkan PUBLIC BROTENSOR_HAS_VULKAN=1 BROTENSOR_HAS_GPU=1)
    target_compile_definitions(brotensor_vulkan PRIVATE VK_NO_PROTOTYPES)
    if(MSVC)
        target_compile_options(brotensor_vulkan PRIVATE /W4)
    else()
        # The Vulkan create-info idiom `VkFooInfo x{VK_STRUCTURE_TYPE_...}` leaves
        # the rest value-initialised on purpose.
        target_compile_options(brotensor_vulkan PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    endif()
    # White-box tests (tests/test_vulkan.cpp) reach the backend's internal
    # headers through this; nothing installed or shipped uses it.
    add_library(brotensor_vulkan_internal INTERFACE)
    target_include_directories(brotensor_vulkan_internal INTERFACE
        "$<BUILD_INTERFACE:${_gen}>" "$<BUILD_INTERFACE:${_src}>" "$<BUILD_INTERFACE:${_shader_dir}>")
    target_compile_definitions(brotensor_vulkan_internal INTERFACE VK_NO_PROTOTYPES)
    if(_vk_headers)
        target_link_libraries(brotensor_vulkan_internal INTERFACE "$<BUILD_INTERFACE:${_vk_headers}>")
    elseif(BROTENSOR_VULKAN_INCLUDE_DIR)
        target_include_directories(brotensor_vulkan_internal INTERFACE "$<BUILD_INTERFACE:${BROTENSOR_VULKAN_INCLUDE_DIR}>")
    endif()
    add_dependencies(brotensor_vulkan_internal brotensor_vulkan)
    # init.cpp probes + registers the backend, so core needs the define too.
    target_compile_definitions(brotensor_core PRIVATE BROTENSOR_HAS_VULKAN=1 BROTENSOR_HAS_GPU=1)
    list(LENGTH BROTENSOR_VK_SHADERS _nshaders)
    message(STATUS "brotensor: Vulkan backend ON (glslc: ${BROTENSOR_GLSLC}, ${_nshaders} shaders)")
endfunction()
