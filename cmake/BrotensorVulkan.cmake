# BrotensorVulkan.cmake — the Vulkan compute backend's build: headers and
# the brotensor_vulkan static library. All kernels compile from single-source
# Brass MIR (src/kernels/mir_*.{h,cpp}) via brass::target::SpirvTarget; no
# external shader compiler (glslc / shaderc) is needed.
#
#   brotensor_vulkan_prepare([REQUIRE_HEADERS])
#                                    finds the Vulkan headers
#                                    (BROTENSOR_VULKAN_HEADERS_TARGET or
#                                    BROTENSOR_VULKAN_INCLUDE_DIR; required only
#                                    with REQUIRE_HEADERS).
#   brotensor_vulkan_detect(<out_var>)
#                                    TRUE when this machine can build and run the
#                                    backend: the Vulkan headers and the Vulkan
#                                    loader (libvulkan) are found. Never fails;
#                                    for a parent's auto-detection (bro).
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
#   * Brass with the SPIR-V target enabled (BRASS_WITH_SPIRV).

include_guard(GLOBAL)

set(_BROTENSOR_VULKAN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE INTERNAL "brotensor source root")

macro(brotensor_vulkan_prepare)
    cmake_parse_arguments(_btvp "REQUIRE_HEADERS" "" "" ${ARGN})
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
    find_path(_btvd_inc vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/include" NO_CACHE)
    # The loader is dlopen()ed at run time, never linked: look for it only to
    # tell whether a binary built here would find a driver.
    find_library(_btvd_loader NAMES vulkan vulkan-1 libvulkan.so.1
                 HINTS "$ENV{VULKAN_SDK}/lib" "$ENV{VULKAN_SDK}/Lib" NO_CACHE)
    if(_btvd_inc AND _btvd_loader)
        set(_ok TRUE)
    endif()
    set(${out_var} ${_ok} PARENT_SCOPE)
endfunction()

function(brotensor_vulkan_add_backend)
    if(NOT (TARGET brass AND BRASS_WITH_SPIRV AND BROTENSOR_HAS_BRASS_JIT))
        message(FATAL_ERROR
            "BROTENSOR_WITH_VULKAN=ON requires Brass with BRASS_WITH_SPIRV=ON "
            "(all Vulkan kernels compile from Brass MIR to SPIR-V).")
    endif()

    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan")
    set(_shader_dir "${_src}/shaders")
    set(_jit "${_BROTENSOR_VULKAN_ROOT}/src/jit")

    brotensor_vulkan_prepare(REQUIRE_HEADERS)
    set(_vk_headers "${BROTENSOR_VULKAN_HEADERS_TARGET}")

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
        ${_src}/kernel_interop_vk.cpp
        ${_src}/vulkan_jit.cpp
        ${_src}/vk_mir_cache.cpp
        ${_jit}/trace_compiler_vulkan.cpp
        ${_src}/register.cpp
        ${_src}/ops_elementwise.cpp
        ${_src}/ops_copy.cpp
        ${_src}/ops_reduce.cpp
        ${_src}/gemm.cpp
        ${_src}/vk_autotune.cpp
        ${_src}/ops_linear.cpp
        ${_src}/ops_norm.cpp
        ${_src}/ops_rope.cpp
        ${_src}/ops_glu.cpp
        ${_src}/ops_attention.cpp
        ${_src}/ops_attention_dense.cpp
        ${_src}/ops_attention_proj.cpp
        ${_src}/ops_attention_bwd.cpp
        ${_src}/ops_fa_bwd.cpp
        ${_src}/ops_scatter.cpp
        ${_src}/ops_spatial_bwd.cpp
        ${_src}/ops_lstm.cpp
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
    )
    target_include_directories(brotensor_vulkan PRIVATE "${_src}" "${_shader_dir}")
    target_link_libraries(brotensor_vulkan PUBLIC brotensor_core
        PRIVATE "$<BUILD_INTERFACE:${_vk_headers}>" "$<BUILD_INTERFACE:brass>" ${CMAKE_DL_LIBS})
    if(BROTENSOR_VULKAN_INCLUDE_DIR AND NOT _vk_headers)
        target_include_directories(brotensor_vulkan PRIVATE "${BROTENSOR_VULKAN_INCLUDE_DIR}")
    endif()
    target_compile_definitions(brotensor_vulkan PUBLIC
        BROTENSOR_HAS_VULKAN=1 BROTENSOR_HAS_GPU=1 BROTENSOR_HAS_VULKAN_TRACE_JIT=1)
    target_compile_definitions(brotensor_vulkan PRIVATE VK_NO_PROTOTYPES)
    set(BROTENSOR_VULKAN_TRACE_JIT ON PARENT_SCOPE)
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
        "$<BUILD_INTERFACE:${_src}>" "$<BUILD_INTERFACE:${_shader_dir}>")
    target_compile_definitions(brotensor_vulkan_internal INTERFACE VK_NO_PROTOTYPES)
    if(_vk_headers)
        target_link_libraries(brotensor_vulkan_internal INTERFACE "$<BUILD_INTERFACE:${_vk_headers}>")
    elseif(BROTENSOR_VULKAN_INCLUDE_DIR)
        target_include_directories(brotensor_vulkan_internal INTERFACE "$<BUILD_INTERFACE:${BROTENSOR_VULKAN_INCLUDE_DIR}>")
    endif()
    add_dependencies(brotensor_vulkan_internal brotensor_vulkan)
    # init.cpp probes + registers the backend, so core needs the define too.
    target_compile_definitions(brotensor_core PRIVATE BROTENSOR_HAS_VULKAN=1 BROTENSOR_HAS_GPU=1)
    message(STATUS "brotensor: Vulkan backend ON (Brass MIR -> SPIR-V, 0 GLSL shaders)")
endfunction()
