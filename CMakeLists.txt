cmake_minimum_required(VERSION 3.24)
project(brotensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- Backends ---
#
# brotensor is layered: a core dispatcher + always-on CPU backend, plus
# optional GPU backends (CUDA, Metal) that compile as independent static
# libraries and register themselves into the dispatcher at static-init.
# Multiple GPU backends are no longer prohibited at the CMake level — they
# remain mutually exclusive in practice because their toolchains do not
# coexist on the same host (nvcc on Linux/Windows; Apple toolchain on Mac).
# CPU has no opt-out.
option(BROTENSOR_WITH_CUDA  "Build GPU backend (CUDA)"  OFF)
option(BROTENSOR_WITH_METAL "Build GPU backend (Metal)" OFF)

if(BROTENSOR_WITH_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    if(MSVC AND DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_CUDA_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif()
endif()
if(BROTENSOR_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "BROTENSOR_WITH_METAL requires Apple platform.")
    endif()
    enable_language(OBJCXX)
endif()

# --- brotensor_core: dispatcher + public op wrappers + Tensor + runtime ---
# Always built. Sources produced by Phase 1A.

add_library(brotensor_core STATIC
    src/tensor.cpp
    src/dispatch.cpp
    src/init.cpp
    src/ops.cpp
    src/safetensors.cpp
)
target_include_directories(brotensor_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brotensor_core PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(brotensor_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brotensor_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# --- brotensor_cpu: CPU backend (always built) ---
# Sources produced by Phase 1B.

add_library(brotensor_cpu STATIC
    src/cpu/alloc.cpp
    src/cpu/ops_impl.cpp
    src/cpu/register.cpp
    src/cpu/elementwise.cpp
    src/cpu/loss.cpp
    src/cpu/embedding.cpp
    src/cpu/concat.cpp
    src/cpu/public_reductions.cpp
    src/cpu/layernorm_inference.cpp
    src/cpu/activations.cpp
    src/cpu/geglu.cpp
    src/cpu/swiglu.cpp
    src/cpu/matmul.cpp
    src/cpu/rope.cpp
    src/cpu/rms_norm.cpp
    src/cpu/conv2d.cpp
    src/cpu/conv3d.cpp
    src/cpu/group_norm.cpp
    src/cpu/resample.cpp
    src/cpu/interp2d.cpp
    src/cpu/pad2d.cpp
    src/cpu/slice2d.cpp
    src/cpu/top_k.cpp
    src/cpu/transpose.cpp
    src/cpu/diffusion_samplers.cpp
    src/cpu/kv_cache.cpp
    src/cpu/cross_attention.cpp
    src/cpu/self_attention.cpp
    src/cpu/attention_moments.cpp
    src/cpu/flash_attention.cpp
    src/cpu/self_attention_inference.cpp
    src/cpu/resblock.cpp
    src/cpu/modulate.cpp
    src/cpu/self_attention_bias.cpp
    src/cpu/fft.cpp
    src/cpu/stft.cpp
    src/cpu/conv1d.cpp
    src/cpu/vocoder_activations.cpp
    src/cpu/codec_quant.cpp
    src/cpu/resample1d.cpp
    src/cpu/log_exp_round.cpp
    src/cpu/sample_logits.cpp
    src/cpu/l2_norm.cpp
    src/cpu/gated_delta_rule.cpp
    src/cpu/spatial_merge.cpp
    src/cpu/rope_mrope.cpp
)
target_link_libraries(brotensor_cpu PUBLIC brotensor_core)
if(MSVC)
    target_compile_options(brotensor_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brotensor_cpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# --- brotensor_cuda: CUDA backend (optional) ---

if(BROTENSOR_WITH_CUDA)
    add_library(brotensor_cuda STATIC
        src/cuda/runtime.cu
        src/cuda/tensor.cu
        src/cuda/gemm.cu
        src/cuda/fp16_matmul.cu
        src/cuda/elementwise.cu
        src/cuda/softmax.cu
        src/cuda/layernorm.cu
        src/cuda/attention.cu
        src/cuda/optim.cu
        src/cuda/adam.cu
        src/cuda/reduce.cu
        src/cuda/loss.cu
        src/cuda/embedding.cu
        src/cuda/concat.cu
        src/cuda/mha.cu
        src/cuda/batched_ops.cu
        src/cuda/conv2d.cu
        src/cuda/conv2d_wmma.cu
        src/cuda/conv3d.cu
        src/cuda/conv2d_int8w_wmma.cu
        src/cuda/linear_int8w_wmma.cu
        src/cuda/group_norm.cu
        src/cuda/resample.cu
        src/cuda/interp2d.cu
        src/cuda/cross_attention.cu
        src/cuda/cross_attention_with_attn.cu
        src/cuda/attention_moments.cu
        src/cuda/flash_attention.cu
        src/cuda/flash_attention_backward.cu
        src/cuda/resblock.cu
        src/cuda/transpose.cu
        src/cuda/matmul.cu
        src/cuda/matmul_backward.cu
        src/cuda/rope.cu
        src/cuda/rms_norm.cu
        src/cuda/swiglu.cu
        src/cuda/kv_cache.cu
        src/cuda/public_reductions.cu
        src/cuda/ddim_step.cu
        src/cuda/euler_step.cu
        src/cuda/dpmpp_2m_step.cu
        src/cuda/timestep_embedding.cu
        src/cuda/int8_quant.cu
        src/cuda/modulate.cu
        src/cuda/self_attention_bias.cu
        src/cuda/vocoder_activations.cu
        src/cuda/log_exp_round.cu
        src/cuda/fft.cu
        src/cuda/stft.cu
        src/cuda/conv1d.cu
        src/cuda/codec_quant.cu
        src/cuda/resample1d.cu
        src/cuda/sample_logits.cu
        src/cuda/l2_norm.cu
        src/cuda/gated_delta_rule.cu
        src/cuda/spatial_merge.cu
        src/cuda/rope_mrope.cu
        src/cuda/register.cu
    )
    target_link_libraries(brotensor_cuda PUBLIC brotensor_core CUDA::cudart)
    target_compile_definitions(brotensor_cuda PUBLIC
        BROTENSOR_HAS_CUDA=1 BROTENSOR_HAS_GPU=1)
    # brotensor_core's init.cpp must know whether to probe + register CUDA.
    target_compile_definitions(brotensor_core PRIVATE
        BROTENSOR_HAS_CUDA=1 BROTENSOR_HAS_GPU=1)
    set_target_properties(brotensor_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        # Run the CUDA device link when the static lib is built so that pure
        # C++ consumers (tests, brogameagent, brodiffusion) do not need to
        # invoke nvlink themselves — without this they fail to resolve
        # __cudaRegisterLinkedBinary_* from the relocatable device code.
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# --- brotensor_metal: Metal backend (optional) ---

if(BROTENSOR_WITH_METAL)
    add_library(brotensor_metal STATIC
        src/metal/runtime.mm
        src/metal/tensor.mm
        src/metal/kernels.mm
        src/metal/gemm.mm
        src/metal/elementwise.mm
        src/metal/fp16_matmul.mm
        src/metal/softmax.mm
        src/metal/layernorm.mm
        src/metal/attention.mm
        src/metal/mha.mm
        src/metal/optim.mm
        src/metal/adam.mm
        src/metal/reduce.mm
        src/metal/loss.mm
        src/metal/embedding.mm
        src/metal/concat.mm
        src/metal/batched_ops.mm
        src/metal/conv2d.mm
        src/metal/conv2d_wmma.mm
        src/metal/conv3d.mm
        src/metal/group_norm.mm
        src/metal/resample.mm
        src/metal/interp2d.mm
        src/metal/cross_attention.mm
        src/metal/cross_attention_with_attn.mm
        src/metal/attention_moments.mm
        src/metal/flash_attention.mm
        src/metal/flash_attention_backward.mm
        src/metal/resblock.mm
        src/metal/transpose.mm
        src/metal/matmul.mm
        src/metal/matmul_backward.mm
        src/metal/rope.mm
        src/metal/rms_norm.mm
        src/metal/swiglu.mm
        src/metal/kv_cache.mm
        src/metal/public_reductions.mm
        src/metal/ddim_step.mm
        src/metal/euler_step.mm
        src/metal/dpmpp_2m_step.mm
        src/metal/timestep_embedding.mm
        src/metal/int8_quant.mm
        src/metal/modulate.mm
        src/metal/self_attention_bias.mm
        src/metal/fft.mm
        src/metal/stft.mm
        src/metal/conv1d.mm
        src/metal/vocoder_activations.mm
        src/metal/codec_quant.mm
        src/metal/resample1d.mm
        src/metal/log_exp_round.mm
        src/metal/sample_logits.mm
        src/metal/l2_norm.mm
        src/metal/gated_delta_rule.mm
        src/metal/spatial_merge.mm
        src/metal/rope_mrope.mm
        src/metal/register.mm
    )
    set_source_files_properties(
        src/metal/runtime.mm
        src/metal/tensor.mm
        src/metal/kernels.mm
        src/metal/gemm.mm
        src/metal/elementwise.mm
        src/metal/fp16_matmul.mm
        src/metal/softmax.mm
        src/metal/layernorm.mm
        src/metal/attention.mm
        src/metal/mha.mm
        src/metal/optim.mm
        src/metal/adam.mm
        src/metal/reduce.mm
        src/metal/loss.mm
        src/metal/embedding.mm
        src/metal/concat.mm
        src/metal/batched_ops.mm
        src/metal/conv2d.mm
        src/metal/conv2d_wmma.mm
        src/metal/conv3d.mm
        src/metal/group_norm.mm
        src/metal/resample.mm
        src/metal/interp2d.mm
        src/metal/cross_attention.mm
        src/metal/cross_attention_with_attn.mm
        src/metal/attention_moments.mm
        src/metal/flash_attention.mm
        src/metal/flash_attention_backward.mm
        src/metal/resblock.mm
        src/metal/transpose.mm
        src/metal/matmul.mm
        src/metal/matmul_backward.mm
        src/metal/rope.mm
        src/metal/rms_norm.mm
        src/metal/swiglu.mm
        src/metal/kv_cache.mm
        src/metal/public_reductions.mm
        src/metal/ddim_step.mm
        src/metal/euler_step.mm
        src/metal/dpmpp_2m_step.mm
        src/metal/timestep_embedding.mm
        src/metal/int8_quant.mm
        src/metal/modulate.mm
        src/metal/self_attention_bias.mm
        src/metal/fft.mm
        src/metal/stft.mm
        src/metal/conv1d.mm
        src/metal/vocoder_activations.mm
        src/metal/codec_quant.mm
        src/metal/resample1d.mm
        src/metal/log_exp_round.mm
        src/metal/sample_logits.mm
        src/metal/l2_norm.mm
        src/metal/gated_delta_rule.mm
        src/metal/spatial_merge.mm
        src/metal/rope_mrope.mm
        src/metal/register.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
            LANGUAGE OBJCXX
    )
    target_link_libraries(brotensor_metal PUBLIC
        brotensor_core
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph"
    )
    target_compile_definitions(brotensor_metal PUBLIC
        BROTENSOR_HAS_METAL=1 BROTENSOR_HAS_GPU=1)
    # brotensor_core's init.cpp must know whether to probe + register Metal.
    target_compile_definitions(brotensor_core PRIVATE
        BROTENSOR_HAS_METAL=1 BROTENSOR_HAS_GPU=1)
endif()

# --- brotensor: final consumed target (INTERFACE) ---
# Links the core + always-on CPU backend + whichever GPU backends are enabled.
# Consumers (brogameagent, brodiffusion, tests) link this one target.
#
# The backend static libs register themselves into the dispatcher from
# static-init objects (src/cpu/register.cpp) or via the extern "C" probe hook
# (src/cuda/register.cu, src/metal/register.mm). Those
# translation units export no symbol that the rest of the program references,
# so a normal static-library link drops their object files entirely and the
# backend is never registered. WHOLE_ARCHIVE forces every object file of the
# backend libs into the link so the registrars run. (CUDA's registrar is also
# reached via the extern "C" probe hook, but whole-archiving keeps the per-
# cluster fill_cuda_vtable_* objects in regardless of cross-TU reachability.)

add_library(brotensor INTERFACE)
add_library(brotensor::brotensor ALIAS brotensor)
target_link_libraries(brotensor INTERFACE
    brotensor_core
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,brotensor_cpu>"
    "$<$<BOOL:${BROTENSOR_WITH_CUDA}>:$<LINK_LIBRARY:WHOLE_ARCHIVE,brotensor_cuda>>"
    "$<$<BOOL:${BROTENSOR_WITH_METAL}>:$<LINK_LIBRARY:WHOLE_ARCHIVE,brotensor_metal>>"
)

# --- Install / export ---
option(BROTENSOR_INSTALL "Generate install targets for brotensor" ${PROJECT_IS_TOP_LEVEL})
if(BROTENSOR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(_brotensor_install_targets brotensor brotensor_core brotensor_cpu)
    if(BROTENSOR_WITH_CUDA)
        list(APPEND _brotensor_install_targets brotensor_cuda)
    endif()
    if(BROTENSOR_WITH_METAL)
        list(APPEND _brotensor_install_targets brotensor_metal)
    endif()

    install(TARGETS ${_brotensor_install_targets}
        EXPORT brotensorTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT brotensorTargets
        FILE brotensorTargets.cmake
        NAMESPACE brotensor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brotensor
    )

    set(_cfg_in "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfig.cmake.in")
    file(WRITE "${_cfg_in}"
        "@PACKAGE_INIT@\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/brotensorTargets.cmake\")\n"
    )
    configure_package_config_file("${_cfg_in}"
        "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brotensor
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brotensor
    )
endif()

# Tests
option(BROTENSOR_TESTS "Build brotensor tests" ON)
if(BROTENSOR_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Configure-time summary ---
set(_brotensor_backends "CPU")
if(BROTENSOR_WITH_CUDA)
    set(_brotensor_backends "${_brotensor_backends} + CUDA")
endif()
if(BROTENSOR_WITH_METAL)
    set(_brotensor_backends "${_brotensor_backends} + Metal")
endif()
message(STATUS "brotensor: building with backends: ${_brotensor_backends}")
