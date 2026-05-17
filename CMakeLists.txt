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

# --- GPU backends (mutually exclusive) ---
#
# brotensor is a GPU-only tensor + ops library. Exactly one backend must be
# selected; flat brotensor:: namespace and identical op signatures across
# backends. Consumer code keys off the umbrella BROTENSOR_HAS_GPU define;
# backend-specific impl keys off BROTENSOR_HAS_CUDA / BROTENSOR_HAS_METAL.
option(BROTENSOR_WITH_CUDA  "Build GPU backend (CUDA)"  OFF)
option(BROTENSOR_WITH_METAL "Build GPU backend (Metal)" OFF)
if(BROTENSOR_WITH_CUDA AND BROTENSOR_WITH_METAL)
    message(FATAL_ERROR
        "BROTENSOR_WITH_CUDA and BROTENSOR_WITH_METAL are mutually exclusive.")
endif()
if(NOT BROTENSOR_WITH_CUDA AND NOT BROTENSOR_WITH_METAL)
    message(FATAL_ERROR
        "brotensor requires one of: -DBROTENSOR_WITH_CUDA=ON or -DBROTENSOR_WITH_METAL=ON")
endif()

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

# --- brotensor library ---

add_library(brotensor STATIC)
add_library(brotensor::brotensor ALIAS brotensor)

target_include_directories(brotensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brotensor PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brotensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brotensor PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

if(BROTENSOR_WITH_CUDA)
    target_sources(brotensor PRIVATE
        src/cuda/runtime.cu
        src/cuda/tensor.cu
        src/cuda/gemm.cu
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
        src/cuda/group_norm.cu
        src/cuda/resample.cu
        src/cuda/cross_attention.cu
        src/cuda/flash_attention.cu
        src/cuda/resblock.cu
        src/cuda/transpose.cu
    )
    target_link_libraries(brotensor PUBLIC CUDA::cudart)
    target_compile_definitions(brotensor PUBLIC BROTENSOR_HAS_CUDA=1 BROTENSOR_HAS_GPU=1)
    set_target_properties(brotensor PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

if(BROTENSOR_WITH_METAL)
    target_sources(brotensor PRIVATE
        src/metal/runtime.mm
        src/metal/tensor.mm
        src/metal/device_buffer.mm
        src/metal/kernels.mm
        src/metal/gemm.mm
        src/metal/elementwise.mm
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
        src/metal/group_norm.mm
        src/metal/resample.mm
        src/metal/cross_attention.mm
        src/metal/flash_attention.mm
        src/metal/resblock.mm
        src/metal/transpose.mm
    )
    set_source_files_properties(
        src/metal/runtime.mm
        src/metal/tensor.mm
        src/metal/device_buffer.mm
        src/metal/kernels.mm
        src/metal/gemm.mm
        src/metal/elementwise.mm
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
        src/metal/group_norm.mm
        src/metal/resample.mm
        src/metal/cross_attention.mm
        src/metal/flash_attention.mm
        src/metal/resblock.mm
        src/metal/transpose.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
            LANGUAGE OBJCXX
    )
    target_link_libraries(brotensor PUBLIC
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph"
    )
    target_compile_definitions(brotensor PUBLIC BROTENSOR_HAS_METAL=1 BROTENSOR_HAS_GPU=1)
endif()

# --- Install / export ---
option(BROTENSOR_INSTALL "Generate install targets for brotensor" ${PROJECT_IS_TOP_LEVEL})
if(BROTENSOR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS brotensor
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
