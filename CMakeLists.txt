cmake_minimum_required(VERSION 3.24)

# Pre-1.0 on purpose: the op surface grows with each sibling project that
# adopts it, and the ABI still moves. Consumers vendor this repo and build it
# from source, so the version is a label for a tag — something to pin a
# FetchContent GIT_TAG to — not a compatibility promise.
project(brotensor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build every backend static lib position-independent. Consumers vendor this
# repo and sometimes link the archives into a shared object (e.g. a Python
# extension module or a plugin .so); without PIC the CPU pool's thread-local
# `t_active` emits a local-exec TLS relocation (R_X86_64_TPOFF32) that the
# linker rejects when the archive lands in a shared object. Scoped to this
# directory's targets — it does not leak to a parent that add_subdirectory()s us.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Static CRT for a top-level build. This has to be chosen BEFORE bronze (and
# through it brass) is add_subdirectory()'d below: CMAKE_MSVC_RUNTIME_LIBRARY
# is read per target at definition time, so anything added earlier gets the
# /MD default and the api executables (which link brotensor /MT + brass +
# bronze_runtime_shared) fail with LNK2038 RuntimeLibrary mismatches.
if(MSVC AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Dependencies (cmake/bro_deps.cmake): an outer project's target wins (under bro
# these are all bro's), else the ../<name> working tree beside the top-level
# project, else the head of its main, fetched at configure.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

# bronze, the JavaScript runtime brotensor_api binds and the compiler that
# builds src/api/js/tensor.js. It brings brass, its code generator, which is
# where the `brass` target the kernel JIT block below picks up comes from. The
# api is bound across bronze's C++ embed boundary, so it has to build against
# the bronze the surrounding tree builds against.
bro_dependency(bronze GITHUB wlejon/bronze TARGET bronze_runtime_shared
               OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
endif()

# --- Coverage instrumentation (gcov / llvm-cov) ---
#
# Scoped to the CPU + core build: GPU Objective-C++ / JIT backends are not
# gcov-instrumented, and CI has no GPU to run them on anyway. So a coverage run
# measures what the CPU + parity tier reaches — the dispatcher, Tensor, the
# safetensors/GGUF readers, and the FP32 reference backend. Optimisation is dropped
# to -O0 so line attribution is not scrambled by inlining.
option(BROTENSOR_COVERAGE "Instrument the core + CPU build for gcov coverage (GCC/Clang)" OFF)
if(BROTENSOR_COVERAGE)
    if(MSVC)
        message(FATAL_ERROR
            "BROTENSOR_COVERAGE requires GCC or Clang (gcov instrumentation); MSVC is unsupported.")
    endif()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:--coverage>
        $<$<COMPILE_LANGUAGE:CXX>:-O0>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
    add_link_options(--coverage)
    message(STATUS "brotensor: coverage instrumentation ON (core + CPU, -O0)")
endif()

# CPU-backend SIMD baseline. AVX2+FMA has been the safe floor for any
# x86-64 machine capable of hosting a CUDA-class GPU dev box since ~2013 —
# targeting it (vs. the compiler's default pre-AVX baseline) lets the
# auto-vectorizer use 256-bit paths on the scalar FP32 loops in src/cpu/.
# Scoped to the CPU-heavy targets only, not forced on GPU backend TUs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
    if(MSVC)
        set(BROTENSOR_CPU_SIMD_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    else()
        set(BROTENSOR_CPU_SIMD_FLAGS
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    endif()
else()
    set(BROTENSOR_CPU_SIMD_FLAGS "")
endif()

# --- Backends ---
#
# brotensor is layered: a core dispatcher + always-on CPU backend, plus
# optional GPU backends (CUDA, Metal, Vulkan) that compile as independent static
# libraries and register themselves into the dispatcher at static-init.
# CUDA and Metal target different platforms (Linux/Windows vs. macOS);
# Metal requires macOS; CUDA compiles as pure C++ (Brass MIR -> PTX + dynamic
# driver loading) with no CUDA Toolkit. CPU has no opt-out.
option(BROTENSOR_WITH_CUDA  "Build GPU backend (CUDA)"  OFF)
option(BROTENSOR_WITH_METAL "Build GPU backend (Metal)" OFF)
# Vulkan compute (hand-written GLSL, any GPU with a stock Vulkan 1.2+ driver).
# Independent of the others; the AMD GPU path. See docs/vulkan.md.
option(BROTENSOR_WITH_VULKAN "Build GPU backend (Vulkan compute)" OFF)
# The HIP (ROCm) backend was removed on 2026-10-03 (docs/vulkan.md): a stale
# cache or a parent project still asking for it fails loudly rather than
# silently building CPU-only.
if(BROTENSOR_WITH_HIP)
    message(FATAL_ERROR
        "BROTENSOR_WITH_HIP was removed (the HIP/ROCm backend is gone); "
        "use -DBROTENSOR_WITH_VULKAN=ON for AMD GPUs and unset BROTENSOR_WITH_HIP.")
endif()

# Backends follow above options; CUDA compiles as pure C++ against Brass MIR.
if(BROTENSOR_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "BROTENSOR_WITH_METAL requires Apple platform.")
    endif()
    enable_language(OBJCXX)
endif()

# --- brotensor_core: dispatcher + public op wrappers + Tensor + runtime ---
# Always built. Sources produced by Phase 1A.

file(GLOB_RECURSE _brotensor_kernel_fp_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/gpu_mir*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/ptx_emit*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/spirv_emit*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/msl_emit*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cu_mir_builders.*"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cu_mir_cache.*")
set(_brotensor_kernel_fp_cpp "${CMAKE_CURRENT_BINARY_DIR}/generated/brotensor_kernel_fingerprint.cpp")
add_custom_command(
    OUTPUT "${_brotensor_kernel_fp_cpp}"
    COMMAND "${CMAKE_COMMAND}" "-DBROTENSOR_ROOT=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUT=${_brotensor_kernel_fp_cpp}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/brotensor_kernel_fingerprint.cmake"
    DEPENDS ${_brotensor_kernel_fp_inputs} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/brotensor_kernel_fingerprint.cmake"
    COMMENT "Fingerprinting brotensor kernel sources"
    VERBATIM)

add_library(brotensor_core STATIC
    src/tensor.cpp
    src/dispatch.cpp
    src/init.cpp
    src/ops.cpp
    src/ops_spatial.cpp
    src/ops_attention.cpp
    src/ops_quant_audio.cpp
    src/ops/fused.cpp
    src/filtered_lrelu.cpp
    src/safetensors.cpp
    src/gguf.cpp
    src/graph.cpp
    src/kernel_interop.cpp
    "${_brotensor_kernel_fp_cpp}"
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
    src/cpu/thread_pool.cpp
    src/cpu/ops_impl.cpp
    src/cpu/ops_mha_batched.cpp
    src/cpu/register.cpp
    src/cpu/register_nn.cpp
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
    src/cpu/lstm.cpp
    src/cpu/rope.cpp
    src/cpu/rms_norm.cpp
    src/cpu/conv2d.cpp
    src/cpu/deform_conv2d.cpp
    src/cpu/conv3d.cpp
    src/cpu/group_norm.cpp
    src/cpu/resample.cpp
    src/cpu/interp2d.cpp
    src/cpu/pad2d.cpp
    src/cpu/slice2d.cpp
    src/cpu/unfold2d.cpp
    src/cpu/l2_normalize.cpp
    src/cpu/convex_upsample.cpp
    src/cpu/top_k.cpp
    src/cpu/pool2d.cpp
    src/cpu/gather_rows.cpp
    src/cpu/packed_encoder.cpp
    src/cpu/linear_wq.cpp
    src/cpu/conv_transpose2d.cpp
    src/cpu/window_partition.cpp
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
    src/cpu/self_attention_decomposed_rel_pos.cpp
    src/cpu/fft.cpp
    src/cpu/stft.cpp
    src/cpu/conv1d.cpp
    src/cpu/vocoder_activations.cpp
    src/cpu/codec_quant.cpp
    src/cpu/resample1d.cpp
    src/cpu/log_exp_round.cpp
    src/cpu/sample_logits.cpp
    src/cpu/masked_diffusion.cpp
    src/cpu/noise.cpp
    src/cpu/l2_norm.cpp
    src/cpu/gated_delta_rule.cpp
    src/cpu/spatial_merge.cpp
    src/cpu/rope_mrope.cpp
    src/cpu/batch_norm.cpp
    src/cpu/image_preproc.cpp
    src/cpu/stylegan_elementwise.cpp
    src/cpu/bias_act.cpp
    src/cpu/upfirdn2d.cpp
    src/cpu/modulated_conv2d.cpp
    src/cpu/cpu_jit.cpp
    src/cpu/cpu_jit_fused.cpp
    src/cpu/cpu_jit_quant.cpp
    src/cpu/cpu_quant_q6k_int8.cpp
    src/jit/trace_dag.cpp
    src/jit/trace_cache.cpp
    src/jit/kernel_disk_cache.cpp
    src/jit/kernel_cache_api.cpp
    src/jit/trace_compiler_cpu.cpp
    src/jit/trace_plan.cpp
    src/jit/trace_eager.cpp
    src/jit/trace_ops.cpp
    src/jit/trace_autodiff.cpp
    src/jit/trace_autodiff_exec.cpp
)
target_link_libraries(brotensor_cpu PUBLIC brotensor_core Threads::Threads)
if(MSVC)
    target_compile_options(brotensor_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brotensor_cpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()
target_compile_options(brotensor_cpu PRIVATE ${BROTENSOR_CPU_SIMD_FLAGS})

# --- Brass JIT acceleration (brass arrives with bronze, above) ---
# A brass with no code generator for this target (BRASS_HOST_BACKEND OFF) is
# no JIT: brotensor_cpu would compile every trace it is handed into bytes the
# CPU cannot execute, and the first op through it is SIGBUS. Such a build
# takes the plain kernels, as if brass were absent.
# The CPU JIT builds brass MIR, not machine code: its f32x8 vectors are AVX2
# ymm registers on x86_64 and NEON q-register pairs on AArch64, so it runs on
# every architecture brass generates code for (x86_64 and AArch64, the same
# list brass's own BRASS_HOST_BACKEND test uses).
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(_brotensor_jit_arch "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(_brotensor_jit_arch "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(_brotensor_jit_arch MATCHES "^(x86_64|AMD64|amd64|x64|aarch64|arm64|ARM64)$")
    set(_brotensor_jit_arch_ok ON)
else()
    set(_brotensor_jit_arch_ok OFF)
endif()
if((TARGET brass AND NOT BRASS_HOST_BACKEND) OR NOT _brotensor_jit_arch_ok)
    message(STATUS "brotensor: brass has no code generator for ${_brotensor_jit_arch}; CPU JIT off")
elseif(TARGET brass)
    message(STATUS "brotensor: found Brass target")
    # BUILD_INTERFACE-only: brass is a sibling add_subdirectory target that is
    # not part of brotensor's export set, and install(EXPORT) refuses any
    # target whose (LINK_ONLY) interface names one it cannot export. The
    # installed package therefore does not carry the brass link; a consumer of
    # the install tree links brass itself.
    set(BROTENSOR_BRASS_LIB "$<BUILD_INTERFACE:brass>")
    target_link_libraries(brotensor_cpu PRIVATE ${BROTENSOR_BRASS_LIB})
    target_compile_definitions(brotensor_cpu PUBLIC BROTENSOR_HAS_BRASS_JIT=1)
    set(BROTENSOR_HAS_BRASS_JIT 1)
    target_sources(brotensor_cpu PRIVATE
        src/backward_index_verify.cpp
        src/backward_shader_audit.cpp
        src/backward_shader_audit_extra.cpp
        src/backward_nan_verify.cpp
        src/jit/gpu_mir_emit_common.cpp
        src/jit/gpu_mir_emit_elementwise.cpp
        src/jit/gpu_mir_emit_rownorm.cpp
        src/jit/ptx_emit.cpp
        src/jit/msl_emit.cpp
        src/kernels/mir_math.cpp
        src/kernels/mir_elementwise.cpp
        src/kernels/mir_reduce.cpp
        src/kernels/mir_norm.cpp
        src/kernels/mir_rope.cpp
        src/kernels/mir_copy.cpp
        src/kernels/mir_scatter.cpp
        src/kernels/mir_gnorm.cpp
        src/kernels/mir_diffusion.cpp
        src/kernels/mir_remap.cpp
        src/kernels/mir_resample.cpp
        src/kernels/mir_resample_bwd.cpp
        src/kernels/mir_lstm.cpp
        src/kernels/mir_delta.cpp
        src/kernels/mir_xent.cpp
        src/kernels/mir_topk.cpp
        src/kernels/mir_select.cpp
        src/kernels/mir_misc.cpp
        src/kernels/mir_vision.cpp
        src/kernels/mir_audio.cpp
        src/kernels/mir_spectral.cpp
        src/kernels/mir_attn_small.cpp
        src/kernels/mir_conv_direct.cpp
        src/kernels/mir_conv_aux.cpp
        src/kernels/mir_conv_igemm_f32.cpp
        src/kernels/mir_conv_int8w.cpp
        src/kernels/mir_conv_mma.cpp
        src/kernels/mir_filtered_lrelu.cpp
        src/kernels/mir_gemm_mma.cpp
        src/kernels/mir_gemm_mma_nn.cpp
        src/kernels/mir_gemm_quant_mma.cpp
        src/kernels/mir_gemm_skinny.cpp
        src/kernels/mir_gemm_splitk.cpp
        src/kernels/mir_linear_bwd.cpp
        src/kernels/mir_quant_helpers.cpp
        src/kernels/mir_gemv.cpp
        src/kernels/mir_gemv_q.cpp
        src/kernels/mir_gemm_simt.cpp
        src/kernels/mir_decode_gqa.cpp
        src/kernels/mir_fa_aux.cpp
        src/kernels/mir_fa_fused_mma.cpp
        src/kernels/mir_fa_packed_mma.cpp
        src/kernels/mir_fa_rows.cpp
        src/kernels/mir_fa_varlen_f32.cpp
        src/kernels/mir_fa_bwd.cpp
        src/kernels/mir_dft64.cpp
        src/kernels/mir_coop_common.cpp
        src/kernels/mir_gemm_cm.cpp
        src/kernels/mir_conv_cm.cpp
        src/kernels/mir_fa_cm.cpp
        src/kernels/mir_fa_bwd_cm.cpp
    )
    if(BRASS_WITH_SPIRV)
        target_sources(brotensor_cpu PRIVATE
            src/jit/spirv_emit.cpp
        )
    endif()
else()
    message(STATUS "brotensor: Brass not found; Brass CPU JIT disabled")
endif()

# --- brotensor_cuda: CUDA backend (optional) ---

if(BROTENSOR_WITH_CUDA)
    if(NOT TARGET brass)
        message(FATAL_ERROR
            "BROTENSOR_WITH_CUDA=ON requires Brass (every CUDA kernel compiles from Brass MIR to PTX).")
    endif()
    add_library(brotensor_cuda STATIC
        src/cuda/runtime.cpp
        src/cuda/tensor.cpp
        src/cuda/cuda_graph.cpp
        src/cuda/gemm.cpp
        src/cuda/fp16_matmul.cpp
        src/cuda/cu_mir_cache.cpp
        src/cuda/cu_mir_builders.cpp
        src/cuda/ops_attention_bwd.cpp
        src/cuda/ops_attention_proj.cpp
        src/cuda/ops_attention_relpos.cpp
        src/cuda/ops_audio.cpp
        src/cuda/ops_conv_direct.cpp
        src/cuda/ops_copy.cpp
        src/cuda/ops_delta.cpp
        src/cuda/ops_diffusion.cpp
        src/cuda/ops_elementwise.cpp
        src/cuda/ops_gnorm.cpp
        src/cuda/ops_lstm.cpp
        src/cuda/ops_misc.cpp
        src/cuda/ops_norm.cpp
        src/cuda/ops_quant.cpp
        src/cuda/ops_gemm_mir.cpp
        src/cuda/ops_gemm_mma_mir.cpp
        src/cuda/ops_fa_mir.cpp
        src/cuda/ops_reduce.cpp
        src/cuda/ops_rope.cpp
        src/cuda/ops_sampling.cpp
        src/cuda/ops_scatter.cpp
        src/cuda/ops_spatial.cpp
        src/cuda/ops_spatial_bwd.cpp
        src/cuda/ops_spectral.cpp
        src/cuda/ops_topk.cpp
        src/cuda/ops_vision.cpp
        src/cuda/ops_xent.cpp
        src/cuda/ops_concat.cpp
        src/cuda/ops_batched.cpp
        src/cuda/conv2d.cpp
        src/cuda/conv2d_wmma.cpp
        src/cuda/ops_conv_transpose2d.cpp
        src/cuda/conv3d.cpp
        src/cuda/ops_quant_mma.cpp
        src/cuda/flash_attention_varlen_f32.cpp
        src/cuda/conv2d_igemm_f32.cpp
        src/cuda/flash_attention.cpp
        src/cuda/ops_fa_qkvo.cpp
        src/cuda/ops_fa_varlen_win.cpp
        src/cuda/flash_attention_fused.cpp
        src/cuda/flash_attention_backward.cpp
        src/cuda/flash_attention_packed.cpp
        src/cuda/ops_fa_packed_bwd.cpp
        src/cuda/gemm_mma.cpp
        src/cuda/gemm_skinny.cpp
        src/cuda/ops_linear_ex.cpp
        src/cuda/resblock.cpp
        src/cuda/matmul.cpp
        src/cuda/ops_matmul_bwd.cpp
        src/cuda/ops_kv_cache.cpp
        src/cuda/decode_gqa.cpp
        src/cuda/int8_quant.cpp
        src/cuda/ops_filtered_lrelu.cpp
        src/cuda/register.cpp
        src/cuda/cuda_driver.cpp
        src/cuda/cuda_jit.cpp
        src/cuda/cuda_jit_gemv.cpp
        src/cuda/cuda_graph_jit.cpp
        src/cuda/cuda_jit_quant.cpp
        src/cuda/kernel_interop_cuda.cpp
        src/jit/trace_compiler_cuda.cpp
    )
    # Dynamic CUDA Driver API: zero CUDA Toolkit headers, zero cudart_static,
    # and zero hard DT_NEEDED / import on libcuda.so.1 / nvcuda.dll.
    # src/cuda/cuda_driver.cpp opens the driver library lazily on first use so
    # one binary runs on GPU machines and falls back cleanly on CPU-only hosts.
    target_link_libraries(brotensor_cuda PUBLIC brotensor_core)
    if(NOT WIN32)
        target_link_libraries(brotensor_cuda PRIVATE ${CMAKE_DL_LIBS})
    endif()
    if(TARGET brass)
        if(BROTENSOR_BRASS_LIB)
            target_link_libraries(brotensor_cuda PUBLIC "${BROTENSOR_BRASS_LIB}")
        endif()
        target_compile_definitions(brotensor_cuda PUBLIC BROTENSOR_HAS_BRASS_CUDA_JIT=1)
    endif()
    target_compile_definitions(brotensor_cuda PUBLIC
        BROTENSOR_HAS_CUDA=1 BROTENSOR_HAS_GPU=1)
    # brotensor_core's init.cpp probes + registers CUDA and dispatch.cpp calls
    # brotensor_cuda_set_device() exported from brotensor_cuda.
    target_compile_definitions(brotensor_core PRIVATE
        BROTENSOR_HAS_CUDA=1 BROTENSOR_HAS_GPU=1)
endif()

# --- brotensor_metal: Metal backend (optional) ---

if(BROTENSOR_WITH_METAL)
    set(_BROTENSOR_METAL_SRCS
        src/metal/runtime.mm
        src/metal/tensor.mm
        src/metal/kernels.mm
        src/metal/gemm.mm
        src/metal/mtl_mir_cache.mm
        src/metal/elementwise.mm
        src/metal/fp16_matmul.mm
        src/metal/gemm_fp32.mm
        src/metal/linear_ex.mm
        src/metal/linear_wq.mm
        src/metal/packed_encoder.mm
        src/metal/layernorm.mm
        src/metal/attention.mm
        src/metal/attention_bwd.mm
        src/metal/loss.mm
        src/metal/misc.mm
        src/metal/vision.mm
        src/metal/embedding.mm
        src/metal/concat.mm
        src/metal/batched_ops.mm
        src/metal/conv2d.mm
        src/metal/conv2d_wmma.mm
        src/metal/conv3d.mm
        src/metal/group_norm.mm
        src/metal/resample.mm
        src/metal/resample_bwd.mm
        src/metal/diffusion.mm
        src/metal/flash_attention.mm
        src/metal/flash_attention_backward.mm
        src/metal/resblock.mm
        src/metal/matmul.mm
        src/metal/matmul_backward.mm
        src/metal/rope.mm
        src/metal/kv_cache.mm
        src/metal/public_reductions.mm
        src/metal/int8_quant.mm
        src/metal/audio.mm
        src/metal/spectral.mm
        src/metal/sample_logits.mm
        src/metal/flash_attention_windowed.mm
        src/metal/gated_delta_rule.mm
        src/metal/top_k.mm
        src/metal/conv_transpose2d.mm
        src/metal/q4k.mm
        src/metal/q6k.mm
        src/metal/q8_0.mm
        src/metal/filtered_lrelu.mm
        src/metal/lstm.mm
        src/metal/kernel_interop_mtl.mm
        src/metal/register.mm
    )
    add_library(brotensor_metal STATIC ${_BROTENSOR_METAL_SRCS})
    set_source_files_properties(${_BROTENSOR_METAL_SRCS}
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
    if(BROTENSOR_HAS_BRASS_JIT)
        target_sources(brotensor_metal PRIVATE
            src/jit/trace_compiler_metal.mm
        )
        set_source_files_properties(src/jit/trace_compiler_metal.mm PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
            LANGUAGE OBJCXX
        )
        target_link_libraries(brotensor_metal PUBLIC "${BROTENSOR_BRASS_LIB}")
        target_compile_definitions(brotensor_metal PUBLIC
            BROTENSOR_HAS_BRASS_JIT=1 BROTENSOR_HAS_METAL_TRACE_JIT=1)
    endif()
    # brotensor_core's init.cpp must know whether to probe + register Metal.
    target_compile_definitions(brotensor_core PRIVATE
        BROTENSOR_HAS_METAL=1 BROTENSOR_HAS_GPU=1)
endif()

# --- brotensor_vulkan: Vulkan compute backend (optional) ---
# Target, shader compilation and embedding: cmake/BrotensorVulkan.cmake.
if(BROTENSOR_WITH_VULKAN)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BrotensorVulkan.cmake")
    brotensor_vulkan_add_backend()
    target_include_directories(brotensor_cpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders")
    target_compile_definitions(brotensor_cpu PUBLIC BROTENSOR_HAS_VULKAN=1 BROTENSOR_HAS_SHADER_TABLE=1)
    add_dependencies(brotensor_cpu brotensor_vulkan)
elseif(BROTENSOR_HAS_BRASS_JIT)
    find_program(_brotensor_audit_glslc NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
    if(_brotensor_audit_glslc)
        include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BrotensorVulkan.cmake")
        brotensor_vulkan_add_shader_table_to_target(brotensor_cpu)
        target_compile_definitions(brotensor_cpu PUBLIC BROTENSOR_HAS_SHADER_TABLE=1)
    endif()
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
    "$<$<BOOL:${BROTENSOR_WITH_VULKAN}>:$<LINK_LIBRARY:WHOLE_ARCHIVE,brotensor_vulkan>>"
)

add_subdirectory(src/api)

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
    if(BROTENSOR_WITH_VULKAN)
        list(APPEND _brotensor_install_targets brotensor_vulkan)
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

    # SameMinorVersion, not the usual SameMajorVersion: under semver a 0.x minor
    # bump is allowed to break, so 0.1 must not satisfy find_package(brotensor 0.2).
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/brotensorConfigVersion.cmake"
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
if(BROTENSOR_WITH_VULKAN)
    set(_brotensor_backends "${_brotensor_backends} + Vulkan")
endif()
message(STATUS "brotensor: building with backends: ${_brotensor_backends}")
