# Test layout:
#   * test_cpu_ops.cpp        — standalone CPU op coverage. Always built.
#   * test_dispatch.cpp       — runtime backend dispatch / registration /
#                                DeviceScope coverage. Always built.
#   * test_cpu_gpu_parity.cpp — CPU↔GPU parity. Only built when a GPU backend
#                                is enabled.
#   * everything else         — GPU smoke tests over the unified Tensor API;
#                                they construct CUDA-resident tensors and call
#                                device-neutral ops, and only build when a GPU
#                                backend is on. Each guards on is_available().

function(brotensor_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brotensor)
    if(BROTENSOR_WITH_CUDA)
        target_include_directories(${name} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
endfunction()

function(brotensor_add_test name)
    brotensor_add_executable(${name} ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- CPU tests (always built) ---
if(BROTENSOR_HAS_BRASS_JIT)
    brotensor_add_test(brotensor_test_jit_trace test_jit_trace.cpp)
    brotensor_add_test(brotensor_test_cpu_jit test_cpu_jit.cpp)
    brotensor_add_test(brotensor_test_cpu_jit_quant test_cpu_jit_quant.cpp)
endif()
# --- Bronze API test ---
#
# Boots a bronze realm, installs the api exactly as bro's host does, and
# checks `bro.tensor` from JavaScript. The JavaScript half is js/tensor.js
# compiled by the bronze CLI, which needs a brass code generator for the
# target (BRASS_HOST_BACKEND; Apple Silicon has none yet), so there the api
# carries a no-op entry instead (src/api/CMakeLists.txt) and there is nothing
# for this test to see: it is not registered there rather than registered to
# fail. Everything else in this directory still runs.
if(NOT BROTENSOR_API_COMPILED_JS)
    message(STATUS "brotensor: brotensor_test_api not registered (no compiled JS on this target)")
endif()
if(BROTENSOR_API_COMPILED_JS)
add_executable(brotensor_test_api
    test_api.cpp
    test_api_batched.cpp
    test_api_attn2.cpp
    test_api_audio.cpp
    test_api_conv.cpp
    test_api_int8.cpp
    test_api_misc.cpp
    test_api_extra.cpp
    test_api_bounds.cpp
)
# brass FIRST, and /INCLUDE a symbol only brass's GC object defines: both
# brass.lib and bronze_runtime_shared.dll define brass_gc_write_barrier, and
# the bronze-compiled tensor.o inside brotensor_api references it the way a
# loaded module would. Naming brass first and forcing its GC object in before
# any static reference exists makes MSVC resolve the barrier against brass
# and never pull the DLL's member — LNK2005 otherwise. Same recipe as bro's
# bronze_host (BRO_BRASS_LIB + /INCLUDE:brass_gc_card_table_base).
target_link_libraries(brotensor_test_api PRIVATE ${BROTENSOR_BRASS_LIB} brotensor_api bronze::runtime_shared)
if(MSVC)
    target_link_options(brotensor_test_api PRIVATE "/INCLUDE:brass_gc_card_table_base")
endif()
# bronze's shared runtime is built into BRONZE_SHARED_RUNTIME_DIR, not beside
# this executable, and the PE loader only looks beside the .exe (0xc0000135 +
# a modal "dll was not found" otherwise). Stage it next to the test as a
# post-build step and put both dirs on ctest's PATH.
if(WIN32)
    add_custom_command(TARGET brotensor_test_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:bronze_runtime_shared>
            $<TARGET_FILE_DIR:brotensor_test_api>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_LINKER_FILE:bronze_runtime_shared>
            $<TARGET_FILE_DIR:brotensor_test_api>
    )
endif()
add_test(NAME brotensor_test_api COMMAND brotensor_test_api)
# The same run with a collection at every allocation and freed space
# poisoned: a binding that holds a raw Value across an allocating call reads
# garbage here instead of passing by luck. `ctest -LE gcstress` skips it.
add_test(NAME brotensor_test_api_gcstress COMMAND brotensor_test_api)
if(WIN32)
    set_tests_properties(brotensor_test_api PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brotensor_test_api>;$ENV{PATH}")
    set_tests_properties(brotensor_test_api_gcstress PROPERTIES
        ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1;PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brotensor_test_api>;$ENV{PATH}"
        LABELS gcstress)
else()
    set_tests_properties(brotensor_test_api_gcstress PROPERTIES
        ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1"
        LABELS gcstress)
endif()
endif() # BROTENSOR_API_COMPILED_JS

brotensor_add_test(brotensor_test_cpu_ops test_cpu_ops.cpp)
brotensor_add_test(brotensor_test_dispatch test_dispatch.cpp)
brotensor_add_test(brotensor_test_thread_pool test_thread_pool.cpp)
brotensor_add_test(brotensor_test_safetensors test_safetensors.cpp)
brotensor_add_test(brotensor_test_gguf test_gguf.cpp)
brotensor_add_test(brotensor_test_bf16_basics test_bf16_basics.cpp)
brotensor_add_test(brotensor_test_fft test_fft.cpp)
brotensor_add_test(brotensor_test_stft test_stft.cpp)
brotensor_add_test(brotensor_test_conv1d test_conv1d.cpp)
brotensor_add_test(brotensor_test_lstm test_lstm.cpp)
brotensor_add_test(brotensor_test_lora test_lora.cpp)
brotensor_add_test(brotensor_test_vocoder_activations test_vocoder_activations.cpp)
brotensor_add_test(brotensor_test_codec_quant test_codec_quant.cpp)
brotensor_add_test(brotensor_test_resample1d test_resample1d.cpp)
brotensor_add_test(brotensor_test_log_exp_round test_log_exp_round.cpp)
brotensor_add_test(brotensor_test_sample_logits test_sample_logits.cpp)
# CPU checks always; the CUDA parity half runs when that backend registers.
brotensor_add_test(brotensor_test_masked_diffusion test_masked_diffusion.cpp)
brotensor_add_test(brotensor_test_noise test_noise.cpp)
brotensor_add_test(brotensor_test_gated_delta_rule test_gated_delta_rule.cpp)
brotensor_add_test(brotensor_test_conv3d test_conv3d.cpp)
brotensor_add_test(brotensor_test_flash_attention_varlen test_flash_attention_varlen.cpp)
brotensor_add_test(brotensor_test_flash_attention_varlen_backward test_flash_attention_varlen_backward.cpp)
brotensor_add_test(brotensor_test_spatial_merge test_spatial_merge.cpp)
brotensor_add_test(brotensor_test_rope_mrope test_rope_mrope.cpp)
brotensor_add_test(brotensor_test_rope_perhead test_rope_perhead.cpp)
brotensor_add_test(brotensor_test_interp2d test_interp2d.cpp)
brotensor_add_test(brotensor_test_interp2d_align_corners test_interp2d_align_corners.cpp)
brotensor_add_test(brotensor_test_pad2d test_pad2d.cpp)
brotensor_add_test(brotensor_test_slice2d test_slice2d.cpp)
brotensor_add_test(brotensor_test_top_k test_top_k.cpp)
brotensor_add_test(brotensor_test_pool2d test_pool2d.cpp)
brotensor_add_test(brotensor_test_gather_rows test_gather_rows.cpp)
brotensor_add_test(brotensor_test_conv_transpose2d test_conv_transpose2d.cpp)
brotensor_add_test(brotensor_test_window_partition test_window_partition.cpp)
brotensor_add_test(brotensor_test_batch_norm test_batch_norm.cpp)
brotensor_add_test(brotensor_test_image_preproc test_image_preproc.cpp)
brotensor_add_test(brotensor_test_bce_with_logits test_bce_with_logits.cpp)
brotensor_add_test(brotensor_test_self_attention_decomposed_rel_pos test_self_attention_decomposed_rel_pos.cpp)
brotensor_add_test(brotensor_test_stylegan_primitives test_stylegan_primitives.cpp)
brotensor_add_test(brotensor_test_bias_act test_bias_act.cpp)
brotensor_add_test(brotensor_test_upfirdn2d test_upfirdn2d.cpp)
brotensor_add_test(brotensor_test_modulated_conv2d test_modulated_conv2d.cpp)
brotensor_add_test(brotensor_test_filtered_lrelu test_filtered_lrelu.cpp)
brotensor_add_test(brotensor_test_stylegan_parity test_stylegan_parity.cpp)
brotensor_add_test(brotensor_test_softmax_rows test_softmax_rows.cpp)
brotensor_add_test(brotensor_test_add_channel_bias test_add_channel_bias.cpp)
brotensor_add_test(brotensor_test_pixel_shuffle_upsample test_pixel_shuffle_upsample.cpp)
brotensor_add_test(brotensor_test_layernorm_stability test_layernorm_stability.cpp)
brotensor_add_test(brotensor_test_modernbert_ops test_modernbert_ops.cpp)

# Coverage-driven: the serializers' writer + error paths, the FP16/BF16 CPU
# matmul_abt (whose only other test is GPU-gated), and the Tensor / dispatch
# error paths. All CPU-only.
brotensor_add_test(brotensor_test_safetensors_extra test_safetensors_extra.cpp)
brotensor_add_test(brotensor_test_gguf_extra test_gguf_extra.cpp)
brotensor_add_test(brotensor_test_matmul_abt_cpu test_matmul_abt_cpu.cpp)
brotensor_add_test(brotensor_test_tensor_errors test_tensor_errors.cpp)

# --- GPU tests (CUDA or Metal) ---
set(BROTENSOR_HAS_GPU OFF)
if(BROTENSOR_WITH_CUDA OR BROTENSOR_WITH_METAL)
    set(BROTENSOR_HAS_GPU ON)
endif()

if(BROTENSOR_HAS_GPU)
    brotensor_add_test(brotensor_test test_tensor.cpp)
    brotensor_add_test(brotensor_test_conv2d test_conv2d.cpp)
    brotensor_add_test(brotensor_test_group_norm test_group_norm.cpp)
    brotensor_add_test(brotensor_test_activations test_activations.cpp)
    brotensor_add_test(brotensor_test_resample test_resample.cpp)
    brotensor_add_test(brotensor_test_cross_attention test_cross_attention.cpp)
    brotensor_add_test(brotensor_test_cross_attention_with_attn test_cross_attention_with_attn.cpp)
    brotensor_add_test(brotensor_test_attention_moments test_attention_moments.cpp)
    brotensor_add_test(brotensor_test_fp16_basics test_fp16_basics.cpp)
    brotensor_add_test(brotensor_test_flash_attention test_flash_attention.cpp)
    brotensor_add_test(brotensor_test_resblock test_resblock.cpp)
    brotensor_add_test(brotensor_test_resblock_int8w test_resblock_int8w.cpp)
    brotensor_add_test(brotensor_test_diffusion_primitives test_diffusion_primitives.cpp)
    brotensor_add_test(brotensor_test_transpose test_transpose.cpp)
    brotensor_add_test(brotensor_test_large_elementwise test_large_elementwise.cpp)
    brotensor_add_test(brotensor_test_matmul test_matmul.cpp)
    brotensor_add_test(brotensor_test_matmul_abt test_matmul_abt.cpp)
    brotensor_add_test(brotensor_test_matmul_backward test_matmul_backward.cpp)
    brotensor_add_test(brotensor_test_flash_attention_backward test_flash_attention_backward.cpp)
    brotensor_add_test(brotensor_test_rope test_rope.cpp)
    brotensor_add_test(brotensor_test_rms_norm test_rms_norm.cpp)
    brotensor_add_test(brotensor_test_swiglu test_swiglu.cpp)
    brotensor_add_test(brotensor_test_kv_cache test_kv_cache.cpp)
    brotensor_add_test(brotensor_test_public_reductions test_public_reductions.cpp)
    brotensor_add_test(brotensor_test_ddim_step test_ddim_step.cpp)
    brotensor_add_test(brotensor_test_sdxl_schedulers test_sdxl_schedulers.cpp)
    brotensor_add_test(brotensor_test_int8_quant test_int8_quant.cpp)
    brotensor_add_test(brotensor_test_int8_conv_wmma test_int8_conv_wmma.cpp)
    brotensor_add_test(brotensor_test_int8_linear_wmma test_int8_linear_wmma.cpp)
    brotensor_add_test(brotensor_test_q4k_parity test_q4k_parity.cpp)
    brotensor_add_test(brotensor_test_q8_0_parity test_q8_0_parity.cpp)
    brotensor_add_test(brotensor_test_q6k_parity test_q6k_parity.cpp)
    # Backend-neutral (runs on Metal too): quantized batched-linear prefill
    # fast path vs the trusted per-token GEMV path.
    brotensor_add_test(brotensor_test_quant_prefill_parity test_quant_prefill_parity.cpp)
    # Hand-run microbench for the matmul_abt simdgroup fast path (not a ctest);
    # runs on CUDA or Metal.
    brotensor_add_executable(brotensor_bench_matmul_abt bench_matmul_abt.cpp)
    brotensor_add_test(brotensor_test_int8_attention test_int8_attention.cpp)
    if(BROTENSOR_WITH_CUDA)
        brotensor_add_test(brotensor_test_streams test_streams.cpp)
        # Hand-run microbench for the plain FP16 matmul WMMA path (not a ctest).
        brotensor_add_executable(brotensor_bench_matmul_rm bench_matmul_rm.cpp)
    endif()

    # CPU↔GPU parity (GPU-only).
    brotensor_add_test(brotensor_test_cpu_gpu_parity test_cpu_gpu_parity.cpp)

    # Per-op CPU↔GPU parity suite — one executable per op group so ctest
    # surfaces failures individually. Complements the monolithic
    # test_cpu_gpu_parity.cpp above; parity_helpers.h is the shared harness.
    set(_brotensor_parity_tests
        linear elementwise softmax layernorm attention optim adam
        reduce loss embedding concat mha batched
        elementwise_extra loss_fused embedding_extra concat_extra
        reduce_extra layernorm_inference layernorm_batched_caches
        activations geglu swiglu matmul matmul_abt rope rmsnorm lstm lora
        conv2d deform_conv2d conv3d group_norm
        resample transpose diffusion kv_cache
        cross_attention self_attention self_attention_decomposed_rel_pos attention_moments
        flash_attention flash_attention_gqa flash_attention_varlen flash_attention_varlen_backward resblock
        spatial_merge patch_unpack rope_mrope
        batched_backward tensor_ops
        modulate
        fft stft
        conv1d vocoder
        codec_quant resample1d log_exp_round sample_logits
        interp2d
        l2_norm gated_delta_rule
        pad2d slice2d unfold2d l2_normalize convex_upsample top_k pool2d gather_rows
        conv_transpose2d window_partition
        batch_norm image_preproc
        softmax_rows add_channel_bias pixel_shuffle_upsample
        masked_mean_pool
        packed_encoder linear_ex
    )
    foreach(_name IN LISTS _brotensor_parity_tests)
        brotensor_add_test(brotensor_test_${_name}_parity test_${_name}_parity.cpp)
    endforeach()

    # Diffusion micro-benchmark. Built but not added to ctest — invoke
    # manually from the build dir (./tests/Release/brotensor_bench_diffusion).
    # CUDA-only: it uses cudaEvent_t for timing.
    if(BROTENSOR_WITH_CUDA)
        brotensor_add_executable(brotensor_bench_diffusion bench_diffusion.cpp)
        brotensor_add_executable(brotensor_bench_filtered_lrelu bench_filtered_lrelu.cpp)
        brotensor_add_executable(brotensor_bench_modulated_conv2d bench_modulated_conv2d.cpp)
        brotensor_add_executable(brotensor_bench_decode_kernels bench_decode_kernels.cpp)
        brotensor_add_executable(brotensor_bench_sam_attention bench_sam_attention.cpp)
        brotensor_add_executable(brotensor_bench_conv_annotators bench_conv_annotators.cpp)
        brotensor_add_executable(brotensor_bench_conv_unet bench_conv_unet.cpp)
        brotensor_add_executable(brotensor_bench_flow_attention bench_flow_attention.cpp)
        # Bandwidth baseline for the elementwise/norm family, the quantized
        # linear paths, and the causal-vs-fused attention gap. These three
        # cover the ops with no prior perf instrumentation.
        brotensor_add_executable(brotensor_bench_elementwise bench_elementwise.cpp)
        brotensor_add_executable(brotensor_bench_quant_matmul bench_quant_matmul.cpp)
        brotensor_add_executable(brotensor_bench_causal_attention bench_causal_attention.cpp)
        # CUDA graph capture/replay (the API is CUDA-only).
        brotensor_add_test(brotensor_test_cuda_graph test_cuda_graph.cpp)
        # Multi-GPU P2P transfer and pipeline execution test.
        brotensor_add_test(brotensor_test_multigpu test_multigpu.cpp)
        # Brass CUDA JIT benchmark & parity test suite
        # Typed (BF16/FP16) I/O, row/scalar broadcast operands, tanh/sigmoid,
        # and the fused norm+chain kernels — the trace JIT surface the DiT and
        # VAE adoption depends on.
        brotensor_add_test(brotensor_test_jit_trace_dtypes test_jit_trace_dtypes.cpp)
        # Per-pattern eager / hand-fused / JIT bandwidth comparison. Built but
        # not a ctest: invoke ./tests/Release/brotensor_bench_jit_fusion.
        brotensor_add_executable(brotensor_bench_jit_fusion bench_jit_fusion.cpp)
        brotensor_add_test(brotensor_test_cuda_jit test_cuda_jit.cpp)
        if(BRASS_ROOT AND EXISTS "${BRASS_ROOT}/include")
            target_include_directories(brotensor_test_cuda_jit PRIVATE "${BRASS_ROOT}/include")
            if(BROTENSOR_BRASS_LIB)
                target_link_libraries(brotensor_test_cuda_jit PRIVATE "${BROTENSOR_BRASS_LIB}")
            endif()
        endif()
        target_include_directories(brotensor_test_cuda_jit PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)
        target_link_libraries(brotensor_test_cuda_jit PRIVATE CUDA::cuda_driver)

        # Fused GEMV JIT benchmark & parity suite
        brotensor_add_test(brotensor_test_cuda_jit_gemv test_cuda_jit_gemv.cpp)
        if(BRASS_ROOT AND EXISTS "${BRASS_ROOT}/include")
            target_include_directories(brotensor_test_cuda_jit_gemv PRIVATE "${BRASS_ROOT}/include")
            if(BROTENSOR_BRASS_LIB)
                target_link_libraries(brotensor_test_cuda_jit_gemv PRIVATE "${BROTENSOR_BRASS_LIB}")
            endif()
        endif()
        target_include_directories(brotensor_test_cuda_jit_gemv PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)
        target_link_libraries(brotensor_test_cuda_jit_gemv PRIVATE CUDA::cuda_driver)

        # CUDA Graph Runner for JIT kernels suite
        brotensor_add_test(brotensor_test_cuda_graph_jit test_cuda_graph_jit.cpp)
        if(BRASS_ROOT AND EXISTS "${BRASS_ROOT}/include")
            target_include_directories(brotensor_test_cuda_graph_jit PRIVATE "${BRASS_ROOT}/include")
            if(BROTENSOR_BRASS_LIB)
                target_link_libraries(brotensor_test_cuda_graph_jit PRIVATE "${BROTENSOR_BRASS_LIB}")
            endif()
        endif()
        target_include_directories(brotensor_test_cuda_graph_jit PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)
        target_link_libraries(brotensor_test_cuda_graph_jit PRIVATE CUDA::cuda_driver)

        # Quantized GEMV (Q8_0 & Q4_K) JIT suite
        brotensor_add_test(brotensor_test_cuda_jit_quant test_cuda_jit_quant.cpp)
        if(BRASS_ROOT AND EXISTS "${BRASS_ROOT}/include")
            target_include_directories(brotensor_test_cuda_jit_quant PRIVATE "${BRASS_ROOT}/include")
            if(BROTENSOR_BRASS_LIB)
                target_link_libraries(brotensor_test_cuda_jit_quant PRIVATE "${BROTENSOR_BRASS_LIB}")
            endif()
        endif()
        target_include_directories(brotensor_test_cuda_jit_quant PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)
        target_link_libraries(brotensor_test_cuda_jit_quant PRIVATE CUDA::cuda_driver)

        # Public Fused Ops (CPU & CUDA) parity test suite
        brotensor_add_test(brotensor_test_ops_fused test_ops_fused.cpp)
        if(BRASS_ROOT AND EXISTS "${BRASS_ROOT}/include")
            target_include_directories(brotensor_test_ops_fused PRIVATE "${BRASS_ROOT}/include")
            if(BROTENSOR_BRASS_LIB)
                target_link_libraries(brotensor_test_ops_fused PRIVATE "${BROTENSOR_BRASS_LIB}")
            endif()
        endif()
        target_include_directories(brotensor_test_ops_fused PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)
        target_link_libraries(brotensor_test_ops_fused PRIVATE CUDA::cuda_driver)
    endif()
endif()
