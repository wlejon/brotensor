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
brotensor_add_test(brotensor_test_cpu_ops test_cpu_ops.cpp)
brotensor_add_test(brotensor_test_dispatch test_dispatch.cpp)
brotensor_add_test(brotensor_test_safetensors test_safetensors.cpp)
brotensor_add_test(brotensor_test_bf16_basics test_bf16_basics.cpp)
brotensor_add_test(brotensor_test_fft test_fft.cpp)
brotensor_add_test(brotensor_test_stft test_stft.cpp)
brotensor_add_test(brotensor_test_conv1d test_conv1d.cpp)
brotensor_add_test(brotensor_test_vocoder_activations test_vocoder_activations.cpp)
brotensor_add_test(brotensor_test_codec_quant test_codec_quant.cpp)
brotensor_add_test(brotensor_test_resample1d test_resample1d.cpp)
brotensor_add_test(brotensor_test_log_exp_round test_log_exp_round.cpp)
brotensor_add_test(brotensor_test_sample_logits test_sample_logits.cpp)
brotensor_add_test(brotensor_test_gated_delta_rule test_gated_delta_rule.cpp)
brotensor_add_test(brotensor_test_conv3d test_conv3d.cpp)
brotensor_add_test(brotensor_test_flash_attention_varlen test_flash_attention_varlen.cpp)
brotensor_add_test(brotensor_test_spatial_merge test_spatial_merge.cpp)
brotensor_add_test(brotensor_test_rope_mrope test_rope_mrope.cpp)
brotensor_add_test(brotensor_test_interp2d test_interp2d.cpp)
brotensor_add_test(brotensor_test_pad2d test_pad2d.cpp)
brotensor_add_test(brotensor_test_slice2d test_slice2d.cpp)
brotensor_add_test(brotensor_test_top_k test_top_k.cpp)
brotensor_add_test(brotensor_test_pool2d test_pool2d.cpp)
brotensor_add_test(brotensor_test_gather_rows test_gather_rows.cpp)
brotensor_add_test(brotensor_test_conv_transpose2d test_conv_transpose2d.cpp)
brotensor_add_test(brotensor_test_window_partition test_window_partition.cpp)
brotensor_add_test(brotensor_test_batch_norm test_batch_norm.cpp)
brotensor_add_test(brotensor_test_image_preproc test_image_preproc.cpp)

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
    brotensor_add_test(brotensor_test_int8_attention test_int8_attention.cpp)
    if(BROTENSOR_WITH_CUDA)
        brotensor_add_test(brotensor_test_streams test_streams.cpp)
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
        reduce_extra layernorm_inference
        activations geglu swiglu matmul rope rmsnorm
        conv2d conv3d group_norm
        resample transpose diffusion kv_cache
        cross_attention self_attention attention_moments
        flash_attention flash_attention_varlen resblock
        spatial_merge rope_mrope
        batched_backward tensor_ops
        modulate
        fft stft
        conv1d vocoder
        codec_quant resample1d log_exp_round sample_logits
        interp2d
        l2_norm gated_delta_rule
        pad2d slice2d top_k pool2d gather_rows
        conv_transpose2d window_partition
    )
    foreach(_name IN LISTS _brotensor_parity_tests)
        brotensor_add_test(brotensor_test_${_name}_parity test_${_name}_parity.cpp)
    endforeach()

    # Diffusion micro-benchmark. Built but not added to ctest — invoke
    # manually from the build dir (./tests/Release/brotensor_bench_diffusion).
    # CUDA-only: it uses cudaEvent_t for timing.
    if(BROTENSOR_WITH_CUDA)
        brotensor_add_executable(brotensor_bench_diffusion bench_diffusion.cpp)
    endif()
endif()
