# Vulkan op coverage

Generated for chunk 6 of the Vulkan backend (docs/vulkan.md). What runs on
`Device::vulkan(i)` today, slot by slot, and what each sibling's code calls.

**Op table: 242 of 270 slots implemented, 28 null.** A null slot
throws "not implemented on vulkan" when dispatched. Every null slot is a training
backward / training-mode forward, a host-only op that never dispatches to a GPU,
or filtered_lrelu, whose public entry point falls back to a composite of two ops
Vulkan implements. No inference path of brolm, brodiffusion, brosoundml,
brovisionml or brogameagent reaches a null slot.

Outside the op table: `conv1d*`, `causal_conv1d`, `conv1d_int8w_fp16` (wrappers over
conv2d / conv_transpose1d / the INT8 conv), `lora_forward` / `lora_backward`
(GEMM compositions) and the five `fused_*` ops (src/ops/fused.cpp; the
stacked-weight SwiGLU GEMV takes the Vulkan / HIP GEMV epilogue, the rest compose
dispatched ops) all run on Vulkan. `jit::*` traces on Vulkan are replayed op by
op (src/jit/trace_eager.cpp), as on HIP. `Device::CUDA` aliases to Vulkan when
neither a CUDA nor a HIP backend is registered (detail::resolve_device_alias).

## Null slots

| Slot | Why |
|---|---|
| `attention_backward` | training backward (brogameagent Attention::backward) |
| `mha_backward` | training backward (brogameagent MultiHeadAttention::backward) |
| `mse_scalar` | host-only: no Tensor operand, ops.cpp always runs it on the CPU table |
| `softmax_xent_segment` | host-only: raw host pointers, ops.cpp always runs it on the CPU table |
| `embedding_lookup_backward` | training backward (scatter-add: needs float atomics or a sort) |
| `bce_with_logits_fused_batched` | training loss (brosoundml BC-ResNet training) |
| `group_norm_backward` | training backward (brosoundml Kokoro decoder backward) |
| `upsample_bilinear_2x_backward` | training backward |
| `interp2d_backward` | training backward |
| `pad2d_backward` | training backward |
| `adaptive_avg_pool2d_backward` | training backward |
| `max_pool2d_backward` | training backward |
| `scatter_rows_add` | training (brolm LayaGrad soft-row gradient; scatter-add) |
| `conv_transpose2d_backward_input` | training backward |
| `conv_transpose2d_backward_weight` | training backward |
| `self_attention_backward` | training backward |
| `cross_attention_backward` | training backward |
| `flash_attention_packed_qkv_backward` | training backward (brolm LayaGrad) |
| `flash_attention_varlen_backward` | training backward |
| `flash_attention_qkvo_backward` | training backward |
| `flash_attention_backward` | training backward |
| `resblock_backward` | training backward |
| `lstm_forward_train` | training forward (no sibling calls it) |
| `lstm_backward` | training backward |
| `batch_norm_forward` | training-mode BatchNorm (batch statistics + running update); inference uses batch_norm_inference |
| `batch_norm_backward` | training backward (brosoundml BC-ResNet / phoneme training) |
| `filtered_lrelu_forward` | by design: composite fallback (bias_act + upfirdn2d), runs on Vulkan |
| `filtered_lrelu_backward` | by design: the public op falls back to its bias_act + upfirdn2d composite (as on HIP), which runs on Vulkan |

## Per sibling

Op-table ops each sibling's `src/` and `include/` call (an unqualified-name scan,
so it over-approximates: a sibling function sharing an op's name counts).
"Training" means the op is a gradient / optimiser / loss op or every call site is
in a training file (LayaGrad, LoRA trainers, BC-ResNet / phoneme training, the
Kokoro decoder backward, brogameagent's learners). Ops new in chunk 6 are marked *.

### brolm

Inference: 54 ops, all implemented.

`tanh_forward`, `sigmoid_forward`, `add_inplace`, `scale_inplace`, `clamp`, `mul_inplace`, `masked_mean_pool_forward`*, `embedding_lookup_forward`*, `copy_d2d`, `copy_d2d_strided`, `cast`, `layernorm_forward_inference_batched`, `linear_forward_batched`, `relu_forward_batched`, `conv2d_forward`, `silu_forward`, `gelu_forward`, `gelu_exact_forward`, `quick_gelu_forward`, `gather_rows`, `scatter_rows`, `linear_forward_batched_fp16`, `linear_forward_batched_ex`, `layernorm_forward_inference_batched_fp16`, `flash_attention_forward`, `flash_attention_gqa_forward`, `flash_attention_packed_qkv_forward`, `flash_attention_qkvo_forward`, `nchw_to_sequence`, `sequence_to_nchw`, `matmul`, `rope_forward`, `rms_norm_forward`, `swiglu_forward`, `kv_cache_append`, `flash_attention_decode`, `flash_attention_decode_masked`, `segment_softmax_stats`, `linear_forward_batched_int8w_fp16`, `dequant_q4k_to_fp16`, `linear_forward_batched_q4k_fp16`, `dequant_q8_0_to_fp16`, `linear_forward_batched_q8_0_fp16`, `dequant_q6k_to_fp16`, `linear_forward_batched_q6k_fp16`, `rope_apply`, `rope_qkv_packed_inplace`, `rope_apply_mrope`, `self_attention_bias_forward`, `self_attention_bias_int8w_fp16`, `causal_conv1d_update`, `l2_norm_forward`, `gated_delta_rule_chunked`*, `gated_delta_rule_step`*

Training: 12 ops, null: `scatter_rows_add`, `flash_attention_packed_qkv_backward`.

`relu_backward`, `masked_mean_pool_backward`*, `layernorm_forward_batched_with_caches`, `layernorm_backward_batched_with_caches`, `adam_step`*, `xavier_init`*, `linear_backward_batched`, `silu_backward`, `gelu_exact_backward`, `scatter_rows_add` **(null)**, `geglu_exact_backward`, `flash_attention_packed_qkv_backward` **(null)**

### brodiffusion

Inference: 63 ops, all implemented.

`relu_forward`, `tanh_forward`, `sigmoid_forward`, `add_inplace`, `axpby_inplace`, `add_scalar_inplace`, `scale_inplace`, `clamp`, `mul_inplace`, `mha_forward`, `concat_rows`, `concat_nchw_channels`, `copy_d2d`, `copy_d2d_strided`, `cast`, `layernorm_forward_inference_batched`, `linear_forward_batched`, `conv2d_forward`, `group_norm_forward`, `silu_forward`, `gelu_forward`, `gelu_exact_forward`, `upsample_nearest_2x`, `downsample_avg_2x`, `pad2d_forward`, `linear_forward_batched_fp16`, `geglu_exact_forward`, `cross_attention_forward_with_attn`, `layernorm_forward_inference_batched_fp16`, `flash_attention_forward`, `flash_attention_gqa_forward`, `flash_attention_windowed_forward`, `flash_attention_varlen_forward`, `flash_attention_qkvo_forward`, `flash_attention_project_kv`, `flash_attention_q_with_kv_cached_forward`, `nchw_to_sequence`, `sequence_to_nchw`, `spatial_merge_2x2_forward`, `pixel_shuffle_upsample_2x_forward`, `patch_unpack_forward`, `resblock_forward`, `resblock_forward_int8w_fp16`, `matmul`, `matmul_abt`, `rms_norm_forward`, `ddim_step`, `timestep_embedding`, `conv2d_int8w_fp16_forward`, `linear_forward_batched_int8w_fp16`, `flash_attention_project_kv_int8w_fp16`, `flash_attention_q_with_kv_cached_int8w_fp16`, `flash_attention_qkvo_int8w_fp16`, `modulate`, `broadcast_mul`, `rope_apply`, `rope_apply_perhead`, `l2_norm_forward`, `randn`, `rsqrt_forward`, `pixel_norm_forward`, `add_channel_bias_inplace`, `softmax_rows_forward`

Training: 1 ops, all implemented.

`adam_step`*

### brosoundml

Inference: 76 ops, all implemented.

`linear_forward`, `relu_forward`, `tanh_forward`, `sigmoid_forward`, `add_inplace`, `axpby_inplace`, `add_scalar_inplace`, `scale_inplace`, `clamp`, `mul_inplace`, `embedding_lookup_forward`*, `concat_rows`, `concat_batched_rows`, `concat_nchw_channels`, `copy_d2d`, `copy_d2d_strided`, `cast`, `layernorm_forward_inference_batched`, `linear_forward_batched`, `relu_forward_batched`, `add_inplace_batched`, `conv2d_forward`, `group_norm_forward`, `silu_forward`, `gelu_forward`, `gelu_exact_forward`, `interp2d_align_corners_forward`, `pad2d_forward`, `slice2d_forward`, `top_k_rows`, `gather_rows`, `scatter_rows`, `linear_forward_batched_fp16`, `linear_forward_batched_ex`, `flash_attention_forward`, `flash_attention_gqa_forward`, `flash_attention_windowed_forward`, `flash_attention_varlen_forward`, `nchw_to_sequence`, `sequence_to_nchw`, `matmul`, `rms_norm_forward`, `sum_cols`, `argmax_rows`, `modulate`, `broadcast_mul`, `rope_apply`, `self_attention_bias_forward`, `rel_pos_bias_xl_forward`, `complex_mul`, `complex_abs`, `complex_angle`, `complex_from_polar`, `rfft`, `irfft`, `stft`, `istft`, `conv_transpose1d_forward`, `pad1d_forward`, `snake_forward`, `elu_forward`, `leaky_relu_forward`, `vq_encode_forward`, `resample1d_forward`, `log_forward`, `exp_forward`, `sample_logits_into`, `masked_diffusion_scores`, `masked_diffusion_commit`, `batch_norm_inference`, `randn`, `rand_uniform`, `sin_forward`, `add_channel_bias_inplace`, `add_row_bias_inplace`, `softmax_rows_forward`

Training: 19 ops, null: `bce_with_logits_fused_batched`, `group_norm_backward`, `batch_norm_forward`, `batch_norm_backward`.

`linear_backward`, `tanh_backward`, `adam_step`*, `xavier_init`*, `linear_backward_batched`, `relu_backward_batched`, `softmax_xent_fused_batched`, `bce_with_logits_fused_batched` **(null)**, `conv2d_backward_input`, `conv2d_backward_weight`*, `conv2d_backward_bias`, `group_norm_backward` **(null)**, `istft_backward`, `conv_transpose1d_backward_input`, `pad1d_backward`, `snake_backward`, `leaky_relu_backward`, `batch_norm_forward` **(null)**, `batch_norm_backward` **(null)**

### brovisionml

StyleGAN3 GAN inversion (`stylegan3.cpp`, fitting W+ to an image) is an inference-time feature that runs a backward: `modulated_conv2d_backward` (frozen weights), the filtered_lrelu / bias_act / upfirdn2d backwards and `adam_step`, all of which run on Vulkan now.

Inference: 54 ops, all implemented.

`relu_forward`, `tanh_forward`, `sigmoid_forward`, `add_inplace`, `add_scalar_inplace`, `scale_inplace`, `clamp`, `mul_inplace`, `mha_forward`, `concat_rows`, `concat_batched_rows`, `concat_nchw_channels`, `copy_d2d`, `copy_d2d_strided`, `cast`, `layernorm_forward_inference_batched`, `linear_forward_batched`, `relu_forward_batched`, `conv2d_forward`, `deform_conv2d_forward`*, `group_norm_forward`, `silu_forward`, `gelu_exact_forward`, `interp2d_forward`, `interp2d_align_corners_forward`, `pad2d_forward`, `slice2d_forward`, `l2_normalize_nchw_forward`, `convex_upsample_forward`, `adaptive_avg_pool2d_forward`, `max_pool2d_forward`, `gather_rows`, `conv_transpose2d_forward`, `linear_forward_batched_fp16`, `linear_forward_batched_fp16_act`, `flash_attention_varlen_forward`, `nchw_to_sequence`, `sequence_to_nchw`, `matmul`, `swiglu_forward`, `argmax_rows`, `rows_count_above`*, `threshold_u8`*, `broadcast_mul`, `rope_apply`, `self_attention_bias_forward`, `self_attention_decomposed_rel_pos_forward`*, `self_attention_decomposed_rel_pos_windowed_forward`*, `leaky_relu_forward`, `batch_norm_inference`, `pixel_norm_forward`, `modulated_conv2d_forward`*, `bias_act_forward`*, `filtered_lrelu_forward` (composite, runs)

Training: 4 ops, all implemented.

`adam_step`*, `matmul_backward`, `modulated_conv2d_backward`*, `filtered_lrelu_backward` (composite, runs)

### brogameagent

The nets construct their weights with `xavier_init` (bit-identical to the CPU stream on Vulkan) and infer through the ops below; the null slots are the learners' backwards.

Inference: 22 ops, all implemented.

`linear_forward`, `relu_forward`, `tanh_forward`, `sigmoid_forward`, `add_inplace`, `add_scalar_inplace`, `scale_inplace`, `clamp`, `build_slot_mask`*, `softmax_forward`, `layernorm_forward`, `attention_forward`*, `mha_forward`, `masked_mean_pool_forward`*, `concat_rows`, `split_rows`, `copy_d2d`, `cast`, `linear_forward_batched`, `relu_forward_batched`, `tanh_forward_batched`, `add_inplace_batched`

Training: 20 ops, null: `attention_backward`, `mha_backward`.

`linear_backward`, `relu_backward`, `tanh_backward`, `sigmoid_backward`, `softmax_backward`, `layernorm_backward`, `attention_backward` **(null)**, `mha_backward` **(null)**, `masked_mean_pool_backward`*, `mse_scalar` (host-only, runs), `softmax_xent`, `softmax_xent_segment` (host-only, runs), `sgd_step`*, `adam_step`*, `xavier_init`*, `linear_backward_batched`, `relu_backward_batched`, `tanh_backward_batched`, `mse_vec_per_sample`*, `softmax_xent_fused_batched`

## What the siblings still need (chunk 7)

The op table is not the whole story: these sibling-side paths name a backend
explicitly and either skip Vulkan or would misbehave on it. None of them is an
op gap in brotensor.

* **Device tests by equality.** `X.device == Device::CUDA || X.device ==
  Device::HIP` (brosoundml `modules.cpp`, `whisper_modules.cpp`,
  `qwen_tts_talker.cpp`, `qwen_tts_code_predictor.cpp`, `supertonic.cpp`;
  brodiffusion `pipeline.cpp`, `triposplat/sampler.cpp`, `flow_model.cpp`;
  brolm `detail/dense_decoder.cpp`, `t5.cpp`) gates GPU-only fast paths and
  graph capture. A tensor created through the `Device::CUDA` alias is tagged
  `vulkan`, so these tests are false on Vulkan and the code takes its
  CPU-style / ungraphed path (correct, slower). They should test
  `!device.is_cpu()` or add `Device::VULKAN`, and use `VulkanGraphCapture` /
  `VulkanGraph` (`<brotensor/vulkan.h>`, the same shape as `CudaGraphCapture`)
  where they capture.
* **Device parsing.** The JS hosts map `"cuda"` / `"hip"` / `"rocm"` strings to
  devices (brolm `api/host_lm_internal.h`, brosoundml
  `api/host_soundml_internal.h`, brodiffusion `native_diffusion_triposplat.cpp`,
  brovisionml `native_vision_ops.cpp`) and pick a default GPU from CUDA / HIP
  only; `"vulkan"` / `"vk"` needs adding (brotensor's own
  `BROTENSOR_DEFAULT_DEVICE` parser already accepts both).
* **T5 on Vulkan.** brolm's T5 re-casts to BF16 only on CUDA / HIP and keeps
  FP16 + the residual clamp elsewhere, which is the right choice on Vulkan:
  the cooperative-matrix GEMM converts BF16 operands to FP16 as it stages them
  (the dtype policy), so a BF16 activation beyond 65504 would become inf
  there. Leave T5 on its FP16 path for Vulkan (or force FP32).
* **Sibling GPU kernels: must be fixed before Vulkan tensors reach them.**
  brodiffusion `fused_resblock.cu`, `fused_transformer.cu` (fused linear +
  GeGLU, row-bias / vector adds) and `triposplat/flow_rope.cu`, brovisionml
  `dsine_ops.cu` (ray ReLU, angular propagation) and `dpt_preprocess_gpu.cu`
  are CUDA / HIP kernels, and their entry points route *any* non-CPU tensor
  to them (`fused.cpp`'s `on_gpu(t)` is `t.device != CPU`; dsine checks
  `== CPU`). With a Vulkan tensor that is a HIP kernel handed Vulkan buffer
  addresses in a HIP + Vulkan build, and the host CPU implementation
  dereferencing them in a Vulkan-only build: a fault either way. Each needs a
  Vulkan branch: the brodiffusion fusions can compose brotensor ops Vulkan has
  (`resblock_forward`, `linear_forward_batched_ex` with the GeGLU epilogue,
  `add_row_bias_inplace`, `add_inplace`, `rope_apply*`); dsine's two ops and
  DPT's preprocessing need small GLSL kernels (or a download / CPU / upload
  fallback).
* **BF16 checkpoints.** `safetensors::upload_compute(_checked)` / `upload_as`
  already do the policy's cast at load on Vulkan (raw BF16 bits up, device
  cast to FP16; tested). `safetensors::upload` and `gguf::upload_raw` keep the
  file dtype by contract, so a BF16 weight loaded through them stays BF16 and
  runs through the GEMM's slower convert-at-load path; `upload_fp16` rejects
  BF16 on every backend. Siblings that hand-convert on the host are unaffected.

## Every slot

Grouped as in include/brotensor/detail/op_table.h; the file is the Vulkan op file
that registers the slot.


**Dense layers + elementwise activations**

| Slot | Vulkan | File |
|---|---|---|
| `linear_forward` | implemented | ops_linear.cpp |
| `linear_backward` | implemented | ops_linear.cpp |
| `relu_forward` | implemented | ops_elementwise.cpp |
| `relu_backward` | implemented | ops_elementwise.cpp |
| `tanh_forward` | implemented | ops_elementwise.cpp |
| `tanh_backward` | implemented | ops_elementwise.cpp |
| `sigmoid_forward` | implemented | ops_elementwise.cpp |
| `sigmoid_backward` | implemented | ops_elementwise.cpp |
| `add_inplace` | implemented | ops_elementwise.cpp |
| `axpby_inplace` | implemented | ops_elementwise.cpp |
| `add_scalar_inplace` | implemented | ops_elementwise.cpp |
| `scale_inplace` | implemented | ops_elementwise.cpp |
| `clamp` | implemented | ops_elementwise.cpp |
| `mul_inplace` | implemented | ops_elementwise.cpp |
| `div_inplace` | implemented | ops_elementwise.cpp |
| `build_slot_mask` | implemented (chunk 6) | ops_misc.cpp |

**Reductions / norm / softmax / attention (training)**

| Slot | Vulkan | File |
|---|---|---|
| `softmax_forward` | implemented | ops_norm.cpp |
| `softmax_backward` | implemented | ops_norm.cpp |
| `layernorm_forward` | implemented | ops_norm.cpp |
| `layernorm_backward` | implemented | ops_norm.cpp |
| `attention_forward` | implemented (chunk 6) | ops_attention_proj.cpp |
| `attention_backward` | null | - |
| `mha_forward` | implemented | ops_attention_proj.cpp |
| `mha_backward` | null | - |

**Pooling / losses / embeddings / concat**

| Slot | Vulkan | File |
|---|---|---|
| `masked_mean_pool_forward` | implemented (chunk 6) | ops_misc.cpp |
| `masked_mean_pool_backward` | implemented (chunk 6) | ops_misc.cpp |
| `mse_vec_forward` | implemented (chunk 6) | ops_misc.cpp |
| `mse_vec_backward` | implemented (chunk 6) | ops_misc.cpp |
| `mse_scalar` | null | - |
| `softmax_xent` | implemented | ops_xent.cpp |
| `softmax_xent_segment` | null | - |
| `softmax_xent_fused` | implemented | ops_xent.cpp |
| `embedding_lookup_forward` | implemented (chunk 6) | ops_misc.cpp |
| `embedding_lookup_backward` | null | - |
| `concat_rows` | implemented | ops_copy.cpp |
| `split_rows` | implemented | ops_copy.cpp |
| `concat_batched_rows` | implemented | ops_spatial.cpp |
| `concat_nchw_channels` | implemented | ops_spatial.cpp |
| `concat_nchw_channels_backward` | implemented | ops_spatial.cpp |
| `copy_d2d` | implemented | ops_copy.cpp |
| `copy_d2d_strided` | implemented | ops_copy.cpp |
| `cast` | implemented | ops_copy.cpp |

**Inference batched + optim**

| Slot | Vulkan | File |
|---|---|---|
| `layernorm_forward_inference_batched` | implemented | ops_norm.cpp |
| `layernorm_forward_batched_with_caches` | implemented | ops_norm.cpp |
| `layernorm_backward_batched_with_caches` | implemented | ops_norm.cpp |
| `sgd_step` | implemented (chunk 6) | ops_misc.cpp |
| `adam_step` | implemented (chunk 6) | ops_misc.cpp |
| `xavier_init` | implemented (chunk 6) | ops_misc.cpp |

**Batched (inference-only) variants**

| Slot | Vulkan | File |
|---|---|---|
| `linear_forward_batched` | implemented | ops_linear.cpp |
| `relu_forward_batched` | implemented | ops_elementwise.cpp |
| `tanh_forward_batched` | implemented | ops_elementwise.cpp |
| `add_inplace_batched` | implemented | ops_elementwise.cpp |

**Batched (training) backward variants**

| Slot | Vulkan | File |
|---|---|---|
| `linear_backward_batched` | implemented | ops_linear.cpp |
| `relu_backward_batched` | implemented | ops_elementwise.cpp |
| `tanh_backward_batched` | implemented | ops_elementwise.cpp |

**Batched per-sample loss kernels**

| Slot | Vulkan | File |
|---|---|---|
| `mse_vec_per_sample` | implemented (chunk 6) | ops_misc.cpp |
| `softmax_xent_fused_batched` | implemented | ops_xent.cpp |
| `bce_with_logits_fused_batched` | null | - |

**Conv2d (forward + backwards)**

| Slot | Vulkan | File |
|---|---|---|
| `conv2d_forward` | implemented | ops_conv.cpp |
| `conv2d_backward_input` | implemented | ops_conv.cpp |
| `conv2d_backward_weight` | implemented (chunk 6) | ops_vision.cpp |
| `conv2d_backward_bias` | implemented | ops_conv.cpp |

**Modulated deformable conv2d (torchvision deform_conv2d v2, fwd)**

| Slot | Vulkan | File |
|---|---|---|
| `deform_conv2d_forward` | implemented (chunk 6) | ops_vision.cpp |

**Conv3d (forward + W8A16 variant for Qwen3-VL patch embed)**

| Slot | Vulkan | File |
|---|---|---|
| `conv3d_forward` | implemented | ops_conv.cpp |
| `conv3d_int8w_fp16_forward` | implemented | ops_quant.cpp |

**GroupNorm**

| Slot | Vulkan | File |
|---|---|---|
| `group_norm_forward` | implemented | ops_gnorm.cpp |
| `group_norm_backward` | null | - |

**Activations: silu, gelu (tanh-approx + exact), quick_gelu**

| Slot | Vulkan | File |
|---|---|---|
| `silu_forward` | implemented | ops_elementwise.cpp |
| `silu_backward` | implemented | ops_elementwise.cpp |
| `gelu_forward` | implemented | ops_elementwise.cpp |
| `gelu_backward` | implemented | ops_elementwise.cpp |
| `gelu_exact_forward` | implemented | ops_elementwise.cpp |
| `gelu_exact_backward` | implemented | ops_elementwise.cpp |
| `quick_gelu_forward` | implemented | ops_elementwise.cpp |
| `quick_gelu_backward` | implemented | ops_elementwise.cpp |

**Resample (NN / bilinear / avgpool)**

| Slot | Vulkan | File |
|---|---|---|
| `upsample_nearest_2x` | implemented | ops_spatial.cpp |
| `upsample_bilinear_2x` | implemented | ops_spatial.cpp |
| `downsample_avg_2x` | implemented | ops_spatial.cpp |
| `upsample_nearest_2x_backward` | implemented | ops_spatial.cpp |
| `upsample_bilinear_2x_backward` | null | - |
| `downsample_avg_2x_backward` | implemented | ops_spatial.cpp |

**Arbitrary-scale 2D resample (nearest / bilinear / bicubic)**

| Slot | Vulkan | File |
|---|---|---|
| `interp2d_forward` | implemented | ops_spatial.cpp |
| `interp2d_backward` | null | - |
| `interp2d_align_corners_forward` | implemented | ops_spatial.cpp |

**2D padding (zero / reflect / replicate) — NCHW**

| Slot | Vulkan | File |
|---|---|---|
| `pad2d_forward` | implemented | ops_spatial.cpp |
| `pad2d_backward` | null | - |

**2D spatial slice / crop on NCHW**

| Slot | Vulkan | File |
|---|---|---|
| `slice2d_forward` | implemented | ops_spatial.cpp |
| `slice2d_backward` | implemented | ops_spatial.cpp |

**2D neighborhood unfold (spatial-preserving im2col) on NCHW**

| Slot | Vulkan | File |
|---|---|---|
| `unfold2d_forward` | implemented | ops_spatial.cpp |

**L2 normalize over channel axis (NCHW), per-pixel unit direction**

| Slot | Vulkan | File |
|---|---|---|
| `l2_normalize_nchw_forward` | implemented | ops_gnorm.cpp |

**Convex (mask-based) upsample (RAFT-style) on NCHW**

| Slot | Vulkan | File |
|---|---|---|
| `convex_upsample_forward` | implemented | ops_spatial.cpp |

**Per-row top-k (descending values + int32 indices)**

| Slot | Vulkan | File |
|---|---|---|
| `top_k_rows` | implemented | ops_topk.cpp |

**Adaptive avg pool 2D (NCHW), arbitrary output spatial size**

| Slot | Vulkan | File |
|---|---|---|
| `adaptive_avg_pool2d_forward` | implemented | ops_spatial.cpp |
| `adaptive_avg_pool2d_backward` | null | - |

**Max pool 2D (NCHW): forward returns Y + int32 flat-spatial Idx**

| Slot | Vulkan | File |
|---|---|---|
| `max_pool2d_forward` | implemented | ops_spatial.cpp |
| `max_pool2d_backward` | null | - |

**Row gather / scatter-add (general 2D — superset of embedding_lookup)**

| Slot | Vulkan | File |
|---|---|---|
| `gather_rows` | implemented | ops_spatial.cpp |
| `scatter_rows_add` | null | - |
| `scatter_rows` | implemented | ops_spatial.cpp |

**2D transposed convolution (NCHW) — forward + three backwards**

| Slot | Vulkan | File |
|---|---|---|
| `conv_transpose2d_forward` | implemented | ops_conv.cpp |
| `conv_transpose2d_backward_input` | null | - |
| `conv_transpose2d_backward_weight` | null | - |
| `conv_transpose2d_backward_bias` | implemented | ops_conv.cpp |

**SAM-style window partition / reverse (NCHW <-> windowed batch)**

| Slot | Vulkan | File |
|---|---|---|
| `window_partition_forward` | implemented | ops_spatial.cpp |
| `window_reverse_forward` | implemented | ops_spatial.cpp |

**FP16 linear (inference-only) + GEGLU family**

| Slot | Vulkan | File |
|---|---|---|
| `linear_forward_batched_fp16` | implemented | ops_linear.cpp |
| `linear_forward_batched_ex` | implemented | ops_linear.cpp |
| `linear_forward_batched_fp16_act` | implemented | ops_linear.cpp |
| `geglu_forward` | implemented | ops_glu.cpp |
| `geglu_backward` | implemented | ops_glu.cpp |
| `geglu_exact_forward` | implemented | ops_glu.cpp |
| `geglu_exact_backward` | implemented | ops_glu.cpp |

**Causal mask helper**

| Slot | Vulkan | File |
|---|---|---|
| `build_causal_mask_row` | implemented (chunk 6) | ops_misc.cpp |

**Cross-attention family (FP16 inference + FP32 train)**

| Slot | Vulkan | File |
|---|---|---|
| `cross_attention_forward` | implemented | ops_attention_proj.cpp |
| `cross_attention_forward_with_attn` | implemented | ops_attention_proj.cpp |
| `self_attention_forward_train` | implemented | ops_attention_proj.cpp |
| `self_attention_backward` | null | - |
| `attention_token_moments` | implemented (chunk 6) | ops_misc.cpp |
| `cross_attention_forward_train` | implemented (chunk 6) | ops_attention_proj.cpp |
| `cross_attention_backward` | null | - |

**FP16 LayerNorm inference + FP16 self-attention**

| Slot | Vulkan | File |
|---|---|---|
| `layernorm_forward_inference_batched_fp16` | implemented | ops_norm.cpp |
| `self_attention_forward` | implemented | ops_attention_proj.cpp |

**Flash attention family**

| Slot | Vulkan | File |
|---|---|---|
| `flash_attention_forward` | implemented | ops_attention.cpp |
| `flash_attention_gqa_forward` | implemented | ops_attention.cpp |
| `flash_attention_windowed_forward` | implemented | ops_attention.cpp |
| `flash_attention_varlen_forward` | implemented | ops_attention.cpp |
| `flash_attention_packed_qkv_forward` | implemented | ops_attention.cpp |
| `flash_attention_packed_qkv_backward` | null | - |
| `flash_attention_varlen_backward` | null | - |
| `flash_attention_qkvo_forward` | implemented | ops_attention.cpp |
| `flash_attention_qkvo_backward` | null | - |
| `flash_attention_backward` | null | - |
| `flash_attention_project_kv` | implemented | ops_attention.cpp |
| `flash_attention_q_with_kv_cached_forward` | implemented | ops_attention.cpp |

**NCHW <-> sequence transposes**

| Slot | Vulkan | File |
|---|---|---|
| `nchw_to_sequence` | implemented | ops_copy.cpp |
| `sequence_to_nchw` | implemented | ops_copy.cpp |

**2x2 pixel-unshuffle: (N,C,H,W) -> (N,4C,H/2,W/2)**

| Slot | Vulkan | File |
|---|---|---|
| `spatial_merge_2x2_forward` | implemented | ops_spatial.cpp |

**DC-AE up-shortcut: repeat_interleave + 2x pixel-shuffle**

| Slot | Vulkan | File |
|---|---|---|
| `pixel_shuffle_upsample_2x_forward` | implemented | ops_spatial.cpp |

**DiT unpatchify: token rows -> image (depth-to-space + channel keep)**

| Slot | Vulkan | File |
|---|---|---|
| `patch_unpack_forward` | implemented | ops_spatial.cpp |

**Diffusion ResBlock (forward + W8A16 + backward)**

| Slot | Vulkan | File |
|---|---|---|
| `resblock_forward` | implemented | ops_diffusion.cpp |
| `resblock_forward_int8w_fp16` | implemented | ops_diffusion.cpp |
| `resblock_backward` | null | - |

**Matmul + RoPE + RMSNorm + SwiGLU + KV-cache + Llama family**

| Slot | Vulkan | File |
|---|---|---|
| `matmul` | implemented | ops_linear.cpp |
| `matmul_abt` | implemented | ops_linear.cpp |
| `matmul_backward` | implemented | ops_linear.cpp |
| `lstm_forward_train` | null | - |
| `lstm_backward` | null | - |
| `rope_forward` | implemented | ops_rope.cpp |
| `rope_backward` | implemented | ops_rope.cpp |
| `rms_norm_forward` | implemented | ops_norm.cpp |
| `rms_norm_backward` | implemented | ops_norm.cpp |
| `swiglu_forward` | implemented | ops_glu.cpp |
| `swiglu_backward` | implemented | ops_glu.cpp |
| `kv_cache_append` | implemented | ops_attention.cpp |
| `flash_attention_decode` | implemented | ops_attention.cpp |
| `flash_attention_decode_masked` | implemented | ops_attention.cpp |

**Public reductions**

| Slot | Vulkan | File |
|---|---|---|
| `sum_rows` | implemented | ops_reduce.cpp |
| `sum_cols` | implemented | ops_reduce.cpp |
| `argmax_rows` | implemented | ops_reduce.cpp |
| `rows_count_above` | implemented (chunk 6) | ops_misc.cpp |
| `segment_softmax_stats` | implemented | ops_topk.cpp |
| `threshold_u8` | implemented (chunk 6) | ops_misc.cpp |

**Diffusion sampler steps + timestep embedding**

| Slot | Vulkan | File |
|---|---|---|
| `ddim_step` | implemented | ops_diffusion.cpp |
| `euler_step` | implemented | ops_diffusion.cpp |
| `dpmpp_2m_step` | implemented | ops_diffusion.cpp |
| `timestep_embedding` | implemented | ops_diffusion.cpp |

**INT8 weight-only quantisation (W8A16)**

| Slot | Vulkan | File |
|---|---|---|
| `matmul_int8w_fp16` | implemented | ops_quant.cpp |
| `conv2d_int8w_fp16_forward` | implemented | ops_quant.cpp |
| `linear_forward_batched_int8w_fp16` | implemented | ops_quant.cpp |
| `dequant_q4k_to_fp16` | implemented | ops_quant.cpp |
| `linear_forward_q4k_fp16` | implemented | ops_quant.cpp |
| `linear_forward_batched_q4k_fp16` | implemented | ops_quant.cpp |
| `dequant_q8_0_to_fp16` | implemented | ops_quant.cpp |
| `linear_forward_q8_0_fp16` | implemented | ops_quant.cpp |
| `linear_forward_batched_q8_0_fp16` | implemented | ops_quant.cpp |
| `dequant_q6k_to_fp16` | implemented | ops_quant.cpp |
| `linear_forward_q6k_fp16` | implemented | ops_quant.cpp |
| `linear_forward_batched_q6k_fp16` | implemented | ops_quant.cpp |
| `flash_attention_project_kv_int8w_fp16` | implemented | ops_quant_attention.cpp |
| `flash_attention_q_with_kv_cached_int8w_fp16` | implemented | ops_quant_attention.cpp |
| `flash_attention_qkvo_int8w_fp16` | implemented | ops_quant_attention.cpp |

**DiT / diffusion extras: AdaLN modulation, axial RoPE, T5-bias attention**

| Slot | Vulkan | File |
|---|---|---|
| `modulate` | implemented | ops_glu.cpp |
| `broadcast_mul` | implemented | ops_glu.cpp |
| `rope_apply` | implemented | ops_rope.cpp |
| `rope_apply_perhead` | implemented | ops_rope.cpp |
| `rope_apply_backward` | implemented | ops_rope.cpp |
| `rope_qkv_packed_inplace` | implemented | ops_rope.cpp |
| `rope_apply_mrope` | implemented | ops_rope.cpp |
| `self_attention_bias_forward` | implemented | ops_attention_proj.cpp |
| `rel_pos_bias_xl_forward` | implemented | ops_attention_proj.cpp |
| `self_attention_bias_int8w_fp16` | implemented | ops_quant_attention.cpp |
| `self_attention_decomposed_rel_pos_forward` | implemented (chunk 6) | ops_vision.cpp |
| `self_attention_decomposed_rel_pos_windowed_forward` | implemented (chunk 6) | ops_vision.cpp |

**Spectral / FFT core (brosoundml)**

| Slot | Vulkan | File |
|---|---|---|
| `complex_mul` | implemented | ops_spectral.cpp |
| `complex_mul_backward` | implemented | ops_spectral.cpp |
| `complex_abs` | implemented | ops_spectral.cpp |
| `complex_abs_backward` | implemented | ops_spectral.cpp |
| `complex_angle` | implemented | ops_spectral.cpp |
| `complex_from_polar` | implemented | ops_spectral.cpp |
| `fft` | implemented | ops_spectral.cpp |
| `ifft` | implemented | ops_spectral.cpp |
| `rfft` | implemented | ops_spectral.cpp |
| `irfft` | implemented | ops_spectral.cpp |
| `rfft_backward` | implemented | ops_spectral.cpp |
| `irfft_backward` | implemented | ops_spectral.cpp |

**STFT / iSTFT (brosoundml)**

| Slot | Vulkan | File |
|---|---|---|
| `stft` | implemented | ops_spectral.cpp |
| `stft_backward` | implemented | ops_spectral.cpp |
| `istft` | implemented | ops_spectral.cpp |
| `istft_backward` | implemented | ops_spectral.cpp |

**1D convolution family (brosoundml)**

| Slot | Vulkan | File |
|---|---|---|
| `conv_transpose1d_forward` | implemented | ops_audio.cpp |
| `conv_transpose1d_backward_input` | implemented | ops_audio.cpp |
| `conv_transpose1d_backward_weight` | implemented | ops_audio.cpp |
| `conv_transpose1d_backward_bias` | implemented | ops_audio.cpp |
| `causal_conv1d_update` | implemented | ops_audio.cpp |
| `pad1d_forward` | implemented | ops_audio.cpp |
| `pad1d_backward` | implemented | ops_audio.cpp |

**Vocoder / codec activations (brosoundml)**

| Slot | Vulkan | File |
|---|---|---|
| `snake_forward` | implemented | ops_audio.cpp |
| `snake_backward` | implemented | ops_audio.cpp |
| `elu_forward` | implemented | ops_elementwise.cpp |
| `elu_backward` | implemented | ops_elementwise.cpp |
| `leaky_relu_forward` | implemented | ops_elementwise.cpp |
| `leaky_relu_backward` | implemented | ops_elementwise.cpp |

**Codec quantization (brosoundml CHUNK 5, family D)**

| Slot | Vulkan | File |
|---|---|---|
| `vq_encode_forward` | implemented | ops_audio.cpp |
| `vq_encode_backward` | implemented | ops_audio.cpp |
| `fsq_quantize_forward` | implemented | ops_audio.cpp |
| `fsq_quantize_backward` | implemented | ops_audio.cpp |

**1D resampling (brosoundml CHUNK 6, family E)**

| Slot | Vulkan | File |
|---|---|---|
| `resample1d_forward` | implemented | ops_audio.cpp |
| `resample1d_backward` | implemented | ops_audio.cpp |

**log / exp / round elementwise (brosoundml CHUNK 6, family G)**

| Slot | Vulkan | File |
|---|---|---|
| `log_forward` | implemented | ops_elementwise.cpp |
| `log_backward` | implemented | ops_elementwise.cpp |
| `exp_forward` | implemented | ops_elementwise.cpp |
| `exp_backward` | implemented | ops_elementwise.cpp |
| `round_forward` | implemented | ops_elementwise.cpp |
| `round_backward` | implemented | ops_elementwise.cpp |

**Autoregressive logit sampling (brosoundml CHUNK 7, family F)**

| Slot | Vulkan | File |
|---|---|---|
| `sample_logits` | implemented | ops_sampling.cpp |
| `sample_logits_into` | implemented | ops_sampling.cpp |

**Masked-diffusion token selection (OmniVoice codebook grids)**

| Slot | Vulkan | File |
|---|---|---|
| `masked_diffusion_scores` | implemented | ops_sampling.cpp |
| `masked_diffusion_commit` | implemented | ops_sampling.cpp |

**L2 norm + Gated Delta Rule (linear-attention text path)**

| Slot | Vulkan | File |
|---|---|---|
| `l2_norm_forward` | implemented | ops_norm.cpp |
| `l2_norm_backward` | implemented | ops_norm.cpp |
| `gated_delta_rule_chunked` | implemented (chunk 6) | ops_delta.cpp |
| `gated_delta_rule_step` | implemented (chunk 6) | ops_delta.cpp |

**BatchNorm (vision backbones: ResNet, DETR-R50, classic Mask2Former)**

| Slot | Vulkan | File |
|---|---|---|
| `batch_norm_forward` | null | - |
| `batch_norm_inference` | implemented | ops_gnorm.cpp |
| `batch_norm_backward` | null | - |

**Image preprocessing helpers (vision-model inference)**

| Slot | Vulkan | File |
|---|---|---|
| `image_normalize` | implemented | ops_diffusion.cpp |
| `image_u8_to_f32_nhwc_to_nchw` | implemented | ops_diffusion.cpp |

**Counter-based noise generation (Philox 4x32-10)**

| Slot | Vulkan | File |
|---|---|---|
| `randn` | implemented | ops_diffusion.cpp |
| `rand_uniform` | implemented | ops_diffusion.cpp |
| `rand_bernoulli` | implemented | ops_diffusion.cpp |
| `randn_truncated` | implemented | ops_diffusion.cpp |

**StyleGAN3-R synthesis primitives — sin/cos/rsqrt + pixel_norm**

| Slot | Vulkan | File |
|---|---|---|
| `sin_forward` | implemented | ops_elementwise.cpp |
| `sin_backward` | implemented | ops_elementwise.cpp |
| `cos_forward` | implemented | ops_elementwise.cpp |
| `cos_backward` | implemented | ops_elementwise.cpp |
| `rsqrt_forward` | implemented | ops_elementwise.cpp |
| `rsqrt_backward` | implemented | ops_elementwise.cpp |
| `pixel_norm_forward` | implemented | ops_norm.cpp |
| `pixel_norm_backward` | implemented | ops_norm.cpp |

**StyleGAN3-R core ops — modulated_conv2d / upfirdn2d / bias_act**

| Slot | Vulkan | File |
|---|---|---|
| `modulated_conv2d_forward` | implemented (chunk 6) | ops_vision.cpp |
| `modulated_conv2d_backward` | implemented (chunk 6) | ops_vision.cpp |
| `upfirdn2d_forward` | implemented (chunk 6) | ops_misc.cpp |
| `upfirdn2d_backward` | implemented (chunk 6) | ops_misc.cpp |
| `bias_act_forward` | implemented (chunk 6) | ops_misc.cpp |
| `bias_act_backward` | implemented (chunk 6) | ops_misc.cpp |
| `filtered_lrelu_forward` | null | - |
| `filtered_lrelu_backward` | null | - |
| `add_channel_bias_inplace` | implemented | ops_elementwise.cpp |
| `add_row_bias_inplace` | implemented | ops_elementwise.cpp |
| `softmax_rows_forward` | implemented | ops_norm.cpp |
