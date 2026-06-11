# Op coverage

All ops are device-neutral and declared in the per-category headers under `<brotensor/ops/>` (`ls include/brotensor/ops/` is the table of contents; `<brotensor/ops.h>` is an umbrella that includes them all).

## Per-header surface

| Header | Surface |
|---|---|
| `activation.h` | relu / tanh / sigmoid, silu, gelu (tanh-approx / exact / quick), GEGLU / GEGLU-exact / SwiGLU, snake (BigVGAN/DAC), elu (EnCodec), leaky_relu (HiFi-GAN) |
| `attention.h` | single-head attention, MHA (optional biases), self/cross attention (train + flash), cross-attention with head-avg map + logit bias, attention token moments, self-attention with T5/ALiBi additive bias, SAM/ViTDet decomposed-rel-pos (incl. windowed), W8A16 bias-attention |
| `flash_attention.h` | tiled flash attention (+ bare-core bwd), windowed (sliding-window causal), packed var-length (+ bwd), fused QKV+O projections (+ bwd), project-KV / Q-with-cached-KV, KV-cache append, causal flash-decode (GQA, + per-key-masked variant), W8A16 variants |
| `linear.h` | linear (single / batched / fp16 / fused-act-epilogue), matmul (+ bwd), W8A16 batched linear |
| `lora.h` | **header-only** LoRA adapter — low-rank delta on a frozen base linear, optional per-rank gate, forward + backward. Pure composition of public ops; runs on any backend |
| `lstm.h` | single-layer LSTM over a sequence (PyTorch `nn.LSTM` layout): training forward with gate/cell caches + full-BPTT backward |
| `norm.h` | LayerNorm (single + batched ±caches), RMSNorm, GroupNorm, BatchNorm (train/infer/bwd), per-head L2-norm (Gated DeltaNet), NCHW channel L2-normalize, pixel_norm (StyleGAN `normalize_2nd_moment`) |
| `conv.h` / `conv1d.h` | conv2d / conv3d (+ W8A16), conv_transpose2d, deform_conv2d (torchvision modulated deformable v2, fwd), and the 1D family (conv1d wrappers, pad1d, conv_transpose1d, causal_conv1d + streaming update) — all with backward where applicable |
| `rope.h` | RoPE forward/backward, rope_apply (explicit cos/sin tables, head-shared) + bwd, rope_apply_perhead (per-head cos/sin tables), M-RoPE (Qwen-VL three-axis) |
| `delta_rule.h` | Gated Delta Rule linear attention — chunked prefill + streaming step |
| `diffusion.h` | AdaLN modulate / broadcast_mul, fused ResBlock (+ W8A16, + bwd), DDIM / Euler / DPM++ 2M sampler steps, sinusoidal timestep embedding |
| `stylegan.h` | StyleGAN3-R generator primitives: modulated_conv2d (style modulation + demod + conv, fwd+bwd), upfirdn2d (up→FIR→down resampler, fwd+bwd), bias_act (fused bias+act+gain+clamp, fwd+bwd), filtered_lrelu (alias-free nonlinearity, fwd+bwd) |
| `spatial.h` | pad2d, slice2d, unfold2d (neighborhood im2col), window partition/reverse (SAM), spatial 2×2 patch merge (Qwen-VL block-major + `pixel_unshuffle` channel-major), NCHW↔sequence transpose |
| `resize.h` | 2× nearest/bilinear up + 2× avg down (+ bwd), arbitrary-scale interp2d (nearest/bilinear/bicubic, half-pixel + align-corners), convex (RAFT) upsample, 1D resample |
| `pooling.h` | masked mean-pool, 2× avg downsample, adaptive avg pool2d, max pool2d (+ index bwd) |
| `embedding.h` | embedding lookup (+ scatter bwd), gather_rows, scatter_rows (overwrite) / scatter_rows_add (accumulate) |
| `concat.h` | concat/split rows, batched column-block concat, NCHW channel concat (+ bwd), copy_d2d + copy_d2d_strided (pitched 2D device copy) |
| `reduction.h` | sum_rows / sum_cols, argmax_rows, top_k_rows, rows_count_above (per-row two-threshold counts → INT32, SAM AMG stability score) |
| `loss.h` | softmax (+ bwd), softmax-xent (+ segment / fused / fused-batched), MSE (vec / scalar / per-sample), BCE-with-logits fused-batched |
| `optim.h` | sgd_step, adam_step, xavier_init |
| `elementwise.h` | add / scale / clamp / mul-inplace, dtype `cast` (FP32↔FP16/BF16), log / exp / round (+ bwd), sin / cos / rsqrt (+ bwd), threshold_u8 (binary threshold → INT8 byte mask, SAM AMG) |
| `sampling.h` | `sample_logits` (temperature / top-k / top-p / greedy), Philox RNG (`randn` / `rand_uniform` / `rand_bernoulli` / `randn_truncated`) |
| `spectral.h` | complex ops, FFT/iFFT, rFFT/irFFT (+ bwd), STFT/iSTFT (+ bwd) |
| `codec.h` | VQ encode (RVQ codeword search) + FSQ quantize (NanoCodec), straight-through bwd |
| `quant.h` | W8A16 host quantizer + matmul, GGUF Q4_K / Q6_K / Q8_0 dequant + fused GEMV + batched matmul |
| `image.h` | per-channel image normalize, uint8 HWC → FP32 NCHW |

## Backend coverage

The **CPU backend** implements essentially the entire FP32 surface — forward *and* backward, including the diffusion samplers, flash attention, the audio family, the vision primitives, LSTM, and the StyleGAN family — as the simple, correct, autovectorize-friendly reference. CPU is **FP32-only by design**: it leaves the FP16 / BF16 / INT8-W8A16 / GGUF-quant vtable slots null, and the dispatcher throws `"brotensor: <op>: not implemented on CPU"` if you call one.

The **CUDA and Metal backends** add the FP16 (and BF16) precision paths, batched-inference variants, the W8A16 and GGUF block-quant kernels, and a handful of GPU-only fused kernels. A few inference-only ops are CPU+CUDA but leave the Metal slot null (noted below). The [audio op family](#audio-op-family) is FP32 on **all three** backends with per-family CPU↔GPU parity tests.

Two "ops" are not vtable entries but device-agnostic compositions of public ops, so they run on any backend automatically: **LoRA** (`ops/lora.h`, header-only) and the **filtered_lrelu composite fallback** (used on CPU/Metal, and on CUDA for configs the fused kernel doesn't cover).

## GPU backends (CUDA / Metal)

FP32 fwd/bwd columns below mirror the CPU surface; the FP16 column is the GPU-only precision path (BF16 follows FP16 where the kernel notes it). INT8-W8A16 and GGUF-quant ops are GPU-only.

| Op | FP32 fwd | FP32 bwd | FP16 fwd | Notes |
|---|---|---|---|---|
| matmul | ✓ | ✓ | ✓ | plain row-major `A @ B` (no bias); dtype-dispatched FP32 + FP16 (FP32 accumulation); backward returns dA/dB (caller zeros, op accumulates; FP16 uses FP32 scratch + fold) |
| matmul_int8w_fp16 | — | — | ✓ | W8A16 weight-only matmul; INT8 weights + per-row FP32 scales, FP16 acts, FP32 accum |
| linear_batched_int8w_fp16 | — | — | ✓ | W8A16 batched linear in (B,in)→(B,out) layout; fused FP16 bias add; mirrors `linear_forward_batched_fp16` shape contract; WMMA fast path for K%8==0 (FP16 tensor cores with INT8→FP16 dequant on shared-mem load), tiled fallback otherwise |
| linear | ✓ | ✓ | ✓ | dense; FP32 single + batched (fwd/bwd), FP16 batched-inference and batched-train backward (dtype-dispatched, FP32 scratch + fold) |
| relu / tanh / sigmoid | ✓ | ✓ | — | elementwise; relu/tanh also have batched fwd+bwd |
| silu / gelu | ✓ | ✓ | ✓ | tanh-approx GELU; dtype-dispatched (FP16 bwd accumulates in FP32) |
| gelu_exact | ✓ | ✓ | ✓ | `0.5*x*(1+erf(x/√2))`, exact PyTorch/diffusers default |
| quick_gelu | ✓ | ✓ | ✓ | `x * sigmoid(1.702*x)`, OpenAI CLIP activation |
| geglu | ✓ | ✓ | ✓ | gated GELU (SD FFN); FP32+FP16 fwd/bwd, dtype-dispatched |
| geglu_exact | ✓ | ✓ | ✓ | gated exact-GELU FFN, matches diffusers GEGLU |
| swiglu | ✓ | ✓ | ✓ | gated SiLU FFN (Llama-style); dtype-dispatched FP32 + FP16 |
| add / scale / mul_inplace | ✓ | n/a | ✓ | dtype-dispatched |
| clamp | ✓ | n/a | ✓ | in-place min/max, dtype-dispatched (VAE epilogue) |
| sin / cos / rsqrt | ✓ | ✓ | ✓ | elementwise (Fourier features, demod/pixel-norm reciprocal-sqrt); GPU dtype-dispatched FP32/FP16/BF16 (FP32 math) |
| threshold_u8 | ✓ | n/a | ✓ | binary threshold → INT8 byte mask (SAM AMG mask binarization); FP32/FP16 input; CPU + CUDA (Metal slot null) |
| build_slot_mask | ✓ | n/a | — | device-side validity mask construction |
| softmax | ✓ | ✓ | — | masked, numerically stable |
| layernorm | ✓ | ✓ | ✓ | FP32 single + batched-infer; FP16 batched-infer + backward (dtype-dispatched, FP32 scratch + fold for dGamma/dBeta) |
| rms_norm | ✓ | ✓ | ✓ | `y = x * gamma / sqrt(mean(x²) + eps)`; dtype-dispatched FP32 + FP16 (FP16 bwd uses FP32 scratch + fold for dGamma) |
| group_norm | ✓ | ✓ | ✓ | NCHW, per-group stats; dtype-dispatched fwd+bwd (FP16 bwd accumulates in FP32) |
| pixel_norm | ✓ | ✓ | ✓ | StyleGAN `normalize_2nd_moment` row normalize; GPU dtype-dispatched FP32/FP16/BF16 |
| attention (single-head) | ✓ | ✓ | — | |
| mha (multi-head) | ✓ | ✓ | — | |
| self_attention | ✓ | ✓ | ✓ | FP32 = training (caches exposed via `_train`); FP16 = flash inference |
| cross_attention | ✓ | ✓ | ✓ | FP32 = training (caches exposed via `_train`, rectangular Wk/Wv); FP16 = flash inference |
| flash_attention | — | ✓ | ✓ | tiled online-softmax, Lk-unbounded, optional causal; FP16 backward via recompute returns dQ/dK/dV (no fwd-time caches). Bare-core bwd enables LoRA training when projections live outside the attention call. |
| flash_attention_qkvo | — | — | ✓ (fwd) / ✓ (bwd) | fused Q/K/V/O projections + biases; rectangular Wk/Wv for cross-attn; optional causal; verified at SD1.5 U-Net head_dims (40/80/160) and CLIP head_dim 64. FP16 backward via recompute (no fwd-time caches); registered on CUDA **and** Metal (and the FP32 path on CPU). **W8A16 variant** (`flash_attention_qkvo_int8w_fp16`) routes all four projections through `linear_forward_batched_int8w_fp16`; attention core stays FP16 |
| flash_attention_varlen | — | ✓ (bwd) | ✓ | packed variable-length MHA (Qwen-VL window attn); per-sequence boundaries from `cu_seqlens` INT32 prefix-sum buffers; optional per-sequence causal; recompute-based backward returns dQ/dK/dV |
| flash_attention_windowed | ✓ | — | — | sliding-window causal self-attention (streaming codecs / decode); window ≤ 0 is unbounded causal; FP32 (CPU + GPU) |
| flash_attention_project_kv | — | — | ✓ | pre-project ctx → K/V for cached cross-attention (SD timesteps reuse). W8A16 variant available |
| flash_attention_q_with_kv_cached | — | — | ✓ | forward against pre-projected K/V; bitwise-equivalent to `flash_attention_qkvo`'s cached path. W8A16 variant available |
| flash_attention_decode | — | — | ✓ | causal-aware decode against a partially-filled K/V cache; supports `L_q ≥ 1` (token-by-token or chunked). Masked variant `flash_attention_decode_masked` (per-key FP32 validity mask) on CPU + CUDA |
| kv_cache_append | — | — | ✓ | append `L_new` projected K/V rows into a pre-allocated `L_max` cache at `cur_len` |
| rope | ✓ | ✓ | ✓ | rotary position embedding; pair-wise rotation per head_dim chunk, `seq_offset` for KV-cache decode |
| resblock | — | ✓ (bwd) | ✓ | fused diffusion ResBlock (GN→SiLU→conv ×2 + skip); FP16 backward via composition of public ops (recomputes h1/h2/h3; no fwd-time caches) |
| conv2d | ✓ | ✓ | ✓ | NCHW, stride/pad/dil; groups ≥ 1 (depthwise supported); backward (dX, dW, dB) dtype-dispatched (FP32+FP16; FP16 dW/dB use FP32 scratch + fold) |
| conv2d_int8w_fp16 | — | — | ✓ | W8A16 weight-only conv2d; INT8 OIHW filter + per-output-channel FP32 scales, FP16 acts; CUDA WMMA fast path for 3x3 s1, 1x1 s1, 3x3 s2 (groups=1, dil=1) — naive fallback otherwise (incl. low-CTA long-K shapes) |
| deform_conv2d | ✓ | — | ✓ | torchvision `deform_conv2d` v2 (modulated deformable, bilinear per-tap offsets), forward-only; FP32/FP16, FP32 accumulation |
| modulated_conv2d | ✓ | ✓ | ✓ | StyleGAN synthesis core: per-sample style modulation + optional demodulation + conv2d; dW optional (skippable for inversion); GPU dtype-dispatched FP32/FP16/BF16, FP32 reductions |
| upfirdn2d | ✓ | ✓ | ✓ | upsample → pad/crop → 2D FIR → downsample (StyleGAN3, incl. non-separable config-R radial filters); backward is upfirdn2d with up/down swapped; GPU FP32/FP16/BF16, FP32 filter math |
| bias_act | ✓ | ✓ | ✓ | fused per-channel bias + activation (linear/lrelu) + gain + clamp; GPU FP32/FP16/BF16 (FP32 math, FP32-scratch dB) |
| filtered_lrelu | ✓ | ✓ | ✓ | alias-free nonlinearity (bias → upsample → lrelu → downsample); fused CUDA kernel + device-agnostic composite over bias_act/upfirdn2d (the path on CPU/Metal and for uncovered CUDA configs) |
| lstm | ✓ | ✓ | — | single-layer training LSTM + full BPTT (PyTorch `nn.LSTM` weight layout, gate order i\|f\|g\|o); FP32 on all three backends |
| upsample_nearest_2x | ✓ | ✓ | ✓ | backward dtype-dispatched (FP32+FP16) |
| upsample_bilinear_2x | ✓ | ✓ | ✓ | align_corners=False; backward dtype-dispatched (FP32+FP16; FP16 uses FP32 scratch + fold) |
| downsample_avg_2x | ✓ | ✓ | ✓ | stride 2, kernel 2; backward dtype-dispatched (FP32+FP16) |
| nchw ↔ sequence transpose | ✓ | n/a | ✓ | gather/scatter between NCHW and (L,D) layouts |
| embedding lookup | ✓ | ✓ | ✓ | FP32/FP16 table dispatch; backward dtype-dispatched (FP16 uses FP32 scratch + fold for atomic-add safety) |
| concat_rows / split_rows | ✓ | ✓ | ✓ | flat byte-aware concat (FP16 supported) |
| concat_batched_rows | ✓ | n/a | ✓ | per-row column-block concat via 2D memcpy |
| concat_nchw_channels | ✓ | ✓ | ✓ | channel-axis concat for U-Net skip merges (N≥1); backward is per-part scatter (overwrites parts) |
| masked_mean_pool | ✓ | ✓ | — | row-wise mean over valid mask |
| sum_rows / sum_cols | ✓ | n/a | ✓ | reductions along rows/cols; dtype-dispatched FP32 + FP16 |
| argmax_rows | ✓ | n/a | ✓ | per-row argmax; FP32/FP16 input, FP32 indices |
| rows_count_above | ✓ | n/a | ✓ | per-row above-threshold counts at two thresholds in one pass → (R,2) INT32 (SAM AMG stability score); FP32/FP16 input; CPU + CUDA (Metal slot null) |
| ddim_step | — | n/a | ✓ | fused DDIM sampler step over FP16 latents; FP32 internal math |
| euler_step | — | n/a | ✓ | fused Euler-discrete step (ε-prediction, σ convention; matches diffusers `EulerDiscreteScheduler`) |
| dpmpp_2m_step | — | n/a | ✓ | fused DPM-Solver++ 2M multistep update; caller supplies linear-combo coefficients and x0 cache. First step falls back to `euler_step` |
| timestep_embedding | ✓ | n/a | — | sinusoidal embedding (FP32) for diffusion timesteps and SDXL added-cond micro-conditioning; diffusers default (flip_sin_to_cos=True) |
| copy_d2d | ✓ | n/a | ✓ | flat-buffer device-to-device chunk copy. Strided variant `copy_d2d_strided` (pitched 2D copy, e.g. NCHW W-axis pad/unpad) on CPU + CUDA |
| build_causal_mask_row | n/a | n/a | ✓ | length-L FP32 mask, CLIP text |
| sgd / adam | ✓ | n/a | — | optimizer steps |
| mse / softmax-xent / bce | ✓ | ✓ | — | per-sample + batched (BCE-with-logits fused-batched added) |
| conv3d | ✓ | — | ✓ | NCTHW (forward only), grouped, FP32/FP16/BF16; W8A16 `conv3d_int8w_fp16` for Qwen-VL patch-embed (GPU-only) |
| conv_transpose1d / 2d | ✓ | ✓ | — | learned upsample (vocoders, SAM mask decoder, DPT heads); FP32 on CPU+CUDA |
| batch_norm | ✓ | ✓ | ✓ (inference) | NCHW train/infer/bwd, running stats (pretrained ResNet/DETR backbones); training fwd + bwd FP32-only; **inference fwd dtype-dispatched FP32/FP16/BF16** (FP32 math per element) |
| l2_norm / l2_normalize_nchw | ✓ | ✓ (l2_norm) | ✓ | per-head q/k L2 (Gated DeltaNet) + channel-axis NCHW normalize (DSINE normals); FP32/FP16/BF16 |
| gated_delta_rule | ✓ | n/a | ✓ | chunked prefill + streaming step (linear-attention text decoders); FP32/FP16, FP32 accumulators |
| rope_apply / rope_apply_perhead / rope_apply_mrope | ✓ | ✓ (apply) | ✓ | explicit cos/sin tables (2D axial RoPE), head-shared or per-head; + Qwen-VL three-axis M-RoPE; FP32/FP16/BF16 (perhead/mrope are inference-only, no bwd) |
| self_attention_bias | ✓ | n/a | ✓ | additive pre-softmax bias (T5 rel-pos / ALiBi); FP32/FP16/BF16; W8A16 variant GPU-only |
| decomposed_rel_pos (± windowed) | ✓ | n/a | ✓ | SAM/ViTDet data-dependent 2D rel-pos attention; FP32/FP16/BF16 |
| modulate / broadcast_mul | ✓ | n/a | ✓ | AdaLN affine + per-channel gate (DiT/SD3/Flux); FP32/FP16/BF16 |
| pad2d / slice2d / unfold2d | ✓ | ✓ (pad/slice) | ✓ | image pad (zero/reflect/replicate), crop, neighborhood im2col |
| window_partition / spatial_merge | ✓ | n/a | ✓ | SAM window tiling (+ reverse) and 2×2 patch merge (Qwen-VL block-major / `pixel_unshuffle` channel-major, e.g. Flux.2 VAE) |
| interp2d / convex_upsample | ✓ | ✓ (interp) | ✓ | arbitrary-scale resize (nearest/bilinear/bicubic, half-pixel + align-corners) + RAFT convex upsample |
| max_pool2d / adaptive_avg_pool2d | ✓ | ✓ | — | indexed max-pool bwd + PyTorch adaptive avg pool; FP32 (CPU+CUDA) |
| gather_rows / scatter_rows / scatter_rows_add | ✓ | ✓ | — | index-driven row gather + scatter (overwrite / accumulate) (SAM prompt encoder, DETR queries); FP32 |
| top_k_rows | ✓ | n/a | — | per-row top-k values + indices; FP32 |
| randn / rand_uniform / rand_bernoulli / randn_truncated | ✓ | n/a | — | Philox 4×32-10 RNG, PyTorch/JAX-compatible, FP32, all three backends |
| GGUF Q4_K / Q6_K / Q8_0 | — | — | ✓ | block-quant dequant + fused GEMV + batched matmul (W4/6/8-A16); CUDA WMMA fast paths, GEMV fallback; registered on CUDA **and** Metal |

## Audio op family

An FP32 op family for TTS / STT / neural-codec inference, consumed by the `brosoundml` sibling. Like the rest of the FP32 surface these are implemented on **all three backends** — CPU, CUDA, and Metal — FP32 throughout, with per-family CPU↔GPU parity tests.

| Group | Ops |
|---|---|
| Spectral core | `fft` / `ifft`, `rfft` / `irfft` (+ backward), `complex_mul` / `complex_abs` / `complex_angle` / `complex_from_polar` — interleaved-complex FP32, mixed-radix + Bluestein |
| STFT | `stft` / `istft` (+ backward) — windowed, COLA-normalised overlap-add |
| 1D convolution | `conv1d` (+ 3 backward halves), `pad1d`, `causal_conv1d`, `conv_transpose1d` (vocoder upsampling), `causal_conv1d_update` (streaming state) |
| Vocoder / codec activations | `snake` / snakebeta (BigVGAN/DAC), `elu` (EnCodec), `leaky_relu` (HiFi-GAN) |
| Codec quantization | `vq_encode` (RVQ codeword search), `fsq_quantize` (NanoCodec FSQ) — straight-through backward |
| Resampling | `resample1d` — arbitrary-scale length resample, nearest / linear (+ backward) |
| Elementwise | `log` / `exp` / `round` (+ backward) — log-mel domain maps |
| Sampling | `sample_logits` — temperature / top-k / top-p autoregressive sampler, counter-based Philox RNG |

The conv1d family is header-only wrappers over the conv2d ops (a 1D conv is a 2D conv with `H=kH=1`); `pad1d`, `conv_transpose1d`, and `causal_conv1d_update` are genuine per-backend kernels. Two exceptions to the FP32-everywhere rule: `conv1d_int8w_fp16` is a GPU-only W8A16 wrapper like its `conv2d_int8w_fp16` parent, and the CUDA `leaky_relu` forward is dtype-dispatched FP32/FP16/BF16 (FP32 math per element).

## Tests

```bash
ctest --test-dir build -C Release
```

Tests live under `tests/`, enabled by `BROTENSOR_TESTS=ON` (default ON when built standalone).

- **Always built (CPU-only):** `test_cpu_ops.cpp`, `test_dispatch.cpp`, plus per-family CPU coverage — the safetensors / GGUF loaders, BF16 basics, the audio family (fft, stft, conv1d, vocoder activations, codec quant, resample1d, log/exp/round, sample_logits, noise), the vision families (conv3d, spatial_merge, interp2d, pad2d/slice2d, pool2d, window_partition, image_preproc, gather_rows, conv_transpose2d, batch_norm), and the training families (lstm, lora, the stylegan primitives, flash_attention_varlen, gated_delta_rule, rope_mrope, top_k, bce_with_logits, decomposed-rel-pos attention).
- **GPU-gated** (built only with a CUDA or Metal backend):
  - `test_cpu_gpu_parity.cpp` — monolithic CPU↔GPU parity;
  - `test_*_parity.cpp` — per-op parity suite, one executable per op group, sharing the `parity_helpers.h` harness. Each runs the same device-neutral op on CPU- and GPU-resident tensors and asserts the results match;
  - dedicated GPU smoke tests for the diffusion / LLM / vision / INT8 / GGUF-quant kernels (conv2d, group_norm, flash attention, RoPE, int8 linear/conv WMMA, q4k/q6k/q8_0 parity, SDXL schedulers, CUDA graphs, streams, …).
- **Benchmarks:** `bench_*.cpp` executables (decode kernels, diffusion, SAM attention, modulated conv2d, filtered_lrelu, conv annotators) for kernel-level timing; built alongside the GPU tests.
