# brotensor

Tensor + ops library. One tensor type, one flat `brotensor::` namespace, three backends — CPU (always built), CUDA and Metal (optional, additive) — selected at runtime per tensor.

Forward + backward primitives for dense layers, elementwise activations, GLU gates (GEGLU/SwiGLU), softmax, LayerNorm/RMSNorm/GroupNorm/BatchNorm, attention (single + multi-head + flash + windowed + var-length + decomposed-rel-pos + T5/ALiBi bias), RoPE / M-RoPE, embedding/gather, concat/split, conv1d/2d/3d (+ transposed), pooling, resampling, SGD + Adam, MSE + cross-entropy + BCE, plus batched-inference variants. CPU runs the whole forward+backward surface in FP32. The GPU backends add an FP16 (and BF16) precision path, a diffusion-oriented op set (conv2d, GroupNorm, SiLU/GELU, 2× up/downsample, cross-attention, AdaLN modulate, fused DDIM/Euler/DPM++ 2M sampler steps, sinusoidal timestep embedding) for downstream `brodiffusion` inference (SD 1.5 + SDXL + DiT), LLM-oriented primitives (RoPE, RMSNorm, SwiGLU, KV-cache append + causal flash-decode with GQA, Gated DeltaNet linear attention) for autoregressive inference, INT8 weight-only matmul/conv (W8A16), and GGUF block-quant GEMV/GEMM (Q4_K / Q6_K / Q8_0, W4/6/8-A16) for memory-bound deployment.

An FP32 **audio-ML op family** — FFT/STFT spectral core, 1D convolution (incl. transposed + streaming), vocoder/codec activations, codec quantization, resampling, and an autoregressive logit sampler — runs on all three backends for downstream `brosoundml` (TTS / STT / neural-codec) inference. A growing set of **vision primitives** (image normalize, NHWC→NCHW, conv3d, window partition, spatial 2×2 merge, decomposed-rel-pos attention, adaptive/max pool, arbitrary-scale interp2d, convex upsample) backs `brovisionml` and the Qwen-VL/SAM-style backbones.

Built as a standalone sibling so multiple downstream projects (`brogameagent`, `brodiffusion`, `brolm`, `brosoundml`, `brovisionml`, …) share one tensor layer. Each vendors it in as an `add_subdirectory` dependency — no system deps, no release process.

## Model

A single `brotensor::Tensor` is a row-major `(rows, cols)` buffer carrying two runtime tags: a `Dtype` and a `Device` (CPU / CUDA / Metal). There is **no separate host/device tensor type**.

`Dtype` is `FP32 / FP16 / BF16 / INT8 / INT32` plus the opaque GGUF block-quant carriers (`Q4_0 … Q8_K`). FP32/FP16/BF16 are the arithmetic dtypes ops dispatch on (BF16 is GPU-only; FP16/BF16 are stored as `uint16_t` bit patterns on the host). INT8/INT32 are pure storage carriers — INT8 backs weight-only quantised matmul/conv (W8A16), INT32 carries device-resident index/offset buffers; no arithmetic op dispatches on them. The GGUF quant dtypes are non-element-addressable block carriers (32-element legacy blocks, 256-element K-quant superblocks) consumed only by the GGUF dequant / fused-matmul ops. Element/block sizing is via `dtype_size_bytes` / `dtype_block_size` / `dtype_block_bytes` / `dtype_storage_bytes` / `dtype_is_quant`.

Every op is device-neutral — `brotensor::linear_forward(W, b, x, y)` — and dispatches to the CPU, CUDA, or Metal backend by its operands' `Device` tag. No `_cpu` / `_gpu` suffixes, no overload set. A backend is a vtable of op + allocator function pointers registered at runtime: the CPU backend self-registers at static-init time and is always present; CUDA / Metal register inside `brotensor::init()` if they were compiled in and probe successfully. Calling an op the operands' backend doesn't implement throws `std::runtime_error`.

## Build

```bash
# CPU-only (any OS)
cmake -B build
cmake --build build --config Release

# CPU + CUDA (NVIDIA, any OS)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# CPU + Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

CPU is always built. CUDA and Metal are additive; CMake no longer forbids enabling both, but they stay exclusive in practice because their toolchains (nvcc vs. the Apple toolchain) don't coexist on one host — so at most one GPU backend per binary. `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` are set per backend; `BROTENSOR_HAS_GPU` is the umbrella. Most code never needs them — the unified `Tensor` and op surface compile identically regardless of backend; reach for the defines only to gate a path that genuinely needs a GPU device present.

Each backend compiles as its own static library (`brotensor_core`, `brotensor_cpu`, `brotensor_cuda`, `brotensor_metal`) and self-registers into the dispatcher; the consumed `brotensor::brotensor` INTERFACE target whole-archives the backend libs so their registration TUs survive the link.

## Tests

```bash
ctest --test-dir build -C Release
```

A large set of tests is **always built** (CPU-only): `test_cpu_ops.cpp`, `test_dispatch.cpp`, plus standalone CPU coverage for the safetensors / GGUF loaders, BF16 basics, the audio family (fft, stft, conv1d, vocoder activations, codec quant, resample1d, log/exp/round, sample_logits, noise), and the newer CPU op families (conv3d, gated_delta_rule, spatial_merge, rope_mrope, interp2d, pad2d/slice2d, top_k, pool2d, gather_rows, conv_transpose2d, window_partition, batch_norm, image_preproc, bce_with_logits, flash_attention_varlen, self_attention_decomposed_rel_pos).

The rest are GPU-gated (built only with a CUDA or Metal backend):

- `test_cpu_gpu_parity.cpp` — monolithic CPU↔GPU parity.
- `test_*_parity.cpp` — per-op CPU↔GPU parity suite, one executable per op group, sharing the `parity_helpers.h` harness. Each runs the same device-neutral op on CPU- and GPU-resident tensors and asserts the results match.
- Diffusion / LLM / vision / INT8 / GGUF-quant kernels have dedicated GPU smoke tests (conv2d, group_norm, flash attention, RoPE, int8 linear/conv WMMA, q4k/q6k/q8_0 parity, SDXL schedulers, …).

## API surface

| Symbol | Meaning |
|---|---|
| `brotensor::Tensor` | Row-major `(rows, cols)` buffer with runtime `Dtype` + `Device` tags. Copyable (device-aware deep copy) + movable. |
| `Tensor::mat(r,c)` / `Tensor::vec(n)` | Zero-filled FP32 **host** (CPU) factories — build params on the host, then migrate. |
| `Tensor::zeros[_on]` / `empty[_on]` | Allocate on the default device (or an explicit one). `zeros` zero-fills; `empty` and `resize()` leave contents **undefined**. |
| `Tensor::from_host[_on]` / `to_host_vector` / `copy_to_host` | Host↔device bootstrap and readback (FP32 / FP16 / BF16 / INT8 variants). |
| `Tensor::to(Device)` | Returns a copy migrated to another backend; source unchanged. `clone()` is a device-preserving deep copy. |
| `Tensor::view(Device, ptr, r, c)` | Non-owning view over an existing backend-resident pointer. |
| `fp32_to_fp16_bits` / `fp16_bits_to_fp32` / `fp32_to_bf16_bits` / `bf16_bits_to_fp32` | Pure-CPU half/bfloat ↔ FP32 bit conversion (tests, small preprocessing). |
| `brotensor::Device { CPU, CUDA, Metal }` / `Dtype { FP32, FP16, BF16, INT8, INT32, Q4_0…Q8_K }` | Runtime tags carried on every tensor. |
| `brotensor::init()` | Idempotent. Probes + registers the CUDA / Metal backends (CPU is always registered). |
| `default_device()` / `set_default_device()` / `DeviceScope` / `compute_dtype()` | Where new `zeros`/`empty`/`from_host` tensors land (best-available: CUDA > Metal > CPU; overridable, also via the `BROTENSOR_DEFAULT_DEVICE` env var). `compute_dtype()` is the dtype a model loader should upload weights at for the current default device (FP32 on CPU, FP16 on a GPU). |
| `available_devices()` / `is_available(Device)` | Backends registered in this binary at runtime. |
| `sync(Device)` / `sync_all()` | Drain pending backend work (no-op on CPU). |
| `<brotensor/ops.h>` | The device-neutral op surface (umbrella over the per-category headers in `<brotensor/ops/>`); every op dispatches on its operands' `Device`. |
| `<brotensor/safetensors.h>` / `<brotensor/gguf.h>` | mmap'd zero-copy weight loaders → `Tensor`. safetensors reads/writes F32/F16/BF16/I32/…; GGUF reads metadata + F32/F16 and the Q4_K/Q6_K/Q8_0 block-quant carriers. |

Backends throw plain `std::runtime_error` (`"brotensor: <op>: <reason>"`) for precondition / dispatch failures.

## Op coverage

All ops are device-neutral and declared in the per-category headers under `<brotensor/ops/>` (the `ls` of that directory is the table of contents; `<brotensor/ops.h>` is an umbrella that includes them all):

| Header | Surface |
|---|---|
| `activation.h` | relu / tanh / sigmoid, silu, gelu (tanh-approx / exact / quick), GEGLU / GEGLU-exact / SwiGLU, snake (BigVGAN/DAC), elu (EnCodec), leaky_relu (HiFi-GAN) |
| `attention.h` | single-head attention, MHA (optional biases), self/cross attention (train + flash), cross-attention with head-avg map + logit bias, attention token moments, self-attention with T5/ALiBi additive bias, SAM/ViTDet decomposed-rel-pos (incl. windowed), W8A16 bias-attention |
| `flash_attention.h` | tiled flash attention (+ bare-core bwd), windowed (sliding-window causal), packed var-length (+ bwd), fused QKV+O projections (+ bwd), project-KV / Q-with-cached-KV, KV-cache append, causal flash-decode (GQA), W8A16 variants |
| `linear.h` | linear (single / batched / fp16 / fused-act-epilogue), matmul (+ bwd), W8A16 batched linear |
| `norm.h` | LayerNorm (single + batched ±caches), RMSNorm, GroupNorm, BatchNorm (train/infer/bwd), per-head L2-norm (Gated DeltaNet), NCHW channel L2-normalize |
| `conv.h` / `conv1d.h` | conv2d / conv3d (+ W8A16), conv_transpose2d, and the 1D family (conv1d wrappers, pad1d, conv_transpose1d, causal_conv1d + streaming update) — all with backward where applicable |
| `rope.h` | RoPE forward/backward, rope_apply (explicit cos/sin tables) + bwd, M-RoPE (Qwen-VL three-axis) |
| `delta_rule.h` | Gated Delta Rule linear attention — chunked prefill + streaming step |
| `diffusion.h` | AdaLN modulate / broadcast_mul, fused ResBlock (+ W8A16, + bwd), DDIM / Euler / DPM++ 2M sampler steps, sinusoidal timestep embedding |
| `spatial.h` | pad2d, slice2d, unfold2d (neighborhood im2col), window partition/reverse (SAM), spatial 2×2 patch merge (Qwen-VL), NCHW↔sequence transpose |
| `resize.h` | 2× nearest/bilinear up + 2× avg down (+ bwd), arbitrary-scale interp2d (nearest/bilinear/bicubic, half-pixel + align-corners), convex (RAFT) upsample, 1D resample |
| `pooling.h` | masked mean-pool, 2× avg downsample, adaptive avg pool2d, max pool2d (+ index bwd) |
| `embedding.h` | embedding lookup (+ scatter bwd), gather_rows / scatter_rows_add |
| `concat.h` | concat/split rows, batched column-block concat, NCHW channel concat (+ bwd), copy_d2d |
| `reduction.h` | sum_rows / sum_cols, argmax_rows, top_k_rows |
| `loss.h` | softmax (+ bwd), softmax-xent (+ segment / fused / fused-batched), MSE (vec / scalar / per-sample), BCE-with-logits fused-batched |
| `optim.h` | sgd_step, adam_step, xavier_init |
| `elementwise.h` | add / scale / clamp / mul-inplace, dtype `cast` (FP32↔FP16/BF16), log / exp / round (+ bwd) |
| `sampling.h` | `sample_logits` (temperature / top-k / top-p / greedy), Philox RNG (`randn` / `rand_uniform` / `rand_bernoulli` / `randn_truncated`) |
| `spectral.h` | complex ops, FFT/iFFT, rFFT/irFFT (+ bwd), STFT/iSTFT (+ bwd) |
| `codec.h` | VQ encode (RVQ codeword search) + FSQ quantize (NanoCodec), straight-through bwd |
| `quant.h` | W8A16 host quantizer + matmul, GGUF Q4_K / Q6_K / Q8_0 dequant + fused GEMV + batched matmul |
| `image.h` | per-channel image normalize, uint8 HWC → FP32 NCHW |

**Backend coverage.** The **CPU backend** implements essentially the entire FP32 surface — forward *and* backward, including the diffusion samplers, flash attention, the audio family, and the vision primitives — as the simple, correct, autovectorize-friendly fallback. CPU is **FP32-only by design**: it leaves the FP16 / BF16 / INT8-W8A16 / GGUF-quant vtable slots null, and the dispatcher throws `"brotensor: <op>: not implemented on CPU"` if you call one. The **CUDA and Metal backends** add the FP16 (and BF16) precision paths, batched-inference variants, the W8A16 and GGUF block-quant kernels, and a handful of GPU-only fused kernels. A few inference-only ops are CPU+CUDA but leave the Metal slot null (noted below). The [audio op family](#audio-op-family) is FP32 on **all three** backends with per-family CPU↔GPU parity tests.

### GPU backends (CUDA / Metal)

FP32 fwd/bwd columns below mirror the CPU surface; the FP16 column is the GPU-only precision path (BF16 follows FP16 where the kernel lists it). INT8-W8A16 and GGUF-quant ops are GPU-only.

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
| build_slot_mask | ✓ | n/a | — | device-side validity mask construction |
| softmax | ✓ | ✓ | — | masked, numerically stable |
| layernorm | ✓ | ✓ | ✓ | FP32 single + batched-infer; FP16 batched-infer + backward (dtype-dispatched, FP32 scratch + fold for dGamma/dBeta) |
| rms_norm | ✓ | ✓ | ✓ | `y = x * gamma / sqrt(mean(x²) + eps)`; dtype-dispatched FP32 + FP16 (FP16 bwd uses FP32 scratch + fold for dGamma) |
| group_norm | ✓ | ✓ | ✓ | NCHW, per-group stats; dtype-dispatched fwd+bwd (FP16 bwd accumulates in FP32) |
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
| flash_attention_decode | — | — | ✓ | causal-aware decode against a partially-filled K/V cache; supports `L_q ≥ 1` (token-by-token or chunked) |
| kv_cache_append | — | — | ✓ | append `L_new` projected K/V rows into a pre-allocated `L_max` cache at `cur_len` |
| rope | ✓ | ✓ | ✓ | rotary position embedding; pair-wise rotation per head_dim chunk, `seq_offset` for KV-cache decode |
| resblock | — | ✓ (bwd) | ✓ | fused diffusion ResBlock (GN→SiLU→conv ×2 + skip); FP16 backward via composition of public ops (recomputes h1/h2/h3; no fwd-time caches) |
| conv2d | ✓ | ✓ | ✓ | NCHW, stride/pad/dil; FP32 fwd ✓ \| FP32 bwd ✓ \| FP16 fwd/bwd ✓ \| groups ≥ 1 (depthwise supported); backward (dX, dW, dB) dtype-dispatched (FP32+FP16; FP16 dW/dB use FP32 scratch + fold) |
| conv2d_int8w_fp16 | — | — | ✓ | W8A16 weight-only conv2d; INT8 OIHW filter + per-output-channel FP32 scales, FP16 acts; CUDA WMMA fast path for 3x3 s1, 1x1 s1, 3x3 s2 (groups=1, dil=1) — naive fallback otherwise |
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
| ddim_step | — | n/a | ✓ | fused DDIM sampler step over FP16 latents; FP32 internal math |
| euler_step | — | n/a | ✓ | fused Euler-discrete step (ε-prediction, σ convention; matches diffusers `EulerDiscreteScheduler`) |
| dpmpp_2m_step | — | n/a | ✓ | fused DPM-Solver++ 2M multistep update; caller supplies linear-combo coefficients and x0 cache. First step falls back to `euler_step` |
| timestep_embedding | ✓ | n/a | — | sinusoidal embedding (FP32) for diffusion timesteps and SDXL added-cond micro-conditioning; diffusers default (flip_sin_to_cos=True) |
| copy_d2d | ✓ | n/a | ✓ | flat-buffer device-to-device chunk copy |
| build_causal_mask_row | n/a | n/a | ✓ | length-L FP32 mask, CLIP text |
| sgd / adam | ✓ | n/a | — | optimizer steps |
| mse / softmax-xent / bce | ✓ | ✓ | — | per-sample + batched (BCE-with-logits fused-batched added) |
| conv3d | ✓ | — | ✓ | NCTHW (forward only), grouped, FP32/FP16/BF16; W8A16 `conv3d_int8w_fp16` for Qwen-VL patch-embed (GPU-only) |
| conv_transpose1d / 2d | ✓ | ✓ | — | learned upsample (vocoders, SAM mask decoder, DPT heads); FP32 on CPU+CUDA |
| batch_norm | ✓ | ✓ | — | NCHW train/infer/bwd, running stats (pretrained ResNet/DETR backbones); FP32 |
| l2_norm / l2_normalize_nchw | ✓ | ✓ (l2_norm) | ✓ | per-head q/k L2 (Gated DeltaNet) + channel-axis NCHW normalize (DSINE normals); FP32/FP16/BF16 |
| gated_delta_rule | ✓ | n/a | ✓ | chunked prefill + streaming step (linear-attention text decoders); FP32/FP16, FP32 accumulators |
| rope_apply / rope_apply_mrope | ✓ | ✓ (apply) | ✓ | explicit cos/sin tables (2D axial RoPE) + Qwen-VL three-axis M-RoPE |
| self_attention_bias | ✓ | n/a | ✓ | additive pre-softmax bias (T5 rel-pos / ALiBi); FP32/FP16/BF16; W8A16 variant GPU-only |
| decomposed_rel_pos (± windowed) | ✓ | n/a | ✓ | SAM/ViTDet data-dependent 2D rel-pos attention; FP32/FP16/BF16 |
| modulate / broadcast_mul | ✓ | n/a | ✓ | AdaLN affine + per-channel gate (DiT/SD3/Flux); FP32/FP16/BF16 |
| pad2d / slice2d / unfold2d | ✓ | ✓ (pad/slice) | ✓ | image pad (zero/reflect/replicate), crop, neighborhood im2col |
| window_partition / spatial_merge | ✓ | n/a | ✓ | SAM window tiling (+ reverse) and Qwen-VL 2×2 patch merge |
| interp2d / convex_upsample | ✓ | ✓ (interp) | ✓ | arbitrary-scale resize (nearest/bilinear/bicubic, half-pixel + align-corners) + RAFT convex upsample |
| max_pool2d / adaptive_avg_pool2d | ✓ | ✓ | — | indexed max-pool bwd + PyTorch adaptive avg pool; FP32 (CPU+CUDA) |
| gather_rows / scatter_rows_add | ✓ | ✓ | — | index-driven row gather/scatter (SAM prompt encoder, DETR queries); FP32 |
| top_k_rows | ✓ | n/a | — | per-row top-k values + indices; FP32 |
| randn / rand_uniform / rand_bernoulli / randn_truncated | ✓ | n/a | — | Philox 4×32-10 RNG, PyTorch/JAX-compatible, FP32, all three backends |
| GGUF Q4_K / Q6_K / Q8_0 | — | — | ✓ | block-quant dequant + fused GEMV + batched matmul (W4/6/8-A16); CUDA WMMA fast paths, GEMV fallback; registered on CUDA **and** Metal |

### Audio op family

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

The conv1d family is header-only wrappers over the conv2d ops (a 1D conv is a 2D conv with `H=kH=1`); `pad1d`, `conv_transpose1d`, and `causal_conv1d_update` are genuine per-backend kernels. The lone exception to the FP32-everywhere rule is `conv1d_int8w_fp16`, a W8A16 wrapper that is GPU-only like its `conv2d_int8w_fp16` parent.

## License

[MIT](LICENSE)