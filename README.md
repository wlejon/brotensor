# brotensor

Tensor + ops library. One tensor type, one flat `brotensor::` namespace, three backends — CPU (always built), CUDA and Metal (optional, additive) — selected at runtime per tensor.

Forward + backward primitives for dense layers, elementwise activations, softmax, layernorm/RMSNorm, attention (single + multi-head + flash), embedding lookup, concat/split, SGD + Adam, MSE + cross-entropy, plus batched inference variants. The GPU backends add an FP16 precision path, a diffusion-oriented op set (conv2d, GroupNorm, SiLU/GELU, 2× up/downsample, cross-attention, fused DDIM/Euler/DPM++ 2M sampler steps, sinusoidal timestep embedding) for downstream `brodiffusion` inference (SD 1.5 + SDXL), LLM-oriented primitives (RoPE, RMSNorm, SwiGLU, KV-cache append + causal flash-decode) for autoregressive inference, and INT8 weight-only matmul/conv2d (W8A16) for memory-bound deployment.

An FP32 **audio-ML op family** — FFT/STFT spectral core, 1D convolution (incl. transposed + streaming), vocoder/codec activations, codec quantization, resampling, and an autoregressive logit sampler — runs on all three backends for downstream `brosoundml` (TTS / STT / neural-codec) inference.

Built as a standalone sibling so multiple downstream projects (`brogameagent`, `brodiffusion`, …) share one tensor layer. Both vendor it in as an `add_subdirectory` dependency — no system deps, no release process.

## Model

A single `brotensor::Tensor` is a row-major `(rows, cols)` buffer carrying two runtime tags: a `Dtype` (FP32 / FP16 / INT8 / INT32) and a `Device` (CPU / CUDA / Metal). There is **no separate host/device tensor type**.

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

CPU is always built. CUDA and Metal are additive and mutually exclusive — at most one GPU backend per binary. `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` are set per backend; `BROTENSOR_HAS_GPU` is the umbrella. Most code never needs them — the unified `Tensor` and op surface compile identically regardless of backend; reach for the defines only to gate a path that genuinely needs a GPU device present.

## Tests

```bash
ctest --test-dir build -C Release
```

`test_cpu_ops.cpp` and `test_dispatch.cpp` are CPU-only and always built. The rest are GPU-gated (built only with a CUDA or Metal backend):

- `test_cpu_gpu_parity.cpp` — monolithic CPU↔GPU parity.
- `test_*_parity.cpp` — per-op CPU↔GPU parity suite, one executable per op group (linear, elementwise, softmax, layernorm, attention, optim, adam, reduce, loss, embedding, concat, mha, batched), sharing the `parity_helpers.h` harness. Each runs the same device-neutral op on CPU- and GPU-resident tensors and asserts the results match.
- Diffusion / LLM kernels (conv2d, group_norm, flash attention, RoPE, INT8, …) have dedicated GPU smoke tests.

## API surface

| Symbol | Meaning |
|---|---|
| `brotensor::Tensor` | Row-major `(rows, cols)` buffer with runtime `Dtype` + `Device` tags. Copyable (device-aware deep copy) + movable. |
| `Tensor::mat(r,c)` / `Tensor::vec(n)` | Zero-filled FP32 **host** (CPU) factories — build params on the host, then migrate. |
| `Tensor::zeros[_on]` / `empty[_on]` | Allocate on the default device (or an explicit one). `zeros` zero-fills; `empty` and `resize()` leave contents **undefined**. |
| `Tensor::from_host[_on]` / `to_host_vector` / `copy_to_host` | Host↔device bootstrap and readback (FP32 + FP16 variants). |
| `Tensor::to(Device)` | Returns a copy migrated to another backend; source unchanged. `clone()` is a device-preserving deep copy. |
| `Tensor::view(Device, ptr, r, c)` | Non-owning view over an existing backend-resident pointer. |
| `brotensor::Device { CPU, CUDA, Metal }` / `Dtype { FP32, FP16, INT8, INT32 }` | Runtime tags carried on every tensor. |
| `brotensor::init()` | Idempotent. Probes + registers the CUDA / Metal backends (CPU is always registered). |
| `default_device()` / `set_default_device()` / `DeviceScope` | Where new `zeros`/`empty`/`from_host` tensors land (best-available: CUDA > Metal > CPU; overridable, also via the `BROTENSOR_DEFAULT_DEVICE` env var). |
| `available_devices()` / `is_available(Device)` | Backends registered in this binary at runtime. |
| `sync(Device)` / `sync_all()` | Drain pending backend work (no-op on CPU). |
| `<brotensor/ops.h>` | The device-neutral op surface; every op dispatches on its operands' `Device`. |

Backends throw plain `std::runtime_error` (`"brotensor: <op>: <reason>"`) for precondition / dispatch failures.

## Op coverage

All ops live in `<brotensor/ops.h>` and are device-neutral. The **CPU backend** implements the dense / attention / loss / optimizer subset that drives autograd-free training (FP32 only). The **CUDA and Metal backends** additionally implement the FP16 / INT8 precision paths, batched-inference variants, and the diffusion / LLM kernel set. A separate FP32 [audio op family](#audio-op-family) is implemented on **all three** backends.

### CPU backend

FP32 only — scalar, autovectorize-friendly loops; the simple, correct fallback. Forward + backward for:

| Group | Ops |
|---|---|
| Dense | `linear` (single + batched) |
| Activations | `relu` / `tanh` / `sigmoid` (single + batched), `add_inplace` / `add_scalar_inplace` / `scale_inplace` |
| Attention | `softmax`, `layernorm`, `attention` (single-head), `mha` (multi-head) |
| Pooling | `masked_mean_pool`, `build_slot_mask` |
| Loss | `softmax_xent` (+ `_segment`, `_fused_batched`), `mse_scalar`, `mse_vec_per_sample` |
| Optimizers | `sgd_step`, `adam_step` |
| Plumbing | `concat_rows` / `split_rows`, `copy_d2d`, `xavier_init` |

This is the code path `brogameagent`'s hand-crafted ExIt circuits use by default; FP16 / INT8 / diffusion / LLM ops are GPU-only by design.

### GPU backends (CUDA / Metal)

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
| flash_attention_qkvo | — | — | ✓ (fwd) / ✓ (bwd) | fused Q/K/V/O projections + biases; rectangular Wk/Wv for cross-attn; optional causal; verified at SD1.5 U-Net head_dims (40/80/160) and CLIP head_dim 64. FP16 backward via recompute (no fwd-time caches); CUDA only — Metal bwd throws. **W8A16 variant** (`flash_attention_qkvo_int8w_fp16`) routes all four projections through `linear_forward_batched_int8w_fp16`; attention core stays FP16 |
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
| mse / softmax-xent | ✓ | ✓ | — | per-sample + batched |

### Audio op family

An FP32 op family for TTS / STT / neural-codec inference, consumed by the `brosoundml` sibling. Unlike the FP16 / INT8 / diffusion / LLM kernels, these are implemented on **all three backends** — CPU, CUDA, and Metal — FP32 throughout, with per-family CPU↔GPU parity tests.

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