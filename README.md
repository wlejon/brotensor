# brotensor

GPU tensor + ops library. CUDA and Metal backends, identical op signatures, flat `brotensor::` namespace.

Forward + backward primitives for dense layers, elementwise activations, softmax, layernorm/RMSNorm, attention (single + multi-head + flash), embedding lookup, concat/split, SGD + Adam, MSE + cross-entropy, plus batched inference variants. FP16 storage tag on `GpuTensor` plus a diffusion-oriented op set (conv2d, GroupNorm, SiLU/GELU, 2x up/downsample, cross-attention, fused DDIM/Euler/DPM++ 2M sampler steps, sinusoidal timestep embedding) for downstream brodiff inference (SD 1.5 + SDXL), and LLM-oriented primitives (RoPE, RMSNorm, SwiGLU, KV-cache append + causal flash-decode) for autoregressive inference. INT8 weight-only matmul/conv2d (W8A16) for memory-bound deployment, and thread-local CUDA stream control for pipelined inference.

Built as a standalone sibling so multiple downstream projects (brogameagent, future brodiff, …) share one GPU layer.

## Build

```bash
# CUDA (NVIDIA, any OS)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

Exactly one backend must be selected at configure time; they are mutually exclusive.

## Namespace + defines

| Symbol | Meaning |
|---|---|
| `brotensor::GpuTensor` | Device-resident (rows, cols) float32 tensor, move-only |
| `brotensor::cuda_init()` / `cuda_sync()` | Backend init / synchronise |
| `brotensor::cuda_set_stream()` / `cuda_current_stream()` / `cuda_stream_sync()` | Thread-local stream control; hot ops (matmul, fp16 matmul, flash_attention causal, conv2d_forward direct) launch on the current stream. Metal: no-op for source compat. |
| `brotensor::Dtype::FP32 / FP16 / INT8` | Tensor dtype tag; INT8 used for W8A16 weight-only quant |
| `brotensor::*_forward_gpu` / `*_backward_gpu` | Op primitives (see `include/brotensor/ops.h`) |
| `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` | Backend identifier defines |
| `BROTENSOR_HAS_GPU` | Umbrella define (true if either backend is on) |
| `BROTENSOR_CUDA_CHECK(expr)` | Error-check macro for CUDA calls |

## Op coverage

| Op | FP32 fwd | FP32 bwd | FP16 fwd | Notes |
|---|---|---|---|---|
| matmul | ✓ | ✓ | ✓ | plain row-major `A @ B` (no bias); dtype-dispatched FP32 + FP16 (FP32 accumulation); backward returns dA/dB (caller zeros, op accumulates; FP16 uses FP32 scratch + fold) |
| matmul_int8w_fp16 | — | — | ✓ | W8A16 weight-only matmul; INT8 weights + per-row FP32 scales, FP16 acts, FP32 accum |
| linear_batched_int8w_fp16 | — | — | ✓ | W8A16 batched linear in (B,in)→(B,out) layout; fused FP16 bias add; mirrors `linear_forward_batched_fp16_gpu` shape contract; WMMA fast path for K%8==0 (FP16 tensor cores with INT8→FP16 dequant on shared-mem load), tiled fallback otherwise |
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
| flash_attention_qkvo | — | — | ✓ (fwd) / ✓ (bwd) | fused Q/K/V/O projections + biases; rectangular Wk/Wv for cross-attn; optional causal; verified at SD1.5 U-Net head_dims (40/80/160) and CLIP head_dim 64. FP16 backward via recompute (no fwd-time caches); CUDA only — Metal bwd throws. **W8A16 variant** (`flash_attention_qkvo_int8w_fp16`) routes all four projections through `linear_forward_batched_int8w_fp16_gpu`; attention core stays FP16 |
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
