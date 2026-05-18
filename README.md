# brotensor

GPU tensor + ops library. CUDA and Metal backends, identical op signatures, flat `brotensor::` namespace.

Forward + backward primitives for dense layers, elementwise activations, softmax, layernorm, attention (single + multi-head), embedding lookup, concat/split, SGD + Adam, MSE + cross-entropy, plus batched inference variants. FP16 storage tag on `GpuTensor` plus a diffusion-oriented op set (conv2d, GroupNorm, SiLU/GELU, 2x up/downsample, cross-attention) for downstream brodiff inference.

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
| `brotensor::*_forward_gpu` / `*_backward_gpu` | Op primitives (see `include/brotensor/ops.h`) |
| `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` | Backend identifier defines |
| `BROTENSOR_HAS_GPU` | Umbrella define (true if either backend is on) |
| `BROTENSOR_CUDA_CHECK(expr)` | Error-check macro for CUDA calls |

## Op coverage

| Op | FP32 fwd | FP32 bwd | FP16 fwd | Notes |
|---|---|---|---|---|
| linear | ✓ | ✓ | ✓ | dense; FP32 single + batched (fwd/bwd), FP16 batched-inference and batched-train backward (dtype-dispatched, FP32 scratch + fold) |
| relu / tanh / sigmoid | ✓ | ✓ | — | elementwise; relu/tanh also have batched fwd+bwd |
| silu / gelu | ✓ | ✓ | ✓ | tanh-approx GELU; dtype-dispatched (FP16 bwd accumulates in FP32) |
| gelu_exact | ✓ | ✓ | ✓ | `0.5*x*(1+erf(x/√2))`, exact PyTorch/diffusers default |
| quick_gelu | ✓ | ✓ | ✓ | `x * sigmoid(1.702*x)`, OpenAI CLIP activation |
| geglu | ✓ | ✓ | ✓ | gated GELU (SD FFN); FP32+FP16 fwd/bwd, dtype-dispatched |
| geglu_exact | ✓ | ✓ | ✓ | gated exact-GELU FFN, matches diffusers GEGLU |
| add / scale / mul_inplace | ✓ | n/a | ✓ | dtype-dispatched |
| clamp | ✓ | n/a | ✓ | in-place min/max, dtype-dispatched (VAE epilogue) |
| build_slot_mask | ✓ | n/a | — | device-side validity mask construction |
| softmax | ✓ | ✓ | — | masked, numerically stable |
| layernorm | ✓ | ✓ | ✓ | FP32 single + batched-infer; FP16 batched-infer + backward (dtype-dispatched, FP32 scratch + fold for dGamma/dBeta) |
| group_norm | ✓ | ✓ | ✓ | NCHW, per-group stats; dtype-dispatched fwd+bwd (FP16 bwd accumulates in FP32) |
| attention (single-head) | ✓ | ✓ | — | |
| mha (multi-head) | ✓ | ✓ | — | |
| self_attention | ✓ | ✓ | ✓ | FP32 = training (caches exposed via `_train`); FP16 = flash inference |
| cross_attention | ✓ | ✓ | ✓ | FP32 = training (caches exposed via `_train`, rectangular Wk/Wv); FP16 = flash inference |
| flash_attention | — | — | ✓ | tiled online-softmax, Lk-unbounded, optional causal |
| flash_attention_qkvo | — | — | ✓ | fused Q/K/V/O projections + biases; rectangular Wk/Wv for cross-attn; optional causal; verified at SD1.5 U-Net head_dims (40/80/160) and CLIP head_dim 64 |
| resblock | — | — | ✓ | fused diffusion ResBlock (GN→SiLU→conv ×2 + skip) |
| conv2d | ✓ | ✓ | ✓ | NCHW, groups=1, stride/pad/dil; backward (dX, dW, dB) dtype-dispatched (FP32+FP16; FP16 dW/dB use FP32 scratch + fold) |
| upsample_nearest_2x | ✓ | ✓ | ✓ | backward dtype-dispatched (FP32+FP16) |
| upsample_bilinear_2x | ✓ | ✓ | ✓ | align_corners=False; backward dtype-dispatched (FP32+FP16; FP16 uses FP32 scratch + fold) |
| downsample_avg_2x | ✓ | ✓ | ✓ | stride 2, kernel 2; backward dtype-dispatched (FP32+FP16) |
| nchw ↔ sequence transpose | ✓ | n/a | ✓ | gather/scatter between NCHW and (L,D) layouts |
| embedding lookup | ✓ | ✓ | ✓ | FP32/FP16 table dispatch; backward dtype-dispatched (FP16 uses FP32 scratch + fold for atomic-add safety) |
| concat_rows / split_rows | ✓ | ✓ | ✓ | flat byte-aware concat (FP16 supported) |
| concat_batched_rows | ✓ | n/a | ✓ | per-row column-block concat via 2D memcpy |
| concat_nchw_channels | ✓ | n/a | ✓ | channel-axis concat for U-Net skip merges (N≥1) |
| masked_mean_pool | ✓ | ✓ | — | row-wise mean over valid mask |
| copy_d2d | ✓ | n/a | ✓ | flat-buffer device-to-device chunk copy |
| build_causal_mask_row | n/a | n/a | ✓ | length-L FP32 mask, CLIP text |
| sgd / adam | ✓ | n/a | — | optimizer steps |
| mse / softmax-xent | ✓ | ✓ | — | per-sample + batched |
