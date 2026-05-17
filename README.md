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
| linear | ✓ | ✓ | — | dense layer |
| relu / tanh / sigmoid | ✓ | ✓ | — | elementwise |
| silu / gelu | ✓ | — | ✓ | tanh-approx GELU |
| softmax | ✓ | ✓ | — | masked, numerically stable |
| layernorm | ✓ | ✓ | — | single-vec + batched-infer |
| group_norm | — | — | ✓ | NCHW, per-group stats |
| attention (single-head) | ✓ | ✓ | — | |
| mha (multi-head) | ✓ | ✓ | — | |
| cross_attention | — | — | ✓ | thin wrapper, FP16 inference |
| conv2d | — | — | ✓ | NCHW, groups=1, stride/pad/dil |
| upsample_nearest_2x | — | — | ✓ | |
| upsample_bilinear_2x | — | — | ✓ | align_corners=False |
| downsample_avg_2x | — | — | ✓ | stride 2, kernel 2 |
| embedding lookup | ✓ | ✓ | — | |
| concat / split | ✓ | ✓ | — | |
| sgd / adam | ✓ | n/a | — | optimizer steps |
| mse / softmax-xent | ✓ | ✓ | — | per-sample + batched |
