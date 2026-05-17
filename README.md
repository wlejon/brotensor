# brotensor

GPU tensor + ops library. CUDA and Metal backends, identical op signatures, flat `brotensor::` namespace.

Forward + backward primitives for dense layers, elementwise activations, softmax, layernorm, attention (single + multi-head), embedding lookup, concat/split, SGD + Adam, MSE + cross-entropy, plus batched inference variants.

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
