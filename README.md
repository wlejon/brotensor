# brotensor

[![CI](https://github.com/wlejon/brotensor/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brotensor/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brotensor/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brotensor/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A C++20 tensor + ops library with **one tensor type** and **three interchangeable backends** — CPU (always built), CUDA, and Metal (both optional). Every op is device-neutral: you write

```cpp
brotensor::linear_forward(W, b, x, y);
```

once, and it runs on whichever device the tensors live on. No `_cpu` / `_gpu` suffixes, no separate host/device tensor types, no template parameters — a `Tensor` carries a runtime `Device` tag and ops dispatch on it.

brotensor is the shared tensor layer for a family of sibling projects (`brodiffusion`, `brolm`, `brosoundml`, `brovisionml`, `brogameagent`, …). Each vendors it via CMake `add_subdirectory`. There are no third-party library dependencies to install — the CPU backend is scalar C++, and the GPU backends use the SDKs that ship with their own toolchains.

## What's inside

- **A forward + backward op surface** covering the dense / attention / normalization / convolution / loss / optimizer core, plus dedicated families for:
  - **LLM inference** — RoPE (incl. M-RoPE), RMSNorm, SwiGLU, KV-cache append, causal flash-decode with GQA, Gated DeltaNet linear attention, GGUF fused quant matmul
  - **Diffusion inference** — conv2d, GroupNorm, cross-attention, fused ResBlock, AdaLN modulate, fused DDIM / Euler / DPM++ 2M sampler steps (SD 1.5, SDXL, DiT)
  - **Audio (TTS / STT / codecs)** — FFT/STFT spectral core, 1D convolution (incl. transposed + streaming causal), vocoder activations, VQ/FSQ codec quantization, resampling, autoregressive logit sampling
  - **Vision** — SAM/ViTDet decomposed-rel-pos attention, window partition, Qwen-VL spatial merge, deformable conv2d, interp2d, image preprocessing
  - **Training building blocks** — flash attention with backward, LSTM with full BPTT, LoRA adapters, StyleGAN3 generator primitives (modulated conv, upfirdn2d, filtered lrelu), SGD/Adam
- **Precision & quantization** — the CPU backend is the complete FP32 reference; the GPU backends add FP16/BF16 paths, INT8 weight-only matmul/conv (W8A16), and GGUF block-quant kernels (Q4_K / Q6_K / Q8_0)
- **Model loading** — mmap'd zero-copy readers for **safetensors** (also writes) and **GGUF**

See [docs/op-coverage.md](docs/op-coverage.md) for the full per-op coverage tables.

## Quick start

Vendor it and link the interface target:

```cmake
add_subdirectory(brotensor)
target_link_libraries(my_app PRIVATE brotensor::brotensor)
```

```cpp
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/ops.h>

using namespace brotensor;

int main() {
    init();  // probe + register GPU backends (CPU works even without this)

    // Host data -> tensors on the best available device (CUDA > Metal > CPU).
    float w[6] = {1, 2, 3, 4, 5, 6};           // W: (2,3), row-major
    float v[3] = {1, 0, -1};
    Tensor W = Tensor::from_host(w, 2, 3);
    Tensor b = Tensor::zeros(2, 1);
    Tensor x = Tensor::from_host(v, 3, 1);

    // Device-neutral op: dispatches on the operands' Device tag.
    // y is uncommitted (no storage yet) -> the op allocates it on the same device.
    Tensor y;
    linear_forward(W, b, x, y);                // y = W x + b

    sync_all();                                // drain GPU work before readback
    std::vector<float> out = y.to_host_vector();   // {-2, -2}
}
```

## Build

Requires CMake ≥ 3.24 and a C++20 compiler. CUDA additionally needs the CUDA Toolkit (nvcc); Metal needs the Apple toolchain and **macOS 15 or newer** — the backend builds offset-backed `MPSGraphTensorData` via `-[MPSNDArray initWithBuffer:offset:descriptor:]`, which the macOS 14 SDK does not declare.

```bash
# CPU-only (any OS)
cmake -B build
cmake --build build --config Release

# CPU + CUDA (NVIDIA)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# CPU + Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

CPU is always built; CUDA and Metal are additive. In practice a binary carries at most one GPU backend, since the nvcc and Apple toolchains don't coexist on one host. Build internals (library targets, preprocessor defines, backend registration) are covered in [docs/architecture.md](docs/architecture.md#build-internals).

## Tests

```bash
ctest --test-dir build -C Release
```

CPU tests always build; CPU↔GPU parity and GPU smoke tests build only when a GPU backend is enabled and skip cleanly otherwise. See [docs/op-coverage.md](docs/op-coverage.md#tests) for the layout.

Every op in the table is exercised by at least one test.

**Coverage.** `-DBROTENSOR_COVERAGE=ON` instruments the core + CPU backend for gcov (GCC/Clang; `-O0`, so use a fresh build dir). Enable a GPU backend alongside it:

```bash
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DBROTENSOR_COVERAGE=ON -DBROTENSOR_WITH_CUDA=ON
cmake --build build-cov
ctest --test-dir build-cov
gcovr --root . --filter src/ --exclude src/cuda/ --exclude src/metal/ --print-summary
```

Enabling a GPU backend matters: most of `src/cpu/` is exercised by the parity suite — every parity test calls the CPU op as its reference — and that suite doesn't build at all without one. A CPU-only coverage run measures the CPU backend with the majority of the tests that exercise it excluded.

The GPU backends themselves are compiled by nvcc / the Apple toolchain and aren't gcov-instrumented, so they're excluded rather than counted as 0%.

**CI.** GitHub Actions builds and tests the CPU tier on Linux (GCC + Clang), Windows (MSVC) and macOS/arm64; builds *and runs* the Metal backend on macOS (the hosted runner has a real Metal device, so CPU↔Metal parity is verified on every push); and compiles the CUDA backend on Linux — compile-only, since hosted runners have no NVIDIA GPU, so CUDA parity runs on real hardware off CI. The coverage numbers land in each run's job summary, with the line-by-line HTML report attached as a build artifact.

**Static analysis.** A [CodeQL](.github/workflows/codeql.yml) run analyses the core and the CPU backend on every push and once a week, with results in the repository's Security tab. It is pointed primarily at the safetensors and GGUF readers: both mmap a file supplied by the user and index into it using header-declared offsets, which is where a memory-safety bug in this library would realistically live.

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The design: the unified `Tensor`, the dtype system, runtime dispatch, backend registration, device policy, streams/sync, error handling, build internals, and how to add an op |
| [docs/api.md](docs/api.md) | API reference: `Tensor` factories and accessors, runtime functions, the safetensors and GGUF loaders |
| [docs/op-coverage.md](docs/op-coverage.md) | Full op surface: per-header table, per-op backend/dtype coverage, the audio op family, test layout |

## License

[MIT](LICENSE)
