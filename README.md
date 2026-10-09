# brotensor

[![CI](https://github.com/wlejon/brotensor/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brotensor/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brotensor/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brotensor/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A C++20 tensor + ops library with **one tensor type** and **interchangeable backends** — CPU (always built), CUDA, Metal and Vulkan (all optional; Vulkan is the AMD GPU path). Every op is device-neutral: you write

```cpp
brotensor::linear_forward(W, b, x, y);
```

once, and it runs on whichever device the tensors live on. No `_cpu` / `_gpu` suffixes, no separate host/device tensor types, no template parameters — a `Tensor` carries a runtime `Device` tag and ops dispatch on it.

brotensor is the shared tensor layer of the [bro ecosystem](https://github.com/wlejon/bro/blob/main/docs/ecosystem.md): brolm, brodiffusion, brosoundml, brovisionml, brogameagent and broimage build on it, and [bro](https://github.com/wlejon/bro) exposes it to apps as `bro.tensor` through the JavaScript binding in `src/api/` (`brotensor_api`). It depends on no other library repository; the binding needs bronze (the JavaScript compiler and runtime) and brass (its code generator), pinned dependencies fetched at configure, and brass also gives the CPU backend a JIT for fused elementwise and row-norm traces. There are no third-party libraries to install — the CPU backend is plain C++ (AVX2 on x86-64), and the GPU backends use the SDKs that ship with their own toolchains.

## Platforms and backends

| Backend | Where | Status in CI |
|---------|-------|--------------|
| CPU | Windows, Linux, macOS (x86-64 and arm64) | built and tested on every push |
| CUDA | Windows, Linux (NVIDIA) | compiled on Linux; tests run on real hardware off CI |
| Metal | macOS 15+ (Apple silicon) | built and tested on the hosted macOS runner, CPU↔Metal parity included |
| Vulkan | any GPU with a Vulkan 1.2+ driver (the AMD path) | not in CI; tested on hardware |

## What's inside

- **A forward + backward op surface** covering the dense / attention / normalization / convolution / loss / optimizer core, plus dedicated families for:
  - **LLM inference** — RoPE (incl. M-RoPE), RMSNorm, SwiGLU, KV-cache append, causal flash-decode with GQA, GQA prefill attention (causal *or* bidirectional, for LLM2Vec-style encoders), Gated DeltaNet linear attention, GGUF fused quant matmul
  - **Diffusion inference** — conv2d, GroupNorm, cross-attention, fused ResBlock, AdaLN modulate, fused DDIM / Euler / DPM++ 2M sampler steps (SD 1.5, SDXL, DiT)
  - **Audio (TTS / STT / codecs)** — FFT/STFT spectral core, 1D convolution (incl. transposed + streaming causal), vocoder activations, VQ/FSQ codec quantization, resampling, autoregressive logit sampling
  - **Vision** — SAM/ViTDet decomposed-rel-pos attention, window partition, Qwen-VL spatial merge, deformable conv2d, interp2d, image preprocessing
  - **Training building blocks** — flash attention with backward, LSTM with full BPTT, LoRA adapters, StyleGAN3 generator primitives (modulated conv, upfirdn2d, filtered lrelu), SGD/Adam
- **Precision & quantization** — the CPU backend is the complete FP32 reference; the GPU backends add FP16/BF16 paths, INT8 weight-only matmul/conv (W8A16), and GGUF block-quant kernels (Q4_K / Q6_K / Q8_0)
- **Model loading** — mmap'd zero-copy readers for **safetensors** (also writes) and **GGUF**
- **Graph capture** (`<brotensor/cuda_graph.h>`, device-neutral: CUDA graphs, Vulkan re-submitted command buffers; gate on `graph_capture_available()`) — capture a fixed-shape step once and replay it with a single launch, amortising per-kernel launch overhead in tight decode loops. `Tensor::resize` keeps device pointers stable across shape cycles so captured buffers stay valid

See [docs/op-coverage.md](docs/op-coverage.md) for the full per-op coverage tables.

## Quick start

Consumers resolve brotensor the way every repo in the ecosystem resolves a sibling: an existing `brotensor` target wins, then a working tree beside the top-level project at `../brotensor`, then the commit the consumer pins with `bro_dependency(brotensor GITHUB wlejon/brotensor REF <sha>)` (`cmake/bro_deps.cmake`), fetched at configure. Then link the interface target:

```cmake
target_link_libraries(my_app PRIVATE brotensor::brotensor)
```

```cpp
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/ops.h>

using namespace brotensor;

int main() {
    init();  // probe + register GPU backends (CPU works even without this)

    // Host data -> tensors on the best available device (CUDA > Metal > Vulkan > CPU).
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

Requires CMake ≥ 3.24 and a C++20 compiler; a plain `git clone` is enough, there are no submodules. [bronze](https://github.com/wlejon/bronze) and, through it, [brass](https://github.com/wlejon/brass) are dependencies resolved by `cmake/bro_deps.cmake`: a working tree at `../bronze` / `../brass` when there is one, otherwise the commit `CMakeLists.txt` pins, fetched at configure (override with `-DFETCHCONTENT_SOURCE_DIR_BRONZE=<path>`). They compile inside the build tree. CUDA additionally needs the CUDA Toolkit (nvcc); Vulkan needs `glslc` (shaderc) plus the Vulkan headers and loader; Metal needs the Apple toolchain and **macOS 15 or newer** — the backend builds offset-backed `MPSGraphTensorData` via `-[MPSNDArray initWithBuffer:offset:descriptor:]`, which the macOS 14 SDK does not declare.

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

# CPU + Vulkan (AMD's backend; any Vulkan 1.2+ GPU)
cmake -B build_vk -DCMAKE_BUILD_TYPE=Release -DBROTENSOR_WITH_VULKAN=ON
cmake --build build_vk
```

CPU is always built; the GPU backends are additive. CUDA / Metal don't share a host; Vulkan sits beside either. Without CUDA or Metal, Vulkan is the default device. Without CUDA, `Device::CUDA` aliases to the GPU that is there: Metal, else Vulkan (`detail::resolve_device_alias`). Vulkan design and coverage: [docs/vulkan.md](docs/vulkan.md), [docs/vulkan-coverage.md](docs/vulkan-coverage.md), performance: [docs/vulkan-perf.md](docs/vulkan-perf.md). Build internals (library targets, preprocessor defines, backend registration) are covered in [docs/architecture.md](docs/architecture.md#build-internals).

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

**CI.** GitHub Actions builds and tests the CPU tier on Linux (GCC + Clang), Windows (MSVC) and macOS/arm64; builds *and runs* the Metal backend on macOS (the hosted runner has a real Metal device, so CPU↔Metal parity is verified on every push); and compiles the CUDA backend on Linux — compile-only, since hosted runners have no NVIDIA GPU, so CUDA parity runs on real hardware off CI. The coverage numbers land in each run's job summary, and the [line-by-line report](https://wlejon.github.io/brotensor/coverage/) is published on every push to `main`.

**Static analysis.** A [CodeQL](.github/workflows/codeql.yml) run analyses the core and the CPU backend on every push and once a week, with results in the repository's Security tab. It is pointed primarily at the safetensors and GGUF readers: both mmap a file supplied by the user and index into it using header-declared offsets, which is where a memory-safety bug in this library would realistically live.

## Documentation

Published at **[wlejon.github.io/brotensor](https://wlejon.github.io/brotensor/)** — the pages below plus the line-by-line coverage report, rebuilt on every push to `main`.

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The design: the unified `Tensor`, the dtype system, runtime dispatch, backend registration, device policy, streams/sync, error handling, build internals, and how to add an op |
| [docs/api.md](docs/api.md) | API reference: `Tensor` factories and accessors, runtime functions, the safetensors and GGUF loaders |
| [docs/op-coverage.md](docs/op-coverage.md) | Full op surface: per-header table, per-op backend/dtype coverage, the audio op family, test layout |

## Versioning

Pre-1.0: the op surface is still growing and the ABI can move between minor versions. Siblings vendor the repo and build from source, so a tag is a pin point rather than a compatibility promise — `find_package(brotensor 0.1)` accepts `0.1.x` and rejects `0.2`.

## License

[MIT](LICENSE)
