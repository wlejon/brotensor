# CLAUDE.md — brotensor

Tensor + ops library shared by sibling projects (`brogameagent`, `brodiffusion`, …). One flat namespace `brotensor::`, three backends: CPU (always built), CUDA, Metal (latter two additive and mutually exclusive).

## Layout

```
include/brotensor/
  tensor.h        Tensor (host, std::vector-backed) + GpuTensor (device) + Dtype + upload/download
  device.h        enum class Device { CPU, GPU } + device_require_gpu()
  ops_cpu.h       CPU op declarations (suffixed _cpu, over Tensor)
  ops.h           GPU op declarations (suffixed _gpu, over GpuTensor)
  runtime.h       cuda_init / sync / stream control (Metal: no-ops)
  device_buffer.h DeviceBuffer<T> helper

src/
  cpu/            *.cpp — scalar FP32 ops (always compiled)
  cuda/           *.cu  — CUDA kernels (gated on BROTENSOR_WITH_CUDA)
  metal/          *.mm  — Metal kernels (gated on BROTENSOR_WITH_METAL)
```

## Build

```sh
# CPU-only
cmake -S . -B build && cmake --build build --config Release

# CPU + CUDA
cmake -S . -B build -DBROTENSOR_WITH_CUDA=ON && cmake --build build --config Release

# CPU + Metal (Apple only)
cmake -S . -B build -DBROTENSOR_WITH_METAL=ON && cmake --build build --config Release
```

`BROTENSOR_WITH_CUDA` and `BROTENSOR_WITH_METAL` are mutually exclusive — CMake errors if both are on. CPU has no opt-out.

Defines: `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` set per-backend, `BROTENSOR_HAS_GPU` is the umbrella. Gate any GPU-conditional code on `BROTENSOR_HAS_GPU` unless you specifically need backend identity.

## Tests

```sh
ctest --test-dir build -C Release
```

Tests live under `tests/`, enabled by `BROTENSOR_TESTS=ON` (default ON when standalone).

## Conventions

- **Two tensor types, no runtime device tag.** `brotensor::Tensor` for host, `brotensor::GpuTensor` for device. Dispatch is compile-time via function name (`*_cpu` vs `*_gpu`), not via a runtime field on the tensor. If you ever want one-tensor-with-internal-device, that's a separate redesign — don't sneak it in piecemeal.
- **Op signatures mirror across CPU and GPU.** Same argument order, same shape contracts, same accumulation semantics for backward (caller zeros dW/dB; op accumulates). When adding a CPU op that already has a GPU counterpart, port the contract verbatim and document any FP32-only restriction.
- **CPU is FP32-only.** Don't add FP16 / INT8 paths on the CPU side — those exist on the GPU because they pay for themselves there. CPU's job is to be the simple, correct fallback.
- **GPU dtype dispatch is on `GpuTensor::dtype`.** Ops select FP32 vs FP16 (vs INT8 for W8A16) internally; the public surface takes a single `GpuTensor&` per arg.
- **No host tensor type exposed to GPU sources.** GPU `.cu` / `.mm` files include `<brotensor/tensor.h>` but should not depend on the host `Tensor` for arithmetic — they take raw float pointers (`upload`/`download`) so non-GPU consumers can transfer without pulling NVCC.
- **Streams.** `cuda_set_stream(void*)` is thread-local. Hot ops (matmul, fp16 matmul, flash_attention, conv2d_forward direct) launch on the current stream. Metal is a no-op for source compatibility.
- **Error checks.** Wrap CUDA calls with `BROTENSOR_CUDA_CHECK(expr)`. Throws `std::runtime_error` with file:line.

## When extending

- Adding a new op: declare in the appropriate `ops.h` / `ops_cpu.h`, implement in `src/cuda/`, `src/metal/`, `src/cpu/`. List the new source file under each backend's `target_sources` block in `CMakeLists.txt`. Match the shape contract across all three.
- Adding a new dtype path: extend `Dtype`, update `dtype_size_bytes`, add the path inside the relevant GPU op kernel — don't add per-dtype public entry points.
- Brodiffusion / brogameagent both vendor brotensor via `add_subdirectory`; assume your changes will be consumed without a release process. Don't break the public ABI casually.
