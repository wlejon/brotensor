# CLAUDE.md — brotensor

Tensor + ops library shared by sibling projects (`brogameagent`, `brodiffusion`, …). One flat namespace `brotensor::`. A single unified `Tensor` type carries a runtime `Device` tag; ops dispatch to the registered backend at runtime. Three backends: CPU (always built), CUDA, Metal (latter two optional).

## Layout

```
include/brotensor/
  tensor.h          Tensor (unified host+device) + Dtype + Device + factories,
                    migration (to/clone), mutators, host accessors
  ops.h             Public op declarations — one per op, runtime-dispatched
  runtime.h         init() / default-device policy / compute_dtype() /
                    DeviceScope / sync
  safetensors.h     safetensors reader + writer — File/TensorView + upload* +
                    write_file. Tensor-container format; output type is Tensor
  metal_interop.h   Public Metal custom-kernel surface (Obj-C++ / .mm only)
  detail/op_table.h  X-macro: the single canonical op list
  detail/dispatch.h  OpsVTable / AllocVTable + register_backend + dispatch()

src/
  tensor.cpp        Tensor impl — alloc/clone/to/resize/zero via AllocVTable
  dispatch.cpp      Backend registry + per-operand device resolution
  init.cpp          Runtime: init(), default device, DeviceScope, sync
  ops.cpp           One thin wrapper per op — resolve device, forward to vtable
  safetensors.cpp   safetensors mmap reader + JSON header parser + writer
  cpu/              *.cpp — scalar FP32 backend (always compiled)
  cuda/             *.cu  — CUDA backend (gated on BROTENSOR_WITH_CUDA)
  metal/            *.mm  — Metal backend (gated on BROTENSOR_WITH_METAL)
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

`BROTENSOR_WITH_CUDA` and `BROTENSOR_WITH_METAL` are no longer mutually exclusive at the CMake level — they stay exclusive in practice because their toolchains (nvcc vs. the Apple toolchain) don't coexist on one host. CPU has no opt-out.

Each backend compiles as its own static library and registers itself into the dispatcher. The consumed `brotensor::brotensor` interface target whole-archives the backend libs so their self-registration TUs survive the link.

Defines: `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` set per-backend, `BROTENSOR_HAS_GPU` is the umbrella. Gate any GPU-conditional code on `BROTENSOR_HAS_GPU` unless you specifically need backend identity.

## Tests

```sh
ctest --test-dir build -C Release
```

Tests live under `tests/`, enabled by `BROTENSOR_TESTS=ON` (default ON when standalone). Most are CPU↔GPU parity tests; they skip cleanly when the GPU backend isn't available.

## Conventions

- **One unified `Tensor`, runtime device tag.** A single `brotensor::Tensor` holds storage on any backend; the `Device device` field (`enum class Device { CPU, CUDA, Metal }`) says where. There is no separate `GpuTensor` — that older two-type design has been replaced. Storage is a single opaque `void* data` allocated through the backend's `AllocVTable`.
- **Dispatch is runtime, per-operand.** Each public op in `ops.h` is a thin wrapper in `src/ops.cpp`. The wrapper calls `detail::dispatch(...)`, which resolves the op's device from the first *committed* operand (`data != nullptr`), verifies every other committed operand agrees (throws on mismatch), and returns that backend's `OpsVTable`. An *uncommitted* output (`data == nullptr`) is a wildcard — skipped by the check, then pinned to the resolved device via `adopt_output` before the backend impl allocates it. A null vtable slot means the backend doesn't implement that op; the wrapper throws "not implemented on <device>".
- **The op list is one X-macro.** `detail/op_table.h`'s `BROTENSOR_FOR_EACH_OP` is the single source of truth. It expands into the `OpsVTable` struct, the `src/ops.cpp` wrappers, and each backend's registration table — so the public surface and every backend stay in sync by construction.
- **Op signatures mirror across CPU and GPU.** The vtable slot signature *is* the public signature. Same argument order, same shape contracts, same accumulation semantics for backward (caller zeros dW/dB; op accumulates). When adding a CPU op that already has a GPU counterpart, port the contract verbatim and document any FP32-only restriction.
- **CPU is FP32-only.** Don't add FP16 / INT8 paths on the CPU side — those exist on the GPU because they pay for themselves there. CPU's job is to be the simple, correct fallback. The CPU backend leaves unimplemented vtable slots null.
- **GPU dtype dispatch is on `Tensor::dtype`.** Ops select FP32 vs FP16 (vs INT8 for W8A16) internally; the public surface takes a single `Tensor&` per arg. `Dtype` is `FP32 / FP16 / INT8 / INT32` (INT8/INT32 are storage carriers for quantised weights and index buffers — no arithmetic op dispatches on them).
- **Backend-resident storage stays opaque.** GPU `.cu` / `.mm` files include `<brotensor/tensor.h>` and treat `Tensor::data` as a raw device pointer (CUDA) or resolve it to its `MTLBuffer` via `metal_interop.h` (Metal). Use `from_host` / `to` / `copy_to_host` for host transfers.
- **Backend registration.** CPU self-registers from a static-init object (`src/cpu/register.cpp`), so CPU tensors work without a prior `init()` call. CUDA / Metal are probed and registered by `brotensor::init()`.
- **Default device.** `default_device()` picks the best available (CUDA > Metal > CPU). Override globally with `set_default_device()`, per-scope with `DeviceScope`, or via the `BROTENSOR_DEFAULT_DEVICE` env var (`cpu` / `cuda` / `metal`). `zeros` / `empty` / `from_host` land on the default; `*_on` variants pin to an explicit device.
- **Streams / async.** CUDA hot ops launch on the current stream. Metal batches command buffers and submits asynchronously (`submit` / `flush` — see `metal_interop.h`). `sync(Device)` / `sync_all()` drain pending work; call before reading GPU results back to host.
- **Error checks.** Backend impls throw `std::runtime_error` with a `"brotensor: <op>: <reason>"` message. Wrap CUDA calls with `BROTENSOR_CUDA_CHECK(expr)`. Negative dimensions are rejected at allocation; `resize()` on a non-owning `view()` throws rather than silently severing the view.

## When extending

- **Adding a new op:** add one row to `BROTENSOR_FOR_EACH_OP` in `detail/op_table.h`; implement it in `src/cpu/`, `src/cuda/`, `src/metal/`; register the slot in each backend's registration file (`src/cpu/register.cpp`, `src/cuda/register*.cu`, `src/metal/register*.mm`); list any new source file under that backend's target in `CMakeLists.txt`. Match the shape contract across all three (a backend may register a null slot if it genuinely can't support the op — the dispatcher throws on null lookups).
- **Adding a new dtype path:** extend `Dtype`, update `dtype_size_bytes`, and add the path inside the relevant GPU op kernel — don't add per-dtype public entry points.
- **ABI:** brodiffusion / brogameagent both vendor brotensor via `add_subdirectory`; assume your changes will be consumed without a release process. Don't break the public ABI casually.
