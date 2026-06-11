# Architecture

How brotensor is put together: one tensor type, a runtime dispatch layer, and three backends behind a single vtable.

## One tensor, runtime tags

A `brotensor::Tensor` is a row-major `(rows, cols)` buffer carrying two runtime tags:

- `Device device` — `CPU`, `CUDA`, or `Metal`: where the storage lives.
- `Dtype dtype` — what the elements are.

There is **no separate host/device tensor type**. Storage is a single opaque `void* data` allocated through the owning backend's allocator vtable. Rank-1 tensors are `(N, 1)`; higher-rank layouts (NCHW images, `(T*B, H)` sequences, packed heads) are flattened into the two dims by each op's documented shape contract.

Backend code treats `data` as whatever its runtime needs: the CUDA backend reads it as a raw device pointer; the Metal backend resolves it to its `MTLBuffer` via `metal_interop.h`. Host code never touches GPU storage directly — it goes through `from_host` / `to` / `copy_to_host` / `to_host_vector`.

## Dtypes

`Dtype` is `FP32 / FP16 / BF16 / INT8 / INT32 / F64` plus the opaque GGUF block-quant carriers (`Q4_0 … Q8_K`).

- **FP32 / FP16 / BF16** are the arithmetic dtypes ops dispatch on. BF16 is GPU-only; FP16/BF16 are stored as `uint16_t` bit patterns on the host.
- **INT8 / INT32 / F64** are pure storage carriers: INT8 backs weight-only quantized matmul/conv (W8A16) and byte masks, INT32 carries device-resident index/offset buffers (e.g. `cu_seqlens`, pool indices), F64 is a reserved 8-byte carrier. No general arithmetic op dispatches on them.
- **GGUF quant dtypes** (`Q4_0…Q8_K`) are non-element-addressable block carriers — 32-element legacy blocks or 256-element K-quant superblocks — consumed only by the GGUF dequant / fused-matmul ops.

Element/block sizing goes through `dtype_size_bytes` / `dtype_block_size` / `dtype_block_bytes` / `dtype_storage_bytes` / `dtype_is_quant`. Quant dtypes return 0 from `dtype_size_bytes` — use `dtype_storage_bytes` for buffer sizes.

GPU ops select their FP32 / FP16 / BF16 (/ INT8) path internally from `Tensor::dtype` — there are no per-dtype public entry points (the few `*_int8w_fp16` / `*_q4k_fp16` names are distinct ops with different operand contracts, not dtype overloads).

## Dispatch

Every public op in `<brotensor/ops.h>` is a thin wrapper that:

1. Resolves the op's device from the first **committed** operand (`data != nullptr`).
2. Verifies every other committed operand agrees — a device mismatch throws.
3. Treats an **uncommitted** output (`data == nullptr`, e.g. a default-constructed `Tensor`) as a wildcard: it's skipped by the check, then pinned to the resolved device before the backend impl allocates it.
4. Calls that backend's function pointer from its `OpsVTable`. A null slot means the backend doesn't implement the op — the wrapper throws `"brotensor: <op>: not implemented on <device>"`.

A backend is just an `OpsVTable` (op function pointers) plus an `AllocVTable` (alloc/free/copy), registered at runtime:

- **CPU** self-registers from a static-init object (`src/cpu/register.cpp`) — CPU tensors work without any `init()` call.
- **CUDA / Metal** are probed and registered inside `brotensor::init()`, if compiled in and a device is actually present.

### The op table is one X-macro

`include/brotensor/detail/op_table.h` defines `BROTENSOR_FOR_EACH_OP`, the single canonical op list. It expands into the `OpsVTable` struct, the public wrappers in `src/ops.cpp`, and each backend's registration table — so the public surface and every backend stay in sync by construction. Adding an op means adding one row there (see [Extending](#extending)).

### Op signatures mirror across backends

The vtable slot signature *is* the public signature: same argument order, same shape contracts, same accumulation semantics on every backend. The convention for backward ops: the caller zeros parameter gradients (`dW`, `dB`); the op **accumulates** into them. Activation/input gradients (`dX`) are overwritten. Per-op headers document any deviation.

## Backends

- **CPU** — scalar FP32, always compiled. It implements essentially the entire FP32 surface, forward *and* backward — including the diffusion samplers, flash attention, the audio family, and the vision primitives. It is the simple, correct, autovectorize-friendly reference that the parity tests measure the GPU backends against. By design it leaves the FP16 / BF16 / INT8-W8A16 / GGUF-quant slots null.
- **CUDA** (`BROTENSOR_WITH_CUDA=ON`) — mirrors the FP32 surface and adds the FP16/BF16 precision paths, batched-inference variants, W8A16 WMMA kernels, GGUF block-quant kernels, and fused inference kernels.
- **Metal** (`BROTENSOR_WITH_METAL=ON`) — same role as CUDA on Apple GPUs. A few inference-only ops are CPU+CUDA with the Metal slot left null (noted in the [coverage tables](op-coverage.md)).

A handful of "ops" are not vtable entries at all but device-agnostic compositions of public ops — LoRA (`ops/lora.h`, header-only) and the `filtered_lrelu` composite fallback — so they run on any backend automatically.

## Default device and scopes

`default_device()` picks the best available backend (CUDA > Metal > CPU). The no-suffix factories (`Tensor::zeros`, `empty`, `from_host`) allocate there; `*_on` variants pin to an explicit device. Override the default:

- globally with `set_default_device(Device)`,
- per-scope with `DeviceScope` (RAII),
- per-process with the `BROTENSOR_DEFAULT_DEVICE` env var (`cpu` / `cuda` / `metal`).

`compute_dtype()` is the dtype a model loader should upload weights at for the current default device: FP32 on CPU, FP16 on a GPU.

## Streams and synchronization

CUDA hot ops launch asynchronously on the current stream. Metal batches command buffers and submits asynchronously (see `metal_interop.h` for `submit` / `flush` and the custom-kernel interop surface). `sync(Device)` / `sync_all()` drain pending work — call one before reading GPU results back to the host. They are no-ops on CPU.

## Error handling

Backend impls throw plain `std::runtime_error` with a `"brotensor: <op>: <reason>"` message for precondition and dispatch failures. CUDA calls are wrapped with `BROTENSOR_CUDA_CHECK(expr)`. Negative dimensions are rejected at allocation; `resize()` on a non-owning `view()` throws rather than silently severing the view.

## Build internals

Four static libraries plus one interface target:

| Target | Role |
|---|---|
| `brotensor_core` | Tensor, dispatcher, runtime, safetensors/GGUF loaders |
| `brotensor_cpu` | CPU backend (always built) |
| `brotensor_cuda` | CUDA backend (`BROTENSOR_WITH_CUDA=ON`) |
| `brotensor_metal` | Metal backend (`BROTENSOR_WITH_METAL=ON`) |
| `brotensor::brotensor` | The INTERFACE target consumers link |

The interface target **whole-archives** the backend libraries so their self-registration translation units survive the link — don't link the backend libs directly.

`BROTENSOR_WITH_CUDA` and `BROTENSOR_WITH_METAL` are independent options (both default OFF); they stay exclusive in practice because the nvcc and Apple toolchains don't coexist on one host. Per-backend preprocessor defines: `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL`, with `BROTENSOR_HAS_GPU` as the umbrella. Most code never needs them — the unified `Tensor` and op surface compile identically regardless of backend; reach for the defines only to gate a path that genuinely needs a GPU present, and prefer `BROTENSOR_HAS_GPU` unless you need backend identity.

`BROTENSOR_TESTS` (default ON when built standalone) enables the test suite.

## Extending

**Adding a new op:**

1. Declare it in the matching `include/brotensor/ops/<category>.h` (the umbrella `ops.h` re-includes every category header — don't add declarations to `ops.h` directly).
2. Add one row to `BROTENSOR_FOR_EACH_OP` in `include/brotensor/detail/op_table.h`.
3. Implement it in `src/cpu/`, `src/cuda/`, `src/metal/` — matching the shape contract exactly across backends. A backend that genuinely can't support the op registers a null slot; the dispatcher throws on lookup.
4. Register the slot in each backend's registration file (`src/cpu/register.cpp`, `src/cuda/register.cu` via a `fill_cuda_vtable_*` function, `src/metal/register.mm`).
5. List any new source file under that backend's target in `CMakeLists.txt`, and add a test (parity test if the op exists on CPU and GPU).

FP32 ops should land on CPU too — it's the parity reference. FP16/BF16/INT8/GGUF-quant variants are GPU-only by design.

**Adding a new dtype path:** extend `Dtype`, update `dtype_size_bytes` (and `dtype_block_size` / `dtype_block_bytes` for a block-quant carrier), and add the path inside the relevant GPU op kernel. Don't add per-dtype public entry points.

**ABI:** downstream siblings vendor brotensor via `add_subdirectory` and consume changes without a release process — don't break the public ABI casually.
