# API reference

The portable public surface lives in five headers:

| Header | Contents |
|---|---|
| `<brotensor/tensor.h>` | `Tensor`, `Dtype`, `Device`, factories, migration, host accessors, dtype + bit-conversion helpers |
| `<brotensor/runtime.h>` | `init()` / `shutdown()`, device policy, `compute_dtype()`, device memory, `sync` |
| `<brotensor/ops.h>` | The device-neutral op surface — an umbrella over the per-category headers in `<brotensor/ops/>` (see [op-coverage.md](op-coverage.md)) |
| `<brotensor/safetensors.h>` | safetensors reader + writer |
| `<brotensor/gguf.h>` | GGUF reader |

Two further headers are backend-specific and compile only in a build that enables that backend:

| Header | Contents |
|---|---|
| `<brotensor/cuda_graph.h>` | CUDA graph capture / replay (`CudaGraph`, `CudaGraphCapture`) — CUDA-only; gate on `BROTENSOR_HAS_CUDA` |
| `<brotensor/metal_interop.h>` | Metal custom-kernel surface — Obj-C++ / `.mm` consumers only |

All preconditions and dispatch failures throw `std::runtime_error` with a `"brotensor: <op>: <reason>"` message.

## Tensor

A row-major `(rows, cols)` buffer with runtime `Dtype` + `Device` tags. Copyable (device-aware deep copy) and movable. Rank-1 data is `(N, 1)`; higher-rank layouts are flattened per each op's shape contract.

```cpp
enum class Device { CPU, CUDA, Metal };
enum class Dtype  { FP32, FP16, INT8, INT32, BF16, F64,
                    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,        // GGUF legacy blocks
                    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K };       // GGUF K-quant superblocks
```

### Factories

| Factory | Meaning |
|---|---|
| `Tensor::zeros(r, c, dt = FP32)` / `Tensor::empty(r, c, dt = FP32)` | Allocate on the **default device**. `zeros` zero-fills; `empty` leaves contents **undefined**. |
| `Tensor::zeros_on(dev, r, c, dt)` / `Tensor::empty_on(dev, r, c, dt)` | Same, pinned to an explicit device. |
| `Tensor::from_host(ptr, r, c)` (+ `_fp16` / `_bf16` / `_int8` variants) | Copy a host buffer to a new tensor on the default device. FP16/BF16 take `uint16_t` bit patterns, INT8 takes `int8_t`. |
| `Tensor::from_host_on(dev, ptr, r, c)` (+ `_fp16_on` / `_bf16_on` / `_int8_on`) | Same, pinned to an explicit device. |
| `Tensor::from_raw_bytes_on(dev, src, r, c, dt, nbytes)` | Dtype-agnostic byte-level bootstrap. Copies raw bytes rather than interpreting elements, so unlike `from_host*` it works for **any** dtype including the opaque GGUF block-quant carriers. `nbytes` must equal `dtype_storage_bytes(dt, r*c)`. |
| `Tensor::mat(r, c)` / `Tensor::vec(n)` | Zero-filled FP32 **host (CPU)** tensors — build parameters on the host, then migrate with `to()`. |
| `Tensor::view(dev, ptr, r, c, dt)` | Non-owning view over an existing backend-resident pointer. `resize()` on a view throws. |

### Migration and readback

| Member | Meaning |
|---|---|
| `t.to(device)` | Returns a copy migrated to another backend; the source is unchanged. |
| `t.clone()` | Device-preserving deep copy. |
| `t.to_host_vector()` (+ `_fp16` / `_bf16`) | Read back to a `std::vector` (`float` / `uint16_t` bits). |
| `t.copy_to_host(dst)` (+ `_fp16` / `_bf16`) | Read back into a caller-owned buffer. |
| `t.zero()` | memset the buffer to zero over `bytes()`. |
| `t.resize(r, c, dt)` | Reshape in place; contents **undefined** afterwards (call `zero()` if needed), device preserved. Throws on a negative dimension or a non-owning view. |

`resize()` reuses storage whenever the requested shape fits the existing allocation — capacity is the high-water mark of the tensor's past sizes — and reallocates only when growing past it. A no-op when the shape and dtype already match. So a scratch buffer cycling through shapes stabilises at its largest size instead of reallocating every call, **and its device pointer stays stable** — which is what makes a tensor reusable across a CUDA-graph-captured op sequence.

Call `sync(device)` / `sync_all()` before reading GPU results back to the host — GPU ops are asynchronous.

CPU-resident tensors additionally expose direct host accessors (`host_f32_mut()`, `at()`, `operator[]`, …) — see `tensor.h`.

### Dtype helpers

Free functions for sizing a buffer without special-casing the quant carriers:

| Helper | Meaning |
|---|---|
| `dtype_size_bytes(dt)` | Bytes per element. **Returns 0 for the block-quant dtypes** — they aren't element-addressable. |
| `dtype_block_size(dt)` | Elements per block (32 for the legacy quants, 256 for the K-quants, 1 otherwise). |
| `dtype_block_bytes(dt)` | Encoded bytes per block. |
| `dtype_storage_bytes(dt, n)` | Bytes needed for `n` elements. **Use this for buffer sizes** — it's correct for quant and non-quant dtypes alike. |
| `dtype_is_quant(dt)` | Whether `dt` is a GGUF block-quant carrier. |
| `device_name(dev)` | The backend kind as a string (`"cpu"` / `"cuda"` / `"metal"`). |

### Bit-conversion helpers

`fp32_to_fp16_bits` / `fp16_bits_to_fp32` / `fp32_to_bf16_bits` / `bf16_bits_to_fp32` — pure-CPU scalar conversions between FP32 and half/bfloat bit patterns, for tests and small host-side preprocessing.

## Runtime

| Function | Meaning |
|---|---|
| `init()` | Idempotent. Probes and registers the CUDA / Metal backends. CPU is always registered (static-init), so CPU-only code works without calling it. |
| `shutdown()` | Joins the CPU backend's worker threads. Idempotent, and safe even if `init()` was never called. See the note below — **call it before returning from `main()`**. |
| `default_device()` | Where no-suffix factories allocate. Best available: CUDA > Metal > CPU. |
| `set_default_device(dev)` | Global override. Also overridable per-process via the `BROTENSOR_DEFAULT_DEVICE` env var (`cpu` / `cuda` / `metal`). |
| `DeviceScope scope(dev)` | RAII per-scope default-device override. |
| `compute_dtype()` | The dtype a model loader should upload weights at for the current default device: FP32 on CPU, FP16 on a GPU. |
| `available_devices()` / `is_available(dev)` | Backends registered in this binary at runtime. |
| `sync(dev)` / `sync_all()` | Drain pending backend work (no-op on CPU). |
| `device_mem_info(dev, free, total)` | Device-wide free/total bytes. Returns `false` (outputs untouched) when the backend can't report; CPU always returns `false`. |
| `device_mem_trim(dev, keep_bytes = 0)` | Return the allocator's cached-but-unused memory to the driver, keeping at most `keep_bytes`. Syncs the device first so stream-ordered frees are reclaimable. |
| `device_product_name(dev)` | The card's human-readable name (e.g. `"NVIDIA GeForce RTX 4090"`) — distinct from `device_name()`, which is the backend kind. `""` if unavailable. |

**Shutdown.** The CPU backend's worker threads otherwise live until the thread pool's Meyers-singleton destructor runs during static destruction, by which point the OS has already suspended every other thread. A worker suspended mid-op while holding a global lock (e.g. the Debug CRT's iterator-checking mutex) can deadlock the main thread's own exit-time destructors on that same lock. `shutdown()` makes the teardown deterministic instead.

**Trimming.** `device_mem_trim` is worth calling between pipeline phases with very different scratch shapes: cached blocks count against device residency, and on Windows (WDDM) sustained near-full commit makes the OS demote large resident allocations to shared memory — silently turning weight reads into PCIe traffic.

## safetensors (`<brotensor/safetensors.h>`)

mmap'd zero-copy reader plus a writer. Namespace `brotensor::safetensors`.

- `File` — opens and mmaps a `.safetensors` file, parses the JSON header, exposes tensors by name as `TensorView`s (name, dtype, shape, raw byte span).
- Upload helpers (view → device `Tensor`). All require an F32 / F16 / BF16 source view; brotensor is 2D-only, so the caller flattens higher-rank weights to the `(rows, cols)` layout the consuming op expects:
  - `upload(view, rows, cols, dst)` — **dtype-preserving**: `dst` gets the brotensor dtype matching the view, zero conversion (a BF16 view yields a BF16 tensor, an F16 view an FP16 tensor);
  - `upload_fp16(view, rows, cols, dst)` — always FP16, converting host-side from F32 if needed;
  - `upload_as(view, rows, cols, want, dst)` — at an **explicit** arithmetic dtype, converting host-side. Lets one module pick a compute dtype different from the global one — e.g. Flux runs BF16 on a pipeline whose dtype is FP16, because its activations overflow FP16;
  - `upload_compute(view, rows, cols, dst)` — at `compute_dtype()` for the current default device, so one checkpoint serves either backend (BF16 widens to FP32 on CPU, narrows to FP16 on a GPU);
  - `upload_compute_checked(view, rows, cols, dst, name)` — same, but first validates the view's dtype and element count, throwing tagged with the caller-supplied `name` and the safetensors key.
- `write_file(path, entries)` — write a `.safetensors` file from host data. Each `WriteEntry` carries name / dtype / shape / host pointer / byte count; dtype defaults to `F16`.
- Supported on-disk dtypes: F32, F16, BF16, I32, I64, U8, BOOL.

## GGUF (`<brotensor/gguf.h>`)

mmap'd reader for GGUF model files. Namespace `brotensor::gguf`.

- `File` — opens and mmaps a `.gguf` file, parses header + metadata, exposes tensors as `TensorInfo` (name, GGUF type, mapped brotensor `dtype`, `dtype_supported`, shape, `numel`, raw data span). Also `find_tensor()` / `get_tensor()` / `tensors()`, `version()`, `alignment()`, `tensor_count()`.
- Metadata: `find_meta(key)` / `get_meta(key)` / `metadata()`, returning GGUF `Value` / `ValueType`.
- `shape_to_2d(shape)` — collapse a GGUF n-d shape to brotensor's `(rows, cols)`. GGUF shapes are innermost-first, so `cols = shape[0]` and `rows = product(shape[1..])`; a 1-D shape gives `(shape[0], 1)`. Throws on an empty shape.
- `upload_raw(info, rows, cols, dst)` — upload a tensor's raw bytes to the device at its carrier dtype, no host-side dequantization. Throws if `info.dtype_supported` is false; for a quant carrier, `cols` must be a multiple of `dtype_block_size(dtype)`.
- **Carriers the reader maps:** F32, F16, BF16, the legacy blocks Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1, and the K-quant superblocks Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K.

Note that carrier support is broader than *op* support: the reader will load any of the above, but only **Q4_K / Q6_K / Q8_0** are consumed by the fused dequant / matmul kernels (see [op-coverage.md](op-coverage.md)). Loading a Q5_K tensor succeeds; calling a matmul on it throws.
