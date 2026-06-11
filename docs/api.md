# API reference

The public surface lives in five headers:

| Header | Contents |
|---|---|
| `<brotensor/tensor.h>` | `Tensor`, `Dtype`, `Device`, factories, migration, host accessors, bit-conversion helpers |
| `<brotensor/runtime.h>` | `init()`, device policy, `compute_dtype()`, `sync` |
| `<brotensor/ops.h>` | The device-neutral op surface — an umbrella over the per-category headers in `<brotensor/ops/>` (see [op-coverage.md](op-coverage.md)) |
| `<brotensor/safetensors.h>` | safetensors reader + writer |
| `<brotensor/gguf.h>` | GGUF reader |

`<brotensor/metal_interop.h>` additionally exposes the Metal custom-kernel surface (Obj-C++ / `.mm` consumers only).

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
| `Tensor::mat(r, c)` / `Tensor::vec(n)` | Zero-filled FP32 **host (CPU)** tensors — build parameters on the host, then migrate with `to()`. |
| `Tensor::view(dev, ptr, r, c, dt)` | Non-owning view over an existing backend-resident pointer. `resize()` on a view throws. |

### Migration and readback

| Member | Meaning |
|---|---|
| `t.to(device)` | Returns a copy migrated to another backend; the source is unchanged. |
| `t.clone()` | Device-preserving deep copy. |
| `t.to_host_vector()` (+ `_fp16` / `_bf16`) | Read back to a `std::vector` (`float` / `uint16_t` bits). |
| `t.copy_to_host(dst)` (+ `_fp16` / `_bf16`) | Read back into a caller-owned buffer. |
| `t.resize(r, c, dt)` | Reallocate in place; contents **undefined** afterwards. Throws on a non-owning view. |

Call `sync(device)` / `sync_all()` before reading GPU results back to the host — GPU ops are asynchronous.

CPU-resident tensors additionally expose direct host accessors (`host_f32_mut()`, `at()`, `operator[]`, …) — see `tensor.h`.

### Bit-conversion helpers

`fp32_to_fp16_bits` / `fp16_bits_to_fp32` / `fp32_to_bf16_bits` / `bf16_bits_to_fp32` — pure-CPU scalar conversions between FP32 and half/bfloat bit patterns, for tests and small host-side preprocessing.

## Runtime

| Function | Meaning |
|---|---|
| `init()` | Idempotent. Probes and registers the CUDA / Metal backends. CPU is always registered (static-init), so CPU-only code works without calling it. |
| `default_device()` | Where no-suffix factories allocate. Best available: CUDA > Metal > CPU. |
| `set_default_device(dev)` | Global override. Also overridable per-process via the `BROTENSOR_DEFAULT_DEVICE` env var (`cpu` / `cuda` / `metal`). |
| `DeviceScope scope(dev)` | RAII per-scope default-device override. |
| `compute_dtype()` | The dtype a model loader should upload weights at for the current default device: FP32 on CPU, FP16 on a GPU. |
| `available_devices()` / `is_available(dev)` | Backends registered in this binary at runtime. |
| `sync(dev)` / `sync_all()` | Drain pending backend work (no-op on CPU). |

## safetensors (`<brotensor/safetensors.h>`)

mmap'd zero-copy reader plus a writer. Namespace `brotensor::safetensors`.

- `File` — opens and mmaps a `.safetensors` file, parses the JSON header, exposes tensors by name as `TensorView`s (name, dtype, shape, raw byte span).
- Upload helpers (view → device `Tensor`):
  - `upload(view, rows, cols, dst)` — as FP32;
  - `upload_fp16(view, rows, cols, dst)` — as FP16;
  - `upload_compute(view, rows, cols, dst)` — at `compute_dtype()` for the current default device;
  - `upload_compute_checked(...)` — same, with shape validation.
- `write_file(path, entries)` — write a `.safetensors` file from host data.
- Supported on-disk dtypes: F32, F16, BF16, I32, I64, U8, BOOL.

## GGUF (`<brotensor/gguf.h>`)

mmap'd reader for GGUF model files. Namespace `brotensor::gguf`.

- `File` — opens and mmaps a `.gguf` file, parses header + metadata, exposes tensors as `TensorInfo` (name, GGUF type, shape, raw data span).
- Metadata: `find_meta(key)` / `get_meta(key)` / `metadata()`.
- `shape_to_2d(shape)` — collapse a GGUF n-d shape to brotensor's `(rows, cols)`.
- `upload_raw(info, rows, cols, dst)` — upload a tensor's raw bytes to the device at its carrier dtype.
- Supported carriers: F32, F16, and the block-quant types Q4_K / Q6_K / Q8_0 — consumed directly by the fused dequant/matmul ops (see [op-coverage.md](op-coverage.md)) without dequantizing on the host.
