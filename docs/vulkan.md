# Vulkan backend

`BROTENSOR_WITH_VULKAN=ON` builds a compute backend made of hand-written GLSL
kernels and a small Vulkan host runtime. It runs on any GPU with a stock Vulkan
1.2+ driver and is meant to replace the HIP backend on AMD. The measurements
behind its design are in `../vk-spike/RESULTS.md` (Radeon 8060S, Mesa RADV).

**Status:** the runtime is complete. The ops registered so far are the elementwise
family, the activation backwards, the bias adds, cast, copies, row concat and
split, the NCHW / sequence transposes, and the row and column reductions. The
backend is never the default device. To select it, use
`set_default_device(Device::vulkan(i))`, a `DeviceScope`, or
`BROTENSOR_DEFAULT_DEVICE=vulkan` (also `vk`, `vulkan:1`). Any other op throws
"not implemented on vulkan".

## Build

```sh
cmake -B build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DBROTENSOR_WITH_HIP=ON -DBROTENSOR_WITH_VULKAN=ON
cmake --build build_vk && ctest --test-dir build_vk -R vulkan
```

The build needs the Vulkan headers and `glslc` (shaderc). If `glslc` is not
found, pass `-DBROTENSOR_GLSLC=`. Nothing links `libvulkan`: the loader is
`dlopen`ed at run time, so a binary built with the backend still starts (and
falls back to the other backends) on a machine without Vulkan. The backend
coexists with HIP (and with CUDA or Metal) in one build. `cmake/BrotensorVulkan.cmake`
holds the whole build: target, shader compilation and embedding.

## File layout

```
cmake/BrotensorVulkan.cmake     target + glslc step + generated shader table
include/brotensor/vulkan.h      public extras: flush, Event, VulkanGraph(Capture), stats, device_info
src/vulkan/
  detail/vk_fns.h, loader.cpp   dlopen'd loader, X-macro function tables (add calls here)
  detail/device.h, instance.cpp Instance (device selection), DeviceCtx (VkDevice, queue, owners)
  detail/allocator.h, .cpp      memory blocks, sub-allocation, address registry
  detail/stream.h, stream.cpp   command recording, batching, timeline serials, capture
  detail/pipelines.h, .cpp      pipeline cache, shared push-constant layout, limit guard
  detail/spirv_reflect.h, .cpp  shared-memory / workgroup-size reflection for the guard
  detail/kernels.h              what an op file uses: device_of, addr, dt_variant, launch
  tensor.cpp                    AllocVTable: alloc/free/transfers/memset/sync/mem queries
  graph.cpp                     VulkanGraph / VulkanGraphCapture, Event
  register.cpp                  probe + vtable fill + public stats
  ops_*.cpp                     ops, one file per family, each with a fill_vulkan_vtable_<family>
  shaders/                      *.comp kernels, common.glsl, op_codes.h, shaders.cmake (the list)
tests/test_vulkan.cpp, test_vulkan_ops.cpp, test_vulkan_common.h
```

Keep every file under 1000 lines: a new op family gets its own `ops_<family>.cpp`.

## Memory model

- **Pointers are device addresses.** A Vulkan tensor's `Tensor::data` holds the
  `VkDeviceAddress` of its first byte (buffer device address), cast to `void*`.
  As a result, views at an offset, `copy_d2d` offsets, and the raw
  `const float* d_mask` / `const int32_t*` operands in the op table all work
  exactly as on CUDA and HIP. The host never dereferences these values. When a
  transfer or `vkCmdCopyBuffer` needs a `(VkBuffer, offset)`, the allocator maps
  the address back with `Allocator::resolve`.
- **Blocks.** Each block is one `VkDeviceMemory` with one `VkBuffer` bound over
  all of it. Requests of 32 MiB or less are sub-allocated from 256 MiB blocks
  (best fit, 256-byte aligned, coalescing). Larger requests get a dedicated
  block. Blocks are mapped persistently when device-local memory is
  host-visible across the whole heap, which covers unified memory and
  resizable BAR. `memory_stats()` reports all of this.
- **4 GiB per tensor.** The allocator enforces min(`maxMemoryAllocationSize`,
  `maxBufferSize`, 4 GiB) per buffer, which RADV reports as 4 GiB - 4. A larger
  tensor is refused with an error asking the caller to split it, for example a
  large embedding table into row ranges. RADV actually accepts larger buffers,
  out of spec, and `BROTENSOR_VK_ALLOW_OVERSIZE=1` lifts the check for that
  case. Total memory is not limited: a device can hold any number of blocks of
  up to 4 GiB each. Shaders index with 32-bit element counts, which is safe
  because no buffer is larger than 4 GiB.
- **Reuse is stream ordered.** All work on a device is recorded in order into
  one stream, with a full barrier between commands. Every host access first
  drains that stream. So a freed sub-allocation can be handed out immediately,
  with the same semantics as `hipFreeAsync` on one stream. A dedicated block is
  returned to the driver only once the GPU has passed the work recorded before
  its free (a timeline serial).
- **Transfers.** Uploads (h2d) take one of four paths:
  - memcpy directly into mapped memory when the stream is idle;
  - `vkCmdUpdateBuffer` in-stream for 64 KiB or less (no wait);
  - drain the stream, then memcpy;
  - a staging buffer when memory is not mapped.

  Downloads (d2h) drain the stream. They read in place for 64 KiB or less, and
  otherwise go through a host-cached staging buffer, 64 MiB at a time, because
  mapped device-local memory is write-combined (slow for the CPU to read). Both
  directions return synchronously, as on HIP. Vulkan↔Vulkan copies across two
  devices go through the host. Vulkan↔HIP/CPU copies use `Tensor::to`, which
  also bounces through the host.

## Dispatch model

- **Descriptor-free.** Every pipeline shares one `VkPipelineLayout` that has a
  push-constant range and no descriptor sets. A kernel receives its tensors as
  `uint64_t` device addresses, plus its scalars, in its push block, and reads
  them through `buffer_reference` types (`common.glsl`). Keep a push block at
  128 B or less (the portable minimum; RADV allows 256). The C++ mirror struct
  puts `uint64_t` fields first, and `launch()` static-asserts the size.
- **Pipelines** are created on first use and cached per device. The cache key
  is (ShaderId, specialisation constants, required subgroup size).
  Specialisation constant 0 usually selects the operation, so each
  (kernel, op) pair is its own pipeline with the switch folded away. Before
  `vkCreateComputePipelines` sees a shader, its SPIR-V is reflected with the
  specialisation applied, and the shared memory and workgroup size are
  checked against the device limits. A shader over the limit throws. Without
  this guard, RADV raises SIGFPE inside the driver for more than 64 KiB of
  shared memory. Shared-array lengths must be constants, spec constants, or
  simple integer arithmetic on them; anything else is reported as unverifiable.
- **Streams.** Each device has one in-order stream. Commands go into an open
  command buffer, which is submitted to the **graphics** queue when it reaches
  `BROTENSOR_VK_BATCH` commands (default 128), on `sync`, before a host access,
  or on `vulkan::flush(d)`. Up to four batches are in flight. Each submission
  signals a timeline semaphore with a new serial, and an `Event` is just such a
  serial. The graphics queue is used, not the compute-only queue, because on
  RADV a pipeline-changing chain costs 0.63 µs per kernel on the graphics queue
  and 1.8 µs on the compute queue. Ordering comes from a global compute and
  transfer barrier between commands and at the start of each command buffer. A
  later chunk may replace that with hazard tracking, but nothing else assumes
  more than in-order execution.
- **Graphs.** `VulkanGraphCapture` / `VulkanGraph` (`vulkan.h`) follow the
  `CudaGraphCapture` / `CudaGraph` contract and are built on what Vulkan does
  cheapest: the captured ops are recorded once into a command buffer with
  `SIMULTANEOUS_USE`, and `launch()` resubmits it unchanged. While a capture is
  recording, `sync`, downloads and uploads on that device throw, and frees are
  held by the graph until it is destroyed. The spike measured 0.74 µs per
  dependent kernel for replay, against 1.9 µs for HIP and its graphs. The
  chunk-1 test reports wall time including the sync. For a 301-command chain
  of 4096-element elementwise ops and copies, that was about 1.1 µs per command
  replayed and 1.4 µs per command eager. Siblings can treat the two graph
  classes as interchangeable shapes.

## Dtype policy

- **FP32**: native.
- **FP16**: stored as `float16_t` with 16-bit storage. Arithmetic is FP32, and
  the store uses `float16_t(x)`, which is round-to-nearest-even. Never use
  `packHalf2x16`, which rounds toward zero on RADV. Casts are bit-identical to
  `brotensor::fp32_to_fp16_bits`, and the tests check this, including 65520 →
  inf.
- **BF16**: RADV has no BF16 shader type (no `VK_KHR_shader_bfloat16`, no BF16
  cooperative matrix). BF16 is therefore carried as `uint16` bits and
  converted in registers (`bf16_to_f32` / `f32_to_bf16` in `common.glsl`,
  RNE, NaN kept quiet, bit-identical to the host helpers). This runs at full
  bandwidth for memory-bound ops, so the elementwise, reduction and norm
  families take BF16 directly. **The policy for matrix-core ops (GEMM,
  attention) is decided here:** `compute_dtype(Device::vulkan(i))` is FP16, so
  a loader holding BF16 weights converts them **once at upload** with
  `cast(…, Dtype::FP16)`, and the coopmat kernels then see only FP16. A BF16
  activation that reaches a GEMM goes through the same `cast` first. The GEMM
  chunk may add an FP32 SIMT path for BF16 values outside FP16's range (±65504).
  It must not grow a native BF16 coopmat path, because the hardware has none.
- **INT8 / INT32**: storage carriers only (8-bit storage is enabled). There is
  no FP8.

GLSL built-ins are not IEEE-exact: `exp` is about 3 ulp plus |x|·ulp, and
`sin`/`cos` have an absolute error bound. `tanh` and `erf` are written out in
`common.glsl` (finite for large |x|, accurate near 0). Tanh-GELU and its
gradient use the identity 0.5(1 + tanh u) = sigmoid(2u), which avoids the
cancellation the CPU reference suffers for x < -3. `axpby` is `precise`
(unfused), like CUDA and HIP. The tests' tolerances record what each op
achieves.

## Adding an op

1. **Kernel.** Write `src/vulkan/shaders/<kernel>.comp`. Start with
   `#include "common.glsl"`, read tensors through `load_elem` / `store_elem`
   (FP32 / FP16 / BF16 by the `DT` define) or the raw `*Buf` types, and use a
   grid-stride loop (`GRID_STRIDE_X`) so any size fits within 65535
   workgroups. Put new op codes in `op_codes.h`, which both GLSL and C++
   include.
2. **List it** in `shaders/shaders.cmake` as `"name|file.comp|-Ddefines"`.
   Dtype families are three consecutive entries `<base>_f32`, `<base>_f16`,
   `<base>_bf16`, and the `_f32` id goes into `kDtypeFamilies`
   (`detail/kernels.h`). Startup checks the order.
3. **Host side.** Write the op in `src/vulkan/ops_<family>.cpp` with the exact
   op-table signature and the CUDA backend's contract (shapes, output resize,
   dtype rules, accumulate versus overwrite). Use `device_of(t)` for the
   device, then `pipelines().get(id, {spec…})`, then `launch(d, k, push, groups…)`.
   Register it in that file's `fill_vulkan_vtable_<family>`. A new file goes
   into the source list in `cmake/BrotensorVulkan.cmake`, and its fill
   function is called from `register.cpp`.
4. **Test.** Add the op to `tests/test_vulkan_ops.cpp` (or a new
   `test_vulkan_<family>.cpp` in the same executable). Compare it against the
   CPU op on several shapes, on FP32, FP16 and BF16, with a stated tolerance.
   Data movement must be bit-exact.

Kernels that need a fixed subgroup size (cooperative matrix) pass
`subgroup = 32` to `Pipelines::get`. The device enables
`VK_KHR_cooperative_matrix` and subgroup size control when they are present.
RADV's default compute subgroup size is 64, and the spike's coopmat kernels
pinned it to 32. The chunk-1 kernels are subgroup-size agnostic.

## Environment

| Variable | Effect |
|---|---|
| `BROTENSOR_DEFAULT_DEVICE=vulkan[:i]` / `vk[:i]` | make Vulkan the default device |
| `BROTENSOR_DISABLE_VULKAN=1` | do not probe Vulkan at all |
| `BROTENSOR_VK_VERBOSE=1` | print why the backend did or did not register |
| `BROTENSOR_VK_ALLOW_CPU=1` | accept CPU implementations (lavapipe) |
| `BROTENSOR_VK_BATCH=n` | commands per submitted batch (default 128) |
| `BROTENSOR_VK_BLOCK_MB=n` | sub-allocation block size (default 256) |
| `BROTENSOR_VK_MAPPED=0` | do not map device memory even when possible |
| `BROTENSOR_VK_ALLOW_OVERSIZE=1` | allow tensors over the per-buffer limit (out of spec) |

Vulkan validation layers were not installed on the development machine, so
the backend has not been run under them.
