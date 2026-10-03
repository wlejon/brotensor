# Vulkan backend

`BROTENSOR_WITH_VULKAN=ON` builds a compute backend made of hand-written GLSL
kernels and a small Vulkan host runtime. It runs on any GPU with a stock Vulkan
1.2+ driver and is meant to replace the HIP backend on AMD. The measurements
behind its design are in `../vk-spike/RESULTS.md` (Radeon 8060S, Mesa RADV).

**Status:** the runtime is complete. The ops registered so far are the elementwise
family, the activation backwards, the bias adds, cast, copies, row concat and
split, the NCHW / sequence transposes, the row and column reductions (chunk 1),
and the transformer core (chunk 2): `matmul`, `matmul_abt`, the linear family
(`linear_forward`, `_batched`, `_batched_fp16`, `_fp16_act`, `_ex` with all four
epilogues), the GEMM backwards (`matmul_backward`, `linear_backward`,
`linear_backward_batched`), the LayerNorm family (vector, batched inference with
and without beta, FP16, with caches, both backwards), RMSNorm, per-head L2 norm,
pixel norm, softmax (masked vector, rows, backward), every RoPE variant
(`rope_forward/backward`, `rope_apply`, `_backward`, `_perhead`,
`rope_qkv_packed_inplace`, `rope_apply_mrope`), SwiGLU / GeGLU forward and
backward, `modulate` and `broadcast_mul`. The backend is never the default device. To select it, use
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
  detail/gemm.h, gemm.cpp       the GEMM dispatcher every matrix op goes through
  ops_*.cpp                     ops, one file per family, each with a fill_vulkan_vtable_<family>
                                (elementwise, copy, reduce, linear, norm, rope, glu)
  shaders/                      *.comp kernels, common.glsl, gemm_common.glsl, op_codes.h,
                                shaders.cmake (the list)
tests/test_vulkan.cpp           runtime; main(), --only=ops|gemm|norm, --bench-gemm
tests/test_vulkan_ops.cpp       chunk-1 op parity
tests/test_vulkan_gemm.cpp      matmul / linear parity, every kernel path
tests/test_vulkan_norm.cpp      norms, softmax, RoPE, GLUs parity
tests/test_vulkan_bench.cpp     GEMM / GEMV throughput (not in ctest)
tests/test_vulkan_common.h
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

## Matrix multiply

Every matrix op is one or more calls to `detail::vulkan::gemm()` (`gemm.cpp`):
C[z](M, N) = op(A[z]) op(B[z]) with A stored (M, K) or transposed, B stored
(N, K) (the linear layer) or (K, N), batch strides taken literally (0
broadcasts), and a fused epilogue r = act(acc + bias) then store, accumulate,
GeGLU (interleaved pairs) or SwiGLU (stacked gate / up rows). It picks one of
three kernels:

| Kernel | When | What |
|---|---|---|
| `gemv.comp` | NT layout, M <= 8, one batch | 64 invocations per output column, 4 chunks of 16 bytes in flight each, all epilogues fused. 235 GB/s of FP16 weights at B = 1-2, 200-215 GB/s at B = 4-8 (256 GB/s peak) |
| `gemm_cm.comp` | FP16 / BF16 operands, `coopmat_f16` device | 16x16x16 FP16 fragments, FP32 accumulation, subgroup 32, double-buffered shared tiles |
| `gemm_simt.comp` | everything else | FP32 FMA, 128x128 or 64x64 tile, 8x8 or 4x4 per invocation |

**Cooperative-matrix kernel.** Tiles (BM x BN x BK / subgroup tile) are
256x128x32/64x64, 128x128x32/64x64, 128x64x32/64x32 and 64x64x32/32x32. The
choice (`pick_cm`) was measured on the spike's shapes: 256x128 when M >= 512
and there are >= 150 tiles (or K >= 8192), 128x128 down to 24 tiles, 64x64 for
M <= 64. `BROTENSOR_VK_GEMM_CFG=256,128,32,64,64` forces one. Any M, N, K works;
three things decide the speed, all learned the hard way on RADV / ACO:

* **Never consume a load where it is issued.** A bounds branch, a zeroing
  select or the BF16 -> FP16 conversion at the load site makes the compiler
  wait for that load there, not after the fragment math: the first version
  ran at half the spike's speed for this alone. With `VEC` (16-byte aligned
  bases, leading dimensions and batch strides, contiguous extents that are
  multiples of 8), chunks are loaded unconditionally at a clamped index with a
  validity bit, and the select and conversion happen when the chunk is
  stored to shared memory. Without `VEC` the kernel takes scalar, bounds-
  checked loads (correct for any view, much slower).
* **The epilogue form is a specialisation, not a branch.** Interior tiles with
  FP16 output store straight from the accumulator (`DIRECT = 1`): the bias is a
  fragment loaded with row stride 0, and activation, SwiGLU and accumulate are
  elementwise over fragments of one type, so the opaque element-to-lane layout
  never matters. Edge tiles, BF16 output and GeGLU go through a per-subgroup
  scratch in shared memory (aliasing the A tile) and a bounds-checked
  per-element pass. The host dispatches the full tiles with `DIRECT = 1` and
  the right / bottom edge strips (if any) separately; choosing the form at run
  time inside one kernel cost 20% even when the slow branch was never taken.
* **Shared memory per workgroup sets occupancy.** 128x128 uses exactly 40 KiB
  (three workgroups per WGP); a few hundred bytes more drops it to two.

Measured through the public `matmul_abt` (FP16, NT, wall clock around 200
back-to-back launches with one sync, median of 7 batches; hipBLAS and spike
figures from `../vk-spike/RESULTS.md`):

| Shape (M x N x K) | hipBLAS | spike | Vulkan backend | vs hipBLAS |
|---|---|---|---|---|
| 4096 x 4096 x 4096 | 22.7 | 24.9 | 23.8 | 1.05 |
| 2048 x 2048 x 2048 | 38.8 | 30.4 | 29.8 | 0.77 |
| 8192 x 8192 x 8192 | 25.2 | 21.2 | 19.6 | 0.78 |
| 512 x 3072 x 1024 | 28.8 | 25.2 | 23.1 | 0.80 |
| 512 x 1024 x 3072 | 33.8 | 26.2 | 24.8 | 0.73 |
| 512 x 4096 x 4096 | 20.6 | 21.0 | 20.3 | 0.98 |
| 512 x 12288 x 4096 | 21.6 | 17.7 | 16.3 | 0.75 |
| 512 x 4096 x 12288 | 16.2 | 17.5 | 16.2 | 1.00 |
| 4096 x 3072 x 3072 | 36.4 | 32.1 | 34.0 | 0.93 |
| 4096 x 12288 x 3072 | 32.5 | 31.7 | 32.6 | 1.00 |

BF16 operands (converted at load, fallback epilogue) run at 10.7-31.9 TF/s on
the same shapes, the SIMT kernel at 3-8 TF/s in FP16 or FP32, and `matmul`'s
NN layout at the NT figures. `brotensor_test_vulkan --bench-gemm` reproduces
the table (`BROTENSOR_VK_BENCH_TILES=1` adds one column per tile,
`BROTENSOR_VK_BENCH_SHAPE=<tag>` runs one shape, `BROTENSOR_VK_BENCH_GEMV=1`
the GEMV table). The GPU shares its power budget with the CPU: long runs
throttle, so compare configurations in separate short processes.

**Contracts.** As the HIP backend where it is wider than the CPU's: FP32 /
FP16 / BF16 throughout; `linear_forward_batched` takes FP32 activations against
FP32 / FP16 / BF16 weights and accumulates in FP32 without rounding the
activations; `matmul_abt` resizes C only when it cannot hold the batch, and
throws when a stride reaches past an operand (a device fault on Vulkan is a
lost device). The SIMT path has no GLU epilogue: it writes an unrounded FP32 r
and gates it in a second pass, so the result matches the fused kernels.

## Norms, softmax, RoPE

One kernel (`norm.comp`) does the row normalisations, softmax and the column
reductions their backwards need (gamma / beta gradients accumulate, FP32 sums,
one rounding into the gradient's dtype). LayerNorm takes the variance from
deviations in a second pass. RMSNorm accepts FP32 gamma against 16-bit X.
BF16 is computed directly, not converted.

RoPE with angles from `theta_base` computes inv_freq = exp(-(2i/d) ln base)
with ln base from the host's `logf` and an exp accurate to about an ulp, and
takes sin / cos through a three-part Cody-Waite reduction, because GLSL's
`sin` is only specified on [-pi, pi] and RADV's loses bits at decode
positions. What remains is the FP32 angle itself: pos * inv_freq carries
ulp(theta), 0.002 rad at position 30000, on the CPU as much as here, so an
inv_freq one ulp off glibc's moves the result by up to that. Table RoPE is
exact. `rope_qkv_packed_inplace` leaves a row whose position is outside the
table untouched rather than reading past it; M-RoPE positions are not
range-checked (device pointers, as on CUDA / HIP).

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
`VK_KHR_cooperative_matrix`, subgroup size control and the Vulkan memory model
(GLSL's coopmat declares it) when they are present; `PhysInfo::coopmat_f16`
says whether the 16x16x16 FP16 -> FP32 subgroup shape exists at subgroup 32.
RADV's default compute subgroup size is 64, and the coopmat kernels pin it to
32. The other kernels are subgroup-size agnostic (`gemv.comp` needs the size
to divide 64; the dispatcher checks).

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
| `BROTENSOR_VK_NO_COOPMAT=1` | do not use cooperative matrix (the GEMMs run the SIMT kernel) |
| `BROTENSOR_VK_GEMM_CFG=bm,bn,bk,wm,wn` | force one cooperative-matrix tile (one of the four in `gemm.cpp`) |

Vulkan validation layers were not installed on the development machine, so
the backend has not been run under them.
