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
backward, `modulate` and `broadcast_mul`, and the attention family (chunk 3):
`flash_attention_forward`, `_gqa`, `_windowed`, `_varlen`, `_packed_qkv`, the
decode ops over a KV cache (`flash_attention_decode`, `_decode_masked`,
`kv_cache_append`), the projection-fused `flash_attention_qkvo_forward`,
`_project_kv`, `_q_with_kv_cached_forward`, the materialised-probability
attentions (`self_attention_bias_forward`, `cross_attention_forward`,
`_with_attn`, `mha_forward`, `self_attention_forward(_train)`),
`rel_pos_bias_xl_forward`, `top_k_rows`, `segment_softmax_stats` and the
softmax cross-entropies (`softmax_xent`, `_fused`, `_fused_batched`), and the
spatial / diffusion family (chunk 4): `conv2d_forward`, `conv2d_backward_input`,
`conv2d_backward_bias`, `conv3d_forward`, `conv_transpose2d_forward` and its bias
gradient, `resblock_forward`, `group_norm_forward`, `batch_norm_inference`,
`l2_normalize_nchw_forward`, the 2x resamples (`upsample_nearest_2x`,
`upsample_bilinear_2x`, `downsample_avg_2x`, the nearest / average backwards),
`interp2d_forward` / `_align_corners_forward` (nearest, bilinear, both
bicubics), `adaptive_avg_pool2d_forward`, `max_pool2d_forward`,
`convex_upsample_forward`, `pad2d_forward`, `slice2d_forward` / `_backward`,
`unfold2d_forward`, `window_partition_forward` / `window_reverse_forward`,
`spatial_merge_2x2_forward`, `pixel_shuffle_upsample_2x_forward`,
`patch_unpack_forward`, `concat_nchw_channels` (+ backward),
`concat_batched_rows`, `gather_rows`, `scatter_rows`, the sampler steps
(`ddim_step`, `euler_step`, `dpmpp_2m_step`), `timestep_embedding`,
`image_normalize`, `image_u8_to_f32_nhwc_to_nchw` and the Philox noise ops
(`randn`, `rand_uniform`, `rand_bernoulli`, `randn_truncated`): 161 slots.
Attention backwards, the INT8-weight variants (attention, `conv2d_int8w_fp16`,
`conv3d_int8w_fp16`, `resblock_forward_int8w_fp16`) and the spatial backwards
other than the ones named are not implemented yet.
The backend is never the default device. To select it, use
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
  detail/attention.h            the attention dispatcher (ops_attention.cpp) and dense path
  detail/spatial.h              conv2d() / group_norm() entry points the spatial files share
  ops_*.cpp                     ops, one file per family, each with a fill_vulkan_vtable_<family>
                                (elementwise, copy, reduce, linear, norm, rope, glu, attention,
                                attention_proj, topk, xent, conv, gnorm, spatial, diffusion;
                                ops_attention_dense.cpp is the materialised path the others call)
  shaders/                      *.comp kernels, common.glsl, gemm_common.glsl, math_acc.glsl
                                (accurate exp / log / sincos), op_codes.h, shaders.cmake (the list)
tests/test_vulkan.cpp           runtime; main(), --only=ops|gemm|norm|attention|conv|spatial,
                                --bench-gemm, --bench-attention, --bench-conv
tests/test_vulkan_ops.cpp       chunk-1 op parity
tests/test_vulkan_gemm.cpp      matmul / linear parity, every kernel path
tests/test_vulkan_norm.cpp      norms, softmax, RoPE, GLUs parity
tests/test_vulkan_attention.cpp flash family, decode, kv cache parity, every attention path
tests/test_vulkan_attention_ops.cpp  fused / materialised attentions, top-k, segment stats, xent
tests/test_vulkan_conv.cpp      conv2d on every path, conv3d, transposed conv, backwards, ResBlock
tests/test_vulkan_spatial.cpp   resamples, pooling, gathers, NCHW norms, samplers, noise, image ops
tests/test_vulkan_bench.cpp     GEMM / GEMV throughput (not in ctest)
tests/test_vulkan_bench_attention.cpp  attention throughput (not in ctest)
tests/test_vulkan_bench_conv.cpp       conv TF/s, GroupNorm / resample GB/s (not in ctest)
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

## Attention

Every attention op builds an `AttnProblem` (`detail/attention.h`): lq query
rows, lk key rows, hq query heads over hkv KV heads (GQA), head width hd,
row strides for Q, K / V and O, and a mode that gives each query row its own
key interval [lo, hi) (`fa_common.glsl`): q_offset + causal / window over
[0, lk) (with an optional key mask), `cu_seqlens_q` / `_k` (varlen, causal
within a sequence), or per-row `seq_bounds` (packed QKV, band window).
Device-resident tables are clamped to [0, lk], so a malformed table cannot
read outside K. `attention()` picks one of three paths:

| Path | When | What |
|---|---|---|
| `fa_cm.comp` | FP16 / BF16, `coopmat_f16`, more than 4 query rows, hd <= 256 | flash attention on 16x16x16 fragments, FP32 scores and accumulation |
| `fa_rows.comp` (+ `fa_combine.comp`) | at most 4 query rows (decode), FP32, no cooperative matrix, soft-capping, masked decode | one query row per workgroup, the G query heads of one KV head together, keys split across workgroups |
| dense (`ops_attention_dense.cpp`) | bidirectional, unwindowed mode-0 problems on the paths above that do not take them (FP32 with lq, lk >= 64; hd > 256) | S = Q K^T (FP32) through `gemm()`, `attn_softmax.comp`, O = P V through `gemm()` |

**`fa_cm.comp`.** One workgroup is BR = 16 NSG query rows of one head; per
block of BC keys, S = Q K^T goes through shared memory (FP32), two lanes per
row run the online softmax and write P as FP16, the accumulator is rescaled
through the "row index" fragment (a 16x16 accumulator-layout matrix of row
numbers, loaded once, which tells each lane which row each of its elements
is in; KHR cooperative matrix does not say) and O += P V. The spike's kernel
(`../vk-spike/shaders/fa_cm.comp`) with these changes: the GEMM's first rule
(the next K / V block is loaded into registers before the current block's
math and stored to shared memory after it, unconditional clamped loads with a
validity bit when `VEC`), per-row key intervals (the workgroup walks the
union of its rows' intervals; causal blocks above the diagonal are never
loaded), a key-mask stage in shared memory, GQA by head index, the head
padded to 16 * ceil(hd / 16) columns of zeros (any width: 40, 33, 160...),
BF16 converted to FP16 at the shared-memory store (the dtype policy), late
query blocks dispatched first (causal balance), and split keys when the query
blocks alone would not fill the GPU (a few query blocks over a long cache:
workgroup z takes a slice of the keys and writes its unnormalised rows and
(max, sum) in `fa_rows`' partial layout for `fa_combine.comp`). Interior
query blocks with FP16 output and hd % 16 == 0 store straight from the
accumulator (`DIRECT`, a specialisation, as in the GEMM); the last partial
block, BF16 and odd widths use a per-subgroup scratch aliasing S.
Tiles (BC keys x NSG subgroups): (16, 2) for hd <= 96 and > 128, (32, 4) for
hd 128, measured; `BROTENSOR_VK_FA_CFG=bc,nsg` forces one. Shared memory is
8-31 KiB at those tiles, and every array length is a specialisation-constant
expression the pipeline guard checks (a forced 64-key tile at hd 256 is
refused there, not in the driver).

**`fa_rows.comp`.** Each thread scores one key of a 128-key tile against the
G (<= 16) queries staged in shared memory, so K is read once for the whole
GQA group; tile maxima go through subgroup ops and shared memory, the
running sums stay per thread until the end, and P V streams V rows 16 at a
time per thread (all loads issued before use; consuming each load where it
was issued made the loop latency-bound, 2-4x slower). Keys are split across
workgroups until the grid has `BROTENSOR_VK_FA_WGS` (default 80) workgroups:
a long cache streams best in a few long splits (160 and more lost 5-30% at
16k-32k keys). A key whose weight is zero contributes nothing even when its V
row is NaN (masked decode over a fixed-capacity cache).

**Dense path.** Q, K, V and O are addressed in place through the GEMM's
leading dimensions and batch strides (head h is a batch stride of hd), GQA
one KV head at a time with K / V broadcast (batch stride 0) over its group.
Scores are FP32 before the max subtraction (`gemm()` takes 16-bit operands
with an FP32 result through the SIMT kernel), as on HIP; passes of heads,
then of query rows, keep the scores under 256 MiB. It is also the engine of
the projection-fused ops below.

**Projection-fused ops** (`ops_attention_proj.cpp`): `self_attention_bias_forward`,
`cross_attention_forward(_with_attn)`, `mha_forward`,
`self_attention_forward(_train)` keep Q, K, V, scores, probabilities and the
per-head outputs in FP32 whatever X's dtype (the CUDA contract for these
ops); O and AttnAvg come out in X's dtype through `attn_aux.comp`, which also
applies the query gating. `flash_attention_qkvo_forward`, `_project_kv` and
`_q_with_kv_cached_forward` project in the activations' dtype (HIP's
contract) and call the flash path. `rel_pos_bias_xl_forward` is one batched
GEMM over all 2T-1 positions plus a strided copy (the rel shift is a row
pitch of 2T-2).

Measured with `brotensor_test_vulkan --bench-attention` (FP16, wall clock
around back-to-back calls, median of 7 batches, one shape per process:
`BROTENSOR_VK_BENCH_PART=prefill|causal|decode`,
`BROTENSOR_VK_BENCH_SHAPE=<tag>`; HIP is the HIP backend's same public op in
the same process, `BROTENSOR_VK_BENCH_NOHIP=1` skips it; spike figures from
`../vk-spike/RESULTS.md`), ms / TF/s:

| Shape (L, heads, hd) | spike | Vulkan | HIP | vs spike | vs HIP |
|---|---|---|---|---|---|
| 542, 48, 128 | 0.58 | 0.520 / 13.9 | 3.02 / 2.4 | 1.11 | 5.8 |
| 1024, 32, 128 | 1.29 | 1.114 / 15.4 | 6.11 / 2.8 | 1.16 | 5.5 |
| 4096, 24, 128 | 16.1 | 13.05 / 15.8 | 57.4 / 3.6 | 1.23 | 4.4 |
| 1024, 16, 64 | 0.43 | 0.306 / 14.1 | 1.93 / 2.2 | 1.41 | 6.3 |
| 4096, 16, 64 | 5.41 | 4.553 / 15.1 | 33.0 / 2.1 | 1.19 | 7.3 |
| 4096, 24, 64 | 9.17 | 6.552 / 15.7 | 45.4 / 2.3 | 1.40 | 6.9 |
| 1024, 16, 80 | - | 0.439 / 12.2 | 1.90 / 2.8 | - | 4.3 |
| 4096, 16, 96 | - | 7.081 / 14.6 | 35.4 / 2.9 | - | 5.0 |
| 1024, 16, 160 | - | 0.702 / 15.3 | 2.66 / 4.0 | - | 3.8 |
| 2048, 8, 256 | - | 3.973 / 8.7 | 10.0 / 3.4 | - | 2.5 |

Causal (TF/s counting half the flops): 9.8-15.5 TF/s for hd 64-160 (HIP's
causal path is its scalar kernel, 0.05-0.17 TF/s, so 60-170x). Decode
(`flash_attention_decode`, Lq = 1): 16 / 8 heads x 128 at 16k-32k keys reads
the cache at 219-225 GB/s (86-88% of 256 GB/s; HIP 14 GB/s); a 4096-key cache
fits the 32 MB Infinity Cache and shows 405 GB/s; 4-64 query rows over 16k
keys take 0.50-0.61 ms (HIP 4.9-20.7 ms).

## Convolution

`conv2d()` (`ops_conv.cpp`, `detail/spatial.h`) takes NCHW activations and
OIHW (grouped) weights in one dtype and picks one of three kernels:

| Kernel | When | What |
|---|---|---|
| `conv_cm.comp` | FP16 / BF16, `coopmat_f16`, C_out / groups >= 16 and K = C_in / groups kH kW >= 16 | implicit GEMM on 16x16x16 fragments, FP32 accumulation |
| `conv_simt.comp` | FP32, or no cooperative matrix, same size rule | implicit GEMM with FP32 FMA (`gemm_simt.comp`'s tiling) |
| `conv_direct.comp` | depthwise and narrow grouped convolutions (the rule fails) | one output per invocation; also conv3d, the transposed convolution and `conv2d_backward_input` |

**Implicit GEMM.** Per image and group (workgroup z), Y[oc, p] = W[oc, k]
col[k, p] + bias[oc] with M = C_out / groups, N = H_out W_out pixels and K
in the weights' own (ic, kh, kw) order, so A is the weight tensor as stored
and C is the NCHW output (ld = H_out W_out). col is never materialised: the
B-tile loader computes each element's input address, and each thread's
8-pixel chunks are fixed for the whole K loop, so their output coordinates
are decoded once. The kernel geometry (kernel, stride, padding, dilation) is
specialised. The B gather is what bounds the kernel, and its loads come in
three forms (spec constant `BVEC`): a plain 1x1 (stride 1, no padding) loads
a chunk as one aligned 16-byte load; any stride-1 kernel over rows of whole
16-byte chunks (W % 8 == 0, W_out % 8 == 0) loads the two aligned chunks
around it and shifts the 8 halves into place when the tile is stored to
shared memory (the GEMM's first rule: loads at clamped addresses with
validity bits, consumed after the fragment math); everything else takes eight
2-byte loads. The vector forms and tiles with fewer B chunks per thread
took 512 channels at 64x64 from 13.5 to 24.5 TF/s and 128 channels at
512x512 from 13.7 to 18.2. The epilogue is the GEMM's two-form one (fragment-form for interior
FP16 tiles, the bias a column-major stride-0 fragment because it is per row;
per-element for edges and BF16) with an optional accumulate (`ACCUM`, used by
the ResBlock to add conv2 onto the skip path without another pass). Tiles
(`pick_cm`, measured): 256x128 (8 subgroups of 64x64) when C_out fills
256-row tiles and there are >= 48 of them, 128x64 (4 of 64x32) when it fills
128-row tiles, else 64x64; `BROTENSOR_VK_CONV_CFG=bm,bn,bk,wm,wn` forces one.

Measured with `brotensor_test_vulkan --bench-conv` (FP16, batch 1 unless
noted, 3x3 stride 1 same padding unless noted, wall clock around
back-to-back calls, median of 7 batches; HIP is the HIP backend's same public
op, which runs im2col + hipBLAS, in the same process), ms / TF/s:

| Shape | Vulkan | HIP | vs HIP |
|---|---|---|---|
| 512 -> 512, 64x64 | 0.788 / 24.5 | 1.086 / 17.8 | 1.38 |
| 512 -> 512, 128x128 | 2.806 / 27.6 | 4.123 / 18.8 | 1.47 |
| 256 -> 256, 256x256 | 2.763 / 28.0 | 6.429 / 12.0 | 2.33 |
| 128 -> 128, 512x512 | 4.242 / 18.2 | 12.01 / 6.4 | 2.83 |
| 256 -> 128, 512x512 | 7.701 / 20.1 | 22.76 / 6.8 | 2.95 |
| 1x1 512 -> 256, 256x256 | 0.765 / 22.5 | 0.847 / 20.3 | 1.11 |
| 320 -> 320, 64x64, batch 2 (U-Net) | 0.803 / 18.8 | 1.047 / 14.4 | 1.30 |
| 640 -> 640, 32x32, batch 2 | 0.718 / 21.0 | 0.721 / 21.0 | 1.00 |
| 1280 -> 1280, 16x16, batch 2 | 0.717 / 21.1 | 0.760 / 19.9 | 1.06 |

BF16 (converted to FP16 at the shared-memory store) runs at 18.7-25.7 TF/s on
the same shapes, FP32 and the SIMT path at 5.4-8 TF/s
(`BROTENSOR_VK_BENCH_DTYPE=f32|bf16`, `BROTENSOR_VK_BENCH_CONV_PATH=simt|direct`).
The 128-channel shapes stay below the GEMM's 30 TF/s because every B element
is reused by only 128 output channels; staging an input patch in shared
memory once per (ic, kh) and building the B tiles from it would cut the
global loads further.

**Other convolutions.** `conv3d_forward` is one GEMM (Y = X Wt^T + bias,
NT) when the kernel covers the whole input (Qwen-VL's patch embedding), the
direct kernel otherwise. `conv_transpose2d_forward` and
`conv2d_backward_input` are the direct kernel's gather form (the latter reads
the conv weights as the transposed layout with C_in and C_out swapped). The
bias gradients are a per-channel sum (`gnorm.comp`). `resblock_forward`
composes GN + SiLU (one fused pass), conv1 with a per-channel time-embedding
shift folded into its bias, GN + SiLU in place, the skip (copy or 1x1 conv)
written to Y, and conv2 accumulated onto it.

## Spatial ops, NCHW norms, diffusion helpers

* **`resample.comp`**: interpolation (nearest, bilinear, bicubic a = -0.5 /
  -0.75; half-pixel or corner-aligned), the 2x average downsample, the
  nearest / average 2x backwards, adaptive average pooling, max pooling (+
  INT32 flat argmax), RAFT's convex upsample. One invocation is one output
  position and walks 16 channels, so coordinates, tap weights and the convex
  softmax are computed once per position. Source coordinates are exact
  rationals in 64-bit integers (half-pixel: ((2o + 1) in - out) / (2 out)),
  floor and round-half-to-even taken exactly, where the CPU and HIP compute
  them in double; the two agree except where the double product rounds
  across a tie, which no test shape does.
* **`remap.comp`** (`_b2` / `_b4` by element size, bit-exact for any dtype):
  pad (zero / reflect / replicate), crop and its backward (a zero pad),
  nearest 2x upsample, unfold, window partition / reverse, the 2x2 pixel
  unshuffle, the DC-AE pixel shuffle, DiT unpatchify, row gather / scatter
  (an out-of-range index is clamped for a gather and skipped for a scatter:
  a device fault would lose the device). The channel concats are strided
  copies.
* **`gnorm.comp`**: GroupNorm in two kernels. The statistics kernel splits
  each (sample, group) tile, which is contiguous in NCHW, into chunks of
  ~8192 elements and reduces each in one pass to (count, mean, M2) with sums
  shifted by the chunk's first element (no E[x^2] - E[x]^2 cancellation);
  the apply kernel combines a tile's partials with Chan's formula in shared
  memory and normalises (+ SiLU for the ResBlock), 4 elements per load when
  aligned. Traffic is one read for the statistics plus a read and a write.
  BatchNorm inference and the per-pixel L2 normalise are elementwise /
  per-pixel passes.
* **`sampler.comp`**: DDIM, Euler and DPM++ 2M steps with the coefficients
  computed on the host as the CPU does, evaluated `precise` (unfused; the
  CPU backend is built with -mfma and GCC contracts them, so FP32 results
  differ by an ulp), the timestep embedding (exp / sin / cos from
  `math_acc.glsl`), image normalisation and the u8 NHWC -> NCHW conversion,
  whose `src` is a Vulkan device address (the data of a tensor the bytes were
  uploaded to), as on CUDA.
* **`philox.comp`**: Philox 4x32-10 as in `src/cpu/noise.cpp`; the uniform
  and Bernoulli draws are bit-identical to the CPU, the normal ones go
  through `log_acc` / `sincos_acc` and 2 pi u rounded once from the exact
  product, within 2e-6 of glibc.

Measured with `--bench-conv` (FP16, batch 1; GB/s of the minimum traffic,
each input read once and each output written once; GroupNorm reads its input
twice, so its actual traffic is 1.5x the figure; inputs of 32 MB and less
partly stay in the 32 MB Infinity Cache, hence figures above the 256 GB/s
peak):

| Op | Vulkan ms / GB/s | HIP ms / GB/s |
|---|---|---|
| GroupNorm 32 groups, 256 ch 256x256 (32 MB) | 0.519 / 129 (194 actual) | 1.847 / 36 |
| GroupNorm, 128 ch 512x512 (64 MB) | 0.995 / 135 (202 actual) | 4.074 / 33 |
| upsample_nearest_2x, 512 ch 128 -> 256 | 0.415 / 202 | 0.655 / 128 |
| upsample_nearest_2x, 256 ch 256 -> 512 | 0.902 / 186 | 1.303 / 129 |
| bilinear, 512 ch 64 -> 128 | 0.116 / 181 | 0.871 / 24 |
| bilinear, 256 ch 100 -> 333 | 0.408 / 152 | 2.975 / 21 |
| bicubic (torch), 64 ch 512 -> 224 | 0.153 / 261 | FP32 only |

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
  Attention follows the same rule inside the kernel: `fa_cm.comp` converts BF16
  Q / K / V to FP16 as it stages them (a BF16 value beyond ±65504 becomes inf)
  and writes O back as BF16; `fa_rows.comp` computes BF16 in FP32, and the
  dense path keeps its scores in FP32 while its P V GEMM follows the GEMM's rule.
  `conv_cm.comp` does the same as it stages A and B; the SIMT and direct
  convolutions and every spatial kernel compute BF16 in FP32.
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
| `BROTENSOR_VK_FA_CFG=bc,nsg` | force the `fa_cm` tile: bc in {16, 32, 64} keys, nsg in {1, 2, 4} subgroups, 16 nsg <= 2 bc |
| `BROTENSOR_VK_FA_WGS=n` | workgroups `fa_rows` splits keys up to (default 80) |
| `BROTENSOR_VK_FA_PATH=rows\|cm\|dense` | force an attention path where it applies (benchmarking) |
| `BROTENSOR_VK_CONV_CFG=bm,bn,bk,wm,wn` | force the `conv_cm` tile (one of the six in `ops_conv.cpp`) |

Vulkan validation layers were not installed on the development machine, so
the backend has not been run under them.
