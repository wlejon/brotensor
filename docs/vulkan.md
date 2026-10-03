# Vulkan backend

`BROTENSOR_WITH_VULKAN=ON` builds a compute backend made of hand-written GLSL
kernels and a small Vulkan host runtime. It runs on any GPU with a stock Vulkan
1.2+ driver and is brotensor's AMD GPU path. The measurements behind its
design are in `../vk-spike/RESULTS.md` (Radeon 8060S, Mesa RADV).

**HIP removal (2026-10-03).** brotensor used to carry a HIP (ROCm) backend
(`BROTENSOR_WITH_HIP`, `src/hip/`, ~17.6k lines plus ~1.9k of HIP-only tests)
that compiled most of `src/cuda/*.cu` a second time through a compat shim. It
was deleted after commit b052cf3, the last one with `src/hip/`, once Vulkan
covered 266 of 270 op-table slots (training included) and was faster on every
workload measured on the same Radeon 8060S: 1.1-3x on SD1.5, PixArt, Sana,
TripoSplat, Depth-Anything and SAM, 2x Qwen3 decode and 10x prefill, the
T5-XXL encode 2x after the range-safe BF16 GEMM (docs/vulkan-bf16.md), the
flash-attention backward 3.2-5.8x. The side-by-side numbers are in
docs/vulkan-perf.md, and the per-op "HIP" columns below are kept as that
historical comparison. `BROTENSOR_WITH_HIP=ON` is now a configure error;
`DeviceType::HIP`, `Device::hip()`, `hip_device_count()`,
`BROTENSOR_HAS_HIP`, `BROTENSOR_PREFER_HIP` and the `hip` / `rocm` device
strings are gone. Use `-DBROTENSOR_WITH_VULKAN=ON` and `Device::vulkan(i)`
(or `Device::CUDA`, which aliases to Vulkan without a CUDA backend).

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
(`randn`, `rand_uniform`, `rand_bernoulli`, `randn_truncated`), then the
quantised weights and the audio family (chunk 5): the GGUF linears
(`linear_forward_{q8_0,q4k,q6k}_fp16`, their `_batched` forms, `dequant_*_to_fp16`),
the INT8 W8A16 family (`linear_forward_batched_int8w_fp16`, `matmul_int8w_fp16`,
`conv2d_int8w_fp16_forward`, `conv3d_int8w_fp16_forward`,
`resblock_forward_int8w_fp16`, `flash_attention_project_kv_int8w_fp16`,
`_q_with_kv_cached_int8w_fp16`, `_qkvo_int8w_fp16`, `self_attention_bias_int8w_fp16`),
`conv_transpose1d_forward` and its three backwards, `causal_conv1d_update`,
`pad1d_forward` / `_backward`, `snake_forward` / `_backward`,
`resample1d_forward` / `_backward`, `vq_encode_*`, `fsq_quantize_*`, the complex
ops, `fft` / `ifft` / `rfft` / `irfft` (+ adjoints), `stft` / `istft` (+
adjoints), `sample_logits(_into)` and `masked_diffusion_scores` / `_commit`, and the
rest of the inference table (chunk 6): `embedding_lookup_forward`, masked
mean pooling (+ backward), the slot / causal masks, `threshold_u8`,
`rows_count_above`, `attention_token_moments`, the gated delta rule
(`gated_delta_rule_step` / `_chunked`), SAM's decomposed-rel-pos attention
(global and windowed), `deform_conv2d_forward`, StyleGAN3's `bias_act`,
`upfirdn2d`, `modulated_conv2d` (forward and backward: GAN inversion runs
one), `conv2d_backward_weight`, `attention_forward`,
`cross_attention_forward_train`, `xavier_init`, `sgd_step`, `adam_step` and
the MSE losses: 242 of 270 slots. Chunk 8 added the attention / MHA
backwards, training BatchNorm and BCE; chunk 9 the rest of the training
surface (the flash-attention backwards, GroupNorm / ResBlock backwards, the
scatter-adds, the transposed-convolution, resample, padding and pooling
backwards, LSTM training): 266 of 270. The 4 null slots are two host-only
ops and filtered_lrelu (a composite of bias_act + upfirdn2d by design), none
of which is ever dispatched to Vulkan. `docs/vulkan-coverage.md` lists every
slot and, per sibling, what its inference and training code calls.
**Default device.** Without a CUDA or Metal backend, Vulkan is the default
device. The order is CUDA, Metal, Vulkan (`pick_default_from_available`,
`src/init.cpp`). Otherwise select it with `set_default_device(Device::vulkan(i))`, a
`DeviceScope`, or `BROTENSOR_DEFAULT_DEVICE=vulkan` (also `vk`, `vulkan:1`).
An op in a null slot throws "not implemented on vulkan"; since chunk 9 no
public op reaches one (training included).
`Device::cuda(i)` aliases to `Device::vulkan(i)` when no CUDA backend is
registered and Vulkan is (`detail::resolve_device_alias`); with Metal
registered as well it aliases to Metal unless the default device is Vulkan
(the same alias makes the generic suites run on Metal in a Metal build). The generic test
suites name the GPU as `Device::CUDA` and pick it through `tests/gpu_select.h`
(the one device rule every suite uses); in a Vulkan-only build they all run
on Vulkan, and in a build with CUDA or Metal as well they run again on Vulkan
as `<name>_vulkan` (`BROTENSOR_TEST_GPU=vulkan`,
`BROTENSOR_DEFAULT_DEVICE=vulkan`).
Trace-JIT (`jit::*`) DAGs on Vulkan are compiled to one SPIR-V kernel each
(see "Trace JIT" below); a DAG with no fusion is replayed op by op through
the dispatched ops (`src/jit/trace_eager.cpp`). The `fused_*` ops need no
Vulkan case (they compose dispatched ops; the stacked-weight SwiGLU GEMV
takes the GEMV epilogue).

## Build

```sh
cmake -B build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DBROTENSOR_WITH_VULKAN=ON
cmake --build build_vk && ctest --test-dir build_vk -R vulkan
```

The build needs the Vulkan headers and `glslc` (shaderc). If `glslc` is not
found, pass `-DBROTENSOR_GLSLC=`. Nothing links `libvulkan`: the loader is
`dlopen`ed at run time, so a binary built with the backend still starts (and
falls back to the other backends) on a machine without Vulkan. The backend
coexists with CUDA or Metal in one build. `cmake/BrotensorVulkan.cmake`
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
  graph.cpp                     VulkanGraph / VulkanGraphCapture, Event, the neutral capture's recorder
  custom.cpp                    custom kernels: register_shader / kernel_info / dispatch
  vulkan_jit.h, .cpp            brass SpirvKernel -> pipeline, for src/jit/trace_compiler_vulkan.cpp
  register.cpp                  probe + vtable fill + public stats
  detail/gemm.h, gemm.cpp       the GEMM dispatcher every matrix op goes through
  detail/attention.h            the attention dispatcher (ops_attention.cpp) and dense path
  detail/spatial.h              conv2d() / group_norm() entry points the spatial files share
  detail/quant.h                quant_weight() / quant_linear() / dequant(): the quantised-weight core
  ops_*.cpp                     ops, one file per family, each with a fill_vulkan_vtable_<family>
                                (elementwise, copy, reduce, linear, norm, rope, glu, attention,
                                attention_proj, topk, xent, conv, gnorm, spatial, diffusion,
                                quant, quant_attention, audio, spectral, sampling, misc, delta,
                                vision, attention_bwd, fa_bwd, scatter, spatial_bwd, lstm;
                                ops_attention_dense.cpp is the materialised path the others call)
  shaders/                      *.comp kernels, common.glsl, gemm_common.glsl, math_acc.glsl
                                (accurate exp / log / sincos), quant_decode.glsl (the quantised
                                chunk loads / decodes), op_codes.h, shaders.cmake (the list)
tests/test_vulkan.cpp           runtime, the Device::CUDA alias; main(),
                                --only=ops|gemm|norm|attention|conv|spatial|quant|audio|misc|vision|capture|train|jit|alias,
                                --bench-gemm, --bench-attention, --bench-conv, --bench-quant, --bench-audio, --bench-jit
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
tests/test_vulkan_quant.cpp     GGUF / INT8 linears on both paths, dequantise, INT8 convolutions
tests/test_vulkan_quant_attention.cpp  INT8-weight attentions and ResBlock
tests/test_vulkan_audio.cpp     1D convs, Snake, pad / resample, codec quantisers, complex, FFT, STFT
tests/test_vulkan_sampling.cpp  sample_logits(_into), masked diffusion
tests/test_vulkan_bench_quant.cpp      quantised linears GB/s / TF/s, audio ops (not in ctest)
tests/test_vulkan_misc.cpp      embedding, pooling / masks, thresholds, init / optimisers / MSE, moments,
                                bias_act / upfirdn2d / filtered_lrelu, gated delta rule, BF16 checkpoint load
tests/test_vulkan_vision.cpp    SAM rel-pos attention, deform / modulated conv (+ backward),
                                conv2d_backward_weight, attention_forward, cross_attention_forward_train
tests/test_vulkan_capture.cpp   the device-neutral CudaGraphCapture on Vulkan, custom kernels
                                (--only=capture; shaders in tests/vulkan_shaders/)
tests/test_vulkan_train.cpp     attention / MHA / self / cross attention backwards, training BatchNorm, BCE,
                                LSTM (--only=train; runs the three below too)
tests/test_vulkan_train_fa.cpp  flash-attention backwards (bare, varlen, packed QKV, QKVO)
tests/test_vulkan_train_spatial.cpp   GroupNorm / ResBlock backwards
tests/test_vulkan_train_spatial2.cpp  scatter-adds, conv_transpose2d backwards, resample / pad / pool backwards
tests/test_vulkan_jit.cpp       the trace JIT's compiler (--only=jit); test_vulkan_bench_jit.cpp: --bench-jit
tests/test_vulkan_common.h
```

Keep every file under 1000 lines: a new op family gets its own `ops_<family>.cpp`.

## Memory model

- **Pointers are device addresses.** A Vulkan tensor's `Tensor::data` holds the
  `VkDeviceAddress` of its first byte (buffer device address), cast to `void*`.
  As a result, views at an offset, `copy_d2d` offsets, and the raw
  `const float* d_mask` / `const int32_t*` operands in the op table all work
  exactly as on CUDA. The host never dereferences these values. When a
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
  with the same semantics as `cudaFreeAsync` on one stream. A dedicated block is
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
  directions return synchronously. Vulkan↔Vulkan copies across two
  devices go through the host. Vulkan↔CPU / other-backend copies use
  `Tensor::to`, which also bounces through the host.

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
  `SIMULTANEOUS_USE`, and `launch()` resubmits it unchanged. The recording
  is cut into segments of `BROTENSOR_VK_BATCH` commands, each launched as its
  own queue submission (in order, barrier-separated like eager batches): the
  amdgpu driver resets the GPU when one submission runs past ~10 s, which a
  whole captured TripoSplat step (two 3.6 s flow forwards) did in one. While a capture is
  recording, `sync`, downloads and uploads on that device throw, and frees are
  held by the graph until it is destroyed. The spike measured 0.74 µs per
  dependent kernel for replay, against 1.9 µs for HIP and its graphs. The
  chunk-1 test reports wall time including the sync. For a 301-command chain
  of 4096-element elementwise ops and copies, that was about 1.1 µs per command
  replayed and 1.4 µs per command eager.
- **Device-neutral capture.** `CudaGraphCapture` / `CudaGraph` (`cuda_graph.h`)
  are defined once in the core (`src/graph.cpp`) and forward to a recorder
  each backend registers at probe time (`detail/graph_backend.h`): CUDA's is
  what those classes used to be (unchanged), Vulkan's is a
  `VulkanGraphCapture`. `CudaGraphCapture()` records on the default device
  when it is a GPU with capture (so `BROTENSOR_DEFAULT_DEVICE=vulkan` or a
  `DeviceScope` makes an unchanged `bt::CudaGraphCapture cap; ...;
  g = cap.finish(); g.launch();` site record on Vulkan); with a CPU / Metal
  default device it falls back to CUDA's current device as before, then
  Vulkan 0. `CudaGraphCapture(Device)` names the device and
  `graph_capture_available(Device)` is the gate a caller checks instead of
  `device == CUDA`. What still differs per backend: a Vulkan
  capture records every thread's ops on its device (CUDA: the
  capturing thread's), and sync, uploads and downloads on it throw while it
  records. Tested by `brotensor_test_cuda_graph_vulkan`,
  `brotensor_test_graph_capture_alloc_vulkan` (the CUDA tests with Vulkan
  as the default device) and `brotensor_test_vulkan --only=capture`.

## Custom kernels

A sibling ships GLSL for a Vulkan device the way it ships MSL for Metal
(`metal_interop.h`). CMake, after including `cmake/BrotensorVulkan.cmake`:

```cmake
brotensor_vulkan_prepare()            # glslc, brotensor's shader include dir
brotensor_vulkan_add_shaders(mylib NAMESPACE mylib::vk HEADER mylib_vk_shaders.h
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
    SHADERS "my_op_f32|my_op.comp|-DDT=0" "my_op_f16|my_op.comp|-DDT=1")
```

compiles each entry with brotensor's glslc flags, brotensor's shader directory
on the include path (`common.glsl`, `math_acc.glsl`), embeds the SPIR-V in
`mylib` and generates `mylib_vk_shaders.h` with `enum class Shader` and
`handle(Shader)`, which registers the SPIR-V with `vulkan::register_shader`
on first use. The host side fills a push block (`vulkan::address(t)` per
tensor, scalars after the 64-bit fields, <= 128 bytes) and calls
`vulkan::dispatch(device, handle(Shader::my_op_f32), &pc, sizeof pc, gx, gy, gz,
spec, nspec)`; the dispatch goes on the device's stream (ordered with
brotensor's ops, recorded by a capture) through the same pipeline cache and
limit guard as the built-in kernels. `DeviceInfo::shader_float64` says
whether `double` is available (the device enables `shaderFloat64` when
present; brovisionml's DSINE kernels use it, as their CPU / CUDA twins do).

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
| `gemm_cm.comp` | FP16 / BF16 operands (C in their dtype or FP32), `coopmat_f16` device | 16x16x16 FP16 fragments, FP32 accumulation, subgroup 32, double-buffered shared tiles; a BF16 A staged at a per-row power-of-two scale (docs/vulkan-bf16.md) |
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
* **Workgroups start their K walk at different blocks** (chunk 5): block
  `((x + 5 y) * 37) mod nk`, then around. With every workgroup at the same K,
  the rows of a tile (a row pitch apart) and of the concurrent tiles fall on
  the same memory channels when the pitch is a multiple of 4 KiB (8B's
  K = 12288 and 14336 rows): INT8 512 x 4096 x 12288 ran at 8.6 TF/s against
  24.9 for K = 12352. The rotation took it to 19.7; in one session, A / B
  with and without it, FP16 8B-up went 15.0 -> 22.0 TF/s, DiT 4096 x 3072 x
  3072 27.1 -> 32.5 and square2k 25.6 -> 29.0 (no shape measured slower). The
  sum is the same in another order, deterministic per tile.

Measured through the public `matmul_abt` (FP16, NT, wall clock around 200
back-to-back launches with one sync, median of 7 batches; hipBLAS and spike
figures from `../vk-spike/RESULTS.md`):

| Shape (M x N x K) | hipBLAS | spike | Vulkan backend | vs hipBLAS |
|---|---|---|---|---|
| 4096 x 4096 x 4096 | 22.7 | 24.9 | 26.9 | 1.18 |
| 2048 x 2048 x 2048 | 38.8 | 30.4 | 28.7 | 0.74 |
| 8192 x 8192 x 8192 | 25.2 | 21.2 | 19.4 | 0.77 |
| 512 x 3072 x 1024 | 28.8 | 25.2 | 23.7 | 0.82 |
| 512 x 1024 x 3072 | 33.8 | 26.2 | 24.5 | 0.72 |
| 512 x 4096 x 4096 | 20.6 | 21.0 | 22.6 | 1.09 |
| 512 x 12288 x 4096 | 21.6 | 17.7 | 22.0 | 1.02 |
| 512 x 4096 x 12288 | 16.2 | 17.5 | 18.8 | 1.16 |
| 4096 x 3072 x 3072 | 36.4 | 32.1 | 31.6 | 0.87 |
| 4096 x 12288 x 3072 | 32.5 | 31.7 | 32.2 | 0.99 |

(Chunk-5 figures, with the K rotation; chunk 2 measured 23.8 / 29.8 / 19.6 /
23.1 / 24.8 / 20.3 / 16.3 / 16.2 / 34.0 / 32.6.)

BF16 operands (converted at load, A at a per-row scale, fallback epilogue;
docs/vulkan-bf16.md) run at 17.9-29.6 TF/s on the same shapes, the SIMT
kernel at 3-8 TF/s in FP16 or FP32, and `matmul`'s
NN layout at the NT figures. `brotensor_test_vulkan --bench-gemm` reproduces
the table (`BROTENSOR_VK_BENCH_TILES=1` adds one column per tile,
`BROTENSOR_VK_BENCH_SHAPE=<tag>` runs one shape, `BROTENSOR_VK_BENCH_GEMV=1`
the GEMV table). The GPU shares its power budget with the CPU: long runs
throttle, so compare configurations in separate short processes.

**Contracts.** As the HIP backend where it is wider than the CPU's: FP32 /
FP16 / BF16 throughout; `linear_forward_batched` takes FP32 activations against
FP32 / FP16 / BF16 weights and accumulates in FP32 without rounding the
activations (SIMT; only an op's own FP32 intermediates may take the scaled
cooperative-matrix path, `GemmArgs::round_a`); `matmul_abt` resizes C only when it cannot hold the batch, and
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
hd 128, measured; for hd <= 96, (32, 2) once K and V together pass 24 MiB
(each query block streams its head's whole K / V, so past the 32 MB
Infinity Cache the bigger key block halves the passes: 16 heads x 64 at 8192 /
12288 keys went from 34.6 / 205 ms to 27.1 / 60.2 ms, 8.0 / 3.0 -> 10.1 / 10.3
TF/s; TripoSplat's flow DiT, 8-13k rows, sampled 2.5x faster);
`BROTENSOR_VK_FA_CFG=bc,nsg` forces one. Shared memory is
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
`BROTENSOR_VK_BENCH_SHAPE=<tag>`; the HIP column is the removed HIP backend's
same public op, measured in the same process before its removal; spike figures from
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
back-to-back calls, median of 7 batches; the HIP column is the removed HIP
backend's same public op, im2col + hipBLAS, measured in the same process), ms / TF/s:

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

## Quantised weights

`detail/quant.h` describes a quantised weight (`quant_weight()`: INT8 with
per-row FP32 scales, or the GGUF block formats Q8_0, Q4_K and Q6_K, the ones
brolm's GGUF loader keeps) and `quant_linear()` computes Y(B, N) = X(B, K)
W^T + bias for FP16 (or, INT8 only, BF16) activations. A weight is never
expanded to FP16 on the matrix paths:

| Path | When | What |
|---|---|---|
| `gemv_q.comp` | B <= 8 rows (decode) | each W row read once, decoded in registers, all B rows of X against it |
| `gemm_cm.comp` with `QB` | B > 8 (prefill), cooperative matrix | the B tile's chunks decoded to FP16 as they are stored to shared memory |
| `dequant.comp`, then the dense GEMM | B > 8 without cooperative matrix; conv3d's direct geometry | an FP16 copy of the weight |

**Decode GEMV.** A lane group of LPR lanes (16 / 32 / 64) walks NR rows of
W together (1, 2 or 4), so every X chunk it loads serves NR rows. A lane's
unit is an *item*: 16 weights for INT8 and Q8_0 (one 16-byte load, half a
Q8_0 block), 32 for the K-quants (16 bytes of Q4_K nibbles = two
sub-blocks' halves; 24 bytes of Q6_K = 8 values of each of the four quads of
a 128-element half), with UNR items per lane loaded before any is decoded.
Each item is decoded once into FP32 weights with the block scales and
minimums folded in (Q4_K: w = d sc q - dmin m), so the inner loop over the B
rows of X is one multiply-add per weight, which ACO emits as `v_fma_mix_f32`
reading the FP16 X directly. Folding the minimum per weight instead of
subtracting dmin m sum(x) per sub-block, and NR > 1 (X loaded once per NR rows:
at B = 8 the X re-reads otherwise saturate the L0 cache), took Q4_K at B = 8
from 100 to 160-206 GB/s. 2-byte-aligned data (Q8_0's 34-byte and Q6_K's
210-byte blocks) is loaded as 16-bit vectors (`u16vec4`; an `Aligned 2`
8-byte load is invalid SPIR-V) and packed by the decode. Configuration
(`gemv_cfg`, measured): wave64; INT8 / Q8_0 rows of K >= 4096 one row per
group with four items in flight, shorter rows two per group; Q4_K LPR 32 /
NR 4; Q6_K NR 2 (4 at B = 8). `BROTENSOR_VK_QGEMV=lpr,unr,sg,nr` forces one.

**Prefill GEMM.** `gemm_cm.comp`'s B loader takes `QB` (`QF_*`): a chunk is
8 weights of one row (one K step of 32 is a Q8_0 block, a Q4_K sub-block,
half a Q6_K quad pair), loaded raw (`qload8`, `quant_decode.glsl`) at the
GEMM's load point and decoded to FP16 (`qdecode8_f16`: d sc q - dmin m in
FP32, one RNE rounding) when the tile is stored to shared memory, so the
decode overlaps the fragment math like the BF16 conversion does. INT8 scales
are applied there too (q scale rounded once to FP16), so the result equals
the FP16 GEMM on the dequantised weight up to summation order.
`matmul_int8w_fp16` is the linear on X^T, transposed back.
`conv2d_int8w_fp16_forward` (and the ResBlock's) decode INT8 A tiles the same
way in `conv_cm.comp` (`QA`, per-output-channel scale); other conv paths
dequantise the (small) weight first. The INT8 attentions are the projections
through `quant_linear()` around the FP16 attention core
(`ops_quant_attention.cpp`).

Measured with `brotensor_test_vulkan --bench-quant` (wall clock around
back-to-back calls, median of 7 batches, one format per process:
`BROTENSOR_VK_BENCH_FMT=int8|q8_0|q4k|q6k`, `BROTENSOR_VK_BENCH_PART=decode|prefill`;
the HIP column is the removed HIP backend's same public op, measured in the
same process). Decode, GB/s of the stored
weight bytes at B = 1 / 2 / 4 / 8; shapes over 32 MB (the Infinity Cache)
are DRAM-bound, smaller ones read partly from the cache:

| Format, shape (N x K, MB) | Vulkan | HIP |
|---|---|---|
| Q8_0 151936 x 1024 (165) | 237 / 236 / 232 / 224 | 233 / 117 / 59 / 30 |
| Q8_0 12288 x 4096 (53) | 226 / 228 / 220 / 202 | 224 / 114 / 58 / 29 |
| Q8_0 4096 x 12288 (53) | 226 / 225 / 220 / 193 | 209 / 112 / 57 / 29 |
| INT8 151936 x 1024 (156) | 239 / 239 / 236 / 232 | 47 / 47 / 48 / 47 |
| INT8 12288 x 4096 (50) | 234 / 229 / 229 / 203 | 29 / 29 / 27 / 21 |
| INT8 4096 x 12288 (50) | 232 / 227 / 227 / 208 | 17 / 17 / 16 / 17 |
| Q4_K 151936 x 1024 (88) | 235 / 233 / 225 / 163 | 122 / 61 / 31 / 15 |
| Q4_K 12288 x 4096 (28) | 515 / 417 / 308 / 206 | 217 / 109 / 54 / 27 |
| Q6_K 151936 x 1024 (128) | 237 / 235 / 228 / 204 | 181 / 91 / 46 / 23 |
| Q6_K 12288 x 4096 (41) | 268 / 262 / 244 / 206 | 197 / 100 / 50 / 25 |

HIP's batched GGUF linear launches its GEMV once per activation row, hence
its 1/B scaling. Prefill, TF/s:

| Shape (M x N x K) | INT8 | Q8_0 | Q4_K | Q6_K | HIP (INT8 / GGUF) |
|---|---|---|---|---|---|
| 512 x 4096 x 4096 | 22.4 | 21.6 | 18.6 | 17.7 | 1.76 / 0.7-1.5 |
| 512 x 12288 x 4096 | 23.3 | 26.9 | 27.8 | 24.2 | 1.48 / 0.4-0.8 |
| 512 x 4096 x 12288 | 19.8 | 22.8 | 22.9 | 20.9 | 1.45 / 0.4-0.8 |
| 4096 x 3072 x 3072 | 28.2 | - | - | - | 1.76 |
| 4096 x 4096 x 4096 | 26.9 | - | - | - | 1.76 |

## Audio, spectral ops, sampling

* **1D convolutions** (`ops_audio.cpp`): conv1d is conv2d with H = 1 (the
  public wrapper). `conv_transpose1d_forward` is a GEMM, cols (C_out / g kL,
  L) = Wt_g^T X_g per image and group (TA + NB layouts, cooperative matrix
  for FP16; FP32 cols for BF16 signals, rounding each product to BF16 cost a
  few ulp), then an overlap-add gather (`AU_COL2IM`): 512 -> 256 channels,
  k16 s8 at L = 1000 takes 0.42 ms in FP16, 0.81 ms in FP32, against 55 ms
  for the direct gather kernel and 64 ms on HIP. Its input gradient is a
  strided, dilated, grouped conv1d of dY with the same weights through
  `conv2d()`; the weight gradient is one workgroup per weight element
  reducing over (n, l) (correct, slow: a GEMM over im2col'd dY is the fix
  when training needs it); the bias gradient is conv2d's.
* **`audio.comp`** (FP32 / FP16 / BF16 signals, FP32 per-channel parameters):
  Snake / SnakeBeta (`sincos_acc`; the backward one workgroup per channel),
  pad1d and its adjoint as a gather (reflect images enumerated), resample1d
  with exact rational source positions (round-half-even nearest as the
  CPU's `nearbyint`) and its adjoint as a gather in output order,
  `causal_conv1d_update` (one invocation per (n, c) row, state rolled in
  increasing order), VQ (one workgroup per row, first minimum), FSQ.
* **Spectral** (`ops_spectral.cpp`, `dft.comp`, `dft64.comp`, FP32 in and
  out). Frame-sized transforms (L <= 2048; C2C <= 1024) on a device with
  `shaderFloat64` run `dft64.comp`: a direct DFT in FP64, one workgroup per
  row, the twiddle table e^{2 pi i m / L} built per workgroup in shared memory
  from an FP64 Taylor sincos on the angle folded to [0, pi / 4] (m reduced
  exactly in integers), every product and sum in double and one rounding to
  FP32 at the store, which is what the CPU (FP64 FFT) and CUDA / HIP (FP64
  direct DFT) references do. A frame's result depends only on its samples,
  so streaming == offline bit for bit. Longer transforms (and devices without
  FP64, or `BROTENSOR_VK_DFT_GEMM=1`) are a matrix product against a DFT
  basis built on the device per call (same twiddle reduction, `sincos_acc`,
  ~1e-7 per coefficient; rows in chunks of <= 16 Mi elements) through the FP32
  GEMM: any length, primes included, but its 400-2048-term FP32 sums carry
  ~5e-5 absolute error at n_fft 512 (CPU: ~1e-6), and the GEMM's K blocking
  rounds a frame differently by its row position. Chunk 8 measured both on
  brosoundml `test_mel` (n_fft 512, win 400, log-mel): GEMM 0.025 log-mel vs
  the CPU and 0.018 streaming vs offline; FP64 DFT below 1e-4 on both. The
  STFT reads its frames in place (the reflect-padded signal with a row pitch
  of hop_length), the window and normalisation folded into the transform;
  the inverse's overlap-add and COLA division, and the two adjoints'
  scatters, are gathers. Whisper's front end (30 s, n_fft 400, hop 160):
  STFT 3.6 ms / iSTFT 4.6 ms on the FP64 DFT (0.20 / 0.23 ms on the FP32
  GEMM), against 53 / 124 ms on HIP (an FP64 O(L^2) DFT kernel).
* **Sampling** (`select.comp`, one workgroup per row): the CPU sorts the
  vocabulary; here (probability descending, index ascending) is a 64-bit key
  per token and each "first rank such that" (top-k, the top-p nucleus, the
  draw's cumulative mass) is a binary search over the key space, one pass
  per step, with masses in 64-bit fixed point (2^-56, exact and
  order-independent where the CPU sums in FP64). The draw is the CPU's
  Philox stream; the masked-diffusion class filter finds the k-th value the
  same way over order-preserving float bits and ranks ties by index with a
  workgroup scan. Tokens match the CPU exactly in the tests (V up to 151936);
  a draw can only differ when the uniform lands within a few ulp of a
  probability boundary. At V = 151936 a row costs ~1-2 ms (up to ~190 passes
  over the row); a radix select would cut that if LLM sampling moves here.

Measured with `brotensor_test_vulkan --bench-audio` (GB/s of minimum traffic):
Snake 512 x 24000 234 (FP32) / 209 (FP16) GB/s (HIP 214, FP32 only), pad1d
230 (HIP 228), resample1d 113 (HIP 110, the per-output 64-bit rational
arithmetic bounds it), complex_abs 685 (cache-resident).

## Chunk-6 ops

* **`misc.comp`** (`ops_misc.cpp`, FP32 / FP16 / BF16 by `DT`): the
  embedding lookup is `gather_rows` over a view of the device index array (an
  out-of-range index is clamped, not a fault); masked mean pooling counts the
  valid rows (mask >= 0.5, the CPU's rule) per invocation; `rows_count_above`
  and `attention_token_moments` are one workgroup per row / key; `xavier_init`
  draws splitmix64 at state + (i + 1) K, bit-identical to the CPU walk, and
  advances the state by n K; SGD / Adam are `precise` FP32 elementwise steps
  (the bias corrections from the host's `powf`); `bias_act`'s dB recomputes
  the gradient in FP32 per channel rather than summing the rounded dX;
  `upfirdn2d` visits the CPU's taps in the CPU's order, and its backward is
  the forward with up / down swapped. filtered_lrelu has no slot (the public
  op's composite of the two runs here).
* **Gated delta rule** (`delta_rule.comp`): every row of a head's (d_v, d_k)
  state evolves on its own, so a group of 32 lanes (the subgroup size is
  pinned when it can be) owns one (head, row), keeps the row in registers for
  the whole token walk, and reduces both dot products with subgroup shuffles;
  the state is read and written once per call whatever L is, and step and
  chunked are one kernel. d_k up to 512.
* **SAM attention** (`ops_vision.cpp`): windows are row gathers from X with
  one zero row appended (the zero padding), projections in X's dtype, Bh =
  Q_h rel_pos_h^T and Bw likewise as one batched FP32 GEMM over the heads,
  then the dense path with `attn_softmax.comp`'s `BIAS = 2` mode adding
  Bh[q, qh - kh + gh - 1] + Bw[q, qw - kw + gw - 1] to each scaled score (the
  (L, L) bias never exists), and O gathered back.
* **Deformable / modulated convolution, weight gradients**: deform_conv2d is
  a bilinear im2col (torchvision's corner rules) per image, group and slab of
  <= 32 Mi col elements, then `gemm()` and the channel bias;
  `modulated_conv2d_forward` builds the per-sample weights and demodulation
  in one pass and runs one grouped conv2d with groups = N over the batch as a
  single image (so the implicit-GEMM paths apply). `conv2d_backward_weight`
  and the modulated backward's dw'' are im2col slabs (the same kernel without
  offsets) and dW += dY col^T through `gemm()`; the backward's dX is
  `conv2d_backward_input` with w'' as one grouped convolution.
* **conv_transpose2d** is now conv_transpose1d's design in 2D: cols (C_out kH
  kW, H W) = Wt_g^T X_g per image batch and group through `gemm()`
  (cooperative matrix for FP16, FP32 cols for BF16), then the 2D overlap-add
  gather (`MI_COL2IM2D`); the direct gather remains for an image whose cols
  exceed 256 MiB and under `set_conv_override(2)`.
* `attention_forward` is `mha_forward` with one head, and
  `cross_attention_forward_train` is `cross_attention_forward` keeping its
  per-head caches (both `ops_attention_proj.cpp`).

These are correctness-first: none of them was benchmarked against HIP in
chunk 6.

## Training ops

The training backwards (chunk 8: attention / MHA / self / cross attention,
training BatchNorm, BCE; chunk 9: flash-attention backwards, GroupNorm /
ResBlock backwards, scatter-adds, transposed-convolution, resample, pad and
pool backwards, LSTM; the flash backward's cooperative-matrix kernel
`fa_bwd_cm.comp`) are described in `docs/vulkan-training.md`, with their
measured errors against the CPU and the flash backward's speed.

## Trace JIT

`begin_trace()` / `end_trace()` on Vulkan tensors compile the DAG to one
kernel (`src/jit/trace_compiler_vulkan.cpp`, hook `register_vulkan_trace_compiler`)
from the CUDA compiler's plans (`src/jit/trace_plan.h`, shared): **elementwise**
(grid-stride, 16-byte accesses, `(1, cols)` rows / `(1, 1)` scalars by column,
per-buffer FP32/FP16/BF16, FP32 math) and **row-norm** (one RMSNorm /
LayerNorm with the chain before and after it, row groups of 32-256 threads, a
32-lane subgroup butterfly plus one shared round, a 2-D grid past 65535
groups; LayerNorm's variance two-pass, the sum of squared deviations, as in
the PTX emitter). Each plan is brass MIR through `KernelBuilder`
(`src/jit/spirv_emit*.cpp`), a vector and a scalar entry lowered by
`SpirvTarget::compile`; bind picks the vector entry when every pointer is
aligned for it and packs the push block once, so a replay is one dispatch on
the stream (barrier-ordered, captured into graphs). `launch_count()` is 1,
`fusion_name()` names the entry, `compile_us()` covers MIR, SPIR-V and the
pipelines. brass's runtime contract is this backend's dispatch model (u64
device addresses in the push block — a tensor's `data` already is one —,
Pipelines' descriptor-free layout, workgroup size as spec constants 0..2,
subgroup 32 required when possible: `src/vulkan/vulkan_jit.cpp`). Modules are
interned by their SPIR-V words (a recompiled trace reuses its pipelines) and
their capabilities checked once per device (`jit_missing_for` lists every
missing feature). BF16 is a shift plus a bit reinterpretation, MIR's
`bitcast.i32` / `bitcast.f32` (one `OpBitcast` each), at no measurable cost
(BF16 runs at the FP16 bandwidth); rounding is RNE with quiet NaNs, as
`common.glsl`. A DAG with no plan (two reductions, an op with no
elementwise form, too many buffers for the push block, >= 2^31 elements) or a
compile failure is replayed op by op (`trace_eager.cpp`; a failure's reason
is printed once, `BROTENSOR_JIT_STRICT=1` throws instead,
`BROTENSOR_JIT_VULKAN=eager` disables the compiler). Built when brass is a
sibling target with its SPIR-V target (`BROTENSOR_HAS_VULKAN_TRACE_JIT`).
Measured (`--bench-jit`): 207-232 GB/s on cache-cold shapes, 3.2-7.8x the
replay on LN-modulate, 5.7-6.7x on the VAE norm + SiLU, 9-15x on a ten-op
chain; a Qwen-Image 2.1 1024 x 1024 denoise step 3.51 s against 3.75 s
replayed and 3.60 s with its trace sites off (docs/vulkan-perf.md).

## Dtype policy

- **FP32**: native.
- **FP16**: stored as `float16_t` with 16-bit storage. Arithmetic is FP32, and
  the store uses `float16_t(x)`, which is round-to-nearest-even. Never use
  `packHalf2x16`, which rounds toward zero on RADV. Casts are bit-identical to
  `brotensor::fp32_to_fp16_bits`, and the tests check this, including 65520 →
  inf.
- **BF16**: RADV has no BF16 shader type by default (no
  `VK_KHR_shader_bfloat16`, no BF16 cooperative matrix; both exist behind
  `RADV_EXPERIMENTAL=bfloat16`, docs/vulkan-bf16.md). BF16 is therefore carried as `uint16` bits and
  converted in registers (`bf16_to_f32` / `f32_to_bf16` in `common.glsl`,
  RNE, NaN kept quiet, bit-identical to the host helpers). This runs at full
  bandwidth for memory-bound ops, so the elementwise, reduction and norm
  families take BF16 directly. **The policy for matrix-core ops (GEMM,
  attention) is decided here:** `compute_dtype(Device::vulkan(i))` is FP16, so
  a loader holding BF16 weights converts them **once at upload** with
  `cast(…, Dtype::FP16)`, and the coopmat kernels then see FP16 weights. A
  BF16 *activation* is range-safe in the GEMM: the cooperative-matrix kernel
  stages a BF16 A operand into FP16 at a per-row power-of-two scale (exact for
  BF16's 8-bit significand, undone in the FP32 epilogue; docs/vulkan-bf16.md),
  so models that need BF16's range (T5-XXL, Sana) run their BF16 forward as
  on CUDA / HIP. The B operand is converted unscaled and must lie within
  ±65504. Attention stages unscaled too: `fa_cm.comp` converts BF16
  Q / K / V to FP16 as it stages them (a BF16 value beyond ±65504 becomes inf)
  and writes O back as BF16; `fa_rows.comp` computes BF16 in FP32, and the
  dense path keeps its scores in FP32 while its P V GEMM follows the GEMM's rule.
  `conv_cm.comp` does the same as it stages A and B; the SIMT and direct
  convolutions and every spatial kernel compute BF16 in FP32.
- **At load**: `safetensors::upload_compute(_checked)` and `upload_as` already
  implement the policy on Vulkan (BF16 bits uploaded raw, cast to the compute
  dtype, FP16, on the device; `test_vulkan_misc.cpp` pins it bit for bit).
  `safetensors::upload`, `gguf::upload_raw` and `Tensor::to` keep the source
  dtype by contract, so a BF16 weight that arrives through them stays BF16 and
  the matrix kernels convert it per load (correct, slower).
- **INT8 / INT32**: storage carriers only (8-bit storage is enabled): INT8
  weights with FP32 scales, INT32 indices / levels / tokens. The GGUF block
  dtypes (Q8_0, Q4_K, Q6_K) are opaque byte carriers decoded by the quantised
  kernels. There is no FP8.

GLSL built-ins are not IEEE-exact. Neither is the optimiser: NIR reassociates
a floating-point expression across a function boundary unless it is
`precise`, so FSQ's `div_rn(idx, h) - 1.0` came out an ulp or two off the
CPU's (chunk 5); `math_acc.glsl` now has `div_rn_precise` for callers whose
next operation must round on its own. It is not a drop-in replacement:
image_normalize's `(x - mean) * div_rn(1, std)` matches the CPU bit for bit
with the plain `div_rn` and is an ulp off with the precise one, so each
kernel keeps the form its exact-parity test pins. Also: `exp` is about 3 ulp plus |x|·ulp, and
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
| `BROTENSOR_DEFAULT_DEVICE=vulkan[:i]` / `vk[:i]` | make Vulkan the default device (it already is without CUDA / Metal) |
| `BROTENSOR_DISABLE_VULKAN=1` | do not probe Vulkan at all |
| `BROTENSOR_VK_VERBOSE=1` | print why the backend did or did not register |
| `BROTENSOR_VK_ALLOW_CPU=1` | accept CPU implementations (lavapipe) |
| `BROTENSOR_VK_BATCH=n` | commands per submitted batch (default 128) |
| `BROTENSOR_VK_BLOCK_MB=n` | sub-allocation block size (default 256) |
| `BROTENSOR_VK_MAPPED=0` | do not map device memory even when possible |
| `BROTENSOR_VK_ALLOW_OVERSIZE=1` | allow tensors over the per-buffer limit (out of spec) |
| `BROTENSOR_VK_NO_COOPMAT=1` | do not use cooperative matrix (the GEMMs run the SIMT kernel, attention and its backward the FMA kernels) |
| `BROTENSOR_VK_GEMM_NOSCALE=1` | stage a BF16 A as plain FP16 (the pre-scaling behaviour: past ±65504 becomes inf; comparison only) |
| `BROTENSOR_VK_GEMM_CFG=bm,bn,bk,wm,wn` | force one cooperative-matrix tile (one of the four in `gemm.cpp`) |
| `BROTENSOR_VK_FA_CFG=bc,nsg` | force the `fa_cm` tile: bc in {16, 32, 64} keys, nsg in {1, 2, 4} subgroups, 16 nsg <= 2 bc |
| `BROTENSOR_VK_FA_WGS=n` | workgroups `fa_rows` splits keys up to (default 80) |
| `BROTENSOR_VK_FA_PATH=rows\|cm\|dense` | force an attention path where it applies (benchmarking) |
| `BROTENSOR_VK_FA_BWD_PATH=fma\|cm` | force the flash-attention backward's kernel (`cm` where eligible: FP16 / BF16, hd <= 128) |
| `BROTENSOR_VK_FA_BWD_CFG=nsg,bn,areg,...` | force the `fa_bwd_cm` tiles: three (subgroups, streamed rows, A fragments in registers) triples for the statistics, dQ and dK / dV dispatches |
| `BROTENSOR_VK_CONV_CFG=bm,bn,bk,wm,wn` | force the `conv_cm` tile (one of the six in `ops_conv.cpp`) |
| `BROTENSOR_VK_QGEMV=lpr,unr,sg,nr` | force the quantised GEMV's lanes per row group, items in flight, subgroup size and rows per group |
| `BROTENSOR_JIT_VULKAN=eager` / `BROTENSOR_JIT_STRICT=1` | replay traces op by op / make a trace-compiler failure throw (see Trace JIT) |
| `BROTENSOR_VK_DFT_GEMM=1` | run every spectral transform through the FP32 basis GEMM instead of the FP64 direct DFT (benchmarking; less accurate) |

Vulkan validation layers were not installed on the development machine, so
the backend has not been run under them.
