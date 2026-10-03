# Removing the HIP backend: inventory

A write-up, not a plan of record (§2, §4 and §5 updated for chunk 9): what deleting brotensor's HIP backend
(`BROTENSOR_WITH_HIP`, `src/hip/`) would remove, what the siblings would lose,
what stays, and what it costs. Line counts are `wc -l` on the tree as of
chunk 8 (2026-10-03). Vulkan is already the default device in a HIP + Vulkan
build and is faster on every sibling model measured except the T5-XXL encode
(docs/vulkan-perf.md).

## 1. What would be deleted

### brotensor

| Item | Files | Lines |
|---|---:|---:|
| `src/hip/*.hip` (native HIP kernels + runtime, graph, register) | 35 | 16,549 |
| `src/hip/detail/*` (per-family internals, `hip_check.h`, `cuda_compat.h`, `capture_arena.h`, `hipblas_handle.h`, `stubs.hip`) | 12 | 978 |
| `src/hip/compat/*.h` (`cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`, `math_constants.h`, `mma.h` → rocWMMA) | 5 | 92 |
| **`src/hip/` total** | **52** | **17,619** |
| `cmake/BrotensorHip.cmake` (ROCm root, arch check, `brotensor_hip_detect_gpus`) | 1 | 128 |
| `CMakeLists.txt`: option + exclusivity check (L124-132), ROCm / hipBLAS / rocWMMA lookup (L152-174), `brotensor_hip` target with the 90-file `.cu`-as-HIP list (L722-877) | — | ~185 |
| `tests/test_hip.cpp` (`brotensor_test_hip`) | 1 | 998 |
| `tests/test_cuda_rocm_parity.cpp` (golden-file CUDA vs ROCm; skipped, no golden on disk) | 1 | 326 |
| `tests/test_vulkan_hip_parity.{cpp,h}` (chunk 8 Vulkan-vs-HIP parity) | 2 | 543 |
| `tests/CMakeLists.txt`: HIP test block, the `BROTENSOR_PREFER_HIP=1` stamping loop, HIP arms of the GPU gates | — | ~35 |
| HIP comparison columns in `tests/test_vulkan_bench_{attention,quant,conv}.cpp` (runtime `hip_ok()` arms) | 3 | ~60 |

HIP-specific lines inside shared core files (they shrink, they do not go):

| File | HIP refs | What |
|---|---:|---|
| `src/init.cpp` | 21 | HIP probe / registration, `prefer_hip()`, `hip`/`rocm` env parsing, HIP in `available_devices()` |
| `src/dispatch.cpp` | 16 | `hipSetDevice` on dispatch, `g_hip_device_count`, the `Device::CUDA` → HIP alias arm |
| `include/brotensor/detail/dispatch.h` | 17 | alias rule comment, `prefer_hip()`, `set_hip_device_count` |
| `include/brotensor/cuda_graph.h`, `detail/graph_backend.h`, `src/graph.cpp` | 21 | the hipGraph backend of the device-neutral capture |
| `include/brotensor/runtime.h` | 7 | `hip_device_count()`, alias doc |
| `include/brotensor/tensor.h`, `src/tensor.cpp` | 9 | `DeviceType::HIP`, `Device::hip()`, `device_name` arm (keep the enum value one release as a parse-only alias, or drop it: every sibling names it, see §2) |
| `src/jit/trace_compiler_cpu.cpp`, `trace_ops.cpp`, `trace_eager.cpp` | 13 | `is_hip` flags shared with Vulkan's eager replay (rename only) |
| `src/vulkan/ops_norm.cpp`, `ops_linear.cpp`, `detail/stream.h`, `cmake/BrotensorVulkan.cmake` | 10 | comments comparing to HIP |

### Siblings

| Repo | File | HIP lines | What goes |
|---|---|---:|---|
| brolm | `CMakeLists.txt` | 1 | `BROTENSOR_WITH_HIP` forward option |
| brolm | `src/api/host_lm_internal.h` | 4 | `"hip"`/`"rocm"` device parsing, `"HIP"` name |
| brolm | `src/laya_scheduler.cpp` | 1 | `hip_device_count()` replica count |
| brolm | `src/t5.cpp` | 1 | BF16 recast on `CUDA \|\| HIP` (HIP arm) |
| brolm | `tests/test_bioclip_hip.cpp` + 1 CMake line | 119 | whole test (HIP-or-CPU BioCLIP integration; becomes a default-GPU test) |
| brosoundml | `CMakeLists.txt` | 1 | forward option |
| brosoundml | `tools/tool_device.h`, `src/api/host_soundml_internal.h`, `src/api/native_soundml_tts.cpp` | 8 | device parsing / names |
| brosoundml | `tests/test_whisper.cpp`, `tests/test_device.h`, `tests/train_pos_tagger.cpp` | 7 | GPU pick lists |
| brogameagent | `CMakeLists.txt` | 11 | `BROGAMEAGENT_WITH_HIP` option, exclusivity, forwarding, GPU tool gate |
| brogameagent | `src/api/host_ai_nn_support.cpp`, `tests/CMakeLists.txt`, `tests/gpu/parity_helpers.h` | 4 | device parsing, test gate |
| brodiffusion | `CMakeLists.txt` | ~38 | option, `enable_language(HIP)` block (L62-80), `.cu`-as-HIP block (L326-343) |
| brodiffusion | `src/fused.cpp` | 8 | `\|\| BROTENSOR_HAS_HIP` in 6 gates (the CUDA arm stays) |
| brodiffusion | `include/brodiffusion/detail/cuda_check.cuh` | ~10 | HIP error-check branch |
| brodiffusion | `src/triposplat/flow_model.cpp`, `src/api/native_diffusion_triposplat.cpp`, `tests/test_device.h` | 6 | gates, device parsing, GPU pick |
| brovisionml | `CMakeLists.txt` | ~30 | option, HIP language block (L50-67), `.cu`-as-HIP block (L256-265) |
| brovisionml | `src/dsine_ops.cpp`, `src/dpt_preprocess.cpp`, `src/dsine_ops.cu`, `src/dsine_ops_cuda.h` | 12 | HIP arms of the CUDA gates |
| brovisionml | `src/api/native_vision_ops.cpp`, `tools/tool_device.h`, `tests/test_device.h` | 10 | device parsing / names / GPU pick |
| bro | `CMakeLists.txt` | ~30 | `BRO_WITH_TENSOR_HIP` option, ROCm auto-detect (L257-273), HIP arm of the forwarding block (L324-349), feature summary |
| bro | `src/bronze_host/host_gpu.cpp`, `native_motion.cpp` | 8 | `'hip'`/`'rocm'` parsing, `"hip"` device name |
| bro | `tests/gpu/test_gpu_binding.js` | ~8 | the HIP + Vulkan default-device check |
| bro | docs (`build-options.md`, `BUILDING.md`, `multi-repo-workflow.md`, `docs/*-api.js`) | ~40 | HIP rows / strings |

No sibling ships a HIP-only source file: every sibling HIP build compiles its
CUDA `.cu` files through `src/hip/compat`. Sibling HIP-specific code totals
~350 lines, almost all CMake and device-name plumbing.

### CMake options removed

| Option | Repo |
|---|---|
| `BROTENSOR_WITH_HIP` | brotensor (plus the forwarding copies in brolm, brosoundml, brodiffusion, brovisionml) |
| `BROGAMEAGENT_WITH_HIP` | brogameagent |
| `BRO_WITH_TENSOR_HIP` + ROCm auto-detect | bro |
| `BROTENSOR_PREFER_HIP` (env), `hip`/`rocm` in `BROTENSOR_DEFAULT_DEVICE` | brotensor runtime |
| Build dependencies dropped | ROCm (hip, hipBLAS, rocWMMA headers), `rocm_agent_enumerator`; the `build_hip` dirs |

## 2. What the siblings would lose

**HIP-only kernels with no Vulkan twin: none on any inference path.** Every
inference op the siblings call is on Vulkan (docs/vulkan-coverage.md), and the
sibling GPU kernels all have Vulkan twins since chunk 7a:

| Sibling kernel (CUDA, built as HIP) | Lines | Vulkan twin |
|---|---:|---|
| brodiffusion `fused_resblock.cu` | 479 | brotensor `resblock_forward` / fused ops (`src/fused.cpp`) |
| brodiffusion `fused_transformer.cu` | 453 | linear + `geglu_exact_forward`, `add_row_bias_inplace` |
| brodiffusion `triposplat/flow_rope.cu` | 108 | `src/vulkan/flow_rope.comp` (37) |
| brovisionml `dsine_ops.cu` | 197 | `src/vulkan/dsine_ops.comp` (225) + `vulkan_ops.cpp` (153) |
| brovisionml `dpt_preprocess_gpu.cu` | 151 | `src/vulkan/dpt_preprocess.comp` (70) |

**The HIP WMMA INT8 path.** `conv2d_int8w_wmma.cu` and `linear_int8w_wmma.cu`
(rocWMMA through `compat/mma.h`) back the 9 INT8 W8A16 ops on HIP. Vulkan
implements all 9 (cooperative-matrix GEMM with INT8 decode in the A/B staging,
docs/vulkan.md "Quantised weights"), so no op is lost; the HIP kernels stay as
CUDA kernels. `brotensor_test_int8_conv_wmma` / `_int8_linear_wmma` test the
public ops and run on Vulkan too since chunk 9 (`*_vulkan`).

**Performance.** One model is slower on Vulkan: the T5-XXL encode (1.9 s vs
0.93 s), because brolm's T5 recasts to BF16 only on CUDA / HIP and runs FP32
activations through the SIMT GEMM on Vulkan (the matrix-core GEMM stages BF16 as
FP16, which overflows T5's activations). Removing HIP makes that 2x the only
choice on AMD unless Vulkan gets a range-safe BF16 GEMM.

**Training backwards: none missing since chunk 9.** Chunk 8 filled 7 slots
(the attention / MHA / self / cross attention backwards, training BatchNorm,
BCE; ~560 lines); chunk 9 filled the 17 that still worked only on HIP. The
op table is 266 of 270 on Vulkan; the 4 null slots are 2 host-only ops
(`mse_scalar`, `softmax_xent_segment`, always run on the CPU table) and
`filtered_lrelu_forward` / `_backward` (composite by design), none reachable
from a Vulkan tensor.

| Slots (chunk 9) | Who calls it | Vulkan (host + GLSL, lines) |
|---|---|---:|
| `flash_attention_backward`, `_varlen_backward`, `_qkvo_backward`, `_packed_qkv_backward` | brolm LayaGrad (`packed_qkv`), training | `ops_fa_bwd.cpp` 325 + `fa_bwd.comp` 411 (one FA-2 kernel pair, four entry points, FP32 / FP16 / BF16) |
| `lstm_forward_train`, `lstm_backward` | none today | `ops_lstm.cpp` 187 + `lstm.comp` 62 (GEMMs around a pointwise cell) |
| `resblock_backward` | training | +120 in `ops_diffusion.cpp` (composition) |
| `group_norm_backward` | brosoundml Kokoro decoder backward | +60 in `ops_gnorm.cpp`, +80 in `gnorm.comp` (no atomics) |
| `conv_transpose2d_backward_input`, `_backward_weight` | training | +110 in `ops_conv.cpp` (conv2d / conv2d_backward_weight) |
| `upsample_bilinear_2x_backward`, `interp2d_backward`, `pad2d_backward`, `adaptive_avg_pool2d_backward`, `max_pool2d_backward` | training | `ops_spatial_bwd.cpp` 140 + `resample_bwd.comp` 158 (gathers) |
| `embedding_lookup_backward`, `scatter_rows_add` | brolm LayaGrad soft-row gradient | `ops_scatter.cpp` 108 + `scatter_add.comp` 91 (sort + segmented sum, deterministic) |
| **Total (17 slots)** | | **~1,850** (estimate was ~2,270) + ~560 lines of `test_vulkan_train*.cpp` |

Design, errors against the CPU and the flash backward's speed (3.2-5.8x HIP's)
are in docs/vulkan-training.md. One behavioural difference from HIP worth
knowing: the Vulkan scatter-adds, GroupNorm backward and pooling backwards
are deterministic (HIP's use float atomics), and FP32 scatter-adds / max-pool
backward match the CPU bit for bit.

## 3. What stays

CUDA stays whole: HIP was compiled from it, not the other way round.
`BROTENSOR_HIP_CUDA_KERNELS` lists 90 of the 94 `src/cuda/*.cu` files (38,170
of 38,833 lines); all 94 remain for the CUDA backend.

Where removal simplifies CUDA code (HIP arms inside `.cu` files):

| File | Block | Lines | After removal |
|---|---|---:|---|
| `src/cuda/gemm_mma.cu` | `#ifndef __HIP__` around the whole mma.sync GEMM | 465 | guard deleted; file unchanged otherwise |
| `src/cuda/flash_attention_packed.cu` | `#ifndef __HIP__` (L76-325, L395-404) | 260 | guards deleted |
| `src/cuda/flash_attention_fused.cu` | `#if __HIP__` 64 KB-LDS tile table vs the CUDA one (L485-565) | 81 | HIP arm (~35) deleted |
| `src/cuda/self_attention_bias.cu` | hipBLAS include + check macro, hipBLAS batched GEMM arm, grid.z cap (L25-35, L382-443) | 68 | HIP arms deleted |
| `src/cuda/fp16_matmul.cu` | `#ifdef __HIP__` (L43-51) | 9 | deleted |

Files that exist only because `.cu` is compiled as HIP (all under `src/hip/`,
deleted with it): `compat/*.h` (92 lines: the `cuda_runtime.h` / `cuda_fp16.h`
/ `mma.h` shims) and `detail/cuda_compat.h` (239). brodiffusion and brovisionml
add `src/hip/compat` to their include path only for this. The CUDA-side
structure that HIP shaped and that would stay as is: the per-family
`fill_cuda_vtable_*` functions (`src/hip/register.hip` calls 45 of them, then
lets 14 `fill_hip_vtable_*` override GEMM, softmax, RoPE, MHA, attention,
conv, norms, audio and quant). Nothing in `src/cuda/` exists solely for HIP.

## 4. Test coverage that changes

| Suite | Today | After removal |
|---|---|---|
| brotensor `build_hip` | 184 ctest entries | gone |
| brotensor `build_vk` | 312 entries since chunk 9: the generic CPU↔GPU block runs on **HIP** (`BROTENSOR_PREFER_HIP=1`) and again on **Vulkan** as 119 `<name>_vulkan` registrations (`BROTENSOR_TEST_GPU=vulkan`, `BROTENSOR_DEFAULT_DEVICE=vulkan`), all passing, plus the 4 earlier Vulkan entries | the HIP half goes; the block is gated on `CUDA OR METAL OR HIP OR VULKAN` and every suite picks its GPU through `tests/gpu_select.h`, so a **Vulkan-only build** builds and runs the whole block on Vulkan (189 entries, all pass). Not re-run on Vulkan: the four suites with their own Vulkan runs (`cuda_graph`, `graph_capture_alloc`, `ops_fused`, `jit_trace_dtypes`) and the CUDA-only ones (streams, linear_rounding, multigpu, cuda_jit*) |
| `brotensor_test_hip` | HIP runtime / alias / smoke | gone |
| `brotensor_test_cuda_rocm_parity` | skipped (no golden) | gone; CUDA golden generation can stay for a future CUDA-vs-Vulkan check |
| `test_vulkan_hip_parity` (chunk 8) | Vulkan vs HIP per op, same process | gone; the CPU oracle in `brotensor_test_vulkan` remains the only reference |
| `brotensor_test_int8_{conv,linear}_wmma` | HIP rocWMMA kernels, and Vulkan's INT8 paths (`_vulkan`) | CUDA + Vulkan |
| `--bench-*` HIP columns | Vulkan vs HIP | Vulkan only |
| brogameagent / brosoundml under `BROTENSOR_PREFER_HIP=1` | all 33 / 48 pass | the 5 / 2 tests that needed HIP pass on Vulkan after chunk 8; no second backend to cross-check |
| brolm LayaGrad (`brolm_test_laya_grad`) | Vulkan since chunk 9 (packed-QKV backward + scatter-add), HIP under `BROTENSOR_PREFER_HIP=1` | Vulkan only |
| brolm `brolm_test_bioclip_hip` | HIP or CPU | retarget to the default GPU |
| bro `tests/gpu/test_gpu_binding.js` | HIP + Vulkan default check | simplified |

## 5. Recommendation

Steps (1) and (2) of the sequence below are done (chunk 9): every op-table
slot a sibling can reach runs on Vulkan, training included, and the generic
CPU↔GPU suite runs on Vulkan (and is the whole GPU suite of a Vulkan-only
build). What remains before deleting HIP: (3) a range-safe BF16 (or
FP32-accumulate split) GEMM so the T5-XXL encode stops being 2x slower; the
flash-attention backward on cooperative-matrix fragments if training
throughput matters (2.5 TF/s now, against the forward's 15); then (4) delete
`src/hip/` and the ~185 + ~350 lines of CMake / plumbing listed above. One
HIP-side wart found while re-gating, moot after removal: in a HIP + Vulkan
build where Vulkan is the alias target, a HIP op called explicitly allocates
its `Device::CUDA` temporaries on Vulkan (e.g. `modulated_conv2d_backward`'s
scratch), so HIP ops are only reliable there under `BROTENSOR_PREFER_HIP=1`.

The original recommendation:

Remove HIP after the training follow-up, not before: Vulkan already wins
every inference workload but one and has twins for every sibling kernel, so the
only users who would lose a GPU path are training ones (flash-attention / LSTM /
pooling / scatter backwards), and the generic 125-test parity suite would drop
from AMD coverage until it is re-gated on Vulkan. Sequence: (1) the ~3,000-line
training surface on Vulkan; (2) re-gate the generic GPU suite on Vulkan and fix
what it finds; (3) a range-safe BF16 (or FP32-accumulate split) GEMM so the
T5-XXL encode stops being 2x slower; (4) delete `src/hip/` and the
~185 + ~350 lines of CMake / plumbing listed above (−19.9k lines in brotensor
including tests, −0.4k across the siblings and bro), keeping `DeviceType::HIP`
as a parse alias for one release.
