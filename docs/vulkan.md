# Vulkan backend

`BROTENSOR_WITH_VULKAN=ON` builds a Vulkan 1.2+ compute backend whose kernels compile from **single-source Brass MIR** (`src/kernels/mir_*.{h,cpp}`) through `brass::target::SpirvTarget` (`0` GLSL shaders; zero `glslc` / shaderc dependency). It runs on any GPU with a stock Vulkan 1.2+ driver and serves as brotensor's AMD GPU path (verified on Radeon 8060S / Mesa RADV).

## Architecture & Single-Source Brass MIR

Every Vulkan compute op — inference and training — is lowered on demand from a `brass::Function` MIR builder in `src/kernels/` into SPIR-V 1.5 with `PhysicalStorageBuffer64` buffer device addresses (BDA) and cached at two levels in `src/vulkan/vk_mir_cache.{h,cpp}`:

1. **In-memory pipeline cache (`vk_mir_cache`)**: keyed by `(MirKind, op, dtypes, vector_lanes, flags, aux)`.
2. **Persistent SPIR-V disk cache (`jit::KernelDiskCache`)**: stores compiled SPIR-V binaries keyed by full `MirKey`, workgroup size, and Brass's binary build fingerprint so warm runs start with zero shader recompilation.

Static formal verification (`brotensor_test_backward_index_proofs` and `brotensor_test_backward_shader_audit`, backed by `brass::verify::prove_spirv_shader_safety` and `verify_spirv_memory_safety`) audits generated SPIR-V modules for structural validity, shared-memory bounds, workgroup barrier uniformity, subgroup collective uniformity, and out-of-bounds hazards.

**Default device & aliasing.** Without a CUDA or Metal backend, Vulkan is the default device (`default_device()`). Otherwise select it with `set_default_device(Device::vulkan(i))`, a `DeviceScope`, or `BROTENSOR_DEFAULT_DEVICE=vulkan` (or `vk`, `vulkan:1`). When no CUDA backend is registered, `Device::CUDA` aliases to Metal (if available) else Vulkan (`detail::resolve_device_alias`).

## Op Coverage

Vulkan populates **270 of 274** vtable slots in `OpsVTable`. The **4 null slots** are intentionally omitted by design and never fail a public call:

| Slot | Why null on Vulkan |
|---|---|
| `mse_scalar` | Host-only op (no `Tensor` operand); `ops.cpp` always dispatches to CPU |
| `softmax_xent_segment` | Host-only op (raw host pointers); `ops.cpp` always dispatches to CPU |
| `filtered_lrelu_forward` | Public entry falls back to composite (`bias_act` + `upfirdn2d`) on Vulkan |
| `filtered_lrelu_backward` | Public entry falls back to composite (`bias_act_bwd` + `upfirdn2d_bwd`) on Vulkan |

All inference, training (including dense/MHA/cross/flash attention backward, GroupNorm/BatchNorm/ResBlock backward, conv/resample/padding/pooling backward, and full-BPTT LSTM), audio/spectral (incl. FP64 direct DFT via `mir_dft64`), vision, StyleGAN3, INT8-W8A16, INT8/FP8 row-quantized, and GGUF (`Q4_K`, `Q6_K`, `Q8_0`) ops run natively on Vulkan.

## Matrix-Core & Cooperative Matrix Kernels

On hardware exposing `VK_KHR_cooperative_matrix` (`16×16×16` FP16 → FP32 at pinned subgroup size 32, `PhysInfo::coopmat_f16`), four Brass MIR cooperative-matrix engines handle compute-heavy workloads (backed by `coop_*` intrinsics in `src/kernels/mir_coop_common.{h,cpp}`):

- **GEMM (`mir_gemm_cm.cpp`, `MirKind::GemmCm`)**: double-buffered shared tiles (`128×128`, `64×64`, `64×128`, `128×64`), FP16/BF16/INT8/FP8/GGUF (`Q8_0`/`Q4_K`/`Q6_K`) `B`-tile decode, per-row power-of-two scaling for wide BF16/FP32 `A` operands ([vulkan-bf16.md](vulkan-bf16.md)), and fused bias + activations + `ACCUM` epilogues.
- **Convolution (`mir_conv_cm.cpp`, `MirKind::ConvCm`)**: `128×128` implicit-GEMM 2D convolution (`FP16`/`BF16`, grouped/strided/dilated, fused bias + activation). Small/depthwise/3D/transpose convolutions route through `mir_conv_simt` or `mir_conv_direct`.
- **FlashAttention Forward (`mir_fa_cm.cpp`, `MirKind::FaCm`)**: `64×64` tiled attention (`FP16`/`BF16`, causal/varlen/windowed/GQA/softcap/mask/split-K); FP32, wide heads, and decode route through `mir_fa_rows`.
- **FlashAttention Backward (`mir_fa_bwd_cm.cpp`, `MirKind::FaBwdCm`)**: deterministic 3-pass recompute backward (`PASS 0` logsumexp/D statistics, `PASS 1` `dK`/`dV`, `PASS 2` `dQ`) with uniform shared control broadcast and in-register online scaling (`hd <= 128`); FP32 and `hd > 128` route through `mir_fa_bwd`.

### Radeon 8060S Benchmark Highlights (Brass MIR, Release)

- **FP16 GEMM (`brotensor_bench_gemm_fp16 --device=vulkan`, `4096³`):**
  - `NN`: **`34.94 TF/s`** (up from `32.68 TF/s` GLSL baseline, +6.9%)
  - `NT`: **`30.79 TF/s`** (up from `26.77 TF/s` GLSL baseline, +15.0%)
  - `ABT`: **`31.04 TF/s`** (up from `27.15 TF/s` GLSL baseline, +14.3%)
- **Decode Quantized Linear (`brotensor_bench_wq_linear`, `2048`-wide layer, `M=1`):**
  - FP16 `190.4 GB/s`, INT8 **`142.3 GB/s`** (+6.5% vs Wave 0), FP8 **`143.7 GB/s`** (+33.3% over pre-LUT).
- **GGUF Decode GEMM (`brotensor_bench_quant_matmul --device=vulkan`, `B=1`):**
  - `q4k` `174.6 GB/s`, `q6k` `189.7 GB/s`, `q8_0` `196.7 GB/s`.

## Trace JIT

`begin_trace()` / `end_trace()` on Vulkan compiles elementwise and row-norm DAGs (`src/jit/trace_compiler_vulkan.cpp`, `src/jit/spirv_emit.cpp`) to single fused SPIR-V kernels dispatched on the Vulkan stream (`VulkanGraph` capture compatible). Unsupported DAGs replay eagerly (`src/jit/trace_eager.cpp`).

## Dtype Policy

- **FP32**: native.
- **FP16**: native 16-bit storage, FP32 accumulation and RNE store.
- **BF16**: stored as `uint16` bits and converted in registers (`RNE`, quiet NaN preservation); matrix-core GEMMs stage BF16 `A` operands with exact power-of-two per-row scaling into FP16 fragments ([vulkan-bf16.md](vulkan-bf16.md)), while `compute_dtype()` converts BF16 model weights to FP16 on upload.
- **INT8 / FP8 / INT32 / GGUF (`Q8_0`, `Q4_K`, `Q6_K`)**: storage carriers decoded in registers/shared memory by quantized kernels (`mir_gemv_q`, `mir_gemm_cm`, `mir_quant_helpers`).
- **FP64**: used when `PhysInfo::shader_float64` is true (e.g., `mir_dft64` direct DFT and `brovisionml` DSINE geometry via `brotensor::interop::device_has_float64`).

## Build & Custom Kernels

```sh
cmake -B build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DBROTENSOR_WITH_VULKAN=ON
cmake --build build_vk -j4
./build_vk/tests/brotensor_test_vulkan
```

The Vulkan loader (`libvulkan`) is `dlopen`ed at runtime, so Vulkan-enabled binaries run (and fall back cleanly) even on machines without a Vulkan ICD. Downstream pure-C++20 libraries compile their own Brass MIR kernels to SPIR-V and dispatch them via `<brotensor/kernel_interop.h>` (`dispatch_spirv`), without linking Vulkan headers or `glslc`; advanced batching and `VulkanGraph` helpers live in `<brotensor/vulkan.h>`.

## File Layout

```text
include/brotensor/kernel_interop.h  Pure-C++20 dispatch_spirv / TensorBufferView / device_has_float64
include/brotensor/vulkan.h          flush, Event, VulkanGraph(Capture), stats, device_info
src/kernels/mir_*.{h,cpp}           71 single-source Brass MIR files (shared across Vulkan/CUDA/Metal)
src/vulkan/
  vk_mir_cache.{h,cpp}              SpirvTarget compiler cache + KernelDiskCache integration
  kernel_interop_vk.cpp             dispatch_spirv & shader_float64 interop hooks
  detail/                           dlopen loader, Instance/DeviceCtx, Allocator, Stream, Pipelines, SPIR-V reflect
  shaders/op_codes.h                Canonical C/C++ op-code definitions shared across backends (0 GLSL shaders)
  ops_*.cpp, gemm.cpp, register.cpp Op dispatchers & vtable registration
```

## Extending via Brass MIR

1. Author the compute kernel builder in `src/kernels/mir_<topic>.{h,cpp}` using `brass::codegen::KernelBuilder` (keep shared-memory indices bounded and barriers workgroup-uniform so `backward_shader_audit` and `backward_index_proofs` prove `0` hazards).
2. Add a `MirKind` entry in `src/vulkan/vk_mir_cache.{h,cpp}` (and share the same builder with `src/cuda/cu_mir_cache.*` and `src/metal/mtl_mir_cache.*`).
3. Dispatch from `src/vulkan/ops_<family>.cpp` and register an audit entry in `src/backward_shader_audit.cpp` plus CPU↔GPU parity tests in `tests/`.

## Environment Variables

| Variable | Effect |
|---|---|
| `BROTENSOR_DEFAULT_DEVICE=vulkan[:i]` / `vk[:i]` | Select Vulkan device `i` as default |
| `BROTENSOR_DISABLE_VULKAN=1` | Disable Vulkan probing |
| `BROTENSOR_VK_VERBOSE=1` | Log Vulkan device selection details |
| `BROTENSOR_VK_ALLOW_CPU=1` | Allow software Vulkan ICDs (e.g. lavapipe) |
| `BROTENSOR_VK_MAPPED=0` | Disable host-visible device memory mapping |
| `BROTENSOR_VK_ALLOW_OVERSIZE=1` | Allow tensors larger than `maxStorageBufferRange` |
| `BROTENSOR_VK_NO_COOPMAT=1` | Force SIMT GEMM / FMA FlashAttention paths (skip `VK_KHR_cooperative_matrix`) |
| `BROTENSOR_JIT_VULKAN=eager` / `BROTENSOR_JIT_STRICT=1` | Replay trace JIT op-by-op / throw on trace compile fallback |
