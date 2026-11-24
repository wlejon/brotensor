# Compiler Consolidation Log: Brass Foundation & Radeon 8060S Vulkan Proof Gate

## Device & Environment
- **Device:** AMD Radeon 8060S Graphics (`RADV STRIX_HALO`)
- **Driver:** `radv Mesa 26.2.6-arch1.1`
- **Enabled JIT Capabilities:** `coopmat_f16=1`, `integer_dot_product=1` (`VK_KHR_shader_integer_dot_product`), `memory_model=1`, `subgroup_size_control=1` (`sg=32..64`)

## `brass` Compiler Foundation (`origin/main` at `20edf14`)
- **`5ba4275`**: 128-bit (`i32x4`) and 16-bit (`u16`) shared memory, `ptx_subgroup_barrier`, `ptx_subgroup_id`, and `MslISel` modularization (`msl_builder.hpp`, `msl_isel.cpp`, `msl_isel_intrinsics.cpp`)
- **`16a7f03`**: `ptx_sdot4` (`OpSDotKHR` / `dp4a.s32.s32` / `_brass_sdot4`), `ptx_pack_f16x2`, `ptx_f16_hi_to_f32`, `ptx_ldexp_f32` across SPIR-V, PTX, and MSL
- **`e58ae6e`**: 16×16×16 cooperative matrix (`coop_alloc_{a,b,acc}`, `coop_fill_acc`, `coop_load_{a,b,acc}`, `coop_mma`, `coop_store_acc`, `coop_scale_acc`, `coop_add_acc`) across SPIR-V (`SPV_KHR_cooperative_matrix`), PTX (`wmma`), and MSL (`metal_simdgroup_matrix`)
- **`484175d`**: Complete `MslTarget` intrinsic parity (trig/extra math, `ptx_shfl_down_sync_f32`, `ptx_{i,u}64_to_f32`, shared `i64` load/store, global/shared atomics; explicit diagnostic on unsupported 64-bit atomic add); full `brass_unit_tests` **2031/2031 passed**, pushed to `origin/main`
- **`7f0dd4a`**: Lowered `ptx_f32_to_f16` and `ptx_pack_f16x2` on SPIR-V via `OpFConvert` (`vec2<float16_t>`) + `OpBitcast` (`u32`) for exact IEEE RNE rounding on RADV
- **`20edf14`** *(Apple Silicon agent)*: Fixed MSL `simdgroup_matrix` FP16/fill conversion, `ptx_shared_load/store_*` & narrow global/shared offset handling, and loop-carried `threadgroup` pointer address spaces (`brass_unit_tests` **2033/2033 passed**, `test_msl_target` **9/9 passed**, live M2 Pro FP16 coop-mat GEMM **5465.0 GFLOP/s = `1.30×`** at `4096³` and Q8_0 `sdot4` GEMV **166.3 GB/s GPU-only = `2.00×`** at `12288×4096`)

## 1. Cooperative-Matrix FP16 NT GEMM (`C = A @ B^T`, `128×128×32`)
Kernel built via `brass::codegen::KernelBuilder` -> `SpirvTarget::compile` (`4838` SPIR-V words, `40960` B shared, pinned `subgroup=32`, double-buffered tiles, `SKA=SKB=5` bank-conflict padding, K-rotation) vs `brotensor::linear_forward_batched_ex` (`gemm_cm.comp`):

| Shape (`M×N×K`) | `max_abs` vs ref | `brass` (ms) | `brass` (TF/s) | `gemm_cm` (ms) | `gemm_cm` (TF/s) | Speed Ratio (`brass / gemm_cm`) |
|---|---|---|---|---|---|---|
| `512×512×512` | `0.0039` | `0.023` | `11.70` | `0.019` | `14.39` | `0.813×` |
| `1024×1024×1024` | `0.0000` | `0.080` | `26.94` | `0.093` | `23.20` | **`1.161×`** |
| `2048×2048×2048` | `0.0000` | `0.519` | `33.11` | `0.604` | `28.46` | **`1.164×`** |
| `4096×4096×4096` | `0.0156` | `5.340` | `25.74` | `5.102` | `26.94` | `0.955×` |

## 2. Hardware `sdot4` Q8_0 GEMV (`y = W_q8_0 * x`, Wave64)
Two-stage dispatch (`quant_x_q8` `785` words at `sg=32` + `gemv_q8_sdot4` `1601` words at `sg=64`, aligned `i32` loads across block pairs + `OpSDotKHR` + `spirv_subgroup_add_f32`) timed end-to-end with 512 MiB L2-defeating ring rotation vs `brotensor::linear_forward_q8_0_fp16` (`gemv_q.comp`):

| Shape (`N×K`) | Cosine vs `gemv_q` | `l2_rel_qq` vs CPU Q8×Q8 | `max_abs_qq` | `brass` (ms) | `brass` (GB/s) | `gemv_q` (ms) | `gemv_q` (GB/s) | Speed Ratio (`brass / gemv_q`) |
|---|---|---|---|---|---|---|---|---|
| `4096×4096` | `0.999993` | `5.45e-04` | `0.0020` | `0.078` | `227.7` | `0.083` | `214.4` | **`1.062×`** |
| `11008×4096` | `0.999993` | `5.13e-04` | `0.0039` | `0.206` | `232.5` | `0.214` | `224.2` | **`1.037×`** |

## 3. Standalone Baseline Suite Snapshots
- **`brotensor_bench_gemm_fp16` (Vulkan):**
  - `4096×4096×4096`: NN `32503.4 GFLOP/s` (`4.228 ms`), NT `27685.1 GFLOP/s` (`4.964 ms`), ABT `27534.6 GFLOP/s` (`4.992 ms`)
  - `1024×1024×1024`: NN `22084.8 GFLOP/s` (`0.097 ms`), NT `23269.1 GFLOP/s` (`0.092 ms`), ABT `23216.7 GFLOP/s` (`0.092 ms`)
  - `512×4096×4096`: NN `23401.6 GFLOP/s`, NT `22631.4 GFLOP/s`
- **`brotensor_bench_quant_matmul` (Vulkan, decode `B=1`):**
  - `q8_0`: `o_proj 4096×4096` `159.2 GB/s` (`112.0 us`), `qkv_proj 6144×4096` `177.3 GB/s` (`150.8 us`), `gate/up 12288×4096` `202.3 GB/s` (`264.4 us`), `down_proj 4096×12288` `201.7 GB/s` (`265.2 us`)
  - `q4k`: `4096×4096` `127.2 GB/s`, `12288×4096` `183.0 GB/s`
  - `q6k`: `4096×4096` `145.2 GB/s`, `12288×4096` `193.6 GB/s`
- **Regression checks:** `brotensor_test_vulkan --only=jit`, `--only=quant`, and `--only=gemm` all passed with `OK: 0 failure(s)`.

---

## Wave 0 Radeon 8060S Op & Real-Model Baselines (vs M2 Pro & RTX 4090)

### Op Microbenchmarks (`brotensor_bench_elementwise`, `gemm_fp32`, `wq_linear`)
- **`brotensor_bench_elementwise` (`37/37` ops `ok`):**
  - Unary forward FP32 (`9217×4096`): `silu 1.379 ms (219.0 GB/s)`, `gelu_tanh 1.687 ms (179.0 GB/s)`, `relu 1.367 ms (221.0 GB/s)`, `tanh 1.367 ms (220.9 GB/s)`, `sigmoid 1.366 ms (221.2 GB/s)`
  - Unary forward FP16 (`18433×4096`): `silu 1.488 ms (203.0 GB/s)`, `gelu 1.492 ms (202.5 GB/s)`, `relu 1.459 ms (206.9 GB/s)`, `tanh 1.486 ms (203.2 GB/s)`, `sigmoid 1.480 ms (204.0 GB/s)`
  - Unary forward BF16 (`18433×4096`): `silu 1.821 ms (165.8 GB/s)`, `gelu 1.805 ms (167.3 GB/s)`, `relu 1.503 ms (201.0 GB/s)`, `tanh 1.495 ms (202.1 GB/s)`, `sigmoid 1.484 ms (203.5 GB/s)`
  - Unary backward (`9217×4096` fp32 / `18433×4096` fp16/bf16): `silu_bwd 2.122/2.329/2.291 ms (213.4/194.5/197.7 GB/s)`, `gelu_bwd 2.142/2.306/2.312 ms (211.5/196.4/196.0 GB/s)`
  - Binary/scalar inplace: `add 2.175/2.224/2.237 ms (208.3/203.7/202.5 GB/s)`, `mul 2.416/2.230/2.246 ms (187.5/203.2/201.7 GB/s)`, `scale 1.527/1.500/1.493 ms (197.8/201.3/202.3 GB/s)`
  - Gated/norm: `geglu_forward fp32/fp16/bf16` `0.990/1.095/1.095 ms (228.8/206.9/207.0 GB/s)`, `rms_norm fp32/fp16 1.543/1.669 ms (195.7/181.0 GB/s)`, `layernorm_inf_fp16 1.984 ms (152.2 GB/s)`, `softmax_rows 1.541 ms (196.0 GB/s)`
- **`brotensor_bench_gemm_fp32` (`4096×4096×4096`):** NN `14.515 ms (9468.8 GFLOP/s)`, NT `19.675 ms (6985.4 GFLOP/s)` (TN 4096³ triggered amdgpu CS TDR timeout `VK_ERROR_DEVICE_LOST`)
- **`brotensor_bench_wq_linear` (2048-wide decoder layer):**
  - `M=1`: FP16 `0.514 ms (195.9 GB/s)`, INT8 `0.377 ms (133.6 GB/s)`, FP8 `0.356 ms (141.5 GB/s)`
  - `M=2`: FP16 `0.524 ms (192.0 GB/s)`, INT8 `0.356 ms (141.5 GB/s)`, FP8 `0.372 ms (135.4 GB/s)`
  - `M=8`: FP16 `0.692 ms (145.5 GB/s)`, INT8 `0.403 ms (124.7 GB/s)`, FP8 `0.409 ms (123.0 GB/s)`

### Real-Model Vulkan Baselines (Radeon 8060S vs M2 Pro Metal / RTX 4090 CUDA)

| Repo | Model | Workload | Radeon 8060S (Vulkan) | Apple M2 Pro (Metal) | RTX 4090 (CUDA) |
|---|---|---|---|---|---|
| `brovisionml` | Depth-Anything-V2-Small | 1024×768 (w2, r5) | **23.42 ms** best (mean 24.25 ms) | 991.50 ms best (mean 995.94 ms) | — |
| `brovisionml` | Depth-Anything-V2-Large | 1024×768 (w2, r5) | **128.78 ms** best (mean 132.91 ms) | — | — |
| `brovisionml` | SAM ViT-Base | 1024×768 encode / decode | **126.28 ms** / **9.17 ms** (mean 127.27 / 9.52 ms) | 4730.35 ms / 55.79 ms | — |
| `brovisionml` | SegFormer-B0 | 1024×768 (w2, r5) | **28.52 ms** best (mean 29.69 ms) | — | — |
| `brodiffusion` | SD 1.5 (`lcm-dreamshaper`) | 512×512, 20 steps, CFG 7.5 | **1.250 s** denoise (`0.063 s/step`), **0.155 s** VAE, `1.43 s` total (`2.64/47.23 GiB`) | 39.045 s denoise, 2.788 s VAE | — |
| `brodiffusion` | Sana-600M | 512×512, 20 steps, CFG 4.5 | **1.055 s** denoise (`0.053 s/step`), **0.200 s** VAE, `1.35 s` total (`8.40/47.23 GiB`) | — | — |
| `brodiffusion` | PixArt-Sigma + T5-XXL | 512×512, 10 steps, CFG 4.5 | **0.626 s** T5, **1.551 s** denoise, **0.153 s** VAE, `2.39 s` total (`12.72/47.23 GiB`) | skipped (>11.84 GiB) | — |
| `brosoundml` | Kokoro TTS | 155 phonemes -> 9.47 s audio | **213.7 ms** mean (`44.33× RT`, best `207.3 ms`) | 3194.1 ms (`2.97× RT`) | 136.5 ms (`69.41× RT`) |
| `brosoundml` | Supertonic-3 TTS | 5 steps -> 6.78 s audio (warm) | **276 ms** best (`24.55× RT`; 2nd run 277 ms) | 1624 ms (`4.17× RT`) | — |
| `brolm` | Qwen3-0.6B-Q8_0 | prefill 256 / decode 128 | **12526.5 tok/s** prefill / **193.6 tok/s** (`5.16 ms/tok`) decode | — | 12897.3 / 374.4 tok/s |

---

## Wave 1: Vulkan Elementwise, Cast, Bias & GLU -> Single-Source Brass MIR

Replaced 6 GLSL compute shaders (`unary.comp`, `binary.comp`, `act_bwd.comp`, `bias.comp`, `glu.comp`, `cast.comp` — 24 shader variants removed, `140 -> 116` compiled Vulkan shaders) with single-source `brass::Function` MIR builders under `src/kernels/{mir_math,mir_elementwise}.{h,cpp}` + runtime `SpirvTarget` cache in `src/vulkan/vk_mir_cache.{h,cpp}`.

### `brotensor_bench_elementwise` (Radeon 8060S, Wave 0 vs Wave 1)

All `37/37` ops verified `ok`.

| Category | Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 1 (ms / GB/s) | Speedup |
|---|---|---|---|---|---|---|
| Unary fwd | `silu_forward` | `9217×4096` | `fp32` | `1.379 ms / 219.0` | `1.352 ms / 223.4` | `+2.0%` |
| Unary fwd | `gelu_forward` | `9217×4096` | `fp32` | `1.687 ms / 179.0` | `1.352 ms / 223.4` | **`+24.8%`** |
| Unary fwd | `relu_forward` | `9217×4096` | `fp32` | `1.367 ms / 221.0` | `1.341 ms / 225.1` | `+1.9%` |
| Unary fwd | `tanh_forward` | `9217×4096` | `fp32` | `1.367 ms / 220.9` | `1.335 ms / 226.2` | `+2.4%` |
| Unary fwd | `sigmoid_forward` | `9217×4096` | `fp32` | `1.366 ms / 221.2` | `1.343 ms / 224.9` | `+1.7%` |
| Unary fwd | `silu_forward` | `18433×4096` | `fp16` | `1.488 ms / 203.0` | `1.335 ms / 226.3` | **`+11.5%`** |
| Unary fwd | `gelu_forward` | `18433×4096` | `fp16` | `1.492 ms / 202.5` | `1.333 ms / 226.6` | **`+11.9%`** |
| Unary fwd | `relu_forward` | `18433×4096` | `fp16` | `1.459 ms / 206.9` | `1.355 ms / 222.8` | `+7.7%` |
| Unary fwd | `tanh_forward` | `18433×4096` | `fp16` | `1.486 ms / 203.2` | `1.357 ms / 222.6` | `+9.5%` |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `fp16` | `1.480 ms / 204.0` | `1.338 ms / 225.8` | `+10.7%` |
| Unary fwd | `silu_forward` | `18433×4096` | `bf16` | `1.821 ms / 165.8` | `1.335 ms / 226.3` | **`+36.5%`** |
| Unary fwd | `gelu_forward` | `18433×4096` | `bf16` | `1.805 ms / 167.3` | `1.357 ms / 222.5` | **`+33.0%`** |
| Unary fwd | `relu_forward` | `18433×4096` | `bf16` | `1.503 ms / 201.0` | `1.343 ms / 224.8` | **`+11.8%`** |
| Unary fwd | `tanh_forward` | `18433×4096` | `bf16` | `1.495 ms / 202.1` | `1.333 ms / 226.6` | **`+12.1%`** |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `bf16` | `1.484 ms / 203.5` | `1.351 ms / 223.5` | `+9.8%` |
| Unary bwd | `silu_backward` | `9217×4096` | `fp32` | `2.122 ms / 213.4` | `2.015 ms / 224.8` | `+5.3%` |
| Unary bwd | `gelu_backward` | `9217×4096` | `fp32` | `2.142 ms / 211.5` | `2.021 ms / 224.1` | `+6.0%` |
| Unary bwd | `silu_backward` | `18433×4096` | `fp16` | `2.329 ms / 194.5` | `2.037 ms / 222.4` | **`+14.3%`** |
| Unary bwd | `gelu_backward` | `18433×4096` | `fp16` | `2.306 ms / 196.4` | `2.052 ms / 220.8` | **`+12.4%`** |
| Unary bwd | `silu_backward` | `18433×4096` | `bf16` | `2.291 ms / 197.7` | `2.055 ms / 220.4` | **`+11.5%`** |
| Unary bwd | `gelu_backward` | `18433×4096` | `bf16` | `2.312 ms / 196.0` | `2.019 ms / 224.4` | **`+14.5%`** |
| Binary/scalar | `add_inplace` | `9217×4096` | `fp32` | `2.175 ms / 208.3` | `2.077 ms / 218.1` | `+4.7%` |
| Binary/scalar | `mul_inplace` | `9217×4096` | `fp32` | `2.416 ms / 187.5` | `2.068 ms / 219.1` | **`+16.9%`** |
| Binary/scalar | `scale_inplace` | `9217×4096` | `fp32` | `1.527 ms / 197.8` | `1.338 ms / 225.7` | **`+14.1%`** |
| Binary/scalar | `add_inplace` | `18433×4096` | `fp16` | `2.224 ms / 203.7` | `2.062 ms / 219.7` | `+7.9%` |
| Binary/scalar | `mul_inplace` | `18433×4096` | `fp16` | `2.230 ms / 203.2` | `2.070 ms / 218.8` | `+7.7%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `fp16` | `1.500 ms / 201.3` | `1.329 ms / 227.3` | **`+12.9%`** |
| Binary/scalar | `add_inplace` | `18433×4096` | `bf16` | `2.237 ms / 202.5` | `2.060 ms / 219.9` | `+8.6%` |
| Binary/scalar | `mul_inplace` | `18433×4096` | `bf16` | `2.246 ms / 201.7` | `2.054 ms / 220.6` | `+9.4%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `bf16` | `1.493 ms / 202.3` | `1.371 ms / 220.2` | `+8.8%` |
| Gated | `geglu_forward` | `4609×8192` | `fp32` | `0.990 ms / 228.8` | `1.013 ms / 223.7` | `-2.2%` |
| Gated | `geglu_forward` | `9217×8192` | `fp16` | `1.095 ms / 206.9` | `1.011 ms / 224.0` | `+8.3%` |
| Gated | `geglu_forward` | `9217×8192` | `bf16` | `1.095 ms / 207.0` | `1.006 ms / 225.1` | `+8.7%` |
| Norm *(Wave 2)* | `rms_norm_forward` | `9217×4096` | `fp32` | `1.543 ms / 195.7` | `1.567 ms / 192.8` | *unchanged* |
| Norm *(Wave 2)* | `rms_norm_forward` | `18433×4096` | `fp16` | `1.669 ms / 181.0` | `1.683 ms / 179.5` | *unchanged* |
| Norm *(Wave 2)* | `layernorm_inf_fp16` | `18433×4096` | `fp16` | `1.984 ms / 152.2` | `1.964 ms / 153.8` | *unchanged* |
| Norm *(Wave 2)* | `softmax_rows` | `9217×4096` | `fp32` | `1.541 ms / 196.0` | `1.535 ms / 196.7` | *unchanged* |

### Wave 1 Correctness, Proof Audit & LOC Delta
- **Correctness & Formal Audits:**
  - `brotensor_test_vulkan` `--only=ops`, `--only=gemm`, `--only=train`: passed (`0` failures)
  - `brotensor_test_backward_index_proofs`: ALL PASSED (including `act_bwd_mir_lifted_spirv_bounds_and_safety`)
  - `brotensor_test_backward_shader_audit`: **135/135 shaders passed** (`126847` SPIR-V/PTX insts, `2509` shared accesses proven, `1717` global accesses proven, `0` failures)
  - CPU-vs-Vulkan Parity: `elementwise` (`42/42`), `elementwise_extra` (`15/15`), `activations` (`31/31`), `geglu` (`15/15`), `swiglu` (`10/10`), `log_exp_round` (`14/14`), `add_channel_bias` (`14/14`) — **all passed**
- **LOC Delta (Vulkan Wave 1):**
  - Deleted 6 GLSL shaders / 24 variants (`unary.comp`, `binary.comp`, `act_bwd.comp`, `bias.comp`, `glu.comp`, `cast.comp`): `src/vulkan/shaders` `8,235 -> 7,830` (**`-405` lines**)
  - Added backend-neutral shared MIR builders: `src/kernels/{mir_math,mir_elementwise}.{h,cpp}` (**`+515` lines**, reused across Metal + CUDA in Wave 1) + Vulkan runtime MIR cache `src/vulkan/vk_mir_cache.{h,cpp}` (**`+219` lines**)
  - Pending Wave 1 cross-backend deletions once Metal & CUDA wire into `src/kernels/mir_elementwise`: `~1,934` Metal lines (`elementwise.mm`, `elementwise_msl.h`, `swiglu.mm`) and `~1,695` CUDA lines (`elementwise.cu`, `elementwise_activations.cu`, `swiglu.cu`)
