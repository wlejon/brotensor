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
  - Wave 1 Metal consolidation completed below (`-2,016` net Metal LOC); pending Wave 1 CUDA deletion: `~1,695` CUDA lines (`elementwise.cu`, `elementwise_activations.cu`, `swiglu.cu`)

---

## Wave 1 (Metal / Apple M2 Pro): Elementwise, Cast, Bias & GLU -> Single-Source Brass MIR

Added Metal runtime MIR cache (`src/metal/mtl_mir_cache.{h,mm}`) compiling `brotensor::kernels::{build_unary_mir, build_binary_mir, build_act_bwd_mir, build_bias_mir, build_cast_mir, build_glu_mir}` via `brass::target::MslTarget::compile` (`kEwBlock = 256`, 128-bit `lanes = 8` for 16B-aligned FP16/BF16 and `lanes = 4` for 16B-aligned FP32, 64-bit `lanes = 4` for 8B-aligned 16-bit, else scalar `1`), wired all Metal Wave 1 elementwise / activation / bias / cast / GLU ops (`relu`, `tanh`, `sigmoid`, `silu`, `gelu`, `gelu_exact`, `quick_gelu`, `exp`, `log`, `sin`, `cos`, `rsqrt`, `round`, `elu`, `leaky_relu`, `add_scalar`, `scale`, `clamp`, `add`, `mul`, `axpby`, `add_row_bias`, `add_channel_bias`, `cast`, `swiglu`, `geglu`, `geglu_exact`, and batched `relu`/`tanh`/`add` delegates) through it, and deleted the duplicated hand-written MSL kernels.

### `brotensor_bench_elementwise` (Apple M2 Pro Metal, Wave 0 vs Wave 1)

All `37/37` ops verified `ok` sequentially on an idle M2 Pro (19-core GPU, 16 GiB unified RAM, ~200 GB/s theoretical bus).

| Category | Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 1 (ms / GB/s) | Speedup |
|---|---|---|---|---|---|---|
| Unary fwd | `silu_forward` | `9217×4096` | `fp32` | `1.743 ms / 173.3` | `1.745 ms / 173.1` | `-0.1%` |
| Unary fwd | `gelu_forward` | `9217×4096` | `fp32` | `1.743 ms / 173.2` | `1.743 ms / 173.3` | `+0.1%` |
| Unary fwd | `relu_forward` | `9217×4096` | `fp32` | `1.742 ms / 173.4` | `1.739 ms / 173.6` | `+0.1%` |
| Unary fwd | `tanh_forward` | `9217×4096` | `fp32` | `1.741 ms / 173.5` | `1.741 ms / 173.4` | `-0.1%` |
| Unary fwd | `sigmoid_forward` | `9217×4096` | `fp32` | `1.738 ms / 173.8` | `1.733 ms / 174.3` | `+0.3%` |
| Unary fwd | `silu_forward` | `18433×4096` | `fp16` | `2.014 ms / 150.0` | `1.745 ms / 173.1` | **`+15.4%`** |
| Unary fwd | `gelu_forward` | `18433×4096` | `fp16` | `2.017 ms / 149.7` | `1.735 ms / 174.0` | **`+16.3%`** |
| Unary fwd | `relu_forward` | `18433×4096` | `fp16` | `2.010 ms / 150.3` | `1.743 ms / 173.3` | **`+15.3%`** |
| Unary fwd | `tanh_forward` | `18433×4096` | `fp16` | `1.998 ms / 151.1` | `1.736 ms / 174.0` | **`+15.1%`** |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `fp16` | `2.011 ms / 150.2` | `1.739 ms / 173.7` | **`+15.6%`** |
| Unary fwd | `silu_forward` | `18433×4096` | `bf16` | `2.016 ms / 149.8` | `1.731 ms / 174.5` | **`+16.5%`** |
| Unary fwd | `gelu_forward` | `18433×4096` | `bf16` | `2.013 ms / 150.0` | `1.736 ms / 174.0` | **`+16.0%`** |
| Unary fwd | `relu_forward` | `18433×4096` | `bf16` | `2.005 ms / 150.7` | `1.729 ms / 174.6` | **`+16.0%`** |
| Unary fwd | `tanh_forward` | `18433×4096` | `bf16` | `2.013 ms / 150.0` | `1.724 ms / 175.2` | **`+16.8%`** |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `bf16` | `2.007 ms / 150.4` | `1.727 ms / 174.8` | **`+16.2%`** |
| Unary bwd | `silu_backward` | `9217×4096` | `fp32` | `2.525 ms / 179.4` | `2.534 ms / 178.8` | `-0.4%` |
| Unary bwd | `gelu_backward` | `9217×4096` | `fp32` | `2.528 ms / 179.2` | `2.527 ms / 179.3` | `+0.0%` |
| Unary bwd | `silu_backward` | `18433×4096` | `fp16` | `2.519 ms / 179.9` | `2.530 ms / 179.0` | `-0.4%` |
| Unary bwd | `gelu_backward` | `18433×4096` | `fp16` | `2.543 ms / 178.2` | `2.523 ms / 179.5` | `+0.8%` |
| Unary bwd | `silu_backward` | `18433×4096` | `bf16` | `2.534 ms / 178.8` | `2.570 ms / 176.2` | `-1.4%` |
| Unary bwd | `gelu_backward` | `18433×4096` | `bf16` | `3.836 ms / 118.1` | `2.513 ms / 180.3` | **`+52.6%`** |
| Binary/scalar | `add_inplace` | `9217×4096` | `fp32` | `2.596 ms / 174.5` | `2.580 ms / 175.6` | `+0.6%` |
| Binary/scalar | `mul_inplace` | `9217×4096` | `fp32` | `2.575 ms / 175.9` | `2.588 ms / 175.0` | `-0.5%` |
| Binary/scalar | `scale_inplace` | `9217×4096` | `fp32` | `1.786 ms / 169.1` | `1.798 ms / 167.9` | `-0.7%` |
| Binary/scalar | `add_inplace` | `18433×4096` | `fp16` | `2.571 ms / 176.2` | `2.616 ms / 173.2` | `-1.7%` |
| Binary/scalar | `mul_inplace` | `18433×4096` | `fp16` | `2.538 ms / 178.5` | `2.596 ms / 174.5` | `-2.2%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `fp16` | `2.058 ms / 146.8` | `1.776 ms / 170.0` | **`+15.9%`** |
| Binary/scalar | `add_inplace` | `18433×4096` | `bf16` | `2.542 ms / 178.2` | `2.570 ms / 176.3` | `-1.1%` |
| Binary/scalar | `mul_inplace` | `18433×4096` | `bf16` | `2.553 ms / 177.4` | `2.570 ms / 176.3` | `-0.7%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `bf16` | `2.015 ms / 149.9` | `1.772 ms / 170.4` | **`+13.7%`** |
| Gated | `geglu_forward` | `4609×8192` | `fp32` | `1.330 ms / 170.4` | `1.316 ms / 172.1` | `+1.1%` |
| Gated | `geglu_forward` | `9217×8192` | `fp16` | `2.038 ms / 111.1` | `1.335 ms / 169.6` | **`+52.7%`** |
| Gated | `geglu_forward` | `9217×8192` | `bf16` | `2.028 ms / 111.7` | `1.298 ms / 174.5` | **`+56.2%`** |
| Norm *(Wave 2)* | `rms_norm_forward` | `9217×4096` | `fp32` | `1.894 ms / 159.5` | `1.872 ms / 161.4` | *unchanged* |
| Norm *(Wave 2)* | `rms_norm_forward` | `18433×4096` | `fp16` | `1.775 ms / 170.1` | `1.752 ms / 172.4` | *unchanged* |
| Norm *(Wave 2)* | `layernorm_inf_fp16` | `18433×4096` | `fp16` | `2.112 ms / 143.0` | `2.123 ms / 142.2` | *unchanged* |
| Norm *(Wave 2)* | `softmax_rows` | `9217×4096` | `fp32` | `2.385 ms / 126.6` | `2.464 ms / 122.6` | *unchanged* |

### Wave 1 Metal Verification & LOC Delta
- **Parity & Full `ctest` Suite:**
  - Targeted Wave 1 suites (`elementwise`, `elementwise_extra`, `large_elementwise`, `activations`, `geglu`, `swiglu`, `log_exp_round`, `add_channel_bias`, `stylegan`, `vocoder`, `batched`): **19/19 passed** (`0` failures).
  - Full Metal `ctest` (`-LE gcstress -j2`): **189/190 passed** (100% of Metal parity, runtime, JIT, and audit tests passed; the single failure is `brotensor_test_gpu_mir_emit`'s SPIR-V vs PTX `f16` Z3 equivalence check, unrelated to Metal).
- **LOC Delta (Metal Wave 1):**
  - Deleted files: `src/metal/elementwise_msl.h` (`-825`), `src/metal/swiglu.mm` (`-229`), `src/metal/log_exp_round.mm` (`-223`) — **`-1,277` lines**.
  - Slimmed files: `src/metal/elementwise.mm` (`880 -> 453`, keeping only `build_slot_mask`, `build_causal_mask_row`, `threshold_u8` outside Wave 1), `src/metal/batched_ops.mm` (`-193`), `src/metal/vocoder_activations.mm` (`-163`), `src/metal/stylegan_elementwise.mm` (`-151`), `src/metal/kernels.mm` (`-86`).
  - Added Metal MIR runtime cache: `src/metal/mtl_mir_cache.{h,mm}` (`+72` / `+209` = **`+281` lines**).
  - **Total Metal diff:** `2,628` lines deleted, `612` lines added (**`-2,016` net Metal LOC**; all modified/created files `< 500` LOC).

---

## Wave 1 (CUDA / NVIDIA RTX 4090): Elementwise, Activation, Bias, Cast & GLU -> Single-Source Brass MIR

Added CUDA runtime MIR cache (`src/cuda/cu_mir_cache.{h,cpp}`) lowering `brotensor::kernels::{build_unary_mir, build_binary_mir, build_act_bwd_mir, build_bias_mir, build_cast_mir, build_glu_mir}` through `brass::target::PtxTarget::emit_function` (`sm_80`, PTX 7.8, `kEwBlock = 256`, alignment-aware `lanes = 8` for 16B-aligned 16-bit, `lanes = 4` for 16B-aligned FP32 or 8B-aligned 16-bit, else `1`) and Driver-API `CudaJitEngine::get_function` (`kCapacity` bumped `512 -> 2048`), wired all Wave 1 CUDA elementwise, scalar, binary (`add`, `mul`, `axpby`), activation forward/backward, row/channel bias, cast, and GLU (SwiGLU / GeGLU / GeGLU-exact fwd/bwd) ops via `src/cuda/ops_elementwise.cpp` (`resblock.cu` direct symbols preserved), split the 3 non-MIR utilities (`threshold_u8`, `build_causal_mask_row`, `build_slot_mask`) into `src/cuda/elementwise_misc.cu`, and added `ptx_clamp` / `clampf` lowering to `brass` (`brass/src/target/ptx/ptx_isel_intrinsics.{hpp,cpp}`).

### `brotensor_bench_elementwise` (RTX 4090 CUDA, `CUDA_VISIBLE_DEVICES=0`, Wave 0 vs Wave 1)

Working sets sized past 72 MB L2 (~1008 GB/s peak HBM). Wave 1 figures reflect sustained `best of 20` timings observed on idle GPU 0 across two verification passes (one single-row Windows WDDM timer/scheduling spike seen once on `add_inplace` `bf16` `[458.4 -> 927.4 GB/s]` and once on `scale_inplace` `fp16` `[913.6 -> 455.9 GB/s]`):

| Category | Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 1 (ms / GB/s) | Speedup |
|---|---|---|---|---|---|---|
| Unary fwd | `silu_forward` | `9217×4096` | `fp32` | `0.321 ms / 942.3` | `0.330 ms / 916.0` | `-2.8%` |
| Unary fwd | `gelu_forward` | `9217×4096` | `fp32` | `0.319 ms / 945.3` | `0.330 ms / 916.0` | `-3.1%` |
| Unary fwd | `relu_forward` | `9217×4096` | `fp32` | `0.340 ms / 888.4` | `0.331 ms / 913.3` | `+2.8%` |
| Unary fwd | `tanh_forward` | `9217×4096` | `fp32` | `0.341 ms / 885.7` | `0.330 ms / 914.0` | `+3.2%` |
| Unary fwd | `sigmoid_forward` | `9217×4096` | `fp32` | `0.343 ms / 880.4` | `0.331 ms / 913.1` | `+3.7%` |
| Unary fwd | `silu_forward` | `18433×4096` | `fp16` | `0.346 ms / 872.6` | `0.331 ms / 913.1` | `+4.6%` |
| Unary fwd | `gelu_forward` | `18433×4096` | `fp16` | `0.348 ms / 867.4` | `0.331 ms / 913.1` | `+5.3%` |
| Unary fwd | `relu_forward` | `18433×4096` | `fp16` | `0.348 ms / 867.4` | `0.329 ms / 918.8` | `+5.9%` |
| Unary fwd | `tanh_forward` | `18433×4096` | `fp16` | `0.340 ms / 888.3` | `0.332 ms / 910.3` | `+2.5%` |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `fp16` | `0.349 ms / 864.9` | `0.330 ms / 915.9` | `+5.9%` |
| Unary fwd | `silu_forward` | `18433×4096` | `bf16` | `0.348 ms / 867.4` | `0.330 ms / 915.9` | `+5.6%` |
| Unary fwd | `gelu_forward` | `18433×4096` | `bf16` | `0.349 ms / 865.2` | `0.331 ms / 911.1` | `+5.3%` |
| Unary fwd | `relu_forward` | `18433×4096` | `bf16` | `0.348 ms / 867.8` | `0.331 ms / 913.1` | `+5.2%` |
| Unary fwd | `tanh_forward` | `18433×4096` | `bf16` | `0.349 ms / 865.8` | `0.330 ms / 915.7` | `+5.8%` |
| Unary fwd | `sigmoid_forward` | `18433×4096` | `bf16` | `0.347 ms / 870.0` | `0.332 ms / 910.6` | `+4.7%` |
| Unary bwd | `silu_backward` | `9217×4096` | `fp32` | `0.499 ms / 908.5` | `0.490 ms / 923.6` | `+1.7%` |
| Unary bwd | `gelu_backward` | `9217×4096` | `fp32` | `0.501 ms / 904.7` | `0.492 ms / 921.7` | `+1.9%` |
| Unary bwd | `silu_backward` | `18433×4096` | `fp16` | `0.507 ms / 894.1` | `0.491 ms / 921.8` | `+3.1%` |
| Unary bwd | `gelu_backward` | `18433×4096` | `fp16` | `0.506 ms / 895.5` | `0.492 ms / 921.7` | `+2.9%` |
| Unary bwd | `silu_backward` | `18433×4096` | `bf16` | `0.504 ms / 899.2` | `0.490 ms / 923.6` | `+2.7%` |
| Unary bwd | `gelu_backward` | `18433×4096` | `bf16` | `1.000 ms / 452.8` | `0.492 ms / 921.7` | **`+103.6%`** |
| Binary/scalar | `add_inplace` | `9217×4096` | `fp32` | `0.501 ms / 904.7` | `0.488 ms / 927.5` | `+2.5%` |
| Binary/scalar | `mul_inplace` | `9217×4096` | `fp32` | `0.479 ms / 945.6` | `0.488 ms / 927.5` | `-1.9%` |
| Binary/scalar | `scale_inplace` | `9217×4096` | `fp32` | `0.308 ms / 979.9` | `0.330 ms / 916.2` | `-6.5%` |
| Binary/scalar | `add_inplace` | `18433×4096` | `fp16` | `0.996 ms / 454.7` | `0.489 ms / 927.3` | **`+103.9%`** |
| Binary/scalar | `mul_inplace` | `18433×4096` | `fp16` | `0.496 ms / 912.6` | `0.488 ms / 927.4` | `+1.6%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `fp16` | `0.337 ms / 896.4` | `0.331 ms / 913.6` | `+1.9%` |
| Binary/scalar | `add_inplace` | `18433×4096` | `bf16` | `0.497 ms / 912.2` | `0.488 ms / 927.4` | `+1.7%` |
| Binary/scalar | `mul_inplace` | `18433×4096` | `bf16` | `0.496 ms / 914.2` | `0.488 ms / 927.6` | `+1.5%` |
| Binary/scalar | `scale_inplace` | `18433×4096` | `bf16` | `0.338 ms / 893.7` | `0.330 ms / 914.0` | `+2.3%` |
| Gated | `geglu_forward` | `4609×8192` | `fp32` | `0.252 ms / 899.3` | `0.247 ms / 918.0` | `+2.1%` |
| Gated | `geglu_forward` | `9217×8192` | `fp16` | `0.255 ms / 889.1` | `0.247 ms / 917.9` | `+3.2%` |
| Gated | `geglu_forward` | `9217×8192` | `bf16` | `0.255 ms / 889.5` | `0.247 ms / 917.9` | `+3.2%` |

*(Norm Wave 0/unchanged baseline on RTX 4090: `rms_norm_forward fp32 0.335 ms / 901.9 GB/s`, `fp16 0.335 ms / 901.6 GB/s`, `layernorm_inf_fp16 0.344 ms / 877.8 GB/s`, `softmax_rows 0.427 ms / 707.5 GB/s`; all `37/37` rows verified `ok`.)*

### Wave 1 CUDA Verification & LOC Delta
- **Correctness & Full `ctest` Suite:**
  - `brass` PTX & GPU proof tests (`-R "^PTX |^GPU - "`): **82/82 passed** (`0` failures).
  - Targeted Wave 1 CUDA parity suites (`elementwise`, `activations`, `geglu`, `swiglu`, `log_exp_round`, `add_channel_bias`, `vocoder_activations`, `large_elementwise`): **13/13 passed** (`0` failures).
  - Full rebased CUDA `ctest` (`-C Release -j 1`): **205/205 passed** (`0` failures).
- **LOC Delta (CUDA Wave 1):**
  - Deleted files: `src/cuda/elementwise.cu` (`-730`), `src/cuda/elementwise_activations.cu` (`-759`), `src/cuda/swiglu.cu` (`-206`), `src/cuda/elementwise_common.cuh` (`-183`) — **`-1,878` lines**.
  - Added CUDA files: `src/cuda/cu_mir_cache.{h,cpp}` (`+58` / `+185` = `+243`), `src/cuda/ops_elementwise.cpp` (`+297`), `src/cuda/elementwise_misc.cu` (`+107`) — **`+647` lines** (all files `< 300` LOC).
  - **Total CUDA diff** (including `CMakeLists.txt` & `cuda_jit_engine.h` capacity bump): `651` insertions, `1,882` deletions (**`-1,231` net LOC**).

---

## Wave 2: Vulkan Normalization, Softmax, Row/Col Reductions, RoPE & Rowvec -> Single-Source Brass MIR

Replaced 5 GLSL shaders / 21 variants (`norm.comp`, `reduce_rows.comp`, `sum_cols.comp`, `rope.comp`, `rowvec.comp`; `116 -> 95` compiled Vulkan shaders) with single-source vectorized `brass::Function` MIR builders in `src/kernels/{mir_common.h, mir_math.{h,cpp}, mir_elementwise.{h,cpp}, mir_reduce.{h,cpp}, mir_rope.{h,cpp}, mir_norm.{h,cpp}}`, dispatched through `src/vulkan/vk_mir_cache.{h,cpp}`. Null-pointer runtime branches and scalar row loops in `norm.comp` were replaced by compile-time flag specialization (`NormMirFlags`), 128-bit `lanes=4/8` vectorization, and `BVSolver`-provable 8×32 block reductions.

### `brotensor_bench_elementwise` Normalization Gains (Radeon 8060S, Wave 0 vs Wave 2)

| Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 2 (ms / GB/s) | Speedup |
|---|---|---|---|---|---|
| `rms_norm_forward` | `9217×4096` | `fp32` | `1.543 ms / 195.7` | `1.488 ms / 202.9` | `+3.7%` |
| `rms_norm_forward` | `18433×4096` | `fp16` | `1.669 ms / 181.0` | `1.402 ms / 215.4` | **`+19.0%`** |
| `layernorm_inf_fp16` | `18433×4096` | `fp16` | `1.984 ms / 152.2` | `1.444 ms / 209.1` | **`+37.4%`** |
| `softmax_rows` | `9217×4096` | `fp32` | `1.541 ms / 196.0` | `1.476 ms / 204.6` | `+4.4%` |

*(All 33 Wave 1 elementwise/gated rows in `brotensor_bench_elementwise` also verified `ok` at `217.6–226.5 GB/s`.)*

### Wave 2 Correctness, Proof Audit & LOC Delta
- **Correctness & Formal Audits:**
  - `brotensor_test_vulkan` `--only=ops`, `--only=train` (`brotensor_test_vulkan --only=gemm` passed prior to the 5-line masked-softmax predicate fix in `mir_norm.cpp`): passed (`0` failures)
  - `brotensor_test_backward_index_proofs`: ALL PASSED (including `rope_lifted_spirv_bounds_and_safety` on 3 MIR binaries and `norm_mir_lifted_spirv_bounds_and_barrier_safety` on 9 MIR backward binaries)
  - `brotensor_test_backward_shader_audit`: **141/141 shaders passed** (`123408` SPIR-V/PTX insts, `4603` shared accesses proven, `2262` global accesses proven, `616` uniform barriers, `541` uniform collectives, `0` failures)
  - Reduction & Parity Suites: `public_reductions` (`0` failures), `softmax` (`10/10`), `layernorm` (`8/8`), `layernorm_inference` (`11/11`), `layernorm_batched_caches` (`8/8`), `rmsnorm` (`21/21`), `softmax_rows` (`17/17`), `l2_norm` (`11/11`), `l2_normalize` (`5/5`), `rope` (`20/20`), `rope_mrope` (`4/4`), `reduce` (`6/6`), `reduce_extra` (`23/23`), `modulate` (`11/11`)
- **Real-Model Gate (Waves 1–2):**
  - `brovisionml` Depth-Anything-V2-Small (`1024×768`, w2/r5): **23.85 ms** best / mean `24.70 ms` (Wave 0: `23.42 / 24.25 ms`)
  - `brosoundml` Kokoro TTS (`155` phonemes -> `9.47 s` audio): **220.4 ms** mean / `42.98× RT` / best `217.4 ms` (Wave 0: `213.7 ms / 44.33× RT / 207.3 ms`)
  - `brolm` Qwen3-0.6B-Q8_0 (prefill `256` / decode `128`, r3): **12403.5 tok/s** prefill / **192.5 tok/s** (`5.19 ms/tok`) decode, deterministic `ids 0bc3383515f9e203` (Wave 0: `12526.5 tok/s` / `193.6 tok/s`)
- **LOC Delta (Vulkan Wave 2):**
  - Deleted 5 GLSL shaders / 21 variants (`norm.comp`, `reduce_rows.comp`, `sum_cols.comp`, `rope.comp`, `rowvec.comp`): `src/vulkan/shaders` `7,830 -> 7,405` (**`-425` lines**; cumulative **`-830` lines** / 11 shaders / 45 variants deleted across Waves 1–2)
  - Shared backend-neutral `src/kernels/*` now `1,445` lines total across Waves 1–2 (`+930` lines in Wave 2); `src/vulkan/vk_mir_cache.{h,cpp}` `356` lines (`+137` lines)
  - Wave 2 Metal consolidation completed below (`-2,379` net Metal LOC); pending Wave 2 CUDA deletion: `~3,874` CUDA lines (`layernorm*`, `rmsnorm*`, `softmax*`, `rope*`, `public_reductions*`, `reduce*`, `l2_norm*`, `l2_normalize*`)

---

## Wave 2 (Metal / Apple M2 Pro): Normalization, Softmax, Row/Col Reductions, RoPE & Rowvec -> Single-Source Brass MIR

Extended Metal's runtime MIR cache (`src/metal/mtl_mir_cache.{h,mm}`) with `Rowvec`, `SumRows`, `ArgmaxRows`, `SumCols`, `Norm` (`kNormBlock = 256`, compile-time `NormMirFlags`, `mir_vec_lanes_norm` clamping max lanes to `4` whenever any active tensor or softmax `s0_mask` is FP32), and `Rope` (`kRopeBlock = 256`), wired all Wave 2 Metal ops through it (`layernorm*`, `rms_norm*`, `pixel_norm*`, `l2_norm*`, `softmax*`, `rope*` including `rope_qkv_packed_inplace` and `rope_apply_mrope`, `sum_rows`, `sum_cols`, `argmax_rows`, `modulate`, `broadcast_mul`), and deleted the duplicated hand-written MSL kernels.

### `brotensor_bench_elementwise` Normalization Gains (Apple M2 Pro Metal, Wave 0 / Wave 1 vs Wave 2)

All `37/37` ops verified `ok` sequentially on an idle M2 Pro (19-core GPU, 16 GiB unified RAM, ~200 GB/s theoretical bus).

| Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 1 (ms / GB/s) | Wave 2 (ms / GB/s) | Speedup (vs Wave 0) |
|---|---|---|---|---|---|---|
| `rms_norm_forward` | `9217×4096` | `fp32` | `1.894 ms / 159.5` | `1.872 ms / 161.4` | `1.760 ms / 171.6` | **`+7.6%`** |
| `rms_norm_forward` | `18433×4096` | `fp16` | `1.775 ms / 170.1` | `1.752 ms / 172.4` | `1.731 ms / 174.4` | **`+2.5%`** |
| `layernorm_inf_fp16` | `18433×4096` | `fp16` | `2.112 ms / 143.0` | `2.123 ms / 142.2` | `1.746 ms / 172.9` | **`+21.0%`** |
| `softmax_rows` | `9217×4096` | `fp32` | `2.385 ms / 126.6` | `2.464 ms / 122.6` | `1.781 ms / 169.6` | **`+33.9%`** |

*(All 33 Wave 1 elementwise/gated rows in `brotensor_bench_elementwise` also verified `ok` at `170.4–179.8 GB/s`.)*

### Wave 2 Metal Verification & LOC Delta
- **Parity & Full `ctest` Suite:**
  - Targeted Wave 2 suites (`public_reductions`, `softmax` `10/10` incl. host-mask rejection, `softmax_rows`, `layernorm`, `layernorm_inference`, `layernorm_batched_caches`, `layernorm_stability`, `rmsnorm`, `l2_norm`, `l2_normalize`, `rope`, `rope_perhead`, `rope_mrope`, `packed_encoder`, `reduce`, `reduce_extra`, `modulate`, `stylegan`): **23/23 passed** (`0` failures).
  - Full Metal `ctest` (`-LE gcstress -j2`, with `brass` `d5608f4`): **188/190 passed** (100% of Metal parity, runtime, JIT, and `brotensor_test_gpu_mir_emit` tests passed; the only 2 failures are the duplicate CTest targets for the CPU-only SPIR-V/PTX `backward_shader_audit` binary timing out at 300s on M2 Pro).
- **LOC Delta (Metal Wave 2):**
  - Deleted files: `src/metal/rms_norm.mm` (`-559`), `src/metal/softmax.mm` (`-351`), `src/metal/l2_norm.mm` (`-237`), `src/metal/rope_mrope.mm` (`-232`), `src/metal/stylegan_elementwise.mm` (`-205`), `src/metal/modulate.mm` (`-178`) — **`-1,762` lines**.
  - Slimmed files: `src/metal/layernorm.mm` (`525 -> 424`, absorbing `rms_norm`, `pixel_norm`, `l2_norm`, `softmax`), `src/metal/rope.mm` (`531 -> 222`, absorbing `rope_mrope` and `rope_qkv_packed_inplace`), `src/metal/public_reductions.mm` (`540 -> 179`, keeping only non-Wave-2 `rows_count_above`), `src/metal/packed_encoder.mm` (`-58`, removing duplicate `rope_qkv_packed_inplace`). Note: `src/metal/reduce.mm` (`masked_mean_pool`) and `src/metal/l2_normalize.mm` (`l2_normalize_nchw_forward`) remain outside Wave 2.
  - Extended cache & rowvec: `src/metal/mtl_mir_cache.{h,mm}` (`+180` net lines -> `462` total), `src/metal/elementwise.mm` (`+43` lines -> `496` total for `modulate` / `broadcast_mul`).
  - **Total Metal diff (Wave 2):** `3,292` lines deleted, `913` lines added (**`-2,379` net Metal LOC**; cumulative Waves 1–2: **`-4,395` net Metal LOC**; all modified/created files `< 500` LOC).

---

## Wave 2 (CUDA / NVIDIA RTX 4090): Normalization, Softmax, Row/Col Reductions, RoPE, Rowvec & Remaining Wave 1 Ops -> Single-Source Brass MIR

Extended CUDA's runtime MIR cache (`src/cuda/cu_mir_cache.{h,cpp}`) to `MirKind::{Rowvec, SumRows, ArgmaxRows, SumCols, Norm, Rope}` (`kCapacity` `2048 -> 4096` in `cuda_jit_engine.h`, alignment-aware `mir_vec_lanes_norm` clamping max lanes to `4` whenever any active operand or `NORM_SOFTMAX` `s0_mask` is FP32, compile-time `NormMirFlags`), wired all Wave 2 CUDA ops plus the remaining separable Wave 1 CUDA ops (`exp`/`log`/`round` from `log_exp_round.cu`, `sin`/`cos`/`rsqrt` from `stylegan_elementwise.cu`, `elu`/`leaky_relu` from `vocoder_activations.cu`, the 5 batched elementwise ops from `batched_ops.cu`, `rope_qkv_packed_inplace` from `packed_encoder.cu`, and `modulate`/`broadcast_mul` via `MirKind::Rowvec`), moved `rows_count_above` into `src/cuda/elementwise_misc.cu`, and deleted 11 obsolete `.cu` files (`src/cuda/ops_{elementwise,norm,reduce,rope}.cpp`).

### `brotensor_bench_elementwise` Normalization Gains (RTX 4090 CUDA, `CUDA_VISIBLE_DEVICES=0`, Wave 0 vs Wave 2)

All `37/37` ops verified `ok` sequentially on an idle RTX 4090 (working sets > 72 MB L2, ~1008 GB/s peak HBM):

| Op | Shape | Dtype | Wave 0 (ms / GB/s) | Wave 2 (ms / GB/s) | Speedup (vs Wave 0) |
|---|---|---|---|---|---|
| `rms_norm_forward` | `9217×4096` | `fp32` | `0.336 ms / 899.2` | `0.332 ms / 910.3` | `+1.2%` |
| `rms_norm_forward` | `18433×4096` | `fp16` | `0.336 ms / 899.2` | `0.331 ms / 913.3` | `+1.6%` |
| `layernorm_inf_fp16` | `18433×4096` | `fp16` | `0.346 ms / 872.6` | `0.331 ms / 913.1` | `+4.6%` |
| `softmax_rows` | `9217×4096` | `fp32` | `0.429 ms / 703.9` | `0.332 ms / 910.7` | **`+29.4%`** |

*(All 33 Wave 1 elementwise/gated rows in `brotensor_bench_elementwise` also verified `ok` at `895.7–927.6 GB/s`.)*

### Wave 2 CUDA Verification & LOC Delta
- **Correctness & Full `ctest` Suite:**
  - Targeted Wave 1 + Wave 2 CUDA parity & unit suites (`gpu_mir_emit`, `elementwise*`, `activations*`, `geglu`, `swiglu*`, `log_exp_round`, `add_channel_bias`, `public_reductions`, `softmax*`, `layernorm*`, `rmsnorm`, `l2_norm`, `l2_normalize`, `rope*`, `reduce*`, `modulate`, `stylegan*`, `vocoder`, `packed_encoder`, `batched*`): **40/40 passed** (`0` failures, `16.42s`).
  - Full CUDA `ctest` (`-C Release -LE gcstress -j 1`): **204/204 passed** (`0` failures).
- **LOC Delta (CUDA Wave 2 + Remaining Wave 1):**
  - Deleted 11 files (`-4,231` lines): `layernorm.cu` (`-875`), `rms_norm.cu` (`-483`), `softmax.cu` (`-478`), `rope.cu` (`-460`), `stylegan_elementwise.cu` (`-396`), `layernorm_batched.cu` (`-316`), `log_exp_round.cu` (`-295`), `public_reductions.cu` (`-275`), `l2_norm.cu` (`-255`), `rope_mrope.cu` (`-225`), `modulate.cu` (`-173`).
  - Slimmed: `batched_ops.cu` (`-174`), `vocoder_activations.cu` (`-148`), `packed_encoder.cu` (`-63`).
  - Added/extended: `src/cuda/cu_mir_cache.{h,cpp}` (`+220`), `src/cuda/ops_elementwise.cpp` (`+130` net -> `371`), `src/cuda/elementwise_misc.cu` (`+65` -> `172`), `src/cuda/ops_norm.cpp` (`+509`), `src/cuda/ops_reduce.cpp` (`+91`), `src/cuda/ops_rope.cpp` (`+213`) (all files `<= 509` LOC).
  - **Total CUDA diff (Wave 2 + remaining Wave 1):** `1,184` lines added, `4,676` lines deleted (**`-3,492` net CUDA LOC**; cumulative CUDA Waves 1–2 net reduction: **`-4,723` net LOC**).

---

## Wave 3: Vulkan GroupNorm, BatchNorm, Copy2D/Transpose/Fill & Scatter-Add -> Single-Source Brass MIR

Replaced 6 GLSL shaders / 13 variants (`gnorm.comp`, `bnorm.comp`, `copy2d.comp`, `transpose.comp`, `fill_bytes.comp`, `scatter_add.comp`; `95 -> 82` compiled Vulkan shaders) with single-source `brass::Function` MIR builders in `src/kernels/{mir_copy.{h,cpp}, mir_scatter.{h,cpp}, mir_gnorm.{h,cpp}}`, dispatched via `src/vulkan/vk_mir_cache.{h,cpp}` across `ops_gnorm.cpp`, `ops_copy.cpp`, `ops_scatter.cpp`, `ops_quant.cpp`, `ops_conv.cpp`, and `tensor.cpp`. Also updated `brass` (`a370f6c`) to handle `vec2<f16>` `OpFConvert` + `OpBitcast` in `gpu_equiv_spirv.cpp`, fixing `brotensor_test_gpu_mir_emit` `f16` Z3 equivalence.

### Wave 3 Correctness, Proof Audit & LOC Delta
- **Correctness & Formal Audits:**
  - `brotensor_test_gpu_mir_emit`: ALL PASSED (including `f32`, `f16`, `bf16` SPIR-V == PTX equivalence + RowNorm)
  - `brotensor_test_backward_index_proofs`: ALL PASSED (`scatter_add_mir_lifted_spirv_bounds_and_safety` on 3 MIR binaries, `norm_mir_lifted_spirv_bounds_and_barrier_safety` on 16 MIR norm/gnorm/bnorm backward binaries)
  - `brotensor_test_backward_shader_audit`: **149/149 shaders passed** (`126722` SPIR-V/PTX insts, `5810` shared accesses proven, `2687` global accesses proven, `643` uniform barriers, `752` uniform collectives, `0` failures)
  - Unit & Parity Suites: `group_norm` (ok), `group_norm_parity` (`18/18`), `batch_norm` (ok), `batch_norm_parity` (`10/10`), `l2_normalize_parity` (`5/5`), `transpose` (ok), `transpose_parity` (`19/19`), `concat_parity` (`13/13`), `concat_extra_parity` (`20/20`), `embedding_parity` (`8/8`), `embedding_extra_parity` (`8/8`), `conv2d_parity` (`47/47`), `linear_wq_parity` (`6/6`), `brotensor_test_vulkan` `--only=ops`, `--only=train`, `--only=quant` (`0` failures)
- **LOC Delta (Vulkan Wave 3):**
  - Deleted 6 GLSL shaders / 13 variants (`gnorm.comp`, `bnorm.comp`, `copy2d.comp`, `transpose.comp`, `fill_bytes.comp`, `scatter_add.comp`): `src/vulkan/shaders` `7,405 -> 6,849` (**`-556` lines**; cumulative **`-1,386` lines** / 17 shaders / 58 variants deleted, `140 -> 82` compiled Vulkan shaders)
  - Shared backend-neutral `src/kernels/*` now `2,217` lines total across Waves 1–3 (`+772` lines in Wave 3, max file `399` lines); `src/vulkan/vk_mir_cache.{h,cpp}` `508` lines (`+152` lines)
  - Wave 3 Metal consolidation completed below (`Waves 3–4 Metal`, `-5,257` combined net Metal LOC); pending Wave 3 CUDA deletion: `~2,434` CUDA lines (`group_norm*`, `batch_norm*`, `transpose*`, `concat*`, `gather_scatter*`, `embedding*`)

---

## Wave 4: Vulkan Resample, ResampleBwd, Remap, Sampler & Philox -> Single-Source Brass MIR

Replaced 5 GLSL shaders / 12 variants (`resample.comp`, `resample_bwd.comp`, `remap.comp`, `sampler.comp`, `philox.comp`; `82 -> 70` compiled Vulkan shaders, **50% of all 140 baseline shader variants removed across Waves 1–4**) with single-source `brass::Function` MIR builders in `src/kernels/{mir_diffusion.{h,cpp}, mir_remap.{h,cpp}, mir_resample.{h,cpp}, mir_resample_bwd.{h,cpp}}` and `emit_log_acc` + helpers in `mir_math` / `mir_common.h`, dispatched through `src/vulkan/vk_mir_cache.{h,cpp}` across `ops_spatial.cpp`, `ops_spatial_bwd.cpp`, and `ops_diffusion.cpp`. Sampler steps (`DDIM`, `Euler`, `DPM++ 2M`) support 128-bit vectorization with a tail loop, `remap` copies raw 16/32-bit patterns bit-for-bit, `resample`/`resample_bwd` preserve exact 64-bit rational coordinates and left-to-right 2×2 summation, and `philox` lowers `umulExtended` via `mul_hi_u32` + `log_acc`/`sincos_acc`.

### Wave 4 Correctness, Proof Audit & LOC Delta
- **Correctness & Formal Audits (Radeon 8060S):**
  - `brotensor_test_backward_index_proofs`: **ALL PASSED** (`resample2d_lifted_spirv_resample_bwd_f32_f16_bf16_bounds_and_safety` on 18 MIR `.spv` binaries: `3788` insts, `21` guarded PSB64 writes, `337` bounds guards; `resample2d_lifted_spirv_resample_fwd_and_remap_bounds_and_safety` on 15 MIR `.spv` binaries: `3090` insts, `18` guarded PSB64 writes, `238` bounds guards)
  - `brotensor_test_backward_shader_audit`: **149/149 shaders passed** (`119897` SPIR-V/PTX insts, `5810` shared accesses proven, `2687` global accesses proven, `643` uniform barriers, `752` uniform collectives, `0` failures)
  - Unit & Parity Suites: `noise` (`OK`), `diffusion_primitives` (`All passed`), `diffusion_parity` (`23/23`), `image_preproc_parity` (`5/5`), `resample_parity` (`39/39`), `interp2d_parity` (`30/30`), `interp2d_align_corners` (`OK`), `pad2d_parity` (`11/11`), `slice2d_parity` (`5/5`), `pool2d_parity` (`17/17`), `unfold2d_parity` (`5/5`), `convex_upsample_parity` (`5/5`), `window_partition_parity` (`7/7`), `spatial_merge_parity` (`6/6`), `pixel_shuffle_upsample_parity` (`18/18`), `patch_unpack_parity` (`4/4`), `gather_rows_parity` (`8/8`), `resblock_parity` (`16/16`), `brotensor_test_vulkan` `--only=spatial`, `--only=ops`, and `--only=train` (`0` failures across all 3 suites)
- **Real-Model Gate (Waves 1–4, Radeon 8060S Vulkan):**
  - `brovisionml` Depth-Anything-V2-Small (`1024×768`, w2/r5): **23.43 ms** best / **24.43 ms** mean (Wave 0: `23.42 / 24.25 ms`)
  - `brosoundml` Kokoro TTS (`155` phonemes -> `9.47 s` audio): **207.2 ms** mean / **45.74× RT** / **205.0 ms** best (Wave 0: `213.7 ms / 44.33× RT / 207.3 ms`)
  - `brodiffusion` SD 1.5 (`lcm-dreamshaper`, 512×512, 20 steps) & `Sana-600M` (512×512, 20 steps): completed cleanly end-to-end
- **LOC Delta (Vulkan Wave 4):**
  - Deleted 5 GLSL shaders / 12 variants (`resample.comp`, `resample_bwd.comp`, `remap.comp`, `sampler.comp`, `philox.comp`): `src/vulkan/shaders/*` `6,849 -> 5,897` (**`-952` lines**; cumulative **`-2,338` lines** / 22 shaders / 70 variants deleted across Waves 1–4, `140 -> 70` compiled Vulkan shaders)
  - Shared backend-neutral `src/kernels/*` now `3,232` lines across 25 files (`+1,015` lines in Wave 4, max file `399` lines); `src/vulkan/vk_mir_cache.{h,cpp}` `657` lines (`+149` lines)
  - Wave 4 Metal consolidation completed below; pending Wave 4 CUDA deletion: **`3,341`** CUDA lines across `resample`, `interp2d`, `pad2d`, `slice2d`, `pool2d`, `unfold2d`, `convex_upsample`, `window_partition`, `spatial_merge`, `pixel_shuffle_upsample`, `patch_unpack`, `gather_scatter`, `diffusion_samplers`, `noise`, `image_preproc`

---

## Waves 3 & 4 (Metal / Apple M2 Pro): GroupNorm, BatchNorm, Copy/Transpose/Fill, Scatter-Add, Resample/Bwd, Remap, Sampler & Philox -> Single-Source Brass MIR

Extended Metal's runtime MIR cache (`src/metal/mtl_mir_cache.{h,mm}`, now `764` LOC total with 2D/3D grid dispatch) with Wave 3 (`FillBytes` at `kFillBlock = 64`, `Copy2d`, `Transpose`, `ScatterAdd`, `Gnorm`, `Bnorm`) and Wave 4 (`Sampler`, `Philox`, `Remap`, `Resample`, `ResampleBwd`), wired every Metal Wave 3 & Wave 4 op through `MslTarget`, made Metal's `copy_d2d`/`copy_d2d_strided`/`concat_*`/`memset_zero` fully asynchronous on the GPU queue (eliminating `cuda_sync()` host-memcpy stalls), and deleted 19 replaced Metal files.

### Waves 3 & 4 Metal Verification & LOC Delta
- **Parity & Full `ctest` Suite (Apple M2 Pro, Metal Release, `brass` `d5608f4`):**
  - Targeted Waves 3 & 4 suites (`group_norm`, `group_norm_parity`, `batch_norm`, `batch_norm_parity`, `l2_normalize_parity`, `resblock`, `resblock_int8w`, `resblock_parity`, `transpose`, `transpose_parity`, `concat_parity`, `concat_extra_parity`, `embedding_parity`, `embedding_extra_parity`, `gather_rows`, `gather_rows_parity`, `ddim_step`, `sdxl_schedulers`, `diffusion_primitives`, `diffusion_parity`, `masked_diffusion`, `noise`, `image_preproc`, `image_preproc_parity`, `pad2d`, `pad2d_parity`, `slice2d`, `slice2d_parity`, `unfold2d_parity`, `window_partition`, `window_partition_parity`, `spatial_merge`, `spatial_merge_parity`, `pixel_shuffle_upsample`, `pixel_shuffle_upsample_parity`, `patch_unpack_parity`, `resample`, `resample_parity`, `interp2d`, `interp2d_parity`, `interp2d_align_corners`, `pool2d`, `pool2d_parity`, `convex_upsample_parity`): **48/48 passed** (`44.65 s`, `0` failures).
  - Full Metal `ctest` (`-LE gcstress -j2`): **188/190 passed** (100% of Metal parity, runtime, JIT, `backward_index_proofs`, and `gpu_mir_emit` tests passed; the only 2 failures are the duplicate CTest targets for the CPU-only SPIR-V/PTX `backward_shader_audit` binary timing out at 300s on M2 Pro).
- **LOC Delta (Metal Waves 3 & 4):**
  - Deleted 19 files: `layernorm_msl.h` (`-575`), `batch_norm.mm` (`-578`), `l2_normalize.mm` (`-123`), `transpose.mm` (`-199`), `ddim_step.mm` (`-127`), `euler_step.mm` (`-108`), `dpmpp_2m_step.mm` (`-138`), `timestep_embedding.mm` (`-100`), `image_preproc.mm` (`-238`), `noise.mm` (`-238`), `gather_rows.mm` (`-233`), `pad2d.mm` (`-281`), `slice2d.mm` (`-229`), `unfold2d.mm` (`-179`), `window_partition.mm` (`-202`), `spatial_merge.mm` (`-347`), `interp2d.mm` (`-383`), `pool2d.mm` (`-540`), `convex_upsample.mm` (`-162`) — **`-4,980` lines** in deleted files alone.
  - Consolidated / rewritten files (all `< 600` LOC): `mtl_mir_cache.{h,mm}` (`182` / `582`), `group_norm.mm` (`320`), `concat.mm` (`253`), `embedding.mm` (`125`), `tensor.mm` (`262`), `diffusion.mm` (`265`), `resample.mm` (`447`), `resample_bwd.mm` (`160`).
  - **Net Metal LOC delta (Waves 3 & 4):** **`-5,257` lines** (`src/metal/` `32,290 -> 27,033` LOC; **cumulative Waves 1–4: `-9,652` net Metal LOC** and 28 Metal files deleted).

---

## Wave 5: Vulkan LSTM, Gated Delta Rule, Cross-Entropy (Softmax/BCE) & Top-K/Segment Stats -> Single-Source Brass MIR

Replaced 4 GLSL shaders / 6 variants (`lstm.comp`, `delta_rule.comp`, `xent.comp`, `topk_seg.comp`; `70 -> 64` compiled Vulkan shaders) with single-source `brass::Function` MIR builders in `src/kernels/{mir_lstm.{h,cpp}, mir_delta.{h,cpp}, mir_xent.{h,cpp}, mir_topk.{h,cpp}}`, dispatched through `src/vulkan/vk_mir_cache.{h,cpp}` across `ops_lstm.cpp`, `ops_delta.cpp`, `ops_xent.cpp`, and `ops_topk.cpp` (`jit_pipeline(..., req_subgroup)` preserved for `delta_rule` `lpr` lane groups with `ne <= 16` loop-carried register state). Also simplified `mir_rope.cpp` to reuse quotient multiplications (`sub(x, mul(q, d))`) instead of redundant `umod`s and removed the redundant Z3 CDCL bitblast on `rope_{f32,f16,bf16}` in `backward_shader_audit.cpp` (already proven in `backward_index_verify.cpp`), cutting `brotensor_test_backward_shader_audit` from ~280s down to **46.95 s**.

### Wave 5 Correctness, Proof Audit & LOC Delta
- **Correctness & Formal Audits (Radeon 8060S):**
  - `brotensor_test_backward_index_proofs`: **ALL PASSED** (including `norm_lstm_deltarule_xent_lifted_spirv_bounds_and_barrier_safety` on 5 MIR `.spv` binaries: `1402` insts, `26` guarded PSB64 writes, `45` bounds comparisons, and `rope_lifted_spirv_bounds_and_safety` on 3 MIR binaries: `504` insts)
  - `brotensor_test_backward_shader_audit`: **151/151 shaders passed** in **46.947 s** (`118484` SPIR-V/PTX insts, `5773` shared / `2675` global accesses proven, `616` uniform barriers, `774` uniform collectives, `0` failures)
  - Unit & Parity Suites: `lstm_parity` (`12/12`), `gated_delta_rule_parity` (`7/7`), `loss_parity` (`9/9`), `loss_fused_parity` (`8/8`), `top_k_parity` (`5/5`), `packed_encoder_parity` (`21/21` incl. `segment_stats` & `rope_packed`), `rope_parity` (`20/20`), `brotensor_test_vulkan` `--only=ops`, `--only=train`, and `--only=attention` (`0` failures)
- **LOC Delta (Vulkan Wave 5):**
  - Deleted 4 GLSL shaders / 6 variants (`lstm.comp`, `delta_rule.comp`, `xent.comp`, `topk_seg.comp`): `src/vulkan/shaders/*` `5,897 -> 5,499` (**`-398` lines**; cumulative **`-2,736` lines** / 26 files / 76 variants deleted across Waves 1–5, `140 -> 64` compiled Vulkan shaders)
  - Shared backend-neutral `src/kernels/*` now `3,811` lines across 33 files (`+579` lines in Wave 5, max file `399` lines); `src/vulkan/vk_mir_cache.{h,cpp}` `770` lines (`+113` lines)
  - Wave 5 Metal consolidation completed below (`-758` net Metal LOC); pending Wave 5 CUDA deletion: `1,896` CUDA lines (`lstm.cu` `494`, `gated_delta_rule.cu` `475`, `top_k.cu` `248`, `loss.cu` `531`, `packed_encoder.cu` `148`)

---

## Wave 5 (Metal / Apple M2 Pro): LSTM, Gated Delta Rule, Cross-Entropy (Softmax/BCE) & Top-K/Segment Stats -> Single-Source Brass MIR

Extended Metal's runtime MIR cache (`src/metal/mtl_mir_cache.{h,mm}`, `207` / `663` LOC) with `MirKind::{Lstm, DeltaRule, Xent, Topk}`, wired every Wave 5 Metal op through `MslTarget`, moved `lstm_forward_train`/`lstm_backward` matrix products onto Metal's simdgroup GEMM (`launch_matmul_abt_mixed`) + `MirKind::Lstm` + `NORM_COL_SUM` (removing the old scalar MSL loops & CAS-atomic backward), switched `gated_delta_rule_{chunked,step}` to 32-lane simdgroup register-resident `MirKind::DeltaRule`, moved `segment_softmax_stats` into `top_k.mm` alongside `top_k_rows` (`MirKind::Topk` across FP32/FP16/BF16), and replaced all 3 hand-written softmax-xent / BCE MSL kernels in `loss.mm` with `MirKind::Xent` (leaving only small `mse_*` kernels for Wave 6+ misc).

### Wave 5 Metal Verification & LOC Delta
- **Parity & Full `ctest` Suite (Apple M2 Pro, Metal Release, `brass` `d5608f4`):**
  - Targeted Wave 5 & audit suites (`lstm`, `lstm_parity`, `gated_delta_rule`, `gated_delta_rule_parity`, `top_k`, `top_k_parity`, `packed_encoder_parity`, `loss_parity`, `loss_fused_parity`, `bce_with_logits`, `batched_backward_parity`, `gpu_mir_emit`, `backward_shader_audit` ×2): **14/14 passed** (`53.19 s`, `0` failures).
  - Full Metal `ctest` (`-LE gcstress -j2 --timeout 300`): **190/190 passed** (`285.34 s`, `0` failures; `backward_shader_audit` `47.54–47.60 s`).
- **LOC Delta (Metal Wave 5):**
  - Slimmed / rewritten files (all `< 700` LOC): `lstm.mm` (`441 -> 245`), `gated_delta_rule.mm` (`241 -> 108`), `top_k.mm` (`220 -> 83`, absorbing `segment_softmax_stats`), `loss.mm` (`647 -> 328`), `packed_encoder.mm` (`452 -> 373`), `mtl_mir_cache.{h,mm}` (`182/582 -> 207/663`).
  - **Net Metal LOC delta (Wave 5):** `522` insertions, `1,280` deletions (**`-758` net Metal LOC**; `src/metal/` `27,033 -> 26,275` LOC; **cumulative Waves 1–5: `-10,410` net Metal LOC** and 28 Metal files deleted).

---

## Wave 3 (CUDA / RTX 4090): GroupNorm, BatchNorm, L2-Normalize, Transpose & Deterministic Scatter-Add -> Single-Source Brass MIR

Extended CUDA's runtime MIR cache (`src/cuda/cu_mir_cache.{h,cpp}`, `113` / `408` LOC with 2D/3D `cuLaunchKernel` grid dispatch) with `FillBytes`, `Copy2d`, `Transpose`, `ScatterAdd`, `Gnorm`, and `Bnorm` lowered via `PtxTarget::emit_function` (`sm_89`), and wired GroupNorm fwd/bwd, BatchNorm fwd/inference/bwd, per-pixel NCHW L2-normalize, NCHW <-> sequence transpose (`ops_copy.cpp`), and deterministic bitonic-sort scatter-add (`ops_scatter.cpp`) through `cu_mir_cache` (`concat.cu` retains async CUDA DMA copies `cudaMemcpyAsync`/`cudaMemcpy2DAsync`).

### Wave 3 CUDA Verification & LOC Delta (RTX 4090 GPU 0, `CUDA_VISIBLE_DEVICES=0`, `brass@d5608f4`)
- **Targeted Wave 3 Parity & Unit Suites:** `18/18` passed (`16.74 s`, `0` failures): `gather_rows`, `conv_transpose2d`, `batch_norm`, `group_norm`, `resblock`, `resblock_int8w`, `transpose`, `embedding_parity`, `concat_parity`, `embedding_extra_parity`, `concat_extra_parity`, `group_norm_parity`, `transpose_parity`, `resblock_parity`, `l2_normalize_parity`, `gather_rows_parity`, `conv_transpose2d_parity`, `batch_norm_parity`.
- **Full CUDA `ctest` (`-LE gcstress -j 1`):** **`204/204` passed** (`906.16 s`, `0` failures).
- **LOC Delta (CUDA Wave 3):**
  - Deleted 5 `.cu` files: `group_norm.cu` (`-789`), `group_norm_split.cu` (`-144`), `batch_norm.cu` (`-451`), `l2_normalize.cu` (`-126`), `transpose.cu` (`-156`) — **`-1,666` lines** deleted cleanly.
  - Slimmed `embedding.cu` (`-119` net diff, now `99` LOC) & `gather_scatter.cu` (`-99` net diff, now `161` LOC); added `ops_gnorm.cpp` (`260`), `ops_scatter.cpp` (`87`), `ops_copy.cpp` (`54`), `cu_mir_cache.{h,cpp}` (`+178`). Net `src/cuda/` Wave 3 delta: **`-1,308` lines** (all edited/created files `<= 408` LOC).

---

## Wave 6: Vulkan Token Selection (`sample_logits` / `masked_diffusion`), Vision (`upfirdn2d` / `mod_conv` / `col2im2d` / `deform_conv2d`) & Small Misc Ops -> Single-Source Brass MIR

Replaced 2 GLSL shaders / 4 variants (`select.comp` `344`L, `misc.comp` `406`L; `64 -> 60` compiled Vulkan shaders) with three focused backend-neutral `brass::Function` MIR modules:
- `src/kernels/mir_select.{h,cpp}` (`561` LOC): `SEL_SAMPLE` (temperature softmax, 64-bit fixed-point top-k/top-p binary search, Philox 4x32-10 inverse-CDF draw, greedy argmax), `SEL_COUNTER`, `SEL_COMMIT`, and `SEL_MD` (guided/unguided log-softmax, 32-bit order-preserving top-k threshold search + rank scan, `splitmix64` Gumbel noise). Shared `philox_words` from `mir_diffusion.h` and added shared `emit_splitmix64_mix_u01` in `mir_math.{h,cpp}`.
- `src/kernels/mir_misc.{h,cpp}` (`389` LOC): `MI_MEAN_POOL`, `MI_MEAN_POOL_BWD`, `MI_SLOT_MASK`, `MI_CAUSAL_ROW`, `MI_THRESHOLD_U8`, `MI_COUNT_ABOVE`, `MI_XAVIER`, `MI_SGD`, `MI_ADAM`, `MI_MSE_SAMPLE`, `MI_MSE_BWD`, `MI_MSE_SUM`, `MI_MOMENTS`, `MI_BIAS_ACT`, `MI_BIAS_ACT_BWD`, `MI_BIAS_ACT_DB`.
- `src/kernels/mir_vision.{h,cpp}` (`476` LOC): `MI_UPFIRDN` (upfirdn2d fwd/bwd), `MI_MODW`, `MI_MODW_BWD`, `MI_MODW_DS`, `MI_MODW_DW`, `MI_COL2IM2D`, `MI_DEFORM_COL`.
- Extended `src/vulkan/vk_mir_cache.{h,cpp}` (`815` LOC) with `SelectSample`, `SelectCounter`, `SelectCommit`, `SelectMd`, `Misc`, `Vision` and generic packed push-constant dispatcher `mir_dispatch_args`, wired `ops_sampling.cpp` (`177` LOC), `ops_misc.cpp` (`391` LOC), and `ops_vision.cpp` (`558` LOC).

### Wave 6 Correctness, Proof Audit, Model Gate & LOC Delta (Radeon 8060S)
- **Formal Audits:**
  - `brotensor_test_backward_index_proofs`: **ALL PASSED** (including `upfirdn2d_lifted_spirv_bounds_and_safety` on 6 MIR `.spv` binaries: `972` lifted SPIR-V insts, `6` guarded PSB64 writes, `58` bounds checks, `0` hazards)
  - `brotensor_test_backward_shader_audit`: **161/161 shaders passed** in **56.283 s** (`117425` SPIR-V/PTX insts, `6595` shared / `2675` global accesses proven, `661` uniform barriers, `810` uniform collectives, `0` failures)
- **Unit, Parity & Vulkan Suites:** `brotensor_test_sample_logits`, `sample_logits_parity` (`16/16`), `masked_diffusion`, `masked_mean_pool_parity` (`10/10`), `attention_moments`, `attention_moments_parity` (`5/5`), `deform_conv2d_parity` (`7/7`), `bias_act`, `upfirdn2d`, `modulated_conv2d`, `filtered_lrelu`, `stylegan_parity`, `stylegan_primitives`, `conv_transpose2d_parity`, `elementwise_extra_parity`, `reduce_extra_parity`, plus `brotensor_test_vulkan` `--only={ops,train,sampling,misc,vision,capture}` — **all passed (`0` failures)**
- **Downstream Real-Model Gate (`Vulkan`, Radeon 8060S):**
  - `Depth-Anything-V2-Small` (`1024×768`): `23.48 ms` best / `25.74 ms` mean
  - `SAM ViT-B` (`1024×768`): encode `126.74 ms` best / `127.54 ms` mean; decode `8.97 ms` best / `9.30 ms` mean
  - `SAM AMG` (`1024×768`): `4852.54 ms` best / `4867.58 ms` mean
  - `Kokoro TTS`: `214.5 ms` mean for `9.47 s` audio (`44.18× realtime`)
  - `SD 1.5 (LCM-Dreamshaper)` & `Sana-600M` (`512×512`, `4` steps): clean generation
- **LOC Delta (Vulkan Wave 6):**
  - `src/vulkan/shaders/*`: `5,499 -> 5,125` (**`-374` lines**; cumulative **`-3,110` lines** / 28 `.comp` files / 80 variants deleted across Waves 1–6, `140 -> 60` compiled Vulkan shaders)
  - Shared backend-neutral `src/kernels/*`: `5,252` lines across 39 files (`+1,441` lines in Wave 6, max file `535` lines); `src/vulkan/vk_mir_cache.{h,cpp}` `815` lines (`+45` lines, max file `637` lines)
  - Wave 6 Metal consolidation completed below (`-2,007` net Metal LOC); pending cross-backend targets: Windows CUDA Waves 4–6.

---

## Wave 6 (Metal / Apple M2 Pro): Token Selection, Vision (`upfirdn2d`/`modulated_conv2d`/`deform_conv2d`/`col2im2d`) & Small Misc Ops -> Single-Source Brass MIR

Extended Metal's runtime MIR cache (`src/metal/mtl_mir_cache.{h,mm}`, `218` / `698` LOC) with `MirKind::{SelectSample, SelectCounter, SelectCommit, SelectMd, Misc, Vision}` and generic `mir_dispatch_args`, wired every Wave 6 Metal op onto `src/kernels/{mir_select,mir_misc,mir_vision}`, removed all remaining hand-written MSL from `elementwise.mm`, `public_reductions.mm`, `loss.mm`, `sample_logits.mm`, and `k_build_slot_mask` from `kernels.mm`, switched `deform_conv2d_forward`, `modulated_conv2d_{forward,backward}`, and `conv_transpose2d_forward` onto Metal's simdgroup GEMM + `MirKind::Vision`, and deleted 10 replaced `.mm` files.

### Wave 6 Metal Verification & LOC Delta
- **Parity & Full `ctest` Suite (Apple M2 Pro, Metal Release, `brass` `d5608f4`):**
  - Targeted Wave 6 suites (`sample_logits`, `sample_logits_parity`, `masked_diffusion`, `masked_mean_pool_parity`, `public_reductions`, `optim_parity`, `adam_parity`, `loss_parity`, `loss_fused_parity`, `batched_backward_parity`, `attention_moments`, `attention_moments_parity`, `bias_act`, `upfirdn2d`, `modulated_conv2d`, `filtered_lrelu`, `stylegan_parity`, `stylegan_primitives`, `deform_conv2d_parity`, `conv_transpose2d`, `conv_transpose2d_parity`, `gpu_mir_emit`, plus resample/pixel-shuffle suites): **29/29 passed** (`56.50 s`, `0` failures).
  - Full Metal `ctest` (`-LE gcstress -j2 --timeout 300`): **190/190 passed** (`341.10 s`, `0` failures; `backward_shader_audit` `60.27–60.49 s`).
- **LOC Delta (Metal Wave 6):**
  - Deleted 10 files (`-2,138` lines): `masked_diffusion.mm` (`-351`), `reduce.mm` (`-193`), `xavier_init.mm` (`-79`), `optim.mm` (`-72`), `adam.mm` (`-96`), `attention_moments.mm` (`-154`), `bias_act.mm` (`-298`), `upfirdn2d.mm` (`-210`), `modulated_conv2d.mm` (`-494`), `deform_conv2d.mm` (`-191`).
  - Added / rewritten / slimmed files (all `< 700` LOC): `mtl_mir_cache.{h,mm}` (`218` / `698`), `sample_logits.mm` (`491 -> 184`, absorbing `masked_diffusion_*`), `misc.mm` (`348`, new), `vision.mm` (`469`, new), `loss.mm` (`328 -> 218`), `elementwise.mm` (`496 -> 367`), `public_reductions.mm` (`179 -> 75`), `conv_transpose2d.mm` (`563 -> 495`), `kernels.mm` (`179 -> 165`).
  - **Net Metal LOC delta (Wave 6):** **`-2,007` net Metal LOC** (`src/metal/` `26,275 -> 24,268` LOC; **cumulative Waves 1–6: `-12,417` net Metal LOC** and 38 Metal files deleted).

