# Compiler Consolidation Log: Brass Foundation & Radeon 8060S Vulkan Proof Gate

## Device & Environment
- **Device:** AMD Radeon 8060S Graphics (`RADV STRIX_HALO`)
- **Driver:** `radv Mesa 26.2.6-arch1.1`
- **Enabled JIT Capabilities:** `coopmat_f16=1`, `integer_dot_product=1` (`VK_KHR_shader_integer_dot_product`), `memory_model=1`, `subgroup_size_control=1` (`sg=32..64`)

## `brass` Compiler Foundation (`origin/main` at `484175d`)
- **`5ba4275`**: 128-bit (`i32x4`) and 16-bit (`u16`) shared memory, `ptx_subgroup_barrier`, `ptx_subgroup_id`, and `MslISel` modularization (`msl_builder.hpp`, `msl_isel.cpp`, `msl_isel_intrinsics.cpp`)
- **`16a7f03`**: `ptx_sdot4` (`OpSDotKHR` / `dp4a.s32.s32` / `_brass_sdot4`), `ptx_pack_f16x2`, `ptx_f16_hi_to_f32`, `ptx_ldexp_f32` across SPIR-V, PTX, and MSL
- **`e58ae6e`**: 16×16×16 cooperative matrix (`coop_alloc_{a,b,acc}`, `coop_fill_acc`, `coop_load_{a,b,acc}`, `coop_mma`, `coop_store_acc`, `coop_scale_acc`, `coop_add_acc`) across SPIR-V (`SPV_KHR_cooperative_matrix`), PTX (`wmma`), and MSL (`metal_simdgroup_matrix`)
- **`484175d`**: Complete `MslTarget` intrinsic parity (trig/extra math, `ptx_shfl_down_sync_f32`, `ptx_{i,u}64_to_f32`, shared `i64` load/store, global/shared atomics; explicit diagnostic on unsupported 64-bit atomic add); full `brass_unit_tests` **2031/2031 passed**, pushed to `origin/main`

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
