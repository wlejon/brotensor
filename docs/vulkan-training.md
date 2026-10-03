# Vulkan backend: training ops

The Vulkan training surface (docs/vulkan.md is the backend's design; this is
the part of it that only training reaches). Every op is tested against the
CPU reference by `brotensor_test_vulkan --only=train`
(`tests/test_vulkan_train*.cpp`), and the generic CPU-vs-GPU suites run on
Vulkan as the `*_vulkan` ctest registrations.

## Chunk 8

* **Attention backwards** (`ops_attention_bwd.cpp`, `attn_bwd.comp`):
  `attention_backward`, `mha_backward` (+ the optional bias gradients),
  `self_attention_backward` and `cross_attention_backward`, the backwards of
  the materialised-probability forwards brogameagent's Attention / MHA /
  transformer layers train through. FP32, as the CPU. Each is FP32 GEMMs
  through `gemm()` (the per-head products are batched GEMMs over the head's
  columns of the (L, D) matrices, so no head split / merge copies) around one
  kernel, the row softmax backward in place over dP (one workgroup per
  (head, query) row; masked keys and gated query rows zero). dW / db
  accumulate through the GEMM's `EPI_ACCUM` and `column_sum_accumulate`; dX /
  dCtx are overwritten. Within 2e-5 of the largest output against the CPU.
* **Training BatchNorm** (`ops_gnorm.cpp`, `bnorm.comp`): `batch_norm_forward`
  (batch statistics with the CPU's two-pass variance, saved mean / rstd, the
  running-stat update with the unbiased variance) and `batch_norm_backward`
  (dGamma / dBeta accumulated), FP32, one workgroup per channel.
* **`bce_with_logits_fused_batched`** (`ops_xent.cpp`, `xent.comp` MODE 1):
  per-element sigmoid cross-entropy with a positive-class weight, the CPU's
  stable softplus / sigmoid forms with `exp_acc` and a log1p from `log_acc`,
  one workgroup per row for the per-sample loss.

## Chunk 9

* **Flash-attention backward** (`ops_fa_bwd.cpp`, `fa_bwd.comp`): one kernel
  pair (FMA; the cooperative-matrix version is the next item) behind `flash_attention_backward`, `_varlen_backward`,
  `_packed_qkv_backward` and the core of `_qkvo_backward`, FA-2 style and
  recompute-based (O is not read; the op table has no saved logsumexp, so
  pass 0 recomputes it). Pass 0, query-major (BR query rows of one head per
  workgroup): walks the rows' keys once for the online softmax statistics
  m, l and D = sum P dP (dP = dO . V, rescaled with l as the maximum moves,
  so O is never formed), stores (m, 1/l, D), then walks them again for
  dQ = scale sum dS K. Pass 1, key-major (BC keys of one head): walks the
  query blocks that can attend its keys (the causal / window / varlen range;
  packed bounds check each block) and accumulates dV = sum P dO and
  dK = scale sum dS Q. Per tile, Q / dO and K / V are staged in shared memory
  in their own encoding (16-bit stays 16-bit: exact, half the room), S and dP
  are computed by 128 invocations of RPT rows x 4 keys each (a row's lanes
  reduce through subgroup shuffles), P and dS go through shared memory to the
  per-thread accumulators. Everything is FP32 after the load; no atomics,
  deterministic. Each query row's keys follow the forward's modes
  (`fa_common.glsl`: causal, window, key mask, varlen, packed bounds). Tiles
  (BR, BC) keep the shared memory under 64 KiB: 16-bit (32, 32) up to head
  width 128, (32, 16) up to 256; FP32 (32, 32) to 64, (32, 16) to 128 and
  (16, 8) with 32 invocations to 256. Wider than HIP: FP32 / FP16 / BF16 on
  every entry point (HIP's bare backward takes 16-bit only). The QKVO
  backward composes the projections (`linear_forward_batched_ex`,
  `flash_attention_forward`, `linear_backward_batched`) around the core.
  Measured with `--bench-attention` (`BROTENSOR_VK_BENCH_PART=bwd`, FP16,
  10 L^2 hd H flops, halved for causal), Vulkan vs HIP's same public op (its
  row path; HIP has no FP32-output tensor-core GEMM): L 512 x 8 heads x 64
  0.59 / 3.20 ms, L 1024 x 16 x 64 3.99 / 22.6 ms (2.7 / 0.48 TF/s), causal
  2.18 / 12.0 ms, L 2048 x 8 x 128 causal 14.0 / 45.0 ms, L 2048 x 16 x 64
  17.1 / 99.1 ms: 3.2-5.8x HIP, but 2.5 TF/s against the forward's 15. This
  kernel is now the fallback (FP32, heads wider than 128, no cooperative
  matrix, `BROTENSOR_VK_NO_COOPMAT=1`); 16-bit takes the one below.

* **Flash-attention backward on cooperative matrices** (`fa_bwd_cm.comp`,
  same host file): FP16 / BF16 on a `coopmat_f16` device, head width up to
  128, every entry point and mode (rows / causal / key mask, varlen, packed
  bounds and windows) through the same per-row intervals. Three dispatches,
  all 16x16x16 FP16 fragments with FP32 accumulation and subgroup 32:
  *statistics* (query-major: S = Q K^T, dP = dO V^T per key block, the online
  m, l, D recurrence; stores lse = m + log2 l and D), *dQ* (query-major: S,
  dP again, dS = P (dP - D) with P = 2^(S scale - lse) elementwise in FP32 on
  the accumulator fragments, stored once to shared memory as FP16, dQ += dS K)
  and *dK / dV* (key-major: S^T = K Q^T and dP^T = V dO^T computed directly
  in the transposed orientation, so P^T and dS^T leave the elementwise step
  as A operands: dV += P^T dO, dK += dS^T Q). That is nine 16-wide GEMM
  sweeps of the score matrix (2 + 3 + 4) against the five the flop count
  credits: there is no saved logsumexp in the op table, and dQ stays
  deterministic (no atomics). Splitting the statistics from dQ (one
  query-major kernel before) freed registers: both had sat at RADV's 256-VGPR
  cap, and the split alone took the L 2048 x 16 x 64 total from 6.2 to 5.2 ms.
  Element positions: KHR cooperative matrix does not say which lane holds
  which element, so an "index" fragment (16 row + col, `fa_cm`'s row-index
  trick) gives each element's row and column. When it shows the clustered
  layout (8 elements per lane, element i in column lane % 16, a row's 16
  elements in one aligned 16-lane cluster: RADV's WMMA layout, checked at run
  time, not assumed), row maxima and sums are `subgroupClustered*` reductions
  on registers and each lane keeps its rows' intervals and statistics in
  registers; otherwise a layout-agnostic path stages the statistics' S / dP
  through shared memory (two lanes per row) and reads per-element data from
  shared memory (the tests force it through an override). Streamed blocks are
  prefetched into registers before the current block's math (`fa_cm`'s
  rule); the own rows' A fragments stay in registers up to head width 64, and
  then the streamed blocks reuse their staging room. Tiles (subgroups x
  streamed rows; statistics / dQ / dK-dV): hd <= 64 (2 x 16) / (2 x 16) /
  (4 x 16); wider (4 x 32) / (4 x 16) / (4 x 16) with dQ's and dK-dV's A
  fragments read from shared memory (in registers next to 128-wide
  accumulators they spill); shared memory 9-54 KiB.
  `BROTENSOR_VK_FA_BWD_CFG=nsg,bn,areg x 3` forces them,
  `BROTENSOR_VK_FA_BWD_PATH=fma|cm` picks a kernel. Head widths 129-256 stay on
  FMA: the dK and dV accumulators alone would be 256 VGPRs. BF16 is staged as
  FP16 unscaled like the forward's Q / K / V (docs/vulkan-bf16.md: inputs
  past +-65504 become inf), and P and dS are rounded to FP16 as A operands
  (|dS| <= 2 max|dP|; FP16 dO and V products past 65504 would overflow where
  the FMA kernel's FP32 does not). Errors match the FMA kernel's (FP16
  max 8.2e-4 / mean 4.2e-4 of the largest gradient over the parity cases,
  BF16 7.6e-3 / 2.9e-3). Measured (`BROTENSOR_VK_BENCH_PART=bwd`, FP16, RADV
  on the Radeon 8060S, idle GPU; FMA = this chunk's kernel forced):

  | Shape | FMA ms / TF/s | coopmat ms / TF/s | Speed-up |
  |---|---:|---:|---:|
  | L 512 x 8 x 64 | 0.646 / 2.08 | 0.182 / 7.4 | 3.5x |
  | L 1024 x 16 x 64 | 4.56 / 2.35 | 1.38 / 7.8 | 3.3x |
  | L 1024 x 16 x 64 causal | 2.49 / 2.15 | 0.78 / 6.9 | 3.2x |
  | L 2048 x 8 x 128 causal | 13.9 / 1.55 | 3.6 / 5.9 | 3.9x |
  | L 2048 x 16 x 64 | 17.2 / 2.50 | 4.85 / 8.9 | 3.5x |
  | L 1024 x 16 x 128 | 13.2 / 1.63 | 2.70 / 7.9 | 4.9x |

  Per dispatch at L 2048 x 16 x 64: statistics 1.6 ms, dQ 1.4, dK / dV 1.9,
  i.e. 9-12 TF/s of executed matrix work each, the forward's efficiency; the
  remaining gap to the forward's TF/s is the recompute (9 sweeps for 5).
* **GroupNorm backward** (`ops_gnorm.cpp`, `gnorm.comp` `GN_BWD_*`): the
  forward's chunked statistics, then one workgroup per (channel, sample) for
  sum dY and sum dY xhat, dX per tile from the group's sums, and the
  parameter gradients summed over the batch per channel (FP32, one rounding,
  accumulated): deterministic, where CUDA / HIP use float atomics.
* **ResBlock backward** (`ops_diffusion.cpp`): the CPU / CUDA composition
  (recompute GN1 / SiLU / conv1 + shift / GN2 / SiLU, then the conv, SiLU and
  GroupNorm backwards and the skip path) over the Vulkan ops, all three
  dtypes (CUDA: 16-bit only).
* **Scatter-adds** (`ops_scatter.cpp`, `scatter_add.comp`):
  `scatter_rows_add` and `embedding_lookup_backward` sort the (row, m) pairs
  as 64-bit keys (bitonic: in shared memory up to 512 pairs, global
  compare-exchange steps beyond) and sum each destination row's run of
  source rows in increasing m, FP32 from the row's current value, one
  rounding into the dtype. Chosen over `VK_EXT_shader_atomic_float` (which
  RADV has) because the result must not depend on scheduling: FP32 now
  matches the CPU reference bit for bit, and 16-bit results are reproducible
  run to run (CUDA / HIP atomics are neither). An index outside [0, R) is
  skipped.
* **conv_transpose2d backwards** (`ops_conv.cpp`): the input gradient is
  `conv2d` of dY with Wt read as OIHW weights (C_in outputs, C_out / groups
  inputs), the weight gradient `conv2d_backward_weight` of that convolution
  with X as its output gradient (accumulated); an output_padding that
  reaches a stride (output_padding < dilation) crops / zero-pads. Both run
  on the implicit-GEMM paths. All three dtypes (CUDA: FP32 only).
* **Resample / pad / pool backwards** (`ops_spatial_bwd.cpp`,
  `resample_bwd.comp`): gathers, one invocation per input element summing
  the outputs that read it (nearest / bilinear interpolation with the
  forward's exact rational coordinates and a candidate window around
  h out / in; adaptive-pool windows; pad's interior, reflected positions and
  replicated edge runs). `max_pool2d_backward` takes only the argmax indices
  (no window geometry), so one invocation per (n, c) plane walks its outputs
  in order into an FP32 plane: the CPU's order, bit-identical in FP32.
* **LSTM** (`ops_lstm.cpp`, `lstm.comp`), FP32 as on CUDA / HIP: X W_ih^T +
  b_ih for all steps in one GEMM, then per step an accumulating
  h_{t-1} W_hh^T GEMM and the pointwise cell; the backward walks the steps in
  reverse (pointwise gradient, dh_{t-1} = dZ_t W_hh), then one GEMM each for
  dX, dW_ih, dW_hh and the bias column sums. 2T dispatches per direction.

Worst errors against the CPU (FP32 reference on the dtype-rounded inputs,
max |diff| / max |want| over every test shape, `--only=train`): FP32 below
1e-5 everywhere (scatter-adds and max-pool backward bit-identical); FP16
4.3e-4 (flash backwards), 1.3e-3 (ResBlock backward, a long 16-bit chain);
BF16 3.6e-3 (flash), 8.4e-3 (ResBlock): at most ~1.4 ulp of the dtype.

Two contract fixes found by running the generic suites on Vulkan:
`segment_softmax_stats` writes its (S, 4) features in the logits' dtype (it
wrote FP32; CUDA / HIP and brolm's Laya batch, which copies them into a
16-bit buffer, use the logits' dtype), and 16-bit `self_attention_forward`
takes the flash route with the mask on keys only, as CUDA / HIP (FP32 stays
`mha_forward`, which also gates query rows).

