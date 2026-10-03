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
  pair behind `flash_attention_backward`, `_varlen_backward`,
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
  17.1 / 99.1 ms: 3.2-5.8x HIP, but 2.5 TF/s against the forward's 15: a
  cooperative-matrix version of the two passes is the next step if training
  throughput matters.
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

