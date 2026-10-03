# Vulkan backend: range-safe BF16 (and FP32) operands in the matrix-core GEMM

The cooperative-matrix GEMM (`gemm_cm.comp`, docs/vulkan.md "Matrix
multiply") multiplies 16x16x16 **FP16** fragments. Until this change a BF16
operand was converted to FP16 as it was staged into shared memory, so any
BF16 value beyond FP16's ±65504 became inf. T5-XXL's FFN activations and
Sana's linear-attention summaries live out there, so brolm's T5 and
brodiffusion's Sana ran FP32 workarounds on the SIMT GEMM on Vulkan only
(brolm `t5.cpp` `f32_stream_`, brodiffusion `sana.cpp` `f32_core`), and the
T5-XXL encode was the one model slower on Vulkan than on HIP. This document
covers what the driver offers, the scaled-FP16 path built instead of a native
BF16 one, its numerics, and what it changed.

## What the driver exposes (Mesa RADV 26.2.3, Radeon 8060S, gfx1151)

`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` lists, all 16x16x16 at
subgroup scope:

| A x B | C / result |
|---|---|
| FP16 x FP16 | FP16, FP32 |
| UINT8 / SINT8 x UINT8 / SINT8 (all four mixes) | SINT32 (plain and saturating), UINT32 |

No BF16 entry and no `VK_KHR_shader_bfloat16` by default, although the
hardware has `V_WMMA_F32_16X16X16_BF16`. RADV has it behind an experimental
switch: with `RADV_EXPERIMENTAL=bfloat16` (formerly `RADV_PERFTEST`) the
driver adds `VK_KHR_shader_bfloat16` (`shaderBFloat16Type` and
`shaderBFloat16CooperativeMatrix` true, `shaderBFloat16DotProduct` false) and
two more 16x16x16 subgroup entries: BF16 x BF16 -> BF16 and BF16 x BF16 ->
FP32. Because it is opt-in per process and marked experimental, brotensor does
not build on it (a native BF16 variant of `gemm_cm.comp` is the natural
follow-up once RADV enables it by default; `glslc` 2026.3 accepts
`GL_EXT_bfloat16`). The probe: `vulkaninfo`, or the 30-line
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` dump this was measured
with.

## The scaled-FP16 path

**Per-row power-of-two scaling of A.** Before a GEMM whose A operand is BF16
(or FP32, below), `gemm_rowscale.comp` writes one int32 exponent per row of A
(per batch slice, once when A is broadcast with stride 0):

    e(m) = E - 141,  E = biased FP32 exponent of max_k |A(m, k)|

so `max_k |A(m, k)| * 2^-e` lies in [2^14, 2^15). The maximum is taken over
the finite elements' bit patterns (no float compares); an all-zero row gets
e = 0; e is clamped to [-100, 100]. The GEMM (`SCALE_A` specialisation)
reads each invocation's row exponents once, multiplies every A element by
2^-e as it converts it to FP16 at the shared-memory store (the one place a
conversion may happen, docs/vulkan.md "never consume a load where it is
issued"), and multiplies the FP32 accumulator of row m by 2^e in the epilogue,
before the bias, activation, GLU and accumulate. B is converted unscaled.

**Numerics.**

* Both multiplications are by powers of two, so they are exact. A BF16 value
  has 8 significant bits; FP16 has 11. Every element of a row within 2^31 of
  the row's maximum (2^-17 after scaling, FP16's subnormal resolution for an
  8-bit significand) therefore reaches the fragment **unrounded**: the
  scaled-FP16 GEMM computes exactly what a native BF16 x BF16 -> FP32 WMMA
  computes, up to FP32 summation order. Smaller elements round, and are 2^-31
  of the row's largest term.
* An FP32 A is rounded to FP16's 11 bits at that per-row scale: relative
  error 2^-12 per element whatever the row's magnitude (better than BF16's
  2^-9).
* A row holding inf or NaN keeps them (inf x 2^-e = inf) and its finite
  elements are still scaled, so the result is what exact arithmetic gives
  (the CPU's inf, not inf - inf = NaN from saturated neighbours).
* Rows that used to be in range get the same result as before, bit for bit,
  except where an element was an FP16 subnormal (now more precise): power-of-two
  scaling commutes with RNE and with the FP32 accumulation away from the
  range limits.
* **B must lie in FP16's range.** It is converted unscaled: scaling B per
  column would need a pass over the weight on every call, as expensive as the
  GEMM itself when M is small. Weights are in range by construction; the
  activation-by-activation products (attention scores Q K^T, P V, Sana's
  S Q^T) have the large operand on the A side in every model measured.
  Values past 65504 in B still become inf. The flash-attention kernels stage
  every BF16 operand this way, unscaled: the forward's Q / K / V (`fa_cm`)
  and the cooperative-matrix backward's Q / K / V / dO (`fa_bwd_cm`, whose
  P and dS also round to FP16); FP32 and `BROTENSOR_VK_NO_COOPMAT=1` keep
  the backward on its FP32 FMA kernel.
* FP16 A is not scaled (it cannot exceed the range), so the FP16 GEMM, its
  speed and its bits are unchanged.

**Which GEMMs take it.** Every `gemm()` call that reaches the
cooperative-matrix kernel with a BF16 A: `matmul`, `matmul_abt` (batched,
broadcast, padded strides), the whole linear family and its fused epilogues
(store, accumulate, GeGLU, SwiGLU, bias + activation), the backwards (the
transposed-A layout: 8 exponents per staged chunk), the INT8 / GGUF
quantised-B GEMMs with a BF16 A, and the projections inside the attention
ops. The GEMV kernel (M <= 8) and the SIMT kernel compute in FP32 and never
had the problem. `BROTENSOR_VK_NO_COOPMAT=1` still runs everything on SIMT.

**Two widenings of the cooperative-matrix kernel came with it,** because the
T5 encode then showed where its time went:

* **FP32 C from 16-bit operands** (`gemm_cm_f16_c32`, `gemm_cm_bf16_c32`):
  the "16-bit operands, FP32 result" form (bias in the operands' dtype, as
  `gemm_simt_*_c32`) used to run on SIMT. The attention ops project 16-bit X
  into FP32 Q / K / V this way (`ops_attention_proj.cpp`), so T5's (and every
  `self_attention_bias_forward` / `cross_attention_forward` / `mha_forward`
  caller's) projections were on the 3-8 TF/s kernel.
* **FP32 A against 16-bit B, opt-in** (`gemm_cm_f32a_w16`,
  `gemm_cm_f32a_wbf16`, `GemmArgs::round_a`): an FP32 A staged as scaled
  FP16 (above). Only for an A that is the op's own intermediate, today the
  attention ops' FP32 head output against Wo. The public FP32-activation ops
  (`linear_forward_batched` with FP32 X against 16-bit W) keep their
  "activations never rounded" contract and stay on SIMT.

**Cost.** One extra dispatch (and barrier) per GEMM, reading A once
(16-byte loads, 64 invocations per row). On the `--bench-gemm` shapes the
BF16 GEMM moves by -12% to +12% against the unscaled staging
(`BROTENSOR_VK_GEMM_NOSCALE=1`), the loss on the small 512-row shapes where
the dispatch is a larger share; the FP16 GEMM is unchanged (table below).

## Consumers switched back

* **brolm `t5.cpp`**: Vulkan takes the CUDA / HIP branch (weights re-cast to
  BF16, BF16 residual stream and FFN); the Vulkan-only FP32-activation stream
  (`f32_stream_`) and its scratch tensors are gone. `out` is BF16, as on HIP.
* **brodiffusion `sana.cpp`**: the linear-attention core runs in BF16 on every
  GPU again; the Vulkan-only FP32 copies of Q / K / V (`f32_core`) are gone.

## Results (Radeon 8060S, same process build, warm)

| | HIP | Vulkan, FP32 workaround | Vulkan, scaled BF16 |
|---|---:|---:|---:|
| T5-XXL encode in PixArt (2 passes) | 1.002 s | 1.865 s | **0.507 s** |
| T5-XXL forward, 24 tokens | 0.186 s | 0.246 s | 0.064 s |
| T5-XXL forward, 160 tokens | 0.30 s | 0.56 s | 0.155 s |
| T5-XXL forward, 507 tokens | 0.52 s | 1.26 s | 0.40 s |
| PixArt-Sigma 1024^2 step | 2.34 s | 0.899 s | 0.848 s |
| PixArt-Sigma, 20 steps end to end | 49.4 s | 20.6 s | 18.5 s |
| Sana 600M 1024^2 step | 0.173 s | 0.153 s | 0.148 s |

(T5 forwards: brolm `TextEncoder::forward` alone, third of three runs. The
long-sequence T5 gain is mostly the FP32-result projections leaving SIMT;
with only the BF16 scaling, 507 tokens took 0.72 s.)

**Accuracy.** T5-XXL, 24 tokens: per-row cosine against HIP's BF16 encoding
0.99977 worst (the FP32 workaround: 0.99976). On a natural 160-token prompt,
against the Vulkan FP32-activation encoding as the high-precision reference:
scaled BF16 0.900 worst row / 0.99917 mean, HIP's BF16 0.799 / 0.99851; Vulkan
against HIP 0.980 / 0.99971. A few rows of a BF16 T5 drift on any backend; the
scaled path drifts less than rocBLAS's BF16. PixArt-Sigma (seed 42, 20
steps, 1024^2): the Vulkan image against HIP's 27.7 dB PSNR (the FP32
workaround's: 26.3 dB), same picture. Sana: the scaled-BF16 image against the
FP32-core one 31.5 dB, visually identical; both differ from HIP's by the same
14.7 dB (HIP's render is hazier), a difference that predates this change and
lies outside the linear-attention core.

## Files

| File | Lines | What |
|---|---:|---|
| `src/vulkan/shaders/gemm_rowscale.comp` | 97 | the per-row exponent pre-pass (BF16 / FP32, plain or transposed A) |
| `src/vulkan/shaders/gemm_cm.comp` | 442 | `SCALE_A`; separate A / B / C / bias dtypes (`-DDTA/-DDTC/-DDTBI`); FP32 A loads (two 16-byte loads per chunk) |
| `src/vulkan/gemm.cpp` | 386 | kernel choice (`cm_shader`), the pre-pass, `round_a`, `set_gemm_scaling` |
| `src/vulkan/ops_attention_proj.cpp` | 351 | sets `round_a` on the Wo GEMM |
| `tests/test_vulkan_gemm_range.cpp` | 387 | the range tests (below) |

## Tests

`brotensor_test_vulkan --only=gemm` runs `test_vulkan_gemm_range.cpp`: BF16
A rows from 1e-6 to 3e7 (an outlier column per fourth row, one zero row)
against the CPU op in FP32 on the same BF16-rounded inputs, per output row
within 2e-5 of the row's largest output (1e-4 for the GLUs) plus two BF16
ulps of the element. Cases: `matmul_abt` with no bias and with bias + each of
the five activations, K % 8 != 0 (scalar loads), an unaligned A view, packed
heads, a broadcast B, a broadcast A, padded batch strides; the linear family
with all four epilogues at 3 (GEMV), 64 and 200 rows; `matmul` (NN); the
backward's dX and dW (transposed A); a row holding inf. Each on the
cooperative-matrix kernel and on SIMT (and the linear cases with GEMV
disabled). Direct `gemm()` cases cover FP32 C from FP16 / BF16 operands (with
a 16-bit bias and an activation, a wide BF16 A included) and an FP32 A
(rows 1e-6 .. 3e7, bits below BF16's) against FP16 / BF16 B: SIMT without
`round_a`, the cooperative-matrix kernel with it, 2e-3 of the row maximum
(measured 3.4e-4). One check turns the scaling off and asserts the same GEMM
overflows, so the inputs do exercise the range. Worst case across them: 0.49
of the tolerance (one BF16 output ulp at a rounding boundary); many are
bit-identical to the CPU.

The Vulkan-vs-HIP parity suite (`test_vulkan_hip_parity --only=gemm`,
removed with the HIP backend) added two large-magnitude BF16 cases
(rows 1e-4 .. 4e7, `linear_forward_batched_fp16` 120 x 4096 x 1024 and
`matmul_abt` 300 x 256 x 512) against HIP's rocBLAS BF16, each row divided by
its own maximum: 2.9e-3 and 2.4e-3 of the row maximum (tolerance 1.6e-2, one
BF16 ulp is 7.8e-3), the same distance each backend has from the CPU.
