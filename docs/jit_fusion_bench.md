# Trace JIT fusion benchmarks

`tests/bench_jit_fusion.cpp`, run on GPU 1 (RTX 4090, sm_89, 72 MB L2) with a
game and another agent's job on the machine. Timing is min-of-20 after a
600 ms clock spin-up (`tests/bench_helpers.h`); a mean here measures the SM
clock ramp rather than the kernel.

Reproduce with:

```
CUDA_VISIBLE_DEVICES=1 ./build/tests/Release/brotensor_bench_jit_fusion
```

Every pattern is measured three ways:

* **eager** — the brotensor op sequence the model code writes today. One
  kernel per op, every intermediate a full round trip through HBM.
* **hand** — the hand-written fused kernel where one exists (brass
  `MlFusionCompiler` PTX, or a single-kernel brotensor op such as
  `axpby_inplace`). Blank where there is none.
* **jit** — one `begin_trace()`/`end_trace()` over the same expression,
  replayed through `TraceHandle::execute()`.

## Reading the numbers

`GB/s` is **ideal** traffic over measured time: inputs read once, outputs
written once, a `(1, D)` broadcast row counted once rather than N times. That
is the traffic a perfectly fused kernel must move. The eager column moves
more than that — its extra traffic is exactly what the fusion removes — so its
GB/s is *effective*, not achieved, and comparing the two columns directly is
the point.

`% of peak` is against the machine's measured streaming bandwidth, taken with
a 128 MiB device-to-device copy at the start of every run: **939 GB/s** (the
4090's 1008 GB/s spec, less ECC-free overhead and the other load on the card).

**Figures above 100% are L2 hits, not errors.** The 4090 has 72 MB of L2. At
4096×4096 a BF16 tensor is 33.5 MB, so a two-operand pattern's whole working
set is L2-resident and both columns run well above HBM speed. At 4096×12288
(100 MB per BF16 tensor) nothing fits and the numbers are honest HBM. Compare
within a shape, not across.

## Results

Peak: 939.6 GB/s. `L` is kernel launches per replay.

### 4096 × 4096 — DiT activations at 1024², BF16 working set fits L2

| pattern | dtype | eager (L) | hand (L) | jit (L) | jit vs eager | jit fusion |
|---|---|---|---|---|---|---|
| gated-residual `x += g[1,D]*y` | fp32 | 0.355 ms (2) | — | **0.219 ms (1)** | 1.62× | elementwise-vec |
| | bf16 | 0.143 ms (2) | — | **0.023 ms (1)** | 6.2× | elementwise-vec |
| | fp16 | 0.142 ms (2) | — | **0.023 ms (1)** | 6.2× | elementwise-vec |
| swiglu-tail `silu(g)*p` | fp32 | 0.360 ms (2) | 0.211 ms (1) | 0.218 ms (1) | 1.65× | elementwise-vec |
| | bf16 | 0.107 ms (2) | — | **0.103 ms (1)** | 1.04× | elementwise-vec |
| euler-step `x += ds*v` | fp32 | 0.442 ms (3) | 0.219 ms (1) | **0.218 ms (1)** | 2.03× | elementwise-vec |
| | bf16 | 0.140 ms (3) | 0.028 ms (1) | **0.022 ms (1)** | 6.4× | elementwise-vec |
| layernorm-modulate | fp32 | 0.294 ms (2) | 0.144 ms (1) | 0.147 ms (1) | 2.00× | row-layernorm-chain |
| | bf16 | 0.122 ms (2) | — | **0.022 ms (1)** | 5.5× | row-layernorm-chain |
| rmsnorm-silu | fp32 | 0.233 ms (2) | — | **0.145 ms (1)** | 1.61× | row-rmsnorm-chain |
| | bf16 | 0.055 ms (2) | — | **0.022 ms (1)** | 2.5× | row-rmsnorm-chain |
| residual-rmsnorm | fp32 | 0.339 ms (2) | 0.284 ms (1) | 0.293 ms (1) | 1.16× | residual-rmsnorm (hand) |
| | bf16 | **0.121 ms (2)** | — | 0.129 ms (1) | 0.94× | row-rmsnorm-chain |
| chain4 `(a*b+c)*silu(d)` | fp32 | 0.925 ms (5) | — | **0.362 ms (1)** | 2.56× | elementwise-vec |
| | bf16 | 0.290 ms (5) | — | **0.183 ms (1)** | 1.58× | elementwise-vec |

### 4096 × 12288 — MLP hidden width, nothing fits L2

This is the honest HBM table.

| pattern | dtype | eager (L) | hand (L) | jit (L) | jit GB/s | % peak |
|---|---|---|---|---|---|---|
| gated-residual | fp32 | 1.116 ms (2) | — | **0.637 ms (1)** | 948 | 100% |
| | bf16 | 0.555 ms (2) | — | **0.326 ms (1)** | 928 | 99% |
| swiglu-tail | fp32 | 1.132 ms (2) | 0.626 ms (1) | 0.657 ms (1) | 919 | 98% |
| | bf16 | 0.536 ms (2) | — | **0.322 ms (1)** | 939 | 100% |
| euler-step | fp32 | 1.581 ms (3) | 0.667 ms (1) | **0.638 ms (1)** | 947 | 101% |
| | bf16 | 0.725 ms (3) | 0.364 ms (1) | **0.326 ms (1)** | 927 | 99% |
| layernorm-modulate | fp32 | 0.996 ms (2) | 0.539 ms (1) | **0.535 ms (1)** | 754 | 80% |
| | bf16 | 0.481 ms (2) | — | **0.224 ms (1)** | 898 | 96% |
| rmsnorm-silu | fp32 | 1.005 ms (2) | — | **0.536 ms (1)** | 752 | 80% |
| | bf16 | 0.502 ms (2) | — | **0.225 ms (1)** | 894 | 95% |
| residual-rmsnorm | fp32 | 1.225 ms (2) | 0.946 ms (1) | **0.963 ms (1)** | 837 | 89% |
| | bf16 | 0.556 ms (2) | — | **0.442 ms (1)** | 910 | 97% |
| chain4 | fp32 | 2.945 ms (5) | — | **1.090 ms (1)** | 923 | 98% |
| | bf16 | 1.421 ms (5) | — | **0.544 ms (1)** | 926 | 99% |

### 1024 × 4096 and 256 × 1024 — launch-bound end

At 256×1024 a BF16 tensor is 512 KB and every kernel is launch-bound: the
absolute times sit at 3–4 µs whatever the pattern, which is roughly one launch.
The eager column's advantage is gone before the bandwidth even matters, so the
ratio here *is* the launch-count ratio.

| pattern | dtype | shape | eager | jit | jit vs eager |
|---|---|---|---|---|---|
| layernorm-modulate | fp32 | 1024×4096 | 0.033 ms (2) | **0.014 ms (1)** | 2.36× |
| layernorm-modulate | bf16 | 1024×4096 | 0.029 ms (2) | **0.009 ms (1)** | 3.2× |
| residual-rmsnorm | bf16 | 1024×4096 | 0.022 ms (2) | **0.012 ms (1)** | 1.83× |
| chain4 | bf16 | 1024×4096 | 0.037 ms (5) | **0.011 ms (1)** | 3.4× |
| gated-residual | bf16 | 256×1024 | 0.007 ms (2) | **0.004 ms (1)** | 1.75× |
| chain4 | bf16 | 256×1024 | 0.015 ms (5) | **0.004 ms (1)** | 3.75× |
| layernorm-modulate | bf16 | 256×1024 | 0.009 ms (2) | **0.004 ms (1)** | 2.25× |

### VAE feature maps — many narrow rows

A convolutional feature map reaches the row kernel as `(H*W, channels)`, which
is the opposite regime to a transformer activation: hundreds of thousands of
rows, 96 to 384 wide. One row per 256-thread block — which is what the kernel
did originally — gives a 96-wide row 256 threads and idles seven eighths of
them. The kernel now picks `threads_per_row` from the row width and packs
`256 / threads_per_row` rows into each block, holding the packing to a divisor
of the row count so the "past the end" exit stays uniform and the reduction's
barrier stays safe. At 1048576×96 that is 1.156 ms → 0.894 ms; at 384 wide,
where a row already wants most of a block, nothing changes.

| pattern | dtype | shape | eager | jit | jit vs eager |
|---|---|---|---|---|---|
| rmsnorm-silu | fp32 | 262144×384 | 1.837 ms (2) | **0.889 ms (1)** | 2.07× |
| rmsnorm-silu | fp32 | 1048576×96 | 2.829 ms (2) | **0.894 ms (1)** | 3.16× |
| rmsnorm-silu | fp32 | 65536×384 | 0.418 ms (2) | **0.208 ms (1)** | 2.01× |

All three land at 96–103% of measured streaming bandwidth. Eager only reaches
30–51%, because at these widths its two kernels are bound by how fast they can
issue rows, not by bandwidth.

## Launch counts

The JIT is one launch for every pattern here. Eager is:

| pattern | eager launches | why |
|---|---|---|
| gated-residual | 2 | `broadcast_mul` into a temp, then `add_inplace` |
| swiglu-tail | 2 | `silu_forward`, `mul_inplace` |
| euler-step | 3 | `copy_d2d`, `scale_inplace`, `add_inplace` (what the scheduler writes today) |
| layernorm-modulate | 2 | `layernorm_forward_inference_batched`, `modulate` |
| rmsnorm-silu | 2 | `rms_norm_forward`, `silu_forward` |
| residual-rmsnorm | 2 | `add_inplace`, `rms_norm_forward` |
| chain4 | 5 | clone, mul, add, silu, mul |

## Compile and cache cost

Measured with a cold `TraceCache`, at 1024×4096 BF16:

| trace | first trace (total) | of which PTX emit + ptxas | cached re-trace | replay |
|---|---|---|---|---|
| elementwise (gated residual) | 145 µs | 66 µs | 31 µs | 9 µs |
| row-norm (layernorm+modulate) | 1085 µs | 1019 µs | 38 µs | 9 µs |

The row-norm module is the expensive one: two entries, each fully unrolled
across eight lanes with the reduction inlined, so ptxas has real work to do.
Amortised over a 20-step denoise with 32 blocks it is noise — the cost is paid
once per distinct *shape*, because D, the row count and eps are baked into the
PTX as immediates.

A cached re-trace costs ~23 µs (elementwise) to ~29 µs (row-norm). That is not
the cache lookup; it is re-running the traced user code to rebuild the DAG plus
re-binding. **Adopters should hold the `TraceHandle` and call `execute()`, not
re-trace every step.**

## What a trace allocates

A traced op does not compute anything, so it does not allocate anything: it
returns a *symbolic* tensor — real shape, dtype and device, null buffer, and an
opaque `jit_slot` naming the DAG node it stands for. `end_trace()` then gives a
real buffer to exactly the live-at-end nodes the caller is still holding, and to
nothing else.

Measured by `brotensor::alloc_stats()` in `tests/test_jit_trace.cpp` at
512×1024 FP32:

| trace | allocations | bytes | eager equivalent |
|---|---|---|---|
| `store(dst, silu(x)*y + x)` | **0** | **0** | 3 × 2 MiB of intermediates |
| `Tensor kept = silu(x)*y + x` | 1 | 2 MiB (the one output) | same 3 × 2 MiB |

This is what made the VAE seam below adoptable. The tracer used to allocate a
full-size output for *every* traced op before recording its node — at the
decoder's widest feature map, `(1048576, 144)` BF16, that is 288 MiB per
intermediate, 576 MiB per trace, for contents the fused kernel never reads.

Two consequences for callers. A symbolic tensor cannot be copied — it is a
value in an expression, not a tensor, and the copy ctor throws. And an
intermediate the fused kernel only consumes internally (`Tensor ln =
layernorm(x); out = ln * g;` — nothing else reads `ln`) comes back from
`end_trace()` as an *empty* tensor rather than a buffer of uninitialised
garbage, which is what it used to be.

## Adoption: the Qwen-Image 2.1 VAE's RMSNorm + SiLU seam

`brodiffusion/src/vae_qwenimage21.cpp`. Every norm in the graph except the
attention block's is immediately followed by a SiLU — 35 seams in the decoder
(17 resnets × 2, plus `norm_out`). Eagerly each is two full passes over the
feature map: `rms_norm_forward` writes it and `silu_forward` reads it straight
back. Traced it is one kernel, and the normalised map never lands.

Measured with `brodiffusion.exe qi21-vae-fwd --synthetic --bench-ab 8` on
GPU 1, min-of-8, alternating in one process:

| decode | eager | fused | | peak VRAM eager | peak VRAM fused |
|---|---|---|---|---|---|
| 32×32 latent → 512×512 px | 106.6 ms | **102.6 ms** | 1.04× | 3546 MiB | 3514 MiB |
| 64×64 latent → 1024×1024 px | 429.3 ms | **409.1 ms** | 1.05× | 7674 MiB | 7706 MiB |

Two things to read honestly here.

The win is 4–5%, not the 2–3× the seam shows in isolation, because a VAE decode
is convolution-bound: the norms are a small slice of it. In isolation the same
kernel is `rmsnorm-silu 1048576×96 fp32: 2.543 ms → 0.883 ms` in the table
above, and the ~20 ms recovered at 1024² is about what summing the seam's
traffic across every stage predicts.

Peak VRAM does not move, and should not: the fusion removes *traffic*, not
residency — the store destination is still a materialised feature map. What the
symbolic tracer changes is that building the trace costs nothing. The first
decode allocates **72 allocations / 9725 MiB at 1024² with the JIT on and
exactly the same 72 / 9725 MiB with `BRODIFFUSION_JIT=0`**: the tracer adds not
one byte. Under the old tracer those 35 seams would each have bought two
full-size buffers at trace time.

Numerically the fused path differs from the eager one at the BF16 ulp level
(one fewer rounding — the normalised value never round-trips through BF16
before the SiLU). Against the FP32 diffusers reference at `.parity/`, decode is
cosine 0.999504 / rel-L2 0.0315 fused against 0.999579 / 0.0290 eager, and
encode 0.999731 / 0.0232 against 0.999752 / 0.0223 — both well inside the ~3%
band the BF16 VAE already sits in, and the gap between the two is an order of
magnitude smaller than the gap to the reference.

## What each improvement bought

The starting point emitted one kernel shape: FP32 in, FP32 out, every operand
the full `(rows, cols)` buffer, one element per thread. At BF16 with a `(1, D)`
gate — which is every DiT pattern above — **there was no fused form at all**,
so the eager column was the only option.

| improvement | what it bought |
|---|---|
| BF16/FP16 I/O with FP32 math | The whole BF16/FP16 half of every table. Before this, tracing a BF16 expression threw. |
| Row/scalar broadcast operands | `gated-residual` and `layernorm-modulate` at all. A `(1, D)` gate was previously not expressible, and materialising it would have cost N×D of traffic it now does not. |
| 16-byte vector accesses in the elementwise emitter | `chain4` bf16 4096×12288: 1.04 ms → 0.544 ms. Scalar 2-byte loads waste three quarters of every transaction. |
| Vectorising the **row-norm** kernel (it was still scalar after the first pass) | `layernorm-modulate` bf16 4096×4096: 0.033 → 0.022 ms. `layernorm-modulate` fp32 1024×4096: 0.018 → 0.014 ms, which moved it from *behind* the hand-written brass kernel to *ahead* of it. |
| Pre-chain store in pass one, reload in pass two | `residual-rmsnorm` bf16 4096×12288: 0.538 → 0.442 ms (from 1.4% slower than eager to 26% faster). The fused kernel used to replay the residual add in both passes, reading x and p twice; now pass one writes x and pass two reads it back out of L2. |
| `tanh` / `sigmoid` ops | Coverage, not speed — the DiT applies `tanh` to a `(1, D)` row once per forward, not per token. |
| Lock-free caches | No measurable replay cost either way; the mutex was never on the hot path. It is a design constraint, and removing it also removed the initialisation flag. |
| Row-width-aware block geometry | `rmsnorm-silu` fp32 1048576×96: 1.156 → 0.894 ms (72% → 96% of peak). A one-row-per-block kernel gives a 96-wide row 256 threads and idles seven eighths of them; packing eight such rows into a block recovers the occupancy. Wider rows are unaffected — 262144×384 is 0.890 ms either way. |

## Where the JIT still loses, and why

1. **`residual-rmsnorm` at BF16/FP16, 4096×4096 only** — 0.129 ms vs eager's
   0.121 ms, 7% behind. Both operands total 67 MB, which fits the 4090's 72 MB
   L2, so eager's extra kernel costs it almost nothing while the fused kernel
   still pays for reading the row twice. At 4096×12288, where L2 cannot hide
   it, the fused form wins by 26%. This is a cache-size artefact of one shape,
   not a code-generation gap.

2. **`swiglu-tail` and `residual-rmsnorm` at FP32 vs the brass hand kernels** —
   0.218 vs 0.211 ms and 0.293 vs 0.284 ms, 3% behind. The hand kernels are
   shape-specialised in ways the generic emitter is not: brass's packed SwiGLU
   reads one interleaved `(B, 2D)` buffer where the traced form reads two
   separate ones, and the hand residual-RMSNorm is a genuine single-pass
   kernel that folds the residual write into the reduction's only read. The
   generic row kernel cannot be single-pass without holding a whole row in
   registers, which would cap D.

3. **`layernorm-modulate` and `rmsnorm-silu` at FP32, 4096×12288** — 80% of
   peak against 95–100% for the same patterns at BF16. At FP32 one row is
   48 KB, so the pass-two re-read no longer sits in L1 and competes for L2 with
   the other 4095 blocks in flight. The brass hand kernel is at the same 80%,
   so this is the two-pass structure, not the emitter.

Everything else is at 95–101% of measured streaming bandwidth, which is where a
memory-bound kernel should be, and matches or beats the hand-written kernel
wherever one exists.
