# Vulkan vs HIP: performance summary

Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, 40 CUs, LPDDR5X 256 GB/s),
Arch Linux, Mesa RADV 26.2.3 (ACO), ROCm 7.2.4. Both backends in one build
(`build_vk`); "HIP" is brotensor's HIP backend (hipBLAS for dense FP16 GEMM),
"Vulkan" its Vulkan backend (docs/vulkan.md). Ratios are HIP time / Vulkan
time, or Vulkan / HIP throughput: above 1, Vulkan is faster. Chunk 8,
2026-10-03.

## Models (end to end, sibling code, same weights)

| Model / step | HIP | Vulkan | HIP / Vulkan |
|---|---:|---:|---:|
| Qwen3-0.6B Q8_0 decode (brolm) | 97 tok/s | 192 tok/s | 1.98 |
| Qwen3-0.6B Q8_0 prefill (brolm) | 1.3k tok/s | 15.9k tok/s | 12.2 |
| SD 1.5 UNet step (brodiffusion) | 0.186 s | 0.062 s | 3.00 |
| PixArt-Sigma, 20 steps | 49.1 s | 19.6 s | 2.51 |
| Sana step | 0.173 s | 0.153 s | 1.13 |
| TripoSplat (image to splat) | 86 s | 52 s | 1.65 |
| Depth-Anything-V2 small (brovisionml) | 47 ms | 23 ms | 2.04 |
| SAM-B mask decode | 11.4 ms | 4.0 ms | 2.85 |
| Qwen3-TTS (brosoundml) | 17.6 s | 5.7 s | 3.09 |
| Whisper-tiny transcription | 0.90 s | 0.31 s | 2.90 |
| Qwen3-ASR | 7.7 s | 2.3 s | 3.35 |
| T5-XXL encode (brolm, PixArt's text encoder) | 0.93 s | 1.9 s | **0.49** |

Measured in chunks 6-7b on the siblings' own benchmarks / tests; the STFT
change of chunk 8 (below) adds ~3.4 ms per 30 s of audio to the Whisper /
Qwen3-ASR front ends (~1% of those rows).

## Kernels (`brotensor_test_vulkan --bench-*`, chunk 8 run)

HIP columns are measured live in the same process (`build_vk` has both
backends), except the dense GEMM, whose HIP column is hipBLAS from the
vk-spike run 2 (brotensor's HIP backend calls hipBLAS for that op).

**Dense FP16 GEMM** (`--bench-gemm`, `matmul_abt`, TF/s)

| Shape | M x N x K | hipBLAS | Vulkan | Vulkan / HIP |
|---|---|---:|---:|---:|
| square4k | 4096^3 | 22.7 | 27.1 | 1.19 |
| square2k | 2048^3 | 38.8 | 29.1 | 0.75 |
| square8k | 8192^3 | 25.2 | 19.4 | 0.77 |
| Qwen0.6B up, pp512 | 512 x 3072 x 1024 | 28.8 | 24.0 | 0.83 |
| Qwen0.6B down, pp512 | 512 x 1024 x 3072 | 33.8 | 25.0 | 0.74 |
| 8B q/o, pp512 | 512 x 4096 x 4096 | 20.6 | 22.6 | 1.10 |
| 8B up, pp512 | 512 x 12288 x 4096 | 21.6 | 21.9 | 1.01 |
| 8B down, pp512 | 512 x 4096 x 12288 | 16.2 | 18.9 | 1.17 |
| DiT 3072, L4096 | 4096 x 3072 x 3072 | 36.4 | 32.2 | 0.88 |
| DiT MLP, L4096 | 4096 x 12288 x 3072 | 32.5 | 32.8 | 1.01 |

Other Vulkan GEMM paths (TF/s, square4k / DiT 3072): BF16 cooperative matrix
26.9 / 30.6, FP16 SIMT 7.9 / 8.1, FP32 SIMT 6.7 / 8.3. FP16 GEMV (decode, W
read): 195-239 GB/s at B = 1-8 on 4096-12288-wide weights (peak 256).

**Flash attention, FP16** (`--bench-attention`, ms)

| Shape | HIP | Vulkan | HIP / Vulkan |
|---|---:|---:|---:|
| bidirectional L542 h48 d128 | 3.03 | 0.517 | 5.85 |
| bidirectional L1024 h32 d128 | 6.10 | 1.11 | 5.49 |
| bidirectional L4096 h24 d128 | 57.7 | 13.1 | 4.42 |
| bidirectional L4096 h16 d64 | 32.8 | 7.05 | 4.66 |
| bidirectional L4096 h24 d64 | 45.6 | 6.94 | 6.57 |
| bidirectional L2048 h8 d256 | 10.0 | 3.93 | 2.55 |
| causal L1024 h32 d128 | 86.2 | 0.652 | 132 |
| causal L4096 h24 d128 | 1144 | 9.59 | 119 |
| causal L4096 h16 d64 | 301 | 6.07 | 49.5 |
| decode 1 x 16/8/128, 4096 keys | 0.741 | 0.0419 | 17.7 |
| decode 1 x 16/8/128, 32768 keys | 9.84 | 0.598 | 16.5 |
| decode 64 x 16/8/128, 16384 keys | 20.7 | 0.609 | 34.0 |

Vulkan bidirectional prefill: 9.8-15.8 TF/s (the spike's shader: 0.77-1.33x
of it). HIP's causal path does not skip masked tiles.

**Convolution and spatial ops, FP16** (`--bench-conv`, throughput)

| Op | HIP | Vulkan | Vulkan / HIP |
|---|---:|---:|---:|
| conv2d VAE 512 ch 64^2 (TF/s) | 17.2 | 24.4 | 1.42 |
| conv2d VAE 128 ch 512^2 | 6.45 | 18.6 | 2.88 |
| conv2d VAE 256 -> 128 ch 512^2 | 6.79 | 20.1 | 2.95 |
| conv2d UNet 320 ch 64^2 b2 | 14.0 | 18.0 | 1.28 |
| conv2d UNet 1280 ch 16^2 b2 | 19.6 | 21.1 | 1.07 |
| GroupNorm 512 ch 128^2 (GB/s) | 56 | 544 | 9.71 |
| GroupNorm 128 ch 512^2 | 32 | 135 | 4.16 |
| nearest 2x 256 ch 256^2 | 129 | 186 | 1.45 |
| bilinear 256 ch 100 -> 333 | 21 | 149 | 7.15 |

**Quantised linears** (`--bench-quant`; decode GB/s of weight read, prefill TF/s)

| Op, shape | HIP | Vulkan | Vulkan / HIP |
|---|---:|---:|---:|
| INT8 W8A16 decode, 8B q/o, B1 | 36.9 | 720 | 19.5 |
| INT8 W8A16 decode, 8B down, B8 | 17.0 | 207 | 12.1 |
| INT8 W8A16 prefill, 8B gate/up M512 | 1.48 | 22.4 | 15.2 |
| Q8_0 decode, 0.6B down, B1 | 486 | 478 | 0.98 |
| Q8_0 decode, 8B gate/up, B1 / B8 | 224 / 29.0 | 226 / 201 | 1.01 / 6.94 |
| Q8_0 prefill, 8B gate/up M512 | 0.44 | 26.4 | 60.1 |
| Q4_K decode, 8B gate/up, B1 / B8 | 216 / 27.0 | 511 / 208 | 2.37 / 7.70 |
| Q4_K prefill, 8B gate/up M512 | 0.77 | 26.1 | 33.8 |
| Q6_K decode, 8B gate/up, B1 / B8 | 197 / 25.3 | 270 / 206 | 1.37 / 8.16 |
| Q6_K prefill, 8B gate/up M512 | 0.50 | 24.0 | 48.0 |

(Decode GB/s above 256 are cache-resident weights.)

**Audio** (`--bench-audio`, ms)

| Op | HIP | Vulkan | HIP / Vulkan |
|---|---:|---:|---:|
| Snake 512 ch x 24000, FP32 | 0.458 | 0.423 | 1.08 |
| pad1d reflect 512 x 24000 | 0.431 | 0.431 | 1.00 |
| resample1d 16k -> 24k, 64 ch x 10 s | 0.932 | 0.919 | 1.01 |
| conv_transpose1d 512 -> 256 k16 s8 L1000, FP32 | 63.7 | 0.835 | 76.3 |
| STFT n_fft 400 hop 160, 30 s (FP64 DFT, default) | 52.8 | 3.69 | 14.3 |
| iSTFT, same (FP64 DFT, default) | 122.6 | 4.59 | 26.7 |
| STFT, same (`BROTENSOR_VK_DFT_GEMM=1`, FP32 GEMM) | 52.8 | 0.198 | 267 |
| iSTFT, same (`BROTENSOR_VK_DFT_GEMM=1`) | 122.6 | 0.212 | 578 |
| complex_abs 3001 x 201 | 0.010 | 0.010 | 0.94 |

`sample_logits_into` with no top-k / top-p at V = 151936 (found by the
Vulkan-vs-HIP parity test): ~21 s per call on HIP (one argmax pass per kept
token, O(keep V)), 9 ms on Vulkan.

## llama.cpp reference (vk-spike, same machine)

One codebase with expert kernels for both backends (`/home/j/projects/vk-spike/RESULTS.md`,
run 2, llama-bench):

| Model | HIP (ROCm) | Vulkan (RADV) | Vulkan / HIP |
|---|---:|---:|---:|
| Qwen3-0.6B Q8_0 decode tg128 @ d0 / d512 | 227.7 / 214.1 tok/s | 282.2 / 255.0 tok/s | 1.24 / 1.19 |
| Qwen3-0.6B Q8_0 prefill pp512 | 12787 ± 1059 tok/s | 14572 ± 69 tok/s | 1.14 |
| Qwen3-30B-A3B Q4_K_M decode / pp512 | 80.6 / 1683 tok/s | 92.7 / 1584 tok/s | 1.15 / 0.94 |
| Qwen3-30B-A3B Q8_0 decode / pp512 | 54.5 / 1452 tok/s | 61.6 / 1812 tok/s | 1.13 / 1.25 |
| per-op MUL_MAT, 204 cases | | | geomean 1.27 (165 faster) |
| per-op FLASH_ATTN_EXT, 24 cases | | | geomean 1.73 (0.36-8.9) |

brotensor against llama.cpp on the same GGUF: decode 192 tok/s (Vulkan) vs
llama.cpp Vulkan 282 (0.68x); prefill 15.9k vs 14.6k tok/s (1.09x).

## Known losses

| Where | HIP | Vulkan | Why |
|---|---:|---:|---|
| T5-XXL encode | 0.93 s | 1.9 s | T5's activations exceed FP16; RADV has no BF16 cooperative matrix, so Vulkan runs T5 with FP32 activations on the SIMT GEMM (docs/vulkan.md "Dtype policy") |
| STFT / iSTFT, 30 s Whisper front end | 53 / 124 ms | 3.6 / 4.6 ms | not a loss against HIP; against the FP32 basis GEMM (0.20 / 0.23 ms) the FP64 direct DFT is 18-20x slower, the price of CPU-exact spectra (log-mel 0.025 -> 1e-6 vs the CPU) |
| BF16 operands in matrix ops | | | staged as FP16 (range 65504) by the matrix-core kernels; FP32 fallbacks where a model needs the range (T5, Sana's linear attention) |
| Training backwards | | | 21 op-table slots still null on Vulkan (docs/vulkan-coverage.md); those paths have no Vulkan timing |
