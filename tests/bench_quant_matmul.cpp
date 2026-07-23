// Micro-benchmark: the GGUF block-quant linear paths — Q4_K / Q6_K / Q8_0.
//
// Two regimes, and they are bound by different things:
//
//   * decode  (B == 1)  — a GEMV. Every weight byte is read once and used for
//     one multiply, so this is pure bandwidth: the score is GB/s of quantized
//     weight streamed, against ~1008 GB/s of HBM on a 4090. Note the
//     denominator counts the *packed* bytes actually moved (Q4_K is 4.5
//     bits/weight), so it is directly comparable to the FP16 GEMV rows in
//     bench_decode_kernels.
//
//   * prefill (B >= 16) — a GEMM. The weight is decoded once and reused across
//     B tokens, so the nibble-unpack arithmetic in the B-tile loader moves onto
//     the critical path and the score becomes GFLOP/s.
//
// The unpack inner loop is the thing worth optimizing here (shift / and /
// int->float / __float2half per nibble today), and prefill is where it shows.
//
// Correctness per row is checked against the dequantize-then-dense path
// (dequant_q*_to_fp16 + linear_forward_batched_fp16) — genuinely independent
// code from the fused kernels, so a fast-but-wrong unpack can't pass silently.
//
// NOT registered with ctest — invoke manually:
//   ./build-cuda/tests/Release/brotensor_bench_quant_matmul

#include <brotensor/ops.h>
#include <brotensor/ops/quant.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

int g_failures = 0;

// Per-format GGUF block geometry.
struct Format {
    const char* name;
    Dtype       dt;
    int         block_bytes;
    int         block_elems;
    void (*gemv)(const Tensor&, const Tensor*, const Tensor&, Tensor&);
    void (*batched)(const Tensor&, const Tensor*, const Tensor&, Tensor&);
    void (*dequant)(const Tensor&, Tensor&);
};

const Format kFormats[] = {
    {"q4k",  Dtype::Q4_K, 144, 256,
     brotensor::linear_forward_q4k_fp16,
     brotensor::linear_forward_batched_q4k_fp16,
     brotensor::dequant_q4k_to_fp16},
    {"q6k",  Dtype::Q6_K, 210, 256,
     brotensor::linear_forward_q6k_fp16,
     brotensor::linear_forward_batched_q6k_fp16,
     brotensor::dequant_q6k_to_fp16},
    {"q8_0", Dtype::Q8_0,  34,  32,
     brotensor::linear_forward_q8_0_fp16,
     brotensor::linear_forward_batched_q8_0_fp16,
     brotensor::dequant_q8_0_to_fp16},
};

uint16_t f16(float v) { return brotensor::fp32_to_fp16_bits(v); }

void put_f16(uint8_t* p, int off, float v) {
    const uint16_t b = f16(v);
    p[off]     = static_cast<uint8_t>(b & 0xFF);
    p[off + 1] = static_cast<uint8_t>(b >> 8);
}

// Build an (out, in) quantized weight from pseudo-random block bytes with the
// scale field(s) pinned to a small finite constant, so decoded values stay
// bounded and non-NaN. The exact values don't matter — both the fused path and
// the dequant reference decode the same bytes. Same construction as
// test_quant_prefill_parity.cpp.
Tensor make_quant_weight(int out, int in, const Format& fmt, uint32_t seed) {
    const int bpr = in / fmt.block_elems;
    const size_t nblocks = static_cast<size_t>(out) * bpr;
    std::vector<uint8_t> bytes(nblocks * fmt.block_bytes);
    uint32_t lcg = seed;
    for (auto& b : bytes) {
        lcg = lcg * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(lcg >> 24);
    }
    for (size_t blk = 0; blk < nblocks; ++blk) {
        uint8_t* p = &bytes[blk * fmt.block_bytes];
        if (fmt.dt == Dtype::Q4_K) {            // d @0, dmin @2
            put_f16(p, 0, 0.02f);
            put_f16(p, 2, 0.008f);
        } else if (fmt.dt == Dtype::Q8_0) {     // d @0
            put_f16(p, 0, 0.01f);
        } else {                                // Q6_K: d @ end
            put_f16(p, fmt.block_bytes - 2, 0.01f);
        }
    }
    Tensor W = Tensor::empty_on(Device::CUDA, out, in, fmt.dt);
    cudaMemcpy(W.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    return W;
}

Tensor upload_fp16(const std::vector<float>& v, int rows, int cols) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = f16(v[i]);
    return Tensor::from_host_fp16_on(Device::CUDA, h.data(), rows, cols);
}

std::vector<float> download_fp16(const Tensor& t) {
    std::vector<uint16_t> h(t.size());
    t.copy_to_host_fp16(h.data());
    brotensor::sync_all();
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return v;
}

std::vector<float> rand_vec(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// Compare against dequant-then-dense. Both consume identical weight bytes;
// they differ only in accumulation order, so the tolerance covers FP16 output
// rounding and FP32 reassociation.
bool check_against_dequant(const Format& fmt, const Tensor& W, const Tensor& X,
                           const std::vector<float>& got, int B, int out,
                           const char* what) {
    Tensor W16;
    fmt.dequant(W, W16);
    brotensor::sync_all();
    Tensor Yref;
    brotensor::linear_forward_batched_fp16(W16, nullptr, X, Yref);
    brotensor::sync_all();
    auto ref = download_fp16(Yref);

    // Normalize the error by the RMS of the reference vector, not by each
    // element's own magnitude.
    //
    // These are K=4096-term dot products over pseudo-random weights, so the
    // results are a random walk and individual outputs land arbitrarily close
    // to zero. A per-element relative error divides by those near-zero values
    // and explodes on outputs that are perfectly fine.
    //
    // The two paths also differ legitimately in precision: the fused GEMV
    // decodes each block to FP32 and accumulates without ever rounding a
    // weight to 16 bits, while this reference rounds every weight through FP16
    // first. That injects a per-weight relative error of ~5e-4, which random-
    // walks into an absolute discrepancy of about eps*sqrt(K)*|w||x| — the
    // same order as the RMS of the result times eps. So "max abs error over
    // RMS of the result" is the quantity that stays ~5e-3 for a correct kernel
    // and goes O(1) the moment an unpack is wrong.
    const size_t n = static_cast<size_t>(B) * out;
    double ss = 0.0;
    for (size_t i = 0; i < n && i < ref.size(); ++i) ss += double(ref[i]) * ref[i];
    const double rms = std::sqrt(ss / double(n)) + 1e-6;

    double worst_abs = 0.0;
    for (size_t i = 0; i < n && i < got.size() && i < ref.size(); ++i) {
        const double e = std::fabs(got[i] - ref[i]);
        if (e > worst_abs) worst_abs = e;
    }
    const double norm = worst_abs / rms;
    if (norm > 3e-2) {
        std::printf("      ^^ MISMATCH %s: max|err| %.3g vs result RMS %.3g "
                    "(ratio %.3g)\n", what, worst_abs, rms, norm);
        ++g_failures;
        return false;
    }
    return true;
}

// AD102 L2 is 72 MB — a decode GEMV replaying one weight matrix would serve it
// from cache and report bandwidth the bus never delivered. Cycle through
// enough replicas to blow past L2, as bench_decode_kernels does.
constexpr double kL2DefeatBytes = 512.0 * 1024.0 * 1024.0;

int replicas_for(double bytes_each) {
    const int n = static_cast<int>(kL2DefeatBytes / bytes_each) + 1;
    return n < 1 ? 1 : (n > 32 ? 32 : n);
}

// ─── decode: B == 1 GEMV, bandwidth-bound ──────────────────────────────────

void bench_gemv(const char* label, const Format& fmt, int out, int in) {
    if (in % fmt.block_elems != 0) {
        std::printf("  %-6s %-20s  skipped (K %% %d != 0)\n", fmt.name, label,
                    fmt.block_elems);
        return;
    }
    const double w_bytes = static_cast<double>(out) *
                           (static_cast<double>(in) / fmt.block_elems) *
                           fmt.block_bytes;
    const int n_rep = replicas_for(w_bytes);

    std::vector<Tensor> Ws;
    Ws.reserve(static_cast<size_t>(n_rep));
    for (int i = 0; i < n_rep; ++i) {
        Ws.push_back(make_quant_weight(out, in, fmt, 0x1234567u + 7919u * i));
    }
    auto xv = rand_vec(static_cast<size_t>(in), 0xC0FFEEu, 0.5f);
    Tensor X = upload_fp16(xv, in, 1);
    Tensor Y;
    fmt.gemv(Ws[0], nullptr, X, Y);
    brotensor::sync_all();

    // The GEMV takes x as (in,1); the dense reference wants (1,in).
    Tensor Xrow = upload_fp16(xv, 1, in);
    auto got = download_fp16(Y);
    const bool ok = check_against_dequant(fmt, Ws[0], Xrow, got, 1, out, label);

    // Round-robin over replicas so each timed call streams a cold weight.
    int idx = 0;
    const float ms = bt_bench::time_min_ms([&] {
        fmt.gemv(Ws[static_cast<size_t>(idx)], nullptr, X, Y);
        idx = (idx + 1) % n_rep;
    });

    std::printf("  %-6s %-20s %6d x %-6d  %9.1f us  %7.1f GB/s  %s\n",
                fmt.name, label, out, in, ms * 1e3,
                w_bytes / (ms * 1e-3) / 1e9, ok ? "ok" : "MISMATCH");
}

// ─── prefill: B >= 16 GEMM, decode-arithmetic-bound ────────────────────────

void bench_prefill(const char* label, const Format& fmt, int B, int out, int in) {
    if (in % fmt.block_elems != 0) {
        std::printf("  %-6s %-20s  skipped (K %% %d != 0)\n", fmt.name, label,
                    fmt.block_elems);
        return;
    }
    Tensor W = make_quant_weight(out, in, fmt, 0x2468ACEu);
    auto xv = rand_vec(static_cast<size_t>(B) * in, 0xBEEF01u, 0.4f);
    Tensor X = upload_fp16(xv, B, in);
    Tensor Y;
    fmt.batched(W, nullptr, X, Y);
    brotensor::sync_all();

    auto got = download_fp16(Y);
    const bool ok = check_against_dequant(fmt, W, X, got, B, out, label);

    const float ms = bt_bench::time_min_ms([&] { fmt.batched(W, nullptr, X, Y); });
    const double flops = 2.0 * B * static_cast<double>(out) * in;

    std::printf("  %-6s %-20s B=%-4d %6d x %-6d  %8.3f ms  %8.1f GFLOP/s  %s\n",
                fmt.name, label, B, out, in, ms,
                flops / (ms * 1e-3) / 1e9, ok ? "ok" : "MISMATCH");
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available — skipping quant matmul bench\n");
        return 0;
    }
    // Pull the SM clock off its P8 idle floor before any timing.
    bt_bench::spin_up();
    std::printf("brotensor_bench_quant_matmul  (warmup %.0f ms/op, best of %d)\n",
                bt_bench::kWarmupMs, bt_bench::kSamples);
    std::printf("GB/s counts packed weight bytes; peak HBM on a 4090 is "
                "~1008 GB/s\n");

    // Llama/Qwen-8B-class projection shapes — the ones a quantized decoder
    // actually issues.
    struct Shape { const char* label; int out, in; };
    const Shape kShapes[] = {
        {"qkv_proj",   6144,  4096},
        {"o_proj",     4096,  4096},
        {"gate/up",   12288,  4096},
        {"down_proj",  4096, 12288},
    };

    std::printf("\n-- decode GEMV (B=1, bandwidth-bound) --\n");
    for (const auto& fmt : kFormats) {
        for (const auto& s : kShapes) bench_gemv(s.label, fmt, s.out, s.in);
    }

    std::printf("\n-- prefill GEMM (weight decoded once, reused across B) --\n");
    for (int B : {16, 128, 512}) {
        for (const auto& fmt : kFormats) {
            bench_prefill("o_proj",   fmt, B, 4096, 4096);
            bench_prefill("gate/up",  fmt, B, 12288, 4096);
        }
        std::printf("\n");
    }

    if (g_failures != 0) {
        std::printf("%d MISMATCH(es)\n", g_failures);
        return 1;
    }
    std::printf("all rows verified\n");
    return 0;
}
