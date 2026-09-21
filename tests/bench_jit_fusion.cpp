// ─── Trace JIT vs eager ops vs hand-written fusions ──────────────────────────
//
// Every pattern here is one the Qwen-Image 2.1 DiT, its VAE, or the flow-match
// scheduler runs on the hot path, measured three ways:
//
//   eager  — the brotensor op sequence the model code writes today. Each op is
//            its own kernel, and every intermediate makes a full round trip
//            through HBM.
//   hand   — the hand-written fused kernel, where one exists (brass's
//            MlFusionCompiler PTX). Blank when there is none.
//   jit    — one begin_trace()/end_trace() over the same expression, replayed
//            through TraceHandle::execute().
//
// These are bandwidth-bound kernels, so the number that matters is GB/s
// against the machine's achievable streaming bandwidth, which the harness
// measures first with a large device-to-device copy rather than quoting a
// spec sheet. `ideal` bytes is the traffic a perfectly fused kernel must move:
// inputs read once, outputs written once, broadcast rows counted once. The
// eager column moves more than that — its extra traffic is exactly what the
// fusion removes — so its GB/s figure is "effective", not achieved.
//
// Timing is min-of-N after a clock spin-up (tests/bench_helpers.h): on a
// desktop GPU these kernels are short enough that a mean measures the clock
// ramp instead of the kernel.

#include "bench_helpers.h"

#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "../src/cuda/cuda_jit.h"
#include "../src/jit/trace_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace brotensor;

namespace {

// ── measurement bookkeeping ─────────────────────────────────────────────────

struct Row {
    std::string pattern;
    std::string shape;
    std::string dtype;
    std::string variant;   // eager | hand | jit
    int launches = 0;
    float ms = 0.0f;
    double gbps = 0.0;
    double roof = 0.0;     // fraction of the measured streaming peak
    std::string note;
};

std::vector<Row> g_rows;
double g_peak_gbps = 0.0;

int esz(Dtype d) { return d == Dtype::FP32 ? 4 : 2; }

const char* dname(Dtype d) {
    switch (d) {
        case Dtype::FP32: return "fp32";
        case Dtype::BF16: return "bf16";
        case Dtype::FP16: return "fp16";
        default: return "?";
    }
}

void record(const std::string& pattern, int rows, int cols, Dtype dt,
            const std::string& variant, int launches, float ms,
            std::int64_t ideal_bytes, const std::string& note = std::string()) {
    Row r;
    r.pattern = pattern;
    r.shape = std::to_string(rows) + "x" + std::to_string(cols);
    r.dtype = dname(dt);
    r.variant = variant;
    r.launches = launches;
    r.ms = ms;
    r.gbps = (ms > 0.0f) ? (static_cast<double>(ideal_bytes) / 1e9) / (ms / 1e3) : 0.0;
    r.roof = g_peak_gbps > 0.0 ? r.gbps / g_peak_gbps : 0.0;
    r.note = note;
    g_rows.push_back(std::move(r));
    const Row& p = g_rows.back();
    std::printf("  %-22s %-12s %-5s %-6s  %2d launch  %8.3f ms  %8.1f GB/s  %5.0f%% %s\n",
                p.pattern.c_str(), p.shape.c_str(), p.dtype.c_str(), p.variant.c_str(),
                p.launches, p.ms, p.gbps, p.roof * 100.0, p.note.c_str());
    std::fflush(stdout);
}

// ── tensor helpers ──────────────────────────────────────────────────────────

struct SplitMix64 {
    std::uint64_t s;
    explicit SplitMix64(std::uint64_t seed) : s(seed) {}
    float next_sym() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
    }
};

Tensor rand_tensor(Device dev, int rows, int cols, Dtype dt, std::uint64_t seed,
                   float bias = 0.0f) {
    SplitMix64 rng(seed);
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    if (dt == Dtype::FP32) {
        std::vector<float> h(n);
        for (auto& v : h) v = rng.next_sym() + bias;
        return Tensor::from_host_on(dev, h.data(), rows, cols);
    }
    std::vector<std::uint16_t> h(n);
    for (auto& v : h) {
        const float f = rng.next_sym() + bias;
        v = (dt == Dtype::BF16) ? fp32_to_bf16_bits(f) : fp32_to_fp16_bits(f);
    }
    return (dt == Dtype::BF16) ? Tensor::from_host_bf16_on(dev, h.data(), rows, cols)
                               : Tensor::from_host_fp16_on(dev, h.data(), rows, cols);
}

// ── the achievable streaming bandwidth of this machine ──────────────────────

void measure_peak(Device dev) {
    const int rows = 8192, cols = 4096;  // 128 MiB at FP32
    Tensor a = Tensor::zeros_on(dev, rows, cols, Dtype::FP32);
    Tensor b = Tensor::zeros_on(dev, rows, cols, Dtype::FP32);
    const std::int64_t bytes = 2LL * rows * cols * 4;
    const float ms = bt_bench::time_min_ms([&] { copy_d2d(a, 0, b, 0, a.size()); });
    g_peak_gbps = (static_cast<double>(bytes) / 1e9) / (ms / 1e3);
    std::printf("Measured streaming bandwidth (128 MiB d2d copy): %.1f GB/s (%.3f ms)\n\n",
                g_peak_gbps, ms);
}

// ── patterns ────────────────────────────────────────────────────────────────
//
// Each returns after recording one row per variant.

// x += gate[1,D] * y — the DiT attention and MLP residual gates. Two of these
// run per block, 32 blocks per step.
void bench_gated_residual(Device dev, int rows, int cols, Dtype dt) {
    Tensor x = rand_tensor(dev, rows, cols, dt, 1);
    Tensor y = rand_tensor(dev, rows, cols, dt, 2);
    Tensor g = rand_tensor(dev, 1, cols, dt, 3);
    const std::int64_t ideal = 3LL * rows * cols * esz(dt) + 1LL * cols * esz(dt);

    Tensor tmp = Tensor::empty_on(dev, rows, cols, dt);
    record("gated-residual", rows, cols, dt, "eager", 2,
           bt_bench::time_min_ms([&] {
               broadcast_mul(y, g, tmp);
               add_inplace(x, tmp);
           }),
           ideal);

    begin_trace();
    x += g * y;
    TraceHandle h = end_trace();
    record("gated-residual", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// out = silu(gate) * proj — the SwiGLU tail, at the MLP's 3x hidden width.
void bench_swiglu(Device dev, int rows, int cols, Dtype dt) {
    Tensor g = rand_tensor(dev, rows, cols, dt, 11);
    Tensor p = rand_tensor(dev, rows, cols, dt, 12);
    Tensor out = Tensor::empty_on(dev, rows, cols, dt);
    const std::int64_t ideal = 3LL * rows * cols * esz(dt);

    Tensor gs = g.clone();
    record("swiglu-tail", rows, cols, dt, "eager", 2,
           bt_bench::time_min_ms([&] {
               silu_forward(g, gs);
               mul_inplace(gs, p);
           }),
           ideal);

    // brass's packed SwiGLU takes gate|up interleaved as (rows, 2*cols) and is
    // FP32-only, so it moves the same bytes from a different layout.
    if (dt == Dtype::FP32) {
        Tensor packed = rand_tensor(dev, rows, 2 * cols, dt, 13);
        record("swiglu-tail", rows, cols, dt, "hand", 1,
               bt_bench::time_min_ms([&] {
                   detail::cuda::jit::launch_swiglu_ptx(
                       static_cast<const float*>(packed.data),
                       static_cast<float*>(out.data), rows, cols, nullptr);
               }),
               ideal, "brass packed");
    }

    begin_trace();
    Tensor jout = jit::silu(g) * p;
    TraceHandle h = end_trace();
    record("swiglu-tail", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// sample += (sigma_next - sigma) * v — one per denoising step.
void bench_euler(Device dev, int rows, int cols, Dtype dt) {
    Tensor x = rand_tensor(dev, rows, cols, dt, 21);
    Tensor v = rand_tensor(dev, rows, cols, dt, 22);
    const float d_sigma = -0.0373f;
    const std::int64_t ideal = 3LL * rows * cols * esz(dt);

    // What the scheduler writes today: copy, scale, add.
    Tensor scratch = Tensor::empty_on(dev, rows, cols, dt);
    record("euler-step", rows, cols, dt, "eager", 3,
           bt_bench::time_min_ms([&] {
               copy_d2d(v, 0, scratch, 0, v.size());
               scale_inplace(scratch, d_sigma);
               add_inplace(x, scratch);
           }),
           ideal);

    // The single-kernel form already available: a fused axpby.
    record("euler-step", rows, cols, dt, "hand", 1,
           bt_bench::time_min_ms([&] { axpby_inplace(x, v, 1.0f, d_sigma); }), ideal,
           "axpby_inplace");

    begin_trace();
    x += v * d_sigma;
    TraceHandle h = end_trace();
    record("euler-step", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// LN(x) * (1 + scale[1,D]) + shift[1,D] — the DiT's AdaLN, twice per block.
void bench_layernorm_modulate(Device dev, int rows, int cols, Dtype dt) {
    Tensor x = rand_tensor(dev, rows, cols, dt, 31);
    Tensor gamma = rand_tensor(dev, 1, cols, dt, 32, 1.0f);
    Tensor beta = rand_tensor(dev, 1, cols, dt, 33);
    Tensor scale = rand_tensor(dev, 1, cols, dt, 34);
    Tensor shift = rand_tensor(dev, 1, cols, dt, 35);
    const std::int64_t ideal = 2LL * rows * cols * esz(dt) + 4LL * cols * esz(dt);

    Tensor ln = Tensor::empty_on(dev, rows, cols, dt);
    Tensor out = Tensor::empty_on(dev, rows, cols, dt);
    record("layernorm-modulate", rows, cols, dt, "eager", 2,
           bt_bench::time_min_ms([&] {
               layernorm_forward_inference_batched(x, gamma, beta, ln, 1e-6f);
               modulate(ln, scale, shift, out);
           }),
           ideal);

    if (dt == Dtype::FP32) {
        record("layernorm-modulate", rows, cols, dt, "hand", 1,
               bt_bench::time_min_ms([&] {
                   detail::cuda::jit::launch_fused_layernorm_modulate_ptx(
                       static_cast<const float*>(x.data),
                       static_cast<const float*>(gamma.data),
                       static_cast<const float*>(beta.data),
                       static_cast<const float*>(scale.data),
                       static_cast<const float*>(shift.data),
                       static_cast<float*>(out.data), rows, cols, 1e-6f, nullptr);
               }),
               ideal, "brass fused");
    }

    begin_trace();
    Tensor jln = jit::layernorm(x, gamma, beta, 1e-6f);
    Tensor jout = jit::modulate(jln, scale, shift);
    TraceHandle h = end_trace();
    record("layernorm-modulate", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// silu(rms_norm(x, gamma)) — the VAE resnet's norm/activation seam.
void bench_rmsnorm_silu(Device dev, int rows, int cols, Dtype dt) {
    Tensor x = rand_tensor(dev, rows, cols, dt, 41);
    Tensor gamma = rand_tensor(dev, 1, cols, dt, 42, 1.0f);
    const std::int64_t ideal = 2LL * rows * cols * esz(dt) + 1LL * cols * esz(dt);

    Tensor n = Tensor::empty_on(dev, rows, cols, dt);
    record("rmsnorm-silu", rows, cols, dt, "eager", 2,
           bt_bench::time_min_ms([&] {
               rms_norm_forward(x, gamma, 1e-6f, n);
               silu_forward(n, n);
           }),
           ideal);

    begin_trace();
    Tensor jout = jit::silu(jit::rms_norm(x, gamma, 1e-6f));
    TraceHandle h = end_trace();
    record("rmsnorm-silu", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// x += p; y = rms_norm(x) — the transformer residual the hand kernel targets.
void bench_residual_rmsnorm(Device dev, int rows, int cols, Dtype dt) {
    Tensor x = rand_tensor(dev, rows, cols, dt, 51);
    Tensor p = rand_tensor(dev, rows, cols, dt, 52);
    Tensor gamma = rand_tensor(dev, 1, cols, dt, 53, 1.0f);
    Tensor y = Tensor::empty_on(dev, rows, cols, dt);
    const std::int64_t ideal = 4LL * rows * cols * esz(dt) + 1LL * cols * esz(dt);

    record("residual-rmsnorm", rows, cols, dt, "eager", 2,
           bt_bench::time_min_ms([&] {
               add_inplace(x, p);
               rms_norm_forward(x, gamma, 1e-6f, y);
           }),
           ideal);

    if (dt == Dtype::FP32) {
        record("residual-rmsnorm", rows, cols, dt, "hand", 1,
               bt_bench::time_min_ms([&] {
                   detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
                       static_cast<float*>(x.data), static_cast<const float*>(p.data),
                       static_cast<const float*>(gamma.data), static_cast<float*>(y.data),
                       rows, cols, 1e-6f, nullptr);
               }),
               ideal, "brass fused");
    }

    begin_trace();
    x += p;
    Tensor jy = jit::rms_norm(x, gamma, 1e-6f);
    TraceHandle h = end_trace();
    record("residual-rmsnorm", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// (a*b + c) * silu(d) — a four-op chain, the shape of expression the tracer
// exists to collapse.
void bench_chain4(Device dev, int rows, int cols, Dtype dt) {
    Tensor a = rand_tensor(dev, rows, cols, dt, 61);
    Tensor b = rand_tensor(dev, rows, cols, dt, 62);
    Tensor c = rand_tensor(dev, rows, cols, dt, 63);
    Tensor d = rand_tensor(dev, rows, cols, dt, 64);
    const std::int64_t ideal = 5LL * rows * cols * esz(dt);

    Tensor t1 = Tensor::empty_on(dev, rows, cols, dt);
    Tensor t2 = Tensor::empty_on(dev, rows, cols, dt);
    record("chain4", rows, cols, dt, "eager", 4,
           bt_bench::time_min_ms([&] {
               copy_d2d(a, 0, t1, 0, a.size());
               mul_inplace(t1, b);
               add_inplace(t1, c);
               silu_forward(d, t2);
               mul_inplace(t1, t2);
           }),
           ideal, "5 launches incl. copy");

    begin_trace();
    Tensor jout = (a * b + c) * jit::silu(d);
    TraceHandle h = end_trace();
    record("chain4", rows, cols, dt, "jit", static_cast<int>(h.launch_count()),
           bt_bench::time_min_ms([&] { h.execute(); }), ideal, h.fusion_name());
}

// ── compile and cache cost ──────────────────────────────────────────────────

void bench_compile_cost(Device dev) {
    std::printf("\n## Compile and cache cost\n\n");
    const int rows = 1024, cols = 4096;
    struct Case { const char* name; Dtype dt; };
    const Case cases[] = {{"elementwise (gated residual)", Dtype::BF16},
                          {"row-norm (layernorm+modulate)", Dtype::BF16}};

    for (const Case& cs : cases) {
        Tensor x = rand_tensor(dev, rows, cols, cs.dt, 71);
        Tensor y = rand_tensor(dev, rows, cols, cs.dt, 72);
        Tensor g = rand_tensor(dev, 1, cols, cs.dt, 73);
        Tensor beta = rand_tensor(dev, 1, cols, cs.dt, 74);
        Tensor shift = rand_tensor(dev, 1, cols, cs.dt, 75);

        // Cold cache so the first trace pays the full PTX emit + ptxas cost.
        jit::TraceCache::instance().clear();

        double first_us = 0.0, hit_us = 0.0;
        const bool row_norm = (std::string(cs.name).rfind("row-norm", 0) == 0);

        auto build = [&]() {
            begin_trace();
            if (row_norm) {
                Tensor ln = jit::layernorm(x, g, beta, 1e-6f);
                Tensor o = jit::modulate(ln, g, shift);
                (void)o;
            } else {
                x += g * y;
            }
            return end_trace();
        };

        const auto t0 = std::chrono::steady_clock::now();
        TraceHandle h1 = build();
        sync_all();
        first_us = std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - t0).count();

        const auto t1 = std::chrono::steady_clock::now();
        TraceHandle h2 = build();
        sync_all();
        hit_us = std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - t1).count();

        std::printf("  %-32s first trace %8.1f us (ptx+ptxas %7.1f us), "
                    "cached re-trace %6.1f us, hit=%d\n",
                    cs.name, first_us, h1.compile_us(), hit_us, h2.is_cache_hit() ? 1 : 0);
    }
    std::fflush(stdout);
}

// ── markdown ────────────────────────────────────────────────────────────────

void print_markdown() {
    std::printf("\n\n<!-- markdown table -->\n\n");
    std::printf("| pattern | shape | dtype | variant | launches | ms | GB/s | %% of peak | note |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");
    for (const Row& r : g_rows) {
        std::printf("| %s | %s | %s | %s | %d | %.3f | %.1f | %.0f%% | %s |\n",
                    r.pattern.c_str(), r.shape.c_str(), r.dtype.c_str(), r.variant.c_str(),
                    r.launches, r.ms, r.gbps, r.roof * 100.0, r.note.c_str());
    }
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::cuda())) {
        std::printf("[SKIP] no CUDA device.\n");
        return 0;
    }
    brotensor::set_default_device(Device::cuda());
    const Device dev = Device::cuda();

    bt_bench::spin_up();
    measure_peak(dev);

    struct Shape { int rows, cols; };
    const Shape shapes[] = {{4096, 4096}, {4096, 12288}, {1024, 4096}, {256, 1024}};
    const Dtype dtypes[] = {Dtype::FP32, Dtype::BF16, Dtype::FP16};

    for (const Shape& s : shapes) {
        std::printf("\n## %dx%d\n\n", s.rows, s.cols);
        for (Dtype dt : dtypes) {
            bench_gated_residual(dev, s.rows, s.cols, dt);
            bench_swiglu(dev, s.rows, s.cols, dt);
            bench_euler(dev, s.rows, s.cols, dt);
            bench_layernorm_modulate(dev, s.rows, s.cols, dt);
            bench_rmsnorm_silu(dev, s.rows, s.cols, dt);
            bench_residual_rmsnorm(dev, s.rows, s.cols, dt);
            bench_chain4(dev, s.rows, s.cols, dt);
        }
    }

    // The VAE's shape: a feature map reaches the row kernel as (H*W, channels),
    // which is a different regime entirely — hundreds of thousands of rows a
    // hundred elements wide, where the reduction's fixed cost per row is what
    // decides whether fusing beats two plain elementwise passes.
    const Shape vae[] = {{262144, 384}, {1048576, 96}, {65536, 384}};
    for (const Shape& s : vae) {
        std::printf("\n## %dx%d (VAE feature map)\n\n", s.rows, s.cols);
        bench_rmsnorm_silu(dev, s.rows, s.cols, Dtype::FP32);
    }

    bench_compile_cost(dev);
    print_markdown();
    return 0;
}
