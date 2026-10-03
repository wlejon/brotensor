// `brotensor_test_vulkan --bench-jit`: the trace JIT's Vulkan compiler against
// the op-by-op replay it replaces, on the seams the DiT / VAE / scheduler code
// traces. Not part of ctest.
//
//   replay  BROTENSOR_JIT_VULKAN=eager: the DAG replayed through the
//           dispatched ops, one launch per traced op, every intermediate a
//           full-size scratch tensor (what Vulkan ran before the compiler)
//   fused   the SPIR-V kernel the compiler emits: one launch
//
// GB/s is over the ideal traffic of a perfectly fused kernel (inputs read
// once, outputs written once, rows counted once), so the replay's figure is
// "effective". Timing: wall clock around a batch of back-to-back executes
// with one sync, median of 7 batches after warm-up (test_vulkan_bench.cpp).

#include "test_vulkan_common.h"

#include <brotensor/jit/trace.h>

#include "../src/jit/trace_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace vkt {

namespace {

namespace jit = brotensor::jit;
using brotensor::TraceHandle;

template <class F>
double median_ms(F&& launch_once) {
    for (int i = 0; i < 3; ++i) launch_once();
    brotensor::sync(vk());
    auto t0 = std::chrono::steady_clock::now();
    launch_once();
    brotensor::sync(vk());
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int reps = std::max(3, std::min(200, static_cast<int>(150.0 / std::max(one, 1e-3))));
    std::vector<double> t;
    for (int b = 0; b < 7; ++b) {
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) launch_once();
        brotensor::sync(vk());
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

void set_eager(bool on) {
#ifdef _WIN32
    _putenv_s("BROTENSOR_JIT_VULKAN", on ? "eager" : "");
#else
    if (on) setenv("BROTENSOR_JIT_VULKAN", "eager", 1);
    else unsetenv("BROTENSOR_JIT_VULKAN");
#endif
}

int esz(Dtype d) { return d == Dtype::FP32 ? 4 : 2; }

// Traces `body` twice — once replayed op by op, once fused — and times both.
void compare(const std::string& pattern, int rows, int cols, Dtype dt, double ideal_bytes,
             const std::function<void()>& body) {
    double ms[2] = {0, 0};
    std::size_t launches[2] = {0, 0};
    std::string fusion;
    for (int fused = 0; fused < 2; ++fused) {
        brotensor::jit::TraceCache::instance().clear();
        set_eager(!fused);
        jit::begin_trace();
        body();
        TraceHandle h = jit::end_trace();
        set_eager(false);
        launches[fused] = h.launch_count();
        if (fused) fusion = h.fusion_name();
        ms[fused] = median_ms([&] { h.execute(); });
    }
    brotensor::jit::TraceCache::instance().clear();
    auto gbps = [&](double m) { return ideal_bytes / 1e9 / (m / 1e3); };
    std::printf("| %s | %dx%d | %s | %zu | %.3f | %.0f | %.3f | %.0f | %.2fx | %s |\n", pattern.c_str(), rows, cols,
                dt_name(dt), launches[0], ms[0], gbps(ms[0]), ms[1], gbps(ms[1]), ms[0] / ms[1], fusion.c_str());
    std::fflush(stdout);
}

Tensor rnd(int rows, int cols, Dtype dt, std::uint64_t seed, float lo = -1, float hi = 1) {
    return upload(random_values(std::size_t(rows) * cols, seed, lo, hi, dt), rows, cols, dt);
}

// LN(x) * (1 + scale[1,D]) + shift[1,D] — the DiT's AdaLN.
void bench_ln_modulate(int rows, int cols, Dtype dt) {
    Tensor x = rnd(rows, cols, dt, 1), g = rnd(1, cols, dt, 2, 0.5f, 1.5f), b = rnd(1, cols, dt, 3);
    Tensor sc = rnd(1, cols, dt, 4), sh = rnd(1, cols, dt, 5);
    Tensor out = Tensor::empty_on(vk(), rows, cols, dt);
    const double ideal = 2.0 * rows * cols * esz(dt) + 4.0 * cols * esz(dt);
    compare("LN-modulate", rows, cols, dt, ideal, [&] {
        jit::store(out, jit::modulate(jit::layernorm(x, g, b, 1e-6f), sc, sh));
    });
}

// silu(rms_norm(x, gamma)) — the VAE resnet's norm/activation seam, on
// (H*W, channels) feature maps.
void bench_norm_silu(int rows, int cols, Dtype dt) {
    Tensor x = rnd(rows, cols, dt, 11), g = rnd(1, cols, dt, 12, 0.5f, 1.5f);
    Tensor out = Tensor::empty_on(vk(), rows, cols, dt);
    const double ideal = 2.0 * rows * cols * esz(dt) + 1.0 * cols * esz(dt);
    compare("VAE norm+silu", rows, cols, dt, ideal, [&] {
        jit::store(out, jit::silu(jit::rms_norm(x, g, 1e-6f)));
    });
}

// A ten-op elementwise chain over four inputs.
void bench_chain(int rows, int cols, Dtype dt) {
    Tensor a = rnd(rows, cols, dt, 21), b = rnd(rows, cols, dt, 22), c = rnd(rows, cols, dt, 23);
    Tensor d = rnd(rows, cols, dt, 24);
    Tensor out = Tensor::empty_on(vk(), rows, cols, dt);
    const double ideal = 5.0 * rows * cols * esz(dt);
    compare("elementwise chain (10 ops)", rows, cols, dt, ideal, [&] {
        jit::store(out, ((a * b + c) * jit::silu(d) - a * 0.5f) * jit::sigmoid(b) + jit::tanh(c) * d);
    });
}

}  // namespace

void run_jit_bench() {
#if BROTENSOR_HAS_VULKAN_TRACE_JIT
    std::printf("Trace JIT on Vulkan: op-by-op replay vs the SPIR-V compiler (ms, GB/s over ideal traffic)\n\n");
    std::printf("| Pattern | Shape | dtype | replay launches | replay ms | replay GB/s | fused ms | fused GB/s | speedup | fusion |\n");
    std::printf("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n");
    for (Dtype dt : {Dtype::FP32, Dtype::BF16, Dtype::FP16}) {
        bench_ln_modulate(4096, 3072, dt);
        bench_ln_modulate(1024, 4096, dt);
    }
    for (Dtype dt : {Dtype::FP32, Dtype::BF16}) {
        bench_norm_silu(262144, 128, dt);
        bench_norm_silu(65536, 384, dt);
        bench_norm_silu(1048576, 96, dt);
    }
    for (Dtype dt : {Dtype::FP32, Dtype::BF16, Dtype::FP16}) {
        bench_chain(4096, 4096, dt);
        bench_chain(1024, 1024, dt);
    }
#else
    std::printf("SKIP: built without the Vulkan trace compiler\n");
#endif
}

}  // namespace vkt
