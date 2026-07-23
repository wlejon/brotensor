// Micro-benchmark: the elementwise / normalization family, reported as
// achieved DRAM bandwidth.
//
// Every op here is memory-bound — it touches each element a constant number
// of times and does a handful of flops on it — so the only interesting number
// is what fraction of HBM peak it reaches (~1008 GB/s on a 4090). A scalar
// kernel that loads one 2-byte __half per thread cannot saturate the bus; a
// vectorised one that loads 16 bytes per thread can. This bench is the
// before/after for that change.
//
// Every individual buffer is sized past the 72 MB L2 on AD102 so the next
// iteration re-reads from DRAM rather than cache — otherwise the numbers are
// fantasy (see rows_for).
//
// Timing goes through bench_helpers.h: an idle 4090 parks at 285 MHz, so a
// few-millisecond measurement samples whatever clock happened to be in effect.
// The harness spins the clocks up first and reports the best of N samples.
//
// Each row also spot-checks the output against a host FP32 reference on a
// strided sample of the elements (a full reference over 67M elements costs
// more than the benchmark), so a fast-but-wrong kernel can't pass silently.
//
// NOT registered with ctest — invoke manually:
//   ./build-cuda/tests/Release/brotensor_bench_elementwise

#include <brotensor/ops.h>
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

// AD102 L2 is 72 MB. Anything below this streams out of cache and reports
// bandwidth the DRAM bus never delivered.
constexpr double kL2Bytes = 72.0 * 1024.0 * 1024.0;

const char* dt_name(Dtype dt) {
    switch (dt) {
        case Dtype::FP32: return "fp32";
        case Dtype::FP16: return "fp16";
        case Dtype::BF16: return "bf16";
        default:          return "????";
    }
}

int dt_bytes(Dtype dt) { return dt == Dtype::FP32 ? 4 : 2; }

// ─── host <-> device helpers ───────────────────────────────────────────────

std::vector<float> rand_vec(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor upload(const std::vector<float>& v, int rows, int cols, Dtype dt) {
    if (dt == Dtype::FP32) {
        return Tensor::from_host_on(Device::CUDA, v.data(), rows, cols);
    }
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        h[i] = dt == Dtype::FP16 ? brotensor::fp32_to_fp16_bits(v[i])
                                 : brotensor::fp32_to_bf16_bits(v[i]);
    }
    return dt == Dtype::FP16
               ? Tensor::from_host_fp16_on(Device::CUDA, h.data(), rows, cols)
               : Tensor::from_host_bf16_on(Device::CUDA, h.data(), rows, cols);
}

std::vector<float> download(const Tensor& t) {
    const size_t n = t.size();
    std::vector<float> v(n);
    if (t.dtype == Dtype::FP32) {
        t.copy_to_host(v.data());
        brotensor::sync_all();
        return v;
    }
    std::vector<uint16_t> h(n);
    if (t.dtype == Dtype::FP16) t.copy_to_host_fp16(h.data());
    else                        t.copy_to_host_bf16(h.data());
    brotensor::sync_all();
    for (size_t i = 0; i < n; ++i) {
        v[i] = t.dtype == Dtype::FP16 ? brotensor::fp16_bits_to_fp32(h[i])
                                      : brotensor::bf16_bits_to_fp32(h[i]);
    }
    return v;
}

// Round-trip a host FP32 value through the storage dtype so the reference
// sees the same operands the kernel read.
float q(float v, Dtype dt) {
    switch (dt) {
        case Dtype::FP16:
            return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
        case Dtype::BF16:
            return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
        default:
            return v;
    }
}

// Error bound in the repo's usual `atol + rtol*|ref|` form (see
// test_activations.cpp). BF16 carries 8 mantissa bits, FP16 11, FP32 24.
//
// The FP32 rtol is looser than the 1e-5 test_activations.cpp uses because
// this bench samples 4096 points per op against that test's 257, so it
// reaches the tail of the input range. gelu_backward is the binding case:
// its gradient evaluates 0.5*(1+t) + 0.5*v*(1-t*t)*du, and `1 - t*t`
// cancels hard once |v| grows and t -> 1 (at v=3, t~=0.995 and 1-t*t~=0.01
// is a difference of two numbers near 1). Losing two decimal digits there,
// then scaling by v*du ~= 5, puts ~2e-5 relative error on the result as
// ordinary FP32 behaviour rather than a kernel defect.
struct Tol { float atol, rtol; };

Tol tol_for(Dtype dt) {
    switch (dt) {
        case Dtype::BF16: return {2e-2f, 4e-2f};
        case Dtype::FP16: return {2e-3f, 5e-3f};
        default:          return {1e-5f, 5e-5f};
    }
}

// Spot-check `got` against ref(i) on a strided sample. Full references over
// 67M elements dominate the benchmark's runtime and add nothing.
bool spot_check(const std::vector<float>& got,
                const std::function<float(size_t)>& ref,
                Dtype dt, const char* what) {
    constexpr size_t kSamples = 4096;
    const size_t n = got.size();
    const size_t stride = n > kSamples ? n / kSamples : 1;
    const Tol tol = tol_for(dt);
    double worst_abs = 0.0, worst_ref = 0.0;
    bool bad = false;
    for (size_t i = 0; i < n; i += stride) {
        const double r = ref(i);
        const double e = std::fabs(got[i] - r);
        if (e > worst_abs) { worst_abs = e; worst_ref = r; }
        if (e > tol.atol + tol.rtol * std::fabs(r)) bad = true;
    }
    if (bad) {
        std::printf("      ^^ MISMATCH %s: worst |err| %.3g at ref %.3g "
                    "(bound %.3g)\n", what, worst_abs, worst_ref,
                    tol.atol + tol.rtol * std::fabs(worst_ref));
        ++g_failures;
        return false;
    }
    return true;
}

void report(const char* op, const char* shape, Dtype dt, double bytes,
            float ms, bool ok) {
    std::printf("  %-22s %-16s %s  %8.3f ms  %7.1f GB/s  %s\n", op, shape,
                dt_name(dt), ms, bytes / (ms * 1e-3) / 1e9,
                ok ? "ok" : "MISMATCH");
}

// Pick a row count that makes ONE buffer exceed L2 by a comfortable margin.
//
// Sizing the *aggregate* working set past L2 is not enough: if each individual
// buffer still fits, a fraction of it stays resident between iterations and
// the reported bandwidth exceeds the DRAM bus (an early version of this bench
// reported 1084 GB/s on a 1008 GB/s part, which is how the bug surfaced).
// Sizing each buffer to 2x L2 means nothing survives to the next iteration.
int rows_for(int cols, Dtype dt) {
    const double per_row = static_cast<double>(cols) * dt_bytes(dt);
    const int r = static_cast<int>((kL2Bytes * 2.0) / per_row) + 1;
    return r < 64 ? 64 : r;
}

// ─── unary forward: y = f(x) ───────────────────────────────────────────────

using UnaryFn  = void (*)(const Tensor&, Tensor&);
using ScalarFn = float (*)(float);

float silu_ref(float v)    { return v / (1.0f + std::exp(-v)); }
float relu_ref(float v)    { return v > 0.0f ? v : 0.0f; }
float sigmoid_ref(float v) { return 1.0f / (1.0f + std::exp(-v)); }
float tanh_ref(float v)    { return std::tanh(v); }
float gelu_ref(float v) {
    const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}

void bench_unary(const char* name, UnaryFn fn, ScalarFn ref, Dtype dt,
                 int cols) {
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv = rand_vec(n, 0x1234u, 3.0f);
    Tensor X = upload(xv, rows, cols, dt);
    Tensor Y;
    fn(X, Y);
    brotensor::sync_all();

    auto yv = download(Y);
    const bool ok = spot_check(yv, [&](size_t i) { return ref(q(xv[i], dt)); },
                               dt, name);

    const float ms = bt_bench::time_min_ms([&] { fn(X, Y); });
    const double bytes = 2.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report(name, shape, dt, bytes, ms, ok);
}

// ─── unary backward: dX = f'(x) * dY ───────────────────────────────────────

using BwdFn = void (*)(const Tensor&, const Tensor&, Tensor&);

float silu_grad_ref(float v) {
    const float s = 1.0f / (1.0f + std::exp(-v));
    return s * (1.0f + v * (1.0f - s));
}
float gelu_grad_ref(float v) {
    const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    const float t = std::tanh(u);
    const float du = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du;
}

void bench_unary_bwd(const char* name, BwdFn fn, ScalarFn grad, Dtype dt,
                     int cols) {
    // reads x and dY, writes dX.
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv  = rand_vec(n, 0x2345u, 3.0f);
    auto dyv = rand_vec(n, 0x3456u, 1.0f);
    Tensor X  = upload(xv,  rows, cols, dt);
    Tensor dY = upload(dyv, rows, cols, dt);
    Tensor dX;
    fn(X, dY, dX);
    brotensor::sync_all();

    auto gv = download(dX);
    const bool ok = spot_check(gv, [&](size_t i) {
        return grad(q(xv[i], dt)) * q(dyv[i], dt);
    }, dt, name);

    const float ms = bt_bench::time_min_ms([&] { fn(X, dY, dX); });
    const double bytes = 3.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report(name, shape, dt, bytes, ms, ok);
}

// ─── binary / scalar in-place ──────────────────────────────────────────────

void bench_add_inplace(Dtype dt, int cols) {
    const int rows = rows_for(cols, dt);   // read y, read x, write y
    const size_t n = static_cast<size_t>(rows) * cols;
    auto yv = rand_vec(n, 0x4567u, 1.0f);
    auto xv = rand_vec(n, 0x5678u, 1.0f);
    Tensor Y = upload(yv, rows, cols, dt);
    Tensor X = upload(xv, rows, cols, dt);
    brotensor::add_inplace(Y, X);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) {
        return q(yv[i], dt) + q(xv[i], dt);
    }, dt, "add_inplace");

    // Re-upload: the timed loop mutates Y, and an unbounded accumulation would
    // drift into inf and change the arithmetic being measured.
    Y = upload(yv, rows, cols, dt);
    const float ms = bt_bench::time_min_ms([&] { brotensor::add_inplace(Y, X); });
    const double bytes = 3.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("add_inplace", shape, dt, bytes, ms, ok);
}

void bench_mul_inplace(Dtype dt, int cols) {
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto yv = rand_vec(n, 0x6789u, 1.0f);
    // Values near 1 so the timed loop's repeated multiply neither vanishes
    // nor overflows.
    std::vector<float> xv(n);
    for (size_t i = 0; i < n; ++i) xv[i] = 1.0f + 0.01f * std::sin(float(i));
    Tensor Y = upload(yv, rows, cols, dt);
    Tensor X = upload(xv, rows, cols, dt);
    brotensor::mul_inplace(Y, X);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) {
        return q(yv[i], dt) * q(xv[i], dt);
    }, dt, "mul_inplace");

    Y = upload(yv, rows, cols, dt);
    const float ms = bt_bench::time_min_ms([&] { brotensor::mul_inplace(Y, X); });
    const double bytes = 3.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("mul_inplace", shape, dt, bytes, ms, ok);
}

void bench_scale_inplace(Dtype dt, int cols) {
    const int rows = rows_for(cols, dt);   // read y, write y
    const size_t n = static_cast<size_t>(rows) * cols;
    auto yv = rand_vec(n, 0x789Au, 1.0f);
    Tensor Y = upload(yv, rows, cols, dt);
    brotensor::scale_inplace(Y, 1.5f);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) { return q(yv[i], dt) * 1.5f; },
                               dt, "scale_inplace");

    // Scale by 1.0 in the timed loop so repeated application is a no-op on the
    // values (same memory traffic, no drift to inf).
    Y = upload(yv, rows, cols, dt);
    const float ms = bt_bench::time_min_ms([&] { brotensor::scale_inplace(Y, 1.0f); });
    const double bytes = 2.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("scale_inplace", shape, dt, bytes, ms, ok);
}

// ─── geglu: (R, 2D) -> (R, D) ──────────────────────────────────────────────

void bench_geglu(Dtype dt, int d_out) {
    const int cols = 2 * d_out;
    const int rows = rows_for(cols, dt);   // read 2D, write D
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv = rand_vec(n, 0x8ABCu, 2.0f);
    Tensor X = upload(xv, rows, cols, dt);
    Tensor Y;
    brotensor::geglu_forward(X, Y);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) {
        const size_t r = i / d_out, c = i % d_out;
        const size_t base = r * static_cast<size_t>(cols);
        return q(xv[base + c], dt) * gelu_ref(q(xv[base + d_out + c], dt));
    }, dt, "geglu_forward");

    const float ms = bt_bench::time_min_ms([&] { brotensor::geglu_forward(X, Y); });
    const double bytes = static_cast<double>(dt_bytes(dt)) *
                         (static_cast<double>(n) + static_cast<double>(rows) * d_out);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("geglu_forward", shape, dt, bytes, ms, ok);
}

// ─── normalizations ────────────────────────────────────────────────────────

void bench_rms_norm(Dtype dt, int cols) {
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv = rand_vec(n, 0x9BCDu, 1.0f);
    auto gv = rand_vec(static_cast<size_t>(cols), 0xACDEu, 0.5f);
    Tensor X = upload(xv, rows, cols, dt);
    Tensor G = upload(gv, cols, 1, dt);
    Tensor Y;
    brotensor::rms_norm_forward(X, G, 1e-5f, Y);
    brotensor::sync_all();

    auto got = download(Y);
    // Reference recomputes the row RMS in FP64 for the sampled rows only.
    const bool ok = spot_check(got, [&](size_t i) {
        const size_t r = i / cols, c = i % cols;
        const size_t base = r * static_cast<size_t>(cols);
        double ss = 0.0;
        for (int j = 0; j < cols; ++j) {
            const double v = q(xv[base + j], dt);
            ss += v * v;
        }
        const double rms = std::sqrt(ss / cols + 1e-5);
        return static_cast<float>(q(xv[base + c], dt) * q(gv[c], dt) / rms);
    }, dt, "rms_norm_forward");

    const float ms = bt_bench::time_min_ms([&] { brotensor::rms_norm_forward(X, G, 1e-5f, Y); });
    const double bytes = 2.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("rms_norm_forward", shape, dt, bytes, ms, ok);
}

void bench_layernorm_fp16(int cols) {
    const Dtype dt = Dtype::FP16;
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv = rand_vec(n, 0xBDEFu, 1.0f);
    auto gv = rand_vec(static_cast<size_t>(cols), 0xCEF0u, 0.5f);
    auto bv = rand_vec(static_cast<size_t>(cols), 0xDF01u, 0.2f);
    Tensor X = upload(xv, rows, cols, dt);
    Tensor G = upload(gv, cols, 1, dt);
    Tensor B = upload(bv, cols, 1, dt);
    Tensor Y;
    brotensor::layernorm_forward_inference_batched_fp16(X, G, B, Y, 1e-5f);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) {
        const size_t r = i / cols, c = i % cols;
        const size_t base = r * static_cast<size_t>(cols);
        double mean = 0.0;
        for (int j = 0; j < cols; ++j) mean += q(xv[base + j], dt);
        mean /= cols;
        // Two-pass variance — sum of squared deviations, never E[x^2]-E[x]^2.
        double var = 0.0;
        for (int j = 0; j < cols; ++j) {
            const double d = q(xv[base + j], dt) - mean;
            var += d * d;
        }
        var /= cols;
        const double xhat = (q(xv[base + c], dt) - mean) / std::sqrt(var + 1e-5);
        return static_cast<float>(xhat * q(gv[c], dt) + q(bv[c], dt));
    }, dt, "layernorm_fp16");

    const float ms = bt_bench::time_min_ms([&] {
        brotensor::layernorm_forward_inference_batched_fp16(X, G, B, Y, 1e-5f);
    });
    const double bytes = 2.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("layernorm_inf_fp16", shape, dt, bytes, ms, ok);
}

void bench_softmax_rows(int cols) {
    const Dtype dt = Dtype::FP32;   // softmax_rows is FP32 on both backends
    const int rows = rows_for(cols, dt);
    const size_t n = static_cast<size_t>(rows) * cols;
    auto xv = rand_vec(n, 0xE012u, 4.0f);
    Tensor X = upload(xv, rows, cols, dt);
    Tensor Y;
    brotensor::softmax_rows_forward(X, Y, rows, cols);
    brotensor::sync_all();

    auto got = download(Y);
    const bool ok = spot_check(got, [&](size_t i) {
        const size_t r = i / cols, c = i % cols;
        const size_t base = r * static_cast<size_t>(cols);
        double m = -1e30;
        for (int j = 0; j < cols; ++j) m = std::max<double>(m, xv[base + j]);
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += std::exp(xv[base + j] - m);
        return static_cast<float>(std::exp(xv[base + c] - m) / s);
    }, dt, "softmax_rows");

    const float ms = bt_bench::time_min_ms([&] {
        brotensor::softmax_rows_forward(X, Y, rows, cols);
    });
    const double bytes = 2.0 * static_cast<double>(n) * dt_bytes(dt);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", rows, cols);
    report("softmax_rows", shape, dt, bytes, ms, ok);
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available — skipping elementwise bench\n");
        return 0;
    }
    // Pull the SM clock off its P8 idle floor before any timing.
    bt_bench::spin_up();
    std::printf("brotensor_bench_elementwise  (warmup %.0f ms/op, best of %d)\n",
                bt_bench::kWarmupMs, bt_bench::kSamples);
    std::printf("working sets sized past the 72 MB L2; peak HBM on a 4090 is "
                "~1008 GB/s\n\n");

    const int kCols = 4096;   // a transformer-ish hidden width

    std::printf("-- unary forward --\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        bench_unary("silu_forward",    brotensor::silu_forward,    silu_ref,    dt, kCols);
        bench_unary("gelu_forward",    brotensor::gelu_forward,    gelu_ref,    dt, kCols);
        bench_unary("relu_forward",    brotensor::relu_forward,    relu_ref,    dt, kCols);
        bench_unary("tanh_forward",    brotensor::tanh_forward,    tanh_ref,    dt, kCols);
        bench_unary("sigmoid_forward", brotensor::sigmoid_forward, sigmoid_ref, dt, kCols);
    }

    std::printf("\n-- unary backward --\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        bench_unary_bwd("silu_backward", brotensor::silu_backward, silu_grad_ref, dt, kCols);
        bench_unary_bwd("gelu_backward", brotensor::gelu_backward, gelu_grad_ref, dt, kCols);
    }

    std::printf("\n-- in-place binary / scalar --\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        bench_add_inplace(dt, kCols);
        bench_mul_inplace(dt, kCols);
        bench_scale_inplace(dt, kCols);
    }

    std::printf("\n-- gated / fused --\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        bench_geglu(dt, kCols);
    }

    std::printf("\n-- normalization --\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16}) {
        bench_rms_norm(dt, kCols);
    }
    bench_layernorm_fp16(kCols);
    bench_softmax_rows(kCols);

    if (g_failures != 0) {
        std::printf("\n%d MISMATCH(es)\n", g_failures);
        return 1;
    }
    std::printf("\nall rows verified\n");
    return 0;
}
