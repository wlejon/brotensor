// Micro-benchmark: the two kernels on the AR-decode critical path, at the
// exact shapes a Qwen3-0.6B decode step issues them.
//
//   * linear_forward_batched_fp16 with B == 1 (the skinny-batch GEMV path)
//     for every projection shape in the layer stack plus the lm_head.
//   * flash_attention_decode_masked at the decode attention shape, short and
//     long caches.
//
// Reports achieved GB/s against the kernel's unavoidable DRAM traffic (the
// weight matrix for the GEMV; the valid K/V rows for attention), so the
// number is directly comparable to the device's HBM bandwidth (~1 TB/s on a
// 4090). Each row also sanity-checks the kernel output against a host FP32
// reference so a fast-but-wrong kernel can't pass silently.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_decode_kernels

#include <brotensor/cuda_graph.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Dtype;
using brotensor::Tensor;

namespace {

constexpr int REPLAYS = 50;

int g_failures = 0;

// Time `body(i)` for i in [0, chunk) by capturing the whole chunk into one
// CUDA graph and replaying it. Eager launches on Windows cost 3-5 us each —
// more than several of these kernels — and the production decode path replays
// these ops inside a captured graph anyway, so graph replay is the honest
// timing. body(i) must be capture-safe (no allocations: run it once eagerly
// first so output tensors exist).
float time_graph_ms(int chunk, const std::function<void(int)>& body) {
    for (int i = 0; i < chunk; ++i) body(i);   // warm-up + allocations
    brotensor::sync_all();
    brotensor::CudaGraph g;
    {
        brotensor::CudaGraphCapture cap;
        for (int i = 0; i < chunk; ++i) body(i);
        g = cap.finish();
    }
    g.launch();
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < REPLAYS; ++r) g.launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / (REPLAYS * chunk);
}

std::vector<float> rand_vec(size_t n, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor upload16(const std::vector<float>& v, int rows, int cols, Dtype dt) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        h[i] = dt == Dtype::FP16 ? brotensor::fp32_to_fp16_bits(v[i])
                                 : brotensor::fp32_to_bf16_bits(v[i]);
    }
    return dt == Dtype::FP16
               ? Tensor::from_host_fp16_on(brotensor::Device::CUDA, h.data(),
                                           rows, cols)
               : Tensor::from_host_bf16_on(brotensor::Device::CUDA, h.data(),
                                           rows, cols);
}

std::vector<float> download16(const Tensor& t) {
    std::vector<uint16_t> h(t.size());
    if (t.dtype == Dtype::FP16) {
        t.copy_to_host_fp16(h.data());
    } else {
        t.copy_to_host_bf16(h.data());
    }
    brotensor::sync_all();
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        v[i] = t.dtype == Dtype::FP16 ? brotensor::fp16_bits_to_fp32(h[i])
                                      : brotensor::bf16_bits_to_fp32(h[i]);
    }
    return v;
}

// Round-trip a host FP32 value through the storage dtype, so references
// see the same quantized operands the kernel reads.
float q16(float v, Dtype dt) {
    return dt == Dtype::FP16
               ? brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v))
               : brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
}

// In steady-state decode every weight read misses L2 (the model is far
// bigger than the 72 MB cache and each matrix is touched once per token).
// A bench that replays ONE matrix back-to-back would instead serve it from
// L2 and report fantasy bandwidth — so cycle through enough replicas that
// the working set exceeds L2.
constexpr double L2_DEFEAT_BYTES = 512.0 * 1024 * 1024;

int replicas_for(double bytes_each) {
    const int n = static_cast<int>(L2_DEFEAT_BYTES / bytes_each) + 1;
    return n < 1 ? 1 : (n > 64 ? 64 : n);
}

// ---------------------------------------------------------------------------
// GEMV: Y = W x + b at B == 1.

void bench_gemv(const char* name, int out_dim, int in_dim, Dtype dt) {
    std::mt19937 rng(0xb01dface);
    auto wv = rand_vec(static_cast<size_t>(out_dim) * in_dim, rng, 0.05f);
    auto xv = rand_vec(static_cast<size_t>(in_dim), rng, 0.5f);
    auto bv = rand_vec(static_cast<size_t>(out_dim), rng, 0.1f);

    const double w_bytes = 2.0 * static_cast<double>(out_dim) * in_dim;
    const int n_rep = replicas_for(w_bytes);
    std::vector<Tensor> Ws;
    Ws.reserve(static_cast<size_t>(n_rep));
    for (int i = 0; i < n_rep; ++i) Ws.push_back(upload16(wv, out_dim, in_dim, dt));
    Tensor X = upload16(xv, 1, in_dim, dt);
    Tensor B = upload16(bv, out_dim, 1, dt);
    Tensor Y;
    brotensor::linear_forward_batched_fp16(Ws[0], &B, X, Y);
    brotensor::sync_all();

    // Host FP32 reference over the dtype-quantized operands. Tolerance covers
    // FP32-accumulation reassociation plus the 16-bit output round.
    auto yv = download16(Y);
    double max_err = 0.0;
    for (int o = 0; o < out_dim; ++o) {
        double ref = q16(bv[static_cast<size_t>(o)], dt);
        for (int k = 0; k < in_dim; ++k) {
            ref += static_cast<double>(
                       q16(wv[static_cast<size_t>(o) * in_dim + k], dt)) *
                   q16(xv[static_cast<size_t>(k)], dt);
        }
        const double err = std::fabs(yv[static_cast<size_t>(o)] - ref) /
                           (std::fabs(ref) + 1e-3);
        if (err > max_err) max_err = err;
    }
    const bool ok = max_err < 5e-2;
    if (!ok) ++g_failures;

    const float ms = time_graph_ms(n_rep, [&](int i) {
        brotensor::linear_forward_batched_fp16(Ws[static_cast<size_t>(i)], &B,
                                               X, Y);
    });
    // Weight matrix dominates; x, bias and y are noise but counted anyway.
    const double bytes = w_bytes + 2.0 * (in_dim + 2.0 * out_dim);
    std::printf("gemv  %-22s %6d x %-6d %s  %9.1f us  %7.1f GB/s  %s\n", name,
                out_dim, in_dim, dt == Dtype::FP16 ? "fp16" : "bf16",
                ms * 1e3, bytes / (ms * 1e-3) / 1e9,
                ok ? "ok" : "MISMATCH");
}

// ---------------------------------------------------------------------------
// Decode attention (masked, the graph-path kernel).

void bench_attn(const char* name, int cap, int valid_len, int n_q, int n_kv,
                int head_dim, Dtype dt) {
    std::mt19937 rng(0xfaceb0a7);
    const int dq  = n_q * head_dim;
    const int dkv = n_kv * head_dim;

    auto qv = rand_vec(static_cast<size_t>(dq), rng, 0.5f);
    auto kv = rand_vec(static_cast<size_t>(cap) * dkv, rng, 0.5f);
    auto vv = rand_vec(static_cast<size_t>(cap) * dkv, rng, 0.5f);

    Tensor Q = upload16(qv, 1, dq, dt);
    const double kv_bytes = 2.0 * 2.0 * static_cast<double>(cap) * dkv;
    const int n_rep = replicas_for(kv_bytes);
    std::vector<Tensor> Ks, Vs;
    Ks.reserve(static_cast<size_t>(n_rep));
    Vs.reserve(static_cast<size_t>(n_rep));
    for (int i = 0; i < n_rep; ++i) {
        Ks.push_back(upload16(kv, cap, dkv, dt));
        Vs.push_back(upload16(vv, cap, dkv, dt));
    }
    const Tensor& K = Ks[0];
    const Tensor& V = Vs[0];

    std::vector<float> mask_h(static_cast<size_t>(cap), 0.0f);
    for (int i = 0; i < valid_len; ++i) mask_h[static_cast<size_t>(i)] = 1.0f;
    Tensor mask = Tensor::from_host_on(brotensor::Device::CUDA, mask_h.data(),
                                       cap, 1);

    Tensor O;
    brotensor::flash_attention_decode_masked(
        Q, K, V, static_cast<const float*>(mask.data), n_q, n_kv, O);
    brotensor::sync_all();

    // The truncated eager kernel is the reference; the masked twin's contract
    // is bit-identical output, so compare at zero tolerance.
    Tensor O_ref;
    brotensor::flash_attention_decode(Q, K, V, valid_len, n_q, n_kv, O_ref);
    brotensor::sync_all();
    auto ov = download16(O);
    auto rv = download16(O_ref);
    bool ok = true;
    for (size_t i = 0; i < ov.size(); ++i) {
        if (ov[i] != rv[i]) { ok = false; break; }
    }
    if (!ok) ++g_failures;

    const float ms = time_graph_ms(n_rep, [&](int i) {
        const size_t s = static_cast<size_t>(i);
        brotensor::flash_attention_decode_masked(
            Q, Ks[s], Vs[s], static_cast<const float*>(mask.data), n_q, n_kv,
            O);
    });
    // Unavoidable traffic: each KV head's valid K rows once per *query group*
    // read (kernels may re-read per query head; the denominator stays the
    // ideal so the number penalizes re-reads), plus Q/O.
    const double bytes =
        2.0 * (2.0 * static_cast<double>(valid_len) * dkv + 2.0 * dq);
    std::printf("attn  %-22s cap %5d valid %5d  %s  %9.1f us  %7.1f GB/s  %s\n",
                name, cap, valid_len, dt == Dtype::FP16 ? "fp16" : "bf16",
                ms * 1e3, bytes / (ms * 1e-3) / 1e9,
                ok ? "ok" : "MISMATCH");
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available — skipping decode kernel bench\n");
        return 0;
    }

    std::printf("-- B=1 GEMV (linear_forward_batched_fp16 skinny path) --\n");
    // Qwen3-0.6B layer shapes (hidden 1024, 16q/8kv hd128, ffn 3072).
    for (Dtype dt : {Dtype::BF16, Dtype::FP16}) {
        bench_gemv("q_proj", 2048, 1024, dt);
        bench_gemv("kv_proj", 1024, 1024, dt);
        bench_gemv("o_proj", 1024, 2048, dt);
        bench_gemv("gate/up_proj", 3072, 1024, dt);
        bench_gemv("down_proj", 1024, 3072, dt);
    }
    bench_gemv("lm_head (151936)", 151936, 1024, Dtype::BF16);
    // Bigger-model shapes (Qwen3-8B-class: hidden 4096, ffn 12288).
    bench_gemv("8B qkv-ish", 6144, 4096, Dtype::BF16);
    bench_gemv("8B down_proj", 4096, 12288, Dtype::BF16);

    std::printf("\n-- decode attention (flash_attention_decode_masked) --\n");
    // Qwen3-0.6B: 16 q heads, 8 kv heads, head_dim 128.
    for (Dtype dt : {Dtype::BF16, Dtype::FP16}) {
        bench_attn("0.6B short", 640, 512, 16, 8, 128, dt);
        bench_attn("0.6B short again", 640, 512, 16, 8, 128, dt);
        bench_attn("0.6B 4k", 4096, 4096, 16, 8, 128, dt);
    }
    bench_attn("0.6B 16k", 16384, 16384, 16, 8, 128, Dtype::BF16);
    // 8B-class: 32 q heads, 8 kv heads.
    bench_attn("8B 4k", 4096, 4096, 32, 8, 128, Dtype::BF16);

    if (g_failures != 0) {
        std::printf("\n%d MISMATCH(es)\n", g_failures);
        return 1;
    }
    return 0;
}
