// `brotensor_test_vulkan --bench-gemm`: GEMM throughput on the vk-spike's
// shapes (../vk-spike/RESULTS.md), through the public matmul_abt (FP16, the
// linear-layer NT layout), next to the spike's hipBLAS and coopmat figures;
// then each tile configuration, BF16 (converted at load), the SIMT fallback,
// FP32, and the GEMV kernel's bandwidth on decode shapes. Not part of ctest.
//
// Timing: wall clock around a batch of back-to-back launches with one device
// sync at the end (each launch is a separate, barrier-ordered dispatch, the
// same serialisation the spike measured), median of 7 batches after warm-up.

#include "test_vulkan_common.h"

#include "detail/gemm.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

struct Shape {
    int M, N, K;
    const char* tag;
    double hipblas, spike;   // TF/s from vk-spike RESULTS.md (run 2)
};

const Shape kShapes[] = {
    {4096, 4096, 4096, "square4k", 22.7, 24.9},
    {2048, 2048, 2048, "square2k", 38.8, 30.4},
    {8192, 8192, 8192, "square8k", 25.2, 21.2},
    {512, 3072, 1024, "qwen0.6B-up-pp512", 28.8, 25.2},
    {512, 1024, 3072, "qwen0.6B-down-pp512", 33.8, 26.2},
    {512, 4096, 4096, "8B-qo-pp512", 20.6, 21.0},
    {512, 12288, 4096, "8B-up-pp512", 21.6, 17.7},
    {512, 4096, 12288, "8B-down-pp512", 16.2, 17.5},
    {4096, 3072, 3072, "DiT-3072-L4096", 36.4, 32.1},
    {4096, 12288, 3072, "DiT-mlp-L4096", 32.5, 31.7},
};

template <class F>
double median_ms(F&& launch_once) {
    // Warm up (pipeline creation, clocks), then size a batch to ~150 ms.
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

// Relative error of a few sampled entries of C = A B^T against a double sum.
double sample_error(const std::vector<float>& a, const std::vector<float>& b, const Tensor& C, int M, int N, int K) {
    const auto c = download(C);
    Rng r(7);
    double num = 0, den = 0;
    for (int t = 0; t < 64; ++t) {
        const int m = static_cast<int>(r.next() % M), n = static_cast<int>(r.next() % N);
        double ref = 0;
        for (int k = 0; k < K; ++k) ref += double(a[std::size_t(m) * K + k]) * b[std::size_t(n) * K + k];
        num += (c[std::size_t(m) * N + n] - ref) * (c[std::size_t(m) * N + n] - ref);
        den += ref * ref;
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

double run_abt(const Tensor& A, const Tensor& B, Tensor& C, int M, int N, int K) {
    return median_ms([&] { brotensor::matmul_abt(A, B, C, 1, M, N, K, 0, 0, 0, nullptr, 0); });
}

void bench_gemv() {
    std::printf("\nGEMV (decode, Y = X W^T + b, W FP16; GB/s of W read)\n%-24s %8s %8s %8s %8s\n", "W (N x K)", "B=1",
                "B=2", "B=4", "B=8");
    const int gemv_shapes[][2] = {{4096, 4096}, {12288, 4096}, {4096, 12288}, {1024, 1024}, {151936, 1024}};
    for (const auto& w : gemv_shapes) {
        const int N = w[0], K = w[1];
        // Rotate through enough weight copies that the 32 MiB Infinity
        // Cache cannot hold them (the spike's rule).
        const std::size_t bytes = std::size_t(N) * K * 2;
        const int copies = static_cast<int>(std::max<std::size_t>(1, (512u << 20) / bytes));
        std::vector<Tensor> Ws;
        const auto wv = random_values(std::size_t(N) * K, 3, -0.5f, 0.5f, Dtype::FP16);
        for (int c = 0; c < copies; ++c) Ws.push_back(upload(wv, N, K, Dtype::FP16));
        char tag[32];
        std::snprintf(tag, sizeof tag, "%dx%d", N, K);
        std::printf("%-24s", tag);
        for (int B : {1, 2, 4, 8}) {
            const auto xv = random_values(std::size_t(B) * K, 4, -0.5f, 0.5f, Dtype::FP16);
            Tensor X = upload(xv, B, K, Dtype::FP16), Y;
            int c = 0;
            const double ms = median_ms([&] {
                brotensor::linear_forward_batched_fp16(Ws[static_cast<std::size_t>(c)], nullptr, X, Y);
                c = (c + 1) % copies;
            });
            std::printf(" %8.0f", double(bytes) / (ms * 1e6));
        }
        std::printf("\n");
    }
}

}  // namespace

void run_gemm_bench() {
    if (std::getenv("BROTENSOR_VK_BENCH_GEMV")) {   // the GEMV table alone
        bench_gemv();
        return;
    }
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("GEMM benchmark on %s (%s)\n", info.name.c_str(), info.driver.c_str());
    std::printf("C(M,N) = A(M,K) B(N,K)^T through matmul_abt; TF/s = 2MNK / time\n\n");
    std::printf("%-22s %-16s %8s %8s %8s %7s %7s %9s\n", "shape", "MxNxK", "hipBLAS", "spike", "vulkan", "vk/hip",
                "vk/spk", "rel.err");
    const bool quick = std::getenv("BROTENSOR_VK_BENCH_QUICK") != nullptr;
    const char* only = std::getenv("BROTENSOR_VK_BENCH_SHAPE");   // one shape by tag
    for (const Shape& s : kShapes) {
        if (quick && s.M == 8192) continue;
        if (only && std::string(only) != s.tag) continue;
        const auto av = random_values(std::size_t(s.M) * s.K, 1, -0.5f, 0.5f, Dtype::FP16);
        const auto bv = random_values(std::size_t(s.N) * s.K, 2, -0.5f, 0.5f, Dtype::FP16);
        Tensor A = upload(av, s.M, s.K, Dtype::FP16), B = upload(bv, s.N, s.K, Dtype::FP16);
        Tensor C = Tensor::empty_on(vk(), s.M, s.N, Dtype::FP16);
        const double ms = run_abt(A, B, C, s.M, s.N, s.K);
        const double tf = 2.0 * s.M * s.N * double(s.K) / (ms * 1e9);
        char dims[32];
        std::snprintf(dims, sizeof dims, "%dx%dx%d", s.M, s.N, s.K);
        std::printf("%-22s %-16s %8.1f %8.1f %8.1f %7.2f %7.2f %9.1e\n", s.tag, dims, s.hipblas, s.spike, tf,
                    tf / s.hipblas, tf / s.spike, sample_error(av, bv, C, s.M, s.N, s.K));
    }

    if (only) return;
    if (std::getenv("BROTENSOR_VK_BENCH_TILES")) {
        std::printf("\nper tile configuration (TF/s)\n%-22s", "shape");
        const int ncfg = dv::set_gemm_cm_config(-1);
        for (int c = 0; c < ncfg; ++c) std::printf(" %16s", dv::gemm_cm_config_name(c));
        std::printf("\n");
        for (const Shape& s : kShapes) {
            if (s.M == 8192) continue;
            const auto av = random_values(std::size_t(s.M) * s.K, 1, -0.5f, 0.5f, Dtype::FP16);
            const auto bv = random_values(std::size_t(s.N) * s.K, 2, -0.5f, 0.5f, Dtype::FP16);
            Tensor A = upload(av, s.M, s.K, Dtype::FP16), B = upload(bv, s.N, s.K, Dtype::FP16);
            Tensor C = Tensor::empty_on(vk(), s.M, s.N, Dtype::FP16);
            std::printf("%-22s", s.tag);
            for (int c = 0; c < ncfg; ++c) {
                dv::set_gemm_cm_config(c);
                const double ms = run_abt(A, B, C, s.M, s.N, s.K);
                std::printf(" %16.1f", 2.0 * s.M * s.N * double(s.K) / (ms * 1e9));
            }
            std::printf("\n");
        }
        dv::set_gemm_cm_config(-1);
    }

    std::printf("\nother paths (TF/s)\n%-22s %10s %10s %10s %10s\n", "shape", "bf16-cm", "f16-simt", "f32-simt",
                "nn-f16");
    for (const Shape& s : kShapes) {
        if (s.M == 8192 || (quick && s.M == 4096 && s.N == 12288)) continue;
        const double flop = 2.0 * s.M * s.N * double(s.K);
        const auto av = random_values(std::size_t(s.M) * s.K, 1, -0.5f, 0.5f, Dtype::FP16);
        const auto bv = random_values(std::size_t(s.N) * s.K, 2, -0.5f, 0.5f, Dtype::FP16);
        double r[4];
        {
            Tensor A = upload(av, s.M, s.K, Dtype::BF16), B = upload(bv, s.N, s.K, Dtype::BF16);
            Tensor C = Tensor::empty_on(vk(), s.M, s.N, Dtype::BF16);
            r[0] = flop / (run_abt(A, B, C, s.M, s.N, s.K) * 1e9);
        }
        {
            dv::set_gemm_override(2);
            Tensor A = upload(av, s.M, s.K, Dtype::FP16), B = upload(bv, s.N, s.K, Dtype::FP16);
            Tensor C = Tensor::empty_on(vk(), s.M, s.N, Dtype::FP16);
            r[1] = flop / (run_abt(A, B, C, s.M, s.N, s.K) * 1e9);
            dv::set_gemm_override(0);
        }
        {
            Tensor A = upload(av, s.M, s.K, Dtype::FP32), B = upload(bv, s.N, s.K, Dtype::FP32);
            Tensor C = Tensor::empty_on(vk(), s.M, s.N, Dtype::FP32);
            r[2] = flop / (run_abt(A, B, C, s.M, s.N, s.K) * 1e9);
        }
        {
            // matmul's NN layout: B stored (K, N).
            Tensor A = upload(av, s.M, s.K, Dtype::FP16), B = upload(bv, s.K, s.N, Dtype::FP16), C;
            r[3] = flop / (median_ms([&] { brotensor::matmul(A, B, C); }) * 1e9);
        }
        std::printf("%-22s %10.1f %10.1f %10.1f %10.1f\n", s.tag, r[0], r[1], r[2], r[3]);
    }

    bench_gemv();
}

}  // namespace vkt
