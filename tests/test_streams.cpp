// Multi-stream sanity for cuda_set_stream / cuda_current_stream /
// cuda_stream_sync: run matmul_gpu on two concurrent streams and verify
// results match the default-stream reference.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void test_two_streams_matmul() {
    std::printf("  two-stream matmul fp16\n");
    const int M = 64, K = 64, N = 64;
    std::mt19937 rng(0x77);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> A1(M*K), B1(K*N), A2(M*K), B2(K*N);
    for (auto& v : A1) v = dist(rng);
    for (auto& v : B1) v = dist(rng);
    for (auto& v : A2) v = dist(rng);
    for (auto& v : B2) v = dist(rng);

    GpuTensor Ag1, Bg1, Cg1, Ag2, Bg2, Cg2;
    auto ah1 = to_fp16(A1), bh1 = to_fp16(B1);
    auto ah2 = to_fp16(A2), bh2 = to_fp16(B2);
    brotensor::upload_fp16(ah1.data(), M, K, Ag1);
    brotensor::upload_fp16(bh1.data(), K, N, Bg1);
    brotensor::upload_fp16(ah2.data(), M, K, Ag2);
    brotensor::upload_fp16(bh2.data(), K, N, Bg2);

    // Reference: default-stream runs.
    GpuTensor Ref1, Ref2;
    brotensor::matmul_gpu(Ag1, Bg1, Ref1);
    brotensor::matmul_gpu(Ag2, Bg2, Ref2);
    std::vector<uint16_t> ref1(M*N), ref2(M*N);
    brotensor::cuda_sync();
    brotensor::download_fp16(Ref1, ref1.data());
    brotensor::download_fp16(Ref2, ref2.data());

    // Now: two streams, one matmul each.
    cudaStream_t s1 = nullptr, s2 = nullptr;
    if (cudaStreamCreate(&s1) != cudaSuccess) { CHECK(false); return; }
    if (cudaStreamCreate(&s2) != cudaSuccess) { CHECK(false); return; }

    brotensor::cuda_set_stream(reinterpret_cast<void*>(s1));
    CHECK(brotensor::cuda_current_stream() == reinterpret_cast<void*>(s1));
    brotensor::matmul_gpu(Ag1, Bg1, Cg1);

    brotensor::cuda_set_stream(reinterpret_cast<void*>(s2));
    brotensor::matmul_gpu(Ag2, Bg2, Cg2);

    // Sync both streams independently, then restore default.
    brotensor::cuda_stream_sync(reinterpret_cast<void*>(s1));
    brotensor::cuda_stream_sync(reinterpret_cast<void*>(s2));
    brotensor::cuda_set_stream(nullptr);
    CHECK(brotensor::cuda_current_stream() == nullptr);

    std::vector<uint16_t> got1(M*N), got2(M*N);
    brotensor::download_fp16(Cg1, got1.data());
    brotensor::download_fp16(Cg2, got2.data());
    brotensor::cuda_sync();

    for (int i = 0; i < M*N; ++i) {
        CHECK(got1[i] == ref1[i]);
        CHECK(got2[i] == ref2[i]);
    }

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_streams\n");
    test_two_streams_matmul();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
