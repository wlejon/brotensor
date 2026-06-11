// CUDA graph capture/replay: capture a fixed-shape rms_norm -> linear step,
// replay it against fresh inputs written in place, and confirm the replayed
// output matches a direct (non-graph) run of the same ops. Also checks that the
// same graph replays correctly for a second, different input — the core
// contract the Qwen3-TTS Code Predictor decode loop relies on.

#include <brotensor/cuda_graph.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// Direct (non-graph) reference: rms_norm(x, gamma) -> linear(W, bias) -> y.
static std::vector<float> direct_step(const std::vector<float>& x,
                                      const Tensor& gamma, const Tensor& W,
                                      const Tensor& bias, int B, int D,
                                      int out_dim, float eps) {
    Tensor X = Tensor::from_host_on(Device::CUDA, x.data(), B, D);
    Tensor normed, Y;
    brotensor::rms_norm_forward(X, gamma, eps, normed);
    brotensor::linear_forward_batched(W, bias, normed, Y);
    brotensor::sync_all();
    std::vector<float> out(static_cast<size_t>(B) * out_dim);
    Y.to(Device::CPU).copy_to_host(out.data());
    return out;
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_cuda_graph\n");

    const int B = 4, D = 64, out_dim = 48;
    const float eps = 1e-5f;
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto rand_vec = [&](int n) {
        std::vector<float> v(n);
        for (auto& e : v) e = dist(rng);
        return v;
    };

    std::vector<float> x0 = rand_vec(B * D);
    std::vector<float> gam = rand_vec(D);
    std::vector<float> w = rand_vec(out_dim * D);
    std::vector<float> bia = rand_vec(out_dim);

    Tensor gamma = Tensor::from_host_on(Device::CUDA, gam.data(), D, 1);
    Tensor W = Tensor::from_host_on(Device::CUDA, w.data(), out_dim, D);
    Tensor bias = Tensor::from_host_on(Device::CUDA, bia.data(), out_dim, 1);

    // Fixed step tensors reused across warm-up, capture, and every replay.
    Tensor X = Tensor::from_host_on(Device::CUDA, x0.data(), B, D);
    Tensor normed, Y;

    auto step = [&]() {
        brotensor::rms_norm_forward(X, gamma, eps, normed);
        brotensor::linear_forward_batched(W, bias, normed, Y);
    };

    // Warm-up: allocates `normed` and `Y` so capture allocates nothing.
    step();
    brotensor::sync_all();

    // Capture the identical step.
    brotensor::CudaGraph g;
    {
        brotensor::CudaGraphCapture cap;
        step();
        g = cap.finish();
    }
    CHECK(g.valid());

    // Replay against two fresh inputs written into X in place; each must match a
    // direct non-graph run of the same ops on that input.
    for (int trial = 0; trial < 2; ++trial) {
        std::vector<float> xn = rand_vec(B * D);
        Tensor tmp = Tensor::from_host_on(Device::CUDA, xn.data(), B, D);
        brotensor::copy_d2d(tmp, 0, X, 0, B * D);  // update X in place

        g.launch();
        brotensor::sync_all();
        std::vector<float> got(static_cast<size_t>(B) * out_dim);
        Y.to(Device::CPU).copy_to_host(got.data());

        std::vector<float> ref =
            direct_step(xn, gamma, W, bias, B, D, out_dim, eps);

        float max_err = 0.0f;
        for (size_t i = 0; i < got.size(); ++i)
            max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
        std::printf("  trial %d  replay vs direct max_err=%g\n", trial, max_err);
        CHECK(max_err < 1e-4f);
    }

    // reset() empties the handle.
    g.reset();
    CHECK(!g.valid());

    // ── FP16 op-sequence capture (group_norm → layout → linear) ───────────
    // Regression coverage for the FP16 inference ops' capture-safety: every
    // launch and copy must land on the capture stream and perform no
    // unpaired allocation.
    {
        const int C = 8, H = 2, Wd = 2;  // tiny SD1.5-fixture shapes
        const int L = H * Wd;
        std::vector<uint16_t> xh(C * H * Wd), gh(C), bh(C), wh(C * C);
        auto f2h = [](float f) { return brotensor::fp32_to_fp16_bits(f); };
        for (size_t i = 0; i < xh.size(); ++i) xh[i] = f2h(dist(rng) * 0.1f);
        for (size_t i = 0; i < gh.size(); ++i) gh[i] = f2h(1.0f);
        for (size_t i = 0; i < bh.size(); ++i) bh[i] = f2h(0.0f);
        for (size_t i = 0; i < wh.size(); ++i) wh[i] = f2h(dist(rng) * 0.1f);
        Tensor Xh  = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), 1, C * H * Wd);
        Tensor Gg  = Tensor::from_host_fp16_on(Device::CUDA, gh.data(), C, 1);
        Tensor Gb  = Tensor::from_host_fp16_on(Device::CUDA, bh.data(), C, 1);
        Tensor Wp  = Tensor::from_host_fp16_on(Device::CUDA, wh.data(), C, C);
        Tensor gn_out, seq, proj;

        auto fp16_step = [&](int upto) {
            if (upto >= 1) brotensor::group_norm_forward(Xh, Gg, Gb, 1, C, H, Wd, 2, 1e-6f, gn_out);
            if (upto >= 2) brotensor::nchw_to_sequence(gn_out, 1, C, H, Wd, seq);
            if (upto >= 3) brotensor::linear_forward_batched_fp16(Wp, nullptr, seq, proj);
        };
        for (int upto = 1; upto <= 3; ++upto) {
            fp16_step(upto);
            brotensor::sync_all();
            try {
                brotensor::CudaGraph g16;
                {
                    brotensor::CudaGraphCapture cap;
                    fp16_step(upto);
                    g16 = cap.finish();
                }
                g16.launch();
                brotensor::sync_all();
                std::printf("  fp16 capture upto=%d OK\n", upto);
            } catch (const std::exception& e) {
                std::printf("  fp16 capture upto=%d FAILED: %s\n", upto, e.what());
                ++g_failures;
            }
        }
    }

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
