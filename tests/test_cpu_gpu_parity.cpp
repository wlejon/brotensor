// CPU↔GPU parity. For each CPU op that has a 1:1 GPU counterpart, run the
// same input through both and compare element-wise within tight FP32
// tolerance. Compiled only when a GPU backend is enabled.

#include <brotensor/ops.h>
#include <brotensor/ops_cpu.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using brotensor::Tensor;
using brotensor::GpuTensor;

static int g_failures = 0;

static bool near_(float a, float b, float abs_eps, float rel_eps) {
    const float d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const float m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

static void compare(const std::string& tag,
                    const std::vector<float>& cpu,
                    const std::vector<float>& gpu,
                    float abs_eps, float rel_eps) {
    if (cpu.size() != gpu.size()) {
        std::printf("  FAIL  [%s]  size mismatch: cpu=%zu gpu=%zu\n",
                    tag.c_str(), cpu.size(), gpu.size());
        ++g_failures;
        return;
    }
    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const float d = std::fabs(cpu[i] - gpu[i]);
        if (d > max_abs) max_abs = d;
        if (!near_(cpu[i], gpu[i], abs_eps, rel_eps)) {
            if (bad < 5) {
                const float m = std::fmax(std::fabs(cpu[i]), std::fabs(gpu[i]));
                const float rd = m > 0.0f ? d / m : 0.0f;
                std::printf("  FAIL  [%s]  i=%zu cpu=%.9g gpu=%.9g abs=%.3g rel=%.3g\n",
                            tag.c_str(), i, cpu[i], gpu[i], d, rd);
            }
            ++bad;
        }
    }
    if (bad > 0) ++g_failures;
    std::printf("  [%s] max_abs=%g bad=%d / %zu\n",
                tag.c_str(), max_abs, bad, cpu.size());
}

// ---- helpers to fill a Tensor ------------------------------------------------

static void fill_random(Tensor& t, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (int i = 0; i < t.size(); ++i) t[i] = d(rng);
}

// ---- linear -----------------------------------------------------------------

static void parity_linear() {
    std::printf("linear parity\n");
    std::mt19937 rng(0xA1B2);

    const int OUT = 5, IN = 7;
    Tensor W(OUT, IN), b = Tensor::vec(OUT), x = Tensor::vec(IN);
    fill_random(W, rng, -0.5f, 0.5f);
    fill_random(b, rng, -0.3f, 0.3f);
    fill_random(x, rng, -1.0f, 1.0f);

    // CPU forward.
    Tensor yCpu = Tensor::vec(OUT);
    brotensor::linear_forward_cpu(W, b, x, yCpu);

    // GPU forward.
    GpuTensor Wg, bg, xg, yg;
    brotensor::upload(W, Wg);
    brotensor::upload(b, bg);
    brotensor::upload(x, xg);
    brotensor::linear_forward_gpu(Wg, bg, xg, yg);
    brotensor::cuda_sync();
    Tensor yGpu;
    brotensor::download(yg, yGpu);

    compare("linear_forward", yCpu.data, yGpu.data, 1e-5f, 1e-5f);

    // Backward.
    Tensor dY = Tensor::vec(OUT);
    fill_random(dY, rng, -1.0f, 1.0f);

    Tensor dXCpu = Tensor::vec(IN);
    Tensor dWCpu(OUT, IN);
    Tensor dBCpu = Tensor::vec(OUT);
    brotensor::linear_backward_cpu(W, x, dY, dXCpu, dWCpu, dBCpu);

    GpuTensor dYg, dXg, dWg(OUT, IN), dBg(OUT, 1);
    dWg.zero();
    dBg.zero();
    brotensor::upload(dY, dYg);
    brotensor::linear_backward_gpu(Wg, xg, dYg, dXg, dWg, dBg);
    brotensor::cuda_sync();

    Tensor dXGpu, dWGpu, dBGpu;
    brotensor::download(dXg, dXGpu);
    brotensor::download(dWg, dWGpu);
    brotensor::download(dBg, dBGpu);

    compare("linear_backward/dX", dXCpu.data, dXGpu.data, 1e-4f, 1e-5f);
    compare("linear_backward/dW", dWCpu.data, dWGpu.data, 1e-4f, 1e-5f);
    compare("linear_backward/dB", dBCpu.data, dBGpu.data, 1e-4f, 1e-5f);
}

// ---- relu / tanh / sigmoid --------------------------------------------------

static void parity_elementwise_act() {
    std::printf("relu/tanh/sigmoid parity\n");
    std::mt19937 rng(0xC3D4);
    const int N = 257;

    Tensor x = Tensor::vec(N), dY = Tensor::vec(N);
    fill_random(x,  rng, -3.0f, 3.0f);
    fill_random(dY, rng, -1.0f, 1.0f);

    GpuTensor xg, dYg;
    brotensor::upload(x,  xg);
    brotensor::upload(dY, dYg);

    // ReLU
    {
        Tensor yCpu = Tensor::vec(N), dXCpu = Tensor::vec(N);
        brotensor::relu_forward_cpu(x, yCpu);
        brotensor::relu_backward_cpu(x, dY, dXCpu);

        GpuTensor yg, dXg;
        brotensor::relu_forward_gpu(xg, yg);
        brotensor::relu_backward_gpu(xg, dYg, dXg);
        brotensor::cuda_sync();
        Tensor yGpu, dXGpu;
        brotensor::download(yg, yGpu);
        brotensor::download(dXg, dXGpu);
        compare("relu_forward",  yCpu.data,  yGpu.data,  1e-6f, 1e-6f);
        compare("relu_backward", dXCpu.data, dXGpu.data, 1e-6f, 1e-6f);
    }

    // tanh — backward consumes y (cached forward), not x.
    {
        Tensor yCpu = Tensor::vec(N), dXCpu = Tensor::vec(N);
        brotensor::tanh_forward_cpu(x, yCpu);
        brotensor::tanh_backward_cpu(yCpu, dY, dXCpu);

        GpuTensor yg, dXg;
        brotensor::tanh_forward_gpu(xg, yg);
        brotensor::tanh_backward_gpu(yg, dYg, dXg);
        brotensor::cuda_sync();
        Tensor yGpu, dXGpu;
        brotensor::download(yg, yGpu);
        brotensor::download(dXg, dXGpu);
        compare("tanh_forward",  yCpu.data,  yGpu.data,  1e-5f, 1e-5f);
        compare("tanh_backward", dXCpu.data, dXGpu.data, 1e-4f, 1e-5f);
    }

    // sigmoid — backward consumes y.
    {
        Tensor yCpu = Tensor::vec(N), dXCpu = Tensor::vec(N);
        brotensor::sigmoid_forward_cpu(x, yCpu);
        brotensor::sigmoid_backward_cpu(yCpu, dY, dXCpu);

        GpuTensor yg, dXg;
        brotensor::sigmoid_forward_gpu(xg, yg);
        brotensor::sigmoid_backward_gpu(yg, dYg, dXg);
        brotensor::cuda_sync();
        Tensor yGpu, dXGpu;
        brotensor::download(yg, yGpu);
        brotensor::download(dXg, dXGpu);
        compare("sigmoid_forward",  yCpu.data,  yGpu.data,  1e-5f, 1e-5f);
        compare("sigmoid_backward", dXCpu.data, dXGpu.data, 1e-4f, 1e-5f);
    }
}

// ---- softmax ----------------------------------------------------------------

static void parity_softmax() {
    std::printf("softmax parity\n");
    std::mt19937 rng(0xE5F6);
    const int N = 129;

    Tensor lg = Tensor::vec(N);
    fill_random(lg, rng, -2.0f, 2.0f);

    // Unmasked forward + backward.
    {
        Tensor pCpu = Tensor::vec(N);
        brotensor::softmax_forward_cpu(lg, pCpu);

        GpuTensor lgg, pg;
        brotensor::upload(lg, lgg);
        brotensor::softmax_forward_gpu(lgg, pg, nullptr);
        brotensor::cuda_sync();
        Tensor pGpu;
        brotensor::download(pg, pGpu);
        compare("softmax_forward(unmasked)", pCpu.data, pGpu.data, 1e-5f, 1e-5f);

        Tensor dP = Tensor::vec(N);
        fill_random(dP, rng, -1.0f, 1.0f);
        Tensor dZCpu = Tensor::vec(N);
        brotensor::softmax_backward_cpu(pCpu, dP, dZCpu);

        GpuTensor dPg, dZg;
        brotensor::upload(dP, dPg);
        brotensor::softmax_backward_gpu(pg, dPg, dZg);
        brotensor::cuda_sync();
        Tensor dZGpu;
        brotensor::download(dZg, dZGpu);
        compare("softmax_backward", dZCpu.data, dZGpu.data, 1e-4f, 1e-5f);
    }

    // Masked forward — mask out a random subset.
    {
        std::vector<float> mask(N, 1.0f);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        int valid = 0;
        for (int i = 0; i < N; ++i) {
            if (u(rng) < 0.3f) mask[i] = 0.0f;
            else ++valid;
        }
        if (valid == 0) mask[0] = 1.0f;  // guarantee at least one legal entry

        Tensor pCpu = Tensor::vec(N);
        brotensor::softmax_forward_cpu(lg, pCpu, mask.data());

        // Upload mask to device.
        GpuTensor lgg, pg, maskg;
        brotensor::upload(lg, lgg);
        brotensor::upload(mask.data(), N, 1, maskg);
        brotensor::softmax_forward_gpu(lgg, pg, maskg.data);
        brotensor::cuda_sync();
        Tensor pGpu;
        brotensor::download(pg, pGpu);
        compare("softmax_forward(masked)", pCpu.data, pGpu.data, 1e-5f, 1e-5f);
    }
}

// ---- add_inplace / add_scalar_inplace ---------------------------------------

static void parity_add() {
    std::printf("add_inplace / add_scalar_inplace parity\n");
    std::mt19937 rng(0x0102);
    const int N = 257;

    Tensor y = Tensor::vec(N), x = Tensor::vec(N);
    fill_random(y, rng, -1.0f, 1.0f);
    fill_random(x, rng, -1.0f, 1.0f);

    Tensor yCpu = y;
    brotensor::add_inplace_cpu(yCpu, x);

    GpuTensor yg, xg;
    brotensor::upload(y, yg);
    brotensor::upload(x, xg);
    brotensor::add_inplace_gpu(yg, xg);
    brotensor::cuda_sync();
    Tensor yGpu;
    brotensor::download(yg, yGpu);
    compare("add_inplace", yCpu.data, yGpu.data, 1e-6f, 1e-6f);

    // Scalar.
    Tensor z = Tensor::vec(N);
    fill_random(z, rng, -1.0f, 1.0f);
    Tensor zCpu = z;
    const float s = 0.375f;
    brotensor::add_scalar_inplace_cpu(zCpu, s);

    GpuTensor zg;
    brotensor::upload(z, zg);
    brotensor::add_scalar_inplace_gpu(zg, s);
    brotensor::cuda_sync();
    Tensor zGpu;
    brotensor::download(zg, zGpu);
    compare("add_scalar_inplace", zCpu.data, zGpu.data, 1e-6f, 1e-6f);
}

// NOTE: no parity check for softmax_xent_cpu / softmax_xent_segment_cpu —
//       the GPU softmax/xent surface is batched (loss.cu / softmax.cu work
//       in (batch, n) form with reductions) and does not expose a single-
//       sample combined op with the same signature. The unmasked/masked
//       softmax_forward and softmax_backward checks above already exercise
//       the underlying numerics.
// NOTE: no parity check for mse_scalar_cpu — there is no scalar-only
//       mse_scalar_gpu; the GPU loss surface (loss.cu) is batched and
//       differs in signature.
// NOTE: no parity check for xavier_init_cpu — the CPU RNG (splitmix64) is
//       deliberately host-only; there is no GPU equivalent that produces
//       bit-identical sequences.

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_cpu_gpu_parity\n");

    parity_linear();
    parity_elementwise_act();
    parity_softmax();
    parity_add();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll parity checks passed.\n");
    return 0;
}
