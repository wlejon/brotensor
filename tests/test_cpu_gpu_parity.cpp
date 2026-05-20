// CPU↔GPU parity. For each op with a CPU and a CUDA implementation, run the
// same input through both backends and compare element-wise within tight
// FP32 tolerance.
//
// With the unified Tensor API both runs call the *same* device-neutral op
// name; dispatch is decided by the operands' Device tag. We build one set of
// inputs as host floats, materialise a CPU-resident and a CUDA-resident copy
// of each, run the op on each set (never mixing devices within a call), and
// download the CUDA result for comparison.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

// GPU backend this binary was built against (CUDA preferred, else Metal).
// Assigned in main() after brotensor::init(); Device::CPU == no GPU backend.
static Device g_gpu = Device::CPU;

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

// ---- helpers ----------------------------------------------------------------
//
// fill_random writes through host_f32_mut, so its argument must be a
// CPU-resident tensor.

static void fill_random(Tensor& t, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.size(); ++i) p[i] = d(rng);
}

// ---- linear -----------------------------------------------------------------

static void parity_linear() {
    std::printf("linear parity\n");
    std::mt19937 rng(0xA1B2);

    const int OUT = 5, IN = 7;

    // Build inputs once on the host.
    Tensor W = Tensor::zeros_on(Device::CPU, OUT, IN);
    Tensor b = Tensor::zeros_on(Device::CPU, OUT, 1);
    Tensor x = Tensor::zeros_on(Device::CPU, IN, 1);
    fill_random(W, rng, -0.5f, 0.5f);
    fill_random(b, rng, -0.3f, 0.3f);
    fill_random(x, rng, -1.0f, 1.0f);

    // CPU forward.
    Tensor yCpu = Tensor::zeros_on(Device::CPU, OUT, 1);
    brotensor::linear_forward(W, b, x, yCpu);

    // GPU forward — CUDA-resident copies of the same data.
    Tensor Wg = W.to(g_gpu);
    Tensor bg = b.to(g_gpu);
    Tensor xg = x.to(g_gpu);
    Tensor yg = Tensor::empty_on(g_gpu, OUT, 1);
    brotensor::linear_forward(Wg, bg, xg, yg);
    brotensor::sync(g_gpu);

    compare("linear_forward", yCpu.to_host_vector(), yg.to_host_vector(),
            1e-5f, 1e-5f);

    // Backward.
    Tensor dY = Tensor::zeros_on(Device::CPU, OUT, 1);
    fill_random(dY, rng, -1.0f, 1.0f);

    Tensor dXCpu = Tensor::zeros_on(Device::CPU, IN, 1);
    Tensor dWCpu = Tensor::zeros_on(Device::CPU, OUT, IN);
    Tensor dBCpu = Tensor::zeros_on(Device::CPU, OUT, 1);
    brotensor::linear_backward(W, x, dY, dXCpu, dWCpu, dBCpu);

    Tensor dYg = dY.to(g_gpu);
    Tensor dXg = Tensor::empty_on(g_gpu, IN, 1);
    Tensor dWg = Tensor::zeros_on(g_gpu, OUT, IN);
    Tensor dBg = Tensor::zeros_on(g_gpu, OUT, 1);
    brotensor::linear_backward(Wg, xg, dYg, dXg, dWg, dBg);
    brotensor::sync(g_gpu);

    compare("linear_backward/dX", dXCpu.to_host_vector(), dXg.to_host_vector(),
            1e-4f, 1e-5f);
    compare("linear_backward/dW", dWCpu.to_host_vector(), dWg.to_host_vector(),
            1e-4f, 1e-5f);
    compare("linear_backward/dB", dBCpu.to_host_vector(), dBg.to_host_vector(),
            1e-4f, 1e-5f);
}

// ---- relu / tanh / sigmoid --------------------------------------------------

static void parity_elementwise_act() {
    std::printf("relu/tanh/sigmoid parity\n");
    std::mt19937 rng(0xC3D4);
    const int N = 257;

    Tensor x  = Tensor::zeros_on(Device::CPU, N, 1);
    Tensor dY = Tensor::zeros_on(Device::CPU, N, 1);
    fill_random(x,  rng, -3.0f, 3.0f);
    fill_random(dY, rng, -1.0f, 1.0f);

    Tensor xg  = x.to(g_gpu);
    Tensor dYg = dY.to(g_gpu);

    // ReLU
    {
        Tensor yCpu = Tensor::zeros_on(Device::CPU, N, 1);
        Tensor dXCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::relu_forward(x, yCpu);
        brotensor::relu_backward(x, dY, dXCpu);

        Tensor yg = Tensor::empty_on(g_gpu, N, 1);
        Tensor dXg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::relu_forward(xg, yg);
        brotensor::relu_backward(xg, dYg, dXg);
        brotensor::sync(g_gpu);
        compare("relu_forward",  yCpu.to_host_vector(),  yg.to_host_vector(),  1e-6f, 1e-6f);
        compare("relu_backward", dXCpu.to_host_vector(), dXg.to_host_vector(), 1e-6f, 1e-6f);
    }

    // tanh — backward consumes y (cached forward), not x.
    {
        Tensor yCpu = Tensor::zeros_on(Device::CPU, N, 1);
        Tensor dXCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::tanh_forward(x, yCpu);
        brotensor::tanh_backward(yCpu, dY, dXCpu);

        Tensor yg = Tensor::empty_on(g_gpu, N, 1);
        Tensor dXg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::tanh_forward(xg, yg);
        brotensor::tanh_backward(yg, dYg, dXg);
        brotensor::sync(g_gpu);
        compare("tanh_forward",  yCpu.to_host_vector(),  yg.to_host_vector(),  1e-5f, 1e-5f);
        compare("tanh_backward", dXCpu.to_host_vector(), dXg.to_host_vector(), 1e-4f, 1e-5f);
    }

    // sigmoid — backward consumes y.
    {
        Tensor yCpu = Tensor::zeros_on(Device::CPU, N, 1);
        Tensor dXCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::sigmoid_forward(x, yCpu);
        brotensor::sigmoid_backward(yCpu, dY, dXCpu);

        Tensor yg = Tensor::empty_on(g_gpu, N, 1);
        Tensor dXg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::sigmoid_forward(xg, yg);
        brotensor::sigmoid_backward(yg, dYg, dXg);
        brotensor::sync(g_gpu);
        compare("sigmoid_forward",  yCpu.to_host_vector(),  yg.to_host_vector(),  1e-5f, 1e-5f);
        compare("sigmoid_backward", dXCpu.to_host_vector(), dXg.to_host_vector(), 1e-4f, 1e-5f);
    }
}

// ---- softmax ----------------------------------------------------------------

static void parity_softmax() {
    std::printf("softmax parity\n");
    std::mt19937 rng(0xE5F6);
    const int N = 129;

    Tensor lg = Tensor::zeros_on(Device::CPU, N, 1);
    fill_random(lg, rng, -2.0f, 2.0f);
    Tensor lgg = lg.to(g_gpu);

    // Unmasked forward + backward.
    {
        Tensor pCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::softmax_forward(lg, pCpu);

        Tensor pg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::softmax_forward(lgg, pg, nullptr);
        brotensor::sync(g_gpu);
        compare("softmax_forward(unmasked)", pCpu.to_host_vector(), pg.to_host_vector(),
                1e-5f, 1e-5f);

        Tensor dP = Tensor::zeros_on(Device::CPU, N, 1);
        fill_random(dP, rng, -1.0f, 1.0f);
        Tensor dZCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::softmax_backward(pCpu, dP, dZCpu);

        Tensor dPg = dP.to(g_gpu);
        Tensor dZg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::softmax_backward(pg, dPg, dZg);
        brotensor::sync(g_gpu);
        compare("softmax_backward", dZCpu.to_host_vector(), dZg.to_host_vector(),
                1e-4f, 1e-5f);
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

        // CPU op: host mask pointer is on the same device as the operands.
        Tensor pCpu = Tensor::zeros_on(Device::CPU, N, 1);
        brotensor::softmax_forward(lg, pCpu, mask.data());

        // GPU op: the mask must be a CUDA-resident buffer too.
        Tensor maskg = Tensor::from_host_on(g_gpu, mask.data(), N, 1);
        Tensor pg = Tensor::empty_on(g_gpu, N, 1);
        brotensor::softmax_forward(lgg, pg,
                                   static_cast<const float*>(maskg.data));
        brotensor::sync(g_gpu);
        compare("softmax_forward(masked)", pCpu.to_host_vector(), pg.to_host_vector(),
                1e-5f, 1e-5f);
    }
}

// ---- add_inplace / add_scalar_inplace ---------------------------------------

static void parity_add() {
    std::printf("add_inplace / add_scalar_inplace parity\n");
    std::mt19937 rng(0x0102);
    const int N = 257;

    Tensor y = Tensor::zeros_on(Device::CPU, N, 1);
    Tensor x = Tensor::zeros_on(Device::CPU, N, 1);
    fill_random(y, rng, -1.0f, 1.0f);
    fill_random(x, rng, -1.0f, 1.0f);

    Tensor yCpu = y.clone();
    brotensor::add_inplace(yCpu, x);

    Tensor yg = y.to(g_gpu);
    Tensor xg = x.to(g_gpu);
    brotensor::add_inplace(yg, xg);
    brotensor::sync(g_gpu);
    compare("add_inplace", yCpu.to_host_vector(), yg.to_host_vector(), 1e-6f, 1e-6f);

    // Scalar.
    Tensor z = Tensor::zeros_on(Device::CPU, N, 1);
    fill_random(z, rng, -1.0f, 1.0f);
    Tensor zCpu = z.clone();
    const float s = 0.375f;
    brotensor::add_scalar_inplace(zCpu, s);

    Tensor zg = z.to(g_gpu);
    brotensor::add_scalar_inplace(zg, s);
    brotensor::sync(g_gpu);
    compare("add_scalar_inplace", zCpu.to_host_vector(), zg.to_host_vector(), 1e-6f, 1e-6f);
}

// NOTE: no parity check for softmax_xent / softmax_xent_segment — the GPU
//       softmax/xent surface is batched (loss / softmax kernels work in
//       (batch, n) form with reductions) and does not expose a single-sample
//       combined op with the same signature. The unmasked/masked
//       softmax_forward and softmax_backward checks above already exercise
//       the underlying numerics.
// NOTE: no parity check for mse_scalar — it is CPU-only (the GPU loss surface
//       is batched and differs in signature); see the ops.h note on
//       mse_scalar.
// NOTE: no parity check for xavier_init — it is CPU-only (the host RNG,
//       splitmix64, has no GPU equivalent that produces bit-identical
//       sequences).

int main() {
    brotensor::init();
    if (brotensor::is_available(Device::CUDA))       g_gpu = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) g_gpu = Device::Metal;
    if (g_gpu == Device::CPU) {
        std::printf("no GPU backend available - skipping\n");
        return 0;
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
