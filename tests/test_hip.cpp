// Test suite for AMD GPU (ROCm/HIP) backend integration in brotensor.
// Exercises Phase 1 runtime/allocator and Phase 2A core math/norms/reductions.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static bool contains(const std::vector<Device>& v, Device d) {
    return std::find(v.begin(), v.end(), d) != v.end();
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9f;
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static bool check_close(const Tensor& hip_tensor, const Tensor& cpu_tensor, float tol, const char* name) {
    std::vector<float> hip_v = hip_tensor.to_host_vector();
    std::vector<float> cpu_v = cpu_tensor.to_host_vector();
    if (hip_v.size() != cpu_v.size()) {
        std::printf("  FAIL  %s: size mismatch (hip=%zu, cpu=%zu)\n", name, hip_v.size(), cpu_v.size());
        ++g_failures;
        return false;
    }
    float diff = max_abs_diff(hip_v, cpu_v);
    if (diff > tol) {
        std::printf("  FAIL  %s: max diff %g > tol %g\n", name, diff, tol);
        ++g_failures;
        return false;
    }
    std::printf("  PASS  %s: max diff %g <= tol %g\n", name, diff, tol);
    return true;
}

static void test_hip_probe_and_registration() {
    std::printf("test_hip_probe_and_registration\n");
    brotensor::init();

    CHECK(brotensor::is_available(Device::HIP));

    int count = brotensor::hip_device_count();
    std::printf("  HIP device count: %d\n", count);
    CHECK(count >= 1);

    std::vector<Device> devs = brotensor::available_devices();
    CHECK(contains(devs, Device::HIP));
    CHECK(contains(devs, Device::hip(0)));

    Device def = brotensor::default_device();
    std::printf("  Default device: %s\n", brotensor::device_name(def));
    CHECK(def == Device::HIP);

    CHECK(std::strcmp(brotensor::device_name(Device::HIP), "hip") == 0);
    CHECK(brotensor::to_string(Device::HIP) == "hip");

    std::string prod = brotensor::device_product_name(Device::HIP);
    std::printf("  HIP device 0 product name: %s\n", prod.c_str());
    CHECK(!prod.empty());
}

static void test_hip_alloc_and_lifecycle() {
    std::printf("test_hip_alloc_and_lifecycle\n");
    Tensor a = Tensor::zeros_on(Device::HIP, 4, 8);
    CHECK(a.rows == 4);
    CHECK(a.cols == 8);
    CHECK(a.size() == 32);
    CHECK(a.data != nullptr);
    CHECK(a.device == Device::HIP);
    CHECK(a.device.is_hip());
    CHECK(a.device.is_gpu());
    CHECK(!a.device.is_cpu());

    a.zero();

    a.resize(2, 3);
    CHECK(a.rows == 2);
    CHECK(a.cols == 3);
    CHECK(a.size() == 6);
    CHECK(a.device == Device::HIP);

    // Move ctor
    Tensor b = std::move(a);
    CHECK(b.rows == 2 && b.cols == 3);
    CHECK(b.device == Device::HIP);
    CHECK(a.data == nullptr);

    // Move assignment
    Tensor c;
    c = std::move(b);
    CHECK(c.rows == 2 && c.cols == 3);
    CHECK(c.device == Device::HIP);
    CHECK(b.data == nullptr);
}

static void test_hip_host_device_transfer() {
    std::printf("test_hip_host_device_transfer\n");
    std::vector<float> host_in = {1.5f, -2.25f, 3.125f, 0.0f, -42.0f, 100.5f};
    Tensor d = Tensor::from_host_on(Device::HIP, host_in.data(), 2, 3);
    CHECK(d.rows == 2);
    CHECK(d.cols == 3);
    CHECK(d.device == Device::HIP);

    std::vector<float> host_out = d.to_host_vector();
    CHECK(host_out.size() == host_in.size());
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(host_out[i] - host_in[i]) < 1e-6f);
    }

    // Clone on HIP
    Tensor d_cloned = d.clone();
    CHECK(d_cloned.rows == 2);
    CHECK(d_cloned.cols == 3);
    CHECK(d_cloned.device == Device::HIP);
    CHECK(d_cloned.data != d.data);

    std::vector<float> clone_out = d_cloned.to_host_vector();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(clone_out[i] - host_in[i]) < 1e-6f);
    }

    // Inter-device transfer to CPU and back to HIP
    Tensor cpu_t = d.to(Device::CPU);
    CHECK(cpu_t.device == Device::CPU);
    CHECK(cpu_t.rows == 2);
    CHECK(cpu_t.cols == 3);
    const float* cp = cpu_t.host_f32();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(cp[i] - host_in[i]) < 1e-6f);
    }

    Tensor back_hip = cpu_t.to(Device::HIP);
    CHECK(back_hip.device == Device::HIP);
    std::vector<float> back_out = back_hip.to_host_vector();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(back_out[i] - host_in[i]) < 1e-6f);
    }
}

static void test_hip_memory_and_sync() {
    std::printf("test_hip_memory_and_sync\n");
    std::size_t free_b = 0, total_b = 0;
    bool ok = brotensor::device_mem_info(Device::HIP, free_b, total_b);
    CHECK(ok);
    std::printf("  HIP mem info: free=%.2f MB, total=%.2f MB\n",
                static_cast<double>(free_b) / (1024.0 * 1024.0),
                static_cast<double>(total_b) / (1024.0 * 1024.0));
    CHECK(total_b > 0);

    brotensor::sync(Device::HIP);
    brotensor::sync_all();

    (void)brotensor::device_mem_trim(Device::HIP, 0);
}

static void test_hip_device_scope() {
    std::printf("test_hip_device_scope\n");
    const Device outer = brotensor::default_device();

    {
        brotensor::DeviceScope s_cpu(Device::CPU);
        CHECK(brotensor::default_device() == Device::CPU);
        Tensor t_cpu = Tensor::zeros(2, 2);
        CHECK(t_cpu.device == Device::CPU);

        {
            brotensor::DeviceScope s_hip(Device::HIP);
            CHECK(brotensor::default_device() == Device::HIP);
            Tensor t_hip = Tensor::zeros(2, 2);
            CHECK(t_hip.device == Device::HIP);
        }
        CHECK(brotensor::default_device() == Device::CPU);
    }
    CHECK(brotensor::default_device() == outer);
}

static void test_hip_elementwise() {
    std::printf("test_hip_elementwise\n");
    const int rows = 8, cols = 16;
    const int n = rows * cols;

    // Test add_inplace
    {
        std::vector<float> vx(n), vy(n);
        for (int i = 0; i < n; ++i) {
            vx[i] = static_cast<float>(i % 7) - 3.0f;
            vy[i] = static_cast<float>(i % 5) * 1.5f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor y_cpu = Tensor::from_host_on(Device::CPU, vy.data(), rows, cols);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor y_hip = y_cpu.to(Device::HIP);

        brotensor::add_inplace(y_cpu, x_cpu);
        brotensor::add_inplace(y_hip, x_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-5f, "add_inplace");
    }

    // Test mul_inplace
    {
        std::vector<float> vx(n), vy(n);
        for (int i = 0; i < n; ++i) {
            vx[i] = static_cast<float>((i % 9) - 4) * 0.25f;
            vy[i] = static_cast<float>((i % 6) + 1) * 0.5f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor y_cpu = Tensor::from_host_on(Device::CPU, vy.data(), rows, cols);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor y_hip = y_cpu.to(Device::HIP);

        brotensor::mul_inplace(y_cpu, x_cpu);
        brotensor::mul_inplace(y_hip, x_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-5f, "mul_inplace");
    }

    // Test relu_forward
    {
        std::vector<float> vx(n);
        for (int i = 0; i < n; ++i) {
            vx[i] = static_cast<float>(i - n / 2) * 0.2f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor out_cpu, out_hip;

        brotensor::relu_forward(x_cpu, out_cpu);
        brotensor::relu_forward(x_hip, out_hip);
        brotensor::sync(Device::HIP);
        check_close(out_hip, out_cpu, 1e-5f, "relu_forward");
    }

    // Test silu_forward
    {
        std::vector<float> vx(n);
        for (int i = 0; i < n; ++i) {
            vx[i] = static_cast<float>(i - n / 2) * 0.1f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor out_cpu, out_hip;

        brotensor::silu_forward(x_cpu, out_cpu);
        brotensor::silu_forward(x_hip, out_hip);
        brotensor::sync(Device::HIP);
        check_close(out_hip, out_cpu, 1e-5f, "silu_forward");
    }

    // Test gelu_forward
    {
        std::vector<float> vx(n);
        for (int i = 0; i < n; ++i) {
            vx[i] = static_cast<float>(i - n / 2) * 0.1f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor out_cpu, out_hip;

        brotensor::gelu_forward(x_cpu, out_cpu);
        brotensor::gelu_forward(x_hip, out_hip);
        brotensor::sync(Device::HIP);
        check_close(out_hip, out_cpu, 1e-5f, "gelu_forward");
    }
}

static void test_hip_norms() {
    std::printf("test_hip_norms\n");
    const int B = 4, D = 16;
    const float eps = 1e-5f;

    // Test RMSNorm
    {
        std::vector<float> vx(B * D), vg(D);
        for (int i = 0; i < B * D; ++i) {
            vx[i] = std::sin(static_cast<float>(i + 1) * 0.3f);
        }
        for (int j = 0; j < D; ++j) {
            vg[j] = 0.5f + static_cast<float>(j % 5) * 0.2f;
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), B, D);
        Tensor gamma_cpu = Tensor::from_host_on(Device::CPU, vg.data(), D, 1);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor gamma_hip = gamma_cpu.to(Device::HIP);
        Tensor y_cpu, y_hip;

        brotensor::rms_norm_forward(x_cpu, gamma_cpu, eps, y_cpu);
        brotensor::rms_norm_forward(x_hip, gamma_hip, eps, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "rms_norm_forward");
    }

    // Test LayerNorm (single vector)
    {
        std::vector<float> vx(D), vg(D), vb(D);
        for (int i = 0; i < D; ++i) {
            vx[i] = std::cos(static_cast<float>(i + 1) * 0.4f);
            vg[i] = 1.0f + 0.1f * static_cast<float>(i % 3);
            vb[i] = 0.05f * static_cast<float>(i % 4);
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), D, 1);
        Tensor gamma_cpu = Tensor::from_host_on(Device::CPU, vg.data(), D, 1);
        Tensor beta_cpu = Tensor::from_host_on(Device::CPU, vb.data(), D, 1);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor gamma_hip = gamma_cpu.to(Device::HIP);
        Tensor beta_hip = beta_cpu.to(Device::HIP);

        Tensor y_cpu, xhat_cpu;
        float mean_cpu = 0.0f, rstd_cpu = 0.0f;
        brotensor::layernorm_forward(x_cpu, gamma_cpu, beta_cpu, y_cpu, xhat_cpu, mean_cpu, rstd_cpu, eps);

        Tensor y_hip, xhat_hip;
        float mean_hip = 0.0f, rstd_hip = 0.0f;
        brotensor::layernorm_forward(x_hip, gamma_hip, beta_hip, y_hip, xhat_hip, mean_hip, rstd_hip, eps);
        brotensor::sync(Device::HIP);

        check_close(y_hip, y_cpu, 1e-4f, "layernorm_forward (y)");
        check_close(xhat_hip, xhat_cpu, 1e-4f, "layernorm_forward (xhat)");
        CHECK(std::fabs(mean_hip - mean_cpu) < 1e-4f);
        CHECK(std::fabs(rstd_hip - rstd_cpu) < 1e-4f);
    }

    // Test LayerNorm (batched inference)
    {
        std::vector<float> vx(B * D), vg(D), vb(D);
        for (int i = 0; i < B * D; ++i) {
            vx[i] = std::sin(static_cast<float>(i + 2) * 0.25f);
        }
        for (int j = 0; j < D; ++j) {
            vg[j] = 0.8f + 0.1f * static_cast<float>(j % 4);
            vb[j] = 0.02f * static_cast<float>(j % 3);
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), B, D);
        Tensor gamma_cpu = Tensor::from_host_on(Device::CPU, vg.data(), D, 1);
        Tensor beta_cpu = Tensor::from_host_on(Device::CPU, vb.data(), D, 1);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor gamma_hip = gamma_cpu.to(Device::HIP);
        Tensor beta_hip = beta_cpu.to(Device::HIP);
        Tensor y_cpu, y_hip;

        brotensor::layernorm_forward_inference_batched(x_cpu, gamma_cpu, beta_cpu, y_cpu, eps);
        brotensor::layernorm_forward_inference_batched(x_hip, gamma_hip, beta_hip, y_hip, eps);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "layernorm_forward_inference_batched");
    }

    // Test l2_norm_forward
    {
        std::vector<float> vx(B * D);
        for (int i = 0; i < B * D; ++i) {
            vx[i] = std::sin(static_cast<float>(i + 1) * 0.5f);
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), B, D);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor y_cpu, y_hip;
        const int head_dim = 8;
        const int num_heads = D / head_dim;

        brotensor::l2_norm_forward(x_cpu, head_dim, num_heads, eps, y_cpu);
        brotensor::l2_norm_forward(x_hip, head_dim, num_heads, eps, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "l2_norm_forward");
    }

    // Test l2_normalize_nchw_forward
    {
        const int N = 2, C = 4, H = 2, W = 2;
        std::vector<float> vx(N * C * H * W);
        for (std::size_t i = 0; i < vx.size(); ++i) {
            vx[i] = std::cos(static_cast<float>(i + 1) * 0.3f);
        }
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), N, C * H * W);
        Tensor x_hip = x_cpu.to(Device::HIP);
        Tensor y_cpu, y_hip;

        brotensor::l2_normalize_nchw_forward(x_cpu, N, C, H, W, eps, y_cpu);
        brotensor::l2_normalize_nchw_forward(x_hip, N, C, H, W, eps, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "l2_normalize_nchw_forward");
    }
}

static void test_hip_reductions() {
    std::printf("test_hip_reductions\n");
    const int M = 7, N = 25;
    std::vector<float> vx(M * N);
    for (int i = 0; i < M * N; ++i) {
        vx[i] = std::sin(static_cast<float>(i + 1) * 0.2f);
    }
    Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), M, N);
    Tensor x_hip = x_cpu.to(Device::HIP);

    // sum_rows
    {
        Tensor y_cpu, y_hip;
        brotensor::sum_rows(x_cpu, y_cpu);
        brotensor::sum_rows(x_hip, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "sum_rows");
    }

    // sum_cols
    {
        Tensor y_cpu, y_hip;
        brotensor::sum_cols(x_cpu, y_cpu);
        brotensor::sum_cols(x_hip, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "sum_cols");
    }

    // masked_mean_pool_forward (all rows valid: mask = nullptr)
    {
        Tensor y_cpu, y_hip;
        brotensor::masked_mean_pool_forward(x_cpu, nullptr, y_cpu);
        brotensor::masked_mean_pool_forward(x_hip, nullptr, y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "masked_mean_pool_forward (mean)");
    }
}

int main() {
    std::printf("test_hip running...\n");
    test_hip_probe_and_registration();
    test_hip_alloc_and_lifecycle();
    test_hip_host_device_transfer();
    test_hip_memory_and_sync();
    test_hip_device_scope();

    // Phase 2A functional tests
    test_hip_elementwise();
    test_hip_norms();
    test_hip_reductions();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll HIP Phase 1 and Phase 2A checks passed successfully.\n");
    return 0;
}
