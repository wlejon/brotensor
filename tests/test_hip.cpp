// Test suite for AMD GPU (ROCm/HIP) backend integration in brotensor.
// Exercises Phase 1 runtime/allocator and Phase 2A core math/norms/reductions.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

static bool check_close_fp16(const Tensor& hip_tensor, const Tensor& cpu_tensor, float tol, const char* name) {
    std::vector<uint16_t> hip_bits = hip_tensor.to_host_vector_fp16();
    std::vector<float> cpu_v = cpu_tensor.to_host_vector();
    if (hip_bits.size() != cpu_v.size()) {
        std::printf("  FAIL  %s: size mismatch (hip=%zu, cpu=%zu)\n", name, hip_bits.size(), cpu_v.size());
        ++g_failures;
        return false;
    }
    std::vector<float> hip_v(hip_bits.size());
    for (std::size_t i = 0; i < hip_bits.size(); ++i) {
        hip_v[i] = brotensor::fp16_bits_to_fp32(hip_bits[i]);
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

static Tensor make_fp16_hip(int rows, int cols, const std::vector<float>& src) {
    std::vector<uint16_t> h(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        h[i] = brotensor::fp32_to_fp16_bits(src[i]);
    }
    return Tensor::from_host_fp16_on(Device::HIP, h.data(), rows, cols);
}

static void round_vector_to_fp16(std::vector<float>& v) {
    for (float& x : v) {
        x = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(x));
    }
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

// ─── Phase 2B Functional Tests ──────────────────────────────────────────────

static void test_hip_matmul() {
    std::printf("test_hip_matmul\n");

    // FP32 Matmul
    {
        const int shapes[][3] = {
            {64, 64, 64},
            {16, 32, 48},
            {1, 64, 32},
        };
        for (const auto& s : shapes) {
            int M = s[0], K = s[1], N = s[2];
            std::vector<float> va(M * K);
            std::vector<float> vb(K * N);
            for (std::size_t i = 0; i < va.size(); ++i) va[i] = std::sin(static_cast<float>(i + 1) * 0.1f);
            for (std::size_t i = 0; i < vb.size(); ++i) vb[i] = std::cos(static_cast<float>(i + 1) * 0.15f);

            Tensor A_cpu = Tensor::from_host_on(Device::CPU, va.data(), M, K);
            Tensor B_cpu = Tensor::from_host_on(Device::CPU, vb.data(), K, N);
            Tensor C_cpu;
            brotensor::matmul(A_cpu, B_cpu, C_cpu);

            Tensor A_hip = A_cpu.to(Device::HIP);
            Tensor B_hip = B_cpu.to(Device::HIP);
            Tensor C_hip;
            brotensor::matmul(A_hip, B_hip, C_hip);
            brotensor::sync(Device::HIP);

            char tag[64];
            std::snprintf(tag, sizeof(tag), "matmul_fp32_%dx%dx%d", M, K, N);
            check_close(C_hip, C_cpu, 1e-4f, tag);
        }
    }

    // FP32 Linear Forward
    {
        const int OUT = 32, IN = 64;
        std::vector<float> vw(OUT * IN);
        std::vector<float> vb(OUT);
        std::vector<float> vx(IN);
        for (std::size_t i = 0; i < vw.size(); ++i) vw[i] = std::sin(static_cast<float>(i + 1) * 0.05f);
        for (std::size_t i = 0; i < vb.size(); ++i) vb[i] = 0.1f * static_cast<float>(i + 1);
        for (std::size_t i = 0; i < vx.size(); ++i) vx[i] = std::cos(static_cast<float>(i + 1) * 0.2f);

        Tensor W_cpu = Tensor::from_host_on(Device::CPU, vw.data(), OUT, IN);
        Tensor b_cpu = Tensor::from_host_on(Device::CPU, vb.data(), OUT, 1);
        Tensor x_cpu = Tensor::from_host_on(Device::CPU, vx.data(), IN, 1);
        Tensor y_cpu;
        brotensor::linear_forward(W_cpu, b_cpu, x_cpu, y_cpu);

        Tensor y_hip;
        brotensor::linear_forward(W_cpu.to(Device::HIP), b_cpu.to(Device::HIP), x_cpu.to(Device::HIP), y_hip);
        brotensor::sync(Device::HIP);
        check_close(y_hip, y_cpu, 1e-4f, "linear_forward_fp32");
    }

    // FP16 Matmul
    {
        const int shapes[][3] = {
            {64, 64, 64},
            {16, 32, 48},
            {1, 64, 32},
        };
        for (const auto& s : shapes) {
            int M = s[0], K = s[1], N = s[2];
            std::vector<float> va(M * K);
            std::vector<float> vb(K * N);
            for (std::size_t i = 0; i < va.size(); ++i) va[i] = std::sin(static_cast<float>(i + 1) * 0.1f);
            for (std::size_t i = 0; i < vb.size(); ++i) vb[i] = std::cos(static_cast<float>(i + 1) * 0.15f);
            round_vector_to_fp16(va);
            round_vector_to_fp16(vb);

            Tensor A_cpu = Tensor::from_host_on(Device::CPU, va.data(), M, K);
            Tensor B_cpu = Tensor::from_host_on(Device::CPU, vb.data(), K, N);
            Tensor C_cpu;
            brotensor::matmul(A_cpu, B_cpu, C_cpu);

            Tensor A_hip = make_fp16_hip(M, K, va);
            Tensor B_hip = make_fp16_hip(K, N, vb);
            Tensor C_hip;
            brotensor::matmul(A_hip, B_hip, C_hip);
            brotensor::sync(Device::HIP);

            char tag[64];
            std::snprintf(tag, sizeof(tag), "matmul_fp16_%dx%dx%d", M, K, N);
            check_close_fp16(C_hip, C_cpu, 2e-3f, tag);
        }
    }
}

static void test_hip_softmax() {
    std::printf("test_hip_softmax\n");

    // softmax_forward (unmasked and masked)
    {
        const int N = 128;
        std::vector<float> vlogits(N);
        for (int i = 0; i < N; ++i) vlogits[i] = std::sin(static_cast<float>(i + 1) * 0.2f) * 2.0f;

        Tensor logits_cpu = Tensor::from_host_on(Device::CPU, vlogits.data(), N, 1);
        Tensor logits_hip = logits_cpu.to(Device::HIP);

        // unmasked
        {
            Tensor probs_cpu, probs_hip;
            brotensor::softmax_forward(logits_cpu, probs_cpu, nullptr);
            brotensor::softmax_forward(logits_hip, probs_hip, nullptr);
            brotensor::sync(Device::HIP);
            check_close(probs_hip, probs_cpu, 1e-4f, "softmax_forward_unmasked");
        }

        // masked
        {
            std::vector<float> mask(N, 1.0f);
            for (int i = N / 2; i < N; ++i) mask[i] = 0.0f;
            Tensor mask_hip = Tensor::from_host_on(Device::HIP, mask.data(), N, 1);

            Tensor probs_cpu, probs_hip;
            brotensor::softmax_forward(logits_cpu, probs_cpu, mask.data());
            brotensor::softmax_forward(logits_hip, probs_hip, static_cast<const float*>(mask_hip.data));
            brotensor::sync(Device::HIP);
            check_close(probs_hip, probs_cpu, 1e-4f, "softmax_forward_masked");
        }
    }

    // softmax_rows_forward (FP32 & FP16)
    {
        const int rows = 8, cols = 64;
        std::vector<float> vx(rows * cols);
        for (std::size_t i = 0; i < vx.size(); ++i) {
            vx[i] = std::cos(static_cast<float>(i + 1) * 0.1f) * 2.0f;
        }

        Tensor X_cpu = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
        Tensor Y_cpu;
        brotensor::softmax_rows_forward(X_cpu, Y_cpu, rows, cols);

        // FP32
        {
            Tensor X_hip = X_cpu.to(Device::HIP);
            Tensor Y_hip;
            brotensor::softmax_rows_forward(X_hip, Y_hip, rows, cols);
            brotensor::sync(Device::HIP);
            check_close(Y_hip, Y_cpu, 1e-4f, "softmax_rows_forward_fp32");
        }

        // FP16
        {
            round_vector_to_fp16(vx);
            Tensor X_cpu_fp16 = Tensor::from_host_on(Device::CPU, vx.data(), rows, cols);
            brotensor::softmax_rows_forward(X_cpu_fp16, Y_cpu, rows, cols);

            Tensor X_hip_fp16 = make_fp16_hip(rows, cols, vx);
            Tensor Y_hip_fp16;
            brotensor::softmax_rows_forward(X_hip_fp16, Y_hip_fp16, rows, cols);
            brotensor::sync(Device::HIP);
            check_close_fp16(Y_hip_fp16, Y_cpu, 1e-2f, "softmax_rows_forward_fp16");
        }
    }
}

static void test_hip_rope() {
    std::printf("test_hip_rope\n");

    const int L = 16, num_heads = 4, head_dim = 64;
    const float theta_base = 10000.0f;
    std::vector<float> vx(L * num_heads * head_dim);
    for (std::size_t i = 0; i < vx.size(); ++i) {
        vx[i] = std::sin(static_cast<float>(i + 1) * 0.1f);
    }
    Tensor X_cpu = Tensor::from_host_on(Device::CPU, vx.data(), L, num_heads * head_dim);
    Tensor X_hip = X_cpu.to(Device::HIP);

    // seq_offset = 0
    {
        Tensor Y_cpu, Y_hip;
        brotensor::rope_forward(X_cpu, head_dim, num_heads, 0, theta_base, Y_cpu);
        brotensor::rope_forward(X_hip, head_dim, num_heads, 0, theta_base, Y_hip);
        brotensor::sync(Device::HIP);
        check_close(Y_hip, Y_cpu, 1e-4f, "rope_forward_offset0");

        Tensor dX_cpu, dX_hip;
        brotensor::rope_backward(Y_cpu, head_dim, num_heads, 0, theta_base, dX_cpu);
        brotensor::rope_backward(Y_hip, head_dim, num_heads, 0, theta_base, dX_hip);
        brotensor::sync(Device::HIP);
        check_close(dX_hip, dX_cpu, 1e-4f, "rope_backward_offset0");
    }

    // seq_offset = 7
    {
        Tensor Y_cpu, Y_hip;
        brotensor::rope_forward(X_cpu, head_dim, num_heads, 7, theta_base, Y_cpu);
        brotensor::rope_forward(X_hip, head_dim, num_heads, 7, theta_base, Y_hip);
        brotensor::sync(Device::HIP);
        check_close(Y_hip, Y_cpu, 1e-4f, "rope_forward_offset7");
    }
}

static void test_hip_attention() {
    std::printf("test_hip_attention\n");

    // Standard attention_forward (FP32)
    {
        const int N = 8, D = 32;
        std::vector<float> vx(N * D), vw(D * D);
        for (std::size_t i = 0; i < vx.size(); ++i) vx[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * 0.2f;
        for (std::size_t i = 0; i < vw.size(); ++i) vw[i] = std::cos(static_cast<float>(i + 1) * 0.15f) * 0.2f;

        Tensor X_c  = Tensor::from_host_on(Device::CPU, vx.data(), N, D);
        Tensor Wq_c = Tensor::from_host_on(Device::CPU, vw.data(), D, D);
        Tensor Wk_c = Tensor::from_host_on(Device::CPU, vw.data(), D, D);
        Tensor Wv_c = Tensor::from_host_on(Device::CPU, vw.data(), D, D);
        Tensor Wo_c = Tensor::from_host_on(Device::CPU, vw.data(), D, D);
        Tensor Q_c, K_c, V_c, Attn_c, Y_pre_c, O_c;

        brotensor::attention_forward(X_c, Wq_c, Wk_c, Wv_c, Wo_c, nullptr,
                                     Q_c, K_c, V_c, Attn_c, Y_pre_c, O_c);

        Tensor X_h  = X_c.to(Device::HIP);
        Tensor Wq_h = Wq_c.to(Device::HIP);
        Tensor Wk_h = Wk_c.to(Device::HIP);
        Tensor Wv_h = Wv_c.to(Device::HIP);
        Tensor Wo_h = Wo_c.to(Device::HIP);
        Tensor Q_h, K_h, V_h, Attn_h, Y_pre_h, O_h;

        brotensor::attention_forward(X_h, Wq_h, Wk_h, Wv_h, Wo_h, nullptr,
                                     Q_h, K_h, V_h, Attn_h, Y_pre_h, O_h);
        brotensor::sync(Device::HIP);
        check_close(O_h, O_c, 1e-3f, "attention_forward_fp32");
    }

    // flash_attention_forward (FP16 on HIP vs FP32 on CPU)
    {
        const int Lq = 32, Lk = 32, D = 64, num_heads = 2;
        std::vector<float> vq(Lq * D), vk(Lk * D), vv(Lk * D);
        for (std::size_t i = 0; i < vq.size(); ++i) vq[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * 0.3f;
        for (std::size_t i = 0; i < vk.size(); ++i) vk[i] = std::cos(static_cast<float>(i + 1) * 0.2f) * 0.3f;
        for (std::size_t i = 0; i < vv.size(); ++i) vv[i] = std::sin(static_cast<float>(i + 1) * 0.3f) * 0.3f;
        round_vector_to_fp16(vq);
        round_vector_to_fp16(vk);
        round_vector_to_fp16(vv);

        Tensor Q_c = Tensor::from_host_on(Device::CPU, vq.data(), Lq, D);
        Tensor K_c = Tensor::from_host_on(Device::CPU, vk.data(), Lk, D);
        Tensor V_c = Tensor::from_host_on(Device::CPU, vv.data(), Lk, D);

        Tensor Q_h = make_fp16_hip(Lq, D, vq);
        Tensor K_h = make_fp16_hip(Lk, D, vk);
        Tensor V_h = make_fp16_hip(Lk, D, vv);

        // Bidirectional (causal = false)
        {
            Tensor O_c, O_h;
            brotensor::flash_attention_forward(Q_c, K_c, V_c, nullptr, num_heads, false, O_c);
            brotensor::flash_attention_forward(Q_h, K_h, V_h, nullptr, num_heads, false, O_h);
            brotensor::sync(Device::HIP);
            check_close_fp16(O_h, O_c, 2e-2f, "flash_attention_forward_bidirectional");
        }

        // Causal (causal = true)
        {
            Tensor O_c, O_h;
            brotensor::flash_attention_forward(Q_c, K_c, V_c, nullptr, num_heads, true, O_c);
            brotensor::flash_attention_forward(Q_h, K_h, V_h, nullptr, num_heads, true, O_h);
            brotensor::sync(Device::HIP);
            check_close_fp16(O_h, O_c, 2e-2f, "flash_attention_forward_causal");
        }
    }
}

// ─── Q4_K Helpers and Tests ──────────────────────────────────────────────────

static constexpr int Q4K_BLOCK = 256;
static constexpr int Q4K_BYTES = 144;

struct Q4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

static void pack_sc_m(uint8_t scales[12], const uint8_t sc[8], const uint8_t m[8]) {
    std::memset(scales, 0, 12);
    for (int j = 0; j < 4; ++j) {
        scales[j]     = sc[j] & 0x3F;
        scales[j + 4] = m[j]  & 0x3F;
    }
    for (int j = 4; j < 8; ++j) {
        scales[j + 4] = static_cast<uint8_t>((sc[j] & 0x0F) | ((m[j] & 0x0F) << 4));
        scales[j - 4] |= static_cast<uint8_t>(((sc[j] >> 4) & 0x03) << 6);
        scales[j]     |= static_cast<uint8_t>(((m[j] >> 4) & 0x03) << 6);
    }
}

static void unpack_sc_m(const uint8_t scales[12], uint8_t* sc, uint8_t* m) {
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = scales[j]     & 0x3F;
            m [j] = scales[j + 4] & 0x3F;
        } else {
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            m [j] = (scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4);
        }
    }
}

static void quantize_q4k_block(const float* src, Q4KBlock& out) {
    float lo[8], hi[8];
    for (int is = 0; is < 8; ++is) {
        lo[is] = hi[is] = src[is * 32];
        for (int l = 1; l < 32; ++l) {
            const float v = src[is * 32 + l];
            if (v < lo[is]) lo[is] = v;
            if (v > hi[is]) hi[is] = v;
        }
    }
    float max_range = 0.0f, max_neg_lo = 0.0f;
    for (int is = 0; is < 8; ++is) {
        max_range  = std::max(max_range, hi[is] - lo[is]);
        max_neg_lo = std::max(max_neg_lo, -lo[is]);
    }
    const float d    = (max_range > 0.0f) ? (max_range / (15.0f * 63.0f)) : 1.0f;
    const float dmin = (max_neg_lo > 0.0f) ? (max_neg_lo / 63.0f) : 1.0f;

    uint8_t sc[8], m[8];
    for (int is = 0; is < 8; ++is) {
        int sc_i = (d > 0.0f) ? static_cast<int>(std::lround((hi[is] - lo[is]) / (15.0f * d))) : 0;
        int m_i  = (dmin > 0.0f) ? static_cast<int>(std::lround(-lo[is] / dmin)) : 0;
        sc[is] = static_cast<uint8_t>(std::clamp(sc_i, 0, 63));
        m[is]  = static_cast<uint8_t>(std::clamp(m_i, 0, 63));
    }

    std::memset(out.qs, 0, 128);
    for (int p = 0; p < 4; ++p) {
        const int is_lo = 2 * p, is_hi = 2 * p + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;
        for (int l = 0; l < 32; ++l) {
            int n_lo = (w_lo > 0.0f) ? static_cast<int>(std::lround((src[is_lo * 32 + l] + b_lo) / w_lo)) : 0;
            int n_hi = (w_hi > 0.0f) ? static_cast<int>(std::lround((src[is_hi * 32 + l] + b_hi) / w_hi)) : 0;
            n_lo = std::clamp(n_lo, 0, 15);
            n_hi = std::clamp(n_hi, 0, 15);
            out.qs[p * 32 + l] = static_cast<uint8_t>((n_lo & 0x0F) | ((n_hi & 0x0F) << 4));
        }
    }
    out.d    = brotensor::fp32_to_fp16_bits(d);
    out.dmin = brotensor::fp32_to_fp16_bits(dmin);
    pack_sc_m(out.scales, sc, m);
}

static void dequant_q4k_block(const Q4KBlock& blk, float* dst) {
    const float d    = brotensor::fp16_bits_to_fp32(blk.d);
    const float dmin = brotensor::fp16_bits_to_fp32(blk.dmin);
    uint8_t sc[8], m[8];
    unpack_sc_m(blk.scales, sc, m);
    for (int t = 0; t < 256; ++t) {
        const int is   = t >> 5;
        const int l    = t & 31;
        const int pair = is >> 1;
        const uint8_t qb = blk.qs[pair * 32 + l];
        const int nib  = (is & 1) ? (qb >> 4) : (qb & 0x0F);
        dst[t] = d * static_cast<float>(sc[is]) * static_cast<float>(nib)
               - dmin * static_cast<float>(m[is]);
    }
}

static void test_hip_q4k() {
    std::printf("test_hip_q4k\n");
    constexpr int OUT = 4;
    constexpr int IN  = 256;
    constexpr int BLOCKS_PER_ROW = IN / Q4K_BLOCK;

    std::vector<float> Wf(static_cast<size_t>(OUT) * IN);
    for (size_t i = 0; i < Wf.size(); ++i) {
        Wf[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * 0.5f;
    }

    std::vector<Q4KBlock> Wq(static_cast<size_t>(OUT) * BLOCKS_PER_ROW);
    std::vector<float> W_deq(static_cast<size_t>(OUT) * IN);
    for (int r = 0; r < OUT; ++r) {
        for (int sb = 0; sb < BLOCKS_PER_ROW; ++sb) {
            quantize_q4k_block(&Wf[r * IN + sb * Q4K_BLOCK], Wq[r * BLOCKS_PER_ROW + sb]);
            dequant_q4k_block(Wq[r * BLOCKS_PER_ROW + sb], &W_deq[r * IN + sb * Q4K_BLOCK]);
        }
    }

    Tensor W_q4k_g = Tensor::from_raw_bytes_on(Device::HIP, Wq.data(), OUT, IN,
                                               Dtype::Q4_K, Wq.size() * sizeof(Q4KBlock));

    // Test dequant_q4k_to_fp16
    {
        Tensor W_fp16_g;
        brotensor::dequant_q4k_to_fp16(W_q4k_g, W_fp16_g);
        CHECK(W_fp16_g.dtype == Dtype::FP16);
        CHECK(W_fp16_g.rows == OUT && W_fp16_g.cols == IN);
        brotensor::sync(Device::HIP);
        std::vector<uint16_t> got = W_fp16_g.to_host_vector_fp16();
        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            float g = brotensor::fp16_bits_to_fp32(got[i]);
            float r = W_deq[i];
            float e = std::fabs(g - r);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  PASS  dequant_q4k_to_fp16: max_abs=%g <= 1e-3\n", max_abs);
        CHECK(max_abs <= 1e-3f);
    }

    // Test linear_forward_q4k_fp16 (GEMV)
    {
        std::vector<float> xf(IN);
        for (int i = 0; i < IN; ++i) {
            xf[i] = std::cos(static_cast<float>(i + 1) * 0.15f) * 0.3f;
        }
        std::vector<float> y_ref(OUT, 0.0f);
        for (int r = 0; r < OUT; ++r) {
            float s = 0.0f;
            for (int k = 0; k < IN; ++k) {
                s += W_deq[r * IN + k] * xf[k];
            }
            y_ref[r] = s;
        }

        Tensor x_g = make_fp16_hip(IN, 1, xf);
        Tensor y_g;
        brotensor::linear_forward_q4k_fp16(W_q4k_g, nullptr, x_g, y_g);
        CHECK(y_g.dtype == Dtype::FP16 && y_g.rows == OUT && y_g.cols == 1);
        brotensor::sync(Device::HIP);
        std::vector<uint16_t> got = y_g.to_host_vector_fp16();
        float max_abs = 0.0f;
        for (int r = 0; r < OUT; ++r) {
            float g = brotensor::fp16_bits_to_fp32(got[r]);
            float e = std::fabs(g - y_ref[r]);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  PASS  linear_forward_q4k_fp16: max_abs=%g < 5e-2\n", max_abs);
        CHECK(max_abs < 5e-2f);
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

    // Phase 2B functional tests
    test_hip_matmul();
    test_hip_softmax();
    test_hip_rope();
    test_hip_attention();
    test_hip_q4k();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll HIP Phase 1, Phase 2A, and Phase 2B checks passed successfully.\n");
    return 0;
}
