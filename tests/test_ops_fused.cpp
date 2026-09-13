// ─── Tests for brotensor::fused_* public API (CPU & CUDA) ──────────────────

#include "parity_helpers.h"
#include <brotensor/ops.h>
#include <brotensor/ops/fused.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using bt_parity::compare_tensors;
using bt_parity::fill_random;
using bt_parity::SplitMix64;

namespace {

void reference_rmsnorm(const float* X, const float* gamma, float eps, float* Y, int B, int D) {
    const float inv_D = 1.0f / static_cast<float>(D);
    for (int b = 0; b < B; ++b) {
        const float* x = X + b * D;
        float* y = Y + b * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += x[i] * x[i];
        float rrms = 1.0f / std::sqrt(sum * inv_D + eps);
        for (int i = 0; i < D; ++i) y[i] = x[i] * gamma[i] * rrms;
    }
}

void reference_layernorm_modulate(const float* X, const float* gamma, const float* beta,
                                  const float* scale, const float* shift, float eps,
                                  float* Y, int R, int D) {
    const float inv_D = 1.0f / static_cast<float>(D);
    for (int r = 0; r < R; ++r) {
        const float* x = X + r * D;
        float* y = Y + r * D;
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) sum += x[d];
        float mean = sum * inv_D;
        float sum_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = x[d] - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq * inv_D;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int d = 0; d < D; ++d) {
            float xhat = (x[d] - mean) * rstd;
            float ln = gamma[d] * xhat + beta[d];
            y[d] = ln * (1.0f + scale[d]) + shift[d];
        }
    }
}

void reference_gemv_swiglu(const float* X, const float* W_gate, const float* W_up,
                           float* Y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float dot_gate = 0.0f;
        float dot_up = 0.0f;
        const float* wg = W_gate + n * K;
        const float* wu = W_up + n * K;
        for (int k = 0; k < K; ++k) {
            dot_gate += wg[k] * X[k];
            dot_up += wu[k] * X[k];
        }
        float silu = dot_gate / (1.0f + std::exp(-dot_gate));
        Y[n] = silu * dot_up;
    }
}

void reference_gemv_residual(const float* X, const float* W_down, const float* res,
                             float* Y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float dot = 0.0f;
        const float* wd = W_down + n * K;
        for (int k = 0; k < K; ++k) {
            dot += wd[k] * X[k];
        }
        Y[n] = dot + res[n];
    }
}

void test_residual_rmsnorm() {
    std::printf("  Testing fused_residual_rmsnorm...\n");
    SplitMix64 rng(42);
    const std::vector<std::pair<int, int>> shapes = {
        {1, 128}, {4, 256}, {1, 1024}, {2, 4096}
    };
    const float eps = 1e-5f;

    for (auto [B, D] : shapes) {
        Tensor h_cpu = Tensor::zeros_on(Device::CPU, B, D);
        Tensor proj_cpu = Tensor::zeros_on(Device::CPU, B, D);
        Tensor gamma_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor out_cpu = Tensor::empty_on(Device::CPU, B, D);

        fill_random(h_cpu, rng, 1.0f);
        fill_random(proj_cpu, rng, 0.5f);
        fill_random(gamma_cpu, rng, 1.0f);

        Tensor h_ref = h_cpu.clone();
        for (int i = 0; i < B * D; ++i) h_ref.ptr()[i] += proj_cpu.ptr()[i];
        std::vector<float> ref_out(B * D);
        reference_rmsnorm(h_ref.ptr(), gamma_cpu.ptr(), eps, ref_out.data(), B, D);

        // Run CPU op
        brotensor::fused_residual_rmsnorm(h_cpu, proj_cpu, gamma_cpu, eps, out_cpu);

        // Verify CPU in-place h and out
        for (int i = 0; i < B * D; ++i) {
            BT_CHECK(std::fabs(h_cpu.ptr()[i] - h_ref.ptr()[i]) < 1e-5f);
            BT_CHECK(std::fabs(out_cpu.ptr()[i] - ref_out[i]) < 1e-4f);
        }

        // Run GPU if available
        if (brotensor::is_available(Device::CUDA)) {
            Tensor h_gpu = h_ref.clone(); // start from pre-residual h
            for (int i = 0; i < B * D; ++i) h_gpu.ptr()[i] -= proj_cpu.ptr()[i];
            Tensor h_gpu_dev = h_gpu.to(Device::CUDA);
            Tensor proj_gpu_dev = proj_cpu.to(Device::CUDA);
            Tensor gamma_gpu_dev = gamma_cpu.to(Device::CUDA);
            Tensor out_gpu_dev = Tensor::empty_on(Device::CUDA, B, D);

            brotensor::fused_residual_rmsnorm(h_gpu_dev, proj_gpu_dev, gamma_gpu_dev, eps, out_gpu_dev);
            brotensor::sync_all();

            Tensor h_gpu_back = h_gpu_dev.to(Device::CPU);
            Tensor out_gpu_back = out_gpu_dev.to(Device::CPU);
            compare_tensors(h_ref, h_gpu_back, "fused_residual_rmsnorm h (CUDA)", 1e-4f, 1e-4f);
            compare_tensors(out_cpu, out_gpu_back, "fused_residual_rmsnorm out (CUDA)", 1e-4f, 1e-4f);
        }
    }
}

void test_layernorm_modulate() {
    std::printf("  Testing fused_layernorm_modulate...\n");
    SplitMix64 rng(1337);
    const std::vector<std::pair<int, int>> shapes = {
        {1, 256}, {4, 512}, {16, 1152}
    };
    const float eps = 1e-5f;

    for (auto [R, D] : shapes) {
        Tensor x_cpu = Tensor::zeros_on(Device::CPU, R, D);
        Tensor gamma_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor beta_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor scale_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor shift_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor out_cpu = Tensor::empty_on(Device::CPU, R, D);

        fill_random(x_cpu, rng, 1.0f);
        fill_random(gamma_cpu, rng, 1.0f);
        fill_random(beta_cpu, rng, 0.2f);
        fill_random(scale_cpu, rng, 0.1f);
        fill_random(shift_cpu, rng, 0.1f);

        std::vector<float> ref_out(R * D);
        reference_layernorm_modulate(x_cpu.ptr(), gamma_cpu.ptr(), beta_cpu.ptr(),
                                     scale_cpu.ptr(), shift_cpu.ptr(), eps,
                                     ref_out.data(), R, D);

        brotensor::fused_layernorm_modulate(x_cpu, gamma_cpu, beta_cpu,
                                            scale_cpu, shift_cpu, eps, out_cpu);

        for (int i = 0; i < R * D; ++i) {
            BT_CHECK(std::fabs(out_cpu.ptr()[i] - ref_out[i]) < 1e-3f);
        }

        if (brotensor::is_available(Device::CUDA)) {
            Tensor x_dev = x_cpu.to(Device::CUDA);
            Tensor gamma_dev = gamma_cpu.to(Device::CUDA);
            Tensor beta_dev = beta_cpu.to(Device::CUDA);
            Tensor scale_dev = scale_cpu.to(Device::CUDA);
            Tensor shift_dev = shift_cpu.to(Device::CUDA);
            Tensor out_dev = Tensor::empty_on(Device::CUDA, R, D);

            brotensor::fused_layernorm_modulate(x_dev, gamma_dev, beta_dev,
                                                scale_dev, shift_dev, eps, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_layernorm_modulate (CUDA)", 1e-4f, 1e-4f);
        }
    }
}

void test_gemv_swiglu() {
    std::printf("  Testing fused_gemv_swiglu...\n");
    SplitMix64 rng(2026);
    const std::vector<std::pair<int, int>> shapes = {
        {128, 256}, {256, 512}, {1024, 2048}
    };

    for (auto [N, K] : shapes) {
        Tensor x_cpu = Tensor::zeros_on(Device::CPU, 1, K);
        Tensor w_gate_cpu = Tensor::zeros_on(Device::CPU, N, K);
        Tensor w_up_cpu = Tensor::zeros_on(Device::CPU, N, K);
        Tensor out_cpu = Tensor::empty_on(Device::CPU, 1, N);

        fill_random(x_cpu, rng, 0.5f);
        fill_random(w_gate_cpu, rng, 0.1f);
        fill_random(w_up_cpu, rng, 0.1f);

        std::vector<float> ref_out(N);
        reference_gemv_swiglu(x_cpu.ptr(), w_gate_cpu.ptr(), w_up_cpu.ptr(),
                              ref_out.data(), N, K);

        brotensor::fused_gemv_swiglu(x_cpu, w_gate_cpu, w_up_cpu, out_cpu);

        for (int i = 0; i < N; ++i) {
            BT_CHECK(std::fabs(out_cpu.ptr()[i] - ref_out[i]) < 1e-3f);
        }

        if (brotensor::is_available(Device::CUDA)) {
            Tensor x_dev = x_cpu.to(Device::CUDA);
            Tensor w_gate_dev = w_gate_cpu.to(Device::CUDA);
            Tensor w_up_dev = w_up_cpu.to(Device::CUDA);
            Tensor out_dev = Tensor::empty_on(Device::CUDA, 1, N);

            brotensor::fused_gemv_swiglu(x_dev, w_gate_dev, w_up_dev, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_gemv_swiglu (CUDA)", 1e-4f, 1e-4f);
        }
    }
}

void test_gemv_residual() {
    std::printf("  Testing fused_gemv_residual...\n");
    SplitMix64 rng(9999);
    const std::vector<std::pair<int, int>> shapes = {
        {128, 256}, {256, 512}, {1024, 2048}
    };

    for (auto [N, K] : shapes) {
        Tensor x_cpu = Tensor::zeros_on(Device::CPU, 1, K);
        Tensor w_down_cpu = Tensor::zeros_on(Device::CPU, N, K);
        Tensor res_cpu = Tensor::zeros_on(Device::CPU, 1, N);
        Tensor out_cpu = Tensor::empty_on(Device::CPU, 1, N);

        fill_random(x_cpu, rng, 0.5f);
        fill_random(w_down_cpu, rng, 0.1f);
        fill_random(res_cpu, rng, 1.0f);

        std::vector<float> ref_out(N);
        reference_gemv_residual(x_cpu.ptr(), w_down_cpu.ptr(), res_cpu.ptr(),
                                ref_out.data(), N, K);

        brotensor::fused_gemv_residual(x_cpu, w_down_cpu, res_cpu, out_cpu);

        for (int i = 0; i < N; ++i) {
            BT_CHECK(std::fabs(out_cpu.ptr()[i] - ref_out[i]) < 1e-3f);
        }

        // Test in-place aliasing (out == res)
        Tensor res_inplace = res_cpu.clone();
        brotensor::fused_gemv_residual(x_cpu, w_down_cpu, res_inplace, res_inplace);
        for (int i = 0; i < N; ++i) {
            BT_CHECK(std::fabs(res_inplace.ptr()[i] - ref_out[i]) < 1e-3f);
        }

        if (brotensor::is_available(Device::CUDA)) {
            Tensor x_dev = x_cpu.to(Device::CUDA);
            Tensor w_down_dev = w_down_cpu.to(Device::CUDA);
            Tensor res_dev = res_cpu.to(Device::CUDA);
            Tensor out_dev = Tensor::empty_on(Device::CUDA, 1, N);

            brotensor::fused_gemv_residual(x_dev, w_down_dev, res_dev, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_gemv_residual (CUDA)", 1e-4f, 1e-4f);

            // In-place on GPU
            Tensor res_dev_inplace = res_cpu.to(Device::CUDA);
            brotensor::fused_gemv_residual(x_dev, w_down_dev, res_dev_inplace, res_dev_inplace);
            brotensor::sync_all();
            Tensor res_back = res_dev_inplace.to(Device::CPU);
            compare_tensors(out_cpu, res_back, "fused_gemv_residual in-place (CUDA)", 1e-4f, 1e-4f);
        }
    }
}

} // namespace

int main() {
    brotensor::init();
    std::printf("========================================================\n");
    std::printf("  brotensor Fused Ops Parity & Verification Suite\n");
    std::printf("  CUDA available: %s\n", brotensor::is_available(Device::CUDA) ? "YES" : "NO");
    std::printf("========================================================\n");

    try {
        test_residual_rmsnorm();
        test_layernorm_modulate();
        test_gemv_swiglu();
        test_gemv_residual();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception during test: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "Unknown exception during test\n");
        return 1;
    }

    std::printf("\n[SUCCESS] All fused ops tests passed!\n");
    return 0;
}
