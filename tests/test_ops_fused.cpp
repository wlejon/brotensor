// ─── Tests for brotensor::fused_* public API (CPU & GPU) ───────────────────

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

void reference_residual_layernorm(const float* X, const float* gamma, const float* beta,
                                  float eps, float* Y, int B, int D) {
    const float inv_D = 1.0f / static_cast<float>(D);
    for (int b = 0; b < B; ++b) {
        const float* x = X + b * D;
        float* y = Y + b * D;
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
            y[d] = gamma[d] * xhat + beta[d];
        }
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
        if (bt_parity::gpu_device() != Device::CPU) {
            Tensor h_gpu = h_ref.clone(); // start from pre-residual h
            for (int i = 0; i < B * D; ++i) h_gpu.ptr()[i] -= proj_cpu.ptr()[i];
            Tensor h_gpu_dev = h_gpu.to(bt_parity::gpu_device());
            Tensor proj_gpu_dev = proj_cpu.to(bt_parity::gpu_device());
            Tensor gamma_gpu_dev = gamma_cpu.to(bt_parity::gpu_device());
            Tensor out_gpu_dev = Tensor::empty_on(bt_parity::gpu_device(), B, D);

            brotensor::fused_residual_rmsnorm(h_gpu_dev, proj_gpu_dev, gamma_gpu_dev, eps, out_gpu_dev);
            brotensor::sync_all();

            Tensor h_gpu_back = h_gpu_dev.to(Device::CPU);
            Tensor out_gpu_back = out_gpu_dev.to(Device::CPU);
            compare_tensors(h_ref, h_gpu_back, "fused_residual_rmsnorm h (GPU)", 1e-4f, 1e-4f);
            compare_tensors(out_cpu, out_gpu_back, "fused_residual_rmsnorm out (GPU)", 1e-4f, 1e-4f);
        }
    }
}

void test_residual_layernorm() {
    std::printf("  Testing fused_residual_layernorm...\n");
    SplitMix64 rng(4242);
    const std::vector<std::pair<int, int>> shapes = {
        {1, 128}, {4, 256}, {1, 1024}, {2, 4096}, {14, 768}, {64, 1024}
    };
    const float eps = 1e-5f;

    for (auto [B, D] : shapes) {
        Tensor x_cpu = Tensor::zeros_on(Device::CPU, B, D);
        Tensor res_cpu = Tensor::zeros_on(Device::CPU, B, D);
        Tensor gamma_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor beta_cpu = Tensor::zeros_on(Device::CPU, D, 1);
        Tensor out_cpu = Tensor::empty_on(Device::CPU, B, D);

        fill_random(x_cpu, rng, 1.0f);
        fill_random(res_cpu, rng, 0.5f);
        fill_random(gamma_cpu, rng, 1.0f);
        fill_random(beta_cpu, rng, 0.2f);

        Tensor x_ref = x_cpu.clone();
        for (int i = 0; i < B * D; ++i) x_ref.ptr()[i] += res_cpu.ptr()[i];
        std::vector<float> ref_out(B * D);
        reference_residual_layernorm(x_ref.ptr(), gamma_cpu.ptr(), beta_cpu.ptr(), eps, ref_out.data(), B, D);

        // Run CPU op
        brotensor::fused_residual_layernorm(x_cpu, res_cpu, gamma_cpu, beta_cpu, eps, out_cpu);

        // Verify CPU in-place x and out
        for (int i = 0; i < B * D; ++i) {
            BT_CHECK(std::fabs(x_cpu.ptr()[i] - x_ref.ptr()[i]) < 1e-4f);
            BT_CHECK(std::fabs(out_cpu.ptr()[i] - ref_out[i]) < 1e-3f);
        }

        // Run GPU if available
        if (bt_parity::gpu_device() != Device::CPU) {
            Tensor x_gpu = x_ref.clone(); // start from pre-residual x
            for (int i = 0; i < B * D; ++i) x_gpu.ptr()[i] -= res_cpu.ptr()[i];
            Tensor x_gpu_dev = x_gpu.to(bt_parity::gpu_device());
            Tensor res_gpu_dev = res_cpu.to(bt_parity::gpu_device());
            Tensor gamma_gpu_dev = gamma_cpu.to(bt_parity::gpu_device());
            Tensor beta_gpu_dev = beta_cpu.to(bt_parity::gpu_device());
            Tensor out_gpu_dev = Tensor::empty_on(bt_parity::gpu_device(), B, D);

            brotensor::fused_residual_layernorm(x_gpu_dev, res_gpu_dev, gamma_gpu_dev, beta_gpu_dev, eps, out_gpu_dev);
            brotensor::sync_all();

            Tensor x_gpu_back = x_gpu_dev.to(Device::CPU);
            Tensor out_gpu_back = out_gpu_dev.to(Device::CPU);
            compare_tensors(x_ref, x_gpu_back, "fused_residual_layernorm x (GPU)", 1e-4f, 1e-4f);
            compare_tensors(out_cpu, out_gpu_back, "fused_residual_layernorm out (GPU)", 1e-3f, 1e-3f);
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

        if (bt_parity::gpu_device() != Device::CPU) {
            Tensor x_dev = x_cpu.to(bt_parity::gpu_device());
            Tensor gamma_dev = gamma_cpu.to(bt_parity::gpu_device());
            Tensor beta_dev = beta_cpu.to(bt_parity::gpu_device());
            Tensor scale_dev = scale_cpu.to(bt_parity::gpu_device());
            Tensor shift_dev = shift_cpu.to(bt_parity::gpu_device());
            Tensor out_dev = Tensor::empty_on(bt_parity::gpu_device(), R, D);

            brotensor::fused_layernorm_modulate(x_dev, gamma_dev, beta_dev,
                                                scale_dev, shift_dev, eps, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_layernorm_modulate (GPU)", 1e-4f, 1e-4f);
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

        if (bt_parity::gpu_device() != Device::CPU) {
            Tensor x_dev = x_cpu.to(bt_parity::gpu_device());
            Tensor w_gate_dev = w_gate_cpu.to(bt_parity::gpu_device());
            Tensor w_up_dev = w_up_cpu.to(bt_parity::gpu_device());
            Tensor out_dev = Tensor::empty_on(bt_parity::gpu_device(), 1, N);

            brotensor::fused_gemv_swiglu(x_dev, w_gate_dev, w_up_dev, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_gemv_swiglu (GPU)", 1e-4f, 1e-4f);
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

        if (bt_parity::gpu_device() != Device::CPU) {
            Tensor x_dev = x_cpu.to(bt_parity::gpu_device());
            Tensor w_down_dev = w_down_cpu.to(bt_parity::gpu_device());
            Tensor res_dev = res_cpu.to(bt_parity::gpu_device());
            Tensor out_dev = Tensor::empty_on(bt_parity::gpu_device(), 1, N);

            brotensor::fused_gemv_residual(x_dev, w_down_dev, res_dev, out_dev);
            brotensor::sync_all();

            Tensor out_back = out_dev.to(Device::CPU);
            compare_tensors(out_cpu, out_back, "fused_gemv_residual (GPU)", 1e-4f, 1e-4f);

            // In-place on GPU
            Tensor res_dev_inplace = res_cpu.to(bt_parity::gpu_device());
            brotensor::fused_gemv_residual(x_dev, w_down_dev, res_dev_inplace, res_dev_inplace);
            brotensor::sync_all();
            Tensor res_back = res_dev_inplace.to(Device::CPU);
            compare_tensors(out_cpu, res_back, "fused_gemv_residual in-place (GPU)", 1e-4f, 1e-4f);
        }
    }
}

// ── 16-bit operands on HIP ─────────────────────────────────────────────────
//
// None of the five ops has a HIP-specific kernel except the stacked-weight
// SwiGLU GEMV (linear_forward_batched_ex's epilogue); everything else is the
// eager composition, which is what has to be right at FP16 / BF16 too. The
// reference is FP32 over the same rounded inputs, so the tolerance only has
// to admit the 16-bit stores along the way.

float round16(Dtype dt, float v) {
    return dt == Dtype::BF16 ? brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v))
                             : brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

std::vector<float> random16(Dtype dt, SplitMix64& rng, size_t n, float scale, float offset = 0.0f) {
    std::vector<float> v(n);
    for (auto& x : v) x = round16(dt, rng.next_unit() * scale + offset);
    return v;
}

Tensor upload16(Dtype dt, const std::vector<float>& v, int rows, int cols) {
    std::vector<uint16_t> bits(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        bits[i] = dt == Dtype::BF16 ? brotensor::fp32_to_bf16_bits(v[i])
                                    : brotensor::fp32_to_fp16_bits(v[i]);
    }
    return dt == Dtype::BF16
        ? Tensor::from_host_bf16_on(bt_parity::gpu_device(), bits.data(), rows, cols)
        : Tensor::from_host_fp16_on(bt_parity::gpu_device(), bits.data(), rows, cols);
}

Tensor download16(const Tensor& t) {
    brotensor::sync_all();
    std::vector<uint16_t> bits = t.dtype == Dtype::BF16 ? t.to_host_vector_bf16()
                                                        : t.to_host_vector_fp16();
    Tensor out = Tensor::zeros_on(Device::CPU, t.rows, t.cols);
    for (size_t i = 0; i < bits.size(); ++i) {
        out.ptr()[i] = t.dtype == Dtype::BF16 ? brotensor::bf16_bits_to_fp32(bits[i])
                                              : brotensor::fp16_bits_to_fp32(bits[i]);
    }
    return out;
}

Tensor host(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::CPU, v.data(), rows, cols);
}

void test_fused_16bit_hip() {
    if (!bt_parity::gpu_device().is_hip()) return;
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        const char* dn = dt == Dtype::BF16 ? "bf16" : "fp16";
        std::printf("  Testing fused ops at %s on HIP...\n", dn);
        const float tol = dt == Dtype::BF16 ? 3e-2f : 4e-3f;
        SplitMix64 rng(0x16B17 + static_cast<uint64_t>(dt));
        const int B = 6, D = 384;

        // residual + rmsnorm
        {
            auto hh = random16(dt, rng, B * D, 1.0f), hp = random16(dt, rng, B * D, 0.5f);
            auto hg = random16(dt, rng, D, 0.3f, 1.0f);
            std::vector<float> sum(B * D), ref(B * D);
            for (int i = 0; i < B * D; ++i) sum[i] = round16(dt, hh[i] + hp[i]);
            reference_rmsnorm(sum.data(), hg.data(), 1e-6f, ref.data(), B, D);
            Tensor h = upload16(dt, hh, B, D), p = upload16(dt, hp, B, D), g = upload16(dt, hg, D, 1);
            Tensor out;
            brotensor::fused_residual_rmsnorm(h, p, g, 1e-6f, out);
            compare_tensors(host(sum, B, D), download16(h), "residual_rmsnorm h (16-bit HIP)", tol, tol);
            compare_tensors(host(ref, B, D), download16(out), "residual_rmsnorm out (16-bit HIP)", tol, tol);
        }
        // residual + layernorm
        {
            auto hx = random16(dt, rng, B * D, 1.0f), hr = random16(dt, rng, B * D, 0.5f);
            auto hg = random16(dt, rng, D, 0.3f, 1.0f), hb = random16(dt, rng, D, 0.2f);
            std::vector<float> sum(B * D), ref(B * D);
            for (int i = 0; i < B * D; ++i) sum[i] = round16(dt, hx[i] + hr[i]);
            reference_residual_layernorm(sum.data(), hg.data(), hb.data(), 1e-6f, ref.data(), B, D);
            Tensor x = upload16(dt, hx, B, D), r = upload16(dt, hr, B, D);
            Tensor g = upload16(dt, hg, D, 1), b = upload16(dt, hb, D, 1);
            Tensor out;
            brotensor::fused_residual_layernorm(x, r, g, b, 1e-6f, out);
            compare_tensors(host(ref, B, D), download16(out), "residual_layernorm (16-bit HIP)", tol, tol);
        }
        // layernorm + modulate
        {
            auto hx = random16(dt, rng, B * D, 1.0f);
            auto hg = random16(dt, rng, D, 0.3f, 1.0f), hb = random16(dt, rng, D, 0.2f);
            auto hs = random16(dt, rng, D, 0.5f), hsh = random16(dt, rng, D, 0.5f);
            std::vector<float> ref(B * D);
            reference_layernorm_modulate(hx.data(), hg.data(), hb.data(), hs.data(), hsh.data(),
                                         1e-6f, ref.data(), B, D);
            Tensor out;
            brotensor::fused_layernorm_modulate(upload16(dt, hx, B, D), upload16(dt, hg, D, 1),
                                                upload16(dt, hb, D, 1), upload16(dt, hs, D, 1),
                                                upload16(dt, hsh, D, 1), 1e-6f, out);
            compare_tensors(host(ref, B, D), download16(out), "layernorm_modulate (16-bit HIP)",
                            2 * tol, 2 * tol);
        }
        // SwiGLU GEMV: separate weights (eager composition) and the two halves
        // of one stacked weight (the fused GEMV kernel).
        {
            const int N = 320, K = 512;
            auto hx = random16(dt, rng, K, 0.5f);
            auto hw = random16(dt, rng, 2 * N * K, 0.1f);
            std::vector<float> ref(N);
            reference_gemv_swiglu(hx.data(), hw.data(), hw.data() + N * K, ref.data(), N, K);
            Tensor x = upload16(dt, hx, 1, K);
            Tensor w = upload16(dt, hw, 2 * N, K);
            const size_t half = static_cast<size_t>(N) * K * 2;
            Tensor wg = Tensor::view(w.device, w.data, N, K, dt);
            Tensor wu = Tensor::view(w.device, static_cast<char*>(w.data) + half, N, K, dt);
            Tensor out_stacked = Tensor::empty_on(w.device, 1, N, dt);
            brotensor::fused_gemv_swiglu(x, wg, wu, out_stacked);
            compare_tensors(host(ref, 1, N), download16(out_stacked),
                            "gemv_swiglu stacked (16-bit HIP)", tol, tol);
            Tensor wg2 = wg.clone(), wu2 = wu.clone();
            Tensor out_split = Tensor::empty_on(w.device, 1, N, dt);
            brotensor::fused_gemv_swiglu(x, wg2, wu2, out_split);
            compare_tensors(host(ref, 1, N), download16(out_split),
                            "gemv_swiglu separate (16-bit HIP)", tol, tol);
        }
        // GEMV + residual
        {
            const int N = 320, K = 512;
            auto hx = random16(dt, rng, K, 0.5f), hw = random16(dt, rng, N * K, 0.1f);
            auto hr = random16(dt, rng, N, 0.5f);
            std::vector<float> ref(N);
            reference_gemv_residual(hx.data(), hw.data(), hr.data(), ref.data(), N, K);
            Tensor out;
            brotensor::fused_gemv_residual(upload16(dt, hx, 1, K), upload16(dt, hw, N, K),
                                           upload16(dt, hr, 1, N), out);
            compare_tensors(host(ref, 1, N), download16(out), "gemv_residual (16-bit HIP)", tol, tol);
        }
    }
}

} // namespace

int main() {
    brotensor::init();
    std::printf("========================================================\n");
    std::printf("  brotensor Fused Ops Parity & Verification Suite\n");
    std::printf("  GPU: %s\n", brotensor::to_string(bt_parity::gpu_device()).c_str());
    std::printf("========================================================\n");

    try {
        test_residual_rmsnorm();
        test_residual_layernorm();
        test_layernorm_modulate();
        test_gemv_swiglu();
        test_gemv_residual();
        test_fused_16bit_hip();
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
