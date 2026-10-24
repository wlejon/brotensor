// Issues #138 & #157: Direct Lifted SPIR-V/PTX Backward Shader Audit &
// Multi-Backend Randomized Numerical Adjoint Dot-Product Oracle.
//
// Exercises:
//   1. Direct static lifter proofs over the shipped Vulkan SPIR-V binaries in
//      `shader_table.h` (`brass::spirv::parse_binary`, `verify_spirv_structural`,
//      `prove_spirv_shader_safety`, `PhysicalStorageBuffer64` & `PushConstant`
//      bounds-guard CFG inspection) plus lowered PTX equivalence & safety proofs
//      (`verify_spirv_ptx_equivalence`, `verify_ptx_equivalence`,
//      `verify_spirv_memory_safety`, `verify_ptx_memory_safety`,
//      `verify_ptx_barrier_uniformity`) and negative OOB refutation.
//   2. Multi-backend randomized numerical inner-product adjoint oracle:
//        <F(X), dY>    == <X, B_X(dY)>
//        <F(X, W), dY> == <W, B_W(X, dY)>
//      across every available runtime backend (`brotensor::available_devices()`:
//      `Device::CPU`, plus `Device::VULKAN` / `Device::CUDA` / `Device::Metal`
//      when available) and GPU-vs-CPU backward agreement across randomized
//      shapes for:
//        - conv2d, conv_transpose2d, modulated_conv2d
//        - upfirdn2d (asymmetric padding, distinct up/down factors, flip=false/true)
//        - resample2d (upsample_nearest_2x, downsample_avg_2x, upsample_bilinear_2x,
//          interp2d nearest/bilinear, pad2d zero/reflect/replicate, adaptive_avg_pool2d)
//        - rope (orthogonal rotation transpose & inverse roundtrip)
//        - flash_attention (packed_qkv sliding-window/bidirectional & varlen causal/non-causal)

#include <brotensor/backward_verify.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

static double dot_f32(const Tensor& a_dev, const Tensor& b_dev) {
    Tensor a = a_dev.is_host() ? a_dev : a_dev.to(Device::CPU);
    Tensor b = b_dev.is_host() ? b_dev : b_dev.to(Device::CPU);
    CHECK(a.size() == b.size());
    double s = 0.0;
    const float* pa = a.host_f32();
    const float* pb = b.host_f32();
    for (int i = 0; i < a.size(); ++i) {
        s += static_cast<double>(pa[i]) * static_cast<double>(pb[i]);
    }
    return s;
}

static double max_abs_diff(const Tensor& a_dev, const Tensor& b_dev) {
    Tensor a = a_dev.is_host() ? a_dev : a_dev.to(Device::CPU);
    Tensor b = b_dev.is_host() ? b_dev : b_dev.to(Device::CPU);
    CHECK(a.size() == b.size());
    double m = 0.0;
    const float* pa = a.host_f32();
    const float* pb = b.host_f32();
    for (int i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(pa[i]) - static_cast<double>(pb[i])));
    }
    return m;
}

static Tensor make_random_tensor(Device dev, int rows, int cols, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> host(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (float& v : host) v = dist(rng);
    return Tensor::from_host_on(dev, host.data(), rows, cols);
}

static void check_adjoint(double lhs, double rhs, double rel_tol = 2e-4) {
    const double diff = std::fabs(lhs - rhs);
    const double bound = rel_tol * (1.0 + std::max(std::fabs(lhs), std::fabs(rhs)));
    if (!(diff <= bound)) {
        std::printf("  ADJOINT MISMATCH: lhs=%.8f rhs=%.8f diff=%.8e bound=%.8e\n",
                    lhs, rhs, diff, bound);
    }
    CHECK(diff <= bound);
}

static void test_bv_proof_suite() {
    auto report = brotensor::verify::run_all_backward_index_proofs();
    CHECK(!report.entries.empty());
    for (const auto& entry : report.entries) {
        std::printf("  [Lifted-SPV/PTX] %-60s : %s (%s)\n",
                    entry.name.c_str(),
                    entry.proven ? "PROVEN" : "FAILED",
                    entry.detail.c_str());
        CHECK(entry.proven);
    }
    CHECK(report.all_passed());
}

static void test_conv2d_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0xC0422D01u);
    for (int trial = 0; trial < 8; ++trial) {
        const int groups = (trial % 3 == 0) ? 1 : ((trial % 3 == 1) ? 2 : 3);
        const int Cg_in  = 1 + (trial % 3);
        const int Cg_out = 1 + ((trial + 1) % 3);
        const int C_in   = groups * Cg_in;
        const int C_out  = groups * Cg_out;
        const int N      = 1 + (trial % 3);
        const int kH     = 1 + (trial % 4);
        const int kW     = 1 + ((trial + 2) % 4);
        const int sh     = 1 + (trial % 3);
        const int sw     = 1 + ((trial + 1) % 3);
        const int dh     = 1 + (trial % 2);
        const int dw     = 1 + ((trial + 1) % 2);
        const int ph     = trial % 3;
        const int pw     = (trial + 1) % 3;
        const int H      = dh * (kH - 1) + sh * (2 + (trial % 4)) + 1;
        const int W      = dw * (kW - 1) + sw * (2 + ((trial + 2) % 4)) + 1;
        const int H_out  = (H + 2 * ph - dh * (kH - 1) - 1) / sh + 1;
        const int W_out  = (W + 2 * pw - dw * (kW - 1) - 1) / sw + 1;

        Tensor X_cpu  = make_random_tensor(Device::CPU, N, C_in * H * W, 0.5f, rng);
        Tensor Wt_cpu = make_random_tensor(Device::CPU, C_out, Cg_in * kH * kW, 0.4f, rng);
        Tensor dY_cpu = make_random_tensor(Device::CPU, N, C_out * H_out * W_out, 0.4f, rng);

        Tensor dX_ref, dW_ref;
        for (const Device& dev : devices) {
            Tensor X  = X_cpu.to(dev);
            Tensor Wt = Wt_cpu.to(dev);
            Tensor dY = dY_cpu.to(dev);

            Tensor Y, dX;
            Tensor dWt = Tensor::zeros_on(dev, C_out, Cg_in * kH * kW);
            brotensor::conv2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, kH, kW,
                                      sh, sw, ph, pw, dh, dw, groups, Y);
            brotensor::conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                             sh, sw, ph, pw, dh, dw, groups, dX);
            brotensor::conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                                              sh, sw, ph, pw, dh, dw, groups, dWt);
            brotensor::sync(dev);

            const double lhs   = dot_f32(Y, dY);
            const double rhs_x = dot_f32(X, dX);
            const double rhs_w = dot_f32(Wt, dWt);
            check_adjoint(lhs, rhs_x);
            check_adjoint(lhs, rhs_w);

            if (dev.is_cpu()) {
                dX_ref = dX;
                dW_ref = dWt;
            } else if (!dX_ref.empty()) {
                CHECK(max_abs_diff(dX, dX_ref) <= 2e-4);
                CHECK(max_abs_diff(dWt, dW_ref) <= 2e-4);
            }
        }
    }
}

static void test_conv_transpose2d_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0xC0427202u);
    for (int trial = 0; trial < 6; ++trial) {
        const int groups = (trial % 2 == 0) ? 1 : 2;
        const int Cg_in  = 2 + (trial % 2);
        const int Cg_out = 2 + ((trial + 1) % 2);
        const int C_in   = groups * Cg_in;
        const int C_out  = groups * Cg_out;
        const int N      = 1 + (trial % 2);
        const int H      = 4 + (trial % 4);
        const int W      = 4 + ((trial + 1) % 4);
        const int kH     = 2 + (trial % 3);
        const int kW     = 2 + ((trial + 1) % 3);
        const int sh     = 1 + (trial % 2);
        const int sw     = 1 + ((trial + 1) % 2);
        const int dh     = 1 + (trial % 2);
        const int dw     = 1;
        const int ph     = trial % 2;
        const int pw     = (trial + 1) % 2;
        const int oph    = (sh > 1) ? (trial % sh) : 0;
        const int opw    = (sw > 1) ? ((trial + 1) % sw) : 0;
        const int H_out  = (H - 1) * sh - 2 * ph + dh * (kH - 1) + oph + 1;
        const int W_out  = (W - 1) * sw - 2 * pw + dw * (kW - 1) + opw + 1;

        Tensor X_cpu  = make_random_tensor(Device::CPU, N, C_in * H * W, 0.5f, rng);
        Tensor Wt_cpu = make_random_tensor(Device::CPU, C_in, Cg_out * kH * kW, 0.4f, rng);
        Tensor dY_cpu = make_random_tensor(Device::CPU, N, C_out * H_out * W_out, 0.4f, rng);

        Tensor dX_ref;
        for (const Device& dev : devices) {
            Tensor X  = X_cpu.to(dev);
            Tensor Wt = Wt_cpu.to(dev);
            Tensor dY = dY_cpu.to(dev);

            Tensor Y, dX;
            brotensor::conv_transpose2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, kH, kW,
                                                sh, sw, ph, pw, oph, opw, dh, dw, groups, Y);
            brotensor::conv_transpose2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                                       sh, sw, ph, pw, oph, opw, dh, dw, groups, dX);
            brotensor::sync(dev);

            check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
            if (dev.is_cpu()) {
                dX_ref = dX;
            } else if (!dX_ref.empty()) {
                CHECK(max_abs_diff(dX, dX_ref) <= 2e-4);
            }
        }
    }
}

static void test_modulated_conv2d_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0x40DC0423u);
    for (int trial = 0; trial < 6; ++trial) {
        const int N     = 1 + (trial % 2);
        const int C_in  = 2 + (trial % 3);
        const int C_out = 2 + ((trial + 1) % 3);
        const int H     = 5 + (trial % 4);
        const int Wd    = 5 + ((trial + 2) % 4);
        const int kH    = (trial % 2 == 0) ? 3 : 1;
        const int kW    = (trial % 3 == 0) ? 1 : 3;
        const int ph    = trial % 2;
        const int pw    = (trial + 1) % 2;
        const bool demod = (trial % 2 == 0);
        const int H_out = H + 2 * ph - (kH - 1);
        const int W_out = Wd + 2 * pw - (kW - 1);

        Tensor X_cpu  = make_random_tensor(Device::CPU, N, C_in * H * Wd, 0.5f, rng);
        Tensor W_cpu  = make_random_tensor(Device::CPU, C_out, C_in * kH * kW, 0.4f, rng);
        Tensor s_cpu  = make_random_tensor(Device::CPU, N, C_in, 0.4f, rng);
        for (int i = 0; i < s_cpu.size(); ++i) s_cpu[i] += 1.0f;
        Tensor dY_cpu = make_random_tensor(Device::CPU, N, C_out * H_out * W_out, 0.4f, rng);

        Tensor dX_ref, dW_ref, ds_ref;
        for (const Device& dev : devices) {
            Tensor X  = X_cpu.to(dev);
            Tensor W  = W_cpu.to(dev);
            Tensor s  = s_cpu.to(dev);
            Tensor dY = dY_cpu.to(dev);

            Tensor dcoef, Y, dX, ds;
            Tensor dW = Tensor::zeros_on(dev, C_out, C_in * kH * kW);
            brotensor::modulated_conv2d_forward(X, W, s, N, C_in, H, Wd, C_out, kH, kW,
                                                ph, pw, demod, 1e-8f, dcoef, Y);
            brotensor::modulated_conv2d_backward(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                                                 ph, pw, demod, 1e-8f, dX, dW, ds);
            brotensor::sync(dev);

            const double lhs = dot_f32(Y, dY);
            check_adjoint(lhs, dot_f32(X, dX));
            if (!demod) {
                // When demodulate=false, modulated_conv2d is also linear in W and s:
                check_adjoint(lhs, dot_f32(W, dW));
                check_adjoint(lhs, dot_f32(s, ds));
            }
            if (dev.is_cpu()) {
                dX_ref = dX;
                dW_ref = dW;
                ds_ref = ds;
            } else if (!dX_ref.empty()) {
                CHECK(max_abs_diff(dX, dX_ref) <= 2e-4);
                CHECK(max_abs_diff(dW, dW_ref) <= 2e-4);
                CHECK(max_abs_diff(ds, ds_ref) <= 2e-4);
            }
        }
    }
}

static void test_upfirdn2d_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0x09F12D44u);
    for (int trial = 0; trial < 8; ++trial) {
        const int N      = 1 + (trial % 2);
        const int C      = 1 + (trial % 3);
        const int H      = 5 + (trial % 5);
        const int Wd     = 5 + ((trial + 2) % 5);
        const int fH     = 2 + (trial % 4);
        const int fW     = 2 + ((trial + 1) % 4);
        const int up_x   = 1 + (trial % 3);
        const int up_y   = 1 + ((trial + 1) % 3);
        const int down_x = 1 + ((trial + 2) % 3);
        const int down_y = 1 + (trial % 3);
        const int px0    = trial % 4;
        const int px1    = (trial + 2) % 4;
        const int py0    = (trial + 1) % 4;
        const int py1    = (trial + 3) % 4;
        const float gain = 0.75f + 0.25f * static_cast<float>(trial % 4);
        const int H_out  = (H * up_y + py0 + py1 - fH) / down_y + 1;
        const int W_out  = (Wd * up_x + px0 + px1 - fW) / down_x + 1;
        if (H_out <= 0 || W_out <= 0) continue;

        Tensor X_cpu  = make_random_tensor(Device::CPU, N, C * H * Wd, 0.5f, rng);
        Tensor f_cpu  = make_random_tensor(Device::CPU, fH, fW, 0.4f, rng);
        Tensor dY_cpu = make_random_tensor(Device::CPU, N, C * H_out * W_out, 0.5f, rng);

        for (bool flip : {false, true}) {
            Tensor dX_ref;
            for (const Device& dev : devices) {
                Tensor X  = X_cpu.to(dev);
                Tensor f  = f_cpu.to(dev);
                Tensor dY = dY_cpu.to(dev);

                Tensor Y, dX;
                brotensor::upfirdn2d_forward(X, f, N, C, H, Wd, fH, fW,
                                             up_x, up_y, down_x, down_y,
                                             px0, px1, py0, py1, flip, gain, Y);
                brotensor::upfirdn2d_backward(dY, f, N, C, H, Wd, fH, fW,
                                              up_x, up_y, down_x, down_y,
                                              px0, px1, py0, py1, flip, gain, dX);
                brotensor::sync(dev);

                check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
                if (dev.is_cpu()) {
                    dX_ref = dX;
                } else if (!dX_ref.empty()) {
                    CHECK(max_abs_diff(dX, dX_ref) <= 2e-4);
                }
            }
        }
    }
}

static void test_resample_pad_pool_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0x2E5A4955u);
    for (int trial = 0; trial < 6; ++trial) {
        const int N = 1 + (trial % 2);
        const int C = 1 + (trial % 3);
        const int H = 2 * (2 + (trial % 4)); // even for downsample_avg_2x
        const int W = 2 * (2 + ((trial + 1) % 4));

        Tensor X_cpu     = make_random_tensor(Device::CPU, N, C * H * W, 0.5f, rng);
        Tensor dY_up_cpu = make_random_tensor(Device::CPU, N, C * (2 * H) * (2 * W), 0.4f, rng);
        Tensor dY_dn_cpu = make_random_tensor(Device::CPU, N, C * (H / 2) * (W / 2), 0.4f, rng);

        // 1. upsample_nearest_2x, upsample_bilinear_2x, downsample_avg_2x
        for (const Device& dev : devices) {
            Tensor X     = X_cpu.to(dev);
            Tensor dY_up = dY_up_cpu.to(dev);
            Tensor dY_dn = dY_dn_cpu.to(dev);

            Tensor Y_near, dX_near;
            brotensor::upsample_nearest_2x(X, N, C, H, W, Y_near);
            brotensor::upsample_nearest_2x_backward(dY_up, N, C, H, W, dX_near);

            Tensor Y_bil, dX_bil;
            brotensor::upsample_bilinear_2x(X, N, C, H, W, Y_bil);
            brotensor::upsample_bilinear_2x_backward(dY_up, N, C, H, W, dX_bil);

            Tensor Y_avg, dX_avg;
            brotensor::downsample_avg_2x(X, N, C, H, W, Y_avg);
            brotensor::downsample_avg_2x_backward(dY_dn, N, C, H, W, dX_avg);
            brotensor::sync(dev);

            check_adjoint(dot_f32(Y_near, dY_up), dot_f32(X, dX_near));
            check_adjoint(dot_f32(Y_bil, dY_up), dot_f32(X, dX_bil));
            check_adjoint(dot_f32(Y_avg, dY_dn), dot_f32(X, dX_avg));
        }

        // 2. Arbitrary-scale interp2d (mode 0 = nearest, mode 1 = bilinear)
        const int H_out = 3 + ((trial * 3) % 9);
        const int W_out = 3 + ((trial * 5 + 2) % 9);
        Tensor dY_interp_cpu = make_random_tensor(Device::CPU, N, C * H_out * W_out, 0.4f, rng);
        for (int mode : {0, 1}) {
            Tensor dX_ref;
            for (const Device& dev : devices) {
                Tensor X  = X_cpu.to(dev);
                Tensor dY = dY_interp_cpu.to(dev);
                Tensor Y, dX;
                brotensor::interp2d_forward(X, N, C, H, W, H_out, W_out, mode, Y);
                brotensor::interp2d_backward(dY, N, C, H, W, H_out, W_out, mode, dX);
                brotensor::sync(dev);

                check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
                if (dev.is_cpu()) {
                    dX_ref = dX;
                } else if (!dX_ref.empty()) {
                    CHECK(max_abs_diff(dX, dX_ref) <= 2e-4);
                }
            }
        }

        // 3. pad2d (mode 0 = zero, mode 1 = reflect, mode 2 = replicate)
        const int pt = 1 + (trial % 2);
        const int pb = trial % 3;
        const int pl = (trial + 1) % 3;
        const int pr = 1 + ((trial + 1) % 2);
        const int Hp = H + pt + pb;
        const int Wp = W + pl + pr;
        Tensor dY_pad_cpu = make_random_tensor(Device::CPU, N, C * Hp * Wp, 0.4f, rng);
        for (int mode : {0, 1, 2}) {
            for (const Device& dev : devices) {
                Tensor X  = X_cpu.to(dev);
                Tensor dY = dY_pad_cpu.to(dev);
                Tensor Y, dX;
                brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, mode, Y);
                brotensor::pad2d_backward(dY, N, C, H, W, pt, pb, pl, pr, mode, dX);
                brotensor::sync(dev);

                check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
            }
        }

        // 4. adaptive_avg_pool2d
        const int Ha = 1 + (trial % 4);
        const int Wa = 1 + ((trial + 1) % 4);
        Tensor dY_pool_cpu = make_random_tensor(Device::CPU, N, C * Ha * Wa, 0.4f, rng);
        for (const Device& dev : devices) {
            Tensor X  = X_cpu.to(dev);
            Tensor dY = dY_pool_cpu.to(dev);
            Tensor Y, dX;
            brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, Ha, Wa, Y);
            brotensor::adaptive_avg_pool2d_backward(dY, N, C, H, W, Ha, Wa, dX);
            brotensor::sync(dev);

            check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
        }
    }
}

static void test_rope_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0x209E0066u);
    for (int trial = 0; trial < 6; ++trial) {
        const int L          = 3 + (trial * 3);
        const int num_heads  = 1 + (trial % 4);
        const int head_dim   = 2 * (2 + (trial % 7)); // any even head_dim
        const int seq_offset = trial * 2;
        const float theta    = (trial % 2 == 0) ? 10000.0f : 500000.0f;

        Tensor X_cpu  = make_random_tensor(Device::CPU, L, num_heads * head_dim, 0.6f, rng);
        Tensor dY_cpu = make_random_tensor(Device::CPU, L, num_heads * head_dim, 0.5f, rng);

        Tensor dX_ref;
        for (const Device& dev : devices) {
            Tensor X  = X_cpu.to(dev);
            Tensor dY = dY_cpu.to(dev);

            Tensor Y, dX, X_rt;
            brotensor::rope_forward(X, head_dim, num_heads, seq_offset, theta, Y);
            brotensor::rope_backward(dY, head_dim, num_heads, seq_offset, theta, dX);
            brotensor::rope_backward(Y, head_dim, num_heads, seq_offset, theta, X_rt);
            brotensor::sync(dev);

            check_adjoint(dot_f32(Y, dY), dot_f32(X, dX));
            CHECK(max_abs_diff(X, X_rt) <= 1e-5);
            if (dev.is_cpu()) {
                dX_ref = dX;
            } else if (!dX_ref.empty()) {
                CHECK(max_abs_diff(dX, dX_ref) <= 1e-4);
            }
        }
    }
}

static void test_flash_attention_randomized_multi_backend(const std::vector<Device>& devices) {
    std::mt19937 rng(0xFA774077u);

    // 1. flash_attention_packed_qkv_forward / backward across randomized multi-sequence
    //    packings and both full (window = 0) and sliding-window (window > 0) masks.
    //    For fixed (Q, K), O(Q, K, V) = P(Q, K) * V is strictly linear in V, and
    //    dV = P(Q, K)^T * dO is its exact linear adjoint: <O, dO> == <V, dV>.
    for (int trial = 0; trial < 6; ++trial) {
        const int len0      = 3 + (trial % 5);
        const int len1      = 4 + ((trial + 2) % 5);
        const int L         = len0 + len1;
        const int num_heads = 1 + (trial % 3);
        const int head_dim  = 16 * (1 + (trial % 2));
        const int D         = num_heads * head_dim;
        const int window    = (trial % 2 == 0) ? 0 : (2 + (trial % 4));

        std::vector<std::int32_t> bounds_host(static_cast<std::size_t>(L) * 2);
        for (int r = 0; r < L; ++r) {
            bounds_host[static_cast<std::size_t>(r) * 2 + 0] = (r < len0) ? 0 : len0;
            bounds_host[static_cast<std::size_t>(r) * 2 + 1] = (r < len0) ? len0 : L;
        }

        Tensor QKV_cpu = make_random_tensor(Device::CPU, L, 3 * D, 0.4f, rng);
        Tensor dO_cpu  = make_random_tensor(Device::CPU, L, D, 0.4f, rng);

        // Extract V slice [2*D, 3*D) on CPU for the linear-in-V adjoint inner product
        Tensor V_cpu = Tensor::mat(L, D);
        for (int r = 0; r < L; ++r) {
            for (int c = 0; c < D; ++c) {
                V_cpu(r, c) = QKV_cpu(r, 2 * D + c);
            }
        }

        Tensor dQKV_ref;
        for (const Device& dev : devices) {
            Tensor QKV = QKV_cpu.to(dev);
            Tensor dO  = dO_cpu.to(dev);
            Tensor seq_bounds = Tensor::from_raw_bytes_on(
                dev, bounds_host.data(), L, 2, Dtype::INT32,
                bounds_host.size() * sizeof(std::int32_t));

            Tensor O, dQKV;
            brotensor::flash_attention_packed_qkv_forward(QKV, seq_bounds, num_heads, window, O);
            brotensor::flash_attention_packed_qkv_backward(QKV, dO, seq_bounds, num_heads, window, dQKV);
            brotensor::sync(dev);

            Tensor dQKV_host = dQKV.is_host() ? dQKV : dQKV.to(Device::CPU);
            Tensor dV_host = Tensor::mat(L, D);
            for (int r = 0; r < L; ++r) {
                for (int c = 0; c < D; ++c) {
                    dV_host(r, c) = dQKV_host(r, 2 * D + c);
                }
            }

            check_adjoint(dot_f32(O, dO), dot_f32(V_cpu, dV_host));
            if (dev.is_cpu()) {
                dQKV_ref = dQKV_host;
            } else if (!dQKV_ref.empty()) {
                CHECK(max_abs_diff(dQKV_host, dQKV_ref) <= 3e-4);
            }
        }
    }

    // 2. flash_attention_varlen_forward / backward across causal and non-causal batches
    for (int trial = 0; trial < 6; ++trial) {
        const bool causal   = (trial % 2 == 1);
        const int s0_q      = 3 + (trial % 4);
        const int s0_k      = causal ? s0_q : (4 + ((trial + 1) % 4));
        const int s1_q      = 4 + ((trial + 2) % 4);
        const int s1_k      = causal ? s1_q : (3 + (trial % 5));
        const int total_q   = s0_q + s1_q;
        const int total_k   = s0_k + s1_k;
        const int max_q     = std::max(s0_q, s1_q);
        const int max_k     = std::max(s0_k, s1_k);
        const int num_heads = 1 + (trial % 3);
        const int head_dim  = 16 * (1 + (trial % 2));
        const int D         = num_heads * head_dim;

        const std::int32_t cu_q_host[3] = {0, s0_q, total_q};
        const std::int32_t cu_k_host[3] = {0, s0_k, total_k};

        Tensor Q_cpu  = make_random_tensor(Device::CPU, total_q, D, 0.4f, rng);
        Tensor K_cpu  = make_random_tensor(Device::CPU, total_k, D, 0.4f, rng);
        Tensor V_cpu  = make_random_tensor(Device::CPU, total_k, D, 0.4f, rng);
        Tensor dO_cpu = make_random_tensor(Device::CPU, total_q, D, 0.4f, rng);

        Tensor dQ_ref, dK_ref, dV_ref;
        for (const Device& dev : devices) {
            Tensor Q  = Q_cpu.to(dev);
            Tensor K  = K_cpu.to(dev);
            Tensor V  = V_cpu.to(dev);
            Tensor dO = dO_cpu.to(dev);
            Tensor cu_q = Tensor::from_raw_bytes_on(
                dev, cu_q_host, 3, 1, Dtype::INT32, sizeof(cu_q_host));
            Tensor cu_k = Tensor::from_raw_bytes_on(
                dev, cu_k_host, 3, 1, Dtype::INT32, sizeof(cu_k_host));

            Tensor O, dQ, dK, dV;
            brotensor::flash_attention_varlen_forward(
                Q, K, V,
                static_cast<const std::int32_t*>(cu_q.data),
                static_cast<const std::int32_t*>(cu_k.data),
                /*batch_size=*/2, max_q, max_k, num_heads, head_dim, causal, O);
            brotensor::flash_attention_varlen_backward(
                Q, K, V, O, dO,
                static_cast<const std::int32_t*>(cu_q.data),
                static_cast<const std::int32_t*>(cu_k.data),
                /*batch_size=*/2, max_q, max_k, num_heads, head_dim, causal, dQ, dK, dV);
            brotensor::sync(dev);

            // Exact linear adjoint in V: <O(Q, K, V), dO> == <V, dV>
            check_adjoint(dot_f32(O, dO), dot_f32(V, dV));

            if (dev.is_cpu()) {
                dQ_ref = dQ;
                dK_ref = dK;
                dV_ref = dV;
            } else if (!dQ_ref.empty()) {
                CHECK(max_abs_diff(dQ, dQ_ref) <= 3e-4);
                CHECK(max_abs_diff(dK, dK_ref) <= 3e-4);
                CHECK(max_abs_diff(dV, dV_ref) <= 3e-4);
            }
        }
    }
}

int main() {
    brotensor::init();
    const std::vector<Device> devices = brotensor::available_devices();
    std::printf("=== Issues #138 & #157: Lifted SPIR-V/PTX Proofs & Multi-Backend Adjoint Oracle ===\n");
    std::printf("  Active runtime backends (%zu):", devices.size());
    for (const Device& d : devices) {
        std::printf(" %s", brotensor::to_string(d).c_str());
    }
    std::printf("\n");

    test_bv_proof_suite();
    test_conv2d_randomized_multi_backend(devices);
    test_conv_transpose2d_randomized_multi_backend(devices);
    test_modulated_conv2d_randomized_multi_backend(devices);
    test_upfirdn2d_randomized_multi_backend(devices);
    test_resample_pad_pool_randomized_multi_backend(devices);
    test_rope_randomized_multi_backend(devices);
    test_flash_attention_randomized_multi_backend(devices);

    brotensor::shutdown();
    if (g_failures == 0) {
        std::printf("ALL PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
