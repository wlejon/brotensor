// CPU<->GPU parity for the grouped and depthwise conv2d paths (forward and
// backward, FP32 + FP16) and a depthwise finite-difference check. Part of
// brotensor_test_conv2d (main in test_conv2d.cpp).

#include "test_conv2d_common.h"

// ─── Grouped/depthwise parity helpers ───────────────────────────────────
// These exercise the conv2d_*_gpu overloads that take an explicit `groups`
// parameter. CPU references use the same `groups`-aware closed form.
void run_grouped_fp32(const char* label,
                      int N, int C_in, int H, int W,
                      int C_out, int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w,
                      int dil_h, int dil_w,
                      int groups) {
    std::printf("  [fp32 grouped] %s  groups=%d  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d\n",
                label, groups, N, C_in, H, W, C_out, kH, kW);
    std::mt19937 rng(0xA110U + groups);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    const int Cg_in = C_in / groups;
    const int x_n  = N * C_in * H * W;
    const int w_n  = C_out * Cg_in * kH * kW;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), Wt(w_n), bias(C_out), dY(dy_n);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : Wt)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);
    for (auto& v : dY)   v = dist(rng);

    // Forward.
    std::vector<float> Y_cpu;
    conv2d_cpu_fp32(X, Wt, bias, /*has_bias*/true,
                    N, C_in, H, W, C_out, kH, kW,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, Y_cpu, groups);

    Tensor Yg;
    Tensor Xg = Tensor::from_host_on(brotensor::Device::CUDA,
                                     X.data(), N, C_in * H * W);
    Tensor Wg = Tensor::from_host_on(brotensor::Device::CUDA,
                                     Wt.data(), C_out, Cg_in * kH * kW);
    Tensor Bg = Tensor::from_host_on(brotensor::Device::CUDA,
                                     bias.data(), C_out, 1);
    brotensor::conv2d_forward(Xg, Wg, &Bg,
                              N, C_in, H, W, C_out, kH, kW,
                              stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                              groups, Yg);
    CHECK(Yg.cols == C_out * H_out * W_out);
    std::vector<float> Y_gpu(static_cast<size_t>(Yg.size()), 0.0f);
    Yg.copy_to_host(Y_gpu.data()); brotensor::sync_all();
    {
        int bad = 0; float me = 0.0f;
        for (size_t i = 0; i < Y_cpu.size(); ++i) {
            const float e = std::fabs(Y_gpu[i] - Y_cpu[i]);
            if (e > me) me = e;
            const float tol = 1e-4f + 1e-5f * std::fabs(Y_cpu[i]);
            if (e > tol) ++bad;
        }
        std::printf("    fwd  max_err=%g bad=%d / %zu\n", me, bad, Y_cpu.size());
        CHECK(bad == 0);
    }

    // dX.
    std::vector<float> dX_cpu;
    conv2d_backward_input_cpu_fp32(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                   H_out, W_out, dX_cpu, groups);
    Tensor dXg;
    Tensor dYg = Tensor::from_host_on(brotensor::Device::CUDA,
                                      dY.data(), N, C_out * H_out * W_out);
    brotensor::conv2d_backward_input(Wg, dYg, N, C_in, H, W, C_out, kH, kW,
                                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                     groups, dXg);
    std::vector<float> dX_gpu(static_cast<size_t>(dXg.size()), 0.0f);
    dXg.copy_to_host(dX_gpu.data()); brotensor::sync_all();
    {
        int bad = 0; float me = 0.0f;
        for (size_t i = 0; i < dX_cpu.size(); ++i) {
            const float e = std::fabs(dX_gpu[i] - dX_cpu[i]);
            if (e > me) me = e;
            const float tol = 1e-4f + 1e-5f * std::fabs(dX_cpu[i]);
            if (e > tol) ++bad;
        }
        std::printf("    dX   max_err=%g bad=%d / %zu\n", me, bad, dX_cpu.size());
        CHECK(bad == 0);
    }

    // dW.
    std::vector<float> dW_cpu;
    conv2d_backward_weight_cpu_fp32(X, dY, N, C_in, H, W, C_out, kH, kW,
                                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                    H_out, W_out, dW_cpu, groups);
    Tensor dWg = Tensor::zeros_on(brotensor::Device::CUDA,
                                  C_out, Cg_in * kH * kW);
    brotensor::conv2d_backward_weight(Xg, dYg, N, C_in, H, W, C_out, kH, kW,
                                      stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                      groups, dWg);
    std::vector<float> dW_gpu(static_cast<size_t>(dWg.size()), 0.0f);
    dWg.copy_to_host(dW_gpu.data()); brotensor::sync_all();
    {
        int bad = 0; float me = 0.0f;
        for (size_t i = 0; i < dW_cpu.size(); ++i) {
            const float e = std::fabs(dW_gpu[i] - dW_cpu[i]);
            if (e > me) me = e;
            const float tol = 1e-4f + 1e-5f * std::fabs(dW_cpu[i]);
            if (e > tol) ++bad;
        }
        std::printf("    dW   max_err=%g bad=%d / %zu\n", me, bad, dW_cpu.size());
        CHECK(bad == 0);
    }

    // dB (no groups argument by spec).
    std::vector<float> dB_cpu;
    conv2d_backward_bias_cpu_fp32(dY, N, C_out, H_out, W_out, dB_cpu);
    Tensor dBg = Tensor::zeros_on(brotensor::Device::CUDA, C_out, 1);
    brotensor::conv2d_backward_bias(dYg, N, C_out, H_out, W_out, dBg);
    std::vector<float> dB_gpu(static_cast<size_t>(dBg.size()), 0.0f);
    dBg.copy_to_host(dB_gpu.data()); brotensor::sync_all();
    {
        int bad = 0; float me = 0.0f;
        for (int c = 0; c < C_out; ++c) {
            const float e = std::fabs(dB_gpu[c] - dB_cpu[c]);
            if (e > me) me = e;
            const float tol = 1e-4f + 1e-5f * std::fabs(dB_cpu[c]);
            if (e > tol) ++bad;
        }
        std::printf("    dB   max_err=%g bad=%d / %d\n", me, bad, C_out);
        CHECK(bad == 0);
    }
}

void run_grouped_fp16(const char* label,
                      int N, int C_in, int H, int W,
                      int C_out, int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w,
                      int dil_h, int dil_w,
                      int groups) {
    std::printf("  [fp16 grouped] %s  groups=%d  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d\n",
                label, groups, N, C_in, H, W, C_out, kH, kW);
    std::mt19937 rng(0xB220U + groups);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    const int Cg_in = C_in / groups;
    const int x_n  = N * C_in * H * W;
    const int w_n  = C_out * Cg_in * kH * kW;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), Wt(w_n), bias(C_out), dY(dy_n);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : Wt)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);
    for (auto& v : dY)   v = dist(rng);

    auto X_q  = quantize_through_fp16(X);
    auto Wt_q = quantize_through_fp16(Wt);
    auto B_q  = quantize_through_fp16(bias);
    auto dY_q = quantize_through_fp16(dY);

    // CPU references (FP32 math on FP16-quantized inputs).
    std::vector<float> Y_cpu, dX_cpu, dW_cpu, dB_cpu;
    conv2d_cpu_fp32(X_q, Wt_q, B_q, /*has_bias*/true,
                    N, C_in, H, W, C_out, kH, kW,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, Y_cpu, groups);
    conv2d_backward_input_cpu_fp32(Wt_q, dY_q, N, C_in, H, W, C_out, kH, kW,
                                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                   H_out, W_out, dX_cpu, groups);
    conv2d_backward_weight_cpu_fp32(X_q, dY_q, N, C_in, H, W, C_out, kH, kW,
                                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                    H_out, W_out, dW_cpu, groups);
    conv2d_backward_bias_cpu_fp32(dY_q, N, C_out, H_out, W_out, dB_cpu);

    // Upload FP16 tensors.
    auto X_h  = to_fp16(X);
    auto Wt_h = to_fp16(Wt);
    auto B_h  = to_fp16(bias);
    auto dY_h = to_fp16(dY);
    Tensor Yg, dXg;
    Tensor Xg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                          X_h.data(), N, C_in * H * W);
    Tensor Wg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                          Wt_h.data(), C_out, Cg_in * kH * kW);
    Tensor Bg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                          B_h.data(), C_out, 1);
    Tensor dYg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                           dY_h.data(), N, C_out * H_out * W_out);

    // fwd
    brotensor::conv2d_forward(Xg, Wg, &Bg,
                              N, C_in, H, W, C_out, kH, kW,
                              stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                              groups, Yg);
    std::vector<uint16_t> Y_h(Yg.size()); Yg.copy_to_host_fp16(Y_h.data());
    brotensor::sync_all();
    { int bad; float me; check_fp16_against(Y_h, Y_cpu, "fp16-grouped-fwd", bad, me); }

    // dX
    brotensor::conv2d_backward_input(Wg, dYg, N, C_in, H, W, C_out, kH, kW,
                                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                     groups, dXg);
    std::vector<uint16_t> dX_h(dXg.size()); dXg.copy_to_host_fp16(dX_h.data());
    brotensor::sync_all();
    { int bad; float me; check_fp16_against(dX_h, dX_cpu, "fp16-grouped-dX", bad, me); }

    // dW (caller zeros)
    std::vector<uint16_t> zeros_w(w_n, brotensor::fp32_to_fp16_bits(0.0f));
    Tensor dWg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                           zeros_w.data(), C_out, Cg_in * kH * kW);
    brotensor::conv2d_backward_weight(Xg, dYg, N, C_in, H, W, C_out, kH, kW,
                                      stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                      groups, dWg);
    std::vector<uint16_t> dW_h(dWg.size()); dWg.copy_to_host_fp16(dW_h.data());
    brotensor::sync_all();
    { int bad; float me; check_fp16_against(dW_h, dW_cpu, "fp16-grouped-dW", bad, me); }

    // dB
    std::vector<uint16_t> zeros_b(C_out, brotensor::fp32_to_fp16_bits(0.0f));
    Tensor dBg = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                           zeros_b.data(), C_out, 1);
    brotensor::conv2d_backward_bias(dYg, N, C_out, H_out, W_out, dBg);
    std::vector<uint16_t> dB_h(dBg.size()); dBg.copy_to_host_fp16(dB_h.data());
    brotensor::sync_all();
    { int bad; float me; check_fp16_against(dB_h, dB_cpu, "fp16-grouped-dB", bad, me); }
}

// Finite-difference check on a depthwise tiny shape, for both dX and dW.
void run_depthwise_finite_diff() {
    const int groups = 4;
    const int N = 1, C_in = 4, C_out = 4, H = 4, W = 4;
    const int kH = 3, kW = 3;
    const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    const int dil_h = 1, dil_w = 1;
    const int Cg_in = C_in / groups;
    std::printf("  [fp32 depthwise finite-diff] tiny 3x3 same-pad  groups=%d\n", groups);

    std::mt19937 rng(0xD8E4U);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    const int x_n = N * C_in * H * W;
    const int w_n = C_out * Cg_in * kH * kW;
    const int H_out = H, W_out = W;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), Wt(w_n), dY(dy_n);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wt) v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    // Analytic dX and dW from GPU.
    Tensor dXg;
    Tensor Xg = Tensor::from_host_on(brotensor::Device::CUDA,
                                     X.data(), N, C_in * H * W);
    Tensor Wg = Tensor::from_host_on(brotensor::Device::CUDA,
                                     Wt.data(), C_out, Cg_in * kH * kW);
    Tensor dYg = Tensor::from_host_on(brotensor::Device::CUDA,
                                      dY.data(), N, C_out * H_out * W_out);

    brotensor::conv2d_backward_input(Wg, dYg,
                                     N, C_in, H, W, C_out, kH, kW,
                                     stride_h, stride_w, pad_h, pad_w,
                                     dil_h, dil_w, groups, dXg);
    std::vector<float> dX_gpu(static_cast<size_t>(dXg.size()), 0.0f);
    dXg.copy_to_host(dX_gpu.data()); brotensor::sync_all();

    Tensor dWg = Tensor::zeros_on(brotensor::Device::CUDA,
                                  C_out, Cg_in * kH * kW);
    brotensor::conv2d_backward_weight(Xg, dYg,
                                      N, C_in, H, W, C_out, kH, kW,
                                      stride_h, stride_w, pad_h, pad_w,
                                      dil_h, dil_w, groups, dWg);
    std::vector<float> dW_gpu(static_cast<size_t>(dWg.size()), 0.0f);
    dWg.copy_to_host(dW_gpu.data()); brotensor::sync_all();

    auto loss_for_X = [&](const std::vector<float>& X_pert) {
        std::vector<float> Y;
        conv2d_cpu_fp32(X_pert, Wt, /*bias*/std::vector<float>{}, /*has_bias*/false,
                        N, C_in, H, W, C_out, kH, kW,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                        H_out, W_out, Y, groups);
        double s = 0.0;
        for (size_t i = 0; i < Y.size(); ++i) s += static_cast<double>(dY[i]) * Y[i];
        return s;
    };
    auto loss_for_W = [&](const std::vector<float>& W_pert) {
        std::vector<float> Y;
        conv2d_cpu_fp32(X, W_pert, /*bias*/std::vector<float>{}, /*has_bias*/false,
                        N, C_in, H, W, C_out, kH, kW,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                        H_out, W_out, Y, groups);
        double s = 0.0;
        for (size_t i = 0; i < Y.size(); ++i) s += static_cast<double>(dY[i]) * Y[i];
        return s;
    };

    const float eps = 1e-3f;
    // dX FD.
    {
        int bad = 0; float me = 0.0f;
        for (int i = 0; i < x_n; ++i) {
            auto Xp = X, Xm = X;
            Xp[i] += eps; Xm[i] -= eps;
            const float num = static_cast<float>((loss_for_X(Xp) - loss_for_X(Xm)) / (2.0 * eps));
            const float ana = dX_gpu[i];
            const float e = std::fabs(num - ana);
            if (e > me) me = e;
            if (e > 1e-3f + 1e-3f * std::fabs(ana)) {
                if (bad < 5)
                    std::printf("    dX fd mismatch i=%d num=%g ana=%g err=%g\n", i, num, ana, e);
                ++bad;
            }
        }
        std::printf("    depthwise FD dX max_err=%g bad=%d / %d\n", me, bad, x_n);
        CHECK(bad == 0);
    }
    // dW FD.
    {
        int bad = 0; float me = 0.0f;
        for (int i = 0; i < w_n; ++i) {
            auto Wp = Wt, Wm = Wt;
            Wp[i] += eps; Wm[i] -= eps;
            const float num = static_cast<float>((loss_for_W(Wp) - loss_for_W(Wm)) / (2.0 * eps));
            const float ana = dW_gpu[i];
            const float e = std::fabs(num - ana);
            if (e > me) me = e;
            if (e > 1e-3f + 1e-3f * std::fabs(ana)) {
                if (bad < 5)
                    std::printf("    dW fd mismatch i=%d num=%g ana=%g err=%g\n", i, num, ana, e);
                ++bad;
            }
        }
        std::printf("    depthwise FD dW max_err=%g bad=%d / %d\n", me, bad, w_n);
        CHECK(bad == 0);
    }
}
