// Vulkan parity for the audio / codec / spectral ops against the CPU backend:
// Snake (+ beta, in place, backward), pad1d (zero / reflect / replicate and
// the adjoints), resample1d (nearest / linear, up and down, odd lengths),
// causal_conv1d_update (state roll, dilation, multi-step), conv_transpose1d
// (stride, padding, output padding, dilation, groups; the three backwards),
// VQ / FSQ, the complex ops, fft / ifft / rfft / irfft and the adjoints over
// prime, odd and power-of-two lengths, and stft / istft (+ adjoints) centred
// and not, odd n_fft, windows shorter than n_fft, normalised, batched.
// Signals in FP32 (and FP16 / BF16 where the op takes them); the reference
// is the CPU op on the same (rounded) values.

#include "test_vulkan_common.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace vkt {

namespace {

Tensor cpu(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

// |got - want| <= rel * max|want| + 1e-6 (transforms: errors scale with the
// largest output, not each element).
void close_scaled(const std::vector<float>& got, const std::vector<float>& want, float rel, const std::string& tag) {
    float m = 0;
    for (float w : want) m = std::max(m, std::fabs(w));
    expect_close(got, want, rel * m + 1e-6f, 0, tag);
}

std::vector<float> rounded(std::vector<float> v, Dtype dt) {
    for (float& x : v) x = round_to(dt, x);
    return v;
}

const Dtype kTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

void test_snake() {
    for (Dtype dt : kTypes) {
        for (bool with_beta : {false, true}) {
            const int N = 2, C = 5, L = 37;
            const auto x = random_values(std::size_t(N) * C * L, 1, -4, 4, dt);
            const auto al = random_values(C, 2, 0.2f, 2.0f, Dtype::FP32);
            const auto be = random_values(C, 3, 0.5f, 1.5f, Dtype::FP32);
            const auto dy = random_values(x.size(), 4, -1, 1, dt);
            Tensor X = upload(x, N, C * L, dt), A = upload(al, C, 1, Dtype::FP32), B = upload(be, C, 1, Dtype::FP32);
            Tensor Y;
            brotensor::snake_forward(X, A, with_beta ? &B : nullptr, N, C, L, Y);
            Tensor Xc = cpu(x, N, C * L), Ac = cpu(al, C, 1), Bc = cpu(be, C, 1), Yc;
            brotensor::snake_forward(Xc, Ac, with_beta ? &Bc : nullptr, N, C, L, Yc);
            const std::string t = std::string("snake ") + dt_name(dt) + (with_beta ? " beta" : "");
            const float tol = dt == Dtype::FP32 ? 2e-6f : 2 * dtype_eps(dt);
            expect_close(download(Y), rounded(Yc.to_host_vector(), dt), 4e-6f, tol, t);
            if (dt == Dtype::FP32) {   // in place, and the backward
                Tensor Xi = upload(x, N, C * L, dt);
                brotensor::snake_forward(Xi, A, with_beta ? &B : nullptr, N, C, L, Xi);
                expect_close(download(Xi), Yc.to_host_vector(), 4e-6f, 2e-6f, t + " in place");
                Tensor dY = upload(dy, N, C * L, dt), dX, dYc = cpu(dy, N, C * L), dXc;
                Tensor dA = Tensor::zeros_on(vk(), C, 1), dB = Tensor::zeros_on(vk(), C, 1);
                Tensor dAc = Tensor::zeros_on(Device::cpu(), C, 1), dBc = Tensor::zeros_on(Device::cpu(), C, 1);
                brotensor::snake_backward(X, A, with_beta ? &B : nullptr, dY, N, C, L, dX, dA, with_beta ? &dB : nullptr);
                brotensor::snake_backward(Xc, Ac, with_beta ? &Bc : nullptr, dYc, N, C, L, dXc, dAc,
                                          with_beta ? &dBc : nullptr);
                expect_close(download(dX), dXc.to_host_vector(), 1e-5f, 1e-5f, t + " backward dX");
                close_scaled(download(dA), dAc.to_host_vector(), 1e-5f, t + " backward dAlpha");
                if (with_beta) close_scaled(download(dB), dBc.to_host_vector(), 1e-5f, t + " backward dBeta");
            }
        }
    }
}

void test_pad1d() {
    for (Dtype dt : kTypes) {
        for (int mode : {0, 1, 2}) {
            const int N = 2, C = 3, L = 11, pl = mode == 1 ? 10 : 13, pr = mode == 1 ? 4 : 2;
            const auto x = random_values(std::size_t(N) * C * L, 10 + mode, -1, 1, dt);
            Tensor X = upload(x, N, C * L, dt), Y, Xc = cpu(x, N, C * L), Yc;
            brotensor::pad1d_forward(X, N, C, L, pl, pr, mode, Y);
            brotensor::pad1d_forward(Xc, N, C, L, pl, pr, mode, Yc);
            const std::string t = std::string("pad1d mode ") + std::to_string(mode) + " " + dt_name(dt);
            expect_close(download(Y), Yc.to_host_vector(), 0, 0, t);
            const int Lp = L + pl + pr;
            const auto dy = random_values(std::size_t(N) * C * Lp, 20 + mode, -1, 1, dt);
            Tensor dY = upload(dy, N, C * Lp, dt), dX, dYc = cpu(dy, N, C * Lp), dXc;
            brotensor::pad1d_backward(dY, N, C, L, pl, pr, mode, dX);
            brotensor::pad1d_backward(dYc, N, C, L, pl, pr, mode, dXc);
            expect_close(download(dX), rounded(dXc.to_host_vector(), dt), 1e-6f, dt == Dtype::FP32 ? 1e-6f : dtype_eps(dt),
                         t + " backward");
        }
    }
}

void test_resample1d() {
    struct Case { int lin, lout; };
    for (const Case c : {Case{7, 23}, Case{23, 7}, Case{100, 37}, Case{16000, 24000}, Case{24000, 16000}, Case{1, 5}}) {
        for (int mode : {0, 1}) {
            for (Dtype dt : {Dtype::FP32, Dtype::BF16}) {
                const int N = 1, C = 2;
                const auto x = random_values(std::size_t(N) * C * c.lin, 30, -1, 1, dt);
                Tensor X = upload(x, N, C * c.lin, dt), Y, Xc = cpu(x, N, C * c.lin), Yc;
                brotensor::resample1d_forward(X, N, C, c.lin, c.lout, mode, Y);
                brotensor::resample1d_forward(Xc, N, C, c.lin, c.lout, mode, Yc);
                char t[80];
                std::snprintf(t, sizeof t, "resample1d %d -> %d %s %s", c.lin, c.lout, mode ? "linear" : "nearest",
                              dt_name(dt));
                const float tol = dt == Dtype::FP32 ? 1e-6f : dtype_eps(dt);
                expect_close(download(Y), rounded(Yc.to_host_vector(), dt), 1e-6f, tol, t);
                if (dt != Dtype::FP32) continue;
                const auto dy = random_values(std::size_t(N) * C * c.lout, 31, -1, 1, dt);
                Tensor dY = upload(dy, N, C * c.lout, dt), dX, dYc = cpu(dy, N, C * c.lout), dXc;
                brotensor::resample1d_backward(dY, N, C, c.lin, c.lout, mode, dX);
                brotensor::resample1d_backward(dYc, N, C, c.lin, c.lout, mode, dXc);
                expect_close(download(dX), dXc.to_host_vector(), 2e-6f, 2e-6f, std::string(t) + " backward");
            }
        }
    }
}

void test_causal_conv1d_update() {
    for (int Ls : {1, 5}) {
        for (int dil : {1, 2}) {
            const int N = 2, C = 6, kL = 4, hist = (kL - 1) * dil;
            const auto w = random_values(std::size_t(C) * kL, 40, -1, 1, Dtype::FP32);
            const auto b = random_values(C, 41, -1, 1, Dtype::FP32);
            const auto s0 = random_values(std::size_t(N) * C * hist, 42, -1, 1, Dtype::FP32);
            Tensor W = upload(w, C, kL, Dtype::FP32), B = upload(b, C, 1, Dtype::FP32);
            Tensor S = upload(s0, N, C * hist, Dtype::FP32), Wc = cpu(w, C, kL), Bc = cpu(b, C, 1);
            Tensor Sc = cpu(s0, N, C * hist);
            for (int step = 0; step < 3; ++step) {   // the state carries across calls
                const auto x = random_values(std::size_t(N) * C * Ls, 43 + step, -1, 1, Dtype::FP32);
                Tensor X = upload(x, N, C * Ls, Dtype::FP32), Y, Xc = cpu(x, N, C * Ls), Yc;
                brotensor::causal_conv1d_update(X, W, &B, N, C, Ls, kL, dil, S, Y);
                brotensor::causal_conv1d_update(Xc, Wc, &Bc, N, C, Ls, kL, dil, Sc, Yc);
                char t[80];
                std::snprintf(t, sizeof t, "causal_conv1d_update L_step=%d dil=%d step %d", Ls, dil, step);
                expect_close(download(Y), Yc.to_host_vector(), 1e-6f, 1e-6f, t);
                expect_close(download(S), Sc.to_host_vector(), 0, 0, std::string(t) + " state");
            }
        }
    }
}

struct ConvT { int n, cin, L, cout, kL, s, p, op, d, g; const char* name; };

void test_conv_transpose1d() {
    const ConvT cases[] = {
        {1, 16, 23, 8, 16, 8, 4, 0, 1, 1, "HiFiGAN-like k16 s8 p4"},
        {2, 6, 17, 4, 5, 3, 2, 1, 1, 1, "k5 s3 p2 output_padding 1"},
        {1, 8, 13, 12, 3, 2, 1, 1, 2, 2, "dilation 2 groups 2"},
        {1, 32, 40, 32, 4, 2, 1, 0, 1, 1, "32 ch k4 s2"},
        {1, 4, 9, 4, 3, 1, 0, 2, 3, 4, "depthwise, output_padding 2 (< dilation)"},
    };
    for (const ConvT& c : cases) {
        for (Dtype dt : kTypes) {
            const int Cg = c.cout / c.g, Lo = (c.L - 1) * c.s - 2 * c.p + c.d * (c.kL - 1) + c.op + 1;
            const auto x = random_values(std::size_t(c.n) * c.cin * c.L, 50, -1, 1, dt);
            const auto w = random_values(std::size_t(c.cin) * Cg * c.kL, 51, -0.5f, 0.5f, dt);
            const auto b = random_values(c.cout, 52, -0.5f, 0.5f, dt);
            Tensor X = upload(x, c.n, c.cin * c.L, dt), W = upload(w, c.cin, Cg * c.kL, dt), B = upload(b, c.cout, 1, dt);
            Tensor Xc = cpu(x, c.n, c.cin * c.L), Wc = cpu(w, c.cin, Cg * c.kL), Bc = cpu(b, c.cout, 1), Y, Yc;
            brotensor::conv_transpose1d_forward(X, W, &B, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d, c.g, Y);
            brotensor::conv_transpose1d_forward(Xc, Wc, &Bc, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d, c.g, Yc);
            const std::string t = std::string("conv_transpose1d ") + c.name + " " + dt_name(dt);
            const float atol = dt == Dtype::FP32 ? 1e-5f : 3e-3f, rtol = dt == Dtype::FP32 ? 1e-5f : 2 * dtype_eps(dt);
            expect_close(download(Y), rounded(Yc.to_host_vector(), dt), atol, rtol, t);
            const auto dy = random_values(std::size_t(c.n) * c.cout * Lo, 53, -1, 1, dt);
            Tensor dY = upload(dy, c.n, c.cout * Lo, dt), dYc = cpu(dy, c.n, c.cout * Lo), dX, dXc;
            brotensor::conv_transpose1d_backward_input(W, dY, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d, c.g, dX);
            brotensor::conv_transpose1d_backward_input(Wc, dYc, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d, c.g,
                                                       dXc);
            expect_close(download(dX), rounded(dXc.to_host_vector(), dt), atol, rtol, t + " backward input");
            if (dt != Dtype::FP32) continue;
            const auto w0 = random_values(w.size(), 54, -0.1f, 0.1f, dt);
            Tensor dW = upload(w0, c.cin, Cg * c.kL, dt), dWc = cpu(w0, c.cin, Cg * c.kL);
            brotensor::conv_transpose1d_backward_weight(X, dY, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d, c.g,
                                                        dW);
            brotensor::conv_transpose1d_backward_weight(Xc, dYc, c.n, c.cin, c.L, c.cout, c.kL, c.s, c.p, c.op, c.d,
                                                        c.g, dWc);
            expect_close(download(dW), dWc.to_host_vector(), 2e-5f, 2e-5f, t + " backward weight (accumulates)");
            const auto b0 = random_values(c.cout, 55, -0.1f, 0.1f, dt);
            Tensor dB = upload(b0, c.cout, 1, dt), dBc = cpu(b0, c.cout, 1);
            brotensor::conv_transpose1d_backward_bias(dY, c.n, c.cout, Lo, dB);
            brotensor::conv_transpose1d_backward_bias(dYc, c.n, c.cout, Lo, dBc);
            expect_close(download(dB), dBc.to_host_vector(), 2e-5f, 2e-5f, t + " backward bias (accumulates)");
        }
    }
}

void test_codec() {
    const int N = 45, D = 7, K = 300;
    auto x = random_values(std::size_t(N) * D, 60, -1.2f, 1.2f, Dtype::FP32);
    const auto cb = random_values(std::size_t(K) * D, 61, -1, 1, Dtype::FP32);
    Tensor X = upload(x, N, D, Dtype::FP32), CB = upload(cb, K, D, Dtype::FP32), I, Q;
    Tensor Xc = cpu(x, N, D), CBc = cpu(cb, K, D), Ic, Qc;
    brotensor::vq_encode_forward(X, CB, I, Q);
    brotensor::vq_encode_forward(Xc, CBc, Ic, Qc);
    std::vector<std::int32_t> iv(N), ic(N);
    I.copy_to_host_raw(iv.data(), iv.size() * 4);
    Ic.copy_to_host_raw(ic.data(), ic.size() * 4);
    expect_equal_bits(iv.data(), ic.data(), iv.size() * 4, "vq_encode indices");
    expect_close(download(Q), Qc.to_host_vector(), 0, 0, "vq_encode quantized");
    Tensor dX;
    brotensor::vq_encode_backward(Q, dX);
    expect_close(download(dX), Qc.to_host_vector(), 0, 0, "vq_encode backward (straight through)");

    const std::vector<std::int32_t> lv = {8, 5, 5, 5, 3, 7, 2};
    for (std::size_t i = 0; i < x.size(); i += 5) x[i] = 0.25f * float(int(i % 9) - 4);   // half-way ties
    Tensor X2 = upload(x, N, D, Dtype::FP32), X2c = cpu(x, N, D);
    Tensor Lv = Tensor::from_raw_bytes_on(vk(), lv.data(), D, 1, Dtype::INT32, lv.size() * 4);
    Tensor Lc = Tensor::from_raw_bytes_on(Device::cpu(), lv.data(), D, 1, Dtype::INT32, lv.size() * 4);
    Tensor FQ, FP, FQc, FPc;
    brotensor::fsq_quantize_forward(X2, Lv, FQ, FP);
    brotensor::fsq_quantize_forward(X2c, Lc, FQc, FPc);
    expect_close(download(FQ), FQc.to_host_vector(), 0, 0, "fsq_quantize quantized");
    std::vector<std::int32_t> pv(N), pc(N);
    FP.copy_to_host_raw(pv.data(), pv.size() * 4);
    FPc.copy_to_host_raw(pc.data(), pc.size() * 4);
    expect_equal_bits(pv.data(), pc.data(), pv.size() * 4, "fsq_quantize packed indices");
}

void test_complex() {
    const int R = 3, C = 41;
    const auto a = random_values(std::size_t(R) * 2 * C, 70, -2, 2, Dtype::FP32);
    auto b = random_values(std::size_t(R) * 2 * C, 71, -2, 2, Dtype::FP32);
    b[0] = 0; b[1] = 0;   // a zero magnitude for the abs backward
    b[2] = -1; b[3] = 0;  // angle pi
    Tensor A = upload(a, R, 2 * C, Dtype::FP32), B = upload(b, R, 2 * C, Dtype::FP32);
    Tensor Ac = cpu(a, R, 2 * C), Bc = cpu(b, R, 2 * C), Y, Yc;
    brotensor::complex_mul(A, B, Y);
    brotensor::complex_mul(Ac, Bc, Yc);
    expect_close(download(Y), Yc.to_host_vector(), 1e-6f, 1e-6f, "complex_mul");
    Tensor dA = Tensor::zeros_on(vk(), R, 2 * C), dB = Tensor::zeros_on(vk(), R, 2 * C);
    Tensor dAc = Tensor::zeros_on(Device::cpu(), R, 2 * C), dBc = Tensor::zeros_on(Device::cpu(), R, 2 * C);
    brotensor::complex_mul_backward(A, B, A, dA, dB);
    brotensor::complex_mul_backward(Ac, Bc, Ac, dAc, dBc);
    expect_close(download(dA), dAc.to_host_vector(), 1e-6f, 1e-6f, "complex_mul_backward dA");
    expect_close(download(dB), dBc.to_host_vector(), 1e-6f, 1e-6f, "complex_mul_backward dB");
    brotensor::complex_abs(B, Y);
    brotensor::complex_abs(Bc, Yc);
    expect_close(download(Y), Yc.to_host_vector(), 1e-6f, 1e-6f, "complex_abs");
    const auto g = random_values(std::size_t(R) * C, 72, -1, 1, Dtype::FP32);
    Tensor G = upload(g, R, C, Dtype::FP32), Gc = cpu(g, R, C), dZ, dZc;
    brotensor::complex_abs_backward(B, G, dZ);
    brotensor::complex_abs_backward(Bc, Gc, dZc);
    expect_close(download(dZ), dZc.to_host_vector(), 1e-6f, 1e-6f, "complex_abs_backward (zero magnitude -> 0)");
    brotensor::complex_angle(B, Y);
    brotensor::complex_angle(Bc, Yc);
    expect_close(download(Y), Yc.to_host_vector(), 1e-6f, 1e-6f, "complex_angle");
    const auto ph = random_values(std::size_t(R) * C, 73, -40, 40, Dtype::FP32);
    Tensor P = upload(ph, R, C, Dtype::FP32), Pc = cpu(ph, R, C);
    brotensor::complex_from_polar(G, P, Y);
    brotensor::complex_from_polar(Gc, Pc, Yc);
    expect_close(download(Y), Yc.to_host_vector(), 1e-6f, 1e-6f, "complex_from_polar");
}

void test_fft() {
    for (int L : {1, 2, 7, 16, 97, 400, 1000}) {
        const int R = 3, C = L / 2 + 1;
        const auto z = random_values(std::size_t(R) * 2 * L, 80 + L, -1, 1, Dtype::FP32);
        const auto r = random_values(std::size_t(R) * L, 81 + L, -1, 1, Dtype::FP32);
        const auto h = random_values(std::size_t(R) * 2 * C, 82 + L, -1, 1, Dtype::FP32);
        Tensor Z = upload(z, R, 2 * L, Dtype::FP32), X = upload(r, R, L, Dtype::FP32), H = upload(h, R, 2 * C, Dtype::FP32);
        Tensor Zc = cpu(z, R, 2 * L), Xc = cpu(r, R, L), Hc = cpu(h, R, 2 * C), Y, Yc;
        const std::string t = " L=" + std::to_string(L);
        const float rel = 2e-6f * std::sqrt(float(L)) + 1e-6f;
        brotensor::fft(Z, Y); brotensor::fft(Zc, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "fft" + t);
        brotensor::ifft(Z, Y); brotensor::ifft(Zc, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "ifft" + t);
        brotensor::rfft(X, Y); brotensor::rfft(Xc, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "rfft" + t);
        brotensor::irfft(H, L, Y); brotensor::irfft(Hc, L, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "irfft" + t);
        brotensor::rfft_backward(H, L, Y); brotensor::rfft_backward(Hc, L, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "rfft_backward" + t);
        brotensor::irfft_backward(X, Y); brotensor::irfft_backward(Xc, Yc);
        close_scaled(download(Y), Yc.to_host_vector(), rel, "irfft_backward" + t);
    }
}

struct Stft { int N, L, n_fft, hop, win; bool center, norm; const char* name; };

void test_stft() {
    const Stft cases[] = {
        {1, 4000, 400, 160, 400, true, false, "Whisper 400/160"},
        {2, 3001, 512, 128, 400, true, true, "n_fft 512 win 400 normalised, batch 2"},
        {1, 1500, 255, 64, 200, true, false, "odd n_fft 255"},
        {1, 2048, 1024, 256, 1024, false, false, "not centred"},
        {1, 129, 128, 33, 97, true, false, "short signal, odd window"},
    };
    for (const Stft& c : cases) {
        std::vector<float> win(c.win);
        for (int i = 0; i < c.win; ++i) win[i] = 0.5f - 0.5f * std::cos(6.283185307179586 * i / c.win);   // Hann
        const auto sig = random_values(std::size_t(c.N) * c.L, 90, -1, 1, Dtype::FP32);
        Tensor S = upload(sig, c.N, c.L, Dtype::FP32), Wn = upload(win, 1, c.win, Dtype::FP32), Sp;
        Tensor Sc = cpu(sig, c.N, c.L), Wc = cpu(win, 1, c.win), Spc;
        const std::string t = std::string(" ") + c.name;
        const float rel = 3e-6f * std::sqrt(float(c.n_fft)) + 1e-6f;
        brotensor::stft(S, Wn, c.N, c.n_fft, c.hop, c.win, c.center, c.norm, Sp);
        brotensor::stft(Sc, Wc, c.N, c.n_fft, c.hop, c.win, c.center, c.norm, Spc);
        close_scaled(download(Sp), Spc.to_host_vector(), rel, "stft" + t);
        const std::vector<float> spec = Spc.to_host_vector();
        Tensor In = upload(spec, Spc.rows, Spc.cols, Dtype::FP32), Inc = cpu(spec, Spc.rows, Spc.cols), Y, Yc;
        brotensor::istft(In, Wn, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, Y);
        brotensor::istft(Inc, Wc, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, Yc);
        // Where the COLA envelope is tiny (uncentred edges under a Hann
        // window) istft divides by it: the CPU's FP64 and this FP32 then
        // disagree by eps / envelope. Compare where the envelope is >= 1e-3.
        std::vector<float> got = download(Y), want = Yc.to_host_vector();
        {
            const int shift = c.center ? c.n_fft / 2 : 0, pad_lo = (c.n_fft - c.win) / 2;
            const int frames = c.center ? 1 + c.L / c.hop : 1 + (c.L - c.n_fft) / c.hop;
            for (int n = 0; n < c.L; ++n) {
                double env = 0;
                for (int f = 0; f < frames; ++f) {
                    const int j = n + shift - f * c.hop - pad_lo;
                    if (j >= 0 && j < c.win) env += double(win[j]) * win[j];
                }
                if (env < 1e-3)
                    for (int b = 0; b < c.N; ++b) got[std::size_t(b) * c.L + n] = want[std::size_t(b) * c.L + n] = 0;
            }
        }
        close_scaled(got, want, rel, "istft" + t);
        if (c.center && c.win == c.n_fft) close_scaled(download(Y), sig, rel * 4, "istft(stft(x)) == x" + t);
        const auto g = random_values(spec.size(), 91, -1, 1, Dtype::FP32);
        Tensor G = upload(g, Spc.rows, Spc.cols, Dtype::FP32), Gc = cpu(g, Spc.rows, Spc.cols), dS, dSc;
        brotensor::stft_backward(G, Wn, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, dS);
        brotensor::stft_backward(Gc, Wc, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, dSc);
        close_scaled(download(dS), dSc.to_host_vector(), rel, "stft_backward" + t);
        brotensor::istft_backward(S, Wn, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, dS);
        brotensor::istft_backward(Sc, Wc, c.N, c.L, c.n_fft, c.hop, c.win, c.center, c.norm, dSc);
        close_scaled(download(dS), dSc.to_host_vector(), rel, "istft_backward" + t);
    }
}

}  // namespace

void run_sampling_tests();   // test_vulkan_sampling.cpp

void run_audio_tests() {
    std::printf("\n[audio / codec / spectral]\n");
    test_snake();
    test_pad1d();
    test_resample1d();
    test_causal_conv1d_update();
    test_conv_transpose1d();
    test_codec();
    test_complex();
    test_fft();
    test_stft();
    run_sampling_tests();
}

}  // namespace vkt
