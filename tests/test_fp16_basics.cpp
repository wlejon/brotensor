// CPU↔GPU parity for the FP16-gap basic ops: linear_forward_batched_fp16,
// add_inplace/scale_inplace/mul_inplace (FP16 dispatch), concat_rows /
// concat_batched_rows (FP16), and layernorm_forward_inference_batched_fp16.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}
static std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

static void check_fp16(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref, const char* label) {
    CHECK(got.size() == ref.size());
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, g, ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

static void test_linear_fp16() {
    std::printf("  linear_forward_batched_fp16\n");
    const int B = 5, in_dim = 11, out_dim = 7;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> Wf(out_dim * in_dim), Bf(out_dim), Xf(B * in_dim);
    for (auto& v : Wf) v = dist(rng);
    for (auto& v : Bf) v = dist(rng);
    for (auto& v : Xf) v = dist(rng);

    auto Wq = rq(Wf), Bq = rq(Bf), Xq = rq(Xf);
    std::vector<float> Ref(B * out_dim, 0.0f);
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out_dim; ++o) {
            double s = Bq[o];
            for (int k = 0; k < in_dim; ++k) s += static_cast<double>(Xq[b*in_dim+k]) * Wq[o*in_dim+k];
            Ref[b*out_dim+o] = static_cast<float>(s);
        }

    Tensor W, Bb, X, Y;
    auto Wh = to_fp16(Wf), Bh = to_fp16(Bf), Xh = to_fp16(Xf);
    W  = Tensor::from_host_fp16_on(Device::CUDA, Wh.data(), out_dim, in_dim);
    Bb = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), out_dim, 1);
    X  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), B, in_dim);
    brotensor::linear_forward_batched_fp16(W, &Bb, X, Y);
    CHECK(Y.rows == B && Y.cols == out_dim && Y.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Y.size());
    Y.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, Ref, "linear");

    // No-bias variant.
    Tensor Y2;
    brotensor::linear_forward_batched_fp16(W, nullptr, X, Y2);
    std::vector<uint16_t> got2(Y2.size());
    Y2.copy_to_host_fp16(got2.data());
    brotensor::sync_all();
    std::vector<float> Ref2 = Ref;
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out_dim; ++o) Ref2[b*out_dim+o] -= Bq[o];
    check_fp16(got2, Ref2, "linear-nobias");
}

// Reference for the fused epilogue activation (FP32 domain), matching
// src/cuda/detail/activations.cuh.
static float ref_linear_act(int act, float v) {
    switch (act) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: { const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
                  return 0.5f * v * (1.0f + std::tanh(u)); }
        case 3: return 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
        case 4: return v / (1.0f + std::exp(-v));
        case 5: return v / (1.0f + std::exp(-1.702f * v));
        default: return v;
    }
}

// Drives linear_forward_batched_fp16_act for one shape/activation and compares
// against an FP32 linear+bias+activation reference.
static void run_linear_act_case(int B, int in_dim, int out_dim,
                                 int act, const char* label) {
    std::mt19937 rng(4321u + static_cast<unsigned>(act));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> Wf(out_dim * in_dim), Bf(out_dim), Xf(B * in_dim);
    for (auto& v : Wf) v = dist(rng);
    for (auto& v : Bf) v = dist(rng);
    for (auto& v : Xf) v = dist(rng);

    auto Wq = rq(Wf), Bq = rq(Bf), Xq = rq(Xf);
    std::vector<float> Ref(B * out_dim, 0.0f);
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out_dim; ++o) {
            double s = Bq[o];
            for (int k = 0; k < in_dim; ++k)
                s += static_cast<double>(Xq[b*in_dim+k]) * Wq[o*in_dim+k];
            Ref[b*out_dim+o] = ref_linear_act(act, static_cast<float>(s));
        }

    Tensor W, Bb, X, Y;
    auto Wh = to_fp16(Wf), Bh = to_fp16(Bf), Xh = to_fp16(Xf);
    W  = Tensor::from_host_fp16_on(Device::CUDA, Wh.data(), out_dim, in_dim);
    Bb = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), out_dim, 1);
    X  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), B, in_dim);
    brotensor::linear_forward_batched_fp16_act(W, &Bb, X, act, Y);
    CHECK(Y.rows == B && Y.cols == out_dim && Y.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Y.size());
    Y.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, Ref, label);
}

static void test_linear_fp16_act() {
    std::printf("  linear_forward_batched_fp16_act\n");
    // WMMA store path: M*N>=256, K>=16, K%8==0, N%8==0 — exercises the fused
    // epilogue in the vectorised store stage.
    run_linear_act_case(16, 32, 24, brotensor::kLinearActRelu,      "wmma-relu");
    run_linear_act_case(16, 32, 24, brotensor::kLinearActGeluTanh,  "wmma-gelu_tanh");
    run_linear_act_case(16, 32, 24, brotensor::kLinearActSilu,      "wmma-silu");
    run_linear_act_case(16, 32, 24, brotensor::kLinearActQuickGelu, "wmma-quick_gelu");
    // Naive fallback path (K%8!=0) — same epilogue applied per element.
    run_linear_act_case(5, 11, 7,   brotensor::kLinearActSilu,      "naive-silu");
    // Skinny-batch GEMV path (B<=32, K%4==0) at the AR-decode shape.
    run_linear_act_case(1, 64, 48,  brotensor::kLinearActSilu,      "gemv-silu");
}

static void test_elementwise_fp16() {
    std::printf("  elementwise fp16 (add/scale/mul inplace)\n");
    const int N = 128;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(N), Bv(N);
    for (auto& v : A)  v = dist(rng);
    for (auto& v : Bv) v = dist(rng);
    auto Aq = rq(A), Bq = rq(Bv);

    // add
    Tensor Ag, Bg;
    auto Ah = to_fp16(A), Bh = to_fp16(Bv);
    Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(), N, 1);
    Bg = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), N, 1);
    brotensor::add_inplace(Ag, Bg);
    std::vector<uint16_t> got(N);
    Ag.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    std::vector<float> Ref(N);
    for (int i = 0; i < N; ++i) Ref[i] = Aq[i] + Bq[i];
    check_fp16(got, Ref, "add_inplace");

    // scale (re-upload)
    Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(), N, 1);
    brotensor::scale_inplace(Ag, 0.25f);
    Ag.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    for (int i = 0; i < N; ++i) Ref[i] = Aq[i] * 0.25f;
    check_fp16(got, Ref, "scale_inplace");

    // mul
    Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(), N, 1);
    brotensor::mul_inplace(Ag, Bg);
    Ag.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    for (int i = 0; i < N; ++i) Ref[i] = Aq[i] * Bq[i];
    check_fp16(got, Ref, "mul_inplace");
}

static void test_concat_fp16() {
    std::printf("  concat_rows / concat_batched_rows fp16\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> P1(13), P2(7), P3(4);
    for (auto& v : P1) v = dist(rng);
    for (auto& v : P2) v = dist(rng);
    for (auto& v : P3) v = dist(rng);
    auto P1q = rq(P1), P2q = rq(P2), P3q = rq(P3);

    Tensor G1, G2, G3, Out;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    G1 = Tensor::from_host_fp16_on(Device::CUDA, P1h.data(), 13, 1);
    G2 = Tensor::from_host_fp16_on(Device::CUDA, P2h.data(), 7, 1);
    G3 = Tensor::from_host_fp16_on(Device::CUDA, P3h.data(), 4, 1);
    brotensor::concat_rows({&G1, &G2, &G3}, Out);
    CHECK(Out.rows == 24 && Out.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Out.size());
    Out.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    std::vector<float> Ref;
    for (auto v : P1q) Ref.push_back(v);
    for (auto v : P2q) Ref.push_back(v);
    for (auto v : P3q) Ref.push_back(v);
    check_fp16(got, Ref, "concat_rows");

    // batched: (B=3, 4) and (B=3, 2)
    std::vector<float> A(3*4), Bv(3*2);
    for (auto& v : A) v = dist(rng);
    for (auto& v : Bv) v = dist(rng);
    auto Aq = rq(A), Bq = rq(Bv);
    Tensor Ag, Bg, OutB;
    auto Ah = to_fp16(A), Bh = to_fp16(Bv);
    Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(), 3, 4);
    Bg = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), 3, 2);
    brotensor::concat_batched_rows({&Ag, &Bg}, OutB);
    CHECK(OutB.rows == 3 && OutB.cols == 6 && OutB.dtype == Dtype::FP16);
    std::vector<uint16_t> gotB(OutB.size());
    OutB.copy_to_host_fp16(gotB.data());
    brotensor::sync_all();
    std::vector<float> RefB(3*6);
    for (int b = 0; b < 3; ++b) {
        for (int j = 0; j < 4; ++j) RefB[b*6+j]     = Aq[b*4+j];
        for (int j = 0; j < 2; ++j) RefB[b*6+4+j]   = Bq[b*2+j];
    }
    check_fp16(gotB, RefB, "concat_batched_rows");
}

static void test_concat_nchw_channels_fp16() {
    // Verify the per-sample channel regrouping that distinguishes
    // concat_nchw_channels_gpu from a flat byte concat. Use N=2 so the
    // sample-interleaving bug in concat_rows_gpu would be visible.
    std::printf("  concat_nchw_channels fp16\n");
    const int N = 2, H = 2, W = 3;
    const int C1 = 2, C2 = 3, C3 = 1;
    const int total_C = C1 + C2 + C3;
    const int HW = H * W;
    std::mt19937 rng(0xC0CAC07A);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> P1(N*C1*HW), P2(N*C2*HW), P3(N*C3*HW);
    for (auto& v : P1) v = dist(rng);
    for (auto& v : P2) v = dist(rng);
    for (auto& v : P3) v = dist(rng);
    auto P1q = rq(P1), P2q = rq(P2), P3q = rq(P3);

    Tensor G1, G2, G3, Out;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    G1 = Tensor::from_host_fp16_on(Device::CUDA, P1h.data(), N, C1*HW);
    G2 = Tensor::from_host_fp16_on(Device::CUDA, P2h.data(), N, C2*HW);
    G3 = Tensor::from_host_fp16_on(Device::CUDA, P3h.data(), N, C3*HW);
    brotensor::concat_nchw_channels({&G1, &G2, &G3}, N, H, W,
                                    {C1, C2, C3}, Out);
    CHECK(Out.rows == N && Out.cols == total_C*HW && Out.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Out.size());
    Out.copy_to_host_fp16(got.data());
    brotensor::sync_all();

    // Reference: per-sample, lay channel blocks of P1 then P2 then P3.
    std::vector<float> Ref(N*total_C*HW);
    for (int n = 0; n < N; ++n) {
        int c_off = 0;
        auto copy_part = [&](const std::vector<float>& src, int Ci) {
            for (int c = 0; c < Ci; ++c)
                for (int p = 0; p < HW; ++p)
                    Ref[n*total_C*HW + (c_off + c)*HW + p] =
                        src[n*Ci*HW + c*HW + p];
            c_off += Ci;
        };
        copy_part(P1q, C1);
        copy_part(P2q, C2);
        copy_part(P3q, C3);
    }
    check_fp16(got, Ref, "concat_nchw_channels");
}

static void test_concat_nchw_channels_backward_fp16() {
    // Inverse round-trip: build a concatenated tensor (forward), upload a
    // random dY of matching shape, scatter via backward, verify each part
    // equals the channel slice of dY.
    std::printf("  concat_nchw_channels_backward fp16\n");
    const int N = 2, H = 3, W = 2;
    const int C1 = 1, C2 = 4, C3 = 2;
    const int total_C = C1 + C2 + C3;
    const int HW = H * W;
    std::mt19937 rng(0xDEADBEEFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> dY_host(N * total_C * HW);
    for (auto& v : dY_host) v = dist(rng);
    auto dY_q = rq(dY_host);
    auto dY_h = to_fp16(dY_host);
    Tensor dY;
    dY = Tensor::from_host_fp16_on(Device::CUDA, dY_h.data(), N, total_C * HW);

    Tensor dP1, dP2, dP3;
    brotensor::concat_nchw_channels_backward(
        dY, N, H, W, {C1, C2, C3}, {&dP1, &dP2, &dP3});
    CHECK(dP1.rows == N && dP1.cols == C1 * HW && dP1.dtype == Dtype::FP16);
    CHECK(dP2.rows == N && dP2.cols == C2 * HW && dP2.dtype == Dtype::FP16);
    CHECK(dP3.rows == N && dP3.cols == C3 * HW && dP3.dtype == Dtype::FP16);

    auto download = [&](const Tensor& g) {
        return g.to_host_vector_fp16();
    };
    auto got1 = download(dP1);
    auto got2 = download(dP2);
    auto got3 = download(dP3);
    brotensor::sync_all();

    auto build_ref = [&](int c_off, int Ci) {
        std::vector<float> ref(N * Ci * HW);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < Ci; ++c)
                for (int p = 0; p < HW; ++p)
                    ref[n * Ci * HW + c * HW + p] =
                        dY_q[n * total_C * HW + (c_off + c) * HW + p];
        return ref;
    };
    check_fp16(got1, build_ref(0,        C1), "concat_bwd part1");
    check_fp16(got2, build_ref(C1,       C2), "concat_bwd part2");
    check_fp16(got3, build_ref(C1 + C2,  C3), "concat_bwd part3");

    // Second shape: single big part (N=1, C=8, H=W=4).
    {
        const int N2 = 1, H2 = 4, W2 = 4, C = 8;
        std::vector<float> dy(N2 * C * H2 * W2);
        for (auto& v : dy) v = dist(rng);
        auto dyh = to_fp16(dy);
        Tensor DY;
        DY = Tensor::from_host_fp16_on(Device::CUDA, dyh.data(), N2, C * H2 * W2);
        Tensor P;
        brotensor::concat_nchw_channels_backward(
            DY, N2, H2, W2, {C}, {&P});
        CHECK(P.rows == N2 && P.cols == C * H2 * W2);
        auto got = download(P);
        brotensor::sync_all();
        auto ref = rq(dy);
        check_fp16(got, ref, "concat_bwd single");
    }
}

static void test_split_and_copy_d2d_fp16() {
    std::printf("  split_rows / copy_d2d fp16\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // split_rows_gpu inverse of concat_rows_gpu. Build a concatenated FP16
    // tensor and split it back into three pieces; values must round-trip.
    std::vector<float> P1(5), P2(11), P3(3);
    for (auto& v : P1) v = dist(rng);
    for (auto& v : P2) v = dist(rng);
    for (auto& v : P3) v = dist(rng);
    auto P1q = rq(P1), P2q = rq(P2), P3q = rq(P3);
    Tensor G1, G2, G3, Cat;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    G1 = Tensor::from_host_fp16_on(Device::CUDA, P1h.data(), 5, 1);
    G2 = Tensor::from_host_fp16_on(Device::CUDA, P2h.data(), 11, 1);
    G3 = Tensor::from_host_fp16_on(Device::CUDA, P3h.data(), 3, 1);
    brotensor::concat_rows({&G1, &G2, &G3}, Cat);

    Tensor S1 = Tensor::zeros_on(Device::CUDA, 5, 1, Dtype::FP16);
    Tensor S2 = Tensor::zeros_on(Device::CUDA, 11, 1, Dtype::FP16);
    Tensor S3 = Tensor::zeros_on(Device::CUDA, 3, 1, Dtype::FP16);
    brotensor::split_rows(Cat, {&S1, &S2, &S3});
    std::vector<uint16_t> g1(5), g2(11), g3(3);
    S1.copy_to_host_fp16(g1.data());
    S2.copy_to_host_fp16(g2.data());
    S3.copy_to_host_fp16(g3.data());
    brotensor::sync_all();
    check_fp16(g1, P1q, "split_rows[0]");
    check_fp16(g2, P2q, "split_rows[1]");
    check_fp16(g3, P3q, "split_rows[2]");

    // copy_d2d: copy a 4-element slice starting at offset 3 of Cat into a
    // fresh FP16 tensor at offset 1.
    Tensor Dst = Tensor::zeros_on(Device::CUDA, 8, 1, Dtype::FP16);
    Dst.zero();
    brotensor::copy_d2d(Cat, /*src_off*/3, Dst, /*dst_off*/1, /*n*/4);
    std::vector<uint16_t> got_dst(8);
    Dst.copy_to_host_fp16(got_dst.data());
    brotensor::sync_all();
    std::vector<float> ref_dst(8, 0.0f);
    // Cat[3..6] = (P1[3], P1[4], P2[0], P2[1])
    ref_dst[1] = P1q[3];
    ref_dst[2] = P1q[4];
    ref_dst[3] = P2q[0];
    ref_dst[4] = P2q[1];
    check_fp16(got_dst, ref_dst, "copy_d2d");
}

static void test_layernorm_fp16() {
    std::printf("  layernorm_forward_inference_batched_fp16\n");
    const int R = 4, D = 32;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(R*D), G(D), B(D);
    for (auto& v : X) v = dist(rng);
    for (auto& v : G) v = 0.5f + 0.5f * dist(rng);
    for (auto& v : B) v = dist(rng);
    auto Xq = rq(X), Gq = rq(G), Bq = rq(B);
    std::vector<float> Ref(R*D);
    const float eps = 1e-5f;
    for (int r = 0; r < R; ++r) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) s += Xq[r*D+j];
        const float mean = static_cast<float>(s / D);
        double sv = 0.0;
        for (int j = 0; j < D; ++j) { const double d = Xq[r*D+j] - mean; sv += d * d; }
        const float rstd = 1.0f / std::sqrt(static_cast<float>(sv/D) + eps);
        for (int j = 0; j < D; ++j) Ref[r*D+j] = (Xq[r*D+j] - mean) * rstd * Gq[j] + Bq[j];
    }
    Tensor Xg, Gg, Bg, Yg;
    auto Xh = to_fp16(X), Gh = to_fp16(G), Bh = to_fp16(B);
    Xg = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), R, D);
    Gg = Tensor::from_host_fp16_on(Device::CUDA, Gh.data(), 1, D);
    Bg = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), 1, D);
    brotensor::layernorm_forward_inference_batched_fp16(Xg, Gg, Bg, Yg, eps);
    CHECK(Yg.rows == R && Yg.cols == D && Yg.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Yg.size());
    Yg.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, Ref, "layernorm");
}

static void test_linear_backward_batched_fp16() {
    std::printf("  linear_backward_batched_fp16\n");
    // Two shapes — small and larger — to exercise both paths.
    for (auto shape : std::vector<std::array<int,3>>{ {3, 7, 5}, {8, 32, 16} }) {
        const int B = shape[0], in_dim = shape[1], out_dim = shape[2];
        std::mt19937 rng(static_cast<uint32_t>(B * 1009 + in_dim));
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> Wf(out_dim * in_dim), Xf(B * in_dim), dYf(B * out_dim);
        for (auto& v : Wf) v = dist(rng);
        for (auto& v : Xf) v = dist(rng);
        for (auto& v : dYf) v = dist(rng);
        auto Wq = rq(Wf), Xq = rq(Xf), dYq = rq(dYf);

        // CPU references in FP32.
        std::vector<float> dX_ref(B * in_dim, 0.0f),
                          dW_ref(out_dim * in_dim, 0.0f),
                          dB_ref(out_dim, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int j = 0; j < in_dim; ++j) {
                double a = 0.0;
                for (int i = 0; i < out_dim; ++i)
                    a += static_cast<double>(Wq[i*in_dim+j]) * dYq[b*out_dim+i];
                dX_ref[b*in_dim+j] = static_cast<float>(a);
            }
        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < in_dim; ++j) {
                double a = 0.0;
                for (int b = 0; b < B; ++b)
                    a += static_cast<double>(dYq[b*out_dim+i]) * Xq[b*in_dim+j];
                dW_ref[i*in_dim+j] = static_cast<float>(a);
            }
        for (int i = 0; i < out_dim; ++i) {
            double a = 0.0;
            for (int b = 0; b < B; ++b) a += dYq[b*out_dim+i];
            dB_ref[i] = static_cast<float>(a);
        }

        Tensor Wg, Xg, dYg, dXg, dWg, dBg;
        auto Wh = to_fp16(Wf), Xh = to_fp16(Xf), dYh = to_fp16(dYf);
        Wg  = Tensor::from_host_fp16_on(Device::CUDA, Wh.data(), out_dim, in_dim);
        Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), B, in_dim);
        dYg = Tensor::from_host_fp16_on(Device::CUDA, dYh.data(), B, out_dim);
        std::vector<uint16_t> z_dw(out_dim * in_dim,
                                   brotensor::fp32_to_fp16_bits(0.0f));
        std::vector<uint16_t> z_db(out_dim,
                                   brotensor::fp32_to_fp16_bits(0.0f));
        dWg = Tensor::from_host_fp16_on(Device::CUDA, z_dw.data(), out_dim, in_dim);
        dBg = Tensor::from_host_fp16_on(Device::CUDA, z_db.data(), out_dim, 1);

        brotensor::linear_backward_batched(Wg, Xg, dYg, dXg, dWg, dBg);
        CHECK(dXg.dtype == Dtype::FP16);
        CHECK(dWg.dtype == Dtype::FP16);
        CHECK(dBg.dtype == Dtype::FP16);

        std::vector<uint16_t> dx_got(dXg.size()), dw_got(dWg.size()), db_got(dBg.size());
        dXg.copy_to_host_fp16(dx_got.data());
        dWg.copy_to_host_fp16(dw_got.data());
        dBg.copy_to_host_fp16(db_got.data());
        brotensor::sync_all();
        std::printf("    shape B=%d in=%d out=%d\n", B, in_dim, out_dim);
        check_fp16(dx_got, dX_ref, "linbwd-dX");
        check_fp16(dw_got, dW_ref, "linbwd-dW");
        check_fp16(db_got, dB_ref, "linbwd-dB");
    }
}

static void test_layernorm_backward_fp16() {
    std::printf("  layernorm_backward_fp16\n");
    for (int n : {16, 64}) {
        std::mt19937 rng(static_cast<uint32_t>(0xABCD + n));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> X(n), G(n), Bf(n), dYf(n);
        for (auto& v : X)  v = dist(rng);
        for (auto& v : G)  v = 0.5f + 0.5f * dist(rng);
        for (auto& v : Bf) v = dist(rng);
        for (auto& v : dYf) v = dist(rng);
        auto Xq = rq(X), Gq = rq(G), Bq = rq(Bf), dYq = rq(dYf);

        // Forward in FP32 (round through fp16) to get xhat and rstd.
        const float eps = 1e-5f;
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += Xq[i];
        const float mean = static_cast<float>(s / n);
        double sv = 0.0;
        for (int i = 0; i < n; ++i) { double d = Xq[i] - mean; sv += d*d; }
        const float rstd = 1.0f / std::sqrt(static_cast<float>(sv/n) + eps);
        std::vector<float> xhat_ref(n);
        for (int i = 0; i < n; ++i) xhat_ref[i] = (Xq[i] - mean) * rstd;
        auto xhat_q = rq(xhat_ref);  // FP16 round-trip

        // CPU reference for backward (using FP16-rounded inputs).
        double sum_dxh = 0.0, sum_dxh_xhat = 0.0;
        for (int i = 0; i < n; ++i) {
            const double dxh = static_cast<double>(dYq[i]) * Gq[i];
            sum_dxh      += dxh;
            sum_dxh_xhat += dxh * xhat_q[i];
        }
        std::vector<float> dX_ref(n), dG_ref(n), dB_ref(n);
        const float nf = static_cast<float>(n);
        const float scale = rstd / nf;
        for (int i = 0; i < n; ++i) {
            const float dxh = dYq[i] * Gq[i];
            dX_ref[i] = scale * (nf * dxh
                                  - static_cast<float>(sum_dxh)
                                  - xhat_q[i] * static_cast<float>(sum_dxh_xhat));
            dG_ref[i] = dYq[i] * xhat_q[i];
            dB_ref[i] = dYq[i];
        }

        Tensor dYg, xhatg, Gg, dXg, dGg, dBg;
        auto dYh = to_fp16(dYf), xh = to_fp16(xhat_ref), Gh = to_fp16(G);
        dYg   = Tensor::from_host_fp16_on(Device::CUDA, dYh.data(), n, 1);
        xhatg = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), n, 1);
        Gg    = Tensor::from_host_fp16_on(Device::CUDA, Gh.data(), n, 1);
        std::vector<uint16_t> zeros(n, brotensor::fp32_to_fp16_bits(0.0f));
        dGg = Tensor::from_host_fp16_on(Device::CUDA, zeros.data(), n, 1);
        dBg = Tensor::from_host_fp16_on(Device::CUDA, zeros.data(), n, 1);

        brotensor::layernorm_backward(dYg, xhatg, Gg, rstd, dXg, dGg, dBg);
        CHECK(dXg.dtype == Dtype::FP16);
        CHECK(dGg.dtype == Dtype::FP16);
        CHECK(dBg.dtype == Dtype::FP16);

        std::vector<uint16_t> dx_got(n), dg_got(n), db_got(n);
        dXg.copy_to_host_fp16(dx_got.data());
        dGg.copy_to_host_fp16(dg_got.data());
        dBg.copy_to_host_fp16(db_got.data());
        brotensor::sync_all();
        std::printf("    shape n=%d\n", n);
        check_fp16(dx_got, dX_ref, "lnbwd-dX");
        check_fp16(dg_got, dG_ref, "lnbwd-dGamma");
        check_fp16(db_got, dB_ref, "lnbwd-dBeta");
    }
}

static void test_embedding_backward_fp16() {
    std::printf("  embedding_lookup_backward_fp16\n");
    for (auto shape : std::vector<std::array<int,3>>{ {5, 4, 8}, {10, 16, 32} }) {
        const int V = shape[0], B = shape[1], D = shape[2];
        std::mt19937 rng(static_cast<uint32_t>(V*131 + B*17 + D));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int>    idx_dist(0, V - 1);
        std::vector<int32_t> idx(B);
        for (auto& v : idx) v = idx_dist(rng);
        // Force at least one duplicate so the atomic-sum path exercises.
        if (B >= 2) idx[1] = idx[0];
        std::vector<float> dOut(B * D);
        for (auto& v : dOut) v = dist(rng);
        auto dOut_q = rq(dOut);

        std::vector<float> dT_ref(V * D, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int j = 0; j < D; ++j)
                dT_ref[idx[b]*D + j] += dOut_q[b*D + j];

        Tensor dOutg, dTg;
        auto dOut_h = to_fp16(dOut);
        dOutg = Tensor::from_host_fp16_on(Device::CUDA, dOut_h.data(), B, D);
        std::vector<uint16_t> zeros(V * D, brotensor::fp32_to_fp16_bits(0.0f));
        dTg = Tensor::from_host_fp16_on(Device::CUDA, zeros.data(), V, D);

        // Upload idx into a device int32 buffer (bit-cast through FP32 upload).
        std::vector<float> idx_as_float(B);
        for (int i = 0; i < B; ++i) {
            int32_t v = idx[i];
            std::memcpy(&idx_as_float[i], &v, sizeof(int32_t));
        }
        Tensor idx_buf =
            Tensor::from_host_on(Device::CUDA, idx_as_float.data(), B, 1);

        brotensor::embedding_lookup_backward(
            dOutg,
            reinterpret_cast<const int32_t*>(idx_buf.data),
            B, dTg);
        CHECK(dTg.dtype == Dtype::FP16);
        std::vector<uint16_t> got(V * D);
        dTg.copy_to_host_fp16(got.data());
        brotensor::sync_all();
        std::printf("    shape V=%d B=%d D=%d\n", V, B, D);
        check_fp16(got, dT_ref, "embbwd-dTable");
    }
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_fp16_basics\n");
    test_linear_fp16();
    test_linear_fp16_act();
    test_elementwise_fp16();
    test_concat_fp16();
    test_concat_nchw_channels_fp16();
    test_concat_nchw_channels_backward_fp16();
    test_split_and_copy_d2d_fp16();
    test_layernorm_fp16();
    test_linear_backward_batched_fp16();
    test_layernorm_backward_fp16();
    test_embedding_backward_fp16();
    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll FP16-basics checks passed.\n");
    return 0;
}
