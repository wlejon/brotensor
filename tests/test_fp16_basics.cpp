// CPU↔GPU parity for the FP16-gap basic ops: linear_forward_batched_fp16,
// add_inplace/scale_inplace/mul_inplace (FP16 dispatch), concat_rows /
// concat_batched_rows (FP16), and layernorm_forward_inference_batched_fp16.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

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

    GpuTensor W, Bb, X, Y;
    auto Wh = to_fp16(Wf), Bh = to_fp16(Bf), Xh = to_fp16(Xf);
    brotensor::upload_fp16(Wh.data(), out_dim, in_dim, W);
    brotensor::upload_fp16(Bh.data(), out_dim, 1, Bb);
    brotensor::upload_fp16(Xh.data(), B, in_dim, X);
    brotensor::linear_forward_batched_fp16_gpu(W, &Bb, X, Y);
    CHECK(Y.rows == B && Y.cols == out_dim && Y.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Y.size());
    brotensor::download_fp16(Y, got.data());
    brotensor::cuda_sync();
    check_fp16(got, Ref, "linear");

    // No-bias variant.
    GpuTensor Y2;
    brotensor::linear_forward_batched_fp16_gpu(W, nullptr, X, Y2);
    std::vector<uint16_t> got2(Y2.size());
    brotensor::download_fp16(Y2, got2.data());
    brotensor::cuda_sync();
    std::vector<float> Ref2 = Ref;
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out_dim; ++o) Ref2[b*out_dim+o] -= Bq[o];
    check_fp16(got2, Ref2, "linear-nobias");
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
    GpuTensor Ag, Bg;
    auto Ah = to_fp16(A), Bh = to_fp16(Bv);
    brotensor::upload_fp16(Ah.data(), N, 1, Ag);
    brotensor::upload_fp16(Bh.data(), N, 1, Bg);
    brotensor::add_inplace_gpu(Ag, Bg);
    std::vector<uint16_t> got(N);
    brotensor::download_fp16(Ag, got.data());
    brotensor::cuda_sync();
    std::vector<float> Ref(N);
    for (int i = 0; i < N; ++i) Ref[i] = Aq[i] + Bq[i];
    check_fp16(got, Ref, "add_inplace");

    // scale (re-upload)
    brotensor::upload_fp16(Ah.data(), N, 1, Ag);
    brotensor::scale_inplace_gpu(Ag, 0.25f);
    brotensor::download_fp16(Ag, got.data());
    brotensor::cuda_sync();
    for (int i = 0; i < N; ++i) Ref[i] = Aq[i] * 0.25f;
    check_fp16(got, Ref, "scale_inplace");

    // mul
    brotensor::upload_fp16(Ah.data(), N, 1, Ag);
    brotensor::mul_inplace_gpu(Ag, Bg);
    brotensor::download_fp16(Ag, got.data());
    brotensor::cuda_sync();
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

    GpuTensor G1, G2, G3, Out;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    brotensor::upload_fp16(P1h.data(), 13, 1, G1);
    brotensor::upload_fp16(P2h.data(), 7, 1, G2);
    brotensor::upload_fp16(P3h.data(), 4, 1, G3);
    brotensor::concat_rows_gpu({&G1, &G2, &G3}, Out);
    CHECK(Out.rows == 24 && Out.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Out.size());
    brotensor::download_fp16(Out, got.data());
    brotensor::cuda_sync();
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
    GpuTensor Ag, Bg, OutB;
    auto Ah = to_fp16(A), Bh = to_fp16(Bv);
    brotensor::upload_fp16(Ah.data(), 3, 4, Ag);
    brotensor::upload_fp16(Bh.data(), 3, 2, Bg);
    brotensor::concat_batched_rows_gpu({&Ag, &Bg}, OutB);
    CHECK(OutB.rows == 3 && OutB.cols == 6 && OutB.dtype == Dtype::FP16);
    std::vector<uint16_t> gotB(OutB.size());
    brotensor::download_fp16(OutB, gotB.data());
    brotensor::cuda_sync();
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

    GpuTensor G1, G2, G3, Out;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    brotensor::upload_fp16(P1h.data(), N, C1*HW, G1);
    brotensor::upload_fp16(P2h.data(), N, C2*HW, G2);
    brotensor::upload_fp16(P3h.data(), N, C3*HW, G3);
    brotensor::concat_nchw_channels_gpu({&G1, &G2, &G3}, N, H, W,
                                        {C1, C2, C3}, Out);
    CHECK(Out.rows == N && Out.cols == total_C*HW && Out.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Out.size());
    brotensor::download_fp16(Out, got.data());
    brotensor::cuda_sync();

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
    GpuTensor G1, G2, G3, Cat;
    auto P1h = to_fp16(P1), P2h = to_fp16(P2), P3h = to_fp16(P3);
    brotensor::upload_fp16(P1h.data(), 5, 1, G1);
    brotensor::upload_fp16(P2h.data(), 11, 1, G2);
    brotensor::upload_fp16(P3h.data(), 3, 1, G3);
    brotensor::concat_rows_gpu({&G1, &G2, &G3}, Cat);

    GpuTensor S1(5, 1, Dtype::FP16), S2(11, 1, Dtype::FP16), S3(3, 1, Dtype::FP16);
    brotensor::split_rows_gpu(Cat, {&S1, &S2, &S3});
    std::vector<uint16_t> g1(5), g2(11), g3(3);
    brotensor::download_fp16(S1, g1.data());
    brotensor::download_fp16(S2, g2.data());
    brotensor::download_fp16(S3, g3.data());
    brotensor::cuda_sync();
    check_fp16(g1, P1q, "split_rows[0]");
    check_fp16(g2, P2q, "split_rows[1]");
    check_fp16(g3, P3q, "split_rows[2]");

    // copy_d2d_gpu: copy a 4-element slice starting at offset 3 of Cat into a
    // fresh FP16 tensor at offset 1.
    GpuTensor Dst(8, 1, Dtype::FP16);
    Dst.zero();
    brotensor::copy_d2d_gpu(Cat, /*src_off*/3, Dst, /*dst_off*/1, /*n*/4);
    std::vector<uint16_t> got_dst(8);
    brotensor::download_fp16(Dst, got_dst.data());
    brotensor::cuda_sync();
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
    GpuTensor Xg, Gg, Bg, Yg;
    auto Xh = to_fp16(X), Gh = to_fp16(G), Bh = to_fp16(B);
    brotensor::upload_fp16(Xh.data(), R, D, Xg);
    brotensor::upload_fp16(Gh.data(), 1, D, Gg);
    brotensor::upload_fp16(Bh.data(), 1, D, Bg);
    brotensor::layernorm_forward_inference_batched_fp16_gpu(Xg, Gg, Bg, Yg, eps);
    CHECK(Yg.rows == R && Yg.cols == D && Yg.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Yg.size());
    brotensor::download_fp16(Yg, got.data());
    brotensor::cuda_sync();
    check_fp16(got, Ref, "layernorm");
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_fp16_basics\n");
    test_linear_fp16();
    test_elementwise_fp16();
    test_concat_fp16();
    test_concat_nchw_channels_fp16();
    test_split_and_copy_d2d_fp16();
    test_layernorm_fp16();
    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll FP16-basics checks passed.\n");
    return 0;
}
