// Tests for cross_attention_forward_with_attn: FP16 cross-attention with
// head-averaged attention map output and optional pre-softmax logit bias.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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
static std::vector<float> from_fp16(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp16_bits_to_fp32(v[i]);
    return o;
}

static float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float e = std::fabs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

// Helper: download an FP16 Tensor to float vector.
static std::vector<float> dl_fp16(const Tensor& g) {
    std::vector<uint16_t> v(static_cast<size_t>(g.size()), 0);
    g.copy_to_host_fp16(v.data());
    return from_fp16(v);
}

// Random input setup shared by tests.
struct AttnInputs {
    int Lq, Lk, D, nh;
    std::vector<uint16_t> X, Ctx, Wq, Wk, Wv, Wo;
};

static AttnInputs make_inputs(int Lq, int Lk, int D, int nh, uint32_t seed) {
    AttnInputs in{Lq, Lk, D, nh, {}, {}, {}, {}, {}, {}};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq * D), Ctx(Lk * D);
    std::vector<float> Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D);
    for (auto& v : X)   v = dist(rng);
    for (auto& v : Ctx) v = dist(rng);
    for (auto& v : Wq)  v = dist(rng);
    for (auto& v : Wk)  v = dist(rng);
    for (auto& v : Wv)  v = dist(rng);
    for (auto& v : Wo)  v = dist(rng);
    in.X   = to_fp16(X);
    in.Ctx = to_fp16(Ctx);
    in.Wq  = to_fp16(Wq);
    in.Wk  = to_fp16(Wk);
    in.Wv  = to_fp16(Wv);
    in.Wo  = to_fp16(Wo);
    return in;
}

static void upload_inputs(const AttnInputs& in,
                          Tensor& Xg, Tensor& Cg,
                          Tensor& Wqg, Tensor& Wkg,
                          Tensor& Wvg, Tensor& Wog) {
    Xg  = Tensor::from_host_fp16_on(Device::CUDA, in.X.data(),   in.Lq, in.D);
    Cg  = Tensor::from_host_fp16_on(Device::CUDA, in.Ctx.data(), in.Lk, in.D);
    Wqg = Tensor::from_host_fp16_on(Device::CUDA, in.Wq.data(),  in.D,  in.D);
    Wkg = Tensor::from_host_fp16_on(Device::CUDA, in.Wk.data(),  in.D,  in.D);
    Wvg = Tensor::from_host_fp16_on(Device::CUDA, in.Wv.data(),  in.D,  in.D);
    Wog = Tensor::from_host_fp16_on(Device::CUDA, in.Wo.data(),  in.D,  in.D);
}

// Test 1: parity vs existing FP16 cross_attention_forward (no bias, no mask).
static void test_parity_no_bias_no_mask() {
    const int Lq = 64, Lk = 16, D = 64, nh = 4;
    std::printf("  parity vs cross_attention_forward  Lq=%d Lk=%d D=%d nh=%d\n",
                Lq, Lk, D, nh);
    auto in = make_inputs(Lq, Lk, D, nh, 0xC0DEu);

    Tensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    upload_inputs(in, Xg, Cg, Wqg, Wkg, Wvg, Wog);

    Tensor O_ref;
    brotensor::cross_attention_forward(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                       nullptr, nh, O_ref);

    Tensor O, AttnAvg;
    brotensor::cross_attention_forward_with_attn(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 nullptr, nullptr, nh,
                                                 O, AttnAvg);
    brotensor::sync_all();

    CHECK(O.rows == Lq && O.cols == D && O.dtype == Dtype::FP16);
    CHECK(AttnAvg.rows == Lq && AttnAvg.cols == Lk && AttnAvg.dtype == Dtype::FP16);

    auto vO = dl_fp16(O);
    auto vR = dl_fp16(O_ref);
    const float me = max_abs_err(vO, vR);
    std::printf("    O max_err=%g\n", me);
    CHECK(me < 1e-2f);
}

// CPU reference for the head-averaged attention map. Uses FP32 throughout
// against requantised FP32 weights so the comparison is fair.
static void attn_avg_cpu(const std::vector<float>& X,
                         const std::vector<float>& Ctx,
                         const std::vector<float>& Wq,
                         const std::vector<float>& Wk,
                         const std::vector<float>* bias,  // (Lq, Lk) or null
                         const std::vector<float>* mask,  // (Lk,) or null
                         int Lq, int Lk, int D, int nh,
                         std::vector<float>& AttnAvg) {
    const int hd = D / nh;
    std::vector<float> Q(Lq * D, 0.0f), K(Lk * D, 0.0f);
    for (int i = 0; i < Lq; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += static_cast<double>(X[i * D + k]) * Wq[j * D + k];
            Q[i * D + j] = static_cast<float>(s);
        }
    for (int i = 0; i < Lk; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += static_cast<double>(Ctx[i * D + k]) * Wk[j * D + k];
            K[i * D + j] = static_cast<float>(s);
        }
    AttnAvg.assign(static_cast<size_t>(Lq) * Lk, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> sc(Lk);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float max_v = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q * D + off + d]) * K[k * D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                if (bias) s += (*bias)[q * Lk + k];
                if (mask && (*mask)[k] <= 0.5f) s = -1e30f;
                sc[k] = s;
                if (s > max_v) max_v = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) { sc[k] = std::exp(sc[k] - max_v); sum += sc[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) {
                AttnAvg[q * Lk + k] += sc[k] * inv;
            }
        }
        for (int k = 0; k < Lk; ++k) AttnAvg[q * Lk + k] /= static_cast<float>(nh);
    }
}

static std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

static void test_attn_avg_correctness() {
    const int Lq = 32, Lk = 16, D = 32, nh = 4;
    std::printf("  AttnAvg correctness  Lq=%d Lk=%d D=%d nh=%d\n", Lq, Lk, D, nh);
    std::mt19937 rng(0xBABEu);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq * D), Ctx(Lk * D);
    std::vector<float> Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D);
    for (auto& v : X)   v = dist(rng);
    for (auto& v : Ctx) v = dist(rng);
    for (auto& v : Wq)  v = dist(rng);
    for (auto& v : Wk)  v = dist(rng);
    for (auto& v : Wv)  v = dist(rng);
    for (auto& v : Wo)  v = dist(rng);

    auto X_q   = rq(X), Ctx_q = rq(Ctx);
    auto Wq_q  = rq(Wq), Wk_q  = rq(Wk);

    std::vector<float> AttnAvg_ref;
    attn_avg_cpu(X_q, Ctx_q, Wq_q, Wk_q, nullptr, nullptr,
                 Lq, Lk, D, nh, AttnAvg_ref);

    auto Xh  = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk),
         Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    Tensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(),  Lk, D);
    Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D,  D);
    Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D,  D);
    Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D,  D);
    Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D,  D);

    Tensor O, AttnAvg;
    brotensor::cross_attention_forward_with_attn(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 nullptr, nullptr, nh,
                                                 O, AttnAvg);
    brotensor::sync_all();
    auto got = dl_fp16(AttnAvg);
    const float me = max_abs_err(got, AttnAvg_ref);
    std::printf("    AttnAvg max_err=%g\n", me);
    CHECK(me < 1e-2f);
}

static void test_bias_injection() {
    const int Lq = 16, Lk = 12, D = 32, nh = 4;
    std::printf("  bias injection (masks token 0)  Lq=%d Lk=%d D=%d nh=%d\n",
                Lq, Lk, D, nh);
    auto in = make_inputs(Lq, Lk, D, nh, 0xFEEDu);
    Tensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    upload_inputs(in, Xg, Cg, Wqg, Wkg, Wvg, Wog);

    std::vector<float> bias(Lq * Lk, 0.0f);
    for (int q = 0; q < Lq; ++q) bias[q * Lk + 0] = -1e4f;  // suppress key 0
    Tensor bias_g = Tensor::from_host_on(Device::CUDA, bias.data(), Lq, Lk);

    Tensor O, AttnAvg;
    brotensor::cross_attention_forward_with_attn(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 nullptr, &bias_g, nh,
                                                 O, AttnAvg);
    brotensor::sync_all();
    auto a = dl_fp16(AttnAvg);

    float max_col0 = 0.0f;
    float min_rowsum = 1e30f, max_rowsum = -1e30f;
    for (int q = 0; q < Lq; ++q) {
        const float c0 = std::fabs(a[q * Lk + 0]);
        if (c0 > max_col0) max_col0 = c0;
        float s = 0.0f;
        for (int k = 0; k < Lk; ++k) s += a[q * Lk + k];
        if (s < min_rowsum) min_rowsum = s;
        if (s > max_rowsum) max_rowsum = s;
    }
    std::printf("    max |AttnAvg[:, 0]|=%g  row-sum range=[%g, %g]\n",
                max_col0, min_rowsum, max_rowsum);
    CHECK(max_col0 < 1e-3f);
    CHECK(min_rowsum > 0.99f && max_rowsum < 1.01f);
}

static void test_mask_compat() {
    const int Lq = 16, Lk = 12, D = 32, nh = 4;
    std::printf("  d_mask compat (masks last key)  Lq=%d Lk=%d D=%d nh=%d\n",
                Lq, Lk, D, nh);
    auto in = make_inputs(Lq, Lk, D, nh, 0xACE1u);
    Tensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    upload_inputs(in, Xg, Cg, Wqg, Wkg, Wvg, Wog);

    std::vector<float> mask(Lk, 1.0f);
    mask[Lk - 1] = 0.0f;
    Tensor mg = Tensor::from_host_on(Device::CUDA, mask.data(), Lk, 1);

    Tensor O, AttnAvg;
    brotensor::cross_attention_forward_with_attn(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 static_cast<const float*>(mg.data),
                                                 nullptr, nh,
                                                 O, AttnAvg);
    brotensor::sync_all();
    auto a = dl_fp16(AttnAvg);

    float max_last = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        const float c = std::fabs(a[q * Lk + (Lk - 1)]);
        if (c > max_last) max_last = c;
    }
    std::printf("    max |AttnAvg[:, Lk-1]|=%g\n", max_last);
    CHECK(max_last < 1e-3f);
}

static void test_sdxl_shape_runs() {
    const int Lq = 1024, Lk = 77, D = 640, nh = 10;
    std::printf("  SDXL-realistic shape  Lq=%d Lk=%d D=%d nh=%d\n",
                Lq, Lk, D, nh);
    auto in = make_inputs(Lq, Lk, D, nh, 0xBEEFu);
    Tensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    upload_inputs(in, Xg, Cg, Wqg, Wkg, Wvg, Wog);

    Tensor O, AttnAvg;
    brotensor::cross_attention_forward_with_attn(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 nullptr, nullptr, nh,
                                                 O, AttnAvg);
    brotensor::sync_all();
    CHECK(O.rows == Lq && O.cols == D && O.dtype == Dtype::FP16);
    CHECK(AttnAvg.rows == Lq && AttnAvg.cols == Lk && AttnAvg.dtype == Dtype::FP16);

    auto a = dl_fp16(AttnAvg);
    int bad_finite = 0;
    int bad_rowsum = 0;
    for (int q = 0; q < Lq; ++q) {
        float s = 0.0f;
        for (int k = 0; k < Lk; ++k) {
            float v = a[q * Lk + k];
            if (!std::isfinite(v)) ++bad_finite;
            s += v;
        }
        if (!(s > 0.95f && s < 1.05f)) ++bad_rowsum;
    }
    auto vO = dl_fp16(O);
    int bad_O = 0;
    for (float v : vO) if (!std::isfinite(v)) ++bad_O;
    std::printf("    bad_finite=%d bad_rowsum=%d bad_O=%d\n",
                bad_finite, bad_rowsum, bad_O);
    CHECK(bad_finite == 0);
    CHECK(bad_rowsum == 0);
    CHECK(bad_O == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_cross_attention_with_attn\n");

    test_parity_no_bias_no_mask();
    test_attn_avg_correctness();
    test_bias_injection();
    test_mask_compat();
    test_sdxl_shape_runs();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll cross_attention_with_attn checks passed.\n");
    return 0;
}
