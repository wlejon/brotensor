// rope_apply_perhead: per-head cos/sin RoPE.
//
// Covers: CPU FP32 exactness vs a hand reference; CUDA FP32/FP16 parity; and the
// reduction property that with tables shared across heads, rope_apply_perhead
// equals rope_apply (cross-check against the existing op).

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

// Reference: per-head rotation. cos/sin are (L*nh, hd/2), row-major, head-minor.
static void perhead_ref(const std::vector<float>& X,
                        const std::vector<float>& C,
                        const std::vector<float>& S,
                        std::vector<float>& Y,
                        int L, int nh, int hd) {
    const int D = nh * hd, half = hd / 2;
    Y.assign(static_cast<size_t>(L) * D, 0.0f);
    for (int row = 0; row < L; ++row)
        for (int h = 0; h < nh; ++h) {
            const int off = row * D + h * hd;
            const int tbl = (row * nh + h) * half;
            for (int i = 0; i < half; ++i) {
                const float c = C[tbl + i], s = S[tbl + i];
                const float x0 = X[off + 2 * i], x1 = X[off + 2 * i + 1];
                Y[off + 2 * i]     = x0 * c - x1 * s;
                Y[off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
}

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    brotensor::init();
    std::printf("test_rope_perhead\n");

    const int L = 5, nh = 3, hd = 8, half = hd / 2;
    std::mt19937 rng(0x51);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(static_cast<size_t>(L) * nh * hd);
    for (auto& v : X) v = dist(rng);
    // Distinct per-(row,head,pair) angles so each head genuinely differs.
    std::vector<float> ang(static_cast<size_t>(L) * nh * half), C(ang.size()), S(ang.size());
    for (size_t i = 0; i < ang.size(); ++i) { ang[i] = dist(rng) * 3.0f; C[i] = std::cos(ang[i]); S[i] = std::sin(ang[i]); }

    std::vector<float> ref;
    perhead_ref(X, C, S, ref, L, nh, hd);

    // ── CPU FP32 exactness ────────────────────────────────────────────────
    {
        Tensor Xc = Tensor::from_host_on(Device::CPU, X.data(), L, nh * hd);
        Tensor Cc = Tensor::from_host_on(Device::CPU, C.data(), L * nh, half);
        Tensor Sc = Tensor::from_host_on(Device::CPU, S.data(), L * nh, half);
        Tensor Yc;
        brotensor::rope_apply_perhead(Xc, Cc, Sc, hd, nh, Yc);
        brotensor::sync_all();
        std::vector<float> got(Yc.size());
        Yc.copy_to_host(got.data());
        float e = max_abs_diff(got, ref);
        std::printf("  cpu fp32 max_err=%g\n", e);
        CHECK(e < 1e-5f);
    }

    // ── reduction: tables shared across heads == rope_apply ───────────────
    {
        std::vector<float> Csh(static_cast<size_t>(L) * half), Ssh(Csh.size());
        for (size_t i = 0; i < Csh.size(); ++i) { Csh[i] = std::cos(0.7f * (i + 1)); Ssh[i] = std::sin(0.7f * (i + 1)); }
        // Per-head table = shared table broadcast over heads.
        std::vector<float> Cph(static_cast<size_t>(L) * nh * half), Sph(Cph.size());
        for (int row = 0; row < L; ++row)
            for (int h = 0; h < nh; ++h)
                for (int i = 0; i < half; ++i) {
                    Cph[(row * nh + h) * half + i] = Csh[row * half + i];
                    Sph[(row * nh + h) * half + i] = Ssh[row * half + i];
                }
        Tensor Xc = Tensor::from_host_on(Device::CPU, X.data(), L, nh * hd);
        Tensor Ya, Yp;
        brotensor::rope_apply(Xc,
            Tensor::from_host_on(Device::CPU, Csh.data(), L, half),
            Tensor::from_host_on(Device::CPU, Ssh.data(), L, half), hd, nh, Ya);
        brotensor::rope_apply_perhead(Xc,
            Tensor::from_host_on(Device::CPU, Cph.data(), L * nh, half),
            Tensor::from_host_on(Device::CPU, Sph.data(), L * nh, half), hd, nh, Yp);
        brotensor::sync_all();
        std::vector<float> a(Ya.size()), p(Yp.size());
        Ya.copy_to_host(a.data()); Yp.copy_to_host(p.data());
        float e = max_abs_diff(a, p);
        std::printf("  reduction (shared==rope_apply) max_err=%g\n", e);
        CHECK(e < 1e-6f);
    }

    // ── CUDA FP32 + FP16 parity ───────────────────────────────────────────
    if (brotensor::is_available(Device::CUDA)) {
        Tensor Cg = Tensor::from_host_on(Device::CUDA, C.data(), L * nh, half);
        Tensor Sg = Tensor::from_host_on(Device::CUDA, S.data(), L * nh, half);
        {
            Tensor Xg = Tensor::from_host_on(Device::CUDA, X.data(), L, nh * hd);
            Tensor Yg;
            brotensor::rope_apply_perhead(Xg, Cg, Sg, hd, nh, Yg);
            brotensor::sync_all();
            std::vector<float> got(Yg.size());
            Yg.copy_to_host(got.data());
            float e = max_abs_diff(got, ref);
            std::printf("  cuda fp32 max_err=%g\n", e);
            CHECK(e < 1e-4f);
        }
        {
            auto Xh = to_fp16(X);
            Tensor Xg = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), L, nh * hd);
            Tensor Yg;
            brotensor::rope_apply_perhead(Xg, Cg, Sg, hd, nh, Yg);
            brotensor::sync_all();
            std::vector<uint16_t> got(Yg.size());
            Yg.copy_to_host_fp16(got.data());
            std::vector<float> gotf(got.size());
            for (size_t i = 0; i < got.size(); ++i) gotf[i] = brotensor::fp16_bits_to_fp32(got[i]);
            float e = max_abs_diff(gotf, ref);
            std::printf("  cuda fp16 max_err=%g\n", e);
            CHECK(e < 3e-2f);
        }
    } else {
        std::printf("  CUDA not available - GPU parity skipped\n");
    }

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
