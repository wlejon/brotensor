// Parity for resblock_forward_int8w_fp16 against the FP16 fused
// resblock_forward using host-dequantised weights as the reference.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
static inline void cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n);
}
#define cudaMemcpyHostToDevice 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void download_to_f(const Tensor& g, std::vector<float>& out) {
    std::vector<uint16_t> tmp(g.size());
    g.copy_to_host_fp16(tmp.data());
    brotensor::sync_all();
    out.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) out[i] = brotensor::fp16_bits_to_fp32(tmp[i]);
}

// Host-quantise (out, in) FP16 weights to INT8 + per-row FP32 scales, and
// produce the corresponding FP16 dequantised view for use as the reference.
static void quantize_and_dequant(const std::vector<uint16_t>& W_h,
                                 int out, int in,
                                 std::vector<int8_t>& W_q,
                                 std::vector<float>& scales,
                                 std::vector<uint16_t>& W_deq) {
    W_q.resize(out * in);
    scales.resize(out);
    brotensor::quantize_int8_per_row_host(W_h.data(), out, in,
                                          W_q.data(), scales.data());
    W_deq.resize(out * in);
    for (int r = 0; r < out; ++r) {
        const float s = scales[r];
        for (int c = 0; c < in; ++c) {
            const float v = static_cast<float>(W_q[r * in + c]) * s;
            W_deq[r * in + c] = brotensor::fp32_to_fp16_bits(v);
        }
    }
}

struct Case {
    const char* label;
    int N, C_in, C_out, H, W, num_groups;
    bool with_temb;
    bool temb_per_N;     // true: (N, C_out); false: (C_out, 1)
    bool need_skip_conv;
    bool no_bias_convs;  // b1 == b2 == null
};

static float run_case(const Case& tc) {
    std::printf("  %s  N=%d Cin=%d Cout=%d H=%d W=%d groups=%d temb=%d temb_per_N=%d skip_conv=%d no_bias=%d\n",
                tc.label, tc.N, tc.C_in, tc.C_out, tc.H, tc.W,
                tc.num_groups, (int)tc.with_temb, (int)tc.temb_per_N,
                (int)tc.need_skip_conv, (int)tc.no_bias_convs);
    std::mt19937 rng(0x9E37 ^ (tc.C_in * 131u + tc.C_out * 17u + (unsigned)tc.H));
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto rand = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };
    const int N = tc.N, C_in = tc.C_in, C_out = tc.C_out, H = tc.H, Wd = tc.W;
    const int spatial = H * Wd;

    auto Xf  = rand(N * C_in * spatial);
    auto g1f = rand(C_in), b1f = rand(C_in);
    auto W1f = rand(C_out * C_in * 9);
    auto bc1f = rand(C_out);
    auto g2f = rand(C_out), b2f = rand(C_out);
    auto W2f = rand(C_out * C_out * 9);
    auto bc2f = rand(C_out);
    auto Wskf = tc.need_skip_conv ? rand(C_out * C_in) : std::vector<float>{};
    auto bskf = tc.need_skip_conv ? rand(C_out) : std::vector<float>{};
    std::vector<float> tembf;
    int temb_rows = 0, temb_cols = 0;
    if (tc.with_temb) {
        if (tc.temb_per_N) {
            tembf = rand(N * C_out);
            temb_rows = N; temb_cols = C_out;
        } else {
            tembf = rand(C_out);
            temb_rows = C_out; temb_cols = 1;
        }
    }

    auto Xh  = to_fp16(Xf);
    auto g1h = to_fp16(g1f), b1h = to_fp16(b1f);
    auto W1h = to_fp16(W1f);
    auto bc1h = to_fp16(bc1f);
    auto g2h = to_fp16(g2f), b2h = to_fp16(b2f);
    auto W2h = to_fp16(W2f);
    auto bc2h = to_fp16(bc2f);
    auto Wskh = tc.need_skip_conv ? to_fp16(Wskf) : std::vector<uint16_t>{};
    auto bskh = tc.need_skip_conv ? to_fp16(bskf) : std::vector<uint16_t>{};
    auto tembh = tc.with_temb ? to_fp16(tembf) : std::vector<uint16_t>{};

    // Host-quantise W1, W2, [Wskip] → INT8 + scales, and matching FP16 dequant
    // versions for the reference FP16 fused op.
    std::vector<int8_t> W1q, W2q, Wskq;
    std::vector<float>  s1v, s2v, sskv;
    std::vector<uint16_t> W1deq, W2deq, Wskdeq;
    quantize_and_dequant(W1h, C_out, C_in * 9, W1q, s1v, W1deq);
    quantize_and_dequant(W2h, C_out, C_out * 9, W2q, s2v, W2deq);
    if (tc.need_skip_conv) {
        quantize_and_dequant(Wskh, C_out, C_in, Wskq, sskv, Wskdeq);
    }

    auto up_fp16 = [&](const std::vector<uint16_t>& v, int r, int c) {
        return Tensor::from_host_fp16_on(Device::CUDA, v.data(), r, c);
    };

    Tensor Xg  = up_fp16(Xh,  N,    C_in*spatial);
    Tensor g1g = up_fp16(g1h, C_in, 1);
    Tensor b1g = up_fp16(b1h, C_in, 1);
    Tensor g2g = up_fp16(g2h, C_out,1);
    Tensor b2g = up_fp16(b2h, C_out,1);

    Tensor bc1g, bc2g, bskg, tembg;
    if (!tc.no_bias_convs) {
        bc1g = up_fp16(bc1h, C_out, 1);
        bc2g = up_fp16(bc2h, C_out, 1);
    }
    if (tc.need_skip_conv) bskg = up_fp16(bskh, C_out, 1);
    if (tc.with_temb) tembg = up_fp16(tembh, temb_rows, temb_cols);

    // FP16 dequant tensors for reference op.
    Tensor W1deq_g = up_fp16(W1deq, C_out, C_in * 9);
    Tensor W2deq_g = up_fp16(W2deq, C_out, C_out * 9);
    Tensor Wskdeq_g;
    if (tc.need_skip_conv) Wskdeq_g = up_fp16(Wskdeq, C_out, C_in);

    // INT8 weight + FP32 scale tensors for the new op.
    Tensor W1int8_g = Tensor::zeros_on(Device::CUDA, C_out, C_in * 9, Dtype::INT8);
    Tensor W2int8_g = Tensor::zeros_on(Device::CUDA, C_out, C_out * 9, Dtype::INT8);
    cudaMemcpy(W1int8_g.data, W1q.data(), W1q.size() * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(W2int8_g.data, W2q.data(), W2q.size() * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    Tensor s1g = Tensor::from_host_on(Device::CUDA, s1v.data(), C_out, 1);
    Tensor s2g = Tensor::from_host_on(Device::CUDA, s2v.data(), C_out, 1);

    Tensor Wskint8_g, sskg;
    if (tc.need_skip_conv) {
        Wskint8_g = Tensor::zeros_on(Device::CUDA, C_out, C_in, Dtype::INT8);
        cudaMemcpy(Wskint8_g.data, Wskq.data(), Wskq.size() * sizeof(int8_t),
                   cudaMemcpyHostToDevice);
        sskg = Tensor::from_host_on(Device::CUDA, sskv.data(), C_out, 1);
    }

    const Tensor* b1p = tc.no_bias_convs ? nullptr : &bc1g;
    const Tensor* b2p = tc.no_bias_convs ? nullptr : &bc2g;
    const Tensor* tembp = tc.with_temb ? &tembg : nullptr;
    const Tensor* Wskdeq_p = tc.need_skip_conv ? &Wskdeq_g : nullptr;
    const Tensor* Wskint8_p = tc.need_skip_conv ? &Wskint8_g : nullptr;
    const Tensor* sskp = tc.need_skip_conv ? &sskg : nullptr;
    const Tensor* bskp = tc.need_skip_conv ? &bskg : nullptr;

    // Reference: FP16 fused op with dequantised weights.
    Tensor Y_ref;
    brotensor::resblock_forward(Xg, g1g, b1g, W1deq_g, b1p,
                                tembp,
                                g2g, b2g, W2deq_g, b2p,
                                Wskdeq_p, bskp,
                                N, C_in, C_out, H, Wd,
                                tc.num_groups, 1e-5f, Y_ref);

    // INT8W path.
    Tensor Y_int8w;
    brotensor::resblock_forward_int8w_fp16(
        Xg, g1g, b1g, W1int8_g, s1g, b1p,
        tembp,
        g2g, b2g, W2int8_g, s2g, b2p,
        Wskint8_p, sskp, bskp,
        N, C_in, C_out, H, Wd,
        tc.num_groups, 1e-5f, Y_int8w);

    CHECK(Y_int8w.rows == N && Y_int8w.cols == C_out * spatial &&
          Y_int8w.dtype == Dtype::FP16);

    std::vector<float> ref_f, got_f;
    download_to_f(Y_ref, ref_f);
    download_to_f(Y_int8w, got_f);
    float max_err = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < ref_f.size(); ++i) {
        const float e = std::fabs(got_f[i] - ref_f[i]);
        if (e > max_err) max_err = e;
        if (e > 1.5e-2f + 1.5e-2f * std::fabs(ref_f[i])) {
            if (bad < 3) std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                                     i, got_f[i], ref_f[i], e);
            ++bad;
        }
    }
    std::printf("    max_err=%g bad=%d / %zu\n", max_err, bad, ref_f.size());
    CHECK(max_err < 1.5e-2f);
    return max_err;
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_resblock_int8w\n");

    Case cases[] = {
        // Identity skip, no t_emb, with biases.
        {"identity-no-temb",   1, 320,  320, 32, 32, 32, false, false, false, false},
        // Identity skip with t_emb (N, C_out).
        {"identity-temb-NC",   1, 640,  640, 16, 16, 32, true,  true,  false, false},
        // Identity skip with t_emb (C_out,) and no conv biases.
        {"identity-temb-C-nobias", 1, 320, 320, 16, 16, 32, true, false, false, true},
        // Wskip 1x1 up-channels with t_emb.
        {"upskip-temb",        1, 320,  640, 16, 16, 32, true,  true,  true,  false},
        // Wskip 1x1 large channels small spatial.
        {"upskip-large",       1, 640, 1280,  8,  8, 32, true,  true,  true,  false},
    };

    for (const auto& c : cases) {
        run_case(c);
    }

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll resblock_int8w checks passed.\n");
    return 0;
}
