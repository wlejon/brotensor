// Parity for flash_attention_backward_gpu (bare attention core, FP16) against
// an analytic CPU reference. Covers unmasked + masked, causal=false +
// causal=true, num_heads in {1, 2}.
//
// We use a CPU analytic reference rather than the qkvo cross-check because
// qkvo couples K and V to a single Ctx tensor — there's no clean way to feed
// independent Q, K, V through it. The analytic backward is the same math the
// qkvo backward implements internally (per-head dV = P^T·dO, dP = dO·V^T,
// D_q reduction, dS = P*(dP-D_q)*inv_sqrt, dQ = dS·K, dK = dS^T·Q), so this
// covers the same ground without the projection wrapper.

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

// CPU forward + backward, fp32 math throughout. Inputs are already
// FP16-rounded values (rq above) to remove the dtype round-trip from the
// error budget. Implements the same per-head math as the CUDA op:
//   S = Q·K^T, P_unmasked = softmax(S * inv_sqrt + mask_terms)
//   O = P · V
//   dV = P^T · dO
//   dP = dO · V^T
//   D_q = sum_k P[q,k]·dP[q,k]
//   dS = P * (dP - D_q) * inv_sqrt
//   dQ = dS · K
//   dK = dS^T · Q
static void cpu_attn_backward(const std::vector<float>& Q,
                              const std::vector<float>& K,
                              const std::vector<float>& V,
                              const std::vector<float>& dO,
                              const std::vector<float>* mask,
                              bool causal,
                              int Lq, int Lk, int D, int nh,
                              std::vector<float>& dQ,
                              std::vector<float>& dK,
                              std::vector<float>& dV,
                              std::vector<float>* O_out = nullptr) {
    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    dQ.assign(static_cast<size_t>(Lq) * D, 0.0f);
    dK.assign(static_cast<size_t>(Lk) * D, 0.0f);
    dV.assign(static_cast<size_t>(Lk) * D, 0.0f);
    if (O_out) O_out->assign(static_cast<size_t>(Lq) * D, 0.0f);

    std::vector<float> P(Lq * Lk), dP(Lq * Lk);
    for (int h = 0; h < nh; ++h) {
        const int off = h * hd;

        // Forward: P[q, k] = softmax_k (Q_h·K_h^T)[q, k] * inv_sqrt
        for (int q = 0; q < Lq; ++q) {
            float row_max = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                if (mask && (*mask)[k] <= 0.5f) s = -1e30f;
                if (causal && k > q) s = -1e30f;
                P[q * Lk + k] = s;
                if (s > row_max) row_max = s;
            }
            const bool empty = (row_max <= -1e29f);
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                const float e = empty ? 0.0f : std::exp(P[q*Lk + k] - row_max);
                P[q*Lk + k] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) P[q*Lk + k] *= inv;
        }

        if (O_out) {
            for (int q = 0; q < Lq; ++q)
                for (int d = 0; d < hd; ++d) {
                    double a = 0.0;
                    for (int k = 0; k < Lk; ++k)
                        a += static_cast<double>(P[q*Lk + k]) * V[k*D + off + d];
                    (*O_out)[q*D + off + d] = static_cast<float>(a);
                }
        }

        // dV_h[k, d] = sum_q P[q, k] * dO[q, d]
        for (int k = 0; k < Lk; ++k)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int q = 0; q < Lq; ++q)
                    a += static_cast<double>(P[q*Lk + k]) * dO[q*D + off + d];
                dV[k*D + off + d] = static_cast<float>(a);
            }

        // dP[q, k] = sum_d dO[q, d] * V[k, d]
        for (int q = 0; q < Lq; ++q)
            for (int k = 0; k < Lk; ++k) {
                double a = 0.0;
                for (int d = 0; d < hd; ++d)
                    a += static_cast<double>(dO[q*D + off + d]) * V[k*D + off + d];
                dP[q*Lk + k] = static_cast<float>(a);
            }

        // dS = P * (dP - D_q) * inv_sqrt  (in-place over P)
        for (int q = 0; q < Lq; ++q) {
            double Dq = 0.0;
            for (int k = 0; k < Lk; ++k)
                Dq += static_cast<double>(P[q*Lk + k]) * dP[q*Lk + k];
            for (int k = 0; k < Lk; ++k) {
                const float p  = P[q*Lk + k];
                const float dp = dP[q*Lk + k];
                P[q*Lk + k] = p * (dp - static_cast<float>(Dq)) * inv_sqrt;
            }
        }

        // dQ_h[q, d] = sum_k dS[q, k] * K[k, d]
        for (int q = 0; q < Lq; ++q)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(P[q*Lk + k]) * K[k*D + off + d];
                dQ[q*D + off + d] = static_cast<float>(a);
            }

        // dK_h[k, d] = sum_q dS[q, k] * Q[q, d]
        for (int k = 0; k < Lk; ++k)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int q = 0; q < Lq; ++q)
                    a += static_cast<double>(P[q*Lk + k]) * Q[q*D + off + d];
                dK[k*D + off + d] = static_cast<float>(a);
            }
    }
}

static void check_fp16(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref,
                       const char* label,
                       float atol = 5e-2f, float rtol = 5e-2f) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, g, ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

static void run_case(const char* label, int Lq, int Lk, int D, int nh,
                     bool use_mask, bool causal) {
    std::printf("  %s Lq=%d Lk=%d D=%d nh=%d mask=%d causal=%d\n",
                label, Lq, Lk, D, nh, (int)use_mask, (int)causal);
    std::mt19937 rng(0xBADD1u + static_cast<unsigned>(Lq*131 + Lk*17 + D + nh + use_mask + (causal?7:0)));
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(Lq*D), K(Lk*D), V(Lk*D), dO(Lq*D);
    for (auto& v : Q)  v = dist(rng);
    for (auto& v : K)  v = dist(rng);
    for (auto& v : V)  v = dist(rng);
    for (auto& v : dO) v = dist(rng);
    auto Qq = rq(Q), Kq = rq(K), Vq = rq(V), dOq = rq(dO);

    std::vector<float> mask_host;
    const std::vector<float>* mask_ptr = nullptr;
    if (use_mask) {
        mask_host.assign(Lk, 1.0f);
        for (int k = 3*Lk/4; k < Lk; ++k) mask_host[k] = 0.0f;
        mask_ptr = &mask_host;
    }

    std::vector<float> dQ_ref, dK_ref, dV_ref, O_ref;
    cpu_attn_backward(Qq, Kq, Vq, dOq, mask_ptr, causal,
                      Lq, Lk, D, nh,
                      dQ_ref, dK_ref, dV_ref, &O_ref);

    // Upload everything.
    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V), dOh = to_fp16(dO);
    GpuTensor Qg, Kg, Vg, dOg, Og;
    brotensor::upload_fp16(Qh.data(),  Lq, D, Qg);
    brotensor::upload_fp16(Kh.data(),  Lk, D, Kg);
    brotensor::upload_fp16(Vh.data(),  Lk, D, Vg);
    brotensor::upload_fp16(dOh.data(), Lq, D, dOg);

    // Run forward to produce O (unused by current bwd impl but part of API).
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg,
                                           use_mask ? nullptr : nullptr, // placeholder
                                           nh, causal, Og);
    // (We re-run with mask below; first call without is just to size Og.)

    GpuTensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        brotensor::upload(mask_host.data(), Lk, 1, mg);
        d_mask = mg.data;
    }
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, d_mask, nh, causal, Og);

    GpuTensor dQg, dKg, dVg;
    brotensor::flash_attention_backward_gpu(Qg, Kg, Vg, Og, dOg,
                                            d_mask, nh, causal,
                                            dQg, dKg, dVg);
    CHECK(dQg.rows == Lq && dQg.cols == D && dQg.dtype == Dtype::FP16);
    CHECK(dKg.rows == Lk && dKg.cols == D && dKg.dtype == Dtype::FP16);
    CHECK(dVg.rows == Lk && dVg.cols == D && dVg.dtype == Dtype::FP16);

    std::vector<uint16_t> dQ_got(Lq*D), dK_got(Lk*D), dV_got(Lk*D);
    brotensor::download_fp16(dQg, dQ_got.data());
    brotensor::download_fp16(dKg, dK_got.data());
    brotensor::download_fp16(dVg, dV_got.data());
    brotensor::cuda_sync();

    check_fp16(dQ_got, dQ_ref, "dQ");
    check_fp16(dK_got, dK_ref, "dK");
    check_fp16(dV_got, dV_ref, "dV");
}

int main() {
    brotensor::cuda_init();
    std::printf("test_flash_attention_backward\n");
    // unmasked / masked, causal off, nh in {1, 2}
    run_case("basic-nh1",    4, 4, 8, 1, false, false);
    run_case("basic-nh2",    4, 4, 8, 2, false, false);
    run_case("masked-nh2",   8, 8, 16, 2, true,  false);
    run_case("rect-nh2",     6, 10, 16, 2, false, false);
    run_case("rect-masked",  6, 10, 16, 2, true,  false);
    // causal
    run_case("causal-nh1",   8, 8, 8, 1, false, true);
    run_case("causal-nh2",   8, 8, 16, 2, false, true);
    run_case("causal-masked",8, 8, 16, 2, true,  true);
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
