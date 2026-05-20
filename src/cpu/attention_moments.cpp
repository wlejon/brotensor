// CPU backend — attention_token_moments (CHUNK 5).
//
// Ground truth: src/cuda/attention_moments.cu.
//
// Per-text-token spatial moments of a cross-attention map. Attn is (Lq, Lk)
// where Lq = h_lat * w_lat spatial query tokens and Lk text tokens. For each
// text token k it computes:
//   mass[k]        = sum_q Attn(q, k)
//   centroid[k,0]  = (sum_q y(q) * Attn(q, k)) / mass[k]   (y = q / w_lat)
//   centroid[k,1]  = (sum_q x(q) * Attn(q, k)) / mass[k]   (x = q % w_lat)
// If mass[k] <= MOM_EPS, both centroid components are set to 0.
//
// DTYPE: the CUDA kernel reads FP16 Attn and writes FP32 mass/centroid. The
// CPU backend is FP32-only — it reads FP32 Attn. The parity test quantises
// the attention map through FP16 so CPU (FP32) and GPU (FP16) start from the
// same input bit patterns, then compares with a loose tolerance (the GPU's
// long FP16 reduction over up to ~1024 spatial tokens accumulates noise).
//
// mass and centroid are OVERWRITTEN (computed fresh, not accumulated).

#include <brotensor/tensor.h>

#include <stdexcept>

namespace brotensor::detail::cpu {

namespace {
constexpr float MOM_EPS = 1e-8f;
}

void attention_token_moments(const ::brotensor::Tensor& Attn,
                             int h_lat, int w_lat,
                             ::brotensor::Tensor& mass,
                             ::brotensor::Tensor& centroid) {
    using ::brotensor::Dtype;
    if (h_lat <= 0 || w_lat <= 0) {
        throw std::runtime_error(
            "attention_token_moments: h_lat and w_lat must be positive");
    }
    const int Lq = h_lat * w_lat;
    const int Lk = Attn.cols;
    if (Attn.rows != Lq) {
        throw std::runtime_error(
            "attention_token_moments: Attn.rows must equal h_lat * w_lat");
    }
    if (mass.rows != Lk || mass.cols != 1 || mass.dtype != Dtype::FP32)
        mass.resize(Lk, 1, Dtype::FP32);
    if (centroid.rows != Lk || centroid.cols != 2 || centroid.dtype != Dtype::FP32)
        centroid.resize(Lk, 2, Dtype::FP32);
    if (Lk == 0) return;

    const float* Ap = Attn.host_f32();
    float* Mp = mass.host_f32_mut();
    float* Cp = centroid.host_f32_mut();

    for (int k = 0; k < Lk; ++k) {
        float am = 0.0f, ay = 0.0f, ax = 0.0f;
        for (int q = 0; q < Lq; ++q) {
            const float a = Ap[static_cast<std::size_t>(q) * Lk + k];
            const int y = q / w_lat;
            const int x = q - y * w_lat;
            am += a;
            ay += static_cast<float>(y) * a;
            ax += static_cast<float>(x) * a;
        }
        Mp[k] = am;
        if (am > MOM_EPS) {
            const float inv = 1.0f / am;
            Cp[static_cast<std::size_t>(k) * 2 + 0] = ay * inv;
            Cp[static_cast<std::size_t>(k) * 2 + 1] = ax * inv;
        } else {
            Cp[static_cast<std::size_t>(k) * 2 + 0] = 0.0f;
            Cp[static_cast<std::size_t>(k) * 2 + 1] = 0.0f;
        }
    }
}

} // namespace brotensor::detail::cpu
