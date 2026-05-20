// ─── CPU diffusion sampler steps + timestep embedding (CHUNK 4) ────────────
//
// FP32 scalar host implementations. Ports the elementwise sampler kernels:
//   src/cuda/ddim_step.cu, euler_step.cu, dpmpp_2m_step.cu,
//   src/cuda/timestep_embedding.cu
//
// IMPORTANT — dtype: the GPU sampler kernels (ddim/euler/dpmpp_2m) run FP16
// internally (their tensors must be FP16). The CPU backend is FP32-only, so
// the CPU impls require FP32 tensors. CPU↔GPU parity for these three ops
// therefore feeds FP16 to the GPU and FP32 to the CPU and compares with a
// loose FP16-driven tolerance (see tests/test_diffusion_parity.cpp). The
// internal arithmetic is identical FP32 math in both backends — the GPU just
// rounds inputs/outputs through FP16 storage. timestep_embedding is FP32 on
// both backends, so it gets a tight tolerance.
//
// ACCUMULATION: every op fully OVERWRITES its outputs (x_prev / x0_out / Y).
//
// ── ddim_step ──
//   x0_pred = (x_t - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
//   dir     = sqrt(max(0, 1 - alpha_prev - sigma_t^2)) * eps_pred
//   x_prev  = sqrt(max(0, alpha_prev)) * x0_pred + dir
//   (inv_sqrt_alpha_t is 0 when sqrt(alpha_t) <= 0, matching the GPU.)
//
// ── euler_step ──
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
//
// ── dpmpp_2m_step ──
//   x0_t   = x_t - sigma_t * eps_pred
//   x_prev = c_xt * x_t + c_x0t * x0_t + c_x0prev * x0_prev
//   x0_out = x0_t
//
// ── timestep_embedding ── (diffusers get_timestep_embedding,
//    flip_sin_to_cos=True, downscale_freq_shift=0):
//   half      = dim / 2
//   freqs[k]  = exp(-log(max_period) * k / half)
//   args[i,j] = timesteps[i] * freqs[k],  k = j (j<half) else j-half
//   Y[i, 0:half]      = cos(args)
//   Y[i, half:2*half] = sin(args)
//   if dim is odd: Y[i, dim-1] = 0

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void ddim_step(const ::brotensor::Tensor& x_t,
               const ::brotensor::Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               ::brotensor::Tensor& x_prev) {
    check_fp32(x_t, "ddim_step", "x_t");
    check_fp32(eps_pred, "ddim_step", "eps_pred");
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("ddim_step: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != Dtype::FP32) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP32);
    }
    const int total = x_t.size();
    if (total == 0) return;

    // Scalar coefficients precomputed in FP32 — identical to the GPU host code.
    const float sqrt_alpha_t     = std::sqrt(alpha_t);
    const float inv_sqrt_alpha_t = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    const float sqrt_1m_alpha_t  = std::sqrt(std::max(0.0f, 1.0f - alpha_t));
    const float sqrt_alpha_prev  = std::sqrt(std::max(0.0f, alpha_prev));
    const float dir_inner        = 1.0f - alpha_prev - sigma_t * sigma_t;
    const float dir_coef         = std::sqrt(std::max(0.0f, dir_inner));

    const float* xtp  = x_t.host_f32();
    const float* epsp = eps_pred.host_f32();
    float* xpp = x_prev.host_f32_mut();

    for (int i = 0; i < total; ++i) {
        const float xt  = xtp[i];
        const float eps = epsp[i];
        const float x0_pred = (xt - sqrt_1m_alpha_t * eps) * inv_sqrt_alpha_t;
        const float dir = dir_coef * eps;
        xpp[i] = sqrt_alpha_prev * x0_pred + dir;
    }
}

void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev) {
    check_fp32(x_t, "euler_step", "x_t");
    check_fp32(eps_pred, "euler_step", "eps_pred");
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("euler_step: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != Dtype::FP32) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP32);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const float dsigma = sigma_prev - sigma_t;

    const float* xtp  = x_t.host_f32();
    const float* epsp = eps_pred.host_f32();
    float* xpp = x_prev.host_f32_mut();

    for (int i = 0; i < total; ++i) {
        xpp[i] = xtp[i] + dsigma * epsp[i];
    }
}

void dpmpp_2m_step(const ::brotensor::Tensor& x_t,
                   const ::brotensor::Tensor& eps_pred,
                   const ::brotensor::Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   ::brotensor::Tensor& x_prev,
                   ::brotensor::Tensor& x0_out) {
    check_fp32(x_t, "dpmpp_2m_step", "x_t");
    check_fp32(eps_pred, "dpmpp_2m_step", "eps_pred");
    check_fp32(x0_prev, "dpmpp_2m_step", "x0_prev");
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols ||
        x_t.rows != x0_prev.rows  || x_t.cols != x0_prev.cols) {
        throw std::runtime_error("dpmpp_2m_step: shape mismatch");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != Dtype::FP32) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP32);
    }
    if (x0_out.rows != x_t.rows || x0_out.cols != x_t.cols ||
        x0_out.dtype != Dtype::FP32) {
        x0_out.resize(x_t.rows, x_t.cols, Dtype::FP32);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const float* xtp  = x_t.host_f32();
    const float* epsp = eps_pred.host_f32();
    const float* x0pp = x0_prev.host_f32();
    float* xpp  = x_prev.host_f32_mut();
    float* x0op = x0_out.host_f32_mut();

    for (int i = 0; i < total; ++i) {
        const float xt  = xtp[i];
        const float eps = epsp[i];
        const float x0p = x0pp[i];
        const float x0t = xt - sigma_t * eps;
        xpp[i]  = c_xt * xt + c_x0t * x0t + c_x0prev * x0p;
        x0op[i] = x0t;
    }
}

void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period,
                        ::brotensor::Tensor& Y) {
    check_fp32(timesteps, "timestep_embedding", "timesteps");
    if (timesteps.cols != 1) {
        throw std::runtime_error("timestep_embedding: timesteps must be (N,1)");
    }
    if (dim <= 0) {
        throw std::runtime_error("timestep_embedding: dim must be positive");
    }
    const int N = timesteps.rows;
    if (Y.rows != N || Y.cols != dim || Y.dtype != Dtype::FP32) {
        Y.resize(N, dim, Dtype::FP32);
    }
    if (N == 0) return;

    const int half = dim / 2;
    const float log_max_period = std::log(max_period);

    const float* tsp = timesteps.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int i = 0; i < N; ++i) {
        const float ts = tsp[i];
        for (int j = 0; j < dim; ++j) {
            if (j >= 2 * half) {
                // Odd-dim tail slot.
                Yp[i * dim + j] = 0.0f;
                continue;
            }
            const int k = j < half ? j : j - half;
            const float freq = std::exp(-log_max_period *
                                        static_cast<float>(k) /
                                        static_cast<float>(half));
            const float arg = ts * freq;
            Yp[i * dim + j] = j < half ? std::cos(arg) : std::sin(arg);
        }
    }
}

} // namespace brotensor::detail::cpu
