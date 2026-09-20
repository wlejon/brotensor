// ─── CPU diffusion sampler steps + timestep embedding (CHUNK 4) ────────────
//
// Scalar host implementations supporting FP32, FP16, and BF16 tensors.
// Ports the elementwise sampler kernels:
//   src/cuda/ddim_step.cu, euler_step.cu, dpmpp_2m_step.cu,
//   src/cuda/timestep_embedding.cu
//
// Arithmetic is evaluated in FP32; inputs/outputs are converted to/from FP32
// when using FP16 or BF16 storage, matching GPU sampler behavior.
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
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_sampler_dtype(Dtype dt, const char* op, const char* name) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32, FP16, or BF16");
    }
}

struct Fp32Accessor {
    static float load(const void* ptr, int i) {
        return static_cast<const float*>(ptr)[i];
    }
    static void store(void* ptr, int i, float v) {
        static_cast<float*>(ptr)[i] = v;
    }
};

struct Fp16Accessor {
    static float load(const void* ptr, int i) {
        return fp16_bits_to_fp32(static_cast<const uint16_t*>(ptr)[i]);
    }
    static void store(void* ptr, int i, float v) {
        static_cast<uint16_t*>(ptr)[i] = fp32_to_fp16_bits(v);
    }
};

struct Bf16Accessor {
    static float load(const void* ptr, int i) {
        return bf16_bits_to_fp32(static_cast<const uint16_t*>(ptr)[i]);
    }
    static void store(void* ptr, int i, float v) {
        static_cast<uint16_t*>(ptr)[i] = fp32_to_bf16_bits(v);
    }
};

template <typename Acc>
void ddim_step_impl(const void* xtp, const void* epsp, void* xpp, int total,
                    float sqrt_alpha_prev, float inv_sqrt_alpha_t,
                    float sqrt_1m_alpha_t, float dir_coef) {
    for (int i = 0; i < total; ++i) {
        const float xt = Acc::load(xtp, i);
        const float eps = Acc::load(epsp, i);
        const float x0_pred = (xt - sqrt_1m_alpha_t * eps) * inv_sqrt_alpha_t;
        const float dir = dir_coef * eps;
        Acc::store(xpp, i, sqrt_alpha_prev * x0_pred + dir);
    }
}

template <typename Acc>
void euler_step_impl(const void* xtp, const void* epsp, void* xpp, int total, float dsigma) {
    for (int i = 0; i < total; ++i) {
        const float xt = Acc::load(xtp, i);
        const float eps = Acc::load(epsp, i);
        Acc::store(xpp, i, xt + dsigma * eps);
    }
}

template <typename Acc>
void dpmpp_2m_step_impl(const void* xtp, const void* epsp, const void* x0pp,
                        void* xpp, void* x0op, int total,
                        float sigma_t, float c_xt, float c_x0t, float c_x0prev) {
    for (int i = 0; i < total; ++i) {
        const float xt = Acc::load(xtp, i);
        const float eps = Acc::load(epsp, i);
        const float x0p = Acc::load(x0pp, i);
        const float x0t = xt - sigma_t * eps;
        Acc::store(xpp, i, c_xt * xt + c_x0t * x0t + c_x0prev * x0p);
        Acc::store(x0op, i, x0t);
    }
}

template <typename InAcc, typename OutAcc>
void timestep_embedding_impl(const void* tsp, void* Yp, int N, int dim, int half, float log_max_period) {
    for (int i = 0; i < N; ++i) {
        const float ts = InAcc::load(tsp, i);
        for (int j = 0; j < dim; ++j) {
            if (j >= 2 * half) {
                OutAcc::store(Yp, i * dim + j, 0.0f);
                continue;
            }
            const int k = j < half ? j : j - half;
            const float freq = std::exp(-log_max_period *
                                        static_cast<float>(k) /
                                        static_cast<float>(half));
            const float arg = ts * freq;
            OutAcc::store(Yp, i * dim + j, j < half ? std::cos(arg) : std::sin(arg));
        }
    }
}

} // namespace

void ddim_step(const ::brotensor::Tensor& x_t,
               const ::brotensor::Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               ::brotensor::Tensor& x_prev) {
    check_sampler_dtype(x_t.dtype, "ddim_step", "x_t");
    if (eps_pred.dtype != x_t.dtype) {
        throw std::runtime_error("ddim_step: x_t and eps_pred must have the same dtype");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("ddim_step: shape mismatch between x_t and eps_pred");
    }
    const Dtype dtype = x_t.dtype;
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != dtype) {
        x_prev.resize(x_t.rows, x_t.cols, dtype);
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

    if (dtype == Dtype::FP32) {
        ddim_step_impl<Fp32Accessor>(x_t.data, eps_pred.data, x_prev.data, total,
                                     sqrt_alpha_prev, inv_sqrt_alpha_t, sqrt_1m_alpha_t, dir_coef);
    } else if (dtype == Dtype::FP16) {
        ddim_step_impl<Fp16Accessor>(x_t.data, eps_pred.data, x_prev.data, total,
                                     sqrt_alpha_prev, inv_sqrt_alpha_t, sqrt_1m_alpha_t, dir_coef);
    } else {
        ddim_step_impl<Bf16Accessor>(x_t.data, eps_pred.data, x_prev.data, total,
                                     sqrt_alpha_prev, inv_sqrt_alpha_t, sqrt_1m_alpha_t, dir_coef);
    }
}

void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev) {
    check_sampler_dtype(x_t.dtype, "euler_step", "x_t");
    if (eps_pred.dtype != x_t.dtype) {
        throw std::runtime_error("euler_step: x_t and eps_pred must have the same dtype");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("euler_step: shape mismatch between x_t and eps_pred");
    }
    const Dtype dtype = x_t.dtype;
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != dtype) {
        x_prev.resize(x_t.rows, x_t.cols, dtype);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const float dsigma = sigma_prev - sigma_t;
    if (dtype == Dtype::FP32) {
        euler_step_impl<Fp32Accessor>(x_t.data, eps_pred.data, x_prev.data, total, dsigma);
    } else if (dtype == Dtype::FP16) {
        euler_step_impl<Fp16Accessor>(x_t.data, eps_pred.data, x_prev.data, total, dsigma);
    } else {
        euler_step_impl<Bf16Accessor>(x_t.data, eps_pred.data, x_prev.data, total, dsigma);
    }
}

void dpmpp_2m_step(const ::brotensor::Tensor& x_t,
                   const ::brotensor::Tensor& eps_pred,
                   const ::brotensor::Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   ::brotensor::Tensor& x_prev,
                   ::brotensor::Tensor& x0_out) {
    check_sampler_dtype(x_t.dtype, "dpmpp_2m_step", "x_t");
    if (eps_pred.dtype != x_t.dtype || x0_prev.dtype != x_t.dtype) {
        throw std::runtime_error("dpmpp_2m_step: all inputs must have the same dtype");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols ||
        x_t.rows != x0_prev.rows  || x_t.cols != x0_prev.cols) {
        throw std::runtime_error("dpmpp_2m_step: shape mismatch");
    }
    const Dtype dtype = x_t.dtype;
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != dtype) {
        x_prev.resize(x_t.rows, x_t.cols, dtype);
    }
    if (x0_out.rows != x_t.rows || x0_out.cols != x_t.cols || x0_out.dtype != dtype) {
        x0_out.resize(x_t.rows, x_t.cols, dtype);
    }
    const int total = x_t.size();
    if (total == 0) return;

    if (dtype == Dtype::FP32) {
        dpmpp_2m_step_impl<Fp32Accessor>(x_t.data, eps_pred.data, x0_prev.data,
                                         x_prev.data, x0_out.data, total,
                                         sigma_t, c_xt, c_x0t, c_x0prev);
    } else if (dtype == Dtype::FP16) {
        dpmpp_2m_step_impl<Fp16Accessor>(x_t.data, eps_pred.data, x0_prev.data,
                                         x_prev.data, x0_out.data, total,
                                         sigma_t, c_xt, c_x0t, c_x0prev);
    } else {
        dpmpp_2m_step_impl<Bf16Accessor>(x_t.data, eps_pred.data, x0_prev.data,
                                         x_prev.data, x0_out.data, total,
                                         sigma_t, c_xt, c_x0t, c_x0prev);
    }
}

void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period,
                        ::brotensor::Tensor& Y) {
    check_sampler_dtype(timesteps.dtype, "timestep_embedding", "timesteps");
    if (timesteps.cols != 1) {
        throw std::runtime_error("timestep_embedding: timesteps must be (N,1)");
    }
    if (dim <= 0) {
        throw std::runtime_error("timestep_embedding: dim must be positive");
    }
    const int N = timesteps.rows;
    Dtype out_dt = (Y.dtype == Dtype::FP16 || Y.dtype == Dtype::BF16 || Y.dtype == Dtype::FP32)
                       ? Y.dtype : timesteps.dtype;
    if (Y.rows != N || Y.cols != dim || Y.dtype != out_dt) {
        Y.resize(N, dim, out_dt);
    }
    if (N == 0) return;

    const int half = dim / 2;
    const float log_max_period = std::log(max_period);

    auto dispatch_out = [&](auto in_acc) {
        using InT = decltype(in_acc);
        if (Y.dtype == Dtype::FP32) {
            timestep_embedding_impl<InT, Fp32Accessor>(timesteps.data, Y.data, N, dim, half, log_max_period);
        } else if (Y.dtype == Dtype::FP16) {
            timestep_embedding_impl<InT, Fp16Accessor>(timesteps.data, Y.data, N, dim, half, log_max_period);
        } else {
            timestep_embedding_impl<InT, Bf16Accessor>(timesteps.data, Y.data, N, dim, half, log_max_period);
        }
    };

    if (timesteps.dtype == Dtype::FP32) {
        dispatch_out(Fp32Accessor{});
    } else if (timesteps.dtype == Dtype::FP16) {
        dispatch_out(Fp16Accessor{});
    } else {
        dispatch_out(Bf16Accessor{});
    }
}

} // namespace brotensor::detail::cpu
