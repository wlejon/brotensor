// Metal diffusion sampler steps (DDIM, Euler, DPM++ 2M), sinusoidal timestep
// embedding, image preprocessing (image_normalize, image_u8_to_f32_nhwc_to_nchw),
// and Philox 4x32-10 noise (randn, rand_uniform, rand_bernoulli, randn_truncated).
//
// Single-source brass MIR kernels (src/kernels/mir_diffusion.cpp) compiled via
// MslTarget and cached in mtl_mir_cache.mm (mirrors src/vulkan/ops_diffusion.cpp).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

struct SamplerArgs {
    const void* x = nullptr;
    const void* e = nullptr;
    const void* p = nullptr;
    const void* y = nullptr;
    const void* z = nullptr;
    std::uint32_t n = 0, dim = 0, half_ = 0, hw = 0, c = 0;
    float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "unsupported dtype (expected FP32/FP16/BF16)");
    }
}

std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

std::uint32_t count32(const Tensor& t, const char* op) {
    const long long n = t.size();
    if (n < 0 || n > 0x7fffffffLL) fail(op, "element count out of range");
    return static_cast<std::uint32_t>(n);
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

void same_shape(const char* op, const Tensor& a, const Tensor& b) {
    need(op, a.rows == b.rows && a.cols == b.cols, "shape mismatch");
}

void run_sampler(const char* op, Dtype dt, Dtype in_dt, std::uint32_t mode,
                 const SamplerArgs& a) {
    if (a.n == 0) return;
    int lanes = 1;
    if (mode == SMP_DDIM || mode == SMP_EULER || mode == SMP_DPMPP_2M) {
        const std::uint32_t check_n = (a.n >= 8u) ? 0u : a.n;
        lanes = mir_vec_lanes(dt, check_n, addr(a.x), addr(a.e), addr(a.p), addr(a.y));
        if (mode == SMP_DPMPP_2M) {
            lanes = std::min(lanes, mir_vec_lanes(dt, check_n, addr(a.z)));
        }
    }
    const MirKernel mk = mir_kernel(
        {MirKind::Sampler, mode, dt, in_dt, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_sampler(mk, mir_groups_1d(a.n, lanes), op,
                         a.x, a.e, a.p, a.y, a.z,
                         a.n, a.dim, a.half_, a.hw, a.c,
                         a.k0, a.k1, a.k2, a.k3);
}

std::size_t check_noise_y(const char* op, const Tensor& Y) {
    need(op, Y.dtype == Dtype::FP32, "Y must be FP32");
    need(op, Y.rows >= 0 && Y.cols >= 0, "Y has negative dimension");
    const std::size_t n =
        static_cast<std::size_t>(Y.rows) * static_cast<std::size_t>(Y.cols);
    need(op, n == 0 || Y.data != nullptr,
         "Y is uncommitted; pre-allocate before calling");
    return n;
}

void run_philox(const char* op, std::uint32_t mode, std::uint64_t key,
                std::uint64_t counter, float p, float lo, float hi, Tensor& Y) {
    const std::size_t n = check_noise_y(op, Y);
    if (n == 0) return;
    const std::uint64_t gx64 = std::min<std::uint64_t>((n + 255u) / 256u, 65535u);
    const MirKernel mk = mir_kernel({MirKind::Philox, mode});
    mir_dispatch_philox(mk, static_cast<std::uint32_t>(gx64), op,
                        Y.data, key, counter, n, p, lo, hi);
}

}  // namespace

// ─── Sampler steps ─────────────────────────────────────────────────────────

void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev) {
    const char* op = "ddim_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    const float sqrt_alpha_t = std::sqrt(alpha_t);
    SamplerArgs a{};
    a.x = x_t.data;
    a.e = eps_pred.data;
    a.y = x_prev.data;
    a.n = count32(x_t, op);
    a.k0 = std::sqrt(std::max(0.0f, alpha_prev));
    a.k1 = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    a.k2 = std::sqrt(std::max(0.0f, 1.0f - alpha_t));
    a.k3 = std::sqrt(std::max(0.0f, 1.0f - alpha_prev - sigma_t * sigma_t));
    run_sampler(op, x_t.dtype, Dtype::FP32, SMP_DDIM, a);
}

void euler_step(const Tensor& x_t, const Tensor& eps_pred,
                float sigma_t, float sigma_prev, Tensor& x_prev) {
    const char* op = "euler_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    SamplerArgs a{};
    a.x = x_t.data;
    a.e = eps_pred.data;
    a.y = x_prev.data;
    a.n = count32(x_t, op);
    a.k0 = sigma_prev - sigma_t;
    run_sampler(op, x_t.dtype, Dtype::FP32, SMP_EULER, a);
}

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred, const Tensor& x0_prev,
                   float sigma_t, float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out) {
    const char* op = "dpmpp_2m_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype && x0_prev.dtype == x_t.dtype,
         "all inputs must have the same dtype");
    same_shape(op, x_t, eps_pred);
    same_shape(op, x_t, x0_prev);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    ensure(x0_out, x_t.rows, x_t.cols, x_t.dtype);
    SamplerArgs a{};
    a.x = x_t.data;
    a.e = eps_pred.data;
    a.p = x0_prev.data;
    a.y = x_prev.data;
    a.z = x0_out.data;
    a.n = count32(x_t, op);
    a.k0 = sigma_t;
    a.k1 = c_xt;
    a.k2 = c_x0t;
    a.k3 = c_x0prev;
    run_sampler(op, x_t.dtype, Dtype::FP32, SMP_DPMPP_2M, a);
}

void timestep_embedding(const Tensor& timesteps, int dim, float max_period, Tensor& Y) {
    const char* op = "timestep_embedding";
    require_fp(timesteps.dtype, op);
    need(op, timesteps.cols == 1, "timesteps must be (N,1)");
    need(op, dim > 0, "dim must be positive");
    const Dtype out = (Y.dtype == Dtype::FP16 || Y.dtype == Dtype::BF16 ||
                       Y.dtype == Dtype::FP32)
                          ? Y.dtype
                          : timesteps.dtype;
    ensure(Y, timesteps.rows, dim, out);
    SamplerArgs a{};
    a.x = timesteps.data;
    a.y = Y.data;
    a.n = count32(Y, op);
    a.dim = static_cast<std::uint32_t>(dim);
    a.half_ = static_cast<std::uint32_t>(dim / 2);
    a.k0 = std::log(max_period);
    run_sampler(op, out, timesteps.dtype, SMP_TIMESTEP_EMB, a);
}

// ─── Image preprocessing ───────────────────────────────────────────────────

void image_normalize(const Tensor& X, const Tensor& mean, const Tensor& std_,
                     int N, int C, int H, int W, Tensor& Y) {
    const char* op = "image_normalize";
    require_fp(X.dtype, op);
    need(op, mean.dtype == Dtype::FP32 && std_.dtype == Dtype::FP32,
         "mean / std must be FP32");
    need(op, mean.size() == C && std_.size() == C, "mean/std must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    ensure(Y, N, static_cast<int>(cols), Dtype::FP32);
    SamplerArgs a{};
    a.x = X.data;
    a.e = mean.data;
    a.p = std_.data;
    a.y = Y.data;
    a.n = count32(Y, op);
    a.hw = static_cast<std::uint32_t>(H * W);
    a.c = static_cast<std::uint32_t>(C);
    if (X.dtype != Dtype::FP32) {
        Tensor xf = Tensor::empty_on(X.device, X.rows, X.cols, Dtype::FP32);
        ::brotensor::cast(X, xf, Dtype::FP32);
        a.x = xf.data;
        run_sampler(op, Dtype::FP32, Dtype::FP32, SMP_IMAGE_NORM, a);
        return;
    }
    run_sampler(op, Dtype::FP32, Dtype::FP32, SMP_IMAGE_NORM, a);
}

void image_u8_to_f32_nhwc_to_nchw(const std::uint8_t* src, int N, int H, int W, int C,
                                  float scale, float bias, Tensor& Y) {
    const char* op = "image_u8_to_f32_nhwc_to_nchw";
    need(op, N >= 0 && H >= 0 && W >= 0 && C >= 0, "negative dim");
    need(op, src != nullptr || static_cast<long long>(N) * H * W * C == 0, "src is null");
    ensure(Y, N, C * H * W, Dtype::FP32);
    if (Y.size() == 0) return;
    NSUInteger ofs = 0;
    metal_impl::pool_require(src, ofs, op, "src");
    SamplerArgs a{};
    a.x = src;
    a.y = Y.data;
    a.n = count32(Y, op);
    a.hw = static_cast<std::uint32_t>(H * W);
    a.c = static_cast<std::uint32_t>(C);
    a.k0 = scale;
    a.k1 = bias;
    run_sampler(op, Dtype::FP32, Dtype::FP32, SMP_U8_NHWC, a);
}

// ─── Philox 4x32-10 noise ──────────────────────────────────────────────────

void randn(std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    run_philox("randn", RNG_NORMAL, key, counter, 0.0f, 0.0f, 0.0f, Y);
}

void rand_uniform(std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    run_philox("rand_uniform", RNG_UNIFORM, key, counter, 0.0f, 0.0f, 0.0f, Y);
}

void rand_bernoulli(float p, std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    need("rand_bernoulli", p >= 0.0f && p <= 1.0f, "p must be in [0, 1]");
    run_philox("rand_bernoulli", RNG_BERNOULLI, key, counter, p, 0.0f, 0.0f, Y);
}

void randn_truncated(float lo, float hi, std::uint64_t key, std::uint64_t counter,
                     Tensor& Y) {
    need("randn_truncated", lo < hi, "lo must be < hi");
    run_philox("randn_truncated", RNG_TRUNCATED, key, counter, 0.0f, lo, hi, Y);
}

}  // namespace brotensor::detail::metal
