// CUDA diffusion samplers, timestep embedding, image preprocessing, and Philox
// 4x32-10 PRNG noise ops backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct SamplerPush {
    const void* x = nullptr;
    const void* e = nullptr;
    const void* p = nullptr;
    void* y = nullptr;
    void* z = nullptr;
    std::uint32_t n = 0, dim = 0, half = 0, hw = 0, c = 0;
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
        fail(op, "must be FP32, FP16, or BF16");
    }
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

void same_shape(const char* op, const Tensor& a, const Tensor& b) {
    need(op, a.rows == b.rows && a.cols == b.cols, "shape mismatch");
}

inline std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

std::uint32_t count32(const Tensor& t, const char* op) {
    const long long n = static_cast<long long>(t.rows) * t.cols;
    if (n < 0 || n > 0xffffffffLL) fail(op, "tensor too large");
    return static_cast<std::uint32_t>(n);
}

void run_sampler(Dtype dt, std::uint32_t mode, const SamplerPush& pc, Dtype in_dt = Dtype::FP32) {
    if (pc.n == 0) return;
    int lanes = 1;
    if (mode == SMP_DDIM || mode == SMP_EULER || mode == SMP_DPMPP_2M) {
        const std::uint32_t check_n = (pc.n >= 8u) ? 0u : pc.n;
        lanes = mir_vec_lanes(dt, check_n, addr(pc.x), addr(pc.e), addr(pc.p), addr(pc.y));
        if (mode == SMP_DPMPP_2M) {
            lanes = (std::min)(lanes, mir_vec_lanes(dt, check_n, addr(pc.z)));
        }
    }
    CUfunction fn = mir_function(
        {MirKind::Sampler, mode, dt, in_dt, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_sampler(
        fn, mir_groups_1d(pc.n, lanes),
        pc.x, pc.e, pc.p, pc.y, pc.z,
        pc.n, pc.dim, pc.half, pc.hw, pc.c, pc.k0, pc.k1, pc.k2, pc.k3);
}

std::size_t check_noise_y(const char* op, const Tensor& Y) {
    need(op, Y.dtype == Dtype::FP32, "Y must be FP32");
    need(op, Y.rows >= 0 && Y.cols >= 0, "Y has negative dimension");
    const std::size_t n = static_cast<std::size_t>(Y.rows) * static_cast<std::size_t>(Y.cols);
    need(op, n == 0 || Y.data != nullptr, "Y is uncommitted; pre-allocate before calling");
    return n;
}

void run_philox(const char* op, std::uint32_t mode, std::uint64_t key, std::uint64_t counter,
                float p, float lo, float hi, Tensor& Y) {
    const std::size_t n = check_noise_y(op, Y);
    if (n == 0) return;
    const std::uint64_t gx64 = std::min((n + 255u) / 256u, std::uint64_t(65535));
    CUfunction fn = mir_function({MirKind::Philox, mode});
    mir_dispatch_philox(fn, static_cast<std::uint32_t>(gx64), Y.data, key, counter, n, p, lo, hi);
}

}  // namespace

void ddim_step(const Tensor& x_t, const Tensor& eps_pred, float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev) {
    constexpr const char* op = "ddim_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    const float sqrt_alpha_t = std::sqrt(alpha_t);
    SamplerPush pc{};
    pc.x = x_t.data;
    pc.e = eps_pred.data;
    pc.y = x_prev.data;
    pc.n = count32(x_t, op);
    pc.k0 = std::sqrt((std::max)(0.0f, alpha_prev));
    pc.k1 = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    pc.k2 = std::sqrt((std::max)(0.0f, 1.0f - alpha_t));
    pc.k3 = std::sqrt((std::max)(0.0f, 1.0f - alpha_prev - sigma_t * sigma_t));
    run_sampler(x_t.dtype, SMP_DDIM, pc);
}

void euler_step(const Tensor& x_t, const Tensor& eps_pred, float sigma_t, float sigma_prev, Tensor& x_prev) {
    constexpr const char* op = "euler_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    SamplerPush pc{};
    pc.x = x_t.data;
    pc.e = eps_pred.data;
    pc.y = x_prev.data;
    pc.n = count32(x_t, op);
    pc.k0 = sigma_prev - sigma_t;
    run_sampler(x_t.dtype, SMP_EULER, pc);
}

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred, const Tensor& x0_prev, float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev, Tensor& x_prev, Tensor& x0_out) {
    constexpr const char* op = "dpmpp_2m_step";
    require_fp(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype && x0_prev.dtype == x_t.dtype, "all inputs must have the same dtype");
    same_shape(op, x_t, eps_pred);
    same_shape(op, x_t, x0_prev);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    ensure(x0_out, x_t.rows, x_t.cols, x_t.dtype);
    SamplerPush pc{};
    pc.x = x_t.data;
    pc.e = eps_pred.data;
    pc.p = x0_prev.data;
    pc.y = x_prev.data;
    pc.z = x0_out.data;
    pc.n = count32(x_t, op);
    pc.k0 = sigma_t;
    pc.k1 = c_xt;
    pc.k2 = c_x0t;
    pc.k3 = c_x0prev;
    run_sampler(x_t.dtype, SMP_DPMPP_2M, pc);
}

void timestep_embedding(const Tensor& timesteps, int dim, float max_period, Tensor& Y) {
    constexpr const char* op = "timestep_embedding";
    require_fp(timesteps.dtype, op);
    need(op, timesteps.cols == 1, "timesteps must be (N,1)");
    need(op, dim > 0, "dim must be positive");
    const Dtype out = (Y.dtype == Dtype::FP16 || Y.dtype == Dtype::BF16 || Y.dtype == Dtype::FP32)
                          ? Y.dtype
                          : timesteps.dtype;
    ensure(Y, timesteps.rows, dim, out);
    SamplerPush pc{};
    pc.x = timesteps.data;
    pc.y = Y.data;
    pc.n = count32(Y, op);
    pc.dim = static_cast<std::uint32_t>(dim);
    pc.half = static_cast<std::uint32_t>(dim / 2);
    pc.k0 = std::log(max_period);
    run_sampler(out, SMP_TIMESTEP_EMB, pc, timesteps.dtype);
}

void image_normalize(const Tensor& X, const Tensor& mean, const Tensor& std_,
                     int N, int C, int H, int W, Tensor& Y) {
    constexpr const char* op = "image_normalize";
    require_fp(X.dtype, op);
    need(op, mean.dtype == Dtype::FP32 && std_.dtype == Dtype::FP32, "mean/std must be FP32");
    need(op, mean.size() == C && std_.size() == C, "mean/std must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    ensure(Y, N, static_cast<int>(cols), Dtype::FP32);
    SamplerPush pc{};
    pc.e = mean.data;
    pc.p = std_.data;
    pc.y = Y.data;
    pc.n = count32(Y, op);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    if (X.dtype != Dtype::FP32) {
        Tensor xf = Tensor::empty_on(X.device, X.rows, X.cols, Dtype::FP32);
        ::brotensor::cast(X, xf, Dtype::FP32);
        pc.x = xf.data;
        run_sampler(Dtype::FP32, SMP_IMAGE_NORM, pc);
        return;
    }
    pc.x = X.data;
    run_sampler(Dtype::FP32, SMP_IMAGE_NORM, pc);
}

void image_u8_to_f32_nhwc_to_nchw(const std::uint8_t* src, int N, int H, int W, int C,
                                  float scale, float bias, Tensor& Y) {
    constexpr const char* op = "image_u8_to_f32_nhwc_to_nchw";
    need(op, N >= 0 && H >= 0 && W >= 0 && C >= 0, "negative dim");
    need(op, src != nullptr || static_cast<long long>(N) * H * W * C == 0, "src is null");
    ensure(Y, N, C * H * W, Dtype::FP32);
    if (Y.size() == 0) return;
    SamplerPush pc{};
    pc.x = src;
    pc.y = Y.data;
    pc.n = count32(Y, op);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.k0 = scale;
    pc.k1 = bias;
    run_sampler(Dtype::FP32, SMP_U8_NHWC, pc);
}

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

void randn_truncated(float lo, float hi, std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    need("randn_truncated", lo < hi, "lo must be < hi");
    run_philox("randn_truncated", RNG_TRUNCATED, key, counter, 0.0f, lo, hi, Y);
}

void fill_cuda_vtable_image_preproc(::brotensor::detail::OpsVTable& v) {
    v.image_normalize = &image_normalize;
    v.image_u8_to_f32_nhwc_to_nchw = &image_u8_to_f32_nhwc_to_nchw;
}

void fill_cuda_vtable_noise(::brotensor::detail::OpsVTable& v) {
    v.randn = &randn;
    v.rand_uniform = &rand_uniform;
    v.rand_bernoulli = &rand_bernoulli;
    v.randn_truncated = &randn_truncated;
}

}  // namespace brotensor::detail::cuda
