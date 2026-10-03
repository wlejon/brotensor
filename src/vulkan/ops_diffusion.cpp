// Vulkan diffusion helpers: the sampler steps (DDIM, Euler, DPM++ 2M), the
// sinusoidal timestep embedding, the image preprocessing helpers, Philox
// noise (randn, rand_uniform, rand_bernoulli, randn_truncated), and the
// diffusion ResBlock (forward and backward) composed from the Vulkan
// GroupNorm and convolution.
// Contracts follow the CPU / CUDA backends (src/cpu/diffusion_samplers.cpp,
// noise.cpp, image_preproc.cpp, resblock.cpp): the sampler steps take FP32 /
// FP16 / BF16 with FP32 arithmetic and overwrite their outputs; the scalar
// coefficients are computed here exactly as the CPU computes them; noise
// fills a pre-sized FP32 Y with the same draws as every other backend.

#include "detail/kernels.h"
#include "detail/spatial.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct SamplerPush {
    std::uint64_t x, e, p, y, z;
    std::uint32_t n, dim, half_, hw, c;
    float k0, k1, k2, k3;
};

struct PhiloxPush {
    std::uint64_t y, key, counter;
    std::uint32_t n_lo, n_hi;
    float p, lo, hi;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

void run_sampler(const char* op, const Tensor& on, Dtype dt, std::uint32_t mode, SamplerPush pc, int dti = 0) {
    if (pc.n == 0) return;
    DeviceCtx& d = device_of(on);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::sampler_f32, dt, op),
                                        {mode, static_cast<std::uint32_t>(dti)});
    launch(d, k, pc, groups_1d(pc.n, k));
}

void same_shape(const char* op, const Tensor& a, const Tensor& b) {
    need(op, a.rows == b.rows && a.cols == b.cols, "shape mismatch");
}

std::size_t check_noise_y(const char* op, const Tensor& Y) {
    need(op, Y.dtype == Dtype::FP32, "Y must be FP32");
    need(op, Y.rows >= 0 && Y.cols >= 0, "Y has negative dimension");
    const std::size_t n = static_cast<std::size_t>(Y.rows) * static_cast<std::size_t>(Y.cols);
    need(op, n == 0 || Y.data != nullptr, "Y is uncommitted; pre-allocate before calling");
    return n;
}

void run_philox(const char* op, std::uint32_t mode, std::uint64_t key, std::uint64_t counter, float p, float lo,
                float hi, Tensor& Y) {
    const std::size_t n = check_noise_y(op, Y);
    if (n == 0) return;
    DeviceCtx& d = device_of(Y);
    const Kernel& k = d.pipelines().get(ShaderId::philox, {mode});
    PhiloxPush pc{addr(Y.data), key, counter, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n >> 32),
                  p, lo, hi};
    launch(d, k, pc, groups_1d(n, k));
}

}  // namespace

// ─── sampler steps ─────────────────────────────────────────────────────────

void ddim_step(const Tensor& x_t, const Tensor& eps_pred, float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev) {
    const char* op = "ddim_step";
    dt_code(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    const float sqrt_alpha_t = std::sqrt(alpha_t);
    SamplerPush pc{};
    pc.x = addr(x_t.data); pc.e = addr(eps_pred.data); pc.y = addr(x_prev.data);
    pc.n = count32(x_t, op);
    pc.k0 = std::sqrt(std::max(0.0f, alpha_prev));
    pc.k1 = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    pc.k2 = std::sqrt(std::max(0.0f, 1.0f - alpha_t));
    pc.k3 = std::sqrt(std::max(0.0f, 1.0f - alpha_prev - sigma_t * sigma_t));
    run_sampler(op, x_t, x_t.dtype, SMP_DDIM, pc);
}

void euler_step(const Tensor& x_t, const Tensor& eps_pred, float sigma_t, float sigma_prev, Tensor& x_prev) {
    const char* op = "euler_step";
    dt_code(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype, "x_t and eps_pred must have the same dtype");
    same_shape(op, x_t, eps_pred);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    SamplerPush pc{};
    pc.x = addr(x_t.data); pc.e = addr(eps_pred.data); pc.y = addr(x_prev.data);
    pc.n = count32(x_t, op);
    pc.k0 = sigma_prev - sigma_t;
    run_sampler(op, x_t, x_t.dtype, SMP_EULER, pc);
}

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred, const Tensor& x0_prev, float sigma_t, float c_xt,
                   float c_x0t, float c_x0prev, Tensor& x_prev, Tensor& x0_out) {
    const char* op = "dpmpp_2m_step";
    dt_code(x_t.dtype, op);
    need(op, eps_pred.dtype == x_t.dtype && x0_prev.dtype == x_t.dtype, "all inputs must have the same dtype");
    same_shape(op, x_t, eps_pred);
    same_shape(op, x_t, x0_prev);
    ensure(x_prev, x_t.rows, x_t.cols, x_t.dtype);
    ensure(x0_out, x_t.rows, x_t.cols, x_t.dtype);
    SamplerPush pc{};
    pc.x = addr(x_t.data); pc.e = addr(eps_pred.data); pc.p = addr(x0_prev.data);
    pc.y = addr(x_prev.data); pc.z = addr(x0_out.data);
    pc.n = count32(x_t, op);
    pc.k0 = sigma_t; pc.k1 = c_xt; pc.k2 = c_x0t; pc.k3 = c_x0prev;
    run_sampler(op, x_t, x_t.dtype, SMP_DPMPP_2M, pc);
}

// Y (N, dim) in Y's dtype when it already is FP32 / FP16 / BF16, else in the
// timesteps' dtype (the CPU rule).
void timestep_embedding(const Tensor& timesteps, int dim, float max_period, Tensor& Y) {
    const char* op = "timestep_embedding";
    const int dti = dt_code(timesteps.dtype, op);
    need(op, timesteps.cols == 1, "timesteps must be (N,1)");
    need(op, dim > 0, "dim must be positive");
    const Dtype out = (Y.dtype == Dtype::FP16 || Y.dtype == Dtype::BF16 || Y.dtype == Dtype::FP32) ? Y.dtype
                                                                                                    : timesteps.dtype;
    ensure(Y, timesteps.rows, dim, out);
    SamplerPush pc{};
    pc.x = addr(timesteps.data); pc.y = addr(Y.data);
    pc.n = count32(Y, op);
    pc.dim = static_cast<std::uint32_t>(dim);
    pc.half_ = static_cast<std::uint32_t>(dim / 2);
    pc.k0 = std::log(max_period);
    run_sampler(op, timesteps, out, SMP_TIMESTEP_EMB, pc, dti);
}

// ─── image preprocessing ───────────────────────────────────────────────────

void image_normalize(const Tensor& X, const Tensor& mean, const Tensor& std_, int N, int C, int H, int W, Tensor& Y) {
    const char* op = "image_normalize";
    dt_code(X.dtype, op);
    need(op, mean.dtype == Dtype::FP32 && std_.dtype == Dtype::FP32, "mean / std must be FP32");
    need(op, mean.size() == C && std_.size() == C, "mean/std must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    ensure(Y, N, static_cast<int>(cols), Dtype::FP32);
    SamplerPush pc{};
    pc.x = addr(X.data); pc.e = addr(mean.data); pc.p = addr(std_.data); pc.y = addr(Y.data);
    pc.n = count32(Y, op);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    if (X.dtype != Dtype::FP32) {   // FP32 output: widen first, then normalise in place
        Tensor xf = Tensor::empty_on(::brotensor::Device::vulkan(device_of(X).index()), X.rows, X.cols, Dtype::FP32);
        ::brotensor::cast(X, xf, Dtype::FP32);
        pc.x = addr(xf.data);
        run_sampler(op, X, Dtype::FP32, SMP_IMAGE_NORM, pc);
        return;
    }
    run_sampler(op, X, Dtype::FP32, SMP_IMAGE_NORM, pc);
}

// `src` is a Vulkan device address of N*H*W*C bytes (as on CUDA, a device
// pointer), e.g. the data of an INT8 tensor the image bytes were uploaded to.
void image_u8_to_f32_nhwc_to_nchw(const std::uint8_t* src, int N, int H, int W, int C, float scale, float bias,
                                  Tensor& Y) {
    const char* op = "image_u8_to_f32_nhwc_to_nchw";
    need(op, N >= 0 && H >= 0 && W >= 0 && C >= 0, "negative dim");
    need(op, src != nullptr || static_cast<long long>(N) * H * W * C == 0, "src is null");
    ensure(Y, N, C * H * W, Dtype::FP32);
    if (Y.size() == 0) return;
    need(op, Y.device.is_vulkan(), "Y must be a Vulkan tensor");
    SamplerPush pc{};
    pc.x = addr(src); pc.y = addr(Y.data);
    pc.n = count32(Y, op);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.k0 = scale; pc.k1 = bias;
    run_sampler(op, Y, Dtype::FP32, SMP_U8_NHWC, pc);
}

// ─── noise ─────────────────────────────────────────────────────────────────

void randn(std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    run_philox("randn", RNG_NORMAL, key, counter, 0, 0, 0, Y);
}

void rand_uniform(std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    run_philox("rand_uniform", RNG_UNIFORM, key, counter, 0, 0, 0, Y);
}

void rand_bernoulli(float p, std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    need("rand_bernoulli", p >= 0.0f && p <= 1.0f, "p must be in [0, 1]");
    run_philox("rand_bernoulli", RNG_BERNOULLI, key, counter, p, 0, 0, Y);
}

void randn_truncated(float lo, float hi, std::uint64_t key, std::uint64_t counter, Tensor& Y) {
    need("randn_truncated", lo < hi, "lo must be < hi");
    run_philox("randn_truncated", RNG_TRUNCATED, key, counter, 0, lo, hi, Y);
}

// ─── ResBlock ──────────────────────────────────────────────────────────────
//
// h1 = SiLU(GN(X)); h2 = conv3x3(h1) + b1 (+ t_emb_shift); h3 = SiLU(GN(h2));
// Y = skip + conv3x3(h3) + b2, skip = X or conv1x1(X) + bskip. GN + SiLU is
// one fused pass; a per-channel t_emb shift (or a per-sample one with N = 1)
// is folded into conv1's bias; the skip is written to Y first and conv2
// accumulates onto it, so no pass re-reads Y.
namespace {

// The ResBlock, with FP16 / BF16 / FP32 weights (scales null) or INT8 ones
// with per-output-channel FP32 scales (resblock_forward_int8w_fp16: X FP16).
void resblock(const char* op, const Tensor& X, const Tensor& gamma1, const Tensor& beta1, const Tensor& W1,
              const Tensor* s1, const Tensor* b1, const Tensor* t_emb_shift, const Tensor& gamma2,
              const Tensor& beta2, const Tensor& W2, const Tensor* s2, const Tensor* b2, const Tensor* Wskip,
              const Tensor* sskip, const Tensor* bskip, int N, int C_in, int C_out, int H, int W, int num_groups,
              float eps, Tensor& Y) {
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    const bool q8 = s1 != nullptr;
    need(op, num_groups > 0 && C_in % num_groups == 0 && C_out % num_groups == 0,
         "num_groups must divide C_in and C_out");
    need(op, Wskip != nullptr || C_in == C_out, "Wskip required when C_in != C_out");
    for (const Tensor* t : {&gamma1, &beta1, &gamma2, &beta2, b1, b2, bskip, t_emb_shift}) {
        need(op, t == nullptr || t->dtype == dt, "all parameters must share X's dtype");
    }
    for (const Tensor* t : {&W1, &W2, Wskip}) {
        need(op, t == nullptr || t->dtype == (q8 ? Dtype::INT8 : dt),
             q8 ? "W1 / W2 / Wskip must be INT8" : "all parameters must share X's dtype");
    }
    if (q8) {
        need(op, dt == Dtype::FP16, "X must be FP16");
        need(op, s2 != nullptr && (!Wskip || sskip != nullptr), "every INT8 weight needs its scales");
        for (const Tensor* t : {s1, s2, Wskip ? sskip : nullptr}) {
            need(op, t == nullptr || (t->dtype == Dtype::FP32 && t->size() == C_out),
                 "scales must be FP32 with C_out elements");
        }
    }
    need(op, gamma1.size() == C_in && beta1.size() == C_in, "gamma1/beta1 must have C_in elements");
    need(op, gamma2.size() == C_out && beta2.size() == C_out, "gamma2/beta2 must have C_out elements");
    need(op, W1.size() == 9LL * C_out * C_in && W2.size() == 9LL * C_out * C_out, "W1 / W2 must be 3x3 OIHW");
    need(op, !Wskip || Wskip->size() == static_cast<long long>(C_out) * C_in, "Wskip must be (C_out, C_in)");
    need(op, !b1 || b1->size() == C_out, "b1 must have C_out elements");
    need(op, !b2 || b2->size() == C_out, "b2 must have C_out elements");
    need(op, !bskip || bskip->size() == C_out, "bskip must have C_out elements");
    const long long hw = static_cast<long long>(H) * W;
    need(op, X.size() >= N * C_in * hw, "X is smaller than N*C_in*H*W");
    bool shift_n = false;
    if (t_emb_shift) {
        const Tensor& t = *t_emb_shift;
        if (t.rows == N && t.cols == C_out) shift_n = true;
        else if (!((t.rows == C_out && t.cols == 1) || (t.rows == 1 && t.cols == C_out) || t.size() == C_out))
            fail(op, "t_emb_shift shape must be (N, C_out) or (C_out,)");
        if (shift_n && N == 1) shift_n = false;
    }
    ensure(Y, N, static_cast<int>(C_out * hw), dt);
    if (N == 0 || hw == 0) return;
    DeviceCtx& d = device_of(X);
    const ::brotensor::Device dev = ::brotensor::Device::vulkan(d.index());

    Tensor h1 = Tensor::empty_on(dev, N, static_cast<int>(C_in * hw), dt);
    group_norm(d, addr(X.data), addr(gamma1.data), addr(beta1.data), addr(h1.data), dt, N, C_in,
               static_cast<int>(hw), num_groups, eps, true);

    // conv1's bias, with a per-channel shift folded in.
    Tensor bias1;
    std::uint64_t b1a = b1 ? addr(b1->data) : 0;
    if (t_emb_shift && !shift_n) {
        bias1 = Tensor::empty_on(dev, C_out, 1, dt);
        if (b1) {
            ::brotensor::copy_d2d(*b1, 0, bias1, 0, C_out);
            ::brotensor::add_inplace(bias1, *t_emb_shift);
        } else {
            ::brotensor::copy_d2d(*t_emb_shift, 0, bias1, 0, C_out);
        }
        b1a = addr(bias1.data);
    }
    Tensor h2 = Tensor::empty_on(dev, N, static_cast<int>(C_out * hw), dt);
    Conv2dArgs c1;
    c1.x = addr(h1.data); c1.w = addr(W1.data); c1.bias = b1a; c1.y = addr(h2.data); c1.dt = dt;
    c1.scale = s1 ? addr(s1->data) : 0;
    c1.n = N; c1.cin = C_in; c1.h = H; c1.wd = W; c1.cout = C_out; c1.kh = c1.kw = 3; c1.ph = c1.pw = 1;
    c1.op = op;
    conv2d(d, c1);
    if (shift_n) ::brotensor::add_channel_bias_inplace(h2, *t_emb_shift, N * C_out, static_cast<int>(hw));
    h1 = Tensor();

    group_norm(d, addr(h2.data), addr(gamma2.data), addr(beta2.data), addr(h2.data), dt, N, C_out,
               static_cast<int>(hw), num_groups, eps, true);

    if (Wskip) {
        Conv2dArgs cs;
        cs.x = addr(X.data); cs.w = addr(Wskip->data); cs.bias = bskip ? addr(bskip->data) : 0;
        cs.y = addr(Y.data); cs.dt = dt;
        cs.scale = sskip ? addr(sskip->data) : 0;
        cs.n = N; cs.cin = C_in; cs.h = H; cs.wd = W; cs.cout = C_out;
        cs.op = op;
        conv2d(d, cs);
    } else {
        ::brotensor::copy_d2d(X, 0, Y, 0, static_cast<int>(N * C_in * hw));
    }
    Conv2dArgs c2;
    c2.x = addr(h2.data); c2.w = addr(W2.data); c2.bias = b2 ? addr(b2->data) : 0; c2.y = addr(Y.data); c2.dt = dt;
    c2.scale = s2 ? addr(s2->data) : 0;
    c2.n = N; c2.cin = C_out; c2.h = H; c2.wd = W; c2.cout = C_out; c2.kh = c2.kw = 3; c2.ph = c2.pw = 1;
    c2.op = op;
    if (std::string(conv2d_path(d, c2)) == "direct") {   // narrow blocks: no accumulating kernel
        Tensor r = Tensor::empty_on(dev, N, static_cast<int>(C_out * hw), dt);
        c2.y = addr(r.data);
        conv2d(d, c2);
        ::brotensor::add_inplace(Y, r);
        return;
    }
    c2.accum = true;
    conv2d(d, c2);
}

}  // namespace

void resblock_forward(const Tensor& X, const Tensor& gamma1, const Tensor& beta1, const Tensor& W1, const Tensor* b1,
                      const Tensor* t_emb_shift, const Tensor& gamma2, const Tensor& beta2, const Tensor& W2,
                      const Tensor* b2, const Tensor* Wskip, const Tensor* bskip, int N, int C_in, int C_out, int H,
                      int W, int num_groups, float eps, Tensor& Y) {
    resblock("resblock_forward", X, gamma1, beta1, W1, nullptr, b1, t_emb_shift, gamma2, beta2, W2, nullptr, b2,
             Wskip, nullptr, bskip, N, C_in, C_out, H, W, num_groups, eps, Y);
}

void resblock_forward_int8w_fp16(const Tensor& X, const Tensor& gamma1, const Tensor& beta1, const Tensor& W1,
                                 const Tensor& s1, const Tensor* b1, const Tensor* t_emb_shift, const Tensor& gamma2,
                                 const Tensor& beta2, const Tensor& W2, const Tensor& s2, const Tensor* b2,
                                 const Tensor* Wskip, const Tensor* sskip, const Tensor* bskip, int N, int C_in,
                                 int C_out, int H, int W, int num_groups, float eps, Tensor& Y) {
    resblock("resblock_forward_int8w_fp16", X, gamma1, beta1, W1, &s1, b1, t_emb_shift, gamma2, beta2, W2, &s2, b2,
             Wskip, sskip, bskip, N, C_in, C_out, H, W, num_groups, eps, Y);
}

// The ResBlock backward, composed as the CPU reference (src/cpu/resblock.cpp)
// and the CUDA one (resblock.cu): recompute h1 = SiLU(GN1(X)), h2 = conv1(h1)
// (+ shift), h3 = SiLU(GN2(h2)), then conv2 / SiLU / GN2 / conv1 / SiLU / GN1
// backwards and the skip path. dX is overwritten; every parameter gradient
// (and dt_emb_shift) accumulates. FP32 / FP16 / BF16 (CUDA: 16-bit only),
// all operands in X's dtype.
void resblock_backward(const Tensor& X, const Tensor& gamma1, const Tensor& beta1, const Tensor& W1, const Tensor* b1,
                       const Tensor* t_emb_shift, const Tensor& gamma2, const Tensor& beta2, const Tensor& W2,
                       const Tensor* /*b2*/, const Tensor* Wskip, const Tensor* /*bskip*/, int N, int C_in, int C_out,
                       int H, int W, int num_groups, float eps, const Tensor& dY, Tensor& dX, Tensor& dGamma1,
                       Tensor& dBeta1, Tensor& dW1, Tensor* db1, Tensor* dt_emb_shift, Tensor& dGamma2,
                       Tensor& dBeta2, Tensor& dW2, Tensor* db2, Tensor* dWskip, Tensor* dbskip) {
    namespace bt = ::brotensor;
    constexpr const char* op = "resblock_backward";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, num_groups > 0 && C_in % num_groups == 0 && C_out % num_groups == 0,
         "num_groups must divide C_in and C_out");
    need(op, Wskip != nullptr || C_in == C_out, "Wskip required when C_in != C_out");
    for (const Tensor* t : {&dY, &gamma1, &beta1, &W1, &gamma2, &beta2, &W2, b1, Wskip, t_emb_shift}) {
        need(op, t == nullptr || t->dtype == dt, "all tensors must share X's dtype");
    }
    need(op, W1.size() == 9LL * C_out * C_in && W2.size() == 9LL * C_out * C_out, "W1 / W2 must be 3x3 OIHW");
    need(op, !Wskip || Wskip->size() == static_cast<long long>(C_out) * C_in, "Wskip must be (C_out, C_in)");
    const int hw = H * W;
    need(op, dY.rows == N && dY.cols == C_out * hw, "dY shape mismatch");
    need(op, X.size() >= static_cast<long long>(N) * C_in * hw, "X is smaller than N*C_in*H*W");
    bool shift_n = false;
    if (t_emb_shift) {
        const Tensor& t = *t_emb_shift;
        if (t.rows == N && t.cols == C_out) shift_n = true;
        else if (!((t.rows == C_out && t.cols == 1) || (t.rows == 1 && t.cols == C_out) || t.size() == C_out))
            fail(op, "t_emb_shift shape must be (N, C_out) or (C_out,)");
        if (dt_emb_shift) need(op, dt_emb_shift->dtype == dt && dt_emb_shift->size() == t.size(),
                               "dt_emb_shift must match t_emb_shift");
    }
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != C_in * hw || dX.dtype != dt) dX.resize(N, C_in * hw, dt);
    if (N == 0 || hw == 0) return;
    DeviceCtx& d = device_of(X);
    const ::brotensor::Device dev = ::brotensor::Device::vulkan(d.index());
    auto tmp = [&](int c) { return Tensor::empty_on(dev, N, c * hw, dt); };

    // Recompute the forward's intermediates.
    Tensor h1p = tmp(C_in), h1 = tmp(C_in), h2 = tmp(C_out), h3p = tmp(C_out), h3 = tmp(C_out);
    group_norm(d, addr(X.data), addr(gamma1.data), addr(beta1.data), addr(h1p.data), dt, N, C_in, hw, num_groups,
               eps, false);
    bt::silu_forward(h1p, h1);
    bt::conv2d_forward(h1, W1, b1, N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        if (shift_n) bt::add_channel_bias_inplace(h2, *t_emb_shift, N * C_out, hw);
        else
            for (int n = 0; n < N; ++n) {
                Tensor row = Tensor::view(dev, static_cast<char*>(h2.data) + std::size_t(n) * C_out * hw * ::brotensor::dtype_size_bytes(dt),
                                          1, C_out * hw, dt);
                bt::add_channel_bias_inplace(row, *t_emb_shift, C_out, hw);
            }
    }
    group_norm(d, addr(h2.data), addr(gamma2.data), addr(beta2.data), addr(h3p.data), dt, N, C_out, hw, num_groups,
               eps, false);
    bt::silu_forward(h3p, h3);

    // conv2, SiLU2, GN2.
    Tensor dh3, dh3p, dh2;
    bt::conv2d_backward_input(W2, dY, N, C_out, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, dh3);
    bt::conv2d_backward_weight(h3, dY, N, C_out, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) bt::conv2d_backward_bias(dY, N, C_out, H, W, *db2);
    bt::silu_backward(h3p, dh3, dh3p);
    bt::group_norm_backward(h2, gamma2, dh3p, N, C_out, H, W, num_groups, eps, dh2, dGamma2, dBeta2);

    // The time-embedding shift: dh2 summed over the pixels (and the batch).
    if (t_emb_shift && dt_emb_shift) {
        if (shift_n) {
            for (int n = 0; n < N; ++n) {
                Tensor row = Tensor::view(dev, static_cast<char*>(dh2.data) + std::size_t(n) * C_out * hw * ::brotensor::dtype_size_bytes(dt),
                                          1, C_out * hw, dt);
                Tensor out = Tensor::view(dev, static_cast<char*>(dt_emb_shift->data) + std::size_t(n) * C_out * ::brotensor::dtype_size_bytes(dt),
                                          C_out, 1, dt);
                bt::conv2d_backward_bias(row, 1, C_out, H, W, out);
            }
        } else {
            Tensor out = Tensor::view(dev, dt_emb_shift->data, C_out, 1, dt);
            bt::conv2d_backward_bias(dh2, N, C_out, H, W, out);
        }
    }

    // conv1, SiLU1, GN1 (dX overwritten), then the skip path.
    Tensor dh1, dh1p;
    bt::conv2d_backward_input(W1, dh2, N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, dh1);
    bt::conv2d_backward_weight(h1, dh2, N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) bt::conv2d_backward_bias(dh2, N, C_out, H, W, *db1);
    bt::silu_backward(h1p, dh1, dh1p);
    bt::group_norm_backward(X, gamma1, dh1p, N, C_in, H, W, num_groups, eps, dX, dGamma1, dBeta1);
    if (!Wskip) {
        bt::add_inplace(dX, dY);
        return;
    }
    Tensor dxs;
    bt::conv2d_backward_input(*Wskip, dY, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1, dxs);
    if (dWskip) bt::conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1, *dWskip);
    if (dbskip) bt::conv2d_backward_bias(dY, N, C_out, H, W, *dbskip);
    bt::add_inplace(dX, dxs);
}

void fill_vulkan_vtable_diffusion(::brotensor::detail::OpsVTable& v) {
    v.ddim_step = &ddim_step;
    v.euler_step = &euler_step;
    v.dpmpp_2m_step = &dpmpp_2m_step;
    v.timestep_embedding = &timestep_embedding;
    v.image_normalize = &image_normalize;
    v.image_u8_to_f32_nhwc_to_nchw = &image_u8_to_f32_nhwc_to_nchw;
    v.randn = &randn;
    v.rand_uniform = &rand_uniform;
    v.rand_bernoulli = &rand_bernoulli;
    v.randn_truncated = &randn_truncated;
    v.resblock_forward = &resblock_forward;
    v.resblock_forward_int8w_fp16 = &resblock_forward_int8w_fp16;
    v.resblock_backward = &resblock_backward;
}

}  // namespace brotensor::detail::vulkan
