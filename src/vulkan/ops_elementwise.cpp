// Vulkan elementwise ops: the unary activations and math functions, the
// in-place binary / scalar family, the activation backwards and the bias adds.
// FP32 / FP16 / BF16 storage, FP32 arithmetic; contracts follow the CUDA / HIP
// backends (output resized to the input's shape and dtype, backwards overwrite
// dX, binary ops require equal sizes and dtypes except FP32 += FP16 in
// add_inplace). Kernels: shaders/unary.comp, binary.comp, act_bwd.comp,
// bias.comp.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct UnaryPush  { std::uint64_t x, y; std::uint32_t n; float a, b; };
struct BinaryPush { std::uint64_t y, x; std::uint32_t n; float a, b; };
struct GradPush   { std::uint64_t p, dy, dx; std::uint32_t n; float a; };
struct BiasPush   { std::uint64_t y, bias; std::uint32_t n, d, l; };

void unary(const char* op, const Tensor& x, Tensor& y, std::uint32_t code,
           float a = 0.0f, float b = 0.0f) {
    dt_code(x.dtype, op);
    if (&x != &y && (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const std::uint32_t n = count32(x, op);
    if (n == 0) return;
    DeviceCtx& d = device_of(x);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::unary_f32, x.dtype, op), {code});
    const UnaryPush pc{addr(x.data), addr(y.data), n, a, b};
    launch(d, k, pc, groups_1d(n, k));
}

void inplace_scalar(const char* op, Tensor& y, std::uint32_t code, float a, float b = 0.0f) {
    unary(op, y, y, code, a, b);
}

void binary(const char* op, Tensor& y, const Tensor& x, std::uint32_t code,
            float a = 0.0f, float b = 0.0f) {
    if (y.size() != x.size()) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": size mismatch y=(" +
                                 std::to_string(y.rows) + "," + std::to_string(y.cols) + ") x=(" +
                                 std::to_string(x.rows) + "," + std::to_string(x.cols) + ")");
    }
    ShaderId id;
    if (y.dtype == x.dtype) {
        id = dt_variant(ShaderId::binary_f32, y.dtype, op);
    } else if (code == BOP_ADD && y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16) {
        id = ShaderId::binary_f32_f16;
    } else {
        throw std::runtime_error(std::string("brotensor: ") + op + ": dtype mismatch");
    }
    const std::uint32_t n = count32(y, op);
    if (n == 0) return;
    DeviceCtx& d = device_of(y);
    const Kernel& k = d.pipelines().get(id, {code});
    const BinaryPush pc{addr(y.data), addr(x.data), n, a, b};
    launch(d, k, pc, groups_1d(n, k));
}

// dX = g(p, dY). `p` may be null (GOP_PASS).
void act_backward(const char* op, const Tensor* p, const Tensor& dY, Tensor& dX,
                  std::uint32_t code, float a = 0.0f) {
    dt_code(dY.dtype, op);
    if (p) {
        if (p->dtype != dY.dtype) {
            throw std::runtime_error(std::string("brotensor: ") + op + ": dY.dtype must match the input's");
        }
        if (p->size() != dY.size()) {
            throw std::runtime_error(std::string("brotensor: ") + op + ": dY size must match the input's");
        }
    }
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    const std::uint32_t n = count32(dY, op);
    if (n == 0) return;
    DeviceCtx& d = device_of(dY);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::act_bwd_f32, dY.dtype, op), {code});
    const GradPush pc{p ? addr(p->data) : 0, addr(dY.data), addr(dX.data), n, a};
    launch(d, k, pc, groups_1d(n, k));
}

void bias_add(const char* op, Tensor& y, const Tensor& bias, std::uint32_t mode,
              std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    if (y.dtype != bias.dtype) throw std::runtime_error(std::string("brotensor: ") + op + ": dtype mismatch");
    if (n == 0 || dim == 0) return;
    DeviceCtx& d = device_of(y);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::bias_f32, y.dtype, op), {mode});
    const BiasPush pc{addr(y.data), addr(bias.data), n, dim, l};
    launch(d, k, pc, groups_1d(n, k));
}

}  // namespace

// ─── forward activations / math ────────────────────────────────────────────

#define BT_VK_UNARY(fn, code) \
    void fn(const Tensor& x, Tensor& y) { unary(#fn, x, y, code); }
BT_VK_UNARY(relu_forward, UOP_RELU)
BT_VK_UNARY(tanh_forward, UOP_TANH)
BT_VK_UNARY(sigmoid_forward, UOP_SIGMOID)
BT_VK_UNARY(silu_forward, UOP_SILU)
BT_VK_UNARY(gelu_forward, UOP_GELU_TANH)
BT_VK_UNARY(gelu_exact_forward, UOP_GELU_EXACT)
BT_VK_UNARY(quick_gelu_forward, UOP_QUICK_GELU)
BT_VK_UNARY(exp_forward, UOP_EXP)
BT_VK_UNARY(log_forward, UOP_LOG)
BT_VK_UNARY(sin_forward, UOP_SIN)
BT_VK_UNARY(cos_forward, UOP_COS)
BT_VK_UNARY(rsqrt_forward, UOP_RSQRT)
BT_VK_UNARY(round_forward, UOP_ROUND)
#undef BT_VK_UNARY

// Qualified calls: unqualified, ADL on Tensor would also find brotensor::<op>.
void relu_forward_batched(const Tensor& X, Tensor& Y) { vulkan::relu_forward(X, Y); }
void tanh_forward_batched(const Tensor& X, Tensor& Y) { vulkan::tanh_forward(X, Y); }
void elu_forward(const Tensor& x, float alpha, Tensor& y) { unary("elu_forward", x, y, UOP_ELU, alpha); }
void leaky_relu_forward(const Tensor& x, float slope, Tensor& y) {
    unary("leaky_relu_forward", x, y, UOP_LEAKY_RELU, slope);
}

// ─── in-place scalar / binary ──────────────────────────────────────────────

void add_scalar_inplace(Tensor& y, float s) { inplace_scalar("add_scalar_inplace", y, UOP_ADD_SCALAR, s); }
void scale_inplace(Tensor& y, float s) { inplace_scalar("scale_inplace", y, UOP_SCALE, s); }
void clamp(Tensor& y, float lo, float hi) { inplace_scalar("clamp", y, UOP_CLAMP, lo, hi); }

void add_inplace(Tensor& y, const Tensor& x) { binary("add_inplace", y, x, BOP_ADD); }
void add_inplace_batched(Tensor& Y, const Tensor& X) { binary("add_inplace_batched", Y, X, BOP_ADD); }
void mul_inplace(Tensor& y, const Tensor& x) { binary("mul_inplace", y, x, BOP_MUL); }
void div_inplace(Tensor& y, const Tensor& x) { binary("div_inplace", y, x, BOP_DIV); }
void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    binary("axpby_inplace", y, x, BOP_AXPBY, a, b);
}

// ─── backwards ─────────────────────────────────────────────────────────────

#define BT_VK_GRAD(fn, code) \
    void fn(const Tensor& p, const Tensor& dY, Tensor& dX) { act_backward(#fn, &p, dY, dX, code); }
BT_VK_GRAD(relu_backward, GOP_RELU)
BT_VK_GRAD(tanh_backward, GOP_TANH)
BT_VK_GRAD(sigmoid_backward, GOP_SIGMOID)
BT_VK_GRAD(silu_backward, GOP_SILU)
BT_VK_GRAD(gelu_backward, GOP_GELU_TANH)
BT_VK_GRAD(gelu_exact_backward, GOP_GELU_EXACT)
BT_VK_GRAD(quick_gelu_backward, GOP_QUICK_GELU)
BT_VK_GRAD(exp_backward, GOP_EXP)
BT_VK_GRAD(log_backward, GOP_LOG)
BT_VK_GRAD(sin_backward, GOP_SIN)
BT_VK_GRAD(cos_backward, GOP_COS)
BT_VK_GRAD(rsqrt_backward, GOP_RSQRT)
#undef BT_VK_GRAD

void relu_backward_batched(const Tensor& X, const Tensor& dY, Tensor& dX) { vulkan::relu_backward(X, dY, dX); }
void tanh_backward_batched(const Tensor& Y, const Tensor& dY, Tensor& dX) { vulkan::tanh_backward(Y, dY, dX); }
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX) {
    act_backward("elu_backward", &x, dY, dX, GOP_ELU, alpha);
}
void leaky_relu_backward(const Tensor& x, const Tensor& dY, float slope, Tensor& dX) {
    act_backward("leaky_relu_backward", &x, dY, dX, GOP_LEAKY_RELU, slope);
}
void round_backward(const Tensor& dY, Tensor& dX) {
    act_backward("round_backward", nullptr, dY, dX, GOP_PASS);
}

// ─── bias ──────────────────────────────────────────────────────────────────

void add_row_bias_inplace(Tensor& Y, const Tensor& bias) {
    if (bias.size() != Y.cols) throw std::runtime_error("brotensor: add_row_bias_inplace: bias size != Y.cols");
    bias_add("add_row_bias_inplace", Y, bias, BIAS_ROW, count32(Y, "add_row_bias_inplace"),
             static_cast<std::uint32_t>(Y.cols), 1);
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    if (C < 0 || L < 0) throw std::runtime_error("brotensor: add_channel_bias_inplace: negative dimension");
    if (static_cast<long long>(C) * L > y.size() || bias.size() < C) {
        throw std::runtime_error("brotensor: add_channel_bias_inplace: C * L exceeds y or C exceeds bias");
    }
    bias_add("add_channel_bias_inplace", y, bias, BIAS_CHANNEL, static_cast<std::uint32_t>(C) * L,
             static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(L));
}

void fill_vulkan_vtable_elementwise(::brotensor::detail::OpsVTable& v) {
    v.relu_forward = &relu_forward;
    v.relu_forward_batched = &relu_forward_batched;
    v.relu_backward = &relu_backward;
    v.relu_backward_batched = &relu_backward_batched;
    v.tanh_forward = &tanh_forward;
    v.tanh_forward_batched = &tanh_forward_batched;
    v.tanh_backward = &tanh_backward;
    v.tanh_backward_batched = &tanh_backward_batched;
    v.sigmoid_forward = &sigmoid_forward;
    v.sigmoid_backward = &sigmoid_backward;
    v.silu_forward = &silu_forward;
    v.silu_backward = &silu_backward;
    v.gelu_forward = &gelu_forward;
    v.gelu_backward = &gelu_backward;
    v.gelu_exact_forward = &gelu_exact_forward;
    v.gelu_exact_backward = &gelu_exact_backward;
    v.quick_gelu_forward = &quick_gelu_forward;
    v.quick_gelu_backward = &quick_gelu_backward;
    v.exp_forward = &exp_forward;
    v.exp_backward = &exp_backward;
    v.log_forward = &log_forward;
    v.log_backward = &log_backward;
    v.sin_forward = &sin_forward;
    v.sin_backward = &sin_backward;
    v.cos_forward = &cos_forward;
    v.cos_backward = &cos_backward;
    v.rsqrt_forward = &rsqrt_forward;
    v.rsqrt_backward = &rsqrt_backward;
    v.round_forward = &round_forward;
    v.round_backward = &round_backward;
    v.elu_forward = &elu_forward;
    v.elu_backward = &elu_backward;
    v.leaky_relu_forward = &leaky_relu_forward;
    v.leaky_relu_backward = &leaky_relu_backward;

    v.add_scalar_inplace = &add_scalar_inplace;
    v.scale_inplace = &scale_inplace;
    v.clamp = &clamp;
    v.add_inplace = &add_inplace;
    v.add_inplace_batched = &add_inplace_batched;
    v.mul_inplace = &mul_inplace;
    v.div_inplace = &div_inplace;
    v.axpby_inplace = &axpby_inplace;

    v.add_row_bias_inplace = &add_row_bias_inplace;
    v.add_channel_bias_inplace = &add_channel_bias_inplace;
}

}  // namespace brotensor::detail::vulkan
