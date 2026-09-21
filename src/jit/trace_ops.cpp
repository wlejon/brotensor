#include <brotensor/jit/trace.h>
#include "trace_dag.h"
#include <brotensor/ops.h>
#include "../cpu/cpu_jit.h"

#if BROTENSOR_HAS_CUDA
#include "../cuda/cuda_jit.h"
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace brotensor::jit {

namespace {

// Result shape of a binary op whose operands may differ in rank: the operand
// with more elements wins, so `gate * x` and `x * gate` both produce x's
// shape when gate is a (1, D) row or a (1, 1) scalar. The trace records the
// broadcast as an addressing mode on that operand; nothing is materialised.
const Tensor& binary_shape(const Tensor& a, const Tensor& b) {
    return (static_cast<int64_t>(b.rows) * b.cols >
            static_cast<int64_t>(a.rows) * a.cols)
               ? b
               : a;
}

bool same_shape(const Tensor& a, const Tensor& b) {
    return a.rows == b.rows && a.cols == b.cols;
}

// An operand the caller did not supply — the default-constructed Tensor that
// means "no gamma" / "no beta". A symbolic intermediate also has a null
// `data`, so emptiness alone does not answer the question.
bool absent(const Tensor& t) {
    return t.jit_slot < 0 && t.empty();
}

[[noreturn]] void no_eager_broadcast(const char* op) {
    throw std::runtime_error(
        std::string("brotensor::jit: eager ") + op +
        " does not broadcast; trace the expression (begin_trace/end_trace) or "
        "call the shaped op directly");
}

}  // namespace

Tensor operator+(const Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        const Tensor& r = binary_shape(a, b);
        return ctx.record_op(TraceOpKind::Add, {slot_a, slot_b}, 0.0f,
                             r.device, r.dtype, r.rows, r.cols);
    }
    if (!same_shape(a, b)) no_eager_broadcast("+");
    Tensor out = a.clone();
    brotensor::add_inplace(out, b);
    return out;
}

Tensor operator+(const Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::AddScalar, {slot_a}, s,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = a.clone();
    brotensor::add_scalar_inplace(out, s);
    return out;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        const Tensor& r = binary_shape(a, b);
        return ctx.record_op(TraceOpKind::Sub, {slot_a, slot_b}, 0.0f,
                             r.device, r.dtype, r.rows, r.cols);
    }
    if (!same_shape(a, b)) no_eager_broadcast("-");
    Tensor out = a.clone();
    brotensor::axpby_inplace(out, b, 1.0f, -1.0f);
    return out;
}

Tensor operator-(const Tensor& a, float s) {
    return a + (-s);
}

Tensor operator-(float s, const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::ScalarSub, {slot_a}, s,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = a.clone();
    brotensor::scale_inplace(out, -1.0f);
    brotensor::add_scalar_inplace(out, s);
    return out;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        const Tensor& r = binary_shape(a, b);
        return ctx.record_op(TraceOpKind::Mul, {slot_a, slot_b}, 0.0f,
                             r.device, r.dtype, r.rows, r.cols);
    }
    if (!same_shape(a, b)) {
        // The one eager broadcast brotensor already has a kernel for: a
        // length-D row against (L, D). Anything else is an error.
        const Tensor& big = (static_cast<int64_t>(b.rows) * b.cols >
                             static_cast<int64_t>(a.rows) * a.cols)
                                ? b
                                : a;
        const Tensor& small = (&big == &a) ? b : a;
        if (small.rows == 1 && small.cols == big.cols) {
            Tensor out;
            brotensor::broadcast_mul(big, small, out);
            return out;
        }
        no_eager_broadcast("*");
    }
    Tensor out = a.clone();
    brotensor::mul_inplace(out, b);
    return out;
}

Tensor operator*(const Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::MulScalar, {slot_a}, s,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = a.clone();
    brotensor::scale_inplace(out, s);
    return out;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        const Tensor& r = binary_shape(a, b);
        return ctx.record_op(TraceOpKind::Div, {slot_a, slot_b}, 0.0f,
                             r.device, r.dtype, r.rows, r.cols);
    }
    if (!same_shape(a, b)) no_eager_broadcast("/");
    Tensor out = a.clone();
    if (a.is_host()) {
        float* p_out = out.ptr();
        const float* p_b = b.ptr();
        int n = a.size();
        for (int i = 0; i < n; ++i) p_out[i] /= p_b[i];
    } else {
#if BROTENSOR_HAS_CUDA
        if (a.device.is_cuda()) {
            detail::cuda::jit::launch_elementwise_div_ptx(
                out.ptr(),
                b.ptr(),
                a.size()
            );
        } else {
            throw std::runtime_error("brotensor: elementwise eager div on GPU requires CUDA backend");
        }
#else
        throw std::runtime_error("brotensor: elementwise eager div on GPU requires CUDA backend");
#endif
    }
    return out;
}

Tensor operator/(const Tensor& a, float s) {
    return a * (1.0f / s);
}

Tensor& operator+=(Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        ctx.record_inplace_op(TraceOpKind::Add, slot_a, slot_b, 0.0f, a);
        return a;
    }
    brotensor::add_inplace(a, b);
    return a;
}

Tensor& operator+=(Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        ctx.record_inplace_op(TraceOpKind::AddScalar, slot_a, -1, s, a);
        return a;
    }
    brotensor::add_scalar_inplace(a, s);
    return a;
}

Tensor& operator*=(Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        ctx.record_inplace_op(TraceOpKind::Mul, slot_a, slot_b, 0.0f, a);
        return a;
    }
    brotensor::mul_inplace(a, b);
    return a;
}

Tensor& operator*=(Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        ctx.record_inplace_op(TraceOpKind::MulScalar, slot_a, -1, s, a);
        return a;
    }
    brotensor::scale_inplace(a, s);
    return a;
}

Tensor silu(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::SiLU, {slot_a}, 0.0f,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::silu_forward(a, out);
    return out;
}

Tensor gelu(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::GELU, {slot_a}, 0.0f,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::gelu_forward(a, out);
    return out;
}

Tensor relu(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::ReLU, {slot_a}, 0.0f,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::relu_forward(a, out);
    return out;
}

void store(Tensor& dst, const Tensor& src) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_src = ctx.get_or_register_slot(src);
        ctx.record_store(slot_src, dst);
        return;
    }
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != src.dtype) {
        throw std::runtime_error("brotensor::jit: store() needs dst and src to agree on "
                                 "shape and dtype");
    }
    brotensor::copy_d2d(src, 0, dst, 0, src.size());
}

Tensor tanh(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::Tanh, {slot_a}, 0.0f,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::tanh_forward(a, out);
    return out;
}

Tensor sigmoid(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        return ctx.record_op(TraceOpKind::Sigmoid, {slot_a}, 0.0f,
                             a.device, a.dtype, a.rows, a.cols);
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::sigmoid_forward(a, out);
    return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_x = ctx.get_or_register_slot(x);
        std::vector<int> inputs = {slot_x};
        if (!absent(gamma)) {
            inputs.push_back(ctx.get_or_register_slot(gamma));
        }
        return ctx.record_op(TraceOpKind::RMSNorm, inputs, eps,
                             x.device, x.dtype, x.rows, x.cols);
    }

    Tensor g = gamma;
    if (g.empty()) {
        std::vector<float> ones(x.cols, 1.0f);
        g = Tensor::from_host_on(x.device, ones.data(), 1, x.cols);
    }
    Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
    brotensor::rms_norm_forward(x, g, eps, out);
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_x = ctx.get_or_register_slot(x);
        std::vector<int> inputs = {slot_x};
        if (!absent(gamma)) inputs.push_back(ctx.get_or_register_slot(gamma));
        if (!absent(beta)) inputs.push_back(ctx.get_or_register_slot(beta));
        return ctx.record_op(TraceOpKind::LayerNorm, inputs, eps,
                             x.device, x.dtype, x.rows, x.cols);
    }

    Tensor g = gamma;
    if (g.empty()) {
        std::vector<float> ones(x.cols, 1.0f);
        g = Tensor::from_host_on(x.device, ones.data(), 1, x.cols);
    }
    Tensor b = beta;
    if (b.empty()) {
        b = Tensor::zeros_on(x.device, 1, x.cols);
    }
    Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
    brotensor::layernorm_forward_inference_batched(x, g, b, out, eps);
    return out;
}

Tensor modulate(const Tensor& x, const Tensor& scale, const Tensor& shift) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_x = ctx.get_or_register_slot(x);
        int slot_s = ctx.get_or_register_slot(scale);
        int slot_sh = ctx.get_or_register_slot(shift);
        return ctx.record_op(TraceOpKind::Modulate, {slot_x, slot_s, slot_sh}, 0.0f,
                             x.device, x.dtype, x.rows, x.cols);
    }

    Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
    if (x.is_host()) {
        detail::cpu::jit::modulate(x.ptr(), scale.ptr(), shift.ptr(), out.ptr(), x.rows, x.cols);
    } else {
#if BROTENSOR_HAS_CUDA
        detail::cuda::jit::launch_modulate_ptx(
            static_cast<const float*>(x.data),
            static_cast<const float*>(scale.data),
            static_cast<const float*>(shift.data),
            static_cast<float*>(out.data),
            x.rows, x.cols
        );
#else
        throw std::runtime_error("brotensor: modulate on GPU requires CUDA backend");
#endif
    }
    return out;
}

} // namespace brotensor::jit
