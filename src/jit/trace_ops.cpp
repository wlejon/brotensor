#include <brotensor/jit/trace.h>
#include "trace_dag.h"
#include <brotensor/ops.h>
#include "../cpu/cpu_jit.h"

#if BROTENSOR_HAS_CUDA
#include "../cuda/cuda_jit.h"
#endif

#include <vector>
#include <stdexcept>

namespace brotensor::jit {

Tensor operator+(const Tensor& a, const Tensor& b) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        int slot_b = ctx.get_or_register_slot(b);
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::Add, {slot_a, slot_b}, 0.0f, out);
        return out;
    }
    Tensor out = a.clone();
    brotensor::add_inplace(out, b);
    return out;
}

Tensor operator+(const Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::AddScalar, {slot_a}, s, out);
        return out;
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
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::Sub, {slot_a, slot_b}, 0.0f, out);
        return out;
    }
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
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::ScalarSub, {slot_a}, s, out);
        return out;
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
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::Mul, {slot_a, slot_b}, 0.0f, out);
        return out;
    }
    Tensor out = a.clone();
    brotensor::mul_inplace(out, b);
    return out;
}

Tensor operator*(const Tensor& a, float s) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::MulScalar, {slot_a}, s, out);
        return out;
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
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::Div, {slot_a, slot_b}, 0.0f, out);
        return out;
    }
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
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::SiLU, {slot_a}, 0.0f, out);
        return out;
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::silu_forward(a, out);
    return out;
}

Tensor gelu(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::GELU, {slot_a}, 0.0f, out);
        return out;
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::gelu_forward(a, out);
    return out;
}

Tensor relu(const Tensor& a) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_a = ctx.get_or_register_slot(a);
        Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
        ctx.record_op(TraceOpKind::ReLU, {slot_a}, 0.0f, out);
        return out;
    }
    Tensor out = Tensor::empty_on(a.device, a.rows, a.cols, a.dtype);
    brotensor::relu_forward(a, out);
    return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    if (is_tracing()) {
        auto& ctx = TraceContext::current();
        int slot_x = ctx.get_or_register_slot(x);
        std::vector<int> inputs = {slot_x};
        if (!gamma.empty()) {
            inputs.push_back(ctx.get_or_register_slot(gamma));
        }
        Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
        ctx.record_op(TraceOpKind::RMSNorm, inputs, eps, out);
        return out;
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
        if (!gamma.empty()) inputs.push_back(ctx.get_or_register_slot(gamma));
        if (!beta.empty()) inputs.push_back(ctx.get_or_register_slot(beta));
        Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
        ctx.record_op(TraceOpKind::LayerNorm, inputs, eps, out);
        return out;
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
        Tensor out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
        ctx.record_op(TraceOpKind::Modulate, {slot_x, slot_s, slot_sh}, 0.0f, out);
        return out;
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
