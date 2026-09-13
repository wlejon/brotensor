#include "brotensor/ops/fused.h"
#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "../cuda/cuda_jit.h"
#endif
#include "../cpu/cpu_jit.h"

#include <stdexcept>
#include <utility>

namespace brotensor {

void fused_residual_rmsnorm(Tensor& h, const Tensor& proj, const Tensor& gamma,
                            float eps, Tensor& out) {
    if (h.device != proj.device || h.device != gamma.device) {
        throw std::runtime_error("fused_residual_rmsnorm: device mismatch between operands");
    }
    if (h.size() != proj.size()) {
        throw std::runtime_error("fused_residual_rmsnorm: h and proj size mismatch");
    }
    int D = gamma.size();
    if (D <= 0 || (h.size() % D != 0)) {
        throw std::runtime_error("fused_residual_rmsnorm: invalid hidden dimension");
    }
    int B = h.size() / D;

    if (out.rows != h.rows || out.cols != h.cols ||
        out.dtype != h.dtype || out.device != h.device) {
        if (out.device != h.device) {
            out = Tensor::empty_on(h.device, h.rows, h.cols, h.dtype);
        } else {
            out.resize(h.rows, h.cols, h.dtype);
        }
    }

#ifdef BROTENSOR_HAS_CUDA
    if (h.device.type == DeviceType::CUDA &&
        h.dtype == Dtype::FP32 && proj.dtype == Dtype::FP32 && gamma.dtype == Dtype::FP32 &&
        detail::cuda::jit::is_cuda_jit_available()) {
        detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
            static_cast<float*>(h.data),
            static_cast<const float*>(proj.data),
            static_cast<const float*>(gamma.data),
            static_cast<float*>(out.data),
            B, D, eps, nullptr
        );
        return;
    }
#endif

    if (h.device.type == DeviceType::CPU &&
        h.dtype == Dtype::FP32 && proj.dtype == Dtype::FP32 && gamma.dtype == Dtype::FP32 &&
        detail::cpu::jit::is_jit_available()) {
        detail::cpu::jit::fused_residual_rmsnorm(
            static_cast<float*>(h.data),
            static_cast<const float*>(proj.data),
            static_cast<const float*>(gamma.data),
            eps,
            static_cast<float*>(out.data),
            B, D
        );
        return;
    }

    add_inplace(h, proj);
    rms_norm_forward(h, gamma, eps, out);
}

void fused_layernorm_modulate(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                              const Tensor& scale, const Tensor& shift, float eps,
                              Tensor& out) {
    if (x.device != gamma.device || x.device != beta.device ||
        x.device != scale.device || x.device != shift.device) {
        throw std::runtime_error("fused_layernorm_modulate: device mismatch between operands");
    }
    int D = gamma.size();
    if (D <= 0 || beta.size() != D || scale.size() != D || shift.size() != D ||
        (x.size() % D != 0)) {
        throw std::runtime_error("fused_layernorm_modulate: dimension mismatch");
    }
    int R = x.size() / D;

    if (out.rows != x.rows || out.cols != x.cols ||
        out.dtype != x.dtype || out.device != x.device) {
        if (out.device != x.device) {
            out = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
        } else {
            out.resize(x.rows, x.cols, x.dtype);
        }
    }

#ifdef BROTENSOR_HAS_CUDA
    if (x.device.type == DeviceType::CUDA &&
        x.dtype == Dtype::FP32 && gamma.dtype == Dtype::FP32 && beta.dtype == Dtype::FP32 &&
        scale.dtype == Dtype::FP32 && shift.dtype == Dtype::FP32 &&
        detail::cuda::jit::is_cuda_jit_available()) {
        detail::cuda::jit::launch_fused_layernorm_modulate_ptx(
            static_cast<const float*>(x.data),
            static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data),
            static_cast<const float*>(scale.data),
            static_cast<const float*>(shift.data),
            static_cast<float*>(out.data),
            R, D, eps, nullptr
        );
        return;
    }
#endif

    if (x.device.type == DeviceType::CPU &&
        x.dtype == Dtype::FP32 && gamma.dtype == Dtype::FP32 && beta.dtype == Dtype::FP32 &&
        scale.dtype == Dtype::FP32 && shift.dtype == Dtype::FP32 &&
        detail::cpu::jit::is_jit_available()) {
        detail::cpu::jit::fused_layernorm_modulate(
            static_cast<const float*>(x.data),
            static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data),
            static_cast<const float*>(scale.data),
            static_cast<const float*>(shift.data),
            eps,
            static_cast<float*>(out.data),
            R, D
        );
        return;
    }

    Tensor temp = Tensor::empty_on(x.device, x.rows, x.cols, x.dtype);
    if (x.dtype == Dtype::FP16) {
        layernorm_forward_inference_batched_fp16(x, gamma, beta, temp, eps);
    } else {
        layernorm_forward_inference_batched(x, gamma, beta, temp, eps);
    }
    modulate(temp, scale, shift, out);
}

void fused_gemv_swiglu(const Tensor& x, const Tensor& w_gate, const Tensor& w_up,
                       Tensor& out) {
    if (x.device != w_gate.device || x.device != w_up.device) {
        throw std::runtime_error("fused_gemv_swiglu: device mismatch between operands");
    }
    int N = w_gate.rows;
    int K = w_gate.cols;
    if (w_up.rows != N || w_up.cols != K || x.size() != K) {
        throw std::runtime_error("fused_gemv_swiglu: dimension mismatch");
    }
    int out_r = (x.rows == 1) ? 1 : N;
    int out_c = (x.rows == 1) ? N : 1;
    if (out.rows != out_r || out.cols != out_c ||
        out.dtype != x.dtype || out.device != x.device) {
        if (out.device != x.device) {
            out = Tensor::empty_on(x.device, out_r, out_c, x.dtype);
        } else {
            out.resize(out_r, out_c, x.dtype);
        }
    }

#ifdef BROTENSOR_HAS_CUDA
    if (x.device.type == DeviceType::CUDA &&
        x.dtype == Dtype::FP32 && w_gate.dtype == Dtype::FP32 && w_up.dtype == Dtype::FP32 &&
        detail::cuda::jit::is_cuda_jit_available()) {
        detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
            static_cast<const float*>(w_gate.data),
            static_cast<const float*>(w_up.data),
            static_cast<const float*>(x.data),
            static_cast<float*>(out.data),
            N, K, nullptr
        );
        return;
    }
#endif

    Tensor zero = Tensor::zeros_on(x.device, N, 1, x.dtype);
    Tensor up = Tensor::empty_on(x.device, (x.rows == 1 ? 1 : N), (x.rows == 1 ? N : 1), x.dtype);
    Tensor x_row = (x.rows == 1) ? x : Tensor::view(x.device, x.data, 1, K, x.dtype);
    linear_forward_batched(w_gate, zero, x_row, out);
    silu_forward(out, out);
    linear_forward_batched(w_up, zero, x_row, up);
    mul_inplace(out, up);
    out.rows = out_r;
    out.cols = out_c;
}

void fused_gemv_residual(const Tensor& x, const Tensor& w_down, const Tensor& res,
                         Tensor& out) {
    if (x.device != w_down.device || x.device != res.device) {
        throw std::runtime_error("fused_gemv_residual: device mismatch between operands");
    }
    int N = w_down.rows;
    int K = w_down.cols;
    if (x.size() != K || res.size() != N) {
        throw std::runtime_error("fused_gemv_residual: dimension mismatch");
    }
    if (out.rows != res.rows || out.cols != res.cols ||
        out.dtype != res.dtype || out.device != res.device) {
        if (out.device != res.device) {
            out = Tensor::empty_on(res.device, res.rows, res.cols, res.dtype);
        } else {
            out.resize(res.rows, res.cols, res.dtype);
        }
    }

#ifdef BROTENSOR_HAS_CUDA
    if (x.device.type == DeviceType::CUDA &&
        x.dtype == Dtype::FP32 && w_down.dtype == Dtype::FP32 && res.dtype == Dtype::FP32 &&
        detail::cuda::jit::is_cuda_jit_available()) {
        detail::cuda::jit::launch_fused_gemv_residual_ptx(
            static_cast<const float*>(w_down.data),
            static_cast<const float*>(x.data),
            static_cast<const float*>(res.data),
            static_cast<float*>(out.data),
            N, K, nullptr
        );
        return;
    }
#endif

    Tensor x_row = (x.rows == 1) ? x : Tensor::view(x.device, x.data, 1, K, x.dtype);
    Tensor bias_col = (res.cols == 1) ? res : Tensor::view(res.device, res.data, N, 1, res.dtype);
    if (out.data == res.data) {
        Tensor tmp = Tensor::empty_on(x.device, res.rows, res.cols, x.dtype);
        linear_forward_batched(w_down, bias_col, x_row, tmp);
        tmp.rows = res.rows;
        tmp.cols = res.cols;
        out = std::move(tmp);
    } else {
        linear_forward_batched(w_down, bias_col, x_row, out);
        out.rows = res.rows;
        out.cols = res.cols;
    }
}

} // namespace brotensor
