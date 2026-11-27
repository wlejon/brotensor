#include <brotensor/runtime.h>

#include "cu_mir_cache.h"
#include "fp16_internal.cuh"
#include "ops_gemm_mir.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

inline void require_fp(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

inline void require_same_dtype(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                               const char* op) {
    if (a.dtype != b.dtype) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": all FP operand dtypes must match");
    }
}

void linear_forward_batched_fp16_impl(const ::brotensor::Tensor& W,
                                      const ::brotensor::Tensor* bias,
                                      const ::brotensor::Tensor& X_BD,
                                      int act,
                                      ::brotensor::Tensor& Y_BD) {
    const Dtype dt = X_BD.dtype;
    if ((dt != Dtype::FP16 && dt != Dtype::BF16) || W.dtype != dt) {
        throw std::runtime_error(
            "linear_forward_batched_fp16: W and X must both be FP16 or both BF16");
    }
    if (bias && bias->dtype != dt) {
        throw std::runtime_error(
            "linear_forward_batched_fp16: bias dtype must match X/W");
    }
    const int B = X_BD.rows;
    const int in_dim = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != dt) {
        Y_BD.resize(B, out_dim, dt);
    }
    if (B == 0 || out_dim == 0) return;

    const bool has_bias = bias && bias->size() > 0;
    if (B <= 4 && in_dim > 0) {
        dispatch_mir_gemv(X_BD.data, dt, W.data, dt, Y_BD.data, dt,
                          has_bias ? bias->data : nullptr,
                          B, out_dim, in_dim, in_dim, in_dim, out_dim, 0, act);
        return;
    }

    if (dt == Dtype::FP16) {
        fp16_internal::launch_matmul_ABT_act(
            static_cast<const __half*>(X_BD.data),
            static_cast<const __half*>(W.data),
            static_cast<__half*>(Y_BD.data),
            B, out_dim, in_dim,
            has_bias ? static_cast<const __half*>(bias->data) : nullptr, act);
    } else {
        fp16_internal::launch_matmul_ABT_act(
            static_cast<const __nv_bfloat16*>(X_BD.data),
            static_cast<const __nv_bfloat16*>(W.data),
            static_cast<__nv_bfloat16*>(Y_BD.data),
            B, out_dim, in_dim,
            has_bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr, act);
    }
}

}  // namespace

void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp(W, "linear_forward", "W");
    require_same_dtype(W, b, "linear_forward");
    require_same_dtype(W, x, "linear_forward");
    const int out_dim = W.rows;
    const int in_dim = W.cols;
    if (y.rows != out_dim || y.cols != 1 || y.dtype != W.dtype) {
        y.resize(out_dim, 1, W.dtype);
    }
    if (out_dim == 0) return;

    dispatch_mir_gemv(x.data, x.dtype, W.data, W.dtype, y.data, y.dtype, b.data,
                      1, out_dim, in_dim, in_dim, in_dim, out_dim, 0, 0);
}

void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD) {
    linear_forward_batched_fp16_impl(W, bias, X_BD, 0, Y_BD);
}

void linear_forward_batched_fp16_act(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     int act,
                                     ::brotensor::Tensor& Y_BD) {
    linear_forward_batched_fp16_impl(W, bias, X_BD, act, Y_BD);
}

void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB) {
    require_fp(W, "linear_backward", "W");
    require_same_dtype(W, x, "linear_backward");
    require_same_dtype(W, dY, "linear_backward");
    require_same_dtype(W, dW, "linear_backward");
    require_same_dtype(W, dB, "linear_backward");
    const int out_dim = W.rows;
    const int in_dim = W.cols;
    const Dtype dt = W.dtype;

    if (dX.rows != in_dim || dX.cols != 1 || dX.dtype != dt) {
        dX.resize(in_dim, 1, dt);
    }

    if (in_dim > 0) {
        CUfunction fn = mir_function({MirKind::LinearBwd, 0u, dt, dt, 1u, dt, 0u});
        const std::uintptr_t ptrs[] = {
            reinterpret_cast<std::uintptr_t>(W.data),
            reinterpret_cast<std::uintptr_t>(dY.data),
            reinterpret_cast<std::uintptr_t>(dX.data),
        };
        const std::int32_t ints[] = {out_dim, in_dim};
        const std::uint32_t gx = static_cast<std::uint32_t>((in_dim + 127) / 128);
        mir_dispatch_args(fn, gx, 128u, ptrs, ints, {});
    }

    if (out_dim > 0 && in_dim > 0) {
        CUfunction fn = mir_function({MirKind::LinearBwd, 1u, dt, dt, 1u, dt, 0u});
        const std::uintptr_t ptrs[] = {
            reinterpret_cast<std::uintptr_t>(dY.data),
            reinterpret_cast<std::uintptr_t>(x.data),
            reinterpret_cast<std::uintptr_t>(dW.data),
        };
        const std::int32_t ints[] = {out_dim, in_dim};
        const std::uint64_t total = static_cast<std::uint64_t>(out_dim) * in_dim;
        const std::uint32_t gx = static_cast<std::uint32_t>((total + 255u) / 256u);
        mir_dispatch_args(fn, gx, 256u, ptrs, ints, {});
    }

    if (out_dim > 0) {
        CUfunction fn = mir_function({MirKind::LinearBwd, 2u, dt, dt, 1u, dt, 0u});
        const std::uintptr_t ptrs[] = {
            reinterpret_cast<std::uintptr_t>(dY.data),
            reinterpret_cast<std::uintptr_t>(dB.data),
        };
        const std::int32_t ints[] = {out_dim};
        const std::uint32_t gx = static_cast<std::uint32_t>((out_dim + 255) / 256);
        mir_dispatch_args(fn, gx, 256u, ptrs, ints, {});
    }
}

}  // namespace brotensor::detail::cuda
