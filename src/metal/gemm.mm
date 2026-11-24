#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

void add_inplace(Tensor& a, const Tensor& b);  // elementwise.mm

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1 || y.dtype != Dtype::FP32) y.resize(out_dim, 1, Dtype::FP32);
    if (out_dim == 0) return;

    metal_impl::AbtMixed g;
    g.A = buffer_for(x); g.ofs_A = buffer_offset_for(x); g.lda = static_cast<uint64_t>(in_dim);
    g.B = buffer_for(W); g.ofs_B = buffer_offset_for(W); g.ldb = static_cast<uint64_t>(in_dim);
    g.C = buffer_for(y); g.ofs_C = buffer_offset_for(y); g.ldc = static_cast<uint64_t>(out_dim);
    g.bias = buffer_for(b); g.ofs_bias = buffer_offset_for(b);
    g.M = 1; g.N = out_dim; g.K = in_dim;
    g.in = g.out = metal_impl::kAbtF32;
    metal_impl::launch_matmul_abt_mixed(g);
}

void linear_backward(const Tensor& W, const Tensor& x,
                     const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (dX.rows != in_dim || dX.cols != 1 || dX.dtype != Dtype::FP32) dX.resize(in_dim, 1, Dtype::FP32);

    // dX = W^T * dY: (1, out) @ (out, in) -> (1, in)
    if (in_dim > 0 && out_dim > 0) {
        metal_impl::AbtMixed g;
        g.A = buffer_for(dY); g.ofs_A = buffer_offset_for(dY); g.lda = static_cast<uint64_t>(out_dim);
        g.B = buffer_for(W);  g.ofs_B = buffer_offset_for(W);  g.ldb = static_cast<uint64_t>(in_dim);
        g.transB = true;
        g.C = buffer_for(dX); g.ofs_C = buffer_offset_for(dX); g.ldc = static_cast<uint64_t>(in_dim);
        g.M = 1; g.N = in_dim; g.K = out_dim;
        g.in = g.out = metal_impl::kAbtF32;
        metal_impl::launch_matmul_abt_mixed(g);
    }

    // dW += dY * x^T: (out, 1) @ (1, in) -> (out, in)
    if (out_dim > 0 && in_dim > 0) {
        metal_impl::AbtMixed g;
        g.A = buffer_for(dY); g.ofs_A = buffer_offset_for(dY); g.lda = 1;
        g.B = buffer_for(x);  g.ofs_B = buffer_offset_for(x);  g.ldb = static_cast<uint64_t>(in_dim);
        g.transB = true;
        g.C = buffer_for(dW); g.ofs_C = buffer_offset_for(dW); g.ldc = static_cast<uint64_t>(in_dim);
        g.M = out_dim; g.N = in_dim; g.K = 1;
        g.in = g.out = metal_impl::kAbtF32;
        g.epilogue = metal_impl::kAbtAccumulate;
        metal_impl::launch_matmul_abt_mixed(g);
    }

    // dB += dY
    if (out_dim > 0) {
        add_inplace(dB, dY);
    }
}

namespace {

void run_linear_16(const char* op, const Tensor& W, const Tensor* bias, const Tensor& X_BD, int act,
                   Tensor& Y_BD) {
    if ((W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16) || X_BD.dtype != W.dtype) {
        throw std::runtime_error(std::string(op) + ": W and X must both be FP16 or both be BF16");
    }
    if (bias && bias->dtype != W.dtype) {
        throw std::runtime_error(std::string(op) + ": bias dtype must match W");
    }
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error(std::string(op) + ": shape mismatch (W.cols != X.cols)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != X_BD.dtype) {
        Y_BD.resize(B, out_dim, X_BD.dtype);
    }
    if (B == 0 || out_dim == 0) return;

    metal_impl::AbtMixed g;
    g.A = buffer_for(X_BD); g.ofs_A = buffer_offset_for(X_BD); g.lda = static_cast<uint64_t>(in_dim);
    g.B = buffer_for(W);    g.ofs_B = buffer_offset_for(W);    g.ldb = static_cast<uint64_t>(in_dim);
    g.C = buffer_for(Y_BD); g.ofs_C = buffer_offset_for(Y_BD); g.ldc = static_cast<uint64_t>(out_dim);
    if (bias && bias->size() > 0) {
        g.bias = buffer_for(*bias);
        g.ofs_bias = buffer_offset_for(*bias);
    }
    g.M = B;
    g.N = out_dim;
    g.K = in_dim;
    g.in = g.out = metal_impl::abt_type(W.dtype);
    g.act = act;
    metal_impl::launch_matmul_abt_mixed(g);
}

}  // namespace

void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD) {
    run_linear_16("linear_forward_batched_fp16", W, bias, X_BD, 0, Y_BD);
}

void linear_forward_batched_fp16_act(const Tensor& W, const Tensor* bias,
                                     const Tensor& X_BD, int act, Tensor& Y_BD) {
    run_linear_16("linear_forward_batched_fp16_act", W, bias, X_BD, act, Y_BD);
}

}  // namespace brotensor::detail::metal
