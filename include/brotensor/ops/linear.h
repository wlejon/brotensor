#pragma once

// brotensor ops/linear.h — Dense / GEMM: linear + matmul (forward, backward, batched, fp16).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {

// y = W*x + b.  W:(out,in)  b:(out,1)  x:(in,1)  y:(out,1) (resized).
void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y);


// Backward of linear_forward. W:(out,in), x:(in,1), dY:(out,1) forward
// weights / input / upstream. dX:(in,1) overwritten. dW:(out,in), dB:(out,1)
// accumulated — caller zeros.
void linear_backward(const Tensor& W, const Tensor& x,
                     const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB);


// B independent forward passes in a single launch, forward-only. Tensors
// carrying B rows are (B, D) row-major: row b holds the b'th sample.

// Y[b,:] = W*X[b,:] + bias for b in [0,B).
//   W: (out,in).  bias: (out,1).  X_BD: (B,in).  Y_BD: (B,out) resized.
// W may be FP32, FP16 or BF16; bias/X/Y are FP32 and accumulation is FP32
// regardless. 16-bit W halves the weight-read bandwidth — the floor of the
// B<=2 autoregressive-decode regime — without touching the FP32 activation
// stream. (For 16-bit activations too, see linear_forward_batched_fp16.)
void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD);


// Linear backward over a B-row minibatch. Dtype-dispatched (FP32/FP16); all
// tensors share dtype.
//   dX[b] = W^T*dY[b];  dW += sum_b dY[b]*X[b]^T;  dB += sum_b dY[b].
//   W: (out,in) forward weights.  X_BD: (B,in) forward input.
//   dY_BD: (B,out) upstream.  dX_BD: (B,in) overwritten (resized + dtype-set).
//   dW: (out,in), dB: (out,1) accumulated — caller zeros.
void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB);


// 16-bit batched linear forward, inference-only. Like linear_forward_batched but
// FP16 *or* BF16 storage throughout — W, X, bias and the produced Y all share
// the operand dtype (FP16 or BF16); the op dispatches internally. (Kept under
// the historical `_fp16` name for ABI; BF16 is fully supported.)
//   W: (out,in).  bias: (out,1) or null.  X_BD: (B,in).  Y_BD: (B,out) resized.
void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD);


// Activation applied in the GEMM epilogue (fused into the output write) by
// linear_forward_batched_fp16_act. Plain-int values so the op surface stays
// int-typed (mirrors the `int mode` convention); kept in sync with the kernel
// selector in src/cuda/detail/activations.cuh.
enum LinearActivation {
    kLinearActNone      = 0,
    kLinearActRelu      = 1,
    kLinearActGeluTanh  = 2,
    kLinearActGeluExact = 3,
    kLinearActSilu      = 4,
    kLinearActQuickGelu = 5,
};

// As linear_forward_batched_fp16 (FP16 or BF16 storage), but fuses bias +
// activation `act` (a LinearActivation value) into the matmul's output-store
// stage — no separate bias-add or activation launch, and no extra HBM
// round-trips over Y.
//   W: (out,in).  bias: (out,1) or null.  X_BD: (B,in).  Y_BD: (B,out) resized.
void linear_forward_batched_fp16_act(const Tensor& W, const Tensor* bias,
                                     const Tensor& X_BD, int act, Tensor& Y_BD);


// Row-major matrix multiply, no bias: C(M,N) = A(M,K) @ B(K,N).
// Dispatched on A.dtype; B and C share it (C resized + dtype-set to match A).
// FP32 accumulation for both the FP32 and FP16 paths.
void matmul(const Tensor& A, const Tensor& B, Tensor& C);


// Batched A @ B^T, 16-bit (FP16 or BF16) with FP32 accumulation. For each
// b in [0,batch):  C[b](M,N) = A[b](M,K) @ B[b](N,K)^T, with the given element
// strides between batch slices (set strides = M*K, N*K, M*N for tightly packed;
// batch=1 + zero strides for the non-batched case). bias may be null; act is a
// LinearActivation value fused into the epilogue. A, B, C share dtype (FP16 or
// BF16); C is NOT auto-resized — caller pre-sizes/dtypes it.
void matmul_abt(const Tensor& A, const Tensor& B, Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const Tensor* bias, int act);


// Backward of matmul. For C = A @ B:
//   dA(M,K) += dC(M,N) @ B^T;   dB(K,N) += A^T @ dC(M,N).
// Dtype-dispatched (FP32/FP16); all five tensors share dtype. dC is read-only;
// dA, dB are accumulated — caller pre-sizes and zeros (mirrors linear_backward).
//   A: (M,K) forward input.  B: (K,N) forward weight.  dC: (M,N) upstream.
void matmul_backward(const Tensor& A,
                     const Tensor& B,
                     const Tensor& dC,
                     Tensor& dA,
                     Tensor& dB);


// W8A16 batched linear: Y(B,out) = X(B,in) @ dequant(W_int8)^T + bias. Same
// (B,in)->(B,out) layout as linear_forward_batched_fp16. FP32 accumulation.
// Activations may be FP16 *or* BF16 — X and bias must share the dtype and Y is
// produced in it; the op dispatches internally. (Kept under the historical
// `_fp16` name for ABI, same naming quirk as linear_forward_batched_fp16.)
//   W_int8: (out,in) INT8.  scales: (out,1) FP32 per-output-row dequant scales.
//   bias: (out,1) or (1,out) FP16/BF16, or null.  X_BD: (B,in) FP16 or BF16.
//   Y_BD: (B,out) same dtype as X, resized as needed.
void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD,
                                       Tensor& Y_BD);

}  // namespace brotensor
