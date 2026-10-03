// Vulkan matrix ops: matmul, matmul_abt, the linear family (with the bias /
// activation / accumulate / GeGLU / SwiGLU epilogues) and the GEMM backwards.
// Every op maps onto one or more detail::vulkan::gemm() calls (gemm.cpp),
// which picks the GEMV, cooperative-matrix or SIMT kernel. Contracts are wider
// than the CPU's:
// FP32 / FP16 / BF16 everywhere, FP32 activations against 16-bit weights in
// linear_forward_batched (accumulated in FP32 without rounding the
// activations), matmul_abt strides taken literally (zero broadcasts).

#include "detail/gemm.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void column_sum_accumulate(const Tensor& X, Tensor& out);   // ops_norm.cpp

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need_float(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        fail(op, std::string(what) + " must be FP32, FP16 or BF16");
    }
}

// A bias operand: absent when null or empty; otherwise `n` elements of `dt`.
std::uint64_t bias_addr(const Tensor* bias, long long n, Dtype dt, const char* op) {
    if (!bias || bias->size() == 0) return 0;
    if (bias->dtype != dt) fail(op, "bias dtype must match the output's");
    if (bias->size() != n) fail(op, "bias must have " + std::to_string(n) + " elements");
    return addr(bias->data);
}

// Extent check: `need` elements must lie inside `t`.
void need_extent(const Tensor& t, long long need, const char* op, const char* what) {
    if (need > t.size()) {
        fail(op, std::string(what) + " is smaller than the shape and strides address (" +
                     std::to_string(t.size()) + " < " + std::to_string(need) + " elements)");
    }
}

// Y(B, out) = X(B, in) W(out, in)^T with `epi` / `act`, Y already shaped.
void linear(const char* op, const Tensor& W, const Tensor* bias, const Tensor& X, int act, int epi,
            Tensor& Y) {
    const int B = X.rows, in = X.cols, out = W.rows;
    GemmArgs g;
    g.op = op;
    g.a = addr(X.data);
    g.b = addr(W.data);
    g.c = addr(Y.data);
    g.da = X.dtype;
    g.db = W.dtype;
    g.dc = Y.dtype;
    g.bias = bias_addr(bias, out, Y.dtype, op);
    g.m = B;
    g.n = out;
    g.k = in;
    g.lda = in;
    g.ldb = in;
    g.ldc = Y.cols;
    g.epi = epi;
    g.act = act;
    gemm(device_of(W), g);
}

void check_act(int act, const char* op) {
    if (act < 0 || act > LACT_QUICK_GELU) fail(op, "unknown activation " + std::to_string(act));
}

}  // namespace

// ─── matmul family ─────────────────────────────────────────────────────────

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    constexpr const char* op = "matmul";
    need_float(A, op, "A");
    if (A.dtype != B.dtype) fail(op, "A and B must share dtype");
    const int M = A.rows, K = A.cols, N = B.cols;
    if (B.rows != K) fail(op, "shape mismatch (A.cols != B.rows)");
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) C.resize(M, N, A.dtype);
    GemmArgs g;
    g.op = op;
    g.a = addr(A.data);
    g.b = addr(B.data);
    g.c = addr(C.data);
    g.da = g.db = g.dc = A.dtype;
    g.m = M;
    g.n = N;
    g.k = K;
    g.lda = K;
    g.ldb = N;
    g.ldc = N;
    g.nb = true;
    gemm(device_of(A), g);
}

void matmul_abt(const Tensor& A, const Tensor& B, Tensor& C, int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC, const Tensor* bias, int act) {
    constexpr const char* op = "matmul_abt";
    need_float(A, op, "A");
    if (A.dtype != B.dtype) fail(op, "A and B must share dtype");
    check_act(act, op);
    if (batch <= 0 || M == 0 || N == 0) return;
    if (M < 0 || N < 0 || K < 0) fail(op, "negative dimension");
    // C is resized only when its dtype differs or it cannot hold
    // the batch at the given stride.
    const long long c_extent = static_cast<long long>(batch - 1) * strideC + static_cast<long long>(M) * N;
    if (C.dtype != A.dtype || C.size() < c_extent) C.resize(batch > 1 ? batch * M : M, N, A.dtype);
    need_extent(A, static_cast<long long>(batch - 1) * strideA + static_cast<long long>(M) * K, op, "A");
    need_extent(B, static_cast<long long>(batch - 1) * strideB + static_cast<long long>(N) * K, op, "B");
    need_extent(C, c_extent, op, "C");
    GemmArgs g;
    g.op = op;
    g.a = addr(A.data);
    g.b = addr(B.data);
    g.c = addr(C.data);
    g.da = g.db = g.dc = A.dtype;
    g.bias = bias_addr(bias, N, A.dtype, op);
    g.m = M;
    g.n = N;
    g.k = K;
    g.lda = K;
    g.ldb = K;
    g.ldc = N;
    g.batch = batch;
    g.sa = strideA;
    g.sb = strideB;
    g.sc = strideC;
    g.act = act;
    gemm(device_of(A), g);
}

void matmul_backward(const Tensor& A, const Tensor& B, const Tensor& dC, Tensor& dA, Tensor& dB) {
    constexpr const char* op = "matmul_backward";
    need_float(A, op, "A");
    if (B.dtype != A.dtype || dC.dtype != A.dtype || dA.dtype != A.dtype || dB.dtype != A.dtype) {
        fail(op, "dtype mismatch");
    }
    const int M = A.rows, K = A.cols, N = B.cols;
    if (B.rows != K) fail(op, "shape mismatch (A.cols != B.rows)");
    if (dC.rows != M || dC.cols != N) fail(op, "dC shape mismatch");
    if (dA.rows != M || dA.cols != K) fail(op, "dA must be pre-sized to (M, K)");
    if (dB.rows != K || dB.cols != N) fail(op, "dB must be pre-sized to (K, N)");
    if (M == 0 || N == 0 || K == 0) return;
    DeviceCtx& d = device_of(A);
    // dA(M, K) += dC(M, N) B(K, N)^T: B is already the NT operand.
    GemmArgs g;
    g.op = op;
    g.da = g.db = g.dc = A.dtype;
    g.a = addr(dC.data); g.lda = N;
    g.b = addr(B.data);  g.ldb = N;
    g.c = addr(dA.data); g.ldc = K;
    g.m = M; g.n = K; g.k = N;
    g.epi = EPI_ACCUM;
    gemm(d, g);
    // dB(K, N) += A(M, K)^T dC(M, N): A transposed, dC stored (K' = M, N).
    GemmArgs h = g;
    h.a = addr(A.data);  h.lda = K;  h.ta = true;
    h.b = addr(dC.data); h.ldb = N;  h.nb = true;
    h.c = addr(dB.data); h.ldc = N;
    h.m = K; h.n = N; h.k = M;
    gemm(d, h);
}

// ─── linear family ─────────────────────────────────────────────────────────

void linear_forward(const Tensor& W, const Tensor& b, const Tensor& x, Tensor& y) {
    constexpr const char* op = "linear_forward";
    need_float(x, op, "x");
    const int out = W.rows, in = W.cols;
    if (x.size() != in) fail(op, "x must have W.cols elements");
    if (W.dtype != x.dtype && !(x.dtype == Dtype::FP32 && (W.dtype == Dtype::FP16 || W.dtype == Dtype::BF16))) {
        fail(op, "W must share x's dtype, or be 16-bit against FP32 x");
    }
    if (y.rows != out || y.cols != 1 || y.dtype != x.dtype) y.resize(out, 1, x.dtype);
    // x as one row (1, in), y as one row (1, out): the GEMV path.
    Tensor xr = Tensor::view(x.device, x.data, 1, in, x.dtype);
    Tensor yr = Tensor::view(y.device, y.data, 1, out, y.dtype);
    linear(op, W, &b, xr, 0, EPI_STORE, yr);
}

void linear_forward_batched(const Tensor& W, const Tensor& bias, const Tensor& X, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched";
    need_float(X, op, "X");
    if (W.cols != X.cols) fail(op, "shape mismatch (W.cols != X.cols)");
    if (W.dtype != X.dtype && !(X.dtype == Dtype::FP32 && (W.dtype == Dtype::FP16 || W.dtype == Dtype::BF16))) {
        fail(op, "W must share X's dtype, or be 16-bit against FP32 X");
    }
    if (Y.rows != X.rows || Y.cols != W.rows || Y.dtype != X.dtype) Y.resize(X.rows, W.rows, X.dtype);
    linear(op, W, &bias, X, 0, EPI_STORE, Y);
}

void linear_forward_batched_fp16_act(const Tensor& W, const Tensor* bias, const Tensor& X, int act, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_fp16_act";
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) fail(op, "X must be FP16 or BF16");
    if (W.dtype != X.dtype) fail(op, "W must share X's dtype");
    if (W.cols != X.cols) fail(op, "shape mismatch (W.cols != X.cols)");
    check_act(act, op);
    if (Y.rows != X.rows || Y.cols != W.rows || Y.dtype != X.dtype) Y.resize(X.rows, W.rows, X.dtype);
    linear(op, W, bias, X, act, EPI_STORE, Y);
}

void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias, const Tensor& X, Tensor& Y) {
    vulkan::linear_forward_batched_fp16_act(W, bias, X, 0, Y);
}

void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X, int act, int epilogue,
                               Tensor* /*workspace: K is never split*/, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_ex";
    const int epi = epilogue & ~16;   // kLinearEpiFastAccum: accumulation is FP32 regardless
    need_float(X, op, "X");
    if (W.dtype != X.dtype) fail(op, "W must share X's dtype");
    if (W.cols != X.cols) fail(op, "X.cols must equal W.cols");
    check_act(act, op);
    if (epi < EPI_STORE || epi > EPI_SWIGLU) fail(op, "unknown epilogue");
    const int B = X.rows, N = W.rows;
    const bool glu = epi == EPI_GEGLU || epi == EPI_SWIGLU;
    if (glu && (act != 0 || N % 2 != 0)) fail(op, "geglu/swiglu needs act 0 and even W.rows");
    const int out = glu ? N / 2 : N;
    if (epi == EPI_ACCUM) {
        if (Y.rows != B || Y.cols != N || Y.dtype != X.dtype) fail(op, "accumulate needs Y (B, out) in X's dtype");
    } else if (Y.rows != B || Y.cols != out || Y.dtype != X.dtype) {
        Y.resize(B, out, X.dtype);
    }
    linear(op, W, bias, X, act, epi, Y);
}

// ─── backwards ─────────────────────────────────────────────────────────────

void linear_backward_batched(const Tensor& W, const Tensor& X, const Tensor& dY, Tensor& dX, Tensor& dW,
                             Tensor& dB) {
    constexpr const char* op = "linear_backward_batched";
    need_float(W, op, "W");
    if (X.dtype != W.dtype || dY.dtype != W.dtype || dW.dtype != W.dtype || dB.dtype != W.dtype) {
        fail(op, "all tensors must share dtype");
    }
    const int B = X.rows, in = W.cols, out = W.rows;
    if (X.cols != in || dY.rows != B || dY.cols != out) fail(op, "shape mismatch");
    if (dW.rows != out || dW.cols != in || dB.size() != out) fail(op, "dW / dB must be pre-sized");
    if (dX.rows != B || dX.cols != in || dX.dtype != W.dtype) dX.resize(B, in, W.dtype);
    if (B == 0) return;
    DeviceCtx& d = device_of(W);
    // dX(B, in) = dY(B, out) W(out, in): W is stored (K' = out, N' = in).
    GemmArgs g;
    g.op = op;
    g.da = g.db = g.dc = W.dtype;
    g.a = addr(dY.data); g.lda = out;
    g.b = addr(W.data);  g.ldb = in;  g.nb = true;
    g.c = addr(dX.data); g.ldc = in;
    g.m = B; g.n = in; g.k = out;
    gemm(d, g);
    // dW(out, in) += dY^T X: dY stored (K' = B, M' = out), X stored (K' = B, N' = in).
    GemmArgs h = g;
    h.a = addr(dY.data); h.lda = out; h.ta = true;
    h.b = addr(X.data);  h.ldb = in;  h.nb = true;
    h.c = addr(dW.data); h.ldc = in;
    h.m = out; h.n = in; h.k = B;
    h.epi = EPI_ACCUM;
    gemm(d, h);
    // dB += column sums of dY (FP32 sum, one rounding).
    column_sum_accumulate(dY, dB);
}

void linear_backward(const Tensor& W, const Tensor& x, const Tensor& dY, Tensor& dX, Tensor& dW, Tensor& dB) {
    constexpr const char* op = "linear_backward";
    const int out = W.rows, in = W.cols;
    if (x.size() != in || dY.size() != out) fail(op, "x / dY size mismatch");
    if (dX.size() != in || dX.dtype != W.dtype) dX.resize(in, 1, W.dtype);
    Tensor xr = Tensor::view(x.device, x.data, 1, in, x.dtype);
    Tensor dyr = Tensor::view(dY.device, dY.data, 1, out, dY.dtype);
    Tensor dxr = Tensor::view(dX.device, dX.data, 1, in, dX.dtype);
    vulkan::linear_backward_batched(W, xr, dyr, dxr, dW, dB);
}

void fill_vulkan_vtable_linear(::brotensor::detail::OpsVTable& v) {
    v.matmul = &matmul;
    v.matmul_abt = &matmul_abt;
    v.matmul_backward = &matmul_backward;
    v.linear_forward = &linear_forward;
    v.linear_forward_batched = &linear_forward_batched;
    v.linear_forward_batched_fp16 = &linear_forward_batched_fp16;
    v.linear_forward_batched_fp16_act = &linear_forward_batched_fp16_act;
    v.linear_forward_batched_ex = &linear_forward_batched_ex;
    v.linear_backward = &linear_backward;
    v.linear_backward_batched = &linear_backward_batched;
}

}  // namespace brotensor::detail::vulkan
