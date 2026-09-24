// linear_forward_batched_ex on Metal: the fused-epilogue batched linear
// (store / residual-accumulate / GeGLU written straight from the GEMM's FP32
// accumulators). Mirrors src/cuda/linear_ex.cu's contract; the GEMM is the
// mixed-precision simdgroup kernel in fp16_matmul.mm.
//
// The workspace is never used: Metal does not split K, so the result is
// deterministic run to run without one. kLinearEpiFastAccum is accepted and
// ignored — accumulation stays FP32 (the flag only ever permits less
// precision, never requires it).

#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

namespace {

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("brotensor: linear_forward_batched_ex: " + reason);
}

}  // namespace

void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X, int act,
                               int epilogue_flags, Tensor* /*workspace*/, Tensor& Y) {
    const int epilogue = epilogue_flags & ~kLinearEpiFastAccum;
    const Dtype dt = W.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail("W must be FP16, BF16 or FP32 on Metal");
    if (X.dtype != dt) fail("X must share W's dtype");
    if (bias && bias->dtype != dt) fail("bias must share W's dtype");
    const int N = W.rows, K = W.cols, M = X.rows;
    if (X.cols != K) fail("X.cols must equal W.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail("bias size must equal W.rows");
    if (epilogue < 0 || epilogue > 2) fail("unknown epilogue");
    if (epilogue == 2 && (act != 0 || N % 2 != 0)) fail("geglu needs act 0 and even W.rows");
    const int out_cols = epilogue == 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != dt) fail("accumulate needs Y (B, out) in W's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(M, out_cols, dt);
    }
    if (M == 0 || N == 0) return;

    metal_impl::AbtMixed g;
    g.A = buffer_for(X);
    g.ofs_A = buffer_offset_for(X);
    g.lda = static_cast<uint64_t>(K);
    g.B = buffer_for(W);
    g.ofs_B = buffer_offset_for(W);
    g.ldb = static_cast<uint64_t>(K);
    g.C = buffer_for(Y);
    g.ofs_C = buffer_offset_for(Y);
    g.ldc = static_cast<uint64_t>(out_cols);
    if (bias) {
        g.bias = buffer_for(*bias);
        g.ofs_bias = buffer_offset_for(*bias);
    }
    g.M = M;
    g.N = N;
    g.K = K;
    g.in = g.out = metal_impl::abt_type(dt);
    g.act = act;
    g.epilogue = epilogue;
    metal_impl::launch_matmul_abt_mixed(g);
}

}  // namespace brotensor::detail::metal
