// linear_forward_batched_ex — 16-bit batched linear with a fused epilogue
// (store / residual accumulate / GeGLU / SwiGLU) and optional split-K
// workspace. One to eight rows take the skinny GEMV below (decode); 9..64 rows
// the weight-streaming tensor-core kernel in gemm_skinny.cu (every epilogue
// fused); wider batches run on the mma.sync GEMM in gemm_mma.cu; shapes that kernel refuses
// (K or N not a multiple of 8, pre-sm_80) and SwiGLU fall back to a stored
// result plus an elementwise epilogue pass. CPU twin in
// src/cpu/packed_encoder.cpp.

#include <cstdint>
#include <type_traits>

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "gemm_mma.h"
#include "gemm_skinny.h"
#include "ops_gemm_mir.h"

#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

CUstream cur_stream() { return reinterpret_cast<CUstream>(::brotensor::cuda_current_stream()); }

[[noreturn]] void fail(const char* reason) {
    throw std::runtime_error(std::string("brotensor: linear_forward_batched_ex: ") + reason);
}

void run_f32(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X,
             int act, int epi, ::brotensor::Tensor& Y) {
    const int N = W.rows, K = W.cols, M = X.rows;
    const int out_cols = epi >= 2 ? N / 2 : N;
    const void* bp = (bias && bias->size() > 0) ? bias->data : nullptr;
    if (M <= 8) {
        dispatch_mir_gemv(X.data, Dtype::FP32, W.data, Dtype::FP32, Y.data, Dtype::FP32,
                          bp, M, N, K, K, K, out_cols, epi, act);
        return;
    }
    dispatch_mir_gemm_simt(X.data, Dtype::FP32, W.data, Dtype::FP32, Y.data, Dtype::FP32,
                           bp, Dtype::FP32, 1, M, N, K, K, K, out_cols, 0, 0, 0,
                           false, false, epi, act, W.device);
}

template <typename T>
void run(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
         int epi, int acc_mode, ::brotensor::Tensor* ws, ::brotensor::Tensor& Y) {
    const int N = W.rows, K = W.cols, M = X.rows;
    const int out_cols = epi >= 2 ? N / 2 : N;
    const T* bp = (bias && bias->size() > 0) ? static_cast<const T*>(bias->data) : nullptr;
    if (M <= 8) {
        dispatch_mir_gemv(X.data, W.dtype, W.data, W.dtype, Y.data, W.dtype, bp,
                          M, N, K, K, K, out_cols, epi, act);
        return;
    }
    if (M <= 64) {
        float* sp = nullptr;
        std::size_t sn = 0;
        const std::size_t need = gemm_skinny::workspace_floats(M, N, K, epi);
        if (ws && need > 0) {
            if (ws->dtype != Dtype::FP32 || static_cast<std::size_t>(ws->rows) * ws->cols < need)
                ws->resize(1, static_cast<int>(need), Dtype::FP32);
            sp = static_cast<float*>(ws->data);
            sn = static_cast<std::size_t>(ws->rows) * ws->cols;
        }
        if (gemm_skinny::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                                static_cast<T*>(Y.data), M, N, K, bp, act, epi, sp, sn, cur_stream()))
            return;
    }
    float* wsp = nullptr;
    std::size_t ws_n = 0;
    if (ws) {
        const std::size_t need = mma_gemm::workspace_floats(M, N, K);
        if (need > 0) {
            if (ws->dtype != Dtype::FP32 || static_cast<std::size_t>(ws->rows) * ws->cols < need)
                ws->resize(1, static_cast<int>(need), Dtype::FP32);
            wsp = static_cast<float*>(ws->data);
            ws_n = static_cast<std::size_t>(ws->rows) * ws->cols;
        }
    }
    if (epi != 3 && mma_gemm::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                                     static_cast<T*>(Y.data), M, N, K, bp, act, epi, wsp, ws_n, cur_stream(),
                                     acc_mode))
        return;
    if (epi == 3) {
        ::brotensor::Tensor r = ::brotensor::Tensor::empty_on(W.device, M, N, W.dtype);
        if (mma_gemm::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                             static_cast<T*>(r.data), M, N, K, bp, act, 0, wsp, ws_n, cur_stream(),
                             acc_mode)) {
            dispatch_mir_swiglu(r, Y, M, N);
            return;
        }
    }
    dispatch_mir_gemm_simt(X.data, W.dtype, W.data, W.dtype, Y.data, W.dtype, bp, W.dtype,
                           1, M, N, K, K, K, out_cols, 0, 0, 0,
                           false, false, epi, act, W.device);
}

}  // namespace

void linear_forward_batched_ex(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                               const ::brotensor::Tensor& X, int act, int epilogue_flags,
                               ::brotensor::Tensor* workspace, ::brotensor::Tensor& Y) {
    const int acc_mode = (epilogue_flags & kLinearEpiFastAccum) ? mma_gemm::kAccHybrid : mma_gemm::kAccF32;
    const int epilogue = epilogue_flags & ~kLinearEpiFastAccum;
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16 && W.dtype != Dtype::FP32)
        fail("W must be FP32, FP16 or BF16 on CUDA");
    if (X.dtype != W.dtype) fail("X must share W's dtype");
    if (bias && bias->dtype != W.dtype) fail("bias must share W's dtype");
    const int N = W.rows, K = W.cols, M = X.rows;
    if (X.cols != K) fail("X.cols must equal W.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail("bias size must equal W.rows");
    if (epilogue < 0 || epilogue > 3) fail("unknown epilogue");
    if (epilogue >= 2 && (act != 0 || N % 2 != 0)) fail("geglu/swiglu needs act 0 and even W.rows");
    const int out_cols = epilogue >= 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != W.dtype) fail("accumulate needs Y (B, out) in W's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != W.dtype) {
        Y.resize(M, out_cols, W.dtype);
    }
    if (M == 0 || N == 0) return;
    if (W.dtype == Dtype::FP32)
        run_f32(W, bias, X, act, epilogue, Y);
    else if (W.dtype == Dtype::FP16)
        run<Fp16>(W, bias, X, act, epilogue, acc_mode, workspace, Y);
    else
        run<Bf16>(W, bias, X, act, epilogue, acc_mode, workspace, Y);
}

void fill_cuda_vtable_linear_ex(::brotensor::detail::OpsVTable& v) {
    v.linear_forward_batched_ex = &linear_forward_batched_ex;
}

}  // namespace brotensor::detail::cuda
