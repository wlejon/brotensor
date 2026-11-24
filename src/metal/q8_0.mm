// Metal Q8_0 (W8A16) dequant + decode GEMV (MirKind::Dequant / MirKind::GemvQ)
// + prefill simdgroup GEMM fast path. Mirrors src/cuda/q8_0.cu.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

namespace {

constexpr int Q8_BLOCK_ELEMS = 32;
constexpr int Q8_BLOCK_BYTES = 34;

void validate_w_q8_0(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q8_0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q8_0");
    }
    if (W.cols % Q8_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 32");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

}  // namespace

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_w_q8_0(W_q8, "dequant_q8_0_to_fp16");
    const int rows = W_q8.rows;
    const int K    = W_q8.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    if (rows == 0 || K == 0) return;
    const int rowbytes = (K / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES;
    mir_dispatch_dequant(QF_Q8_0, "dequant_q8_0_to_fp16", W_q8.data, W_fp16.data,
                         nullptr, rows, K, rowbytes, true);
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y) {
    validate_w_q8_0(W_q8, "linear_forward_q8_0_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x must be FP16");
    }
    const int out = W_q8.rows;
    const int K   = W_q8.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    const int rowbytes = (K / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES;
    mir_dispatch_gemv_q(QF_Q8_0, Dtype::FP16, "linear_forward_q8_0_fp16",
                        x.data, W_q8.data, y.data,
                        (bias && bias->size() > 0) ? bias->data : nullptr, nullptr,
                        out, K, 1, K, out, rowbytes, LACT_NONE, EPI_STORE);
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q8_0(W_q8, "linear_forward_batched_q8_0_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q8.rows;
    if (W_q8.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q8_0_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    if (B >= kQuantPrefillMinB) {
        Tensor Wf = Tensor::zeros_on(Device::Metal, out, K, Dtype::FP16);
        dequant_q8_0_to_fp16(W_q8, Wf);
        quant_prefill_gemm_fp16(Wf, bias, X_BD, Y_BD);
        return;
    }
    const int rowbytes = (K / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES;
    mir_dispatch_gemv_q(QF_Q8_0, Dtype::FP16, "linear_forward_batched_q8_0_fp16",
                        X_BD.data, W_q8.data, Y_BD.data,
                        (bias && bias->size() > 0) ? bias->data : nullptr, nullptr,
                        out, K, B, K, out, rowbytes, LACT_NONE, EPI_STORE);
}

}  // namespace brotensor::detail::metal
