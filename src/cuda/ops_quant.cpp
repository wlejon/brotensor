// CUDA Wave 9 & Wave 11 quantised kernels backed by single-source brass MIR:
//   - quantize_rows / dequantize_rows (MirKind::WqRows)
//   - dequant_{q8_0,q4k,q6k}_to_fp16 (MirKind::Dequant)
//   - linear_forward_batched_wq_ex + GGUF/INT8 decode GEMV (MirKind::GemvQ)
// Prefill GEMMs still dispatch to q*_wmma_internal WMMA kernels first.

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"

#include "../jit/gpu_mir_emit_detail.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace q8_0_wmma_internal {
bool launch_linear_q8_0_fp16_wmma(const __half* X, const uint8_t* W_q8,
                                  const __half* bias, __half* Y,
                                  int B, int M, int K, cudaStream_t stream);
}

namespace q4k_wmma_internal {
bool launch_linear_q4k_fp16_wmma(const __half* X, const uint8_t* W_q4k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream);
}

namespace q6k_wmma_internal {
bool launch_linear_q6k_fp16_wmma(const __half* X, const uint8_t* W_q6k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream);
}

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

bool byte_fmt(std::uint32_t fmt) { return fmt == QF_INT8 || fmt == QF_FP8; }

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

struct GemvCfg { std::uint32_t lpr, unr, nr; };

GemvCfg gemv_cfg(std::uint32_t fmt, int k, int b) {
    const std::uint32_t ie = (byte_fmt(fmt) || fmt == QF_Q8_0) ? 16u : 32u;
    const std::uint64_t items = static_cast<std::uint64_t>(k) / ie;
    GemvCfg c{32u, 1u, 1u};
    if (byte_fmt(fmt) || fmt == QF_Q8_0) {
        if (k >= 4096) c = {64u, 4u, 1u};
        else c = {32u, 2u, 2u};
    } else if (fmt == QF_Q4K) {
        c = {32u, 1u, 4u};
    } else {
        c = {32u, 1u, b >= 8 ? 4u : 2u};
    }
    if (items < 32u) c.lpr = 16u;
    return c;
}

struct QuantDesc {
    std::uint32_t fmt = QF_INT8;
    int rows = 0;
    int k = 0;
    std::uint32_t rowbytes = 0;
    std::uintptr_t w = 0;
    std::uintptr_t scale = 0;
};

void dispatch_gemv_q_slice(const QuantDesc& q, std::uintptr_t x, Dtype dt, int mb, int ldx,
                           std::uintptr_t bias, std::uintptr_t y, int ldy, int act, int epi) {
    const bool vec = !byte_fmt(q.fmt) || ((q.k % 16 == 0) && (q.w % 16u == 0));
    std::uintptr_t x_use = x;
    int ldx_use = ldx;
    Tensor xa;
    if (vec && ((x % 16u != 0) || (ldx % 8 != 0))) {
        xa = Tensor::empty_on(Device::CUDA, mb, q.k, dt);
        const std::size_t row_bytes = static_cast<std::size_t>(q.k) * 2u;
        if (ldx == q.k) {
            cudaMemcpyAsync(xa.data, reinterpret_cast<const void*>(x),
                            static_cast<std::size_t>(mb) * row_bytes, cudaMemcpyDeviceToDevice, cur_stream());
        } else {
            cudaMemcpy2DAsync(xa.data, row_bytes, reinterpret_cast<const void*>(x),
                              static_cast<std::size_t>(ldx) * 2u, row_bytes, static_cast<std::size_t>(mb),
                              cudaMemcpyDeviceToDevice, cur_stream());
        }
        x_use = addr(xa.data);
        ldx_use = q.k;
    }

    GemvCfg c = gemv_cfg(q.fmt, q.k, mb);
    if ((epi == EPI_GEGLU || epi == EPI_SWIGLU) && c.nr < 2u) c.nr = 2u;
    const std::uint32_t op = (q.fmt & 0x0fu) |
                             ((static_cast<std::uint32_t>(mb) & 0x0fu) << 4) |
                             ((c.unr & 0x0fu) << 8) |
                             ((c.nr & 0x0fu) << 12) |
                             ((static_cast<std::uint32_t>(act) & 0x0fu) << 16) |
                             ((static_cast<std::uint32_t>(epi) & 0x0fu) << 20);
    const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (bias != 0 ? 2u : 0u));
    CUfunction fn = mir_function(
        {MirKind::GemvQ, op, dt, Dtype::FP32, static_cast<std::uint8_t>(c.lpr), Dtype::FP32, flags});
    const std::uint64_t groups = cdiv(static_cast<std::uint64_t>(q.rows), (256u / c.lpr) * c.nr);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::max<std::uint64_t>(1u, std::min<std::uint64_t>(groups, 65535u)));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(groups, gx));
    const std::uintptr_t ptrs[] = {x_use, q.w, y, bias, q.scale};
    const std::int32_t ints[] = {q.rows, q.k, ldx_use, ldy, static_cast<std::int32_t>(q.rowbytes)};
    mir_dispatch_args_3d(fn, gx, gy, 1u, jit::gpu_mir::kEwBlock, ptrs, ints, {});
}

void run_gemv_q_loop(const QuantDesc& q, std::uintptr_t x, Dtype dt, int B, int ldx,
                     std::uintptr_t bias, std::uintptr_t y, int ldy, int act, int epi) {
    const int out_cols = (epi == EPI_GEGLU || epi == EPI_SWIGLU) ? q.rows / 2 : q.rows;
    for (int m0 = 0; m0 < B; m0 += 8) {
        const int mr = std::min(8, B - m0);
        const std::uintptr_t x_p = x + static_cast<std::size_t>(m0) * ldx * 2u;
        const std::uintptr_t y_p = y + static_cast<std::size_t>(m0) * out_cols * 2u;
        dispatch_gemv_q_slice(q, x_p, dt, mr, ldx, bias, y_p, ldy, act, epi);
    }
}

void run_gguf_dequant(std::uint32_t fmt, const Tensor& W, Tensor& W_fp16, int block_elems, int block_bytes) {
    const int rows = W.rows;
    const int K = W.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    if (rows == 0 || K == 0) return;
    const int rowbytes = (K / block_elems) * block_bytes;
    CUfunction fn = mir_function({MirKind::Dequant, fmt, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 1u});
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(K / 8);
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
    const std::uintptr_t ptrs[] = {addr(W.data), addr(W_fp16.data), 0u};
    const std::int32_t ints[] = {rows, K, rowbytes};
    mir_dispatch_args(fn, gx, jit::gpu_mir::kEwBlock, ptrs, ints, {});
}

void validate_gguf(const Tensor& W, Dtype expected, int block_elems, const char* op) {
    if (W.dtype != expected) {
        fail(op, std::string("W must be Dtype::") + (expected == Dtype::Q8_0 ? "Q8_0" : expected == Dtype::Q4_K ? "Q4_K" : "Q6_K"));
    }
    if (W.cols % block_elems != 0) {
        fail(op, "W.cols must be a multiple of " + std::to_string(block_elems));
    }
    if (W.rows <= 0 || W.cols <= 0) {
        fail(op, "W has non-positive shape");
    }
}

void validate_gguf_bias_and_out(const Tensor* bias, int out, int B, Tensor& Y, const char* op, bool batched) {
    if (bias) {
        if (bias->dtype != Dtype::FP16) fail(op, "bias must be FP16");
        const bool ok = batched
            ? ((bias->rows == out && bias->cols == 1) || (bias->rows == 1 && bias->cols == out))
            : (bias->rows == out && bias->cols == 1);
        if (!ok) {
            fail(op, batched ? "bias shape must be (out,1) or (1,out)" : "bias shape must be (out, 1)");
        }
    }
    const int exp_r = batched ? B : out;
    const int exp_c = batched ? out : 1;
    if (Y.rows != exp_r || Y.cols != exp_c || Y.dtype != Dtype::FP16) {
        Y.resize(exp_r, exp_c, Dtype::FP16);
    }
}

}  // namespace

// ── Public helpers for int8_quant.cu fallback ────────────────────────────────

void gemv_int8w_cuda(const Tensor& W_int8, const Tensor& scales, const Tensor* bias,
                     const Tensor& X, Tensor& Y) {
    QuantDesc q;
    q.fmt = QF_INT8;
    q.rows = W_int8.rows;
    q.k = W_int8.cols;
    q.rowbytes = static_cast<std::uint32_t>(W_int8.cols);
    q.w = addr(W_int8.data);
    q.scale = addr(scales.data);
    const std::uintptr_t b_p = (bias && bias->size() > 0) ? addr(bias->data) : 0u;
    run_gemv_q_loop(q, addr(X.data), X.dtype, X.rows, X.cols, b_p, addr(Y.data), W_int8.rows, LACT_NONE, EPI_STORE);
}

// ── linear_wq ops (Wave 9 & Wave 11) ─────────────────────────────────────────

void quantize_rows(const Tensor& W, Dtype fmt, Tensor& Wq, Tensor& scales) {
    constexpr const char* op = "quantize_rows";
    if (fmt != Dtype::INT8 && fmt != Dtype::FP8_E4M3) fail(op, "fmt must be INT8 or FP8_E4M3");
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16 && W.dtype != Dtype::FP32)
        fail(op, "W must be FP16, BF16 or FP32");
    const int N = W.rows, K = W.cols;
    if (Wq.rows != N || Wq.cols != K || Wq.dtype != fmt) Wq.resize(N, K, fmt);
    if (scales.rows != N || scales.cols != 1 || scales.dtype != Dtype::FP32) scales.resize(N, 1, Dtype::FP32);
    if (N == 0 || K == 0) return;
    CUfunction fn = mir_function(
        {MirKind::WqRows, 0u, W.dtype, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(fmt == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min(N, 65535));
    const std::uint32_t gy = static_cast<std::uint32_t>((static_cast<std::uint64_t>(N) + gx - 1u) / gx);
    const std::uintptr_t ptrs[] = {addr(W.data), addr(Wq.data), addr(scales.data)};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args_3d(fn, gx, gy, 1u, jit::gpu_mir::kEwBlock, ptrs, ints, {});
}

void dequantize_rows(const Tensor& Wq, const Tensor& scales, Dtype out, Tensor& W) {
    constexpr const char* op = "dequantize_rows";
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    if (out != Dtype::FP16 && out != Dtype::BF16 && out != Dtype::FP32) fail(op, "out must be FP16, BF16 or FP32");
    const int N = Wq.rows, K = Wq.cols;
    if (W.rows != N || W.cols != K || W.dtype != out) W.resize(N, K, out);
    const std::uint64_t n = static_cast<std::uint64_t>(N) * static_cast<std::uint64_t>(K);
    if (n == 0) return;
    CUfunction fn = mir_function(
        {MirKind::WqRows, 1u, out, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(Wq.dtype == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((n + 255u) / 256u, 1u, 65535u));
    const std::uintptr_t ptrs[] = {addr(W.data), addr(Wq.data), addr(scales.data)};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args(fn, gx, jit::gpu_mir::kEwBlock, ptrs, ints, {});
}

void linear_forward_batched_wq_ex(const Tensor& Wq, const Tensor& scales, const Tensor* bias, const Tensor& X,
                                  int act, int epilogue_flags, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_wq_ex";
    const int epilogue = epilogue_flags & ~kLinearEpiFastAccum;
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) fail(op, "X must be FP16 or BF16 on CUDA");
    if (bias && bias->dtype != X.dtype) fail(op, "bias must share X's dtype");
    const int N = Wq.rows, K = Wq.cols, M = X.rows;
    if (X.cols != K) fail(op, "X.cols must equal Wq.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail(op, "bias size must equal Wq.rows");
    if (epilogue < 0 || epilogue > 3) fail(op, "unknown epilogue");
    if (epilogue >= 2 && (act != 0 || N % 2 != 0)) fail(op, "geglu/swiglu needs act 0 and even Wq.rows");
    const int out_cols = epilogue >= 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != X.dtype) fail(op, "accumulate needs Y (B, out) in X's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(M, out_cols, X.dtype);
    }
    if (M == 0 || N == 0) return;

    QuantDesc q;
    q.fmt = Wq.dtype == Dtype::FP8_E4M3 ? QF_FP8 : QF_INT8;
    q.rows = N;
    q.k = K;
    q.rowbytes = static_cast<std::uint32_t>(K);
    q.w = addr(Wq.data);
    q.scale = addr(scales.data);
    const std::uintptr_t b_p = (bias && bias->size() > 0) ? addr(bias->data) : 0u;
    run_gemv_q_loop(q, addr(X.data), X.dtype, M, K, b_p, addr(Y.data), out_cols, act, epilogue);
}

// ── GGUF ops (Q8_0, Q4_K, Q6_K) ──────────────────────────────────────────────

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_gguf(W_q8, Dtype::Q8_0, 32, "dequant_q8_0_to_fp16");
    run_gguf_dequant(QF_Q8_0, W_q8, W_fp16, 32, 34);
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias, const Tensor& x, Tensor& y) {
    constexpr const char* op = "linear_forward_q8_0_fp16";
    validate_gguf(W_q8, Dtype::Q8_0, 32, op);
    if (x.dtype != Dtype::FP16) fail(op, "x must be FP16");
    const int out = W_q8.rows, K = W_q8.cols;
    if (x.rows != K || x.cols != 1) fail(op, "x shape must be (in, 1)");
    validate_gguf_bias_and_out(bias, out, 1, y, op, false);
    if (out == 0) return;
    QuantDesc q{QF_Q8_0, out, K, static_cast<std::uint32_t>((K / 32) * 34), addr(W_q8.data), 0u};
    const std::uintptr_t b_p = (bias && bias->size() > 0) ? addr(bias->data) : 0u;
    dispatch_gemv_q_slice(q, addr(x.data), Dtype::FP16, 1, K, b_p, addr(y.data), out, LACT_NONE, EPI_STORE);
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD) {
    constexpr const char* op = "linear_forward_batched_q8_0_fp16";
    validate_gguf(W_q8, Dtype::Q8_0, 32, op);
    if (X_BD.dtype != Dtype::FP16) fail(op, "X must be FP16");
    const int B = X_BD.rows, K = X_BD.cols, out = W_q8.rows;
    if (W_q8.cols != K) fail(op, "shape mismatch (W.cols != X.cols)");
    validate_gguf_bias_and_out(bias, out, B, Y_BD, op, true);
    if (B == 0 || out == 0) return;
    const __half* b_h = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;
    if (q8_0_wmma_internal::launch_linear_q8_0_fp16_wmma(
            static_cast<const __half*>(X_BD.data), static_cast<const uint8_t*>(W_q8.data),
            b_h, static_cast<__half*>(Y_BD.data), B, out, K, cur_stream())) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }
    QuantDesc q{QF_Q8_0, out, K, static_cast<std::uint32_t>((K / 32) * 34), addr(W_q8.data), 0u};
    run_gemv_q_loop(q, addr(X_BD.data), Dtype::FP16, B, K, addr(b_h), addr(Y_BD.data), out, LACT_NONE, EPI_STORE);
}

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    validate_gguf(W_q4k, Dtype::Q4_K, 256, "dequant_q4k_to_fp16");
    run_gguf_dequant(QF_Q4K, W_q4k, W_fp16, 256, 144);
}

void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias, const Tensor& x, Tensor& y) {
    constexpr const char* op = "linear_forward_q4k_fp16";
    validate_gguf(W_q4k, Dtype::Q4_K, 256, op);
    if (x.dtype != Dtype::FP16) fail(op, "x must be FP16");
    const int out = W_q4k.rows, K = W_q4k.cols;
    if (x.rows != K || x.cols != 1) fail(op, "x shape must be (in, 1)");
    validate_gguf_bias_and_out(bias, out, 1, y, op, false);
    if (out == 0) return;
    QuantDesc q{QF_Q4K, out, K, static_cast<std::uint32_t>((K / 256) * 144), addr(W_q4k.data), 0u};
    const std::uintptr_t b_p = (bias && bias->size() > 0) ? addr(bias->data) : 0u;
    dispatch_gemv_q_slice(q, addr(x.data), Dtype::FP16, 1, K, b_p, addr(y.data), out, LACT_NONE, EPI_STORE);
}

void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD) {
    constexpr const char* op = "linear_forward_batched_q4k_fp16";
    validate_gguf(W_q4k, Dtype::Q4_K, 256, op);
    if (X_BD.dtype != Dtype::FP16) fail(op, "X must be FP16");
    const int B = X_BD.rows, K = X_BD.cols, out = W_q4k.rows;
    if (W_q4k.cols != K) fail(op, "shape mismatch (W.cols != X.cols)");
    validate_gguf_bias_and_out(bias, out, B, Y_BD, op, true);
    if (B == 0 || out == 0) return;
    const __half* b_h = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;
    if (q4k_wmma_internal::launch_linear_q4k_fp16_wmma(
            static_cast<const __half*>(X_BD.data), static_cast<const uint8_t*>(W_q4k.data),
            b_h, static_cast<__half*>(Y_BD.data), B, out, K, cur_stream())) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }
    QuantDesc q{QF_Q4K, out, K, static_cast<std::uint32_t>((K / 256) * 144), addr(W_q4k.data), 0u};
    run_gemv_q_loop(q, addr(X_BD.data), Dtype::FP16, B, K, addr(b_h), addr(Y_BD.data), out, LACT_NONE, EPI_STORE);
}

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    validate_gguf(W_q6k, Dtype::Q6_K, 256, "dequant_q6k_to_fp16");
    run_gguf_dequant(QF_Q6K, W_q6k, W_fp16, 256, 210);
}

void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias, const Tensor& x, Tensor& y) {
    constexpr const char* op = "linear_forward_q6k_fp16";
    validate_gguf(W_q6k, Dtype::Q6_K, 256, op);
    if (x.dtype != Dtype::FP16) fail(op, "x must be FP16");
    const int out = W_q6k.rows, K = W_q6k.cols;
    if (x.rows != K || x.cols != 1) fail(op, "x shape must be (in, 1)");
    validate_gguf_bias_and_out(bias, out, 1, y, op, false);
    if (out == 0) return;
    QuantDesc q{QF_Q6K, out, K, static_cast<std::uint32_t>((K / 256) * 210), addr(W_q6k.data), 0u};
    const std::uintptr_t b_p = (bias && bias->size() > 0) ? addr(bias->data) : 0u;
    dispatch_gemv_q_slice(q, addr(x.data), Dtype::FP16, 1, K, b_p, addr(y.data), out, LACT_NONE, EPI_STORE);
}

void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD) {
    constexpr const char* op = "linear_forward_batched_q6k_fp16";
    validate_gguf(W_q6k, Dtype::Q6_K, 256, op);
    if (X_BD.dtype != Dtype::FP16) fail(op, "X must be FP16");
    const int B = X_BD.rows, K = X_BD.cols, out = W_q6k.rows;
    if (W_q6k.cols != K) fail(op, "shape mismatch (W.cols != X.cols)");
    validate_gguf_bias_and_out(bias, out, B, Y_BD, op, true);
    if (B == 0 || out == 0) return;
    const __half* b_h = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;
    if (q6k_wmma_internal::launch_linear_q6k_fp16_wmma(
            static_cast<const __half*>(X_BD.data), static_cast<const uint8_t*>(W_q6k.data),
            b_h, static_cast<__half*>(Y_BD.data), B, out, K, cur_stream())) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }
    QuantDesc q{QF_Q6K, out, K, static_cast<std::uint32_t>((K / 256) * 210), addr(W_q6k.data), 0u};
    run_gemv_q_loop(q, addr(X_BD.data), Dtype::FP16, B, K, addr(b_h), addr(Y_BD.data), out, LACT_NONE, EPI_STORE);
}

void fill_cuda_vtable_linear_wq(::brotensor::detail::OpsVTable& v) {
    v.linear_forward_batched_wq_ex = &linear_forward_batched_wq_ex;
    v.quantize_rows = &quantize_rows;
    v.dequantize_rows = &dequantize_rows;
}

}  // namespace brotensor::detail::cuda
