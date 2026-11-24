// Weight-only INT8 / FP8 E4M3 on Metal: quantize_rows, dequantize_rows, and
// linear_forward_batched_wq_ex — all backed by single-source brass MIR
// (MirKind::WqRows and MirKind::GemvQ). Mirrors src/cuda/linear_wq.cu.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X, int act,
                               int epilogue_flags, Tensor* workspace, Tensor& Y);  // linear_ex.mm

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

}  // namespace

void mir_dispatch_gemv_q(std::uint32_t fmt, Dtype dt, const char* op,
                         const void* x, const void* w, const void* y,
                         const void* bias, const void* scale,
                         int rows, int k, int B, int ldx, int ldy,
                         int rowbytes, int act, int epi) {
    if (rows <= 0 || k <= 0 || B <= 0) return;
    const bool byte_fmt = (fmt == QF_INT8 || fmt == QF_FP8);
    const bool glu = (epi == EPI_GEGLU || epi == EPI_SWIGLU);
    const std::uint32_t ie = (byte_fmt || fmt == QF_Q8_0) ? 16u : 32u;
    const std::uint64_t items = static_cast<std::uint64_t>(k) / ie;
    const std::uint32_t lpr = (items < 32u) ? 16u : 32u;
    const std::uint32_t unr = (byte_fmt || fmt == QF_Q8_0) ? (k >= 4096 ? 4u : 2u) : 1u;

    for (int m0 = 0; m0 < B; m0 += 8) {
        const int mb = std::min(8, B - m0);
        std::uint32_t nr = 2u;
        if (byte_fmt || fmt == QF_Q8_0) nr = (k >= 4096) ? 1u : 2u;
        else if (fmt == QF_Q4K) nr = 4u;
        else nr = (mb >= 8) ? 4u : 2u;
        if (glu && (nr < 2u || (nr & 1u) != 0u)) nr = 2u;

        const void* x_ptr = static_cast<const char*>(x) + static_cast<std::size_t>(m0) * ldx * 2u;
        const void* y_ptr = static_cast<const char*>(y) + static_cast<std::size_t>(m0) * ldy * 2u;
        int eff_ldx = ldx;
        Tensor xa;
        const bool w_vec = !byte_fmt ||
            (k % 16 == 0 && (reinterpret_cast<std::uintptr_t>(w) % 16u == 0u));
        bool vec = w_vec;
        if (vec && (reinterpret_cast<std::uintptr_t>(x_ptr) % 16u != 0u || eff_ldx % 8 != 0)) {
            if (byte_fmt) {
                vec = false;
            } else {
                xa = Tensor::empty_on(Device::Metal, mb, k, dt);
                const int cl = mir_copy2d_lanes(2, static_cast<std::uint32_t>(k),
                    static_cast<std::uint32_t>(eff_ldx), static_cast<std::uint32_t>(k),
                    reinterpret_cast<std::uintptr_t>(x_ptr), reinterpret_cast<std::uintptr_t>(xa.data));
                const MirKernel ck = mir_kernel({MirKind::Copy2d, 2u, Dtype::FP32, Dtype::FP32,
                                                 static_cast<std::uint8_t>(cl)});
                const std::uint32_t cgx = static_cast<std::uint32_t>(
                    ((static_cast<std::uint64_t>(k) / cl) + 255u) / 256u);
                mir_dispatch_copy2d(ck, std::max<std::uint32_t>(1u, cgx), static_cast<std::uint32_t>(mb),
                                    op, x_ptr, xa.data, static_cast<std::uint32_t>(eff_ldx),
                                    static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(k),
                                    static_cast<std::uint32_t>(mb));
                x_ptr = xa.data;
                eff_ldx = k;
            }
        }

        const std::uint32_t op_code = (fmt & 0x0fu) |
                                      ((static_cast<std::uint32_t>(mb) & 0x0fu) << 4) |
                                      ((unr & 0x0fu) << 8) |
                                      ((nr & 0x0fu) << 12) |
                                      ((static_cast<std::uint32_t>(act) & 0x0fu) << 16) |
                                      ((static_cast<std::uint32_t>(epi) & 0x0fu) << 20);
        const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (bias ? 2u : 0u));
        const MirKernel mk = mir_kernel(
            {MirKind::GemvQ, op_code, dt, Dtype::FP32, static_cast<std::uint8_t>(lpr), Dtype::FP32, flags});
        const std::uint64_t per_wg = static_cast<std::uint64_t>(256u / lpr) * nr;
        const std::uint64_t groups = (static_cast<std::uint64_t>(rows) + per_wg - 1u) / per_wg;
        const std::uint32_t gx = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(1u, std::min<std::uint64_t>(groups, 65535u)));
        const std::uint32_t gy = static_cast<std::uint32_t>((groups + gx - 1u) / gx);
        const void* ptrs[] = {x_ptr, w, y_ptr, bias, scale};
        const std::int32_t ints[] = {rows, k, eff_ldx, ldy, rowbytes};
        mir_dispatch_args_3d(mk, gx, gy, 1u, op, ptrs, ints, {});
    }
}

void quantize_rows(const Tensor& W, Dtype fmt, Tensor& Wq, Tensor& scales) {
    constexpr const char* op = "quantize_rows";
    if (fmt != Dtype::INT8 && fmt != Dtype::FP8_E4M3) fail(op, "fmt must be INT8 or FP8_E4M3");
    if (W.dtype != Dtype::FP32 && W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16)
        fail(op, "W must be FP32, FP16 or BF16");
    const int N = W.rows, K = W.cols;
    if (Wq.rows != N || Wq.cols != K || Wq.dtype != fmt) Wq.resize(N, K, fmt);
    if (scales.rows != N || scales.cols != 1 || scales.dtype != Dtype::FP32) scales.resize(N, 1, Dtype::FP32);
    if (N == 0) return;
    const MirKernel mk = mir_kernel(
        {MirKind::WqRows, 0u, W.dtype, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(fmt == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min(N, 65535));
    const std::uint32_t gy = static_cast<std::uint32_t>((static_cast<std::uint64_t>(N) + gx - 1u) / gx);
    const void* ptrs[] = {W.data, Wq.data, scales.data};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args_3d(mk, gx, gy, 1u, op, ptrs, ints, {});
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
    const MirKernel mk = mir_kernel(
        {MirKind::WqRows, 1u, out, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(Wq.dtype == Dtype::FP8_E4M3 ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((n + 255u) / 256u, 1u, 65535u));
    const void* ptrs[] = {W.data, Wq.data, scales.data};
    const std::int32_t ints[] = {N, K};
    mir_dispatch_args(mk, gx, op, ptrs, ints, {});
}

void linear_forward_batched_wq_ex(const Tensor& Wq, const Tensor& scales, const Tensor* bias, const Tensor& X,
                                  int act, int epilogue_flags, Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_wq_ex";
    const int epi = epilogue_flags & ~kLinearEpiFastAccum;
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    const int N = Wq.rows, K = Wq.cols, M = X.rows;
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != N)
        fail(op, "scales must be (out,1) FP32");
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail(op, "X must be FP16, BF16 or FP32");
    if (X.cols != K) fail(op, "X.cols must equal Wq.cols");
    if (bias && (bias->dtype != dt || static_cast<long long>(bias->rows) * bias->cols != N))
        fail(op, "bias must be (out,1) in X's dtype");
    if (epi < kLinearEpiStore || epi > kLinearEpiSwiglu) fail(op, "unknown epilogue");
    const bool gated = epi == kLinearEpiGeglu || epi == kLinearEpiSwiglu;
    if (gated && (act != 0 || N % 2 != 0)) fail(op, "geglu/swiglu needs act 0 and even Wq.rows");
    const int out_cols = gated ? N / 2 : N;
    if (epi == kLinearEpiAccumulate) {
        if (Y.rows != M || Y.cols != N || Y.dtype != dt) fail(op, "accumulate needs Y (B, out) in X's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(M, out_cols, dt);
    }
    if (M == 0 || out_cols == 0) return;

    if (dt == Dtype::FP32) {
        Tensor Wf = Tensor::empty_on(Device::Metal, N, K, Dtype::FP32);
        dequantize_rows(Wq, scales, Dtype::FP32, Wf);
        metal::linear_forward_batched_ex(Wf, bias, X, act, epilogue_flags, nullptr, Y);
        return;
    }

    const std::uint32_t fmt = (Wq.dtype == Dtype::FP8_E4M3) ? QF_FP8 : QF_INT8;
    mir_dispatch_gemv_q(fmt, dt, op, X.data, Wq.data, Y.data,
                        bias ? bias->data : nullptr, scales.data,
                        N, K, M, K, out_cols, K, act, epi);
}

}  // namespace brotensor::detail::metal
