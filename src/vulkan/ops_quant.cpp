// Vulkan quantised-weight linears: the GGUF formats (Q8_0, Q4_K, Q6_K: GEMV,
// batched linear, dequantise), the INT8 W8A16 linear and matmul, and the
// INT8 convolutions. Contracts follow the HIP backend (src/hip/quant.hip,
// quant_gguf.hip): FP16 activations for GGUF (FP16 or BF16 for the INT8
// linear), FP32 accumulation, outputs resized.
//
// Kernels (detail/quant.h, docs/vulkan.md "Quantised weights"):
//   <= 8 activation rows   gemv_q.comp: each W row read once, decoded in
//                          registers, MB rows of X against it;
//   more rows (prefill)    gemm_cm.comp with QB: the weight decoded to FP16
//                          as each B chunk is stored to the shared tile;
//   no cooperative matrix  dequant.comp to an FP16 copy, then the dense GEMM.

#include "detail/gemm.h"
#include "detail/kernels.h"
#include "detail/quant.h"
#include "detail/spatial.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);   // ops_copy.cpp
void conv3d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW, int stride_t, int stride_h, int stride_w, int pad_t,
                    int pad_h, int pad_w, int dil_t, int dil_h, int dil_w, int groups, Tensor& Y);   // ops_conv.cpp

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

struct GemvQPush {
    std::uint64_t x, w, y, bias, scale;
    std::uint32_t n, k, ldx, ldy, rowbytes;
};

struct DequantPush {
    std::uint64_t w, y, scale;
    std::uint32_t rows, k, rowbytes;
};

struct TransposePush { std::uint64_t src, dst; std::uint32_t batch, rows, cols; };

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

// gemv_q.comp's lanes per row group, items in flight per lane, subgroup
// size and W rows per group. BROTENSOR_VK_QGEMV=lpr,unr,sg,nr overrides
// (benchmarking).
struct GemvCfg { std::uint32_t lpr, unr, sg, nr; };

// Measured on the Radeon 8060S (docs/vulkan.md "Quantised weights"): wave64,
// about 64 bytes of weights in flight per lane, and several W rows per lane
// group once there are several rows of X to reuse each X chunk across. Long
// INT8 / Q8_0 rows (K >= 4096) stream best one row per group with four items
// in flight; with short rows two rows per group keep more bytes in flight.
GemvCfg gemv_cfg(DeviceCtx& d, int fmt, int k, int B) {
    static const GemvCfg forced = [] {
        GemvCfg c{0, 0, 0, 0};
        if (const char* e = std::getenv("BROTENSOR_VK_QGEMV"))
            std::sscanf(e, "%u,%u,%u,%u", &c.lpr, &c.unr, &c.sg, &c.nr);
        return c;
    }();
    const auto& info = d.info();
    std::uint32_t sg = 0;   // 0: the device's default, not pinned
    if (info.subgroup_size_control) {
        if (info.min_subgroup <= 64 && info.max_subgroup >= 64) sg = 64;
        else if (info.min_subgroup <= 32 && info.max_subgroup >= 32) sg = 32;
    }
    const std::uint32_t ie = fmt == QF_INT8 || fmt == QF_Q8_0 ? 16u : 32u;
    const std::uint64_t items = static_cast<std::uint64_t>(k) / ie;
    GemvCfg c{32, 1, sg, 1};
    if (fmt == QF_INT8 || fmt == QF_Q8_0) {
        if (k >= 4096) c = {64, 4, sg, 1};
        else c = {32, 2, sg, 2};
    } else if (fmt == QF_Q4K) {
        c = {32, 1, sg, 4};
    } else {
        c = {32, 1, sg, B >= 8 ? 4u : 2u};
    }
    if (items < 32) c.lpr = 16;
    if (forced.lpr) c.lpr = forced.lpr;
    if (forced.unr) c.unr = forced.unr;
    if (forced.sg) c.sg = forced.sg;
    if (forced.nr) c.nr = forced.nr;
    return c;
}

const char* fmt_name(int fmt) {
    switch (fmt) {
        case QF_INT8: return "INT8";
        case QF_Q8_0: return "Q8_0";
        case QF_Q4K: return "Q4_K";
        default: return "Q6_K";
    }
}

void gemv_q(DeviceCtx& d, const QuantW& q, std::uint64_t x, Dtype dt, int B, int ldx, std::uint64_t bias,
            std::uint64_t y, int ldy) {
    const bool vec = q.fmt != QF_INT8 || (q.k % 16 == 0 && q.w % 16 == 0);
    const GemvCfg c = gemv_cfg(d, q.fmt, q.k, B);
    const std::uint32_t sg = c.sg ? c.sg : d.info().subgroup_size;
    const std::uint32_t spec[] = {static_cast<std::uint32_t>(q.fmt), static_cast<std::uint32_t>(B), c.lpr,
                                  vec ? 1u : 0u, bias ? 1u : 0u, c.unr, sg, c.nr};
    const ShaderId id = dt == Dtype::FP16 ? ShaderId::gemv_q_f16 : ShaderId::gemv_q_bf16;
    const Kernel& k = d.pipelines().get(id, spec, static_cast<std::uint32_t>(std::size(spec)), c.sg);
    GemvQPush pc{x, q.w, y, bias, q.scale, static_cast<std::uint32_t>(q.rows), static_cast<std::uint32_t>(q.k),
                 static_cast<std::uint32_t>(ldx), static_cast<std::uint32_t>(ldy), q.rowbytes};
    const std::uint64_t groups = cdiv(q.rows, 256u / c.lpr * c.nr);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(groups, gx));
    launch(d, k, pc, gx, gy);
}

}  // namespace

QuantW quant_weight(const Tensor& W, const Tensor* scales, const char* op) {
    QuantW q;
    switch (W.dtype) {
        case Dtype::INT8: q.fmt = QF_INT8; break;
        case Dtype::Q8_0: q.fmt = QF_Q8_0; break;
        case Dtype::Q4_K: q.fmt = QF_Q4K; break;
        case Dtype::Q6_K: q.fmt = QF_Q6K; break;
        default: fail(op, "W must be INT8, Q8_0, Q4_K or Q6_K");
    }
    need(op, W.rows > 0 && W.cols > 0, "W has a non-positive shape");
    q.rows = W.rows;
    q.k = W.cols;
    q.w = addr(W.data);
    if (q.fmt == QF_INT8) {
        need(op, scales != nullptr && scales->dtype == Dtype::FP32, "scales must be FP32");
        need(op, scales->rows == W.rows && scales->cols == 1, "scales shape must be (out, 1)");
        q.scale = addr(scales->data);
        q.rowbytes = static_cast<std::uint32_t>(W.cols);
    } else {
        const int blk = ::brotensor::dtype_block_size(W.dtype);
        if (W.cols % blk != 0) fail(op, "W.cols must be a multiple of " + std::to_string(blk));
        q.rowbytes = static_cast<std::uint32_t>(W.cols / blk * ::brotensor::dtype_block_bytes(W.dtype));
        // Q4_K is read in 16-byte words (144-byte blocks keep rows aligned);
        // Q8_0 and Q6_K in 2-byte ones.
        need(op, q.fmt == QF_Q4K ? q.w % 16 == 0 : q.w % 2 == 0, "W is not aligned for its block format");
    }
    if (static_cast<std::uint64_t>(q.rowbytes) * q.rows > 0xffffffffULL) fail(op, "W larger than 4 GiB");
    return q;
}

void dequant(DeviceCtx& d, const QuantW& q, std::uint64_t y) {
    // 8-element chunks when rows are whole chunks (always for GGUF); single
    // elements for INT8 rows of any other length (a depthwise kernel's 9).
    const bool vec = q.k % 8 == 0 && (q.fmt != QF_INT8 || q.w % 8 == 0);
    const Kernel& k = d.pipelines().get(ShaderId::dequant, {static_cast<std::uint32_t>(q.fmt), vec ? 1u : 0u});
    const DequantPush pc{q.w, y, q.scale, static_cast<std::uint32_t>(q.rows), static_cast<std::uint32_t>(q.k),
                         q.rowbytes};
    launch(d, k, pc, groups_1d(std::uint64_t(q.rows) * (vec ? q.k / 8 : q.k), k));
}

const char* quant_linear_path(DeviceCtx& d, int B) {
    if (B <= 8) return "gemv";
    return d.info().coopmat_f16 ? "coopmat" : "dequant";
}

void quant_linear(DeviceCtx& d, const QuantW& q, std::uint64_t x, Dtype dt, int B, int ldx, std::uint64_t bias,
                  std::uint64_t y, int ldy, const char* op) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16) fail(op, "activations must be FP16 or BF16");
    if (B <= 0 || q.rows == 0) return;
    const Device dev = Device::vulkan(d.index());
    if (B <= 8) {
        // The vector paths read X in 16-byte words; a view that is not
        // aligned is copied (B x K elements, small next to W).
        const bool scalar = q.fmt == QF_INT8 && !(q.k % 16 == 0 && q.w % 16 == 0);
        if (!scalar && (x % 16 != 0 || ldx % 8 != 0)) {
            Tensor xa = Tensor::empty_on(dev, B, q.k, dt);
            for (int r = 0; r < B; ++r) {
                const Span a = d.allocator().resolve(x + std::uint64_t(r) * ldx * 2u, std::size_t(q.k) * 2u);
                const Span b = d.allocator().resolve(addr(xa.data) + std::uint64_t(r) * q.k * 2u, std::size_t(q.k) * 2u);
                d.stream().copy(a.buf, a.offset, b.buf, b.offset, std::size_t(q.k) * 2u);
            }
            gemv_q(d, q, addr(xa.data), dt, B, q.k, bias, y, ldy);
            return;
        }
        gemv_q(d, q, x, dt, B, ldx, bias, y, ldy);
        return;
    }
    GemmArgs g;
    g.op = op;
    g.a = x;
    g.c = y;
    g.bias = bias;
    g.da = g.dc = dt;
    g.m = B;
    g.n = q.rows;
    g.k = q.k;
    g.lda = ldx;
    g.ldc = ldy;
    if (d.info().coopmat_f16) {
        g.b = q.w;
        g.db = dt;
        g.qb = q.fmt;
        g.scale = q.scale;
        g.ldb = static_cast<int>(q.rowbytes);
        gemm(d, g);
        return;
    }
    Tensor w16 = Tensor::empty_on(dev, q.rows, q.k, Dtype::FP16);
    dequant(d, q, addr(w16.data));
    g.b = addr(w16.data);
    g.ldb = q.k;
    if (dt == Dtype::FP16) {
        g.db = dt;
        gemm(d, g);
        return;
    }
    // BF16 activations: an FP32 GEMM against the FP16 weight (rounding the
    // weight to BF16 would cost 2^-8 per weight), cast back once.
    auto view = [&](std::uint64_t a, int rows, int cols, int ld) {
        Tensor t = Tensor::empty_on(dev, rows, cols, Dtype::BF16);
        for (int r = 0; r < rows; ++r) {
            const Span sa = d.allocator().resolve(a + std::uint64_t(r) * ld * 2u, std::size_t(cols) * 2u);
            const Span sb = d.allocator().resolve(addr(t.data) + std::uint64_t(r) * cols * 2u, std::size_t(cols) * 2u);
            d.stream().copy(sa.buf, sa.offset, sb.buf, sb.offset, std::size_t(cols) * 2u);
        }
        return t;
    };
    Tensor x16 = view(x, B, q.k, ldx), x32 = Tensor::empty_on(dev, B, q.k, Dtype::FP32);
    cast(x16, x32, Dtype::FP32);
    Tensor b32, y32 = Tensor::empty_on(dev, B, q.rows, Dtype::FP32);
    if (bias) {
        Tensor b16 = view(bias, 1, q.rows, q.rows);
        b32 = Tensor::empty_on(dev, 1, q.rows, Dtype::FP32);
        cast(b16, b32, Dtype::FP32);
    }
    g.a = addr(x32.data); g.lda = q.k;
    g.c = addr(y32.data); g.ldc = q.rows;
    g.bias = bias ? addr(b32.data) : 0;
    g.da = g.dc = Dtype::FP32;
    g.db = Dtype::FP16;
    gemm(d, g);
    Tensor yb = Tensor::empty_on(dev, B, q.rows, Dtype::BF16);
    cast(y32, yb, Dtype::BF16);
    for (int r = 0; r < B; ++r) {
        const Span sa = d.allocator().resolve(addr(yb.data) + std::uint64_t(r) * q.rows * 2u, std::size_t(q.rows) * 2u);
        const Span sb = d.allocator().resolve(y + std::uint64_t(r) * ldy * 2u, std::size_t(q.rows) * 2u);
        d.stream().copy(sa.buf, sa.offset, sb.buf, sb.offset, std::size_t(q.rows) * 2u);
    }
}

namespace {

// ─── GGUF ops ───────────────────────────────────────────────────────────────

void check_fmt(const Tensor& W, Dtype want, const char* op) {
    if (W.dtype != want) fail(op, std::string("W must be Dtype::") + fmt_name(want == Dtype::Q8_0 ? QF_Q8_0 : want == Dtype::Q4_K ? QF_Q4K : QF_Q6K));
}

void dequant_op(const Tensor& Wq, Tensor& W16, Dtype want, const char* op) {
    check_fmt(Wq, want, op);
    const QuantW q = quant_weight(Wq, nullptr, op);
    if (W16.rows != q.rows || W16.cols != q.k || W16.dtype != Dtype::FP16) W16.resize(q.rows, q.k, Dtype::FP16);
    dequant(device_of(Wq), q, addr(W16.data));
}

std::uint64_t fp16_bias(const Tensor* bias, int out, bool batched, const char* op) {
    if (!bias || bias->size() == 0) return 0;
    need(op, bias->dtype == Dtype::FP16, "bias must be FP16");
    const bool ok = (bias->rows == out && bias->cols == 1) || (batched && bias->rows == 1 && bias->cols == out);
    need(op, ok, batched ? "bias shape must be (out,1) or (1,out)" : "bias shape must be (out, 1)");
    return addr(bias->data);
}

void gguf_gemv(const Tensor& Wq, const Tensor* bias, const Tensor& x, Tensor& y, Dtype want, const char* op) {
    check_fmt(Wq, want, op);
    const QuantW q = quant_weight(Wq, nullptr, op);
    need(op, x.dtype == Dtype::FP16, "x must be FP16");
    need(op, x.rows == q.k && x.cols == 1, "x shape must be (in, 1)");
    const std::uint64_t b = fp16_bias(bias, q.rows, false, op);
    if (y.rows != q.rows || y.cols != 1 || y.dtype != Dtype::FP16) y.resize(q.rows, 1, Dtype::FP16);
    quant_linear(device_of(Wq), q, addr(x.data), Dtype::FP16, 1, q.k, b, addr(y.data), q.rows, op);
}

void gguf_batched(const Tensor& Wq, const Tensor* bias, const Tensor& X, Tensor& Y, Dtype want, const char* op) {
    check_fmt(Wq, want, op);
    const QuantW q = quant_weight(Wq, nullptr, op);
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    need(op, X.cols == q.k, "shape mismatch (W.cols != X.cols)");
    const std::uint64_t b = fp16_bias(bias, q.rows, true, op);
    if (Y.rows != X.rows || Y.cols != q.rows || Y.dtype != Dtype::FP16) Y.resize(X.rows, q.rows, Dtype::FP16);
    if (X.rows == 0) return;
    quant_linear(device_of(Wq), q, addr(X.data), Dtype::FP16, X.rows, q.k, b, addr(Y.data), q.rows, op);
}

void dequant_q4k_to_fp16(const Tensor& W, Tensor& Y) { dequant_op(W, Y, Dtype::Q4_K, "dequant_q4k_to_fp16"); }
void dequant_q8_0_to_fp16(const Tensor& W, Tensor& Y) { dequant_op(W, Y, Dtype::Q8_0, "dequant_q8_0_to_fp16"); }
void dequant_q6k_to_fp16(const Tensor& W, Tensor& Y) { dequant_op(W, Y, Dtype::Q6_K, "dequant_q6k_to_fp16"); }
void linear_forward_q4k_fp16(const Tensor& W, const Tensor* b, const Tensor& x, Tensor& y) {
    gguf_gemv(W, b, x, y, Dtype::Q4_K, "linear_forward_q4k_fp16");
}
void linear_forward_q8_0_fp16(const Tensor& W, const Tensor* b, const Tensor& x, Tensor& y) {
    gguf_gemv(W, b, x, y, Dtype::Q8_0, "linear_forward_q8_0_fp16");
}
void linear_forward_q6k_fp16(const Tensor& W, const Tensor* b, const Tensor& x, Tensor& y) {
    gguf_gemv(W, b, x, y, Dtype::Q6_K, "linear_forward_q6k_fp16");
}
void linear_forward_batched_q4k_fp16(const Tensor& W, const Tensor* b, const Tensor& X, Tensor& Y) {
    gguf_batched(W, b, X, Y, Dtype::Q4_K, "linear_forward_batched_q4k_fp16");
}
void linear_forward_batched_q8_0_fp16(const Tensor& W, const Tensor* b, const Tensor& X, Tensor& Y) {
    gguf_batched(W, b, X, Y, Dtype::Q8_0, "linear_forward_batched_q8_0_fp16");
}
void linear_forward_batched_q6k_fp16(const Tensor& W, const Tensor* b, const Tensor& X, Tensor& Y) {
    gguf_batched(W, b, X, Y, Dtype::Q6_K, "linear_forward_batched_q6k_fp16");
}

// ─── INT8 (W8A16) ───────────────────────────────────────────────────────────

void linear_forward_batched_int8w_fp16(const Tensor& W, const Tensor& scales, const Tensor* bias, const Tensor& X,
                                       Tensor& Y) {
    const char* op = "linear_forward_batched_int8w_fp16";
    need(op, W.dtype == Dtype::INT8, "W must be INT8");
    const QuantW q = quant_weight(W, &scales, op);
    need(op, X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16, "X must be FP16 or BF16");
    need(op, X.cols == q.k, "shape mismatch (W.cols != X.cols)");
    std::uint64_t b = 0;
    if (bias && bias->size() > 0) {
        need(op, bias->dtype == X.dtype, "bias dtype must match X");
        need(op, bias->size() == q.rows, "bias must have out elements");
        b = addr(bias->data);
    }
    if (Y.rows != X.rows || Y.cols != q.rows || Y.dtype != X.dtype) Y.resize(X.rows, q.rows, X.dtype);
    if (X.rows == 0) return;
    quant_linear(device_of(W), q, addr(X.data), X.dtype, X.rows, q.k, b, addr(Y.data), q.rows, op);
}

// Y(M, Nb) = dequant(W)(M, K) X(K, Nb): the linear on X^T, transposed back
// (the transposes move (K + M) Nb elements, small next to the M K Nb MACs).
void matmul_int8w_fp16(const Tensor& W, const Tensor& scales, const Tensor& X, Tensor& Y) {
    const char* op = "matmul_int8w_fp16";
    need(op, W.dtype == Dtype::INT8, "W_int8 must be INT8");
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    const QuantW q = quant_weight(W, &scales, op);
    need(op, X.rows == q.k, "K mismatch (W.cols != X.rows)");
    const int Nb = X.cols;
    if (Y.rows != q.rows || Y.cols != Nb || Y.dtype != Dtype::FP16) Y.resize(q.rows, Nb, Dtype::FP16);
    if (Nb == 0) return;
    DeviceCtx& d = device_of(W);
    const Device dev = Device::vulkan(d.index());
    const Kernel& tk = d.pipelines().get(ShaderId::transpose_b2);
    auto transpose = [&](std::uint64_t src, std::uint64_t dst, int rows, int cols) {
        const TransposePush pc{src, dst, 1u, static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
        launch(d, tk, pc, std::min<std::uint32_t>((pc.cols + 31) / 32, 65535),
               std::min<std::uint32_t>((pc.rows + 31) / 32, 65535), 1);
    };
    need(op, (q.k + 31) / 32 <= 65535 && (Nb + 31) / 32 <= 65535 && (q.rows + 31) / 32 <= 65535,
         "matrix too large for one transpose");
    Tensor Xt = Tensor::empty_on(dev, Nb, q.k, Dtype::FP16);
    Tensor Yt = Tensor::empty_on(dev, Nb, q.rows, Dtype::FP16);
    transpose(addr(X.data), addr(Xt.data), q.k, Nb);
    quant_linear(d, q, addr(Xt.data), Dtype::FP16, Nb, q.k, 0, addr(Yt.data), q.rows, op);
    transpose(addr(Yt.data), addr(Y.data), Nb, q.rows);
}

void conv2d_int8w_fp16_forward(const Tensor& X, const Tensor& W, const Tensor& scales, const Tensor* bias, int N,
                               int C_in, int H, int Wd, int C_out, int kH, int kW, int stride_h, int stride_w,
                               int pad_h, int pad_w, int dil_h, int dil_w, int groups, Tensor& Y) {
    const char* op = "conv2d_int8w_fp16_forward";
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    need(op, W.dtype == Dtype::INT8, "W must be INT8");
    need(op, scales.dtype == Dtype::FP32, "scales must be FP32");
    need(op, !bias || bias->dtype == Dtype::FP16, "bias must be FP16");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >=1 and divide both C_in and C_out");
    need(op, N >= 0 && C_in > 0 && H > 0 && Wd > 0 && C_out > 0, "bad dimension");
    const int K = C_in / groups * kH * kW;
    need(op, W.rows == C_out && W.cols == K, "W shape mismatch");
    need(op, scales.rows == C_out && scales.cols == 1, "scales shape mismatch");
    need(op, !bias || bias->size() == C_out, "bias must have C_out elements");
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * Wd, "X is smaller than N*C_in*H*W");
    Conv2dArgs a;
    a.op = op;
    a.dt = Dtype::FP16;
    a.n = N; a.cin = C_in; a.h = H; a.wd = Wd; a.cout = C_out; a.kh = kH; a.kw = kW;
    a.sh = stride_h; a.sw = stride_w; a.ph = pad_h; a.pw = pad_w; a.dh = dil_h; a.dw = dil_w;
    a.groups = groups;
    need(op, kH >= 1 && kW >= 1 && stride_h >= 1 && stride_w >= 1 && dil_h >= 1 && dil_w >= 1 && pad_h >= 0 &&
                 pad_w >= 0, "kernel, stride and dilation must be >= 1 and padding >= 0");
    const int Ho = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wo = (Wd + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    const long long out_cols = static_cast<long long>(C_out) * Ho * Wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) Y.resize(N, static_cast<int>(out_cols), Dtype::FP16);
    if (N == 0) return;
    a.x = addr(X.data); a.w = addr(W.data); a.y = addr(Y.data); a.bias = bias ? addr(bias->data) : 0;
    a.scale = addr(scales.data);
    conv2d(device_of(X), a);
}

void conv3d_int8w_fp16_forward(const Tensor& X, const Tensor& W, const Tensor& scales, const Tensor* bias, int N,
                               int C_in, int T, int H, int Wd, int C_out, int kT, int kH, int kW, int stride_t,
                               int stride_h, int stride_w, int pad_t, int pad_h, int pad_w, int dil_t, int dil_h,
                               int dil_w, int groups, Tensor& Y) {
    const char* op = "conv3d_int8w_fp16_forward";
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    need(op, W.dtype == Dtype::INT8, "W must be INT8");
    need(op, !bias || bias->dtype == Dtype::FP16, "bias must be FP16");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >= 1 and divide both C_in and C_out");
    const long long K = static_cast<long long>(C_in / groups) * kT * kH * kW;
    need(op, W.rows == C_out && W.cols == K, "W shape must be (C_out, (C_in/groups)*kT*kH*kW)");
    const QuantW q = quant_weight(W, &scales, op);
    DeviceCtx& d = device_of(X);
    if (groups == 1 && kT == T && kH == H && kW == Wd && pad_t == 0 && pad_h == 0 && pad_w == 0) {
        // The patch embedding (kernel = whole input): one quantised linear.
        need(op, X.size() >= static_cast<long long>(N) * K, "X is smaller than N*C_in*T*H*W");
        need(op, !bias || bias->size() == C_out, "bias must have C_out elements");
        if (Y.rows != N || Y.cols != C_out || Y.dtype != Dtype::FP16) Y.resize(N, C_out, Dtype::FP16);
        if (N == 0) return;
        quant_linear(d, q, addr(X.data), Dtype::FP16, N, static_cast<int>(K), bias ? addr(bias->data) : 0,
                     addr(Y.data), C_out, op);
        return;
    }
    // Any other geometry: the direct kernel on an FP16 copy of the (small)
    // weight.
    Tensor W16 = Tensor::empty_on(Device::vulkan(d.index()), C_out, static_cast<int>(K), Dtype::FP16);
    dequant(d, q, addr(W16.data));
    conv3d_forward(X, W16, bias, N, C_in, T, H, Wd, C_out, kT, kH, kW, stride_t, stride_h, stride_w, pad_t, pad_h,
                   pad_w, dil_t, dil_h, dil_w, groups, Y);
}

}  // namespace

void fill_vulkan_vtable_quant(::brotensor::detail::OpsVTable& v) {
    v.dequant_q4k_to_fp16 = &dequant_q4k_to_fp16;
    v.dequant_q8_0_to_fp16 = &dequant_q8_0_to_fp16;
    v.dequant_q6k_to_fp16 = &dequant_q6k_to_fp16;
    v.linear_forward_q4k_fp16 = &linear_forward_q4k_fp16;
    v.linear_forward_q8_0_fp16 = &linear_forward_q8_0_fp16;
    v.linear_forward_q6k_fp16 = &linear_forward_q6k_fp16;
    v.linear_forward_batched_q4k_fp16 = &linear_forward_batched_q4k_fp16;
    v.linear_forward_batched_q8_0_fp16 = &linear_forward_batched_q8_0_fp16;
    v.linear_forward_batched_q6k_fp16 = &linear_forward_batched_q6k_fp16;
    v.linear_forward_batched_int8w_fp16 = &linear_forward_batched_int8w_fp16;
    v.matmul_int8w_fp16 = &matmul_int8w_fp16;
    v.conv2d_int8w_fp16_forward = &conv2d_int8w_fp16_forward;
    v.conv3d_int8w_fp16_forward = &conv3d_int8w_fp16_forward;
}

}  // namespace brotensor::detail::vulkan
