// Vulkan vision-model ops for brovisionml: SAM's attention with the
// decomposed 2D relative position (global and windowed), torchvision's
// modulated deformable convolution (BiRefNet), StyleGAN3's modulated_conv2d,
// and conv_transpose2d as a GEMM plus overlap-add (ops_conv.cpp's entry point
// calls it; the direct gather stays as the fallback). Contracts follow the
// CUDA backend (src/cuda/self_attention_bias.cu, deform_conv2d.cu,
// modulated_conv2d.cu, conv_transpose2d.cu): every operand in X's dtype
// (FP32 / FP16 / BF16), FP32 accumulation, outputs resized and overwritten.
//
// * SAM attention: per unit (the whole grid, or one window of the padded
//   grid gathered by rows), Q / K / V = X W^T + b in X's dtype (gemm()), Bh =
//   Q_h rel_pos_h^T and Bw = Q_h rel_pos_w^T (FP32, one batched GEMM over the
//   heads each), then the dense attention path with attn_softmax.comp adding
//   Bh[q, qh - kh + gh - 1] + Bw[q, qw - kw + gw - 1] to the scaled scores (the
//   (L, L) bias is never formed), O = Yc Wo^T + bo.
// * deform_conv2d: per image, group and slab of output pixels, a bilinear
//   im2col (misc.comp MI_DEFORM_COL) and the GEMM W_g col; bias by channel.
// * modulated_conv2d: the per-sample weights and demodulation coefficients in
//   one pass (MI_MODW), then one grouped conv2d with groups = N over the batch
//   seen as a single N * C_in channel image.
// * conv2d_backward_weight and modulated_conv2d_backward (GAN inversion in
//   brovisionml needs the latter): im2col slabs (MI_DEFORM_COL without
//   offsets) and dW += dY col^T through gemm(); the demodulation's chain rule,
//   ds and dW in misc.comp; dX is conv2d_backward_input with w''.
// * conv_transpose2d: cols (C_out kH kW, H W) = Wt_g^T X_g per image and group
//   through gemm(), then the 2D overlap-add gather (MI_COL2IM2D).

#include "detail/attention.h"
#include "detail/gemm.h"
#include "detail/kernels.h"
#include "detail/spatial.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y);              // ops_spatial.cpp
void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n);   // ops_copy.cpp
void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L);   // ops_elementwise.cpp
void conv2d_backward_input(const Tensor& Wt, const Tensor& dY, int N, int C_in, int H, int W, int C_out, int kH,
                           int kW, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,
                           Tensor& dX);                                       // ops_conv.cpp

namespace {

struct MiscPush {
    std::uint64_t a, b, c, d, e;
    std::uint32_t u[16];
    float f[4];
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint32_t u32(long long v) { return static_cast<std::uint32_t>(v); }
std::uint64_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }

void run_misc(DeviceCtx& d, Dtype dt, std::uint32_t code, const MiscPush& pc, std::uint64_t items, bool per_group,
              const char* op) {
    if (items == 0) return;
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::misc_f32, dt, op), {code});
    if (per_group) {
        const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535));
        const std::uint32_t gy = static_cast<std::uint32_t>(std::min<std::uint64_t>((items + gx - 1) / gx, 65535));
        launch(d, k, pc, gx, gy);
    } else {
        launch(d, k, pc, groups_1d(items, k));
    }
}

const void* opt(const Tensor* t) { return t && t->data && t->size() > 0 ? t->data : nullptr; }

// ─── SAM attention ─────────────────────────────────────────────────────────

// Y (rows, D) in dt = A (rows, D) W^T + b.
void project(DeviceCtx& d, const char* op, std::uint64_t a, int rows, const Tensor& W, const void* b, std::uint64_t y,
             Dtype dt) {
    GemmArgs g;
    g.op = op;
    g.a = a; g.b = addr(W.data); g.c = y; g.bias = addr(b);
    g.da = g.db = g.dc = dt;
    g.m = rows; g.n = W.rows; g.k = W.cols;
    g.lda = W.cols; g.ldb = W.cols; g.ldc = W.rows;
    gemm(d, g);
}

void sam_attention(const char* op, const Tensor& X, const Tensor& Wq, const Tensor* bq, const Tensor& Wk,
                   const Tensor* bk, const Tensor& Wv, const Tensor* bv, const Tensor& Wo, const Tensor* bo,
                   const Tensor& rel_h, const Tensor& rel_w, int H, int grid_h, int grid_w, int window, float scale,
                   Tensor& O) {
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    for (const Tensor* t : {&Wq, &Wk, &Wv, &Wo, &rel_h, &rel_w}) {
        need(op, t->dtype == dt, "Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    }
    const int L = X.rows, D = X.cols;
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    need(op, grid_h > 0 && grid_w > 0 && grid_h * grid_w == L, "grid_h*grid_w must equal X.rows");
    for (const Tensor* t : {&Wq, &Wk, &Wv, &Wo}) need(op, t->rows == D && t->cols == D, "Wq/Wk/Wv/Wo must be (D, D)");
    const int dh = D / H;
    const int gh = window > 0 ? window : grid_h, gw = window > 0 ? window : grid_w;
    need(op, rel_h.rows == 2 * gh - 1 && rel_h.cols == dh, window > 0 ? "rel_pos_h must be (2*window-1, head_dim)"
                                                                      : "rel_pos_h must be (2*grid_h-1, head_dim)");
    need(op, rel_w.rows == 2 * gw - 1 && rel_w.cols == dh, window > 0 ? "rel_pos_w must be (2*window-1, head_dim)"
                                                                      : "rel_pos_w must be (2*grid_w-1, head_dim)");
    for (const Tensor* b : {bq, bk, bv, bo}) {
        if (opt(b)) need(op, b->dtype == dt && b->size() == D, "biases must have D entries of X's dtype");
    }
    if (O.rows != L || O.cols != D || O.dtype != dt) O.resize(L, D, dt);
    if (L == 0 || D == 0) return;
    need(op, gh <= 0xffff && gw <= 0xffff, "grid too large");

    DeviceCtx& d = device_of(X);
    const ::brotensor::Device dev = X.device;
    const std::uint64_t es = esize(dt);

    // Units: the grid, or the windows of the zero-padded grid as row gathers
    // from X with one zero row appended (index L).
    int U = 1;
    const int Lu = gh * gw;
    Tensor Xw, back;
    if (window > 0) {
        const int nwh = (grid_h + window - 1) / window, nww = (grid_w + window - 1) / window;
        U = nwh * nww;
        std::vector<std::int32_t> fwd(static_cast<std::size_t>(U) * Lu), inv(static_cast<std::size_t>(L));
        for (int a = 0; a < nwh; ++a) {
            for (int b = 0; b < nww; ++b) {
                for (int lh = 0; lh < window; ++lh) {
                    for (int lw = 0; lw < window; ++lw) {
                        const int h = a * window + lh, w = b * window + lw;
                        const int r = (a * nww + b) * Lu + lh * window + lw;
                        const bool in = h < grid_h && w < grid_w;
                        fwd[static_cast<std::size_t>(r)] = in ? h * grid_w + w : L;
                        if (in) inv[static_cast<std::size_t>(h * grid_w + w)] = r;
                    }
                }
            }
        }
        Tensor Xp = Tensor::zeros_on(dev, L + 1, D, dt);
        vulkan::copy_d2d(X, 0, Xp, 0, L * D);
        const Tensor fi = Tensor::from_raw_bytes_on(dev, fwd.data(), U * Lu, 1, Dtype::INT32,
                                                         fwd.size() * sizeof(std::int32_t));
        back = Tensor::from_raw_bytes_on(dev, inv.data(), L, 1, Dtype::INT32, inv.size() * sizeof(std::int32_t));
        Xw = Tensor::empty_on(dev, U * Lu, D, dt);
        vulkan::gather_rows(Xp, fi, Xw);
    }
    const Tensor& Xu = window > 0 ? Xw : X;
    const int R = U * Lu;

    Tensor Q = Tensor::empty_on(dev, R, D, dt), K = Tensor::empty_on(dev, R, D, dt), V = Tensor::empty_on(dev, R, D, dt);
    project(d, op, addr(Xu.data), R, Wq, opt(bq), addr(Q.data), dt);
    project(d, op, addr(Xu.data), R, Wk, opt(bk), addr(K.data), dt);
    project(d, op, addr(Xu.data), R, Wv, opt(bv), addr(V.data), dt);

    // Bh (U, H, Lu, 2 gh - 1) and Bw likewise, FP32, from the unscaled Q.
    const int ph = 2 * gh - 1, pw = 2 * gw - 1;
    Tensor Bh = Tensor::empty_on(dev, U * H * Lu, ph, Dtype::FP32);
    Tensor Bw = Tensor::empty_on(dev, U * H * Lu, pw, Dtype::FP32);
    Tensor Yc = Tensor::empty_on(dev, R, D, dt);
    for (int u = 0; u < U; ++u) {
        const std::uint64_t qu = addr(Q.data) + std::uint64_t(u) * Lu * D * es;
        for (int axis = 0; axis < 2; ++axis) {
            const int P = axis == 0 ? ph : pw;
            GemmArgs g;
            g.op = op;
            g.a = qu; g.b = addr(axis == 0 ? rel_h.data : rel_w.data);
            g.c = addr(axis == 0 ? Bh.data : Bw.data) + std::uint64_t(u) * H * Lu * P * 4u;
            g.da = g.db = dt; g.dc = Dtype::FP32;
            g.m = Lu; g.n = P; g.k = dh;
            g.lda = D; g.ldb = dh; g.ldc = P;
            g.batch = H; g.sa = dh; g.sb = 0; g.sc = static_cast<long long>(Lu) * P;
            gemm(d, g);
        }
        AttnProblem p;
        p.op = op;
        p.dt = dt;
        const std::uint64_t off = std::uint64_t(u) * Lu * D * es;
        p.q = qu; p.k = addr(K.data) + off; p.v = addr(V.data) + off; p.o = addr(Yc.data) + off;
        p.lq = p.lk = Lu;
        p.ldq = p.ldk = p.ldo = D;
        p.hd = dh; p.hq = p.hkv = H;
        DenseExtras x;
        x.scale = scale;
        x.rel_h = addr(Bh.data) + std::uint64_t(u) * H * Lu * ph * 4u;
        x.rel_w = addr(Bw.data) + std::uint64_t(u) * H * Lu * pw * 4u;
        x.grid_h = gh; x.grid_w = gw;
        dense_attention(d, p, x);
    }

    if (window > 0) {
        Tensor Ow = Tensor::empty_on(dev, R, D, dt);
        project(d, op, addr(Yc.data), R, Wo, opt(bo), addr(Ow.data), dt);
        vulkan::gather_rows(Ow, back, O);
    } else {
        project(d, op, addr(Yc.data), R, Wo, opt(bo), addr(O.data), dt);
    }
}

}  // namespace

void self_attention_decomposed_rel_pos_forward(const Tensor& X, const Tensor& Wq, const Tensor* bq, const Tensor& Wk,
                                               const Tensor* bk, const Tensor& Wv, const Tensor* bv, const Tensor& Wo,
                                               const Tensor* bo, const Tensor& rel_pos_h, const Tensor& rel_pos_w,
                                               int num_heads, int grid_h, int grid_w, float scale, Tensor& O) {
    sam_attention("self_attention_decomposed_rel_pos_forward", X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h,
                  rel_pos_w, num_heads, grid_h, grid_w, 0, scale, O);
}

void self_attention_decomposed_rel_pos_windowed_forward(const Tensor& X, const Tensor& Wq, const Tensor* bq,
                                                        const Tensor& Wk, const Tensor* bk, const Tensor& Wv,
                                                        const Tensor* bv, const Tensor& Wo, const Tensor* bo,
                                                        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
                                                        int num_heads, int grid_h, int grid_w, int window,
                                                        float scale, Tensor& O) {
    const char* op = "self_attention_decomposed_rel_pos_windowed_forward";
    need(op, window > 0, "window must be >= 1");
    sam_attention(op, X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, window,
                  scale, O);
}

// ─── deformable convolution ────────────────────────────────────────────────

void deform_conv2d_forward(const Tensor& X, const Tensor& offset, const Tensor* mask, const Tensor& Wt,
                           const Tensor* bias, int N, int C_in, int H, int W, int C_out, int kH, int kW, int stride_h,
                           int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups, int deform_groups,
                           Tensor& Y) {
    const char* op = "deform_conv2d_forward";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, Wt.dtype == dt && offset.dtype == dt && (!opt(mask) || mask->dtype == dt) && (!opt(bias) || bias->dtype == dt),
         "X, offset, mask, Wt, bias dtype must match");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0, "groups must divide C_in and C_out");
    need(op, deform_groups >= 1 && C_in % deform_groups == 0, "deform_groups must divide C_in");
    need(op, N >= 0 && H > 0 && W > 0 && kH > 0 && kW > 0 && stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0 &&
                 pad_h >= 0 && pad_w >= 0,
         "bad geometry");
    need(op, stride_h < 65536 && stride_w < 65536 && pad_h < 65536 && pad_w < 65536 && dil_h < 65536 && dil_w < 65536,
         "geometry above 65535");
    const int Ho = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wo = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    const int Cgi = C_in / groups, Cgo = C_out / groups, kk = kH * kW, P = Ho * Wo;
    const long long Kd = static_cast<long long>(Cgi) * kk;
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * W, "X is smaller than N*C_in*H*W");
    need(op, Wt.size() == C_out * Kd, "Wt must be (C_out, C_in/groups*kH*kW)");
    need(op, offset.size() >= static_cast<long long>(N) * deform_groups * 2 * kk * P, "offset is too small");
    need(op, !opt(mask) || mask->size() >= static_cast<long long>(N) * deform_groups * kk * P, "mask is too small");
    need(op, !opt(bias) || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * P;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) Y.resize(N, static_cast<int>(out_cols), dt);
    if (N == 0) return;

    DeviceCtx& d = device_of(X);
    const std::uint64_t es = esize(dt);
    // Pixel slabs keep col at <= 32 Mi elements (a multiple of 64 pixels).
    long long pcn = std::max<long long>(1, (32ll << 20) / Kd);
    if (pcn < P) pcn = std::max<long long>(64, pcn / 64 * 64);
    pcn = std::min<long long>(pcn, P);
    Tensor col = Tensor::empty_on(X.device, static_cast<int>(Kd), static_cast<int>(pcn), dt);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int p0 = 0; p0 < P; p0 += static_cast<int>(pcn)) {
                const int np = static_cast<int>(std::min<long long>(pcn, P - p0));
                MiscPush pc{};
                pc.a = addr(X.data) + std::uint64_t(n) * C_in * H * W * es;
                pc.b = addr(offset.data) + std::uint64_t(n) * deform_groups * 2 * kk * P * es;
                pc.c = opt(mask) ? addr(mask->data) + std::uint64_t(n) * deform_groups * kk * P * es : 0;
                pc.d = addr(col.data);
                const std::uint32_t u[] = {u32(Cgi), u32(H), u32(W), u32(Wo), u32(P), u32(kH), u32(kW),
                                           u32(stride_h) | (u32(stride_w) << 16), u32(pad_h) | (u32(pad_w) << 16),
                                           u32(dil_h) | (u32(dil_w) << 16), u32(C_in / deform_groups),
                                           u32(g * Cgi), u32(p0), u32(np), 0, 0};
                std::copy(std::begin(u), std::end(u), pc.u);
                run_misc(d, dt, MI_DEFORM_COL, pc, std::uint64_t(Kd) * np, false, op);
                GemmArgs gm;
                gm.op = op;
                gm.a = addr(Wt.data) + std::uint64_t(g) * Cgo * Kd * es;
                gm.b = addr(col.data);
                gm.c = addr(Y.data) + (std::uint64_t(n) * C_out + std::uint64_t(g) * Cgo) * P * es + std::uint64_t(p0) * es;
                gm.da = gm.db = gm.dc = dt;
                gm.m = Cgo; gm.n = np; gm.k = static_cast<int>(Kd);
                gm.lda = static_cast<int>(Kd); gm.ldb = np; gm.ldc = P;
                gm.nb = true;
                gemm(d, gm);
            }
        }
    }
    if (opt(bias)) {
        for (int n = 0; n < N; ++n) {
            Tensor yn = Tensor::view(Y.device, static_cast<char*>(Y.data) + std::size_t(n) * out_cols * es, 1,
                                     static_cast<int>(out_cols), dt);
            vulkan::add_channel_bias_inplace(yn, *bias, C_out, P);
        }
    }
}

// ─── StyleGAN3 modulated convolution ───────────────────────────────────────

void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s, int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW, int pad_h, int pad_w, bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y) {
    const char* op = "modulated_conv2d_forward";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, W.dtype == dt && s.dtype == dt, "W/s dtype must match X");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 && pad_h >= 0 && pad_w >= 0, "bad dimension");
    need(op, X.rows == N && X.cols == static_cast<long long>(C_in) * H * Wd, "X shape mismatch");
    need(op, W.rows == C_out && W.cols == wk, "W shape mismatch");
    need(op, s.rows == N && s.cols == C_in, "s shape mismatch");
    const int Ho = H + 2 * pad_h - (kH - 1), Wo = Wd + 2 * pad_w - (kW - 1);
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    if (dcoef.rows != N || dcoef.cols != C_out || dcoef.dtype != Dtype::FP32) dcoef.resize(N, C_out, Dtype::FP32);
    const long long out_cols = static_cast<long long>(C_out) * Ho * Wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) Y.resize(N, static_cast<int>(out_cols), dt);
    if (N == 0) return;
    need(op, N <= 65535, "N above 65535");

    DeviceCtx& d = device_of(X);
    Tensor Wn = Tensor::empty_on(X.device, N * C_out, static_cast<int>(wk), dt);
    MiscPush pc{};
    pc.a = addr(W.data); pc.b = addr(s.data); pc.c = addr(Wn.data); pc.d = addr(dcoef.data);
    pc.u[0] = u32(N); pc.u[1] = u32(C_out); pc.u[2] = u32(C_in); pc.u[3] = u32(kH * kW); pc.u[4] = demodulate ? 1u : 0u;
    pc.f[0] = eps;
    run_misc(d, dt, MI_MODW, pc, std::uint64_t(N) * C_out, true, op);

    Conv2dArgs a;
    a.op = op;
    a.x = addr(X.data); a.w = addr(Wn.data); a.y = addr(Y.data);
    a.dt = dt;
    a.n = 1; a.cin = N * C_in; a.h = H; a.wd = Wd; a.cout = N * C_out;
    a.kh = kH; a.kw = kW; a.ph = pad_h; a.pw = pad_w;
    a.groups = N;
    conv2d(d, a);
}

// ─── weight gradients through im2col ───────────────────────────────────────

namespace {

// col (Cgi kH kW, np) of image `x` (C_in, H, W), input channels from
// ic_base, output pixels [p0, p0 + np): MI_DEFORM_COL without offsets.
void im2col(DeviceCtx& d, Dtype dt, std::uint64_t x, std::uint64_t col, int Cgi, int H, int W, int Ho, int Wo, int kH,
            int kW, int sh, int sw, int ph, int pw, int dh, int dw, int ic_base, int p0, int np, const char* op) {
    MiscPush pc{};
    pc.a = x; pc.d = col;
    const std::uint32_t u[] = {u32(Cgi), u32(H), u32(W), u32(Wo), u32(static_cast<long long>(Ho) * Wo), u32(kH),
                               u32(kW), u32(sh) | (u32(sw) << 16), u32(ph) | (u32(pw) << 16),
                               u32(dh) | (u32(dw) << 16), 1u, u32(ic_base), u32(p0), u32(np), 0, 0};
    std::copy(std::begin(u), std::end(u), pc.u);
    run_misc(d, dt, MI_DEFORM_COL, pc, std::uint64_t(Cgi) * kH * kW * np, false, op);
}

// Pixel slab for a col of `rows` rows: <= 32 Mi elements, a multiple of 64.
int slab(long long rows, int P) {
    long long pcn = std::max<long long>(1, (32ll << 20) / std::max<long long>(rows, 1));
    if (pcn < P) pcn = std::max<long long>(64, pcn / 64 * 64);
    return static_cast<int>(std::min<long long>(pcn, P));
}

// dW_g (Cgo, Kd) (+)= dY_g (Cgo, P) col^T over every image, pixel slab by
// slab: C in `dc` at address `dw` (row pitch Kd), A / col in `dt`.
void weight_grad(DeviceCtx& d, Dtype dt, Dtype dc, std::uint64_t x, std::uint64_t dy, std::uint64_t dw, int n_img,
                 std::uint64_t x_img_stride, std::uint64_t dy_img_stride, std::uint64_t dw_img_stride, int C_in,
                 int H, int W, int C_out, int kH, int kW, int sh, int sw, int ph, int pw, int dh, int dwl, int groups,
                 bool accumulate, const char* op) {
    const int Ho = (H + 2 * ph - dh * (kH - 1) - 1) / sh + 1, Wo = (W + 2 * pw - dwl * (kW - 1) - 1) / sw + 1;
    const int P = Ho * Wo, Cgi = C_in / groups, Cgo = C_out / groups;
    const long long Kd = static_cast<long long>(Cgi) * kH * kW;
    const int np_max = slab(Kd, P);
    const std::uint64_t es = esize(dt), ces = esize(dc);
    Tensor col = Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(Kd), np_max, dt);
    for (int n = 0; n < n_img; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int p0 = 0; p0 < P; p0 += np_max) {
                const int np = std::min(np_max, P - p0);
                im2col(d, dt, x + n * x_img_stride, addr(col.data), Cgi, H, W, Ho, Wo, kH, kW, sh, sw, ph, pw, dh, dwl,
                       g * Cgi, p0, np, op);
                GemmArgs gm;
                gm.op = op;
                gm.a = dy + n * dy_img_stride + (std::uint64_t(g) * Cgo * P + p0) * es;
                gm.b = addr(col.data);
                gm.c = dw + n * dw_img_stride + std::uint64_t(g) * Cgo * Kd * ces;
                gm.da = gm.db = dt; gm.dc = dc;
                gm.m = Cgo; gm.n = static_cast<int>(Kd); gm.k = np;
                gm.lda = P; gm.ldb = np; gm.ldc = static_cast<int>(Kd);
                gm.epi = (accumulate || p0 > 0) ? EPI_ACCUM : EPI_STORE;
                gemm(d, gm);
            }
        }
    }
}

}  // namespace

// dWt (C_out, C_in / groups kH kW) += sum over images of dY_g col_g^T,
// accumulated in dWt's (= X's) dtype one pixel slab at a time.
void conv2d_backward_weight(const Tensor& X, const Tensor& dY, int N, int C_in, int H, int W, int C_out, int kH,
                            int kW, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,
                            Tensor& dWt) {
    const char* op = "conv2d_backward_weight";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, dY.dtype == dt && dWt.dtype == dt, "X, dY, dWt dtype must match");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0, "groups must divide C_in and C_out");
    need(op, N >= 0 && H > 0 && W > 0 && kH > 0 && kW > 0 && stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0 &&
                 pad_h >= 0 && pad_w >= 0 && stride_h < 65536 && stride_w < 65536 && pad_h < 65536 &&
                 pad_w < 65536 && dil_h < 65536 && dil_w < 65536,
         "bad geometry");
    const int Ho = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wo = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    const long long Kd = static_cast<long long>(C_in / groups) * kH * kW;
    need(op, dWt.size() == C_out * Kd, "dWt shape mismatch");
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * W, "X is smaller than N*C_in*H*W");
    need(op, dY.size() >= static_cast<long long>(N) * C_out * Ho * Wo, "dY is smaller than N*C_out*H_out*W_out");
    if (N == 0) return;
    const std::uint64_t es = esize(dt);
    // Each image accumulates onto dWt (the contract: caller zeros).
    weight_grad(device_of(X), dt, dt, addr(X.data), addr(dY.data), addr(dWt.data), N,
                std::uint64_t(C_in) * H * W * es, std::uint64_t(C_out) * Ho * Wo * es, 0, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, true, op);
}

// Per sample: dw'' (FP32) from im2col, dw' through the demodulation, ds and
// (when dW is committed) dW from dw', dX through conv2d_backward_input with
// w'' over the batch as one grouped convolution.
void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s, const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd, int C_out, int kH, int kW, int pad_h, int pad_w,
                               bool demodulate, float eps, Tensor& dX, Tensor& dW, Tensor& ds) {
    const char* op = "modulated_conv2d_backward";
    (void)eps;   // the forward's coefficients come in as dcoef
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, W.dtype == dt && s.dtype == dt && dY.dtype == dt, "W/s/dY dtype must match X");
    need(op, dcoef.dtype == Dtype::FP32, "dcoef must be FP32");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 && pad_h >= 0 && pad_w >= 0 && N <= 65535,
         "bad dimension");
    need(op, W.rows == C_out && W.cols == wk, "W shape mismatch");
    need(op, s.rows == N && s.cols == C_in, "s shape mismatch");
    need(op, dcoef.rows == N && dcoef.cols == C_out, "dcoef must be (N, C_out)");
    const int Ho = H + 2 * pad_h - (kH - 1), Wo = Wd + 2 * pad_w - (kW - 1);
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * Wd, "X shape mismatch");
    need(op, dY.size() >= static_cast<long long>(N) * C_out * Ho * Wo, "dY shape mismatch");
    const bool want_dW = dW.data != nullptr;
    if (want_dW) need(op, dW.dtype == dt && dW.rows == C_out && dW.cols == wk, "dW must be (C_out, C_in kH kW) in X's dtype");
    if (dX.rows != N || dX.cols != static_cast<long long>(C_in) * H * Wd || dX.dtype != dt) {
        dX.resize(N, static_cast<int>(static_cast<long long>(C_in) * H * Wd), dt);
    }
    if (ds.rows != N || ds.cols != C_in || ds.dtype != dt) ds.resize(N, C_in, dt);
    if (N == 0) return;
    DeviceCtx& d = device_of(X);
    const ::brotensor::Device dev = X.device;
    const std::uint64_t es = esize(dt);

    // w'' from the forward's coefficients, then dX = conv2d_backward_input.
    Tensor Wn = Tensor::empty_on(dev, N * C_out, static_cast<int>(wk), dt);
    MiscPush pm{};
    pm.a = addr(W.data); pm.b = addr(s.data); pm.c = addr(Wn.data); pm.d = addr(dcoef.data);
    pm.u[0] = u32(N); pm.u[1] = u32(C_out); pm.u[2] = u32(C_in); pm.u[3] = u32(kH * kW);
    pm.u[4] = demodulate ? 1u : 0u; pm.u[5] = 1u;
    run_misc(d, dt, MI_MODW, pm, std::uint64_t(N) * C_out, true, op);
    Tensor dXv = Tensor::view(dev, dX.data, 1, static_cast<int>(static_cast<long long>(N) * C_in * H * Wd), dt);
    const Tensor dYv = Tensor::view(dev, dY.data, 1, static_cast<int>(static_cast<long long>(N) * C_out * Ho * Wo), dt);
    vulkan::conv2d_backward_input(Wn, dYv, 1, N * C_in, H, Wd, N * C_out, kH, kW, 1, 1, pad_h, pad_w, 1, 1, N, dXv);

    // dw'' (N, C_out, wk) FP32: each sample its own weight gradient.
    Tensor dwp = Tensor::empty_on(dev, N * C_out, static_cast<int>(wk), Dtype::FP32);
    weight_grad(d, dt, Dtype::FP32, addr(X.data), addr(dY.data), addr(dwp.data), N,
                std::uint64_t(C_in) * H * Wd * es, std::uint64_t(C_out) * Ho * Wo * es,
                std::uint64_t(C_out) * wk * 4u, C_in, H, Wd, C_out, kH, kW, 1, 1, pad_h, pad_w, 1, 1, 1, false, op);
    pm.c = addr(dwp.data);
    run_misc(d, dt, MI_MODW_BWD, pm, std::uint64_t(N) * C_out, true, op);   // dw'' -> dw'
    MiscPush pd = pm;
    pd.a = addr(W.data); pd.c = addr(dwp.data); pd.d = addr(ds.data);
    run_misc(d, dt, MI_MODW_DS, pd, std::uint64_t(N) * C_in, false, op);
    if (want_dW) {
        MiscPush pw = pm;
        pw.b = addr(s.data); pw.c = addr(dwp.data); pw.e = addr(dW.data);
        run_misc(d, dt, MI_MODW_DW, pw, std::uint64_t(C_out) * wk, false, op);
    }
}

// ─── conv_transpose2d as GEMM + overlap-add ────────────────────────────────

// Called by ops_conv.cpp's conv_transpose2d_forward after its checks, with Y
// sized. Returns false (nothing recorded) when one image's columns would
// exceed the 256 MiB budget; the caller then takes the direct gather.
bool conv_transpose2d_gemm(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int H, int W,
                           int C_out, int kH, int kW, int sh, int sw, int ph, int pw, int dh, int dw, int groups,
                           int Ho, int Wo, Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    const Dtype dt = X.dtype;
    const Dtype cdt = dt == Dtype::BF16 ? Dtype::FP32 : dt;
    const std::uint64_t es = esize(dt), ces = esize(cdt);
    const int kk = kH * kW, HW = H * W, Cgi = C_in / groups, Cgo = C_out / groups, M = Cgo * kk;
    const std::uint64_t per_image = std::uint64_t(C_out) * kk * HW;
    constexpr std::uint64_t kBudget = 256ull << 20;
    if (per_image * ces > kBudget || per_image > 0x7fffffffull) return false;
    DeviceCtx& d = device_of(X);
    const int nb = static_cast<int>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(N, kBudget / (per_image * ces))));
    Tensor cols = Tensor::empty_on(X.device, nb, static_cast<int>(per_image), cdt);
    for (int n0 = 0; n0 < N; n0 += nb) {
        const int nn = std::min(nb, N - n0);
        for (int g = 0; g < groups; ++g) {
            GemmArgs gm;
            gm.op = op;
            gm.da = gm.db = dt;
            gm.dc = cdt;
            gm.a = addr(Wt.data) + std::uint64_t(g) * Cgi * M * es;
            gm.ta = true;
            gm.lda = M;
            gm.b = addr(X.data) + (std::uint64_t(n0) * C_in + std::uint64_t(g) * Cgi) * HW * es;
            gm.nb = true;
            gm.ldb = HW;
            gm.c = addr(cols.data) + std::uint64_t(g) * M * HW * ces;
            gm.ldc = HW;
            gm.m = M; gm.n = HW; gm.k = Cgi;
            gm.batch = nn;
            gm.sa = 0;
            gm.sb = static_cast<long long>(C_in) * HW;
            gm.sc = static_cast<long long>(per_image);
            gemm(d, gm);
        }
        MiscPush pc{};
        pc.a = addr(cols.data); pc.b = opt(bias) ? addr(bias->data) : 0;
        pc.c = addr(Y.data) + std::uint64_t(n0) * C_out * Ho * Wo * es;
        const std::uint32_t u[] = {u32(nn), u32(C_out), u32(H), u32(W), u32(kH), u32(kW), u32(sh), u32(sw), u32(ph),
                                   u32(pw), u32(dh), u32(dw), u32(Cgo), u32(Ho), u32(Wo), cdt != dt ? 1u : 0u};
        std::copy(std::begin(u), std::end(u), pc.u);
        run_misc(d, dt, MI_COL2IM2D, pc, std::uint64_t(nn) * C_out * Ho * Wo, false, op);
    }
    return true;
}

void fill_vulkan_vtable_vision(::brotensor::detail::OpsVTable& v) {
    v.self_attention_decomposed_rel_pos_forward = &self_attention_decomposed_rel_pos_forward;
    v.self_attention_decomposed_rel_pos_windowed_forward = &self_attention_decomposed_rel_pos_windowed_forward;
    v.deform_conv2d_forward = &deform_conv2d_forward;
    v.modulated_conv2d_forward = &modulated_conv2d_forward;
    v.modulated_conv2d_backward = &modulated_conv2d_backward;
    v.conv2d_backward_weight = &conv2d_backward_weight;
}

}  // namespace brotensor::detail::vulkan
