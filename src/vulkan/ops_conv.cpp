// Vulkan convolutions: conv2d_forward (implicit GEMM through cooperative
// matrix for FP16 / BF16, a SIMT implicit GEMM for FP32 and devices without
// it, a direct kernel for depthwise and narrow grouped convolutions),
// conv3d_forward (a GEMM when the kernel covers the whole input, the direct
// kernel otherwise), conv_transpose2d_forward (direct, gather form) and the
// two bias gradients. Contracts follow the CUDA backend (src/cuda/conv2d.cu,
// conv3d.cu, conv_transpose2d.cu): X, Wt and bias share a dtype (FP32 /
// FP16 / BF16), Y is resized to (N, C_out * H_out * W_out) in X's dtype and
// overwritten; bias gradients accumulate. Design: docs/vulkan.md
// "Convolution".

#include "detail/gemm.h"
#include "detail/kernels.h"
#include "detail/quant.h"
#include "detail/spatial.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct ConvPush {
    std::uint64_t x, w, y, bias, scale;
    std::uint32_t m, n, k, h, wd, wo, cin, cout, groups, tm0, tn0;
};

struct DirectPush {
    std::uint64_t x, w, y, bias;
    std::uint32_t n, cin, t, h, wd, cout, to, ho, wo;
    std::uint32_t kt, kh, kw;
    std::int32_t st, sh, sw, pt, ph, pw, dt, dh, dw;
    std::uint32_t groups;
};

struct GnPush {
    std::uint64_t x, y, g, b, part, rm, rv;
    std::uint32_t tile, hw, cpg, groups, splits, chunk, total, c;
    float eps;
};

// Cooperative-matrix tiles (BM x BN x BK, WM x WN per subgroup).
struct CmCfg { std::uint32_t bm, bn, bk, wm, wn; };
constexpr CmCfg kCmCfgs[] = {
    {256, 128, 32, 64, 64},
    {128, 64, 32, 64, 32},
    {64, 64, 32, 32, 32},
    {128, 128, 32, 64, 64},   // the rest only by BROTENSOR_VK_CONV_CFG
    {128, 128, 32, 64, 32},
    {256, 64, 32, 64, 32},
};
struct SimtCfg { std::uint32_t bm, bn, bk, tm, tn; };
constexpr SimtCfg kSimtCfgs[] = {
    {128, 128, 16, 8, 8},
    {64, 64, 16, 4, 4},
};

std::atomic<int> g_override{0};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

std::uint32_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }

int out_dim(int in, int pad, int dil, int k, int stride) { return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1; }

const CmCfg* forced_cm() {
    static const CmCfg* f = []() -> const CmCfg* {
        const char* e = std::getenv("BROTENSOR_VK_CONV_CFG");
        if (!e || !*e) return nullptr;
        for (const CmCfg& c : kCmCfgs) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%u,%u,%u,%u,%u", c.bm, c.bn, c.bk, c.wm, c.wn);
            if (std::string(buf) == e) return &c;
        }
        return nullptr;
    }();
    return f;
}

// Tile choice on (M = Cg_out, N = H_out W_out, Z = images * groups),
// measured on the VAE / U-Net shapes (docs/vulkan.md "Convolution"): the
// implicit GEMM is bound by gathering B, so the winners are the tiles with
// the fewest B chunks per thread: 256x128 (8 subgroups of 64x64, 25-28 TF/s)
// when M fills 256-row tiles and there are >= 48 of them, 128x64 (4 of
// 64x32, 20-22 TF/s) when M fills 128-row tiles, else 64x64. A tile row
// more than 10% empty loses more than the bigger tile gains.
const CmCfg& pick_cm(std::uint64_t m, std::uint64_t n, std::uint64_t z) {
    if (const CmCfg* f = forced_cm()) return *f;
    auto fill = [&](std::uint64_t bm) { return double(m) / double(cdiv(m, bm) * bm); };
    if (m >= 256 && fill(256) >= 0.9 && cdiv(m, 256) * cdiv(n, 128) * z >= 48) return kCmCfgs[0];
    if (m >= 128 && fill(128) >= 0.9) return kCmCfgs[1];
    return kCmCfgs[2];
}

struct Geo {
    int ho, wo, cgi, cgo, k;
};

Geo geometry(const Conv2dArgs& a) {
    Geo g{};
    if (a.groups < 1 || a.cin % a.groups != 0 || a.cout % a.groups != 0) {
        fail(a.op, "groups must be >= 1 and divide both C_in and C_out");
    }
    if (a.kh < 1 || a.kw < 1 || a.sh < 1 || a.sw < 1 || a.dh < 1 || a.dw < 1 || a.ph < 0 || a.pw < 0) {
        fail(a.op, "kernel, stride and dilation must be >= 1 and padding >= 0");
    }
    g.ho = out_dim(a.h, a.ph, a.dh, a.kh, a.sh);
    g.wo = out_dim(a.wd, a.pw, a.dw, a.kw, a.sw);
    if (g.ho <= 0 || g.wo <= 0) fail(a.op, "non-positive output shape");
    g.cgi = a.cin / a.groups;
    g.cgo = a.cout / a.groups;
    g.k = g.cgi * a.kh * a.kw;
    return g;
}

bool implicit_gemm_ok(const Conv2dArgs&, const Geo& g) {
    return g_override.load(std::memory_order_relaxed) != 2 && g.cgo >= 16 && g.k >= 16;
}

bool coopmat_ok(DeviceCtx& d, const Conv2dArgs& a, const Geo& g) {
    return implicit_gemm_ok(a, g) && g_override.load(std::memory_order_relaxed) == 0 && d.info().coopmat_f16 &&
           a.dt != Dtype::FP32;
}

void conv_direct(DeviceCtx& d, std::uint32_t mode, Dtype dt, const DirectPush& pc) {
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::conv_direct_f32, dt), {mode});
    const std::uint64_t total = std::uint64_t(pc.n) * pc.cout * pc.to * pc.ho * pc.wo;
    if (total == 0) return;
    if (total > 0xffffffffULL) fail("conv", "output too large for one dispatch");
    launch(d, k, pc, groups_1d(total, k));
}

void conv_coopmat(DeviceCtx& d, const Conv2dArgs& a, const Geo& g) {
    const std::uint32_t P = static_cast<std::uint32_t>(g.ho * g.wo);
    const std::uint32_t M = static_cast<std::uint32_t>(g.cgo);
    const std::uint32_t K = static_cast<std::uint32_t>(g.k);
    const CmCfg& c = pick_cm(M, P, std::uint64_t(a.n) * a.groups);
    const std::uint32_t wg = (c.bm / c.wm) * (c.bn / c.wn) * 32u;
    const bool avec = a.scale ? a.w % 8 == 0 && K % 8 == 0 : a.w % 16 == 0 && K % 8 == 0;
    const bool rowch = g.wo % 8 == 0;
    // B chunk loads (conv_cm.comp BVEC): one aligned 16-byte load for a
    // plain 1x1, two shifted ones for any stride-1 kernel over rows of whole
    // 16-byte chunks, eight 2-byte loads otherwise.
    std::uint32_t bvec = 0;
    if (a.kh == 1 && a.kw == 1 && a.sh == 1 && a.sw == 1 && a.ph == 0 && a.pw == 0 && (a.h * a.wd) % 8 == 0 &&
        a.x % 16 == 0) {
        bvec = 1;
    } else if (a.sw == 1 && rowch && a.wd % 8 == 0 && a.x % 16 == 0) {
        bvec = 2;
    }
    const bool direct_ok = a.dt == Dtype::FP16 && P % 8 == 0 && a.y % 16 == 0 &&
                           (a.bias == 0 || (a.bias % 16 == 0 && (a.groups == 1 || M % 8 == 0)));
    const ShaderId id = a.dt == Dtype::FP16 ? ShaderId::conv_cm_f16 : ShaderId::conv_cm_bf16;
    auto kernel = [&](bool direct) -> const Kernel& {
        const std::uint32_t spec[] = {c.bm, c.bn, c.bk, c.wm, c.wn, wg, 0u, avec ? 1u : 0u, direct ? 1u : 0u,
                                      a.bias ? 1u : 0u,
                                      static_cast<std::uint32_t>(a.kh), static_cast<std::uint32_t>(a.kw),
                                      static_cast<std::uint32_t>(a.sh), static_cast<std::uint32_t>(a.sw),
                                      static_cast<std::uint32_t>(a.ph), static_cast<std::uint32_t>(a.pw),
                                      static_cast<std::uint32_t>(a.dh), static_cast<std::uint32_t>(a.dw),
                                      rowch ? 1u : 0u, a.accum ? 1u : 0u, bvec,
                                      a.scale ? static_cast<std::uint32_t>(QF_INT8) : 0u};
        return d.pipelines().get(id, spec, static_cast<std::uint32_t>(std::size(spec)), 32);
    };
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv(P, c.bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(M, c.bm));
    if (gx > 65535 || gy > 65535) fail(a.op, "image too large for one dispatch");
    const std::uint32_t fx = direct_ok ? P / c.bn : 0, fy = direct_ok ? M / c.bm : 0;
    // z = image * groups + group, in chunks of whole images.
    const int per = std::max(1, 65535 / a.groups);
    for (int i0 = 0; i0 < a.n; i0 += per) {
        const int ni = std::min(per, a.n - i0);
        ConvPush pc{};
        pc.x = a.x + std::uint64_t(i0) * a.cin * a.h * a.wd * 2u;
        pc.w = a.w;
        pc.y = a.y + std::uint64_t(i0) * a.cout * P * 2u;
        pc.bias = a.bias;
        pc.scale = a.scale;
        pc.m = M; pc.n = P; pc.k = K;
        pc.h = static_cast<std::uint32_t>(a.h);
        pc.wd = static_cast<std::uint32_t>(a.wd);
        pc.wo = static_cast<std::uint32_t>(g.wo);
        pc.cin = static_cast<std::uint32_t>(a.cin);
        pc.cout = static_cast<std::uint32_t>(a.cout);
        pc.groups = static_cast<std::uint32_t>(a.groups);
        const std::uint32_t gz = static_cast<std::uint32_t>(ni * a.groups);
        if (fx > 0 && fy > 0) {
            launch(d, kernel(true), pc, fx, fy, gz);
            if (fx < gx) {
                ConvPush p = pc;
                p.tn0 = fx;
                launch(d, kernel(false), p, gx - fx, gy, gz);
            }
            if (fy < gy) {
                ConvPush p = pc;
                p.tm0 = fy;
                launch(d, kernel(false), p, fx, gy - fy, gz);
            }
        } else {
            launch(d, kernel(false), pc, gx, gy, gz);
        }
    }
}

void conv_simt(DeviceCtx& d, const Conv2dArgs& a, const Geo& g) {
    const std::uint32_t P = static_cast<std::uint32_t>(g.ho * g.wo);
    const std::uint32_t M = static_cast<std::uint32_t>(g.cgo);
    const std::uint64_t tiles_big = cdiv(M, kSimtCfgs[0].bm) * cdiv(P, kSimtCfgs[0].bn) * std::uint64_t(a.n) * a.groups;
    const SimtCfg& c = tiles_big >= 160 ? kSimtCfgs[0] : kSimtCfgs[1];
    const std::uint32_t wg = (c.bm / c.tm) * (c.bn / c.tn);
    const std::uint32_t spec[] = {c.bm, c.bn, c.bk, c.tm, c.tn, wg,
                                  static_cast<std::uint32_t>(a.kh), static_cast<std::uint32_t>(a.kw),
                                  static_cast<std::uint32_t>(a.sh), static_cast<std::uint32_t>(a.sw),
                                  static_cast<std::uint32_t>(a.ph), static_cast<std::uint32_t>(a.pw),
                                  static_cast<std::uint32_t>(a.dh), static_cast<std::uint32_t>(a.dw),
                                  a.accum ? 1u : 0u};
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::conv_simt_f32, a.dt, a.op), spec,
                                        static_cast<std::uint32_t>(std::size(spec)));
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv(P, c.bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(M, c.bm));
    if (gx > 65535 || gy > 65535) fail(a.op, "image too large for one dispatch");
    const std::uint32_t es = esize(a.dt);
    const int per = std::max(1, 65535 / a.groups);
    for (int i0 = 0; i0 < a.n; i0 += per) {
        const int ni = std::min(per, a.n - i0);
        ConvPush pc{};
        pc.x = a.x + std::uint64_t(i0) * a.cin * a.h * a.wd * es;
        pc.w = a.w;
        pc.y = a.y + std::uint64_t(i0) * a.cout * P * es;
        pc.bias = a.bias;
        pc.m = M; pc.n = P; pc.k = static_cast<std::uint32_t>(g.k);
        pc.h = static_cast<std::uint32_t>(a.h);
        pc.wd = static_cast<std::uint32_t>(a.wd);
        pc.wo = static_cast<std::uint32_t>(g.wo);
        pc.cin = static_cast<std::uint32_t>(a.cin);
        pc.cout = static_cast<std::uint32_t>(a.cout);
        pc.groups = static_cast<std::uint32_t>(a.groups);
        launch(d, k, pc, gx, gy, static_cast<std::uint32_t>(ni * a.groups));
    }
}

void check_same_dtype(const char* op, const Tensor& X, const Tensor& Wt, const Tensor* bias) {
    dt_code(X.dtype, op);
    if (Wt.dtype != X.dtype) fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) fail(op, "bias dtype must match X");
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

}  // namespace

void set_conv_override(int mode) { g_override.store(mode, std::memory_order_relaxed); }

const char* conv2d_path(DeviceCtx& d, const Conv2dArgs& a) {
    const Geo g = geometry(a);
    if (coopmat_ok(d, a, g)) return "coopmat";
    if (implicit_gemm_ok(a, g)) return "simt";
    return "direct";
}

void conv2d(DeviceCtx& d, const Conv2dArgs& a) {
    const Geo g = geometry(a);
    if (a.n == 0) return;
    if (a.scale != 0) {
        // INT8 weights: decoded in the cooperative-matrix kernel's A tiles;
        // any other path runs on an FP16 copy of the (small) weight.
        if (a.dt == Dtype::FP16 && coopmat_ok(d, a, g)) return conv_coopmat(d, a, g);
        QuantW q;
        q.fmt = QF_INT8; q.w = a.w; q.scale = a.scale; q.rows = a.cout; q.k = g.k;
        q.rowbytes = static_cast<std::uint32_t>(g.k);
        Tensor w16 = Tensor::empty_on(Device::vulkan(d.index()), a.cout, g.k, Dtype::FP16);
        dequant(d, q, addr(w16.data));
        Conv2dArgs b = a;
        b.w = addr(w16.data);
        b.scale = 0;
        return conv2d(d, b);
    }
    if (coopmat_ok(d, a, g)) return conv_coopmat(d, a, g);
    if (implicit_gemm_ok(a, g)) return conv_simt(d, a, g);
    if (a.accum) fail(a.op, "accumulating direct convolution");
    DirectPush pc{};
    pc.x = a.x; pc.w = a.w; pc.y = a.y; pc.bias = a.bias;
    pc.n = static_cast<std::uint32_t>(a.n);
    pc.cin = static_cast<std::uint32_t>(a.cin);
    pc.t = 1;
    pc.h = static_cast<std::uint32_t>(a.h);
    pc.wd = static_cast<std::uint32_t>(a.wd);
    pc.cout = static_cast<std::uint32_t>(a.cout);
    pc.to = 1;
    pc.ho = static_cast<std::uint32_t>(g.ho);
    pc.wo = static_cast<std::uint32_t>(g.wo);
    pc.kt = 1;
    pc.kh = static_cast<std::uint32_t>(a.kh);
    pc.kw = static_cast<std::uint32_t>(a.kw);
    pc.st = 1; pc.sh = a.sh; pc.sw = a.sw;
    pc.pt = 0; pc.ph = a.ph; pc.pw = a.pw;
    pc.dt = 1; pc.dh = a.dh; pc.dw = a.dw;
    pc.groups = static_cast<std::uint32_t>(a.groups);
    conv_direct(d, 0, a.dt, pc);
}

void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int H, int W,
                    int C_out, int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h,
                    int dil_w, int groups, Tensor& Y) {
    const char* op = "conv2d_forward";
    check_same_dtype(op, X, Wt, bias);
    Conv2dArgs a;
    a.dt = X.dtype;
    a.n = N; a.cin = C_in; a.h = H; a.wd = W; a.cout = C_out; a.kh = kH; a.kw = kW;
    a.sh = stride_h; a.sw = stride_w; a.ph = pad_h; a.pw = pad_w; a.dh = dil_h; a.dw = dil_w;
    a.groups = groups;
    need(op, N >= 0 && C_in > 0 && H > 0 && W > 0 && C_out > 0, "bad dimension");
    const Geo g = geometry(a);
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * W, "X is smaller than N*C_in*H*W");
    need(op, Wt.size() == static_cast<long long>(C_out) * g.k, "Wt must have C_out*(C_in/groups)*kH*kW elements");
    need(op, !bias || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * g.ho * g.wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(out_cols), X.dtype);
    if (N == 0) return;
    a.x = addr(X.data); a.w = addr(Wt.data); a.y = addr(Y.data); a.bias = bias ? addr(bias->data) : 0;
    conv2d(device_of(X), a);
}

void conv3d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW, int stride_t, int stride_h, int stride_w, int pad_t,
                    int pad_h, int pad_w, int dil_t, int dil_h, int dil_w, int groups, Tensor& Y) {
    const char* op = "conv3d_forward";
    check_same_dtype(op, X, Wt, bias);
    need(op, N >= 0 && C_in > 0 && T > 0 && H > 0 && W > 0 && C_out > 0, "bad dimension");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >= 1 and divide both C_in and C_out");
    need(op, kT >= 1 && kH >= 1 && kW >= 1 && stride_t >= 1 && stride_h >= 1 && stride_w >= 1 && dil_t >= 1 &&
                 dil_h >= 1 && dil_w >= 1 && pad_t >= 0 && pad_h >= 0 && pad_w >= 0,
         "kernel, stride and dilation must be >= 1 and padding >= 0");
    const int To = out_dim(T, pad_t, dil_t, kT, stride_t);
    const int Ho = out_dim(H, pad_h, dil_h, kH, stride_h);
    const int Wo = out_dim(W, pad_w, dil_w, kW, stride_w);
    need(op, To > 0 && Ho > 0 && Wo > 0, "non-positive output shape");
    const long long K = static_cast<long long>(C_in / groups) * kT * kH * kW;
    need(op, X.size() >= static_cast<long long>(N) * C_in * T * H * W, "X is smaller than N*C_in*T*H*W");
    need(op, Wt.size() == C_out * K, "Wt must have C_out*(C_in/groups)*kT*kH*kW elements");
    need(op, !bias || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * To * Ho * Wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(out_cols), X.dtype);
    if (N == 0) return;
    DeviceCtx& d = device_of(X);
    // The kernel covers the whole input (a patch embedding: Qwen-VL's
    // (T, H, W) = (2, 14, 14) patches): Y (N, C_out) = X (N, K) Wt^T + bias,
    // one GEMM.
    if (groups == 1 && kT == T && kH == H && kW == W && pad_t == 0 && pad_h == 0 && pad_w == 0) {
        GemmArgs gm;
        gm.a = addr(X.data); gm.b = addr(Wt.data); gm.c = addr(Y.data);
        gm.bias = bias ? addr(bias->data) : 0;
        gm.da = gm.db = gm.dc = X.dtype;
        gm.m = N; gm.n = C_out; gm.k = static_cast<int>(K);
        gm.lda = static_cast<int>(K); gm.ldb = static_cast<int>(K); gm.ldc = C_out;
        gm.op = op;
        gemm(d, gm);
        return;
    }
    DirectPush pc{};
    pc.x = addr(X.data); pc.w = addr(Wt.data); pc.y = addr(Y.data); pc.bias = bias ? addr(bias->data) : 0;
    pc.n = static_cast<std::uint32_t>(N);
    pc.cin = static_cast<std::uint32_t>(C_in);
    pc.t = static_cast<std::uint32_t>(T);
    pc.h = static_cast<std::uint32_t>(H);
    pc.wd = static_cast<std::uint32_t>(W);
    pc.cout = static_cast<std::uint32_t>(C_out);
    pc.to = static_cast<std::uint32_t>(To);
    pc.ho = static_cast<std::uint32_t>(Ho);
    pc.wo = static_cast<std::uint32_t>(Wo);
    pc.kt = static_cast<std::uint32_t>(kT);
    pc.kh = static_cast<std::uint32_t>(kH);
    pc.kw = static_cast<std::uint32_t>(kW);
    pc.st = stride_t; pc.sh = stride_h; pc.sw = stride_w;
    pc.pt = pad_t; pc.ph = pad_h; pc.pw = pad_w;
    pc.dt = dil_t; pc.dh = dil_h; pc.dw = dil_w;
    pc.groups = static_cast<std::uint32_t>(groups);
    conv_direct(d, 0, X.dtype, pc);
}

void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int H, int W,
                              int C_out, int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w, int dil_h, int dil_w, int groups,
                              Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    check_same_dtype(op, X, Wt, bias);
    need(op, N >= 0 && C_in > 0 && H > 0 && W > 0 && C_out > 0, "bad dimension");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >= 1 and divide both C_in and C_out");
    need(op, kH >= 1 && kW >= 1 && stride_h >= 1 && stride_w >= 1 && dil_h >= 1 && dil_w >= 1 && pad_h >= 0 &&
                 pad_w >= 0 && output_padding_h >= 0 && output_padding_w >= 0,
         "kernel, stride and dilation must be >= 1, padding >= 0");
    need(op, output_padding_h < std::max(stride_h, dil_h) && output_padding_w < std::max(stride_w, dil_w),
         "output_padding must be smaller than stride or dilation");
    const int Ho = (H - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + output_padding_h + 1;
    const int Wo = (W - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + output_padding_w + 1;
    need(op, Ho > 0 && Wo > 0, "non-positive output spatial size");
    const int Cgo = C_out / groups;
    need(op, Wt.rows == C_in && Wt.cols == Cgo * kH * kW, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    need(op, X.rows == N && X.cols == C_in * H * W, "X shape must be (N, C_in*H*W)");
    need(op, !bias || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * Ho * Wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(out_cols), X.dtype);
    if (N == 0) return;
    DirectPush pc{};
    pc.x = addr(X.data); pc.w = addr(Wt.data); pc.y = addr(Y.data); pc.bias = bias ? addr(bias->data) : 0;
    pc.n = static_cast<std::uint32_t>(N);
    pc.cin = static_cast<std::uint32_t>(C_in);
    pc.t = 1;
    pc.h = static_cast<std::uint32_t>(H);
    pc.wd = static_cast<std::uint32_t>(W);
    pc.cout = static_cast<std::uint32_t>(C_out);
    pc.to = 1;
    pc.ho = static_cast<std::uint32_t>(Ho);
    pc.wo = static_cast<std::uint32_t>(Wo);
    pc.kt = 1;
    pc.kh = static_cast<std::uint32_t>(kH);
    pc.kw = static_cast<std::uint32_t>(kW);
    pc.st = 1; pc.sh = stride_h; pc.sw = stride_w;
    pc.ph = pad_h; pc.pw = pad_w;
    pc.dt = 1; pc.dh = dil_h; pc.dw = dil_w;
    pc.groups = static_cast<std::uint32_t>(groups);
    conv_direct(device_of(X), 1, X.dtype, pc);
}

// dX = the transposed convolution of dY with the same weights: the direct
// kernel's gather form with dY as its input (C_out channels), Wt read as
// (C_out, Cg_in, kH, kW), which is the transposed layout's (C_in', Cg_out')
// for C_in' = C_out, and the output plane given (H, W) rather than derived.
void conv2d_backward_input(const Tensor& Wt, const Tensor& dY, int N, int C_in, int H, int W, int C_out, int kH,
                           int kW, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,
                           Tensor& dX) {
    const char* op = "conv2d_backward_input";
    dt_code(dY.dtype, op);
    need(op, Wt.dtype == dY.dtype, "dY dtype must match Wt");
    Conv2dArgs a;
    a.n = N; a.cin = C_in; a.h = H; a.wd = W; a.cout = C_out; a.kh = kH; a.kw = kW;
    a.sh = stride_h; a.sw = stride_w; a.ph = pad_h; a.pw = pad_w; a.dh = dil_h; a.dw = dil_w; a.groups = groups;
    a.op = op;
    need(op, N >= 0 && C_in > 0 && H > 0 && W > 0 && C_out > 0, "bad dimension");
    const Geo g = geometry(a);
    need(op, Wt.size() == static_cast<long long>(C_out) * g.k, "Wt must have C_out*(C_in/groups)*kH*kW elements");
    need(op, dY.size() >= static_cast<long long>(N) * C_out * g.ho * g.wo, "dY is smaller than N*C_out*H_out*W_out");
    const long long in_cols = static_cast<long long>(C_in) * H * W;
    need(op, in_cols <= 0x7fffffffLL, "dX too large");
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != dY.dtype) dX.resize(N, static_cast<int>(in_cols), dY.dtype);
    if (N == 0) return;
    DirectPush pc{};
    pc.x = addr(dY.data); pc.w = addr(Wt.data); pc.y = addr(dX.data); pc.bias = 0;
    pc.n = static_cast<std::uint32_t>(N);
    pc.cin = static_cast<std::uint32_t>(C_out);
    pc.t = 1;
    pc.h = static_cast<std::uint32_t>(g.ho);
    pc.wd = static_cast<std::uint32_t>(g.wo);
    pc.cout = static_cast<std::uint32_t>(C_in);
    pc.to = 1;
    pc.ho = static_cast<std::uint32_t>(H);
    pc.wo = static_cast<std::uint32_t>(W);
    pc.kt = 1;
    pc.kh = static_cast<std::uint32_t>(kH);
    pc.kw = static_cast<std::uint32_t>(kW);
    pc.st = 1; pc.sh = stride_h; pc.sw = stride_w;
    pc.ph = pad_h; pc.pw = pad_w;
    pc.dt = 1; pc.dh = dil_h; pc.dw = dil_w;
    pc.groups = static_cast<std::uint32_t>(groups);
    conv_direct(device_of(dY), 1, dY.dtype, pc);
}

// dB[c] += sum over (n, hw) of dY[n, c, hw]; dB kept when it has C_out
// elements of dY's dtype, otherwise resized to (C_out, 1) and zeroed.
void conv2d_backward_bias(const Tensor& dY, int N, int C_out, int H_out, int W_out, Tensor& dB) {
    const char* op = "conv2d_backward_bias";
    dt_code(dY.dtype, op);
    need(op, N >= 0 && C_out > 0 && H_out >= 0 && W_out >= 0, "bad dimension");
    need(op, dY.size() >= static_cast<long long>(N) * C_out * H_out * W_out, "dY is smaller than N*C_out*H_out*W_out");
    if (dB.size() != C_out || dB.dtype != dY.dtype) {
        dB.resize(C_out, 1, dY.dtype);
        dB.zero();
    }
    if (N == 0 || H_out * W_out == 0) return;
    DeviceCtx& d = device_of(dY);
    GnPush pc{};
    pc.x = addr(dY.data);
    pc.y = addr(dB.data);
    pc.hw = static_cast<std::uint32_t>(H_out * W_out);
    pc.c = static_cast<std::uint32_t>(C_out);
    pc.total = static_cast<std::uint32_t>(N);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::gnorm_f32, dY.dtype, op), {GN_BIAS_GRAD});
    launch(d, k, pc, std::min<std::uint32_t>(pc.c, 65535), (pc.c + 65534) / 65535);
}

void conv_transpose2d_backward_bias(const Tensor& dY, int N, int C_out, int H_out, int W_out, Tensor& dB) {
    conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

void fill_vulkan_vtable_conv(::brotensor::detail::OpsVTable& v) {
    v.conv2d_forward = &conv2d_forward;
    v.conv2d_backward_input = &conv2d_backward_input;
    v.conv2d_backward_bias = &conv2d_backward_bias;
    v.conv3d_forward = &conv3d_forward;
    v.conv_transpose2d_forward = &conv_transpose2d_forward;
    v.conv_transpose2d_backward_bias = &conv_transpose2d_backward_bias;
}

}  // namespace brotensor::detail::vulkan
