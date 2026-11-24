// Metal Wave 8 dense & projection-fused attention forward ops via single-source
// brass MIR kernels (MirKind::{AttnSoftmax, AttnAux}) + Metal simdgroup GEMM:
//   attention_forward, mha_forward, self_attention_forward(_train),
//   cross_attention_forward(_train, _with_attn),
//   self_attention_bias_forward, self_attention_bias_int8w_fp16,
//   self_attention_decomposed_rel_pos(_windowed)_forward,
//   rel_pos_bias_xl_forward.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/op_table.h>
#include <brotensor/runtime.h>

#import "fp16_matmul.h"
#import "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::metal {

#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::pool_require;

namespace {

constexpr std::size_t kScoreBudget = std::size_t(64) << 20;

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

bool present(const Tensor* t) { return t && t->data && t->size() > 0; }

void require_fp(const char* op, Dtype dt) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

std::size_t esize(Dtype dt) { return dt == Dtype::FP32 ? 4u : 2u; }
std::int32_t i32(long long v) { return static_cast<std::int32_t>(v); }

void ensure(Tensor& t, int rows, int cols, Dtype dt, Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

void* byte_off_mut(void* p, std::size_t bytes) {
    return static_cast<char*>(p) + bytes;
}

void dispatch_attn_aux(const char* op, std::uint32_t mode, const void* x, const Tensor& y,
                       const void* mask, std::uint32_t rows, std::uint32_t n) {
    const MirKernel mk = mir_kernel(
        {MirKind::AttnAux, mode, y.dtype, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(mask != nullptr ? 1u : 0u)});
    const std::uint64_t total = mode == AUX_ROW_GATE ? std::uint64_t(rows) * n : n;
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
    const void* ptrs[] = {x, y.data, mask};
    const std::int32_t ints[] = {static_cast<std::int32_t>(rows), static_cast<std::int32_t>(n)};
    mir_dispatch_args(mk, gx, op, ptrs, ints, {});
}

// Y(M, N) = X(M, K) @ W(N, K)^T + b; out_dt may be FP32 or X.dtype.
void project_mm(const char* op, const Tensor& X, const Tensor& W, const Tensor* b,
                Tensor& Y, Dtype out_dt) {
    metal_impl::AbtMixed g;
    g.A = buffer_for(X); g.ofs_A = buffer_offset_for(X); g.lda = static_cast<std::uint64_t>(X.cols);
    g.B = buffer_for(W); g.ofs_B = buffer_offset_for(W); g.ldb = static_cast<std::uint64_t>(W.cols);
    g.C = buffer_for(Y); g.ofs_C = buffer_offset_for(Y); g.ldc = static_cast<std::uint64_t>(W.rows);
    if (present(b)) {
        need(op, b->dtype == X.dtype && b->size() == W.rows,
             "a projection bias must have W.rows elements of X's dtype");
        g.bias = buffer_for(*b);
        g.ofs_bias = buffer_offset_for(*b);
    }
    g.M = X.rows; g.N = W.rows; g.K = X.cols;
    g.in = metal_impl::abt_type(X.dtype);
    g.out = metal_impl::abt_type(out_dt);
    metal_impl::launch_matmul_abt_mixed(g);
}

struct ProjSpec {
    const char* op;
    const Tensor* X;
    const Tensor* C;
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const Tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    const float* kmask = nullptr;
    const float* qmask = nullptr;
    const void* bias = nullptr;
    long long bias_hs = -1;
    int H = 1;
    float scale = NAN;
    Tensor *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;
    Tensor *Attnh = nullptr;
    Tensor *Yc = nullptr;
    Tensor *AttnAvg = nullptr;
    Tensor* O = nullptr;
};

}  // namespace

void dense_attention_metal(const DenseAttnProblem& p, const DenseAttnExtras& x) {
    if (p.lq <= 0) return;
    require_fp(p.op, p.dt);
    const int group = p.hq / p.hkv;
    const Dtype dt = p.dt;
    const bool f32 = (dt == Dtype::FP32);
    if (x.probs && !f32) fail(p.op, "probabilities need FP32");
    const std::size_t es = esize(dt);
    const float scale = (x.auto_scale || std::isnan(x.scale))
        ? static_cast<float>(1.0 / std::sqrt(double(p.hd)))
        : x.scale;

    NSUInteger oQ = 0, oK = 0, oV = 0, oO = 0, oProbs = 0;
    id<MTLBuffer> bQ = pool_require(p.q, oQ, p.op, "Q");
    id<MTLBuffer> bK = pool_require(p.k, oK, p.op, "K");
    id<MTLBuffer> bV = pool_require(p.v, oV, p.op, "V");
    id<MTLBuffer> bO = pool_require(p.o, oO, p.op, "O");
    id<MTLBuffer> bProbs = x.probs ? pool_require(x.probs, oProbs, p.op, "probs") : nil;
    if (p.mask)  { NSUInteger o = 0; pool_require(p.mask, o, p.op, "mask"); }
    if (x.qmask) { NSUInteger o = 0; pool_require(x.qmask, o, p.op, "qmask"); }

    const std::size_t per_head = static_cast<std::size_t>(p.lq) * static_cast<std::size_t>(p.lk);
    const int hpp = static_cast<int>(std::max<std::size_t>(
        1, std::min<std::size_t>(group == 1 ? p.hq : group,
                                 kScoreBudget / std::max<std::size_t>(per_head, 1))));
    int rows = p.lq;
    if (per_head > kScoreBudget) {
        rows = static_cast<int>(std::max<std::size_t>(1, kScoreBudget / static_cast<std::size_t>(p.lk)));
        if (rows >= 64) rows = rows / 64 * 64;
    }

    Tensor S, P;
    if (!x.probs) {
        const long long n = static_cast<long long>(rows == p.lq ? hpp : 1) * rows * p.lk;
        S = Tensor::empty_on(p.dev, static_cast<int>(n), 1, Dtype::FP32);
        if (!f32) P = Tensor::empty_on(p.dev, static_cast<int>(n), 1, dt);
    }

    const bool rel = (x.rel_h != nullptr);
    if (rel && (p.mask || x.bias || x.grid_h <= 0 || x.grid_w <= 0 ||
                x.grid_h > 0xffff || x.grid_w > 0xffff || x.grid_h * x.grid_w != p.lq)) {
        fail(p.op, "bad decomposed rel-pos problem");
    }
    const std::uint32_t bias_mode = rel ? 2u : (x.bias ? 1u : 0u);
    const std::uint8_t sm_flags = static_cast<std::uint8_t>(
        (p.mask ? 1u : 0u) | (x.qmask ? 2u : 0u) | (x.mask_ge ? 4u : 0u));
    const MirKernel ks = mir_kernel(
        {MirKind::AttnSoftmax, bias_mode, dt, Dtype::FP32, 1, Dtype::FP32, sm_flags});
    const void* sm_bias = rel ? x.rel_h : x.bias;
    const void* sm_mask = rel ? x.rel_w : p.mask;
    const std::uint32_t sm_bhs = rel
        ? static_cast<std::uint32_t>(x.grid_w) | (static_cast<std::uint32_t>(x.grid_h) << 16)
        : static_cast<std::uint32_t>(x.bias_hs < 0 ? static_cast<long long>(p.lq) * p.lk : x.bias_hs);

    const metal_impl::AbtType in_ty = metal_impl::abt_type(dt);
    const metal_impl::AbtType p_ty = (x.probs || f32) ? metal_impl::kAbtF32 : in_ty;

    auto pass = [&](int h0, int nh, int r0, int nr) {
        const std::size_t kv_off = static_cast<std::size_t>(h0 / group) * p.hd * es;
        const long long nrk = static_cast<long long>(nr) * p.lk;
        const std::size_t probs_off = (static_cast<std::size_t>(h0) * p.lq + r0) * p.lk * sizeof(float);
        void* s_ptr = x.probs ? byte_off_mut(x.probs, probs_off) : S.data;
        void* p_ptr = (x.probs || f32) ? s_ptr : P.data;
        id<MTLBuffer> bS = x.probs ? bProbs : buffer_for(S);
        const NSUInteger oS = x.probs ? (oProbs + static_cast<NSUInteger>(probs_off)) : buffer_offset_for(S);
        id<MTLBuffer> bP = (x.probs || f32) ? bS : buffer_for(P);
        const NSUInteger oP = (x.probs || f32) ? oS : buffer_offset_for(P);

        metal_impl::AbtMixed g;
        g.A = bQ;
        g.ofs_A = oQ + static_cast<NSUInteger>((std::size_t(r0) * p.ldq + std::size_t(h0) * p.hd) * es);
        g.lda = static_cast<std::uint64_t>(p.ldq);
        g.B = bK;
        g.ofs_B = oK + static_cast<NSUInteger>(kv_off);
        g.ldb = static_cast<std::uint64_t>(p.ldk);
        g.C = bS;
        g.ofs_C = oS;
        g.ldc = static_cast<std::uint64_t>(p.lk);
        g.M = nr; g.N = p.lk; g.K = p.hd;
        g.in = in_ty; g.out = metal_impl::kAbtF32;
        g.batch = nh;
        g.strideA = static_cast<std::uint64_t>(p.hd);
        g.strideB = (group == 1) ? static_cast<std::uint64_t>(p.hd) : 0u;
        g.strideC = static_cast<std::uint64_t>(nrk);
        metal_impl::launch_matmul_abt_mixed(g);

        const std::uint32_t sm_rows = static_cast<std::uint32_t>(nh * nr);
        const void* ptrs[] = {s_ptr, p_ptr, sm_bias, sm_mask, x.qmask};
        const std::int32_t ints[] = {
            i32(sm_rows), i32(p.lk), i32(nr), i32(h0),
            i32(p.lk), i32(p.lk), i32(r0), static_cast<std::int32_t>(sm_bhs),
        };
        const float flts[] = {scale};
        mir_dispatch_args(ks, std::max<std::uint32_t>(1u, std::min<std::uint32_t>(65535u, sm_rows)),
                          p.op, ptrs, ints, flts);

        metal_impl::AbtMixed o;
        o.A = bP; o.ofs_A = oP; o.lda = static_cast<std::uint64_t>(p.lk);
        o.B = bV; o.ofs_B = oV + static_cast<NSUInteger>(kv_off);
        o.ldb = static_cast<std::uint64_t>(p.ldk); o.transB = true;
        o.C = bO;
        o.ofs_C = oO + static_cast<NSUInteger>((std::size_t(r0) * p.ldo + std::size_t(h0) * p.hd) * es);
        o.ldc = static_cast<std::uint64_t>(p.ldo);
        o.M = nr; o.N = p.hd; o.K = p.lk;
        o.in = p_ty;
        o.in_B = (p_ty == in_ty) ? -1 : static_cast<int>(in_ty);
        o.out = in_ty;
        o.batch = nh;
        o.strideA = static_cast<std::uint64_t>(nrk);
        o.strideB = (group == 1) ? static_cast<std::uint64_t>(p.hd) : 0u;
        o.strideC = static_cast<std::uint64_t>(p.hd);
        metal_impl::launch_matmul_abt_mixed(o);
    };

    for (int h = 0; h < p.hq;) {
        const int nh = (rows < p.lq) ? 1 : std::min(hpp, group == 1 ? p.hq - h : group - h % group);
        for (int r0 = 0; r0 < p.lq; r0 += rows) pass(h, nh, r0, std::min(rows, p.lq - r0));
        h += nh;
    }
}

namespace {

void proj_attention(const ProjSpec& s) {
    const char* op = s.op;
    const Tensor &X = *s.X, &C = *s.C;
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    for (const Tensor* t : {s.C, s.Wq, s.Wk, s.Wv, s.Wo}) {
        need(op, t->dtype == dt, "X, Ctx and the weights must share one dtype");
    }
    const int Lq = X.rows, Lk = C.rows, D = s.Wq->rows, H = s.H;
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    need(op, s.Wq->cols == X.cols && s.Wk->rows == D && s.Wv->rows == D &&
             s.Wk->cols == C.cols && s.Wv->cols == C.cols &&
             s.Wo->rows == D && s.Wo->cols == D,
         "weight shapes do not match X / Ctx");
    if (s.kmask) { NSUInteger o = 0; pool_require(s.kmask, o, op, "mask"); }
    if (s.qmask) { NSUInteger o = 0; pool_require(s.qmask, o, op, "mask"); }

    const int hd = D / H;
    const Device dev = X.device;
    ensure(*s.O, Lq, D, dt, dev);
    if (s.Qh) ensure(*s.Qh, H * Lq, hd, Dtype::FP32, dev);
    if (s.Kh) ensure(*s.Kh, H * Lk, hd, Dtype::FP32, dev);
    if (s.Vh) ensure(*s.Vh, H * Lk, hd, Dtype::FP32, dev);
    if (s.Attnh) ensure(*s.Attnh, H * Lq, Lk, Dtype::FP32, dev);
    if (s.Yc) ensure(*s.Yc, Lq, D, Dtype::FP32, dev);
    if (s.AttnAvg) ensure(*s.AttnAvg, Lq, Lk, dt, dev);
    if (Lq == 0 || D == 0) return;
    if (Lk == 0) {
        for (Tensor* t : {s.O, s.Yc, s.AttnAvg}) {
            if (t && t->size() > 0) t->zero();
        }
        return;
    }

    Tensor Q = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    Tensor K = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    Tensor V = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    project_mm(op, X, *s.Wq, s.bq, Q, Dtype::FP32);
    project_mm(op, C, *s.Wk, s.bk, K, Dtype::FP32);
    project_mm(op, C, *s.Wv, s.bv, V, Dtype::FP32);
    auto extract_heads = [&](const Tensor& src, Tensor* dst, int L) {
        if (!dst) return;
        for (int h = 0; h < H; ++h) copy_d2d_strided(src, h * hd, D, *dst, h * L * hd, hd, hd, L);
    };
    extract_heads(Q, s.Qh, Lq);
    extract_heads(K, s.Kh, Lk);
    extract_heads(V, s.Vh, Lk);

    Tensor yc_tmp, probs_tmp;
    Tensor& Yc = s.Yc ? *s.Yc : (yc_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    Tensor* probs = s.Attnh;
    if (!probs && s.AttnAvg) {
        probs_tmp = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
        probs = &probs_tmp;
    }

    DenseAttnProblem p;
    p.op = op; p.dev = dev; p.dt = Dtype::FP32;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = Yc.data;
    p.mask = s.kmask;
    p.lq = Lq; p.lk = Lk;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = hd; p.hq = p.hkv = H;
    DenseAttnExtras x;
    x.scale = s.scale;
    x.auto_scale = std::isnan(s.scale);
    x.bias = s.bias;
    x.bias_hs = s.bias_hs;
    x.qmask = s.qmask;
    x.mask_ge = true;
    x.probs = probs ? probs->data : nullptr;
    dense_attention_metal(p, x);

    if (s.AttnAvg) {
        dispatch_attn_aux(op, AUX_HEAD_MEAN, probs->data, *s.AttnAvg, nullptr,
                          static_cast<std::uint32_t>(H),
                          static_cast<std::uint32_t>(Lq) * static_cast<std::uint32_t>(Lk));
    }

    Tensor o32_tmp, bo32;
    Tensor& O32 = (dt == Dtype::FP32) ? *s.O : (o32_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    metal_impl::AbtMixed g;
    g.A = buffer_for(Yc);     g.ofs_A = buffer_offset_for(Yc);     g.lda = static_cast<std::uint64_t>(D);
    g.B = buffer_for(*s.Wo);  g.ofs_B = buffer_offset_for(*s.Wo);  g.ldb = static_cast<std::uint64_t>(D);
    g.C = buffer_for(O32);    g.ofs_C = buffer_offset_for(O32);    g.ldc = static_cast<std::uint64_t>(D);
    g.M = Lq; g.N = D; g.K = D;
    g.in = metal_impl::kAbtF32;
    g.in_B = (dt == Dtype::FP32) ? -1 : static_cast<int>(metal_impl::abt_type(dt));
    g.out = metal_impl::kAbtF32;
    if (present(s.bo)) {
        need(op, s.bo->dtype == dt && s.bo->size() == D, "bo must have D elements of X's dtype");
        if (dt == Dtype::FP32) {
            g.bias = buffer_for(*s.bo);
            g.ofs_bias = buffer_offset_for(*s.bo);
        } else {
            bo32 = Tensor::empty_on(dev, D, 1, Dtype::FP32);
            cast(*s.bo, bo32, Dtype::FP32);
            g.bias = buffer_for(bo32);
            g.ofs_bias = buffer_offset_for(bo32);
        }
    }
    metal_impl::launch_matmul_abt_mixed(g);

    if (dt != Dtype::FP32 || s.qmask) {
        dispatch_attn_aux(op, AUX_ROW_GATE, O32.data, *s.O, s.qmask,
                          static_cast<std::uint32_t>(Lq), static_cast<std::uint32_t>(D));
    }
}

void sam_attention(const char* op, const Tensor& X,
                   const Tensor& Wq, const Tensor* bq,
                   const Tensor& Wk, const Tensor* bk,
                   const Tensor& Wv, const Tensor* bv,
                   const Tensor& Wo, const Tensor* bo,
                   const Tensor& rel_h, const Tensor& rel_w,
                   int num_heads, int grid_h, int grid_w, int window,
                   float scale, Tensor& O) {
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    const int L = X.rows, D = X.cols, H = num_heads;
    need(op, Wq.dtype == dt && Wk.dtype == dt && Wv.dtype == dt && Wo.dtype == dt &&
             rel_h.dtype == dt && rel_w.dtype == dt,
         "Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    need(op, Wq.rows == D && Wq.cols == D && Wk.rows == D && Wk.cols == D &&
             Wv.rows == D && Wv.cols == D && Wo.rows == D && Wo.cols == D,
         "Wq/Wk/Wv/Wo must be (D, D)");
    need(op, grid_h > 0 && grid_w > 0 && grid_h * grid_w == L,
         "grid_h*grid_w must equal X.rows");
    const int dh = D / H;
    const int gh = window > 0 ? window : grid_h;
    const int gw = window > 0 ? window : grid_w;
    need(op, rel_h.rows == 2 * gh - 1 && rel_h.cols == dh,
         "rel_pos_h must be (2*grid_h-1, head_dim)");
    need(op, rel_w.rows == 2 * gw - 1 && rel_w.cols == dh,
         "rel_pos_w must be (2*grid_w-1, head_dim)");
    for (const Tensor* b : {bq, bk, bv, bo}) {
        if (present(b)) need(op, b->dtype == dt && b->size() == D, "bias must have D elements of X's dtype");
    }
    const Device dev = X.device;
    ensure(O, L, D, dt, dev);
    if (L == 0 || D == 0) return;

    int U = 1;
    const int Lu = gh * gw;
    Tensor Xw, back;
    if (window > 0) {
        const int nwh = (grid_h + window - 1) / window;
        const int nww = (grid_w + window - 1) / window;
        U = nwh * nww;
        std::vector<std::int32_t> fwd(static_cast<std::size_t>(U) * Lu);
        std::vector<std::int32_t> inv(static_cast<std::size_t>(L));
        for (int a = 0; a < nwh; ++a) {
            for (int b = 0; b < nww; ++b) {
                for (int lh = 0; lh < window; ++lh) {
                    for (int lw = 0; lw < window; ++lw) {
                        const int h = a * window + lh, w = b * window + lw;
                        const int r = (a * nww + b) * Lu + lh * window + lw;
                        const bool in = (h < grid_h && w < grid_w);
                        fwd[static_cast<std::size_t>(r)] = in ? h * grid_w + w : L;
                        if (in) inv[static_cast<std::size_t>(h * grid_w + w)] = r;
                    }
                }
            }
        }
        Tensor Xp = Tensor::zeros_on(dev, L + 1, D, dt);
        copy_d2d(X, 0, Xp, 0, L * D);
        const Tensor fi = Tensor::from_raw_bytes_on(dev, fwd.data(), U * Lu, 1, Dtype::INT32,
                                                    fwd.size() * sizeof(std::int32_t));
        back = Tensor::from_raw_bytes_on(dev, inv.data(), L, 1, Dtype::INT32,
                                         inv.size() * sizeof(std::int32_t));
        Xw = Tensor::empty_on(dev, U * Lu, D, dt);
        gather_rows(Xp, fi, Xw);
    }
    const Tensor& Xu = window > 0 ? Xw : X;
    const int R = U * Lu;
    const std::size_t es = esize(dt);

    Tensor Q = Tensor::empty_on(dev, R, D, dt);
    Tensor K = Tensor::empty_on(dev, R, D, dt);
    Tensor V = Tensor::empty_on(dev, R, D, dt);
    project_mm(op, Xu, Wq, bq, Q, dt);
    project_mm(op, Xu, Wk, bk, K, dt);
    project_mm(op, Xu, Wv, bv, V, dt);

    const int ph = 2 * gh - 1, pw = 2 * gw - 1;
    Tensor Bh = Tensor::empty_on(dev, U * H * Lu, ph, Dtype::FP32);
    Tensor Bw = Tensor::empty_on(dev, U * H * Lu, pw, Dtype::FP32);
    Tensor Yc = Tensor::empty_on(dev, R, D, dt);
    const metal_impl::AbtType ty = metal_impl::abt_type(dt);

    for (int u = 0; u < U; ++u) {
        const std::size_t off = std::size_t(u) * Lu * D * es;
        for (int axis = 0; axis < 2; ++axis) {
            const int P = axis == 0 ? ph : pw;
            const Tensor& rel_t = axis == 0 ? rel_h : rel_w;
            Tensor& B_t = axis == 0 ? Bh : Bw;
            metal_impl::AbtMixed g;
            g.A = buffer_for(Q);     g.ofs_A = buffer_offset_for(Q) + static_cast<NSUInteger>(off);
            g.lda = static_cast<std::uint64_t>(D);
            g.B = buffer_for(rel_t); g.ofs_B = buffer_offset_for(rel_t);
            g.ldb = static_cast<std::uint64_t>(dh);
            g.C = buffer_for(B_t);
            g.ofs_C = buffer_offset_for(B_t) +
                      static_cast<NSUInteger>(std::size_t(u) * H * Lu * P * sizeof(float));
            g.ldc = static_cast<std::uint64_t>(P);
            g.M = Lu; g.N = P; g.K = dh;
            g.in = ty; g.out = metal_impl::kAbtF32;
            g.batch = H; g.strideA = static_cast<std::uint64_t>(dh);
            g.strideB = 0; g.strideC = static_cast<std::uint64_t>(Lu) * P;
            metal_impl::launch_matmul_abt_mixed(g);
        }
        DenseAttnProblem p;
        p.op = op; p.dev = dev; p.dt = dt;
        p.q = byte_off_mut(Q.data, off);
        p.k = byte_off_mut(K.data, off);
        p.v = byte_off_mut(V.data, off);
        p.o = byte_off_mut(Yc.data, off);
        p.lq = p.lk = Lu;
        p.ldq = p.ldk = p.ldo = D;
        p.hd = dh; p.hq = p.hkv = H;
        DenseAttnExtras x;
        x.scale = scale;
        x.auto_scale = std::isnan(scale);
        x.rel_h = byte_off_mut(Bh.data, std::size_t(u) * H * Lu * ph * sizeof(float));
        x.rel_w = byte_off_mut(Bw.data, std::size_t(u) * H * Lu * pw * sizeof(float));
        x.grid_h = gh; x.grid_w = gw;
        dense_attention_metal(p, x);
    }

    if (window > 0) {
        Tensor Ow = Tensor::empty_on(dev, R, D, dt);
        project_mm(op, Yc, Wo, bo, Ow, dt);
        gather_rows(Ow, back, O);
    } else {
        project_mm(op, Yc, Wo, bo, O, dt);
    }
}

}  // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo,
                                 const Tensor* bq, const Tensor* bk,
                                 const Tensor* bv, const Tensor* bo,
                                 const float* d_mask, const Tensor* attn_bias,
                                 int num_heads, float scale, Tensor& O) {
    constexpr const char* op = "self_attention_bias_forward";
    ProjSpec s{op, &X, &X, &Wq, &Wk, &Wv, &Wo, bq, bk, bv, bo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.scale = scale;
    s.O = &O;
    if (present(attn_bias)) {
        const long long L = X.rows;
        need(op, attn_bias->dtype == Dtype::FP32, "attn_bias must be FP32");
        need(op, num_heads > 0 && attn_bias->size() == num_heads * L * L,
             "attn_bias must be (num_heads*L, L)");
        s.bias = attn_bias->data;
    }
    proj_attention(s);
}

void self_attention_bias_int8w_fp16(const Tensor& X,
                                    const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask, const Tensor* attn_bias,
                                    int num_heads, float scale, Tensor& O) {
    constexpr const char* op = "self_attention_bias_int8w_fp16";
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    need(op, Wq_int8.dtype == Dtype::INT8 && Wk_int8.dtype == Dtype::INT8 &&
             Wv_int8.dtype == Dtype::INT8 && Wo_int8.dtype == Dtype::INT8,
         "Wq/Wk/Wv/Wo must be INT8");
    const int L = X.rows, D = X.cols, H = num_heads;
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    for (const Tensor* w : {&Wq_int8, &Wk_int8, &Wv_int8, &Wo_int8}) {
        need(op, w->rows == D && w->cols == D, "Wq/Wk/Wv/Wo must be (D, D)");
    }
    for (const Tensor* sc : {&sq, &sk, &sv, &so}) {
        need(op, sc->dtype == Dtype::FP32 && sc->size() == D,
             "each scale tensor must have D FP32 entries");
    }
    const void* bias_ptr = nullptr;
    if (present(attn_bias)) {
        need(op, attn_bias->dtype == Dtype::FP32, "attn_bias must be FP32");
        need(op, attn_bias->size() == static_cast<long long>(H) * L * L,
             "attn_bias must be (num_heads*L, L)");
        bias_ptr = attn_bias->data;
    }
    ensure(O, L, D, Dtype::FP16, X.device);
    if (L == 0 || D == 0) return;

    Tensor Q = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    Tensor K = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    Tensor V = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    Tensor Yc = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    linear_forward_batched_int8w_fp16(Wq_int8, sq, nullptr, X, Q);
    linear_forward_batched_int8w_fp16(Wk_int8, sk, nullptr, X, K);
    linear_forward_batched_int8w_fp16(Wv_int8, sv, nullptr, X, V);

    DenseAttnProblem p;
    p.op = op; p.dev = X.device; p.dt = Dtype::FP16;
    p.q = Q.data; p.k = K.data; p.v = V.data; p.o = Yc.data;
    p.mask = d_mask;
    p.lq = p.lk = L;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = D / H; p.hq = p.hkv = H;
    DenseAttnExtras x;
    x.scale = scale;
    x.auto_scale = std::isnan(scale);
    x.bias = bias_ptr;
    x.qmask = d_mask;
    x.mask_ge = true;
    dense_attention_metal(p, x);
    linear_forward_batched_int8w_fp16(Wo_int8, so, nullptr, Yc, O);
}

void cross_attention_forward(const Tensor& X, const Tensor& Ctx,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads, Tensor& O) {
    ProjSpec s{"cross_attention_forward", &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = (X.rows == Ctx.rows) ? d_mask : nullptr;
    s.H = num_heads;
    s.O = &O;
    proj_attention(s);
}

void cross_attention_forward_with_attn(const Tensor& X, const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask, const Tensor* attn_logit_bias,
                                       int num_heads, Tensor& O, Tensor& AttnAvg) {
    constexpr const char* op = "cross_attention_forward_with_attn";
    ProjSpec s{op, &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.H = num_heads;
    s.O = &O;
    s.AttnAvg = &AttnAvg;
    if (present(attn_logit_bias)) {
        need(op, attn_logit_bias->dtype == Dtype::FP32, "attn_logit_bias must be FP32");
        need(op, attn_logit_bias->size() == static_cast<long long>(X.rows) * Ctx.rows,
             "attn_logit_bias must be (Lq, Lk)");
        s.bias = attn_logit_bias->data;
        s.bias_hs = 0;
    }
    proj_attention(s);
}

void mha_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                 const Tensor& Wv, const Tensor& Wo,
                 const Tensor* bq, const Tensor* bk,
                 const Tensor* bv, const Tensor* bo,
                 const float* d_mask, int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"mha_forward", &X, &X, &Wq, &Wk, &Wv, &Wo, bq, bk, bv, bo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

void attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                       const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                       Tensor& Q, Tensor& K, Tensor& V,
                       Tensor& Attn, Tensor& Y_pre_Wo, Tensor& O) {
    ProjSpec s{"attention_forward", &X, &X, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.Qh = &Q; s.Kh = &K; s.Vh = &V; s.Attnh = &Attn; s.Yc = &Y_pre_Wo;
    s.O = &O;
    proj_attention(s);
}

void cross_attention_forward_train(const Tensor& X, const Tensor& Ctx,
                                   const Tensor& Wq, const Tensor& Wk,
                                   const Tensor& Wv, const Tensor& Wo,
                                   const float* d_mask, int num_heads,
                                   Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                   Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"cross_attention_forward_train", &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = (X.rows == Ctx.rows) ? d_mask : nullptr;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

void self_attention_forward_train(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask, int num_heads,
                                  Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo, nullptr, nullptr, nullptr, nullptr, d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask, int num_heads, Tensor& O) {
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        flash_attention_qkvo_forward(X, nullptr, Wq, nullptr, Wk, nullptr, Wv, nullptr,
                                     Wo, nullptr, d_mask, num_heads, false, O);
        return;
    }
    ProjSpec s{"self_attention_forward", &X, &X, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.O = &O;
    proj_attention(s);
}

void self_attention_decomposed_rel_pos_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale, Tensor& O) {
    sam_attention("self_attention_decomposed_rel_pos_forward",
                  X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
                  num_heads, grid_h, grid_w, 0, scale, O);
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale, Tensor& O) {
    const char* op = "self_attention_decomposed_rel_pos_windowed_forward";
    need(op, window > 0, "window must be >= 1");
    sam_attention(op, X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
                  num_heads, grid_h, grid_w, window, scale, O);
}

void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk,
                             int num_heads, int head_dim, Tensor& Bias) {
    constexpr const char* op = "rel_pos_bias_xl_forward";
    need(op, Qv.dtype == Dtype::FP32 && Pk.dtype == Dtype::FP32, "Qv and Pk must be FP32");
    const int T = Qv.rows, D = Qv.cols;
    need(op, num_heads > 0 && head_dim > 0 && num_heads * head_dim == D,
         "num_heads*head_dim must equal Qv.cols");
    need(op, Pk.cols == D && Pk.rows == 2 * T - 1, "Pk must be (2*Qv.rows - 1, Qv.cols)");
    ensure(Bias, num_heads * T, T, Dtype::FP32, Qv.device);
    if (T == 0) return;
    const int P = 2 * T - 1;
    Tensor R = Tensor::empty_on(Qv.device, num_heads * T, P, Dtype::FP32);
    metal_impl::AbtMixed g;
    g.A = buffer_for(Qv); g.ofs_A = buffer_offset_for(Qv); g.lda = static_cast<std::uint64_t>(D);
    g.B = buffer_for(Pk); g.ofs_B = buffer_offset_for(Pk); g.ldb = static_cast<std::uint64_t>(D);
    g.C = buffer_for(R);  g.ofs_C = buffer_offset_for(R);  g.ldc = static_cast<std::uint64_t>(P);
    g.M = T; g.N = P; g.K = head_dim;
    g.in = g.out = metal_impl::kAbtF32;
    g.batch = num_heads;
    g.strideA = static_cast<std::uint64_t>(head_dim);
    g.strideB = static_cast<std::uint64_t>(head_dim);
    g.strideC = static_cast<std::uint64_t>(T) * P;
    metal_impl::launch_matmul_abt_mixed(g);
    for (int h = 0; h < num_heads; ++h) {
        copy_d2d_strided(R, h * T * P + (T - 1), P - 1, Bias, h * T * T, T, T, T);
    }
}

}  // namespace brotensor::detail::metal
