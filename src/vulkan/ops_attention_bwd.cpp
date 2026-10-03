// Vulkan backwards of the materialised-probability attention ops:
// attention_backward (one head, brogameagent's Attention layer),
// mha_backward (+ optional bias gradients), self_attention_backward and
// cross_attention_backward. Contracts follow the CPU reference
// (src/cpu/ops_impl.cpp, cross_attention.cpp, self_attention.cpp): every
// operand FP32; the forward's caches Qh / Kh / Vh (H * L, hd), Attnh
// (H * lq, lk) and Yconcat (lq, D); softmax scale 1 / sqrt(hd); masks valid
// where >= 0.5, gating query rows (self-attention, and cross-attention when
// lq == lk) and key columns; dW* and db* ACCUMULATE, dX / dCtx are
// overwritten.
//
// Each backward is GEMMs (gemm(), FP32 SIMT) around one kernel,
// attn_bwd.comp's in-place row softmax backward:
//   dOg = dO with gated query rows zeroed (attn_aux.comp's ROW_GATE)
//   dYc = dOg Wo,  dWo += dOg^T Yc,  dbo += colsum dOg
//   per head (batched GEMMs, the head's columns of the (L, D) matrices):
//     dP_h = dYc_h Vh_h^T,  dV_h = P_h^T dYc_h
//     dS_h = softmax_bwd(P_h, dP_h) * scale (masked)
//     dQ_h = dS_h Kh_h,  dK_h = dS_h^T Qh_h
//   dWq += dQ^T X,  dWk += dK^T C,  dWv += dV^T C,  db* += colsum
//   dX = dQ Wq (+ dK Wk + dV Wv for self-attention),  dCtx = dK Wk + dV Wv

#include "detail/gemm.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void column_sum_accumulate(const Tensor& X, Tensor& out);   // ops_norm.cpp

namespace {

struct AuxPush {
    std::uint64_t x, y, mask;
    std::uint32_t rows, n;
};

struct SmBwdPush {
    std::uint64_t p, dp, kmask, qmask;
    std::uint32_t rows, lq, lk;
    float scale;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void ensure(Tensor& t, int rows, int cols, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) t.resize(rows, cols, Dtype::FP32);
}

// One FP32 GEMM: C(m, n) (+)= op(A)(m, k) op(B)(k, n), batched over `batch`
// with element strides sa / sb / sc.
struct G {
    std::uint64_t a, b, c;
    int m, n, k, lda, ldb, ldc;
    bool ta = false, nb = false, acc = false;
    int batch = 1;
    long long sa = 0, sb = 0, sc = 0;
};

void run(DeviceCtx& d, const char* op, const G& x) {
    GemmArgs g;
    g.op = op;
    g.a = x.a; g.b = x.b; g.c = x.c;
    g.m = x.m; g.n = x.n; g.k = x.k;
    g.lda = x.lda; g.ldb = x.ldb; g.ldc = x.ldc;
    g.ta = x.ta; g.nb = x.nb;
    g.epi = x.acc ? EPI_ACCUM : EPI_STORE;
    g.batch = x.batch; g.sa = x.sa; g.sb = x.sb; g.sc = x.sc;
    gemm(d, g);
}

std::uint64_t at(const Tensor& t, long long elems = 0) { return addr(t.data) + std::uint64_t(elems) * 4u; }

struct BwdSpec {
    const char* op;
    const Tensor *dO, *X, *C;                 // C: key / value source (X for self-attention)
    const Tensor *Qh, *Kh, *Vh, *Attnh, *Yc;
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const float* kmask = nullptr;
    const float* qmask = nullptr;
    int H = 1;
    Tensor* dX;
    Tensor* dC = nullptr;                     // null: self-attention, dK / dV fold into dX
    Tensor *dWq, *dWk, *dWv, *dWo;
    Tensor *dbq = nullptr, *dbk = nullptr, *dbv = nullptr, *dbo = nullptr;
};

void check(const BwdSpec& s, int Lq, int Lk, int D, int Dx, int Dc, int hd) {
    const char* op = s.op;
    for (const Tensor* t : {s.dO, s.X, s.C, s.Qh, s.Kh, s.Vh, s.Attnh, s.Yc, s.Wq, s.Wk, s.Wv, s.Wo,
                            static_cast<const Tensor*>(s.dWq), static_cast<const Tensor*>(s.dWk),
                            static_cast<const Tensor*>(s.dWv), static_cast<const Tensor*>(s.dWo)}) {
        if (t->dtype != Dtype::FP32) fail(op, "every operand must be FP32");
    }
    auto shape = [&](const Tensor& t, int r, int c, const char* what) {
        if (t.rows != r || t.cols != c) fail(op, std::string(what) + " has the wrong shape");
    };
    shape(*s.dO, Lq, D, "dO");
    shape(*s.Qh, s.H * Lq, hd, "Qh");
    shape(*s.Kh, s.H * Lk, hd, "Kh");
    shape(*s.Vh, s.H * Lk, hd, "Vh");
    shape(*s.Attnh, s.H * Lq, Lk, "Attnh");
    shape(*s.Yc, Lq, D, "Yconcat");
    shape(*s.Wq, D, Dx, "Wq");
    shape(*s.Wk, D, Dc, "Wk");
    shape(*s.Wv, D, Dc, "Wv");
    shape(*s.Wo, D, D, "Wo");
    shape(*s.dWq, D, Dx, "dWq");
    shape(*s.dWk, D, Dc, "dWk");
    shape(*s.dWv, D, Dc, "dWv");
    shape(*s.dWo, D, D, "dWo");
    for (Tensor* b : {s.dbq, s.dbk, s.dbv, s.dbo}) {
        if (b && (b->dtype != Dtype::FP32 || b->size() != D)) fail(op, "a bias gradient must have D FP32 elements");
    }
}

void attention_bwd(const BwdSpec& s) {
    const char* op = s.op;
    const Tensor &X = *s.X, &C = *s.C;
    const int Lq = X.rows, Lk = C.rows, Dx = X.cols, Dc = C.cols, D = s.Wq->rows, H = s.H;
    const ::brotensor::Device dev = X.device;
    ensure(*s.dX, Lq, Dx, dev);
    if (s.dC) ensure(*s.dC, Lk, Dc, dev);
    if (Lq == 0 || D == 0 || H == 0) return;
    if (H < 0 || D % H != 0) fail(op, "num_heads must divide D");
    const int hd = D / H;
    if (Lk == 0) {   // nothing was attended: no gradient reaches X or Ctx
        s.dX->zero();
        if (s.dC && s.dC->size() > 0) s.dC->zero();
        return;
    }
    check(s, Lq, Lk, D, Dx, Dc, hd);
    DeviceCtx& d = device_of(X);

    // dOg: dO with gated query rows zeroed.
    Tensor dog_tmp;
    std::uint64_t dog = addr(s.dO->data);
    if (s.qmask) {
        dog_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
        const Kernel& k = d.pipelines().get(ShaderId::attn_aux_f32, {std::uint32_t(AUX_ROW_GATE)});
        const AuxPush pc{dog, addr(dog_tmp.data), addr(s.qmask), static_cast<std::uint32_t>(Lq),
                         static_cast<std::uint32_t>(D)};
        launch(d, k, pc, groups_1d(std::uint64_t(Lq) * D, k));
        dog = addr(dog_tmp.data);
    }

    Tensor dYc = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    run(d, op, {dog, addr(s.Wo->data), addr(dYc.data), Lq, D, D, D, D, D, false, true});
    run(d, op, {dog, addr(s.Yc->data), addr(s.dWo->data), D, D, Lq, D, D, D, true, true, true});
    if (s.dbo) column_sum_accumulate(Tensor::view(dev, reinterpret_cast<void*>(dog), Lq, D, Dtype::FP32), *s.dbo);

    const long long pq = static_cast<long long>(Lq) * Lk;
    Tensor dP = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
    Tensor dQ = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    Tensor dK = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    Tensor dV = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    auto per_head = [&](G g, long long sa, long long sb) {
        g.batch = H; g.sa = sa; g.sb = sb;
        return g;
    };
    // dP_h (Lq, Lk) = dYc_h Vh_h^T
    {
        G g = per_head({addr(dYc.data), addr(s.Vh->data), addr(dP.data), Lq, Lk, hd, D, hd, Lk}, hd,
                       static_cast<long long>(Lk) * hd);
        g.sc = pq;
        run(d, op, g);
    }
    // dV_h (Lk, hd) = P_h^T dYc_h, into dV's head columns
    {
        G g = per_head({addr(s.Attnh->data), addr(dYc.data), addr(dV.data), Lk, hd, Lq, Lk, D, D, true, true}, pq,
                       hd);
        g.sc = hd;
        run(d, op, g);
    }
    // dS = softmax backward, in place over dP
    {
        const Kernel& k = d.pipelines().get(ShaderId::attn_bwd);
        const SmBwdPush pc{addr(s.Attnh->data), addr(dP.data), addr(s.kmask), addr(s.qmask),
                           static_cast<std::uint32_t>(H * Lq), static_cast<std::uint32_t>(Lq),
                           static_cast<std::uint32_t>(Lk), 1.0f / std::sqrt(static_cast<float>(hd))};
        launch(d, k, pc, static_cast<std::uint32_t>(std::min(H * Lq, 65535)));
    }
    // dQ_h (Lq, hd) = dS_h Kh_h;  dK_h (Lk, hd) = dS_h^T Qh_h
    {
        G g = per_head({addr(dP.data), addr(s.Kh->data), addr(dQ.data), Lq, hd, Lk, Lk, hd, D, false, true}, pq,
                       static_cast<long long>(Lk) * hd);
        g.sc = hd;
        run(d, op, g);
        G h = per_head({addr(dP.data), addr(s.Qh->data), addr(dK.data), Lk, hd, Lq, Lk, hd, D, true, true}, pq,
                       static_cast<long long>(Lq) * hd);
        h.sc = hd;
        run(d, op, h);
    }
    // Weight and bias gradients (accumulated).
    run(d, op, {addr(dQ.data), addr(X.data), addr(s.dWq->data), D, Dx, Lq, D, Dx, Dx, true, true, true});
    run(d, op, {addr(dK.data), addr(C.data), addr(s.dWk->data), D, Dc, Lk, D, Dc, Dc, true, true, true});
    run(d, op, {addr(dV.data), addr(C.data), addr(s.dWv->data), D, Dc, Lk, D, Dc, Dc, true, true, true});
    if (s.dbq) column_sum_accumulate(dQ, *s.dbq);
    if (s.dbk) column_sum_accumulate(dK, *s.dbk);
    if (s.dbv) column_sum_accumulate(dV, *s.dbv);
    // Input gradients (overwritten).
    run(d, op, {addr(dQ.data), addr(s.Wq->data), addr(s.dX->data), Lq, Dx, D, D, Dx, Dx, false, true});
    Tensor& dKV = s.dC ? *s.dC : *s.dX;
    run(d, op, {addr(dK.data), addr(s.Wk->data), addr(dKV.data), Lk, Dc, D, D, Dc, Dc, false, true, s.dC == nullptr});
    run(d, op, {addr(dV.data), addr(s.Wv->data), addr(dKV.data), Lk, Dc, D, D, Dc, Dc, false, true, true});
}

}  // namespace

void attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Q, const Tensor& K, const Tensor& V,
                        const Tensor& Attn, const Tensor& Y_pre_Wo, const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo, const float* d_mask, Tensor& dX, Tensor& dWq,
                        Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"attention_backward", &dO, &X, &X, &Q, &K, &V, &Attn, &Y_pre_Wo, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

void mha_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                  const Tensor& Attnh, const Tensor& Yconcat, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                  const Tensor& Wo, const float* d_mask, int num_heads, Tensor& dX, Tensor& dWq, Tensor& dWk,
                  Tensor& dWv, Tensor& dWo, Tensor* dbq, Tensor* dbk, Tensor* dbv, Tensor* dbo) {
    BwdSpec s{"mha_backward", &dO, &X, &X, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    s.dbq = dbq; s.dbk = dbk; s.dbv = dbv; s.dbo = dbo;
    attention_bwd(s);
}

void self_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                             const Tensor& Attnh, const Tensor& Yconcat, const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo, const float* d_mask, int num_heads, Tensor& dX,
                             Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, d_mask, num_heads, dX, dWq, dWk, dWv, dWo,
                 nullptr, nullptr, nullptr, nullptr);
}

void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx, const Tensor& Qh,
                              const Tensor& Kh, const Tensor& Vh, const Tensor& Attnh, const Tensor& Yconcat,
                              const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads, Tensor& dX, Tensor& dCtx, Tensor& dWq,
                              Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"cross_attention_backward", &dO, &X, &Ctx, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = X.rows == Ctx.rows ? d_mask : nullptr;   // query gating only when Lq == Lk (CPU / CUDA)
    s.H = num_heads;
    s.dX = &dX;
    s.dC = &dCtx;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

void fill_vulkan_vtable_attention_bwd(::brotensor::detail::OpsVTable& v) {
    v.attention_backward = &attention_backward;
    v.mha_backward = &mha_backward;
    v.self_attention_backward = &self_attention_backward;
    v.cross_attention_backward = &cross_attention_backward;
}

}  // namespace brotensor::detail::vulkan
