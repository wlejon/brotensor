// Metal Wave 8 materialised-probability attention backwards via single-source
// brass MIR kernels (MirKind::{AttnAux, AttnBwd, Norm}) + Metal FP32 simdgroup GEMM:
//   attention_backward, mha_backward, self_attention_backward, cross_attention_backward.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "fp16_matmul.h"
#import "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::pool_require;

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void ensure(Tensor& t, int rows, int cols, Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) {
        t.resize(rows, cols, Dtype::FP32);
    }
}

struct G {
    const Tensor* a;
    const Tensor* b;
    Tensor* c;
    int m, n, k, lda, ldb, ldc;
    bool ta = false, tb = false, acc = false;
    int batch = 1;
    long long sa = 0, sb = 0, sc = 0;
};

void run_gemm(const G& x) {
    metal_impl::AbtMixed g;
    g.A = buffer_for(*x.a); g.ofs_A = buffer_offset_for(*x.a);
    g.lda = static_cast<std::uint64_t>(x.lda); g.transA = x.ta;
    g.B = buffer_for(*x.b); g.ofs_B = buffer_offset_for(*x.b);
    g.ldb = static_cast<std::uint64_t>(x.ldb); g.transB = x.tb;
    g.C = buffer_for(*x.c); g.ofs_C = buffer_offset_for(*x.c);
    g.ldc = static_cast<std::uint64_t>(x.ldc);
    g.M = x.m; g.N = x.n; g.K = x.k;
    g.in = g.out = metal_impl::kAbtF32;
    g.epilogue = x.acc ? metal_impl::kAbtAccumulate : metal_impl::kAbtStore;
    g.batch = x.batch;
    g.strideA = static_cast<std::uint64_t>(x.sa);
    g.strideB = static_cast<std::uint64_t>(x.sb);
    g.strideC = static_cast<std::uint64_t>(x.sc);
    metal_impl::launch_matmul_abt_mixed(g);
}

void col_sum_accum(const char* op, const Tensor& X, Tensor& out) {
    const std::uint32_t rows = static_cast<std::uint32_t>(X.rows);
    const std::uint32_t cols = static_cast<std::uint32_t>(X.cols);
    if (rows == 0 || cols == 0) return;
    const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
    const MirKernel mk = mir_kernel(
        {MirKind::Norm, std::uint32_t(NORM_COL_SUM), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t groups = std::min<std::uint32_t>((cols + 31u) / 32u, 65535u);
    mir_dispatch_norm(mk, groups, op,
                      X.data, nullptr, out.data, nullptr, nullptr, nullptr, nullptr,
                      rows, cols, 0.0f, 0.0f);
}

struct BwdSpec {
    const char* op;
    const Tensor *dO, *X, *C;
    const Tensor *Qh, *Kh, *Vh, *Attnh, *Yc;
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const float* kmask = nullptr;
    const float* qmask = nullptr;
    int H = 1;
    Tensor* dX;
    Tensor* dC = nullptr;
    Tensor *dWq, *dWk, *dWv, *dWo;
    Tensor *dbq = nullptr, *dbk = nullptr, *dbv = nullptr, *dbo = nullptr;
};

void check_bwd(const BwdSpec& s, int Lq, int Lk, int D, int Dx, int Dc, int hd) {
    const char* op = s.op;
    for (const Tensor* t : {s.dO, s.X, s.C, s.Qh, s.Kh, s.Vh, s.Attnh, s.Yc,
                            s.Wq, s.Wk, s.Wv, s.Wo,
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
        if (b && (b->dtype != Dtype::FP32 || b->size() != D)) {
            fail(op, "a bias gradient must have D FP32 elements");
        }
    }
}

void attention_bwd(const BwdSpec& s) {
    const char* op = s.op;
    const Tensor &X = *s.X, &C = *s.C;
    const int Lq = X.rows, Lk = C.rows, Dx = X.cols, Dc = C.cols, D = s.Wq->rows, H = s.H;
    const Device dev = X.device;
    ensure(*s.dX, Lq, Dx, dev);
    if (s.dC) ensure(*s.dC, Lk, Dc, dev);
    if (Lq == 0 || D == 0 || H == 0) return;
    if (H < 0 || D % H != 0) fail(op, "num_heads must divide D");
    const int hd = D / H;
    if (Lk == 0) {
        s.dX->zero();
        if (s.dC && s.dC->size() > 0) s.dC->zero();
        return;
    }
    check_bwd(s, Lq, Lk, D, Dx, Dc, hd);
    if (s.kmask) { NSUInteger o = 0; pool_require(s.kmask, o, op, "mask"); }
    if (s.qmask) { NSUInteger o = 0; pool_require(s.qmask, o, op, "mask"); }

    Tensor dog_tmp;
    const Tensor* dOg = s.dO;
    if (s.qmask) {
        dog_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
        const MirKernel mk = mir_kernel(
            {MirKind::AttnAux, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 1u});
        const std::uint64_t total = static_cast<std::uint64_t>(Lq) * D;
        const std::uint32_t gx = static_cast<std::uint32_t>(
            std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
        const void* ptrs[] = {s.dO->data, dog_tmp.data, s.qmask};
        const std::int32_t ints[] = {static_cast<std::int32_t>(Lq), static_cast<std::int32_t>(D)};
        mir_dispatch_args(mk, gx, op, ptrs, ints, {});
        dOg = &dog_tmp;
    }

    Tensor dYc = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    run_gemm({dOg, s.Wo, &dYc, Lq, D, D, D, D, D, false, true});
    run_gemm({dOg, s.Yc, s.dWo, D, D, Lq, D, D, D, true, true, true});
    if (s.dbo) col_sum_accum(op, *dOg, *s.dbo);

    const long long pq = static_cast<long long>(Lq) * Lk;
    Tensor dP = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
    Tensor dQ = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    Tensor dK = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    Tensor dV = Tensor::empty_on(dev, Lk, D, Dtype::FP32);

    // dP_h (Lq, Lk) = dYc_h @ Vh_h^T
    run_gemm({&dYc, s.Vh, &dP, Lq, Lk, hd, D, hd, Lk, false, false, false,
              H, hd, static_cast<long long>(Lk) * hd, pq});
    // dV_h (Lk, hd) = P_h^T @ dYc_h
    run_gemm({s.Attnh, &dYc, &dV, Lk, hd, Lq, Lk, D, D, true, true, false,
              H, pq, hd, hd});

    // dS = softmax backward in place over dP
    {
        const std::uint8_t flags = static_cast<std::uint8_t>(
            (s.kmask ? 1u : 0u) | (s.qmask ? 2u : 0u));
        const MirKernel mk = mir_kernel(
            {MirKind::AttnBwd, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
        const void* ptrs[] = {s.Attnh->data, dP.data, s.kmask, s.qmask};
        const std::int32_t ints[] = {
            static_cast<std::int32_t>(H * Lq),
            static_cast<std::int32_t>(Lq),
            static_cast<std::int32_t>(Lk),
        };
        const float flts[] = {1.0f / std::sqrt(static_cast<float>(hd))};
        mir_dispatch_args(mk, static_cast<std::uint32_t>(std::clamp(H * Lq, 1, 65535)),
                          op, ptrs, ints, flts);
    }

    // dQ_h = dS_h @ Kh_h;  dK_h = dS_h^T @ Qh_h
    run_gemm({&dP, s.Kh, &dQ, Lq, hd, Lk, Lk, hd, D, false, true, false,
              H, pq, static_cast<long long>(Lk) * hd, hd});
    run_gemm({&dP, s.Qh, &dK, Lk, hd, Lq, Lk, hd, D, true, true, false,
              H, pq, static_cast<long long>(Lq) * hd, hd});

    // Weight and bias gradients (accumulated).
    run_gemm({&dQ, s.X, s.dWq, D, Dx, Lq, D, Dx, Dx, true, true, true});
    run_gemm({&dK, s.C, s.dWk, D, Dc, Lk, D, Dc, Dc, true, true, true});
    run_gemm({&dV, s.C, s.dWv, D, Dc, Lk, D, Dc, Dc, true, true, true});
    if (s.dbq) col_sum_accum(op, dQ, *s.dbq);
    if (s.dbk) col_sum_accum(op, dK, *s.dbk);
    if (s.dbv) col_sum_accum(op, dV, *s.dbv);

    // Input gradients (overwritten).
    run_gemm({&dQ, s.Wq, s.dX, Lq, Dx, D, D, Dx, Dx, false, true, false});
    Tensor* dKV = s.dC ? s.dC : s.dX;
    run_gemm({&dK, s.Wk, dKV, Lk, Dc, D, D, Dc, Dc, false, true, s.dC == nullptr});
    run_gemm({&dV, s.Wv, dKV, Lk, Dc, D, D, Dc, Dc, false, true, true});
}

}  // namespace

void attention_backward(const Tensor& dO, const Tensor& X,
                        const Tensor& Q, const Tensor& K, const Tensor& V,
                        const Tensor& Attn, const Tensor& Y_pre_Wo,
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask,
                        Tensor& dX, Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"attention_backward", &dO, &X, &X, &Q, &K, &V, &Attn, &Y_pre_Wo, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

void mha_backward(const Tensor& dO, const Tensor& X,
                  const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                  const Tensor& Attnh, const Tensor& Yconcat,
                  const Tensor& Wq, const Tensor& Wk,
                  const Tensor& Wv, const Tensor& Wo,
                  const float* d_mask, int num_heads,
                  Tensor& dX, Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo,
                  Tensor* dbq, Tensor* dbk, Tensor* dbv, Tensor* dbo) {
    BwdSpec s{"mha_backward", &dO, &X, &X, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    s.dbq = dbq; s.dbk = dbk; s.dbv = dbv; s.dbo = dbo;
    attention_bwd(s);
}

void self_attention_backward(const Tensor& dO, const Tensor& X,
                             const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                             const Tensor& Attnh, const Tensor& Yconcat,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads,
                             Tensor& dX, Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, d_mask, num_heads,
                 dX, dWq, dWk, dWv, dWo, nullptr, nullptr, nullptr, nullptr);
}

void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                              const Tensor& Attnh, const Tensor& Yconcat,
                              const Tensor& Wq, const Tensor& Wk,
                              const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads,
                              Tensor& dX, Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"cross_attention_backward", &dO, &X, &Ctx, &Qh, &Kh, &Vh, &Attnh, &Yconcat,
              &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = (X.rows == Ctx.rows) ? d_mask : nullptr;
    s.H = num_heads;
    s.dX = &dX;
    s.dC = &dCtx;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

}  // namespace brotensor::detail::metal
