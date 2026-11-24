// CUDA backwards of materialised-probability attention ops backed by Brass MIR -> PTX
// (MirKind::AttnAux, AttnBwd, Transpose, Binary, Norm) + CUDA GEMMs:
//   attention_backward, mha_backward, self_attention_backward, cross_attention_backward.

#include "cu_mir_cache.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_norm.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in batched_ops.cu / matmul.cu / concat.cu.
void linear_forward_batched(const Tensor& W, const Tensor& bias, const Tensor& X_BD, Tensor& Y_BD);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch, int width, int height);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void ensure(Tensor& t, int rows, int cols, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) {
        t.resize(rows, cols, Dtype::FP32);
    }
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

void* byte_off(void* p, std::size_t bytes) noexcept {
    return static_cast<char*>(p) + bytes;
}

void transpose_f32(const Tensor& src, Tensor& dst) {
    const std::uint32_t r = static_cast<std::uint32_t>(src.rows);
    const std::uint32_t c = static_cast<std::uint32_t>(src.cols);
    if (r == 0 || c == 0) return;
    CUfunction fn = mir_function({MirKind::Transpose, 4u});
    const std::uint32_t gx = std::max<std::uint32_t>(1u, (c + 31u) / 32u);
    const std::uint32_t gy = std::clamp((r + 31u) / 32u, 1u, 65535u);
    mir_dispatch_transpose(fn, gx, gy, 1u, src.data, dst.data, 1u, r, c);
}

void add_inplace_f32(Tensor& dst, const Tensor& src) {
    const std::uint32_t n = static_cast<std::uint32_t>(dst.size());
    if (n == 0) return;
    const int lanes = mir_vec_lanes(Dtype::FP32, n, addr(dst.data), addr(src.data));
    CUfunction fn = mir_function(
        {MirKind::Binary, std::uint32_t(BOP_ADD), Dtype::FP32, Dtype::FP32,
         static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(fn, mir_groups_1d(n, lanes), dst.data, src.data, n, 0.0f, 0.0f);
}

// A(m, k) @ B(k, n), or A(k, m)^T @ B(k, n), or A(m, k) @ B(n, k)^T via matmul /
// linear_forward_batched; when acc is true accumulates += into C.
void gemm_f32(const Tensor& A, const Tensor& B, Tensor& C,
              bool ta, bool nb, bool acc) {
    const int M = ta ? A.cols : A.rows;
    const int N = nb ? B.rows : B.cols;
    const int K = ta ? A.rows : A.cols;
    if (M == 0 || N == 0 || K == 0) return;

    Tensor AT;
    const Tensor* Ap = &A;
    if (ta) {
        AT = Tensor::empty_on(C.device, M, K, Dtype::FP32);
        transpose_f32(A, AT);
        Ap = &AT;
    }

    Tensor C_tmp;
    Tensor& dst = acc ? (C_tmp = Tensor::empty_on(C.device, M, N, Dtype::FP32)) : C;
    if (nb) {
        Tensor zb = Tensor::zeros_on(B.device, N, 1, Dtype::FP32);
        linear_forward_batched(B, zb, *Ap, dst);
    } else {
        matmul(*Ap, B, dst);
    }
    if (acc) add_inplace_f32(C, dst);
}

void col_sum_accum(const Tensor& X, Tensor& out) {
    const std::uint32_t rows = static_cast<std::uint32_t>(X.rows);
    const std::uint32_t cols = static_cast<std::uint32_t>(X.cols);
    if (rows == 0 || cols == 0) return;
    const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
    CUfunction fn = mir_function(
        {MirKind::Norm, std::uint32_t(NORM_COL_SUM), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t groups = std::min((cols + 31u) / 32u, 65535u);
    mir_dispatch_norm(fn, groups, X.data, nullptr, out.data, nullptr, nullptr,
                      nullptr, nullptr, rows, cols, 0.0f, 0.0f);
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
        if (b && (b->dtype != Dtype::FP32 || b->size() != D)) {
            fail(op, "a bias gradient must have D FP32 elements");
        }
    }
}

void attention_bwd(const BwdSpec& s) {
    const Tensor &X = *s.X, &C = *s.C;
    const int Lq = X.rows, Lk = C.rows, Dx = X.cols, Dc = C.cols, D = s.Wq->rows, H = s.H;
    const ::brotensor::Device dev = X.device;
    ensure(*s.dX, Lq, Dx, dev);
    if (s.dC) ensure(*s.dC, Lk, Dc, dev);
    if (Lq == 0 || D == 0 || H == 0) return;
    if (H < 0 || D % H != 0) fail(s.op, "num_heads must divide D");
    const int hd = D / H;
    if (Lk == 0) {
        s.dX->zero();
        if (s.dC && s.dC->size() > 0) s.dC->zero();
        return;
    }
    check(s, Lq, Lk, D, Dx, Dc, hd);

    Tensor dog_tmp;
    const Tensor* dOg = s.dO;
    if (s.qmask) {
        dog_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
        CUfunction fn = mir_function(
            {MirKind::AttnAux, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 1u});
        const std::uint64_t total = static_cast<std::uint64_t>(Lq) * D;
        const std::uint32_t gx = static_cast<std::uint32_t>(
            std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
        const std::uintptr_t ptrs[] = {addr(s.dO->data), addr(dog_tmp.data), addr(s.qmask)};
        const std::int32_t ints[] = {Lq, D};
        mir_dispatch_args(fn, gx, 256u, ptrs, ints, {});
        dOg = &dog_tmp;
    }

    Tensor dYc = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    gemm_f32(*dOg, *s.Wo, dYc, false, false, false);
    gemm_f32(*dOg, *s.Yc, *s.dWo, true, false, true);
    if (s.dbo) col_sum_accum(*dOg, *s.dbo);

    Tensor dP = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
    Tensor dQh = Tensor::empty_on(dev, H * Lq, hd, Dtype::FP32);
    Tensor dKh = Tensor::empty_on(dev, H * Lk, hd, Dtype::FP32);
    Tensor dVh = Tensor::empty_on(dev, H * Lk, hd, Dtype::FP32);
    Tensor dYc_h = Tensor::empty_on(dev, H * Lq, hd, Dtype::FP32);
    for (int h = 0; h < H; ++h) {
        copy_d2d_strided(dYc, h * hd, D, dYc_h, h * Lq * hd, hd, hd, Lq);
    }

    const std::size_t q_stride = static_cast<std::size_t>(Lq) * hd * sizeof(float);
    const std::size_t k_stride = static_cast<std::size_t>(Lk) * hd * sizeof(float);
    const std::size_t p_stride = static_cast<std::size_t>(Lq) * Lk * sizeof(float);

    for (int h = 0; h < H; ++h) {
        const Tensor dyc_v = Tensor::view(dev, byte_off(dYc_h.data, h * q_stride), Lq, hd, Dtype::FP32);
        const Tensor vh_v = Tensor::view(dev, byte_off(s.Vh->data, h * k_stride), Lk, hd, Dtype::FP32);
        Tensor dp_v = Tensor::view(dev, byte_off(dP.data, h * p_stride), Lq, Lk, Dtype::FP32);
        gemm_f32(dyc_v, vh_v, dp_v, false, true, false);

        const Tensor att_v = Tensor::view(dev, byte_off(s.Attnh->data, h * p_stride), Lq, Lk, Dtype::FP32);
        Tensor dvh_v = Tensor::view(dev, byte_off(dVh.data, h * k_stride), Lk, hd, Dtype::FP32);
        gemm_f32(att_v, dyc_v, dvh_v, true, false, false);
    }

    {
        const std::uint8_t flags = static_cast<std::uint8_t>(
            (s.kmask ? 1u : 0u) | (s.qmask ? 2u : 0u));
        CUfunction fn = mir_function(
            {MirKind::AttnBwd, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
        const std::uintptr_t ptrs[] = {
            addr(s.Attnh->data), addr(dP.data), addr(s.kmask), addr(s.qmask)};
        const std::int32_t ints[] = {H * Lq, Lq, Lk};
        const float flts[] = {1.0f / std::sqrt(static_cast<float>(hd))};
        mir_dispatch_args(fn, static_cast<std::uint32_t>(std::clamp(H * Lq, 1, 65535)),
                          256u, ptrs, ints, flts);
    }

    for (int h = 0; h < H; ++h) {
        const Tensor ds_v = Tensor::view(dev, byte_off(dP.data, h * p_stride), Lq, Lk, Dtype::FP32);
        const Tensor kh_v = Tensor::view(dev, byte_off(s.Kh->data, h * k_stride), Lk, hd, Dtype::FP32);
        const Tensor qh_v = Tensor::view(dev, byte_off(s.Qh->data, h * q_stride), Lq, hd, Dtype::FP32);
        Tensor dqh_v = Tensor::view(dev, byte_off(dQh.data, h * q_stride), Lq, hd, Dtype::FP32);
        Tensor dkh_v = Tensor::view(dev, byte_off(dKh.data, h * k_stride), Lk, hd, Dtype::FP32);
        gemm_f32(ds_v, kh_v, dqh_v, false, false, false);
        gemm_f32(ds_v, qh_v, dkh_v, true, false, false);
    }

    Tensor dQ = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    Tensor dK = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    Tensor dV = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    for (int h = 0; h < H; ++h) {
        copy_d2d_strided(dQh, h * Lq * hd, hd, dQ, h * hd, D, hd, Lq);
        copy_d2d_strided(dKh, h * Lk * hd, hd, dK, h * hd, D, hd, Lk);
        copy_d2d_strided(dVh, h * Lk * hd, hd, dV, h * hd, D, hd, Lk);
    }

    gemm_f32(dQ, X, *s.dWq, true, false, true);
    gemm_f32(dK, C, *s.dWk, true, false, true);
    gemm_f32(dV, C, *s.dWv, true, false, true);
    if (s.dbq) col_sum_accum(dQ, *s.dbq);
    if (s.dbk) col_sum_accum(dK, *s.dbk);
    if (s.dbv) col_sum_accum(dV, *s.dbv);

    gemm_f32(dQ, *s.Wq, *s.dX, false, false, false);
    Tensor& dKV = s.dC ? *s.dC : *s.dX;
    gemm_f32(dK, *s.Wk, dKV, false, false, s.dC == nullptr);
    gemm_f32(dV, *s.Wv, dKV, false, false, true);
}

}  // namespace

void attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Q, const Tensor& K,
                        const Tensor& V, const Tensor& Attn, const Tensor& Y_pre_Wo,
                        const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask, Tensor& dX, Tensor& dWq, Tensor& dWk,
                        Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"attention_backward", &dO, &X, &X, &Q, &K, &V, &Attn, &Y_pre_Wo, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

void mha_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh, const Tensor& Kh,
                  const Tensor& Vh, const Tensor& Attnh, const Tensor& Yconcat,
                  const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                  const float* d_mask, int num_heads, Tensor& dX, Tensor& dWq, Tensor& dWk,
                  Tensor& dWv, Tensor& dWo, Tensor* dbq, Tensor* dbk, Tensor* dbv, Tensor* dbo) {
    BwdSpec s{"mha_backward", &dO, &X, &X, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.dX = &dX;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    s.dbq = dbq; s.dbk = dbk; s.dbv = dbv; s.dbo = dbo;
    attention_bwd(s);
}

void self_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh,
                             const Tensor& Kh, const Tensor& Vh, const Tensor& Attnh,
                             const Tensor& Yconcat, const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                             int num_heads, Tensor& dX, Tensor& dWq, Tensor& dWk,
                             Tensor& dWv, Tensor& dWo) {
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, d_mask, num_heads,
                 dX, dWq, dWk, dWv, dWo, nullptr, nullptr, nullptr, nullptr);
}

void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                              const Tensor& Attnh, const Tensor& Yconcat, const Tensor& Wq,
                              const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads, Tensor& dX, Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo) {
    BwdSpec s{"cross_attention_backward", &dO, &X, &Ctx, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = (X.rows == Ctx.rows) ? d_mask : nullptr;
    s.H = num_heads;
    s.dX = &dX;
    s.dC = &dCtx;
    s.dWq = &dWq; s.dWk = &dWk; s.dWv = &dWv; s.dWo = &dWo;
    attention_bwd(s);
}

}  // namespace brotensor::detail::cuda
