// CUDA projection-fused attention with materialised probabilities backed by
// Brass MIR -> PTX (MirKind::AttnAux, AttnSoftmax, Cast) + CUDA GEMMs:
//   self_attention_bias_forward, self_attention_bias_int8w_fp16,
//   cross_attention_forward, cross_attention_forward_with_attn,
//   cross_attention_forward_train, mha_forward, attention_forward,
//   self_attention_forward, self_attention_forward_train.

#include "cu_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in batched_ops.cu / gemm.cu / int8_quant.cu / matmul.cu / concat.cu /
// flash_attention.cu / ops_misc.cpp / ops_attention_bwd.cpp / ops_attention_relpos.cpp.
void linear_forward_batched(const Tensor& W, const Tensor& bias, const Tensor& X_BD, Tensor& Y_BD);
void linear_forward_batched_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                                       const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch, int width, int height);
void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask, int num_heads, bool causal, Tensor& O);
void attention_token_moments(const Tensor& Attn, int h_lat, int w_lat,
                             Tensor& mass, Tensor& centroid);
void attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Q, const Tensor& K,
                        const Tensor& V, const Tensor& Attn, const Tensor& Y_pre_Wo,
                        const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask, Tensor& dX, Tensor& dWq, Tensor& dWk,
                        Tensor& dWv, Tensor& dWo);
void mha_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh, const Tensor& Kh,
                  const Tensor& Vh, const Tensor& Attnh, const Tensor& Yconcat,
                  const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                  const float* d_mask, int num_heads, Tensor& dX, Tensor& dWq, Tensor& dWk,
                  Tensor& dWv, Tensor& dWo, Tensor* dbq, Tensor* dbk, Tensor* dbv, Tensor* dbo);
void self_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Qh,
                             const Tensor& Kh, const Tensor& Vh, const Tensor& Attnh,
                             const Tensor& Yconcat, const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                             int num_heads, Tensor& dX, Tensor& dWq, Tensor& dWk,
                             Tensor& dWv, Tensor& dWo);
void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh, const Tensor& Vh,
                              const Tensor& Attnh, const Tensor& Yconcat, const Tensor& Wq,
                              const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads, Tensor& dX, Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk, Tensor& dWv, Tensor& dWo);
void self_attention_decomposed_rel_pos_forward(const Tensor& X, const Tensor& Wq, const Tensor* bq,
                                               const Tensor& Wk, const Tensor* bk,
                                               const Tensor& Wv, const Tensor* bv,
                                               const Tensor& Wo, const Tensor* bo,
                                               const Tensor& rel_pos_h, const Tensor& rel_pos_w,
                                               int num_heads, int grid_h, int grid_w,
                                               float scale, Tensor& O);
void self_attention_decomposed_rel_pos_windowed_forward(
    const Tensor& X, const Tensor& Wq, const Tensor* bq, const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv, const Tensor& Wo, const Tensor* bo,
    const Tensor& rel_pos_h, const Tensor& rel_pos_w, int num_heads, int grid_h, int grid_w,
    int window, float scale, Tensor& O);
void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk, int num_heads,
                             int head_dim, Tensor& Bias);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

bool present(const Tensor* t) noexcept {
    return t && t->data && t->size() > 0;
}

void ensure(Tensor& t, int rows, int cols, Dtype dt, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

void* byte_off(void* p, std::size_t bytes) noexcept {
    return static_cast<char*>(p) + bytes;
}

void cast_tensor(const Tensor& src, Tensor& dst) {
    const long long n_ll = src.size();
    if (n_ll <= 0) return;
    const std::uint32_t n = static_cast<std::uint32_t>(n_ll);
    const int lanes = mir_vec_lanes_mixed(src.dtype, dst.dtype, n, addr(src.data), addr(dst.data));
    CUfunction fn = mir_function(
        {MirKind::Cast, 0u, src.dtype, dst.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_cast(fn, mir_groups_1d(n, lanes), src.data, dst.data, n);
}

void run_attn_aux(std::uint32_t mode, Dtype out_dt, const void* x, void* y,
                  const void* mask, std::uint32_t rows, std::uint32_t n) {
    const std::uint64_t total = (mode == 0u) ? static_cast<std::uint64_t>(rows) * n : n;
    if (total == 0) return;
    CUfunction fn = mir_function(
        {MirKind::AttnAux, mode, out_dt, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(mask != nullptr ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
    const std::uintptr_t ptrs[] = {addr(x), addr(y), addr(mask)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(rows), static_cast<std::int32_t>(n)};
    mir_dispatch_args(fn, gx, 256u, ptrs, ints, {});
}

void project_to_f32(const char* op, const Tensor& X, const Tensor& W,
                    const Tensor* b, Tensor& Y_f32) {
    Tensor X32;
    const Tensor* Xp = &X;
    if (X.dtype != Dtype::FP32) {
        X32 = Tensor::empty_on(X.device, X.rows, X.cols, Dtype::FP32);
        cast_tensor(X, X32);
        Xp = &X32;
    }
    Tensor b32;
    if (present(b)) {
        if (b->dtype != X.dtype || b->size() != W.rows) {
            fail(op, "a projection bias must have W.rows elements of X's dtype");
        }
        if (b->dtype == Dtype::FP32) {
            linear_forward_batched(W, *b, *Xp, Y_f32);
            return;
        }
        b32 = Tensor::empty_on(X.device, W.rows, 1, Dtype::FP32);
        cast_tensor(*b, b32);
    } else {
        b32 = Tensor::zeros_on(X.device, W.rows, 1, Dtype::FP32);
    }
    linear_forward_batched(W, b32, *Xp, Y_f32);
}

void dense_attention_f32(::brotensor::Device dev, const Tensor& Qh, const Tensor& Kh,
                         const Tensor& Vh, Tensor& Yc, Tensor* probs_out,
                         int Lq, int Lk, int H, int hd, float scale_in,
                         const float* kmask, const float* qmask,
                         const float* bias, long long bias_hs, bool mask_ge) {
    if (Lq <= 0 || Lk <= 0) return;
    const float scale = std::isnan(scale_in)
        ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)))
        : scale_in;
    Tensor S_tmp;
    Tensor& S = probs_out ? *probs_out : (S_tmp = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32));
    Tensor zb_k = Tensor::zeros_on(dev, Lk, 1, Dtype::FP32);
    const std::size_t q_stride = static_cast<std::size_t>(Lq) * hd * sizeof(float);
    const std::size_t k_stride = static_cast<std::size_t>(Lk) * hd * sizeof(float);
    const std::size_t s_stride = static_cast<std::size_t>(Lq) * Lk * sizeof(float);
    for (int h = 0; h < H; ++h) {
        const Tensor qh = Tensor::view(dev, byte_off(Qh.data, h * q_stride), Lq, hd, Dtype::FP32);
        const Tensor kh = Tensor::view(dev, byte_off(Kh.data, h * k_stride), Lk, hd, Dtype::FP32);
        Tensor sh = Tensor::view(dev, byte_off(S.data, h * s_stride), Lq, Lk, Dtype::FP32);
        linear_forward_batched(kh, zb_k, qh, sh);
    }

    const std::uint8_t flags = static_cast<std::uint8_t>(
        (kmask ? 1u : 0u) | (qmask ? 2u : 0u) | (mask_ge ? 4u : 0u));
    const std::uint32_t bias_mode = bias ? 1u : 0u;
    CUfunction fn = mir_function(
        {MirKind::AttnSoftmax, bias_mode, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t sm_rows = static_cast<std::uint32_t>(H * Lq);
    const std::int32_t bhs = static_cast<std::int32_t>(
        bias_hs < 0 ? static_cast<long long>(Lq) * Lk : bias_hs);
    const std::uintptr_t ptrs[] = {
        addr(S.data), addr(S.data), addr(bias), addr(bias), addr(qmask)};
    // Note ptrs[3] is kmask for bias_mode != 2; override below:
    const std::uintptr_t sm_ptrs[] = {
        addr(S.data), addr(S.data), addr(bias), addr(kmask), addr(qmask)};
    (void)ptrs;
    const std::int32_t sm_ints[] = {
        static_cast<std::int32_t>(sm_rows), Lk, Lq, 0, Lk, Lk, 0, bhs};
    const float flts[] = {scale};
    mir_dispatch_args(fn, std::clamp(sm_rows, 1u, 65535u), 256u, sm_ptrs, sm_ints, flts);

    Tensor Yh = Tensor::empty_on(dev, H * Lq, hd, Dtype::FP32);
    for (int h = 0; h < H; ++h) {
        const Tensor ph = Tensor::view(dev, byte_off(S.data, h * s_stride), Lq, Lk, Dtype::FP32);
        const Tensor vh = Tensor::view(dev, byte_off(Vh.data, h * k_stride), Lk, hd, Dtype::FP32);
        Tensor yh = Tensor::view(dev, byte_off(Yh.data, h * q_stride), Lq, hd, Dtype::FP32);
        matmul(ph, vh, yh);
    }
    const int D = H * hd;
    for (int h = 0; h < H; ++h) {
        copy_d2d_strided(Yh, h * Lq * hd, hd, Yc, h * hd, D, hd, Lq);
    }
}

struct ProjSpec {
    const char* op;
    const Tensor* X;
    const Tensor* C;
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const Tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    const float* kmask = nullptr;
    const float* qmask = nullptr;
    const float* bias = nullptr;
    long long bias_hs = -1;
    int H = 1;
    float scale = NAN;
    Tensor *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;
    Tensor *Attnh = nullptr, *Yc = nullptr, *AttnAvg = nullptr, *O = nullptr;
};

void proj_attention(const ProjSpec& s) {
    const char* op = s.op;
    const Tensor &X = *s.X, &C = *s.C;
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    for (const Tensor* t : {s.C, s.Wq, s.Wk, s.Wv, s.Wo}) {
        if (t->dtype != dt) fail(op, "X, Ctx and the weights must share one dtype");
    }
    const int Lq = X.rows, Lk = C.rows, D = s.Wq->rows, H = s.H;
    if (H <= 0 || D % H != 0) fail(op, "num_heads must divide D");
    if (s.Wq->cols != X.cols || s.Wk->rows != D || s.Wv->rows != D ||
        s.Wk->cols != C.cols || s.Wv->cols != C.cols ||
        s.Wo->rows != D || s.Wo->cols != D) {
        fail(op, "weight shapes do not match X / Ctx");
    }
    const int hd = D / H;
    const ::brotensor::Device dev = X.device;
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
    project_to_f32(op, X, *s.Wq, s.bq, Q);
    project_to_f32(op, C, *s.Wk, s.bk, K);
    project_to_f32(op, C, *s.Wv, s.bv, V);

    Tensor qh_tmp, kh_tmp, vh_tmp;
    Tensor& Qh = s.Qh ? *s.Qh : (qh_tmp = Tensor::empty_on(dev, H * Lq, hd, Dtype::FP32));
    Tensor& Kh = s.Kh ? *s.Kh : (kh_tmp = Tensor::empty_on(dev, H * Lk, hd, Dtype::FP32));
    Tensor& Vh = s.Vh ? *s.Vh : (vh_tmp = Tensor::empty_on(dev, H * Lk, hd, Dtype::FP32));
    for (int h = 0; h < H; ++h) {
        copy_d2d_strided(Q, h * hd, D, Qh, h * Lq * hd, hd, hd, Lq);
        copy_d2d_strided(K, h * hd, D, Kh, h * Lk * hd, hd, hd, Lk);
        copy_d2d_strided(V, h * hd, D, Vh, h * Lk * hd, hd, hd, Lk);
    }

    Tensor yc_tmp, probs_tmp;
    Tensor& Yc = s.Yc ? *s.Yc : (yc_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    Tensor* probs = s.Attnh;
    if (!probs && s.AttnAvg) {
        probs_tmp = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
        probs = &probs_tmp;
    }
    dense_attention_f32(dev, Qh, Kh, Vh, Yc, probs, Lq, Lk, H, hd, s.scale,
                        s.kmask, s.qmask, s.bias, s.bias_hs, true);

    if (s.AttnAvg) {
        run_attn_aux(1u, dt, probs->data, s.AttnAvg->data, nullptr,
                     static_cast<std::uint32_t>(H),
                     static_cast<std::uint32_t>(Lq) * static_cast<std::uint32_t>(Lk));
    }

    Tensor o32_tmp;
    Tensor& O32 = (dt == Dtype::FP32) ? *s.O : (o32_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    project_to_f32(op, Yc, *s.Wo, s.bo, O32);
    if (dt != Dtype::FP32 || s.qmask) {
        run_attn_aux(0u, dt, O32.data, s.O->data, s.qmask,
                     static_cast<std::uint32_t>(Lq), static_cast<std::uint32_t>(D));
    }
}

}  // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo, const Tensor* bq,
                                 const Tensor* bk, const Tensor* bv, const Tensor* bo,
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
        if (attn_bias->dtype != Dtype::FP32) fail(op, "attn_bias must be FP32");
        if (num_heads <= 0 || attn_bias->size() != num_heads * L * L) {
            fail(op, "attn_bias must be (num_heads*L, L)");
        }
        s.bias = static_cast<const float*>(attn_bias->data);
    }
    proj_attention(s);
}

void self_attention_bias_int8w_fp16(const Tensor& X, const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask, const Tensor* attn_bias,
                                    int num_heads, float scale, Tensor& O) {
    constexpr const char* op = "self_attention_bias_int8w_fp16";
    if (X.dtype != Dtype::FP16) fail(op, "X must be FP16");
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        fail(op, "Wq/Wk/Wv/Wo must be INT8");
    }
    if (sq.dtype != Dtype::FP32 || sk.dtype != Dtype::FP32 ||
        sv.dtype != Dtype::FP32 || so.dtype != Dtype::FP32) {
        fail(op, "scales must be FP32");
    }
    const int L = X.rows, D = X.cols, H = num_heads;
    if (H <= 0 || D % H != 0) fail(op, "num_heads must divide D");
    if (Wq_int8.rows != D || Wq_int8.cols != D || Wk_int8.rows != D || Wk_int8.cols != D ||
        Wv_int8.rows != D || Wv_int8.cols != D || Wo_int8.rows != D || Wo_int8.cols != D) {
        fail(op, "Wq/Wk/Wv/Wo must be (D, D)");
    }
    if (sq.size() != D || sk.size() != D || sv.size() != D || so.size() != D) {
        fail(op, "each scale tensor must have D entries");
    }
    const float* bias_p = nullptr;
    if (present(attn_bias)) {
        if (attn_bias->dtype != Dtype::FP32) fail(op, "attn_bias must be FP32");
        if (attn_bias->size() != static_cast<long long>(H) * L * L) {
            fail(op, "attn_bias must be (num_heads*L, L)");
        }
        bias_p = static_cast<const float*>(attn_bias->data);
    }
    ensure(O, L, D, Dtype::FP16, X.device);
    if (L == 0 || D == 0) return;

    const int hd = D / H;
    Tensor Q16 = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    Tensor K16 = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    Tensor V16 = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    linear_forward_batched_int8w_fp16(Wq_int8, sq, nullptr, X, Q16);
    linear_forward_batched_int8w_fp16(Wk_int8, sk, nullptr, X, K16);
    linear_forward_batched_int8w_fp16(Wv_int8, sv, nullptr, X, V16);

    Tensor Q = Tensor::empty_on(X.device, L, D, Dtype::FP32);
    Tensor K = Tensor::empty_on(X.device, L, D, Dtype::FP32);
    Tensor V = Tensor::empty_on(X.device, L, D, Dtype::FP32);
    cast_tensor(Q16, Q);
    cast_tensor(K16, K);
    cast_tensor(V16, V);

    Tensor Qh = Tensor::empty_on(X.device, H * L, hd, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(X.device, H * L, hd, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(X.device, H * L, hd, Dtype::FP32);
    for (int h = 0; h < H; ++h) {
        copy_d2d_strided(Q, h * hd, D, Qh, h * L * hd, hd, hd, L);
        copy_d2d_strided(K, h * hd, D, Kh, h * L * hd, hd, hd, L);
        copy_d2d_strided(V, h * hd, D, Vh, h * L * hd, hd, hd, L);
    }
    Tensor Yc = Tensor::empty_on(X.device, L, D, Dtype::FP32);
    dense_attention_f32(X.device, Qh, Kh, Vh, Yc, nullptr, L, L, H, hd, scale,
                        d_mask, d_mask, bias_p, static_cast<long long>(L) * L, true);

    Tensor Yc16 = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    cast_tensor(Yc, Yc16);
    linear_forward_batched_int8w_fp16(Wo_int8, so, nullptr, Yc16, O);
    if (d_mask) {
        Tensor O32 = Tensor::empty_on(X.device, L, D, Dtype::FP32);
        cast_tensor(O, O32);
        run_attn_aux(0u, Dtype::FP16, O32.data, O.data, d_mask,
                     static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(D));
    }
}

void cross_attention_forward(const Tensor& X, const Tensor& Ctx, const Tensor& Wq,
                             const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads, Tensor& O) {
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        if (Ctx.dtype != X.dtype) fail("cross_attention_forward", "Ctx dtype must match X dtype");
        flash_attention_qkvo_forward(X, &Ctx, Wq, nullptr, Wk, nullptr, Wv, nullptr,
                                     Wo, nullptr, d_mask, num_heads, false, O);
        return;
    }
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
        if (attn_logit_bias->dtype != Dtype::FP32) fail(op, "attn_logit_bias must be FP32");
        if (attn_logit_bias->rows != X.rows || attn_logit_bias->cols != Ctx.rows) {
            fail(op, "attn_logit_bias must be (Lq, Lk)");
        }
        s.bias = static_cast<const float*>(attn_logit_bias->data);
        s.bias_hs = 0;
    }
    proj_attention(s);
}

void mha_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                 const Tensor& Wo, const Tensor* bq, const Tensor* bk, const Tensor* bv,
                 const Tensor* bo, const float* d_mask, int num_heads, Tensor& Qh,
                 Tensor& Kh, Tensor& Vh, Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"mha_forward", &X, &X, &Wq, &Wk, &Wv, &Wo, bq, bk, bv, bo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

void attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                       const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                       Tensor& Q, Tensor& K, Tensor& V, Tensor& Attn,
                       Tensor& Y_pre_Wo, Tensor& O) {
    ProjSpec s{"attention_forward", &X, &X, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.Qh = &Q; s.Kh = &K; s.Vh = &V; s.Attnh = &Attn; s.Yc = &Y_pre_Wo;
    s.O = &O;
    proj_attention(s);
}

void cross_attention_forward_train(const Tensor& X, const Tensor& Ctx, const Tensor& Wq,
                                   const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                                   const float* d_mask, int num_heads, Tensor& Qh,
                                   Tensor& Kh, Tensor& Vh, Tensor& Attnh,
                                   Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"cross_attention_forward_train", &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = (X.rows == Ctx.rows) ? d_mask : nullptr;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

void self_attention_forward_train(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                                  int num_heads, Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo, nullptr, nullptr, nullptr, nullptr, d_mask,
                num_heads, Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                            int num_heads, Tensor& O) {
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

void fill_cuda_vtable_attention(::brotensor::detail::OpsVTable& v) {
    v.mha_forward = &mha_forward;
    v.mha_backward = &mha_backward;
    v.attention_forward = &attention_forward;
    v.attention_backward = &attention_backward;
    v.cross_attention_forward = &cross_attention_forward;
    v.cross_attention_forward_train = &cross_attention_forward_train;
    v.cross_attention_backward = &cross_attention_backward;
    v.cross_attention_forward_with_attn = &cross_attention_forward_with_attn;
    v.self_attention_forward = &self_attention_forward;
    v.self_attention_bias_forward = &self_attention_bias_forward;
    v.rel_pos_bias_xl_forward = &rel_pos_bias_xl_forward;
    v.self_attention_bias_int8w_fp16 = &self_attention_bias_int8w_fp16;
    v.self_attention_decomposed_rel_pos_forward = &self_attention_decomposed_rel_pos_forward;
    v.self_attention_decomposed_rel_pos_windowed_forward =
        &self_attention_decomposed_rel_pos_windowed_forward;
    v.self_attention_forward_train = &self_attention_forward_train;
    v.self_attention_backward = &self_attention_backward;
    v.attention_token_moments = &attention_token_moments;
}

}  // namespace brotensor::detail::cuda
