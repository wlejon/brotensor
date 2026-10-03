// Vulkan projection-fused attention with materialised probabilities:
// self_attention_bias_forward (T5 / ALiBi biases), cross_attention_forward,
// cross_attention_forward_with_attn (head-averaged map + logit bias),
// cross_attention_forward_train, mha_forward, attention_forward (one head), self_attention_forward(_train), and rel_pos_bias_xl_forward.
// Contracts follow the CPU reference (src/cpu/self_attention_bias.cpp,
// cross_attention.cpp, ops_impl.cpp) and the CUDA backend's dtype rule
// for these ops: projections, scores, probabilities and the per-head
// outputs are FP32 whatever the activations' dtype; X, the weights and the
// biases share one dtype (FP32 / FP16 / BF16); O (and AttnAvg) come out in
// X's dtype; masks are valid where >= 0.5.
//
// Every op is: Q, K, V = X Wq^T + bq, ... (gemm(), FP32 result), the dense
// path (ops_attention_dense.cpp) into an FP32 Yconcat, O = Yconcat Wo^T + bo
// (FP32), then attn_aux.comp's cast with the query gating.

#include "detail/attention.h"
#include "detail/gemm.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);   // ops_copy.cpp
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch, Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height);                 // ops_copy.cpp

namespace {

struct AuxPush {
    std::uint64_t x, y, mask;
    std::uint32_t rows, n;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

bool present(const Tensor* t) { return t && t->data && t->size() > 0; }

void ensure(Tensor& t, int rows, int cols, Dtype dt, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;   // a fresh output from inside the backend
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

// Y(M, N) FP32 = X(M, K) W(N, K)^T + b, X and W of one dtype; b (N) in that
// dtype too (the 16-bit -> FP32 GEMM reads its bias in the operands' dtype).
void project32(const char* op, const Tensor& X, const Tensor& W, const Tensor* b, Tensor& Y) {
    GemmArgs g;
    g.op = op;
    g.a = addr(X.data); g.b = addr(W.data); g.c = addr(Y.data);
    g.da = X.dtype; g.db = W.dtype; g.dc = Dtype::FP32;
    if (present(b)) {
        if (b->dtype != X.dtype || b->size() != W.rows) fail(op, "a projection bias must have W.rows elements of X's dtype");
        g.bias = addr(b->data);
    }
    g.m = X.rows; g.n = W.rows; g.k = X.cols;
    g.lda = X.cols; g.ldb = W.cols; g.ldc = W.rows;
    gemm(device_of(X), g);
}

void aux(DeviceCtx& d, std::uint32_t mode, std::uint64_t x, const Tensor& y, std::uint64_t mask, std::uint32_t rows,
         std::uint32_t n) {
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::attn_aux_f32, y.dtype), {mode});
    const AuxPush pc{x, addr(y.data), mask, rows, n};
    const std::uint64_t total = mode == AUX_ROW_GATE ? std::uint64_t(rows) * n : n;
    launch(d, k, pc, groups_1d(total, k));
}

struct ProjSpec {
    const char* op;
    const Tensor* X;
    const Tensor* C;                 // key / value source (X for self-attention)
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const Tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    const float* kmask = nullptr;    // key validity (lk)
    const float* qmask = nullptr;    // query gating (lq): zero probabilities and O rows
    std::uint64_t bias = 0;          // FP32 logit bias
    long long bias_hs = -1;
    int H = 1;
    float scale = NAN;               // NaN = 1 / sqrt(hd)
    Tensor *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;   // FP32 (H * L, hd)
    Tensor *Attnh = nullptr;         // FP32 (H * lq, lk)
    Tensor *Yc = nullptr;            // FP32 (lq, D)
    Tensor *AttnAvg = nullptr;       // X's dtype (lq, lk)
    Tensor* O = nullptr;             // X's dtype (lq, D)
};

void proj_attention(const ProjSpec& s) {
    const char* op = s.op;
    const Tensor &X = *s.X, &C = *s.C;
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    for (const Tensor* t : {s.C, s.Wq, s.Wk, s.Wv, s.Wo}) {
        if (t->dtype != dt) fail(op, "X, Ctx and the weights must share one dtype");
    }
    const int Lq = X.rows, Lk = C.rows, D = s.Wq->rows, H = s.H;
    if (H <= 0 || D % H != 0) fail(op, "num_heads must divide D");
    if (s.Wq->cols != X.cols || s.Wk->rows != D || s.Wv->rows != D || s.Wk->cols != C.cols ||
        s.Wv->cols != C.cols || s.Wo->rows != D || s.Wo->cols != D) {
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
    DeviceCtx& d = device_of(X);
    if (Lk == 0) {   // nothing to attend: the CPU leaves the outputs as resized; zero them
        for (Tensor* t : {s.O, s.Yc, s.AttnAvg}) if (t && t->size() > 0) t->zero();
        return;
    }

    Tensor Q = Tensor::empty_on(dev, Lq, D, Dtype::FP32);
    Tensor K = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    Tensor V = Tensor::empty_on(dev, Lk, D, Dtype::FP32);
    project32(op, X, *s.Wq, s.bq, Q);
    project32(op, C, *s.Wk, s.bk, K);
    project32(op, C, *s.Wv, s.bv, V);
    auto heads = [&](const Tensor& src, Tensor* dst, int L) {
        if (!dst) return;
        for (int h = 0; h < H; ++h) copy_d2d_strided(src, h * hd, D, *dst, h * L * hd, hd, hd, L);
    };
    heads(Q, s.Qh, Lq);
    heads(K, s.Kh, Lk);
    heads(V, s.Vh, Lk);

    Tensor yc_tmp, probs_tmp;
    Tensor& Yc = s.Yc ? *s.Yc : (yc_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    Tensor* probs = s.Attnh;
    if (!probs && s.AttnAvg) {
        probs_tmp = Tensor::empty_on(dev, H * Lq, Lk, Dtype::FP32);
        probs = &probs_tmp;
    }

    AttnProblem p;
    p.op = op;
    p.dt = Dtype::FP32;
    p.q = addr(Q.data); p.k = addr(K.data); p.v = addr(V.data); p.o = addr(Yc.data);
    p.mask = addr(s.kmask);
    p.lq = Lq; p.lk = Lk;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = hd; p.hq = p.hkv = H;
    DenseExtras x;
    x.scale = s.scale;
    x.bias = s.bias;
    x.bias_hs = s.bias_hs;
    x.qmask = addr(s.qmask);
    x.mask_ge = true;
    x.probs = probs ? addr(probs->data) : 0;
    dense_attention(d, p, x);

    if (s.AttnAvg) aux(d, AUX_HEAD_MEAN, addr(probs->data), *s.AttnAvg, 0, static_cast<std::uint32_t>(H),
                       static_cast<std::uint32_t>(Lq) * static_cast<std::uint32_t>(Lk));

    // O = Yc Wo^T + bo in FP32 (in place in O for FP32), then the gated cast.
    Tensor o32_tmp, bo32;
    Tensor& O32 = dt == Dtype::FP32 ? *s.O : (o32_tmp = Tensor::empty_on(dev, Lq, D, Dtype::FP32));
    GemmArgs g;
    g.op = op;
    g.a = addr(Yc.data); g.b = addr(s.Wo->data); g.c = addr(O32.data);
    g.da = Dtype::FP32; g.db = dt; g.dc = Dtype::FP32;
    g.round_a = true;   // Yc is ours: a 16-bit Wo takes it at FP16 with a per-row scale (matrix cores)
    if (present(s.bo)) {
        if (s.bo->dtype != dt || s.bo->size() != D) fail(op, "bo must have D elements of X's dtype");
        if (dt == Dtype::FP32) {
            g.bias = addr(s.bo->data);
        } else {
            bo32 = Tensor::empty_on(dev, D, 1, Dtype::FP32);   // a default Tensor would resize on the host
            cast(*s.bo, bo32, Dtype::FP32);
            g.bias = addr(bo32.data);
        }
    }
    g.m = Lq; g.n = D; g.k = D;
    g.lda = D; g.ldb = D; g.ldc = D;
    gemm(d, g);
    if (dt != Dtype::FP32 || s.qmask) {
        aux(d, AUX_ROW_GATE, addr(O32.data), *s.O, addr(s.qmask), static_cast<std::uint32_t>(Lq),
            static_cast<std::uint32_t>(D));
    }
}

}  // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                                 const Tensor& Wo, const Tensor* bq, const Tensor* bk, const Tensor* bv,
                                 const Tensor* bo, const float* d_mask, const Tensor* attn_bias, int num_heads,
                                 float scale, Tensor& O) {
    constexpr const char* op = "self_attention_bias_forward";
    ProjSpec s{op, &X, &X, &Wq, &Wk, &Wv, &Wo, bq, bk, bv, bo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.scale = scale;
    s.O = &O;
    if (present(attn_bias)) {
        const long long L = X.rows;
        if (attn_bias->dtype != Dtype::FP32) fail(op, "attn_bias must be FP32");
        if (num_heads <= 0 || attn_bias->size() != num_heads * L * L) fail(op, "attn_bias must be (num_heads*L, L)");
        s.bias = addr(attn_bias->data);
    }
    proj_attention(s);
}

void cross_attention_forward(const Tensor& X, const Tensor& Ctx, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                             const Tensor& Wo, const float* d_mask, int num_heads, Tensor& O) {
    ProjSpec s{"cross_attention_forward", &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = X.rows == Ctx.rows ? d_mask : nullptr;   // query gating only when Lq == Lk (CPU / CUDA)
    s.H = num_heads;
    s.O = &O;
    proj_attention(s);
}

void cross_attention_forward_with_attn(const Tensor& X, const Tensor& Ctx, const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo, const float* d_mask,
                                       const Tensor* attn_logit_bias, int num_heads, Tensor& O, Tensor& AttnAvg) {
    constexpr const char* op = "cross_attention_forward_with_attn";
    ProjSpec s{op, &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;   // keys only: no query gating
    s.H = num_heads;
    s.O = &O;
    s.AttnAvg = &AttnAvg;
    if (present(attn_logit_bias)) {
        if (attn_logit_bias->dtype != Dtype::FP32) fail(op, "attn_logit_bias must be FP32");
        if (attn_logit_bias->size() != static_cast<long long>(X.rows) * Ctx.rows) {
            fail(op, "attn_logit_bias must be (Lq, Lk)");
        }
        s.bias = addr(attn_logit_bias->data);
        s.bias_hs = 0;   // one bias for every head
    }
    proj_attention(s);
}

void mha_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                 const Tensor* bq, const Tensor* bk, const Tensor* bv, const Tensor* bo, const float* d_mask,
                 int num_heads, Tensor& Qh, Tensor& Kh, Tensor& Vh, Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"mha_forward", &X, &X, &Wq, &Wk, &Wv, &Wo, bq, bk, bv, bo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

// Single-head self-attention with every intermediate kept (brogameagent's
// Attention layer): mha_forward with one head, so Q / K / V are the (N, D)
// projections, Attn (N, N), Y_pre_Wo = Attn V and O, all FP32.
void attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                       const float* d_mask, Tensor& Q, Tensor& K, Tensor& V, Tensor& Attn, Tensor& Y_pre_Wo,
                       Tensor& O) {
    ProjSpec s{"attention_forward", &X, &X, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = 1;
    s.Qh = &Q; s.Kh = &K; s.Vh = &V; s.Attnh = &Attn; s.Yc = &Y_pre_Wo;
    s.O = &O;
    proj_attention(s);
}

// cross_attention_forward with the per-head caches the backward reads.
void cross_attention_forward_train(const Tensor& X, const Tensor& Ctx, const Tensor& Wq, const Tensor& Wk,
                                   const Tensor& Wv, const Tensor& Wo, const float* d_mask, int num_heads, Tensor& Qh,
                                   Tensor& Kh, Tensor& Vh, Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    ProjSpec s{"cross_attention_forward_train", &X, &Ctx, &Wq, &Wk, &Wv, &Wo};
    s.kmask = d_mask;
    s.qmask = X.rows == Ctx.rows ? d_mask : nullptr;
    s.H = num_heads;
    s.Qh = &Qh; s.Kh = &Kh; s.Vh = &Vh; s.Attnh = &Attnh; s.Yc = &Yconcat;
    s.O = &O;
    proj_attention(s);
}

void self_attention_forward_train(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                                  const Tensor& Wo, const float* d_mask, int num_heads, Tensor& Qh, Tensor& Kh,
                                  Tensor& Vh, Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo, nullptr, nullptr, nullptr, nullptr, d_mask, num_heads, Qh, Kh, Vh, Attnh, Yconcat,
                O);
}

void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx, const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk, const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo, const float* d_mask, int num_heads, bool causal,
                                  Tensor& O);   // ops_attention.cpp

// FP32: mha_forward's semantics (the mask gates query rows too). FP16 / BF16:
// the flash route, keys only, as on CUDA / HIP (cross_attention.cu).
void self_attention_forward(const Tensor& X, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask, int num_heads, Tensor& O) {
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        flash_attention_qkvo_forward(X, nullptr, Wq, nullptr, Wk, nullptr, Wv, nullptr, Wo, nullptr, d_mask, num_heads,
                                     false, O);
        return;
    }
    ProjSpec s{"self_attention_forward", &X, &X, &Wq, &Wk, &Wv, &Wo};
    s.kmask = s.qmask = d_mask;
    s.H = num_heads;
    s.O = &O;
    proj_attention(s);
}

// Bias[h*T + q, k] = Qv[q, h] . Pk[(T-1-q) + k, h]: per head the full
// product R = Qv_h Pk_h^T (T x (2T-1), one batched GEMM), then the
// Transformer-XL "rel shift" Bias[q, k] = R[q, T-1-q+k], which is a strided
// copy: R's element q (2T-1) + T-1-q + k = (T-1) + q (2T-2) + k.
void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk, int num_heads, int head_dim, Tensor& Bias) {
    constexpr const char* op = "rel_pos_bias_xl_forward";
    if (Qv.dtype != Dtype::FP32 || Pk.dtype != Dtype::FP32) fail(op, "Qv and Pk must be FP32");
    const int T = Qv.rows, D = Qv.cols;
    if (num_heads <= 0 || head_dim <= 0 || num_heads * head_dim != D) fail(op, "num_heads*head_dim must equal Qv.cols");
    if (Pk.cols != D || Pk.rows != 2 * T - 1) fail(op, "Pk must be (2*Qv.rows - 1, Qv.cols)");
    ensure(Bias, num_heads * T, T, Dtype::FP32, Qv.device);
    if (T == 0) return;
    const int P = 2 * T - 1;
    Tensor R = Tensor::empty_on(Qv.device, num_heads * T, P, Dtype::FP32);
    GemmArgs g;
    g.op = op;
    g.a = addr(Qv.data); g.b = addr(Pk.data); g.c = addr(R.data);
    g.m = T; g.n = P; g.k = head_dim;
    g.lda = D; g.ldb = D; g.ldc = P;
    g.batch = num_heads;
    g.sa = head_dim; g.sb = head_dim; g.sc = static_cast<long long>(T) * P;
    gemm(device_of(Qv), g);
    for (int h = 0; h < num_heads; ++h) {
        copy_d2d_strided(R, h * T * P + (T - 1), P - 1, Bias, h * T * T, T, T, T);
    }
}

void fill_vulkan_vtable_attention_proj(::brotensor::detail::OpsVTable& v) {
    v.self_attention_bias_forward = &self_attention_bias_forward;
    v.cross_attention_forward = &cross_attention_forward;
    v.cross_attention_forward_with_attn = &cross_attention_forward_with_attn;
    v.mha_forward = &mha_forward;
    v.attention_forward = &attention_forward;
    v.cross_attention_forward_train = &cross_attention_forward_train;
    v.self_attention_forward_train = &self_attention_forward_train;
    v.self_attention_forward = &self_attention_forward;
    v.rel_pos_bias_xl_forward = &rel_pos_bias_xl_forward;
}

}  // namespace brotensor::detail::vulkan
