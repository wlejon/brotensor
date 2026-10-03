// Vulkan INT8-weight (W8A16) attention: the flash family's projection-fused
// ops (flash_attention_project_kv_int8w_fp16, _q_with_kv_cached_int8w_fp16,
// _qkvo_int8w_fp16) and T5's self_attention_bias_int8w_fp16. Contracts follow
// the CUDA backend (src/cuda/flash_attention.cu, self_attention_bias.cu): X,
// K, V and O FP16, weights INT8 (out, in) with (out) FP32 scales, FP16
// biases. Every projection is quant_linear() (ops_quant.cpp: the weight is
// decoded inside the GEMV / GEMM tiles, never expanded), the attention core
// the FP16 path the dense-weight ops take: flash_attention_forward for the
// flash family, the dense path with T5's scale, logit bias and key / query
// mask for self_attention_bias (scores and softmax in FP32, as on CUDA; Q, K,
// V and the per-head outputs are FP16 here, where CUDA keeps them in FP32).

#include "detail/attention.h"
#include "detail/kernels.h"
#include "detail/quant.h"

#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const float* d_mask, int num_heads,
                             bool causal, Tensor& O);   // ops_attention.cpp

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void ensure(Tensor& t, int rows, int cols, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP16) t.resize(rows, cols, Dtype::FP16);
}

// Y (X.rows, W.rows) FP16 = X W^T + b through the quantised linear.
void qproject(const char* op, const Tensor& X, const Tensor& W, const Tensor& s, const Tensor* b, Tensor& Y) {
    need(op, W.dtype == Dtype::INT8, "weights must be INT8");
    need(op, X.dtype == Dtype::FP16, "activations must be FP16");
    const QuantW q = quant_weight(W, &s, op);
    need(op, X.cols == q.k, "weight shape does not match the activations");
    std::uint64_t bias = 0;
    if (b && b->data && b->size() > 0) {
        need(op, b->dtype == Dtype::FP16 && b->size() == q.rows, "a bias must have W.rows FP16 elements");
        bias = addr(b->data);
    }
    if (Y.data == nullptr) Y.device = X.device;
    if (Y.rows != X.rows || Y.cols != q.rows || Y.dtype != Dtype::FP16) Y.resize(X.rows, q.rows, Dtype::FP16);
    if (X.rows == 0) return;
    quant_linear(device_of(X), q, addr(X.data), Dtype::FP16, X.rows, q.k, bias, addr(Y.data), q.rows, op);
}

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx, const Tensor& Wk, const Tensor& sk, const Tensor* bk,
                                           const Tensor& Wv, const Tensor& sv, const Tensor* bv, Tensor& K_out,
                                           Tensor& V_out) {
    constexpr const char* op = "flash_attention_project_kv_int8w_fp16";
    need(op, ctx.dtype == Dtype::FP16, "ctx must be FP16");
    need(op, Wk.cols == ctx.cols && Wv.rows == Wk.rows && Wv.cols == ctx.cols, "Wk/Wv shape mismatch");
    ensure(K_out, ctx.rows, Wk.rows, ctx.device);
    ensure(V_out, ctx.rows, Wk.rows, ctx.device);
    if (ctx.rows == 0 || Wk.rows == 0) return;
    qproject(op, ctx, Wk, sk, bk, K_out);
    qproject(op, ctx, Wv, sv, bv, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X, const Tensor& K, const Tensor& V,
                                                 const Tensor& Wq, const Tensor& sq, const Tensor* bq,
                                                 const Tensor& Wo, const Tensor& so, const Tensor* bo,
                                                 const float* d_mask, int num_heads, bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_q_with_kv_cached_int8w_fp16";
    need(op, X.dtype == Dtype::FP16 && K.dtype == Dtype::FP16 && V.dtype == Dtype::FP16, "X/K/V must be FP16");
    const int D = X.cols;
    need(op, K.cols == D && V.rows == K.rows && V.cols == D, "K/V shape mismatch");
    need(op, Wq.rows == D && Wq.cols == D && Wo.rows == D && Wo.cols == D, "Wq/Wo shape mismatch");
    need(op, num_heads > 0 && D % num_heads == 0, "num_heads must divide D");
    ensure(O, X.rows, D, X.device);
    if (X.rows == 0 || K.rows == 0 || D == 0) return;
    Tensor Q, A = Tensor::empty_on(X.device, X.rows, D, Dtype::FP16);
    qproject(op, X, Wq, sq, bq, Q);
    flash_attention_forward(Q, K, V, d_mask, num_heads, causal, A);
    qproject(op, A, Wo, so, bo, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X, const Tensor* Ctx, const Tensor& Wq, const Tensor& sq,
                                     const Tensor* bq, const Tensor& Wk, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv, const Tensor& sv, const Tensor* bv, const Tensor& Wo,
                                     const Tensor& so, const Tensor* bo, const float* d_mask, int num_heads,
                                     bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_qkvo_int8w_fp16";
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    need(op, !Ctx || Ctx->dtype == Dtype::FP16, "Ctx must be FP16");
    const Tensor& kv = Ctx ? *Ctx : X;
    const int D = X.cols;
    need(op, Wq.rows == D && Wq.cols == D && Wk.rows == D && Wk.cols == kv.cols && Wv.rows == D &&
                 Wv.cols == kv.cols && Wo.rows == D && Wo.cols == D, "shape mismatch");
    need(op, num_heads > 0 && D % num_heads == 0, "num_heads must divide D");
    ensure(O, X.rows, D, X.device);
    if (X.rows == 0 || kv.rows == 0 || D == 0) return;
    Tensor Kp, Vp;
    flash_attention_project_kv_int8w_fp16(kv, Wk, sk, bk, Wv, sv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_int8w_fp16(X, Kp, Vp, Wq, sq, bq, Wo, so, bo, d_mask, num_heads, causal, O);
}

void self_attention_bias_int8w_fp16(const Tensor& X, const Tensor& Wq, const Tensor& sq, const Tensor& Wk,
                                    const Tensor& sk, const Tensor& Wv, const Tensor& sv, const Tensor& Wo,
                                    const Tensor& so, const float* d_mask, const Tensor* attn_bias, int num_heads,
                                    float scale, Tensor& O) {
    constexpr const char* op = "self_attention_bias_int8w_fp16";
    need(op, X.dtype == Dtype::FP16, "X must be FP16");
    const int L = X.rows, D = X.cols, H = num_heads;
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    for (const Tensor* w : {&Wq, &Wk, &Wv, &Wo}) need(op, w->rows == D && w->cols == D, "Wq/Wk/Wv/Wo must be (D, D)");
    for (const Tensor* s : {&sq, &sk, &sv, &so}) {
        need(op, s->dtype == Dtype::FP32 && s->size() == D, "each scale tensor must have D FP32 entries");
    }
    std::uint64_t bias = 0;
    if (attn_bias && attn_bias->data && attn_bias->size() > 0) {
        need(op, attn_bias->dtype == Dtype::FP32, "attn_bias must be FP32");
        need(op, attn_bias->size() == static_cast<long long>(H) * L * L, "attn_bias must be (num_heads*L, L)");
        bias = addr(attn_bias->data);
    }
    ensure(O, L, D, X.device);
    if (L == 0 || D == 0) return;
    Tensor Q, K, V, Yc = Tensor::empty_on(X.device, L, D, Dtype::FP16);
    qproject(op, X, Wq, sq, nullptr, Q);
    qproject(op, X, Wk, sk, nullptr, K);
    qproject(op, X, Wv, sv, nullptr, V);
    AttnProblem p;
    p.op = op;
    p.dt = Dtype::FP16;
    p.q = addr(Q.data); p.k = addr(K.data); p.v = addr(V.data); p.o = addr(Yc.data);
    p.mask = addr(d_mask);
    p.lq = p.lk = L;
    p.ldq = p.ldk = p.ldo = D;
    p.hd = D / H;
    p.hq = p.hkv = H;
    DenseExtras x;
    x.scale = scale;
    x.bias = bias;
    x.qmask = addr(d_mask);   // masked query rows are zero, so their O rows are too (no output bias)
    x.mask_ge = true;
    dense_attention(device_of(X), p, x);
    qproject(op, Yc, Wo, so, nullptr, O);
}

}  // namespace

void fill_vulkan_vtable_quant_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_project_kv_int8w_fp16 = &flash_attention_project_kv_int8w_fp16;
    v.flash_attention_q_with_kv_cached_int8w_fp16 = &flash_attention_q_with_kv_cached_int8w_fp16;
    v.flash_attention_qkvo_int8w_fp16 = &flash_attention_qkvo_int8w_fp16;
    v.self_attention_bias_int8w_fp16 = &self_attention_bias_int8w_fp16;
}

}  // namespace brotensor::detail::vulkan
