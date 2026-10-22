// brotensor public op wrappers — core dense, elementwise, norm, losses, embedding, concat, optim.
//
// Spatial/conv/pool/StyleGAN wrappers live in src/ops_spatial.cpp;
// attention/matmul/RoPE/RMSNorm/reductions live in src/ops_attention.cpp;
// quantisation/diffusion/audio/sampling/noise live in src/ops_quant_audio.cpp.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor {

namespace {

[[noreturn]] void throw_empty_parts(const char* op) {
    std::string m = "brotensor: ";
    m += op;
    m += ": parts must be non-empty";
    throw std::runtime_error(m);
}

const detail::OpsVTable& vtable_from_parts(
        const std::vector<const Tensor*>& parts, const char* op) {
    if (parts.empty() || parts[0] == nullptr) throw_empty_parts(op);
    Device d = parts[0]->device;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] && parts[i]->device != d) {
            std::string m = "brotensor: ";
            m += op;
            m += ": parts[";
            m += std::to_string(i);
            m += "] on different device";
            throw std::runtime_error(m);
        }
    }
    return detail::ops_for(d);
}

} // namespace

// ─── Dense layers + elementwise ────────────────────────────────────────────

void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(W, b, x, y);
    if (!v.linear_forward) detail::throw_not_implemented("linear_forward", W.device);
    detail::adopt_output(y, W.device);
    v.linear_forward(W, b, x, y);
}

void linear_backward(const Tensor& W, const Tensor& x, const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB) {
    const auto& v = detail::dispatch(W, x, dY, dX, dW, dB);
    if (!v.linear_backward) detail::throw_not_implemented("linear_backward", W.device);
    detail::adopt_output(dX, W.device);
    detail::adopt_output(dW, W.device);
    detail::adopt_output(dB, W.device);
    v.linear_backward(W, x, dY, dX, dW, dB);
}

void relu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.relu_forward) detail::throw_not_implemented("relu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.relu_forward(x, y);
}

void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.relu_backward) detail::throw_not_implemented("relu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.relu_backward(x, dY, dX);
}

void tanh_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.tanh_forward) detail::throw_not_implemented("tanh_forward", x.device);
    detail::adopt_output(y, x.device);
    v.tanh_forward(x, y);
}

void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(y, dY, dX);
    if (!v.tanh_backward) detail::throw_not_implemented("tanh_backward", y.device);
    detail::adopt_output(dX, y.device);
    v.tanh_backward(y, dY, dX);
}

void sigmoid_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.sigmoid_forward) detail::throw_not_implemented("sigmoid_forward", x.device);
    detail::adopt_output(y, x.device);
    v.sigmoid_forward(x, y);
}

void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(y, dY, dX);
    if (!v.sigmoid_backward) detail::throw_not_implemented("sigmoid_backward", y.device);
    detail::adopt_output(dX, y.device);
    v.sigmoid_backward(y, dY, dX);
}

void add_inplace(Tensor& y, const Tensor& x) {
    const auto& v = detail::dispatch(y, x);
    if (!v.add_inplace) detail::throw_not_implemented("add_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.add_inplace(y, x);
}

void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    const auto& v = detail::dispatch(y, x);
    if (!v.axpby_inplace) detail::throw_not_implemented("axpby_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.axpby_inplace(y, x, a, b);
}

void add_scalar_inplace(Tensor& y, float s) {
    const auto& v = detail::dispatch(y);
    if (!v.add_scalar_inplace) detail::throw_not_implemented("add_scalar_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.add_scalar_inplace(y, s);
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    const auto& v = detail::dispatch(y, bias);
    if (!v.add_channel_bias_inplace)
        detail::throw_not_implemented("add_channel_bias_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.add_channel_bias_inplace(y, bias, C, L);
}

void add_row_bias_inplace(Tensor& Y, const Tensor& bias) {
    const auto& v = detail::dispatch(Y, bias);
    if (!v.add_row_bias_inplace)
        detail::throw_not_implemented("add_row_bias_inplace", Y.device);
    detail::adopt_output(Y, Y.device);
    v.add_row_bias_inplace(Y, bias);
}

void scale_inplace(Tensor& y, float s) {
    const auto& v = detail::dispatch(y);
    if (!v.scale_inplace) detail::throw_not_implemented("scale_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.scale_inplace(y, s);
}

void clamp(Tensor& y, float lo, float hi) {
    if (detail::jit_clamp_inplace_hook && detail::jit_clamp_inplace_hook(y, lo, hi)) {
        return;
    }
    const auto& v = detail::dispatch(y);
    if (!v.clamp) detail::throw_not_implemented("clamp", y.device);
    detail::adopt_output(y, y.device);
    v.clamp(y, lo, hi);
}

void mul_inplace(Tensor& y, const Tensor& x) {
    const auto& v = detail::dispatch(y, x);
    if (!v.mul_inplace) detail::throw_not_implemented("mul_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.mul_inplace(y, x);
}

void div_inplace(Tensor& y, const Tensor& x) {
    const auto& v = detail::dispatch(y, x);
    if (!v.div_inplace) detail::throw_not_implemented("div_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.div_inplace(y, x);
}

void modulate(const Tensor& X, const Tensor& scale, const Tensor& shift,
              Tensor& Y) {
    const auto& v = detail::dispatch(X, scale, shift, Y);
    if (!v.modulate) detail::throw_not_implemented("modulate", X.device);
    detail::adopt_output(Y, X.device);
    v.modulate(X, scale, shift, Y);
}

void broadcast_mul(const Tensor& X, const Tensor& vv, Tensor& Y) {
    const auto& v = detail::dispatch(X, vv, Y);
    if (!v.broadcast_mul) detail::throw_not_implemented("broadcast_mul", X.device);
    detail::adopt_output(Y, X.device);
    v.broadcast_mul(X, vv, Y);
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask) {
    const auto& v = detail::dispatch(x, mask);
    if (!v.build_slot_mask) detail::throw_not_implemented("build_slot_mask", x.device);
    detail::adopt_output(mask, x.device);
    v.build_slot_mask(x, offset, K, stride, mask);
}

// ─── Reductions / norm / softmax / attention (training) ────────────────────

void softmax_forward(const Tensor& logits, Tensor& probs, const float* mask) {
    const auto& v = detail::dispatch(logits, probs);
    if (!v.softmax_forward) detail::throw_not_implemented("softmax_forward", logits.device);
    detail::adopt_output(probs, logits.device);
    v.softmax_forward(logits, probs, mask);
}

void softmax_rows_forward(const Tensor& X, Tensor& Y, int rows, int cols) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.softmax_rows_forward) detail::throw_not_implemented("softmax_rows_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.softmax_rows_forward(X, Y, rows, cols);
}

void softmax_backward(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits) {
    const auto& v = detail::dispatch(probs, dProbs, dLogits);
    if (!v.softmax_backward) detail::throw_not_implemented("softmax_backward", probs.device);
    detail::adopt_output(dLogits, probs.device);
    v.softmax_backward(probs, dProbs, dLogits);
}

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out, float eps) {
    const auto& v = detail::dispatch(x, gamma, beta, y, xhat);
    if (!v.layernorm_forward) detail::throw_not_implemented("layernorm_forward", x.device);
    detail::adopt_output(y, x.device);
    detail::adopt_output(xhat, x.device);
    v.layernorm_forward(x, gamma, beta, y, xhat, mean_out, rstd_out, eps);
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat,
                        const Tensor& gamma, float rstd,
                        Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const auto& v = detail::dispatch(dY, xhat, gamma, dX, dGamma, dBeta);
    if (!v.layernorm_backward) detail::throw_not_implemented("layernorm_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    detail::adopt_output(dGamma, dY.device);
    detail::adopt_output(dBeta, dY.device);
    v.layernorm_backward(dY, xhat, gamma, rstd, dX, dGamma, dBeta);
}

void attention_forward(const Tensor& X,
                       const Tensor& Wq, const Tensor& Wk,
                       const Tensor& Wv, const Tensor& Wo,
                       const float* d_mask,
                       Tensor& Q, Tensor& K, Tensor& V,
                       Tensor& Attn, Tensor& Y_pre_Wo, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {&Wk, &Wv, &Wo, &Q, &K, &V, &Attn, &Y_pre_Wo, &O});
    if (!v.attention_forward) detail::throw_not_implemented("attention_forward", X.device);
    detail::adopt_output(Q, X.device);
    detail::adopt_output(K, X.device);
    detail::adopt_output(V, X.device);
    detail::adopt_output(Attn, X.device);
    detail::adopt_output(Y_pre_Wo, X.device);
    detail::adopt_output(O, X.device);
    v.attention_forward(X, Wq, Wk, Wv, Wo, d_mask, Q, K, V, Attn, Y_pre_Wo, O);
}

void attention_backward(const Tensor& dO, const Tensor& X,
                        const Tensor& Q, const Tensor& K,
                        const Tensor& V, const Tensor& Attn,
                        const Tensor& Y_pre_Wo,
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask,
                        Tensor& dX,
                        Tensor& dWq, Tensor& dWk,
                        Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch_with_opts(
        dO, X, {&Q, &K, &V, &Attn, &Y_pre_Wo, &Wq, &Wk, &Wv, &Wo,
                &dX, &dWq, &dWk, &dWv, &dWo});
    if (!v.attention_backward) detail::throw_not_implemented("attention_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.attention_backward(dO, X, Q, K, V, Attn, Y_pre_Wo,
                         Wq, Wk, Wv, Wo, d_mask,
                         dX, dWq, dWk, dWv, dWo);
}

void mha_forward(const Tensor& X,
                 const Tensor& Wq, const Tensor& Wk,
                 const Tensor& Wv, const Tensor& Wo,
                 const Tensor* bq, const Tensor* bk,
                 const Tensor* bv, const Tensor* bo,
                 const float* d_mask, int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {&Wk, &Wv, &Wo, bq, bk, bv, bo, &Qh, &Kh, &Vh, &Attnh, &Yconcat, &O});
    if (!v.mha_forward) detail::throw_not_implemented("mha_forward", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.mha_forward(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, d_mask, num_heads,
                  Qh, Kh, Vh, Attnh, Yconcat, O);
}

void mha_backward(const Tensor& dO, const Tensor& X,
                  const Tensor& Qh, const Tensor& Kh,
                  const Tensor& Vh, const Tensor& Attnh,
                  const Tensor& Yconcat,
                  const Tensor& Wq, const Tensor& Wk,
                  const Tensor& Wv, const Tensor& Wo,
                  const float* d_mask, int num_heads,
                  Tensor& dX,
                  Tensor& dWq, Tensor& dWk,
                  Tensor& dWv, Tensor& dWo,
                  Tensor* dbq, Tensor* dbk,
                  Tensor* dbv, Tensor* dbo) {
    const auto& v = detail::dispatch_with_opts(
        dO, X, {&Qh, &Kh, &Vh, &Attnh, &Yconcat, &Wq, &Wk, &Wv, &Wo,
                &dX, &dWq, &dWk, &dWv, &dWo, dbq, dbk, dbv, dbo});
    if (!v.mha_backward) detail::throw_not_implemented("mha_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    if (dbq) detail::adopt_output(*dbq, dO.device);
    if (dbk) detail::adopt_output(*dbk, dO.device);
    if (dbv) detail::adopt_output(*dbv, dO.device);
    if (dbo) detail::adopt_output(*dbo, dO.device);
    v.mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                   Wq, Wk, Wv, Wo, d_mask, num_heads,
                   dX, dWq, dWk, dWv, dWo,
                   dbq, dbk, dbv, dbo);
}

// ─── Pooling / losses / embedding / concat ─────────────────────────────────

void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y) {
    const auto& v = detail::dispatch(X, y);
    if (!v.masked_mean_pool_forward) detail::throw_not_implemented("masked_mean_pool_forward", X.device);
    detail::adopt_output(y, X.device);
    v.masked_mean_pool_forward(X, d_mask, y);
}

void masked_mean_pool_backward(const Tensor& dY, const float* d_mask,
                               int K, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.masked_mean_pool_backward) detail::throw_not_implemented("masked_mean_pool_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.masked_mean_pool_backward(dY, d_mask, K, dX);
}

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    const auto& v = detail::dispatch(pred, target);
    if (!v.mse_vec_forward) detail::throw_not_implemented("mse_vec_forward", pred.device);
    return v.mse_vec_forward(pred, target);
}

void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    const auto& v = detail::dispatch(pred, target, dPred);
    if (!v.mse_vec_backward) detail::throw_not_implemented("mse_vec_backward", pred.device);
    detail::adopt_output(dPred, pred.device);
    v.mse_vec_backward(pred, target, dPred);
}

float mse_scalar(float pred, float target, float& dPred) {
    const auto& v = detail::ops_for(Device::CPU);
    if (!v.mse_scalar) detail::throw_not_implemented("mse_scalar", Device::CPU);
    return v.mse_scalar(pred, target, dPred);
}

float softmax_xent(const Tensor& logits, const Tensor& target,
                   Tensor& probs, Tensor& dLogits, const float* mask) {
    const auto& v = detail::dispatch(logits, target, probs, dLogits);
    if (!v.softmax_xent) detail::throw_not_implemented("softmax_xent", logits.device);
    detail::adopt_output(probs, logits.device);
    detail::adopt_output(dLogits, logits.device);
    return v.softmax_xent(logits, target, probs, dLogits, mask);
}

float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits,
                           int n, const float* mask) {
    const auto& v = detail::ops_for(Device::CPU);
    if (!v.softmax_xent_segment) detail::throw_not_implemented("softmax_xent_segment", Device::CPU);
    return v.softmax_xent_segment(logits, target, probs, dLogits, n, mask);
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask,
                         Tensor& probs, Tensor& dLogits) {
    const auto& v = detail::dispatch(logits, target, probs, dLogits);
    if (!v.softmax_xent_fused) detail::throw_not_implemented("softmax_xent_fused", logits.device);
    detail::adopt_output(probs, logits.device);
    detail::adopt_output(dLogits, logits.device);
    return v.softmax_xent_fused(logits, target, d_mask, probs, dLogits);
}

void embedding_lookup_forward(const Tensor& table, const int32_t* d_idx,
                              int B, Tensor& out) {
    const auto& v = detail::dispatch(table, out);
    if (!v.embedding_lookup_forward) detail::throw_not_implemented("embedding_lookup_forward", table.device);
    detail::adopt_output(out, table.device);
    v.embedding_lookup_forward(table, d_idx, B, out);
}

void embedding_lookup_backward(const Tensor& dOut, const int32_t* d_idx,
                               int B, Tensor& dTable) {
    const auto& v = detail::dispatch(dOut, dTable);
    if (!v.embedding_lookup_backward) detail::throw_not_implemented("embedding_lookup_backward", dOut.device);
    detail::adopt_output(dTable, dOut.device);
    v.embedding_lookup_backward(dOut, d_idx, B, dTable);
}

void concat_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    const auto& v = vtable_from_parts(parts, "concat_rows");
    if (out.data != nullptr && parts[0]->device != out.device) {
        throw std::runtime_error("brotensor: concat_rows: out on different device");
    }
    detail::adopt_output(out, parts[0]->device);
    if (!v.concat_rows) detail::throw_not_implemented("concat_rows", parts[0]->device);
    v.concat_rows(parts, out);
}

void split_rows(const Tensor& in, const std::vector<Tensor*>& parts) {
    Device d = in.device;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (parts[i] && parts[i]->data != nullptr && parts[i]->device != d) {
            throw std::runtime_error("brotensor: split_rows: parts on different device");
        }
    }
    const auto& v = detail::ops_for(d);
    if (!v.split_rows) detail::throw_not_implemented("split_rows", d);
    for (Tensor* _p : parts) if (_p) detail::adopt_output(*_p, d);
    v.split_rows(in, parts);
}

void concat_batched_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    const auto& v = vtable_from_parts(parts, "concat_batched_rows");
    if (out.data != nullptr && parts[0]->device != out.device) {
        throw std::runtime_error("brotensor: concat_batched_rows: out on different device");
    }
    detail::adopt_output(out, parts[0]->device);
    if (!v.concat_batched_rows) detail::throw_not_implemented("concat_batched_rows", parts[0]->device);
    v.concat_batched_rows(parts, out);
}

void concat_nchw_channels(const std::vector<const Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          Tensor& out) {
    const auto& v = vtable_from_parts(parts, "concat_nchw_channels");
    if (out.data != nullptr && parts[0]->device != out.device) {
        throw std::runtime_error("brotensor: concat_nchw_channels: out on different device");
    }
    detail::adopt_output(out, parts[0]->device);
    if (!v.concat_nchw_channels) detail::throw_not_implemented("concat_nchw_channels", parts[0]->device);
    v.concat_nchw_channels(parts, N, H, W, C_per_part, out);
}

void concat_nchw_channels_backward(const Tensor& dY, int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts) {
    Device d = dY.device;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (parts[i] && parts[i]->data != nullptr && parts[i]->device != d) {
            throw std::runtime_error("brotensor: concat_nchw_channels_backward: parts on different device");
        }
    }
    const auto& v = detail::ops_for(d);
    if (!v.concat_nchw_channels_backward) detail::throw_not_implemented("concat_nchw_channels_backward", d);
    for (Tensor* _p : parts) if (_p) detail::adopt_output(*_p, d);
    v.concat_nchw_channels_backward(dY, N, H, W, C_per_part, parts);
}

void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n) {
    const auto& v = detail::dispatch(src, dst);
    if (!v.copy_d2d) detail::throw_not_implemented("copy_d2d", src.device);
    detail::adopt_output(dst, src.device);
    v.copy_d2d(src, src_off, dst, dst_off, n);
}

void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height) {
    if (width > src_pitch || width > dst_pitch) {
        throw std::runtime_error("brotensor: copy_d2d_strided: width exceeds pitch");
    }
    const auto& v = detail::dispatch(src, dst);
    if (!v.copy_d2d_strided) detail::throw_not_implemented("copy_d2d_strided", src.device);
    detail::adopt_output(dst, src.device);
    v.copy_d2d_strided(src, src_off, src_pitch, dst, dst_off, dst_pitch, width, height);
}

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (detail::jit_cast_out_hook && detail::jit_cast_out_hook(src, dst, out_dtype)) {
        return;
    }
    const auto& v = detail::dispatch(src, dst);
    if (!v.cast) detail::throw_not_implemented("cast", src.device);
    detail::adopt_output(dst, src.device);
    v.cast(src, dst, out_dtype);
}

// ─── Inference batched + optim ─────────────────────────────────────────────

void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         const Tensor& beta,
                                         Tensor& Y_RD, float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, beta, Y_RD);
    if (!v.layernorm_forward_inference_batched)
        detail::throw_not_implemented("layernorm_forward_inference_batched", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    v.layernorm_forward_inference_batched(X_RD, gamma, &beta, Y_RD, eps);
}

void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         Tensor& Y_RD, float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, Y_RD);
    if (!v.layernorm_forward_inference_batched)
        detail::throw_not_implemented("layernorm_forward_inference_batched", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    v.layernorm_forward_inference_batched(X_RD, gamma, nullptr, Y_RD, eps);
}

void layernorm_forward_batched_with_caches(const Tensor& X_RD,
                                           const Tensor& gamma,
                                           const Tensor& beta,
                                           Tensor& Y_RD, Tensor& Xhat_RD,
                                           Tensor& Mean_R, Tensor& Rstd_R,
                                           float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, beta, Y_RD, Xhat_RD, Mean_R, Rstd_R);
    if (!v.layernorm_forward_batched_with_caches)
        detail::throw_not_implemented("layernorm_forward_batched_with_caches", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    detail::adopt_output(Xhat_RD, X_RD.device);
    detail::adopt_output(Mean_R, X_RD.device);
    detail::adopt_output(Rstd_R, X_RD.device);
    v.layernorm_forward_batched_with_caches(X_RD, gamma, beta, Y_RD, Xhat_RD,
                                            Mean_R, Rstd_R, eps);
}

void layernorm_backward_batched_with_caches(const Tensor& dY_RD,
                                            const Tensor& Xhat_RD,
                                            const Tensor& gamma,
                                            const Tensor& Rstd_R,
                                            Tensor& dX_RD,
                                            Tensor& dGamma, Tensor& dBeta) {
    const auto& v = detail::dispatch(dY_RD, Xhat_RD, gamma, Rstd_R, dX_RD, dGamma, dBeta);
    if (!v.layernorm_backward_batched_with_caches)
        detail::throw_not_implemented("layernorm_backward_batched_with_caches", dY_RD.device);
    detail::adopt_output(dX_RD, dY_RD.device);
    detail::adopt_output(dGamma, dY_RD.device);
    detail::adopt_output(dBeta, dY_RD.device);
    v.layernorm_backward_batched_with_caches(dY_RD, Xhat_RD, gamma, Rstd_R,
                                             dX_RD, dGamma, dBeta);
}

void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity,
              float lr, float momentum) {
    const auto& v = detail::dispatch(param, grad, velocity);
    if (!v.sgd_step) detail::throw_not_implemented("sgd_step", param.device);
    detail::adopt_output(param, param.device);
    detail::adopt_output(grad, param.device);
    detail::adopt_output(velocity, param.device);
    v.sgd_step(param, grad, velocity, lr, momentum);
}

void adam_step(Tensor& param, const Tensor& grad,
               Tensor& m, Tensor& v_buf,
               float lr, float beta1, float beta2, float eps, int step) {
    const auto& v = detail::dispatch(param, grad, m, v_buf);
    if (!v.adam_step) detail::throw_not_implemented("adam_step", param.device);
    detail::adopt_output(param, param.device);
    detail::adopt_output(m, param.device);
    detail::adopt_output(v_buf, param.device);
    v.adam_step(param, grad, m, v_buf, lr, beta1, beta2, eps, step);
}

void xavier_init(Tensor& W, uint64_t& rng_state) {
    const auto& v = detail::dispatch(W);
    if (!v.xavier_init) detail::throw_not_implemented("xavier_init", W.device);
    detail::adopt_output(W, W.device);
    v.xavier_init(W, rng_state);
}

// ─── Batched inference variants ────────────────────────────────────────────

void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch(W, bias, X_BD, Y_BD);
    if (!v.linear_forward_batched) detail::throw_not_implemented("linear_forward_batched", W.device);
    detail::adopt_output(Y_BD, W.device);
    v.linear_forward_batched(W, bias, X_BD, Y_BD);
}

void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch(X_BD, Y_BD);
    if (!v.relu_forward_batched) detail::throw_not_implemented("relu_forward_batched", X_BD.device);
    detail::adopt_output(Y_BD, X_BD.device);
    v.relu_forward_batched(X_BD, Y_BD);
}

void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch(X_BD, Y_BD);
    if (!v.tanh_forward_batched) detail::throw_not_implemented("tanh_forward_batched", X_BD.device);
    detail::adopt_output(Y_BD, X_BD.device);
    v.tanh_forward_batched(X_BD, Y_BD);
}

void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD) {
    const auto& v = detail::dispatch(Y_BD, X_BD);
    if (!v.add_inplace_batched) detail::throw_not_implemented("add_inplace_batched", Y_BD.device);
    detail::adopt_output(Y_BD, Y_BD.device);
    v.add_inplace_batched(Y_BD, X_BD);
}

// ─── Batched backward variants ─────────────────────────────────────────────

void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD, Tensor& dW, Tensor& dB) {
    const auto& v = detail::dispatch(W, X_BD, dY_BD, dX_BD, dW, dB);
    if (!v.linear_backward_batched) detail::throw_not_implemented("linear_backward_batched", W.device);
    detail::adopt_output(dX_BD, W.device);
    detail::adopt_output(dW, W.device);
    detail::adopt_output(dB, W.device);
    v.linear_backward_batched(W, X_BD, dY_BD, dX_BD, dW, dB);
}

void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD, Tensor& dX_BD) {
    const auto& v = detail::dispatch(X_BD, dY_BD, dX_BD);
    if (!v.relu_backward_batched) detail::throw_not_implemented("relu_backward_batched", X_BD.device);
    detail::adopt_output(dX_BD, X_BD.device);
    v.relu_backward_batched(X_BD, dY_BD, dX_BD);
}

void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD, Tensor& dX_BD) {
    const auto& v = detail::dispatch(Y_BD, dY_BD, dX_BD);
    if (!v.tanh_backward_batched) detail::throw_not_implemented("tanh_backward_batched", Y_BD.device);
    detail::adopt_output(dX_BD, Y_BD.device);
    v.tanh_backward_batched(Y_BD, dY_BD, dX_BD);
}

// ─── Batched per-sample losses ─────────────────────────────────────────────

void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample) {
    const auto& v = detail::dispatch(pred, target, dPred, loss_per_sample);
    if (!v.mse_vec_per_sample) detail::throw_not_implemented("mse_vec_per_sample", pred.device);
    detail::adopt_output(dPred, pred.device);
    detail::adopt_output(loss_per_sample, pred.device);
    v.mse_vec_per_sample(pred, target, dPred, loss_per_sample);
}

void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets, int n_heads,
                                Tensor& probs_BL, Tensor& dLogits_BL,
                                Tensor& loss_per_sample) {
    const auto& v = detail::dispatch(logits_BL, target_BL, probs_BL, dLogits_BL, loss_per_sample);
    if (!v.softmax_xent_fused_batched)
        detail::throw_not_implemented("softmax_xent_fused_batched", logits_BL.device);
    detail::adopt_output(probs_BL, logits_BL.device);
    detail::adopt_output(dLogits_BL, logits_BL.device);
    detail::adopt_output(loss_per_sample, logits_BL.device);
    v.softmax_xent_fused_batched(logits_BL, target_BL, d_mask_BL,
                                 d_head_offsets, n_heads,
                                 probs_BL, dLogits_BL, loss_per_sample);
}

void bce_with_logits_fused_batched(const Tensor& logits_BL, const Tensor& target_BL,
                                   const float* d_mask_BL,
                                   float pos_weight,
                                   Tensor& probs_BL, Tensor& dLogits_BL,
                                   Tensor& loss_per_sample) {
    const auto& v = detail::dispatch(logits_BL, target_BL, probs_BL, dLogits_BL, loss_per_sample);
    if (!v.bce_with_logits_fused_batched)
        detail::throw_not_implemented("bce_with_logits_fused_batched", logits_BL.device);
    detail::adopt_output(probs_BL, logits_BL.device);
    detail::adopt_output(dLogits_BL, logits_BL.device);
    detail::adopt_output(loss_per_sample, logits_BL.device);
    v.bce_with_logits_fused_batched(logits_BL, target_BL, d_mask_BL,
                                    pos_weight,
                                    probs_BL, dLogits_BL, loss_per_sample);
}

} // namespace brotensor
