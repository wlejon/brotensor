// brotensor public op wrappers. Phase 1A.
//
// One wrapper per public op declared in <brotensor/ops.h>. Each wrapper:
//   1. Validates that all operand tensors share a Device (via detail::dispatch
//      overloads, which throw on mismatch).
//   2. Looks up the function pointer in the resolved OpsVTable.
//   3. If null, throws "not implemented on <device>".
//   4. Otherwise forwards the call.
//
// Special cases:
//   * `mse_scalar`, `softmax_xent_segment` have no Tensor operands. They are
//     host/CPU-only by nature; the wrapper routes directly to the CPU vtable.
//     Backends other than CPU leave these slots null.
//   * Ops with optional `const Tensor*` operands dispatch on the first
//     guaranteed-non-null Tensor (typically the first positional arg).
//   * Ops taking std::vector<const Tensor*> dispatch on parts[0]->device
//     (or, for backward variants, on dY).
//   * `quantize_int8_per_row_host` is a host helper over raw buffers — not in
//     the vtable. Implemented inline here.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cmath>
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

const detail::OpsVTable& vtable_from_parts(
        const std::vector<Tensor*>& parts, const char* op) {
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

#define DISPATCH_REQUIRE(opname, vt)                                         \
    do {                                                                     \
        if (!(vt).opname) detail::throw_not_implemented(#opname, _disp_dev); \
    } while (0)

} // namespace

// Helper to extract device for use after dispatch():
//   const auto& ops = detail::dispatch(...);
//   Device _disp_dev = X.device;   // for DISPATCH_REQUIRE message
//
// We forgo that pattern and inline the throw with a fresh device argument
// where needed.

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

void add_scalar_inplace(Tensor& y, float s) {
    const auto& v = detail::dispatch(y);
    if (!v.add_scalar_inplace) detail::throw_not_implemented("add_scalar_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.add_scalar_inplace(y, s);
}

void scale_inplace(Tensor& y, float s) {
    const auto& v = detail::dispatch(y);
    if (!v.scale_inplace) detail::throw_not_implemented("scale_inplace", y.device);
    detail::adopt_output(y, y.device);
    v.scale_inplace(y, s);
}

void clamp(Tensor& y, float lo, float hi) {
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
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, Q, K, V);
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
    const auto& v = detail::dispatch(dO, X, Q, K, V, Attn, Y_pre_Wo);
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
                 const float* d_mask, int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, Qh, Kh, Vh);
    if (!v.mha_forward) detail::throw_not_implemented("mha_forward", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.mha_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
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
                  Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch(dO, X, Qh, Kh, Vh, Attnh, Yconcat);
    if (!v.mha_backward) detail::throw_not_implemented("mha_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                   Wq, Wk, Wv, Wo, d_mask, num_heads,
                   dX, dWq, dWk, dWv, dWo);
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

// CPU-only host helper. No tensors involved — routes through the CPU vtable.
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

// CPU-only host helper. Raw pointers — CPU vtable directly.
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

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
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
    v.layernorm_forward_inference_batched(X_RD, gamma, beta, Y_RD, eps);
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

// ─── Conv2d ────────────────────────────────────────────────────────────────

void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups, Tensor& Y) {
    const auto& v = detail::dispatch_with_opts(X, Wt, {bias, &Y});
    if (!v.conv2d_forward) detail::throw_not_implemented("conv2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Y);
}

void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, Tensor& dX) {
    const auto& v = detail::dispatch(Wt, dY, dX);
    if (!v.conv2d_backward_input) detail::throw_not_implemented("conv2d_backward_input", Wt.device);
    detail::adopt_output(dX, Wt.device);
    v.conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dX);
}

void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups, Tensor& dWt) {
    const auto& v = detail::dispatch(X, dY, dWt);
    if (!v.conv2d_backward_weight) detail::throw_not_implemented("conv2d_backward_weight", X.device);
    detail::adopt_output(dWt, X.device);
    v.conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                             stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB) {
    const auto& v = detail::dispatch(dY, dB);
    if (!v.conv2d_backward_bias) detail::throw_not_implemented("conv2d_backward_bias", dY.device);
    detail::adopt_output(dB, dY.device);
    v.conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

// ─── GroupNorm ─────────────────────────────────────────────────────────────

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups,
                        float eps, Tensor& Y) {
    const auto& v = detail::dispatch(X, gamma, beta, Y);
    if (!v.group_norm_forward) detail::throw_not_implemented("group_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.group_norm_forward(X, gamma, beta, N, C, H, W, num_groups, eps, Y);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const auto& v = detail::dispatch(X, gamma, dY, dX, dGamma, dBeta);
    if (!v.group_norm_backward) detail::throw_not_implemented("group_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma, X.device);
    detail::adopt_output(dBeta, X.device);
    v.group_norm_backward(X, gamma, dY, N, C, H, W, num_groups, eps,
                          dX, dGamma, dBeta);
}

// ─── Activations ───────────────────────────────────────────────────────────

void silu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.silu_forward) detail::throw_not_implemented("silu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.silu_forward(x, y);
}
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.silu_backward) detail::throw_not_implemented("silu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.silu_backward(x, dY, dX);
}
void gelu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.gelu_forward) detail::throw_not_implemented("gelu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.gelu_forward(x, y);
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.gelu_backward) detail::throw_not_implemented("gelu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.gelu_backward(x, dY, dX);
}
void gelu_exact_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.gelu_exact_forward) detail::throw_not_implemented("gelu_exact_forward", x.device);
    detail::adopt_output(y, x.device);
    v.gelu_exact_forward(x, y);
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.gelu_exact_backward) detail::throw_not_implemented("gelu_exact_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.gelu_exact_backward(x, dY, dX);
}
void quick_gelu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.quick_gelu_forward) detail::throw_not_implemented("quick_gelu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.quick_gelu_forward(x, y);
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.quick_gelu_backward) detail::throw_not_implemented("quick_gelu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.quick_gelu_backward(x, dY, dX);
}

// ─── Resample ──────────────────────────────────────────────────────────────

void upsample_nearest_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.upsample_nearest_2x) detail::throw_not_implemented("upsample_nearest_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.upsample_nearest_2x(X, N, C, H, W, Y);
}
void upsample_bilinear_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.upsample_bilinear_2x) detail::throw_not_implemented("upsample_bilinear_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.upsample_bilinear_2x(X, N, C, H, W, Y);
}
void downsample_avg_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.downsample_avg_2x) detail::throw_not_implemented("downsample_avg_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.downsample_avg_2x(X, N, C, H, W, Y);
}
void upsample_nearest_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.upsample_nearest_2x_backward) detail::throw_not_implemented("upsample_nearest_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.upsample_nearest_2x_backward(dY, N, C, H, W, dX);
}
void upsample_bilinear_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.upsample_bilinear_2x_backward) detail::throw_not_implemented("upsample_bilinear_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.upsample_bilinear_2x_backward(dY, N, C, H, W, dX);
}
void downsample_avg_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.downsample_avg_2x_backward) detail::throw_not_implemented("downsample_avg_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.downsample_avg_2x_backward(dY, N, C, H, W, dX);
}

// ─── FP16 linear + GEGLU ───────────────────────────────────────────────────

void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W, X_BD, {bias, &Y_BD});
    if (!v.linear_forward_batched_fp16) detail::throw_not_implemented("linear_forward_batched_fp16", W.device);
    detail::adopt_output(Y_BD, W.device);
    v.linear_forward_batched_fp16(W, bias, X_BD, Y_BD);
}

void geglu_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.geglu_forward) detail::throw_not_implemented("geglu_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.geglu_forward(X, Y);
}
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.geglu_backward) detail::throw_not_implemented("geglu_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.geglu_backward(X, dY, dX);
}
void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.geglu_exact_forward) detail::throw_not_implemented("geglu_exact_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.geglu_exact_forward(X, Y);
}
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.geglu_exact_backward) detail::throw_not_implemented("geglu_exact_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.geglu_exact_backward(X, dY, dX);
}

// ─── Causal mask helper ────────────────────────────────────────────────────

void build_causal_mask_row(int L, int q, Tensor& mask) {
    const auto& v = detail::dispatch(mask);
    if (!v.build_causal_mask_row) detail::throw_not_implemented("build_causal_mask_row", mask.device);
    detail::adopt_output(mask, mask.device);
    v.build_causal_mask_row(L, q, mask);
}

// ─── Cross-attention family ────────────────────────────────────────────────

void cross_attention_forward(const Tensor& X, const Tensor& Ctx,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads, Tensor& O) {
    const auto& v = detail::dispatch(X, Ctx, Wq, Wk, Wv, Wo, O);
    if (!v.cross_attention_forward) detail::throw_not_implemented("cross_attention_forward", X.device);
    detail::adopt_output(O, X.device);
    v.cross_attention_forward(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads, O);
}

void cross_attention_forward_with_attn(const Tensor& X, const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask,
                                       const Tensor* attn_logit_bias,
                                       int num_heads, Tensor& O, Tensor& AttnAvg) {
    const auto& v = detail::dispatch_with_opts(X, Ctx, {&Wq, &Wk, &Wv, &Wo, attn_logit_bias, &O, &AttnAvg});
    if (!v.cross_attention_forward_with_attn)
        detail::throw_not_implemented("cross_attention_forward_with_attn", X.device);
    detail::adopt_output(O, X.device);
    detail::adopt_output(AttnAvg, X.device);
    v.cross_attention_forward_with_attn(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                        attn_logit_bias, num_heads, O, AttnAvg);
}

void self_attention_forward_train(const Tensor& X,
                                  const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask, int num_heads,
                                  Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, Qh, Kh, Vh);
    if (!v.self_attention_forward_train)
        detail::throw_not_implemented("self_attention_forward_train", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_forward_train(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                   Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_backward(const Tensor& dO, const Tensor& X,
                             const Tensor& Qh, const Tensor& Kh,
                             const Tensor& Vh, const Tensor& Attnh,
                             const Tensor& Yconcat,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask, int num_heads,
                             Tensor& dX,
                             Tensor& dWq, Tensor& dWk,
                             Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch(dO, X, Qh, Kh, Vh, Attnh, Yconcat);
    if (!v.self_attention_backward)
        detail::throw_not_implemented("self_attention_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.self_attention_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                              Wq, Wk, Wv, Wo, d_mask, num_heads,
                              dX, dWq, dWk, dWv, dWo);
}

void attention_token_moments(const Tensor& Attn, int h_lat, int w_lat,
                             Tensor& mass, Tensor& centroid) {
    const auto& v = detail::dispatch(Attn, mass, centroid);
    if (!v.attention_token_moments) detail::throw_not_implemented("attention_token_moments", Attn.device);
    detail::adopt_output(mass, Attn.device);
    detail::adopt_output(centroid, Attn.device);
    v.attention_token_moments(Attn, h_lat, w_lat, mass, centroid);
}

void cross_attention_forward_train(const Tensor& X, const Tensor& Ctx,
                                   const Tensor& Wq, const Tensor& Wk,
                                   const Tensor& Wv, const Tensor& Wo,
                                   const float* d_mask, int num_heads,
                                   Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                   Tensor& Attnh, Tensor& Yconcat, Tensor& O) {
    const auto& v = detail::dispatch(X, Ctx, Wq, Wk, Wv, Wo, Qh, Kh);
    if (!v.cross_attention_forward_train)
        detail::throw_not_implemented("cross_attention_forward_train", X.device);
    detail::adopt_output(Qh, X.device);
    detail::adopt_output(Kh, X.device);
    detail::adopt_output(Vh, X.device);
    detail::adopt_output(Attnh, X.device);
    detail::adopt_output(Yconcat, X.device);
    detail::adopt_output(O, X.device);
    v.cross_attention_forward_train(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                    Qh, Kh, Vh, Attnh, Yconcat, O);
}

void cross_attention_backward(const Tensor& dO, const Tensor& X, const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh,
                              const Tensor& Vh, const Tensor& Attnh,
                              const Tensor& Yconcat,
                              const Tensor& Wq, const Tensor& Wk,
                              const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask, int num_heads,
                              Tensor& dX, Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk,
                              Tensor& dWv, Tensor& dWo) {
    const auto& v = detail::dispatch(dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat);
    if (!v.cross_attention_backward)
        detail::throw_not_implemented("cross_attention_backward", dO.device);
    detail::adopt_output(dX, dO.device);
    detail::adopt_output(dCtx, dO.device);
    detail::adopt_output(dWq, dO.device);
    detail::adopt_output(dWk, dO.device);
    detail::adopt_output(dWv, dO.device);
    detail::adopt_output(dWo, dO.device);
    v.cross_attention_backward(dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat,
                               Wq, Wk, Wv, Wo, d_mask, num_heads,
                               dX, dCtx, dWq, dWk, dWv, dWo);
}

// ─── FP16 LN inference + FP16 self-attention ───────────────────────────────

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD, float eps) {
    const auto& v = detail::dispatch(X_RD, gamma, beta, Y_RD);
    if (!v.layernorm_forward_inference_batched_fp16)
        detail::throw_not_implemented("layernorm_forward_inference_batched_fp16", X_RD.device);
    detail::adopt_output(Y_RD, X_RD.device);
    v.layernorm_forward_inference_batched_fp16(X_RD, gamma, beta, Y_RD, eps);
}

void self_attention_forward(const Tensor& X,
                            const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask, int num_heads, Tensor& O) {
    const auto& v = detail::dispatch(X, Wq, Wk, Wv, Wo, O);
    if (!v.self_attention_forward) detail::throw_not_implemented("self_attention_forward", X.device);
    detail::adopt_output(O, X.device);
    v.self_attention_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads, O);
}

// ─── Flash attention family ────────────────────────────────────────────────

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                             const float* d_mask, int num_heads, bool causal,
                             Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, O);
    if (!v.flash_attention_forward) detail::throw_not_implemented("flash_attention_forward", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_forward(Q, K, V, d_mask, num_heads, causal, O);
}

void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask, int num_heads,
                                  bool causal, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {Ctx, bq, &Wk, bk, &Wv, bv, &Wo, bo, &O});
    if (!v.flash_attention_qkvo_forward)
        detail::throw_not_implemented("flash_attention_qkvo_forward", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_qkvo_forward(X, Ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                                   d_mask, num_heads, causal, O);
}

void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask, int num_heads, bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq, {Ctx, bq, &Wk, bk, &Wv, bv, &Wo, bo, &dO,
                &dX, dCtx, &dWq, dbq, &dWk, dbk, &dWv, dbv, &dWo, dbo});
    if (!v.flash_attention_qkvo_backward)
        detail::throw_not_implemented("flash_attention_qkvo_backward", X.device);
    detail::adopt_output(dX, X.device);
    if (dCtx) detail::adopt_output(*dCtx, X.device);
    detail::adopt_output(dWq, X.device);
    if (dbq) detail::adopt_output(*dbq, X.device);
    detail::adopt_output(dWk, X.device);
    if (dbk) detail::adopt_output(*dbk, X.device);
    detail::adopt_output(dWv, X.device);
    if (dbv) detail::adopt_output(*dbv, X.device);
    detail::adopt_output(dWo, X.device);
    if (dbo) detail::adopt_output(*dbo, X.device);
    v.flash_attention_qkvo_backward(X, Ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                                    d_mask, num_heads, causal, dO,
                                    dX, dCtx, dWq, dbq, dWk, dbk,
                                    dWv, dbv, dWo, dbo);
}

void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& O, const Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              Tensor& dQ, Tensor& dK, Tensor& dV) {
    const auto& v = detail::dispatch(Q, K, V, O, dO, dQ, dK, dV);
    if (!v.flash_attention_backward) detail::throw_not_implemented("flash_attention_backward", Q.device);
    detail::adopt_output(dQ, Q.device);
    detail::adopt_output(dK, Q.device);
    detail::adopt_output(dV, Q.device);
    v.flash_attention_backward(Q, K, V, O, dO, d_mask, num_heads, causal,
                               dQ, dK, dV);
}

void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out, Tensor& V_out) {
    const auto& v = detail::dispatch_with_opts(ctx, Wk, {bk, &Wv, bv, &K_out, &V_out});
    if (!v.flash_attention_project_kv)
        detail::throw_not_implemented("flash_attention_project_kv", ctx.device);
    detail::adopt_output(K_out, ctx.device);
    detail::adopt_output(V_out, ctx.device);
    v.flash_attention_project_kv(ctx, Wk, bk, Wv, bv, K_out, V_out);
}

void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K, const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask, int num_heads,
                                              bool causal, Tensor& O) {
    const auto& v = detail::dispatch_with_opts(X, K, {&V, &Wq, bq, &Wo, bo, &O});
    if (!v.flash_attention_q_with_kv_cached_forward)
        detail::throw_not_implemented("flash_attention_q_with_kv_cached_forward", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_q_with_kv_cached_forward(X, K, V, Wq, bq, Wo, bo,
                                               d_mask, num_heads, causal, O);
}

// ─── NCHW <-> sequence ─────────────────────────────────────────────────────

void nchw_to_sequence(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.nchw_to_sequence) detail::throw_not_implemented("nchw_to_sequence", X.device);
    detail::adopt_output(Y, X.device);
    v.nchw_to_sequence(X, N, C, H, W, Y);
}

void sequence_to_nchw(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.sequence_to_nchw) detail::throw_not_implemented("sequence_to_nchw", X.device);
    detail::adopt_output(Y, X.device);
    v.sequence_to_nchw(X, N, C, H, W, Y);
}

// ─── ResBlock ──────────────────────────────────────────────────────────────

void resblock_forward(const Tensor& X,
                      const Tensor& gamma1, const Tensor& beta1,
                      const Tensor& W1, const Tensor* b1,
                      const Tensor* t_emb_shift,
                      const Tensor& gamma2, const Tensor& beta2,
                      const Tensor& W2, const Tensor* b2,
                      const Tensor* Wskip, const Tensor* bskip,
                      int N, int C_in, int C_out, int H, int W,
                      int num_groups, float eps, Tensor& Y) {
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1, b1, t_emb_shift, &gamma2, &beta2, &W2, b2,
                    Wskip, bskip, &Y});
    if (!v.resblock_forward) detail::throw_not_implemented("resblock_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.resblock_forward(X, gamma1, beta1, W1, b1, t_emb_shift,
                       gamma2, beta2, W2, b2, Wskip, bskip,
                       N, C_in, C_out, H, W, num_groups, eps, Y);
}

void resblock_forward_int8w_fp16(const Tensor& X,
                                 const Tensor& gamma1, const Tensor& beta1,
                                 const Tensor& W1_int8, const Tensor& s1,
                                 const Tensor* b1,
                                 const Tensor* t_emb_shift,
                                 const Tensor& gamma2, const Tensor& beta2,
                                 const Tensor& W2_int8, const Tensor& s2,
                                 const Tensor* b2,
                                 const Tensor* Wskip_int8, const Tensor* sskip,
                                 const Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int W,
                                 int num_groups, float eps, Tensor& Y) {
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1_int8, &s1, b1, t_emb_shift,
                    &gamma2, &beta2, &W2_int8, &s2, b2,
                    Wskip_int8, sskip, bskip, &Y});
    if (!v.resblock_forward_int8w_fp16)
        detail::throw_not_implemented("resblock_forward_int8w_fp16", X.device);
    detail::adopt_output(Y, X.device);
    v.resblock_forward_int8w_fp16(X, gamma1, beta1, W1_int8, s1, b1, t_emb_shift,
                                  gamma2, beta2, W2_int8, s2, b2,
                                  Wskip_int8, sskip, bskip,
                                  N, C_in, C_out, H, W, num_groups, eps, Y);
}

void resblock_backward(const Tensor& X,
                       const Tensor& gamma1, const Tensor& beta1,
                       const Tensor& W1, const Tensor* b1,
                       const Tensor* t_emb_shift,
                       const Tensor& gamma2, const Tensor& beta2,
                       const Tensor& W2, const Tensor* b2,
                       const Tensor* Wskip, const Tensor* bskip,
                       int N, int C_in, int C_out, int H, int W,
                       int num_groups, float eps,
                       const Tensor& dY,
                       Tensor& dX,
                       Tensor& dGamma1, Tensor& dBeta1,
                       Tensor& dW1, Tensor* db1,
                       Tensor* dt_emb_shift,
                       Tensor& dGamma2, Tensor& dBeta2,
                       Tensor& dW2, Tensor* db2,
                       Tensor* dWskip, Tensor* dbskip) {
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1, b1, t_emb_shift, &gamma2, &beta2, &W2, b2,
                    Wskip, bskip, &dY, &dX, &dGamma1, &dBeta1, &dW1, db1,
                    dt_emb_shift, &dGamma2, &dBeta2, &dW2, db2, dWskip, dbskip});
    if (!v.resblock_backward) detail::throw_not_implemented("resblock_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma1, X.device);
    detail::adopt_output(dBeta1, X.device);
    detail::adopt_output(dW1, X.device);
    if (db1) detail::adopt_output(*db1, X.device);
    if (dt_emb_shift) detail::adopt_output(*dt_emb_shift, X.device);
    detail::adopt_output(dGamma2, X.device);
    detail::adopt_output(dBeta2, X.device);
    detail::adopt_output(dW2, X.device);
    if (db2) detail::adopt_output(*db2, X.device);
    if (dWskip) detail::adopt_output(*dWskip, X.device);
    if (dbskip) detail::adopt_output(*dbskip, X.device);
    v.resblock_backward(X, gamma1, beta1, W1, b1, t_emb_shift,
                        gamma2, beta2, W2, b2, Wskip, bskip,
                        N, C_in, C_out, H, W, num_groups, eps,
                        dY, dX, dGamma1, dBeta1, dW1, db1, dt_emb_shift,
                        dGamma2, dBeta2, dW2, db2, dWskip, dbskip);
}

// ─── Matmul + RoPE + RMSNorm + SwiGLU + KV-cache + Llama ───────────────────

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    const auto& v = detail::dispatch(A, B, C);
    if (!v.matmul) detail::throw_not_implemented("matmul", A.device);
    detail::adopt_output(C, A.device);
    v.matmul(A, B, C);
}

void matmul_backward(const Tensor& A, const Tensor& B, const Tensor& dC,
                     Tensor& dA, Tensor& dB) {
    const auto& v = detail::dispatch(A, B, dC, dA, dB);
    if (!v.matmul_backward) detail::throw_not_implemented("matmul_backward", A.device);
    detail::adopt_output(dA, A.device);
    detail::adopt_output(dB, A.device);
    v.matmul_backward(A, B, dC, dA, dB);
}

void rope_forward(const Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.rope_forward) detail::throw_not_implemented("rope_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.rope_forward(X, head_dim, num_heads, seq_offset, theta_base, Y);
}

void rope_backward(const Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.rope_backward) detail::throw_not_implemented("rope_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.rope_backward(dY, head_dim, num_heads, seq_offset, theta_base, dX);
}

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    const auto& v = detail::dispatch(X, gamma, Y);
    if (!v.rms_norm_forward) detail::throw_not_implemented("rms_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.rms_norm_forward(X, gamma, eps, Y);
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma) {
    const auto& v = detail::dispatch(X, gamma, dY, dX, dGamma);
    if (!v.rms_norm_backward) detail::throw_not_implemented("rms_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma, X.device);
    v.rms_norm_backward(X, gamma, dY, eps, dX, dGamma);
}

void swiglu_forward(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.swiglu_forward) detail::throw_not_implemented("swiglu_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.swiglu_forward(X, Y);
}
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.swiglu_backward) detail::throw_not_implemented("swiglu_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.swiglu_backward(X, dY, dX);
}

void kv_cache_append(const Tensor& K_new, const Tensor& V_new, int cur_len,
                     Tensor& K_cache, Tensor& V_cache) {
    const auto& v = detail::dispatch(K_new, V_new, K_cache, V_cache);
    if (!v.kv_cache_append) detail::throw_not_implemented("kv_cache_append", K_new.device);
    detail::adopt_output(K_cache, K_new.device);
    detail::adopt_output(V_cache, K_new.device);
    v.kv_cache_append(K_new, V_new, cur_len, K_cache, V_cache);
}

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_heads, Tensor& O) {
    const auto& v = detail::dispatch(Q, K_cache, V_cache, O);
    if (!v.flash_attention_decode) detail::throw_not_implemented("flash_attention_decode", Q.device);
    detail::adopt_output(O, Q.device);
    v.flash_attention_decode(Q, K_cache, V_cache, valid_len, num_heads, O);
}

// ─── Public reductions ─────────────────────────────────────────────────────

void sum_rows(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.sum_rows) detail::throw_not_implemented("sum_rows", X.device);
    detail::adopt_output(Y, X.device);
    v.sum_rows(X, Y);
}
void sum_cols(const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.sum_cols) detail::throw_not_implemented("sum_cols", X.device);
    detail::adopt_output(Y, X.device);
    v.sum_cols(X, Y);
}
void argmax_rows(const Tensor& X, Tensor& Idx) {
    const auto& v = detail::dispatch(X, Idx);
    if (!v.argmax_rows) detail::throw_not_implemented("argmax_rows", X.device);
    detail::adopt_output(Idx, X.device);
    v.argmax_rows(X, Idx);
}

// ─── Diffusion sampler steps + timestep embedding ──────────────────────────

void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t, Tensor& x_prev) {
    const auto& v = detail::dispatch(x_t, eps_pred, x_prev);
    if (!v.ddim_step) detail::throw_not_implemented("ddim_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    v.ddim_step(x_t, eps_pred, alpha_t, alpha_prev, sigma_t, x_prev);
}

void euler_step(const Tensor& x_t, const Tensor& eps_pred,
                float sigma_t, float sigma_prev, Tensor& x_prev) {
    const auto& v = detail::dispatch(x_t, eps_pred, x_prev);
    if (!v.euler_step) detail::throw_not_implemented("euler_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    v.euler_step(x_t, eps_pred, sigma_t, sigma_prev, x_prev);
}

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred,
                   const Tensor& x0_prev, float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out) {
    const auto& v = detail::dispatch(x_t, eps_pred, x0_prev, x_prev, x0_out);
    if (!v.dpmpp_2m_step) detail::throw_not_implemented("dpmpp_2m_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    detail::adopt_output(x0_out, x_t.device);
    v.dpmpp_2m_step(x_t, eps_pred, x0_prev, sigma_t, c_xt, c_x0t, c_x0prev,
                    x_prev, x0_out);
}

void timestep_embedding(const Tensor& timesteps, int dim, float max_period,
                        Tensor& Y) {
    const auto& v = detail::dispatch(timesteps, Y);
    if (!v.timestep_embedding) detail::throw_not_implemented("timestep_embedding", timesteps.device);
    detail::adopt_output(Y, timesteps.device);
    v.timestep_embedding(timesteps, dim, max_period, Y);
}

// ─── INT8 weight-only quantisation (W8A16) ─────────────────────────────────

// Host helper — pure host buffers, no device dispatch.
void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    for (int r = 0; r < out; ++r) {
        const uint16_t* row = W_fp16 + static_cast<std::size_t>(r) * static_cast<std::size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + static_cast<std::size_t>(r) * static_cast<std::size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    }
}

void matmul_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                       const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(W_int8, scales, X, Y);
    if (!v.matmul_int8w_fp16) detail::throw_not_implemented("matmul_int8w_fp16", W_int8.device);
    detail::adopt_output(Y, W_int8.device);
    v.matmul_int8w_fp16(W_int8, scales, X, Y);
}

void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8, const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    const auto& v = detail::dispatch_with_opts(X, W_int8, {&scales, bias, &Y});
    if (!v.conv2d_int8w_fp16_forward)
        detail::throw_not_implemented("conv2d_int8w_fp16_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv2d_int8w_fp16_forward(X, W_int8, scales, bias,
                                N, C_in, H, W, C_out, kH, kW,
                                stride_h, stride_w, pad_h, pad_w,
                                dil_h, dil_w, groups, Y);
}

void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W_int8, scales, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_int8w_fp16)
        detail::throw_not_implemented("linear_forward_batched_int8w_fp16", W_int8.device);
    detail::adopt_output(Y_BD, W_int8.device);
    v.linear_forward_batched_int8w_fp16(W_int8, scales, bias, X_BD, Y_BD);
}

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8,
                                           const Tensor& sk,
                                           const Tensor* bk,
                                           const Tensor& Wv_int8,
                                           const Tensor& sv,
                                           const Tensor* bv,
                                           Tensor& K_out, Tensor& V_out) {
    const auto& v = detail::dispatch_with_opts(
        ctx, Wk_int8, {&sk, bk, &Wv_int8, &sv, bv, &K_out, &V_out});
    if (!v.flash_attention_project_kv_int8w_fp16)
        detail::throw_not_implemented("flash_attention_project_kv_int8w_fp16", ctx.device);
    detail::adopt_output(K_out, ctx.device);
    detail::adopt_output(V_out, ctx.device);
    v.flash_attention_project_kv_int8w_fp16(ctx, Wk_int8, sk, bk,
                                            Wv_int8, sv, bv, K_out, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                 const Tensor& K, const Tensor& V,
                                                 const Tensor& Wq_int8,
                                                 const Tensor& sq,
                                                 const Tensor* bq,
                                                 const Tensor& Wo_int8,
                                                 const Tensor& so,
                                                 const Tensor* bo,
                                                 const float* d_mask,
                                                 int num_heads, bool causal,
                                                 Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, K, {&V, &Wq_int8, &sq, bq, &Wo_int8, &so, bo, &O});
    if (!v.flash_attention_q_with_kv_cached_int8w_fp16)
        detail::throw_not_implemented("flash_attention_q_with_kv_cached_int8w_fp16", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_q_with_kv_cached_int8w_fp16(X, K, V, Wq_int8, sq, bq,
                                                  Wo_int8, so, bo,
                                                  d_mask, num_heads, causal, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X, const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask, int num_heads, bool causal,
                                     Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq_int8, {Ctx, &sq, bq, &Wk_int8, &sk, bk, &Wv_int8, &sv, bv,
                     &Wo_int8, &so, bo, &O});
    if (!v.flash_attention_qkvo_int8w_fp16)
        detail::throw_not_implemented("flash_attention_qkvo_int8w_fp16", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_qkvo_int8w_fp16(X, Ctx, Wq_int8, sq, bq,
                                      Wk_int8, sk, bk, Wv_int8, sv, bv,
                                      Wo_int8, so, bo,
                                      d_mask, num_heads, causal, O);
}

} // namespace brotensor
