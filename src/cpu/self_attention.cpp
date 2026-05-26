// CPU backend — self-attention training ops (CHUNK 5).
//
// Ground truth: src/cuda/cross_attention.cu — self_attention_forward_train
// and self_attention_backward are thin wrappers over mha_forward /
// mha_backward (the self-attention degenerate case: Lq == Lk, Ctx == X,
// D_ctx == D). The CPU backend follows the same delegation: these forward
// straight to the existing CPU mha_forward / mha_backward in ops_impl.cpp.
//
// DTYPE: FP32 on both backends — straightforward FP32<->FP32 parity.
//
// CONVENTIONS (inherited from the CPU mha impl, which matches the CUDA mha
// kernels):
//   * Weight layout: Wq/Wk/Wv/Wo are all (D, D); per-head projection takes
//     contiguous weight rows hh*dh..hh*dh+dh.
//   * Per-head split: Qh/Kh/Vh are (H*L, dh) row-major by (head, token);
//     Attnh is (H*L, L); Yconcat is (L, D) with head hh in columns
//     hh*dh..hh*dh+dh; O is (L, D).
//   * Softmax scale 1/sqrt(dh); mask is length-L (self-attention so a single
//     buffer gates both query rows and key columns).
//   * Backward: dWq/dWk/dWv/dWo ACCUMULATE (+=); dX is OVERWRITTEN.

#include <brotensor/tensor.h>

namespace brotensor::detail::cpu {

// Implemented in ops_impl.cpp.
void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                 const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                 const float* d_mask, int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                 ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                 ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O);
void mha_backward(const ::brotensor::Tensor& dO,
                  const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                  const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                  const ::brotensor::Tensor& Yconcat,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                  const float* d_mask, int num_heads,
                  ::brotensor::Tensor& dX,
                  ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo,
                  ::brotensor::Tensor* dbq, ::brotensor::Tensor* dbk,
                  ::brotensor::Tensor* dbv, ::brotensor::Tensor* dbo);

void self_attention_forward_train(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor& Wq,
                                  const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor& Wv,
                                  const ::brotensor::Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  ::brotensor::Tensor& Qh,
                                  ::brotensor::Tensor& Kh,
                                  ::brotensor::Tensor& Vh,
                                  ::brotensor::Tensor& Attnh,
                                  ::brotensor::Tensor& Yconcat,
                                  ::brotensor::Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo,
                nullptr, nullptr, nullptr, nullptr,
                d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_backward(const ::brotensor::Tensor& dO,
                             const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Qh,
                             const ::brotensor::Tensor& Kh,
                             const ::brotensor::Tensor& Vh,
                             const ::brotensor::Tensor& Attnh,
                             const ::brotensor::Tensor& Yconcat,
                             const ::brotensor::Tensor& Wq,
                             const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv,
                             const ::brotensor::Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             ::brotensor::Tensor& dX,
                             ::brotensor::Tensor& dWq,
                             ::brotensor::Tensor& dWk,
                             ::brotensor::Tensor& dWv,
                             ::brotensor::Tensor& dWo) {
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                 Wq, Wk, Wv, Wo, d_mask, num_heads,
                 dX, dWq, dWk, dWv, dWo,
                 nullptr, nullptr, nullptr, nullptr);
}

} // namespace brotensor::detail::cpu
