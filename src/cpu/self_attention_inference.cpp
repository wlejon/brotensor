// CPU backend — self_attention_forward (CHUNK 6).
//
// Ground truth: src/cuda/cross_attention.cu — self_attention_forward.
// The CUDA op runs the FP16 flash path for FP16 inputs (delegating to
// flash_attention_qkvo_forward, self-attention, no biases, non-causal) and
// the FP32 mha_forward train-core for FP32 inputs. Either way the underlying
// math is identical to multi-head self-attention.
//
// The CPU backend is FP32-only (per CLAUDE.md). This op delegates straight to
// the existing CPU mha_forward (ops_impl.cpp) — the same FP32 train-core the
// CUDA op uses on its FP32 path — discarding the per-head intermediates that
// the forward-only public surface does not expose.
//
// DTYPE: GPU runs FP16 internally, CPU runs FP32. The parity test quantises
// inputs through FP16 so both backends start from identical values and
// compares with a loose FP16-scale tolerance.
//
// CONVENTIONS (inherited from the CPU mha impl, matching the CUDA mha
// kernels): Wq/Wk/Wv/Wo are all (D, D); per-head split takes contiguous
// weight rows hh*dh..hh*dh+dh; softmax scale 1/sqrt(dh); mask is a length-L
// key-validity buffer.

#include <brotensor/tensor.h>

namespace brotensor::detail::cpu {

// Implemented in ops_impl.cpp.
void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const float* d_mask, int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                 ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                 ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O);

void self_attention_forward(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& Wq,
                            const ::brotensor::Tensor& Wk,
                            const ::brotensor::Tensor& Wv,
                            const ::brotensor::Tensor& Wo,
                            const float* d_mask,
                            int num_heads,
                            ::brotensor::Tensor& O) {
    ::brotensor::Tensor Qh, Kh, Vh, Attnh, Yconcat;
    mha_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}

} // namespace brotensor::detail::cpu
