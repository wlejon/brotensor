#include <brotensor/ops.h>

namespace brotensor {

// cross_attention and self_attention delegate to the flash-attention kernel.
// An earlier hand-rolled core kernel in this file produced incorrect outputs
// at large block counts on this architecture (the failure threshold lived at
// (Lq * num_heads) ≳ 400 with head_dim ≥ 64); the flash kernel's tiled
// online-softmax path is numerically robust at every shape exercised by
// the U-Net / cross-attn pipeline, so cross_attention_forward_gpu now stays
// in the public API as a thin alias.

void cross_attention_forward_gpu(const GpuTensor& X,
                                 const GpuTensor& Ctx,
                                 const GpuTensor& Wq, const GpuTensor& Wk,
                                 const GpuTensor& Wv, const GpuTensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 GpuTensor& O) {
    flash_attention_qkvo_forward_gpu(X, &Ctx, Wq, Wk, Wv, Wo,
                                     d_mask, num_heads, O);
}

void self_attention_forward_gpu(const GpuTensor& X,
                                const GpuTensor& Wq, const GpuTensor& Wk,
                                const GpuTensor& Wv, const GpuTensor& Wo,
                                const float* d_mask,
                                int num_heads,
                                GpuTensor& O) {
    flash_attention_qkvo_forward_gpu(X, nullptr, Wq, Wk, Wv, Wo,
                                     d_mask, num_heads, O);
}

} // namespace brotensor
