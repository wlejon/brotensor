// Weak stubs for CUDA kernel clusters not yet ported to HIP in Phase 2A.
//
// These weak stubs allow whole-archive linking of partially ported clusters
// without linker errors. When subsequent phases add the real implementations,
// the linker's strong symbol resolution overrides these automatically.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <vector>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;

// ── Modulate cluster (modulate.cu) ──

__attribute__((weak)) void modulate(const Tensor&, const Tensor&,
                                    const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: modulate not yet implemented on HIP");
}

__attribute__((weak)) void broadcast_mul(const Tensor&, const Tensor&,
                                         Tensor&) {
    throw std::runtime_error("brotensor: broadcast_mul not yet implemented on HIP");
}

// ── GroupNorm & ResBlock cluster (group_norm.cu, resblock.cu) ──

__attribute__((weak)) void group_norm_forward(const Tensor&, const Tensor&,
                                              const Tensor&, int, int, int, int,
                                              int, float, Tensor&) {
    throw std::runtime_error("brotensor: group_norm_forward not yet implemented on HIP");
}

__attribute__((weak)) void group_norm_backward(const Tensor&, const Tensor&,
                                               const Tensor&, int, int, int, int,
                                               int, float, Tensor&, Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: group_norm_backward not yet implemented on HIP");
}

__attribute__((weak)) void resblock_forward(const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, const Tensor*, const Tensor*,
                                            const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor*, const Tensor*, const Tensor*,
                                            int, int, int, int, int, int, float,
                                            Tensor&) {
    throw std::runtime_error("brotensor: resblock_forward not yet implemented on HIP");
}

__attribute__((weak)) void resblock_forward_int8w_fp16(const Tensor&, const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, const Tensor*,
                                                       const Tensor*, const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, const Tensor*,
                                                       const Tensor*, const Tensor*, const Tensor*,
                                                       int, int, int, int, int, int, float,
                                                       Tensor&) {
    throw std::runtime_error("brotensor: resblock_forward_int8w_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void resblock_backward(const Tensor&, const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor*, const Tensor*,
                                             const Tensor&, const Tensor&, const Tensor&,
                                             const Tensor*, const Tensor*, const Tensor*,
                                             int, int, int, int, int, int, float,
                                             const Tensor&, Tensor&, Tensor&, Tensor&,
                                             Tensor&, Tensor*, Tensor*, Tensor&, Tensor&,
                                             Tensor&, Tensor*, Tensor*, Tensor*) {
    throw std::runtime_error("brotensor: resblock_backward not yet implemented on HIP");
}

// ── Batched cluster (batched_ops.cu) ──

__attribute__((weak)) void linear_forward_batched(const Tensor&, const Tensor&,
                                                  const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_batched not yet implemented on HIP");
}

__attribute__((weak)) void relu_forward_batched(const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: relu_forward_batched not yet implemented on HIP");
}

__attribute__((weak)) void tanh_forward_batched(const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: tanh_forward_batched not yet implemented on HIP");
}

__attribute__((weak)) void add_inplace_batched(Tensor&, const Tensor&) {
    throw std::runtime_error("brotensor: add_inplace_batched not yet implemented on HIP");
}

__attribute__((weak)) void relu_backward_batched(const Tensor&, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: relu_backward_batched not yet implemented on HIP");
}

__attribute__((weak)) void tanh_backward_batched(const Tensor&, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: tanh_backward_batched not yet implemented on HIP");
}

__attribute__((weak)) void linear_backward_batched(const Tensor&, const Tensor&,
                                                   const Tensor&, Tensor&, Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_backward_batched not yet implemented on HIP");
}

// ── KV-cache & FlashAttention decode (kv_cache.cu) ──

__attribute__((weak)) void kv_cache_append(const Tensor&, const Tensor&,
                                           int, Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: kv_cache_append not yet implemented on HIP");
}

__attribute__((weak)) void flash_attention_decode(const Tensor&, const Tensor&, const Tensor&,
                                                  int, int, int, Tensor&, float, int) {
    throw std::runtime_error("brotensor: flash_attention_decode not yet implemented on HIP");
}

__attribute__((weak)) void flash_attention_decode_masked(const Tensor&, const Tensor&,
                                                         const Tensor&, const float*,
                                                         int, int, Tensor&, float, int) {
    throw std::runtime_error("brotensor: flash_attention_decode_masked not yet implemented on HIP");
}

// ── INT8 weight & Quantized cluster (int8_quant.cu, q4k.cu, q8_0.cu, q6k.cu) ──

__attribute__((weak)) void matmul_int8w_fp16(const Tensor&, const Tensor&, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: matmul_int8w_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_batched_int8w_fp16(const Tensor&, const Tensor&,
                                                             const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_batched_int8w_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void conv2d_int8w_fp16_forward(const Tensor&, const Tensor&,
                                                     const Tensor&, const Tensor*,
                                                     int, int, int, int, int, int, int,
                                                     int, int, int, int, int, int, int, Tensor&) {
    throw std::runtime_error("brotensor: conv2d_int8w_fp16_forward not yet implemented on HIP");
}

__attribute__((weak)) void dequant_q4k_to_fp16(const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: dequant_q4k_to_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_q4k_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_q4k_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_batched_q4k_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void dequant_q8_0_to_fp16(const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: dequant_q8_0_to_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_q8_0_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_q8_0_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_batched_q8_0_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void dequant_q6k_to_fp16(const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: dequant_q6k_to_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_q6k_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_q6k_fp16 not yet implemented on HIP");
}

__attribute__((weak)) void linear_forward_batched_q6k_fp16(const Tensor&, const Tensor*, const Tensor&, Tensor&) {
    throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16 not yet implemented on HIP");
}

// ── Utils cluster (matmul.cu) ──

void sum_rows(const Tensor& X, Tensor& Y);
void sum_cols(const Tensor& X, Tensor& Y);
void argmax_rows(const Tensor& X, Tensor& Idx);
void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts);

__attribute__((weak)) void fill_cuda_vtable_utils(::brotensor::detail::OpsVTable& v) {
    v.sum_rows = &sum_rows;
    v.sum_cols = &sum_cols;
    v.argmax_rows = &argmax_rows;
    v.rows_count_above = &rows_count_above;
}

} // namespace brotensor::detail::cuda
