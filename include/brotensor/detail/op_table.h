#pragma once

// ─── brotensor op table ────────────────────────────────────────────────────
//
// X-macro list of every public op in brotensor. Each row carries:
//   - the public op name (the function in `<brotensor/ops.h>`)
//   - the return type
//   - the parameter list (parenthesised, exactly as it appears in ops.h)
//
// This single list is expanded in:
//   * detail/dispatch.h        — into the `OpsVTable` struct (one fn-pointer
//                                per op).
//   * src/ops.cpp              — into thin dispatcher wrappers (one per op).
//   * src/cpu/register.cpp     — into the CPU backend's vtable initialiser.
//   * src/cuda/register_*.cu   — likewise for CUDA per-cluster registration.
//   * src/metal/register_*.mm  — likewise for Metal.
//
// Adding a new op = adding one row here and implementing it in every backend
// (or providing a null slot if the backend doesn't support it — the
// dispatcher will throw on null lookups).
//
// Forward declarations of `brotensor::Tensor` and `brotensor::Dtype` (etc.)
// are required at the point of expansion; including <brotensor/tensor.h>
// before this file is sufficient. <vector> is also required for the
// std::vector-of-pointer ops (concat/split family).

// Parenthesise the params so commas inside the param list don't break the
// macro expansion. Each consumer of BROTENSOR_FOR_EACH_OP defines its own
// X(name, ret, params) macro then includes this file or invokes the macro.

#define BROTENSOR_FOR_EACH_OP(X)                                                                                                                                       \
    /* ─── Dense layers + elementwise activations ─── */                                                                                                               \
    X(linear_forward,                          void,  (const ::brotensor::Tensor& W, const ::brotensor::Tensor& b, const ::brotensor::Tensor& x, ::brotensor::Tensor& y)) \
    X(linear_backward,                         void,  (const ::brotensor::Tensor& W, const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,                       \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dW, ::brotensor::Tensor& dB))                                      \
    X(relu_forward,                            void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(relu_backward,                           void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(tanh_forward,                            void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(tanh_backward,                           void,  (const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(sigmoid_forward,                         void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(sigmoid_backward,                        void,  (const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(add_inplace,                             void,  (::brotensor::Tensor& y, const ::brotensor::Tensor& x))                                                           \
    X(add_scalar_inplace,                      void,  (::brotensor::Tensor& y, float s))                                                                                \
    X(scale_inplace,                           void,  (::brotensor::Tensor& y, float s))                                                                                \
    X(clamp,                                   void,  (::brotensor::Tensor& y, float lo, float hi))                                                                     \
    X(mul_inplace,                             void,  (::brotensor::Tensor& y, const ::brotensor::Tensor& x))                                                           \
    X(build_slot_mask,                         void,  (const ::brotensor::Tensor& x, int offset, int K, int stride, ::brotensor::Tensor& mask))                         \
    /* ─── Reductions / norm / softmax / attention (training) ─── */                                                                                                    \
    X(softmax_forward,                         void,  (const ::brotensor::Tensor& logits, ::brotensor::Tensor& probs, const float* mask))                               \
    X(softmax_backward,                        void,  (const ::brotensor::Tensor& probs, const ::brotensor::Tensor& dProbs, ::brotensor::Tensor& dLogits))              \
    X(layernorm_forward,                       void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,                 \
                                                       ::brotensor::Tensor& y, ::brotensor::Tensor& xhat, float& mean_out, float& rstd_out, float eps))                 \
    X(layernorm_backward,                      void,  (const ::brotensor::Tensor& dY, const ::brotensor::Tensor& xhat, const ::brotensor::Tensor& gamma,                \
                                                       float rstd, ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta))                   \
    X(attention_forward,                       void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                      \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask,                               \
                                                       ::brotensor::Tensor& Q, ::brotensor::Tensor& K, ::brotensor::Tensor& V,                                          \
                                                       ::brotensor::Tensor& Attn, ::brotensor::Tensor& Y_pre_Wo, ::brotensor::Tensor& O))                               \
    X(attention_backward,                      void,  (const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X, const ::brotensor::Tensor& Q,                       \
                                                       const ::brotensor::Tensor& K, const ::brotensor::Tensor& V, const ::brotensor::Tensor& Attn,                     \
                                                       const ::brotensor::Tensor& Y_pre_Wo, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,               \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask,                               \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,                                     \
                                                       ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo))                                                             \
    X(mha_forward,                             void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                      \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask, int num_heads,                \
                                                       ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,                                       \
                                                       ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O))                               \
    X(mha_backward,                            void,  (const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X, const ::brotensor::Tensor& Qh,                      \
                                                       const ::brotensor::Tensor& Kh, const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,                  \
                                                       const ::brotensor::Tensor& Yconcat, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask, int num_heads,                \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,                                     \
                                                       ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo))                                                             \
    /* ─── Pooling / losses / embeddings / concat ─── */                                                                                                                \
    X(masked_mean_pool_forward,                void,  (const ::brotensor::Tensor& X, const float* d_mask, ::brotensor::Tensor& y))                                      \
    X(masked_mean_pool_backward,               void,  (const ::brotensor::Tensor& dY, const float* d_mask, int K, ::brotensor::Tensor& dX))                             \
    X(mse_vec_forward,                         float, (const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target))                                             \
    X(mse_vec_backward,                        void,  (const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target, ::brotensor::Tensor& dPred))                 \
    X(mse_scalar,                              float, (float pred, float target, float& dPred))                                                                         \
    X(softmax_xent,                            float, (const ::brotensor::Tensor& logits, const ::brotensor::Tensor& target,                                            \
                                                       ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits, const float* mask))                                    \
    X(softmax_xent_segment,                    float, (const float* logits, const float* target, float* probs, float* dLogits, int n, const float* mask))               \
    X(softmax_xent_fused,                      float, (const ::brotensor::Tensor& logits, const ::brotensor::Tensor& target, const float* d_mask,                       \
                                                       ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits))                                                       \
    X(embedding_lookup_forward,                void,  (const ::brotensor::Tensor& table, const int32_t* d_idx, int B, ::brotensor::Tensor& out))                        \
    X(embedding_lookup_backward,               void,  (const ::brotensor::Tensor& dOut, const int32_t* d_idx, int B, ::brotensor::Tensor& dTable))                      \
    X(concat_rows,                             void,  (const std::vector<const ::brotensor::Tensor*>& parts, ::brotensor::Tensor& out))                                 \
    X(split_rows,                              void,  (const ::brotensor::Tensor& in, const std::vector<::brotensor::Tensor*>& parts))                                  \
    X(concat_batched_rows,                     void,  (const std::vector<const ::brotensor::Tensor*>& parts, ::brotensor::Tensor& out))                                 \
    X(concat_nchw_channels,                    void,  (const std::vector<const ::brotensor::Tensor*>& parts, int N, int H, int W,                                       \
                                                       const std::vector<int>& C_per_part, ::brotensor::Tensor& out))                                                   \
    X(concat_nchw_channels_backward,           void,  (const ::brotensor::Tensor& dY, int N, int H, int W,                                                              \
                                                       const std::vector<int>& C_per_part, const std::vector<::brotensor::Tensor*>& parts))                             \
    X(copy_d2d,                                void,  (const ::brotensor::Tensor& src, int src_off, ::brotensor::Tensor& dst, int dst_off, int n))                      \
    X(cast,                                    void,  (const ::brotensor::Tensor& src, ::brotensor::Tensor& dst, ::brotensor::Dtype out_dtype))                        \
    /* ─── Inference batched + optim ─── */                                                                                                                             \
    X(layernorm_forward_inference_batched,     void,  (const ::brotensor::Tensor& X_RD, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,              \
                                                       ::brotensor::Tensor& Y_RD, float eps))                                                                           \
    X(sgd_step,                                void,  (::brotensor::Tensor& param, ::brotensor::Tensor& grad, ::brotensor::Tensor& velocity, float lr, float momentum)) \
    X(adam_step,                               void,  (::brotensor::Tensor& param, const ::brotensor::Tensor& grad, ::brotensor::Tensor& m, ::brotensor::Tensor& v,     \
                                                       float lr, float beta1, float beta2, float eps, int step))                                                        \
    X(xavier_init,                             void,  (::brotensor::Tensor& W, uint64_t& rng_state))                                                                    \
    /* ─── Batched (inference-only) variants ─── */                                                                                                                     \
    X(linear_forward_batched,                  void,  (const ::brotensor::Tensor& W, const ::brotensor::Tensor& bias, const ::brotensor::Tensor& X_BD,                  \
                                                       ::brotensor::Tensor& Y_BD))                                                                                      \
    X(relu_forward_batched,                    void,  (const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD))                                                     \
    X(tanh_forward_batched,                    void,  (const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD))                                                     \
    X(add_inplace_batched,                     void,  (::brotensor::Tensor& Y_BD, const ::brotensor::Tensor& X_BD))                                                     \
    /* ─── Batched (training) backward variants ─── */                                                                                                                  \
    X(linear_backward_batched,                 void,  (const ::brotensor::Tensor& W, const ::brotensor::Tensor& X_BD, const ::brotensor::Tensor& dY_BD,                 \
                                                       ::brotensor::Tensor& dX_BD, ::brotensor::Tensor& dW, ::brotensor::Tensor& dB))                                   \
    X(relu_backward_batched,                   void,  (const ::brotensor::Tensor& X_BD, const ::brotensor::Tensor& dY_BD, ::brotensor::Tensor& dX_BD))                  \
    X(tanh_backward_batched,                   void,  (const ::brotensor::Tensor& Y_BD, const ::brotensor::Tensor& dY_BD, ::brotensor::Tensor& dX_BD))                  \
    /* ─── Batched per-sample loss kernels ─── */                                                                                                                       \
    X(mse_vec_per_sample,                      void,  (const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target,                                              \
                                                       ::brotensor::Tensor& dPred, ::brotensor::Tensor& loss_per_sample))                                               \
    X(softmax_xent_fused_batched,              void,  (const ::brotensor::Tensor& logits_BL, const ::brotensor::Tensor& target_BL, const float* d_mask_BL,              \
                                                       const int* d_head_offsets, int n_heads, ::brotensor::Tensor& probs_BL, ::brotensor::Tensor& dLogits_BL,          \
                                                       ::brotensor::Tensor& loss_per_sample))                                                                           \
    /* ─── Conv2d (forward + backwards) ─── */                                                                                                                          \
    X(conv2d_forward,                          void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt, const ::brotensor::Tensor* bias,                    \
                                                       int N, int C_in, int H, int W, int C_out, int kH, int kW,                                                        \
                                                       int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,                              \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(conv2d_backward_input,                   void,  (const ::brotensor::Tensor& Wt, const ::brotensor::Tensor& dY,                                                    \
                                                       int N, int C_in, int H, int W, int C_out, int kH, int kW,                                                        \
                                                       int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,                              \
                                                       ::brotensor::Tensor& dX))                                                                                        \
    X(conv2d_backward_weight,                  void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,                                                     \
                                                       int N, int C_in, int H, int W, int C_out, int kH, int kW,                                                        \
                                                       int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,                              \
                                                       ::brotensor::Tensor& dWt))                                                                                       \
    X(conv2d_backward_bias,                    void,  (const ::brotensor::Tensor& dY, int N, int C_out, int H_out, int W_out, ::brotensor::Tensor& dB))                 \
    /* ─── Conv3d (forward + W8A16 variant for Qwen3-VL patch embed) ─── */                                                                                              \
    X(conv3d_forward,                          void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt, const ::brotensor::Tensor* bias,                    \
                                                       int N, int C_in, int T, int H, int W, int C_out, int kT, int kH, int kW,                                          \
                                                       int stride_t, int stride_h, int stride_w, int pad_t, int pad_h, int pad_w,                                        \
                                                       int dil_t, int dil_h, int dil_w, int groups,                                                                      \
                                                       ::brotensor::Tensor& Y))                                                                                          \
    X(conv3d_int8w_fp16_forward,               void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,              \
                                                       const ::brotensor::Tensor* bias,                                                                                  \
                                                       int N, int C_in, int T, int H, int W, int C_out, int kT, int kH, int kW,                                          \
                                                       int stride_t, int stride_h, int stride_w, int pad_t, int pad_h, int pad_w,                                        \
                                                       int dil_t, int dil_h, int dil_w, int groups,                                                                      \
                                                       ::brotensor::Tensor& Y))                                                                                          \
    /* ─── GroupNorm ─── */                                                                                                                                             \
    X(group_norm_forward,                      void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,                 \
                                                       int N, int C, int H, int W, int num_groups, float eps, ::brotensor::Tensor& Y))                                  \
    X(group_norm_backward,                     void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& dY,                   \
                                                       int N, int C, int H, int W, int num_groups, float eps,                                                           \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta))                               \
    /* ─── Activations: silu, gelu (tanh-approx + exact), quick_gelu ─── */                                                                                             \
    X(silu_forward,                            void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(silu_backward,                           void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(gelu_forward,                            void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(gelu_backward,                           void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(gelu_exact_forward,                      void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(gelu_exact_backward,                     void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(quick_gelu_forward,                      void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                           \
    X(quick_gelu_backward,                     void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    /* ─── Resample (NN / bilinear / avgpool) ─── */                                                                                                                    \
    X(upsample_nearest_2x,                     void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    X(upsample_bilinear_2x,                    void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    X(downsample_avg_2x,                       void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    X(upsample_nearest_2x_backward,            void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX))                             \
    X(upsample_bilinear_2x_backward,           void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX))                             \
    X(downsample_avg_2x_backward,              void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX))                             \
    /* ─── Arbitrary-scale 2D resample (nearest / bilinear / bicubic) ─── */                                                                                            \
    X(interp2d_forward,                        void,  (const ::brotensor::Tensor& X, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode,                 \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(interp2d_backward,                       void,  (const ::brotensor::Tensor& dY, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode,                \
                                                       ::brotensor::Tensor& dX))                                                                                        \
    /* ─── 2D padding (zero / reflect / replicate) — NCHW ─── */                                                                                                        \
    X(pad2d_forward,                           void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W,                                                       \
                                                       int pad_top, int pad_bottom, int pad_left, int pad_right, int mode,                                             \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(pad2d_backward,                          void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W,                                                      \
                                                       int pad_top, int pad_bottom, int pad_left, int pad_right, int mode,                                             \
                                                       ::brotensor::Tensor& dX))                                                                                        \
    /* ─── 2D spatial slice / crop on NCHW ─── */                                                                                                                       \
    X(slice2d_forward,                         void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W,                                                       \
                                                       int h0, int w0, int H_out, int W_out, ::brotensor::Tensor& Y))                                                  \
    X(slice2d_backward,                        void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W,                                                      \
                                                       int h0, int w0, int H_out, int W_out, ::brotensor::Tensor& dX))                                                 \
    /* ─── Per-row top-k (descending values + int32 indices) ─── */                                                                                                     \
    X(top_k_rows,                              void,  (const ::brotensor::Tensor& X, int k, ::brotensor::Tensor& Vals, ::brotensor::Tensor& Idx))                       \
    /* ─── Adaptive avg pool 2D (NCHW), arbitrary output spatial size ─── */                                                                                            \
    X(adaptive_avg_pool2d_forward,             void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, int H_out, int W_out,                                 \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(adaptive_avg_pool2d_backward,            void,  (const ::brotensor::Tensor& dY, int N, int C, int H, int W, int H_out, int W_out,                                \
                                                       ::brotensor::Tensor& dX))                                                                                        \
    /* ─── Max pool 2D (NCHW): forward returns Y + int32 flat-spatial Idx ─── */                                                                                        \
    X(max_pool2d_forward,                      void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W,                                                       \
                                                       int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w,                                                \
                                                       ::brotensor::Tensor& Y, ::brotensor::Tensor& Idx))                                                               \
    X(max_pool2d_backward,                     void,  (const ::brotensor::Tensor& dY, const ::brotensor::Tensor& Idx,                                                  \
                                                       int N, int C, int H, int W, int H_out, int W_out, ::brotensor::Tensor& dX))                                     \
    /* ─── Row gather / scatter-add (general 2D — superset of embedding_lookup) ─── */                                                                                  \
    X(gather_rows,                             void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Idx, ::brotensor::Tensor& Y))                          \
    X(scatter_rows_add,                        void,  (const ::brotensor::Tensor& dY, const ::brotensor::Tensor& Idx, int R, ::brotensor::Tensor& dX))                 \
    /* ─── FP16 linear (inference-only) + GEGLU family ─── */                                                                                                           \
    X(linear_forward_batched_fp16,             void,  (const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X_BD,                  \
                                                       ::brotensor::Tensor& Y_BD))                                                                                      \
    X(geglu_forward,                           void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(geglu_backward,                          void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(geglu_exact_forward,                     void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(geglu_exact_backward,                    void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    /* ─── Causal mask helper ─── */                                                                                                                                    \
    X(build_causal_mask_row,                   void,  (int L, int q, ::brotensor::Tensor& mask))                                                                        \
    /* ─── Cross-attention family (FP16 inference + FP32 train) ─── */                                                                                                  \
    X(cross_attention_forward,                 void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,                                                    \
                                                       const float* d_mask, int num_heads, ::brotensor::Tensor& O))                                                     \
    X(cross_attention_forward_with_attn,       void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,                                                    \
                                                       const float* d_mask, const ::brotensor::Tensor* attn_logit_bias,                                                 \
                                                       int num_heads, ::brotensor::Tensor& O, ::brotensor::Tensor& AttnAvg))                                            \
    X(self_attention_forward_train,            void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                      \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask, int num_heads,                \
                                                       ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,                                       \
                                                       ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O))                               \
    X(self_attention_backward,                 void,  (const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X, const ::brotensor::Tensor& Qh,                      \
                                                       const ::brotensor::Tensor& Kh, const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,                  \
                                                       const ::brotensor::Tensor& Yconcat, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask, int num_heads,                \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,                                     \
                                                       ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo))                                                             \
    X(attention_token_moments,                 void,  (const ::brotensor::Tensor& Attn, int h_lat, int w_lat,                                                           \
                                                       ::brotensor::Tensor& mass, ::brotensor::Tensor& centroid))                                                       \
    X(cross_attention_forward_train,           void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,                                                    \
                                                       const float* d_mask, int num_heads,                                                                              \
                                                       ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,                                       \
                                                       ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O))                               \
    X(cross_attention_backward,                void,  (const ::brotensor::Tensor& dO, const ::brotensor::Tensor& X, const ::brotensor::Tensor& Ctx,                     \
                                                       const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh, const ::brotensor::Tensor& Vh,                     \
                                                       const ::brotensor::Tensor& Attnh, const ::brotensor::Tensor& Yconcat,                                            \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,                                                    \
                                                       const float* d_mask, int num_heads,                                                                              \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dCtx,                                                              \
                                                       ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,                                                              \
                                                       ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo))                                                             \
    /* ─── FP16 LayerNorm inference + FP16 self-attention ─── */                                                                                                        \
    X(layernorm_forward_inference_batched_fp16, void, (const ::brotensor::Tensor& X_RD, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,              \
                                                       ::brotensor::Tensor& Y_RD, float eps))                                                                           \
    X(self_attention_forward,                  void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                      \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask, int num_heads,                \
                                                       ::brotensor::Tensor& O))                                                                                         \
    /* ─── Flash attention family ─── */                                                                                                                                \
    X(flash_attention_forward,                 void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,                        \
                                                       const float* d_mask, int num_heads, bool causal, ::brotensor::Tensor& O))                                        \
    X(flash_attention_varlen_forward,          void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,                        \
                                                       const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,                                                        \
                                                       int batch_size, int max_seqlen_q, int max_seqlen_k,                                                              \
                                                       int num_heads, int head_dim, bool causal,                                                                        \
                                                       ::brotensor::Tensor& O))                                                                                         \
    X(flash_attention_qkvo_forward,            void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor* Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,                                                    \
                                                       const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,                                                    \
                                                       const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,                                                    \
                                                       const float* d_mask, int num_heads, bool causal, ::brotensor::Tensor& O))                                        \
    X(flash_attention_qkvo_backward,           void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor* Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,                                                    \
                                                       const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,                                                    \
                                                       const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,                                                    \
                                                       const float* d_mask, int num_heads, bool causal,                                                                 \
                                                       const ::brotensor::Tensor& dO, ::brotensor::Tensor& dX, ::brotensor::Tensor* dCtx,                               \
                                                       ::brotensor::Tensor& dWq, ::brotensor::Tensor* dbq,                                                              \
                                                       ::brotensor::Tensor& dWk, ::brotensor::Tensor* dbk,                                                              \
                                                       ::brotensor::Tensor& dWv, ::brotensor::Tensor* dbv,                                                              \
                                                       ::brotensor::Tensor& dWo, ::brotensor::Tensor* dbo))                                                             \
    X(flash_attention_backward,                void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,                        \
                                                       const ::brotensor::Tensor& O, const ::brotensor::Tensor& dO,                                                     \
                                                       const float* d_mask, int num_heads, bool causal,                                                                 \
                                                       ::brotensor::Tensor& dQ, ::brotensor::Tensor& dK, ::brotensor::Tensor& dV))                                      \
    X(flash_attention_project_kv,              void,  (const ::brotensor::Tensor& ctx,                                                                                  \
                                                       const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,                                                    \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,                                                    \
                                                       ::brotensor::Tensor& K_out, ::brotensor::Tensor& V_out))                                                         \
    X(flash_attention_q_with_kv_cached_forward, void, (const ::brotensor::Tensor& X, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,                        \
                                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,                                                    \
                                                       const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,                                                    \
                                                       const float* d_mask, int num_heads, bool causal, ::brotensor::Tensor& O))                                        \
    /* ─── NCHW <-> sequence transposes ─── */                                                                                                                          \
    X(nchw_to_sequence,                        void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    X(sequence_to_nchw,                        void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    /* ─── Qwen3-VL patch merger: (N,C,H,W) -> (N,4C,H/2,W/2) ─── */                                                                                                    \
    X(spatial_merge_2x2_forward,               void,  (const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y))                               \
    /* ─── Diffusion ResBlock (forward + W8A16 + backward) ─── */                                                                                                       \
    X(resblock_forward,                        void,  (const ::brotensor::Tensor& X,                                                                                    \
                                                       const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,                                             \
                                                       const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,                                                    \
                                                       const ::brotensor::Tensor* t_emb_shift,                                                                          \
                                                       const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,                                             \
                                                       const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,                                                    \
                                                       const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,                                              \
                                                       int N, int C_in, int C_out, int H, int W, int num_groups, float eps, ::brotensor::Tensor& Y))                    \
    X(resblock_forward_int8w_fp16,             void,  (const ::brotensor::Tensor& X,                                                                                    \
                                                       const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,                                             \
                                                       const ::brotensor::Tensor& W1_int8, const ::brotensor::Tensor& s1, const ::brotensor::Tensor* b1,                \
                                                       const ::brotensor::Tensor* t_emb_shift,                                                                          \
                                                       const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,                                             \
                                                       const ::brotensor::Tensor& W2_int8, const ::brotensor::Tensor& s2, const ::brotensor::Tensor* b2,                \
                                                       const ::brotensor::Tensor* Wskip_int8, const ::brotensor::Tensor* sskip,                                         \
                                                       const ::brotensor::Tensor* bskip,                                                                                \
                                                       int N, int C_in, int C_out, int H, int W, int num_groups, float eps, ::brotensor::Tensor& Y))                    \
    X(resblock_backward,                       void,  (const ::brotensor::Tensor& X,                                                                                    \
                                                       const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,                                             \
                                                       const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,                                                    \
                                                       const ::brotensor::Tensor* t_emb_shift,                                                                          \
                                                       const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,                                             \
                                                       const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,                                                    \
                                                       const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,                                              \
                                                       int N, int C_in, int C_out, int H, int W, int num_groups, float eps,                                             \
                                                       const ::brotensor::Tensor& dY,                                                                                   \
                                                       ::brotensor::Tensor& dX,                                                                                         \
                                                       ::brotensor::Tensor& dGamma1, ::brotensor::Tensor& dBeta1,                                                       \
                                                       ::brotensor::Tensor& dW1, ::brotensor::Tensor* db1,                                                              \
                                                       ::brotensor::Tensor* dt_emb_shift,                                                                               \
                                                       ::brotensor::Tensor& dGamma2, ::brotensor::Tensor& dBeta2,                                                       \
                                                       ::brotensor::Tensor& dW2, ::brotensor::Tensor* db2,                                                              \
                                                       ::brotensor::Tensor* dWskip, ::brotensor::Tensor* dbskip))                                                       \
    /* ─── Matmul + RoPE + RMSNorm + SwiGLU + KV-cache + Llama family ─── */                                                                                            \
    X(matmul,                                  void,  (const ::brotensor::Tensor& A, const ::brotensor::Tensor& B, ::brotensor::Tensor& C))                             \
    X(matmul_backward,                         void,  (const ::brotensor::Tensor& A, const ::brotensor::Tensor& B, const ::brotensor::Tensor& dC,                       \
                                                       ::brotensor::Tensor& dA, ::brotensor::Tensor& dB))                                                               \
    X(rope_forward,                            void,  (const ::brotensor::Tensor& X, int head_dim, int num_heads, int seq_offset, float theta_base,                     \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(rope_backward,                           void,  (const ::brotensor::Tensor& dY, int head_dim, int num_heads, int seq_offset, float theta_base,                    \
                                                       ::brotensor::Tensor& dX))                                                                                        \
    X(rms_norm_forward,                        void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma, float eps, ::brotensor::Tensor& Y))              \
    X(rms_norm_backward,                       void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& dY, float eps,        \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma))                                                           \
    X(swiglu_forward,                          void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(swiglu_backward,                         void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                           \
    X(kv_cache_append,                         void,  (const ::brotensor::Tensor& K_new, const ::brotensor::Tensor& V_new, int cur_len,                                 \
                                                       ::brotensor::Tensor& K_cache, ::brotensor::Tensor& V_cache))                                                     \
    X(flash_attention_decode,                  void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K_cache, const ::brotensor::Tensor& V_cache,            \
                                                       int valid_len, int num_q_heads, int num_kv_heads, ::brotensor::Tensor& O))                                       \
    /* ─── Public reductions ─── */                                                                                                                                     \
    X(sum_rows,                                void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(sum_cols,                                void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(argmax_rows,                             void,  (const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx))                                                         \
    /* ─── Diffusion sampler steps + timestep embedding ─── */                                                                                                          \
    X(ddim_step,                               void,  (const ::brotensor::Tensor& x_t, const ::brotensor::Tensor& eps_pred,                                             \
                                                       float alpha_t, float alpha_prev, float sigma_t, ::brotensor::Tensor& x_prev))                                    \
    X(euler_step,                              void,  (const ::brotensor::Tensor& x_t, const ::brotensor::Tensor& eps_pred,                                             \
                                                       float sigma_t, float sigma_prev, ::brotensor::Tensor& x_prev))                                                   \
    X(dpmpp_2m_step,                           void,  (const ::brotensor::Tensor& x_t, const ::brotensor::Tensor& eps_pred,                                             \
                                                       const ::brotensor::Tensor& x0_prev, float sigma_t,                                                               \
                                                       float c_xt, float c_x0t, float c_x0prev,                                                                         \
                                                       ::brotensor::Tensor& x_prev, ::brotensor::Tensor& x0_out))                                                       \
    X(timestep_embedding,                      void,  (const ::brotensor::Tensor& timesteps, int dim, float max_period, ::brotensor::Tensor& Y))                        \
    /* ─── INT8 weight-only quantisation (W8A16) ─── */                                                                                                                 \
    X(matmul_int8w_fp16,                       void,  (const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,                                            \
                                                       const ::brotensor::Tensor& X, ::brotensor::Tensor& Y))                                                           \
    X(conv2d_int8w_fp16_forward,               void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,              \
                                                       const ::brotensor::Tensor* bias,                                                                                 \
                                                       int N, int C_in, int H, int W, int C_out, int kH, int kW,                                                        \
                                                       int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups,                              \
                                                       ::brotensor::Tensor& Y))                                                                                         \
    X(linear_forward_batched_int8w_fp16,       void,  (const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,                                            \
                                                       const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD))                    \
    X(flash_attention_project_kv_int8w_fp16,   void,  (const ::brotensor::Tensor& ctx,                                                                                  \
                                                       const ::brotensor::Tensor& Wk_int8, const ::brotensor::Tensor& sk, const ::brotensor::Tensor* bk,                \
                                                       const ::brotensor::Tensor& Wv_int8, const ::brotensor::Tensor& sv, const ::brotensor::Tensor* bv,                \
                                                       ::brotensor::Tensor& K_out, ::brotensor::Tensor& V_out))                                                         \
    X(flash_attention_q_with_kv_cached_int8w_fp16, void, (const ::brotensor::Tensor& X, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,                     \
                                                       const ::brotensor::Tensor& Wq_int8, const ::brotensor::Tensor& sq, const ::brotensor::Tensor* bq,                \
                                                       const ::brotensor::Tensor& Wo_int8, const ::brotensor::Tensor& so, const ::brotensor::Tensor* bo,                \
                                                       const float* d_mask, int num_heads, bool causal, ::brotensor::Tensor& O))                                        \
    X(flash_attention_qkvo_int8w_fp16,         void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor* Ctx,                                                    \
                                                       const ::brotensor::Tensor& Wq_int8, const ::brotensor::Tensor& sq, const ::brotensor::Tensor* bq,                \
                                                       const ::brotensor::Tensor& Wk_int8, const ::brotensor::Tensor& sk, const ::brotensor::Tensor* bk,                \
                                                       const ::brotensor::Tensor& Wv_int8, const ::brotensor::Tensor& sv, const ::brotensor::Tensor* bv,                \
                                                       const ::brotensor::Tensor& Wo_int8, const ::brotensor::Tensor& so, const ::brotensor::Tensor* bo,                \
                                                       const float* d_mask, int num_heads, bool causal, ::brotensor::Tensor& O))                                        \
    /* ─── DiT / diffusion extras: AdaLN modulation, axial RoPE, T5-bias attention ─── */                                                                               \
    X(modulate,                                void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& scale,                                                   \
                                                       const ::brotensor::Tensor& shift, ::brotensor::Tensor& Y))                                                        \
    X(broadcast_mul,                           void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& v, ::brotensor::Tensor& Y))         \
    X(rope_apply,                              void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,                                                  \
                                                       const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads, ::brotensor::Tensor& Y))                         \
    X(rope_apply_backward,                     void,  (const ::brotensor::Tensor& dY, const ::brotensor::Tensor& cos_tbl,                                                 \
                                                       const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads, ::brotensor::Tensor& dX))                        \
    X(rope_apply_mrope,                        void,  (const ::brotensor::Tensor& X,                                                                                       \
                                                       const ::brotensor::Tensor& cos_t, const ::brotensor::Tensor& sin_t,                                                 \
                                                       const ::brotensor::Tensor& cos_h, const ::brotensor::Tensor& sin_h,                                                 \
                                                       const ::brotensor::Tensor& cos_w, const ::brotensor::Tensor& sin_w,                                                 \
                                                       const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,                                                   \
                                                       int head_dim, int num_heads, int d_t, int d_h, int d_w,                                                             \
                                                       ::brotensor::Tensor& Y))                                                                                            \
    X(self_attention_bias_forward,             void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,                        \
                                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo, const float* d_mask,                                \
                                                       const ::brotensor::Tensor* attn_bias, int num_heads, float scale, ::brotensor::Tensor& O))                         \
    X(self_attention_bias_int8w_fp16,          void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wq_int8, const ::brotensor::Tensor& sq,                   \
                                                       const ::brotensor::Tensor& Wk_int8, const ::brotensor::Tensor& sk,                                                 \
                                                       const ::brotensor::Tensor& Wv_int8, const ::brotensor::Tensor& sv,                                                 \
                                                       const ::brotensor::Tensor& Wo_int8, const ::brotensor::Tensor& so,                                                 \
                                                       const float* d_mask, const ::brotensor::Tensor* attn_bias,                                                         \
                                                       int num_heads, float scale, ::brotensor::Tensor& O))                                                               \
    /* ─── Spectral / FFT core (brosoundml) ─── */                                                                                                                        \
    X(complex_mul,                             void,  (const ::brotensor::Tensor& a, const ::brotensor::Tensor& b, ::brotensor::Tensor& y))                               \
    X(complex_mul_backward,                    void,  (const ::brotensor::Tensor& a, const ::brotensor::Tensor& b, const ::brotensor::Tensor& dY,                         \
                                                       ::brotensor::Tensor& dA, ::brotensor::Tensor& dB))                                                                 \
    X(complex_abs,                             void,  (const ::brotensor::Tensor& z, ::brotensor::Tensor& y))                                                             \
    X(complex_abs_backward,                    void,  (const ::brotensor::Tensor& z, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dZ))                             \
    X(complex_angle,                           void,  (const ::brotensor::Tensor& z, ::brotensor::Tensor& y))                                                             \
    X(complex_from_polar,                      void,  (const ::brotensor::Tensor& mag, const ::brotensor::Tensor& phase, ::brotensor::Tensor& y))                         \
    X(fft,                                     void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                             \
    X(ifft,                                    void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                             \
    X(rfft,                                    void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                                             \
    X(irfft,                                   void,  (const ::brotensor::Tensor& x, int L, ::brotensor::Tensor& y))                                                      \
    X(rfft_backward,                           void,  (const ::brotensor::Tensor& dY, int L, ::brotensor::Tensor& dX))                                                    \
    X(irfft_backward,                          void,  (const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                                                           \
    /* ─── STFT / iSTFT (brosoundml) ─── */                                                                                                                              \
    X(stft,                                    void,  (const ::brotensor::Tensor& signal, const ::brotensor::Tensor& window,                                              \
                                                       int N, int n_fft, int hop_length, int win_length,                                                                 \
                                                       bool center, bool normalized, ::brotensor::Tensor& spec))                                                         \
    X(stft_backward,                           void,  (const ::brotensor::Tensor& dSpec, const ::brotensor::Tensor& window,                                               \
                                                       int N, int signal_len, int n_fft, int hop_length, int win_length,                                                 \
                                                       bool center, bool normalized, ::brotensor::Tensor& dSignal))                                                      \
    X(istft,                                   void,  (const ::brotensor::Tensor& spec, const ::brotensor::Tensor& window,                                                \
                                                       int N, int signal_len, int n_fft, int hop_length, int win_length,                                                 \
                                                       bool center, bool normalized, ::brotensor::Tensor& signal))                                                       \
    X(istft_backward,                          void,  (const ::brotensor::Tensor& dSignal, const ::brotensor::Tensor& window,                                             \
                                                       int N, int signal_len, int n_fft, int hop_length, int win_length,                                                 \
                                                       bool center, bool normalized, ::brotensor::Tensor& dSpec))                                                        \
    /* ─── 1D convolution family (brosoundml) ─── */                                                                                                                      \
    X(conv_transpose1d_forward,                void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt, const ::brotensor::Tensor* bias,                       \
                                                       int N, int C_in, int L, int C_out, int kL,                                                                         \
                                                       int stride, int padding, int output_padding, int dilation, int groups,                                             \
                                                       ::brotensor::Tensor& Y))                                                                                           \
    X(conv_transpose1d_backward_input,         void,  (const ::brotensor::Tensor& Wt, const ::brotensor::Tensor& dY,                                                       \
                                                       int N, int C_in, int L, int C_out, int kL,                                                                         \
                                                       int stride, int padding, int output_padding, int dilation, int groups,                                             \
                                                       ::brotensor::Tensor& dX))                                                                                          \
    X(conv_transpose1d_backward_weight,        void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,                                                        \
                                                       int N, int C_in, int L, int C_out, int kL,                                                                         \
                                                       int stride, int padding, int output_padding, int dilation, int groups,                                             \
                                                       ::brotensor::Tensor& dWt))                                                                                         \
    X(conv_transpose1d_backward_bias,          void,  (const ::brotensor::Tensor& dY, int N, int C_out, int L_out, ::brotensor::Tensor& dB))                               \
    X(causal_conv1d_update,                    void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt, const ::brotensor::Tensor* bias,                       \
                                                       int N, int C, int L_step, int kL, int dilation,                                                                    \
                                                       ::brotensor::Tensor& state, ::brotensor::Tensor& Y))                                                               \
    X(pad1d_forward,                           void,  (const ::brotensor::Tensor& X, int N, int C, int L,                                                                  \
                                                       int pad_left, int pad_right, int mode, ::brotensor::Tensor& Y))                                                     \
    X(pad1d_backward,                          void,  (const ::brotensor::Tensor& dY, int N, int C, int L,                                                                 \
                                                       int pad_left, int pad_right, int mode, ::brotensor::Tensor& dX))                                                     \
    /* ─── Vocoder / codec activations (brosoundml) ─── */                                                                                                                  \
    X(snake_forward,                           void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& alpha,                                                       \
                                                       const ::brotensor::Tensor* beta, int N, int C, int L, ::brotensor::Tensor& Y))                                       \
    X(snake_backward,                          void,  (const ::brotensor::Tensor& X, const ::brotensor::Tensor& alpha,                                                       \
                                                       const ::brotensor::Tensor* beta, const ::brotensor::Tensor& dY,                                                      \
                                                       int N, int C, int L,                                                                                                 \
                                                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dAlpha, ::brotensor::Tensor* dBeta))                                    \
    X(elu_forward,                             void,  (const ::brotensor::Tensor& x, float alpha, ::brotensor::Tensor& y))                                                   \
    X(elu_backward,                            void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, float alpha, ::brotensor::Tensor& dX))                   \
    X(leaky_relu_forward,                      void,  (const ::brotensor::Tensor& x, float negative_slope, ::brotensor::Tensor& y))                                          \
    X(leaky_relu_backward,                     void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, float negative_slope,                                    \
                                                       ::brotensor::Tensor& dX))                                                                                          \
    /* ─── Codec quantization (brosoundml CHUNK 5, family D) ─── */                                                                                                        \
    X(vq_encode_forward,                       void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& codebook,                                                   \
                                                       ::brotensor::Tensor& indices, ::brotensor::Tensor& quantized))                                                      \
    X(vq_encode_backward,                      void,  (const ::brotensor::Tensor& dQuantized, ::brotensor::Tensor& dX))                                                     \
    X(fsq_quantize_forward,                    void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& levels,                                                     \
                                                       ::brotensor::Tensor& quantized, ::brotensor::Tensor& packed_indices))                                               \
    X(fsq_quantize_backward,                   void,  (const ::brotensor::Tensor& dQuantized, ::brotensor::Tensor& dX))                        \
    /* ─── 1D resampling (brosoundml CHUNK 6, family E) ─── */                                                                                 \
    X(resample1d_forward,                      void,  (const ::brotensor::Tensor& X, int N, int C, int L_in, int L_out, int mode,              \
                                                       ::brotensor::Tensor& Y))                                                                \
    X(resample1d_backward,                     void,  (const ::brotensor::Tensor& dY, int N, int C, int L_in, int L_out, int mode,             \
                                                       ::brotensor::Tensor& dX))                                                               \
    /* ─── log / exp / round elementwise (brosoundml CHUNK 6, family G) ─── */                                                                 \
    X(log_forward,                             void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                  \
    X(log_backward,                            void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))  \
    X(exp_forward,                             void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                  \
    X(exp_backward,                            void,  (const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))  \
    X(round_forward,                           void,  (const ::brotensor::Tensor& x, ::brotensor::Tensor& y))                                  \
    X(round_backward,                          void,  (const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                               \
    /* ─── Autoregressive logit sampling (brosoundml CHUNK 7, family F) ─── */                                                                \
    X(sample_logits,                           void,  (const ::brotensor::Tensor& logits, float temperature, int top_k, float top_p,          \
                                                       uint64_t key, uint64_t counter, ::brotensor::Tensor& indices))                          \
    /* ─── L2 norm + Gated Delta Rule (linear-attention text path) ─── */                                                                      \
    X(l2_norm_forward,                         void,  (const ::brotensor::Tensor& X, int head_dim, int num_heads, float eps,                   \
                                                       ::brotensor::Tensor& Y))                                                                \
    X(l2_norm_backward,                        void,  (const ::brotensor::Tensor& X, int head_dim, int num_heads, float eps,                   \
                                                       const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX))                                \
    X(gated_delta_rule_chunked,                void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,\
                                                       const ::brotensor::Tensor& a_raw, const ::brotensor::Tensor& beta,                      \
                                                       const ::brotensor::Tensor& log_A,                                                       \
                                                       int num_heads, int d_k, int d_v,                                                        \
                                                       ::brotensor::Tensor& state, ::brotensor::Tensor& O))                                    \
    X(gated_delta_rule_step,                   void,  (const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K, const ::brotensor::Tensor& V,\
                                                       const ::brotensor::Tensor& a_raw, const ::brotensor::Tensor& beta,                      \
                                                       const ::brotensor::Tensor& log_A,                                                       \
                                                       int num_heads, int d_k, int d_v,                                                        \
                                                       ::brotensor::Tensor& state, ::brotensor::Tensor& O))
