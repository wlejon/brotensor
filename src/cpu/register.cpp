// ─── CPU backend registration ──────────────────────────────────────────────
//
// Builds the CPU OpsVTable (only the slots we implement; everything else
// stays nullptr — the dispatcher throws "not implemented on this backend"
// on null lookups), pairs it with the CPU AllocVTable from alloc.cpp, and
// hands them to the registry at static-init time. CPU is therefore always
// available without a prior brotensor::init() call.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brotensor::detail::cpu {

// ── alloc.cpp ──
const AllocVTable& cpu_alloc_table();

// ── ops_impl.cpp — forward decls of the 16 implemented ops ──
void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB);
void relu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void relu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void tanh_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void tanh_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void sigmoid_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void sigmoid_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX);
void softmax_forward(const ::brotensor::Tensor& logits, ::brotensor::Tensor& probs,
                     const float* mask);
void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits);
float softmax_xent_segment(const float* lp, const float* tp,
                           float* pp, float* dz,
                           int n, const float* mask);
float softmax_xent(const ::brotensor::Tensor& logits,
                   const ::brotensor::Tensor& target,
                   ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits,
                   const float* mask);
float mse_scalar(float pred, float target, float& dPred);
void add_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);
void add_scalar_inplace(::brotensor::Tensor& y, float s);
void xavier_init(::brotensor::Tensor& W, uint64_t& rng_state);

// ── ops_impl.cpp — forward decls of the 20 newly implemented ops ──
void sgd_step(::brotensor::Tensor& param, ::brotensor::Tensor& grad,
              ::brotensor::Tensor& velocity, float lr, float momentum);
void adam_step(::brotensor::Tensor& param, const ::brotensor::Tensor& grad,
               ::brotensor::Tensor& m, ::brotensor::Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);
void scale_inplace(::brotensor::Tensor& y, float s);
void layernorm_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& gamma,
                       const ::brotensor::Tensor& beta,
                       ::brotensor::Tensor& y, ::brotensor::Tensor& xhat,
                       float& mean_out, float& rstd_out, float eps);
void layernorm_backward(const ::brotensor::Tensor& dY,
                        const ::brotensor::Tensor& xhat,
                        const ::brotensor::Tensor& gamma, float rstd,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta);
void attention_forward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                       const float* d_mask,
                       ::brotensor::Tensor& Q, ::brotensor::Tensor& K,
                       ::brotensor::Tensor& V, ::brotensor::Tensor& Attn,
                       ::brotensor::Tensor& Y_pre_Wo, ::brotensor::Tensor& O);
void attention_backward(const ::brotensor::Tensor& dO,
                        const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                        const ::brotensor::Tensor& V, const ::brotensor::Tensor& Attn,
                        const ::brotensor::Tensor& Y_pre_Wo,
                        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                        const float* d_mask,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                        ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
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
void concat_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                 ::brotensor::Tensor& out);
void split_rows(const ::brotensor::Tensor& in,
                const std::vector<::brotensor::Tensor*>& parts);
void masked_mean_pool_forward(const ::brotensor::Tensor& X, const float* d_mask,
                              ::brotensor::Tensor& y);
void masked_mean_pool_backward(const ::brotensor::Tensor& dY, const float* d_mask,
                               int K, ::brotensor::Tensor& dX);
void build_slot_mask(const ::brotensor::Tensor& x, int offset, int K, int stride,
                     ::brotensor::Tensor& mask);
void copy_d2d(const ::brotensor::Tensor& src, int src_off,
              ::brotensor::Tensor& dst, int dst_off, int n);
void add_inplace_batched(::brotensor::Tensor& Y_BD,
                         const ::brotensor::Tensor& X_BD);
void linear_forward_batched(const ::brotensor::Tensor& W,
                            const ::brotensor::Tensor& bias,
                            const ::brotensor::Tensor& X_BD,
                            ::brotensor::Tensor& Y_BD);
void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB);
void relu_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD);
void relu_backward_batched(const ::brotensor::Tensor& X_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD);
void tanh_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD);
void tanh_backward_batched(const ::brotensor::Tensor& Y_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD);
void mse_vec_per_sample(const ::brotensor::Tensor& pred,
                        const ::brotensor::Tensor& target,
                        ::brotensor::Tensor& dPred,
                        ::brotensor::Tensor& loss_per_sample);
void softmax_xent_fused_batched(const ::brotensor::Tensor& logits_BL,
                                const ::brotensor::Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                ::brotensor::Tensor& probs_BL,
                                ::brotensor::Tensor& dLogits_BL,
                                ::brotensor::Tensor& loss_per_sample);
void bce_with_logits_fused_batched(const ::brotensor::Tensor& logits_BL,
                                   const ::brotensor::Tensor& target_BL,
                                   const float* d_mask_BL,
                                   float pos_weight,
                                   ::brotensor::Tensor& probs_BL,
                                   ::brotensor::Tensor& dLogits_BL,
                                   ::brotensor::Tensor& loss_per_sample);

// ── CHUNK 1 — elementwise.cpp / loss.cpp / embedding.cpp / concat.cpp /
//    public_reductions.cpp / layernorm_inference.cpp ──
void clamp(::brotensor::Tensor& y, float lo, float hi);
void mul_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);
void cast(const ::brotensor::Tensor& src, ::brotensor::Tensor& dst,
          ::brotensor::Dtype out_dtype);
float mse_vec_forward(const ::brotensor::Tensor& pred,
                      const ::brotensor::Tensor& target);
void mse_vec_backward(const ::brotensor::Tensor& pred,
                      const ::brotensor::Tensor& target,
                      ::brotensor::Tensor& dPred);
float softmax_xent_fused(const ::brotensor::Tensor& logits,
                         const ::brotensor::Tensor& target,
                         const float* d_mask,
                         ::brotensor::Tensor& probs,
                         ::brotensor::Tensor& dLogits);
void embedding_lookup_forward(const ::brotensor::Tensor& table,
                              const int32_t* d_idx, int B,
                              ::brotensor::Tensor& out);
void embedding_lookup_backward(const ::brotensor::Tensor& dOut,
                               const int32_t* d_idx, int B,
                               ::brotensor::Tensor& dTable);
void concat_batched_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                         ::brotensor::Tensor& out);
void concat_nchw_channels(const std::vector<const ::brotensor::Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          ::brotensor::Tensor& out);
void concat_nchw_channels_backward(const ::brotensor::Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<::brotensor::Tensor*>& parts);
void sum_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void sum_cols(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx);
void layernorm_forward_inference_batched(const ::brotensor::Tensor& X_RD,
                                         const ::brotensor::Tensor& gamma,
                                         const ::brotensor::Tensor& beta,
                                         ::brotensor::Tensor& Y_RD, float eps);
void layernorm_forward_batched_with_caches(const ::brotensor::Tensor& X_RD,
                                           const ::brotensor::Tensor& gamma,
                                           const ::brotensor::Tensor& beta,
                                           ::brotensor::Tensor& Y_RD,
                                           ::brotensor::Tensor& Xhat_RD,
                                           ::brotensor::Tensor& Mean_R,
                                           ::brotensor::Tensor& Rstd_R,
                                           float eps);
void layernorm_backward_batched_with_caches(const ::brotensor::Tensor& dY_RD,
                                            const ::brotensor::Tensor& Xhat_RD,
                                            const ::brotensor::Tensor& gamma,
                                            const ::brotensor::Tensor& Rstd_R,
                                            ::brotensor::Tensor& dX_RD,
                                            ::brotensor::Tensor& dGamma,
                                            ::brotensor::Tensor& dBeta);
void build_causal_mask_row(int L, int q, ::brotensor::Tensor& mask);

// ── CHUNK 2 — activations.cpp / geglu.cpp / swiglu.cpp / matmul.cpp /
//    rope.cpp / rms_norm.cpp ──
void silu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void silu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void gelu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void gelu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void gelu_exact_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void gelu_exact_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         ::brotensor::Tensor& dX);
void quick_gelu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void quick_gelu_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         ::brotensor::Tensor& dX);
void geglu_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void geglu_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX);
void geglu_exact_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void geglu_exact_backward(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dX);
void swiglu_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void swiglu_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX);
void matmul(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
            ::brotensor::Tensor& C);
void matmul_backward(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA, ::brotensor::Tensor& dB);
void rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, ::brotensor::Tensor& Y);
void rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, ::brotensor::Tensor& dX);
void rms_norm_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma,
                      float eps, ::brotensor::Tensor& Y);
void rms_norm_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma,
                       const ::brotensor::Tensor& dY, float eps,
                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma);

// ── CHUNK 3 — conv2d.cpp / group_norm.cpp ──
void conv2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, ::brotensor::Tensor& Y);
void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups,
                           ::brotensor::Tensor& dX);
void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups,
                            ::brotensor::Tensor& dWt);
void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB);
void deform_conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& offset,
                           const ::brotensor::Tensor* mask,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W, int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, int deform_groups,
                           ::brotensor::Tensor& Y);
void conv3d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w, int groups,
                    ::brotensor::Tensor& Y);
void group_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        int N, int C, int H, int W, int num_groups,
                        float eps, ::brotensor::Tensor& Y);
void group_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta);

// ── CHUNK 4 — resample.cpp / transpose.cpp / diffusion_samplers.cpp /
//    kv_cache.cpp ──
void upsample_nearest_2x(const ::brotensor::Tensor& X,
                         int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_bilinear_2x(const ::brotensor::Tensor& X,
                          int N, int C, int H, int W, ::brotensor::Tensor& Y);
void downsample_avg_2x(const ::brotensor::Tensor& X,
                       int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_nearest_2x_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  ::brotensor::Tensor& dX);
void upsample_bilinear_2x_backward(const ::brotensor::Tensor& dY,
                                   int N, int C, int H, int W,
                                   ::brotensor::Tensor& dX);
void downsample_avg_2x_backward(const ::brotensor::Tensor& dY,
                                int N, int C, int H, int W,
                                ::brotensor::Tensor& dX);
void interp2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H_in, int W_in,
                      int H_out, int W_out, int mode,
                      ::brotensor::Tensor& Y);
void interp2d_backward(const ::brotensor::Tensor& dY,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode,
                       ::brotensor::Tensor& dX);
void interp2d_align_corners_forward(const ::brotensor::Tensor& X,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode,
                                    ::brotensor::Tensor& Y);
void pad2d_forward(const ::brotensor::Tensor& X,
                   int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, ::brotensor::Tensor& Y);
void pad2d_backward(const ::brotensor::Tensor& dY,
                    int N, int C, int H, int W,
                    int pad_top, int pad_bottom, int pad_left, int pad_right,
                    int mode, ::brotensor::Tensor& dX);
void unfold2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_top, int pad_bottom, int pad_left, int pad_right,
                      int mode, ::brotensor::Tensor& Y);
void l2_normalize_nchw_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W, float eps,
                               ::brotensor::Tensor& Y);
void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y);
void slice2d_forward(const ::brotensor::Tensor& X,
                     int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out,
                     ::brotensor::Tensor& Y);
void slice2d_backward(const ::brotensor::Tensor& dY,
                      int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out,
                      ::brotensor::Tensor& dX);
void top_k_rows(const ::brotensor::Tensor& X, int k,
                ::brotensor::Tensor& Vals, ::brotensor::Tensor& Idx);
void adaptive_avg_pool2d_forward(const ::brotensor::Tensor& X,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out,
                                 ::brotensor::Tensor& Y);
void adaptive_avg_pool2d_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  ::brotensor::Tensor& dX);
void max_pool2d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& Idx);
void max_pool2d_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& Idx,
                         int N, int C, int H, int W, int H_out, int W_out,
                         ::brotensor::Tensor& dX);
void gather_rows(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Idx,
                 ::brotensor::Tensor& Y);
void scatter_rows_add(const ::brotensor::Tensor& dY,
                      const ::brotensor::Tensor& Idx, int R,
                      ::brotensor::Tensor& dX);
void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y);
void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     ::brotensor::Tensor& dX);
void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      ::brotensor::Tensor& dWt);
void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB);
void window_partition_forward(const ::brotensor::Tensor& X,
                              int N, int C, int H, int W, int window,
                              ::brotensor::Tensor& Y);
void window_reverse_forward(const ::brotensor::Tensor& X,
                            int N, int C, int H, int W, int window,
                            ::brotensor::Tensor& Y);
void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);
void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);
void ddim_step(const ::brotensor::Tensor& x_t,
               const ::brotensor::Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               ::brotensor::Tensor& x_prev);
void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev);
void dpmpp_2m_step(const ::brotensor::Tensor& x_t,
                   const ::brotensor::Tensor& eps_pred,
                   const ::brotensor::Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   ::brotensor::Tensor& x_prev, ::brotensor::Tensor& x0_out);
void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period, ::brotensor::Tensor& Y);
void kv_cache_append(const ::brotensor::Tensor& K_new,
                     const ::brotensor::Tensor& V_new, int cur_len,
                     ::brotensor::Tensor& K_cache, ::brotensor::Tensor& V_cache);
void flash_attention_decode(const ::brotensor::Tensor& Q,
                            const ::brotensor::Tensor& K_cache,
                            const ::brotensor::Tensor& V_cache,
                            int valid_len,
                            int num_q_heads, int num_kv_heads,
                            ::brotensor::Tensor& O);

// ── CHUNK 5 — cross_attention.cpp / self_attention.cpp /
//    attention_moments.cpp ──
void cross_attention_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Ctx,
                             const ::brotensor::Tensor& Wq,
                             const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv,
                             const ::brotensor::Tensor& Wo,
                             const float* d_mask, int num_heads,
                             ::brotensor::Tensor& O);
void cross_attention_forward_with_attn(const ::brotensor::Tensor& X,
                                       const ::brotensor::Tensor& Ctx,
                                       const ::brotensor::Tensor& Wq,
                                       const ::brotensor::Tensor& Wk,
                                       const ::brotensor::Tensor& Wv,
                                       const ::brotensor::Tensor& Wo,
                                       const float* d_mask,
                                       const ::brotensor::Tensor* attn_logit_bias,
                                       int num_heads,
                                       ::brotensor::Tensor& O,
                                       ::brotensor::Tensor& AttnAvg);
void self_attention_forward_train(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor& Wq,
                                  const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor& Wv,
                                  const ::brotensor::Tensor& Wo,
                                  const float* d_mask, int num_heads,
                                  ::brotensor::Tensor& Qh,
                                  ::brotensor::Tensor& Kh,
                                  ::brotensor::Tensor& Vh,
                                  ::brotensor::Tensor& Attnh,
                                  ::brotensor::Tensor& Yconcat,
                                  ::brotensor::Tensor& O);
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
                             const float* d_mask, int num_heads,
                             ::brotensor::Tensor& dX,
                             ::brotensor::Tensor& dWq,
                             ::brotensor::Tensor& dWk,
                             ::brotensor::Tensor& dWv,
                             ::brotensor::Tensor& dWo);
void attention_token_moments(const ::brotensor::Tensor& Attn,
                             int h_lat, int w_lat,
                             ::brotensor::Tensor& mass,
                             ::brotensor::Tensor& centroid);
void cross_attention_forward_train(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& Ctx,
                                   const ::brotensor::Tensor& Wq,
                                   const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor& Wv,
                                   const ::brotensor::Tensor& Wo,
                                   const float* d_mask, int num_heads,
                                   ::brotensor::Tensor& Qh,
                                   ::brotensor::Tensor& Kh,
                                   ::brotensor::Tensor& Vh,
                                   ::brotensor::Tensor& Attnh,
                                   ::brotensor::Tensor& Yconcat,
                                   ::brotensor::Tensor& O);
void cross_attention_backward(const ::brotensor::Tensor& dO,
                              const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Ctx,
                              const ::brotensor::Tensor& Qh,
                              const ::brotensor::Tensor& Kh,
                              const ::brotensor::Tensor& Vh,
                              const ::brotensor::Tensor& Attnh,
                              const ::brotensor::Tensor& Yconcat,
                              const ::brotensor::Tensor& Wq,
                              const ::brotensor::Tensor& Wk,
                              const ::brotensor::Tensor& Wv,
                              const ::brotensor::Tensor& Wo,
                              const float* d_mask, int num_heads,
                              ::brotensor::Tensor& dX,
                              ::brotensor::Tensor& dCtx,
                              ::brotensor::Tensor& dWq,
                              ::brotensor::Tensor& dWk,
                              ::brotensor::Tensor& dWv,
                              ::brotensor::Tensor& dWo);

// ── CHUNK 6 — flash_attention.cpp / self_attention_inference.cpp /
//    resblock.cpp ──
void flash_attention_forward(const ::brotensor::Tensor& Q,
                             const ::brotensor::Tensor& K,
                             const ::brotensor::Tensor& V,
                             const float* d_mask, int num_heads, bool causal,
                             ::brotensor::Tensor& O);
void flash_attention_windowed_forward(const ::brotensor::Tensor& Q,
                                      const ::brotensor::Tensor& K,
                                      const ::brotensor::Tensor& V,
                                      const float* d_mask, int num_heads, int window,
                                      ::brotensor::Tensor& O);
void flash_attention_varlen_forward(const ::brotensor::Tensor& Q,
                                    const ::brotensor::Tensor& K,
                                    const ::brotensor::Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size, int max_seqlen_q, int max_seqlen_k,
                                    int num_heads, int head_dim, bool causal,
                                    ::brotensor::Tensor& O);
void flash_attention_varlen_backward(const ::brotensor::Tensor& Q,
                                     const ::brotensor::Tensor& K,
                                     const ::brotensor::Tensor& V,
                                     const ::brotensor::Tensor& O,
                                     const ::brotensor::Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     ::brotensor::Tensor& dQ,
                                     ::brotensor::Tensor& dK,
                                     ::brotensor::Tensor& dV);
void flash_attention_qkvo_forward(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor* Ctx,
                                  const ::brotensor::Tensor& Wq,
                                  const ::brotensor::Tensor* bq,
                                  const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor* bk,
                                  const ::brotensor::Tensor& Wv,
                                  const ::brotensor::Tensor* bv,
                                  const ::brotensor::Tensor& Wo,
                                  const ::brotensor::Tensor* bo,
                                  const float* d_mask, int num_heads,
                                  bool causal, ::brotensor::Tensor& O);
void flash_attention_qkvo_backward(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor* Ctx,
                                   const ::brotensor::Tensor& Wq,
                                   const ::brotensor::Tensor* bq,
                                   const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor* bk,
                                   const ::brotensor::Tensor& Wv,
                                   const ::brotensor::Tensor* bv,
                                   const ::brotensor::Tensor& Wo,
                                   const ::brotensor::Tensor* bo,
                                   const float* d_mask, int num_heads,
                                   bool causal,
                                   const ::brotensor::Tensor& dO,
                                   ::brotensor::Tensor& dX,
                                   ::brotensor::Tensor* dCtx,
                                   ::brotensor::Tensor& dWq,
                                   ::brotensor::Tensor* dbq,
                                   ::brotensor::Tensor& dWk,
                                   ::brotensor::Tensor* dbk,
                                   ::brotensor::Tensor& dWv,
                                   ::brotensor::Tensor* dbv,
                                   ::brotensor::Tensor& dWo,
                                   ::brotensor::Tensor* dbo);
void flash_attention_backward(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O,
                              const ::brotensor::Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              ::brotensor::Tensor& dQ,
                              ::brotensor::Tensor& dK,
                              ::brotensor::Tensor& dV);
void flash_attention_project_kv(const ::brotensor::Tensor& ctx,
                                const ::brotensor::Tensor& Wk,
                                const ::brotensor::Tensor* bk,
                                const ::brotensor::Tensor& Wv,
                                const ::brotensor::Tensor* bv,
                                ::brotensor::Tensor& K_out,
                                ::brotensor::Tensor& V_out);
void flash_attention_q_with_kv_cached_forward(const ::brotensor::Tensor& X,
                                              const ::brotensor::Tensor& K,
                                              const ::brotensor::Tensor& V,
                                              const ::brotensor::Tensor& Wq,
                                              const ::brotensor::Tensor* bq,
                                              const ::brotensor::Tensor& Wo,
                                              const ::brotensor::Tensor* bo,
                                              const float* d_mask,
                                              int num_heads, bool causal,
                                              ::brotensor::Tensor& O);
void self_attention_forward(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& Wq,
                            const ::brotensor::Tensor& Wk,
                            const ::brotensor::Tensor& Wv,
                            const ::brotensor::Tensor& Wo,
                            const float* d_mask, int num_heads,
                            ::brotensor::Tensor& O);
void resblock_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma1,
                      const ::brotensor::Tensor& beta1,
                      const ::brotensor::Tensor& W1,
                      const ::brotensor::Tensor* b1,
                      const ::brotensor::Tensor* t_emb_shift,
                      const ::brotensor::Tensor& gamma2,
                      const ::brotensor::Tensor& beta2,
                      const ::brotensor::Tensor& W2,
                      const ::brotensor::Tensor* b2,
                      const ::brotensor::Tensor* Wskip,
                      const ::brotensor::Tensor* bskip,
                      int N, int C_in, int C_out, int H, int W,
                      int num_groups, float eps, ::brotensor::Tensor& Y);
void resblock_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma1,
                       const ::brotensor::Tensor& beta1,
                       const ::brotensor::Tensor& W1,
                       const ::brotensor::Tensor* b1,
                       const ::brotensor::Tensor* t_emb_shift,
                       const ::brotensor::Tensor& gamma2,
                       const ::brotensor::Tensor& beta2,
                       const ::brotensor::Tensor& W2,
                       const ::brotensor::Tensor* b2,
                       const ::brotensor::Tensor* Wskip,
                       const ::brotensor::Tensor* bskip,
                       int N, int C_in, int C_out, int H, int W,
                       int num_groups, float eps,
                       const ::brotensor::Tensor& dY,
                       ::brotensor::Tensor& dX,
                       ::brotensor::Tensor& dGamma1,
                       ::brotensor::Tensor& dBeta1,
                       ::brotensor::Tensor& dW1,
                       ::brotensor::Tensor* db1,
                       ::brotensor::Tensor* dt_emb_shift,
                       ::brotensor::Tensor& dGamma2,
                       ::brotensor::Tensor& dBeta2,
                       ::brotensor::Tensor& dW2,
                       ::brotensor::Tensor* db2,
                       ::brotensor::Tensor* dWskip,
                       ::brotensor::Tensor* dbskip);

// ── DiT / diffusion extras — modulate.cpp ──
void modulate(const ::brotensor::Tensor& X, const ::brotensor::Tensor& scale,
              const ::brotensor::Tensor& shift, ::brotensor::Tensor& Y);
void broadcast_mul(const ::brotensor::Tensor& X, const ::brotensor::Tensor& v,
                   ::brotensor::Tensor& Y);

// ── RoPE with explicit cos/sin tables — rope.cpp ──
void rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                ::brotensor::Tensor& Y);
void rope_apply_perhead(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                        const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                        ::brotensor::Tensor& Y);
void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX);

// ── Self-attention with additive bias — self_attention_bias.cpp ──
void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const ::brotensor::Tensor* bq,
                                 const ::brotensor::Tensor* bk,
                                 const ::brotensor::Tensor* bv,
                                 const ::brotensor::Tensor* bo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O);

// ── Self-attention with decomposed 2D rel-pos bias (SAM/ViTDet) —
//    self_attention_decomposed_rel_pos.cpp ──
void self_attention_decomposed_rel_pos_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h, const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        ::brotensor::Tensor& O);
void self_attention_decomposed_rel_pos_windowed_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h, const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        ::brotensor::Tensor& O);

// ── Spectral / FFT core (brosoundml) — fft.cpp ──
void complex_mul(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                 ::brotensor::Tensor& y);
void complex_mul_backward(const ::brotensor::Tensor& a,
                          const ::brotensor::Tensor& b,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dA, ::brotensor::Tensor& dB);
void complex_abs(const ::brotensor::Tensor& z, ::brotensor::Tensor& y);
void complex_abs_backward(const ::brotensor::Tensor& z,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dZ);
void complex_angle(const ::brotensor::Tensor& z, ::brotensor::Tensor& y);
void complex_from_polar(const ::brotensor::Tensor& mag,
                        const ::brotensor::Tensor& phase,
                        ::brotensor::Tensor& y);
void fft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void ifft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void rfft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void irfft(const ::brotensor::Tensor& x, int L, ::brotensor::Tensor& y);
void rfft_backward(const ::brotensor::Tensor& dY, int L,
                   ::brotensor::Tensor& dX);
void irfft_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX);

// ── STFT / iSTFT (brosoundml) — stft.cpp ──
void stft(const ::brotensor::Tensor& signal, const ::brotensor::Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, ::brotensor::Tensor& spec);
void stft_backward(const ::brotensor::Tensor& dSpec,
                   const ::brotensor::Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   ::brotensor::Tensor& dSignal);
void istft(const ::brotensor::Tensor& spec, const ::brotensor::Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, ::brotensor::Tensor& signal);
void istft_backward(const ::brotensor::Tensor& dSignal,
                    const ::brotensor::Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    ::brotensor::Tensor& dSpec);

// ── 1D convolution family (brosoundml) — conv1d.cpp ──
void conv_transpose1d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, ::brotensor::Tensor& Y);
void conv_transpose1d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, ::brotensor::Tensor& dX);
void conv_transpose1d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, ::brotensor::Tensor& dWt);
void conv_transpose1d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int L_out,
                                    ::brotensor::Tensor& dB);
void causal_conv1d_update(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& Wt,
                          const ::brotensor::Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          ::brotensor::Tensor& state, ::brotensor::Tensor& Y);
void pad1d_forward(const ::brotensor::Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode,
                   ::brotensor::Tensor& Y);
void pad1d_backward(const ::brotensor::Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode,
                    ::brotensor::Tensor& dX);

// ── Vocoder / codec activations (brosoundml) — vocoder_activations.cpp ──
void snake_forward(const ::brotensor::Tensor& X,
                   const ::brotensor::Tensor& alpha,
                   const ::brotensor::Tensor* beta,
                   int N, int C, int L, ::brotensor::Tensor& Y);
void snake_backward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& alpha,
                    const ::brotensor::Tensor* beta,
                    const ::brotensor::Tensor& dY,
                    int N, int C, int L,
                    ::brotensor::Tensor& dX, ::brotensor::Tensor& dAlpha,
                    ::brotensor::Tensor* dBeta);
void elu_forward(const ::brotensor::Tensor& x, float alpha,
                 ::brotensor::Tensor& y);
void elu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  float alpha, ::brotensor::Tensor& dX);
void leaky_relu_forward(const ::brotensor::Tensor& x, float negative_slope,
                        ::brotensor::Tensor& y);
void leaky_relu_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         float negative_slope, ::brotensor::Tensor& dX);

// ── Codec quantization (brosoundml CHUNK 5, family D) — codec_quant.cpp ──
void vq_encode_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& codebook,
                       ::brotensor::Tensor& indices,
                       ::brotensor::Tensor& quantized);
void vq_encode_backward(const ::brotensor::Tensor& dQuantized,
                        ::brotensor::Tensor& dX);
void fsq_quantize_forward(const ::brotensor::Tensor& x,
                          const ::brotensor::Tensor& levels,
                          ::brotensor::Tensor& quantized,
                          ::brotensor::Tensor& packed_indices);
void fsq_quantize_backward(const ::brotensor::Tensor& dQuantized,
                           ::brotensor::Tensor& dX);

// ── 1D resampling (brosoundml CHUNK 6, family E) — resample1d.cpp ──
void resample1d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int L_in, int L_out, int mode,
                        ::brotensor::Tensor& Y);
void resample1d_backward(const ::brotensor::Tensor& dY,
                         int N, int C, int L_in, int L_out, int mode,
                         ::brotensor::Tensor& dX);

// ── log / exp / round elementwise (brosoundml CHUNK 6, family G)
//    — log_exp_round.cpp ──
void log_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void log_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX);
void exp_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void exp_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX);
void round_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void round_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX);

// ── Autoregressive logit sampling (brosoundml CHUNK 7, family F)
//    — sample_logits.cpp ──
void sample_logits(const ::brotensor::Tensor& logits, float temperature,
                   int top_k, float top_p, uint64_t key, uint64_t counter,
                   ::brotensor::Tensor& indices);

// ── Counter-based noise generation (Philox 4x32-10) — noise.cpp ──
void randn(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y);
void rand_uniform(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y);
void rand_bernoulli(float p, uint64_t key, uint64_t counter,
                    ::brotensor::Tensor& Y);
void randn_truncated(float lo, float hi, uint64_t key, uint64_t counter,
                     ::brotensor::Tensor& Y);

// ── L2 norm + Gated Delta Rule (linear-attention text path) ──
//    l2_norm.cpp, gated_delta_rule.cpp
void l2_norm_forward(const ::brotensor::Tensor& X,
                     int head_dim, int num_heads, float eps,
                     ::brotensor::Tensor& Y);
void l2_norm_backward(const ::brotensor::Tensor& X,
                      int head_dim, int num_heads, float eps,
                      const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX);
void gated_delta_rule_chunked(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& a_raw,
                              const ::brotensor::Tensor& beta,
                              const ::brotensor::Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              ::brotensor::Tensor& state,
                              ::brotensor::Tensor& O);
void gated_delta_rule_step(const ::brotensor::Tensor& Q,
                           const ::brotensor::Tensor& K,
                           const ::brotensor::Tensor& V,
                           const ::brotensor::Tensor& a_raw,
                           const ::brotensor::Tensor& beta,
                           const ::brotensor::Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           ::brotensor::Tensor& state,
                           ::brotensor::Tensor& O);

// ── Qwen3-VL polish: spatial_merge_2x2 + M-RoPE — spatial_merge.cpp /
//    rope_mrope.cpp ──
void spatial_merge_2x2_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               bool channel_major,
                               ::brotensor::Tensor& Y);
void rope_apply_mrope(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& cos_t,
                      const ::brotensor::Tensor& sin_t,
                      const ::brotensor::Tensor& cos_h,
                      const ::brotensor::Tensor& sin_h,
                      const ::brotensor::Tensor& cos_w,
                      const ::brotensor::Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      ::brotensor::Tensor& Y);

// ── BatchNorm — batch_norm.cpp ──
void batch_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        ::brotensor::Tensor& running_mean,
                        ::brotensor::Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        ::brotensor::Tensor& Y,
                        ::brotensor::Tensor& saved_mean,
                        ::brotensor::Tensor& saved_rstd);
void batch_norm_inference(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& gamma,
                          const ::brotensor::Tensor& beta,
                          const ::brotensor::Tensor& running_mean,
                          const ::brotensor::Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          ::brotensor::Tensor& Y);
void batch_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& saved_mean,
                         const ::brotensor::Tensor& saved_rstd,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta);

// ── Image preprocessing helpers — image_preproc.cpp ──
void image_normalize(const ::brotensor::Tensor& X,
                     const ::brotensor::Tensor& mean,
                     const ::brotensor::Tensor& std_,
                     int N, int C, int H, int W,
                     ::brotensor::Tensor& Y);
void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  ::brotensor::Tensor& Y);
void lstm_forward_train(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                        const ::brotensor::Tensor& W_hh,
                        const ::brotensor::Tensor* b_ih, const ::brotensor::Tensor* b_hh,
                        const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                        int T, int B,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& gates,
                        ::brotensor::Tensor& C,
                        ::brotensor::Tensor* hT, ::brotensor::Tensor* cT);
void lstm_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                   const ::brotensor::Tensor& W_hh,
                   const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                   const ::brotensor::Tensor& Y, const ::brotensor::Tensor& gates,
                   const ::brotensor::Tensor& C,
                   const ::brotensor::Tensor& dY, int T, int B,
                   ::brotensor::Tensor& dX, ::brotensor::Tensor& dW_ih,
                   ::brotensor::Tensor& dW_hh,
                   ::brotensor::Tensor* db_ih, ::brotensor::Tensor* db_hh,
                   ::brotensor::Tensor* dh0, ::brotensor::Tensor* dc0);

// ── StyleGAN3-R primitives — stylegan_elementwise.cpp / bias_act.cpp /
//    upfirdn2d.cpp / modulated_conv2d.cpp ──
void sin_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void sin_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX);
void cos_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void cos_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX);
void rsqrt_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void rsqrt_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX);
void pixel_norm_forward(const ::brotensor::Tensor& X, float eps,
                        ::brotensor::Tensor& Y);
void pixel_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& dY, float eps,
                         ::brotensor::Tensor& dX);
void bias_act_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, ::brotensor::Tensor& Y);
void bias_act_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp,
                       ::brotensor::Tensor& dX, ::brotensor::Tensor* dB);
void upfirdn2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, ::brotensor::Tensor& Y);
void upfirdn2d_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, ::brotensor::Tensor& dX);
void modulated_conv2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& W,
                              const ::brotensor::Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW, int pad_h, int pad_w,
                              bool demodulate, float eps,
                              ::brotensor::Tensor& dcoef, ::brotensor::Tensor& Y);
void modulated_conv2d_backward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W,
                               const ::brotensor::Tensor& s,
                               const ::brotensor::Tensor& dcoef,
                               const ::brotensor::Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW, int pad_h, int pad_w,
                               bool demodulate, float eps,
                               ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                               ::brotensor::Tensor& ds);

} // namespace brotensor::detail::cpu

namespace {

struct CpuStaticRegistrar {
    CpuStaticRegistrar() {
        using namespace ::brotensor;
        using namespace ::brotensor::detail;

        OpsVTable ops{};   // zero-init — every slot nullptr by default

        ops.linear_forward       = &detail::cpu::linear_forward;
        ops.linear_backward      = &detail::cpu::linear_backward;
        ops.relu_forward         = &detail::cpu::relu_forward;
        ops.relu_backward        = &detail::cpu::relu_backward;
        ops.tanh_forward         = &detail::cpu::tanh_forward;
        ops.tanh_backward        = &detail::cpu::tanh_backward;
        ops.sigmoid_forward      = &detail::cpu::sigmoid_forward;
        ops.sigmoid_backward     = &detail::cpu::sigmoid_backward;
        ops.softmax_forward      = &detail::cpu::softmax_forward;
        ops.softmax_backward     = &detail::cpu::softmax_backward;
        ops.softmax_xent         = &detail::cpu::softmax_xent;
        ops.softmax_xent_segment = &detail::cpu::softmax_xent_segment;
        ops.mse_scalar           = &detail::cpu::mse_scalar;
        ops.add_inplace          = &detail::cpu::add_inplace;
        ops.add_scalar_inplace   = &detail::cpu::add_scalar_inplace;
        ops.xavier_init          = &detail::cpu::xavier_init;

        ops.sgd_step                 = &detail::cpu::sgd_step;
        ops.adam_step                = &detail::cpu::adam_step;
        ops.scale_inplace            = &detail::cpu::scale_inplace;
        ops.layernorm_forward        = &detail::cpu::layernorm_forward;
        ops.layernorm_backward       = &detail::cpu::layernorm_backward;
        ops.attention_forward        = &detail::cpu::attention_forward;
        ops.attention_backward       = &detail::cpu::attention_backward;
        ops.mha_forward              = &detail::cpu::mha_forward;
        ops.mha_backward             = &detail::cpu::mha_backward;
        ops.concat_rows              = &detail::cpu::concat_rows;
        ops.split_rows               = &detail::cpu::split_rows;
        ops.add_inplace_batched      = &detail::cpu::add_inplace_batched;
        ops.linear_forward_batched   = &detail::cpu::linear_forward_batched;
        ops.linear_backward_batched  = &detail::cpu::linear_backward_batched;
        ops.relu_forward_batched     = &detail::cpu::relu_forward_batched;
        ops.relu_backward_batched    = &detail::cpu::relu_backward_batched;
        ops.tanh_forward_batched     = &detail::cpu::tanh_forward_batched;
        ops.tanh_backward_batched    = &detail::cpu::tanh_backward_batched;
        ops.mse_vec_per_sample       = &detail::cpu::mse_vec_per_sample;
        ops.softmax_xent_fused_batched = &detail::cpu::softmax_xent_fused_batched;
        ops.bce_with_logits_fused_batched = &detail::cpu::bce_with_logits_fused_batched;
        ops.masked_mean_pool_forward  = &detail::cpu::masked_mean_pool_forward;
        ops.masked_mean_pool_backward = &detail::cpu::masked_mean_pool_backward;
        ops.build_slot_mask           = &detail::cpu::build_slot_mask;
        ops.copy_d2d                  = &detail::cpu::copy_d2d;

        // ── CHUNK 1 ──
        ops.clamp                      = &detail::cpu::clamp;
        ops.mul_inplace                = &detail::cpu::mul_inplace;
        ops.cast                       = &detail::cpu::cast;
        ops.mse_vec_forward            = &detail::cpu::mse_vec_forward;
        ops.mse_vec_backward           = &detail::cpu::mse_vec_backward;
        ops.softmax_xent_fused         = &detail::cpu::softmax_xent_fused;
        ops.embedding_lookup_forward   = &detail::cpu::embedding_lookup_forward;
        ops.embedding_lookup_backward  = &detail::cpu::embedding_lookup_backward;
        ops.concat_batched_rows        = &detail::cpu::concat_batched_rows;
        ops.concat_nchw_channels       = &detail::cpu::concat_nchw_channels;
        ops.concat_nchw_channels_backward
                                       = &detail::cpu::concat_nchw_channels_backward;
        ops.sum_rows                   = &detail::cpu::sum_rows;
        ops.sum_cols                   = &detail::cpu::sum_cols;
        ops.argmax_rows                = &detail::cpu::argmax_rows;
        ops.layernorm_forward_inference_batched
                                       = &detail::cpu::layernorm_forward_inference_batched;
        ops.layernorm_forward_batched_with_caches
                                       = &detail::cpu::layernorm_forward_batched_with_caches;
        ops.layernorm_backward_batched_with_caches
                                       = &detail::cpu::layernorm_backward_batched_with_caches;
        ops.build_causal_mask_row      = &detail::cpu::build_causal_mask_row;

        // ── CHUNK 2 ──
        ops.silu_forward               = &detail::cpu::silu_forward;
        ops.silu_backward              = &detail::cpu::silu_backward;
        ops.gelu_forward               = &detail::cpu::gelu_forward;
        ops.gelu_backward              = &detail::cpu::gelu_backward;
        ops.gelu_exact_forward         = &detail::cpu::gelu_exact_forward;
        ops.gelu_exact_backward        = &detail::cpu::gelu_exact_backward;
        ops.quick_gelu_forward         = &detail::cpu::quick_gelu_forward;
        ops.quick_gelu_backward        = &detail::cpu::quick_gelu_backward;
        ops.geglu_forward              = &detail::cpu::geglu_forward;
        ops.geglu_backward             = &detail::cpu::geglu_backward;
        ops.geglu_exact_forward        = &detail::cpu::geglu_exact_forward;
        ops.geglu_exact_backward       = &detail::cpu::geglu_exact_backward;
        ops.swiglu_forward             = &detail::cpu::swiglu_forward;
        ops.swiglu_backward            = &detail::cpu::swiglu_backward;
        ops.matmul                     = &detail::cpu::matmul;
        ops.matmul_backward            = &detail::cpu::matmul_backward;
        ops.lstm_forward_train         = &detail::cpu::lstm_forward_train;
        ops.lstm_backward              = &detail::cpu::lstm_backward;
        ops.rope_forward               = &detail::cpu::rope_forward;
        ops.rope_backward              = &detail::cpu::rope_backward;
        ops.rms_norm_forward           = &detail::cpu::rms_norm_forward;
        ops.rms_norm_backward          = &detail::cpu::rms_norm_backward;

        // ── CHUNK 3 ──
        ops.conv2d_forward             = &detail::cpu::conv2d_forward;
        ops.conv2d_backward_input      = &detail::cpu::conv2d_backward_input;
        ops.conv2d_backward_weight     = &detail::cpu::conv2d_backward_weight;
        ops.conv2d_backward_bias       = &detail::cpu::conv2d_backward_bias;
        ops.deform_conv2d_forward      = &detail::cpu::deform_conv2d_forward;
        ops.conv3d_forward             = &detail::cpu::conv3d_forward;
        // conv3d_int8w_fp16_forward is GPU-only; CPU slot left null.
        ops.group_norm_forward         = &detail::cpu::group_norm_forward;
        ops.group_norm_backward        = &detail::cpu::group_norm_backward;

        // ── CHUNK 4 ──
        ops.upsample_nearest_2x          = &detail::cpu::upsample_nearest_2x;
        ops.upsample_bilinear_2x         = &detail::cpu::upsample_bilinear_2x;
        ops.downsample_avg_2x            = &detail::cpu::downsample_avg_2x;
        ops.upsample_nearest_2x_backward = &detail::cpu::upsample_nearest_2x_backward;
        ops.upsample_bilinear_2x_backward
                                         = &detail::cpu::upsample_bilinear_2x_backward;
        ops.downsample_avg_2x_backward   = &detail::cpu::downsample_avg_2x_backward;
        ops.interp2d_forward             = &detail::cpu::interp2d_forward;
        ops.interp2d_backward            = &detail::cpu::interp2d_backward;
        ops.interp2d_align_corners_forward = &detail::cpu::interp2d_align_corners_forward;
        ops.pad2d_forward                = &detail::cpu::pad2d_forward;
        ops.pad2d_backward               = &detail::cpu::pad2d_backward;
        ops.slice2d_forward              = &detail::cpu::slice2d_forward;
        ops.slice2d_backward             = &detail::cpu::slice2d_backward;
        ops.unfold2d_forward             = &detail::cpu::unfold2d_forward;
        ops.l2_normalize_nchw_forward    = &detail::cpu::l2_normalize_nchw_forward;
        ops.convex_upsample_forward      = &detail::cpu::convex_upsample_forward;
        ops.top_k_rows                   = &detail::cpu::top_k_rows;
        ops.adaptive_avg_pool2d_forward  = &detail::cpu::adaptive_avg_pool2d_forward;
        ops.adaptive_avg_pool2d_backward = &detail::cpu::adaptive_avg_pool2d_backward;
        ops.max_pool2d_forward           = &detail::cpu::max_pool2d_forward;
        ops.max_pool2d_backward          = &detail::cpu::max_pool2d_backward;
        ops.gather_rows                  = &detail::cpu::gather_rows;
        ops.scatter_rows_add             = &detail::cpu::scatter_rows_add;
        ops.conv_transpose2d_forward     = &detail::cpu::conv_transpose2d_forward;
        ops.conv_transpose2d_backward_input
                                         = &detail::cpu::conv_transpose2d_backward_input;
        ops.conv_transpose2d_backward_weight
                                         = &detail::cpu::conv_transpose2d_backward_weight;
        ops.conv_transpose2d_backward_bias
                                         = &detail::cpu::conv_transpose2d_backward_bias;
        ops.window_partition_forward     = &detail::cpu::window_partition_forward;
        ops.window_reverse_forward       = &detail::cpu::window_reverse_forward;
        ops.nchw_to_sequence             = &detail::cpu::nchw_to_sequence;
        ops.sequence_to_nchw             = &detail::cpu::sequence_to_nchw;
        ops.ddim_step                    = &detail::cpu::ddim_step;
        ops.euler_step                   = &detail::cpu::euler_step;
        ops.dpmpp_2m_step                = &detail::cpu::dpmpp_2m_step;
        ops.timestep_embedding           = &detail::cpu::timestep_embedding;
        ops.kv_cache_append              = &detail::cpu::kv_cache_append;
        ops.flash_attention_decode       = &detail::cpu::flash_attention_decode;

        // ── CHUNK 5 ──
        ops.cross_attention_forward      = &detail::cpu::cross_attention_forward;
        ops.cross_attention_forward_with_attn
                                         = &detail::cpu::cross_attention_forward_with_attn;
        ops.self_attention_forward_train = &detail::cpu::self_attention_forward_train;
        ops.self_attention_backward      = &detail::cpu::self_attention_backward;
        ops.attention_token_moments      = &detail::cpu::attention_token_moments;
        ops.cross_attention_forward_train
                                         = &detail::cpu::cross_attention_forward_train;
        ops.cross_attention_backward     = &detail::cpu::cross_attention_backward;

        // ── CHUNK 6 ──
        ops.flash_attention_forward      = &detail::cpu::flash_attention_forward;
        ops.flash_attention_windowed_forward
                                         = &detail::cpu::flash_attention_windowed_forward;
        ops.flash_attention_varlen_forward
                                         = &detail::cpu::flash_attention_varlen_forward;
        ops.flash_attention_varlen_backward
                                         = &detail::cpu::flash_attention_varlen_backward;
        ops.flash_attention_qkvo_forward = &detail::cpu::flash_attention_qkvo_forward;
        ops.flash_attention_qkvo_backward
                                         = &detail::cpu::flash_attention_qkvo_backward;
        ops.flash_attention_backward     = &detail::cpu::flash_attention_backward;
        ops.flash_attention_project_kv   = &detail::cpu::flash_attention_project_kv;
        ops.flash_attention_q_with_kv_cached_forward
                                         = &detail::cpu::flash_attention_q_with_kv_cached_forward;
        ops.self_attention_forward       = &detail::cpu::self_attention_forward;
        ops.resblock_forward             = &detail::cpu::resblock_forward;
        ops.resblock_backward            = &detail::cpu::resblock_backward;

        // ── DiT / diffusion extras ──
        ops.modulate                     = &detail::cpu::modulate;
        ops.broadcast_mul                = &detail::cpu::broadcast_mul;
        ops.rope_apply                   = &detail::cpu::rope_apply;
        ops.rope_apply_perhead           = &detail::cpu::rope_apply_perhead;
        ops.rope_apply_backward          = &detail::cpu::rope_apply_backward;
        ops.self_attention_bias_forward  = &detail::cpu::self_attention_bias_forward;
        ops.self_attention_decomposed_rel_pos_forward =
            &detail::cpu::self_attention_decomposed_rel_pos_forward;
        ops.self_attention_decomposed_rel_pos_windowed_forward =
            &detail::cpu::self_attention_decomposed_rel_pos_windowed_forward;

        // ── Spectral / FFT core (brosoundml) ──
        ops.complex_mul                  = &detail::cpu::complex_mul;
        ops.complex_mul_backward         = &detail::cpu::complex_mul_backward;
        ops.complex_abs                  = &detail::cpu::complex_abs;
        ops.complex_abs_backward         = &detail::cpu::complex_abs_backward;
        ops.complex_angle                = &detail::cpu::complex_angle;
        ops.complex_from_polar           = &detail::cpu::complex_from_polar;
        ops.fft                          = &detail::cpu::fft;
        ops.ifft                         = &detail::cpu::ifft;
        ops.rfft                         = &detail::cpu::rfft;
        ops.irfft                        = &detail::cpu::irfft;
        ops.rfft_backward                = &detail::cpu::rfft_backward;
        ops.irfft_backward               = &detail::cpu::irfft_backward;

        // ── STFT / iSTFT (brosoundml) ──
        ops.stft                         = &detail::cpu::stft;
        ops.stft_backward                = &detail::cpu::stft_backward;
        ops.istft                        = &detail::cpu::istft;
        ops.istft_backward               = &detail::cpu::istft_backward;

        // ── 1D convolution family (brosoundml) ──
        ops.conv_transpose1d_forward     = &detail::cpu::conv_transpose1d_forward;
        ops.conv_transpose1d_backward_input
                                         = &detail::cpu::conv_transpose1d_backward_input;
        ops.conv_transpose1d_backward_weight
                                         = &detail::cpu::conv_transpose1d_backward_weight;
        ops.conv_transpose1d_backward_bias
                                         = &detail::cpu::conv_transpose1d_backward_bias;
        ops.causal_conv1d_update         = &detail::cpu::causal_conv1d_update;
        ops.pad1d_forward                = &detail::cpu::pad1d_forward;
        ops.pad1d_backward               = &detail::cpu::pad1d_backward;

        // ── Vocoder / codec activations (brosoundml) ──
        ops.snake_forward                = &detail::cpu::snake_forward;
        ops.snake_backward               = &detail::cpu::snake_backward;
        ops.elu_forward                  = &detail::cpu::elu_forward;
        ops.elu_backward                 = &detail::cpu::elu_backward;
        ops.leaky_relu_forward           = &detail::cpu::leaky_relu_forward;
        ops.leaky_relu_backward          = &detail::cpu::leaky_relu_backward;

        // ── Codec quantization (brosoundml) ──
        ops.vq_encode_forward            = &detail::cpu::vq_encode_forward;
        ops.vq_encode_backward           = &detail::cpu::vq_encode_backward;
        ops.fsq_quantize_forward         = &detail::cpu::fsq_quantize_forward;
        ops.fsq_quantize_backward        = &detail::cpu::fsq_quantize_backward;

        // ── 1D resampling (brosoundml CHUNK 6, family E) ──
        ops.resample1d_forward           = &detail::cpu::resample1d_forward;
        ops.resample1d_backward          = &detail::cpu::resample1d_backward;

        // ── log / exp / round elementwise (brosoundml CHUNK 6, family G) ──
        ops.log_forward                  = &detail::cpu::log_forward;
        ops.log_backward                 = &detail::cpu::log_backward;
        ops.exp_forward                  = &detail::cpu::exp_forward;
        ops.exp_backward                 = &detail::cpu::exp_backward;
        ops.round_forward                = &detail::cpu::round_forward;
        ops.round_backward               = &detail::cpu::round_backward;

        // ── Autoregressive logit sampling (brosoundml CHUNK 7, family F) ──
        ops.sample_logits                = &detail::cpu::sample_logits;

        // ── Counter-based noise generation (Philox 4x32-10) ──
        ops.randn                        = &detail::cpu::randn;
        ops.rand_uniform                 = &detail::cpu::rand_uniform;
        ops.rand_bernoulli               = &detail::cpu::rand_bernoulli;
        ops.randn_truncated              = &detail::cpu::randn_truncated;

        // ── L2 norm + Gated Delta Rule (linear-attention text path) ──
        ops.l2_norm_forward              = &detail::cpu::l2_norm_forward;
        ops.l2_norm_backward             = &detail::cpu::l2_norm_backward;
        ops.gated_delta_rule_chunked     = &detail::cpu::gated_delta_rule_chunked;
        ops.gated_delta_rule_step        = &detail::cpu::gated_delta_rule_step;

        // ── Qwen3-VL polish: spatial_merge_2x2 + M-RoPE ──
        ops.spatial_merge_2x2_forward    = &detail::cpu::spatial_merge_2x2_forward;
        ops.rope_apply_mrope             = &detail::cpu::rope_apply_mrope;

        // ── BatchNorm (vision backbones) ──
        ops.batch_norm_forward           = &detail::cpu::batch_norm_forward;
        ops.batch_norm_inference         = &detail::cpu::batch_norm_inference;
        ops.batch_norm_backward          = &detail::cpu::batch_norm_backward;

        // ── Image preprocessing helpers ──
        ops.image_normalize              = &detail::cpu::image_normalize;
        ops.image_u8_to_f32_nhwc_to_nchw = &detail::cpu::image_u8_to_f32_nhwc_to_nchw;

        // ── StyleGAN3-R synthesis-input primitives ──
        ops.sin_forward                  = &detail::cpu::sin_forward;
        ops.sin_backward                 = &detail::cpu::sin_backward;
        ops.cos_forward                  = &detail::cpu::cos_forward;
        ops.cos_backward                 = &detail::cpu::cos_backward;
        ops.rsqrt_forward                = &detail::cpu::rsqrt_forward;
        ops.rsqrt_backward               = &detail::cpu::rsqrt_backward;
        ops.pixel_norm_forward           = &detail::cpu::pixel_norm_forward;
        ops.pixel_norm_backward          = &detail::cpu::pixel_norm_backward;
        ops.bias_act_forward             = &detail::cpu::bias_act_forward;
        ops.bias_act_backward            = &detail::cpu::bias_act_backward;
        ops.upfirdn2d_forward            = &detail::cpu::upfirdn2d_forward;
        ops.upfirdn2d_backward           = &detail::cpu::upfirdn2d_backward;
        ops.modulated_conv2d_forward     = &detail::cpu::modulated_conv2d_forward;
        ops.modulated_conv2d_backward    = &detail::cpu::modulated_conv2d_backward;

        detail::register_backend(Device::CPU, ops,
                                 detail::cpu::cpu_alloc_table());
    }
};

static CpuStaticRegistrar g_cpu_registrar{};

} // anonymous namespace
