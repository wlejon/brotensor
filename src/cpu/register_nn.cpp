// ─── CPU backend registration (Part 2: Attention, Spectral, Vision & Quant) ───
//
// Populates the remaining CPU OpsVTable slots from `register_cpu_nn_ops`,
// called by `CpuStaticRegistrar` in `src/cpu/register.cpp`.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brotensor::detail::cpu {

// ── packed_encoder.cpp ──
void flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV,
                                        const ::brotensor::Tensor& seq_bounds,
                                        int num_heads, int window,
                                        ::brotensor::Tensor& O);
void flash_attention_packed_qkv_backward(const ::brotensor::Tensor& QKV, const ::brotensor::Tensor& dO,
                                         const ::brotensor::Tensor& seq_bounds, int num_heads, int window,
                                         ::brotensor::Tensor& dQKV);
void rope_qkv_packed_inplace(::brotensor::Tensor& QKV, const ::brotensor::Tensor& cos_tbl,
                             const ::brotensor::Tensor& sin_tbl, const ::brotensor::Tensor& pos,
                             int num_heads, int head_dim);
void segment_softmax_stats(const ::brotensor::Tensor& logits,
                           const ::brotensor::Tensor& seg_offsets,
                           ::brotensor::Tensor& out);
void linear_forward_batched_ex(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                               const ::brotensor::Tensor& X, int act, int epilogue,
                               ::brotensor::Tensor* workspace, ::brotensor::Tensor& Y);
void quantize_rows(const ::brotensor::Tensor& W, ::brotensor::Dtype fmt, ::brotensor::Tensor& Wq,
                   ::brotensor::Tensor& scales);
void linear_forward_batched_wq_ex(const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& scales,
                                  const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
                                  int epilogue, ::brotensor::Tensor& Y);
void dequantize_rows(const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& scales, ::brotensor::Dtype out,
                     ::brotensor::Tensor& W);

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
                                      ::brotensor::Tensor& O, bool causal);
void flash_attention_gqa_forward(const ::brotensor::Tensor& Q,
                                 const ::brotensor::Tensor& K,
                                 const ::brotensor::Tensor& V,
                                 const float* d_mask, int num_q_heads, int num_kv_heads,
                                 bool causal, ::brotensor::Tensor& O);
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
void rel_pos_bias_xl_forward(const ::brotensor::Tensor& Qv,
                             const ::brotensor::Tensor& Pk,
                             int num_heads, int head_dim,
                             ::brotensor::Tensor& Bias);

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
void sample_logits_into(const ::brotensor::Tensor& logits, float temperature,
                        int top_k, float top_p, uint64_t key,
                        ::brotensor::Tensor& counter,
                        ::brotensor::Tensor& scratch,
                        ::brotensor::Tensor& indices);

// ── Masked-diffusion token selection — masked_diffusion.cpp ──
void masked_diffusion_scores(const ::brotensor::Tensor& logits,
                             const ::brotensor::Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, uint64_t seed,
                             ::brotensor::Tensor& pred, ::brotensor::Tensor& scores,
                             ::brotensor::Tensor& confidence);
void masked_diffusion_commit(const ::brotensor::Tensor& pred,
                             const ::brotensor::Tensor& idx, int k, int step,
                             ::brotensor::Tensor& tokens,
                             ::brotensor::Tensor& unmask_step);

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
void pixel_shuffle_upsample_2x_forward(const ::brotensor::Tensor& X,
                                       int N, int C_in, int H, int W,
                                       int C_out, ::brotensor::Tensor& Y);
void patch_unpack_forward(const ::brotensor::Tensor& tokens,
                          int hp, int wp, int P, int C_total, int C_keep,
                          bool channel_major, ::brotensor::Tensor& Y);
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

// ── Quantized linear forward & dequant (Q8_0, Q4_K, Q6_K & INT8) via Brass AVX2/FMA JIT ──
void dequant_q8_0_to_fp16(const ::brotensor::Tensor& W_q8, ::brotensor::Tensor& W_fp16);
void linear_forward_q8_0_fp16(const ::brotensor::Tensor& W_q8, const ::brotensor::Tensor* bias,
                              const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_forward_batched_q8_0_fp16(const ::brotensor::Tensor& W_q8, const ::brotensor::Tensor* bias,
                                      const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD);
void dequant_q4k_to_fp16(const ::brotensor::Tensor& W_q4k, ::brotensor::Tensor& W_fp16);
void linear_forward_q4k_fp16(const ::brotensor::Tensor& W_q4k, const ::brotensor::Tensor* bias,
                             const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_forward_batched_q4k_fp16(const ::brotensor::Tensor& W_q4k, const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD);
void dequant_q6k_to_fp16(const ::brotensor::Tensor& W_q6k, ::brotensor::Tensor& W_fp16);
void linear_forward_q6k_fp16(const ::brotensor::Tensor& W_q6k, const ::brotensor::Tensor* bias,
                             const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_forward_batched_q6k_fp16(const ::brotensor::Tensor& W_q6k, const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD, ::brotensor::Tensor& Y_BD);
void matmul_int8w_fp16(const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,
                       const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void linear_forward_batched_int8w_fp16(const ::brotensor::Tensor& W_int8, const ::brotensor::Tensor& scales,
                                       const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X_BD,
                                       ::brotensor::Tensor& Y_BD);

void register_cpu_nn_ops(::brotensor::detail::OpsVTable& ops) {
    // ── LSTM ──
    ops.lstm_forward_train         = &lstm_forward_train;
    ops.lstm_backward              = &lstm_backward;

    // ── CHUNK 5 ──
    ops.cross_attention_forward      = &cross_attention_forward;
    ops.cross_attention_forward_with_attn
                                     = &cross_attention_forward_with_attn;
    ops.self_attention_forward_train = &self_attention_forward_train;
    ops.self_attention_backward      = &self_attention_backward;
    ops.attention_token_moments      = &attention_token_moments;
    ops.cross_attention_forward_train
                                     = &cross_attention_forward_train;
    ops.cross_attention_backward     = &cross_attention_backward;

    // ── CHUNK 6 ──
    ops.flash_attention_forward      = &flash_attention_forward;
    ops.flash_attention_gqa_forward  = &flash_attention_gqa_forward;
    ops.flash_attention_windowed_forward
                                     = &flash_attention_windowed_forward;
    ops.flash_attention_varlen_forward
                                     = &flash_attention_varlen_forward;
    ops.flash_attention_varlen_backward
                                     = &flash_attention_varlen_backward;
    ops.flash_attention_packed_qkv_forward
                                     = &flash_attention_packed_qkv_forward;
    ops.flash_attention_packed_qkv_backward
                                     = &flash_attention_packed_qkv_backward;
    ops.rope_qkv_packed_inplace      = &rope_qkv_packed_inplace;
    ops.segment_softmax_stats        = &segment_softmax_stats;
    ops.linear_forward_batched_ex    = &linear_forward_batched_ex;
    ops.quantize_rows                = &quantize_rows;
    ops.dequantize_rows              = &dequantize_rows;
    ops.linear_forward_batched_wq_ex = &linear_forward_batched_wq_ex;
    ops.flash_attention_qkvo_forward = &flash_attention_qkvo_forward;
    ops.flash_attention_qkvo_backward
                                     = &flash_attention_qkvo_backward;
    ops.flash_attention_backward     = &flash_attention_backward;
    ops.flash_attention_project_kv   = &flash_attention_project_kv;
    ops.flash_attention_q_with_kv_cached_forward
                                     = &flash_attention_q_with_kv_cached_forward;
    ops.self_attention_forward       = &self_attention_forward;
    ops.resblock_forward             = &resblock_forward;
    ops.resblock_backward            = &resblock_backward;

    // ── DiT / diffusion extras ──
    ops.modulate                     = &modulate;
    ops.broadcast_mul                = &broadcast_mul;
    ops.rope_apply                   = &rope_apply;
    ops.rope_apply_perhead           = &rope_apply_perhead;
    ops.rope_apply_backward          = &rope_apply_backward;
    ops.self_attention_bias_forward  = &self_attention_bias_forward;
    ops.rel_pos_bias_xl_forward      = &rel_pos_bias_xl_forward;
    ops.self_attention_decomposed_rel_pos_forward =
        &self_attention_decomposed_rel_pos_forward;
    ops.self_attention_decomposed_rel_pos_windowed_forward =
        &self_attention_decomposed_rel_pos_windowed_forward;

    // ── Spectral / FFT core (brosoundml) ──
    ops.complex_mul                  = &complex_mul;
    ops.complex_mul_backward         = &complex_mul_backward;
    ops.complex_abs                  = &complex_abs;
    ops.complex_abs_backward         = &complex_abs_backward;
    ops.complex_angle                = &complex_angle;
    ops.complex_from_polar           = &complex_from_polar;
    ops.fft                          = &fft;
    ops.ifft                         = &ifft;
    ops.rfft                         = &rfft;
    ops.irfft                        = &irfft;
    ops.rfft_backward                = &rfft_backward;
    ops.irfft_backward               = &irfft_backward;

    // ── STFT / iSTFT (brosoundml) ──
    ops.stft                         = &stft;
    ops.stft_backward                = &stft_backward;
    ops.istft                        = &istft;
    ops.istft_backward               = &istft_backward;

    // ── 1D convolution family (brosoundml) ──
    ops.conv_transpose1d_forward     = &conv_transpose1d_forward;
    ops.conv_transpose1d_backward_input
                                     = &conv_transpose1d_backward_input;
    ops.conv_transpose1d_backward_weight
                                     = &conv_transpose1d_backward_weight;
    ops.conv_transpose1d_backward_bias
                                     = &conv_transpose1d_backward_bias;
    ops.causal_conv1d_update         = &causal_conv1d_update;
    ops.pad1d_forward                = &pad1d_forward;
    ops.pad1d_backward               = &pad1d_backward;

    // ── Vocoder / codec activations (brosoundml) ──
    ops.snake_forward                = &snake_forward;
    ops.snake_backward               = &snake_backward;
    ops.elu_forward                  = &elu_forward;
    ops.elu_backward                 = &elu_backward;
    ops.leaky_relu_forward           = &leaky_relu_forward;
    ops.leaky_relu_backward          = &leaky_relu_backward;

    // ── Codec quantization (brosoundml) ──
    ops.vq_encode_forward            = &vq_encode_forward;
    ops.vq_encode_backward           = &vq_encode_backward;
    ops.fsq_quantize_forward         = &fsq_quantize_forward;
    ops.fsq_quantize_backward        = &fsq_quantize_backward;

    // ── 1D resampling (brosoundml CHUNK 6, family E) ──
    ops.resample1d_forward           = &resample1d_forward;
    ops.resample1d_backward          = &resample1d_backward;

    // ── log / exp / round elementwise (brosoundml CHUNK 6, family G) ──
    ops.log_forward                  = &log_forward;
    ops.log_backward                 = &log_backward;
    ops.exp_forward                  = &exp_forward;
    ops.exp_backward                 = &exp_backward;
    ops.round_forward                = &round_forward;
    ops.round_backward               = &round_backward;

    // ── Autoregressive logit sampling (brosoundml CHUNK 7, family F) ──
    ops.sample_logits                = &sample_logits;
    ops.sample_logits_into           = &sample_logits_into;

    // ── Masked-diffusion token selection ──
    ops.masked_diffusion_scores      = &masked_diffusion_scores;
    ops.masked_diffusion_commit      = &masked_diffusion_commit;

    // ── Counter-based noise generation (Philox 4x32-10) ──
    ops.randn                        = &randn;
    ops.rand_uniform                 = &rand_uniform;
    ops.rand_bernoulli               = &rand_bernoulli;
    ops.randn_truncated              = &randn_truncated;

    // ── L2 norm + Gated Delta Rule (linear-attention text path) ──
    ops.l2_norm_forward              = &l2_norm_forward;
    ops.l2_norm_backward             = &l2_norm_backward;
    ops.gated_delta_rule_chunked     = &gated_delta_rule_chunked;
    ops.gated_delta_rule_step        = &gated_delta_rule_step;

    // ── Qwen3-VL polish: spatial_merge_2x2 + M-RoPE ──
    ops.spatial_merge_2x2_forward    = &spatial_merge_2x2_forward;
    ops.pixel_shuffle_upsample_2x_forward = &pixel_shuffle_upsample_2x_forward;
    ops.patch_unpack_forward         = &patch_unpack_forward;
    ops.rope_apply_mrope             = &rope_apply_mrope;

    // ── BatchNorm (vision backbones) ──
    ops.batch_norm_forward           = &batch_norm_forward;
    ops.batch_norm_inference         = &batch_norm_inference;
    ops.batch_norm_backward          = &batch_norm_backward;

    // ── Image preprocessing helpers ──
    ops.image_normalize              = &image_normalize;
    ops.image_u8_to_f32_nhwc_to_nchw = &image_u8_to_f32_nhwc_to_nchw;

    // ── StyleGAN3-R synthesis-input primitives ──
    ops.sin_forward                  = &sin_forward;
    ops.sin_backward                 = &sin_backward;
    ops.cos_forward                  = &cos_forward;
    ops.cos_backward                 = &cos_backward;
    ops.rsqrt_forward                = &rsqrt_forward;
    ops.rsqrt_backward               = &rsqrt_backward;
    ops.pixel_norm_forward           = &pixel_norm_forward;
    ops.pixel_norm_backward          = &pixel_norm_backward;
    ops.bias_act_forward             = &bias_act_forward;
    ops.bias_act_backward            = &bias_act_backward;
    ops.upfirdn2d_forward            = &upfirdn2d_forward;
    ops.upfirdn2d_backward           = &upfirdn2d_backward;
    ops.modulated_conv2d_forward     = &modulated_conv2d_forward;
    ops.modulated_conv2d_backward    = &modulated_conv2d_backward;

    // ── Quantized linear forward & dequant (Q8_0, Q4_K, Q6_K & INT8) via Brass AVX2/FMA JIT ──
    ops.dequant_q8_0_to_fp16         = &dequant_q8_0_to_fp16;
    ops.linear_forward_q8_0_fp16     = &linear_forward_q8_0_fp16;
    ops.linear_forward_batched_q8_0_fp16 = &linear_forward_batched_q8_0_fp16;
    ops.dequant_q4k_to_fp16          = &dequant_q4k_to_fp16;
    ops.linear_forward_q4k_fp16      = &linear_forward_q4k_fp16;
    ops.linear_forward_batched_q4k_fp16 = &linear_forward_batched_q4k_fp16;
    ops.dequant_q6k_to_fp16          = &dequant_q6k_to_fp16;
    ops.linear_forward_q6k_fp16      = &linear_forward_q6k_fp16;
    ops.linear_forward_batched_q6k_fp16 = &linear_forward_batched_q6k_fp16;
    ops.matmul_int8w_fp16            = &matmul_int8w_fp16;
    ops.linear_forward_batched_int8w_fp16 = &linear_forward_batched_int8w_fp16;
}

} // namespace brotensor::detail::cpu
