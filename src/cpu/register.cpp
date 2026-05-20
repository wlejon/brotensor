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
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
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

// ── CHUNK 1 — elementwise.cpp / loss.cpp / embedding.cpp / concat.cpp /
//    public_reductions.cpp / layernorm_inference.cpp ──
void clamp(::brotensor::Tensor& y, float lo, float hi);
void mul_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);
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
        ops.masked_mean_pool_forward  = &detail::cpu::masked_mean_pool_forward;
        ops.masked_mean_pool_backward = &detail::cpu::masked_mean_pool_backward;
        ops.build_slot_mask           = &detail::cpu::build_slot_mask;
        ops.copy_d2d                  = &detail::cpu::copy_d2d;

        // ── CHUNK 1 ──
        ops.clamp                      = &detail::cpu::clamp;
        ops.mul_inplace                = &detail::cpu::mul_inplace;
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
        ops.rope_forward               = &detail::cpu::rope_forward;
        ops.rope_backward              = &detail::cpu::rope_backward;
        ops.rms_norm_forward           = &detail::cpu::rms_norm_forward;
        ops.rms_norm_backward          = &detail::cpu::rms_norm_backward;

        detail::register_backend(Device::CPU, ops,
                                 detail::cpu::cpu_alloc_table());
    }
};

static CpuStaticRegistrar g_cpu_registrar{};

} // anonymous namespace
