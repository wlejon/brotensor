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

        detail::register_backend(Device::CPU, ops,
                                 detail::cpu::cpu_alloc_table());
    }
};

static CpuStaticRegistrar g_cpu_registrar{};

} // anonymous namespace
