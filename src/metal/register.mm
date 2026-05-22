// Metal backend master registration.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_METAL is
// defined. Probes for a Metal device; on success initialises the runtime
// (MTLDevice + command queue + precompiled MSL pipelines), builds the
// OpsVTable from the per-op implementations across the Metal TUs, pairs it
// with the Metal AllocVTable, and hands the pair to the dispatcher.
//
// A null slot in the vtable means "this op is not implemented on Metal" — the
// dispatcher throws on null lookups. Metal does not implement the host-scalar
// ops `mse_scalar`, `softmax_xent`, `softmax_xent_segment`, or `xavier_init`;
// those slots stay null.

#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/op_table.h>
#include <brotensor/tensor.h>

#import "internal.h"

namespace brotensor::detail::metal {

// Forward-declare every public op in the Metal backend namespace. The op
// table is the single source of truth for the signatures; the implementations
// live one-per-cluster across the Metal .mm TUs. (The four ops Metal does not
// implement are declared here too but never defined or referenced — harmless.)
#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

// Defined in tensor.mm.
const ::brotensor::detail::AllocVTable& metal_alloc_table();

} // namespace brotensor::detail::metal

extern "C" void brotensor_probe_and_register_metal() {
    // Probe: a Mac without a usable Metal device should not register the
    // backend. Mirror register.cu — return quietly rather than throw so a
    // headless / unsupported host can still load a Metal-enabled binary.
    @autoreleasepool {
        id<MTLDevice> probe = MTLCreateSystemDefaultDevice();
        if (!probe) return;
    }

    // Bring up the device, command queue, and precompiled MSL pipelines.
    ::brotensor::cuda_init();

    using ::brotensor::Device;
    using ::brotensor::detail::OpsVTable;
    namespace dm = ::brotensor::detail::metal;

    OpsVTable ops{};   // zero-init: every slot starts as nullptr

    ops.adam_step                                   = &dm::adam_step;
    ops.add_inplace                                 = &dm::add_inplace;
    ops.add_inplace_batched                         = &dm::add_inplace_batched;
    ops.add_scalar_inplace                          = &dm::add_scalar_inplace;
    ops.argmax_rows                                 = &dm::argmax_rows;
    ops.attention_backward                          = &dm::attention_backward;
    ops.attention_forward                           = &dm::attention_forward;
    ops.attention_token_moments                     = &dm::attention_token_moments;
    ops.broadcast_mul                               = &dm::broadcast_mul;
    ops.build_causal_mask_row                       = &dm::build_causal_mask_row;
    ops.build_slot_mask                             = &dm::build_slot_mask;
    ops.cast                                        = &dm::cast;
    ops.causal_conv1d_update                        = &dm::causal_conv1d_update;
    ops.clamp                                       = &dm::clamp;
    ops.complex_abs                                 = &dm::complex_abs;
    ops.complex_abs_backward                        = &dm::complex_abs_backward;
    ops.complex_angle                               = &dm::complex_angle;
    ops.complex_from_polar                          = &dm::complex_from_polar;
    ops.complex_mul                                 = &dm::complex_mul;
    ops.complex_mul_backward                        = &dm::complex_mul_backward;
    ops.concat_batched_rows                         = &dm::concat_batched_rows;
    ops.concat_nchw_channels                        = &dm::concat_nchw_channels;
    ops.concat_nchw_channels_backward               = &dm::concat_nchw_channels_backward;
    ops.concat_rows                                 = &dm::concat_rows;
    ops.conv2d_backward_bias                        = &dm::conv2d_backward_bias;
    ops.conv2d_backward_input                       = &dm::conv2d_backward_input;
    ops.conv2d_backward_weight                      = &dm::conv2d_backward_weight;
    ops.conv2d_forward                              = &dm::conv2d_forward;
    ops.conv2d_int8w_fp16_forward                   = &dm::conv2d_int8w_fp16_forward;
    ops.conv_transpose1d_backward_bias              = &dm::conv_transpose1d_backward_bias;
    ops.conv_transpose1d_backward_input             = &dm::conv_transpose1d_backward_input;
    ops.conv_transpose1d_backward_weight            = &dm::conv_transpose1d_backward_weight;
    ops.conv_transpose1d_forward                    = &dm::conv_transpose1d_forward;
    ops.copy_d2d                                    = &dm::copy_d2d;
    ops.cross_attention_backward                    = &dm::cross_attention_backward;
    ops.cross_attention_forward                     = &dm::cross_attention_forward;
    ops.cross_attention_forward_train               = &dm::cross_attention_forward_train;
    ops.cross_attention_forward_with_attn           = &dm::cross_attention_forward_with_attn;
    ops.ddim_step                                   = &dm::ddim_step;
    ops.downsample_avg_2x                           = &dm::downsample_avg_2x;
    ops.downsample_avg_2x_backward                  = &dm::downsample_avg_2x_backward;
    ops.dpmpp_2m_step                               = &dm::dpmpp_2m_step;
    ops.elu_backward                                = &dm::elu_backward;
    ops.elu_forward                                 = &dm::elu_forward;
    ops.embedding_lookup_backward                   = &dm::embedding_lookup_backward;
    ops.embedding_lookup_forward                    = &dm::embedding_lookup_forward;
    ops.euler_step                                  = &dm::euler_step;
    ops.fft                                         = &dm::fft;
    ops.flash_attention_backward                    = &dm::flash_attention_backward;
    ops.flash_attention_decode                      = &dm::flash_attention_decode;
    ops.flash_attention_forward                     = &dm::flash_attention_forward;
    ops.flash_attention_project_kv                  = &dm::flash_attention_project_kv;
    ops.flash_attention_project_kv_int8w_fp16       = &dm::flash_attention_project_kv_int8w_fp16;
    ops.flash_attention_q_with_kv_cached_forward    = &dm::flash_attention_q_with_kv_cached_forward;
    ops.flash_attention_q_with_kv_cached_int8w_fp16 = &dm::flash_attention_q_with_kv_cached_int8w_fp16;
    ops.flash_attention_qkvo_backward               = &dm::flash_attention_qkvo_backward;
    ops.flash_attention_qkvo_forward                = &dm::flash_attention_qkvo_forward;
    ops.flash_attention_qkvo_int8w_fp16             = &dm::flash_attention_qkvo_int8w_fp16;
    ops.geglu_backward                              = &dm::geglu_backward;
    ops.geglu_exact_backward                        = &dm::geglu_exact_backward;
    ops.geglu_exact_forward                         = &dm::geglu_exact_forward;
    ops.geglu_forward                               = &dm::geglu_forward;
    ops.gelu_backward                               = &dm::gelu_backward;
    ops.gelu_exact_backward                         = &dm::gelu_exact_backward;
    ops.gelu_exact_forward                          = &dm::gelu_exact_forward;
    ops.gelu_forward                                = &dm::gelu_forward;
    ops.group_norm_backward                         = &dm::group_norm_backward;
    ops.group_norm_forward                          = &dm::group_norm_forward;
    ops.ifft                                        = &dm::ifft;
    ops.irfft                                       = &dm::irfft;
    ops.irfft_backward                              = &dm::irfft_backward;
    ops.istft                                       = &dm::istft;
    ops.istft_backward                              = &dm::istft_backward;
    ops.kv_cache_append                             = &dm::kv_cache_append;
    ops.layernorm_backward                          = &dm::layernorm_backward;
    ops.layernorm_forward                           = &dm::layernorm_forward;
    ops.layernorm_forward_inference_batched         = &dm::layernorm_forward_inference_batched;
    ops.layernorm_forward_inference_batched_fp16    = &dm::layernorm_forward_inference_batched_fp16;
    ops.leaky_relu_backward                         = &dm::leaky_relu_backward;
    ops.leaky_relu_forward                          = &dm::leaky_relu_forward;
    ops.linear_backward                             = &dm::linear_backward;
    ops.linear_backward_batched                     = &dm::linear_backward_batched;
    ops.linear_forward                              = &dm::linear_forward;
    ops.linear_forward_batched                      = &dm::linear_forward_batched;
    ops.linear_forward_batched_fp16                 = &dm::linear_forward_batched_fp16;
    ops.linear_forward_batched_int8w_fp16           = &dm::linear_forward_batched_int8w_fp16;
    ops.masked_mean_pool_backward                   = &dm::masked_mean_pool_backward;
    ops.masked_mean_pool_forward                    = &dm::masked_mean_pool_forward;
    ops.matmul                                      = &dm::matmul;
    ops.matmul_backward                             = &dm::matmul_backward;
    ops.matmul_int8w_fp16                           = &dm::matmul_int8w_fp16;
    ops.mha_backward                                = &dm::mha_backward;
    ops.mha_forward                                 = &dm::mha_forward;
    ops.modulate                                    = &dm::modulate;
    ops.mse_vec_backward                            = &dm::mse_vec_backward;
    ops.mse_vec_forward                             = &dm::mse_vec_forward;
    ops.mse_vec_per_sample                          = &dm::mse_vec_per_sample;
    ops.mul_inplace                                 = &dm::mul_inplace;
    ops.nchw_to_sequence                            = &dm::nchw_to_sequence;
    ops.pad1d_backward                              = &dm::pad1d_backward;
    ops.pad1d_forward                               = &dm::pad1d_forward;
    ops.quick_gelu_backward                         = &dm::quick_gelu_backward;
    ops.quick_gelu_forward                          = &dm::quick_gelu_forward;
    ops.relu_backward                               = &dm::relu_backward;
    ops.relu_backward_batched                       = &dm::relu_backward_batched;
    ops.relu_forward                                = &dm::relu_forward;
    ops.relu_forward_batched                        = &dm::relu_forward_batched;
    ops.resblock_backward                           = &dm::resblock_backward;
    ops.resblock_forward                            = &dm::resblock_forward;
    ops.resblock_forward_int8w_fp16                 = &dm::resblock_forward_int8w_fp16;
    ops.rfft                                        = &dm::rfft;
    ops.rfft_backward                               = &dm::rfft_backward;
    ops.rms_norm_backward                           = &dm::rms_norm_backward;
    ops.rms_norm_forward                            = &dm::rms_norm_forward;
    ops.rope_apply                                  = &dm::rope_apply;
    ops.rope_apply_backward                         = &dm::rope_apply_backward;
    ops.rope_backward                               = &dm::rope_backward;
    ops.rope_forward                                = &dm::rope_forward;
    ops.scale_inplace                               = &dm::scale_inplace;
    ops.self_attention_backward                     = &dm::self_attention_backward;
    ops.self_attention_bias_forward                 = &dm::self_attention_bias_forward;
    ops.self_attention_bias_int8w_fp16              = &dm::self_attention_bias_int8w_fp16;
    ops.self_attention_forward                      = &dm::self_attention_forward;
    ops.self_attention_forward_train                = &dm::self_attention_forward_train;
    ops.sequence_to_nchw                            = &dm::sequence_to_nchw;
    ops.sgd_step                                    = &dm::sgd_step;
    ops.sigmoid_backward                            = &dm::sigmoid_backward;
    ops.sigmoid_forward                             = &dm::sigmoid_forward;
    ops.silu_backward                               = &dm::silu_backward;
    ops.silu_forward                                = &dm::silu_forward;
    ops.snake_backward                              = &dm::snake_backward;
    ops.snake_forward                               = &dm::snake_forward;
    ops.softmax_backward                            = &dm::softmax_backward;
    ops.softmax_forward                             = &dm::softmax_forward;
    ops.softmax_xent_fused                          = &dm::softmax_xent_fused;
    ops.softmax_xent_fused_batched                  = &dm::softmax_xent_fused_batched;
    ops.split_rows                                  = &dm::split_rows;
    ops.stft                                        = &dm::stft;
    ops.stft_backward                               = &dm::stft_backward;
    ops.sum_cols                                    = &dm::sum_cols;
    ops.sum_rows                                    = &dm::sum_rows;
    ops.swiglu_backward                             = &dm::swiglu_backward;
    ops.swiglu_forward                              = &dm::swiglu_forward;
    ops.tanh_backward                               = &dm::tanh_backward;
    ops.tanh_backward_batched                       = &dm::tanh_backward_batched;
    ops.tanh_forward                                = &dm::tanh_forward;
    ops.tanh_forward_batched                        = &dm::tanh_forward_batched;
    ops.timestep_embedding                          = &dm::timestep_embedding;
    ops.upsample_bilinear_2x                        = &dm::upsample_bilinear_2x;
    ops.upsample_bilinear_2x_backward               = &dm::upsample_bilinear_2x_backward;
    ops.upsample_nearest_2x                         = &dm::upsample_nearest_2x;
    ops.upsample_nearest_2x_backward                = &dm::upsample_nearest_2x_backward;

    ::brotensor::detail::register_backend(Device::Metal, ops,
                                          dm::metal_alloc_table());
}
