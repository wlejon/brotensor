// CUDA backend master registration (pure C++ via dynamic Driver API).

#include "cuda_driver.h"
#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

namespace brotensor::detail::cuda {

// ── per-cluster vtable-fill entry points ──
void register_cuda_graph_backend();
void register_cuda_interop_backend();
void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_attention      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_deform_conv    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv3d         (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_elementwise    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_norms          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_utils          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_specialised    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_vocoder        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_log_exp_round  (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_fft            (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_stft           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv1d         (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_codec_quant    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_resample1d     (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_sample_logits  (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_masked_diffusion(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_l2_norm        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_gated_delta_rule(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_qwen3_vl_polish(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_interp2d       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_pool2d         (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_pad2d          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_slice2d        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_unfold2d       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_l2_normalize   (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_convex_upsample(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_gather_scatter (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_top_k          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_window_partition(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv_transpose2d(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_loss_legacy    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_xavier_init    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_batch_norm     (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_image_preproc  (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_noise          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_lstm           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_stylegan_elementwise(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_bias_act       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_upfirdn2d      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_modulated_conv2d(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_filtered_lrelu (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_packed_encoder (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_linear_ex      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_linear_wq      (::brotensor::detail::OpsVTable&);

const ::brotensor::detail::AllocVTable& cuda_alloc_table();
void init_async_pool_for_device(int dev);

}  // namespace brotensor::detail::cuda

extern "C" void brotensor_probe_and_register_cuda() {
    namespace dc = ::brotensor::detail::cuda;
    const int dev_count = dc::device_count();
    if (dev_count <= 0) {
        return;
    }

    ::brotensor::detail::set_cuda_device_count(dev_count);

    for (int i = 0; i < dev_count; ++i) {
        if (dc::set_device(i) != CUDA_SUCCESS) continue;
        dc::init_async_pool_for_device(i);

        CUdevice dev_i = 0;
        if (dc::drv::cuDeviceGet(&dev_i, i) != CUDA_SUCCESS) continue;
        for (int j = 0; j < dev_count; ++j) {
            if (i == j) continue;
            CUdevice dev_j = 0;
            if (dc::drv::cuDeviceGet(&dev_j, j) != CUDA_SUCCESS) continue;
            int can = 0;
            if (dc::drv::cuDeviceCanAccessPeer(&can, dev_i, dev_j) == CUDA_SUCCESS && can) {
                if (CUcontext peer_ctx = dc::primary_context(j)) {
                    (void)dc::drv::cuCtxEnablePeerAccess(peer_ctx, 0);
                }
            }
        }
    }

    (void)dc::set_device(0);

    using ::brotensor::DeviceType;
    using ::brotensor::detail::OpsVTable;

    OpsVTable ops{};
    dc::fill_cuda_vtable_flash_attention(ops);
    dc::fill_cuda_vtable_attention(ops);
    dc::fill_cuda_vtable_conv(ops);
    dc::fill_cuda_vtable_deform_conv(ops);
    dc::fill_cuda_vtable_conv3d(ops);
    dc::fill_cuda_vtable_elementwise(ops);
    dc::fill_cuda_vtable_norms(ops);
    dc::fill_cuda_vtable_utils(ops);
    dc::fill_cuda_vtable_specialised(ops);
    dc::fill_cuda_vtable_vocoder(ops);
    dc::fill_cuda_vtable_log_exp_round(ops);
    dc::fill_cuda_vtable_fft(ops);
    dc::fill_cuda_vtable_stft(ops);
    dc::fill_cuda_vtable_conv1d(ops);
    dc::fill_cuda_vtable_codec_quant(ops);
    dc::fill_cuda_vtable_resample1d(ops);
    dc::fill_cuda_vtable_sample_logits(ops);
    dc::fill_cuda_vtable_masked_diffusion(ops);
    dc::fill_cuda_vtable_l2_norm(ops);
    dc::fill_cuda_vtable_gated_delta_rule(ops);
    dc::fill_cuda_vtable_qwen3_vl_polish(ops);
    dc::fill_cuda_vtable_interp2d(ops);
    dc::fill_cuda_vtable_pool2d(ops);
    dc::fill_cuda_vtable_pad2d(ops);
    dc::fill_cuda_vtable_slice2d(ops);
    dc::fill_cuda_vtable_unfold2d(ops);
    dc::fill_cuda_vtable_l2_normalize(ops);
    dc::fill_cuda_vtable_convex_upsample(ops);
    dc::fill_cuda_vtable_gather_scatter(ops);
    dc::fill_cuda_vtable_top_k(ops);
    dc::fill_cuda_vtable_window_partition(ops);
    dc::fill_cuda_vtable_conv_transpose2d(ops);
    dc::fill_cuda_vtable_loss_legacy(ops);
    dc::fill_cuda_vtable_xavier_init(ops);
    dc::fill_cuda_vtable_batch_norm(ops);
    dc::fill_cuda_vtable_image_preproc(ops);
    dc::fill_cuda_vtable_noise(ops);
    dc::fill_cuda_vtable_lstm(ops);
    dc::fill_cuda_vtable_stylegan_elementwise(ops);
    dc::fill_cuda_vtable_bias_act(ops);
    dc::fill_cuda_vtable_upfirdn2d(ops);
    dc::fill_cuda_vtable_modulated_conv2d(ops);
    dc::fill_cuda_vtable_filtered_lrelu(ops);
    dc::fill_cuda_vtable_packed_encoder(ops);
    dc::fill_cuda_vtable_linear_ex(ops);
    dc::fill_cuda_vtable_linear_wq(ops);

    ::brotensor::detail::register_backend(DeviceType::CUDA, ops, dc::cuda_alloc_table());
    dc::register_cuda_graph_backend();
    dc::register_cuda_interop_backend();
}
