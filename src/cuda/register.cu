// CUDA backend master registration.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_CUDA is
// defined. Probes the driver via cudaGetDeviceCount(); on success, builds
// the OpsVTable by calling each cluster's per-cluster fill function,
// pairs it with the CUDA AllocVTable, and hands the pair to the registry.
//
// Every per-cluster fill function lives in `brotensor::detail::cuda` and is
// defined in its own TU. A null slot in the vtable means "this op is not
// implemented on CUDA" — the dispatcher throws on null lookups.

#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

namespace brotensor::detail::cuda {

// ── per-cluster vtable-fill entry points ──
void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_attention      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_deform_conv    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv3d         (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_elementwise    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_norms          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_utils          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_specialised    (::brotensor::detail::OpsVTable&);
// ── brosoundml audio-ML clusters (one fill fn per src/cuda/<family>.cu) ──
void fill_cuda_vtable_vocoder        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_log_exp_round  (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_fft            (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_stft           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv1d         (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_codec_quant    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_resample1d     (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_sample_logits  (::brotensor::detail::OpsVTable&);
// ── linear-attention text-path clusters ──
void fill_cuda_vtable_l2_norm        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_gated_delta_rule(::brotensor::detail::OpsVTable&);
// ── Qwen3-VL polish (spatial_merge_2x2 + M-RoPE) — rope_mrope.cu ──
void fill_cuda_vtable_qwen3_vl_polish(::brotensor::detail::OpsVTable&);
// ── Vision-encoder arbitrary-scale 2D resample (SAM / DPT / depth) ──
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
// ── StyleGAN3-R synthesis ops (sin/cos/rsqrt/pixel_norm, bias_act,
//    upfirdn2d, modulated_conv2d) ──
void fill_cuda_vtable_stylegan_elementwise(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_bias_act       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_upfirdn2d      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_modulated_conv2d(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_filtered_lrelu (::brotensor::detail::OpsVTable&);

// ── alloc table (defined in tensor.cu) ──
const ::brotensor::detail::AllocVTable& cuda_alloc_table();
void init_async_pool_for_device(int dev);

} // namespace brotensor::detail::cuda

extern "C" void brotensor_probe_and_register_cuda() {
    // Probe: count CUDA devices. If the call fails or returns 0 the runtime
    // / driver is missing — bail without registering. We intentionally do
    // NOT throw here so a CPU-only host can still load a binary that was
    // built with CUDA support.
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return;
    }

    ::brotensor::detail::set_cuda_device_count(device_count);

    // Enable peer access between all GPU pairs and initialize async pools
    for (int i = 0; i < device_count; ++i) {
        if (cudaSetDevice(i) != cudaSuccess) continue;
        brotensor::detail::cuda::init_async_pool_for_device(i);
        for (int j = 0; j < device_count; ++j) {
            if (i != j) {
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess && can) {
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaGetLastError(); // Clear any benign sticky error
                }
            }
        }
    }

    // Set default active device to 0
    cudaSetDevice(0);

    using ::brotensor::DeviceType;
    using ::brotensor::detail::OpsVTable;
    namespace dc = ::brotensor::detail::cuda;

    OpsVTable ops{};   // zero-init: every slot starts as nullptr
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

    ::brotensor::detail::register_backend(DeviceType::CUDA, ops, dc::cuda_alloc_table());
}
