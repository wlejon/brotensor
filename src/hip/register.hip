// HIP backend master registration.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_HIP is
// defined. Probes the driver via hipGetDeviceCount(); on success, builds
// the OpsVTable by calling each cluster's per-cluster fill function,
// pairs it with the HIP AllocVTable, and hands the pair to the registry.

#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_runtime.h>

#include <cstdio>

namespace brotensor::detail::hip {

// ── alloc table (defined in tensor.hip) ──
const ::brotensor::detail::AllocVTable& hip_alloc_table();
void init_async_pool_for_device(int dev);
void fill_hip_vtable_gemm(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_softmax(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_rope(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_mha(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_flash_attention(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_attention(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_conv(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_interp(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_pool(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_spatial(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_norms(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_elementwise(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_audio(::brotensor::detail::OpsVTable& v);
void fill_hip_vtable_quant(::brotensor::detail::OpsVTable& v);
void register_hip_graph_backend();

} // namespace brotensor::detail::hip

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
void fill_cuda_vtable_masked_diffusion(::brotensor::detail::OpsVTable&);
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
// ── packed variable-length encoder batching (flash_attention_packed.cu, packed_encoder.cu) ──
void fill_cuda_vtable_packed_encoder (::brotensor::detail::OpsVTable&);
// ── fused-epilogue / split-K linear (linear_ex.cu, gemm_mma.cu) ──
void fill_cuda_vtable_linear_ex      (::brotensor::detail::OpsVTable&);

} // namespace brotensor::detail::cuda

extern "C" void brotensor_probe_and_register_hip() {
    // Probe: count HIP devices. If the call fails or returns 0 the runtime
    // / driver is missing — bail without registering. We intentionally do
    // NOT throw here so a CPU-only host can still load a binary that was
    // built with HIP support.
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return;
    }

    // Every kernel is written for 32-lane wavefronts (see
    // cmake/BrotensorHip.cmake). A device that runs wider waves would get
    // wrong answers, not errors, so it gets no backend at all.
    for (int i = 0; i < device_count; ++i) {
        int warp = 0;
        if (hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, i) == hipSuccess && warp != 32) {
            std::fprintf(stderr,
                         "brotensor: HIP device %d runs %d-wide wavefronts; the HIP backend "
                         "needs Wave32 (RDNA 3+). HIP disabled.\n", i, warp);
            return;
        }
    }

    ::brotensor::detail::set_hip_device_count(device_count);

    // Enable peer access between all GPU pairs and initialize async pools
    for (int i = 0; i < device_count; ++i) {
        if (hipSetDevice(i) != hipSuccess) continue;
        brotensor::detail::hip::init_async_pool_for_device(i);
        for (int j = 0; j < device_count; ++j) {
            if (i != j) {
                int can = 0;
                if (hipDeviceCanAccessPeer(&can, i, j) == hipSuccess && can) {
                    (void)hipDeviceEnablePeerAccess(j, 0);
                    (void)hipGetLastError(); // Clear any benign sticky error
                }
            }
        }
    }

    // Set default active device to 0
    (void)hipSetDevice(0);

    using ::brotensor::DeviceType;
    using ::brotensor::detail::OpsVTable;
    namespace dh = ::brotensor::detail::hip;
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
    dh::fill_hip_vtable_gemm(ops);
    dh::fill_hip_vtable_softmax(ops);
    dh::fill_hip_vtable_rope(ops);
    dh::fill_hip_vtable_mha(ops);
    dh::fill_hip_vtable_flash_attention(ops);
    dh::fill_hip_vtable_attention(ops);
    dh::fill_hip_vtable_conv(ops);
    dh::fill_hip_vtable_interp(ops);
    dh::fill_hip_vtable_pool(ops);
    dh::fill_hip_vtable_spatial(ops);
    dh::fill_hip_vtable_norms(ops);
    dh::fill_hip_vtable_elementwise(ops);
    dh::fill_hip_vtable_audio(ops);
    dh::fill_hip_vtable_quant(ops);

    ::brotensor::detail::register_backend(DeviceType::HIP, ops, dh::hip_alloc_table());
    dh::register_hip_graph_backend();
}
