// HIP backend master registration.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_HIP is
// defined. Probes the driver via hipGetDeviceCount(); on success, builds
// the OpsVTable, pairs it with the HIP AllocVTable, and hands the pair to the registry.

#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_runtime.h>
#include <vector>

namespace brotensor::detail::hip {

// ── alloc table (defined in tensor.hip) ──
const ::brotensor::detail::AllocVTable& hip_alloc_table();
void init_async_pool_for_device(int dev);

} // namespace brotensor::detail::hip

namespace brotensor::detail::cuda {

// ── per-cluster vtable-fill entry points ──
void fill_cuda_vtable_elementwise (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_norms       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_utils       (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_l2_norm     (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_l2_normalize(::brotensor::detail::OpsVTable&);

// ── Embedding (src/cuda/embedding.cu) ──
void embedding_lookup_forward(const ::brotensor::Tensor& table, const int32_t* d_idx,
                              int B, ::brotensor::Tensor& out);
void embedding_lookup_backward(const ::brotensor::Tensor& dOut, const int32_t* d_idx,
                               int B, ::brotensor::Tensor& dTable);

// ── Concat / split / copy (src/cuda/concat.cu) ──
void concat_rows(const std::vector<const ::brotensor::Tensor*>& parts, ::brotensor::Tensor& out);
void split_rows(const ::brotensor::Tensor& in, const std::vector<::brotensor::Tensor*>& parts);
void concat_batched_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                         ::brotensor::Tensor& out);
void concat_nchw_channels(const std::vector<const ::brotensor::Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          ::brotensor::Tensor& out);
void concat_nchw_channels_backward(const ::brotensor::Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<::brotensor::Tensor*>& dParts);
void copy_d2d(const ::brotensor::Tensor& src, int src_row,
              ::brotensor::Tensor& dst, int dst_row, int n_rows);
void copy_d2d_strided(const ::brotensor::Tensor& src, int src_offset, int src_pitch,
                      ::brotensor::Tensor& dst, int dst_offset, int dst_pitch,
                      int width, int height);

// ── Loss (src/cuda/loss.cu) ──
float mse_vec_forward(const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target);
void mse_vec_backward(const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target, ::brotensor::Tensor& dPred);
void mse_vec_per_sample(const ::brotensor::Tensor& pred, const ::brotensor::Tensor& target,
                        ::brotensor::Tensor& dPred, ::brotensor::Tensor& loss_per_sample);
float softmax_xent_fused(const ::brotensor::Tensor& logits, const ::brotensor::Tensor& target,
                         const float* d_mask,
                         ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits);
void softmax_xent_fused_batched(const ::brotensor::Tensor& logits_BL, const ::brotensor::Tensor& target_BL,
                                const float* d_mask_BL, const int* d_head_offsets,
                                int n_heads, ::brotensor::Tensor& probs_BL, ::brotensor::Tensor& dLogits_BL,
                                ::brotensor::Tensor& loss_per_sample);
void bce_with_logits_fused_batched(const ::brotensor::Tensor& logits_BL, const ::brotensor::Tensor& target_BL,
                                   const float* d_mask_BL, float pos_weight,
                                   ::brotensor::Tensor& probs_BL, ::brotensor::Tensor& dLogits_BL,
                                   ::brotensor::Tensor& loss_per_sample);

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

    OpsVTable ops{}; // zero-init: every slot starts as nullptr

    // Wire per-cluster fill functions
    dc::fill_cuda_vtable_elementwise(ops);
    dc::fill_cuda_vtable_norms(ops);
    dc::fill_cuda_vtable_utils(ops);
    dc::fill_cuda_vtable_l2_norm(ops);
    dc::fill_cuda_vtable_l2_normalize(ops);

    // Embedding
    ops.embedding_lookup_forward  = &dc::embedding_lookup_forward;
    ops.embedding_lookup_backward = &dc::embedding_lookup_backward;

    // Concat / split / copy
    ops.concat_rows                   = &dc::concat_rows;
    ops.split_rows                    = &dc::split_rows;
    ops.concat_batched_rows           = &dc::concat_batched_rows;
    ops.concat_nchw_channels          = &dc::concat_nchw_channels;
    ops.concat_nchw_channels_backward = &dc::concat_nchw_channels_backward;
    ops.copy_d2d                      = &dc::copy_d2d;
    ops.copy_d2d_strided              = &dc::copy_d2d_strided;

    // Loss
    ops.mse_vec_forward               = &dc::mse_vec_forward;
    ops.mse_vec_backward              = &dc::mse_vec_backward;
    ops.mse_vec_per_sample            = &dc::mse_vec_per_sample;
    ops.softmax_xent_fused            = &dc::softmax_xent_fused;
    ops.softmax_xent_fused_batched    = &dc::softmax_xent_fused_batched;
    ops.bce_with_logits_fused_batched = &dc::bce_with_logits_fused_batched;

    ::brotensor::detail::register_backend(DeviceType::HIP, ops, dh::hip_alloc_table());
}
