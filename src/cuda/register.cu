// CUDA backend master registration. Phase 2G.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_CUDA is
// defined. Probes the driver via cudaGetDeviceCount(); on success, builds
// the OpsVTable by calling each Phase 2 cluster's per-cluster fill function,
// pairs it with the CUDA AllocVTable, and hands the pair to the registry.
//
// Every per-cluster fill function lives in `brotensor::detail::cuda` and is
// defined in its own TU (one per Phase 2 agent). A null slot in the vtable
// means "this op is not implemented on CUDA" — the dispatcher throws on
// null lookups.

#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

namespace brotensor::detail::cuda {

// ── per-cluster vtable-fill entry points ──
void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_attention      (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_conv           (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_elementwise    (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_norms          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_utils          (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_specialised    (::brotensor::detail::OpsVTable&);
// ── brosoundml audio-ML clusters (one fill fn per src/cuda/<family>.cu) ──
void fill_cuda_vtable_vocoder        (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_log_exp_round  (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_fft            (::brotensor::detail::OpsVTable&);
void fill_cuda_vtable_stft           (::brotensor::detail::OpsVTable&);

// ── alloc table (defined in tensor.cu) ──
const ::brotensor::detail::AllocVTable& cuda_alloc_table();

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

    // Pick device 0; let lazy context creation happen the first time a kernel
    // launches. (Old behaviour from `cuda_init()`; kept implicit.)
    err = cudaSetDevice(0);
    if (err != cudaSuccess) return;

    using ::brotensor::Device;
    using ::brotensor::detail::OpsVTable;
    namespace dc = ::brotensor::detail::cuda;

    OpsVTable ops{};   // zero-init: every slot starts as nullptr
    dc::fill_cuda_vtable_flash_attention(ops);
    dc::fill_cuda_vtable_attention(ops);
    dc::fill_cuda_vtable_conv(ops);
    dc::fill_cuda_vtable_elementwise(ops);
    dc::fill_cuda_vtable_norms(ops);
    dc::fill_cuda_vtable_utils(ops);
    dc::fill_cuda_vtable_specialised(ops);
    dc::fill_cuda_vtable_vocoder(ops);
    dc::fill_cuda_vtable_log_exp_round(ops);
    dc::fill_cuda_vtable_fft(ops);
    dc::fill_cuda_vtable_stft(ops);

    ::brotensor::detail::register_backend(Device::CUDA, ops, dc::cuda_alloc_table());
}
