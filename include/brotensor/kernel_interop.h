#pragma once
// Zero-copy interop between brotensor::Tensor and external compute kernels
// (e.g. brokernel CPU/SPIR-V kernels):
//   - tensor buffer views (host pointer or Vulkan BDA device address)
//   - byte-level H2D / D2H transfers preserving existing Tensor storage
//   - push-constant SPIR-V compute dispatch on brotensor's Vulkan stream

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace brotensor::interop {

struct TensorBufferView {
    void* host_ptr = nullptr;
    std::uint64_t device_address = 0;
    std::size_t bytes = 0;
    std::size_t numel = 0;
    Dtype dtype = Dtype::FP32;
    Device device = Device::CPU;

    bool is_gpu() const noexcept { return device.is_gpu(); }
    bool is_vulkan() const noexcept { return device.is_vulkan(); }
    bool is_cuda() const noexcept { return device.is_cuda(); }
};

/// Returns a zero-copy buffer view of `t` on its owning device.
/// For CPU tensors, `host_ptr` is `t.data`; for Vulkan tensors, `device_address`
/// is the buffer device address (`VkDeviceAddress`) carried in `t.data`.
TensorBufferView view(const Tensor& t);

/// Copies `nbytes` from host memory `src_host` into `dst` in-place (preserving
/// `dst.data` allocation/device address). Requires `dst.data != nullptr` and `nbytes <= dst.bytes()`.
bool copy_bytes_to_tensor(const void* src_host, std::size_t nbytes, Tensor& dst, std::string* err = nullptr);

/// Copies `nbytes` from `src` (CPU or GPU, synced) into host memory `dst_host`.
bool copy_bytes_from_tensor(const Tensor& src, void* dst_host, std::size_t nbytes, std::string* err = nullptr);

struct SpirvDispatchDesc {
    const std::uint32_t* words = nullptr;
    std::size_t word_count = 0;
    const char* entry_name = "main";
    std::uint32_t local_size[3] = {64, 1, 1};
    std::uint32_t groups[3] = {1, 1, 1};
    const void* push_data = nullptr;
    std::uint32_t push_bytes = 0;
    bool local_size_spec_constants = true;
};

/// Returns true when `dev` is a Vulkan device ready for `dispatch_spirv`.
bool vulkan_spirv_dispatch_available(Device dev = Device::vulkan(0));

/// Dispatches an arbitrary BDA push-constant SPIR-V kernel directly on `dev`'s
/// brotensor Vulkan stream (ordered with preceding/subsequent brotensor ops).
bool dispatch_spirv(Device dev, const SpirvDispatchDesc& desc, std::string* err = nullptr);

struct PtxDispatchDesc {
    const char* ptx = nullptr;
    std::size_t ptx_bytes = 0;
    const char* entry_name = nullptr;
    std::uint32_t local_size[3] = {64, 1, 1};
    std::uint32_t groups[3] = {1, 1, 1};
    void** kernel_params = nullptr;
    std::uint32_t shared_bytes = 0;
};

/// Returns true when `dev` is a CUDA device ready for zero-copy `dispatch_ptx`.
bool cuda_ptx_dispatch_available(Device dev = Device::cuda(0));

/// JIT-compiles (cached) and launches a PTX kernel directly on `dev`'s
/// brotensor CUDA stream (ordered with preceding/subsequent brotensor ops).
bool dispatch_ptx(Device dev, const PtxDispatchDesc& desc, std::string* err = nullptr);

namespace detail {
using VulkanDispatchHook = bool (*)(int device_index, const SpirvDispatchDesc& desc, std::string* err);
using VulkanAvailableHook = bool (*)(int device_index);
void register_vulkan_interop_hooks(VulkanAvailableHook avail_fn, VulkanDispatchHook dispatch_fn);

using CudaDispatchHook = bool (*)(int device_index, const PtxDispatchDesc& desc, std::string* err);
using CudaAvailableHook = bool (*)(int device_index);
void register_cuda_interop_hooks(CudaAvailableHook avail_fn, CudaDispatchHook dispatch_fn);
} // namespace detail

} // namespace brotensor::interop
