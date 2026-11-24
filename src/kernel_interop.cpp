#include "brotensor/kernel_interop.h"
#include "brotensor/detail/dispatch.h"
#include "brotensor/runtime.h"

#include <atomic>
#include <cstring>
#include <stdexcept>

namespace brotensor::interop {

namespace {

std::atomic<detail::VulkanAvailableHook> g_vk_avail{nullptr};
std::atomic<detail::VulkanDispatchHook> g_vk_dispatch{nullptr};
std::atomic<detail::CudaAvailableHook> g_cuda_avail{nullptr};
std::atomic<detail::CudaDispatchHook> g_cuda_dispatch{nullptr};

} // namespace

namespace detail {

void register_vulkan_interop_hooks(VulkanAvailableHook avail_fn, VulkanDispatchHook dispatch_fn) {
    g_vk_avail.store(avail_fn, std::memory_order_release);
    g_vk_dispatch.store(dispatch_fn, std::memory_order_release);
}

void register_cuda_interop_hooks(CudaAvailableHook avail_fn, CudaDispatchHook dispatch_fn) {
    g_cuda_avail.store(avail_fn, std::memory_order_release);
    g_cuda_dispatch.store(dispatch_fn, std::memory_order_release);
}

} // namespace detail

TensorBufferView view(const Tensor& t) {
    TensorBufferView out;
    out.bytes = t.bytes();
    out.numel = static_cast<std::size_t>(t.size());
    out.dtype = t.dtype;
    out.device = t.device;
    if (t.device.is_cpu()) {
        out.host_ptr = t.data;
    } else if (t.device.is_vulkan() || t.device.is_cuda()) {
        out.device_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.data));
    }
    return out;
}

bool copy_bytes_to_tensor(const void* src_host, std::size_t nbytes, Tensor& dst, std::string* err) {
    if (nbytes == 0) return true;
    if (!src_host) {
        if (err) *err = "copy_bytes_to_tensor: null source host pointer";
        return false;
    }
    if (!dst.data) {
        if (err) *err = "copy_bytes_to_tensor: destination tensor has no allocated storage";
        return false;
    }
    const std::size_t cap = dst.bytes();
    if (nbytes > cap) {
        if (err) *err = "copy_bytes_to_tensor: source byte size exceeds tensor byte size";
        return false;
    }
    if (dst.device.is_cpu()) {
        std::memcpy(dst.data, src_host, nbytes);
        return true;
    }
    try {
        const Device resolved = brotensor::detail::resolve_device_alias(dst.device);
        const auto& alloc = brotensor::detail::alloc_for(resolved);
        if (!alloc.memcpy_h2d) {
            if (err) *err = "copy_bytes_to_tensor: backend does not support memcpy_h2d";
            return false;
        }
        alloc.memcpy_h2d(dst.data, src_host, nbytes, resolved.index);
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return false;
    }
}

bool copy_bytes_from_tensor(const Tensor& src, void* dst_host, std::size_t nbytes, std::string* err) {
    if (nbytes == 0) return true;
    if (!dst_host) {
        if (err) *err = "copy_bytes_from_tensor: null destination host pointer";
        return false;
    }
    if (!src.data) {
        if (err) *err = "copy_bytes_from_tensor: source tensor has no allocated storage";
        return false;
    }
    const std::size_t cap = src.bytes();
    if (nbytes > cap) {
        if (err) *err = "copy_bytes_from_tensor: requested byte size exceeds tensor byte size";
        return false;
    }
    if (src.device.is_cpu()) {
        std::memcpy(dst_host, src.data, nbytes);
        return true;
    }
    try {
        const Device resolved = brotensor::detail::resolve_device_alias(src.device);
        const auto& alloc = brotensor::detail::alloc_for(resolved);
        if (!alloc.memcpy_d2h) {
            if (err) *err = "copy_bytes_from_tensor: backend does not support memcpy_d2h";
            return false;
        }
        alloc.memcpy_d2h(dst_host, src.data, nbytes, resolved.index);
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return false;
    }
}

bool vulkan_spirv_dispatch_available(Device dev) {
    auto fn = g_vk_avail.load(std::memory_order_acquire);
    if (!fn) return false;
    const Device resolved = brotensor::detail::resolve_device_alias(dev);
    if (!resolved.is_vulkan()) return false;
    return fn(resolved.index);
}

bool dispatch_spirv(Device dev, const SpirvDispatchDesc& desc, std::string* err) {
    auto fn = g_vk_dispatch.load(std::memory_order_acquire);
    if (!fn) {
        if (err) *err = "brotensor: Vulkan SPIR-V interop backend is not available";
        return false;
    }
    const Device resolved = brotensor::detail::resolve_device_alias(dev);
    if (!resolved.is_vulkan()) {
        if (err) *err = "brotensor: dispatch_spirv requires a Vulkan-capable device";
        return false;
    }
    return fn(resolved.index, desc, err);
}

bool cuda_ptx_dispatch_available(Device dev) {
    auto fn = g_cuda_avail.load(std::memory_order_acquire);
    if (!fn) return false;
    const Device resolved = brotensor::detail::resolve_device_alias(dev);
    if (!resolved.is_cuda()) return false;
    return fn(resolved.index);
}

bool dispatch_ptx(Device dev, const PtxDispatchDesc& desc, std::string* err) {
    auto fn = g_cuda_dispatch.load(std::memory_order_acquire);
    if (!fn) {
        if (err) *err = "brotensor: CUDA PTX interop backend is not available";
        return false;
    }
    const Device resolved = brotensor::detail::resolve_device_alias(dev);
    if (!resolved.is_cuda()) {
        if (err) *err = "brotensor: dispatch_ptx requires a CUDA device";
        return false;
    }
    return fn(resolved.index, desc, err);
}

} // namespace brotensor::interop
