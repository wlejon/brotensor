// Vulkan AllocVTable: allocation, host <-> device transfers, device copies,
// memset, sync and the memory / name queries.
//
// Transfers keep CUDA-style stream semantics: a host <-> device copy returns with
// the host buffer free to reuse and the data in place in stream order.
//   upload   * device memory mapped and the stream idle: memcpy straight in.
//            * <= 64 KiB, 4-byte aligned: vkCmdUpdateBuffer, recorded in the
//              stream (the bytes are copied into the command buffer), so no
//              wait at all.
//            * mapped, stream busy: drain the stream, then memcpy.
//            * not mapped: through a host staging buffer, 64 MiB at a time.
//   download * drain the stream; <= 64 KiB from mapped memory is read in
//              place; anything else is copied by the GPU into a host-cached
//              staging buffer and memcpy'd out, because device-local mapped
//              memory is write-combined on the CPU side (fast to write, slow
//              to read).
// Device-to-device copies and memsets are recorded in the stream.

#include "detail/device.h"
#include "detail/kernels.h"
#include "detail/vk_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

constexpr std::size_t kInlineUpdateMax = 65536;
constexpr std::size_t kDirectReadMax = 65536;
constexpr VkDeviceSize kStagingMax = VkDeviceSize(64) << 20;

// One pair of staging buffers per device, grown on demand up to kStagingMax.
struct Staging {
    std::mutex mu;
    std::unique_ptr<Block> down;   // host cached: GPU writes, CPU reads
    std::unique_ptr<Block> up;     // host visible: CPU writes, GPU reads
};

Staging& staging(int dev) {
    static Staging s[16];
    return s[dev & 15];
}

Block& staging_buffer(DeviceCtx& d, std::unique_ptr<Block>& slot, VkDeviceSize want, MemKind kind) {
    if (!slot || slot->size < want) {
        if (slot) {
            d.stream().sync();   // the old one may still be in flight
            d.allocator().destroy_raw(std::move(slot));
        }
        VkDeviceSize sz = VkDeviceSize(1) << 20;
        while (sz < want) sz <<= 1;
        slot = d.allocator().create_raw(std::min(sz, kStagingMax), kind);
    }
    return *slot;
}

void require_not_capturing(DeviceCtx& d, const char* what) {
    if (d.stream().capturing()) {
        throw std::runtime_error(std::string("brotensor: vulkan: ") + what +
                                 " on a device whose graph capture is recording "
                                 "(it would make the host wait; do transfers outside the capture)");
    }
}

}  // namespace

void* vk_alloc(std::size_t bytes, int dev) {
    if (bytes == 0) return nullptr;
    return device(dev).allocator().alloc(bytes);
}

void vk_free(void* p, int dev) {
    if (!p) return;
    device(dev).allocator().free(p);
}

void vk_memcpy_h2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    DeviceCtx& d = device(dev);
    require_not_capturing(d, "host-to-device copy");
    const Span s = d.allocator().resolve(addr(dst), n);
    Stream& st = d.stream();
    if (s.host && st.idle()) {
        std::memcpy(s.host, src, n);
        return;
    }
    if (n <= kInlineUpdateMax && s.offset % 4 == 0 && n % 4 == 0) {
        st.update(s.buf, s.offset, n, src);
        return;
    }
    if (s.host) {
        st.sync();
        std::memcpy(s.host, src, n);
        return;
    }
    Staging& stg = staging(dev);
    std::lock_guard<std::mutex> lk(stg.mu);
    Block& up = staging_buffer(d, stg.up, std::min<VkDeviceSize>(n, kStagingMax), MemKind::HostUpload);
    const auto* in = static_cast<const std::uint8_t*>(src);
    for (std::size_t done = 0; done < n;) {
        const std::size_t chunk = std::min<std::size_t>(n - done, up.size);
        st.sync();   // the staging buffer is free once the previous chunk landed
        std::memcpy(up.map, in + done, chunk);
        st.copy(up.buf, 0, s.buf, s.offset + done, chunk);
        done += chunk;
    }
    st.sync();
}

void vk_memcpy_d2h(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    DeviceCtx& d = device(dev);
    require_not_capturing(d, "device-to-host copy");
    const Span s = d.allocator().resolve(addr(src), n);
    Stream& st = d.stream();
    if (s.host && s.coherent && n <= kDirectReadMax) {
        st.sync();
        std::memcpy(dst, s.host, n);
        return;
    }
    Staging& stg = staging(dev);
    std::lock_guard<std::mutex> lk(stg.mu);
    Block& down = staging_buffer(d, stg.down, std::min<VkDeviceSize>(n, kStagingMax), MemKind::HostCached);
    auto* out = static_cast<std::uint8_t*>(dst);
    for (std::size_t done = 0; done < n;) {
        const std::size_t chunk = std::min<std::size_t>(n - done, down.size);
        st.copy(s.buf, s.offset + done, down.buf, 0, chunk);
        st.sync();
        if (!down.coherent) {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = down.mem;
            r.offset = 0;
            r.size = VK_WHOLE_SIZE;
            BT_VK_CHECK(d.fn().vkInvalidateMappedMemoryRanges(d.device(), 1, &r));
        }
        std::memcpy(out + done, down.map, chunk);
        done += chunk;
    }
}

void vk_memcpy_d2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0 || dst == src) return;
    DeviceCtx& d = device(dev);
    const Span a = d.allocator().resolve(addr(src), n);
    const Span b = d.allocator().resolve(addr(dst), n);
    d.stream().copy(a.buf, a.offset, b.buf, b.offset, n);
}

void vk_memcpy_peer(void* dst, int dst_dev, const void* src, int src_dev, std::size_t n) {
    if (n == 0) return;
    if (dst_dev == src_dev) {
        vk_memcpy_d2d(dst, src, n, dst_dev);
        return;
    }
    // Two VkDevices share nothing; bounce through the host.
    std::vector<std::uint8_t> host(n);
    vk_memcpy_d2h(host.data(), src, n, src_dev);
    vk_memcpy_h2d(dst, host.data(), n, dst_dev);
}

// Byte-exact zero fill of [dst, dst + n): vkCmdFillBuffer for the 4-byte
// aligned middle, a byte kernel for the unaligned edges (views at odd element
// offsets of 1- and 2-byte dtypes).
void vk_memset_zero(void* dst, std::size_t n, int dev) {
    if (n == 0) return;
    DeviceCtx& d = device(dev);
    const std::uint64_t a = addr(dst);
    const Span s = d.allocator().resolve(a, n);
    const std::uint64_t lo = std::min<std::uint64_t>((a + 3) & ~std::uint64_t(3), a + n);
    const std::uint64_t hi = std::max<std::uint64_t>((a + n) & ~std::uint64_t(3), lo);
    auto bytes = [&](std::uint64_t from, std::uint64_t to) {
        if (to <= from) return;
        struct Push { std::uint64_t dst; std::uint32_t n, value; } pc{from, static_cast<std::uint32_t>(to - from), 0};
        const Kernel& k = d.pipelines().get(ShaderId::fill_bytes);
        launch(d, k, pc, groups_1d(to - from, k));
    };
    bytes(a, lo);
    if (hi > lo) d.stream().fill(s.buf, s.offset + (lo - a), hi - lo, 0u);
    bytes(hi, a + n);
}

// Syncing a device nobody has used is a no-op; it does not create the
// VkDevice (sync_all() visits every registered device).
void vk_sync(int dev) {
    if (dev >= 0) {
        if (DeviceCtx* d = device_if_created(dev)) d->stream().sync();
        return;
    }
    for (int i = 0; i < vulkan_device_count(); ++i) {
        if (DeviceCtx* d = device_if_created(i)) d->stream().sync();
    }
}

bool vk_mem_info(std::size_t* free_bytes, std::size_t* total_bytes, int dev) {
    const Instance& in = Instance::get();
    if (dev < 0 || dev >= in.count()) return false;
    const PhysInfo& info = in.phys(dev);
    // The heap tensors live in: the largest device-local heap.
    std::uint32_t heap = 0;
    VkDeviceSize best = 0;
    for (std::uint32_t h = 0; h < info.mem.memoryHeapCount; ++h) {
        if ((info.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            info.mem.memoryHeaps[h].size > best) {
            best = info.mem.memoryHeaps[h].size;
            heap = h;
        }
    }
    std::size_t total = static_cast<std::size_t>(best);
    std::size_t avail = total;
    if (info.memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp.pNext = &budget;
        in.fns().vkGetPhysicalDeviceMemoryProperties2(info.pd, &mp);
        const VkDeviceSize b = budget.heapBudget[heap], u = budget.heapUsage[heap];
        avail = static_cast<std::size_t>(b > u ? b - u : 0);
    }
    if (free_bytes) *free_bytes = avail;
    if (total_bytes) *total_bytes = total;
    return true;
}

bool vk_mem_trim(std::size_t keep_bytes, int dev) {
    if (dev < 0 || dev >= vulkan_device_count()) return false;
    DeviceCtx* dp = device_if_created(dev);
    if (!dp) return true;   // nothing allocated, nothing cached
    DeviceCtx& d = *dp;
    if (d.stream().capturing()) return false;
    d.stream().sync();
    d.allocator().collect(d.stream().completed());
    d.allocator().trim(keep_bytes);
    return true;
}

bool vk_device_name(char* out, std::size_t cap, int dev) {
    if (!out || cap == 0) return false;
    const Instance& in = Instance::get();
    if (dev < 0 || dev >= in.count()) return false;
    std::snprintf(out, cap, "%s", in.phys(dev).name.c_str());
    return true;
}

const ::brotensor::detail::AllocVTable& vulkan_alloc_table() {
    static const ::brotensor::detail::AllocVTable t = {
        &vk_alloc,
        &vk_free,
        &vk_memcpy_h2d,
        &vk_memcpy_d2h,
        &vk_memcpy_d2d,
        &vk_memcpy_peer,
        &vk_memset_zero,
        &vk_sync,
        &vk_mem_info,
        &vk_mem_trim,
        &vk_device_name,
    };
    return t;
}

}  // namespace brotensor::detail::vulkan
