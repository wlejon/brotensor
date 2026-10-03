// Vulkan device-memory allocator. Design notes: detail/allocator.h.

#include "detail/allocator.h"
#include "detail/device.h"
#include "detail/vk_check.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

namespace {

constexpr VkDeviceSize kSpecMaxBuffer = VkDeviceSize(4) << 30;     // 4 GiB
constexpr VkDeviceSize kDedicatedGranule = VkDeviceSize(64) << 10;  // 64 KiB

constexpr VkMemoryPropertyFlags kAmdUncached =
    VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;

VkDeviceSize round_up(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

const char* env(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

std::string mib(VkDeviceSize b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(b) / (1024.0 * 1024.0));
    return buf;
}

}  // namespace

Allocator::Allocator(DeviceCtx& dev) : dev_(dev) {
    const PhysInfo& info = dev.info();
    const LoaderFns& lf = Instance::get().fns();

    // Per-buffer limit: min(maxMemoryAllocationSize, maxBufferSize, 4 GiB).
    VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    const bool v13 = info.props.apiVersion >= VK_API_VERSION_1_3;
    if (v13) m3.pNext = &m4;
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &m3;
    lf.vkGetPhysicalDeviceProperties2(info.pd, &p2);
    max_buffer_ = std::min<VkDeviceSize>(m3.maxMemoryAllocationSize, kSpecMaxBuffer);
    if (v13) max_buffer_ = std::min<VkDeviceSize>(max_buffer_, m4.maxBufferSize);
    allow_oversize_ = env("BROTENSOR_VK_ALLOW_OVERSIZE") && std::strcmp(env("BROTENSOR_VK_ALLOW_OVERSIZE"), "0") != 0;

    // Map device memory when a device-local, host-visible, coherent type covers
    // (nearly) the whole of the largest device-local heap: unified memory
    // (APUs) and resizable-BAR discrete GPUs. Then uploads are a memcpy.
    VkDeviceSize largest_dl = 0;
    for (std::uint32_t h = 0; h < info.mem.memoryHeapCount; ++h) {
        if (info.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            largest_dl = std::max(largest_dl, info.mem.memoryHeaps[h].size);
    }
    bool mappable = false;
    for (std::uint32_t t = 0; t < info.mem.memoryTypeCount; ++t) {
        const auto& mt = info.mem.memoryTypes[t];
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mt.propertyFlags & want) != want || (mt.propertyFlags & kAmdUncached)) continue;
        if (info.mem.memoryHeaps[mt.heapIndex].size * 10 >= largest_dl * 9) mappable = true;
    }
    mapped_ = mappable;
    if (const char* m = env("BROTENSOR_VK_MAPPED")) mapped_ = mappable && std::strcmp(m, "0") != 0;

    if (const char* b = env("BROTENSOR_VK_BLOCK_MB")) {
        const long long mb = std::atoll(b);
        if (mb > 0) block_bytes_ = VkDeviceSize(mb) << 20;
    }
    block_bytes_ = std::min(block_bytes_, round_up(max_buffer_ / 2, kAlign));
    block_bytes_ = std::max(block_bytes_, kSubAllocMax);
}

Allocator::~Allocator() {
    for (auto& d : deferred_) destroy_block(d.block.get());
    for (auto& kv : blocks_) destroy_block(kv.second.get());
}

int Allocator::pick_memory_type(std::uint32_t bits, MemKind kind) const {
    const auto& mem = dev_.info().mem;
    auto find = [&](VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) -> int {
        for (std::uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
            if (!(bits & (1u << t))) continue;
            const VkMemoryPropertyFlags f = mem.memoryTypes[t].propertyFlags;
            if ((f & want) != want) continue;
            if (f & (avoid | kAmdUncached)) continue;
            return static_cast<int>(t);
        }
        return -1;
    };
    switch (kind) {
        case MemKind::Device: {
            if (mapped_) {
                const int t = find(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
                if (t >= 0) return t;
            }
            int t = find(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            if (t < 0) t = find(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
            if (t < 0) t = find(0, 0);
            return t;
        }
        case MemKind::HostCached: {
            int t = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (t < 0) t = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
            if (t < 0) t = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
            return t;
        }
        case MemKind::HostUpload: {
            int t = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (t < 0) t = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
            return t;
        }
    }
    return -1;
}

std::unique_ptr<Block> Allocator::create_block(VkDeviceSize bytes, MemKind kind, bool dedicated,
                                               VkResult* err) {
    const DeviceFns& f = dev_.fn();
    VkDevice d = dev_.device();
    auto b = std::make_unique<Block>();
    b->size = bytes;
    b->dedicated = dedicated;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (kind == MemKind::Device) {
        bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = f.vkCreateBuffer(d, &bci, nullptr, &b->buf);
    if (r != VK_SUCCESS) { *err = r; return nullptr; }

    VkMemoryRequirements req;
    f.vkGetBufferMemoryRequirements(d, b->buf, &req);
    const int type = pick_memory_type(req.memoryTypeBits, kind);
    if (type < 0) {
        f.vkDestroyBuffer(d, b->buf, nullptr);
        *err = VK_ERROR_FEATURE_NOT_PRESENT;
        return nullptr;
    }
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = b->buf;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    const void** tail = &mai.pNext;
    if (kind == MemKind::Device) { *tail = &flags; tail = &flags.pNext; }
    if (dedicated) { *tail = &ded; }
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = static_cast<std::uint32_t>(type);
    r = f.vkAllocateMemory(d, &mai, nullptr, &b->mem);
    if (r != VK_SUCCESS) {
        f.vkDestroyBuffer(d, b->buf, nullptr);
        *err = r;
        return nullptr;
    }
    r = f.vkBindBufferMemory(d, b->buf, b->mem, 0);
    if (r != VK_SUCCESS) {
        f.vkFreeMemory(d, b->mem, nullptr);
        f.vkDestroyBuffer(d, b->buf, nullptr);
        *err = r;
        return nullptr;
    }
    const VkMemoryPropertyFlags pf = dev_.info().mem.memoryTypes[type].propertyFlags;
    if (pf & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        void* p = nullptr;
        if (f.vkMapMemory(d, b->mem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
            b->map = static_cast<std::uint8_t*>(p);
            b->coherent = (pf & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        }
    }
    if (kind == MemKind::Device) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b->buf;
        b->addr = f.vkGetBufferDeviceAddress(d, &ai);
    }
    *err = VK_SUCCESS;
    return b;
}

void Allocator::destroy_block(Block* b) {
    if (!b) return;
    const DeviceFns& f = dev_.fn();
    VkDevice d = dev_.device();
    if (b->map) f.vkUnmapMemory(d, b->mem);
    if (b->buf) f.vkDestroyBuffer(d, b->buf, nullptr);
    if (b->mem) f.vkFreeMemory(d, b->mem, nullptr);
    b->map = nullptr;
    b->buf = VK_NULL_HANDLE;
    b->mem = VK_NULL_HANDLE;
}

std::unique_ptr<Block> Allocator::create_raw(VkDeviceSize bytes, MemKind kind) {
    VkResult err = VK_SUCCESS;
    auto b = create_block(bytes, kind, /*dedicated=*/true, &err);
    if (!b) {
        throw std::runtime_error(std::string("brotensor: vulkan: cannot allocate ") + mib(bytes) +
                                 " staging buffer: " + vk_result_name(err));
    }
    return b;
}

void Allocator::destroy_raw(std::unique_ptr<Block> b) { destroy_block(b.get()); }

// ─── free-range bookkeeping ────────────────────────────────────────────────

void Allocator::erase_free_index(Block* b, VkDeviceSize off, VkDeviceSize size) {
    auto range = by_size_.equal_range(size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.first == b && it->second.second == off) {
            by_size_.erase(it);
            return;
        }
    }
}

void Allocator::insert_free(Block* b, VkDeviceSize off, VkDeviceSize size) {
    auto& fr = b->free_ranges;
    auto next = fr.lower_bound(off);
    if (next != fr.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == off) {
            erase_free_index(b, prev->first, prev->second);
            off = prev->first;
            size += prev->second;
            fr.erase(prev);
        }
    }
    if (next != fr.end() && off + size == next->first) {
        erase_free_index(b, next->first, next->second);
        size += next->second;
        fr.erase(next);
    }
    fr.emplace(off, size);
    by_size_.emplace(size, std::make_pair(b, off));
}

// ─── alloc / free ──────────────────────────────────────────────────────────

void* Allocator::alloc_locked(std::size_t bytes, VkResult* err) {
    *err = VK_SUCCESS;
    const VkDeviceSize size = round_up(bytes, kAlign);
    if (size <= kSubAllocMax) {
        auto it = by_size_.lower_bound(size);
        if (it == by_size_.end()) {
            auto blk = create_block(block_bytes_, MemKind::Device, /*dedicated=*/false, err);
            if (!blk) return nullptr;
            Block* raw = blk.get();
            blocks_.emplace(raw->addr, std::move(blk));
            insert_free(raw, 0, raw->size);
            it = by_size_.lower_bound(size);
        }
        Block* b = it->second.first;
        const VkDeviceSize off = it->second.second;
        const VkDeviceSize have = it->first;
        by_size_.erase(it);
        b->free_ranges.erase(off);
        if (have > size) {
            b->free_ranges.emplace(off + size, have - size);
            by_size_.emplace(have - size, std::make_pair(b, off + size));
        }
        b->used += size;
        const std::uint64_t a = b->addr + off;
        live_[a] = Live{b, off, size};
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(a));
    }

    VkDeviceSize dsize = round_up(bytes, kDedicatedGranule);
    if (bytes <= max_buffer_) dsize = std::min(dsize, max_buffer_);
    auto blk = create_block(dsize, MemKind::Device, /*dedicated=*/true, err);
    if (!blk) return nullptr;
    Block* raw = blk.get();
    raw->used = dsize;
    blocks_.emplace(raw->addr, std::move(blk));
    live_[raw->addr] = Live{raw, 0, dsize};
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(raw->addr));
}

void* Allocator::alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > max_buffer_ && !allow_oversize_) {
        throw std::runtime_error(
            "brotensor: vulkan: cannot allocate " + mib(bytes) + " as one tensor: the per-buffer "
            "limit on this device is " + mib(max_buffer_) + " (Vulkan caps a buffer at 4 GiB). "
            "Split the tensor (e.g. by row ranges); BROTENSOR_VK_ALLOW_OVERSIZE=1 lifts the "
            "check on drivers that accept larger buffers out of spec (docs/vulkan.md).");
    }
    if (has_deferred_.load(std::memory_order_acquire)) collect(dev_.stream().completed());
    VkResult err = VK_SUCCESS;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (void* p = alloc_locked(bytes, &err)) return p;
    }
    if (err == VK_ERROR_OUT_OF_DEVICE_MEMORY || err == VK_ERROR_OUT_OF_HOST_MEMORY) {
        // Retry once with everything reclaimable given back to the driver.
        if (!dev_.stream().capturing()) {
            dev_.stream().sync();
            collect(dev_.stream().completed());
            trim(0);
            std::lock_guard<std::mutex> lk(mu_);
            if (void* p = alloc_locked(bytes, &err)) return p;
        }
    }
    throw std::runtime_error("brotensor: vulkan: out of device memory allocating " + mib(bytes) +
                             " (" + vk_result_name(err) + ")");
}

void Allocator::free(void* p) {
    if (!p) return;
    const std::uint64_t a = addr(p);
    const std::uint64_t tag = dev_.stream().work_serial();
    std::unique_ptr<Block> release_now;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (hold_) {
            hold_->push_back(p);
            return;
        }
        auto it = live_.find(a);
        if (it == live_.end()) {
            throw std::runtime_error("brotensor: vulkan: free of an address this device did not allocate");
        }
        const Live l = it->second;
        live_.erase(it);
        if (!l.block->dedicated) {
            l.block->used -= l.size;
            insert_free(l.block, l.offset, l.size);
            return;
        }
        auto bit = blocks_.find(l.block->addr);
        std::unique_ptr<Block> blk = std::move(bit->second);
        blocks_.erase(bit);
        // (completed() reads the timeline without the stream lock.)
        if (dev_.stream().completed() >= tag) {
            release_now = std::move(blk);
        } else {
            deferred_.push_back(Deferred{tag, std::move(blk)});
            has_deferred_.store(true, std::memory_order_release);
        }
    }
    if (release_now) destroy_block(release_now.get());
}

void Allocator::collect(std::uint64_t completed) {
    std::vector<std::unique_ptr<Block>> done;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (deferred_.empty()) return;
        auto keep = deferred_.begin();
        for (auto it = deferred_.begin(); it != deferred_.end(); ++it) {
            if (it->serial <= completed) done.push_back(std::move(it->block));
            else *keep++ = std::move(*it);
        }
        deferred_.erase(keep, deferred_.end());
        has_deferred_.store(!deferred_.empty(), std::memory_order_release);
    }
    for (auto& b : done) destroy_block(b.get());
}

std::size_t Allocator::trim(std::size_t keep) {
    std::vector<std::unique_ptr<Block>> victims;
    std::size_t released = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        std::size_t kept = 0;
        for (auto it = blocks_.begin(); it != blocks_.end();) {
            Block* b = it->second.get();
            if (b->dedicated || b->used != 0) { ++it; continue; }
            if (kept + b->size <= keep) { kept += b->size; ++it; continue; }
            for (const auto& fr : b->free_ranges) erase_free_index(b, fr.first, fr.second);
            b->free_ranges.clear();
            released += b->size;
            victims.push_back(std::move(it->second));
            it = blocks_.erase(it);
        }
    }
    for (auto& b : victims) destroy_block(b.get());
    return released;
}

Span Allocator::resolve(std::uint64_t a, std::size_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.upper_bound(a);
    if (it != blocks_.begin()) {
        --it;
        const Block* b = it->second.get();
        if (a >= b->addr && a + n <= b->addr + b->size) {
            Span s;
            s.buf = b->buf;
            s.offset = a - b->addr;
            s.host = b->map ? b->map + s.offset : nullptr;
            s.coherent = b->coherent;
            return s;
        }
    }
    throw std::runtime_error("brotensor: vulkan: address range is not inside a live allocation "
                             "of this device (use after free, or a range spanning two tensors)");
}

AllocatorStats Allocator::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    AllocatorStats s;
    for (const auto& kv : blocks_) {
        const Block* b = kv.second.get();
        if (b->dedicated) { ++s.dedicated_blocks; s.dedicated_bytes += b->size; }
        else { ++s.blocks; s.block_bytes += b->size; }
        s.used_bytes += b->used;
    }
    s.live_allocations = live_.size();
    s.deferred_frees = deferred_.size();
    s.max_buffer_bytes = max_buffer_;
    s.host_mapped = mapped_;
    return s;
}

void Allocator::set_hold(std::vector<void*>* hold) {
    std::lock_guard<std::mutex> lk(mu_);
    hold_ = hold;
}

std::vector<void*>* Allocator::hold() const {
    std::lock_guard<std::mutex> lk(mu_);
    return hold_;
}

}  // namespace brotensor::detail::vulkan
