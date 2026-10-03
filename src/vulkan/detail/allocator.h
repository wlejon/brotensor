#pragma once

// Device memory for the Vulkan backend.
//
// Pointer model. A Vulkan tensor's `Tensor::data` is the VkDeviceAddress of
// its first byte (buffer device address), cast to void*. That makes the
// address arithmetic the rest of brotensor does on `data` (views at an element
// offset, the raw `const float* d_mask` / `const int32_t*` operands of the op
// table) work unchanged, and lets a kernel take any number of tensors, at any
// offset, through push constants alone. The host never dereferences it; the
// allocator maps an address back to its (VkBuffer, offset, host mapping) when
// a transfer command or a host copy needs one.
//
// Blocks. Memory comes from the driver in blocks: one VkDeviceMemory with one
// VkBuffer bound over all of it (STORAGE | TRANSFER_SRC/DST | SHADER_DEVICE_
// ADDRESS), mapped persistently when the memory type is host visible.
//   * Requests up to kSubAllocMax are sub-allocated (best fit, 256-byte
//     aligned, coalescing free list) from kBlockBytes blocks.
//   * Larger requests get a dedicated block of their own.
//   * No block is ever larger than the device's per-buffer limit:
//     min(maxMemoryAllocationSize, maxBufferSize), which the spec caps at
//     4 GiB and RADV reports as 4 GiB - 4. A tensor larger than that is
//     refused with an error (callers split it: a large embedding table as
//     several row ranges). BROTENSOR_VK_ALLOW_OVERSIZE=1 lifts the check for
//     drivers that accept bigger buffers out of spec (RADV does), see
//     docs/vulkan.md. Total memory is not limited by this: it is any number
//     of <= 4 GiB blocks.
//
// Reuse is stream ordered. Every command on a device goes through its one
// Stream in submission order with a full memory barrier between commands, and
// every host access to device memory first drains the stream (transfer.cpp).
// So a sub-allocated range can be handed out again the moment it is freed,
// exactly like cudaMallocAsync / cudaFreeAsync on one stream. A dedicated block
// is different only because its VkDeviceMemory is returned to the driver: its
// destruction waits until the GPU has finished the work recorded before the
// free (Stream::work_serial()).
//
// Graph capture holds frees: while a capture is recording, free() does not
// release anything but hands the pointer to the capture, which releases it
// when the replayable graph is destroyed (graph.cpp). Replays therefore never
// touch memory that has been given to somebody else.

#include "vk_fns.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::vulkan {

class DeviceCtx;

struct Block {
    VkDeviceMemory  mem  = VK_NULL_HANDLE;
    VkBuffer        buf  = VK_NULL_HANDLE;
    VkDeviceAddress addr = 0;
    VkDeviceSize    size = 0;
    std::uint8_t*   map  = nullptr;   // persistent host mapping, or null
    bool            dedicated = false;
    bool            coherent  = true; // host mapping needs no invalidate
    VkDeviceSize    used = 0;         // sub-allocated bytes in use
    std::map<VkDeviceSize, VkDeviceSize> free_ranges;  // offset -> size
};

// Where an address range lives. `host` is null when the block is not mapped.
struct Span {
    VkBuffer       buf    = VK_NULL_HANDLE;
    VkDeviceSize   offset = 0;
    std::uint8_t*  host   = nullptr;
    bool           coherent = true;
};

struct AllocatorStats {
    std::size_t blocks           = 0;  // sub-allocation blocks
    std::size_t dedicated_blocks = 0;
    std::size_t block_bytes      = 0;  // driver memory held by sub-allocation blocks
    std::size_t dedicated_bytes  = 0;
    std::size_t used_bytes       = 0;  // bytes handed out (rounded), both kinds
    std::size_t live_allocations = 0;
    std::size_t deferred_frees   = 0;  // dedicated blocks waiting on the GPU
    std::size_t max_buffer_bytes = 0;  // the per-buffer limit
    bool        host_mapped      = false;  // device memory is host visible
};

enum class MemKind {
    Device,        // tensor storage: device local (+ host visible when mapped_)
    HostCached,    // download staging: host visible + cached
    HostUpload,    // upload staging: host visible (+ coherent)
};

class Allocator {
public:
    static constexpr VkDeviceSize kAlign       = 256;
    static constexpr VkDeviceSize kBlockBytes  = VkDeviceSize(256) << 20;   // 256 MiB
    static constexpr VkDeviceSize kSubAllocMax = VkDeviceSize(32) << 20;    // 32 MiB

    explicit Allocator(DeviceCtx& dev);
    ~Allocator();
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    // Tensor storage. Returns the device address as void*; null for 0 bytes.
    // Throws std::runtime_error past the per-buffer limit or when the device
    // is out of memory (after draining the stream and trimming once).
    void* alloc(std::size_t bytes);
    void  free(void* p);

    // Resolve [addr, addr + n) to its block. Throws if the range is not inside
    // one live block of this device.
    Span resolve(std::uint64_t addr, std::size_t n) const;

    // Destroy dedicated blocks whose last use has completed on the GPU.
    void collect(std::uint64_t completed_serial);

    // Release fully free sub-allocation blocks, keeping at most `keep` bytes
    // of them. Caller has drained the stream. Returns bytes released.
    std::size_t trim(std::size_t keep);

    AllocatorStats stats() const;
    VkDeviceSize max_buffer_bytes() const { return max_buffer_; }
    bool host_mapped() const { return mapped_; }

    // Capture support (graph.cpp): while a hold list is installed, free()
    // appends to it instead of releasing.
    void set_hold(std::vector<void*>* hold);
    std::vector<void*>* hold() const;

    // A raw block outside the address registry (staging buffers). Throws on
    // failure. destroy_raw() releases it immediately (caller ensures idle).
    std::unique_ptr<Block> create_raw(VkDeviceSize bytes, MemKind kind);
    void destroy_raw(std::unique_ptr<Block> b);

private:
    std::unique_ptr<Block> create_block(VkDeviceSize bytes, MemKind kind,
                                        bool dedicated, VkResult* err);
    void destroy_block(Block* b);
    void* alloc_locked(std::size_t bytes, VkResult* err);
    void insert_free(Block* b, VkDeviceSize off, VkDeviceSize size);
    void erase_free_index(Block* b, VkDeviceSize off, VkDeviceSize size);
    int  pick_memory_type(std::uint32_t type_bits, MemKind kind) const;

    DeviceCtx& dev_;
    mutable std::mutex mu_;
    bool mapped_ = false;           // MemKind::Device is host visible
    VkDeviceSize max_buffer_ = 0;   // per-buffer limit actually enforced
    bool allow_oversize_ = false;
    VkDeviceSize block_bytes_ = kBlockBytes;

    std::map<std::uint64_t, std::unique_ptr<Block>> blocks_;   // by base address
    // Free-range index across all sub-allocation blocks: size -> (block, offset).
    std::multimap<VkDeviceSize, std::pair<Block*, VkDeviceSize>> by_size_;
    struct Live { Block* block; VkDeviceSize offset; VkDeviceSize size; };
    std::unordered_map<std::uint64_t, Live> live_;               // address -> allocation
    struct Deferred { std::uint64_t serial; std::unique_ptr<Block> block; };
    std::vector<Deferred> deferred_;
    std::atomic<bool> has_deferred_{false};   // lets alloc() skip collect()
    std::vector<void*>* hold_ = nullptr;
};

}  // namespace brotensor::detail::vulkan
