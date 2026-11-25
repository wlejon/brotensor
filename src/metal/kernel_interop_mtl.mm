// Zero-copy MSL compute interop on brotensor's Metal backend.
#include "brotensor/kernel_interop.h"
#include "brotensor/runtime.h"

#import "internal.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::metal {

namespace {

struct MslCacheKey {
    std::uint64_t hash = 0;
    std::size_t bytes = 0;
    std::string entry;

    bool operator==(const MslCacheKey& o) const noexcept {
        return hash == o.hash && bytes == o.bytes && entry == o.entry;
    }
};

struct MslCacheKeyHash {
    std::size_t operator()(const MslCacheKey& k) const noexcept {
        std::uint64_t h = k.hash ^ (static_cast<std::uint64_t>(k.bytes) * 0x9e3779b97f4a7c15ULL);
        for (unsigned char c : k.entry) h = (h * 1099511628211ULL) ^ c;
        return static_cast<std::size_t>(h);
    }
};

struct MslCacheEntry {
    std::string source;
    id<MTLComputePipelineState> pso = nil;
};

struct MslInteropCache {
    std::mutex mu;
    std::unordered_map<MslCacheKey, std::vector<MslCacheEntry>, MslCacheKeyHash> by_key;
};

MslInteropCache& msl_interop_cache() {
    static auto* cache = new MslInteropCache();
    return *cache;
}

std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

id<MTLComputePipelineState> get_or_compile_msl(const char* src, std::size_t len, const char* entry) {
    const std::string entry_str(entry);
    const MslCacheKey key{fnv1a_bytes(src, len), len, entry_str};
    MslInteropCache& cache = msl_interop_cache();
    {
        std::lock_guard<std::mutex> lk(cache.mu);
        auto it = cache.by_key.find(key);
        if (it != cache.by_key.end()) {
            for (const auto& e : it->second) {
                if (e.source.size() == len && std::memcmp(e.source.data(), src, len) == 0) {
                    return e.pso;
                }
            }
        }
    }

    std::string src_str(src, len);
    id<MTLComputePipelineState> pso = nil;
    @autoreleasepool {
        NSString* ns_src = [NSString stringWithUTF8String:src_str.c_str()];
        NSString* ns_entry = [NSString stringWithUTF8String:entry_str.c_str()];
        pso = metal_impl::compile_pipeline(ns_src, ns_entry);
    }

    std::lock_guard<std::mutex> lk(cache.mu);
    auto& bucket = cache.by_key[key];
    for (const auto& e : bucket) {
        if (e.source.size() == len && std::memcmp(e.source.data(), src, len) == 0) {
            return e.pso;
        }
    }
    bucket.push_back(MslCacheEntry{std::move(src_str), pso});
    return pso;
}

bool mtl_interop_available(int device_index) {
    return device_index == 0 && brotensor::is_available(Device::Metal) &&
           metal_impl::device() != nil;
}

bool mtl_interop_dispatch_msl(int device_index, const interop::MslDispatchDesc& desc, std::string* err) {
    if (!desc.msl || desc.msl_bytes == 0) {
        if (err) *err = "dispatch_msl: empty MSL source";
        return false;
    }
    if (!desc.entry_name || !desc.entry_name[0]) {
        if (err) *err = "dispatch_msl: missing kernel entry name";
        return false;
    }
    if (desc.buffer_count > 31u || (desc.buffer_count > 0 && !desc.buffer_ptrs)) {
        if (err) *err = "dispatch_msl: invalid buffer_ptrs/buffer_count";
        return false;
    }
    if (desc.scalar_bytes > 4096u || (desc.scalar_bytes > 0 && !desc.scalar_data)) {
        if (err) *err = "dispatch_msl: invalid scalar_data/scalar_bytes (max 4096 bytes)";
        return false;
    }
    if (desc.groups[0] == 0 || desc.groups[1] == 0 || desc.groups[2] == 0) {
        return true;
    }
    if (!mtl_interop_available(device_index)) {
        if (err) *err = "dispatch_msl: Metal device is not available";
        return false;
    }

    try {
        const std::size_t msl_len =
            (desc.msl[desc.msl_bytes - 1] == '\0') ? (desc.msl_bytes - 1) : desc.msl_bytes;
        id<MTLComputePipelineState> pso = get_or_compile_msl(desc.msl, msl_len, desc.entry_name);

        @autoreleasepool {
            std::array<id<MTLBuffer>, 31> bufs{};
            std::array<NSUInteger, 31> offs{};
            id<MTLBuffer> fb_buf = nil;
            NSUInteger fb_ofs = 0;
            for (std::uint32_t i = 0; i < desc.buffer_count; ++i) {
                if (desc.buffer_ptrs[i]) {
                    bufs[i] = metal_impl::pool_require(desc.buffer_ptrs[i], offs[i],
                                                       "dispatch_msl", "buffer");
                    if (!fb_buf) {
                        fb_buf = bufs[i];
                        fb_ofs = offs[i];
                    }
                }
            }

            id<MTLCommandBuffer> cmd = metal_impl::new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (std::uint32_t i = 0; i < desc.buffer_count; ++i) {
                if (bufs[i] || fb_buf) {
                    [enc setBuffer:(bufs[i] ? bufs[i] : fb_buf)
                            offset:(bufs[i] ? offs[i] : fb_ofs)
                           atIndex:i];
                }
            }
            if (desc.scalar_bytes > 0) {
                const NSUInteger scalar_idx =
                    desc.scalar_buffer_index ? desc.scalar_buffer_index : desc.buffer_count;
                [enc setBytes:desc.scalar_data length:desc.scalar_bytes atIndex:scalar_idx];
            }
            const NSUInteger bx = desc.local_size[0] ? desc.local_size[0] : 1u;
            const NSUInteger by = desc.local_size[1] ? desc.local_size[1] : 1u;
            const NSUInteger bz = desc.local_size[2] ? desc.local_size[2] : 1u;
            [enc dispatchThreadgroups:MTLSizeMake(desc.groups[0], desc.groups[1], desc.groups[2])
                threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
            [enc endEncoding];
            metal_impl::submit(cmd);
        }
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("dispatch_msl: ") + ex.what();
        return false;
    }
}

} // namespace

void register_metal_interop_backend() {
    interop::detail::register_metal_interop_hooks(mtl_interop_available, mtl_interop_dispatch_msl);
}

} // namespace brotensor::detail::metal
