// Zero-copy SPIR-V compute interop on brotensor's Vulkan backend.
#include "brotensor/kernel_interop.h"
#include "brotensor/runtime.h"
#include "detail/device.h"
#include "detail/pipelines.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

constexpr std::uint32_t kInteropKeyBase = 0x60000000u;

struct InteropBlobKey {
    std::uint64_t word_hash = 0;
    std::size_t word_count = 0;
    std::string entry;

    bool operator==(const InteropBlobKey& o) const noexcept {
        return word_hash == o.word_hash && word_count == o.word_count && entry == o.entry;
    }
};

struct InteropBlobKeyHash {
    std::size_t operator()(const InteropBlobKey& k) const noexcept {
        std::uint64_t h = k.word_hash ^ (static_cast<std::uint64_t>(k.word_count) * 0x9e3779b97f4a7c15ULL);
        for (unsigned char c : k.entry) h = (h * 1099511628211ULL) ^ c;
        return static_cast<std::size_t>(h);
    }
};

struct InteropShaderEntry {
    std::uint32_t key_id = 0;
    std::string debug_name;
    std::string entry;
    std::vector<std::uint32_t> words;
};

struct InteropRegistry {
    std::mutex mu;
    std::unordered_map<InteropBlobKey, std::size_t, InteropBlobKeyHash> by_key;
    std::deque<InteropShaderEntry> entries;
};

InteropRegistry& interop_registry() {
    static auto* reg = new InteropRegistry();
    return *reg;
}

std::uint64_t fnv1a_words(const std::uint32_t* words, std::size_t count) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(words);
    const std::size_t nbytes = count * sizeof(std::uint32_t);
    for (std::size_t i = 0; i < nbytes; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

const InteropShaderEntry& intern_shader(const std::uint32_t* words, std::size_t nwords, const char* entry) {
    const std::string entry_str = (entry && *entry) ? entry : "main";
    const InteropBlobKey bk{fnv1a_words(words, nwords), nwords, entry_str};
    InteropRegistry& reg = interop_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    auto it = reg.by_key.find(bk);
    if (it != reg.by_key.end()) {
        const InteropShaderEntry& existing = reg.entries[it->second];
        if (existing.words.size() == nwords &&
            std::memcmp(existing.words.data(), words, nwords * sizeof(std::uint32_t)) == 0) {
            return existing;
        }
    }
    const std::size_t idx = reg.entries.size();
    InteropShaderEntry e;
    e.key_id = kInteropKeyBase + static_cast<std::uint32_t>(idx);
    e.debug_name = "brotensor_interop_" + entry_str + "_" + std::to_string(idx);
    e.entry = entry_str;
    e.words.assign(words, words + nwords);
    reg.entries.push_back(std::move(e));
    reg.by_key.emplace(bk, idx);
    return reg.entries.back();
}

bool vk_interop_available(int device_index) {
    return device_index >= 0 && device_index < brotensor::vulkan_device_count() &&
           brotensor::is_available(Device::vulkan(device_index));
}

bool vk_interop_dispatch(int device_index, const brotensor::interop::SpirvDispatchDesc& desc, std::string* err) {
    if (!desc.words || desc.word_count < 5) {
        if (err) *err = "brotensor: interop dispatch requires valid SPIR-V words";
        return false;
    }
    if (desc.push_bytes > Pipelines::kMaxPushBytes || (desc.push_bytes % 4) != 0 ||
        (desc.push_bytes > 0 && !desc.push_data)) {
        if (err) *err = "brotensor: push constants must be a multiple of 4 bytes and at most 128 bytes";
        return false;
    }
    if (desc.groups[0] == 0 || desc.groups[1] == 0 || desc.groups[2] == 0) {
        return true;
    }
    if (desc.groups[0] > 65535 || desc.groups[1] > 65535 || desc.groups[2] > 65535) {
        if (err) *err = "brotensor: interop grid exceeds 65535 workgroups per axis";
        return false;
    }
    if (!vk_interop_available(device_index)) {
        if (err) *err = "brotensor: Vulkan device is not available";
        return false;
    }
    try {
        DeviceCtx& ctx = device(device_index);
        const InteropShaderEntry& entry = intern_shader(desc.words, desc.word_count, desc.entry_name);
        const ShaderBlob blob{entry.debug_name.c_str(), entry.words.data(), entry.words.size()};
        const std::uint32_t spec[3] = {
            desc.local_size[0] ? desc.local_size[0] : 1u,
            desc.local_size[1] ? desc.local_size[1] : 1u,
            desc.local_size[2] ? desc.local_size[2] : 1u,
        };
        const std::uint32_t nspec = desc.local_size_spec_constants ? 3u : 0u;
        const Kernel& k = ctx.pipelines().get_jit(entry.key_id, blob, entry.entry.c_str(), spec, nspec, 0);
        ctx.stream().dispatch(k.pipe, desc.push_data, desc.push_bytes, desc.groups[0], desc.groups[1], desc.groups[2]);
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return false;
    }
}

struct VulkanInteropRegistrar {
    VulkanInteropRegistrar() {
        brotensor::interop::detail::register_vulkan_interop_hooks(vk_interop_available, vk_interop_dispatch);
    }
} g_vulkan_interop_registrar;

} // namespace

} // namespace brotensor::detail::vulkan
