// Run-time compiled (brass SPIR-V) kernels on the Vulkan backend. Design
// notes: vulkan_jit.h.

#include "vulkan_jit.h"

#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

// One interned module: its words, entry point, and the devices it has
// already been checked against.
struct Entry {
    std::vector<std::uint32_t> words;
    std::string entry;
    std::string name;                 // stable storage for Kernel::name
    std::uint32_t key_id = 0;
    std::vector<int> checked;         // device indices the capability check passed on
};

struct Registry {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::vector<Entry*>> by_hash;
    std::deque<std::unique_ptr<Entry>> entries;   // never shrinks: pipelines refer to names
};

Registry& registry() {
    static Registry* r = new Registry();   // outlives every DeviceCtx (never destroyed)
    return *r;
}

std::uint64_t hash_words(const std::vector<std::uint32_t>& w, const std::string& entry) {
    std::uint64_t h = 14695981039346656037ULL;
    auto mix = [&h](std::uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    for (std::uint32_t x : w) mix(x);
    for (char c : entry) mix(static_cast<unsigned char>(c));
    return h;
}

Entry& intern(const brass::target::SpirvKernel& k) {
    Registry& r = registry();
    const std::uint64_t h = hash_words(k.words, k.entry);
    std::lock_guard<std::mutex> lk(r.mu);
    std::vector<Entry*>& bucket = r.by_hash[h];
    for (Entry* e : bucket) {
        if (e->entry == k.entry && e->words == k.words) return *e;
    }
    if (r.entries.size() >= kCustomKeyBase - kJitKeyBase) {
        throw std::runtime_error("brotensor: vulkan: too many run-time compiled kernels");
    }
    auto e = std::make_unique<Entry>();
    e->words = k.words;
    e->entry = k.entry;
    e->name = "jit:" + k.entry;
    e->key_id = kJitKeyBase | static_cast<std::uint32_t>(r.entries.size());
    Entry* raw = e.get();
    r.entries.push_back(std::move(e));
    bucket.push_back(raw);
    return *raw;
}

}  // namespace

std::string jit_missing_for(DeviceCtx& ctx, const brass::target::SpirvKernel& k) {
    const PhysInfo& info = ctx.info();
    std::ostringstream out;
    for (const std::string& cap : k.capabilities) {
        // Enabled on every device the backend accepts (instance.cpp: the
        // eligibility list and the features it turns on).
        if (cap == "Shader" || cap == "Int64" || cap == "Int16" || cap == "Int8" ||
            cap == "Float16" || cap == "PhysicalStorageBufferAddresses" ||
            cap == "StorageBuffer8BitAccess" || cap == "StorageBuffer16BitAccess" ||
            cap == "GroupNonUniform") {
            continue;
        }
        if (cap == "Float64") {
            if (!info.shader_float64) out << "  capability Float64 needs shaderFloat64\n";
            continue;
        }
        if (cap == "GroupNonUniformShuffle") {
            if (!(info.subgroup_ops & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) {
                out << "  capability GroupNonUniformShuffle needs subgroup SHUFFLE operations in compute\n";
            }
            continue;
        }
        out << "  capability " << cap << " is not one brotensor's Vulkan device enables\n";
    }
    const std::uint32_t push_limit = ctx.pipelines().push_bytes();
    if (k.push_constant_bytes > push_limit) {
        out << "  push constants take " << k.push_constant_bytes << " bytes; the pipeline layout has "
            << push_limit << "\n";
    }
    const VkPhysicalDeviceLimits& lim = info.props.limits;
    if (k.shared_bytes > lim.maxComputeSharedMemorySize) {
        out << "  shared memory takes " << k.shared_bytes << " bytes; the device allows "
            << lim.maxComputeSharedMemorySize << "\n";
    }
    return out.str();
}

JitPipeline jit_pipeline(DeviceCtx& ctx, const brass::target::SpirvKernel& k, std::uint32_t block_x) {
    if (k.words.empty()) {
        throw std::runtime_error("brotensor: vulkan: kernel '" + k.entry + "' has no SPIR-V words");
    }
    Entry& e = intern(k);

    bool checked = false;
    {
        std::lock_guard<std::mutex> lk(registry().mu);
        for (int d : e.checked) checked = checked || d == ctx.index();
    }
    if (!checked) {
        const std::string missing = jit_missing_for(ctx, k);
        if (!missing.empty()) {
            throw std::runtime_error("brotensor: vulkan: run-time compiled kernel '" + k.entry +
                                     "' cannot run on " + ctx.info().name + ":\n" + missing);
        }
        std::lock_guard<std::mutex> lk(registry().mu);
        e.checked.push_back(ctx.index());
    }

    const PhysInfo& info = ctx.info();
    std::uint32_t subgroup = 0;
    if (info.subgroup_size_control && info.min_subgroup <= 32 && info.max_subgroup >= 32 &&
        block_x % 32 == 0) {
        subgroup = 32;
    }
    std::uint32_t spec[3] = {block_x, 1, 1};
    if (!k.local_size_spec_constants && k.local_size[0] != block_x) {
        throw std::runtime_error("brotensor: vulkan: kernel '" + k.entry +
                                 "' has a fixed workgroup size other than the one requested");
    }
    const ShaderBlob blob{e.name.c_str(), e.words.data(), e.words.size()};
    const Kernel& kern = ctx.pipelines().get_jit(e.key_id, blob, e.entry.c_str(), spec,
                                                 k.local_size_spec_constants ? 3u : 0u, subgroup);
    JitPipeline p;
    p.pipe = kern.pipe;
    p.block = block_x;
    p.push_bytes = k.push_constant_bytes;
    p.subgroup = subgroup;
    return p;
}

std::size_t jit_module_count() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    return r.entries.size();
}

}  // namespace brotensor::detail::vulkan
