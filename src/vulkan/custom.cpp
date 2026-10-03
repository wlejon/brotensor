// Custom kernels: SPIR-V supplied by a library built on brotensor
// (vulkan::register_shader / kernel_info / dispatch, include/brotensor/vulkan.h).
// They share the built-in kernels' pipeline layout, cache, limit guard and
// stream, so a custom dispatch is ordered with brotensor's ops and recorded
// by a graph capture like them. The SPIR-V is compiled by the consumer's
// build with cmake/BrotensorVulkan.cmake's brotensor_vulkan_add_shaders().

#include "detail/device.h"
#include "detail/pipelines.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/vulkan.h>

#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brotensor {

namespace detail::vulkan {

namespace {

struct Registry {
    std::mutex mu;
    std::deque<std::string> names;      // stable storage for ShaderBlob::name
    std::deque<ShaderBlob> blobs;       // handle h is blobs[h - 1]
    std::unordered_map<const std::uint32_t*, std::uint32_t> by_words;
};

Registry& registry() {
    static Registry r;
    return r;
}

}  // namespace

ShaderBlob custom_shader_blob(std::uint32_t handle) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    if (handle == 0 || handle > r.blobs.size()) {
        throw std::runtime_error("brotensor: vulkan: unknown custom shader handle " +
                                 std::to_string(handle));
    }
    return r.blobs[handle - 1];
}

}  // namespace detail::vulkan

namespace vulkan {

namespace {

namespace dv = ::brotensor::detail::vulkan;

dv::DeviceCtx& vulkan_device(Device d, const char* who) {
    d = detail::resolve_device_alias(d);
    if (!d.is_vulkan() || !detail::is_registered(d)) {
        throw std::runtime_error(std::string("brotensor: vulkan::") + who + ": " + to_string(d) +
                                 " is not an available Vulkan device");
    }
    return dv::device(d.index);
}

}  // namespace

ShaderHandle register_shader(const char* name, const std::uint32_t* words, std::size_t nwords) {
    if (!words || nwords < 5 || words[0] != 0x07230203u) {
        throw std::runtime_error(std::string("brotensor: vulkan::register_shader: ") +
                                 (name ? name : "?") + " is not SPIR-V");
    }
    dv::Registry& r = dv::registry();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_words.find(words);
    if (it != r.by_words.end()) return it->second;
    if (r.blobs.size() >= dv::kCustomKeyBase - 1) {
        throw std::runtime_error("brotensor: vulkan::register_shader: too many shaders");
    }
    r.names.emplace_back(name ? name : "custom");
    r.blobs.push_back(dv::ShaderBlob{r.names.back().c_str(), words, nwords});
    const auto h = static_cast<ShaderHandle>(r.blobs.size());
    r.by_words.emplace(words, h);
    return h;
}

KernelInfo kernel_info(Device d, ShaderHandle shader, const std::uint32_t* spec,
                       std::uint32_t nspec, std::uint32_t subgroup) {
    dv::DeviceCtx& ctx = vulkan_device(d, "kernel_info");
    const dv::Kernel& k = ctx.pipelines().get_custom(shader, spec, nspec, subgroup);
    KernelInfo info;
    info.local[0] = k.local[0];
    info.local[1] = k.local[1];
    info.local[2] = k.local[2];
    info.shared_bytes = k.shared_bytes;
    return info;
}

void dispatch(Device d, ShaderHandle shader, const void* push, std::size_t push_bytes,
              std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              const std::uint32_t* spec, std::uint32_t nspec, std::uint32_t subgroup) {
    if (push_bytes > dv::Pipelines::kMaxPushBytes || push_bytes % 4 != 0 ||
        (push_bytes && !push)) {
        throw std::runtime_error("brotensor: vulkan::dispatch: push block must be a multiple of 4 "
                                 "bytes and at most 128");
    }
    if (gx > 65535 || gy > 65535 || gz > 65535) {
        throw std::runtime_error("brotensor: vulkan::dispatch: more than 65535 workgroups in one "
                                 "dimension (use a grid-stride loop or a second dimension)");
    }
    dv::DeviceCtx& ctx = vulkan_device(d, "dispatch");
    const dv::Kernel& k = ctx.pipelines().get_custom(shader, spec, nspec, subgroup);
    ctx.stream().dispatch(k.pipe, push, static_cast<std::uint32_t>(push_bytes), gx, gy, gz);
}

}  // namespace vulkan

}  // namespace brotensor
