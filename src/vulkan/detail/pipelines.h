#pragma once

// Compute pipelines, created on first use and cached per device.
//
// Shaders are GLSL under src/vulkan/shaders/, compiled to SPIR-V by glslc at
// build time and embedded in the library (the generated shader_table.h /
// shader_table.cpp in the build tree); nothing is read from disk at run time.
// A pipeline is identified by (ShaderId, specialisation constants, required
// subgroup size) and created once per device.
//
// Descriptor-free. Every pipeline shares one VkPipelineLayout that has no
// descriptor sets, only a push-constant range: kernels receive tensors as
// 64-bit buffer device addresses inside their push-constant block (plus the
// scalars), so binding a kernel is vkCmdBindPipeline + vkCmdPushConstants and
// nothing else. Keep a kernel's push block <= 128 bytes (the Vulkan minimum
// guarantee; RADV allows 256): 16 addresses, or fewer plus scalars.
//
// Limit guard. Before vkCreateComputePipelines the SPIR-V is reflected
// (spirv_reflect.h) with the specialisation applied, and the workgroup size and
// the total shared (Workgroup storage) memory are checked against the device
// limits; a violation throws instead of reaching the driver. RADV raises
// SIGFPE inside vkCreateComputePipelines for > 64 KiB of shared memory rather
// than returning an error (vk-spike RESULTS.md).

#include "vk_fns.h"
#include "shader_table.h"   // generated: enum class ShaderId, shader_spirv()

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace brotensor::detail::vulkan {

class DeviceCtx;

struct Kernel {
    VkPipeline    pipe = VK_NULL_HANDLE;
    std::uint32_t local[3] = {1, 1, 1};   // workgroup size after specialisation
    std::uint32_t shared_bytes = 0;       // reflected shared memory
    const char*   name = "";
};

class Pipelines {
public:
    static constexpr std::uint32_t kMaxPushBytes = 128;

    explicit Pipelines(DeviceCtx& dev);
    ~Pipelines();
    Pipelines(const Pipelines&) = delete;
    Pipelines& operator=(const Pipelines&) = delete;

    // The pipeline for `id` with specialisation constants spec[0..n) bound to
    // constant_id 0..n-1. `subgroup` != 0 requests that exact subgroup size
    // (VK_EXT_subgroup_size_control / Vulkan 1.3; throws if unsupported).
    const Kernel& get(ShaderId id, const std::uint32_t* spec = nullptr,
                      std::uint32_t nspec = 0, std::uint32_t subgroup = 0);
    const Kernel& get(ShaderId id, std::initializer_list<std::uint32_t> spec,
                      std::uint32_t subgroup = 0) {
        return get(id, spec.begin(), static_cast<std::uint32_t>(spec.size()), subgroup);
    }

    VkPipelineLayout layout() const { return layout_; }
    std::uint32_t push_bytes() const { return push_bytes_; }
    std::size_t size() const;

private:
    struct Key {
        std::uint32_t id;
        std::uint32_t subgroup;
        std::vector<std::uint32_t> spec;
        bool operator<(const Key& o) const {
            if (id != o.id) return id < o.id;
            if (subgroup != o.subgroup) return subgroup < o.subgroup;
            return spec < o.spec;
        }
    };

    DeviceCtx& dev_;
    mutable std::mutex mu_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::uint32_t push_bytes_ = 0;
    std::map<Key, std::unique_ptr<Kernel>> kernels_;
};

}  // namespace brotensor::detail::vulkan
