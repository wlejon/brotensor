// Compute pipeline cache + the shared push-constant-only pipeline layout.
// Design notes: detail/pipelines.h.

#include "detail/device.h"
#include "detail/pipelines.h"
#include "detail/spirv_reflect.h"
#include "detail/vk_check.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

Pipelines::Pipelines(DeviceCtx& dev) : dev_(dev) {
    const DeviceFns& f = dev.fn();
    push_bytes_ = std::min<std::uint32_t>(dev.info().props.limits.maxPushConstantsSize, 256);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    BT_VK_CHECK(f.vkCreatePipelineLayout(dev.device(), &pl, nullptr, &layout_));
    // In-memory cache only: the driver keeps its own on-disk shader cache
    // (Mesa: ~/.cache/mesa_shader_cache), so a warm process start is already
    // ~0 ms per pipeline (vk-spike RESULTS.md).
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    BT_VK_CHECK(f.vkCreatePipelineCache(dev.device(), &pci, nullptr, &cache_));
}

Pipelines::~Pipelines() {
    const DeviceFns& f = dev_.fn();
    for (auto& kv : kernels_) f.vkDestroyPipeline(dev_.device(), kv.second->pipe, nullptr);
    f.vkDestroyPipelineCache(dev_.device(), cache_, nullptr);
    f.vkDestroyPipelineLayout(dev_.device(), layout_, nullptr);
}

std::size_t Pipelines::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return kernels_.size();
}

const Kernel& Pipelines::get(ShaderId id, const std::uint32_t* spec, std::uint32_t nspec,
                             std::uint32_t subgroup) {
    return get_blob(static_cast<std::uint32_t>(id), shader_blob(id), spec, nspec, subgroup);
}

const Kernel& Pipelines::get_custom(std::uint32_t handle, const std::uint32_t* spec,
                                    std::uint32_t nspec, std::uint32_t subgroup) {
    {
        // Fast path without building the blob (custom_shader_blob locks too).
        Key key{kCustomKeyBase | handle, subgroup, std::vector<std::uint32_t>(spec, spec + nspec)};
        std::lock_guard<std::mutex> lk(mu_);
        auto it = kernels_.find(key);
        if (it != kernels_.end()) return *it->second;
    }
    return get_blob(kCustomKeyBase | handle, custom_shader_blob(handle), spec, nspec, subgroup);
}

const Kernel& Pipelines::get_blob(std::uint32_t key_id, const ShaderBlob& blob,
                                  const std::uint32_t* spec, std::uint32_t nspec,
                                  std::uint32_t subgroup, const char* entry) {
    Key key{key_id, subgroup, std::vector<std::uint32_t>(spec, spec + nspec)};
    std::lock_guard<std::mutex> lk(mu_);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) return *it->second;

    const PhysInfo& info = dev_.info();
    const VkPhysicalDeviceLimits& lim = info.props.limits;

    // ── limit guard (before the driver sees anything) ──
    SpirvInfo refl;
    std::string why;
    if (!reflect_spirv(blob.words, blob.nwords, spec, nspec, &refl, &why)) {
        throw std::runtime_error(std::string("brotensor: vulkan: shader ") + blob.name +
                                 ": cannot verify its limits: " + why);
    }
    if (refl.shared_bytes > lim.maxComputeSharedMemorySize) {
        throw std::runtime_error(std::string("brotensor: vulkan: shader ") + blob.name + " needs " +
                                 std::to_string(refl.shared_bytes) + " bytes of shared memory; the device allows " +
                                 std::to_string(lim.maxComputeSharedMemorySize));
    }
    const std::uint64_t invocations = std::uint64_t(refl.local[0]) * refl.local[1] * refl.local[2];
    if (invocations == 0 || invocations > lim.maxComputeWorkGroupInvocations ||
        refl.local[0] > lim.maxComputeWorkGroupSize[0] || refl.local[1] > lim.maxComputeWorkGroupSize[1] ||
        refl.local[2] > lim.maxComputeWorkGroupSize[2]) {
        throw std::runtime_error(std::string("brotensor: vulkan: shader ") + blob.name +
                                 ": workgroup size " + std::to_string(refl.local[0]) + "x" +
                                 std::to_string(refl.local[1]) + "x" + std::to_string(refl.local[2]) +
                                 " exceeds the device limits");
    }
    if (subgroup) {
        if (!info.subgroup_size_control || subgroup < info.min_subgroup || subgroup > info.max_subgroup) {
            throw std::runtime_error(std::string("brotensor: vulkan: shader ") + blob.name +
                                     ": subgroup size " + std::to_string(subgroup) +
                                     " is not available on this device");
        }
        if (refl.local[0] % subgroup != 0) {
            throw std::runtime_error(std::string("brotensor: vulkan: shader ") + blob.name +
                                     ": local_size_x must be a multiple of the required subgroup size");
        }
    }

    const DeviceFns& f = dev_.fn();
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = blob.nwords * sizeof(std::uint32_t);
    smi.pCode = blob.words;
    VkShaderModule sm = VK_NULL_HANDLE;
    BT_VK_CHECK(f.vkCreateShaderModule(dev_.device(), &smi, nullptr, &sm));

    std::vector<VkSpecializationMapEntry> entries(nspec);
    for (std::uint32_t i = 0; i < nspec; ++i) entries[i] = {i, i * 4u, 4u};
    VkSpecializationInfo si{nspec, entries.data(), nspec * sizeof(std::uint32_t), spec};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = subgroup;

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.pNext = subgroup ? &rss : nullptr;
    ci.stage.flags = subgroup ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = sm;
    ci.stage.pName = entry;
    ci.stage.pSpecializationInfo = nspec ? &si : nullptr;
    ci.layout = layout_;
    VkPipeline pipe = VK_NULL_HANDLE;
    const VkResult r = f.vkCreateComputePipelines(dev_.device(), cache_, 1, &ci, nullptr, &pipe);
    f.vkDestroyShaderModule(dev_.device(), sm, nullptr);
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("brotensor: vulkan: creating pipeline ") + blob.name +
                                 " failed: " + vk_result_name(r));
    }

    auto k = std::make_unique<Kernel>();
    k->pipe = pipe;
    k->local[0] = refl.local[0];
    k->local[1] = refl.local[1];
    k->local[2] = refl.local[2];
    k->shared_bytes = static_cast<std::uint32_t>(refl.shared_bytes);
    k->name = blob.name;
    const Kernel& ref = *k;
    kernels_.emplace(std::move(key), std::move(k));
    return ref;
}

}  // namespace brotensor::detail::vulkan
