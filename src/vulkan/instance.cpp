// Vulkan instance, physical-device selection and logical-device creation.
// See detail/device.h for the eligibility rules and object lifetimes.

#include "detail/device.h"
#include "detail/vk_check.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

bool env_true(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

// Fills `info` and returns true when `pd` can run the backend; otherwise
// returns false with a reason.
bool probe_physical(const LoaderFns& f, VkPhysicalDevice pd, PhysInfo& info, std::string& why) {
    info.pd = pd;
    f.vkGetPhysicalDeviceProperties(pd, &info.props);
    f.vkGetPhysicalDeviceMemoryProperties(pd, &info.mem);
    info.name = info.props.deviceName;
    if (info.props.apiVersion < VK_API_VERSION_1_2) {
        why = info.name + ": Vulkan 1.2 required";
        return false;
    }
    if (info.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !env_true("BROTENSOR_VK_ALLOW_CPU")) {
        why = info.name + ": CPU implementation skipped (BROTENSOR_VK_ALLOW_CPU=1 allows it)";
        return false;
    }
    const bool v13 = info.props.apiVersion >= VK_API_VERSION_1_3;

    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    p11.pNext = &drv;
    if (v13) drv.pNext = &p13;
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &p11;
    f.vkGetPhysicalDeviceProperties2(pd, &p2);
    info.driver = std::string(drv.driverName) + " " + drv.driverInfo;
    info.subgroup_size = p11.subgroupSize;
    info.min_subgroup = v13 ? p13.minSubgroupSize : p11.subgroupSize;
    info.max_subgroup = v13 ? p13.maxSubgroupSize : p11.subgroupSize;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.pNext = &f12;
    if (v13) f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    f.vkGetPhysicalDeviceFeatures2(pd, &f2);

    struct Need { bool ok; const char* what; };
    const Need needs[] = {
        {f12.bufferDeviceAddress == VK_TRUE, "bufferDeviceAddress"},
        {f12.timelineSemaphore == VK_TRUE, "timelineSemaphore"},
        {f12.shaderFloat16 == VK_TRUE, "shaderFloat16"},
        {f12.shaderInt8 == VK_TRUE, "shaderInt8"},
        {f12.storageBuffer8BitAccess == VK_TRUE, "storageBuffer8BitAccess"},
        {f11.storageBuffer16BitAccess == VK_TRUE, "storageBuffer16BitAccess"},
        {f2.features.shaderInt64 == VK_TRUE, "shaderInt64"},
        {f2.features.shaderInt16 == VK_TRUE, "shaderInt16"},
        {(p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0, "compute subgroups"},
        {(p11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0, "subgroup arithmetic"},
        {(p11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0, "subgroup ballot"},
    };
    for (const Need& n : needs) {
        if (!n.ok) {
            why = info.name + ": missing " + n.what;
            return false;
        }
    }
    info.subgroup_size_control = v13 && f13.subgroupSizeControl == VK_TRUE &&
                                 f13.computeFullSubgroups == VK_TRUE;

    // Queue: graphics + compute (see detail/stream.h for why not compute-only).
    std::uint32_t nq = 0;
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    int gfx = -1, comp = -1;
    for (std::uint32_t i = 0; i < nq; ++i) {
        const bool g = (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool c = (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (g && c && gfx < 0) gfx = static_cast<int>(i);
        if (c && comp < 0) comp = static_cast<int>(i);
    }
    if (gfx < 0 && comp < 0) {
        why = info.name + ": no compute queue";
        return false;
    }
    info.queue_family = static_cast<std::uint32_t>(gfx >= 0 ? gfx : comp);
    info.graphics_queue = gfx >= 0;

    std::uint32_t ne = 0;
    f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, exts.data());
    info.memory_budget = has_extension(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    info.cooperative_matrix = has_extension(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    info.memory_model = f12.vulkanMemoryModel == VK_TRUE && f12.vulkanMemoryModelDeviceScope == VK_TRUE;

    // The GEMM kernels use 16x16x16 FP16 x FP16 -> FP32 subgroup fragments
    // at a pinned subgroup size of 32 (shaders/gemm_cm.comp). Without that
    // shape, the memory model GLSL's coopmat needs, or the subgroup size,
    // the matrix ops run the SIMT kernels.
    if (info.cooperative_matrix && info.memory_model && info.subgroup_size_control &&
        info.min_subgroup <= 32 && info.max_subgroup >= 32 &&
        f.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        std::uint32_t ncm = 0;
        f.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &ncm, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cm(
            ncm, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        f.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &ncm, cm.data());
        for (const auto& p : cm) {
            if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
                p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                p.scope == VK_SCOPE_SUBGROUP_KHR) {
                info.coopmat_f16 = true;
            }
        }
    }
    if (env_true("BROTENSOR_VK_NO_COOPMAT")) info.coopmat_f16 = false;
    return true;
}

int type_rank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:          return 3;
        default:                                     return 4;  // CPU
    }
}

}  // namespace

Instance::Instance() {
    if (env_true("BROTENSOR_DISABLE_VULKAN")) {
        why_ = "disabled by BROTENSOR_DISABLE_VULKAN";
        return;
    }
    const char* why = nullptr;
    if (!load_vulkan_loader(fns_, &why)) {
        why_ = why ? why : "no Vulkan loader";
        return;
    }
    std::uint32_t loader_api = VK_API_VERSION_1_0;
    if (fns_.vkEnumerateInstanceVersion) fns_.vkEnumerateInstanceVersion(&loader_api);
    if (loader_api < VK_API_VERSION_1_2) {
        why_ = "the Vulkan loader is older than 1.2";
        return;
    }
    api_ = std::min<std::uint32_t>(loader_api, VK_API_VERSION_1_3);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "brotensor";
    app.pEngineName = "brotensor";
    app.apiVersion = api_;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const VkResult r = fns_.vkCreateInstance(&ici, nullptr, &inst_);
    if (r != VK_SUCCESS) {
        why_ = std::string("vkCreateInstance failed: ") + vk_result_name(r);
        inst_ = VK_NULL_HANDLE;
        return;
    }
    load_instance_fns(fns_, inst_);

    std::uint32_t n = 0;
    fns_.vkEnumeratePhysicalDevices(inst_, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    fns_.vkEnumeratePhysicalDevices(inst_, &n, pds.data());
    std::string rejected;
    for (VkPhysicalDevice pd : pds) {
        PhysInfo info;
        std::string reason;
        if (probe_physical(fns_, pd, info, reason)) {
            phys_.push_back(std::move(info));
        } else {
            if (!rejected.empty()) rejected += "; ";
            rejected += reason;
        }
    }
    // GPUs before anything else, enumeration order otherwise.
    std::stable_sort(phys_.begin(), phys_.end(), [](const PhysInfo& a, const PhysInfo& b) {
        return type_rank(a.props.deviceType) < type_rank(b.props.deviceType);
    });
    if (phys_.empty()) {
        why_ = rejected.empty() ? "no Vulkan devices" : rejected;
    }
}

Instance& Instance::get() {
    static Instance* inst = new Instance();   // never destroyed, see device.h
    return *inst;
}

// ─── DeviceCtx ──────────────────────────────────────────────────────────────

DeviceCtx::DeviceCtx(int index, const PhysInfo& info) : index_(index), info_(info) {
    const Instance& in = Instance::get();
    const LoaderFns& f = in.fns();
    const bool v13 = info.props.apiVersion >= VK_API_VERSION_1_3;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cmf.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.subgroupSizeControl = info.subgroup_size_control ? VK_TRUE : VK_FALSE;
    f13.computeFullSubgroups = info.subgroup_size_control ? VK_TRUE : VK_FALSE;
    if (info.cooperative_matrix) f13.pNext = &cmf;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    // GL_KHR_cooperative_matrix pulls in GL_KHR_memory_scope_semantics, whose
    // SPIR-V declares the VulkanMemoryModel capability.
    if (info.memory_model) {
        f12.vulkanMemoryModel = VK_TRUE;
        f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    }
    if (v13) f12.pNext = &f13;
    else if (info.cooperative_matrix) f12.pNext = &cmf;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;
    f2.pNext = &f11;

    std::vector<const char*> exts;
    if (info.memory_budget) exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (info.cooperative_matrix) exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = info.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    BT_VK_CHECK(f.vkCreateDevice(info.pd, &dci, nullptr, &dev_));
    load_device_fns(f, dev_, fn_);
    fn_.vkGetDeviceQueue(dev_, info.queue_family, 0, &queue_);

    stream_ = std::make_unique<Stream>(*this);
    alloc_  = std::make_unique<Allocator>(*this);
    pipes_  = std::make_unique<Pipelines>(*this);
}

DeviceCtx::~DeviceCtx() {
    // Never reached for the contexts device() hands out (see device.h); kept
    // correct for completeness.
    if (!dev_) return;
    fn_.vkDeviceWaitIdle(dev_);
    pipes_.reset();
    alloc_.reset();
    stream_.reset();
    fn_.vkDestroyDevice(dev_, nullptr);
}

namespace {
constexpr int kMaxDevices = 16;
std::mutex g_ctx_mu;
std::atomic<DeviceCtx*> g_ctxs[kMaxDevices] = {};
}  // namespace

DeviceCtx& device(int index) {
    if (index < 0 || index >= kMaxDevices) {
        throw std::runtime_error("brotensor: vulkan: device index " + std::to_string(index) +
                                 " out of range");
    }
    if (DeviceCtx* c = g_ctxs[index].load(std::memory_order_acquire)) return *c;
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    if (DeviceCtx* c = g_ctxs[index].load(std::memory_order_acquire)) return *c;
    const Instance& in = Instance::get();
    if (index >= in.count()) {
        throw std::runtime_error("brotensor: vulkan: no device " + std::to_string(index) +
                                 " (" + std::to_string(in.count()) + " available)");
    }
    auto* c = new DeviceCtx(index, in.phys(index));   // never destroyed, see device.h
    g_ctxs[index].store(c, std::memory_order_release);
    return *c;
}

DeviceCtx* device_if_created(int index) {
    if (index < 0 || index >= kMaxDevices) return nullptr;
    return g_ctxs[index].load(std::memory_order_acquire);
}

DeviceCtx& device_of(const ::brotensor::Tensor& t) {
    if (t.device.type != ::brotensor::DeviceType::VULKAN) {
        throw std::runtime_error(std::string("brotensor: vulkan: tensor is on ") +
                                 ::brotensor::device_name(t.device));
    }
    return device(t.device.index);
}

}  // namespace brotensor::detail::vulkan
