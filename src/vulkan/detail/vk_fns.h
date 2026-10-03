#pragma once

// Vulkan entry points, loaded at run time.
//
// The backend never links libvulkan: a binary built with BROTENSOR_WITH_VULKAN
// has to start (and fall back to CPU / HIP) on a machine with no Vulkan loader
// at all, the same reason the CUDA backend links cudart statically and never
// libcuda. So VK_NO_PROTOTYPES is set, the loader library is dlopen()ed once
// (loader.cpp), and every call goes through one of the tables below.
//
// Device-level functions are fetched per VkDevice with vkGetDeviceProcAddr,
// which also skips the loader's dispatch trampoline on the recording hot path.
//
// Adding a call: add it to the matching X-list. Global = callable before an
// instance exists; instance = takes a VkInstance / VkPhysicalDevice; device =
// everything else.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define BT_VK_GLOBAL_FNS(X)                     \
    X(vkCreateInstance)                         \
    X(vkEnumerateInstanceVersion)               \
    X(vkEnumerateInstanceExtensionProperties)

#define BT_VK_INSTANCE_FNS(X)                   \
    X(vkDestroyInstance)                        \
    X(vkEnumeratePhysicalDevices)               \
    X(vkGetPhysicalDeviceProperties)            \
    X(vkGetPhysicalDeviceProperties2)           \
    X(vkGetPhysicalDeviceFeatures2)             \
    X(vkGetPhysicalDeviceMemoryProperties)      \
    X(vkGetPhysicalDeviceMemoryProperties2)     \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkEnumerateDeviceExtensionProperties)     \
    X(vkCreateDevice)                           \
    X(vkGetDeviceProcAddr)

#define BT_VK_DEVICE_FNS(X)                     \
    X(vkDestroyDevice)                          \
    X(vkGetDeviceQueue)                         \
    X(vkDeviceWaitIdle)                         \
    X(vkAllocateMemory)                         \
    X(vkFreeMemory)                             \
    X(vkMapMemory)                              \
    X(vkUnmapMemory)                            \
    X(vkInvalidateMappedMemoryRanges)           \
    X(vkCreateBuffer)                           \
    X(vkDestroyBuffer)                          \
    X(vkGetBufferMemoryRequirements)            \
    X(vkBindBufferMemory)                       \
    X(vkGetBufferDeviceAddress)                 \
    X(vkCreateCommandPool)                      \
    X(vkDestroyCommandPool)                     \
    X(vkAllocateCommandBuffers)                 \
    X(vkFreeCommandBuffers)                     \
    X(vkResetCommandBuffer)                     \
    X(vkBeginCommandBuffer)                     \
    X(vkEndCommandBuffer)                       \
    X(vkQueueSubmit)                            \
    X(vkCreateSemaphore)                        \
    X(vkDestroySemaphore)                       \
    X(vkWaitSemaphores)                         \
    X(vkGetSemaphoreCounterValue)               \
    X(vkCreateShaderModule)                     \
    X(vkDestroyShaderModule)                    \
    X(vkCreatePipelineLayout)                   \
    X(vkDestroyPipelineLayout)                  \
    X(vkCreateComputePipelines)                 \
    X(vkDestroyPipeline)                        \
    X(vkCreatePipelineCache)                    \
    X(vkDestroyPipelineCache)                   \
    X(vkCmdBindPipeline)                        \
    X(vkCmdPushConstants)                       \
    X(vkCmdDispatch)                            \
    X(vkCmdPipelineBarrier)                     \
    X(vkCmdCopyBuffer)                          \
    X(vkCmdFillBuffer)                          \
    X(vkCmdUpdateBuffer)

namespace brotensor::detail::vulkan {

#define BT_VK_DECLARE_FN(name) PFN_##name name = nullptr;

// Process-wide: the loader's vkGetInstanceProcAddr, the global functions and,
// once the instance exists, the instance-level functions.
struct LoaderFns {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    BT_VK_GLOBAL_FNS(BT_VK_DECLARE_FN)
    BT_VK_INSTANCE_FNS(BT_VK_DECLARE_FN)
};

// One per VkDevice.
struct DeviceFns {
    BT_VK_DEVICE_FNS(BT_VK_DECLARE_FN)
};

#undef BT_VK_DECLARE_FN

// Opens the Vulkan loader library and resolves the global functions. False
// (and a reason in *why) when there is no loader on this machine. Idempotent.
bool load_vulkan_loader(LoaderFns& fns, const char** why);

// Resolves the instance-level table for `inst`.
void load_instance_fns(LoaderFns& fns, VkInstance inst);

// Resolves the device-level table for `dev`. Throws if a function is missing.
void load_device_fns(const LoaderFns& lf, VkDevice dev, DeviceFns& out);

}  // namespace brotensor::detail::vulkan
