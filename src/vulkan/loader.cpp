// Vulkan loader: dlopen()s the system Vulkan loader and resolves the entry
// points listed in detail/vk_fns.h. See that header for why the backend does
// not link libvulkan.

#include "detail/vk_check.h"
#include "detail/vk_fns.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace brotensor::detail::vulkan {

namespace {

void* open_loader() {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA("vulkan-1.dll"));
#elif defined(__APPLE__)
    const char* names[] = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"};
    for (const char* n : names) {
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    }
    return nullptr;
#else
    const char* names[] = {"libvulkan.so.1", "libvulkan.so"};
    for (const char* n : names) {
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    }
    return nullptr;
#endif
}

void* find_symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

}  // namespace

bool load_vulkan_loader(LoaderFns& fns, const char** why) {
    if (fns.vkGetInstanceProcAddr) return true;
    void* lib = open_loader();   // kept open for the life of the process
    if (!lib) {
        if (why) *why = "no Vulkan loader library (libvulkan) on this system";
        return false;
    }
    fns.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        find_symbol(lib, "vkGetInstanceProcAddr"));
    if (!fns.vkGetInstanceProcAddr) {
        if (why) *why = "the Vulkan loader does not export vkGetInstanceProcAddr";
        return false;
    }
#define BT_VK_LOAD_GLOBAL(name) \
    fns.name = reinterpret_cast<PFN_##name>(fns.vkGetInstanceProcAddr(VK_NULL_HANDLE, #name));
    BT_VK_GLOBAL_FNS(BT_VK_LOAD_GLOBAL)
#undef BT_VK_LOAD_GLOBAL
    if (!fns.vkCreateInstance) {
        if (why) *why = "the Vulkan loader does not provide vkCreateInstance";
        return false;
    }
    return true;
}

void load_instance_fns(LoaderFns& fns, VkInstance inst) {
#define BT_VK_LOAD_INSTANCE(name) \
    fns.name = reinterpret_cast<PFN_##name>(fns.vkGetInstanceProcAddr(inst, #name));
    BT_VK_INSTANCE_FNS(BT_VK_LOAD_INSTANCE)
#undef BT_VK_LOAD_INSTANCE
}

void load_device_fns(const LoaderFns& lf, VkDevice dev, DeviceFns& out) {
#define BT_VK_LOAD_DEVICE(name)                                                       \
    out.name = reinterpret_cast<PFN_##name>(lf.vkGetDeviceProcAddr(dev, #name));     \
    if (!out.name) throw std::runtime_error("brotensor: vulkan: device function " #name " missing");
    BT_VK_DEVICE_FNS(BT_VK_LOAD_DEVICE)
#undef BT_VK_LOAD_DEVICE
}

const char* vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        default: return "VkResult";
    }
}

void vk_fail(VkResult r, const char* expr, const char* file, int line) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "brotensor: vulkan: %s (%d) at %s:%d in `%s`",
                  vk_result_name(r), static_cast<int>(r), file ? file : "?", line,
                  expr ? expr : "?");
    throw std::runtime_error(buf);
}

}  // namespace brotensor::detail::vulkan
