#pragma once

// Error handling for the Vulkan backend: every failing Vulkan call becomes a
// std::runtime_error carrying the VkResult, the expression and the location,
// the same shape as BROTENSOR_CUDA_CHECK.

#include "vk_fns.h"

namespace brotensor::detail::vulkan {

[[noreturn]] void vk_fail(VkResult r, const char* expr, const char* file, int line);
const char* vk_result_name(VkResult r);

}  // namespace brotensor::detail::vulkan

#define BT_VK_CHECK(expr)                                                         \
    do {                                                                          \
        const VkResult bt_vk_r_ = (expr);                                         \
        if (bt_vk_r_ != VK_SUCCESS)                                               \
            ::brotensor::detail::vulkan::vk_fail(bt_vk_r_, #expr, __FILE__, __LINE__); \
    } while (0)
