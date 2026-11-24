#pragma once

// What an op file needs to launch a kernel: the tensor's device context, its
// address and a typed dispatch.
//
// Push blocks are plain structs mirroring the shader's push_constant block
// under std430 (uint64_t fields first, 8-byte aligned; sizeof rounds to 8).

#include "device.h"
#include "op_codes.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

// 0 = FP32, 1 = FP16, 2 = BF16 (the shaders' DT codes). Throws for any other
// dtype, naming the op.
inline int dt_code(::brotensor::Dtype dt, const char* op) {
    switch (dt) {
        case ::brotensor::Dtype::FP32: return 0;
        case ::brotensor::Dtype::FP16: return 1;
        case ::brotensor::Dtype::BF16: return 2;
        default:
            throw std::runtime_error(std::string("brotensor: ") + op +
                                     ": unsupported dtype on Vulkan (FP32 / FP16 / BF16)");
    }
}

// Workgroups for a grid-stride kernel over n items, one item per invocation
// up to a cap that keeps the dispatch inside every device's limit.
inline std::uint32_t groups_1d(std::uint64_t n, const Kernel& k, std::uint32_t cap = 65535) {
    const std::uint64_t g = (n + k.local[0] - 1) / k.local[0];
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(g, cap)));
}

template <class Push>
inline void launch(DeviceCtx& d, const Kernel& k, const Push& pc, std::uint32_t gx,
                   std::uint32_t gy = 1, std::uint32_t gz = 1) {
    static_assert(sizeof(Push) <= Pipelines::kMaxPushBytes,
                  "push-constant block above the portable 128-byte minimum");
    d.stream().dispatch(k.pipe, &pc, static_cast<std::uint32_t>(sizeof(Push)), gx, gy, gz);
}

// Element count of a tensor as the 32-bit count the kernels take.
inline std::uint32_t count32(const ::brotensor::Tensor& t, const char* op) {
    const long long n = static_cast<long long>(t.rows) * t.cols;
    if (n < 0 || n > 0x7fffffffLL) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": tensor too large");
    }
    return static_cast<std::uint32_t>(n);
}

}  // namespace brotensor::detail::vulkan
