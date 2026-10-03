#pragma once

// What an op file needs to launch a kernel: the tensor's device context, its
// address, the dtype -> shader-variant mapping and a typed dispatch.
//
//   DeviceCtx& d = device_of(x);
//   const Kernel& k = d.pipelines().get(dt_variant(ShaderId::unary_f32, x.dtype), {UOP_RELU});
//   UnaryPush pc{addr(x.data), addr(y.data), n, 0.f, 0.f};
//   launch(d, k, pc, groups_1d(n, k));
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

// Shader families selected by dtype: <base>_f32, <base>_f16, <base>_bf16,
// consecutive in shaders.cmake. A new family is added here too; register.cpp
// checks the order of every one at start-up.
inline constexpr ShaderId kDtypeFamilies[] = {
    ShaderId::unary_f32,  ShaderId::binary_f32,   ShaderId::act_bwd_f32,     ShaderId::bias_f32,
    ShaderId::sum_rows_f32, ShaderId::argmax_rows_f32, ShaderId::sum_cols_f32,
    ShaderId::gemm_simt_f32, ShaderId::gemv_f32, ShaderId::glu_f32, ShaderId::norm_f32,
    ShaderId::rope_f32, ShaderId::rowvec_f32, ShaderId::fa_rows_f32, ShaderId::fa_combine_f32,
    ShaderId::attn_softmax_f32, ShaderId::attn_aux_f32, ShaderId::topk_seg_f32,
    ShaderId::conv_simt_f32, ShaderId::conv_direct_f32, ShaderId::resample_f32, ShaderId::gnorm_f32,
    ShaderId::sampler_f32,
};

// The <base>_f16 / <base>_bf16 sibling of a <base>_f32 family shader.
inline ShaderId dt_variant(ShaderId f32_id, ::brotensor::Dtype dt, const char* op = "op") {
    return static_cast<ShaderId>(static_cast<std::uint32_t>(f32_id) +
                                 static_cast<std::uint32_t>(dt_code(dt, op)));
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
