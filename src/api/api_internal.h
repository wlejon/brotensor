#pragma once

#include "brotensor/tensor.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "embed/embed.h"

#include <cstring>
#include <string>

namespace brotensor::api {

struct GpuTensorHandle {
    brotensor::Tensor tensor;
};

inline brotensor::Tensor* toTensor(void* ptr) {
    if (!ptr) return nullptr;
    return &static_cast<GpuTensorHandle*>(ptr)->tensor;
}

inline const brotensor::Tensor* constTensor(const void* ptr) {
    if (!ptr) return nullptr;
    return &static_cast<const GpuTensorHandle*>(ptr)->tensor;
}

inline brotensor::Dtype parseDtype(const char* dt) {
    if (!dt) return brotensor::Dtype::FP32;
    if (std::strcmp(dt, "fp16") == 0) return brotensor::Dtype::FP16;
    if (std::strcmp(dt, "bf16") == 0) return brotensor::Dtype::BF16;
    if (std::strcmp(dt, "int8") == 0) return brotensor::Dtype::INT8;
    if (std::strcmp(dt, "int32") == 0) return brotensor::Dtype::INT32;
    if (std::strcmp(dt, "f64") == 0) return brotensor::Dtype::F64;
    return brotensor::Dtype::FP32;
}

inline const char* dtypeToString(brotensor::Dtype dt) {
    switch (dt) {
        case brotensor::Dtype::FP32: return "fp32";
        case brotensor::Dtype::FP16: return "fp16";
        case brotensor::Dtype::BF16: return "bf16";
        case brotensor::Dtype::INT8: return "int8";
        case brotensor::Dtype::INT32: return "int32";
        case brotensor::Dtype::F64: return "f64";
        default: return "unknown";
    }
}

} // namespace brotensor::api
