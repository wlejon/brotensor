#pragma once

#include "brotensor/tensor.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "embed/embed.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

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

// The old QuickJS binding accepted "fp32"/"f32", "fp16"/"f16", "bf16",
// "int8"/"i8" (and the numeric Dtype enum, which js/tensor.js maps to the
// string before the call).
inline brotensor::Dtype parseDtype(const char* dt) {
    if (!dt) return brotensor::Dtype::FP32;
    if (std::strcmp(dt, "fp16") == 0 || std::strcmp(dt, "f16") == 0) return brotensor::Dtype::FP16;
    if (std::strcmp(dt, "bf16") == 0) return brotensor::Dtype::BF16;
    if (std::strcmp(dt, "int8") == 0 || std::strcmp(dt, "i8") == 0) return brotensor::Dtype::INT8;
    if (std::strcmp(dt, "int32") == 0 || std::strcmp(dt, "i32") == 0) return brotensor::Dtype::INT32;
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

// ---- error channel ---------------------------------------------------------
//
// A native is called directly from generated code, so a C++ exception must
// not escape it (bronze's fatal.h: those frames carry no unwind metadata).
// A native that fails records the message here instead; the JS wrapper reads
// it back through `__bro_native.tensor.takeError()` right after the call and
// throws. Per thread, like the realm the natives serve.
std::string& lastErrorSlot();
inline void setError(const std::string& msg) { lastErrorSlot() = msg; }

// ---- `dynamic` arguments ---------------------------------------------------
//
// Nullable tensor slots (mask|null, bias|null, ...) cross as `dynamic`: the
// raw Value bits. null / undefined give nullptr. js/tensor*.js checks
// `instanceof GpuTensor` before the call, so the handle data is ours.
inline brotensor::Tensor* tensorFromValue(uint64_t bits) {
    void* d = bronze::embed::handleData(bronze::Value{bits});
    return d ? toTensor(d) : nullptr;
}

// The `const float* d_mask` convention of the ops: the device pointer of an
// optional FP32 GpuTensor, or nullptr.
inline const float* maskPtr(uint64_t bits) {
    auto* t = tensorFromValue(bits);
    return t ? static_cast<const float*>(t->data) : nullptr;
}

// A JS array of GpuTensors (concatRows parts, ...) as tensor pointers. False
// with the error set when an element is not a GpuTensor.
bool readTensorArray(uint64_t bits, std::vector<brotensor::Tensor*>& out, const char* label);

// True when every pointer is non-null; otherwise records "<label>: GpuTensor
// argument missing" and answers false, so a body can `if (!need(...)) return;`.
inline bool need(const char* label, std::initializer_list<const void*> ptrs) {
    for (const void* p : ptrs) {
        if (!p) {
            setError(std::string(label) + ": GpuTensor argument missing");
            return false;
        }
    }
    return true;
}

// Path resolution for the file loaders (api.h setPathResolver).
std::string resolvePath(const std::string& path);

} // namespace brotensor::api

// Guard a native body that calls into brotensor: the ops throw
// std::runtime_error on a shape or dtype contract violation, and that has to
// become a recorded error, never an unwind through generated code.
#define BROTENSOR_API_TRY try {
#define BROTENSOR_API_CATCH(label) \
    } catch (const std::exception& e) { \
        ::brotensor::api::setError(std::string(label) + ": " + e.what()); \
    }
