#pragma once

#include "brotensor/tensor.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "embed/embed.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>

namespace brotensor::api {

struct GpuTensorHandle {
    brotensor::Tensor tensor;
    std::vector<int64_t> shape;
};

inline GpuTensorHandle* toHandle(void* ptr) {
    if (!ptr) return nullptr;
    return static_cast<GpuTensorHandle*>(ptr);
}

inline const GpuTensorHandle* constHandle(const void* ptr) {
    if (!ptr) return nullptr;
    return static_cast<const GpuTensorHandle*>(ptr);
}

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

// ---- input-size contract ---------------------------------------------------
//
// brotensor ops trust their caller's explicit dims: conv2d_forward reads
// N*C_in*H*W elements of X because the caller said so, whatever X holds, and
// an elementwise backward reads dY.size() == x.size() elements without
// checking. From C++ that is the library's contract; from JS it is an
// out-of-bounds read (or write) on the host or the device. So every native
// that hands brotensor a dim or a second operand checks the operand first.
//
// Outputs are exempt: the ops resize them.

// `t` holds at least `n` elements. `n` is computed in 64 bits by the caller
// (dims multiply past INT_MAX easily); a negative `n` means a negative dim.
inline bool needElems(const char* label, const char* name,
                      const brotensor::Tensor* t, int64_t n) {
    if (!t) {
        setError(std::string(label) + ": " + name + " must be a GpuTensor");
        return false;
    }
    if (n < 0) {
        setError(std::string(label) + ": negative dimension for " + name);
        return false;
    }
    if (static_cast<int64_t>(t->size()) < n || (n > 0 && t->data == nullptr)) {
        setError(std::string(label) + ": " + name + " holds " + std::to_string(t->size()) +
                 " elements, the dims need " + std::to_string(n));
        return false;
    }
    return true;
}

// `t` has exactly `ref`'s element count (the elementwise-pair case).
inline bool needSameSize(const char* label, const char* name,
                         const brotensor::Tensor* t, const brotensor::Tensor* ref) {
    if (!t || !ref) {
        setError(std::string(label) + ": " + name + " must be a GpuTensor");
        return false;
    }
    if (t->size() != ref->size()) {
        setError(std::string(label) + ": " + name + " has " + std::to_string(t->size()) +
                 " elements, expected " + std::to_string(ref->size()));
        return false;
    }
    return true;
}

// `t` carries `ref`'s dtype (a kernel dispatches on one operand's dtype and
// reads the others with that element size).
inline bool needSameDtype(const char* label, const char* name,
                          const brotensor::Tensor* t, const brotensor::Tensor* ref) {
    if (!t || !ref) {
        setError(std::string(label) + ": " + name + " must be a GpuTensor");
        return false;
    }
    if (t->dtype != ref->dtype) {
        setError(std::string(label) + ": " + name + " is " + dtypeToString(t->dtype) +
                 ", expected " + dtypeToString(ref->dtype));
        return false;
    }
    return true;
}

// Product of dims in 64 bits; negative when any dim is negative.
inline int64_t elems(std::initializer_list<int64_t> dims) {
    int64_t n = 1;
    for (int64_t d : dims) {
        if (d < 0) return -1;
        n *= d;
    }
    return n;
}

// An elementwise partner: same element count and dtype as `ref`.
inline bool needPair(const char* label, const char* name,
                     const brotensor::Tensor* t, const brotensor::Tensor* ref) {
    return needSameSize(label, name, t, ref) && needSameDtype(label, name, t, ref);
}

// Each value is > 0 (kernel sizes, strides, dilations, groups, heads: a zero
// is a division by zero inside the op, which on the CPU backend is a crash).
inline bool needPositive(const char* label, std::initializer_list<int64_t> vals) {
    for (int64_t v : vals) {
        if (v <= 0) {
            setError(std::string(label) + ": kernel/stride/dilation/group/head counts must be positive");
            return false;
        }
    }
    return true;
}

// Each value is >= 0 (dims, paddings, offsets).
inline bool needNonNegative(const char* label, std::initializer_list<int64_t> vals) {
    for (int64_t v : vals) {
        if (v < 0) {
            setError(std::string(label) + ": dims, paddings and offsets must be non-negative");
            return false;
        }
    }
    return true;
}

// 2-D convolution geometry: positive kernel/stride/dilation/groups, groups
// dividing both channel counts, non-negative dims and paddings. Answers the
// output extent the op will produce (0 when the window does not fit).
inline bool convGeom2d(const char* label, int64_t C_in, int64_t C_out, int64_t H, int64_t W,
                       int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                       int64_t pH, int64_t pW, int64_t dH, int64_t dW, int64_t groups,
                       int64_t& H_out, int64_t& W_out) {
    if (!needPositive(label, {kH, kW, sH, sW, dH, dW, groups})) return false;
    if (!needNonNegative(label, {C_in, C_out, H, W, pH, pW})) return false;
    if (C_in % groups != 0 || C_out % groups != 0) {
        setError(std::string(label) + ": groups must divide C_in and C_out");
        return false;
    }
    const int64_t spanH = H + 2 * pH - dH * (kH - 1) - 1;
    const int64_t spanW = W + 2 * pW - dW * (kW - 1) - 1;
    H_out = spanH < 0 ? 0 : spanH / sH + 1;
    W_out = spanW < 0 ? 0 : spanW / sW + 1;
    return true;
}

// One operand of a multi-tensor op: `t` must hold `n` elements. A null `t`
// is an absent optional operand and passes (need() has already rejected a
// missing required one).
struct Operand {
    const char* name;
    const brotensor::Tensor* t;
    int64_t n;
};

// Every operand holds its element count and, when `ref` is given, carries
// ref's dtype.
inline bool needOperands(const char* label, const brotensor::Tensor* ref,
                         std::initializer_list<Operand> ops) {
    for (const Operand& o : ops) {
        if (!o.t) continue;
        if (!needElems(label, o.name, o.t, o.n)) return false;
        if (ref && !needSameDtype(label, o.name, o.t, ref)) return false;
    }
    return true;
}

// The ops' `const float* d_mask` operands (and the FP32 bias tables): an
// optional FP32 tensor of at least `n` elements.
inline bool needMask(const char* label, const brotensor::Tensor* mt, int64_t n, const char* name = "mask") {
    if (!mt) return true;
    if (!needElems(label, name, mt, n)) return false;
    if (mt->dtype != brotensor::Dtype::FP32) {
        setError(std::string(label) + ": " + name + " must be FP32");
        return false;
    }
    return true;
}

// numHeads is positive and divides D.
inline bool needHeads(const char* label, int64_t D, int64_t numHeads) {
    if (!needPositive(label, {numHeads})) return false;
    if (D % numHeads != 0) {
        setError(std::string(label) + ": numHeads (" + std::to_string(numHeads) + ") must divide D (" +
                 std::to_string(D) + ")");
        return false;
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
