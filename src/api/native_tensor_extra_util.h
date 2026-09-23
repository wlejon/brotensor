#pragma once

// native_tensor_extra_util.h — argument checks shared by the native_tensor_extra*
// files, on top of api_internal.h's needElems / needSameSize / needSameDtype.
//
// Two things the extra group needs beyond those:
//   - dims products that cannot overflow (a JS caller can pass any int32, and
//     five of them multiply past INT64_MAX), and output extents that must fit
//     the int the ops size tensors with;
//   - INT32 index operands whose VALUES address memory (scatter rows, RoPE
//     positions, sequence bounds, segment offsets). The kernels do not
//     bounds-check those, so the native reads them on the host and checks
//     them before the launch. The same read lets an FP32 tensor of whole
//     numbers (what GpuTensor.prototype.upload always produces) stand in for
//     an INT32 one, converted to an INT32 copy on the operand's device.

#include "api_internal.h"

#include <climits>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace brotensor::api::extra {

// Product of dims in 64 bits: -1 when any dim is negative, INT64_MAX when the
// product would overflow (needElems then reports it as too large).
inline int64_t prod(std::initializer_list<int64_t> dims) {
    int64_t n = 1;
    for (int64_t d : dims) if (d < 0) return -1;
    for (int64_t d : dims) {
        if (d == 0) return 0;
        if (n > INT64_MAX / d) return INT64_MAX;
        n *= d;
    }
    return n;
}

// An output extent (or element count) the op will allocate: positive and
// within the int the ops size tensors with.
inline bool needExtent(const char* label, const char* what, int64_t n) {
    if (n <= 0) {
        setError(std::string(label) + ": " + what + " is " + std::to_string(n) + ", must be positive");
        return false;
    }
    if (n > INT_MAX) {
        setError(std::string(label) + ": " + what + " is too large");
        return false;
    }
    return true;
}

// A scalar argument in [lo, hi].
inline bool needRange(const char* label, const char* what, int64_t v, int64_t lo, int64_t hi) {
    if (v < lo || v > hi) {
        setError(std::string(label) + ": " + what + " is " + std::to_string(v) + ", must be in [" +
                 std::to_string(lo) + ", " + std::to_string(hi) + "]");
        return false;
    }
    return true;
}

inline bool needDtype(const char* label, const char* name, const brotensor::Tensor* t, brotensor::Dtype dt) {
    if (!t) {
        setError(std::string(label) + ": " + name + " must be a GpuTensor");
        return false;
    }
    if (t->dtype != dt) {
        setError(std::string(label) + ": " + name + " is " + dtypeToString(t->dtype) + ", expected " +
                 dtypeToString(dt));
        return false;
    }
    return true;
}

// FP32 / FP16 / BF16 — the float storage the dtype-dispatched kernels read.
inline bool needFloat(const char* label, const char* name, const brotensor::Tensor* t) {
    if (!t) {
        setError(std::string(label) + ": " + name + " must be a GpuTensor");
        return false;
    }
    if (t->dtype != brotensor::Dtype::FP32 && t->dtype != brotensor::Dtype::FP16 &&
        t->dtype != brotensor::Dtype::BF16) {
        setError(std::string(label) + ": " + name + " is " + dtypeToString(t->dtype) +
                 ", expected fp32, fp16 or bf16");
        return false;
    }
    return true;
}

// An optional operand: null passes; otherwise it must hold `n` elements of
// `ref`'s dtype (ref null = any dtype).
inline bool optElems(const char* label, const char* name, const brotensor::Tensor* t, int64_t n,
                     const brotensor::Tensor* ref) {
    if (!t) return true;
    if (!needElems(label, name, t, n)) return false;
    return !ref || needSameDtype(label, name, t, ref);
}

// An optional FP32 operand (the `const float* d_mask` slots): null passes.
inline bool optF32Elems(const char* label, const char* name, const brotensor::Tensor* t, int64_t n) {
    if (!t) return true;
    return needDtype(label, name, t, brotensor::Dtype::FP32) && needElems(label, name, t, n);
}

// The values of an index operand on the host: INT32 read as-is, FP32 whole
// numbers converted. Anything else (or fewer than `n` elements) is an error.
bool hostInt32(const char* label, const char* name, const brotensor::Tensor* t, int64_t n,
               std::vector<int32_t>& out);

// `t` itself when it is already INT32, else an INT32 copy of `vals` shaped
// like `t` on `t`'s device, held in `scratch`.
const brotensor::Tensor& int32Operand(const brotensor::Tensor& t, const std::vector<int32_t>& vals,
                                      brotensor::Tensor& scratch);

// Every one of the first `n` values lies in [lo, hi).
bool needIndicesIn(const char* label, const char* name, const std::vector<int32_t>& vals, int64_t n,
                   int64_t lo, int64_t hi);

} // namespace brotensor::api::extra
