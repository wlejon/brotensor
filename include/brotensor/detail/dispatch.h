#pragma once

#include "../tensor.h"
#include "op_table.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brotensor::detail {

// ─── OpsVTable ─────────────────────────────────────────────────────────────
//
// One function-pointer slot per public op. The slot signature matches the
// public op signature in <brotensor/ops.h> exactly. Generated from the
// single op list in detail/op_table.h so the public surface, the wrappers
// in src/ops.cpp, and each backend's registration table stay in sync.
//
// A null slot means "this backend does not implement this op". The
// dispatcher (`brotensor::detail::dispatch(...)` callers, or the wrapper
// itself) checks for null and throws a "not implemented on this backend"
// std::runtime_error.

#define BROTENSOR_VTABLE_FIELD(name, ret, params) ret (*name) params;

struct OpsVTable {
    BROTENSOR_FOR_EACH_OP(BROTENSOR_VTABLE_FIELD)
};

#undef BROTENSOR_VTABLE_FIELD

// ─── AllocVTable ───────────────────────────────────────────────────────────
//
// Per-backend memory operations. Used by Tensor's destructor / factories /
// to() / clone() / resize() / zero() to manage storage without backend
// awareness leaking into the public Tensor API.
//
// Conventions:
//   * `alloc(0)` returns nullptr; `free(nullptr)` is a no-op.
//   * `memset_zero` is byte-wise zero (matches the GpuTensor::zero contract
//     — fine because FP32, FP16, and INT8 all encode +0 as all-bits-zero).
//   * `sync` waits for pending work on this backend to drain. CPU's slot is
//     a no-op.
struct AllocVTable {
    void* (*alloc)(std::size_t bytes);
    void  (*free)(void* ptr);
    void  (*memcpy_h2d)(void* dst, const void* src, std::size_t n);
    void  (*memcpy_d2h)(void* dst, const void* src, std::size_t n);
    void  (*memcpy_d2d)(void* dst, const void* src, std::size_t n);
    void  (*memset_zero)(void* dst, std::size_t n);
    void  (*sync)();
};

// ─── Backend registration ──────────────────────────────────────────────────
//
// Each backend registers itself once. CPU does so from a static-init object
// (so CPU tensors work without a prior `init()` call); CUDA / Metal do so
// from `brotensor::init()` after a successful driver probe. Registering the
// same Device twice replaces the previous tables — callers should not rely
// on that (it exists for testability).
void register_backend(Device d, const OpsVTable& ops, const AllocVTable& alloc);

// Dispatcher lookups. Throw std::runtime_error if `d` is not currently
// registered.
const OpsVTable&   ops_for(Device d);
const AllocVTable& alloc_for(Device d);

// True iff `d` has been registered (CPU is always true after static init).
bool is_registered(Device d);

// ─── Operand-consistency helpers ───────────────────────────────────────────
//
// Every public op wrapper in `src/ops.cpp` starts with one of these calls.
// They resolve the op's device from the first *committed* operand (one whose
// `data != nullptr`) and verify every other committed operand agrees,
// returning that backend's vtable. On mismatch they throw std::runtime_error
// with a human-readable "operand <i> is on <dev> but the op resolved to
// <dev>" message.
//
// An *uncommitted* operand — a freshly default-constructed or resized-to-empty
// tensor with `data == nullptr` — has no device affinity yet and is treated as
// a wildcard: it is skipped by the consistency check. This is what lets a
// caller pass an unsized output tensor to any op without first pinning it to
// the right backend. The wrapper then calls `adopt_output` (below) so the
// tensor is tagged with the resolved device before the backend impl allocates
// it.
//
// Overloads cover up to 8 operands — enough for the heavier ops (attention,
// resblock, conv2d backward). Optional `const Tensor*` operands should be
// passed via the `_opt` variants, which skip null pointers.
const OpsVTable& dispatch(const Tensor& a);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f,
                          const Tensor& g);
const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f,
                          const Tensor& g, const Tensor& h);

// Variant for ops whose surface includes both required and optional
// operands. Pass any number of required operands followed by a (possibly
// empty) list of optional pointers; null pointers are skipped. The dispatch
// device is the first required operand's device.
const OpsVTable& dispatch_with_opts(const Tensor& a,
                                    std::initializer_list<const Tensor*> opts);
const OpsVTable& dispatch_with_opts(const Tensor& a, const Tensor& b,
                                    std::initializer_list<const Tensor*> opts);

// Pin an uncommitted output tensor to the op's device. If `t` already owns
// storage (data != nullptr) this is a no-op — its device is left untouched so
// a genuine mismatch would have been caught by dispatch(). The wrapper calls
// this for every non-const Tensor& output, after dispatch and before invoking
// the backend impl, so the impl resizes/allocates on the right backend.
void adopt_output(Tensor& t, Device d);

// Throw a "<op>: not implemented on <device>" std::runtime_error. Called
// by the public wrapper when the chosen vtable slot is null.
[[noreturn]] void throw_not_implemented(const char* op_name, Device d);

} // namespace brotensor::detail
