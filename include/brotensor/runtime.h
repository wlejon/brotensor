#pragma once

#include "tensor.h"

#include <vector>

namespace brotensor {

// ─── Backend lifecycle ─────────────────────────────────────────────────────
//
// Idempotent. Probes available backends and registers them. Safe to call
// repeatedly. The CPU backend self-registers at static-init time, so it is
// always available even if `init()` is never called; `init()` is responsible
// for the CUDA / Metal driver probes that need explicit invocation.
//
// If init() is never called, CPU is the only available backend.
void init();

// ─── Default-device policy ─────────────────────────────────────────────────

// Returns the device the next zeros/empty/from_host call will land on.
// Default policy: best available — CUDA > Metal > CPU. Overridable via
// set_default_device() or the BROTENSOR_DEFAULT_DEVICE environment variable
// (one of "cpu", "cuda", "metal").
Device default_device();

// Globally override the default device. Throws std::runtime_error if `d` is
// not currently registered. Affects every thread that doesn't have an
// active DeviceScope.
void set_default_device(Device d);

// Backends actually registered in this binary at runtime. CPU is always
// present; CUDA / Metal appear only if their backend was both compiled in
// and successfully probed by init().
std::vector<Device> available_devices();
bool is_available(Device);

// Thread-local scope override. Default device for tensor construction
// inside the scope is `d`. Restored on destruction. Throws on construction
// if `d` is not currently registered.
class DeviceScope {
    Device prev_;
public:
    explicit DeviceScope(Device d);
    ~DeviceScope();

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;
};

// ─── Synchronisation ───────────────────────────────────────────────────────
//
// Wait for pending work on a backend to drain. No-op on CPU. Throws
// std::runtime_error if `d` isn't registered.
void sync(Device d);

// Sync every registered backend.
void sync_all();

// ─── Errors ────────────────────────────────────────────────────────────────
//
// Backend impls throw plain std::runtime_error with a readable
// "brotensor: <op>: <reason>" message for op precondition / dispatch
// failures. No named exception type for now — kept as std::runtime_error
// for ABI continuity and to avoid a public error-type hierarchy.

} // namespace brotensor
