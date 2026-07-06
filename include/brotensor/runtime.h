#pragma once

#include "tensor.h"

#include <string>
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

// Joins the CPU backend's background worker threads. Call this
// deterministically as part of your own process/engine shutdown sequence,
// before returning from main() — worker threads are otherwise daemon
// threads that live until the pool's Meyers-singleton destructor runs
// during the process's static-destruction phase, by which point every
// other thread has already been suspended by the OS. A worker suspended
// mid-op while holding some global lock (e.g. the Debug CRT's
// iterator-checking mutex) can deadlock the main thread's own exit-time
// destructors waiting on that same lock forever. Idempotent; safe to call
// even if init() was never called or no CPU work ever ran.
void shutdown();

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

// ─── Compute-precision policy ──────────────────────────────────────────────
//
// The dtype a backend computes in: FP32 on the CPU backend (CPU is FP32-only
// by design), FP16 on a GPU backend (where the half-precision kernels pay for
// themselves). Derived from default_device(), so it tracks set_default_device()
// and any active DeviceScope. This is the single decision point a model loader
// uses to pick the dtype it uploads weights at.
Dtype compute_dtype();

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

// ─── Memory ────────────────────────────────────────────────────────────────
//
// Device-wide free/total memory in bytes for `d` (e.g. cudaMemGetInfo).
// Returns false — leaving the outputs untouched — when the backend is not
// registered or cannot report (CPU always returns false).
bool device_mem_info(Device d, std::size_t& free_bytes,
                     std::size_t& total_bytes);

// Return the backend allocator's cached-but-unused memory to the driver,
// keeping at most `keep_bytes` cached. Synchronizes the device first so
// stream-ordered frees are actually reclaimable. Use between pipeline phases
// with very different scratch shapes: cached blocks count against device
// residency, and on Windows (WDDM) sustained near-full commit makes the OS
// demote large resident allocations to shared memory — which silently turns
// weight reads into PCIe traffic. Returns false when the backend is not
// registered or has no trimmable allocator (CPU always returns false).
bool device_mem_trim(Device d, std::size_t keep_bytes = 0);

// The device's human-readable product name (e.g. cudaDeviceProp.name, "NVIDIA
// GeForce RTX 4090") — distinct from device_name(), which is the backend kind
// ("cuda"/"metal"/"cpu"). Returns "" when the backend is not registered or
// cannot report (CPU always returns "").
std::string device_product_name(Device d);

// ─── Errors ────────────────────────────────────────────────────────────────
//
// Backend impls throw plain std::runtime_error with a readable
// "brotensor: <op>: <reason>" message for op precondition / dispatch
// failures. No named exception type for now — kept as std::runtime_error
// for ABI continuity and to avoid a public error-type hierarchy.

} // namespace brotensor
