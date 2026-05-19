#pragma once

// ─── Device tag for backend-aware layers ───────────────────────────────────
//
// Lightweight enum + guard helper so consumer libraries (brogameagent, …)
// can answer "where do my weights currently live?" without inventing their
// own Device type. Layers that mirror parameters across host/device hold a
// `Device device_` field plus optional GpuTensor mirrors of every host
// Tensor they own. `to(Device)` migrates parameters/grads/velocities/caches
// host↔device.
//
// Calling to(Device::GPU) on a CPU-only build (BROTENSOR_HAS_GPU undefined)
// is a runtime error: see device_require_gpu() below.

#include <stdexcept>
#include <string>

namespace brotensor {

enum class Device { CPU, GPU };

// Throws std::runtime_error with a readable message when GPU is requested
// but no GPU backend was compiled in. Layers call this from `to(GPU)`.
inline void device_require_gpu(const char* layer_name) {
#ifndef BROTENSOR_HAS_GPU
    throw std::runtime_error(
        std::string("brotensor: cannot move ") + layer_name +
        " to Device::GPU — built without BROTENSOR_HAS_GPU (no GPU backend compiled in)");
#else
    (void)layer_name;
#endif
}

} // namespace brotensor
