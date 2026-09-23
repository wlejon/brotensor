#pragma once

// Opt a kernel in to more than the default 48 KB of dynamic shared memory,
// once per DEVICE. cudaFuncSetAttribute acts on the kernel as loaded into the
// current device's context, so a process-wide "done" flag leaves every device
// but the first one without the opt-in — a launch there fails with
// cudaErrorInvalidValue as soon as a second GPU runs the kernel. `done_mask`
// is the call site's own static bit set (bit = device index, 32 devices).

#include "cuda_check.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace brotensor::detail::cuda {

template <typename Kernel>
inline void opt_in_dynamic_smem(Kernel* kernel, int bytes, std::atomic<std::uint32_t>& done_mask) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) dev = 0;
    const std::uint32_t bit = 1u << dev;
    if (done_mask.load(std::memory_order_acquire) & bit) return;
    BROTENSOR_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    done_mask.fetch_or(bit, std::memory_order_acq_rel);
}

}  // namespace brotensor::detail::cuda
