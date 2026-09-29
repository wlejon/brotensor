// HIP backend master registration.
//
// Called from `brotensor::init()` (src/init.cpp) when BROTENSOR_HAS_HIP is
// defined. Probes the driver via hipGetDeviceCount(); on success, builds
// the OpsVTable, pairs it with the HIP AllocVTable, and hands the pair to the registry.

#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_runtime.h>

namespace brotensor::detail::hip {

// ── alloc table (defined in tensor.hip) ──
const ::brotensor::detail::AllocVTable& hip_alloc_table();
void init_async_pool_for_device(int dev);

} // namespace brotensor::detail::hip

extern "C" void brotensor_probe_and_register_hip() {
    // Probe: count HIP devices. If the call fails or returns 0 the runtime
    // / driver is missing — bail without registering. We intentionally do
    // NOT throw here so a CPU-only host can still load a binary that was
    // built with HIP support.
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return;
    }

    ::brotensor::detail::set_hip_device_count(device_count);

    // Enable peer access between all GPU pairs and initialize async pools
    for (int i = 0; i < device_count; ++i) {
        if (hipSetDevice(i) != hipSuccess) continue;
        brotensor::detail::hip::init_async_pool_for_device(i);
        for (int j = 0; j < device_count; ++j) {
            if (i != j) {
                int can = 0;
                if (hipDeviceCanAccessPeer(&can, i, j) == hipSuccess && can) {
                    (void)hipDeviceEnablePeerAccess(j, 0);
                    (void)hipGetLastError(); // Clear any benign sticky error
                }
            }
        }
    }

    // Set default active device to 0
    (void)hipSetDevice(0);

    using ::brotensor::DeviceType;
    using ::brotensor::detail::OpsVTable;
    namespace dh = ::brotensor::detail::hip;

    OpsVTable ops{}; // zero-init: every slot starts as nullptr
    // Per-cluster ops will be wired here in subsequent phases

    ::brotensor::detail::register_backend(DeviceType::HIP, ops, dh::hip_alloc_table());
}
