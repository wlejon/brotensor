#pragma once

// Internal header — exposes BROTENSOR_HIP_CHECK for the HIP backend TUs.

#include <hip/hip_runtime.h>

// The kernels assume Wave32, RDNA's 64 KB workgroup LDS and WMMA matrix
// cores: RDNA 3+ only (cmake/BrotensorHip.cmake rejects other targets; this
// catches a build that bypassed it).
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX8__) || defined(__GFX9__) || defined(__GFX10__))
#error "brotensor HIP kernels need an RDNA 3+ (gfx11xx / gfx12xx) target"
#endif

namespace brotensor::detail::hip {
void hip_check_throw(hipError_t err, const char* expr_text, const char* file, int line);
}

#define BROTENSOR_HIP_CHECK(expr)                                                   \
    do {                                                                            \
        hipError_t _bga_err = (expr);                                               \
        if (_bga_err != hipSuccess) {                                               \
            ::brotensor::detail::hip::hip_check_throw(                              \
                _bga_err, #expr, __FILE__, __LINE__);                               \
        }                                                                           \
    } while (0)
