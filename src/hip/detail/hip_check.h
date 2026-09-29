#pragma once

// Internal header — exposes BROTENSOR_HIP_CHECK for the HIP backend TUs.

#include <hip/hip_runtime.h>

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
