#pragma once

// Internal header — re-exposes BROTENSOR_CUDA_CHECK for the CUDA backend TUs
// after the public <brotensor/runtime.h> stopped carrying it.

namespace brotensor::detail::cuda {
void cuda_check_throw(int err, const char* expr_text, const char* file, int line);
}

#define BROTENSOR_CUDA_CHECK(expr)                                                  \
    do {                                                                            \
        int _bga_err = static_cast<int>(expr);                                      \
        if (_bga_err != 0) {                                                        \
            ::brotensor::detail::cuda::cuda_check_throw(                            \
                _bga_err, #expr, __FILE__, __LINE__);                               \
        }                                                                           \
    } while (0)
