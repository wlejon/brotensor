#pragma once

#include <hipblas/hipblas.h>

namespace brotensor::detail::hip {

hipblasHandle_t hipblas_handle(int dev = -1);

} // namespace brotensor::detail::hip
