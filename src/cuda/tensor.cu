// CUDA AllocVTable. Phase 2G.
//
// All Tensor allocation / freeing / memcpy / zero / sync on the CUDA device
// is routed through `cuda_alloc_table()` (see brotensor/detail/dispatch.h).
// The vtable is exposed via a function-local-static so static-init order
// does not matter — the first call from `register.cu` constructs it.
//
// The old `GpuTensor` constructor/destructor + upload/download free
// functions + fp16/fp32 bit conversion helpers that used to live in this
// file are gone — the new `brotensor::Tensor` in src/tensor.cpp subsumes
// every one of them, and the conversion helpers moved with it.

#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace brotensor::detail::cuda {

void* cuda_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* p = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void cuda_memcpy_h2d(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_d2d(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
}

void cuda_memset_zero(void* dst, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemset(dst, 0, n));
}

void cuda_sync() {
    BROTENSOR_CUDA_CHECK(cudaDeviceSynchronize());
}

const ::brotensor::detail::AllocVTable& cuda_alloc_table() {
    static const ::brotensor::detail::AllocVTable t = {
        &cuda_alloc,
        &cuda_free,
        &cuda_memcpy_h2d,
        &cuda_memcpy_d2h,
        &cuda_memcpy_d2d,
        &cuda_memset_zero,
        &cuda_sync,
    };
    return t;
}

} // namespace brotensor::detail::cuda
