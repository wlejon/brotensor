// CUDA AllocVTable.
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
#include <cstdint>
#include <cstdio>

namespace brotensor {
// Forward decl: thread-local current stream from runtime.cu.
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

// Tensor allocation is the per-op fixed cost in the inference loops: every op
// allocates its output and frees its temporaries, so a plain cudaMalloc /
// cudaFree pair — both of which synchronize the device — serializes the whole
// pipeline (thousands of tiny ops per frame). The stream-ordered allocator
// (cudaMallocAsync / cudaFreeAsync, CUDA 11.2+) draws from a memory pool that
// caches freed blocks for reuse without a device sync, so steady-state
// allocation costs nothing on the host timeline. We raise the pool's release
// threshold so freed blocks stay resident across iterations instead of being
// handed back to the driver. Devices without pool support fall back to the
// synchronous path.
bool init_async_pool() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev)
            != cudaSuccess || !supported)
        return false;
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) return false;
    std::uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    return true;
}

// Resolved once on the first allocation (a CUDA context exists by then). Stable
// for the rest of the run, so every pointer is allocated and freed on the same
// (async or sync) path — never a cross-path mismatch.
bool async_pool_ready() {
    static const bool ok = init_async_pool();
    return ok;
}

}  // namespace

void* cuda_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* p = nullptr;
    if (async_pool_ready()) {
        auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
        BROTENSOR_CUDA_CHECK(cudaMallocAsync(&p, bytes, s));
    } else {
        BROTENSOR_CUDA_CHECK(cudaMalloc(&p, bytes));
    }
    return p;
}

void cuda_free(void* ptr) {
    if (!ptr) return;
    if (async_pool_ready()) {
        // Best-effort, stream-ordered; unchecked so teardown after context
        // destruction stays quiet, matching the old cudaFree behaviour.
        auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
        cudaError_t err = cudaFreeAsync(ptr, s);
        if (err != cudaSuccess) {
            // Quiet by design at teardown (context may already be gone) —
            // but clear the sticky error so it cannot surface at an
            // unrelated later CUDA call and misattribute the failure. A
            // free failing while the stream is CAPTURING is a real bug
            // (freeing non-graph memory inside a capture poisons the
            // graph), so that case gets a loud warning with the pointer.
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(s, &cs);
            if (cs != cudaStreamCaptureStatusNone) {
                std::fprintf(stderr,
                    "brotensor: WARNING: cudaFreeAsync(%p) failed (err=%d) "
                    "during stream capture — a non-graph allocation was "
                    "freed inside a captured region\n",
                    ptr, static_cast<int>(err));
            }
            cudaGetLastError();
        }
    } else {
        cudaFree(ptr);
    }
}

void cuda_memcpy_h2d(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

// Device-to-device copy and device fill are stream-ordered (async on the
// current stream): both touch only device memory, so there is no host-buffer
// lifetime hazard, and a same-stream consumer (e.g. the attention kernel reading
// a freshly written KV-cache slot) sees the result in order. Synchronous
// cudaMemcpy/cudaMemset here would stall the host on every call — and the
// inference loops issue these per layer (KV-cache writes), so the stalls
// serialized the whole pipeline. h2d stays synchronous (its host source may be
// a temporary the caller frees on return).
void cuda_memcpy_d2d(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s));
}

void cuda_memset_zero(void* dst, std::size_t n) {
    if (n == 0) return;
    auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dst, 0, n, s));
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
