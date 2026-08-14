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
void* cuda_current_stream(int dev = -1);
}

namespace brotensor::detail::cuda {

namespace {

constexpr int kMaxCudaDevices = 16;

struct DevicePool {
    bool initialized = false;
    bool supported = false;
    cudaMemPool_t pool = nullptr;
};

DevicePool g_device_pools[kMaxCudaDevices];

int resolve_device_index(int dev) {
    if (dev < 0) {
        int d = 0;
        if (cudaGetDevice(&d) == cudaSuccess && d >= 0 && d < kMaxCudaDevices) {
            return d;
        }
        return 0;
    }
    if (dev >= kMaxCudaDevices) return 0;
    return dev;
}

}  // namespace

void init_async_pool_for_device(int dev) {
    dev = resolve_device_index(dev);
    if (g_device_pools[dev].initialized) return;
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev) == cudaSuccess && supported) {
        cudaMemPool_t pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
            std::uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            g_device_pools[dev].supported = true;
            g_device_pools[dev].pool = pool;
        }
    }
    g_device_pools[dev].initialized = true;
    if (prior != dev) cudaSetDevice(prior);
}

bool async_pool_ready_for(int dev) {
    dev = resolve_device_index(dev);
    if (!g_device_pools[dev].initialized) {
        init_async_pool_for_device(dev);
    }
    return g_device_pools[dev].supported;
}

void* cuda_alloc(std::size_t bytes, int dev) {
    if (bytes == 0) return nullptr;
    dev = resolve_device_index(dev);
    void* p = nullptr;
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(dev));
        BROTENSOR_CUDA_CHECK(cudaMallocAsync(&p, bytes, s));
    } else {
        BROTENSOR_CUDA_CHECK(cudaMalloc(&p, bytes));
    }
    if (prior != dev) cudaSetDevice(prior);
    return p;
}

void* cuda_alloc(std::size_t bytes) {
    return cuda_alloc(bytes, -1);
}

void cuda_free(void* ptr, int dev) {
    if (!ptr) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(dev));
        cudaError_t err = cudaFreeAsync(ptr, s);
        if (err != cudaSuccess) {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(s, &cs);
            if (cs != cudaStreamCaptureStatusNone) {
                std::fprintf(stderr,
                    "brotensor: WARNING: cudaFreeAsync(%p) failed (err=%d) on dev %d during stream capture\n",
                    ptr, static_cast<int>(err), dev);
            }
            cudaGetLastError();
        }
    } else {
        cudaFree(ptr);
    }
    if (prior != dev) cudaSetDevice(prior);
}

void cuda_free(void* ptr) {
    cuda_free(ptr, -1);
}

void cuda_memcpy_h2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
    if (prior != dev) cudaSetDevice(prior);
}

void cuda_memcpy_d2h(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
    if (prior != dev) cudaSetDevice(prior);
}

void cuda_memcpy_d2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(dev));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s));
    if (prior != dev) cudaSetDevice(prior);
}

void cuda_memcpy_peer(void* dst, int dst_dev, const void* src, int src_dev, std::size_t n) {
    if (n == 0) return;
    dst_dev = resolve_device_index(dst_dev);
    src_dev = resolve_device_index(src_dev);
    if (dst_dev == src_dev) {
        cuda_memcpy_d2d(dst, src, n, dst_dev);
        return;
    }
    int prior = 0;
    cudaGetDevice(&prior);
    cudaSetDevice(dst_dev);
    auto dst_stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(dst_dev));
    auto src_stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(src_dev));

    // Event on src_stream ensures prior writes on src_stream land before dst_stream reads
    cudaEvent_t ev = nullptr;
    BROTENSOR_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    BROTENSOR_CUDA_CHECK(cudaEventRecord(ev, src_stream));
    BROTENSOR_CUDA_CHECK(cudaStreamWaitEvent(dst_stream, ev, 0));
    BROTENSOR_CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, n, dst_stream));
    BROTENSOR_CUDA_CHECK(cudaEventDestroy(ev));
    if (prior != dst_dev) cudaSetDevice(prior);
}

void cuda_memset_zero(void* dst, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    auto s = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream(dev));
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dst, 0, n, s));
    if (prior != dev) cudaSetDevice(prior);
}

void cuda_sync(int dev) {
    int prior = 0;
    cudaGetDevice(&prior);
    if (dev >= 0) {
        cudaSetDevice(dev);
        BROTENSOR_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        int count = 0;
        if (cudaGetDeviceCount(&count) == cudaSuccess) {
            for (int i = 0; i < count; ++i) {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaSetDevice(prior);
}

bool cuda_mem_info(std::size_t* free_bytes, std::size_t* total_bytes, int dev) {
    dev = resolve_device_index(dev);
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    std::size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        cudaGetLastError();
        if (prior != dev) cudaSetDevice(prior);
        return false;
    }
    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    if (prior != dev) cudaSetDevice(prior);
    return true;
}

bool cuda_device_name(char* out, std::size_t cap, int dev) {
    if (!out || cap == 0) return false;
    dev = resolve_device_index(dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    std::snprintf(out, cap, "%s", prop.name);
    return true;
}

bool cuda_mem_trim(std::size_t keep_bytes, int dev) {
    dev = resolve_device_index(dev);
    if (!async_pool_ready_for(dev)) return false;
    int prior = 0;
    cudaGetDevice(&prior);
    if (prior != dev) cudaSetDevice(dev);
    cudaMemPool_t pool = g_device_pools[dev].pool;
    if (!pool && cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) {
        if (prior != dev) cudaSetDevice(prior);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (prior != dev) cudaSetDevice(prior);
        return false;
    }
    if (cudaMemPoolTrimTo(pool, keep_bytes) != cudaSuccess) {
        cudaGetLastError();
        if (prior != dev) cudaSetDevice(prior);
        return false;
    }
    if (prior != dev) cudaSetDevice(prior);
    return true;
}

const ::brotensor::detail::AllocVTable& cuda_alloc_table() {
    static const ::brotensor::detail::AllocVTable t = {
        &cuda_alloc,
        &cuda_free,
        &cuda_memcpy_h2d,
        &cuda_memcpy_d2h,
        &cuda_memcpy_d2d,
        &cuda_memcpy_peer,
        &cuda_memset_zero,
        &cuda_sync,
        &cuda_mem_info,
        &cuda_mem_trim,
        &cuda_device_name,
    };
    return t;
}

} // namespace brotensor::detail::cuda
