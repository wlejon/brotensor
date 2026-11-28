// CUDA AllocVTable (pure C++ backed by the dynamic CUDA Driver API).

#include "cuda_driver.h"
#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace brotensor {
void* cuda_current_stream(int dev = -1);
}

namespace brotensor::detail::cuda {

namespace {

constexpr int kMaxCudaDevices = 16;

struct DevicePool {
    bool initialized = false;
    bool supported = false;
    CUmemoryPool pool = nullptr;
};

DevicePool g_device_pools[kMaxCudaDevices];

int resolve_device_index(int dev) {
    if (dev < 0) dev = current_device();
    if (dev < 0 || dev >= kMaxCudaDevices) return 0;
    return dev;
}

CUdeviceptr as_devptr(const void* p) noexcept {
    return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(p));
}

void* from_devptr(CUdeviceptr p) noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(p));
}

struct DeviceScopeGuard {
    int prior = 0;
    bool switched = false;
    explicit DeviceScopeGuard(int dev) : prior(current_device()) {
        if (prior != dev && set_device(dev) == CUDA_SUCCESS) {
            switched = true;
        }
    }
    ~DeviceScopeGuard() {
        if (switched) (void)set_device(prior);
    }
};

}  // namespace

void init_async_pool_for_device(int dev) {
    dev = resolve_device_index(dev);
    if (g_device_pools[dev].initialized) return;
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    CUdevice cu_dev = 0;
    if (drv::cuDeviceGet(&cu_dev, dev) == CUDA_SUCCESS) {
        int supported = 0;
        if (drv::cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, cu_dev) == CUDA_SUCCESS &&
            supported) {
            CUmemoryPool pool = nullptr;
            if (drv::cuDeviceGetDefaultMemPool(&pool, cu_dev) == CUDA_SUCCESS && pool) {
                std::uint64_t threshold = UINT64_MAX;
                (void)drv::cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);
                g_device_pools[dev].supported = true;
                g_device_pools[dev].pool = pool;
            }
        }
    }
    g_device_pools[dev].initialized = true;
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
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    CUdeviceptr p = 0;
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dev));
        BROTENSOR_CUDA_CHECK(drv::cuMemAllocAsync(&p, bytes, s));
    } else {
        BROTENSOR_CUDA_CHECK(drv::cuMemAlloc(&p, bytes));
    }
    return from_devptr(p);
}

void* cuda_alloc(std::size_t bytes) {
    return cuda_alloc(bytes, -1);
}

void cuda_free(void* ptr, int dev) {
    if (!ptr) return;
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    const CUdeviceptr dptr = as_devptr(ptr);
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dev));
        const CUresult err = drv::cuMemFreeAsync(dptr, s);
        if (err != CUDA_SUCCESS) {
            CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
            drv::cuStreamIsCapturing(s, &cs);
            if (cs != CU_STREAM_CAPTURE_STATUS_NONE) {
                std::fprintf(stderr,
                    "brotensor: WARNING: cuMemFreeAsync(%p) failed (err=%d) on dev %d during stream capture\n",
                    ptr, static_cast<int>(err), dev);
            }
        }
    } else {
        (void)drv::cuMemFree(dptr);
    }
}

void cuda_free(void* ptr) {
    cuda_free(ptr, -1);
}

void cuda_memcpy_h2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    BROTENSOR_CUDA_CHECK(drv::cuMemcpyHtoD(as_devptr(dst), src, n));
}

void cuda_memcpy_d2h(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    BROTENSOR_CUDA_CHECK(drv::cuMemcpyDtoH(dst, as_devptr(src), n));
}

void cuda_memcpy_d2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    auto s = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dev));
    BROTENSOR_CUDA_CHECK(drv::cuMemcpyDtoDAsync(as_devptr(dst), as_devptr(src), n, s));
}

void cuda_memcpy_peer(void* dst, int dst_dev, const void* src, int src_dev, std::size_t n) {
    if (n == 0) return;
    dst_dev = resolve_device_index(dst_dev);
    src_dev = resolve_device_index(src_dev);
    if (dst_dev == src_dev) {
        cuda_memcpy_d2d(dst, src, n, dst_dev);
        return;
    }
    DeviceScopeGuard guard(dst_dev);
    CUcontext dst_ctx = primary_context(dst_dev);
    CUcontext src_ctx = primary_context(src_dev);
    auto dst_stream = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dst_dev));
    auto src_stream = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(src_dev));

    BROTENSOR_CUDA_CHECK(set_device(src_dev));
    CUevent ev = nullptr;
    BROTENSOR_CUDA_CHECK(drv::cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
    BROTENSOR_CUDA_CHECK(drv::cuEventRecord(ev, src_stream));
    BROTENSOR_CUDA_CHECK(set_device(dst_dev));
    BROTENSOR_CUDA_CHECK(drv::cuStreamWaitEvent(dst_stream, ev, 0));
    BROTENSOR_CUDA_CHECK(drv::cuMemcpyPeerAsync(as_devptr(dst), dst_ctx, as_devptr(src), src_ctx, n, dst_stream));
    BROTENSOR_CUDA_CHECK(drv::cuEventDestroy(ev));
}

void cuda_memset_zero(void* dst, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    auto s = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dev));
    BROTENSOR_CUDA_CHECK(drv::cuMemsetD8Async(as_devptr(dst), 0, n, s));
}

void cuda_sync(int dev) {
    const int prior = current_device();
    if (dev >= 0) {
        BROTENSOR_CUDA_CHECK(set_device(dev));
        BROTENSOR_CUDA_CHECK(drv::cuCtxSynchronize());
    } else {
        const int count = device_count();
        for (int i = 0; i < count; ++i) {
            if (set_device(i) == CUDA_SUCCESS) {
                (void)drv::cuCtxSynchronize();
            }
        }
    }
    (void)set_device(prior);
}

bool cuda_mem_info(std::size_t* free_bytes, std::size_t* total_bytes, int dev) {
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    std::size_t f = 0, t = 0;
    if (drv::cuMemGetInfo(&f, &t) != CUDA_SUCCESS) {
        return false;
    }
    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return true;
}

bool cuda_device_name(char* out, std::size_t cap, int dev) {
    if (!out || cap == 0) return false;
    dev = resolve_device_index(dev);
    CUdevice cu_dev = 0;
    if (drv::cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return false;
    const int ncap = static_cast<int>(cap > 256 ? 256 : cap);
    if (drv::cuDeviceGetName(out, ncap, cu_dev) != CUDA_SUCCESS) return false;
    out[cap - 1] = '\0';
    return true;
}

bool cuda_mem_trim(std::size_t keep_bytes, int dev) {
    dev = resolve_device_index(dev);
    if (!async_pool_ready_for(dev)) return false;
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    CUmemoryPool pool = g_device_pools[dev].pool;
    if (!pool) {
        CUdevice cu_dev = 0;
        if (drv::cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return false;
        if (drv::cuDeviceGetDefaultMemPool(&pool, cu_dev) != CUDA_SUCCESS || !pool) return false;
    }
    if (drv::cuCtxSynchronize() != CUDA_SUCCESS) return false;
    return drv::cuMemPoolTrimTo(pool, keep_bytes) == CUDA_SUCCESS;
}

bool cuda_idle(int dev) {
    dev = resolve_device_index(dev);
    DeviceScopeGuard guard(dev);
    ensure_context(dev);
    auto s = reinterpret_cast<CUstream>(::brotensor::cuda_current_stream(dev));
    const CUresult e = drv::cuStreamQuery(s);
    if (e == CUDA_ERROR_NOT_READY) return false;
    BROTENSOR_CUDA_CHECK(e);
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
        &cuda_idle,
    };
    return t;
}

}  // namespace brotensor::detail::cuda
