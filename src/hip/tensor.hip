// HIP AllocVTable.
//
// All Tensor allocation / freeing / memcpy / zero / sync on HIP devices
// is routed through `hip_alloc_table()` (see brotensor/detail/dispatch.h).

#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace brotensor {
// Forward decl: thread-local current stream from runtime.hip.
void* hip_current_stream(int dev = -1);
}

namespace brotensor::detail::hip {

namespace {

constexpr int kMaxHipDevices = 16;

struct DevicePool {
    bool initialized = false;
    bool supported = false;
    hipMemPool_t pool = nullptr;
};

DevicePool g_device_pools[kMaxHipDevices];

int resolve_device_index(int dev) {
    if (dev < 0) {
        int d = 0;
        if (hipGetDevice(&d) == hipSuccess && d >= 0 && d < kMaxHipDevices) {
            return d;
        }
        return 0;
    }
    if (dev >= kMaxHipDevices) return 0;
    return dev;
}

}  // namespace

void init_async_pool_for_device(int dev) {
    dev = resolve_device_index(dev);
    if (g_device_pools[dev].initialized) return;
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    int supported = 0;
    if (hipDeviceGetAttribute(&supported, hipDeviceAttributeMemoryPoolsSupported, dev) == hipSuccess && supported) {
        hipMemPool_t pool = nullptr;
        if (hipDeviceGetDefaultMemPool(&pool, dev) == hipSuccess) {
            std::uint64_t threshold = UINT64_MAX;
            (void)hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold);
            g_device_pools[dev].supported = true;
            g_device_pools[dev].pool = pool;
        }
    }
    g_device_pools[dev].initialized = true;
    if (prior != dev) (void)hipSetDevice(prior);
}

bool async_pool_ready_for(int dev) {
    dev = resolve_device_index(dev);
    if (!g_device_pools[dev].initialized) {
        init_async_pool_for_device(dev);
    }
    return g_device_pools[dev].supported;
}

void* hip_alloc(std::size_t bytes, int dev) {
    if (bytes == 0) return nullptr;
    dev = resolve_device_index(dev);
    void* p = nullptr;
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(dev));
        BROTENSOR_HIP_CHECK(hipMallocAsync(&p, bytes, s));
    } else {
        BROTENSOR_HIP_CHECK(hipMalloc(&p, bytes));
    }
    if (prior != dev) (void)hipSetDevice(prior);
    return p;
}

void* hip_alloc(std::size_t bytes) {
    return hip_alloc(bytes, -1);
}

void hip_free(void* ptr, int dev) {
    if (!ptr) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    if (async_pool_ready_for(dev)) {
        auto s = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(dev));
        hipError_t err = hipFreeAsync(ptr, s);
        if (err != hipSuccess) {
            hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
            (void)hipStreamIsCapturing(s, &cs);
            if (cs != hipStreamCaptureStatusNone) {
                std::fprintf(stderr,
                    "brotensor: WARNING: hipFreeAsync(%p) failed (err=%d) on dev %d during stream capture\n",
                    ptr, static_cast<int>(err), dev);
            }
            (void)hipGetLastError();
        }
    } else {
        (void)hipFree(ptr);
    }
    if (prior != dev) (void)hipSetDevice(prior);
}

void hip_free(void* ptr) {
    hip_free(ptr, -1);
}

void hip_memcpy_h2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    BROTENSOR_HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
    if (prior != dev) (void)hipSetDevice(prior);
}

void hip_memcpy_d2h(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    BROTENSOR_HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyDeviceToHost));
    if (prior != dev) (void)hipSetDevice(prior);
}

void hip_memcpy_d2d(void* dst, const void* src, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    auto s = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(dev));
    BROTENSOR_HIP_CHECK(hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToDevice, s));
    if (prior != dev) (void)hipSetDevice(prior);
}

void hip_memcpy_peer(void* dst, int dst_dev, const void* src, int src_dev, std::size_t n) {
    if (n == 0) return;
    dst_dev = resolve_device_index(dst_dev);
    src_dev = resolve_device_index(src_dev);
    if (dst_dev == src_dev) {
        hip_memcpy_d2d(dst, src, n, dst_dev);
        return;
    }
    int prior = 0;
    (void)hipGetDevice(&prior);
    (void)hipSetDevice(dst_dev);
    auto dst_stream = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(dst_dev));
    auto src_stream = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(src_dev));

    hipEvent_t ev = nullptr;
    BROTENSOR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    BROTENSOR_HIP_CHECK(hipEventRecord(ev, src_stream));
    BROTENSOR_HIP_CHECK(hipStreamWaitEvent(dst_stream, ev, 0));
    BROTENSOR_HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, n, dst_stream));
    BROTENSOR_HIP_CHECK(hipEventDestroy(ev));
    if (prior != dst_dev) (void)hipSetDevice(prior);
}

void hip_memset_zero(void* dst, std::size_t n, int dev) {
    if (n == 0) return;
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    auto s = reinterpret_cast<hipStream_t>(::brotensor::hip_current_stream(dev));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dst, 0, n, s));
    if (prior != dev) (void)hipSetDevice(prior);
}

void hip_sync(int dev) {
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (dev >= 0) {
        (void)hipSetDevice(dev);
        BROTENSOR_HIP_CHECK(hipDeviceSynchronize());
    } else {
        int count = 0;
        if (hipGetDeviceCount(&count) == hipSuccess) {
            for (int i = 0; i < count; ++i) {
                (void)hipSetDevice(i);
                (void)hipDeviceSynchronize();
            }
        }
    }
    (void)hipSetDevice(prior);
}

bool hip_mem_info(std::size_t* free_bytes, std::size_t* total_bytes, int dev) {
    dev = resolve_device_index(dev);
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    std::size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) {
        (void)hipGetLastError();
        if (prior != dev) (void)hipSetDevice(prior);
        return false;
    }
    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    if (prior != dev) (void)hipSetDevice(prior);
    return true;
}

bool hip_device_name(char* out, std::size_t cap, int dev) {
    if (!out || cap == 0) return false;
    dev = resolve_device_index(dev);
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    std::snprintf(out, cap, "%s", prop.name);
    return true;
}

bool hip_mem_trim(std::size_t keep_bytes, int dev) {
    dev = resolve_device_index(dev);
    if (!async_pool_ready_for(dev)) return false;
    int prior = 0;
    (void)hipGetDevice(&prior);
    if (prior != dev) (void)hipSetDevice(dev);
    hipMemPool_t pool = g_device_pools[dev].pool;
    if (!pool && hipDeviceGetDefaultMemPool(&pool, dev) != hipSuccess) {
        if (prior != dev) (void)hipSetDevice(prior);
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        if (prior != dev) (void)hipSetDevice(prior);
        return false;
    }
    if (hipMemPoolTrimTo(pool, keep_bytes) != hipSuccess) {
        (void)hipGetLastError();
        if (prior != dev) (void)hipSetDevice(prior);
        return false;
    }
    if (prior != dev) (void)hipSetDevice(prior);
    return true;
}

const ::brotensor::detail::AllocVTable& hip_alloc_table() {
    static const ::brotensor::detail::AllocVTable t = {
        &hip_alloc,
        &hip_free,
        &hip_memcpy_h2d,
        &hip_memcpy_d2h,
        &hip_memcpy_d2d,
        &hip_memcpy_peer,
        &hip_memset_zero,
        &hip_sync,
        &hip_mem_info,
        &hip_mem_trim,
        &hip_device_name,
    };
    return t;
}

} // namespace brotensor::detail::hip
