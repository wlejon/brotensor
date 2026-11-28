// Pure-C++ CUDA backend runtime helpers backed by the dynamic Driver API.

#include "cuda_driver.h"
#include "detail/cuda_check.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace brotensor::detail::cuda {

namespace {

constexpr int kMaxCudaDevices = 16;

std::mutex g_ctx_mu;
CUcontext g_primary_ctxs[kMaxCudaDevices] = {};

thread_local int t_current_device = 0;
thread_local CUcontext t_bound_ctx = nullptr;
thread_local CUstream g_current_streams[kMaxCudaDevices] = {nullptr};

bool ensure_driver_init() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] { ok = (drv::cuInit(0) == CUDA_SUCCESS); });
    return ok;
}

int clamp_dev_index(int dev) {
    if (dev < 0 || dev >= kMaxCudaDevices) return 0;
    return dev;
}

int resolve_device_index(int dev) {
    if (dev < 0) return clamp_dev_index(current_device());
    return clamp_dev_index(dev);
}

}  // namespace

// ─── Error check throw ─────────────────────────────────────────────────────

void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    if (err == 0) return;
    const char* name = nullptr;
    const char* desc = nullptr;
    if (drv::cuGetErrorName(err, &name) != CUDA_SUCCESS || !name) name = "?";
    if (drv::cuGetErrorString(err, &desc) != CUDA_SUCCESS || !desc) desc = "?";
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "CUDA error %d (%s: %s) at %s:%d in `%s`",
                  err, name, desc, file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

// ─── Device & primary context management ───────────────────────────────────

int device_count() {
    if (!ensure_driver_init()) return 0;
    int n = 0;
    if (drv::cuDeviceGetCount(&n) != CUDA_SUCCESS || n < 0) return 0;
    return n;
}

CUcontext primary_context(int dev) {
    if (dev < 0 || dev >= kMaxCudaDevices) return nullptr;
    if (!ensure_driver_init()) return nullptr;
    std::lock_guard<std::mutex> lock(g_ctx_mu);
    if (g_primary_ctxs[dev]) return g_primary_ctxs[dev];
    CUdevice cu_dev = 0;
    if (drv::cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return nullptr;
    CUcontext ctx = nullptr;
    if (drv::cuDevicePrimaryCtxRetain(&ctx, cu_dev) != CUDA_SUCCESS) return nullptr;
    g_primary_ctxs[dev] = ctx;
    return ctx;
}

CUresult set_device(int dev) {
    if (dev < 0 || dev >= kMaxCudaDevices) return CUDA_ERROR_INVALID_VALUE;
    CUcontext ctx = primary_context(dev);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;
    const CUresult r = drv::cuCtxSetCurrent(ctx);
    if (r == CUDA_SUCCESS) {
        t_current_device = dev;
        t_bound_ctx = ctx;
    }
    return r;
}

void ensure_context(int dev) {
    if (dev >= 0) {
        (void)set_device(dev);
        return;
    }
    if (t_bound_ctx) return;
    if (!ensure_driver_init()) return;
    CUcontext cur = nullptr;
    if (drv::cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur != nullptr) {
        CUdevice cd = 0;
        if (drv::cuCtxGetDevice(&cd) == CUDA_SUCCESS && cd >= 0 && cd < kMaxCudaDevices) {
            t_current_device = cd;
        }
        t_bound_ctx = cur;
        return;
    }
    if (CUcontext ctx = primary_context(t_current_device)) {
        if (drv::cuCtxSetCurrent(ctx) == CUDA_SUCCESS) {
            t_bound_ctx = ctx;
        }
    }
}

int current_device() {
    ensure_context(-1);
    return t_current_device;
}

int device_attribute(int dev, CUdevice_attribute attr) {
    if (!ensure_driver_init()) return 0;
    if (dev < 0) dev = current_device();
    CUdevice cu_dev = 0;
    if (drv::cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return 0;
    int val = 0;
    if (drv::cuDeviceGetAttribute(&val, attr, cu_dev) != CUDA_SUCCESS) return 0;
    return val;
}

// ─── Async memory copy / memset / pointer query helpers ────────────────────

CUresult memcpy_d2d_async(void* dst, const void* src, std::size_t bytes, CUstream stream) {
    if (bytes == 0) return CUDA_SUCCESS;
    ensure_context(-1);
    return drv::cuMemcpyDtoDAsync(
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)),
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src)),
        bytes, stream);
}

CUresult memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, CUstream stream) {
    if (bytes == 0) return CUDA_SUCCESS;
    ensure_context(-1);
    return drv::cuMemcpyDtoHAsync(
        dst, static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src)), bytes, stream);
}

CUresult memcpy_2d_d2d_async(void* dst, std::size_t dst_pitch,
                             const void* src, std::size_t src_pitch,
                             std::size_t width_bytes, std::size_t height,
                             CUstream stream) {
    if (width_bytes == 0 || height == 0) return CUDA_SUCCESS;
    ensure_context(-1);
    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src));
    cp.srcPitch = src_pitch;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst));
    cp.dstPitch = dst_pitch;
    cp.WidthInBytes = width_bytes;
    cp.Height = height;
    return drv::cuMemcpy2DAsync(&cp, stream);
}

CUresult memset_zero_async(void* dst, std::size_t bytes, CUstream stream) {
    if (bytes == 0) return CUDA_SUCCESS;
    ensure_context(-1);
    return drv::cuMemsetD8Async(
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)), 0, bytes, stream);
}

bool is_device_or_managed_ptr(const void* ptr) {
    if (!ptr) return false;
    ensure_context(-1);
    const auto dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(ptr));
    CUmemorytype mem_type{};
    if (drv::cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr) == CUDA_SUCCESS) {
        if (mem_type == CU_MEMORYTYPE_DEVICE || mem_type == CU_MEMORYTYPE_UNIFIED) {
            return true;
        }
    }
    int managed = 0;
    if (drv::cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, dptr) == CUDA_SUCCESS) {
        return managed != 0;
    }
    return false;
}

// ─── Current stream (CUDA-internal) ────────────────────────────────────────

void* cuda_current_stream(int dev) {
    const int idx = resolve_device_index(dev);
    return reinterpret_cast<void*>(g_current_streams[idx]);
}

void* cuda_current_stream() {
    return cuda_current_stream(-1);
}

void cuda_set_stream(void* stream, int dev) {
    const int idx = resolve_device_index(dev);
    g_current_streams[idx] = reinterpret_cast<CUstream>(stream);
}

void cuda_set_stream(void* stream) {
    cuda_set_stream(stream, -1);
}

}  // namespace brotensor::detail::cuda

extern "C" void brotensor_cuda_set_device(int dev_index) {
    (void)::brotensor::detail::cuda::set_device(dev_index);
}

namespace brotensor {
void* cuda_current_stream(int dev) { return ::brotensor::detail::cuda::cuda_current_stream(dev); }
void* cuda_current_stream() { return ::brotensor::detail::cuda::cuda_current_stream(-1); }
}  // namespace brotensor
