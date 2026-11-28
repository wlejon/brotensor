#pragma once

// Self-contained CUDA Driver API declarations and dynamic loader interface.
//
// Zero CUDA Toolkit headers (<cuda.h>, <cuda_runtime.h>) are included
// anywhere in brotensor. On CUDA-enabled builds the NVIDIA driver library
// (nvcuda.dll / libcuda.so.1) is loaded dynamically on first use; on hosts
// without an NVIDIA driver every wrapper cleanly returns
// CUDA_ERROR_NOT_INITIALIZED and brotensor::init() falls back to CPU/Vulkan.

#include <cstddef>
#include <cstdint>

#if defined(_WIN32) && !defined(_WIN64)
#define BROTENSOR_CUDAAPI __stdcall
#else
#define BROTENSOR_CUDAAPI
#endif

// ─── Minimal CUDA Driver API ABI types & constants ─────────────────────────

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = std::uint64_t;

struct CUctx_st;       using CUcontext    = CUctx_st*;
struct CUmod_st;       using CUmodule     = CUmod_st*;
struct CUfunc_st;      using CUfunction   = CUfunc_st*;
struct CUstream_st;    using CUstream     = CUstream_st*;
struct CUevent_st;     using CUevent      = CUevent_st*;
struct CUarray_st;     using CUarray      = CUarray_st*;
struct CUlinkState_st; using CUlinkState  = CUlinkState_st*;
struct CUgraph_st;     using CUgraph      = CUgraph_st*;
struct CUgraphExec_st; using CUgraphExec  = CUgraphExec_st*;
struct CUmemPool_st;   using CUmemoryPool = CUmemPool_st*;

enum : CUresult {
    CUDA_SUCCESS                              = 0,
    CUDA_ERROR_INVALID_VALUE                  = 1,
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,
    CUDA_ERROR_NOT_INITIALIZED                = 3,
    CUDA_ERROR_NOT_READY                      = 600,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
};

enum CUdevice_attribute : int {
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT     = 16,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED   = 115,
};

enum CUfunction_attribute : int {
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

enum CUjit_option : int {
    CU_JIT_INFO_LOG_BUFFER             = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES  = 4,
    CU_JIT_ERROR_LOG_BUFFER            = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL          = 7,
};

enum CUjitInputType : int {
    CU_JIT_INPUT_PTX = 1,
};

enum : unsigned int {
    CU_STREAM_DEFAULT      = 0x0u,
    CU_STREAM_NON_BLOCKING = 0x1u,
};

enum : unsigned int {
    CU_EVENT_DEFAULT        = 0x0u,
    CU_EVENT_DISABLE_TIMING = 0x2u,
};

enum CUstreamCaptureMode : int {
    CU_STREAM_CAPTURE_MODE_GLOBAL       = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED      = 2,
};

enum CUstreamCaptureStatus : int {
    CU_STREAM_CAPTURE_STATUS_NONE        = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE      = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
};

enum CUmemPool_attribute : int {
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
};

enum CUpointer_attribute : int {
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_IS_MANAGED  = 8,
};

enum CUmemorytype : int {
    CU_MEMORYTYPE_HOST    = 1,
    CU_MEMORYTYPE_DEVICE  = 2,
    CU_MEMORYTYPE_ARRAY   = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
};

struct CUDA_MEMCPY2D {
    std::size_t  srcXInBytes;
    std::size_t  srcY;
    CUmemorytype srcMemoryType;
    const void*  srcHost;
    CUdeviceptr  srcDevice;
    CUarray      srcArray;
    std::size_t  srcPitch;

    std::size_t  dstXInBytes;
    std::size_t  dstY;
    CUmemorytype dstMemoryType;
    void*        dstHost;
    CUdeviceptr  dstDevice;
    CUarray      dstArray;
    std::size_t  dstPitch;

    std::size_t  WidthInBytes;
    std::size_t  Height;
};

namespace brotensor::detail::cuda {

// Host-side 2-byte tag types replacing <cuda_fp16.h> __half and <cuda_bf16.h>
// __nv_bfloat16 purely for typed GPU buffer pointer signatures.
struct alignas(2) Fp16 { std::uint16_t bits; };
struct alignas(2) Bf16 { std::uint16_t bits; };
static_assert(sizeof(Fp16) == 2 && alignof(Fp16) == 2);
static_assert(sizeof(Bf16) == 2 && alignof(Bf16) == 2);

// Runtime context / device / copy helpers implemented in src/cuda/runtime.cpp.
int current_device();
CUresult set_device(int dev);
CUcontext primary_context(int dev);
void ensure_context(int dev = -1);
int device_count();
int device_attribute(int dev, CUdevice_attribute attr);

inline bool supports_sm80(int dev = -1) {
    static int cached_major[16] = {};
    if (dev < 0) dev = current_device();
    if (dev < 0 || dev >= 16) dev = 0;
    if (cached_major[dev] == 0) {
        cached_major[dev] = device_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    }
    return cached_major[dev] >= 8;
}

CUresult memcpy_d2d_async(void* dst, const void* src, std::size_t bytes, CUstream stream);
CUresult memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, CUstream stream);
CUresult memcpy_2d_d2d_async(void* dst, std::size_t dst_pitch,
                             const void* src, std::size_t src_pitch,
                             std::size_t width_bytes, std::size_t height,
                             CUstream stream);
CUresult memset_zero_async(void* dst, std::size_t bytes, CUstream stream);
bool is_device_or_managed_ptr(const void* ptr);

namespace drv {

// ─── Initialization & error reporting ──────────────────────────────────────
CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuGetErrorName(CUresult error, const char** pStr);

// ─── Device, primary context & memory pool management ──────────────────────
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev);
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
CUresult cuMemPoolTrimTo(CUmemoryPool pool, std::size_t minBytesToKeep);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxSynchronize();
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
CUresult cuCtxGetDevice(CUdevice* device);

// ─── Memory allocation, copies & pointer queries ───────────────────────────
CUresult cuMemAlloc(CUdeviceptr* dptr, std::size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemAllocAsync(CUdeviceptr* dptr, std::size_t bytesize, CUstream hStream);
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult cuMemGetInfo(std::size_t* free, std::size_t* total);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, std::size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, std::size_t ByteCount);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           std::size_t ByteCount, CUstream hStream);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                           std::size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                           std::size_t ByteCount, CUstream hStream);
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           std::size_t ByteCount, CUstream hStream);
CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                         std::size_t N, CUstream hStream);
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);

// ─── Streams, events & graphs ──────────────────────────────────────────────
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus);
CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph,
                            unsigned long long flags);
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult cuGraphDestroy(CUgraph hGraph);
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);

// ─── Modules, linker & kernel launch ───────────────────────────────────────
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int numOptions, CUjit_option* options,
                            void** optionValues);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                      void** optionValues, CUlinkState* stateOut);
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
                       void* data, std::size_t size, const char* name,
                       unsigned int numOptions, CUjit_option* options,
                       void** optionValues);
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, std::size_t* sizeOut);
CUresult cuLinkDestroy(CUlinkState state);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);

}  // namespace drv
}  // namespace brotensor::detail::cuda
