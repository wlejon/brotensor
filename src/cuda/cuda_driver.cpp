#include "cuda_driver.h"

#include <mutex>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace brotensor::detail::cuda::drv {

namespace {

using FnCuInit                    = CUresult(BROTENSOR_CUDAAPI*)(unsigned int);
using FnCuDriverGetVersion        = CUresult(BROTENSOR_CUDAAPI*)(int*);
using FnCuGetErrorString          = CUresult(BROTENSOR_CUDAAPI*)(CUresult, const char**);
using FnCuGetErrorName            = CUresult(BROTENSOR_CUDAAPI*)(CUresult, const char**);

using FnCuDeviceGetCount          = CUresult(BROTENSOR_CUDAAPI*)(int*);
using FnCuDeviceGet               = CUresult(BROTENSOR_CUDAAPI*)(CUdevice*, int);
using FnCuDeviceGetName           = CUresult(BROTENSOR_CUDAAPI*)(char*, int, CUdevice);
using FnCuDeviceGetAttribute      = CUresult(BROTENSOR_CUDAAPI*)(int*, CUdevice_attribute, CUdevice);
using FnCuDeviceCanAccessPeer     = CUresult(BROTENSOR_CUDAAPI*)(int*, CUdevice, CUdevice);
using FnCuDevicePrimaryCtxRetain  = CUresult(BROTENSOR_CUDAAPI*)(CUcontext*, CUdevice);
using FnCuDevicePrimaryCtxRelease = CUresult(BROTENSOR_CUDAAPI*)(CUdevice);
using FnCuDeviceGetDefaultMemPool = CUresult(BROTENSOR_CUDAAPI*)(CUmemoryPool*, CUdevice);
using FnCuMemPoolSetAttribute     = CUresult(BROTENSOR_CUDAAPI*)(CUmemoryPool, CUmemPool_attribute, void*);
using FnCuMemPoolTrimTo           = CUresult(BROTENSOR_CUDAAPI*)(CUmemoryPool, std::size_t);
using FnCuCtxGetCurrent           = CUresult(BROTENSOR_CUDAAPI*)(CUcontext*);
using FnCuCtxSetCurrent           = CUresult(BROTENSOR_CUDAAPI*)(CUcontext);
using FnCuCtxSynchronize          = CUresult(BROTENSOR_CUDAAPI*)();
using FnCuCtxEnablePeerAccess     = CUresult(BROTENSOR_CUDAAPI*)(CUcontext, unsigned int);
using FnCuCtxGetDevice            = CUresult(BROTENSOR_CUDAAPI*)(CUdevice*);

using FnCuMemAlloc                = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr*, std::size_t);
using FnCuMemFree                 = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr);
using FnCuMemAllocAsync           = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr*, std::size_t, CUstream);
using FnCuMemFreeAsync            = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, CUstream);
using FnCuMemGetInfo              = CUresult(BROTENSOR_CUDAAPI*)(std::size_t*, std::size_t*);
using FnCuMemcpyHtoD              = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, const void*, std::size_t);
using FnCuMemcpyDtoH              = CUresult(BROTENSOR_CUDAAPI*)(void*, CUdeviceptr, std::size_t);
using FnCuMemcpyDtoDAsync         = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, CUdeviceptr, std::size_t, CUstream);
using FnCuMemcpyHtoDAsync         = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, const void*, std::size_t, CUstream);
using FnCuMemcpyDtoHAsync         = CUresult(BROTENSOR_CUDAAPI*)(void*, CUdeviceptr, std::size_t, CUstream);
using FnCuMemcpyPeerAsync         = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext,
                                                                 std::size_t, CUstream);
using FnCuMemcpy2DAsync           = CUresult(BROTENSOR_CUDAAPI*)(const CUDA_MEMCPY2D*, CUstream);
using FnCuMemsetD8Async           = CUresult(BROTENSOR_CUDAAPI*)(CUdeviceptr, unsigned char, std::size_t, CUstream);
using FnCuPointerGetAttribute     = CUresult(BROTENSOR_CUDAAPI*)(void*, CUpointer_attribute, CUdeviceptr);

using FnCuStreamCreate            = CUresult(BROTENSOR_CUDAAPI*)(CUstream*, unsigned int);
using FnCuStreamDestroy           = CUresult(BROTENSOR_CUDAAPI*)(CUstream);
using FnCuStreamSynchronize       = CUresult(BROTENSOR_CUDAAPI*)(CUstream);
using FnCuStreamQuery             = CUresult(BROTENSOR_CUDAAPI*)(CUstream);
using FnCuStreamWaitEvent         = CUresult(BROTENSOR_CUDAAPI*)(CUstream, CUevent, unsigned int);
using FnCuStreamIsCapturing       = CUresult(BROTENSOR_CUDAAPI*)(CUstream, CUstreamCaptureStatus*);
using FnCuStreamBeginCapture      = CUresult(BROTENSOR_CUDAAPI*)(CUstream, CUstreamCaptureMode);
using FnCuStreamEndCapture        = CUresult(BROTENSOR_CUDAAPI*)(CUstream, CUgraph*);
using FnCuEventCreate             = CUresult(BROTENSOR_CUDAAPI*)(CUevent*, unsigned int);
using FnCuEventDestroy            = CUresult(BROTENSOR_CUDAAPI*)(CUevent);
using FnCuEventRecord             = CUresult(BROTENSOR_CUDAAPI*)(CUevent, CUstream);
using FnCuEventSynchronize        = CUresult(BROTENSOR_CUDAAPI*)(CUevent);
using FnCuEventElapsedTime        = CUresult(BROTENSOR_CUDAAPI*)(float*, CUevent, CUevent);
using FnCuGraphInstantiate        = CUresult(BROTENSOR_CUDAAPI*)(CUgraphExec*, CUgraph, unsigned long long);
using FnCuGraphLaunch             = CUresult(BROTENSOR_CUDAAPI*)(CUgraphExec, CUstream);
using FnCuGraphDestroy            = CUresult(BROTENSOR_CUDAAPI*)(CUgraph);
using FnCuGraphExecDestroy        = CUresult(BROTENSOR_CUDAAPI*)(CUgraphExec);

using FnCuModuleLoadData          = CUresult(BROTENSOR_CUDAAPI*)(CUmodule*, const void*);
using FnCuModuleLoadDataEx        = CUresult(BROTENSOR_CUDAAPI*)(CUmodule*, const void*, unsigned int,
                                                                 CUjit_option*, void**);
using FnCuModuleGetFunction       = CUresult(BROTENSOR_CUDAAPI*)(CUfunction*, CUmodule, const char*);
using FnCuModuleUnload            = CUresult(BROTENSOR_CUDAAPI*)(CUmodule);
using FnCuFuncSetAttribute        = CUresult(BROTENSOR_CUDAAPI*)(CUfunction, CUfunction_attribute, int);
using FnCuLinkCreate              = CUresult(BROTENSOR_CUDAAPI*)(unsigned int, CUjit_option*, void**, CUlinkState*);
using FnCuLinkAddData             = CUresult(BROTENSOR_CUDAAPI*)(CUlinkState, CUjitInputType, void*, std::size_t,
                                                                 const char*, unsigned int, CUjit_option*, void**);
using FnCuLinkComplete            = CUresult(BROTENSOR_CUDAAPI*)(CUlinkState, void**, std::size_t*);
using FnCuLinkDestroy             = CUresult(BROTENSOR_CUDAAPI*)(CUlinkState);
using FnCuLaunchKernel            = CUresult(BROTENSOR_CUDAAPI*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                                                 unsigned int, unsigned int, unsigned int,
                                                                 unsigned int, CUstream, void**, void**);

struct Table {
    FnCuInit                    cuInit                    = nullptr;
    FnCuDriverGetVersion        cuDriverGetVersion        = nullptr;
    FnCuGetErrorString          cuGetErrorString          = nullptr;
    FnCuGetErrorName            cuGetErrorName            = nullptr;

    FnCuDeviceGetCount          cuDeviceGetCount          = nullptr;
    FnCuDeviceGet               cuDeviceGet               = nullptr;
    FnCuDeviceGetName           cuDeviceGetName           = nullptr;
    FnCuDeviceGetAttribute      cuDeviceGetAttribute      = nullptr;
    FnCuDeviceCanAccessPeer     cuDeviceCanAccessPeer     = nullptr;
    FnCuDevicePrimaryCtxRetain  cuDevicePrimaryCtxRetain  = nullptr;
    FnCuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease = nullptr;
    FnCuDeviceGetDefaultMemPool cuDeviceGetDefaultMemPool = nullptr;
    FnCuMemPoolSetAttribute     cuMemPoolSetAttribute     = nullptr;
    FnCuMemPoolTrimTo           cuMemPoolTrimTo           = nullptr;
    FnCuCtxGetCurrent           cuCtxGetCurrent           = nullptr;
    FnCuCtxSetCurrent           cuCtxSetCurrent           = nullptr;
    FnCuCtxSynchronize          cuCtxSynchronize          = nullptr;
    FnCuCtxEnablePeerAccess     cuCtxEnablePeerAccess     = nullptr;
    FnCuCtxGetDevice            cuCtxGetDevice            = nullptr;

    FnCuMemAlloc                cuMemAlloc                = nullptr;
    FnCuMemFree                 cuMemFree                 = nullptr;
    FnCuMemAllocAsync           cuMemAllocAsync           = nullptr;
    FnCuMemFreeAsync            cuMemFreeAsync            = nullptr;
    FnCuMemGetInfo              cuMemGetInfo              = nullptr;
    FnCuMemcpyHtoD              cuMemcpyHtoD              = nullptr;
    FnCuMemcpyDtoH              cuMemcpyDtoH              = nullptr;
    FnCuMemcpyDtoDAsync         cuMemcpyDtoDAsync         = nullptr;
    FnCuMemcpyHtoDAsync         cuMemcpyHtoDAsync         = nullptr;
    FnCuMemcpyDtoHAsync         cuMemcpyDtoHAsync         = nullptr;
    FnCuMemcpyPeerAsync         cuMemcpyPeerAsync         = nullptr;
    FnCuMemcpy2DAsync           cuMemcpy2DAsync           = nullptr;
    FnCuMemsetD8Async           cuMemsetD8Async           = nullptr;
    FnCuPointerGetAttribute     cuPointerGetAttribute     = nullptr;

    FnCuStreamCreate            cuStreamCreate            = nullptr;
    FnCuStreamDestroy           cuStreamDestroy           = nullptr;
    FnCuStreamSynchronize       cuStreamSynchronize       = nullptr;
    FnCuStreamQuery             cuStreamQuery             = nullptr;
    FnCuStreamWaitEvent         cuStreamWaitEvent         = nullptr;
    FnCuStreamIsCapturing       cuStreamIsCapturing       = nullptr;
    FnCuStreamBeginCapture      cuStreamBeginCapture      = nullptr;
    FnCuStreamEndCapture        cuStreamEndCapture        = nullptr;
    FnCuEventCreate             cuEventCreate             = nullptr;
    FnCuEventDestroy            cuEventDestroy            = nullptr;
    FnCuEventRecord             cuEventRecord             = nullptr;
    FnCuEventSynchronize        cuEventSynchronize        = nullptr;
    FnCuEventElapsedTime        cuEventElapsedTime        = nullptr;
    FnCuGraphInstantiate        cuGraphInstantiate        = nullptr;
    FnCuGraphLaunch             cuGraphLaunch             = nullptr;
    FnCuGraphDestroy            cuGraphDestroy            = nullptr;
    FnCuGraphExecDestroy        cuGraphExecDestroy        = nullptr;

    FnCuModuleLoadData          cuModuleLoadData          = nullptr;
    FnCuModuleLoadDataEx        cuModuleLoadDataEx        = nullptr;
    FnCuModuleGetFunction       cuModuleGetFunction       = nullptr;
    FnCuModuleUnload            cuModuleUnload            = nullptr;
    FnCuFuncSetAttribute        cuFuncSetAttribute        = nullptr;
    FnCuLinkCreate              cuLinkCreate              = nullptr;
    FnCuLinkAddData             cuLinkAddData             = nullptr;
    FnCuLinkComplete            cuLinkComplete            = nullptr;
    FnCuLinkDestroy             cuLinkDestroy             = nullptr;
    FnCuLaunchKernel            cuLaunchKernel            = nullptr;
};

void* open_driver_library() {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#else
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    return lib;
#endif
}

template <typename Fn>
void resolve(void* lib, const char* symbol, Fn& out) {
    if (!lib) return;
#if defined(_WIN32)
    FARPROC proc = GetProcAddress(reinterpret_cast<HMODULE>(lib), symbol);
    out = reinterpret_cast<Fn>(proc);
#else
    void* proc = dlsym(lib, symbol);
    out = reinterpret_cast<Fn>(proc);
#endif
}

const Table& table() {
    static Table t;
    static std::once_flag once;
    std::call_once(once, [] {
        // Intentionally kept open for process lifetime — unloading nvcuda.dll
        // during static teardown races with driver worker threads.
        void* lib = open_driver_library();
        if (!lib) return;

        resolve(lib, "cuInit", t.cuInit);
        resolve(lib, "cuDriverGetVersion", t.cuDriverGetVersion);
        resolve(lib, "cuGetErrorString", t.cuGetErrorString);
        resolve(lib, "cuGetErrorName", t.cuGetErrorName);

        resolve(lib, "cuDeviceGetCount", t.cuDeviceGetCount);
        resolve(lib, "cuDeviceGet", t.cuDeviceGet);
        resolve(lib, "cuDeviceGetName", t.cuDeviceGetName);
        resolve(lib, "cuDeviceGetAttribute", t.cuDeviceGetAttribute);
        resolve(lib, "cuDeviceCanAccessPeer", t.cuDeviceCanAccessPeer);
        resolve(lib, "cuDevicePrimaryCtxRetain", t.cuDevicePrimaryCtxRetain);
        resolve(lib, "cuDevicePrimaryCtxRelease_v2", t.cuDevicePrimaryCtxRelease);
        resolve(lib, "cuDeviceGetDefaultMemPool", t.cuDeviceGetDefaultMemPool);
        resolve(lib, "cuMemPoolSetAttribute", t.cuMemPoolSetAttribute);
        resolve(lib, "cuMemPoolTrimTo", t.cuMemPoolTrimTo);
        resolve(lib, "cuCtxGetCurrent", t.cuCtxGetCurrent);
        resolve(lib, "cuCtxSetCurrent", t.cuCtxSetCurrent);
        resolve(lib, "cuCtxSynchronize", t.cuCtxSynchronize);
        resolve(lib, "cuCtxEnablePeerAccess", t.cuCtxEnablePeerAccess);
        resolve(lib, "cuCtxGetDevice", t.cuCtxGetDevice);

        resolve(lib, "cuMemAlloc_v2", t.cuMemAlloc);
        resolve(lib, "cuMemFree_v2", t.cuMemFree);
        resolve(lib, "cuMemAllocAsync", t.cuMemAllocAsync);
        resolve(lib, "cuMemFreeAsync", t.cuMemFreeAsync);
        resolve(lib, "cuMemGetInfo_v2", t.cuMemGetInfo);
        resolve(lib, "cuMemcpyHtoD_v2", t.cuMemcpyHtoD);
        resolve(lib, "cuMemcpyDtoH_v2", t.cuMemcpyDtoH);
        resolve(lib, "cuMemcpyDtoDAsync_v2", t.cuMemcpyDtoDAsync);
        resolve(lib, "cuMemcpyHtoDAsync_v2", t.cuMemcpyHtoDAsync);
        resolve(lib, "cuMemcpyDtoHAsync_v2", t.cuMemcpyDtoHAsync);
        resolve(lib, "cuMemcpyPeerAsync", t.cuMemcpyPeerAsync);
        resolve(lib, "cuMemcpy2DAsync_v2", t.cuMemcpy2DAsync);
        resolve(lib, "cuMemsetD8Async", t.cuMemsetD8Async);
        resolve(lib, "cuPointerGetAttribute", t.cuPointerGetAttribute);

        resolve(lib, "cuStreamCreate", t.cuStreamCreate);
        resolve(lib, "cuStreamDestroy_v2", t.cuStreamDestroy);
        resolve(lib, "cuStreamSynchronize", t.cuStreamSynchronize);
        resolve(lib, "cuStreamQuery", t.cuStreamQuery);
        resolve(lib, "cuStreamWaitEvent", t.cuStreamWaitEvent);
        resolve(lib, "cuStreamIsCapturing", t.cuStreamIsCapturing);
        resolve(lib, "cuStreamBeginCapture_v2", t.cuStreamBeginCapture);
        resolve(lib, "cuStreamEndCapture", t.cuStreamEndCapture);
        resolve(lib, "cuEventCreate", t.cuEventCreate);
        resolve(lib, "cuEventDestroy_v2", t.cuEventDestroy);
        resolve(lib, "cuEventRecord", t.cuEventRecord);
        resolve(lib, "cuEventSynchronize", t.cuEventSynchronize);
        resolve(lib, "cuEventElapsedTime", t.cuEventElapsedTime);
        // Always resolve the 3-arg flags ABI (unversioned cuGraphInstantiate is the CUDA 10 5-arg ABI).
        resolve(lib, "cuGraphInstantiateWithFlags", t.cuGraphInstantiate);
        resolve(lib, "cuGraphLaunch", t.cuGraphLaunch);
        resolve(lib, "cuGraphDestroy", t.cuGraphDestroy);
        resolve(lib, "cuGraphExecDestroy", t.cuGraphExecDestroy);

        resolve(lib, "cuModuleLoadData", t.cuModuleLoadData);
        resolve(lib, "cuModuleLoadDataEx", t.cuModuleLoadDataEx);
        resolve(lib, "cuModuleGetFunction", t.cuModuleGetFunction);
        resolve(lib, "cuModuleUnload", t.cuModuleUnload);
        resolve(lib, "cuFuncSetAttribute", t.cuFuncSetAttribute);
        resolve(lib, "cuLinkCreate_v2", t.cuLinkCreate);
        resolve(lib, "cuLinkAddData_v2", t.cuLinkAddData);
        resolve(lib, "cuLinkComplete", t.cuLinkComplete);
        resolve(lib, "cuLinkDestroy", t.cuLinkDestroy);
        resolve(lib, "cuLaunchKernel", t.cuLaunchKernel);
    });
    return t;
}

template <typename Fn, typename... Args>
CUresult call_drv(Fn fn, Args... args) {
    return fn ? fn(args...) : CUDA_ERROR_NOT_INITIALIZED;
}

}  // namespace

CUresult cuInit(unsigned int flags) { return call_drv(table().cuInit, flags); }
CUresult cuDriverGetVersion(int* driverVersion) { return call_drv(table().cuDriverGetVersion, driverVersion); }
CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return call_drv(table().cuGetErrorString, error, pStr);
}
CUresult cuGetErrorName(CUresult error, const char** pStr) {
    return call_drv(table().cuGetErrorName, error, pStr);
}

CUresult cuDeviceGetCount(int* count) { return call_drv(table().cuDeviceGetCount, count); }
CUresult cuDeviceGet(CUdevice* device, int ordinal) { return call_drv(table().cuDeviceGet, device, ordinal); }
CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    return call_drv(table().cuDeviceGetName, name, len, dev);
}
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    return call_drv(table().cuDeviceGetAttribute, pi, attrib, dev);
}
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) {
    return call_drv(table().cuDeviceCanAccessPeer, canAccessPeer, dev, peerDev);
}
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    return call_drv(table().cuDevicePrimaryCtxRetain, pctx, dev);
}
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return call_drv(table().cuDevicePrimaryCtxRelease, dev);
}
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) {
    return call_drv(table().cuDeviceGetDefaultMemPool, pool_out, dev);
}
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    return call_drv(table().cuMemPoolSetAttribute, pool, attr, value);
}
CUresult cuMemPoolTrimTo(CUmemoryPool pool, std::size_t minBytesToKeep) {
    return call_drv(table().cuMemPoolTrimTo, pool, minBytesToKeep);
}
CUresult cuCtxGetCurrent(CUcontext* pctx) { return call_drv(table().cuCtxGetCurrent, pctx); }
CUresult cuCtxSetCurrent(CUcontext ctx) { return call_drv(table().cuCtxSetCurrent, ctx); }
CUresult cuCtxSynchronize() { return call_drv(table().cuCtxSynchronize); }
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    return call_drv(table().cuCtxEnablePeerAccess, peerContext, Flags);
}
CUresult cuCtxGetDevice(CUdevice* device) { return call_drv(table().cuCtxGetDevice, device); }

CUresult cuMemAlloc(CUdeviceptr* dptr, std::size_t bytesize) {
    return call_drv(table().cuMemAlloc, dptr, bytesize);
}
CUresult cuMemFree(CUdeviceptr dptr) { return call_drv(table().cuMemFree, dptr); }
CUresult cuMemAllocAsync(CUdeviceptr* dptr, std::size_t bytesize, CUstream hStream) {
    return call_drv(table().cuMemAllocAsync, dptr, bytesize, hStream);
}
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return call_drv(table().cuMemFreeAsync, dptr, hStream);
}
CUresult cuMemGetInfo(std::size_t* free, std::size_t* total) {
    return call_drv(table().cuMemGetInfo, free, total);
}
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, std::size_t ByteCount) {
    return call_drv(table().cuMemcpyHtoD, dstDevice, srcHost, ByteCount);
}
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, std::size_t ByteCount) {
    return call_drv(table().cuMemcpyDtoH, dstHost, srcDevice, ByteCount);
}
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           std::size_t ByteCount, CUstream hStream) {
    return call_drv(table().cuMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount, hStream);
}
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                           std::size_t ByteCount, CUstream hStream) {
    return call_drv(table().cuMemcpyHtoDAsync, dstDevice, srcHost, ByteCount, hStream);
}
CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                           std::size_t ByteCount, CUstream hStream) {
    return call_drv(table().cuMemcpyDtoHAsync, dstHost, srcDevice, ByteCount, hStream);
}
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           std::size_t ByteCount, CUstream hStream) {
    return call_drv(table().cuMemcpyPeerAsync, dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}
CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    return call_drv(table().cuMemcpy2DAsync, pCopy, hStream);
}
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, std::size_t N, CUstream hStream) {
    return call_drv(table().cuMemsetD8Async, dstDevice, uc, N, hStream);
}
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    return call_drv(table().cuPointerGetAttribute, data, attribute, ptr);
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    return call_drv(table().cuStreamCreate, phStream, Flags);
}
CUresult cuStreamDestroy(CUstream hStream) { return call_drv(table().cuStreamDestroy, hStream); }
CUresult cuStreamSynchronize(CUstream hStream) { return call_drv(table().cuStreamSynchronize, hStream); }
CUresult cuStreamQuery(CUstream hStream) { return call_drv(table().cuStreamQuery, hStream); }
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    return call_drv(table().cuStreamWaitEvent, hStream, hEvent, Flags);
}
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
    return call_drv(table().cuStreamIsCapturing, hStream, captureStatus);
}
CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    return call_drv(table().cuStreamBeginCapture, hStream, mode);
}
CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) {
    return call_drv(table().cuStreamEndCapture, hStream, phGraph);
}
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    return call_drv(table().cuEventCreate, phEvent, Flags);
}
CUresult cuEventDestroy(CUevent hEvent) { return call_drv(table().cuEventDestroy, hEvent); }
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return call_drv(table().cuEventRecord, hEvent, hStream);
}
CUresult cuEventSynchronize(CUevent hEvent) { return call_drv(table().cuEventSynchronize, hEvent); }
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    return call_drv(table().cuEventElapsedTime, pMilliseconds, hStart, hEnd);
}
CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) {
    return call_drv(table().cuGraphInstantiate, phGraphExec, hGraph, flags);
}
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    return call_drv(table().cuGraphLaunch, hGraphExec, hStream);
}
CUresult cuGraphDestroy(CUgraph hGraph) { return call_drv(table().cuGraphDestroy, hGraph); }
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    return call_drv(table().cuGraphExecDestroy, hGraphExec);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    return call_drv(table().cuModuleLoadData, module, image);
}
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                            CUjit_option* options, void** optionValues) {
    return call_drv(table().cuModuleLoadDataEx, module, image, numOptions, options, optionValues);
}
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    return call_drv(table().cuModuleGetFunction, hfunc, hmod, name);
}
CUresult cuModuleUnload(CUmodule hmod) { return call_drv(table().cuModuleUnload, hmod); }
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    return call_drv(table().cuFuncSetAttribute, hfunc, attrib, value);
}
CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                      void** optionValues, CUlinkState* stateOut) {
    return call_drv(table().cuLinkCreate, numOptions, options, optionValues, stateOut);
}
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, std::size_t size,
                       const char* name, unsigned int numOptions, CUjit_option* options,
                       void** optionValues) {
    return call_drv(table().cuLinkAddData, state, type, data, size, name, numOptions, options, optionValues);
}
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, std::size_t* sizeOut) {
    return call_drv(table().cuLinkComplete, state, cubinOut, sizeOut);
}
CUresult cuLinkDestroy(CUlinkState state) { return call_drv(table().cuLinkDestroy, state); }
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    return call_drv(table().cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

}  // namespace brotensor::detail::cuda::drv
