#include "cuda_driver.h"

#include <cuda_runtime.h>

#include <mutex>

namespace brotensor::detail::cuda::drv {

namespace {

// One pointer per entry point, all resolved together on first use. A null
// pointer after resolution means "no usable driver on this machine", and
// every wrapper then answers CUDA_ERROR_NOT_INITIALIZED — the same thing
// cuInit() itself says on such a box — so callers need no second code path.
struct Table {
    decltype(&::cuInit)                     cuInit = nullptr;
    decltype(&::cuDriverGetVersion)         cuDriverGetVersion = nullptr;
    decltype(&::cuDeviceGetCount)           cuDeviceGetCount = nullptr;
    decltype(&::cuDeviceGet)                cuDeviceGet = nullptr;
    decltype(&::cuDeviceGetAttribute)       cuDeviceGetAttribute = nullptr;
    decltype(&::cuDevicePrimaryCtxRetain)   cuDevicePrimaryCtxRetain = nullptr;
    decltype(&::cuCtxGetCurrent)            cuCtxGetCurrent = nullptr;
    decltype(&::cuCtxSetCurrent)            cuCtxSetCurrent = nullptr;
    decltype(&::cuModuleLoadData)           cuModuleLoadData = nullptr;
    decltype(&::cuModuleLoadDataEx)         cuModuleLoadDataEx = nullptr;
    decltype(&::cuModuleGetFunction)        cuModuleGetFunction = nullptr;
    decltype(&::cuModuleUnload)             cuModuleUnload = nullptr;
    decltype(&::cuLinkCreate)               cuLinkCreate = nullptr;
    decltype(&::cuLinkAddData)              cuLinkAddData = nullptr;
    decltype(&::cuLinkComplete)             cuLinkComplete = nullptr;
    decltype(&::cuLinkDestroy)              cuLinkDestroy = nullptr;
    decltype(&::cuLaunchKernel)             cuLaunchKernel = nullptr;
};

template <typename Fn>
void resolve(const char* symbol, Fn& out) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    // cudart asks the driver for the entry point matching the runtime's own
    // ABI version, which is the prototype cuda.h declared for us, so the cast
    // below is to the same function type the header names.
#if CUDART_VERSION >= 12050
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        symbol, &fn, CUDART_VERSION, cudaEnableDefault, &status);
#else
    const cudaError_t err = cudaGetDriverEntryPoint(symbol, &fn, cudaEnableDefault, &status);
#endif
    if (err == cudaSuccess && status == cudaDriverEntryPointSuccess && fn) {
        out = reinterpret_cast<Fn>(fn);
    }
}

const Table& table() {
    static Table t;
    static std::once_flag once;
    std::call_once(once, [] {
        resolve("cuInit", t.cuInit);
        resolve("cuDriverGetVersion", t.cuDriverGetVersion);
        resolve("cuDeviceGetCount", t.cuDeviceGetCount);
        resolve("cuDeviceGet", t.cuDeviceGet);
        resolve("cuDeviceGetAttribute", t.cuDeviceGetAttribute);
        resolve("cuDevicePrimaryCtxRetain", t.cuDevicePrimaryCtxRetain);
        resolve("cuCtxGetCurrent", t.cuCtxGetCurrent);
        resolve("cuCtxSetCurrent", t.cuCtxSetCurrent);
        resolve("cuModuleLoadData", t.cuModuleLoadData);
        resolve("cuModuleLoadDataEx", t.cuModuleLoadDataEx);
        resolve("cuModuleGetFunction", t.cuModuleGetFunction);
        resolve("cuModuleUnload", t.cuModuleUnload);
        resolve("cuLinkCreate", t.cuLinkCreate);
        resolve("cuLinkAddData", t.cuLinkAddData);
        resolve("cuLinkComplete", t.cuLinkComplete);
        resolve("cuLinkDestroy", t.cuLinkDestroy);
        resolve("cuLaunchKernel", t.cuLaunchKernel);
    });
    return t;
}

}  // namespace

CUresult cuInit(unsigned int flags) {
    const auto& t = table();
    return t.cuInit ? t.cuInit(flags) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    const auto& t = table();
    return t.cuDriverGetVersion ? t.cuDriverGetVersion(driverVersion)
                                : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuDeviceGetCount(int* count) {
    const auto& t = table();
    return t.cuDeviceGetCount ? t.cuDeviceGetCount(count) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    const auto& t = table();
    return t.cuDeviceGet ? t.cuDeviceGet(device, ordinal) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    const auto& t = table();
    return t.cuDeviceGetAttribute ? t.cuDeviceGetAttribute(pi, attrib, dev)
                                  : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    const auto& t = table();
    return t.cuDevicePrimaryCtxRetain ? t.cuDevicePrimaryCtxRetain(pctx, dev)
                                      : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    const auto& t = table();
    return t.cuCtxGetCurrent ? t.cuCtxGetCurrent(pctx) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    const auto& t = table();
    return t.cuCtxSetCurrent ? t.cuCtxSetCurrent(ctx) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    const auto& t = table();
    return t.cuModuleLoadData ? t.cuModuleLoadData(module, image)
                              : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int numOptions, CUjit_option* options,
                            void** optionValues) {
    const auto& t = table();
    return t.cuModuleLoadDataEx
        ? t.cuModuleLoadDataEx(module, image, numOptions, options, optionValues)
        : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    const auto& t = table();
    return t.cuModuleGetFunction ? t.cuModuleGetFunction(hfunc, hmod, name)
                                 : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuModuleUnload(CUmodule hmod) {
    const auto& t = table();
    return t.cuModuleUnload ? t.cuModuleUnload(hmod) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                      void** optionValues, CUlinkState* stateOut) {
    const auto& t = table();
    return t.cuLinkCreate ? t.cuLinkCreate(numOptions, options, optionValues, stateOut)
                          : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
                       void* data, std::size_t size, const char* name,
                       unsigned int numOptions, CUjit_option* options,
                       void** optionValues) {
    const auto& t = table();
    return t.cuLinkAddData
        ? t.cuLinkAddData(state, type, data, size, name, numOptions, options, optionValues)
        : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, std::size_t* sizeOut) {
    const auto& t = table();
    return t.cuLinkComplete ? t.cuLinkComplete(state, cubinOut, sizeOut)
                            : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuLinkDestroy(CUlinkState state) {
    const auto& t = table();
    return t.cuLinkDestroy ? t.cuLinkDestroy(state) : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    const auto& t = table();
    return t.cuLaunchKernel
        ? t.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, hStream, kernelParams, extra)
        : CUDA_ERROR_NOT_INITIALIZED;
}

}  // namespace brotensor::detail::cuda::drv
