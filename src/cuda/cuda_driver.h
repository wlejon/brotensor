#pragma once

// The CUDA *driver* API entry points the JIT uses, resolved at run time.
//
// brotensor_cuda links cudart statically, and cudart loads the driver
// (libcuda.so.1 / nvcuda.dll) lazily: on a machine with no NVIDIA driver it
// reports no device and init.cpp falls back to CPU, so one build runs
// everywhere. Linking CUDA::cuda_driver as well would undo that — it is a
// hard DT_NEEDED / import of the driver library, and the executable then does
// not *load* without one (a CPU-only box, a hosted CI runner, the manifest
// tool that runs at build time). So the ten driver functions below are
// fetched through cudart's cudaGetDriverEntryPoint the first time any of them
// is called, which fails cleanly — CUDA_ERROR_NOT_INITIALIZED, and the JIT
// engine reports itself unavailable — instead of failing at process start.
//
// The wrappers carry the exact prototypes cuda.h declares, so a call site
// reads as `drv::cuLaunchKernel(...)` and nothing else changes.

#include <cstddef>
#include <cuda.h>

namespace brotensor::detail::cuda::drv {

CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
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

}  // namespace brotensor::detail::cuda::drv
