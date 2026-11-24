#include "brotensor/kernel_interop.h"
#include "brotensor/runtime.h"
#include "cuda_driver.h"
#include "cuda_jit_engine.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace brotensor::detail::cuda {

namespace {

std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

bool cuda_interop_available(int device_index) {
    if (device_index < 0 || !jit::CudaJitEngine::instance().is_available()) {
        return false;
    }
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && device_index < count;
}

bool cuda_interop_dispatch_ptx(int device_index, const interop::PtxDispatchDesc& desc, std::string* err) {
    if (!desc.ptx || desc.ptx_bytes == 0) {
        if (err) *err = "dispatch_ptx: empty PTX module";
        return false;
    }
    if (!desc.entry_name || !desc.entry_name[0]) {
        if (err) *err = "dispatch_ptx: missing kernel entry name";
        return false;
    }

    const cudaError_t devErr = cudaSetDevice(device_index);
    if (devErr != cudaSuccess) {
        if (err) *err = std::string("dispatch_ptx: cudaSetDevice failed: ") + cudaGetErrorString(devErr);
        return false;
    }

    try {
        const std::size_t ptxLen =
            (desc.ptx[desc.ptx_bytes - 1] == '\0') ? (desc.ptx_bytes - 1) : desc.ptx_bytes;
        const std::uint64_t hash = fnv1a_bytes(desc.ptx, ptxLen);
        char keyBuf[96];
        std::snprintf(keyBuf, sizeof(keyBuf), "ptx_interop_%016llx_%zu_%s",
                      static_cast<unsigned long long>(hash), ptxLen, desc.entry_name);

        std::string ptxStr(desc.ptx, ptxLen);
        CUfunction fn = jit::CudaJitEngine::instance().get_function(keyBuf, ptxStr, desc.entry_name);

        CUstream stream = jit::resolve_stream(nullptr);
        const unsigned int gx = desc.groups[0] ? desc.groups[0] : 1u;
        const unsigned int gy = desc.groups[1] ? desc.groups[1] : 1u;
        const unsigned int gz = desc.groups[2] ? desc.groups[2] : 1u;
        const unsigned int bx = desc.local_size[0] ? desc.local_size[0] : 1u;
        const unsigned int by = desc.local_size[1] ? desc.local_size[1] : 1u;
        const unsigned int bz = desc.local_size[2] ? desc.local_size[2] : 1u;

        const CUresult res = drv::cuLaunchKernel(
            fn, gx, gy, gz, bx, by, bz, desc.shared_bytes, stream, desc.kernel_params, nullptr);
        if (res != CUDA_SUCCESS) {
            if (err) *err = "dispatch_ptx: cuLaunchKernel failed with CUDA error " + std::to_string(static_cast<int>(res));
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("dispatch_ptx: ") + ex.what();
        return false;
    }
}

} // namespace

void register_cuda_interop_backend() {
    interop::detail::register_cuda_interop_hooks(cuda_interop_available, cuda_interop_dispatch_ptx);
}

} // namespace brotensor::detail::cuda
