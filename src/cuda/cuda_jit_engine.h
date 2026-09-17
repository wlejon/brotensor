#pragma once

#include "cuda_driver.h"

#include <cuda.h>
#include <cuda_runtime.h>
#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/target/ptx_target.hpp>
#endif

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda::jit {

class CudaJitEngine {
public:
    static CudaJitEngine& instance() {
        static CudaJitEngine engine;
        return engine;
    }

    bool is_available() {
        std::lock_guard<std::mutex> lock(mu_);
        ensure_initialized_locked();
        return available_;
    }

    CUfunction get_function(const std::string& key, const std::string& ptx, const std::string& entry_name) {
        std::lock_guard<std::mutex> lock(mu_);
        ensure_initialized_locked();
        if (!available_) {
            throw std::runtime_error("CUDA Driver JIT is not available on this system");
        }

        auto it = function_cache_.find(key);
        if (it != function_cache_.end()) {
            return it->second;
        }

        // Ensure active context
        CUcontext current_ctx = nullptr;
        drv::cuCtxGetCurrent(&current_ctx);
        if (!current_ctx) {
            drv::cuCtxSetCurrent(context_);
        }

        // Setup JIT options with maximum optimization level 4
        constexpr size_t LOG_SIZE = 8192;
        std::vector<char> error_log(LOG_SIZE, 0);
        std::vector<char> info_log(LOG_SIZE, 0);

        CUjit_option options[] = {
            CU_JIT_OPTIMIZATION_LEVEL,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
        };

        void* option_values[] = {
            reinterpret_cast<void*>(static_cast<uintptr_t>(4)),
            error_log.data(),
            reinterpret_cast<void*>(static_cast<uintptr_t>(LOG_SIZE)),
            info_log.data(),
            reinterpret_cast<void*>(static_cast<uintptr_t>(LOG_SIZE))
        };

        CUmodule module = nullptr;
        CUresult res = drv::cuModuleLoadDataEx(&module, ptx.c_str(), 5, options, option_values);
        if (res != CUDA_SUCCESS) {
            std::string err_msg = "CUDA JIT compilation failed for " + entry_name + ": " + error_log.data();
            throw std::runtime_error(err_msg);
        }
        modules_.push_back(module);

        CUfunction fn = nullptr;
        res = drv::cuModuleGetFunction(&fn, module, entry_name.c_str());
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to find kernel entry point '" + entry_name + "' in compiled PTX module");
        }

        function_cache_[key] = fn;
        return fn;
    }

private:
    CudaJitEngine() = default;

    ~CudaJitEngine() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto mod : modules_) {
            if (mod) {
                drv::cuModuleUnload(mod);
            }
        }
        modules_.clear();
        function_cache_.clear();
    }

    void ensure_initialized_locked() {
        if (initialized_) return;
        initialized_ = true;

        // Without a driver on the machine every drv:: call answers
        // CUDA_ERROR_NOT_INITIALIZED, so this is also the "no GPU" exit.
        CUresult res = drv::cuInit(0);
        if (res != CUDA_SUCCESS) {
            available_ = false;
            return;
        }

        int dev_count = 0;
        res = drv::cuDeviceGetCount(&dev_count);
        if (res != CUDA_SUCCESS || dev_count <= 0) {
            available_ = false;
            return;
        }

        res = drv::cuCtxGetCurrent(&context_);
        if (res != CUDA_SUCCESS || !context_) {
            CUdevice dev;
            if (drv::cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
                if (drv::cuDevicePrimaryCtxRetain(&context_, dev) == CUDA_SUCCESS) {
                    drv::cuCtxSetCurrent(context_);
                }
            }
        }

        available_ = (context_ != nullptr);
    }

    std::mutex mu_;
    bool initialized_ = false;
    bool available_ = false;
    CUcontext context_ = nullptr;
    std::unordered_map<std::string, CUfunction> function_cache_;
    std::vector<CUmodule> modules_;
};

inline CUstream resolve_stream(void* stream) {
    if (stream) return reinterpret_cast<CUstream>(stream);
    return reinterpret_cast<CUstream>(::brotensor::cuda_current_stream());
}

#if BROTENSOR_HAS_BRASS_CUDA_JIT
inline brass::target::PtxOptions make_ptx_opts() {
    brass::target::PtxOptions opts;
    opts.sm_arch = "sm_89";
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 8;
    return opts;
}
#endif

} // namespace brotensor::detail::cuda::jit
