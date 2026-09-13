#include "cuda_jit.h"
#include <brotensor/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/ptx_target.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda::jit {

namespace {

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
        cuCtxGetCurrent(&current_ctx);
        if (!current_ctx) {
            cuCtxSetCurrent(context_);
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
        CUresult res = cuModuleLoadDataEx(&module, ptx.c_str(), 5, options, option_values);
        if (res != CUDA_SUCCESS) {
            std::string err_msg = "CUDA JIT compilation failed for " + entry_name + ": " + error_log.data();
            throw std::runtime_error(err_msg);
        }
        modules_.push_back(module);

        CUfunction fn = nullptr;
        res = cuModuleGetFunction(&fn, module, entry_name.c_str());
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
                cuModuleUnload(mod);
            }
        }
        modules_.clear();
        function_cache_.clear();
    }

    void ensure_initialized_locked() {
        if (initialized_) return;
        initialized_ = true;

        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            available_ = false;
            return;
        }

        int dev_count = 0;
        res = cuDeviceGetCount(&dev_count);
        if (res != CUDA_SUCCESS || dev_count <= 0) {
            available_ = false;
            return;
        }

        res = cuCtxGetCurrent(&context_);
        if (res != CUDA_SUCCESS || !context_) {
            CUdevice dev;
            if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
                if (cuDevicePrimaryCtxRetain(&context_, dev) == CUDA_SUCCESS) {
                    cuCtxSetCurrent(context_);
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

inline brass::target::PtxOptions make_ptx_opts() {
    brass::target::PtxOptions opts;
    opts.sm_arch = "sm_89";
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 8;
    return opts;
}

} // namespace

bool is_cuda_jit_available() {
    return CudaJitEngine::instance().is_available();
}

void launch_fused_residual_rmsnorm_ptx(
    float* X,
    const float* res,
    const float* gamma,
    float* Y,
    int B,
    int D,
    float eps,
    void* stream
) {
    if (B <= 0 || D <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_residual_rms_norm(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_residual_rms_norm",
        ptx,
        "fused_residual_rms_norm_kernel"
    );

    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    CUdeviceptr d_res = reinterpret_cast<CUdeviceptr>(res);
    CUdeviceptr d_gamma = reinterpret_cast<CUdeviceptr>(gamma);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    uint32_t b_arg = static_cast<uint32_t>(B);
    uint32_t d_arg = static_cast<uint32_t>(D);
    float eps_arg = eps;

    void* kernel_params[] = {
        &d_x,
        &d_res,
        &d_gamma,
        &d_y,
        &b_arg,
        &d_arg,
        &eps_arg
    };

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        static_cast<unsigned int>(B), 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_residual_rms_norm_kernel");
    }
}

void launch_fused_layernorm_modulate_ptx(
    const float* X,
    const float* gamma,
    const float* beta,
    const float* scale,
    const float* shift,
    float* Y,
    int R,
    int D,
    float eps,
    void* stream
) {
    if (R <= 0 || D <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_layernorm_modulate(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_layernorm_modulate",
        ptx,
        "fused_layernorm_modulate_kernel"
    );

    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    CUdeviceptr d_gamma = reinterpret_cast<CUdeviceptr>(gamma);
    CUdeviceptr d_beta = reinterpret_cast<CUdeviceptr>(beta);
    CUdeviceptr d_scale = reinterpret_cast<CUdeviceptr>(scale);
    CUdeviceptr d_shift = reinterpret_cast<CUdeviceptr>(shift);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    uint32_t r_arg = static_cast<uint32_t>(R);
    uint32_t d_arg = static_cast<uint32_t>(D);
    float eps_arg = eps;

    void* kernel_params[] = {
        &d_x,
        &d_gamma,
        &d_beta,
        &d_scale,
        &d_shift,
        &d_y,
        &r_arg,
        &d_arg,
        &eps_arg
    };

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        static_cast<unsigned int>(R), 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_layernorm_modulate_kernel");
    }
}

void launch_swiglu_ptx(
    const float* X,
    float* Y,
    int B,
    int D,
    void* stream
) {
    if (B <= 0 || D <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_swiglu(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "swiglu",
        ptx,
        "swiglu_kernel"
    );

    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    uint32_t b_arg = static_cast<uint32_t>(B);
    uint32_t d_arg = static_cast<uint32_t>(D);

    void* kernel_params[] = {
        &d_x,
        &d_y,
        &b_arg,
        &d_arg
    };

    uint32_t total_vec = (static_cast<uint32_t>(B) * static_cast<uint32_t>(D)) / 4;
    unsigned int grid_x = std::min(65535u, (total_vec + 255u) / 256u);
    if (grid_x == 0) grid_x = 1;

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        grid_x, 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for swiglu_kernel");
    }
}

void launch_modulate_ptx(
    const float* X,
    const float* scale,
    const float* shift,
    float* Y,
    int L,
    int D,
    void* stream
) {
    if (L <= 0 || D <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_adaln_modulate(false, make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "adaln_modulate",
        ptx,
        "adaln_modulate_kernel"
    );

    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    CUdeviceptr d_scale = reinterpret_cast<CUdeviceptr>(scale);
    CUdeviceptr d_shift = reinterpret_cast<CUdeviceptr>(shift);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    uint32_t l_arg = static_cast<uint32_t>(L);
    uint32_t d_arg = static_cast<uint32_t>(D);

    void* kernel_params[] = {
        &d_x,
        &d_scale,
        &d_shift,
        &d_y,
        &l_arg,
        &d_arg
    };

    unsigned int grid_x = static_cast<unsigned int>(std::min(65535, L));
    if (grid_x == 0) grid_x = 1;

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        grid_x, 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for adaln_modulate_kernel");
    }
}

} // namespace brotensor::detail::cuda::jit
