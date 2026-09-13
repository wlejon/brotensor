#include "cuda_jit.h"
#include "cuda_jit_engine.h"

#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/ptx_target.hpp>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda::jit {

void launch_fused_gemv_q8_0_ptx(
    const void* W_q8_0,
    const float* x,
    float* y,
    int N,
    int K,
    void* stream
) {
    if (N <= 0 || K <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_gemv_q8_0(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_gemv_q8_0",
        ptx,
        "fused_gemv_q8_0_kernel"
    );

    CUdeviceptr d_w = reinterpret_cast<CUdeviceptr>(W_q8_0);
    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(y);
    uint32_t n_arg = static_cast<uint32_t>(N);
    uint32_t k_arg = static_cast<uint32_t>(K);

    void* kernel_params[] = {
        &d_w,
        &d_x,
        &d_y,
        &n_arg,
        &k_arg
    };

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        static_cast<unsigned int>(N), 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_gemv_q8_0_kernel");
    }
}

void launch_fused_gemv_q4_k_ptx(
    const void* W_q4_k,
    const float* x,
    float* y,
    int N,
    int K,
    void* stream
) {
    if (N <= 0 || K <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_gemv_q4_k(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_gemv_q4_k",
        ptx,
        "fused_gemv_q4_k_kernel"
    );

    CUdeviceptr d_w = reinterpret_cast<CUdeviceptr>(W_q4_k);
    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(y);
    uint32_t n_arg = static_cast<uint32_t>(N);
    uint32_t k_arg = static_cast<uint32_t>(K);

    void* kernel_params[] = {
        &d_w,
        &d_x,
        &d_y,
        &n_arg,
        &k_arg
    };

    CUstream custream = resolve_stream(stream);
    CUresult status = cuLaunchKernel(
        fn,
        static_cast<unsigned int>(N), 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_gemv_q4_k_kernel");
    }
}

} // namespace brotensor::detail::cuda::jit
