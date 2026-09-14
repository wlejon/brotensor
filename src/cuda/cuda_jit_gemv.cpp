#include "cuda_jit.h"
#include "cuda_jit_engine.h"

#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/ptx_target.hpp>
#endif

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda::jit {

#if BROTENSOR_HAS_BRASS_CUDA_JIT

void launch_fused_gemv_swiglu_ptx(
    const float* W_gate,
    const float* W_up,
    const float* x,
    float* y,
    int N,
    int K,
    void* stream
) {
    if (N <= 0 || K <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_gemv_swiglu(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_gemv_swiglu",
        ptx,
        "fused_gemv_swiglu_kernel"
    );

    CUdeviceptr d_w_gate = reinterpret_cast<CUdeviceptr>(W_gate);
    CUdeviceptr d_w_up = reinterpret_cast<CUdeviceptr>(W_up);
    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(y);
    uint32_t n_arg = static_cast<uint32_t>(N);
    uint32_t k_arg = static_cast<uint32_t>(K);

    void* kernel_params[] = {
        &d_w_gate,
        &d_w_up,
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
        throw std::runtime_error("cuLaunchKernel failed for fused_gemv_swiglu_kernel");
    }
}

void launch_fused_gemv_residual_ptx(
    const float* W_down,
    const float* x,
    const float* res,
    float* y,
    int N,
    int K,
    void* stream
) {
    if (N <= 0 || K <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_gemv_residual(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_gemv_residual",
        ptx,
        "fused_gemv_residual_kernel"
    );

    CUdeviceptr d_w_down = reinterpret_cast<CUdeviceptr>(W_down);
    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr d_res = reinterpret_cast<CUdeviceptr>(res);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(y);
    uint32_t n_arg = static_cast<uint32_t>(N);
    uint32_t k_arg = static_cast<uint32_t>(K);

    void* kernel_params[] = {
        &d_w_down,
        &d_x,
        &d_res,
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
        throw std::runtime_error("cuLaunchKernel failed for fused_gemv_residual_kernel");
    }
}

#else // !BROTENSOR_HAS_BRASS_CUDA_JIT

void launch_fused_gemv_swiglu_ptx(const float*, const float*, const float*, float*, int, int, void*) {}
void launch_fused_gemv_residual_ptx(const float*, const float*, const float*, float*, int, int, void*) {}

#endif // BROTENSOR_HAS_BRASS_CUDA_JIT

} // namespace brotensor::detail::cuda::jit
