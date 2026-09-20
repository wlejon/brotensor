#include "cuda_jit.h"
#include "cuda_jit_engine.h"
#include <brotensor/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/ptx_target.hpp>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace brotensor::detail::cuda::jit {


bool is_cuda_jit_available() {
    return CudaJitEngine::instance().is_available();
}

#if BROTENSOR_HAS_BRASS_CUDA_JIT

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
    CUresult status = drv::cuLaunchKernel(
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

void launch_fused_residual_layernorm_ptx(
    float* X,
    const float* res,
    const float* gamma,
    const float* beta,
    float* Y,
    int B,
    int D,
    float eps,
    void* stream
) {
    if (B <= 0 || D <= 0) return;

    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_fused_residual_layernorm(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "fused_residual_layernorm",
        ptx,
        "fused_residual_layernorm_kernel"
    );

    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    CUdeviceptr d_res = reinterpret_cast<CUdeviceptr>(res);
    CUdeviceptr d_gamma = reinterpret_cast<CUdeviceptr>(gamma);
    CUdeviceptr d_beta = reinterpret_cast<CUdeviceptr>(beta);
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    uint32_t b_arg = static_cast<uint32_t>(B);
    uint32_t d_arg = static_cast<uint32_t>(D);
    float eps_arg = eps;

    void* kernel_params[] = {
        &d_x,
        &d_res,
        &d_gamma,
        &d_beta,
        &d_y,
        &b_arg,
        &d_arg,
        &eps_arg
    };

    CUstream custream = resolve_stream(stream);
    CUresult status = drv::cuLaunchKernel(
        fn,
        static_cast<unsigned int>(B), 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_residual_layernorm_kernel");
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
    CUresult status = drv::cuLaunchKernel(
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

    // X is one packed gate|up projection, [B, 2D] — the layout swiglu_forward
    // is defined on — so this is brass's packed kernel, not
    // fused_swiglu_kernel(gate, up, out, n), which wants two flat buffers.
    static const std::string ptx = []() {
        brass::codegen::MlFusionCompiler comp;
        return comp.emit_ptx_swiglu_packed(make_ptx_opts());
    }();

    CUfunction fn = CudaJitEngine::instance().get_function(
        "swiglu_packed",
        ptx,
        "fused_swiglu_packed_kernel"
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
    CUresult status = drv::cuLaunchKernel(
        fn,
        grid_x, 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_swiglu_packed_kernel");
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
        "fused_adaln_modulate_kernel"
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
    CUresult status = drv::cuLaunchKernel(
        fn,
        grid_x, 1, 1,
        256, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for fused_adaln_modulate_kernel");
    }
}

void launch_elementwise_div_ptx(
    float* Y,
    const float* X,
    int n,
    void* stream
) {
    if (n <= 0) return;

    static const std::string ptx = R"(
.version 7.8
.target sm_89
.address_size 64

.visible .entry elementwise_div_kernel(
    .param .u64 param_y,
    .param .u64 param_x,
    .param .u32 param_n
) {
    .reg .b32 %r_tid, %r_ntid, %r_ctaid, %r_ncta, %r_stride, %r_idx, %r_n;
    .reg .pred %p_oob;
    .reg .b64 %r_off64;
    .reg .b64 %r_p_y, %r_addr_y;
    .reg .b64 %r_p_x, %r_addr_x;
    .reg .f32 %f_y, %f_x, %f_res;

    ld.param.u32 %r_n, [param_n];
    ld.param.u64 %r_p_y, [param_y];
    ld.param.u64 %r_p_x, [param_x];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ncta, %nctaid.x;
    mad.lo.u32 %r_idx, %r_ctaid, %r_ntid, %r_tid;
    mul.lo.u32 %r_stride, %r_ntid, %r_ncta;

LOOP:
    setp.ge.u32 %p_oob, %r_idx, %r_n;
    @%p_oob bra EXIT;

    cvt.u64.u32 %r_off64, %r_idx;
    shl.b64 %r_off64, %r_off64, 2;

    add.u64 %r_addr_y, %r_p_y, %r_off64;
    ld.global.f32 %f_y, [%r_addr_y];

    add.u64 %r_addr_x, %r_p_x, %r_off64;
    ld.global.f32 %f_x, [%r_addr_x];

    div.rn.f32 %f_res, %f_y, %f_x;
    st.global.f32 [%r_addr_y], %f_res;

    add.u32 %r_idx, %r_idx, %r_stride;
    bra LOOP;

EXIT:
    ret;
}
)";

    CUfunction fn = CudaJitEngine::instance().get_function(
        "elementwise_div",
        ptx,
        "elementwise_div_kernel"
    );

    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(Y);
    CUdeviceptr d_x = reinterpret_cast<CUdeviceptr>(X);
    uint32_t n_arg = static_cast<uint32_t>(n);

    void* kernel_params[] = {
        &d_y,
        &d_x,
        &n_arg
    };

    unsigned int block_size = 256;
    unsigned int grid_size = (n_arg + block_size - 1) / block_size;
    if (grid_size == 0) grid_size = 1;
    if (grid_size > 65535) grid_size = 65535;

    CUstream custream = resolve_stream(stream);
    CUresult status = drv::cuLaunchKernel(
        fn,
        grid_size, 1, 1,
        block_size, 1, 1,
        0,
        custream,
        kernel_params,
        nullptr
    );

    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for elementwise_div_kernel");
    }
}

#else // !BROTENSOR_HAS_BRASS_CUDA_JIT

void launch_fused_residual_rmsnorm_ptx(float*, const float*, const float*, float*, int, int, float, void*) {}
void launch_fused_residual_layernorm_ptx(float*, const float*, const float*, const float*, float*, int, int, float, void*) {}
void launch_fused_layernorm_modulate_ptx(const float*, const float*, const float*, const float*, const float*, float*, int, int, float, void*) {}
void launch_swiglu_ptx(const float*, float*, int, int, void*) {}
void launch_modulate_ptx(const float*, const float*, const float*, float*, int, int, void*) {}
void launch_elementwise_div_ptx(float*, const float*, int, void*) {}

#endif // BROTENSOR_HAS_BRASS_CUDA_JIT

} // namespace brotensor::detail::cuda::jit
