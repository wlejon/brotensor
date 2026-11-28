// CUDA fused filtered_lrelu backed by Brass MIR -> PTX kernels.

#include "cu_mir_cache.h"
#include "../gpu_common/filtered_lrelu.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace brotensor {

void filtered_lrelu_forward_composite(const Tensor& X, const Tensor& fu,
                                      const Tensor& fd, const Tensor* b,
                                      int N, int C, int H, int W, int up, int down,
                                      int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                                      float gain, float slope, float clamp,
                                      Tensor& up_buf, Tensor& act_buf, Tensor& Y);

namespace detail::cuda {

void* cuda_alloc(std::size_t);
void  cuda_free(void*);

void filtered_lrelu_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& fu,
                            const ::brotensor::Tensor& fd, const ::brotensor::Tensor* b,
                            int N, int C, int H, int W, int up, int down,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                            float gain, float slope, float clamp,
                            ::brotensor::Tensor& up_buf, ::brotensor::Tensor& act_buf,
                            ::brotensor::Tensor& Y) {
    const gpu_common::FilteredLreluPlan p = gpu_common::plan_filtered_lrelu(
        X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
        up_buf, act_buf, 48u * 1024u, /*shmem_uses_float_stage=*/true);
    if (!p.supported) {
        ::brotensor::filtered_lrelu_forward_composite(
            X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, act_buf, Y);
        return;
    }
    if (!gpu_common::prepare_filtered_lrelu_out(X, N, C, H, W, p, Y)) return;

    const Dtype dt = X.dtype;
    const std::size_t esz = gpu_common::fl_elem_size(dt);
    const std::size_t polyu_n = static_cast<std::size_t>(up) * up * p.JyMax * p.JxMax;
    const std::size_t polyd_n = static_cast<std::size_t>(p.fdH) * p.fdW;
    void* poly_u = cuda_alloc(polyu_n * esz);
    void* poly_d = cuda_alloc(polyd_n * esz);

    CUfunction polyu_fn = mir_function({MirKind::FilteredLrelu, 0u, dt});
    const std::uintptr_t polyu_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(fu.data),
        reinterpret_cast<std::uintptr_t>(poly_u),
    };
    const std::int32_t polyu_ints[] = {p.fuH, p.fuW, up, p.JyMax, p.JxMax};
    mir_dispatch_args_3d(
        polyu_fn,
        static_cast<std::uint32_t>((polyu_n + 127u) / 128u), 1u, 1u, 128u,
        polyu_ptrs, polyu_ints, {});

    CUfunction polyd_fn = mir_function({MirKind::FilteredLrelu, 1u, dt});
    const std::uintptr_t polyd_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(fd.data),
        reinterpret_cast<std::uintptr_t>(poly_d),
    };
    const std::int32_t polyd_ints[] = {p.fdH, p.fdW};
    mir_dispatch_args_3d(
        polyd_fn,
        static_cast<std::uint32_t>((polyd_n + 127u) / 128u), 1u, 1u, 128u,
        polyd_ptrs, polyd_ints, {});

    const bool has_bias = (b != nullptr && b->data != nullptr);
    const bool has_clamp = (clamp >= 0.0f);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (has_bias ? 1u : 0u) | (has_clamp ? 2u : 0u));
    const std::uint64_t aux = gpu_common::filtered_lrelu_main_aux(p, up, down);

    CUfunction main_fn = mir_function({MirKind::FilteredLrelu, 2u, dt, dt, 1u, dt, flags, aux});
    const float up_gain = static_cast<float>(up) * static_cast<float>(up);
    const std::uintptr_t main_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(X.data),
        reinterpret_cast<std::uintptr_t>(poly_u),
        reinterpret_cast<std::uintptr_t>(poly_d),
        has_bias ? reinterpret_cast<std::uintptr_t>(b->data) : 0u,
        reinterpret_cast<std::uintptr_t>(Y.data),
    };
    const std::int32_t main_ints[] = {C, H, W, p.Hout, p.Wout, pad_x0, pad_y0};
    const float main_flts[] = {up_gain, gain, slope, clamp};
    const std::uint32_t gx = static_cast<std::uint32_t>((p.Wout + gpu_common::kFlTileW - 1) / gpu_common::kFlTileW);
    const std::uint32_t gy = static_cast<std::uint32_t>((p.Hout + gpu_common::kFlTileH - 1) / gpu_common::kFlTileH);
    const std::uint32_t gz = static_cast<std::uint32_t>(N) * static_cast<std::uint32_t>(C);
    mir_dispatch_args_3d(main_fn, gx, gy, gz, gpu_common::kFlBlockThreads, main_ptrs, main_ints, main_flts);

    cuda_free(poly_u);
    cuda_free(poly_d);
}

void fill_cuda_vtable_filtered_lrelu(::brotensor::detail::OpsVTable& v) {
    v.filtered_lrelu_forward = &filtered_lrelu_forward;
}

}  // namespace detail::cuda
}  // namespace brotensor
