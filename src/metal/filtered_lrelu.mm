// ─── Metal fused filtered_lrelu (StyleGAN3 alias-free nonlinearity) ─────────
//
// Forward uses single-source Brass MIR -> MSL kernels (MirKind::FilteredLrelu:
// polyphase up-FIR, leaky ReLU + gain + clamp into threadgroup memory, dense
// down-FIR). Backward and uncached/oversize configs forward to the composite
// (upfirdn2d + bias_act on Metal GPU). Zero raw MSL shaders.

#include "mtl_mir_cache.h"
#include "../gpu_common/filtered_lrelu.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor {
void filtered_lrelu_forward_composite(const Tensor& X, const Tensor& fu,
                                      const Tensor& fd, const Tensor* b,
                                      int N, int C, int H, int W, int up, int down,
                                      int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                                      float gain, float slope, float clamp,
                                      Tensor& up_buf, Tensor& act_buf, Tensor& Y);
void filtered_lrelu_backward_composite(const Tensor& dY, const Tensor& X,
                                       const Tensor& fu, const Tensor& fd,
                                       const Tensor* b, int N, int C, int H, int W,
                                       int up, int down, int pad_x0, int pad_x1,
                                       int pad_y0, int pad_y1, float gain, float slope,
                                       float clamp, const Tensor& up_buf,
                                       Tensor& dX, Tensor* dB);
} // namespace brotensor

namespace brotensor::detail::metal {

void filtered_lrelu_forward(const Tensor& X, const Tensor& fu, const Tensor& fd,
                            const Tensor* b, int N, int C, int H, int W,
                            int up, int down, int pad_x0, int pad_x1,
                            int pad_y0, int pad_y1, float gain, float slope,
                            float clamp, Tensor& up_buf, Tensor& act_buf,
                            Tensor& Y) {
    const gpu_common::FilteredLreluPlan p = gpu_common::plan_filtered_lrelu(
        X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
        up_buf, act_buf, 16u * 1024u, /*shmem_uses_float_stage=*/false);
    if (!p.supported) {
        ::brotensor::filtered_lrelu_forward_composite(
            X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, act_buf, Y);
        return;
    }
    if (!gpu_common::prepare_filtered_lrelu_out(X, N, C, H, W, p, Y)) return;

    const Dtype dt = X.dtype;
    const int nu = up * up * p.JyMax * p.JxMax;
    const int nd = p.fdH * p.fdW;
    Tensor poly_u = Tensor::empty_on(Device::Metal, 1, nu, dt);
    Tensor poly_d = Tensor::empty_on(Device::Metal, 1, nd, dt);

    const MirKernel polyu_mk = mir_kernel({MirKind::FilteredLrelu, 0u, dt});
    const void* polyu_ptrs[] = {fu.data, poly_u.data};
    const std::int32_t polyu_ints[] = {p.fuH, p.fuW, up, p.JyMax, p.JxMax};
    mir_dispatch_args_3d(
        polyu_mk, static_cast<std::uint32_t>((nu + 127) / 128), 1u, 1u,
        "filtered_lrelu_polyu", polyu_ptrs, polyu_ints, {});

    const MirKernel polyd_mk = mir_kernel({MirKind::FilteredLrelu, 1u, dt});
    const void* polyd_ptrs[] = {fd.data, poly_d.data};
    const std::int32_t polyd_ints[] = {p.fdH, p.fdW};
    mir_dispatch_args_3d(
        polyd_mk, static_cast<std::uint32_t>((nd + 127) / 128), 1u, 1u,
        "filtered_lrelu_polyd", polyd_ptrs, polyd_ints, {});

    const bool has_bias = (b != nullptr && b->data != nullptr);
    const bool has_clamp = (clamp >= 0.0f);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (has_bias ? 1u : 0u) | (has_clamp ? 2u : 0u));
    const std::uint64_t aux = gpu_common::filtered_lrelu_main_aux(p, up, down);

    const MirKernel main_mk = mir_kernel({MirKind::FilteredLrelu, 2u, dt, dt, 1u, dt, flags, aux});
    const float up_gain = static_cast<float>(up) * static_cast<float>(up);
    const void* main_ptrs[] = {
        X.data, poly_u.data, poly_d.data, has_bias ? b->data : nullptr, Y.data,
    };
    const std::int32_t main_ints[] = {C, H, W, p.Hout, p.Wout, pad_x0, pad_y0};
    const float main_flts[] = {up_gain, gain, slope, clamp};
    const std::uint32_t gx = static_cast<std::uint32_t>((p.Wout + gpu_common::kFlTileW - 1) / gpu_common::kFlTileW);
    const std::uint32_t gy = static_cast<std::uint32_t>((p.Hout + gpu_common::kFlTileH - 1) / gpu_common::kFlTileH);
    const std::uint32_t gz = static_cast<std::uint32_t>(N) * static_cast<std::uint32_t>(C);
    mir_dispatch_args_3d(main_mk, gx, gy, gz, "filtered_lrelu_forward",
                         main_ptrs, main_ints, main_flts);
}

void filtered_lrelu_backward(const Tensor& dY, const Tensor& X,
                             const Tensor& fu, const Tensor& fd,
                             const Tensor* b, int N, int C, int H, int W,
                             int up, int down, int pad_x0, int pad_x1,
                             int pad_y0, int pad_y1, float gain, float slope,
                             float clamp, const Tensor& up_buf,
                             Tensor& dX, Tensor* dB) {
    ::brotensor::filtered_lrelu_backward_composite(
        dY, X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
        gain, slope, clamp, up_buf, dX, dB);
}

} // namespace brotensor::detail::metal
