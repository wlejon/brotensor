// ─── Metal fused filtered_lrelu (StyleGAN3 alias-free nonlinearity) ─────────
//
// Forward uses single-source Brass MIR -> MSL kernels (MirKind::FilteredLrelu:
// polyphase up-FIR, leaky ReLU + gain + clamp into threadgroup memory, dense
// down-FIR). Backward and uncached/oversize configs forward to the composite
// (upfirdn2d + bias_act on Metal GPU). Zero raw MSL shaders.

#include "mtl_mir_cache.h"

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

namespace {

constexpr int kFlTh = 16;
constexpr int kFlTw = 16;

inline bool is_fp(Dtype d) {
    return d == Dtype::FP32 || d == Dtype::FP16 || d == Dtype::BF16;
}

} // namespace

void filtered_lrelu_forward(const Tensor& X, const Tensor& fu, const Tensor& fd,
                            const Tensor* b, int N, int C, int H, int W,
                            int up, int down, int pad_x0, int pad_x1,
                            int pad_y0, int pad_y1, float gain, float slope,
                            float clamp, Tensor& up_buf, Tensor& act_buf,
                            Tensor& Y) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;
    const int Hu = H * up, Wu = W * up;
    const int Huo = Hu + pad_y0 + pad_y1 - fuH + 1;
    const int Wuo = Wu + pad_x0 + pad_x1 - fuW + 1;
    const int Hout = down >= 1 ? (Huo - fdH) / down + 1 : 0;
    const int Wout = down >= 1 ? (Wuo - fdW) / down + 1 : 0;

    const int JyMax = (up >= 1) ? (fuH + up - 1) / up : 0;
    const int JxMax = (up >= 1) ? (fuW + up - 1) / up : 0;
    const int AH = (kFlTh - 1) * down + fdH;
    const int AW = (kFlTw - 1) * down + fdW;
    const std::size_t tg_bytes =
        static_cast<std::size_t>(AH) * static_cast<std::size_t>(AW) *
        (X.dtype == Dtype::FP32 ? 4u : 2u);

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool fits_aux =
        up >= 1 && up < 256 && down >= 1 && down < 256 &&
        JyMax >= 1 && JyMax < 256 && JxMax >= 1 && JxMax < 256 &&
        fdH >= 1 && fdH < 256 && fdW >= 1 && fdW < 256;
    const bool supported =
        is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) &&
        Huo > 0 && Wuo > 0 && Hout > 0 && Wout > 0 &&
        fits_aux && tg_bytes <= 16u * 1024u &&
        static_cast<long long>(N) * C <= 65535;
    if (want_cache || !supported) {
        ::brotensor::filtered_lrelu_forward_composite(
            X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, act_buf, Y);
        return;
    }

    if (X.rows != N || X.cols != C * H * W)
        throw std::runtime_error("brotensor: filtered_lrelu_forward: X shape mismatch");
    const int out_cols = C * Hout * Wout;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype)
        Y.resize(N, out_cols, X.dtype);
    if (N == 0 || out_cols == 0) return;

    const Dtype dt = X.dtype;
    const int nu = up * up * JyMax * JxMax;
    const int nd = fdH * fdW;
    Tensor poly_u = Tensor::empty_on(Device::Metal, 1, nu, dt);
    Tensor poly_d = Tensor::empty_on(Device::Metal, 1, nd, dt);

    const MirKernel polyu_mk = mir_kernel({MirKind::FilteredLrelu, 0u, dt});
    const void* polyu_ptrs[] = {fu.data, poly_u.data};
    const std::int32_t polyu_ints[] = {fuH, fuW, up, JyMax, JxMax};
    mir_dispatch_args_3d(
        polyu_mk, static_cast<std::uint32_t>((nu + 127) / 128), 1u, 1u,
        "filtered_lrelu_polyu", polyu_ptrs, polyu_ints, {});

    const MirKernel polyd_mk = mir_kernel({MirKind::FilteredLrelu, 1u, dt});
    const void* polyd_ptrs[] = {fd.data, poly_d.data};
    const std::int32_t polyd_ints[] = {fdH, fdW};
    mir_dispatch_args_3d(
        polyd_mk, static_cast<std::uint32_t>((nd + 127) / 128), 1u, 1u,
        "filtered_lrelu_polyd", polyd_ptrs, polyd_ints, {});

    const bool has_bias = (b != nullptr && b->data != nullptr);
    const bool has_clamp = (clamp >= 0.0f);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (has_bias ? 1u : 0u) | (has_clamp ? 2u : 0u));
    const std::uint64_t aux =
        static_cast<std::uint64_t>(up) |
        (static_cast<std::uint64_t>(down) << 8) |
        (static_cast<std::uint64_t>(JyMax) << 16) |
        (static_cast<std::uint64_t>(JxMax) << 24) |
        (static_cast<std::uint64_t>(fdH) << 32) |
        (static_cast<std::uint64_t>(fdW) << 40);

    const MirKernel main_mk = mir_kernel({MirKind::FilteredLrelu, 2u, dt, dt, 1u, dt, flags, aux});
    const float up_gain = static_cast<float>(up) * static_cast<float>(up);
    const void* main_ptrs[] = {
        X.data, poly_u.data, poly_d.data, has_bias ? b->data : nullptr, Y.data,
    };
    const std::int32_t main_ints[] = {C, H, W, Hout, Wout, pad_x0, pad_y0};
    const float main_flts[] = {up_gain, gain, slope, clamp};
    const std::uint32_t gx = static_cast<std::uint32_t>((Wout + kFlTw - 1) / kFlTw);
    const std::uint32_t gy = static_cast<std::uint32_t>((Hout + kFlTh - 1) / kFlTh);
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
