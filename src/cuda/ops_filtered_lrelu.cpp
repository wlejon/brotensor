// CUDA fused filtered_lrelu backed by Brass MIR -> PTX kernels.

#include "cu_mir_cache.h"

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

namespace {

constexpr int kFlTh = 16;
constexpr int kFlTw = 16;
constexpr std::uint32_t kFlThreads = static_cast<std::uint32_t>(kFlTh * kFlTw);

inline bool is_fp(::brotensor::Dtype d) {
    return d == ::brotensor::Dtype::FP32 || d == ::brotensor::Dtype::FP16 ||
           d == ::brotensor::Dtype::BF16;
}

inline std::size_t fl_elem_size(::brotensor::Dtype d) {
    return (d == ::brotensor::Dtype::FP32) ? 4u : 2u;
}

}  // namespace

void filtered_lrelu_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& fu,
                            const ::brotensor::Tensor& fd, const ::brotensor::Tensor* b,
                            int N, int C, int H, int W, int up, int down,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                            float gain, float slope, float clamp,
                            ::brotensor::Tensor& up_buf, ::brotensor::Tensor& act_buf,
                            ::brotensor::Tensor& Y) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;
    const int Hu = H * up, Wu = W * up;
    const int Huo = Hu + pad_y0 + pad_y1 - fuH + 1;
    const int Wuo = Wu + pad_x0 + pad_x1 - fuW + 1;
    const int Hout = (Huo - fdH) / down + 1;
    const int Wout = (Wuo - fdW) / down + 1;

    const int JyMax = (fuH + up - 1) / up;
    const int JxMax = (fuW + up - 1) / up;
    const int AH = (kFlTh - 1) * down + fdH;
    const int AW = (kFlTw - 1) * down + fdW;
    const std::size_t shmem_max = static_cast<std::size_t>(AH) * AW * sizeof(float);

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool fits_aux =
        up < 256 && down < 256 && JyMax >= 1 && JyMax < 256 &&
        JxMax >= 1 && JxMax < 256 && fdH >= 1 && fdH < 256 && fdW >= 1 && fdW < 256;
    const bool supported =
        is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) && up >= 1 && down >= 1 &&
        Huo > 0 && Wuo > 0 && Hout > 0 && Wout > 0 &&
        fits_aux &&
        shmem_max <= 48u * 1024u &&
        static_cast<long long>(N) * C <= 65535;
    if (want_cache || !supported) {
        ::brotensor::filtered_lrelu_forward_composite(
            X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, act_buf, Y);
        return;
    }

    if (X.rows != N || X.cols != C * H * W) {
        throw std::runtime_error("brotensor: filtered_lrelu_forward: X shape mismatch");
    }
    const int out_cols = C * Hout * Wout;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const long long total = static_cast<long long>(N) * out_cols;
    if (total == 0) return;

    const Dtype dt = X.dtype;
    const std::size_t esz = fl_elem_size(dt);
    const std::size_t polyu_n = static_cast<std::size_t>(up) * up * JyMax * JxMax;
    const std::size_t polyd_n = static_cast<std::size_t>(fdH) * fdW;
    void* poly_u = cuda_alloc(polyu_n * esz);
    void* poly_d = cuda_alloc(polyd_n * esz);

    CUfunction polyu_fn = mir_function({MirKind::FilteredLrelu, 0u, dt});
    const std::uintptr_t polyu_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(fu.data),
        reinterpret_cast<std::uintptr_t>(poly_u),
    };
    const std::int32_t polyu_ints[] = {fuH, fuW, up, JyMax, JxMax};
    mir_dispatch_args_3d(
        polyu_fn,
        static_cast<std::uint32_t>((polyu_n + 127u) / 128u), 1u, 1u, 128u,
        polyu_ptrs, polyu_ints, {});

    CUfunction polyd_fn = mir_function({MirKind::FilteredLrelu, 1u, dt});
    const std::uintptr_t polyd_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(fd.data),
        reinterpret_cast<std::uintptr_t>(poly_d),
    };
    const std::int32_t polyd_ints[] = {fdH, fdW};
    mir_dispatch_args_3d(
        polyd_fn,
        static_cast<std::uint32_t>((polyd_n + 127u) / 128u), 1u, 1u, 128u,
        polyd_ptrs, polyd_ints, {});

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

    CUfunction main_fn = mir_function({MirKind::FilteredLrelu, 2u, dt, dt, 1u, dt, flags, aux});
    const float up_gain = static_cast<float>(up) * static_cast<float>(up);
    const std::uintptr_t main_ptrs[] = {
        reinterpret_cast<std::uintptr_t>(X.data),
        reinterpret_cast<std::uintptr_t>(poly_u),
        reinterpret_cast<std::uintptr_t>(poly_d),
        has_bias ? reinterpret_cast<std::uintptr_t>(b->data) : 0u,
        reinterpret_cast<std::uintptr_t>(Y.data),
    };
    const std::int32_t main_ints[] = {C, H, W, Hout, Wout, pad_x0, pad_y0};
    const float main_flts[] = {up_gain, gain, slope, clamp};
    const std::uint32_t gx = static_cast<std::uint32_t>((Wout + kFlTw - 1) / kFlTw);
    const std::uint32_t gy = static_cast<std::uint32_t>((Hout + kFlTh - 1) / kFlTh);
    const std::uint32_t gz = static_cast<std::uint32_t>(N) * static_cast<std::uint32_t>(C);
    mir_dispatch_args_3d(main_fn, gx, gy, gz, kFlThreads, main_ptrs, main_ints, main_flts);

    cuda_free(poly_u);
    cuda_free(poly_d);
}

void fill_cuda_vtable_filtered_lrelu(::brotensor::detail::OpsVTable& v) {
    v.filtered_lrelu_forward = &filtered_lrelu_forward;
}

}  // namespace detail::cuda
}  // namespace brotensor
