// CUDA vision ops backed by Brass MIR -> PTX (MirKind::Vision):
// deform_conv2d_forward and modulated_conv2d_{forward,backward}.

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"
#include "../kernels/mir_vision.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in ops_elementwise.cpp / matmul.cu / conv2d.cu.
void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, Tensor& Y);
void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                           int N, int C_in, int H, int W, int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, Tensor& dX);
void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                            int N, int C_in, int H, int W, int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups, Tensor& dWt);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::int32_t i32(long long v) noexcept {
    return static_cast<std::int32_t>(v);
}

std::size_t esize(Dtype t) noexcept {
    return static_cast<std::size_t>(::brotensor::dtype_size_bytes(t));
}

const void* opt(const Tensor* t) noexcept {
    return (t && t->data && t->size() > 0) ? t->data : nullptr;
}

void* byte_off(void* base, std::size_t off_bytes) noexcept {
    return static_cast<char*>(base) + off_bytes;
}

const void* byte_off(const void* base, std::size_t off_bytes) noexcept {
    return static_cast<const char*>(base) + off_bytes;
}

void run_vision(std::uint32_t code, Dtype dt, std::uint8_t flags, std::uint64_t items,
                bool per_group, std::span<const std::uintptr_t> ptrs,
                std::span<const std::int32_t> ints, std::span<const float> flts) {
    if (items == 0) return;
    CUfunction fn = mir_function(MirKey{MirKind::Vision, code, dt, dt, 1, dt, flags});
    const std::uint32_t gx = per_group
        ? static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535))
        : mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kVisionBlock);
    mir_dispatch_args(fn, gx, kernels::kVisionBlock, ptrs, ints, flts);
}

}  // namespace

void deform_conv2d_forward(const Tensor& X, const Tensor& offset, const Tensor* mask,
                           const Tensor& Wt, const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, int deform_groups,
                           Tensor& Y) {
    constexpr const char* op = "deform_conv2d_forward";
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    need(op, Wt.dtype == dt && offset.dtype == dt &&
                 (!opt(mask) || mask->dtype == dt) && (!opt(bias) || bias->dtype == dt),
         "X, offset, mask, Wt, bias dtype must match");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must divide C_in and C_out");
    need(op, deform_groups >= 1 && C_in % deform_groups == 0,
         "deform_groups must divide C_in");
    need(op, N >= 0 && H > 0 && W > 0 && kH > 0 && kW > 0 &&
                 stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0 &&
                 pad_h >= 0 && pad_w >= 0,
         "bad geometry");
    need(op, stride_h < 65536 && stride_w < 65536 && pad_h < 65536 &&
                 pad_w < 65536 && dil_h < 65536 && dil_w < 65536,
         "geometry above 65535");
    const int Ho = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wo = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    const int Cgi = C_in / groups, Cgo = C_out / groups, kk = kH * kW, P = Ho * Wo;
    const long long Kd = static_cast<long long>(Cgi) * kk;
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * W,
         "X is smaller than N*C_in*H*W");
    need(op, Wt.size() == C_out * Kd, "Wt must be (C_out, C_in/groups*kH*kW)");
    need(op, offset.size() >= static_cast<long long>(N) * deform_groups * 2 * kk * P,
         "offset is too small");
    need(op, !opt(mask) || mask->size() >= static_cast<long long>(N) * deform_groups * kk * P,
         "mask is too small");
    need(op, !opt(bias) || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * P;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(N, static_cast<int>(out_cols), dt);
    }
    if (N == 0) return;

    const std::size_t es = esize(dt);
    long long pcn = std::max<long long>(1, (32ll << 20) / Kd);
    if (pcn < P) pcn = std::max<long long>(64, (pcn / 64) * 64);
    pcn = std::min<long long>(pcn, P);
    Tensor col = Tensor::empty_on(X.device, static_cast<int>(Kd), static_cast<int>(pcn), dt);
    Tensor y_slab;
    if (pcn < P) {
        y_slab = Tensor::empty_on(X.device, Cgo, static_cast<int>(pcn), dt);
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());

    for (int n = 0; n < N; ++n) {
        const std::size_t off_n_bytes =
            static_cast<std::size_t>(n) * deform_groups * 2 * kk * P * es;
        const std::size_t mask_n_bytes =
            static_cast<std::size_t>(n) * deform_groups * kk * P * es;
        const std::uintptr_t x_ptr =
            addr(byte_off(X.data, static_cast<std::size_t>(n) * C_in * H * W * es));
        const std::uintptr_t off_ptr = addr(byte_off(offset.data, off_n_bytes));
        const std::uintptr_t mask_ptr =
            opt(mask) ? addr(byte_off(mask->data, mask_n_bytes)) : 0;

        for (int g = 0; g < groups; ++g) {
            const Tensor wg = Tensor::view(
                Wt.device,
                const_cast<void*>(byte_off(Wt.data, static_cast<std::size_t>(g) * Cgo * Kd * es)),
                Cgo, static_cast<int>(Kd), dt);
            for (int p0 = 0; p0 < P; p0 += static_cast<int>(pcn)) {
                const int np = static_cast<int>(std::min<long long>(pcn, P - p0));
                const std::uintptr_t ptrs[] = {x_ptr, off_ptr, mask_ptr, addr(col.data)};
                const std::int32_t ints[] = {
                    i32(Cgi), i32(H), i32(W), i32(Wo), i32(P), i32(kH), i32(kW),
                    static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(stride_h) | (static_cast<std::uint32_t>(stride_w) << 16)),
                    static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(pad_h) | (static_cast<std::uint32_t>(pad_w) << 16)),
                    static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(dil_h) | (static_cast<std::uint32_t>(dil_w) << 16)),
                    i32(C_in / deform_groups), i32(g * Cgi), i32(p0), i32(np)};
                const std::uint8_t flags = static_cast<std::uint8_t>(1u | (mask_ptr != 0 ? 2u : 0u));
                run_vision(MI_DEFORM_COL, dt, flags, static_cast<std::uint64_t>(Kd) * np,
                           false, ptrs, ints, {});

                const Tensor col_v = Tensor::view(X.device, col.data, static_cast<int>(Kd), np, dt);
                void* y_ng_p0 = byte_off(
                    Y.data, ((static_cast<std::size_t>(n) * C_out + static_cast<std::size_t>(g) * Cgo) * P +
                             static_cast<std::size_t>(p0)) * es);
                if (np == P) {
                    Tensor y_v = Tensor::view(Y.device, y_ng_p0, Cgo, P, dt);
                    matmul(wg, col_v, y_v);
                } else {
                    Tensor y_sub = Tensor::view(Y.device, y_slab.data, Cgo, np, dt);
                    matmul(wg, col_v, y_sub);
                    BROTENSOR_CUDA_CHECK(cudaMemcpy2DAsync(
                        y_ng_p0, static_cast<std::size_t>(P) * es,
                        y_sub.data, static_cast<std::size_t>(np) * es,
                        static_cast<std::size_t>(np) * es, static_cast<std::size_t>(Cgo),
                        cudaMemcpyDeviceToDevice, stream));
                }
            }
        }
    }
    if (opt(bias)) {
        for (int n = 0; n < N; ++n) {
            Tensor yn = Tensor::view(
                Y.device, byte_off(Y.data, static_cast<std::size_t>(n) * out_cols * es),
                1, static_cast<int>(out_cols), dt);
            add_channel_bias_inplace(yn, *bias, C_out, P);
        }
    }
}

void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW,
                              int pad_h, int pad_w, bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y) {
    constexpr const char* op = "modulated_conv2d_forward";
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    need(op, W.dtype == dt && s.dtype == dt, "W/s dtype must match X");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 &&
                 pad_h >= 0 && pad_w >= 0 && N <= 65535,
         "bad dimension");
    need(op, X.rows == N && X.cols == static_cast<long long>(C_in) * H * Wd, "X shape mismatch");
    need(op, W.rows == C_out && W.cols == wk, "W shape mismatch");
    need(op, s.rows == N && s.cols == C_in, "s shape mismatch");
    const int Ho = H + 2 * pad_h - (kH - 1), Wo = Wd + 2 * pad_w - (kW - 1);
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    if (dcoef.rows != N || dcoef.cols != C_out || dcoef.dtype != Dtype::FP32) {
        dcoef.resize(N, C_out, Dtype::FP32);
    }
    const long long out_cols = static_cast<long long>(C_out) * Ho * Wo;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(N, static_cast<int>(out_cols), dt);
    }
    if (N == 0) return;

    Tensor Wn = Tensor::empty_on(X.device, N * C_out, static_cast<int>(wk), dt);
    const std::uintptr_t ptrs[] = {addr(W.data), addr(s.data), addr(Wn.data), addr(dcoef.data)};
    const std::int32_t ints[] = {i32(N), i32(C_out), i32(C_in), i32(kH * kW)};
    const float flts[] = {eps};
    run_vision(MI_MODW, dt, static_cast<std::uint8_t>(demodulate ? 1u : 0u),
               static_cast<std::uint64_t>(N) * C_out, true, ptrs, ints, flts);

    const std::size_t es = esize(dt);
    const std::size_t x_stride = static_cast<std::size_t>(C_in) * H * Wd * es;
    const std::size_t w_stride = static_cast<std::size_t>(C_out) * wk * es;
    const std::size_t y_stride = static_cast<std::size_t>(out_cols) * es;
    for (int n = 0; n < N; ++n) {
        const Tensor Xn = Tensor::view(
            X.device, const_cast<void*>(byte_off(X.data, static_cast<std::size_t>(n) * x_stride)),
            1, C_in * H * Wd, dt);
        const Tensor Wn_n = Tensor::view(
            Wn.device, byte_off(Wn.data, static_cast<std::size_t>(n) * w_stride),
            C_out, static_cast<int>(wk), dt);
        Tensor Yn = Tensor::view(
            Y.device, byte_off(Y.data, static_cast<std::size_t>(n) * y_stride),
            1, static_cast<int>(out_cols), dt);
        conv2d_forward(Xn, Wn_n, nullptr, 1, C_in, H, Wd, C_out, kH, kW,
                       1, 1, pad_h, pad_w, 1, 1, 1, Yn);
    }
}

void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s,
                               const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW,
                               int pad_h, int pad_w, bool demodulate, float eps,
                               Tensor& dX, Tensor& dW, Tensor& ds) {
    constexpr const char* op = "modulated_conv2d_backward";
    (void)eps;
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    need(op, W.dtype == dt && s.dtype == dt && dY.dtype == dt, "W/s/dY dtype must match X");
    need(op, dcoef.dtype == Dtype::FP32, "dcoef must be FP32");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 &&
                 pad_h >= 0 && pad_w >= 0 && N <= 65535,
         "bad dimension");
    need(op, W.rows == C_out && W.cols == wk, "W shape mismatch");
    need(op, s.rows == N && s.cols == C_in, "s shape mismatch");
    need(op, dcoef.rows == N && dcoef.cols == C_out, "dcoef must be (N, C_out)");
    const int Ho = H + 2 * pad_h - (kH - 1), Wo = Wd + 2 * pad_w - (kW - 1);
    need(op, Ho > 0 && Wo > 0, "non-positive output shape");
    need(op, X.size() >= static_cast<long long>(N) * C_in * H * Wd, "X shape mismatch");
    need(op, dY.size() >= static_cast<long long>(N) * C_out * Ho * Wo, "dY shape mismatch");
    const bool want_dW = dW.data != nullptr;
    if (want_dW) {
        need(op, dW.dtype == dt && dW.rows == C_out && dW.cols == wk,
             "dW must be (C_out, C_in kH kW) in X's dtype");
    }
    if (dX.rows != N || dX.cols != static_cast<long long>(C_in) * H * Wd || dX.dtype != dt) {
        dX.resize(N, static_cast<int>(static_cast<long long>(C_in) * H * Wd), dt);
    }
    if (ds.rows != N || ds.cols != C_in || ds.dtype != dt) {
        ds.resize(N, C_in, dt);
    }
    if (N == 0) return;

    Tensor Wn = Tensor::empty_on(X.device, N * C_out, static_cast<int>(wk), dt);
    const std::uintptr_t modw_ptrs[] = {addr(W.data), addr(s.data), addr(Wn.data), addr(dcoef.data)};
    const std::int32_t mod_ints[] = {i32(N), i32(C_out), i32(C_in), i32(kH * kW)};
    const float modw_flts[] = {0.0f};
    const std::uint8_t modw_flags = static_cast<std::uint8_t>((demodulate ? 1u : 0u) | 2u);
    run_vision(MI_MODW, dt, modw_flags, static_cast<std::uint64_t>(N) * C_out,
               true, modw_ptrs, mod_ints, modw_flts);

    Tensor dwp = Tensor::zeros_on(X.device, N * C_out, static_cast<int>(wk), Dtype::FP32);
    Tensor dWpp_tmp;
    if (dt != Dtype::FP32) {
        dWpp_tmp = Tensor::empty_on(X.device, C_out, static_cast<int>(wk), dt);
    }
    const std::size_t es = esize(dt);
    const std::size_t x_stride = static_cast<std::size_t>(C_in) * H * Wd * es;
    const std::size_t dy_stride = static_cast<std::size_t>(C_out) * Ho * Wo * es;
    const std::size_t w_stride = static_cast<std::size_t>(C_out) * wk * es;
    const std::size_t dwp_stride = static_cast<std::size_t>(C_out) * wk * sizeof(float);
    const std::uint32_t total_w = static_cast<std::uint32_t>(C_out * wk);

    for (int n = 0; n < N; ++n) {
        const Tensor Xn = Tensor::view(
            X.device, const_cast<void*>(byte_off(X.data, static_cast<std::size_t>(n) * x_stride)),
            1, C_in * H * Wd, dt);
        const Tensor dYn = Tensor::view(
            dY.device, const_cast<void*>(byte_off(dY.data, static_cast<std::size_t>(n) * dy_stride)),
            1, C_out * Ho * Wo, dt);
        const Tensor Wn_n = Tensor::view(
            Wn.device, byte_off(Wn.data, static_cast<std::size_t>(n) * w_stride),
            C_out, static_cast<int>(wk), dt);
        Tensor dXn = Tensor::view(
            dX.device, byte_off(dX.data, static_cast<std::size_t>(n) * x_stride),
            1, C_in * H * Wd, dt);
        conv2d_backward_input(Wn_n, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                              1, 1, pad_h, pad_w, 1, 1, 1, dXn);

        Tensor dwp_n = Tensor::view(
            dwp.device, byte_off(dwp.data, static_cast<std::size_t>(n) * dwp_stride),
            C_out, static_cast<int>(wk), Dtype::FP32);
        if (dt == Dtype::FP32) {
            conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                                   1, 1, pad_h, pad_w, 1, 1, 1, dwp_n);
        } else {
            dWpp_tmp.zero();
            conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                                   1, 1, pad_h, pad_w, 1, 1, 1, dWpp_tmp);
            const int lanes = mir_vec_lanes_mixed(
                dt, Dtype::FP32, total_w, addr(dWpp_tmp.data), addr(dwp_n.data));
            CUfunction fn = mir_function(
                MirKey{MirKind::Cast, 0, dt, Dtype::FP32, static_cast<std::uint8_t>(lanes)});
            mir_dispatch_cast(fn, mir_groups_1d(total_w, lanes), dWpp_tmp.data, dwp_n.data, total_w);
        }
    }

    if (demodulate) {
        const std::uintptr_t bwd_ptrs[] = {addr(W.data), addr(s.data), addr(dwp.data), addr(dcoef.data)};
        run_vision(MI_MODW_BWD, dt, 0, static_cast<std::uint64_t>(N) * C_out,
                   true, bwd_ptrs, mod_ints, {});
    }
    const std::uintptr_t ds_ptrs[] = {addr(W.data), addr(dwp.data), addr(ds.data)};
    run_vision(MI_MODW_DS, dt, 0, static_cast<std::uint64_t>(N) * C_in,
               false, ds_ptrs, mod_ints, {});
    if (want_dW) {
        const std::uintptr_t dw_ptrs[] = {addr(s.data), addr(dwp.data), addr(dW.data)};
        run_vision(MI_MODW_DW, dt, 0, static_cast<std::uint64_t>(C_out) * wk,
                   false, dw_ptrs, mod_ints, {});
    }
}

void fill_cuda_vtable_deform_conv(::brotensor::detail::OpsVTable& v) {
    v.deform_conv2d_forward = &deform_conv2d_forward;
}

void fill_cuda_vtable_modulated_conv2d(::brotensor::detail::OpsVTable& v) {
    v.modulated_conv2d_forward = &modulated_conv2d_forward;
    v.modulated_conv2d_backward = &modulated_conv2d_backward;
}

}  // namespace brotensor::detail::cuda
