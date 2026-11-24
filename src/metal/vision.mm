// Metal Wave 6 vision ops via single-source brass MIR kernels
// (src/kernels/mir_vision.cpp -> mtl_mir_cache.mm) + Metal simdgroup GEMM:
//   deform_conv2d_forward        (MI_DEFORM_COL + GEMM + channel bias)
//   modulated_conv2d_forward     (MI_MODW + grouped conv2d)
//   modulated_conv2d_backward    (MI_MODW + conv2d_bwd_input + im2col GEMM + MI_MODW_{BWD,DS,DW})
//   conv_transpose2d_gemm        (batched GEMM + MI_COL2IM2D)

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "fp16_matmul.h"
#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L);
void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, Tensor& Y);
void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                           int N, int C_in, int H, int W, int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, Tensor& dX);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(const char* op, Dtype dt) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

std::int32_t i32(long long v) { return static_cast<std::int32_t>(v); }
std::size_t esize(Dtype dt) { return dt == Dtype::FP32 ? 4u : 2u; }

const void* byte_off(const void* p, std::size_t bytes) {
    return static_cast<const char*>(p) + bytes;
}

void* byte_off_mut(void* p, std::size_t bytes) {
    return static_cast<char*>(p) + bytes;
}

const void* opt_ptr(const Tensor* t) {
    return (t && t->data && t->size() > 0) ? t->data : nullptr;
}

void run_vision(const char* op, std::uint32_t code, Dtype dt, std::uint8_t flags,
                std::uint64_t items, bool per_group,
                std::span<const void* const> ptrs,
                std::span<const std::int32_t> ints,
                std::span<const float> flts) {
    if (items == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Vision, code, dt, dt, 1, dt, flags});
    const std::uint32_t gx = per_group
        ? static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535u))
        : mir_groups_1d(static_cast<std::uint32_t>(items), 1);
    mir_dispatch_args(mk, gx, op, ptrs, ints, flts);
}

int pixel_slab(long long rows, int P) {
    long long pcn = std::max<long long>(1, (32ll << 20) / std::max<long long>(rows, 1));
    if (pcn < P) pcn = std::max<long long>(64, pcn / 64 * 64);
    return static_cast<int>(std::min<long long>(pcn, P));
}

void im2col_mir(const char* op, Dtype dt, const void* x, void* col,
                int Cgi, int H, int W, int Ho, int Wo, int kH, int kW,
                int sh, int sw, int ph, int pw, int dh, int dw,
                int ic_base, int p0, int np) {
    const void* ptrs[] = {x, nullptr, nullptr, col};
    const std::int32_t ints[] = {
        i32(Cgi), i32(H), i32(W), i32(Wo), i32(static_cast<long long>(Ho) * Wo),
        i32(kH), i32(kW),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(sh) | (static_cast<std::uint32_t>(sw) << 16)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(ph) | (static_cast<std::uint32_t>(pw) << 16)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(dh) | (static_cast<std::uint32_t>(dw) << 16)),
        1, i32(ic_base), i32(p0), i32(np),
    };
    run_vision(op, MI_DEFORM_COL, dt, 0u,
               static_cast<std::uint64_t>(Cgi) * kH * kW * np, false,
               ptrs, ints, {});
}

// Per-image weight gradient: dw_img (C_out, Kd) (+)= dY_img @ col^T, slabbed.
void weight_grad_mir(const char* op, Dtype dt, Dtype dc,
                     const Tensor& X, const Tensor& dY, Tensor& dW_out,
                     int n_img, std::size_t x_img_stride, std::size_t dy_img_stride,
                     std::size_t dw_img_stride, int C_in, int H, int W, int C_out,
                     int kH, int kW, int sh, int sw, int ph, int pw,
                     int dh, int dwl, int groups, bool accumulate) {
    const int Ho = (H + 2 * ph - dh * (kH - 1) - 1) / sh + 1;
    const int Wo = (W + 2 * pw - dwl * (kW - 1) - 1) / sw + 1;
    const int P = Ho * Wo, Cgi = C_in / groups, Cgo = C_out / groups;
    const long long Kd = static_cast<long long>(Cgi) * kH * kW;
    const int np_max = pixel_slab(Kd, P);
    const std::size_t es = esize(dt), ces = esize(dc);
    Tensor col = Tensor::empty_on(Device::Metal, static_cast<int>(Kd), np_max, dt);

    id<MTLBuffer> bDY = buffer_for(dY);
    const NSUInteger oDY = buffer_offset_for(dY);
    id<MTLBuffer> bCol = buffer_for(col);
    const NSUInteger oCol = buffer_offset_for(col);
    id<MTLBuffer> bDW = buffer_for(dW_out);
    const NSUInteger oDW = buffer_offset_for(dW_out);

    for (int n = 0; n < n_img; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int p0 = 0; p0 < P; p0 += np_max) {
                const int np = std::min(np_max, P - p0);
                im2col_mir(op, dt, byte_off(X.data, static_cast<std::size_t>(n) * x_img_stride),
                           col.data, Cgi, H, W, Ho, Wo, kH, kW, sh, sw, ph, pw,
                           dh, dwl, g * Cgi, p0, np);
                metal_impl::AbtMixed gm;
                gm.A = bDY;
                gm.ofs_A = oDY + static_cast<NSUInteger>(
                    static_cast<std::size_t>(n) * dy_img_stride +
                    (static_cast<std::size_t>(g) * Cgo * P + p0) * es);
                gm.lda = static_cast<std::uint64_t>(P);
                gm.B = bCol;
                gm.ofs_B = oCol;
                gm.ldb = static_cast<std::uint64_t>(np);
                gm.C = bDW;
                gm.ofs_C = oDW + static_cast<NSUInteger>(
                    static_cast<std::size_t>(n) * dw_img_stride +
                    static_cast<std::size_t>(g) * Cgo * Kd * ces);
                gm.ldc = static_cast<std::uint64_t>(Kd);
                gm.M = Cgo;
                gm.N = static_cast<int>(Kd);
                gm.K = np;
                gm.in = metal_impl::abt_type(dt);
                gm.out = metal_impl::abt_type(dc);
                gm.epilogue = (accumulate || p0 > 0)
                    ? metal_impl::kAbtAccumulate
                    : metal_impl::kAbtStore;
                metal_impl::launch_matmul_abt_mixed(gm);
            }
        }
    }
}

}  // namespace

// ─── Deformable convolution ────────────────────────────────────────────────

void deform_conv2d_forward(const Tensor& X, const Tensor& offset,
                           const Tensor* mask, const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dil_h, int dil_w,
                           int groups, int deform_groups, Tensor& Y) {
    constexpr const char* op = "deform_conv2d_forward";
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    need(op, Wt.dtype == dt && offset.dtype == dt &&
             (!opt_ptr(mask) || mask->dtype == dt) &&
             (!opt_ptr(bias) || bias->dtype == dt),
         "X, offset, mask, Wt, bias dtype must match");
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must divide C_in and C_out");
    need(op, deform_groups >= 1 && C_in % deform_groups == 0,
         "deform_groups must divide C_in");
    need(op, N >= 0 && H > 0 && W > 0 && kH > 0 && kW > 0 &&
             stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0 &&
             pad_h >= 0 && pad_w >= 0, "bad geometry");
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
    need(op, !opt_ptr(mask) ||
             mask->size() >= static_cast<long long>(N) * deform_groups * kk * P,
         "mask is too small");
    need(op, !opt_ptr(bias) || bias->size() == C_out, "bias must have C_out elements");
    const long long out_cols = static_cast<long long>(C_out) * P;
    need(op, out_cols <= 0x7fffffffLL, "output too large");
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(N, static_cast<int>(out_cols), dt);
    }
    if (N == 0) return;

    const std::size_t es = esize(dt);
    const int pcn = pixel_slab(Kd, P);
    Tensor col = Tensor::empty_on(Device::Metal, static_cast<int>(Kd), pcn, dt);

    id<MTLBuffer> bWt = buffer_for(Wt);
    const NSUInteger oWt = buffer_offset_for(Wt);
    id<MTLBuffer> bCol = buffer_for(col);
    const NSUInteger oCol = buffer_offset_for(col);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oY = buffer_offset_for(Y);
    const bool has_mask = opt_ptr(mask) != nullptr;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int p0 = 0; p0 < P; p0 += pcn) {
                const int np = std::min(pcn, P - p0);
                const void* x_p = byte_off(X.data, static_cast<std::size_t>(n) * C_in * H * W * es);
                const void* off_p = byte_off(offset.data, static_cast<std::size_t>(n) * deform_groups * 2 * kk * P * es);
                const void* mask_p = has_mask
                    ? byte_off(mask->data, static_cast<std::size_t>(n) * deform_groups * kk * P * es)
                    : nullptr;
                const void* ptrs[] = {x_p, off_p, mask_p, col.data};
                const std::int32_t ints[] = {
                    i32(Cgi), i32(H), i32(W), i32(Wo), i32(P), i32(kH), i32(kW),
                    static_cast<std::int32_t>(static_cast<std::uint32_t>(stride_h) | (static_cast<std::uint32_t>(stride_w) << 16)),
                    static_cast<std::int32_t>(static_cast<std::uint32_t>(pad_h) | (static_cast<std::uint32_t>(pad_w) << 16)),
                    static_cast<std::int32_t>(static_cast<std::uint32_t>(dil_h) | (static_cast<std::uint32_t>(dil_w) << 16)),
                    i32(C_in / deform_groups), i32(g * Cgi), i32(p0), i32(np),
                };
                const std::uint8_t flags = static_cast<std::uint8_t>(1u | (has_mask ? 2u : 0u));
                run_vision(op, MI_DEFORM_COL, dt, flags,
                           static_cast<std::uint64_t>(Kd) * np, false,
                           ptrs, ints, {});
                metal_impl::AbtMixed gm;
                gm.A = bWt;
                gm.ofs_A = oWt + static_cast<NSUInteger>(static_cast<std::size_t>(g) * Cgo * Kd * es);
                gm.lda = static_cast<std::uint64_t>(Kd);
                gm.B = bCol;
                gm.ofs_B = oCol;
                gm.ldb = static_cast<std::uint64_t>(np);
                gm.transB = true;
                gm.C = bY;
                gm.ofs_C = oY + static_cast<NSUInteger>(
                    ((static_cast<std::size_t>(n) * C_out + static_cast<std::size_t>(g) * Cgo) * P + p0) * es);
                gm.ldc = static_cast<std::uint64_t>(P);
                gm.M = Cgo;
                gm.N = np;
                gm.K = static_cast<int>(Kd);
                gm.in = gm.out = metal_impl::abt_type(dt);
                metal_impl::launch_matmul_abt_mixed(gm);
            }
        }
    }
    if (opt_ptr(bias)) {
        for (int n = 0; n < N; ++n) {
            Tensor yn = Tensor::view(Device::Metal,
                                     byte_off_mut(Y.data, static_cast<std::size_t>(n) * out_cols * es),
                                     1, static_cast<int>(out_cols), dt);
            add_channel_bias_inplace(yn, *bias, C_out, P);
        }
    }
}

// ─── StyleGAN3 modulated convolution ───────────────────────────────────────

void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s,
                              int N, int C_in, int H, int Wd, int C_out,
                              int kH, int kW, int pad_h, int pad_w,
                              bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y) {
    constexpr const char* op = "modulated_conv2d_forward";
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    need(op, W.dtype == dt && s.dtype == dt, "W/s dtype must match X");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 &&
             pad_h >= 0 && pad_w >= 0, "bad dimension");
    need(op, X.rows == N && X.cols == static_cast<long long>(C_in) * H * Wd,
         "X shape mismatch");
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
    need(op, N <= 65535, "N above 65535");

    Tensor Wn = Tensor::empty_on(Device::Metal, N * C_out, static_cast<int>(wk), dt);
    const void* ptrs[] = {W.data, s.data, Wn.data, dcoef.data};
    const std::int32_t ints[] = {i32(N), i32(C_out), i32(C_in), i32(kH * kW)};
    const float flts[] = {eps};
    run_vision(op, MI_MODW, dt, static_cast<std::uint8_t>(demodulate ? 1u : 0u),
               static_cast<std::uint64_t>(N) * C_out, true, ptrs, ints, flts);

    const Tensor Xv = Tensor::view(Device::Metal, const_cast<void*>(X.data), 1,
                                   static_cast<int>(static_cast<long long>(N) * C_in * H * Wd), dt);
    Tensor Yv = Tensor::view(Device::Metal, Y.data, 1,
                             static_cast<int>(static_cast<long long>(N) * out_cols), dt);
    conv2d_forward(Xv, Wn, nullptr, 1, N * C_in, H, Wd, N * C_out, kH, kW,
                   1, 1, pad_h, pad_w, 1, 1, N, Yv);
}

void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s,
                               const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd, int C_out,
                               int kH, int kW, int pad_h, int pad_w,
                               bool demodulate, float eps,
                               Tensor& dX, Tensor& dW, Tensor& ds) {
    constexpr const char* op = "modulated_conv2d_backward";
    (void)eps;
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    need(op, W.dtype == dt && s.dtype == dt && dY.dtype == dt,
         "W/s/dY dtype must match X");
    need(op, dcoef.dtype == Dtype::FP32, "dcoef must be FP32");
    const long long wk = static_cast<long long>(C_in) * kH * kW;
    need(op, N >= 0 && C_in > 0 && C_out > 0 && kH > 0 && kW > 0 &&
             pad_h >= 0 && pad_w >= 0 && N <= 65535, "bad dimension");
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

    const std::size_t es = esize(dt);
    Tensor Wn = Tensor::empty_on(Device::Metal, N * C_out, static_cast<int>(wk), dt);
    const void* modw_ptrs[] = {W.data, s.data, Wn.data, dcoef.data};
    const std::int32_t mod_ints[] = {i32(N), i32(C_out), i32(C_in), i32(kH * kW)};
    const float modw_flts[] = {0.0f};
    const std::uint8_t modw_flags = static_cast<std::uint8_t>((demodulate ? 1u : 0u) | 2u);
    run_vision(op, MI_MODW, dt, modw_flags,
               static_cast<std::uint64_t>(N) * C_out, true,
               modw_ptrs, mod_ints, modw_flts);

    Tensor dXv = Tensor::view(Device::Metal, dX.data, 1,
                              static_cast<int>(static_cast<long long>(N) * C_in * H * Wd), dt);
    const Tensor dYv = Tensor::view(Device::Metal, const_cast<void*>(dY.data), 1,
                                    static_cast<int>(static_cast<long long>(N) * C_out * Ho * Wo), dt);
    conv2d_backward_input(Wn, dYv, 1, N * C_in, H, Wd, N * C_out, kH, kW,
                          1, 1, pad_h, pad_w, 1, 1, N, dXv);

    Tensor dwp = Tensor::empty_on(Device::Metal, N * C_out, static_cast<int>(wk), Dtype::FP32);
    weight_grad_mir(op, dt, Dtype::FP32, X, dY, dwp, N,
                    static_cast<std::size_t>(C_in) * H * Wd * es,
                    static_cast<std::size_t>(C_out) * Ho * Wo * es,
                    static_cast<std::size_t>(C_out) * wk * 4u,
                    C_in, H, Wd, C_out, kH, kW, 1, 1, pad_h, pad_w, 1, 1, 1, false);
    if (demodulate) {
        const void* bwd_ptrs[] = {W.data, s.data, dwp.data, dcoef.data};
        run_vision(op, MI_MODW_BWD, dt, 0u,
                   static_cast<std::uint64_t>(N) * C_out, true,
                   bwd_ptrs, mod_ints, {});
    }
    const void* ds_ptrs[] = {W.data, dwp.data, ds.data};
    run_vision(op, MI_MODW_DS, dt, 0u,
               static_cast<std::uint64_t>(N) * C_in, false,
               ds_ptrs, mod_ints, {});
    if (want_dW) {
        const void* dw_ptrs[] = {s.data, dwp.data, dW.data};
        run_vision(op, MI_MODW_DW, dt, 0u,
                   static_cast<std::uint64_t>(C_out) * wk, false,
                   dw_ptrs, mod_ints, {});
    }
}

// ─── conv_transpose2d GEMM + overlap-add (MI_COL2IM2D) ─────────────────────

bool conv_transpose2d_gemm(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int sh, int sw, int ph, int pw,
                           int dh, int dw, int groups, int Ho, int Wo, Tensor& Y) {
    constexpr const char* op = "conv_transpose2d_forward";
    const Dtype dt = X.dtype;
    const Dtype cdt = (dt == Dtype::BF16) ? Dtype::FP32 : dt;
    const std::size_t es = esize(dt), ces = esize(cdt);
    const int kk = kH * kW, HW = H * W;
    const int Cgi = C_in / groups, Cgo = C_out / groups, M = Cgo * kk;
    const std::uint64_t per_image = static_cast<std::uint64_t>(C_out) * kk * HW;
    constexpr std::uint64_t kBudget = 256ull << 20;
    if (per_image * ces > kBudget || per_image > 0x7fffffffull) return false;
    const int nb = static_cast<int>(
        std::max<std::uint64_t>(1, std::min<std::uint64_t>(N, kBudget / std::max<std::uint64_t>(per_image * ces, 1u))));
    Tensor cols = Tensor::empty_on(Device::Metal, nb, static_cast<int>(per_image), cdt);

    id<MTLBuffer> bWt = buffer_for(Wt);
    const NSUInteger oWt = buffer_offset_for(Wt);
    id<MTLBuffer> bX = buffer_for(X);
    const NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bCols = buffer_for(cols);
    const NSUInteger oCols = buffer_offset_for(cols);
    const void* bias_p = opt_ptr(bias);

    for (int n0 = 0; n0 < N; n0 += nb) {
        const int nn = std::min(nb, N - n0);
        for (int g = 0; g < groups; ++g) {
            metal_impl::AbtMixed gm;
            gm.A = bWt;
            gm.ofs_A = oWt + static_cast<NSUInteger>(static_cast<std::size_t>(g) * Cgi * M * es);
            gm.lda = static_cast<std::uint64_t>(M);
            gm.transA = true;
            gm.B = bX;
            gm.ofs_B = oX + static_cast<NSUInteger>(
                (static_cast<std::size_t>(n0) * C_in + static_cast<std::size_t>(g) * Cgi) * HW * es);
            gm.ldb = static_cast<std::uint64_t>(HW);
            gm.transB = true;
            gm.C = bCols;
            gm.ofs_C = oCols + static_cast<NSUInteger>(static_cast<std::size_t>(g) * M * HW * ces);
            gm.ldc = static_cast<std::uint64_t>(HW);
            gm.M = M;
            gm.N = HW;
            gm.K = Cgi;
            gm.in = metal_impl::abt_type(dt);
            gm.out = metal_impl::abt_type(cdt);
            gm.batch = nn;
            gm.strideA = 0;
            gm.strideB = static_cast<std::uint64_t>(C_in) * HW;
            gm.strideC = per_image;
            metal_impl::launch_matmul_abt_mixed(gm);
        }
        const void* ptrs[] = {
            cols.data, bias_p,
            byte_off_mut(Y.data, static_cast<std::size_t>(n0) * C_out * Ho * Wo * es),
        };
        const std::int32_t ints[] = {
            i32(nn), i32(C_out), i32(H), i32(W), i32(kH), i32(kW),
            i32(sh), i32(sw), i32(ph), i32(pw), i32(dh), i32(dw),
            i32(Ho), i32(Wo),
        };
        const std::uint8_t flags = static_cast<std::uint8_t>(
            (bias_p != nullptr ? 1u : 0u) | (cdt != dt ? 2u : 0u));
        run_vision(op, MI_COL2IM2D, dt, flags,
                   static_cast<std::uint64_t>(nn) * C_out * Ho * Wo, false,
                   ptrs, ints, {});
    }
    return true;
}

}  // namespace brotensor::detail::metal
