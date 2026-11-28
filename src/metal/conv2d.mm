// Metal 2D convolution backed 100% by single-source Brass MIR kernels:
//   conv2d_forward         — FP16 MirKind::ConvCm (simdgroup_matrix implicit-GEMM),
//                            pointwise 1x1 GEMM fast path, MirKind::ConvSimt,
//                            else MirKind::ConvDirect (mode 0)
//   conv2d_backward_input  — pointwise 1x1 GEMM fast path, else MirKind::ConvDirect (mode 1)
//   conv2d_backward_weight — pointwise 1x1 split-K GEMM + MirKind::ConvAux fold (op 7),
//                            else MirKind::ConvAux bwd_weight (ops 0/1)
//   conv2d_backward_bias   — MirKind::Gnorm (GN_BIAS_GRAD)

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

namespace {

std::uint64_t cdiv_u64(std::uint64_t a, std::uint64_t b) { return (a + b - 1u) / b; }

bool conv2d_is_pointwise(int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int groups) {
    return kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 &&
           pad_h == 0 && pad_w == 0 && groups == 1;
}

bool conv2d_cm_eligible(int C_in, int C_out, int H_out, int W_out,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w, int dil_h, int dil_w, int groups, int N) {
    const int P = H_out * W_out;
    const int M = C_out / groups;
    const int K = (C_in / groups) * kH * kW;
    if (M < 16 || K < 16 || static_cast<std::size_t>(M) * static_cast<std::size_t>(P) < 1024u) {
        return false;
    }
    if (kH > 255 || kW > 255 || stride_h > 255 || stride_w > 255 ||
        pad_h > 255 || pad_w > 255 || dil_h > 255 || dil_w > 255) {
        return false;
    }
    if (P % 8 != 0) return false;
    if (cdiv_u64(P, 64u) > 65535u || cdiv_u64(M, 64u) > 65535u) return false;
    (void)N;
    return true;
}

void run_conv_cm(const void* x, const void* w, const void* y, const void* bias,
                 int n, int cin, int h, int wd, int cout, int ho, int wo,
                 int kh, int kw, int sh, int sw, int ph, int pw,
                 int dh, int dw, int groups, bool accum = false) {
    const std::uint32_t P = static_cast<std::uint32_t>(ho * wo);
    const std::uint32_t M = static_cast<std::uint32_t>(cout / groups);
    const std::uint32_t K = static_cast<std::uint32_t>((cin / groups) * kh * kw);
    // Use 128x64 (cfg_idx=2, ~30 KiB shared) when M fills 128 rows cleanly,
    // else 64x64 (cfg_idx=3, ~20 KiB shared); both stay within Metal's 32 KiB cap.
    const double fill128 = double(M) / double(cdiv_u64(M, 128u) * 128u);
    const std::uint32_t cfg_idx = (M >= 128u && fill128 >= 0.9 &&
                                   cdiv_u64(M, 128u) * cdiv_u64(P, 64u) * std::uint64_t(n) * groups >= 24u)
                                  ? 2u : 3u;
    const MetalCmTile tile = metal_cm_tile(cfg_idx);
    const std::uint32_t bm = tile.bm;
    const std::uint32_t bn = tile.bn;
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(w);
    const bool avec = (w_addr % 16u == 0u) && (K % 8u == 0u);
    const bool rowch = (wo % 8 == 0);
    std::uint32_t bvec = 0u;
    if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 &&
        (h * wd) % 8 == 0 && (x_addr % 16u == 0u)) {
        bvec = 1u;
    } else if (sw == 1 && rowch && wd % 8 == 0 && (x_addr % 16u == 0u)) {
        bvec = 2u;
    }
    const std::uint64_t aux =
        (static_cast<std::uint64_t>(kh & 0xff) << 0) | (static_cast<std::uint64_t>(kw & 0xff) << 8) |
        (static_cast<std::uint64_t>(sh & 0xff) << 16) | (static_cast<std::uint64_t>(sw & 0xff) << 24) |
        (static_cast<std::uint64_t>(ph & 0xff) << 32) | (static_cast<std::uint64_t>(pw & 0xff) << 40) |
        (static_cast<std::uint64_t>(dh & 0xff) << 48) | (static_cast<std::uint64_t>(dw & 0xff) << 56);
    const std::uint32_t op = (cfg_idx & 3u) | ((bvec & 3u) << 8);
    // direct=false uses the shared-scratch epilogue so any edge alignment and bias layout works.
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (avec ? 1u : 0u) | (bias ? 4u : 0u) | (rowch ? 8u : 0u) | (accum ? 16u : 0u));
    const MirKernel mk = mir_kernel({MirKind::ConvCm, op, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, flags, aux});
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv_u64(P, bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv_u64(M, bm));
    const int per = std::max(1, 65535 / groups);
    for (int i0 = 0; i0 < n; i0 += per) {
        const int ni = std::min(per, n - i0);
        const void* ptrs[] = {
            static_cast<const char*>(x) + std::size_t(i0) * cin * h * wd * 2u,
            w,
            static_cast<const char*>(y) + std::size_t(i0) * cout * P * 2u,
            bias,
            nullptr,
        };
        const std::int32_t ints[] = {
            static_cast<std::int32_t>(M), static_cast<std::int32_t>(P), static_cast<std::int32_t>(K),
            h, wd, wo, cin, cout, groups, 0, 0,
        };
        mir_dispatch_args_3d(mk, gx, gy, static_cast<std::uint32_t>(ni * groups),
                             "conv2d_forward", ptrs, ints, {});
    }
}

bool conv2d_simt_eligible(int C_in, int C_out, int H_out, int W_out,
                          int kH, int kW, int stride_h, int stride_w,
                          int pad_h, int pad_w, int dil_h, int dil_w, int groups) {
    const int cgo = C_out / groups;
    const int k = (C_in / groups) * kH * kW;
    if (cgo < 16 || k < 16) return false;
    if (kH > 255 || kW > 255 || stride_h > 255 || stride_w > 255 ||
        pad_h > 255 || pad_w > 255 || dil_h > 255 || dil_w > 255) return false;
    const std::uint64_t P = static_cast<std::uint64_t>(H_out) * static_cast<std::uint64_t>(W_out);
    return (P + 63u) / 64u <= 65535u && (static_cast<std::uint64_t>(cgo) + 63u) / 64u <= 65535u;
}

void run_conv_simt(Dtype dt, const char* op,
                   const void* x, const void* w, const void* y, const void* bias,
                   int n, int cin, int h, int wd, int cout, int ho, int wo,
                   int kh, int kw, int sh, int sw, int ph, int pw,
                   int dh, int dw, int groups, bool accum = false) {
    const std::uint32_t P = static_cast<std::uint32_t>(ho * wo);
    const std::uint32_t M = static_cast<std::uint32_t>(cout / groups);
    const std::uint32_t K = static_cast<std::uint32_t>((cin / groups) * kh * kw);
    const std::uint64_t tiles_big = cdiv_u64(M, 128u) * cdiv_u64(P, 128u) * std::uint64_t(n) * groups;
    const bool big128 = tiles_big >= 160u;
    const std::uint32_t bm = big128 ? 128u : 64u;
    const std::uint32_t bn = big128 ? 128u : 64u;
    const std::uint64_t aux =
        (static_cast<std::uint64_t>(kh & 0xff) << 0) | (static_cast<std::uint64_t>(kw & 0xff) << 8) |
        (static_cast<std::uint64_t>(sh & 0xff) << 16) | (static_cast<std::uint64_t>(sw & 0xff) << 24) |
        (static_cast<std::uint64_t>(ph & 0xff) << 32) | (static_cast<std::uint64_t>(pw & 0xff) << 40) |
        (static_cast<std::uint64_t>(dh & 0xff) << 48) | (static_cast<std::uint64_t>(dw & 0xff) << 56);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (big128 ? 1u : 0u) | (accum ? 2u : 0u) | (bias ? 4u : 0u));
    const MirKernel mk = mir_kernel({MirKind::ConvSimt, 0u, dt, Dtype::FP32, 1, Dtype::FP32, flags, aux});
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv_u64(P, bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv_u64(M, bm));
    const std::size_t es = dtype_size_bytes(dt);
    const int per = std::max(1, 65535 / groups);
    for (int i0 = 0; i0 < n; i0 += per) {
        const int ni = std::min(per, n - i0);
        const void* ptrs[] = {
            static_cast<const char*>(x) + std::size_t(i0) * cin * h * wd * es,
            w,
            static_cast<const char*>(y) + std::size_t(i0) * cout * P * es,
            bias,
        };
        const std::int32_t ints[] = {
            static_cast<std::int32_t>(M), static_cast<std::int32_t>(P), static_cast<std::int32_t>(K),
            h, wd, wo, cin, cout, groups,
        };
        mir_dispatch_args_3d(mk, gx, gy, static_cast<std::uint32_t>(ni * groups), op, ptrs, ints, {});
    }
}

}  // namespace

void conv2d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_forward: X must be FP16, BF16, or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    // Fast path 1: pointwise 1x1 GEMM.
    if (!bias && conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
        const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
        metal_impl::AbtMixed g;
        g.A = buffer_for(Wt); g.ofs_A = buffer_offset_for(Wt); g.lda = static_cast<std::uint64_t>(C_in);
        g.B = buffer_for(X);  g.ofs_B = buffer_offset_for(X);  g.ldb = HW; g.transB = true;
        g.C = buffer_for(Y);  g.ofs_C = buffer_offset_for(Y);  g.ldc = HW;
        g.M = C_out; g.N = static_cast<int>(HW); g.K = C_in;
        g.in = g.out = metal_impl::abt_type(X.dtype);
        g.batch = N;
        g.strideA = 0;
        g.strideB = static_cast<std::uint64_t>(C_in) * HW;
        g.strideC = static_cast<std::uint64_t>(C_out) * HW;
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    // Fast path 2: FP16 simdgroup-matrix implicit-GEMM via MirKind::ConvCm.
    if (X.dtype == Dtype::FP16 &&
        conv2d_cm_eligible(C_in, C_out, H_out, W_out, kH, kW, stride_h, stride_w,
                           pad_h, pad_w, dil_h, dil_w, groups, N)) {
        run_conv_cm(X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
                    N, C_in, H, W, C_out, H_out, W_out,
                    kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
        return;
    }

    // Fast path 3: tiled SIMT implicit-GEMM (64x64 or 128x128) via MirKind::ConvSimt.
    if (conv2d_simt_eligible(C_in, C_out, H_out, W_out, kH, kW, stride_h, stride_w,
                             pad_h, pad_w, dil_h, dil_w, groups)) {
        run_conv_simt(X.dtype, "conv2d_forward",
                      X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
                      N, C_in, H, W, C_out, H_out, W_out,
                      kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
        return;
    }

    mir_dispatch_conv_direct(
        0u, X.dtype, "conv2d_forward",
        X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, H_out, W_out,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_input(const Tensor& Wt,
                           const Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           Tensor& dX) {
    if (Wt.dtype != Dtype::FP16 && Wt.dtype != Dtype::FP32 && Wt.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_input: Wt must be FP16, BF16, or FP32");
    }
    if (dY.dtype != Wt.dtype) {
        throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_input: groups must be >=1 and divide both C_in and C_out");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups)) {
        const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
        metal_impl::AbtMixed g;
        g.A = buffer_for(Wt); g.ofs_A = buffer_offset_for(Wt);
        g.lda = static_cast<std::uint64_t>(C_in); g.transA = true;
        g.B = buffer_for(dY); g.ofs_B = buffer_offset_for(dY); g.ldb = HW; g.transB = true;
        g.C = buffer_for(dX); g.ofs_C = buffer_offset_for(dX); g.ldc = HW;
        g.M = C_in; g.N = static_cast<int>(HW); g.K = C_out;
        g.in = g.out = metal_impl::abt_type(Wt.dtype);
        g.batch = N;
        g.strideA = 0;
        g.strideB = static_cast<std::uint64_t>(C_out) * HW;
        g.strideC = static_cast<std::uint64_t>(C_in) * HW;
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    mir_dispatch_conv_direct(
        1u, dY.dtype, "conv2d_backward_input",
        dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, H_out, W_out, C_in, 1, H, W,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_weight(const Tensor& X,
                            const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            Tensor& dWt) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_weight: X must be FP16, BF16, or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight: X, dY, dWt dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_weight: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    const std::uint32_t total = static_cast<std::uint32_t>(C_out) * Cg_in * kH * kW;
    if (total == 0) return;

    if (conv2d_is_pointwise(kH, kW, stride_h, stride_w, pad_h, pad_w, groups) &&
        (X.dtype == Dtype::FP32 || N == 1)) {
        const std::int64_t HW = static_cast<std::int64_t>(H) * W;
        const std::int64_t tiles = static_cast<std::int64_t>((C_out + 63) / 64) * ((C_in + 63) / 64);
        std::int64_t S = tiles >= 256 ? 1 : (256 + tiles - 1) / tiles;
        S = std::min<std::int64_t>(S, std::max<std::int64_t>(1, HW / 1024));
        const std::int64_t Kc = ((HW + S - 1) / S + 31) / 32 * 32;
        S = (HW + Kc - 1) / Kc;
        const std::int64_t S_full = HW / Kc;
        const std::int64_t K_tail = HW - S_full * Kc;
        const std::uint32_t MN = static_cast<std::uint32_t>(C_out) * static_cast<std::uint32_t>(C_in);
        const metal_impl::AbtType ty = metal_impl::abt_type(X.dtype);
        const std::size_t esz = dtype_size_bytes(X.dtype);
        Tensor part;
        if (S > 1) part = Tensor::empty_on(Device::Metal, static_cast<int>(S), static_cast<int>(MN), Dtype::FP32);
        const MirKernel fold_mk = (S > 1) ? mir_kernel({MirKind::ConvAux, 7u, X.dtype}) : MirKernel{};
        for (int n = 0; n < N; ++n) {
            const NSUInteger oA = buffer_offset_for(dY) +
                static_cast<NSUInteger>(static_cast<std::uint64_t>(n) * C_out * HW * esz);
            const NSUInteger oB = buffer_offset_for(X) +
                static_cast<NSUInteger>(static_cast<std::uint64_t>(n) * C_in * HW * esz);
            metal_impl::AbtMixed g;
            g.A = buffer_for(dY); g.lda = static_cast<std::uint64_t>(HW);
            g.B = buffer_for(X);  g.ldb = static_cast<std::uint64_t>(HW);
            g.ldc = static_cast<std::uint64_t>(C_in);
            g.M = C_out; g.N = C_in;
            g.in = ty;
            if (S == 1) {
                g.ofs_A = oA; g.ofs_B = oB;
                g.C = buffer_for(dWt); g.ofs_C = buffer_offset_for(dWt);
                g.K = static_cast<int>(HW);
                g.out = ty;
                g.epilogue = metal_impl::kAbtAccumulate;
                metal_impl::launch_matmul_abt_mixed(g);
                continue;
            }
            g.C = buffer_for(part);
            g.out = metal_impl::kAbtF32;
            g.epilogue = metal_impl::kAbtStore;
            if (S_full > 0) {
                g.ofs_A = oA; g.ofs_B = oB; g.ofs_C = buffer_offset_for(part);
                g.K = static_cast<int>(Kc);
                g.batch = static_cast<int>(S_full);
                g.strideA = g.strideB = static_cast<std::uint64_t>(Kc);
                g.strideC = MN;
                metal_impl::launch_matmul_abt_mixed(g);
            }
            if (K_tail > 0) {
                const NSUInteger koff = static_cast<NSUInteger>(S_full * Kc * esz);
                g.ofs_A = oA + koff; g.ofs_B = oB + koff;
                g.ofs_C = buffer_offset_for(part) +
                          static_cast<NSUInteger>(static_cast<std::uint64_t>(S_full) * MN * sizeof(float));
                g.K = static_cast<int>(K_tail);
                g.batch = 1;
                g.strideA = g.strideB = g.strideC = 0;
                metal_impl::launch_matmul_abt_mixed(g);
            }
            const void* fptrs[] = {part.data, dWt.data};
            const std::int32_t fints[] = {static_cast<std::int32_t>(MN), static_cast<std::int32_t>(S)};
            const std::uint32_t gx = static_cast<std::uint32_t>((MN + 255u) / 256u);
            mir_dispatch_args_3d(fold_mk, gx, 1u, 1u, "conv2d_splitk_fold", fptrs, fints, {});
        }
        return;
    }

    const bool block_per_elem = (total < 65536u);
    const MirKernel mk = mir_kernel({MirKind::ConvAux, block_per_elem ? 0u : 1u, X.dtype});
    const void* ptrs[] = {X.data, dY.data, dWt.data};
    if (block_per_elem) {
        const std::int32_t ints[] = {
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Cg_in, Cg_out,
        };
        mir_dispatch_args_3d(mk, total, 1u, 1u, "conv2d_backward_weight", ptrs, ints, {});
    } else {
        const std::int32_t ints[] = {
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Cg_in, Cg_out,
            static_cast<std::int32_t>(total),
        };
        const std::uint32_t gx = static_cast<std::uint32_t>((static_cast<std::uint64_t>(total) + 255u) / 256u);
        mir_dispatch_args_3d(mk, gx, 1u, 1u, "conv2d_backward_weight", ptrs, ints, {});
    }
}

void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_bias: dY must be FP16, BF16, or FP32");
    }
    if (dB.dtype != dY.dtype) {
        throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    const MirKernel mk = mir_kernel({MirKind::Gnorm, std::uint32_t(GN_BIAS_GRAD),
                                     dY.dtype, dY.dtype, 1});
    const std::uint32_t uc = static_cast<std::uint32_t>(C_out);
    mir_dispatch_gnorm(mk, std::min<std::uint32_t>(uc, 65535u), (uc + 65534u) / 65535u,
                       "conv2d_backward_bias",
                       dY.data, dB.data, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0u, static_cast<std::uint32_t>(H_out * W_out), 0u, 0u, 0u, 0u,
                       static_cast<std::uint32_t>(N), uc, 0.0f);
}

}  // namespace brotensor::detail::metal
