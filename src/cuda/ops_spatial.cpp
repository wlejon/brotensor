// CUDA NCHW spatial forward, remap, gather/scatter, and embedding forward ops
// backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in ops_scatter.cpp.
void scatter_rows_add(const Tensor& dY, const Tensor& Idx, int R, Tensor& dX);
// Defined in ops_spatial_bwd.cpp.
void interp2d_backward(const Tensor& dY, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode, Tensor& dX);
void adaptive_avg_pool2d_backward(const Tensor& dY, int N, int C, int H, int W, int H_out, int W_out, Tensor& dX);
void max_pool2d_backward(const Tensor& dY, const Tensor& Idx, int N, int C, int H, int W,
                         int H_out, int W_out, Tensor& dX);
void pad2d_backward(const Tensor& dY, int N, int C, int H, int W, int pad_top, int pad_bottom,
                    int pad_left, int pad_right, int mode, Tensor& dX);

namespace {

constexpr std::uint32_t kCpt = 16;

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

void ensure(Tensor& t, long long r, long long c, Dtype dt, const char* op) {
    if (r < 0 || c < 0 || r > 0x7fffffffLL || c > 0x7fffffffLL) fail(op, "output too large");
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(static_cast<int>(r), static_cast<int>(c), dt);
}

bool plane_dims(std::uint64_t plane, std::uint64_t channels, std::uint64_t images,
                std::uint32_t& gx, std::uint32_t& gy, std::uint32_t& gz, const char* op) {
    if (plane == 0 || channels == 0 || images == 0) return false;
    const std::uint64_t gy64 = cdiv(channels, kCpt);
    if (gy64 > 65535 || images > 65535) fail(op, "too many channels or images for one dispatch");
    gx = static_cast<std::uint32_t>(std::min(cdiv(plane, 256), std::uint64_t(65535)));
    gy = static_cast<std::uint32_t>(gy64);
    gz = static_cast<std::uint32_t>(images);
    return true;
}

CUfunction resample_kernel(Dtype dt, std::uint32_t mode, std::uint32_t align, std::uint32_t imode, const char* op) {
    require_fp(dt, op);
    const std::uint8_t flags = static_cast<std::uint8_t>(((imode & 3u) << 1) | (align & 1u));
    return mir_function({MirKind::Resample, mode, dt, dt, 1, Dtype::FP32, flags});
}

CUfunction remap_kernel(Dtype dt, std::uint32_t mode, const char* op) {
    const int es = ::brotensor::dtype_size_bytes(dt);
    if (es != 2 && es != 4) fail(op, "FP32 / FP16 / BF16 / INT32 only");
    return mir_function({MirKind::Remap, mode, Dtype::FP32, Dtype::FP32, static_cast<std::uint8_t>(es)});
}

void check_nchw(const char* op, const Tensor& X, int N, int C, int H, int W) {
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0, "negative dimension");
    need(op, X.size() >= static_cast<long long>(N) * C * H * W, "X is smaller than N*C*H*W");
}

void run_resample(const char* op, const Tensor& X, int N, int C, int hi, int wi, int ho, int wo,
                  std::uint32_t mode, std::uint32_t align, std::uint32_t imode, Tensor& Y) {
    require_fp(X.dtype, op);
    check_nchw(op, X, N, C, hi, wi);
    ensure(Y, N, static_cast<long long>(C) * ho * wo, X.dtype, op);
    std::uint32_t gx = 0, gy = 0, gz = 0;
    if (!plane_dims(std::uint64_t(ho) * wo, C, N, gx, gy, gz, op)) return;
    CUfunction fn = resample_kernel(X.dtype, mode, align, imode, op);
    mir_dispatch_resample(
        fn, gx, gy, gz, X.data, Y.data, nullptr,
        static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(hi), static_cast<std::uint32_t>(wi),
        static_cast<std::uint32_t>(ho), static_cast<std::uint32_t>(wo),
        0, 0, 0, 0, 0, 0);
}

void run_remap(const char* op, const Tensor& X, std::uint32_t mode, int images, int c, int cin,
               int hi, int wi, int ho, int wo, const std::int32_t (&a)[7], Tensor& Y) {
    std::uint32_t gx = 0, gy = 0, gz = 0;
    if (!plane_dims(std::uint64_t(ho) * wo, c, images, gx, gy, gz, op)) return;
    CUfunction fn = remap_kernel(X.dtype, mode, op);
    mir_dispatch_remap(
        fn, gx, gy, gz, X.data, Y.data, nullptr,
        static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(cin),
        static_cast<std::uint32_t>(hi), static_cast<std::uint32_t>(wi),
        static_cast<std::uint32_t>(ho), static_cast<std::uint32_t>(wo),
        a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
}

void check_interp(const char* op, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode) {
    need(op, N >= 0 && C >= 0 && H_in >= 0 && W_in >= 0 && H_out >= 0 && W_out >= 0,
         "N, C, H_in, W_in, H_out, W_out must be non-negative");
    need(op, mode >= 0 && mode <= 3,
         "mode must be 0 (nearest), 1 (bilinear), 2 (bicubic a=-0.5, PIL), or 3 (bicubic a=-0.75, torch)");
    need(op, !((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)),
         "input spatial dims must be > 0 when output spatial dims are > 0");
}

void check_pad_mode(const char* op, int mode) {
    need(op, mode >= 0 && mode <= 2, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
}

void row_gather_dispatch(const Tensor& src_rows, const std::int32_t* d_idx, int M, int C,
                         Tensor& dst, const char* op) {
    CUfunction fn = remap_kernel(src_rows.dtype, RM_GATHER_ROWS, op);
    const std::size_t es = static_cast<std::size_t>(::brotensor::dtype_size_bytes(src_rows.dtype));
    for (int m0 = 0; m0 < M; m0 += 65535) {
        std::uint32_t gx = 0, gy = 0, gz = 0;
        if (!plane_dims(static_cast<std::uint64_t>(C), 1, (std::min)(65535, M - m0), gx, gy, gz, op)) break;
        char* out_ptr = static_cast<char*>(dst.data) + static_cast<std::size_t>(m0) * C * es;
        mir_dispatch_remap(
            fn, gx, gy, gz,
            src_rows.data, out_ptr, d_idx + m0,
            1, 1, 1, static_cast<std::uint32_t>(C), 1, static_cast<std::uint32_t>(C),
            src_rows.rows, 0, 0, 0, 0, 0, 0);
    }
}

}  // namespace

// ─── fixed 2x resamples ────────────────────────────────────────────────────

void upsample_nearest_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    constexpr const char* op = "upsample_nearest_2x";
    require_fp(X.dtype, op);
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, 4LL * C * H * W, X.dtype, op);
    run_remap(op, X, RM_UP2, N, C, C, H, W, 2 * H, 2 * W, {0, 0, 0, 0, 0, 0, 0}, Y);
}

void upsample_bilinear_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    run_resample("upsample_bilinear_2x", X, N, C, H, W, 2 * H, 2 * W, RS_INTERP, 0, 1, Y);
}

void downsample_avg_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    constexpr const char* op = "downsample_avg_2x";
    need(op, H % 2 == 0 && W % 2 == 0, "H and W must be even");
    run_resample(op, X, N, C, H, W, H / 2, W / 2, RS_DOWN_AVG2, 0, 0, Y);
}

void upsample_nearest_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    run_resample("upsample_nearest_2x_backward", dY, N, C, 2 * H, 2 * W, H, W, RS_UP_NEAREST2_BWD, 0, 0, dX);
}

void downsample_avg_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    constexpr const char* op = "downsample_avg_2x_backward";
    need(op, H % 2 == 0 && W % 2 == 0, "H and W must be even");
    run_resample(op, dY, N, C, H / 2, W / 2, H, W, RS_DOWN_AVG2_BWD, 0, 0, dX);
}

// ─── arbitrary-scale interpolation ─────────────────────────────────────────

void interp2d_forward(const Tensor& X, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode, Tensor& Y) {
    constexpr const char* op = "interp2d_forward";
    check_interp(op, N, C, H_in, W_in, H_out, W_out, mode);
    run_resample(op, X, N, C, H_in, W_in, H_out, W_out, RS_INTERP, 0, static_cast<std::uint32_t>(mode), Y);
}

void interp2d_align_corners_forward(const Tensor& X, int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode, Tensor& Y) {
    constexpr const char* op = "interp2d_align_corners_forward";
    check_interp(op, N, C, H_in, W_in, H_out, W_out, mode);
    run_resample(op, X, N, C, H_in, W_in, H_out, W_out, RS_INTERP, 1, static_cast<std::uint32_t>(mode), Y);
}

// ─── pooling & convex upsample ─────────────────────────────────────────────

void adaptive_avg_pool2d_forward(const Tensor& X, int N, int C, int H, int W, int H_out, int W_out, Tensor& Y) {
    constexpr const char* op = "adaptive_avg_pool2d_forward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 1 && W_out >= 1, "H_out and W_out must be >= 1");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    run_resample(op, X, N, C, H, W, H_out, W_out, RS_ADAPTIVE_AVG, 0, 0, Y);
}

void max_pool2d_forward(const Tensor& X, int N, int C, int H, int W, int kH, int kW,
                        int stride_h, int stride_w, int pad_h, int pad_w, Tensor& Y, Tensor& Idx) {
    constexpr const char* op = "max_pool2d_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, kH >= 1 && kW >= 1, "kH and kW must be >= 1");
    need(op, stride_h >= 1 && stride_w >= 1, "strides must be >= 1");
    need(op, pad_h >= 0 && pad_w >= 0, "pads must be >= 0");
    need(op, kH <= H + 2 * pad_h && kW <= W + 2 * pad_w, "kernel larger than padded input");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int Ho = (H + 2 * pad_h - kH) / stride_h + 1, Wo = (W + 2 * pad_w - kW) / stride_w + 1;
    const long long cols = static_cast<long long>(C) * Ho * Wo;
    ensure(Y, N, cols, X.dtype, op);
    ensure(Idx, N, cols, Dtype::INT32, op);
    if (N == 0) return;
    std::uint32_t gx = 0, gy = 0, gz = 0;
    if (!plane_dims(std::uint64_t(Ho) * Wo, C, N, gx, gy, gz, op)) return;
    CUfunction fn = resample_kernel(X.dtype, RS_MAXPOOL, 0, 0, op);
    mir_dispatch_resample(
        fn, gx, gy, gz, X.data, Y.data, Idx.data,
        static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(W),
        static_cast<std::uint32_t>(Ho), static_cast<std::uint32_t>(Wo),
        static_cast<std::uint32_t>(kH), static_cast<std::uint32_t>(kW),
        stride_h, stride_w, pad_h, pad_w);
}

void convex_upsample_forward(const Tensor& X, const Tensor& Mask, int N, int C, int H, int W, int scale, Tensor& Y) {
    constexpr const char* op = "convex_upsample_forward";
    require_fp(X.dtype, op);
    need(op, Mask.dtype == X.dtype, "Mask dtype must match X");
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, scale >= 1, "scale must be >=1");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    need(op, Mask.rows == N && Mask.cols == 9LL * scale * scale * H * W, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int Ho = scale * H, Wo = scale * W;
    ensure(Y, N, static_cast<long long>(C) * Ho * Wo, X.dtype, op);
    if (N == 0) return;
    std::uint32_t gx = 0, gy = 0, gz = 0;
    if (!plane_dims(std::uint64_t(Ho) * Wo, C, N, gx, gy, gz, op)) return;
    CUfunction fn = resample_kernel(X.dtype, RS_CONVEX, 0, 0, op);
    mir_dispatch_resample(
        fn, gx, gy, gz, X.data, Y.data, Mask.data,
        static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(W),
        static_cast<std::uint32_t>(Ho), static_cast<std::uint32_t>(Wo),
        static_cast<std::uint32_t>(scale), 0, 0, 0, 0, 0);
}

// ─── remaps (pad / slice / unfold / window / merge / shuffle / patch / rows) ─

void pad2d_forward(const Tensor& X, int N, int C, int H, int W, int pad_top, int pad_bottom,
                   int pad_left, int pad_right, int mode, Tensor& Y) {
    constexpr const char* op = "pad2d_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    check_pad_mode(op, mode);
    if (mode == 1) {
        need(op, pad_top < H && pad_bottom < H, "reflect padding requires pad_top and pad_bottom < H");
        need(op, pad_left < W && pad_right < W, "reflect padding requires pad_left and pad_right < W");
    }
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int Hp = H + pad_top + pad_bottom, Wp = W + pad_left + pad_right;
    ensure(Y, N, static_cast<long long>(C) * Hp * Wp, X.dtype, op);
    run_remap(op, X, RM_PAD, N, C, C, H, W, Hp, Wp, {pad_top, pad_left, mode, 0, 0, 0, 0}, Y);
}

void slice2d_forward(const Tensor& X, int N, int C, int H, int W, int h0, int w0, int H_out, int W_out, Tensor& Y) {
    constexpr const char* op = "slice2d_forward";
    require_fp(X.dtype, op);
    need(op, C >= 1 && N >= 0 && H >= 0 && W >= 0, "C must be >=1; N, H, W must be >=0");
    need(op, H_out >= 0 && W_out >= 0, "H_out and W_out must be >=0");
    need(op, h0 >= 0 && w0 >= 0, "h0 and w0 must be >=0");
    need(op, h0 + H_out <= H && w0 + W_out <= W, "slice out of bounds");
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, static_cast<long long>(C) * H_out * W_out, X.dtype, op);
    run_remap(op, X, RM_SLICE, N, C, C, H, W, H_out, W_out, {h0, w0, 0, 0, 0, 0, 0}, Y);
}

void slice2d_backward(const Tensor& dY, int N, int C, int H, int W, int h0, int w0,
                      int H_out, int W_out, Tensor& dX) {
    constexpr const char* op = "slice2d_backward";
    require_fp(dY.dtype, op);
    need(op, C >= 1 && N >= 0 && H >= 0 && W >= 0, "C must be >=1; N, H, W must be >=0");
    need(op, H_out >= 0 && W_out >= 0, "H_out and W_out must be >=0");
    need(op, h0 >= 0 && w0 >= 0, "h0 and w0 must be >=0");
    need(op, h0 + H_out <= H && w0 + W_out <= W, "slice outside the plane");
    check_nchw(op, dY, N, C, H_out, W_out);
    ensure(dX, N, static_cast<long long>(C) * H * W, dY.dtype, op);
    if (H_out == 0 || W_out == 0) {
        dX.zero();
        return;
    }
    run_remap(op, dY, RM_PAD, N, C, C, H_out, W_out, H, W, {h0, w0, 0, 0, 0, 0, 0}, dX);
}

void unfold2d_forward(const Tensor& X, int N, int C, int H, int W, int kH, int kW,
                      int stride_h, int stride_w, int pad_top, int pad_bottom,
                      int pad_left, int pad_right, int mode, Tensor& Y) {
    constexpr const char* op = "unfold2d_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, kH >= 1 && kW >= 1 && stride_h >= 1 && stride_w >= 1, "kH/kW/stride must be >=1");
    need(op, pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    check_pad_mode(op, mode);
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int Ho = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int Wo = (W + pad_left + pad_right - kW) / stride_w + 1;
    need(op, H + pad_top + pad_bottom >= kH && W + pad_left + pad_right >= kW && Ho > 0 && Wo > 0,
         "kernel/padding/stride yield empty output");
    const long long co = static_cast<long long>(C) * kH * kW;
    ensure(Y, N, co * Ho * Wo, X.dtype, op);
    run_remap(op, X, RM_UNFOLD, N, static_cast<int>(co), C, H, W, Ho, Wo,
              {kH, kW, stride_h, stride_w, pad_top, pad_left, mode}, Y);
}

void window_partition_forward(const Tensor& X, int N, int C, int H, int W, int window, Tensor& Y) {
    constexpr const char* op = "window_partition_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1 && window >= 1, "bad dimension");
    need(op, H % window == 0 && W % window == 0, "H and W must be multiples of window");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int nh = H / window, nw = W / window;
    ensure(Y, static_cast<long long>(N) * nh * nw, static_cast<long long>(C) * window * window, X.dtype, op);
    run_remap(op, X, RM_WIN_PART, N * nh * nw, C, C, H, W, window, window, {window, nh, nw, 0, 0, 0, 0}, Y);
}

void window_reverse_forward(const Tensor& X, int N, int C, int H, int W, int window, Tensor& Y) {
    constexpr const char* op = "window_reverse_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1 && window >= 1, "bad dimension");
    need(op, H % window == 0 && W % window == 0, "H and W must be multiples of window");
    const int nh = H / window, nw = W / window;
    need(op, X.rows == static_cast<long long>(N) * nh * nw && X.cols == C * window * window,
         "X shape must be (N*nw_h*nw_w, C*window*window)");
    ensure(Y, N, static_cast<long long>(C) * H * W, X.dtype, op);
    run_remap(op, X, RM_WIN_REV, N, C, C, window, window, H, W, {window, nh, nw, 0, 0, 0, 0}, Y);
}

void spatial_merge_2x2_forward(const Tensor& X, int N, int C, int H, int W, bool channel_major, Tensor& Y) {
    constexpr const char* op = "spatial_merge_2x2_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0 && H % 2 == 0 && W % 2 == 0, "bad dimension");
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, static_cast<long long>(C) * H * W, X.dtype, op);
    run_remap(op, X, RM_MERGE2, N, 4 * C, C, H, W, H / 2, W / 2, {channel_major ? 1 : 0, 0, 0, 0, 0, 0, 0}, Y);
}

void pixel_shuffle_upsample_2x_forward(const Tensor& X, int N, int C_in, int H, int W, int C_out, Tensor& Y) {
    constexpr const char* op = "pixel_shuffle_upsample_2x_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C_in > 0 && H >= 0 && W >= 0 && C_out > 0 && (4 * C_out) % C_in == 0, "bad dimension");
    check_nchw(op, X, N, C_in, H, W);
    ensure(Y, N, 4LL * C_out * H * W, X.dtype, op);
    run_remap(op, X, RM_PSHUF, N, C_out, C_in, H, W, 2 * H, 2 * W, {(4 * C_out) / C_in, 0, 0, 0, 0, 0, 0}, Y);
}

void patch_unpack_forward(const Tensor& tokens, int hp, int wp, int P, int C_total, int C_keep,
                          bool channel_major, Tensor& Y) {
    constexpr const char* op = "patch_unpack_forward";
    require_fp(tokens.dtype, op);
    need(op, hp >= 0 && wp >= 0 && P > 0 && C_total > 0 && C_keep > 0 && C_keep <= C_total, "bad dimension");
    need(op, tokens.rows == hp * wp && tokens.cols == P * P * C_total, "tokens shape mismatch");
    const int H = hp * P, W = wp * P;
    ensure(Y, 1, static_cast<long long>(C_keep) * H * W, tokens.dtype, op);
    if (hp * wp == 0) return;
    run_remap(op, tokens, RM_PATCH, 1, C_keep, C_total, 1, 1, H, W,
              {P, C_total, channel_major ? 1 : 0, wp, 0, 0, 0}, Y);
}

void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y) {
    constexpr const char* op = "gather_rows";
    require_fp(X.dtype, op);
    need(op, Idx.dtype == Dtype::INT32 && Idx.cols == 1, "Idx must be (M, 1) INT32");
    const int M = Idx.rows, C = X.cols;
    ensure(Y, M, C, X.dtype, op);
    if (M == 0 || C == 0) return;
    need(op, X.rows > 0, "X has no rows");
    row_gather_dispatch(X, static_cast<const std::int32_t*>(Idx.data), M, C, Y, op);
}

void embedding_lookup_forward(const Tensor& table, const std::int32_t* d_idx, int B, Tensor& out) {
    constexpr const char* op = "embedding_lookup_forward";
    require_fp(table.dtype, op);
    const int D = table.cols;
    ensure(out, B, D, table.dtype, op);
    if (B <= 0 || D == 0) return;
    need(op, table.rows > 0 && d_idx != nullptr, "empty table or null index pointer");
    row_gather_dispatch(table, d_idx, B, D, out, op);
}

void scatter_rows(const Tensor& Y, const Tensor& Idx, Tensor& X) {
    constexpr const char* op = "scatter_rows";
    require_fp(Y.dtype, op);
    need(op, Idx.dtype == Dtype::INT32 && Idx.cols == 1, "Idx must be (M, 1) INT32");
    const int M = Idx.rows, C = Y.cols;
    need(op, Y.rows == M && X.dtype == Y.dtype && X.cols == C, "X/Y shape or dtype mismatch");
    if (M == 0 || C == 0) return;
    CUfunction fn = remap_kernel(Y.dtype, RM_SCATTER_ROWS, op);
    const std::size_t es = static_cast<std::size_t>(::brotensor::dtype_size_bytes(Y.dtype));
    const auto* idx_p = static_cast<const std::int32_t*>(Idx.data);
    for (int m0 = 0; m0 < M; m0 += 65535) {
        std::uint32_t gx = 0, gy = 0, gz = 0;
        if (!plane_dims(static_cast<std::uint64_t>(C), 1, (std::min)(65535, M - m0), gx, gy, gz, op)) break;
        const char* in_ptr = static_cast<const char*>(Y.data) + static_cast<std::size_t>(m0) * C * es;
        mir_dispatch_remap(
            fn, gx, gy, gz,
            in_ptr, X.data, idx_p + m0,
            1, 1, 1, static_cast<std::uint32_t>(C), 1, static_cast<std::uint32_t>(C),
            X.rows, 0, 0, 0, 0, 0, 0);
    }
}

// ─── vtable registrations ──────────────────────────────────────────────────

void fill_cuda_vtable_interp2d(::brotensor::detail::OpsVTable& v) {
    v.interp2d_forward = &interp2d_forward;
    v.interp2d_align_corners_forward = &interp2d_align_corners_forward;
    v.interp2d_backward = &interp2d_backward;
}

void fill_cuda_vtable_pool2d(::brotensor::detail::OpsVTable& v) {
    v.adaptive_avg_pool2d_forward = &adaptive_avg_pool2d_forward;
    v.adaptive_avg_pool2d_backward = &adaptive_avg_pool2d_backward;
    v.max_pool2d_forward = &max_pool2d_forward;
    v.max_pool2d_backward = &max_pool2d_backward;
}

void fill_cuda_vtable_pad2d(::brotensor::detail::OpsVTable& v) {
    v.pad2d_forward = &pad2d_forward;
    v.pad2d_backward = &pad2d_backward;
}

void fill_cuda_vtable_slice2d(::brotensor::detail::OpsVTable& v) {
    v.slice2d_forward = &slice2d_forward;
    v.slice2d_backward = &slice2d_backward;
}

void fill_cuda_vtable_unfold2d(::brotensor::detail::OpsVTable& v) {
    v.unfold2d_forward = &unfold2d_forward;
}

void fill_cuda_vtable_convex_upsample(::brotensor::detail::OpsVTable& v) {
    v.convex_upsample_forward = &convex_upsample_forward;
}

void fill_cuda_vtable_window_partition(::brotensor::detail::OpsVTable& v) {
    v.window_partition_forward = &window_partition_forward;
    v.window_reverse_forward = &window_reverse_forward;
}

void fill_cuda_vtable_gather_scatter(::brotensor::detail::OpsVTable& v) {
    v.gather_rows = &gather_rows;
    v.scatter_rows_add = &scatter_rows_add;
    v.scatter_rows = &scatter_rows;
}

}  // namespace brotensor::detail::cuda
