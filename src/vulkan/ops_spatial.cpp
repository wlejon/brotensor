// Vulkan NCHW spatial ops: the 2x resamples and their two gather backwards,
// arbitrary-scale interpolation (nearest / bilinear / bicubic, half-pixel or
// corner-aligned), adaptive average and max pooling (shaders/resample.comp),
// the convex upsample, and the pure gathers of shaders/remap.comp: padding,
// crop and its backward, unfold, window partition / reverse, pixel
// (un)shuffles, DiT unpatchify, row gather / scatter, plus the channel
// concat (strided copies). Contracts follow the CUDA backend (src/cuda/
// resample.cu, interp2d.cu, pool2d.cu, pad2d.cu, slice2d.cu, unfold2d.cu,
// window_partition.cu, spatial_merge.cu, convex_upsample.cu, concat.cu,
// gather_scatter.cu): FP32 / FP16 / BF16 storage (FP32 arithmetic for the
// resamples, bit-exact copies for the gathers), outputs resized to the
// op's shape in the input's dtype and overwritten.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct ResamplePush {
    std::uint64_t x, y, aux;
    std::uint32_t c, hi, wi, ho, wo, kh, kw;
    std::int32_t sh, sw, ph, pw;
};

struct RemapPush {
    std::uint64_t x, y, idx;
    std::uint32_t c, cin, hi, wi, ho, wo;
    std::int32_t a0, a1, a2, a3, a4, a5, a6;
};

constexpr std::uint32_t kCpt = 16;   // channels per invocation (both kernels' CPT)

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

void ensure(Tensor& t, long long r, long long c, Dtype dt, const char* op) {
    if (r < 0 || c < 0 || c > 0x7fffffffLL || r > 0x7fffffffLL) fail(op, "output too large");
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(static_cast<int>(r), static_cast<int>(c), dt);
}

// Grid for a plane-per-x, CPT-channels-per-y, image-per-z kernel.
void plane_launch(DeviceCtx& d, const Kernel& k, const void* pc_ptr, std::uint32_t pc_size, std::uint64_t plane,
                  std::uint64_t channels, std::uint64_t images, const char* op) {
    if (plane == 0 || channels == 0 || images == 0) return;
    const std::uint64_t gy = cdiv(channels, kCpt);
    if (gy > 65535 || images > 65535) fail(op, "too many channels or images for one dispatch");
    // Fill the GPU when the plane is small: more groups in x than the plane
    // needs only idle, so x covers the plane once.
    const std::uint64_t gx = std::min<std::uint64_t>(cdiv(plane, 256), 65535);
    d.stream().dispatch(k.pipe, pc_ptr, pc_size, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy),
                        static_cast<std::uint32_t>(images));
}

template <class Push>
void plane_run(DeviceCtx& d, const Kernel& k, const Push& pc, std::uint64_t plane, std::uint64_t channels,
               std::uint64_t images, const char* op) {
    static_assert(sizeof(Push) <= Pipelines::kMaxPushBytes);
    plane_launch(d, k, &pc, static_cast<std::uint32_t>(sizeof(Push)), plane, channels, images, op);
}

const Kernel& resample_kernel(DeviceCtx& d, Dtype dt, std::uint32_t mode, std::uint32_t align, std::uint32_t imode,
                              const char* op) {
    return d.pipelines().get(dt_variant(ShaderId::resample_f32, dt, op), {mode, kCpt, align, imode});
}

const Kernel& remap_kernel(DeviceCtx& d, Dtype dt, std::uint32_t mode, const char* op) {
    const int es = ::brotensor::dtype_size_bytes(dt);
    if (es != 2 && es != 4) fail(op, "FP32 / FP16 / BF16 / INT32 only on Vulkan");
    return d.pipelines().get(es == 4 ? ShaderId::remap_b4 : ShaderId::remap_b2, {mode, kCpt});
}

void check_nchw(const char* op, const Tensor& X, int N, int C, int H, int W) {
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0, "negative dimension");
    need(op, X.size() >= static_cast<long long>(N) * C * H * W, "X is smaller than N*C*H*W");
}

// Resample X (N, C, hi, wi) -> Y (N, C, ho, wo) with resample.comp.
void resample(const char* op, const Tensor& X, int N, int C, int hi, int wi, int ho, int wo, std::uint32_t mode,
              std::uint32_t align, std::uint32_t imode, Tensor& Y) {
    dt_code(X.dtype, op);
    check_nchw(op, X, N, C, hi, wi);
    ensure(Y, N, static_cast<long long>(C) * ho * wo, X.dtype, op);
    if (N == 0 || C == 0 || ho == 0 || wo == 0) return;
    DeviceCtx& d = device_of(X);
    ResamplePush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data);
    pc.c = static_cast<std::uint32_t>(C);
    pc.hi = static_cast<std::uint32_t>(hi); pc.wi = static_cast<std::uint32_t>(wi);
    pc.ho = static_cast<std::uint32_t>(ho); pc.wo = static_cast<std::uint32_t>(wo);
    plane_run(d, resample_kernel(d, X.dtype, mode, align, imode, op), pc, std::uint64_t(ho) * wo, C, N, op);
}

// Remap X (N, cin, hi, wi) -> Y (N, c, ho, wo) with remap.comp.
void remap(const char* op, const Tensor& X, std::uint32_t mode, int images, int c, int cin, int hi, int wi, int ho,
           int wo, const std::int32_t (&a)[7], Tensor& Y) {
    if (images == 0 || c == 0 || ho == 0 || wo == 0) return;
    DeviceCtx& d = device_of(X);
    RemapPush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data);
    pc.c = static_cast<std::uint32_t>(c); pc.cin = static_cast<std::uint32_t>(cin);
    pc.hi = static_cast<std::uint32_t>(hi); pc.wi = static_cast<std::uint32_t>(wi);
    pc.ho = static_cast<std::uint32_t>(ho); pc.wo = static_cast<std::uint32_t>(wo);
    pc.a0 = a[0]; pc.a1 = a[1]; pc.a2 = a[2]; pc.a3 = a[3]; pc.a4 = a[4]; pc.a5 = a[5]; pc.a6 = a[6];
    plane_run(d, remap_kernel(d, X.dtype, mode, op), pc, std::uint64_t(ho) * wo, c, images, op);
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

}  // namespace

// ─── fixed 2x resamples ────────────────────────────────────────────────────

void upsample_nearest_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const char* op = "upsample_nearest_2x";
    dt_code(X.dtype, op);
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, 4LL * C * H * W, X.dtype, op);
    remap(op, X, RM_UP2, N, C, C, H, W, 2 * H, 2 * W, {0, 0, 0, 0, 0, 0, 0}, Y);
}

void upsample_bilinear_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    resample("upsample_bilinear_2x", X, N, C, H, W, 2 * H, 2 * W, RS_INTERP, 0, 1, Y);
}

void downsample_avg_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    const char* op = "downsample_avg_2x";
    need(op, H % 2 == 0 && W % 2 == 0, "H and W must be even");
    resample(op, X, N, C, H, W, H / 2, W / 2, RS_DOWN_AVG2, 0, 0, Y);
}

void upsample_nearest_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    resample("upsample_nearest_2x_backward", dY, N, C, 2 * H, 2 * W, H, W, RS_UP_NEAREST2_BWD, 0, 0, dX);
}

void downsample_avg_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    const char* op = "downsample_avg_2x_backward";
    need(op, H % 2 == 0 && W % 2 == 0, "H and W must be even");
    resample(op, dY, N, C, H / 2, W / 2, H, W, RS_DOWN_AVG2_BWD, 0, 0, dX);
}

// ─── arbitrary-scale interpolation ─────────────────────────────────────────

void interp2d_forward(const Tensor& X, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode, Tensor& Y) {
    const char* op = "interp2d_forward";
    check_interp(op, N, C, H_in, W_in, H_out, W_out, mode);
    resample(op, X, N, C, H_in, W_in, H_out, W_out, RS_INTERP, 0, static_cast<std::uint32_t>(mode), Y);
}

void interp2d_align_corners_forward(const Tensor& X, int N, int C, int H_in, int W_in, int H_out, int W_out,
                                    int mode, Tensor& Y) {
    const char* op = "interp2d_align_corners_forward";
    check_interp(op, N, C, H_in, W_in, H_out, W_out, mode);
    resample(op, X, N, C, H_in, W_in, H_out, W_out, RS_INTERP, 1, static_cast<std::uint32_t>(mode), Y);
}

// ─── pooling ───────────────────────────────────────────────────────────────

void adaptive_avg_pool2d_forward(const Tensor& X, int N, int C, int H, int W, int H_out, int W_out, Tensor& Y) {
    const char* op = "adaptive_avg_pool2d_forward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 1 && W_out >= 1, "H_out and W_out must be >= 1");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    resample(op, X, N, C, H, W, H_out, W_out, RS_ADAPTIVE_AVG, 0, 0, Y);
}

void max_pool2d_forward(const Tensor& X, int N, int C, int H, int W, int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w, Tensor& Y, Tensor& Idx) {
    const char* op = "max_pool2d_forward";
    dt_code(X.dtype, op);
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
    DeviceCtx& d = device_of(X);
    ResamplePush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data); pc.aux = addr(Idx.data);
    pc.c = static_cast<std::uint32_t>(C);
    pc.hi = static_cast<std::uint32_t>(H); pc.wi = static_cast<std::uint32_t>(W);
    pc.ho = static_cast<std::uint32_t>(Ho); pc.wo = static_cast<std::uint32_t>(Wo);
    pc.kh = static_cast<std::uint32_t>(kH); pc.kw = static_cast<std::uint32_t>(kW);
    pc.sh = stride_h; pc.sw = stride_w; pc.ph = pad_h; pc.pw = pad_w;
    plane_run(d, resample_kernel(d, X.dtype, RS_MAXPOOL, 0, 0, op), pc, std::uint64_t(Ho) * Wo, C, N, op);
}

void convex_upsample_forward(const Tensor& X, const Tensor& Mask, int N, int C, int H, int W, int scale, Tensor& Y) {
    const char* op = "convex_upsample_forward";
    dt_code(X.dtype, op);
    need(op, Mask.dtype == X.dtype, "Mask dtype must match X");
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, scale >= 1, "scale must be >=1");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    need(op, Mask.rows == N && Mask.cols == 9LL * scale * scale * H * W, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int Ho = scale * H, Wo = scale * W;
    ensure(Y, N, static_cast<long long>(C) * Ho * Wo, X.dtype, op);
    if (N == 0) return;
    DeviceCtx& d = device_of(X);
    ResamplePush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data); pc.aux = addr(Mask.data);
    pc.c = static_cast<std::uint32_t>(C);
    pc.hi = static_cast<std::uint32_t>(H); pc.wi = static_cast<std::uint32_t>(W);
    pc.ho = static_cast<std::uint32_t>(Ho); pc.wo = static_cast<std::uint32_t>(Wo);
    pc.kh = static_cast<std::uint32_t>(scale);
    plane_run(d, resample_kernel(d, X.dtype, RS_CONVEX, 0, 0, op), pc, std::uint64_t(Ho) * Wo, C, N, op);
}

// ─── gathers ───────────────────────────────────────────────────────────────

void pad2d_forward(const Tensor& X, int N, int C, int H, int W, int pad_top, int pad_bottom, int pad_left,
                   int pad_right, int mode, Tensor& Y) {
    const char* op = "pad2d_forward";
    dt_code(X.dtype, op);
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
    remap(op, X, RM_PAD, N, C, C, H, W, Hp, Wp, {pad_top, pad_left, mode, 0, 0, 0, 0}, Y);
}

void slice2d_forward(const Tensor& X, int N, int C, int H, int W, int h0, int w0, int H_out, int W_out, Tensor& Y) {
    const char* op = "slice2d_forward";
    dt_code(X.dtype, op);
    need(op, C >= 1 && N >= 0 && H >= 0 && W >= 0, "C must be >=1; N, H, W must be >=0");
    need(op, H_out >= 0 && W_out >= 0, "H_out and W_out must be >=0");
    need(op, h0 >= 0 && w0 >= 0, "h0 and w0 must be >=0");
    need(op, h0 + H_out <= H, "h0 + H_out must be <= H");
    need(op, w0 + W_out <= W, "w0 + W_out must be <= W");
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, static_cast<long long>(C) * H_out * W_out, X.dtype, op);
    remap(op, X, RM_SLICE, N, C, C, H, W, H_out, W_out, {h0, w0, 0, 0, 0, 0, 0}, Y);
}

// dX = dY placed at (h0, w0) in a zero (H, W) plane: zero padding.
void slice2d_backward(const Tensor& dY, int N, int C, int H, int W, int h0, int w0, int H_out, int W_out, Tensor& dX) {
    const char* op = "slice2d_backward";
    dt_code(dY.dtype, op);
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
    remap(op, dY, RM_PAD, N, C, C, H_out, W_out, H, W, {h0, w0, 0, 0, 0, 0, 0}, dX);
}

void unfold2d_forward(const Tensor& X, int N, int C, int H, int W, int kH, int kW, int stride_h, int stride_w,
                      int pad_top, int pad_bottom, int pad_left, int pad_right, int mode, Tensor& Y) {
    const char* op = "unfold2d_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, kH >= 1 && kW >= 1, "kH/kW must be >=1");
    need(op, stride_h >= 1 && stride_w >= 1, "stride must be >=1");
    need(op, pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    check_pad_mode(op, mode);
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int Ho = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int Wo = (W + pad_left + pad_right - kW) / stride_w + 1;
    need(op, H + pad_top + pad_bottom >= kH && W + pad_left + pad_right >= kW && Ho > 0 && Wo > 0,
         "kernel/padding/stride yield empty output");
    const long long co = static_cast<long long>(C) * kH * kW;
    ensure(Y, N, co * Ho * Wo, X.dtype, op);
    remap(op, X, RM_UNFOLD, N, static_cast<int>(co), C, H, W, Ho, Wo,
          {kH, kW, stride_h, stride_w, pad_top, pad_left, mode}, Y);
}

void window_partition_forward(const Tensor& X, int N, int C, int H, int W, int window, Tensor& Y) {
    const char* op = "window_partition_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, window >= 1, "window must be >=1");
    need(op, H % window == 0 && W % window == 0, "H and W must be multiples of window (use pad2d first if needed)");
    need(op, X.rows == N && X.cols == C * H * W, "X shape must be (N, C*H*W)");
    const int nh = H / window, nw = W / window;
    ensure(Y, static_cast<long long>(N) * nh * nw, static_cast<long long>(C) * window * window, X.dtype, op);
    remap(op, X, RM_WIN_PART, N * nh * nw, C, C, H, W, window, window, {window, nh, nw, 0, 0, 0, 0}, Y);
}

void window_reverse_forward(const Tensor& X, int N, int C, int H, int W, int window, Tensor& Y) {
    const char* op = "window_reverse_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, window >= 1, "window must be >=1");
    need(op, H % window == 0 && W % window == 0, "H and W must be multiples of window (use pad2d first if needed)");
    const int nh = H / window, nw = W / window;
    need(op, X.rows == static_cast<long long>(N) * nh * nw && X.cols == C * window * window,
         "X shape must be (N*nw_h*nw_w, C*window*window)");
    ensure(Y, N, static_cast<long long>(C) * H * W, X.dtype, op);
    remap(op, X, RM_WIN_REV, N, C, C, window, window, H, W, {window, nh, nw, 0, 0, 0, 0}, Y);
}

void spatial_merge_2x2_forward(const Tensor& X, int N, int C, int H, int W, bool channel_major, Tensor& Y) {
    const char* op = "spatial_merge_2x2_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0, "negative dimension");
    need(op, H % 2 == 0 && W % 2 == 0, "H and W must be even");
    check_nchw(op, X, N, C, H, W);
    ensure(Y, N, static_cast<long long>(C) * H * W, X.dtype, op);
    remap(op, X, RM_MERGE2, N, 4 * C, C, H, W, H / 2, W / 2, {channel_major ? 1 : 0, 0, 0, 0, 0, 0, 0}, Y);
}

void pixel_shuffle_upsample_2x_forward(const Tensor& X, int N, int C_in, int H, int W, int C_out, Tensor& Y) {
    const char* op = "pixel_shuffle_upsample_2x_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C_in > 0 && H >= 0 && W >= 0 && C_out > 0, "bad dimension");
    need(op, (4 * C_out) % C_in == 0, "C_in must divide 4*C_out");
    check_nchw(op, X, N, C_in, H, W);
    ensure(Y, N, 4LL * C_out * H * W, X.dtype, op);
    remap(op, X, RM_PSHUF, N, C_out, C_in, H, W, 2 * H, 2 * W, {(4 * C_out) / C_in, 0, 0, 0, 0, 0, 0}, Y);
}

void patch_unpack_forward(const Tensor& tokens, int hp, int wp, int P, int C_total, int C_keep, bool channel_major,
                          Tensor& Y) {
    const char* op = "patch_unpack_forward";
    dt_code(tokens.dtype, op);
    need(op, hp >= 0 && wp >= 0 && P > 0 && C_total > 0 && C_keep > 0 && C_keep <= C_total, "bad dimension");
    need(op, tokens.rows == hp * wp && tokens.cols == P * P * C_total, "tokens shape mismatch");
    const int H = hp * P, W = wp * P;
    ensure(Y, 1, static_cast<long long>(C_keep) * H * W, tokens.dtype, op);
    if (hp * wp == 0) return;
    remap(op, tokens, RM_PATCH, 1, C_keep, C_total, 1, 1, H, W, {P, C_total, channel_major ? 1 : 0, wp, 0, 0, 0}, Y);
}

void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y) {
    const char* op = "gather_rows";
    need(op, Idx.dtype == Dtype::INT32, "Idx must be INT32");
    need(op, Idx.cols == 1, "Idx must be (M, 1)");
    const int M = Idx.rows, C = X.cols;
    ensure(Y, M, C, X.dtype, op);
    if (M == 0 || C == 0) return;
    need(op, X.rows > 0, "X has no rows");
    DeviceCtx& d = device_of(X);
    const Kernel& k = remap_kernel(d, X.dtype, RM_GATHER_ROWS, op);
    const std::uint64_t es = static_cast<std::uint64_t>(::brotensor::dtype_size_bytes(X.dtype));
    for (int m0 = 0; m0 < M; m0 += 65535) {
        RemapPush pc{};
        pc.x = addr(X.data);
        pc.y = addr(Y.data) + std::uint64_t(m0) * C * es;
        pc.idx = addr(Idx.data) + std::uint64_t(m0) * 4u;
        pc.c = 1; pc.cin = 1; pc.hi = 1; pc.wi = static_cast<std::uint32_t>(C);
        pc.ho = 1; pc.wo = static_cast<std::uint32_t>(C);
        pc.a0 = X.rows;
        plane_run(d, k, pc, static_cast<std::uint64_t>(C), 1, std::min(65535, M - m0), op);
    }
}

void scatter_rows(const Tensor& Y, const Tensor& Idx, Tensor& X) {
    const char* op = "scatter_rows";
    need(op, Idx.dtype == Dtype::INT32, "Idx must be INT32");
    need(op, Idx.cols == 1, "Idx must be (M, 1)");
    const int M = Idx.rows;
    need(op, Y.rows == M, "Y.rows must equal Idx.rows");
    need(op, X.dtype == Y.dtype, "X dtype must match Y");
    need(op, X.cols == Y.cols, "X.cols must equal Y.cols");
    const int C = Y.cols;
    if (M == 0 || C == 0) return;
    DeviceCtx& d = device_of(Y);
    const Kernel& k = remap_kernel(d, Y.dtype, RM_SCATTER_ROWS, op);
    const std::uint64_t es = static_cast<std::uint64_t>(::brotensor::dtype_size_bytes(Y.dtype));
    for (int m0 = 0; m0 < M; m0 += 65535) {
        RemapPush pc{};
        pc.x = addr(Y.data) + std::uint64_t(m0) * C * es;
        pc.y = addr(X.data);
        pc.idx = addr(Idx.data) + std::uint64_t(m0) * 4u;
        pc.c = 1; pc.cin = 1; pc.hi = 1; pc.wi = static_cast<std::uint32_t>(C);
        pc.ho = 1; pc.wo = static_cast<std::uint32_t>(C);
        pc.a0 = X.rows;
        plane_run(d, k, pc, static_cast<std::uint64_t>(C), 1, std::min(65535, M - m0), op);
    }
}

// ─── channel concat (strided copies) ───────────────────────────────────────

void concat_nchw_channels(const std::vector<const Tensor*>& parts, int N, int H, int W,
                          const std::vector<int>& C_per_part, Tensor& out) {
    const char* op = "concat_nchw_channels";
    need(op, parts.size() == C_per_part.size(), "parts.size() != C_per_part.size()");
    need(op, !parts.empty() && parts[0], "no parts");
    const Dtype dt = parts[0]->dtype;
    long long total_C = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        need(op, parts[i] != nullptr, "null part");
        need(op, parts[i]->dtype == dt, "parts must share a dtype");
        need(op, parts[i]->size() == static_cast<long long>(N) * C_per_part[i] * H * W,
             "part size mismatch (expected N*C_i*H*W)");
        total_C += C_per_part[i];
    }
    const long long HW = static_cast<long long>(H) * W;
    ensure(out, N, total_C * HW, dt, op);
    long long c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const long long chunk = C_per_part[i] * HW;
        if (chunk > 0 && N > 0) {
            ::brotensor::copy_d2d_strided(*parts[i], 0, static_cast<int>(chunk), out, static_cast<int>(c_off * HW),
                                          static_cast<int>(total_C * HW), static_cast<int>(chunk), N);
        }
        c_off += C_per_part[i];
    }
}

void concat_nchw_channels_backward(const Tensor& dY, int N, int H, int W, const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts) {
    const char* op = "concat_nchw_channels_backward";
    need(op, parts.size() == C_per_part.size(), "parts.size() != C_per_part.size()");
    long long total_C = 0;
    for (int c : C_per_part) total_C += c;
    const long long HW = static_cast<long long>(H) * W;
    need(op, dY.size() == N * total_C * HW, "dY size mismatch");
    long long c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        need(op, parts[i] != nullptr, "null part");
        const long long chunk = C_per_part[i] * HW;
        ensure(*parts[i], N, chunk, dY.dtype, op);
        if (chunk > 0 && N > 0) {
            ::brotensor::copy_d2d_strided(dY, static_cast<int>(c_off * HW), static_cast<int>(total_C * HW),
                                          *parts[i], 0, static_cast<int>(chunk), static_cast<int>(chunk), N);
        }
        c_off += C_per_part[i];
    }
}

// out (B, sum of widths) = the parts side by side, row by row (null parts
// skipped), in the first part's dtype.
void concat_batched_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    const char* op = "concat_batched_rows";
    int B = -1;
    long long total = 0;
    Dtype dt = Dtype::FP32;
    for (const Tensor* p : parts) {
        if (!p) continue;
        if (B < 0) { B = p->rows; dt = p->dtype; }
        need(op, p->rows == B, "parts must have the same number of rows");
        need(op, p->dtype == dt, "parts must share a dtype");
        total += p->cols;
    }
    if (B < 0) {
        ensure(out, 0, 0, Dtype::FP32, op);
        return;
    }
    ensure(out, B, total, dt, op);
    long long off = 0;
    for (const Tensor* p : parts) {
        if (!p || p->cols == 0) continue;
        if (B > 0) {
            ::brotensor::copy_d2d_strided(*p, 0, p->cols, out, static_cast<int>(off), static_cast<int>(total), p->cols,
                                          B);
        }
        off += p->cols;
    }
}

void fill_vulkan_vtable_spatial(::brotensor::detail::OpsVTable& v) {
    v.upsample_nearest_2x = &upsample_nearest_2x;
    v.upsample_bilinear_2x = &upsample_bilinear_2x;
    v.downsample_avg_2x = &downsample_avg_2x;
    v.upsample_nearest_2x_backward = &upsample_nearest_2x_backward;
    v.downsample_avg_2x_backward = &downsample_avg_2x_backward;
    v.interp2d_forward = &interp2d_forward;
    v.interp2d_align_corners_forward = &interp2d_align_corners_forward;
    v.adaptive_avg_pool2d_forward = &adaptive_avg_pool2d_forward;
    v.max_pool2d_forward = &max_pool2d_forward;
    v.convex_upsample_forward = &convex_upsample_forward;
    v.pad2d_forward = &pad2d_forward;
    v.slice2d_forward = &slice2d_forward;
    v.slice2d_backward = &slice2d_backward;
    v.unfold2d_forward = &unfold2d_forward;
    v.window_partition_forward = &window_partition_forward;
    v.window_reverse_forward = &window_reverse_forward;
    v.spatial_merge_2x2_forward = &spatial_merge_2x2_forward;
    v.pixel_shuffle_upsample_2x_forward = &pixel_shuffle_upsample_2x_forward;
    v.patch_unpack_forward = &patch_unpack_forward;
    v.gather_rows = &gather_rows;
    v.scatter_rows = &scatter_rows;
    v.concat_nchw_channels = &concat_nchw_channels;
    v.concat_nchw_channels_backward = &concat_nchw_channels_backward;
    v.concat_batched_rows = &concat_batched_rows;
}

}  // namespace brotensor::detail::vulkan
