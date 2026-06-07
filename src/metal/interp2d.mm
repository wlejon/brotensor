// ─── Metal arbitrary-scale 2D resample ──────────────────────────────────────
//
// Metal counterpart of src/cpu/interp2d.cpp. FP32-only on this backend (first
// pass — FP16/BF16 dispatch can be added later if vision inference workloads
// land on Metal hosts). Arbitrary-scale resampling on an NCHW tensor with
// nearest / bilinear forward and backward, plus bicubic forward (mode 2 =
// Catmull-Rom a=-0.5 / PIL; mode 3 = a=-0.75 / torch); bicubic backward throws.
//
// Memory layout (NCHW flat — consistent with resample.mm / conv2d.mm):
//   element (n, c, h, w) at ((n*C + c)*H + h)*W + w
//
// Sampling convention — PyTorch align_corners=False / half-pixel; identical
// to interp2d.cpp / interp2d.cu. At (H_out, W_out) == (2H, 2W) the output
// matches upsample_*_2x exactly (relied on by test_interp2d_parity).
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   interp2d_forward  — Y  OVERWRITTEN (one thread per output element).
//   interp2d_backward — dX OVERWRITTEN. The CPU op is a zero-then-scatter
//     adjoint; Metal instead runs the exact transpose as a *gather*: one
//     thread per input element walks the H_out*W_out outputs and sums any
//     that sampled it. Avoids atomics and lets the result be FP32 bit-stable
//     against the CPU when the same sum order is used.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) +
                 " must be FP32 (interp2d is FP32-only on Metal)");
    }
}

void check_args(const char* op,
                int N, int C, int H_in, int W_in,
                int H_out, int W_out, int mode,
                bool allow_bicubic) {
    if (N < 0 || C < 0 || H_in < 0 || W_in < 0 ||
        H_out < 0 || W_out < 0) {
        fail(op, "N, C, H_in, W_in, H_out, W_out must be non-negative");
    }
    const int max_mode = allow_bicubic ? 3 : 1;
    if (mode < 0 || mode > max_mode) {
        fail(op, allow_bicubic
            ? "mode must be 0 (nearest), 1 (bilinear), 2 (bicubic a=-0.5, "
              "PIL), or 3 (bicubic a=-0.75, torch)"
            : "mode must be 0 (nearest) or 1 (bilinear) — bicubic backward "
              "is not implemented");
    }
    if ((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)) {
        fail(op, "input spatial dims must be > 0 when output spatial "
                 "dims are > 0");
    }
}

// Parameter block — must match the MSL struct below.
struct I2dParams {
    uint32_t N, C, H_in, W_in, H_out, W_out;
    uint32_t mode;     // 0 nearest, 1 bilinear, 2 bicubic a=-0.5, 3 bicubic a=-0.75 (fwd only)
    uint32_t align;    // 0 = half-pixel (align_corners=False), 1 = align_corners=True
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct I2dParams {
    uint N, C, H_in, W_in, H_out, W_out;
    uint mode;
    uint align;
    uint total;
};

// Keys cubic-convolution kernel with coefficient `a`. a = -0.5 is Catmull-Rom
// (matches PIL/Pillow BICUBIC); a = -0.75 matches torch interpolate("bicubic")
// and OpenCV. The two differ only in that constant.
static inline float cubic_keys(float t, float a) {
    float at = fabs(t);
    if (at < 1.0f) {
        return ((a + 2.0f) * at - (a + 3.0f)) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        // a*t^3 - 5a*t^2 + 8a*t - 4a, Horner in |t|.
        return ((a * at - 5.0f * a) * at + 8.0f * a) * at - 4.0f * a;
    }
    return 0.0f;
}

// ── forward: one thread per output element (n, c, oh, ow) ────────────────────
kernel void k_interp2d_forward(device const float* X [[buffer(0)]],
                               device float*       Y [[buffer(1)]],
                               constant I2dParams& P [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint ow   = gid % P.W_out;
    uint t1   = gid / P.W_out;
    uint oh   = t1 % P.H_out;
    uint t2   = t1 / P.H_out;
    uint c    = t2 % P.C;
    uint n    = t2 / P.C;
    uint xbase = (n * P.C + c) * P.H_in * P.W_in;
    float sy = float(P.H_in) / float(P.H_out);
    float sx = float(P.W_in) / float(P.W_out);
    int Hi = int(P.H_in), Wi = int(P.W_in);
    // align_corners=True maps o -> o*(in-1)/(out-1); else half-pixel.
    float src_y, src_x;
    if (P.align != 0u) {
        src_y = (P.H_out > 1u) ? float(oh) * float(Hi - 1) / float(int(P.H_out) - 1) : 0.0f;
        src_x = (P.W_out > 1u) ? float(ow) * float(Wi - 1) / float(int(P.W_out) - 1) : 0.0f;
    } else {
        src_y = (float(oh) + 0.5f) * sy - 0.5f;
        src_x = (float(ow) + 0.5f) * sx - 0.5f;
    }

    if (P.mode == 0u) {
        int iy = max(0, min(int(rint(src_y)), Hi - 1));
        int ix = max(0, min(int(rint(src_x)), Wi - 1));
        Y[gid] = X[xbase + uint(iy) * P.W_in + uint(ix)];
    } else if (P.mode == 1u) {
        int y0 = int(floor(src_y));
        int x0 = int(floor(src_x));
        float fy = src_y - float(y0);
        float fx = src_x - float(x0);
        int y0c = max(0, min(y0,     Hi - 1));
        int y1c = max(0, min(y0 + 1, Hi - 1));
        int x0c = max(0, min(x0,     Wi - 1));
        int x1c = max(0, min(x0 + 1, Wi - 1));
        float v00 = X[xbase + uint(y0c) * P.W_in + uint(x0c)];
        float v01 = X[xbase + uint(y0c) * P.W_in + uint(x1c)];
        float v10 = X[xbase + uint(y1c) * P.W_in + uint(x0c)];
        float v11 = X[xbase + uint(y1c) * P.W_in + uint(x1c)];
        float top = v00 + (v01 - v00) * fx;
        float bot = v10 + (v11 - v10) * fx;
        Y[gid] = top + (bot - top) * fy;
    } else {
        // bicubic — mode 2: a=-0.5 (PIL); mode 3: a=-0.75 (torch/OpenCV).
        float a = (P.mode == 3u) ? -0.75f : -0.5f;
        int y0 = int(floor(src_y));
        int x0 = int(floor(src_x));
        float fy = src_y - float(y0);
        float fx = src_x - float(x0);
        float wy[4], wx[4];
        for (int k = 0; k < 4; ++k) {
            wy[k] = cubic_keys(fy - float(k - 1), a);
            wx[k] = cubic_keys(fx - float(k - 1), a);
        }
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int iy = max(0, min(y0 + j - 1, Hi - 1));
            float row = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int ix = max(0, min(x0 + i - 1, Wi - 1));
                row += wx[i] * X[xbase + uint(iy) * P.W_in + uint(ix)];
            }
            acc += wy[j] * row;
        }
        Y[gid] = acc;
    }
}

// ── backward: one thread per input element (n, c, iy, ix) — gather adjoint ──
//
// Walks every output position and checks whether the forward sampling for that
// output landed on (iy, ix), adding the matching weight if so. O(H_out*W_out)
// per input pixel — fine at typical vision-encoder spatial sizes. The dst
// loop iterates in ascending (oh, ow) order, matching the CPU scatter order;
// for nearest the sum is one term, for bilinear up to four taps per output
// pixel may touch this input, summed left-to-right top-to-bottom.
kernel void k_interp2d_backward(device const float* dY [[buffer(0)]],
                                device float*       dX [[buffer(1)]],
                                constant I2dParams& P  [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint ix   = gid % P.W_in;
    uint t1   = gid / P.W_in;
    uint iy   = t1 % P.H_in;
    uint t2   = t1 / P.H_in;
    uint c    = t2 % P.C;
    uint n    = t2 / P.C;
    uint ybase = (n * P.C + c) * P.H_out * P.W_out;
    float sy = float(P.H_in) / float(P.H_out);
    float sx = float(P.W_in) / float(P.W_out);
    int Hi = int(P.H_in), Wi = int(P.W_in);
    int ixi = int(ix), iyi = int(iy);

    float acc = 0.0f;
    for (uint oh = 0u; oh < P.H_out; ++oh) {
        float src_y = (float(oh) + 0.5f) * sy - 0.5f;
        for (uint ow = 0u; ow < P.W_out; ++ow) {
            float src_x = (float(ow) + 0.5f) * sx - 0.5f;
            float g = dY[ybase + oh * P.W_out + ow];
            if (P.mode == 0u) {
                int sy_i = max(0, min(int(rint(src_y)), Hi - 1));
                int sx_i = max(0, min(int(rint(src_x)), Wi - 1));
                if (sy_i == iyi && sx_i == ixi) acc += g;
            } else {
                int y0 = int(floor(src_y));
                int x0 = int(floor(src_x));
                float fy = src_y - float(y0);
                float fx = src_x - float(x0);
                int y0c = max(0, min(y0,     Hi - 1));
                int y1c = max(0, min(y0 + 1, Hi - 1));
                int x0c = max(0, min(x0,     Wi - 1));
                int x1c = max(0, min(x0 + 1, Wi - 1));
                // Sum order must match the CPU scatter (00, 01, 10, 11).
                if (y0c == iyi && x0c == ixi) acc += (1.0f - fy) * (1.0f - fx) * g;
                if (y0c == iyi && x1c == ixi) acc += (1.0f - fy) * fx        * g;
                if (y1c == iyi && x0c == ixi) acc += fy          * (1.0f - fx) * g;
                if (y1c == iyi && x1c == ixi) acc += fy          * fx        * g;
            }
        }
    }
    dX[gid] = acc;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                            \
    }
DEF_PSO(pso_interp2d_forward,  @"k_interp2d_forward")
DEF_PSO(pso_interp2d_backward, @"k_interp2d_backward")
#undef DEF_PSO

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger total,
                void (^binders)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        binders(enc);
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

// ─── Forward ─────────────────────────────────────────────────────────────────
void interp2d_forward(const Tensor& X,
                      int N, int C, int H_in, int W_in,
                      int H_out, int W_out, int mode, Tensor& Y) {
    const char* op = "interp2d_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/true);

    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    I2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H_in = static_cast<uint32_t>(H_in);
    p.W_in = static_cast<uint32_t>(W_in);
    p.H_out = static_cast<uint32_t>(H_out);
    p.W_out = static_cast<uint32_t>(W_out);
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);

    dispatch1d(pso_interp2d_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(I2dParams) atIndex:2];
    });
}

// ─── Forward, align_corners=True (inference resample for DPT / seg heads) ─────
// Reuses the forward kernel; only the source-coordinate map differs (align=1).
void interp2d_align_corners_forward(const Tensor& X,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode, Tensor& Y) {
    const char* op = "interp2d_align_corners_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/true);

    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    I2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H_in = static_cast<uint32_t>(H_in);
    p.W_in = static_cast<uint32_t>(W_in);
    p.H_out = static_cast<uint32_t>(H_out);
    p.W_out = static_cast<uint32_t>(W_out);
    p.mode = static_cast<uint32_t>(mode);
    p.align = 1u;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);

    dispatch1d(pso_interp2d_forward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(I2dParams) atIndex:2];
    });
}

// ─── Backward ────────────────────────────────────────────────────────────────
void interp2d_backward(const Tensor& dY,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode, Tensor& dX) {
    const char* op = "interp2d_backward";
    req_fp32(op, dY, "dY");
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/false);

    const int cols_in = C * H_in * W_in;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    // No upstream gradient -> every input gradient is zero. dY may be an
    // empty tensor with no backing buffer, so don't dispatch the gather.
    if (H_out == 0 || W_out == 0) { dX.zero(); return; }

    I2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H_in = static_cast<uint32_t>(H_in);
    p.W_in = static_cast<uint32_t>(W_in);
    p.H_out = static_cast<uint32_t>(H_out);
    p.W_out = static_cast<uint32_t>(W_out);
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    dispatch1d(pso_interp2d_backward(), p.total,
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(I2dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
