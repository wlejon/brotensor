// ─── Metal upfirdn2d (StyleGAN3-R) ──────────────────────────────────────────
//
// Metal port of src/cuda/upfirdn2d.cu. Upsample (zero-insert) → pad/crop → 2D
// FIR correlation → downsample → gain. General non-separable 2D path mirroring
// NVlabs `_upfirdn2d_ref`. The filter is a constant shared across channels
// (depthwise); there is no gradient to it.
//
// Both forward and backward funnel through `upfirdn2d_run`: the op is linear in
// X, so the backward is itself a forward with up/down swapped, the filter flip
// inverted, and padding recomputed (mirrors `_upfirdn2d_cuda`).
//
// One thread per output element (n,c,oh,ow); the per-tap accumulation runs in
// FP32 regardless of storage dtype (FP32/FP16/BF16 supported).

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

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct UFParams {
    uint total;
    int N, C, Hin, Win, Hout, Wout;
    int fH, fW, up_x, up_y, down_x, down_y;
    int px0, py0, Hu, Wu, flip_filter;
    float gain;
};

#define UPFIRDN_KERNEL(NAME, T)                                               \
kernel void NAME(device const T* In   [[buffer(0)]],                          \
                 device const T* Fp   [[buffer(1)]],                          \
                 device T*       Out  [[buffer(2)]],                          \
                 constant UFParams& p [[buffer(3)]],                          \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= p.total) return;                                                 \
    int ow = int(i % uint(p.Wout));                                           \
    uint t = i / uint(p.Wout);                                                \
    int oh = int(t % uint(p.Hout));                                           \
    t /= uint(p.Hout);                                                        \
    int c = int(t % uint(p.C));                                               \
    int n = int(t / uint(p.C));                                               \
    ulong in_base = ((ulong)n * p.C + c) * p.Hin * p.Win;                     \
    int py_base = oh * p.down_y;                                              \
    int px_base = ow * p.down_x;                                              \
    float acc = 0.0f;                                                         \
    for (int kh = 0; kh < p.fH; ++kh) {                                       \
        int uy = py_base + kh - p.py0;                                        \
        if (uy < 0 || uy >= p.Hu || (uy % p.up_y) != 0) continue;             \
        int iy = uy / p.up_y;                                                 \
        int frow = p.flip_filter ? kh : (p.fH - 1 - kh);                      \
        for (int kw = 0; kw < p.fW; ++kw) {                                   \
            int ux = px_base + kw - p.px0;                                    \
            if (ux < 0 || ux >= p.Wu || (ux % p.up_x) != 0) continue;         \
            int ix = ux / p.up_x;                                             \
            int fcol = p.flip_filter ? kw : (p.fW - 1 - kw);                  \
            acc += float(In[in_base + (ulong)iy * p.Win + ix]) *              \
                   float(Fp[(ulong)frow * p.fW + fcol]);                      \
        }                                                                     \
    }                                                                         \
    ulong out_base = ((ulong)n * p.C + c) * p.Hout * p.Wout;                  \
    Out[out_base + (ulong)oh * p.Wout + ow] = T(acc * p.gain);                \
}

UPFIRDN_KERNEL(k_upfirdn2d_fp32, float)
UPFIRDN_KERNEL(k_upfirdn2d_fp16, half)
UPFIRDN_KERNEL(k_upfirdn2d_bf16, bfloat)
)msl";

struct UFParams {
    uint32_t total;
    int32_t N, C, Hin, Win, Hout, Wout;
    int32_t fH, fW, up_x, up_y, down_x, down_y;
    int32_t px0, py0, Hu, Wu, flip_filter;
    float gain;
};

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_fp32, @"k_upfirdn2d_fp32")
DEF_PSO(pso_fp16, @"k_upfirdn2d_fp16")
DEF_PSO(pso_bf16, @"k_upfirdn2d_bf16")
#undef DEF_PSO

void require_fp(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

// Shared engine. `In` is (N, C*Hin*Win); resizes `Out` to (N, C*Hout*Wout).
void upfirdn2d_run(const Tensor& In, int N, int C, int Hin, int Win,
                   const Tensor& f, int fH, int fW,
                   int up_x, int up_y, int down_x, int down_y,
                   int px0, int px1, int py0, int py1,
                   bool flip_filter, float gain, Tensor& Out, const char* op) {
    require_fp(op, In, "input");
    require_fp(op, f, "f");
    if (f.dtype != In.dtype)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": f.dtype must match input.dtype");
    if (up_x < 1 || up_y < 1 || down_x < 1 || down_y < 1)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": up/down factors must be >= 1");
    if (In.rows != N || In.cols != C * Hin * Win)
        throw std::runtime_error(std::string("brotensor: ") + op + ": input shape mismatch");
    if (f.rows != fH || f.cols != fW)
        throw std::runtime_error(std::string("brotensor: ") + op + ": filter shape mismatch");
    const int Hu = Hin * up_y, Wu = Win * up_x;
    const int Hp = Hu + py0 + py1, Wp = Wu + px0 + px1;
    if (Hp < fH || Wp < fW)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": padded input smaller than filter");
    const int Hc = Hp - fH + 1, Wc = Wp - fW + 1;
    const int Hout = (Hc - 1) / down_y + 1;
    const int Wout = (Wc - 1) / down_x + 1;
    const int out_cols = C * Hout * Wout;
    if (Out.rows != N || Out.cols != out_cols || Out.dtype != In.dtype)
        Out.resize(N, out_cols, In.dtype);
    if (N == 0 || out_cols == 0) return;

    const NSUInteger total = static_cast<NSUInteger>(N) * out_cols;
    UFParams p{static_cast<uint32_t>(total), N, C, Hin, Win, Hout, Wout,
               fH, fW, up_x, up_y, down_x, down_y, px0, py0, Hu, Wu,
               flip_filter ? 1 : 0, gain};
    id<MTLComputePipelineState> pso =
        (In.dtype == Dtype::FP16) ? pso_fp16()
      : (In.dtype == Dtype::BF16) ? pso_bf16() : pso_fp32();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(In)  offset:buffer_offset_for(In)  atIndex:0];
        [enc setBuffer:buffer_for(f)   offset:buffer_offset_for(f)   atIndex:1];
        [enc setBuffer:buffer_for(Out) offset:buffer_offset_for(Out) atIndex:2];
        [enc setBytes:&p length:sizeof(UFParams) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

inline int out_dim(int in, int up, int down, int pad0, int pad1, int fdim) {
    return (in * up + pad0 + pad1 - fdim) / down + 1;
}

} // namespace

void upfirdn2d_forward(const Tensor& X, const Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, Tensor& Y) {
    upfirdn2d_run(X, N, C, H, Wd, f, fH, fW, up_x, up_y, down_x, down_y,
                  pad_x0, pad_x1, pad_y0, pad_y1, flip_filter, gain, Y,
                  "upfirdn2d_forward");
}

void upfirdn2d_backward(const Tensor& dY, const Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, Tensor& dX) {
    const int Hout = out_dim(H,  up_y, down_y, pad_y0, pad_y1, fH);
    const int Wout = out_dim(Wd, up_x, down_x, pad_x0, pad_x1, fW);
    // Backward padding (NVlabs _upfirdn2d_cuda): swap up<->down, flip the flip.
    const int p_x0 = fW - pad_x0 - 1;
    const int p_x1 = Wd * up_x - Wout * down_x + pad_x0 - up_x + 1;
    const int p_y0 = fH - pad_y0 - 1;
    const int p_y1 = H * up_y - Hout * down_y + pad_y0 - up_y + 1;
    upfirdn2d_run(dY, N, C, Hout, Wout, f, fH, fW,
                  /*up=*/down_x, down_y, /*down=*/up_x, up_y,
                  p_x0, p_x1, p_y0, p_y1,
                  !flip_filter, gain, dX, "upfirdn2d_backward");
    if (dX.rows != N || dX.cols != C * H * Wd)
        throw std::runtime_error("upfirdn2d_backward: internal dX shape "
                                 "mismatch (param inconsistency)");
}

} // namespace brotensor::detail::metal
