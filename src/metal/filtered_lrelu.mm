// ─── Metal fused filtered_lrelu (StyleGAN3 alias-free nonlinearity) ─────────
//
// Metal port of src/cuda/filtered_lrelu.cu. The composite is bias_act →
// upfirdn2d(up, gain=up²) → bias_act(lrelu) → upfirdn2d(down): four global
// passes, three of them at the up-sampled (2x-4x) resolution, and it leaves
// two full-size up-rate intermediates (up_buf, act_buf) live per layer. This
// fuses the chain into one kernel that reads X (+bias) and writes Y, with the
// up-FIR, leaky-ReLU/clamp and down-FIR all in registers / threadgroup memory.
//
// Same structure as the CUDA kernel: one threadgroup per (n, c, 16x16 output
// tile). Phase A computes the tile's activation footprint exactly once into
// threadgroup memory using dense polyphase banks of the up filter (built on
// the GPU per call, pre-flipped so both operands of the inner loop are read in
// input order); phase B downsamples from threadgroup memory with the
// pre-flipped dense down filter. FP32 accumulation, storage dtype T for the
// activation tile (FP32 is therefore the same taps in the same order as the
// composite). FP32 / FP16 / BF16.
//
// Cache contract (identical to CUDA): the fused path produces NO up_buf /
// act_buf. A caller that commits either (signalling it wants the caches), an
// unsupported dtype/config, or a tile that would not fit threadgroup memory
// takes the composite instead.
//
// The backward (FP32, dB not requested) is fused too, so it needs no cache: it
// recomputes the pre-activation from X for the lrelu mask (see k_fl_bwd_fp32).
// Other cases take the composite backward, which rebuilds up_buf from X when it
// arrives uncommitted. Together these replace the composite's five up-rate
// passes per layer, and keep a GAN inversion (forward_cached + backward over
// every layer) from holding an up-rate cache per layer.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor {
// Composite fallback (src/filtered_lrelu.cpp).
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

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr int FL_TH = 16;
constexpr int FL_TW = 16;
constexpr int FL_THREADS = FL_TH * FL_TW;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant constexpr int FL_TH = 16;
constant constexpr int FL_TW = 16;
constant constexpr int FL_THREADS = FL_TH * FL_TW;

struct FLPolyParams {
    int fuH, fuW, fdH, fdW, up, JyMax, JxMax, nu, nd;
};

struct FLParams {
    int N, C, H, W, Hout, Wout;
    int JyMax, JxMax, fdH, fdW, up, down, px0, py0, AH, AW, has_bias;
    float up_gain, gain, slope, clamp;
};

// poly_u[(ry*up + rx)][j][i] = fu[fuH-1-(ry+j*up)][fuW-1-(rx+i*up)] (zero past
// the filter extent); poly_d = fd flipped in both axes. idx < nu fills poly_u,
// the rest fills poly_d.
template <typename T>
kernel void k_fl_build_poly(device const T* fu  [[buffer(0)]],
                            device const T* fd  [[buffer(1)]],
                            device T*       pu  [[buffer(2)]],
                            device T*       pd  [[buffer(3)]],
                            constant FLPolyParams& q [[buffer(4)]],
                            uint idx [[thread_position_in_grid]]) {
    if (int(idx) < q.nu) {
        int t = int(idx);
        const int i = t % q.JxMax; t /= q.JxMax;
        const int j = t % q.JyMax; t /= q.JyMax;
        const int rx = t % q.up;   t /= q.up;
        const int ry = t;
        const int fuy = ry + j * q.up, fux = rx + i * q.up;
        pu[idx] = (fuy < q.fuH && fux < q.fuW)
                      ? fu[(q.fuH - 1 - fuy) * q.fuW + (q.fuW - 1 - fux)] : T(0);
        return;
    }
    const int k = int(idx) - q.nu;
    if (k >= q.nd) return;
    const int fdx = k % q.fdW, fdy = k / q.fdW;
    pd[k] = fd[(q.fdH - 1 - fdy) * q.fdW + (q.fdW - 1 - fdx)];
}

template <typename T>
kernel void k_fl_fwd(device const T* X      [[buffer(0)]],
                     device const T* poly_u [[buffer(1)]],
                     device const T* poly_d [[buffer(2)]],
                     device const T* bias   [[buffer(3)]],
                     device T*       Y      [[buffer(4)]],
                     constant FLParams& p   [[buffer(5)]],
                     threadgroup T* act     [[threadgroup(0)]],
                     uint3 tg  [[threadgroup_position_in_grid]],
                     uint  tid [[thread_index_in_threadgroup]]) {
    const int nc = int(tg.z);
    const int c  = nc % p.C;
    const int n  = nc / p.C;
    const int oh0 = int(tg.y) * FL_TH;
    const int ow0 = int(tg.x) * FL_TW;
    const int ay0 = oh0 * p.down;
    const int ax0 = ow0 * p.down;
    const int up = p.up;

    device const T* Xc = X + ((ulong)n * p.C + c) * (ulong)p.H * p.W;
    const float bias_v = p.has_bias ? float(bias[c]) : 0.0f;

    // Phase A: the activation footprint of this output tile, each cell once.
    const int acells = p.AH * p.AW;
    const int bank = p.JyMax * p.JxMax;
    for (int a = int(tid); a < acells; a += FL_THREADS) {
        const int ly = a / p.AW, lx = a - ly * p.AW;
        const int ay = ay0 + ly, ax = ax0 + lx;
        const int ry = (((p.py0 - ay) % up) + up) % up;   // first contributing residue
        const int rx = (((p.px0 - ax) % up) + up) % up;
        const int iyb = (ay + ry - p.py0) / up;           // input row at j = 0 (exact)
        const int ixb = (ax + rx - p.px0) / up;
        const int jlo = iyb < 0 ? -iyb : 0;
        const int jhi = min(p.H - iyb, p.JyMax);
        const int ilo = ixb < 0 ? -ixb : 0;
        const int ihi = min(p.W - ixb, p.JxMax);
        device const T* polbase = poly_u + (ry * up + rx) * bank;
        float u = 0.0f;
        for (int j = jlo; j < jhi; ++j) {
            device const T* xr  = Xc + (ulong)(iyb + j) * p.W + ixb;
            device const T* pol = polbase + j * p.JxMax;
            float s = 0.0f;
            for (int i = ilo; i < ihi; ++i) s += float(pol[i]) * (float(xr[i]) + bias_v);
            u += s;
        }
        u *= p.up_gain;
        float av = (u > 0.0f ? u : p.slope * u) * p.gain;
        if (p.clamp >= 0.0f) av = clamp(av, -p.clamp, p.clamp);
        act[a] = T(av);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: one thread per output pixel of the tile, downsampling from act.
    const int oh = oh0 + int(tid) / FL_TW;
    const int ow = ow0 + int(tid) % FL_TW;
    if (oh < p.Hout && ow < p.Wout) {
        const int ly0 = (oh - oh0) * p.down;
        const int lx0 = (ow - ow0) * p.down;
        float acc = 0.0f;
        for (int fdy = 0; fdy < p.fdH; ++fdy) {
            threadgroup const T* ar = act + (ly0 + fdy) * p.AW + lx0;
            device const T* pol = poly_d + fdy * p.fdW;
            float s = 0.0f;
            for (int i = 0; i < p.fdW; ++i) s += float(pol[i]) * float(ar[i]);
            acc += s;
        }
        Y[((ulong)n * p.C + c) * (ulong)p.Hout * p.Wout + (ulong)oh * p.Wout + ow] = T(acc);
    }
}

// Fused backward (FP32). One threadgroup per (n, c, TH x TW tile of dX).
// Phase 0 stages the tile's input footprints in threadgroup memory: X + bias
// (input rate, zero outside the image — the composite's zero padding) and dY
// (output rate, zero outside). Phase A fills the tile's up-rate footprint
// (RH x RW cells) with
//   dU = dA * gain * lrelu'(U) * [|gain*lrelu(U)| <= clamp]
// where U (the pre-activation) is recomputed with the polyphase up banks as
// the forward does, and dA is the down-FIR transpose of dY. Phase B is the
// up-FIR transpose from threadgroup memory:
//   dX[iy,ix] = up^2 * sum_{m,k} fu[m][k] * dU[up*ly + m][up*lx + k].
struct FLBwdParams {
    int N, C, H, W, Hout, Wout, Huo, Wuo;
    int fuH, fuW, JyMax, JxMax, fdH, fdW, up, down, px0, py0;
    int TH, TW, RH, RW, XH, XW, YH, YW, JdY, JdX, has_bias;
    float up_gain, gain, slope, clamp;
};

// Down-filter transpose banks for the backward: for output-cell residue
// (ry, rx) = (ay mod down, ax mod down), the contributing taps are
// kh = ry + j*down (dY row oh = (ay - ry)/down - j). Stored reversed in j/i so
// the inner loop walks dY forward from ohb = (ay - ry)/down - (JdY - 1):
//   bank[(ry*down + rx)][j'][i'] = fd[fdH-1-kh][fdW-1-kw],
//   kh = ry + (JdY-1-j')*down, kw = rx + (JdX-1-i')*down (zero past fd).
kernel void k_fl_build_dbank(device const float* fd [[buffer(0)]],
                             device float*       db [[buffer(1)]],
                             constant int4& q      [[buffer(2)]],   // fdH, fdW, down, JdY
                             constant int& JdX     [[buffer(3)]],
                             uint idx [[thread_position_in_grid]]) {
    const int fdH = q.x, fdW = q.y, down = q.z, JdY = q.w;
    const int total = down * down * JdY * JdX;
    if (int(idx) >= total) return;
    int t = int(idx);
    const int ip = t % JdX; t /= JdX;
    const int jp = t % JdY; t /= JdY;
    const int rx = t % down; t /= down;
    const int ry = t;
    const int kh = ry + (JdY - 1 - jp) * down, kw = rx + (JdX - 1 - ip) * down;
    db[idx] = (kh < fdH && kw < fdW) ? fd[(fdH - 1 - kh) * fdW + (fdW - 1 - kw)] : 0.0f;
}

inline int fl_ceil_div(int a, int b) { return a >= 0 ? (a + b - 1) / b : -((-a) / b); }
inline int fl_floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

kernel void k_fl_bwd_fp32(device const float* dY     [[buffer(0)]],
                          device const float* X      [[buffer(1)]],
                          device const float* poly_u [[buffer(2)]],
                          device const float* fu     [[buffer(3)]],
                          device const float* dbank  [[buffer(4)]],
                          device const float* bias   [[buffer(5)]],
                          device float*       dX     [[buffer(6)]],
                          constant FLBwdParams& p    [[buffer(7)]],
                          threadgroup float* smem    [[threadgroup(0)]],
                          uint3 tg  [[threadgroup_position_in_grid]],
                          uint  tid [[thread_index_in_threadgroup]]) {
    const int nc = int(tg.z);
    const int c  = nc % p.C;
    const int n  = nc / p.C;
    const int iy0 = int(tg.y) * p.TH;
    const int ix0 = int(tg.x) * p.TW;
    const int up = p.up, down = p.down;
    // Up-rate origin of the footprint: dX[iy] reads dU[up*iy + py0 - kh],
    // kh in [0, fuH).
    const int uy0 = up * iy0 + p.py0 - (p.fuH - 1);
    const int ux0 = up * ix0 + p.px0 - (p.fuW - 1);
    // Input-rate origin of the X footprint (the first iyb of the region) and
    // output-rate origin of the dY footprint (the first oh any cell reads).
    const int xy0 = fl_ceil_div(uy0 - p.py0, up);
    const int xx0 = fl_ceil_div(ux0 - p.px0, up);
    const int oy0 = fl_floor_div(uy0, down) - (p.JdY - 1);
    const int ox0 = fl_floor_div(ux0, down) - (p.JdX - 1);

    threadgroup float* dU = smem;
    threadgroup float* Xs = dU + p.RH * p.RW;
    threadgroup float* Ys = Xs + p.XH * p.XW;

    device const float* Xc  = X  + ((ulong)n * p.C + c) * (ulong)p.H * p.W;
    device const float* dYc = dY + ((ulong)n * p.C + c) * (ulong)p.Hout * p.Wout;
    const float bias_v = p.has_bias ? bias[c] : 0.0f;

    for (int a = int(tid); a < p.XH * p.XW; a += FL_THREADS) {
        const int r = a / p.XW, q = a - r * p.XW;
        const int iy = xy0 + r, ix = xx0 + q;
        Xs[a] = (iy >= 0 && iy < p.H && ix >= 0 && ix < p.W)
                    ? Xc[(ulong)iy * p.W + ix] + bias_v : 0.0f;
    }
    for (int a = int(tid); a < p.YH * p.YW; a += FL_THREADS) {
        const int r = a / p.YW, q = a - r * p.YW;
        const int oh = oy0 + r, ow = ox0 + q;
        Ys[a] = (oh >= 0 && oh < p.Hout && ow >= 0 && ow < p.Wout)
                    ? dYc[(ulong)oh * p.Wout + ow] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int bank = p.JyMax * p.JxMax;
    const int cells = p.RH * p.RW;
    for (int a = int(tid); a < cells; a += FL_THREADS) {
        const int ly = a / p.RW, lx = a - ly * p.RW;
        const int ay = uy0 + ly, ax = ux0 + lx;
        float du = 0.0f;
        if (ay >= 0 && ay < p.Huo && ax >= 0 && ax < p.Wuo) {
            // dA: the transpose-bank dot with the staged (zero-padded) dY.
            const int ry_d = ay % down, rx_d = ax % down;
            threadgroup const float* yb =
                Ys + ((ay - ry_d) / down - (p.JdY - 1) - oy0) * p.YW +
                     ((ax - rx_d) / down - (p.JdX - 1) - ox0);
            device const float* db = dbank + (ry_d * down + rx_d) * (p.JdY * p.JdX);
            float da = 0.0f;
            for (int j = 0; j < p.JdY; ++j) {
                threadgroup const float* yr = yb + j * p.YW;
                device const float* dr = db + j * p.JdX;
                float s = 0.0f;
                for (int i = 0; i < p.JdX; ++i) s += dr[i] * yr[i];
                da += s;
            }
            // U: the forward's phase A for this cell, from the staged X + b.
            const int ry = (((p.py0 - ay) % up) + up) % up;
            const int rx = (((p.px0 - ax) % up) + up) % up;
            const int iyb = (ay + ry - p.py0) / up;
            const int ixb = (ax + rx - p.px0) / up;
            device const float* polbase = poly_u + (ry * up + rx) * bank;
            threadgroup const float* xb = Xs + (iyb - xy0) * p.XW + (ixb - xx0);
            float u = 0.0f;
            for (int j = 0; j < p.JyMax; ++j) {
                threadgroup const float* xr = xb + j * p.XW;
                device const float* pol = polbase + j * p.JxMax;
                float s = 0.0f;
                for (int i = 0; i < p.JxMax; ++i) s += pol[i] * xr[i];
                u += s;
            }
            u *= p.up_gain;
            du = da * p.gain * (u > 0.0f ? 1.0f : p.slope);
            if (p.clamp >= 0.0f) {
                const float y_pre = p.gain * (u > 0.0f ? u : p.slope * u);
                if (y_pre < -p.clamp || y_pre > p.clamp) du = 0.0f;
            }
        }
        dU[a] = du;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B. With fewer outputs than threads (the 8x8 tiles), P adjacent
    // lanes share one output, each summing every P-th filter row, and are
    // combined with a butterfly (P divides the 32-lane SIMD group).
    const int outs = p.TH * p.TW;
    const int P = outs >= FL_THREADS ? 1 : FL_THREADS / outs;
    const int part = int(tid) % P;
    for (int o = int(tid) / P; o < outs; o += FL_THREADS / P) {
        const int ly = o / p.TW, lx = o - ly * p.TW;
        const int iy = iy0 + ly, ix = ix0 + lx;
        float acc = 0.0f;
        for (int m = part; m < p.fuH; m += P) {
            threadgroup const float* dr = dU + (up * ly + m) * p.RW + up * lx;
            device const float* fr = fu + m * p.fuW;
            float s = 0.0f;
            for (int k = 0; k < p.fuW; ++k) s += fr[k] * dr[k];
            acc += s;
        }
        for (int off = 1; off < P; off <<= 1) acc += simd_shuffle_xor(acc, ushort(off));
        if (part == 0 && iy < p.H && ix < p.W)
            dX[((ulong)n * p.C + c) * (ulong)p.H * p.W + (ulong)iy * p.W + ix] = acc * p.up_gain;
    }
}

#define FL_INST(T, SUF)                                                                        \
template [[host_name("k_fl_build_poly_" SUF)]] kernel void k_fl_build_poly<T>(                 \
    device const T*, device const T*, device T*, device T*, constant FLPolyParams&, uint);     \
template [[host_name("k_fl_fwd_" SUF)]] kernel void k_fl_fwd<T>(                               \
    device const T*, device const T*, device const T*, device const T*, device T*,             \
    constant FLParams&, threadgroup T*, uint3, uint);
FL_INST(float, "fp32")
FL_INST(half, "fp16")
FL_INST(bfloat, "bf16")
)msl";

// Must match the MSL structs byte-for-byte.
struct FLPolyParams {
    int32_t fuH, fuW, fdH, fdW, up, JyMax, JxMax, nu, nd;
};
struct FLParams {
    int32_t N, C, H, W, Hout, Wout;
    int32_t JyMax, JxMax, fdH, fdW, up, down, px0, py0, AH, AW, has_bias;
    float up_gain, gain, slope, clamp;
};

struct FLBwdParams {
    int32_t N, C, H, W, Hout, Wout, Huo, Wuo;
    int32_t fuH, fuW, JyMax, JxMax, fdH, fdW, up, down, px0, py0;
    int32_t TH, TW, RH, RW, XH, XW, YH, YW, JdY, JdX, has_bias;
    float up_gain, gain, slope, clamp;
};

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_poly_fp32, @"k_fl_build_poly_fp32")
DEF_PSO(pso_poly_fp16, @"k_fl_build_poly_fp16")
DEF_PSO(pso_poly_bf16, @"k_fl_build_poly_bf16")
DEF_PSO(pso_fwd_fp32, @"k_fl_fwd_fp32")
DEF_PSO(pso_fwd_fp16, @"k_fl_fwd_fp16")
DEF_PSO(pso_fwd_bf16, @"k_fl_fwd_bf16")
DEF_PSO(pso_bwd_fp32, @"k_fl_bwd_fp32")
DEF_PSO(pso_dbank_fp32, @"k_fl_build_dbank")
#undef DEF_PSO

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

    const int AH = (FL_TH - 1) * down + fdH;   // threadgroup activation tile extent
    const int AW = (FL_TW - 1) * down + fdW;
    const size_t tg_bytes = static_cast<size_t>(AH) * AW * dtype_size_bytes(X.dtype);

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool supported =
        is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) && up >= 1 && down >= 1 &&
        Huo >= fdH && Wuo >= fdW && Hout > 0 && Wout > 0 &&
        tg_bytes <= 16u * 1024u &&
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

    FLPolyParams q{};
    q.fuH = fuH; q.fuW = fuW; q.fdH = fdH; q.fdW = fdW; q.up = up;
    q.JyMax = (fuH + up - 1) / up;
    q.JxMax = (fuW + up - 1) / up;
    q.nu = up * up * q.JyMax * q.JxMax;
    q.nd = fdH * fdW;
    // Transient polyphase banks; empty_on (no zero-fill, so no queue flush) —
    // the build kernel writes every element before the main kernel reads it.
    Tensor poly_u = Tensor::empty_on(Device::Metal, 1, q.nu, X.dtype);
    Tensor poly_d = Tensor::empty_on(Device::Metal, 1, q.nd, X.dtype);

    FLParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W; p.Hout = Hout; p.Wout = Wout;
    p.JyMax = q.JyMax; p.JxMax = q.JxMax; p.fdH = fdH; p.fdW = fdW;
    p.up = up; p.down = down; p.px0 = pad_x0; p.py0 = pad_y0; p.AH = AH; p.AW = AW;
    p.has_bias = b ? 1 : 0;
    p.up_gain = static_cast<float>(up) * static_cast<float>(up);
    p.gain = gain; p.slope = slope; p.clamp = clamp;

    const bool h = X.dtype == Dtype::FP16, bf = X.dtype == Dtype::BF16;
    id<MTLComputePipelineState> ppoly = h ? pso_poly_fp16() : bf ? pso_poly_bf16() : pso_poly_fp32();
    id<MTLComputePipelineState> pfwd  = h ? pso_fwd_fp16()  : bf ? pso_fwd_bf16()  : pso_fwd_fp32();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ppoly];
        [enc setBuffer:buffer_for(fu) offset:buffer_offset_for(fu) atIndex:0];
        [enc setBuffer:buffer_for(fd) offset:buffer_offset_for(fd) atIndex:1];
        [enc setBuffer:buffer_for(poly_u) offset:buffer_offset_for(poly_u) atIndex:2];
        [enc setBuffer:buffer_for(poly_d) offset:buffer_offset_for(poly_d) atIndex:3];
        [enc setBytes:&q length:sizeof(q) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(q.nu + q.nd), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:pfwd];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(poly_u) offset:buffer_offset_for(poly_u) atIndex:1];
        [enc setBuffer:buffer_for(poly_d) offset:buffer_offset_for(poly_d) atIndex:2];
        if (b) [enc setBuffer:buffer_for(*b) offset:buffer_offset_for(*b) atIndex:3];
        else   [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:3];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc setThreadgroupMemoryLength:((tg_bytes + 15) / 16) * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((Wout + FL_TW - 1) / FL_TW),
                                              static_cast<NSUInteger>((Hout + FL_TH - 1) / FL_TH),
                                              static_cast<NSUInteger>(N) * C)
            threadsPerThreadgroup:MTLSizeMake(FL_THREADS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void filtered_lrelu_backward(const Tensor& dY, const Tensor& X,
                             const Tensor& fu, const Tensor& fd,
                             const Tensor* b, int N, int C, int H, int W,
                             int up, int down, int pad_x0, int pad_x1,
                             int pad_y0, int pad_y1, float gain, float slope,
                             float clamp, const Tensor& up_buf,
                             Tensor& dX, Tensor* dB) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;
    const int Huo = H * up + pad_y0 + pad_y1 - fuH + 1;
    const int Wuo = W * up + pad_x0 + pad_x1 - fuW + 1;
    const int Hout = down >= 1 ? (Huo - fdH) / down + 1 : 0;
    const int Wout = down >= 1 ? (Wuo - fdW) / down + 1 : 0;

    // Tile: 16x16 dX outputs unless the up-rate footprint would not fit, then
    // 8x8 (the 4x-upsampling layers' 24-tap filters).
    // Threadgroup footprint for a T x T tile: the up-rate dU region plus the
    // staged X (input rate) and dY (output rate) windows it reads. The window
    // extents are upper bounds (any tile origin fits).
    const int JyMax = (fuH + up - 1) / up, JxMax = (fuW + up - 1) / up;
    const int JdY = (fdH + down - 1) / down, JdX = (fdW + down - 1) / down;
    auto dims = [&](int T, int& RH, int& RW, int& XH, int& XW, int& YH, int& YW) {
        RH = up * (T - 1) + fuH;          RW = up * (T - 1) + fuW;
        XH = (RH - 1) / up + 2 + JyMax;   XW = (RW - 1) / up + 2 + JxMax;
        YH = (RH - 1) / down + 2 + JdY;   YW = (RW - 1) / down + 2 + JdX;
    };
    auto region = [&](int T) {
        int RH, RW, XH, XW, YH, YW;
        dims(T, RH, RW, XH, XW, YH, YW);
        return static_cast<size_t>(RH * RW + XH * XW + YH * YW) * sizeof(float);
    };
    const int T = region(16) <= 20u * 1024u ? 16 : 8;
    const bool supported =
        X.dtype == Dtype::FP32 && dY.dtype == Dtype::FP32 && fu.dtype == Dtype::FP32 &&
        fd.dtype == Dtype::FP32 && (!b || b->dtype == Dtype::FP32) && dB == nullptr &&
        up >= 1 && down >= 1 && Huo >= fdH && Wuo >= fdW && Hout > 0 && Wout > 0 &&
        region(T) <= 20u * 1024u && static_cast<long long>(N) * C <= 65535;
    if (!supported) {
        ::brotensor::filtered_lrelu_backward_composite(
            dY, X, fu, fd, b, N, C, H, W, up, down, pad_x0, pad_x1, pad_y0, pad_y1,
            gain, slope, clamp, up_buf, dX, dB);
        return;
    }
    if (X.rows != N || X.cols != C * H * W)
        throw std::runtime_error("brotensor: filtered_lrelu_backward: X shape mismatch");
    if (dY.rows != N || dY.cols != C * Hout * Wout)
        throw std::runtime_error("brotensor: filtered_lrelu_backward: dY shape mismatch");
    if (dX.rows != N || dX.cols != C * H * W || dX.dtype != Dtype::FP32)
        dX.resize(N, C * H * W, Dtype::FP32);
    if (N == 0 || C == 0) return;

    FLPolyParams q{};
    q.fuH = fuH; q.fuW = fuW; q.fdH = fdH; q.fdW = fdW; q.up = up;
    q.JyMax = (fuH + up - 1) / up;
    q.JxMax = (fuW + up - 1) / up;
    q.nu = up * up * q.JyMax * q.JxMax;
    q.nd = fdH * fdW;
    Tensor poly_u = Tensor::empty_on(Device::Metal, 1, q.nu, Dtype::FP32);
    Tensor poly_d = Tensor::empty_on(Device::Metal, 1, q.nd, Dtype::FP32);
    const int nb = down * down * JdY * JdX;
    Tensor dbank = Tensor::empty_on(Device::Metal, 1, nb, Dtype::FP32);

    FLBwdParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W; p.Hout = Hout; p.Wout = Wout;
    p.Huo = Huo; p.Wuo = Wuo;
    p.fuH = fuH; p.fuW = fuW; p.JyMax = q.JyMax; p.JxMax = q.JxMax;
    p.fdH = fdH; p.fdW = fdW; p.up = up; p.down = down;
    p.px0 = pad_x0; p.py0 = pad_y0;
    p.TH = T; p.TW = T;
    dims(T, p.RH, p.RW, p.XH, p.XW, p.YH, p.YW);
    p.JdY = JdY; p.JdX = JdX;
    p.has_bias = b ? 1 : 0;
    p.up_gain = static_cast<float>(up) * static_cast<float>(up);
    p.gain = gain; p.slope = slope; p.clamp = clamp;
    const size_t tg_bytes = region(T);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_poly_fp32()];
        [enc setBuffer:buffer_for(fu) offset:buffer_offset_for(fu) atIndex:0];
        [enc setBuffer:buffer_for(fd) offset:buffer_offset_for(fd) atIndex:1];
        [enc setBuffer:buffer_for(poly_u) offset:buffer_offset_for(poly_u) atIndex:2];
        [enc setBuffer:buffer_for(poly_d) offset:buffer_offset_for(poly_d) atIndex:3];
        [enc setBytes:&q length:sizeof(q) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(q.nu + q.nd), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        const int32_t dq[4] = {fdH, fdW, down, JdY};
        [enc setComputePipelineState:pso_dbank_fp32()];
        [enc setBuffer:buffer_for(fd) offset:buffer_offset_for(fd) atIndex:0];
        [enc setBuffer:buffer_for(dbank) offset:buffer_offset_for(dbank) atIndex:1];
        [enc setBytes:dq length:sizeof(dq) atIndex:2];
        [enc setBytes:&JdX length:sizeof(int32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nb), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:pso_bwd_fp32()];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:1];
        [enc setBuffer:buffer_for(poly_u) offset:buffer_offset_for(poly_u) atIndex:2];
        [enc setBuffer:buffer_for(fu) offset:buffer_offset_for(fu) atIndex:3];
        [enc setBuffer:buffer_for(dbank) offset:buffer_offset_for(dbank) atIndex:4];
        if (b) [enc setBuffer:buffer_for(*b) offset:buffer_offset_for(*b) atIndex:5];
        else   [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:5];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:6];
        [enc setBytes:&p length:sizeof(p) atIndex:7];
        [enc setThreadgroupMemoryLength:((tg_bytes + 15) / 16) * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((W + T - 1) / T),
                                              static_cast<NSUInteger>((H + T - 1) / T),
                                              static_cast<NSUInteger>(N) * C)
            threadsPerThreadgroup:MTLSizeMake(FL_THREADS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
