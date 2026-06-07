// ─── Metal LSTM cell (ops/lstm.h) ───────────────────────────────────────────
//
// Metal port of src/cuda/lstm.cu — single-layer, single-direction, PyTorch
// nn.LSTM layout: W_ih (4H,I), W_hh (4H,H), gate-row blocks ordered [input,
// forget, cell, output]; gate g uses tanh, the rest sigmoid. FP32-only,
// matching the CPU/CUDA contract.
//
// The recurrence is sequential in time, so the host loops over timesteps and
// submits one kernel per step; steps serialize on the queue (step t reads the
// hidden/cell state step t-1 wrote). Within a step the (B*H) units are
// independent and run one-thread-per-unit.
//
// Backward scatters into shared parameter grads / dX / the recurrent carry with
// cross-(b)/cross-(n) collisions. Metal lacks a guaranteed native float
// atomic-add, so we accumulate with a compare-exchange loop on atomic_uint
// (bit-cast), which is correct on every Metal GPU. The per-unit cell carry has
// no cross-n coupling and is written plainly.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger LSTM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float safe_tanh(float x) { return tanh(clamp(x, -9.0f, 9.0f)); }

inline void atomic_add_f(device atomic_uint* p, float v) {
    uint old = atomic_load_explicit(p, memory_order_relaxed);
    uint cur;
    do {
        float f = as_type<float>(old) + v;
        cur = as_type<uint>(f);
    } while (!atomic_compare_exchange_weak_explicit(
                p, &old, cur, memory_order_relaxed, memory_order_relaxed));
}

struct LParams {
    int t, B, I, H;
    uint has_bih, has_bhh, has_h0, has_c0;
};

kernel void k_lstm_fwd(device const float* xp   [[buffer(0)]],
                       device const float* wih  [[buffer(1)]],
                       device const float* whh  [[buffer(2)]],
                       device const float* bih  [[buffer(3)]],
                       device const float* bhh  [[buffer(4)]],
                       device const float* h0p  [[buffer(5)]],
                       device const float* c0p  [[buffer(6)]],
                       device float*       yp   [[buffer(7)]],
                       device float*       gp   [[buffer(8)]],
                       device float*       cp   [[buffer(9)]],
                       constant LParams& p       [[buffer(10)]],
                       uint idx [[thread_position_in_grid]]) {
    int B = p.B, I = p.I, H = p.H, t = p.t;
    int BH = B * H;
    if (int(idx) >= BH) return;
    int b = int(idx) / H;
    int n = int(idx) % H;
    int G = 4 * H;
    ulong row = (ulong)t * B + b;

    device const float* x = xp + row * I;
    device const float* hprev = (t == 0)
        ? (p.has_h0 ? h0p + (ulong)b * H : (device const float*)0)
        : yp + ((ulong)(t - 1) * B + b) * H;
    device const float* cprev = (t == 0)
        ? (p.has_c0 ? c0p + (ulong)b * H : (device const float*)0)
        : cp + ((ulong)(t - 1) * B + b) * H;

    float zi = p.has_bih ? bih[n]         : 0.0f;
    float zf = p.has_bih ? bih[H + n]     : 0.0f;
    float zg = p.has_bih ? bih[2 * H + n] : 0.0f;
    float zo = p.has_bih ? bih[3 * H + n] : 0.0f;
    if (p.has_bhh) { zi += bhh[n]; zf += bhh[H + n]; zg += bhh[2 * H + n]; zo += bhh[3 * H + n]; }

    device const float* wi = wih + (ulong)n * I;
    device const float* wf = wih + (ulong)(H + n) * I;
    device const float* wg = wih + (ulong)(2 * H + n) * I;
    device const float* wo = wih + (ulong)(3 * H + n) * I;
    for (int j = 0; j < I; ++j) {
        float xj = x[j];
        zi += wi[j] * xj; zf += wf[j] * xj; zg += wg[j] * xj; zo += wo[j] * xj;
    }
    if (hprev != 0) {
        device const float* ui = whh + (ulong)n * H;
        device const float* uf = whh + (ulong)(H + n) * H;
        device const float* ug = whh + (ulong)(2 * H + n) * H;
        device const float* uo = whh + (ulong)(3 * H + n) * H;
        for (int m = 0; m < H; ++m) {
            float hm = hprev[m];
            zi += ui[m] * hm; zf += uf[m] * hm; zg += ug[m] * hm; zo += uo[m] * hm;
        }
    }

    float ig = sigmoidf(zi);
    float fg = sigmoidf(zf);
    float gg = safe_tanh(zg);
    float og = sigmoidf(zo);
    float cprev_n = (cprev != 0) ? cprev[n] : 0.0f;
    float cn = fg * cprev_n + ig * gg;

    device float* grow = gp + row * G;
    grow[n] = ig; grow[H + n] = fg; grow[2 * H + n] = gg; grow[3 * H + n] = og;
    cp[row * H + n] = cn;
    yp[row * H + n] = og * safe_tanh(cn);
}

kernel void k_lstm_bwd(device const float* xp   [[buffer(0)]],
                       device const float* yp   [[buffer(1)]],
                       device const float* gp   [[buffer(2)]],
                       device const float* cp   [[buffer(3)]],
                       device const float* dyp  [[buffer(4)]],
                       device const float* wih  [[buffer(5)]],
                       device const float* whh  [[buffer(6)]],
                       device const float* h0p  [[buffer(7)]],
                       device const float* c0p  [[buffer(8)]],
                       device const float* dh_in [[buffer(9)]],
                       device atomic_uint* dh_out [[buffer(10)]],
                       device float*       dc   [[buffer(11)]],
                       device atomic_uint* dxp  [[buffer(12)]],
                       device atomic_uint* dwih [[buffer(13)]],
                       device atomic_uint* dwhh [[buffer(14)]],
                       device atomic_uint* dbih [[buffer(15)]],
                       device atomic_uint* dbhh [[buffer(16)]],
                       constant LParams& p       [[buffer(17)]],
                       constant uint& has_dbih   [[buffer(18)]],
                       constant uint& has_dbhh   [[buffer(19)]],
                       uint idx [[thread_position_in_grid]]) {
    int B = p.B, I = p.I, H = p.H, t = p.t;
    int BH = B * H;
    if (int(idx) >= BH) return;
    int b = int(idx) / H;
    int n = int(idx) % H;
    int G = 4 * H;
    ulong row = (ulong)t * B + b;

    device const float* grow = gp + row * G;
    float ig = grow[n], fg = grow[H + n], gg = grow[2 * H + n], og = grow[3 * H + n];
    float tc = safe_tanh(cp[row * H + n]);

    float dh = dyp[row * H + n] + dh_in[(ulong)b * H + n];
    float do_ = dh * tc;
    float dcv = dh * og * (1.0f - tc * tc) + dc[(ulong)b * H + n];

    device const float* cprev = (t == 0)
        ? (p.has_c0 ? c0p + (ulong)b * H : (device const float*)0)
        : cp + ((ulong)(t - 1) * B + b) * H;
    float cprev_n = (cprev != 0) ? cprev[n] : 0.0f;
    float df = dcv * cprev_n;
    float di = dcv * gg;
    float dg = dcv * ig;
    float dcprev = dcv * fg;

    float dzi = di * ig * (1.0f - ig);
    float dzf = df * fg * (1.0f - fg);
    float dzg = dg * (1.0f - gg * gg);
    float dzo = do_ * og * (1.0f - og);

    if (has_dbih) { atomic_add_f(&dbih[n], dzi); atomic_add_f(&dbih[H + n], dzf); atomic_add_f(&dbih[2 * H + n], dzg); atomic_add_f(&dbih[3 * H + n], dzo); }
    if (has_dbhh) { atomic_add_f(&dbhh[n], dzi); atomic_add_f(&dbhh[H + n], dzf); atomic_add_f(&dbhh[2 * H + n], dzg); atomic_add_f(&dbhh[3 * H + n], dzo); }

    device const float* x = xp + row * I;
    device const float* wi = wih + (ulong)n * I;
    device const float* wf = wih + (ulong)(H + n) * I;
    device const float* wg = wih + (ulong)(2 * H + n) * I;
    device const float* wo = wih + (ulong)(3 * H + n) * I;
    device atomic_uint* dxrow = dxp + row * I;
    device atomic_uint* dwi = dwih + (ulong)n * I;
    device atomic_uint* dwf = dwih + (ulong)(H + n) * I;
    device atomic_uint* dwg = dwih + (ulong)(2 * H + n) * I;
    device atomic_uint* dwo = dwih + (ulong)(3 * H + n) * I;
    for (int j = 0; j < I; ++j) {
        atomic_add_f(&dxrow[j], wi[j] * dzi + wf[j] * dzf + wg[j] * dzg + wo[j] * dzo);
        float xj = x[j];
        atomic_add_f(&dwi[j], dzi * xj); atomic_add_f(&dwf[j], dzf * xj);
        atomic_add_f(&dwg[j], dzg * xj); atomic_add_f(&dwo[j], dzo * xj);
    }

    device const float* hprev = (t == 0)
        ? (p.has_h0 ? h0p + (ulong)b * H : (device const float*)0)
        : yp + ((ulong)(t - 1) * B + b) * H;
    device const float* ui = whh + (ulong)n * H;
    device const float* uf = whh + (ulong)(H + n) * H;
    device const float* ug = whh + (ulong)(2 * H + n) * H;
    device const float* uo = whh + (ulong)(3 * H + n) * H;
    device atomic_uint* dui = dwhh + (ulong)n * H;
    device atomic_uint* duf = dwhh + (ulong)(H + n) * H;
    device atomic_uint* dug = dwhh + (ulong)(2 * H + n) * H;
    device atomic_uint* duo = dwhh + (ulong)(3 * H + n) * H;
    device atomic_uint* dh_out_b = dh_out + (ulong)b * H;
    for (int m = 0; m < H; ++m) {
        atomic_add_f(&dh_out_b[m], ui[m] * dzi + uf[m] * dzf + ug[m] * dzg + uo[m] * dzo);
        if (hprev != 0) {
            float hm = hprev[m];
            atomic_add_f(&dui[m], dzi * hm); atomic_add_f(&duf[m], dzf * hm);
            atomic_add_f(&dug[m], dzg * hm); atomic_add_f(&duo[m], dzo * hm);
        }
    }

    dc[(ulong)b * H + n] = dcprev;  // per-unit, no cross-n coupling
}

// Copy `n` floats from src (bound at its own offset) to dst.
kernel void k_lstm_copy(device const float* src [[buffer(0)]],
                        device float*       dst [[buffer(1)]],
                        constant uint& n         [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = src[i];
}
)msl";

struct LParams {
    int32_t t, B, I, H;
    uint32_t has_bih, has_bhh, has_h0, has_c0;
};

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_fwd, @"k_lstm_fwd")
DEF_PSO(pso_bwd, @"k_lstm_bwd")
DEF_PSO(pso_copy, @"k_lstm_copy")
#undef DEF_PSO

[[noreturn]] void fail(const char* op, const std::string& r) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + r);
}
void require_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32 (lstm is FP32-only)");
}
const Tensor* opt(const char* op, const Tensor* t, const char* name, int rows, int cols) {
    if (!t || t->empty()) return nullptr;
    require_fp32(op, *t, name);
    if (t->rows != rows || t->cols != cols) fail(op, std::string(name) + " has wrong shape");
    return t;
}

// dummy bind helper for an absent optional buffer
id<MTLBuffer> opt_buf(const Tensor* t, const Tensor& fallback) {
    return t ? buffer_for(*t) : buffer_for(fallback);
}
NSUInteger opt_off(const Tensor* t, const Tensor& fallback) {
    return t ? buffer_offset_for(*t) : buffer_offset_for(fallback);
}

void copy_floats(const Tensor& src, NSUInteger src_byte_off, Tensor& dst, int n) {
    if (n == 0) return;
    const uint32_t nu = static_cast<uint32_t>(n);
    id<MTLComputePipelineState> pso = pso_copy();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(src) offset:src_byte_off atIndex:0];
        [enc setBuffer:buffer_for(dst) offset:buffer_offset_for(dst) atIndex:1];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:2];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                        const Tensor* b_ih, const Tensor* b_hh,
                        const Tensor* h0, const Tensor* c0, int T, int B,
                        Tensor& Y, Tensor& gates, Tensor& C,
                        Tensor* hT, Tensor* cT) {
    const char* op = "lstm_forward_train";
    require_fp32(op, X, "X");
    require_fp32(op, W_ih, "W_ih");
    require_fp32(op, W_hh, "W_hh");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");

    const Tensor* bih = opt(op, b_ih, "b_ih", G, 1);
    const Tensor* bhh = opt(op, b_hh, "b_hh", G, 1);
    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);

    Y.resize(T * B, H, Dtype::FP32);
    gates.resize(T * B, G, Dtype::FP32);
    C.resize(T * B, H, Dtype::FP32);

    id<MTLComputePipelineState> pso = pso_fwd();
    for (int t = 0; t < T; ++t) {
        LParams p{t, B, I, H, bih ? 1u : 0u, bhh ? 1u : 0u, h0p ? 1u : 0u, c0p ? 1u : 0u};
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:buffer_for(X)    offset:buffer_offset_for(X)    atIndex:0];
            [enc setBuffer:buffer_for(W_ih) offset:buffer_offset_for(W_ih) atIndex:1];
            [enc setBuffer:buffer_for(W_hh) offset:buffer_offset_for(W_hh) atIndex:2];
            [enc setBuffer:opt_buf(bih, X) offset:opt_off(bih, X) atIndex:3];
            [enc setBuffer:opt_buf(bhh, X) offset:opt_off(bhh, X) atIndex:4];
            [enc setBuffer:opt_buf(h0p, X) offset:opt_off(h0p, X) atIndex:5];
            [enc setBuffer:opt_buf(c0p, X) offset:opt_off(c0p, X) atIndex:6];
            [enc setBuffer:buffer_for(Y)     offset:buffer_offset_for(Y)     atIndex:7];
            [enc setBuffer:buffer_for(gates) offset:buffer_offset_for(gates) atIndex:8];
            [enc setBuffer:buffer_for(C)     offset:buffer_offset_for(C)     atIndex:9];
            [enc setBytes:&p length:sizeof(LParams) atIndex:10];
            [enc dispatchThreads:MTLSizeMake(B * H, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(LSTM_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
    }

    const size_t bh = static_cast<size_t>(B) * H;
    if (hT) {
        hT->resize(B, H, Dtype::FP32);
        copy_floats(Y, buffer_offset_for(Y) + (NSUInteger)((size_t)(T - 1) * bh * sizeof(float)), *hT, B * H);
    }
    if (cT) {
        cT->resize(B, H, Dtype::FP32);
        copy_floats(C, buffer_offset_for(C) + (NSUInteger)((size_t)(T - 1) * bh * sizeof(float)), *cT, B * H);
    }
}

void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                   const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C,
                   const Tensor& dY, int T, int B,
                   Tensor& dX, Tensor& dW_ih, Tensor& dW_hh,
                   Tensor* db_ih, Tensor* db_hh, Tensor* dh0, Tensor* dc0) {
    const char* op = "lstm_backward";
    require_fp32(op, X, "X");
    require_fp32(op, W_ih, "W_ih");
    require_fp32(op, W_hh, "W_hh");
    require_fp32(op, Y, "Y");
    require_fp32(op, gates, "gates");
    require_fp32(op, C, "C");
    require_fp32(op, dY, "dY");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");
    if (dY.rows != T * B || dY.cols != H) fail(op, "dY must be (T*B, H)");

    require_fp32(op, dW_ih, "dW_ih");
    require_fp32(op, dW_hh, "dW_hh");
    if (dW_ih.rows != G || dW_ih.cols != I) fail(op, "dW_ih must be (4H, I), zeroed");
    if (dW_hh.rows != G || dW_hh.cols != H) fail(op, "dW_hh must be (4H, H), zeroed");
    if (db_ih) { require_fp32(op, *db_ih, "db_ih"); if (db_ih->rows != G || db_ih->cols != 1) fail(op, "db_ih must be (4H, 1), zeroed"); }
    if (db_hh) { require_fp32(op, *db_hh, "db_hh"); if (db_hh->rows != G || db_hh->cols != 1) fail(op, "db_hh must be (4H, 1), zeroed"); }

    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);

    dX.resize(T * B, I, Dtype::FP32);
    dX.zero();  // overwritten via accumulating scatter

    Tensor dhA = Tensor::zeros_on(Device::Metal, B, H, Dtype::FP32);
    Tensor dhB = Tensor::zeros_on(Device::Metal, B, H, Dtype::FP32);
    Tensor dcBuf = Tensor::zeros_on(Device::Metal, B, H, Dtype::FP32);

    id<MTLComputePipelineState> pso = pso_bwd();
    const uint32_t has_dbih = db_ih ? 1u : 0u;
    const uint32_t has_dbhh = db_hh ? 1u : 0u;

    for (int t = T - 1; t >= 0; --t) {
        dhB.zero();  // fresh accumulator for dL/dh_{t-1}
        LParams p{t, B, I, H, 0u, 0u, h0p ? 1u : 0u, c0p ? 1u : 0u};
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:buffer_for(X)     offset:buffer_offset_for(X)     atIndex:0];
            [enc setBuffer:buffer_for(Y)     offset:buffer_offset_for(Y)     atIndex:1];
            [enc setBuffer:buffer_for(gates) offset:buffer_offset_for(gates) atIndex:2];
            [enc setBuffer:buffer_for(C)     offset:buffer_offset_for(C)     atIndex:3];
            [enc setBuffer:buffer_for(dY)    offset:buffer_offset_for(dY)    atIndex:4];
            [enc setBuffer:buffer_for(W_ih)  offset:buffer_offset_for(W_ih)  atIndex:5];
            [enc setBuffer:buffer_for(W_hh)  offset:buffer_offset_for(W_hh)  atIndex:6];
            [enc setBuffer:opt_buf(h0p, X) offset:opt_off(h0p, X) atIndex:7];
            [enc setBuffer:opt_buf(c0p, X) offset:opt_off(c0p, X) atIndex:8];
            [enc setBuffer:buffer_for(dhA)   offset:buffer_offset_for(dhA)   atIndex:9];
            [enc setBuffer:buffer_for(dhB)   offset:buffer_offset_for(dhB)   atIndex:10];
            [enc setBuffer:buffer_for(dcBuf) offset:buffer_offset_for(dcBuf) atIndex:11];
            [enc setBuffer:buffer_for(dX)    offset:buffer_offset_for(dX)    atIndex:12];
            [enc setBuffer:buffer_for(dW_ih) offset:buffer_offset_for(dW_ih) atIndex:13];
            [enc setBuffer:buffer_for(dW_hh) offset:buffer_offset_for(dW_hh) atIndex:14];
            [enc setBuffer:opt_buf(db_ih, X) offset:opt_off(db_ih, X) atIndex:15];
            [enc setBuffer:opt_buf(db_hh, X) offset:opt_off(db_hh, X) atIndex:16];
            [enc setBytes:&p length:sizeof(LParams) atIndex:17];
            [enc setBytes:&has_dbih length:sizeof(uint32_t) atIndex:18];
            [enc setBytes:&has_dbhh length:sizeof(uint32_t) atIndex:19];
            [enc dispatchThreads:MTLSizeMake(B * H, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(LSTM_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        std::swap(dhA, dhB);  // dhA becomes the incoming carry for step t-1
    }

    // After the t=0 step, dhA holds dL/dh0 and dcBuf holds dL/dc0.
    if (dh0) {
        dh0->resize(B, H, Dtype::FP32);
        copy_floats(dhA, buffer_offset_for(dhA), *dh0, B * H);
    }
    if (dc0) {
        dc0->resize(B, H, Dtype::FP32);
        copy_floats(dcBuf, buffer_offset_for(dcBuf), *dc0, B * H);
    }
}

} // namespace brotensor::detail::metal
