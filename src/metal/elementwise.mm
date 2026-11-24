// Metal elementwise, activation, bias, cast, and GLU ops: single-source brass
// MIR kernels compiled via MslTarget and cached in mtl_mir_cache.mm.
//
// Specialized non-Wave-1 ops (build_slot_mask, build_causal_mask_row,
// threshold_u8) remain at the bottom of this file.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::dispatch1d_sync;
using metal_impl::new_command_buffer;

namespace {

std::uint64_t dev_addr(const Tensor& t) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.data));
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": unsupported dtype (expected FP32/FP16/BF16)");
    }
}

std::uint32_t count32(const Tensor& t, const char* op) {
    const long long n = t.size();
    if (n < 0 || n > 0x7fffffffLL) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": element count out of range");
    }
    return static_cast<std::uint32_t>(n);
}

void unary(const char* op, const Tensor& x, Tensor& y, std::uint32_t code,
           float a = 0.0f, float b = 0.0f) {
    require_fp(x.dtype, op);
    if (&x != &y && (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const std::uint32_t n = count32(x, op);
    if (n == 0) return;
    const int lanes = mir_vec_lanes(x.dtype, n, dev_addr(x), dev_addr(y));
    const MirKernel mk = mir_kernel(
        {MirKind::Unary, code, x.dtype, x.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(mk, mir_groups_1d(n, lanes), x, y, n, a, b);
}

void inplace_scalar(const char* op, Tensor& y, std::uint32_t code,
                    float a, float b = 0.0f) {
    unary(op, y, y, code, a, b);
}

void binary(const char* op, Tensor& y, const Tensor& x, std::uint32_t code,
            float a = 0.0f, float b = 0.0f) {
    if (y.size() != x.size()) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": size mismatch y=(" +
                                 std::to_string(y.rows) + "," + std::to_string(y.cols) + ") x=(" +
                                 std::to_string(x.rows) + "," + std::to_string(x.cols) + ")");
    }
    require_fp(y.dtype, op);
    require_fp(x.dtype, op);
    if (y.dtype != x.dtype &&
        !(code == BOP_ADD && y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16)) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": dtype mismatch");
    }
    const std::uint32_t n = count32(y, op);
    if (n == 0) return;
    const int lanes = mir_vec_lanes_mixed(y.dtype, x.dtype, n, dev_addr(y), dev_addr(x));
    const MirKernel mk = mir_kernel(
        {MirKind::Binary, code, y.dtype, x.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(mk, mir_groups_1d(n, lanes), y, x, n, a, b);
}

void act_backward(const char* op, const Tensor* p, const Tensor& dY, Tensor& dX,
                  std::uint32_t code, float a = 0.0f) {
    require_fp(dY.dtype, op);
    if (p) {
        if (p->dtype != dY.dtype) {
            throw std::runtime_error(std::string("brotensor: ") + op +
                                     ": dY.dtype must match the input's");
        }
        if (p->size() != dY.size()) {
            throw std::runtime_error(std::string("brotensor: ") + op +
                                     ": dY size must match the input's");
        }
    }
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    const std::uint32_t n = count32(dY, op);
    if (n == 0) return;
    const std::uint64_t ap = p ? dev_addr(*p) : 0;
    const int lanes = mir_vec_lanes(dY.dtype, n, dev_addr(dY), dev_addr(dX), ap);
    const MirKernel mk = mir_kernel(
        {MirKind::ActBwd, code, dY.dtype, dY.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_act_bwd(mk, mir_groups_1d(n, lanes), p, dY, dX, n, a);
}

void bias_add(const char* op, Tensor& y, const Tensor& bias, std::uint32_t mode,
              std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    require_fp(y.dtype, op);
    if (y.dtype != bias.dtype) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": dtype mismatch");
    }
    if (n == 0 || dim == 0) return;
    const int lanes = (mode == BIAS_ROW && (dim % 4u == 0u))
                          ? mir_vec_lanes(y.dtype, dim, dev_addr(y), dev_addr(bias))
                          : 1;
    const MirKernel mk = mir_kernel(
        {MirKind::Bias, mode, y.dtype, y.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_bias(mk, mir_groups_1d(n, lanes), y, bias, n, dim, l);
}

void glu(const char* op, std::uint32_t code, const Tensor& X,
         const Tensor* dY, Tensor& out) {
    require_fp(X.dtype, op);
    if (X.cols % 2 != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dY) {
        if (dY->dtype != X.dtype || dY->size() != static_cast<long long>(B) * D) {
            throw std::runtime_error(std::string("brotensor: ") + op +
                                     ": dY must be (B, D) in X's dtype");
        }
        if (out.rows != B || out.cols != 2 * D || out.dtype != X.dtype) {
            out.resize(B, 2 * D, X.dtype);
        }
    } else if (out.rows != B || out.cols != D || out.dtype != X.dtype) {
        out.resize(B, D, X.dtype);
    }
    const std::uint32_t n = count32(out, op) / (dY ? 2u : 1u);
    if (n == 0) return;
    const std::uint64_t ax = dev_addr(X);
    const std::uint64_t ady = dY ? dev_addr(*dY) : 0;
    const std::uint64_t aout = dev_addr(out);
    int lanes = 1;
    if (code != GLU_GEGLU_PAIRS && (D % 4 == 0)) {
        const std::uint32_t du = static_cast<std::uint32_t>(D);
        const int max_l = mir_vec_lanes(X.dtype, du, ax, aout, ady);
        if (du % static_cast<std::uint32_t>(max_l) == 0u) lanes = max_l;
    }
    const MirKernel mk = mir_kernel(
        {MirKind::Glu, code, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_glu(mk, mir_groups_1d(n, lanes), X, dY, out, n,
                     static_cast<std::uint32_t>(D));
}

// ─── Specialized non-Wave-1 Metal kernels (mask / threshold_u8) ────────────

NSString* const kMiscSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_causal_mask_row(device float* mask [[buffer(0)]],
                              constant uint& L   [[buffer(1)]],
                              constant uint& q   [[buffer(2)]],
                              uint k [[thread_position_in_grid]]) {
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
}

kernel void k_threshold_u8_fp32(device const float* x [[buffer(0)]],
                                constant float& t     [[buffer(1)]],
                                device char* y        [[buffer(2)]],
                                constant uint& n      [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = (x[i] > t) ? char(1) : char(0);
}

kernel void k_threshold_u8_fp16(device const half* x [[buffer(0)]],
                                constant float& t    [[buffer(1)]],
                                device char* y       [[buffer(2)]],
                                constant uint& n     [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = (float(x[i]) > t) ? char(1) : char(0);
}
)msl";

#define DEF_MISC_PSO(NAME, FN)                                                 \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kMiscSrc, FN); });      \
        return pso;                                                            \
    }
DEF_MISC_PSO(pso_causal_mask_row, @"k_causal_mask_row")
DEF_MISC_PSO(pso_threshold_u8_fp32, @"k_threshold_u8_fp32")
DEF_MISC_PSO(pso_threshold_u8_fp16, @"k_threshold_u8_fp16")
#undef DEF_MISC_PSO

void launch_1d(id<MTLComputePipelineState> pso, std::uint32_t n,
               void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace

// ─── Forward activations / math ────────────────────────────────────────────

#define BT_MTL_UNARY(fn, code) \
    void fn(const Tensor& x, Tensor& y) { unary(#fn, x, y, code); }
BT_MTL_UNARY(relu_forward,       UOP_RELU)
BT_MTL_UNARY(tanh_forward,       UOP_TANH)
BT_MTL_UNARY(sigmoid_forward,    UOP_SIGMOID)
BT_MTL_UNARY(silu_forward,       UOP_SILU)
BT_MTL_UNARY(gelu_forward,       UOP_GELU_TANH)
BT_MTL_UNARY(gelu_exact_forward, UOP_GELU_EXACT)
BT_MTL_UNARY(quick_gelu_forward, UOP_QUICK_GELU)
BT_MTL_UNARY(exp_forward,        UOP_EXP)
BT_MTL_UNARY(log_forward,        UOP_LOG)
BT_MTL_UNARY(sin_forward,        UOP_SIN)
BT_MTL_UNARY(cos_forward,        UOP_COS)
BT_MTL_UNARY(rsqrt_forward,      UOP_RSQRT)
BT_MTL_UNARY(round_forward,      UOP_ROUND)
#undef BT_MTL_UNARY

void relu_forward_batched(const Tensor& X, Tensor& Y) { metal::relu_forward(X, Y); }
void tanh_forward_batched(const Tensor& X, Tensor& Y) { metal::tanh_forward(X, Y); }
void elu_forward(const Tensor& x, float alpha, Tensor& y) {
    unary("elu_forward", x, y, UOP_ELU, alpha);
}
void leaky_relu_forward(const Tensor& x, float slope, Tensor& y) {
    unary("leaky_relu_forward", x, y, UOP_LEAKY_RELU, slope);
}

// ─── In-place scalar / binary ──────────────────────────────────────────────

void add_scalar_inplace(Tensor& y, float s) {
    inplace_scalar("add_scalar_inplace", y, UOP_ADD_SCALAR, s);
}
void scale_inplace(Tensor& y, float s) {
    inplace_scalar("scale_inplace", y, UOP_SCALE, s);
}
void clamp(Tensor& y, float lo, float hi) {
    inplace_scalar("clamp", y, UOP_CLAMP, lo, hi);
}
void add_inplace(Tensor& y, const Tensor& x) {
    binary("add_inplace", y, x, BOP_ADD);
}
void add_inplace_batched(Tensor& Y, const Tensor& X) {
    binary("add_inplace_batched", Y, X, BOP_ADD);
}
void mul_inplace(Tensor& y, const Tensor& x) {
    binary("mul_inplace", y, x, BOP_MUL);
}
void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    binary("axpby_inplace", y, x, BOP_AXPBY, a, b);
}

// ─── Backwards ─────────────────────────────────────────────────────────────

#define BT_MTL_GRAD(fn, code) \
    void fn(const Tensor& p, const Tensor& dY, Tensor& dX) { act_backward(#fn, &p, dY, dX, code); }
BT_MTL_GRAD(relu_backward,       GOP_RELU)
BT_MTL_GRAD(tanh_backward,       GOP_TANH)
BT_MTL_GRAD(sigmoid_backward,    GOP_SIGMOID)
BT_MTL_GRAD(silu_backward,       GOP_SILU)
BT_MTL_GRAD(gelu_backward,       GOP_GELU_TANH)
BT_MTL_GRAD(gelu_exact_backward, GOP_GELU_EXACT)
BT_MTL_GRAD(quick_gelu_backward, GOP_QUICK_GELU)
BT_MTL_GRAD(exp_backward,        GOP_EXP)
BT_MTL_GRAD(log_backward,        GOP_LOG)
BT_MTL_GRAD(sin_backward,        GOP_SIN)
BT_MTL_GRAD(cos_backward,        GOP_COS)
BT_MTL_GRAD(rsqrt_backward,      GOP_RSQRT)
#undef BT_MTL_GRAD

void relu_backward_batched(const Tensor& X, const Tensor& dY, Tensor& dX) {
    metal::relu_backward(X, dY, dX);
}
void tanh_backward_batched(const Tensor& Y, const Tensor& dY, Tensor& dX) {
    metal::tanh_backward(Y, dY, dX);
}
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX) {
    act_backward("elu_backward", &x, dY, dX, GOP_ELU, alpha);
}
void leaky_relu_backward(const Tensor& x, const Tensor& dY, float slope, Tensor& dX) {
    act_backward("leaky_relu_backward", &x, dY, dX, GOP_LEAKY_RELU, slope);
}
void round_backward(const Tensor& dY, Tensor& dX) {
    act_backward("round_backward", nullptr, dY, dX, GOP_PASS);
}

// ─── Broadcast bias ────────────────────────────────────────────────────────

void add_row_bias_inplace(Tensor& y, const Tensor& bias) {
    if (bias.size() != y.cols) {
        throw std::runtime_error("brotensor: add_row_bias_inplace: bias size != y.cols");
    }
    bias_add("add_row_bias_inplace", y, bias, BIAS_ROW,
             count32(y, "add_row_bias_inplace"),
             static_cast<std::uint32_t>(y.cols), 1);
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    if (C < 0 || L < 0) {
        throw std::runtime_error("brotensor: add_channel_bias_inplace: negative dimension");
    }
    if (static_cast<long long>(C) * L > y.size() || bias.size() < C) {
        throw std::runtime_error(
            "brotensor: add_channel_bias_inplace: C * L exceeds y or C exceeds bias");
    }
    bias_add("add_channel_bias_inplace", y, bias, BIAS_CHANNEL,
             static_cast<std::uint32_t>(C) * static_cast<std::uint32_t>(L),
             static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(L));
}

// ─── Cast ──────────────────────────────────────────────────────────────────

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const std::uint32_t n = count32(src, "cast");
    if (n == 0) return;

    if (src.dtype == out_dtype) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:buffer_for(src) sourceOffset:buffer_offset_for(src)
                        toBuffer:buffer_for(dst) destinationOffset:buffer_offset_for(dst)
                            size:src.bytes()];
            [blit endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    require_fp(src.dtype, "cast");
    require_fp(out_dtype, "cast");
    const int lanes = mir_vec_lanes_mixed(src.dtype, out_dtype, n,
                                          dev_addr(src), dev_addr(dst));
    const MirKernel mk = mir_kernel(
        {MirKind::Cast, 0, src.dtype, out_dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_cast(mk, mir_groups_1d(n, lanes), src, dst, n);
}

// ─── Gated linear units (SwiGLU / GeGLU / GeGLU-exact) ────────────────────

void swiglu_forward(const Tensor& X, Tensor& Y) {
    glu("swiglu_forward", GLU_SWIGLU, X, nullptr, Y);
}
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("swiglu_backward", GLU_SWIGLU_BWD, X, &dY, dX);
}
void geglu_forward(const Tensor& X, Tensor& Y) {
    glu("geglu_forward", GLU_GEGLU_TANH, X, nullptr, Y);
}
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_backward", GLU_GEGLU_TANH_BWD, X, &dY, dX);
}
void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    glu("geglu_exact_forward", GLU_GEGLU_EXACT, X, nullptr, Y);
}
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_exact_backward", GLU_GEGLU_EXACT_BWD, X, &dY, dX);
}

// ─── Specialized non-Wave-1 ops ────────────────────────────────────────────

void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask) {
    if (mask.rows != K || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(K, 1, Dtype::FP32);
    }
    if (K <= 0) return;
    id<MTLBuffer> bx = buffer_for(x);
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger off_x = buffer_offset_for(x);
    const NSUInteger off_m = buffer_offset_for(mask);
    const std::uint32_t uo = static_cast<std::uint32_t>(offset);
    const std::uint32_t uK = static_cast<std::uint32_t>(K);
    const std::uint32_t us = static_cast<std::uint32_t>(stride);
    dispatch1d_sync(@"k_build_slot_mask", uK, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:off_x atIndex:0];
        [enc setBuffer:bm offset:off_m atIndex:1];
        [enc setBytes:&uo length:sizeof(std::uint32_t) atIndex:2];
        [enc setBytes:&uK length:sizeof(std::uint32_t) atIndex:3];
        [enc setBytes:&us length:sizeof(std::uint32_t) atIndex:4];
    });
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger om = buffer_offset_for(mask);
    const std::uint32_t Lu = static_cast<std::uint32_t>(L);
    const std::uint32_t qu = static_cast<std::uint32_t>(q);
    launch_1d(pso_causal_mask_row(), Lu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bm offset:om atIndex:0];
        [enc setBytes:&Lu length:sizeof(std::uint32_t) atIndex:1];
        [enc setBytes:&qu length:sizeof(std::uint32_t) atIndex:2];
    });
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("threshold_u8: X must be FP32 or FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const std::uint32_t n = count32(X, "threshold_u8");
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_threshold_u8_fp16() : pso_threshold_u8_fp32();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBytes:&t length:sizeof(float) atIndex:1];
        [enc setBuffer:by offset:oy atIndex:2];
        [enc setBytes:&n length:sizeof(std::uint32_t) atIndex:3];
    });
}

}  // namespace brotensor::detail::metal
