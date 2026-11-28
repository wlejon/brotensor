// CUDA elementwise, activation, math, bias, cast, GLU, and rowvec ops backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

inline void require_fp(const char* op, Dtype dt) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": unsupported dtype (expected FP32/FP16/BF16)");
    }
}

inline std::uintptr_t addr(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}

inline std::uint32_t count32(const Tensor& t, const char* op) {
    const long long sz = t.size();
    if (sz < 0 || sz > static_cast<long long>((std::numeric_limits<std::uint32_t>::max)())) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": tensor too large for 32-bit index");
    }
    return static_cast<std::uint32_t>(sz);
}

void unary(const char* op, const Tensor& x, Tensor& y, std::uint32_t code,
           float a = 0.0f, float b = 0.0f) {
    require_fp(op, x.dtype);
    if (&x != &y && (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const std::uint32_t n = count32(x, op);
    if (n == 0) return;
    const int lanes = mir_vec_lanes(x.dtype, n, addr(x.data), addr(y.data));
    CUfunction fn = mir_function({MirKind::Unary, code, x.dtype, x.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(fn, mir_groups_1d(n, lanes), x.data, y.data, n, a, b);
}

void inplace_scalar(const char* op, Tensor& y, std::uint32_t code, float a, float b = 0.0f) {
    unary(op, y, y, code, a, b);
}

void binary(const char* op, Tensor& y, const Tensor& x, std::uint32_t code,
            float a = 0.0f, float b = 0.0f) {
    if (y.size() != x.size() || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error(std::string(op) + ": shape/dtype mismatch");
    }
    require_fp(op, y.dtype);
    require_fp(op, x.dtype);
    if (y.dtype != x.dtype && !(code == BOP_ADD && y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16)) {
        throw std::runtime_error(std::string(op) + ": dtype mismatch");
    }
    const std::uint32_t n = count32(y, op);
    if (n == 0) return;
    const int lanes = mir_vec_lanes_mixed(y.dtype, x.dtype, n, addr(y.data), addr(x.data));
    CUfunction fn = mir_function({MirKind::Binary, code, y.dtype, x.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(fn, mir_groups_1d(n, lanes), y.data, x.data, n, a, b);
}

void act_backward(const char* op, const Tensor* p, const Tensor& dY, Tensor& dX,
                  std::uint32_t code, float a = 0.0f) {
    require_fp(op, dY.dtype);
    if (p) {
        require_fp(op, p->dtype);
        if (dY.dtype != p->dtype || dY.size() != p->size()) {
            throw std::runtime_error(std::string(op) + ": dY shape/dtype mismatch");
        }
    }
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    const std::uint32_t n = count32(dY, op);
    if (n == 0) return;
    const int lanes = mir_vec_lanes(dY.dtype, n, addr(dY.data), addr(dX.data), p ? addr(p->data) : 0);
    CUfunction fn = mir_function({MirKind::ActBwd, code, dY.dtype, dY.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_act_bwd(fn, mir_groups_1d(n, lanes), p ? p->data : nullptr, dY.data, dX.data, n, a);
}

void glu(const char* op, std::uint32_t code, const Tensor& X, const Tensor* dY, Tensor& out) {
    require_fp(op, X.dtype);
    if (X.cols % 2 != 0) {
        throw std::runtime_error(std::string(op) + ": X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dY) {
        if (dY->dtype != X.dtype || dY->size() != static_cast<long long>(B) * D) {
            throw std::runtime_error(std::string(op) + ": dY must be (B, D) in X's dtype");
        }
        if (out.rows != B || out.cols != 2 * D || out.dtype != X.dtype) {
            out.resize(B, 2 * D, X.dtype);
        }
    } else if (out.rows != B || out.cols != D || out.dtype != X.dtype) {
        out.resize(B, D, X.dtype);
    }
    const std::uint32_t n = count32(out, op) / (dY ? 2u : 1u);
    if (n == 0) return;
    int lanes = 1;
    if (D % 4 == 0) {
        const std::uint32_t du = static_cast<std::uint32_t>(D);
        const int max_l = mir_vec_lanes(X.dtype, du, addr(X.data), addr(out.data), dY ? addr(dY->data) : 0);
        if (du % static_cast<std::uint32_t>(max_l) == 0u) lanes = max_l;
    }
    CUfunction fn = mir_function({MirKind::Glu, code, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_glu(fn, mir_groups_1d(n, lanes), X.data, dY ? dY->data : nullptr, out.data, n, static_cast<std::uint32_t>(D));
}

void rowvec(const char* op, std::uint32_t code, const Tensor& X, const Tensor& v, const Tensor* w, Tensor& Y) {
    require_fp(op, X.dtype);
    if (v.dtype != X.dtype || (w && w->dtype != X.dtype)) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": vector operands must share X's dtype");
    }
    if (v.size() != X.cols || (w && w->size() != X.cols)) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": vector operands have wrong element count");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
    const std::uint32_t n = count32(X, op);
    if (n == 0) return;
    const std::uint32_t D = static_cast<std::uint32_t>(X.cols);
    const int lanes = (D % 4u == 0u) ? mir_vec_lanes(X.dtype, D, addr(X.data), addr(v.data), w ? addr(w->data) : 0, addr(Y.data)) : 1;
    CUfunction fn = mir_function({MirKind::Rowvec, code, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_rowvec(fn, mir_groups_1d(n, lanes), X.data, v.data, w ? w->data : nullptr, Y.data, n, D);
}

}  // namespace

// ─── forward unary activations & math ──────────────────────────────────────

void relu_forward(const Tensor& x, Tensor& y)       { unary("relu_forward", x, y, UOP_RELU); }
void tanh_forward(const Tensor& x, Tensor& y)       { unary("tanh_forward", x, y, UOP_TANH); }
void sigmoid_forward(const Tensor& x, Tensor& y)    { unary("sigmoid_forward", x, y, UOP_SIGMOID); }
void silu_forward(const Tensor& x, Tensor& y)       { unary("silu_forward", x, y, UOP_SILU); }
void gelu_forward(const Tensor& x, Tensor& y)       { unary("gelu_forward", x, y, UOP_GELU_TANH); }
void gelu_exact_forward(const Tensor& x, Tensor& y) { unary("gelu_exact_forward", x, y, UOP_GELU_EXACT); }
void quick_gelu_forward(const Tensor& x, Tensor& y) { unary("quick_gelu_forward", x, y, UOP_QUICK_GELU); }
void exp_forward(const Tensor& x, Tensor& y)        { unary("exp_forward", x, y, UOP_EXP); }
void log_forward(const Tensor& x, Tensor& y)        { unary("log_forward", x, y, UOP_LOG); }
void sin_forward(const Tensor& x, Tensor& y)        { unary("sin_forward", x, y, UOP_SIN); }
void cos_forward(const Tensor& x, Tensor& y)        { unary("cos_forward", x, y, UOP_COS); }
void rsqrt_forward(const Tensor& x, Tensor& y)      { unary("rsqrt_forward", x, y, UOP_RSQRT); }
void round_forward(const Tensor& x, Tensor& y)      { unary("round_forward", x, y, UOP_ROUND); }
void elu_forward(const Tensor& x, float alpha, Tensor& y)        { unary("elu_forward", x, y, UOP_ELU, alpha); }
void leaky_relu_forward(const Tensor& x, float slope, Tensor& y) { unary("leaky_relu_forward", x, y, UOP_LEAKY_RELU, slope); }

// ─── backward activations & math ───────────────────────────────────────────

void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX)       { act_backward("relu_backward", &x, dY, dX, GOP_RELU); }
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX)       { act_backward("tanh_backward", &y, dY, dX, GOP_TANH); }
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX)    { act_backward("sigmoid_backward", &y, dY, dX, GOP_SIGMOID); }
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX)       { act_backward("silu_backward", &x, dY, dX, GOP_SILU); }
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX)       { act_backward("gelu_backward", &x, dY, dX, GOP_GELU_TANH); }
void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX) { act_backward("gelu_exact_backward", &x, dY, dX, GOP_GELU_EXACT); }
void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) { act_backward("quick_gelu_backward", &x, dY, dX, GOP_QUICK_GELU); }
void exp_backward(const Tensor& x, const Tensor& dY, Tensor& dX)        { act_backward("exp_backward", &x, dY, dX, GOP_EXP); }
void log_backward(const Tensor& x, const Tensor& dY, Tensor& dX)        { act_backward("log_backward", &x, dY, dX, GOP_LOG); }
void sin_backward(const Tensor& x, const Tensor& dY, Tensor& dX)        { act_backward("sin_backward", &x, dY, dX, GOP_SIN); }
void cos_backward(const Tensor& x, const Tensor& dY, Tensor& dX)        { act_backward("cos_backward", &x, dY, dX, GOP_COS); }
void rsqrt_backward(const Tensor& y, const Tensor& dY, Tensor& dX)      { act_backward("rsqrt_backward", &y, dY, dX, GOP_RSQRT); }
void round_backward(const Tensor& dY, Tensor& dX)                       { act_backward("round_backward", nullptr, dY, dX, GOP_PASS); }
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX)        { act_backward("elu_backward", &x, dY, dX, GOP_ELU, alpha); }
void leaky_relu_backward(const Tensor& x, const Tensor& dY, float slope, Tensor& dX) { act_backward("leaky_relu_backward", &x, dY, dX, GOP_LEAKY_RELU, slope); }

// ─── batched elementwise delegates ─────────────────────────────────────────

void add_inplace(Tensor& y, const Tensor& x);

void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD) { cuda::relu_forward(X_BD, Y_BD); }
void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD) { cuda::tanh_forward(X_BD, Y_BD); }
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD) {
    if (Y_BD.dtype != X_BD.dtype) throw std::runtime_error("add_inplace_batched: dtype mismatch");
    cuda::add_inplace(Y_BD, X_BD);
}
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD, Tensor& dX_BD) { cuda::relu_backward(X_BD, dY_BD, dX_BD); }
void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD, Tensor& dX_BD) { cuda::tanh_backward(Y_BD, dY_BD, dX_BD); }

// ─── in-place scalar & binary ──────────────────────────────────────────────

void add_scalar_inplace(Tensor& y, float s) { inplace_scalar("add_scalar_inplace", y, UOP_ADD_SCALAR, s); }
void scale_inplace(Tensor& y, float s)      { inplace_scalar("scale_inplace", y, UOP_SCALE, s); }
void clamp(Tensor& y, float lo, float hi)   { inplace_scalar("clamp", y, UOP_CLAMP, lo, hi); }

void add_inplace(Tensor& y, const Tensor& x) {
    if (y.size() != x.size()) {
        throw std::runtime_error("add_inplace: size mismatch");
    }
    require_fp("add_inplace", y.dtype);
    require_fp("add_inplace", x.dtype);
    if (y.dtype != x.dtype && !(y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16)) {
        throw std::runtime_error("add_inplace: dtype mismatch");
    }
    const std::uint32_t n = count32(y, "add_inplace");
    if (n == 0) return;
    const int lanes = mir_vec_lanes_mixed(y.dtype, x.dtype, n, addr(y.data), addr(x.data));
    CUfunction fn = mir_function({MirKind::Binary, BOP_ADD, y.dtype, x.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_5(fn, mir_groups_1d(n, lanes), y.data, x.data, n, 0.0f, 0.0f);
}

void mul_inplace(Tensor& y, const Tensor& x)                   { binary("mul_inplace", y, x, BOP_MUL); }
void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) { binary("axpby_inplace", y, x, BOP_AXPBY, a, b); }

// ─── bias ──────────────────────────────────────────────────────────────────

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    if (C < 0 || L < 0) {
        throw std::runtime_error("add_channel_bias_inplace: negative dimension");
    }
    const long long n64 = static_cast<long long>(C) * L;
    if (n64 == 0) return;
    if (y.dtype != bias.dtype) {
        throw std::runtime_error("add_channel_bias_inplace: dtype mismatch");
    }
    if (y.size() != n64) {
        throw std::runtime_error("add_channel_bias_inplace: y size != C*L");
    }
    if (bias.size() < C) {
        throw std::runtime_error("add_channel_bias_inplace: C exceeds bias size");
    }
    require_fp("add_channel_bias_inplace", y.dtype);
    const std::uint32_t n = static_cast<std::uint32_t>(n64);
    CUfunction fn = mir_function({MirKind::Bias, BIAS_CHANNEL, y.dtype, y.dtype, 1});
    mir_dispatch_bias(fn, mir_groups_1d(n, 1), y.data, bias.data, n, static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(L));
}

void add_row_bias_inplace(Tensor& Y, const Tensor& bias) {
    if (Y.dtype != bias.dtype) {
        throw std::runtime_error("add_row_bias_inplace: dtype mismatch");
    }
    if (bias.size() != Y.cols) {
        throw std::runtime_error("add_row_bias_inplace: bias size != Y.cols");
    }
    require_fp("add_row_bias_inplace", Y.dtype);
    const std::uint32_t n = count32(Y, "add_row_bias_inplace");
    if (n == 0) return;
    const std::uint32_t D = static_cast<std::uint32_t>(Y.cols);
    const int lanes = (D % 4u == 0u) ? mir_vec_lanes(Y.dtype, D, addr(Y.data), addr(bias.data)) : 1;
    CUfunction fn = mir_function({MirKind::Bias, BIAS_ROW, Y.dtype, Y.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_bias(fn, mir_groups_1d(n, lanes), Y.data, bias.data, n, D, 1);
}

// ─── cast ──────────────────────────────────────────────────────────────────

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const std::uint32_t n = count32(src, "cast");
    if (n == 0) return;
    if (src.dtype == out_dtype) {
        BROTENSOR_CUDA_CHECK(memcpy_d2d_async(
            dst.data, src.data, src.bytes(),
            reinterpret_cast<CUstream>(::brotensor::cuda_current_stream())));
        return;
    }
    require_fp("cast", src.dtype);
    require_fp("cast", out_dtype);
    const int lanes = mir_vec_lanes_mixed(src.dtype, out_dtype, n, addr(src.data), addr(dst.data));
    CUfunction fn = mir_function({MirKind::Cast, 0, src.dtype, out_dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_cast(fn, mir_groups_1d(n, lanes), src.data, dst.data, n);
}

// ─── gated linear units & rowvec ───────────────────────────────────────────

void swiglu_forward(const Tensor& X, Tensor& Y)       { glu("swiglu_forward", GLU_SWIGLU, X, nullptr, Y); }
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("swiglu_backward", GLU_SWIGLU_BWD, X, &dY, dX);
}
void geglu_forward(const Tensor& X, Tensor& Y)        { glu("geglu_forward", GLU_GEGLU_TANH, X, nullptr, Y); }
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_backward", GLU_GEGLU_TANH_BWD, X, &dY, dX);
}
void geglu_exact_forward(const Tensor& X, Tensor& Y)  { glu("geglu_exact_forward", GLU_GEGLU_EXACT, X, nullptr, Y); }
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_exact_backward", GLU_GEGLU_EXACT_BWD, X, &dY, dX);
}

void modulate(const Tensor& X, const Tensor& scale, const Tensor& shift, Tensor& Y) {
    rowvec("modulate", ROWVEC_MODULATE, X, scale, &shift, Y);
}
void broadcast_mul(const Tensor& X, const Tensor& v, Tensor& Y) {
    rowvec("broadcast_mul", ROWVEC_MUL, X, v, nullptr, Y);
}

// ─── vtable registration ───────────────────────────────────────────────────

void threshold_u8(const Tensor& X, float t, Tensor& Y);
void build_causal_mask_row(int L, int q, Tensor& mask);
void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask);
void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y);
void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma);
void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y);
void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX);

void fill_cuda_vtable_elementwise(::brotensor::detail::OpsVTable& v) {
    v.relu_forward             = &relu_forward;
    v.modulate                 = &modulate;
    v.broadcast_mul            = &broadcast_mul;
    v.relu_backward            = &relu_backward;
    v.tanh_forward             = &tanh_forward;
    v.tanh_backward            = &tanh_backward;
    v.sigmoid_forward          = &sigmoid_forward;
    v.sigmoid_backward         = &sigmoid_backward;
    v.add_inplace              = &add_inplace;
    v.axpby_inplace            = &axpby_inplace;
    v.add_scalar_inplace       = &add_scalar_inplace;
    v.add_channel_bias_inplace = &add_channel_bias_inplace;
    v.add_row_bias_inplace     = &add_row_bias_inplace;
    v.cast                     = &cast;
    v.scale_inplace            = &scale_inplace;
    v.mul_inplace              = &mul_inplace;
    v.threshold_u8             = &threshold_u8;
    v.clamp                    = &clamp;
    v.silu_forward             = &silu_forward;
    v.silu_backward            = &silu_backward;
    v.gelu_forward             = &gelu_forward;
    v.gelu_backward            = &gelu_backward;
    v.gelu_exact_forward       = &gelu_exact_forward;
    v.gelu_exact_backward      = &gelu_exact_backward;
    v.quick_gelu_forward       = &quick_gelu_forward;
    v.quick_gelu_backward      = &quick_gelu_backward;
    v.geglu_forward            = &geglu_forward;
    v.geglu_backward           = &geglu_backward;
    v.geglu_exact_forward      = &geglu_exact_forward;
    v.geglu_exact_backward     = &geglu_exact_backward;
    v.build_slot_mask          = &build_slot_mask;
    v.build_causal_mask_row    = &build_causal_mask_row;
    v.rms_norm_forward         = &rms_norm_forward;
    v.rms_norm_backward        = &rms_norm_backward;
    v.swiglu_forward           = &swiglu_forward;
    v.swiglu_backward          = &swiglu_backward;
}

void fill_cuda_vtable_log_exp_round(::brotensor::detail::OpsVTable& v) {
    v.log_forward    = &log_forward;
    v.log_backward   = &log_backward;
    v.exp_forward    = &exp_forward;
    v.exp_backward   = &exp_backward;
    v.round_forward  = &round_forward;
    v.round_backward = &round_backward;
}

void fill_cuda_vtable_stylegan_elementwise(::brotensor::detail::OpsVTable& v) {
    v.sin_forward         = &sin_forward;
    v.sin_backward        = &sin_backward;
    v.cos_forward         = &cos_forward;
    v.cos_backward        = &cos_backward;
    v.rsqrt_forward       = &rsqrt_forward;
    v.rsqrt_backward      = &rsqrt_backward;
    v.pixel_norm_forward  = &pixel_norm_forward;
    v.pixel_norm_backward = &pixel_norm_backward;
}

}  // namespace brotensor::detail::cuda
