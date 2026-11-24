// Vulkan gated linear units and AdaLN helpers: swiglu / geglu (tanh and
// exact) forward and backward over (R, 2D) rows (shaders/glu.comp), and
// modulate / broadcast_mul over (L, D) with a per-column vector
// (shaders/rowvec.comp). FP32 / FP16 / BF16 storage, FP32 math; every operand
// shares X's dtype (the CUDA contract); outputs are resized.

#include "detail/kernels.h"
#include "vk_mir_cache.h"

#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct RowvecPush { std::uint64_t x, v, w, y; std::uint32_t n, d; };

void glu(const char* op, std::uint32_t code, const Tensor& X, const Tensor* dY, Tensor& out) {
    dt_code(X.dtype, op);
    if (X.cols % 2 != 0) throw std::runtime_error(std::string("brotensor: ") + op + ": X.cols must be even (2*D)");
    const int B = X.rows, D = X.cols / 2;
    if (dY) {
        if (dY->dtype != X.dtype || dY->size() != static_cast<long long>(B) * D) {
            throw std::runtime_error(std::string("brotensor: ") + op + ": dY must be (B, D) in X's dtype");
        }
        if (out.rows != B || out.cols != 2 * D || out.dtype != X.dtype) out.resize(B, 2 * D, X.dtype);
    } else if (out.rows != B || out.cols != D || out.dtype != X.dtype) {
        out.resize(B, D, X.dtype);
    }
    const std::uint32_t n = count32(out, op) / (dY ? 2u : 1u);
    if (n == 0) return;
    DeviceCtx& d = device_of(X);
    const std::uint64_t ax = addr(X.data), ady = dY ? addr(dY->data) : 0, aout = addr(out.data);
    int lanes = 1;
    if (code != GLU_GEGLU_PAIRS && (D % 4 == 0)) {
        const std::uint32_t du = static_cast<std::uint32_t>(D);
        const int max_l = mir_vec_lanes(X.dtype, du, ax, aout, ady);
        if (du % static_cast<std::uint32_t>(max_l) == 0u) lanes = max_l;
    }
    const MirKernel mk = mir_kernel(d, {MirKind::Glu, code, X.dtype, X.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_glu(d, mk, mir_groups_1d(n, lanes), ax, ady, aout, n, static_cast<std::uint32_t>(D));
}

void rowvec(const char* op, std::uint32_t code, const Tensor& X, const Tensor& v, const Tensor* w, Tensor& Y) {
    dt_code(X.dtype, op);
    if (v.dtype != X.dtype || (w && w->dtype != X.dtype)) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": vector operands must share X's dtype");
    }
    if (v.size() != X.cols || (w && w->size() != X.cols)) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": vector operands have wrong element count");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) Y.resize(X.rows, X.cols, X.dtype);
    const std::uint32_t n = count32(X, op);
    if (n == 0) return;
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::rowvec_f32, X.dtype, op), {code});
    const RowvecPush pc{addr(X.data), addr(v.data), w ? addr(w->data) : 0, addr(Y.data), n,
                        static_cast<std::uint32_t>(X.cols)};
    launch(d, k, pc, groups_1d(n, k));
}

}  // namespace

void swiglu_forward(const Tensor& X, Tensor& Y) { glu("swiglu_forward", GLU_SWIGLU, X, nullptr, Y); }
void geglu_forward(const Tensor& X, Tensor& Y) { glu("geglu_forward", GLU_GEGLU_TANH, X, nullptr, Y); }
void geglu_exact_forward(const Tensor& X, Tensor& Y) { glu("geglu_exact_forward", GLU_GEGLU_EXACT, X, nullptr, Y); }
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("swiglu_backward", GLU_SWIGLU_BWD, X, &dY, dX);
}
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_backward", GLU_GEGLU_TANH_BWD, X, &dY, dX);
}
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    glu("geglu_exact_backward", GLU_GEGLU_EXACT_BWD, X, &dY, dX);
}

void modulate(const Tensor& X, const Tensor& scale, const Tensor& shift, Tensor& Y) {
    rowvec("modulate", ROWVEC_MODULATE, X, scale, &shift, Y);
}
void broadcast_mul(const Tensor& X, const Tensor& v, Tensor& Y) { rowvec("broadcast_mul", ROWVEC_MUL, X, v, nullptr, Y); }

void fill_vulkan_vtable_glu(::brotensor::detail::OpsVTable& v) {
    v.swiglu_forward = &swiglu_forward;
    v.swiglu_backward = &swiglu_backward;
    v.geglu_forward = &geglu_forward;
    v.geglu_backward = &geglu_backward;
    v.geglu_exact_forward = &geglu_exact_forward;
    v.geglu_exact_backward = &geglu_exact_backward;
    v.modulate = &modulate;
    v.broadcast_mul = &broadcast_mul;
}

}  // namespace brotensor::detail::vulkan
