#include "native_tensor_decl.h"
#include "api_internal.h"
#include "brotensor/detail/dispatch.h"

#include <cstring>

using namespace brotensor::api;
using brotensor::Tensor;

// Every body checks its operands against the dims before brotensor sees them
// (api_internal.h "input-size contract") and runs under BROTENSOR_API_TRY:
// an op that throws records the message for the wrapper's chk().
//
// Outputs an op resizes are not checked; outputs it ACCUMULATES into (the
// dW / dB of a backward, "caller zeros") are, because the op writes them at
// the size the dims say.

namespace {

// A unary map: y = f(x). The op resizes y.
template <class Op>
void unary(const char* L, void* x, void* y, Op op) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!need(L, {xt, yt})) return;
    BROTENSOR_API_TRY
        op(*xt, *yt);
    BROTENSOR_API_CATCH(L)
}

// Its backward: dX = f'(a) * dY, `a` being the forward input or output.
template <class Op>
void unaryBackward(const char* L, void* a, void* dY, void* dX, Op op) {
    auto* at = toTensor(a);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {at, dyt, dxt})) return;
    if (!needPair(L, "dY", dyt, at)) return;
    BROTENSOR_API_TRY
        op(*at, *dyt, *dxt);
    BROTENSOR_API_CATCH(L)
}

// A gated unit's input is (B, 2*D): an even last dim.
bool needGatedInput(const char* L, const Tensor* xt) {
    if (xt->cols % 2 != 0) {
        setError(std::string(L) + ": X must be (B, 2*D) — an even column count");
        return false;
    }
    return true;
}

template <class Op>
void gatedForward(const char* L, void* X, void* Y, Op op) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, yt})) return;
    if (!needGatedInput(L, xt)) return;
    BROTENSOR_API_TRY
        op(*xt, *yt);
    BROTENSOR_API_CATCH(L)
}

template <class Op>
void gatedBackward(const char* L, void* X, void* dY, void* dX, Op op) {
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {xt, dyt, dxt})) return;
    if (!needGatedInput(L, xt)) return;
    if (!needElems(L, "dY", dyt, elems({xt->rows, xt->cols / 2}))) return;
    if (!needSameDtype(L, "dY", dyt, xt)) return;
    BROTENSOR_API_TRY
        op(*xt, *dyt, *dxt);
    BROTENSOR_API_CATCH(L)
}

// An NCHW resample X:(N, C*H*W) -> Y. The op resizes Y.
template <class Op>
void nchwMap(const char* L, void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y, Op op) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, yt})) return;
    if (!needElems(L, "X", xt, elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        op(*xt, *yt);
    BROTENSOR_API_CATCH(L)
}

// downsample_avg_2x wants an even plane.
bool needEvenPlane(const char* L, int32_t H, int32_t W) {
    if (H % 2 != 0 || W % 2 != 0) {
        setError(std::string(L) + ": H and W must be even");
        return false;
    }
    return true;
}

} // namespace

extern "C" {

void bro_tensor_linearForward(void* W, void* b, void* x, void* y) {
    const char* L = "linearForward";
    auto* wt = toTensor(W);
    auto* bt = toTensor(b);
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!need(L, {wt, bt, xt, yt})) return;
    if (!needElems(L, "b", bt, wt->rows) || !needElems(L, "x", xt, wt->cols)) return;
    if (!needSameDtype(L, "b", bt, wt) || !needSameDtype(L, "x", xt, wt)) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward(*wt, *bt, *xt, *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_linearBackward(void* W, void* x, void* dY, void* dX, void* dW, void* dB) {
    const char* L = "linearBackward";
    auto* wt = toTensor(W);
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dwt = toTensor(dW);
    auto* dbt = toTensor(dB);
    if (!need(L, {wt, xt, dyt, dxt, dwt, dbt})) return;
    if (!needElems(L, "x", xt, wt->cols) || !needElems(L, "dY", dyt, wt->rows)) return;
    // dW / dB accumulate at the forward's size — "caller zeros".
    if (!needElems(L, "dW", dwt, elems({wt->rows, wt->cols})) || !needElems(L, "dB", dbt, wt->rows)) return;
    for (const Tensor* t : {xt, dyt, dwt, dbt}) {
        if (!needSameDtype(L, "every operand", t, wt)) return;
    }
    BROTENSOR_API_TRY
        brotensor::linear_backward(*wt, *xt, *dyt, *dxt, *dwt, *dbt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_reluForward(void* x, void* y) {
    unary("reluForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::relu_forward(a, b); });
}

void bro_tensor_reluBackward(void* x, void* dY, void* dX) {
    unaryBackward("reluBackward", x, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::relu_backward(a, g, o); });
}

void bro_tensor_tanhForward(void* x, void* y) {
    unary("tanhForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::tanh_forward(a, b); });
}

void bro_tensor_tanhBackward(void* y, void* dY, void* dX) {
    unaryBackward("tanhBackward", y, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::tanh_backward(a, g, o); });
}

void bro_tensor_sigmoidForward(void* x, void* y) {
    unary("sigmoidForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::sigmoid_forward(a, b); });
}

void bro_tensor_sigmoidBackward(void* y, void* dY, void* dX) {
    unaryBackward("sigmoidBackward", y, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::sigmoid_backward(a, g, o); });
}

void bro_tensor_addInplace(void* y, void* x) {
    const char* L = "addInplace";
    auto* yt = toTensor(y);
    auto* xt = toTensor(x);
    if (!need(L, {yt, xt})) return;
    if (!needPair(L, "x", xt, yt)) return;
    BROTENSOR_API_TRY
        brotensor::add_inplace(*yt, *xt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_addScalarInplace(void* y, double s) {
    const char* L = "addScalarInplace";
    auto* yt = toTensor(y);
    if (!need(L, {yt})) return;
    BROTENSOR_API_TRY
        brotensor::add_scalar_inplace(*yt, static_cast<float>(s));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_scaleInplace(void* y, double s) {
    const char* L = "scaleInplace";
    auto* yt = toTensor(y);
    if (!need(L, {yt})) return;
    BROTENSOR_API_TRY
        brotensor::scale_inplace(*yt, static_cast<float>(s));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_mulInplace(void* y, void* x) {
    const char* L = "mulInplace";
    auto* yt = toTensor(y);
    auto* xt = toTensor(x);
    if (!need(L, {yt, xt})) return;
    if (!needPair(L, "x", xt, yt)) return;
    BROTENSOR_API_TRY
        brotensor::mul_inplace(*yt, *xt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_clamp(void* y, double lo, double hi) {
    const char* L = "clamp";
    auto* yt = toTensor(y);
    if (!need(L, {yt})) return;
    BROTENSOR_API_TRY
        brotensor::clamp(*yt, static_cast<float>(lo), static_cast<float>(hi));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_siluForward(void* x, void* y) {
    unary("siluForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::silu_forward(a, b); });
}

void bro_tensor_siluBackward(void* x, void* dY, void* dX) {
    unaryBackward("siluBackward", x, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::silu_backward(a, g, o); });
}

void bro_tensor_geluForward(void* x, void* y) {
    unary("geluForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::gelu_forward(a, b); });
}

void bro_tensor_geluBackward(void* x, void* dY, void* dX) {
    unaryBackward("geluBackward", x, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::gelu_backward(a, g, o); });
}

void bro_tensor_geluExactForward(void* x, void* y) {
    unary("geluExactForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::gelu_exact_forward(a, b); });
}

void bro_tensor_geluExactBackward(void* x, void* dY, void* dX) {
    unaryBackward("geluExactBackward", x, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::gelu_exact_backward(a, g, o); });
}

void bro_tensor_quickGeluForward(void* x, void* y) {
    unary("quickGeluForward", x, y, [](const Tensor& a, Tensor& b) { brotensor::quick_gelu_forward(a, b); });
}

void bro_tensor_quickGeluBackward(void* x, void* dY, void* dX) {
    unaryBackward("quickGeluBackward", x, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::quick_gelu_backward(a, g, o); });
}

void bro_tensor_swigluForward(void* X, void* Y) {
    gatedForward("swigluForward", X, Y, [](const Tensor& a, Tensor& b) { brotensor::swiglu_forward(a, b); });
}

void bro_tensor_swigluBackward(void* X, void* dY, void* dX) {
    gatedBackward("swigluBackward", X, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::swiglu_backward(a, g, o); });
}

void bro_tensor_gegluForward(void* X, void* Y) {
    gatedForward("gegluForward", X, Y, [](const Tensor& a, Tensor& b) { brotensor::geglu_forward(a, b); });
}

void bro_tensor_gegluBackward(void* X, void* dY, void* dX) {
    gatedBackward("gegluBackward", X, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::geglu_backward(a, g, o); });
}

void bro_tensor_gegluExactForward(void* X, void* Y) {
    gatedForward("gegluExactForward", X, Y, [](const Tensor& a, Tensor& b) { brotensor::geglu_exact_forward(a, b); });
}

void bro_tensor_gegluExactBackward(void* X, void* dY, void* dX) {
    gatedBackward("gegluExactBackward", X, dY, dX,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::geglu_exact_backward(a, g, o); });
}

void bro_tensor_softmaxForward(void* logits, void* probs, double temp) {
    const char* L = "softmaxForward";
    auto* lt = toTensor(logits);
    auto* pt = toTensor(probs);
    if (!need(L, {lt, pt})) return;
    BROTENSOR_API_TRY
        if (temp != 1.0 && temp > 0.0) {
            auto scaled = lt->clone();
            brotensor::scale_inplace(scaled, static_cast<float>(1.0 / temp));
            brotensor::softmax_forward(scaled, *pt);
        } else {
            brotensor::softmax_forward(*lt, *pt);
        }
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_softmaxBackward(void* probs, void* dProbs, void* dLogits) {
    unaryBackward("softmaxBackward", probs, dProbs, dLogits,
                  [](const Tensor& a, const Tensor& g, Tensor& o) { brotensor::softmax_backward(a, g, o); });
}

void bro_tensor_matmul(void* A, void* B, void* C) {
    const char* L = "matmul";
    auto* at = toTensor(A);
    auto* bt = toTensor(B);
    auto* ct = toTensor(C);
    if (!need(L, {at, bt, ct})) return;
    if (at->cols != bt->rows) {
        setError("matmul: A is (" + std::to_string(at->rows) + "," + std::to_string(at->cols) +
                 "), B is (" + std::to_string(bt->rows) + "," + std::to_string(bt->cols) + "); A.cols must equal B.rows");
        return;
    }
    if (!needSameDtype(L, "B", bt, at)) return;
    BROTENSOR_API_TRY
        brotensor::matmul(*at, *bt, *ct);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_matmulBackward(void* A, void* B, void* dC, void* dA, void* dB) {
    const char* L = "matmulBackward";
    auto* at = toTensor(A);
    auto* bt = toTensor(B);
    auto* dct = toTensor(dC);
    auto* dat = toTensor(dA);
    auto* dbt = toTensor(dB);
    if (!need(L, {at, bt, dct, dat, dbt})) return;
    if (at->cols != bt->rows) {
        setError("matmulBackward: A.cols must equal B.rows");
        return;
    }
    if (!needElems(L, "dC", dct, elems({at->rows, bt->cols}))) return;
    // dA / dB accumulate — "caller pre-sizes and zeros".
    if (!needSameSize(L, "dA", dat, at) || !needSameSize(L, "dB", dbt, bt)) return;
    for (const Tensor* t : {bt, dct, dat, dbt}) {
        if (!needSameDtype(L, "every operand", t, at)) return;
    }
    BROTENSOR_API_TRY
        brotensor::matmul_backward(*at, *bt, *dct, *dat, *dbt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_conv2dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* Y) {
    const char* L = "conv2dForward";
    auto* xt = toTensor(X);
    auto* wt = toTensor(Wt);
    auto* yt = toTensor(Y);
    if (!need(L, {xt, wt, yt})) return;
    const Tensor* bias_t = tensorFromValue(bias_bits);
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N})) return;
    if (!needElems(L, "X", xt, elems({N, C_in, H, W}))) return;
    if (!needElems(L, "Wt", wt, elems({C_out, C_in / groups, kH, kW}))) return;
    if (!needSameDtype(L, "Wt", wt, xt)) return;
    if (bias_t && (!needElems(L, "bias", bias_t, C_out) || !needSameDtype(L, "bias", bias_t, xt))) return;
    BROTENSOR_API_TRY
        brotensor::conv2d_forward(*xt, *wt, bias_t, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *yt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_conv2dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dX) {
    const char* L = "conv2dBackwardInput";
    auto* wt = toTensor(Wt);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {wt, dyt, dxt})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N})) return;
    if (!needElems(L, "Wt", wt, elems({C_out, C_in / groups, kH, kW}))) return;
    if (!needElems(L, "dY", dyt, elems({N, C_out, H_out, W_out}))) return;
    if (!needSameDtype(L, "dY", dyt, wt)) return;
    BROTENSOR_API_TRY
        brotensor::conv2d_backward_input(*wt, *dyt, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_conv2dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dWt) {
    const char* L = "conv2dBackwardWeight";
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dwt = toTensor(dWt);
    if (!need(L, {xt, dyt, dwt})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N})) return;
    if (!needElems(L, "X", xt, elems({N, C_in, H, W}))) return;
    if (!needElems(L, "dY", dyt, elems({N, C_out, H_out, W_out}))) return;
    // dWt accumulates — "caller zeros".
    if (!needElems(L, "dWt", dwt, elems({C_out, C_in / groups, kH, kW}))) return;
    if (!needSameDtype(L, "dY", dyt, xt) || !needSameDtype(L, "dWt", dwt, xt)) return;
    BROTENSOR_API_TRY
        brotensor::conv2d_backward_weight(*xt, *dyt, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *dwt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_conv2dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t H_out, int32_t W_out, void* dB) {
    const char* L = "conv2dBackwardBias";
    auto* dyt = toTensor(dY);
    auto* dbt = toTensor(dB);
    if (!need(L, {dyt, dbt})) return;
    if (!needElems(L, "dY", dyt, elems({N, C_out, H_out, W_out}))) return;
    if (!needElems(L, "dB", dbt, C_out) || !needSameDtype(L, "dB", dbt, dyt)) return;
    BROTENSOR_API_TRY
        brotensor::conv2d_backward_bias(*dyt, N, C_out, H_out, W_out, *dbt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_upsampleNearest2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    nchwMap("upsampleNearest2xForward", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::upsample_nearest_2x(x, N, C, H, W, y); });
}

// The 2x backwards take the INPUT dims; dY is the upsampled (N, C*2H*2W).
void bro_tensor_upsampleNearest2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    const char* L = "upsampleNearest2xBackward";
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {dyt, dxt})) return;
    if (!needNonNegative(L, {N, C, H, W})) return;
    if (!needElems(L, "dY", dyt, elems({N, C, 2 * static_cast<int64_t>(H), 2 * static_cast<int64_t>(W)}))) return;
    BROTENSOR_API_TRY
        brotensor::upsample_nearest_2x_backward(*dyt, N, C, H, W, *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_upsampleBilinear2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    nchwMap("upsampleBilinear2xForward", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::upsample_bilinear_2x(x, N, C, H, W, y); });
}

void bro_tensor_upsampleBilinear2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    const char* L = "upsampleBilinear2xBackward";
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {dyt, dxt})) return;
    if (!needNonNegative(L, {N, C, H, W})) return;
    if (!needElems(L, "dY", dyt, elems({N, C, 2 * static_cast<int64_t>(H), 2 * static_cast<int64_t>(W)}))) return;
    BROTENSOR_API_TRY
        brotensor::upsample_bilinear_2x_backward(*dyt, N, C, H, W, *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_downsampleAvg2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    if (!needEvenPlane("downsampleAvg2xForward", H, W)) return;
    nchwMap("downsampleAvg2xForward", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::downsample_avg_2x(x, N, C, H, W, y); });
}

void bro_tensor_downsampleAvg2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    const char* L = "downsampleAvg2xBackward";
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!need(L, {dyt, dxt})) return;
    if (!needNonNegative(L, {N, C, H, W}) || !needEvenPlane(L, H, W)) return;
    if (!needElems(L, "dY", dyt, elems({N, C, H / 2, W / 2}))) return;
    BROTENSOR_API_TRY
        brotensor::downsample_avg_2x_backward(*dyt, N, C, H, W, *dxt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_sumRows(void* X, void* Y) {
    unary("sumRows", X, Y, [](const Tensor& a, Tensor& b) { brotensor::sum_rows(a, b); });
}

void bro_tensor_sumCols(void* X, void* Y) {
    unary("sumCols", X, Y, [](const Tensor& a, Tensor& b) { brotensor::sum_cols(a, b); });
}

void bro_tensor_argmaxRows(void* X, void* Idx) {
    unary("argmaxRows", X, Idx, [](const Tensor& a, Tensor& b) { brotensor::argmax_rows(a, b); });
}

// A flat element copy between two tensors of one dtype on one device. Both
// ranges are checked against their tensor: this is a raw memcpy.
void bro_tensor_copyD2D(void* src, int32_t srcOff, void* dst, int32_t dstOff, int32_t n) {
    const char* L = "copyD2D";
    auto* s = toTensor(src);
    auto* d = toTensor(dst);
    if (!need(L, {s, d})) return;
    if (!needNonNegative(L, {srcOff, dstOff, n})) return;
    if (n == 0) return;
    if (!needSameDtype(L, "dst", d, s)) return;
    if (s->device != d->device) {
        setError("copyD2D: src and dst live on different devices");
        return;
    }
    const int64_t srcEnd = static_cast<int64_t>(srcOff) + n;
    const int64_t dstEnd = static_cast<int64_t>(dstOff) + n;
    if (srcEnd > static_cast<int64_t>(s->size()) || dstEnd > static_cast<int64_t>(d->size())) {
        setError("copyD2D: [" + std::to_string(srcOff) + ", " + std::to_string(srcEnd) + ") of a " +
                 std::to_string(s->size()) + "-element src into [" + std::to_string(dstOff) + ", " +
                 std::to_string(dstEnd) + ") of a " + std::to_string(d->size()) + "-element dst is out of range");
        return;
    }
    int elemSize = brotensor::dtype_size_bytes(s->dtype);
    if (elemSize <= 0) elemSize = 4;
    const std::size_t bytes = static_cast<std::size_t>(n) * elemSize;
    const std::size_t s_byte_offset = static_cast<std::size_t>(srcOff) * elemSize;
    const std::size_t d_byte_offset = static_cast<std::size_t>(dstOff) * elemSize;
    BROTENSOR_API_TRY
        if (s->is_host()) {
            std::memmove(reinterpret_cast<char*>(d->data) + d_byte_offset,
                         reinterpret_cast<const char*>(s->data) + s_byte_offset,
                         bytes);
        } else {
            brotensor::detail::alloc_for(d->device).memcpy_d2d(
                reinterpret_cast<char*>(d->data) + d_byte_offset,
                reinterpret_cast<const char*>(s->data) + s_byte_offset,
                bytes, d->device.index);
        }
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_nchwToSequence(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    nchwMap("nchwToSequence", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::nchw_to_sequence(x, N, C, H, W, y); });
}

void bro_tensor_sequenceToNchw(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    nchwMap("sequenceToNchw", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::sequence_to_nchw(x, N, C, H, W, y); });
}

void bro_tensor_interp2dForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y) {
    if (!needNonNegative("interp2dForward", {H_out, W_out})) return;
    nchwMap("interp2dForward", X, N, C, H_in, W_in, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::interp2d_forward(x, N, C, H_in, W_in, H_out, W_out, mode, y); });
}

void bro_tensor_interp2dAlignCornersForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y) {
    if (!needNonNegative("interp2dAlignCornersForward", {H_out, W_out})) return;
    nchwMap("interp2dAlignCornersForward", X, N, C, H_in, W_in, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::interp2d_align_corners_forward(x, N, C, H_in, W_in, H_out, W_out, mode, y); });
}

void bro_tensor_unfold2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padT, int32_t padB, int32_t padL, int32_t padR, int32_t mode, void* Y) {
    const char* L = "unfold2dForward";
    if (!needPositive(L, {kH, kW, sH, sW}) || !needNonNegative(L, {padT, padB, padL, padR})) return;
    nchwMap(L, X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::unfold2d_forward(x, N, C, H, W, kH, kW, sH, sW, padT, padB, padL, padR, mode, y); });
}

void bro_tensor_l2NormalizeNchwForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y) {
    nchwMap("l2NormalizeNchwForward", X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::l2_normalize_nchw_forward(x, N, C, H, W, static_cast<float>(eps), y); });
}

void bro_tensor_convexUpsampleForward(void* X, void* Mask, int32_t N, int32_t C, int32_t H, int32_t W, int32_t scale, void* Y) {
    const char* L = "convexUpsampleForward";
    auto* mt = toTensor(Mask);
    if (!need(L, {mt})) return;
    if (!needPositive(L, {scale})) return;
    // Mask is (N, 9*scale*scale*H*W) and shares X's dtype.
    if (!needElems(L, "Mask", mt, elems({N, 9, scale, scale, H, W}))) return;
    if (auto* xt = toTensor(X); xt && !needSameDtype(L, "Mask", mt, xt)) return;
    nchwMap(L, X, N, C, H, W, Y,
            [&](const Tensor& x, Tensor& y) { brotensor::convex_upsample_forward(x, *mt, N, C, H, W, scale, y); });
}

double bro_tensor_mseVecForward(void* pred, void* target) {
    const char* L = "mseVecForward";
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    if (!need(L, {pt, tt})) return 0.0;
    if (!needPair(L, "target", tt, pt)) return 0.0;
    BROTENSOR_API_TRY
        return static_cast<double>(brotensor::mse_vec_forward(*pt, *tt));
    BROTENSOR_API_CATCH(L)
    return 0.0;
}

void bro_tensor_mseVecBackward(void* pred, void* target, void* dPred) {
    const char* L = "mseVecBackward";
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    auto* dpt = toTensor(dPred);
    if (!need(L, {pt, tt, dpt})) return;
    if (!needPair(L, "target", tt, pt)) return;
    BROTENSOR_API_TRY
        brotensor::mse_vec_backward(*pt, *tt, *dpt);
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_mseVecPerSample(void* pred, void* target, void* dPred, void* lossPerSample) {
    const char* L = "mseVecPerSample";
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    auto* dpt = toTensor(dPred);
    auto* lt = toTensor(lossPerSample);
    if (!need(L, {pt, tt, dpt, lt})) return;
    if (!needPair(L, "target", tt, pt)) return;
    BROTENSOR_API_TRY
        brotensor::mse_vec_per_sample(*pt, *tt, *dpt, *lt);
    BROTENSOR_API_CATCH(L)
}

} // extern "C"
