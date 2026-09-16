#include "native_tensor_decl.h"
#include "api_internal.h"
#include "brotensor/detail/dispatch.h"

#include <cstring>

using namespace brotensor::api;

extern "C" {

void bro_tensor_linearForward(void* W, void* b, void* x, void* y) {
    auto* wt = toTensor(W);
    auto* bt = toTensor(b);
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!wt || !bt || !xt || !yt) return;
    brotensor::linear_forward(*wt, *bt, *xt, *yt);
}

void bro_tensor_linearBackward(void* W, void* x, void* dY, void* dX, void* dW, void* dB) {
    auto* wt = toTensor(W);
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    auto* dwt = toTensor(dW);
    auto* dbt = toTensor(dB);
    if (!wt || !xt || !dyt || !dxt || !dwt || !dbt) return;
    brotensor::linear_backward(*wt, *xt, *dyt, *dxt, *dwt, *dbt);
}

void bro_tensor_reluForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::relu_forward(*xt, *yt);
}

void bro_tensor_reluBackward(void* x, void* dY, void* dX) {
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::relu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_tanhForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::tanh_forward(*xt, *yt);
}

void bro_tensor_tanhBackward(void* y, void* dY, void* dX) {
    auto* yt = toTensor(y);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!yt || !dyt || !dxt) return;
    brotensor::tanh_backward(*yt, *dyt, *dxt);
}

void bro_tensor_sigmoidForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::sigmoid_forward(*xt, *yt);
}

void bro_tensor_sigmoidBackward(void* y, void* dY, void* dX) {
    auto* yt = toTensor(y);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!yt || !dyt || !dxt) return;
    brotensor::sigmoid_backward(*yt, *dyt, *dxt);
}

void bro_tensor_addInplace(void* y, void* x) {
    auto* yt = toTensor(y);
    auto* xt = toTensor(x);
    if (!yt || !xt) return;
    brotensor::add_inplace(*yt, *xt);
}

void bro_tensor_addScalarInplace(void* y, double s) {
    auto* yt = toTensor(y);
    if (!yt) return;
    brotensor::add_scalar_inplace(*yt, static_cast<float>(s));
}

void bro_tensor_scaleInplace(void* y, double s) {
    auto* yt = toTensor(y);
    if (!yt) return;
    brotensor::scale_inplace(*yt, static_cast<float>(s));
}

void bro_tensor_mulInplace(void* y, void* x) {
    auto* yt = toTensor(y);
    auto* xt = toTensor(x);
    if (!yt || !xt) return;
    brotensor::mul_inplace(*yt, *xt);
}

void bro_tensor_clamp(void* y, double lo, double hi) {
    auto* yt = toTensor(y);
    if (!yt) return;
    brotensor::clamp(*yt, static_cast<float>(lo), static_cast<float>(hi));
}

void bro_tensor_siluForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::silu_forward(*xt, *yt);
}

void bro_tensor_siluBackward(void* x, void* dY, void* dX) {
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::silu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_geluForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::gelu_forward(*xt, *yt);
}

void bro_tensor_geluBackward(void* x, void* dY, void* dX) {
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::gelu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_geluExactForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::gelu_exact_forward(*xt, *yt);
}

void bro_tensor_geluExactBackward(void* x, void* dY, void* dX) {
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::gelu_exact_backward(*xt, *dyt, *dxt);
}

void bro_tensor_quickGeluForward(void* x, void* y) {
    auto* xt = toTensor(x);
    auto* yt = toTensor(y);
    if (!xt || !yt) return;
    brotensor::quick_gelu_forward(*xt, *yt);
}

void bro_tensor_quickGeluBackward(void* x, void* dY, void* dX) {
    auto* xt = toTensor(x);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::quick_gelu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_swigluForward(void* X, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::swiglu_forward(*xt, *yt);
}

void bro_tensor_swigluBackward(void* X, void* dY, void* dX) {
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::swiglu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_gegluForward(void* X, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::geglu_forward(*xt, *yt);
}

void bro_tensor_gegluBackward(void* X, void* dY, void* dX) {
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::geglu_backward(*xt, *dyt, *dxt);
}

void bro_tensor_gegluExactForward(void* X, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::geglu_exact_forward(*xt, *yt);
}

void bro_tensor_gegluExactBackward(void* X, void* dY, void* dX) {
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!xt || !dyt || !dxt) return;
    brotensor::geglu_exact_backward(*xt, *dyt, *dxt);
}

void bro_tensor_softmaxForward(void* logits, void* probs, double temp) {
    auto* lt = toTensor(logits);
    auto* pt = toTensor(probs);
    if (!lt || !pt) return;
    if (temp != 1.0 && temp > 0.0) {
        auto scaled = lt->clone();
        brotensor::scale_inplace(scaled, static_cast<float>(1.0 / temp));
        brotensor::softmax_forward(scaled, *pt);
    } else {
        brotensor::softmax_forward(*lt, *pt);
    }
}

void bro_tensor_softmaxBackward(void* probs, void* dProbs, void* dLogits) {
    auto* pt = toTensor(probs);
    auto* dpt = toTensor(dProbs);
    auto* dlt = toTensor(dLogits);
    if (!pt || !dpt || !dlt) return;
    brotensor::softmax_backward(*pt, *dpt, *dlt);
}

void bro_tensor_matmul(void* A, void* B, void* C) {
    auto* at = toTensor(A);
    auto* bt = toTensor(B);
    auto* ct = toTensor(C);
    if (!at || !bt || !ct) return;
    brotensor::matmul(*at, *bt, *ct);
}

void bro_tensor_matmulBackward(void* A, void* B, void* dC, void* dA, void* dB) {
    auto* at = toTensor(A);
    auto* bt = toTensor(B);
    auto* dct = toTensor(dC);
    auto* dat = toTensor(dA);
    auto* dbt = toTensor(dB);
    if (!at || !bt || !dct || !dat || !dbt) return;
    brotensor::matmul_backward(*at, *bt, *dct, *dat, *dbt);
}

void bro_tensor_conv2dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* Y) {
    auto* xt = toTensor(X);
    auto* wt = toTensor(Wt);
    auto* yt = toTensor(Y);
    if (!xt || !wt || !yt) return;
    const brotensor::Tensor* bias_t = nullptr;
    void* data = bronze::embed::handleData(bronze::Value{bias_bits});
    if (data) bias_t = toTensor(data);
    brotensor::conv2d_forward(*xt, *wt, bias_t, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *yt);
}

void bro_tensor_conv2dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dX) {
    auto* wt = toTensor(Wt);
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!wt || !dyt || !dxt) return;
    brotensor::conv2d_backward_input(*wt, *dyt, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *dxt);
}

void bro_tensor_conv2dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in, int32_t H, int32_t W, int32_t C_out, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW, int32_t groups, void* dWt) {
    auto* xt = toTensor(X);
    auto* dyt = toTensor(dY);
    auto* dwt = toTensor(dWt);
    if (!xt || !dyt || !dwt) return;
    brotensor::conv2d_backward_weight(*xt, *dyt, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, *dwt);
}

void bro_tensor_conv2dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t H_out, int32_t W_out, void* dB) {
    auto* dyt = toTensor(dY);
    auto* dbt = toTensor(dB);
    if (!dyt || !dbt) return;
    brotensor::conv2d_backward_bias(*dyt, N, C_out, H_out, W_out, *dbt);
}

void bro_tensor_upsampleNearest2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::upsample_nearest_2x(*xt, N, C, H, W, *yt);
}

void bro_tensor_upsampleNearest2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!dyt || !dxt) return;
    brotensor::upsample_nearest_2x_backward(*dyt, N, C, H, W, *dxt);
}

void bro_tensor_upsampleBilinear2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::upsample_bilinear_2x(*xt, N, C, H, W, *yt);
}

void bro_tensor_upsampleBilinear2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!dyt || !dxt) return;
    brotensor::upsample_bilinear_2x_backward(*dyt, N, C, H, W, *dxt);
}

void bro_tensor_downsampleAvg2xForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::downsample_avg_2x(*xt, N, C, H, W, *yt);
}

void bro_tensor_downsampleAvg2xBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W, void* dX) {
    auto* dyt = toTensor(dY);
    auto* dxt = toTensor(dX);
    if (!dyt || !dxt) return;
    brotensor::downsample_avg_2x_backward(*dyt, N, C, H, W, *dxt);
}

void bro_tensor_sumRows(void* X, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::sum_rows(*xt, *yt);
}

void bro_tensor_sumCols(void* X, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::sum_cols(*xt, *yt);
}

void bro_tensor_argmaxRows(void* X, void* Idx) {
    auto* xt = toTensor(X);
    auto* idxt = toTensor(Idx);
    if (!xt || !idxt) return;
    brotensor::argmax_rows(*xt, *idxt);
}

void bro_tensor_copyD2D(void* src, int32_t srcOff, void* dst, int32_t dstOff, int32_t n) {
    auto* s = toTensor(src);
    auto* d = toTensor(dst);
    if (!s || !d || n <= 0) return;
    int elemSize = brotensor::dtype_size_bytes(s->dtype);
    if (elemSize <= 0) elemSize = 4;
    std::size_t bytes = static_cast<std::size_t>(n) * elemSize;
    std::size_t s_byte_offset = static_cast<std::size_t>(srcOff) * elemSize;
    std::size_t d_byte_offset = static_cast<std::size_t>(dstOff) * elemSize;
    if (s->is_host() && d->is_host()) {
        std::memmove(reinterpret_cast<char*>(d->data) + d_byte_offset,
                     reinterpret_cast<const char*>(s->data) + s_byte_offset,
                     bytes);
    } else {
        brotensor::detail::alloc_for(d->device).memcpy_d2d(
            reinterpret_cast<char*>(d->data) + d_byte_offset,
            reinterpret_cast<const char*>(s->data) + s_byte_offset,
            bytes, d->device.index);
    }
}

void bro_tensor_nchwToSequence(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::nchw_to_sequence(*xt, N, C, H, W, *yt);
}

void bro_tensor_sequenceToNchw(void* X, int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::sequence_to_nchw(*xt, N, C, H, W, *yt);
}

void bro_tensor_interp2dForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::interp2d_forward(*xt, N, C, H_in, W_in, H_out, W_out, mode, *yt);
}

void bro_tensor_interp2dAlignCornersForward(void* X, int32_t N, int32_t C, int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, int32_t mode, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::interp2d_align_corners_forward(*xt, N, C, H_in, W_in, H_out, W_out, mode, *yt);
}

void bro_tensor_unfold2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padT, int32_t padB, int32_t padL, int32_t padR, int32_t mode, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::unfold2d_forward(*xt, N, C, H, W, kH, kW, sH, sW, padT, padB, padL, padR, mode, *yt);
}

void bro_tensor_l2NormalizeNchwForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y) {
    auto* xt = toTensor(X);
    auto* yt = toTensor(Y);
    if (!xt || !yt) return;
    brotensor::l2_normalize_nchw_forward(*xt, N, C, H, W, static_cast<float>(eps), *yt);
}

void bro_tensor_convexUpsampleForward(void* X, void* Mask, int32_t N, int32_t C, int32_t H, int32_t W, int32_t scale, void* Y) {
    auto* xt = toTensor(X);
    auto* mt = toTensor(Mask);
    auto* yt = toTensor(Y);
    if (!xt || !mt || !yt) return;
    brotensor::convex_upsample_forward(*xt, *mt, N, C, H, W, scale, *yt);
}

double bro_tensor_mseVecForward(void* pred, void* target) {
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    if (!pt || !tt) return 0.0;
    return static_cast<double>(brotensor::mse_vec_forward(*pt, *tt));
}

void bro_tensor_mseVecBackward(void* pred, void* target, void* dPred) {
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    auto* dpt = toTensor(dPred);
    if (!pt || !tt || !dpt) return;
    brotensor::mse_vec_backward(*pt, *tt, *dpt);
}

void bro_tensor_mseVecPerSample(void* pred, void* target, void* dPred, void* lossPerSample) {
    auto* pt = toTensor(pred);
    auto* tt = toTensor(target);
    auto* dpt = toTensor(dPred);
    auto* lt = toTensor(lossPerSample);
    if (!pt || !tt || !dpt || !lt) return;
    brotensor::mse_vec_per_sample(*pt, *tt, *dpt, *lt);
}

} // extern "C"
