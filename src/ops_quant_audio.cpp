// brotensor public op wrappers — diffusion, quantisation, spectral/FFT, 1D audio, sampling, noise.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor {

namespace {

inline void check_quant_block_cols(const char* op, const Tensor& t,
                                   const char* name, Dtype expected_dt) {
    detail::check_quant_cols(op, t, name);
    const int bs = dtype_block_size(expected_dt);
    if (t.rows <= 0 || t.cols <= 0 || (t.cols % bs) != 0) {
        detail::throw_quant_cols_error(op, name, expected_dt, t.rows, t.cols);
    }
}

} // namespace

// ─── Diffusion sampler steps + timestep embedding ──────────────────────────

void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t, Tensor& x_prev) {
    const auto& v = detail::dispatch(x_t, eps_pred, x_prev);
    if (!v.ddim_step) detail::throw_not_implemented("ddim_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    v.ddim_step(x_t, eps_pred, alpha_t, alpha_prev, sigma_t, x_prev);
}

void euler_step(const Tensor& x_t, const Tensor& eps_pred,
                float sigma_t, float sigma_prev, Tensor& x_prev) {
    const auto& v = detail::dispatch(x_t, eps_pred, x_prev);
    if (!v.euler_step) detail::throw_not_implemented("euler_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    v.euler_step(x_t, eps_pred, sigma_t, sigma_prev, x_prev);
}

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred,
                   const Tensor& x0_prev, float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out) {
    const auto& v = detail::dispatch(x_t, eps_pred, x0_prev, x_prev, x0_out);
    if (!v.dpmpp_2m_step) detail::throw_not_implemented("dpmpp_2m_step", x_t.device);
    detail::adopt_output(x_prev, x_t.device);
    detail::adopt_output(x0_out, x_t.device);
    v.dpmpp_2m_step(x_t, eps_pred, x0_prev, sigma_t, c_xt, c_x0t, c_x0prev,
                    x_prev, x0_out);
}

void timestep_embedding(const Tensor& timesteps, int dim, float max_period,
                        Tensor& Y) {
    const auto& v = detail::dispatch(timesteps, Y);
    if (!v.timestep_embedding) detail::throw_not_implemented("timestep_embedding", timesteps.device);
    detail::adopt_output(Y, timesteps.device);
    v.timestep_embedding(timesteps, dim, max_period, Y);
}

// ─── INT8 / block-quant weight-only quantisation ───────────────────────────

void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
        const uint16_t* row = W_fp16 + r * static_cast<std::size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + r * static_cast<std::size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    });
}

void matmul_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                       const Tensor& X, Tensor& Y) {
    const auto& v = detail::dispatch(W_int8, scales, X, Y);
    if (!v.matmul_int8w_fp16) detail::throw_not_implemented("matmul_int8w_fp16", W_int8.device);
    detail::adopt_output(Y, W_int8.device);
    v.matmul_int8w_fp16(W_int8, scales, X, Y);
}

void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD, Tensor& Y_BD) {
    const auto& v = detail::dispatch_with_opts(W_int8, scales, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_int8w_fp16)
        detail::throw_not_implemented("linear_forward_batched_int8w_fp16", W_int8.device);
    detail::adopt_output(Y_BD, W_int8.device);
    v.linear_forward_batched_int8w_fp16(W_int8, scales, bias, X_BD, Y_BD);
}

void quantize_rows(const Tensor& W, Dtype fmt, Tensor& Wq, Tensor& scales) {
    const auto& v = detail::dispatch_with_opts(W, {&Wq, &scales});
    if (!v.quantize_rows) detail::throw_not_implemented("quantize_rows", W.device);
    detail::adopt_output(Wq, W.device);
    detail::adopt_output(scales, W.device);
    v.quantize_rows(W, fmt, Wq, scales);
}

void dequantize_rows(const Tensor& Wq, const Tensor& scales, Dtype out, Tensor& W) {
    BROTENSOR_CHECK_QUANT_COLS("dequantize_rows", Wq);
    const auto& v = detail::dispatch_with_opts(Wq, {&W});
    if (!v.dequantize_rows) detail::throw_not_implemented("dequantize_rows", Wq.device);
    detail::adopt_output(W, Wq.device);
    v.dequantize_rows(Wq, scales, out, W);
}

void linear_forward_batched_wq_ex(const Tensor& Wq, const Tensor& scales, const Tensor* bias,
                                  const Tensor& X_BD, int act, int epilogue, Tensor& Y_BD) {
    BROTENSOR_CHECK_QUANT_COLS("linear_forward_batched_wq_ex", Wq);
    const auto& v = detail::dispatch_with_opts(Wq, scales, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_wq_ex)
        detail::throw_not_implemented("linear_forward_batched_wq_ex", Wq.device);
    detail::adopt_output(Y_BD, Wq.device);
    v.linear_forward_batched_wq_ex(Wq, scales, bias, X_BD, act, epilogue, Y_BD);
}

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    check_quant_block_cols("dequant_q4k_to_fp16", W_q4k, "W_q4k", Dtype::Q4_K);
    const auto& v = detail::dispatch(W_q4k, W_fp16);
    if (!v.dequant_q4k_to_fp16)
        detail::throw_not_implemented("dequant_q4k_to_fp16", W_q4k.device);
    detail::adopt_output(W_fp16, W_q4k.device);
    v.dequant_q4k_to_fp16(W_q4k, W_fp16);
}

void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    check_quant_block_cols("linear_forward_q4k_fp16", W_q4k, "W_q4k", Dtype::Q4_K);
    const auto& v = detail::dispatch_with_opts(W_q4k, {bias, &x, &y});
    if (!v.linear_forward_q4k_fp16)
        detail::throw_not_implemented("linear_forward_q4k_fp16", W_q4k.device);
    detail::adopt_output(y, W_q4k.device);
    v.linear_forward_q4k_fp16(W_q4k, bias, x, y);
}

void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    check_quant_block_cols("linear_forward_batched_q4k_fp16", W_q4k, "W_q4k", Dtype::Q4_K);
    const auto& v = detail::dispatch_with_opts(W_q4k, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_q4k_fp16)
        detail::throw_not_implemented("linear_forward_batched_q4k_fp16", W_q4k.device);
    detail::adopt_output(Y_BD, W_q4k.device);
    v.linear_forward_batched_q4k_fp16(W_q4k, bias, X_BD, Y_BD);
}

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    check_quant_block_cols("dequant_q8_0_to_fp16", W_q8, "W_q8", Dtype::Q8_0);
    const auto& v = detail::dispatch(W_q8, W_fp16);
    if (!v.dequant_q8_0_to_fp16)
        detail::throw_not_implemented("dequant_q8_0_to_fp16", W_q8.device);
    detail::adopt_output(W_fp16, W_q8.device);
    v.dequant_q8_0_to_fp16(W_q8, W_fp16);
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y) {
    check_quant_block_cols("linear_forward_q8_0_fp16", W_q8, "W_q8", Dtype::Q8_0);
    const auto& v = detail::dispatch_with_opts(W_q8, {bias, &x, &y});
    if (!v.linear_forward_q8_0_fp16)
        detail::throw_not_implemented("linear_forward_q8_0_fp16", W_q8.device);
    detail::adopt_output(y, W_q8.device);
    v.linear_forward_q8_0_fp16(W_q8, bias, x, y);
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD) {
    check_quant_block_cols("linear_forward_batched_q8_0_fp16", W_q8, "W_q8", Dtype::Q8_0);
    const auto& v = detail::dispatch_with_opts(W_q8, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_q8_0_fp16)
        detail::throw_not_implemented("linear_forward_batched_q8_0_fp16", W_q8.device);
    detail::adopt_output(Y_BD, W_q8.device);
    v.linear_forward_batched_q8_0_fp16(W_q8, bias, X_BD, Y_BD);
}

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    check_quant_block_cols("dequant_q6k_to_fp16", W_q6k, "W_q6k", Dtype::Q6_K);
    const auto& v = detail::dispatch(W_q6k, W_fp16);
    if (!v.dequant_q6k_to_fp16)
        detail::throw_not_implemented("dequant_q6k_to_fp16", W_q6k.device);
    detail::adopt_output(W_fp16, W_q6k.device);
    v.dequant_q6k_to_fp16(W_q6k, W_fp16);
}

void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    check_quant_block_cols("linear_forward_q6k_fp16", W_q6k, "W_q6k", Dtype::Q6_K);
    const auto& v = detail::dispatch_with_opts(W_q6k, {bias, &x, &y});
    if (!v.linear_forward_q6k_fp16)
        detail::throw_not_implemented("linear_forward_q6k_fp16", W_q6k.device);
    detail::adopt_output(y, W_q6k.device);
    v.linear_forward_q6k_fp16(W_q6k, bias, x, y);
}

void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    check_quant_block_cols("linear_forward_batched_q6k_fp16", W_q6k, "W_q6k", Dtype::Q6_K);
    const auto& v = detail::dispatch_with_opts(W_q6k, {bias, &X_BD, &Y_BD});
    if (!v.linear_forward_batched_q6k_fp16)
        detail::throw_not_implemented("linear_forward_batched_q6k_fp16", W_q6k.device);
    detail::adopt_output(Y_BD, W_q6k.device);
    v.linear_forward_batched_q6k_fp16(W_q6k, bias, X_BD, Y_BD);
}

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8,
                                           const Tensor& sk,
                                           const Tensor* bk,
                                           const Tensor& Wv_int8,
                                           const Tensor& sv,
                                           const Tensor* bv,
                                           Tensor& K_out, Tensor& V_out) {
    const auto& v = detail::dispatch_with_opts(
        ctx, Wk_int8, {&sk, bk, &Wv_int8, &sv, bv, &K_out, &V_out});
    if (!v.flash_attention_project_kv_int8w_fp16)
        detail::throw_not_implemented("flash_attention_project_kv_int8w_fp16", ctx.device);
    detail::adopt_output(K_out, ctx.device);
    detail::adopt_output(V_out, ctx.device);
    v.flash_attention_project_kv_int8w_fp16(ctx, Wk_int8, sk, bk,
                                            Wv_int8, sv, bv, K_out, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                 const Tensor& K, const Tensor& V,
                                                 const Tensor& Wq_int8,
                                                 const Tensor& sq,
                                                 const Tensor* bq,
                                                 const Tensor& Wo_int8,
                                                 const Tensor& so,
                                                 const Tensor* bo,
                                                 const float* d_mask,
                                                 int num_heads, bool causal,
                                                 Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, K, {&V, &Wq_int8, &sq, bq, &Wo_int8, &so, bo, &O});
    if (!v.flash_attention_q_with_kv_cached_int8w_fp16)
        detail::throw_not_implemented("flash_attention_q_with_kv_cached_int8w_fp16", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_q_with_kv_cached_int8w_fp16(X, K, V, Wq_int8, sq, bq,
                                                  Wo_int8, so, bo,
                                                  d_mask, num_heads, causal, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X, const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask, int num_heads, bool causal,
                                     Tensor& O) {
    const auto& v = detail::dispatch_with_opts(
        X, Wq_int8, {Ctx, &sq, bq, &Wk_int8, &sk, bk, &Wv_int8, &sv, bv,
                     &Wo_int8, &so, bo, &O});
    if (!v.flash_attention_qkvo_int8w_fp16)
        detail::throw_not_implemented("flash_attention_qkvo_int8w_fp16", X.device);
    detail::adopt_output(O, X.device);
    v.flash_attention_qkvo_int8w_fp16(X, Ctx, Wq_int8, sq, bq,
                                      Wk_int8, sk, bk, Wv_int8, sv, bv,
                                      Wo_int8, so, bo,
                                      d_mask, num_heads, causal, O);
}

// ─── Spectral / FFT core (brosoundml) ──────────────────────────────────────

void complex_mul(const Tensor& a, const Tensor& b, Tensor& y) {
    const auto& v = detail::dispatch(a, b, y);
    if (!v.complex_mul) detail::throw_not_implemented("complex_mul", a.device);
    detail::adopt_output(y, a.device);
    v.complex_mul(a, b, y);
}
void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB) {
    const auto& v = detail::dispatch(a, b, dY, dA, dB);
    if (!v.complex_mul_backward)
        detail::throw_not_implemented("complex_mul_backward", a.device);
    detail::adopt_output(dA, a.device);
    detail::adopt_output(dB, a.device);
    v.complex_mul_backward(a, b, dY, dA, dB);
}
void complex_abs(const Tensor& z, Tensor& y) {
    const auto& v = detail::dispatch(z, y);
    if (!v.complex_abs) detail::throw_not_implemented("complex_abs", z.device);
    detail::adopt_output(y, z.device);
    v.complex_abs(z, y);
}
void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ) {
    const auto& v = detail::dispatch(z, dY, dZ);
    if (!v.complex_abs_backward)
        detail::throw_not_implemented("complex_abs_backward", z.device);
    detail::adopt_output(dZ, z.device);
    v.complex_abs_backward(z, dY, dZ);
}
void complex_angle(const Tensor& z, Tensor& y) {
    const auto& v = detail::dispatch(z, y);
    if (!v.complex_angle) detail::throw_not_implemented("complex_angle", z.device);
    detail::adopt_output(y, z.device);
    v.complex_angle(z, y);
}
void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y) {
    const auto& v = detail::dispatch(mag, phase, y);
    if (!v.complex_from_polar)
        detail::throw_not_implemented("complex_from_polar", mag.device);
    detail::adopt_output(y, mag.device);
    v.complex_from_polar(mag, phase, y);
}
void fft(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.fft) detail::throw_not_implemented("fft", x.device);
    detail::adopt_output(y, x.device);
    v.fft(x, y);
}
void ifft(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.ifft) detail::throw_not_implemented("ifft", x.device);
    detail::adopt_output(y, x.device);
    v.ifft(x, y);
}
void rfft(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.rfft) detail::throw_not_implemented("rfft", x.device);
    detail::adopt_output(y, x.device);
    v.rfft(x, y);
}
void irfft(const Tensor& x, int L, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.irfft) detail::throw_not_implemented("irfft", x.device);
    detail::adopt_output(y, x.device);
    v.irfft(x, L, y);
}
void rfft_backward(const Tensor& dY, int L, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.rfft_backward) detail::throw_not_implemented("rfft_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.rfft_backward(dY, L, dX);
}
void irfft_backward(const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.irfft_backward)
        detail::throw_not_implemented("irfft_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.irfft_backward(dY, dX);
}

void stft(const Tensor& signal, const Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, Tensor& spec) {
    const auto& v = detail::dispatch(signal, window, spec);
    if (!v.stft) detail::throw_not_implemented("stft", signal.device);
    detail::adopt_output(spec, signal.device);
    v.stft(signal, window, N, n_fft, hop_length, win_length,
           center, normalized, spec);
}
void stft_backward(const Tensor& dSpec, const Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   Tensor& dSignal) {
    const auto& v = detail::dispatch(dSpec, window, dSignal);
    if (!v.stft_backward)
        detail::throw_not_implemented("stft_backward", dSpec.device);
    detail::adopt_output(dSignal, dSpec.device);
    v.stft_backward(dSpec, window, N, signal_len, n_fft, hop_length,
                    win_length, center, normalized, dSignal);
}
void istft(const Tensor& spec, const Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, Tensor& signal) {
    const auto& v = detail::dispatch(spec, window, signal);
    if (!v.istft) detail::throw_not_implemented("istft", spec.device);
    detail::adopt_output(signal, spec.device);
    v.istft(spec, window, N, signal_len, n_fft, hop_length, win_length,
            center, normalized, signal);
}
void istft_backward(const Tensor& dSignal, const Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    Tensor& dSpec) {
    const auto& v = detail::dispatch(dSignal, window, dSpec);
    if (!v.istft_backward)
        detail::throw_not_implemented("istft_backward", dSignal.device);
    detail::adopt_output(dSpec, dSignal.device);
    v.istft_backward(dSignal, window, N, signal_len, n_fft, hop_length,
                     win_length, center, normalized, dSpec);
}

// ─── 1D convolution family (brosoundml) ────────────────────────────────────

void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("conv_transpose1d_forward", X, N, C_in, L);
    const auto& v = detail::dispatch_with_opts(X, Wt, {bias, &Y});
    if (!v.conv_transpose1d_forward)
        detail::throw_not_implemented("conv_transpose1d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv_transpose1d_forward(X, Wt, bias, N, C_in, L, C_out, kL, stride,
                               padding, output_padding, dilation, groups, Y);
}

void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, Tensor& dX) {
    if (N <= 0 || C_in <= 0 || L <= 0 || C_out <= 0) {
        detail::throw_spatial_1d_error("conv_transpose1d_backward_input", "dY",
                                       N, C_in, L, dY.rows, dY.cols);
    }
    const auto& v = detail::dispatch(Wt, dY, dX);
    if (!v.conv_transpose1d_backward_input)
        detail::throw_not_implemented("conv_transpose1d_backward_input", Wt.device);
    detail::adopt_output(dX, Wt.device);
    v.conv_transpose1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride,
                                      padding, output_padding, dilation,
                                      groups, dX);
}

void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, Tensor& dWt) {
    BROTENSOR_CHECK_SPATIAL_1D("conv_transpose1d_backward_weight", X, N, C_in, L);
    const auto& v = detail::dispatch(X, dY, dWt);
    if (!v.conv_transpose1d_backward_weight)
        detail::throw_not_implemented("conv_transpose1d_backward_weight", X.device);
    detail::adopt_output(dWt, X.device);
    v.conv_transpose1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride,
                                       padding, output_padding, dilation,
                                       groups, dWt);
}

void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int L_out, Tensor& dB) {
    BROTENSOR_CHECK_SPATIAL_1D("conv_transpose1d_backward_bias", dY, N, C_out, L_out);
    const auto& v = detail::dispatch(dY, dB);
    if (!v.conv_transpose1d_backward_bias)
        detail::throw_not_implemented("conv_transpose1d_backward_bias", dY.device);
    detail::adopt_output(dB, dY.device);
    v.conv_transpose1d_backward_bias(dY, N, C_out, L_out, dB);
}

void causal_conv1d_update(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          Tensor& state, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("causal_conv1d_update", X, N, C, L_step);
    const auto& v = detail::dispatch_with_opts(X, Wt, {bias, &state, &Y});
    if (!v.causal_conv1d_update)
        detail::throw_not_implemented("causal_conv1d_update", X.device);
    detail::adopt_output(state, X.device);
    detail::adopt_output(Y, X.device);
    v.causal_conv1d_update(X, Wt, bias, N, C, L_step, kL, dilation, state, Y);
}

void pad1d_forward(const Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("pad1d_forward", X, N, C, L);
    const auto& v = detail::dispatch(X, Y);
    if (!v.pad1d_forward)
        detail::throw_not_implemented("pad1d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.pad1d_forward(X, N, C, L, pad_left, pad_right, mode, Y);
}

void pad1d_backward(const Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode, Tensor& dX) {
    if (N <= 0 || C <= 0 || L <= 0) {
        detail::throw_spatial_1d_error("pad1d_backward", "dY",
                                       N, C, L, dY.rows, dY.cols);
    }
    const int L_out = L + pad_left + pad_right;
    if (L_out > 0) {
        BROTENSOR_CHECK_SPATIAL_1D("pad1d_backward", dY, N, C, L_out);
    }
    const auto& v = detail::dispatch(dY, dX);
    if (!v.pad1d_backward)
        detail::throw_not_implemented("pad1d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.pad1d_backward(dY, N, C, L, pad_left, pad_right, mode, dX);
}

// ─── Vocoder / codec activations (brosoundml) ──────────────────────────────

void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("snake_forward", X, N, C, L);
    const auto& v = detail::dispatch_with_opts(X, alpha, {beta, &Y});
    if (!v.snake_forward)
        detail::throw_not_implemented("snake_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.snake_forward(X, alpha, beta, N, C, L, Y);
}

void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta) {
    BROTENSOR_CHECK_SPATIAL_1D("snake_backward", X, N, C, L);
    BROTENSOR_CHECK_SPATIAL_1D("snake_backward", dY, N, C, L);
    const auto& v = detail::dispatch_with_opts(
        X, alpha, {beta, &dY, &dX, &dAlpha, dBeta});
    if (!v.snake_backward)
        detail::throw_not_implemented("snake_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dAlpha, X.device);
    if (dBeta) detail::adopt_output(*dBeta, X.device);
    v.snake_backward(X, alpha, beta, dY, N, C, L, dX, dAlpha, dBeta);
}

void elu_forward(const Tensor& x, float alpha, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.elu_forward) detail::throw_not_implemented("elu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.elu_forward(x, alpha, y);
}

void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.elu_backward) detail::throw_not_implemented("elu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.elu_backward(x, dY, alpha, dX);
}

void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.leaky_relu_forward)
        detail::throw_not_implemented("leaky_relu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.leaky_relu_forward(x, negative_slope, y);
}

void leaky_relu_backward(const Tensor& x, const Tensor& dY,
                         float negative_slope, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.leaky_relu_backward)
        detail::throw_not_implemented("leaky_relu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.leaky_relu_backward(x, dY, negative_slope, dX);
}

// ─── Codec quantization (brosoundml) ───────────────────────────────────────

void vq_encode_forward(const Tensor& x, const Tensor& codebook,
                       Tensor& indices, Tensor& quantized) {
    const auto& v = detail::dispatch(x, codebook, indices, quantized);
    if (!v.vq_encode_forward)
        detail::throw_not_implemented("vq_encode_forward", x.device);
    detail::adopt_output(indices, x.device);
    detail::adopt_output(quantized, x.device);
    v.vq_encode_forward(x, codebook, indices, quantized);
}

void vq_encode_backward(const Tensor& dQuantized, Tensor& dX) {
    const auto& v = detail::dispatch(dQuantized, dX);
    if (!v.vq_encode_backward)
        detail::throw_not_implemented("vq_encode_backward", dQuantized.device);
    detail::adopt_output(dX, dQuantized.device);
    v.vq_encode_backward(dQuantized, dX);
}

void fsq_quantize_forward(const Tensor& x, const Tensor& levels,
                          Tensor& quantized, Tensor& packed_indices) {
    const auto& v = detail::dispatch(x, levels, quantized, packed_indices);
    if (!v.fsq_quantize_forward)
        detail::throw_not_implemented("fsq_quantize_forward", x.device);
    detail::adopt_output(quantized, x.device);
    detail::adopt_output(packed_indices, x.device);
    v.fsq_quantize_forward(x, levels, quantized, packed_indices);
}

void fsq_quantize_backward(const Tensor& dQuantized, Tensor& dX) {
    const auto& v = detail::dispatch(dQuantized, dX);
    if (!v.fsq_quantize_backward)
        detail::throw_not_implemented("fsq_quantize_backward", dQuantized.device);
    detail::adopt_output(dX, dQuantized.device);
    v.fsq_quantize_backward(dQuantized, dX);
}

// ─── 1D resampling (brosoundml) ────────────────────────────────────────────

void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out,
                        int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("resample1d_forward", X, N, C, L_in);
    const auto& v = detail::dispatch(X, Y);
    if (!v.resample1d_forward)
        detail::throw_not_implemented("resample1d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.resample1d_forward(X, N, C, L_in, L_out, mode, Y);
}

void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out,
                         int mode, Tensor& dX) {
    if (N <= 0 || C <= 0 || L_in <= 0) {
        detail::throw_spatial_1d_error("resample1d_backward", "dY",
                                       N, C, L_in, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_1D("resample1d_backward", dY, N, C, L_out);
    const auto& v = detail::dispatch(dY, dX);
    if (!v.resample1d_backward)
        detail::throw_not_implemented("resample1d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.resample1d_backward(dY, N, C, L_in, L_out, mode, dX);
}

// ─── log / exp / round elementwise (brosoundml) ────────────────────────────

void log_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.log_forward) detail::throw_not_implemented("log_forward", x.device);
    detail::adopt_output(y, x.device);
    v.log_forward(x, y);
}

void log_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.log_backward) detail::throw_not_implemented("log_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.log_backward(x, dY, dX);
}

void exp_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.exp_forward) detail::throw_not_implemented("exp_forward", x.device);
    detail::adopt_output(y, x.device);
    v.exp_forward(x, y);
}

void exp_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.exp_backward) detail::throw_not_implemented("exp_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.exp_backward(x, dY, dX);
}

void round_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.round_forward) detail::throw_not_implemented("round_forward", x.device);
    detail::adopt_output(y, x.device);
    v.round_forward(x, y);
}

void round_backward(const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(dY, dX);
    if (!v.round_backward) detail::throw_not_implemented("round_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.round_backward(dY, dX);
}

// ─── Autoregressive logit sampling (brosoundml) ────────────────────────────

void sample_logits(const Tensor& logits, float temperature, int top_k,
                   float top_p, uint64_t key, uint64_t counter,
                   Tensor& indices) {
    const auto& v = detail::dispatch(logits, indices);
    if (!v.sample_logits) detail::throw_not_implemented("sample_logits", logits.device);
    detail::adopt_output(indices, logits.device);
    v.sample_logits(logits, temperature, top_k, top_p, key, counter, indices);
}

void sample_logits_into(const Tensor& logits, float temperature, int top_k,
                        float top_p, uint64_t key, Tensor& counter,
                        Tensor& scratch, Tensor& indices) {
    const auto& v = detail::dispatch(logits, counter, scratch, indices);
    if (!v.sample_logits_into)
        detail::throw_not_implemented("sample_logits_into", logits.device);
    detail::adopt_output(counter, logits.device);
    detail::adopt_output(scratch, logits.device);
    detail::adopt_output(indices, logits.device);
    v.sample_logits_into(logits, temperature, top_k, top_p, key, counter,
                         scratch, indices);
}

// ─── Masked-diffusion token selection (OmniVoice codebook grids) ────────────

void masked_diffusion_scores(const Tensor& logits, const Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, std::uint64_t seed,
                             Tensor& pred, Tensor& scores, Tensor& confidence) {
    const auto& v = detail::dispatch(logits, tokens, pred, scores, confidence);
    if (!v.masked_diffusion_scores)
        detail::throw_not_implemented("masked_diffusion_scores", logits.device);
    detail::adopt_output(pred, logits.device);
    detail::adopt_output(scores, logits.device);
    detail::adopt_output(confidence, logits.device);
    v.masked_diffusion_scores(logits, tokens, T, C, V, mask_id,
                              guidance_scale, layer_penalty,
                              position_temperature, class_temperature,
                              class_top_frac, seed, pred, scores, confidence);
}

void masked_diffusion_commit(const Tensor& pred, const Tensor& idx, int k, int step,
                             Tensor& tokens, Tensor& unmask_step) {
    const auto& v = detail::dispatch(pred, idx, tokens, unmask_step);
    if (!v.masked_diffusion_commit)
        detail::throw_not_implemented("masked_diffusion_commit", pred.device);
    v.masked_diffusion_commit(pred, idx, k, step, tokens, unmask_step);
}

// ─── L2 norm + Gated Delta Rule (linear-attention text path) ───────────────

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads,
                     float eps, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.l2_norm_forward) detail::throw_not_implemented("l2_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.l2_norm_forward(X, head_dim, num_heads, eps, Y);
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads,
                      float eps, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.l2_norm_backward) detail::throw_not_implemented("l2_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.l2_norm_backward(X, head_dim, num_heads, eps, dY, dX);
}

void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta,
                              const Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, a_raw, beta, log_A, state, O);
    if (!v.gated_delta_rule_chunked)
        detail::throw_not_implemented("gated_delta_rule_chunked", Q.device);
    detail::adopt_output(state, Q.device);
    detail::adopt_output(O, Q.device);
    v.gated_delta_rule_chunked(Q, K, V, a_raw, beta, log_A,
                               num_heads, d_k, d_v, state, O);
}

void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& a_raw, const Tensor& beta,
                           const Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           Tensor& state, Tensor& O) {
    const auto& v = detail::dispatch(Q, K, V, a_raw, beta, log_A, state, O);
    if (!v.gated_delta_rule_step)
        detail::throw_not_implemented("gated_delta_rule_step", Q.device);
    detail::adopt_output(state, Q.device);
    detail::adopt_output(O, Q.device);
    v.gated_delta_rule_step(Q, K, V, a_raw, beta, log_A,
                            num_heads, d_k, d_v, state, O);
}

// ─── Counter-based noise generation (Philox 4x32-10) ───────────────────────

void randn(uint64_t key, uint64_t counter, Tensor& Y) {
    const auto& v = detail::dispatch(Y);
    if (!v.randn) detail::throw_not_implemented("randn", Y.device);
    detail::adopt_output(Y, Y.device);
    v.randn(key, counter, Y);
}

void rand_uniform(uint64_t key, uint64_t counter, Tensor& Y) {
    const auto& v = detail::dispatch(Y);
    if (!v.rand_uniform) detail::throw_not_implemented("rand_uniform", Y.device);
    detail::adopt_output(Y, Y.device);
    v.rand_uniform(key, counter, Y);
}

void rand_bernoulli(float p, uint64_t key, uint64_t counter, Tensor& Y) {
    const auto& v = detail::dispatch(Y);
    if (!v.rand_bernoulli) detail::throw_not_implemented("rand_bernoulli", Y.device);
    detail::adopt_output(Y, Y.device);
    v.rand_bernoulli(p, key, counter, Y);
}

void randn_truncated(float lo, float hi, uint64_t key, uint64_t counter,
                     Tensor& Y) {
    const auto& v = detail::dispatch(Y);
    if (!v.randn_truncated) detail::throw_not_implemented("randn_truncated", Y.device);
    detail::adopt_output(Y, Y.device);
    v.randn_truncated(lo, hi, key, counter, Y);
}

// ─── StyleGAN3 synthesis-input primitives — sin/cos/rsqrt + pixel_norm ──────

void sin_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.sin_forward) detail::throw_not_implemented("sin_forward", x.device);
    detail::adopt_output(y, x.device);
    v.sin_forward(x, y);
}

void sin_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.sin_backward) detail::throw_not_implemented("sin_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.sin_backward(x, dY, dX);
}

void cos_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.cos_forward) detail::throw_not_implemented("cos_forward", x.device);
    detail::adopt_output(y, x.device);
    v.cos_forward(x, y);
}

void cos_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.cos_backward) detail::throw_not_implemented("cos_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.cos_backward(x, dY, dX);
}

void rsqrt_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.rsqrt_forward) detail::throw_not_implemented("rsqrt_forward", x.device);
    detail::adopt_output(y, x.device);
    v.rsqrt_forward(x, y);
}

void rsqrt_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(y, dY, dX);
    if (!v.rsqrt_backward) detail::throw_not_implemented("rsqrt_backward", y.device);
    detail::adopt_output(dX, y.device);
    v.rsqrt_backward(y, dY, dX);
}

void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y) {
    const auto& v = detail::dispatch(X, Y);
    if (!v.pixel_norm_forward) detail::throw_not_implemented("pixel_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.pixel_norm_forward(X, eps, Y);
}

void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX) {
    const auto& v = detail::dispatch(X, dY, dX);
    if (!v.pixel_norm_backward) detail::throw_not_implemented("pixel_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    v.pixel_norm_backward(X, dY, eps, dX);
}

} // namespace brotensor
