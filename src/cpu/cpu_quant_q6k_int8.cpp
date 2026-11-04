// brotensor/src/cpu/cpu_quant_q6k_int8.cpp — CPU Q6_K and W8A16 (int8w_fp16) quantized ops.

#include "cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>
#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

static void validate_w_q6k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q6_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q6_K");
    }
    if (W.cols % 256 != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W cols must be a multiple of 256");
    }
}

static void validate_w_int8(const Tensor& W, const Tensor& scales, const char* op) {
    if (W.dtype != Dtype::INT8) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": scales must be Dtype::FP32");
    }
    if (scales.size() != W.rows) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": scales size must match W.rows");
    }
}

static inline void local_dequant_q6k_block(const void* blk_ptr, float* out256) {
    const uint8_t* p = static_cast<const uint8_t*>(blk_ptr);
    const uint8_t* ql = p;
    const uint8_t* qh = p + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(p + 192);
    uint16_t d_raw;
    std::memcpy(&d_raw, p + 208, 2);
    const float d = fp16_bits_to_fp32(d_raw);

    for (int e = 0; e < 256; ++e) {
        const int group = e >> 7;
        const int local = e & 127;
        const int quad  = local >> 5;
        const int l     = local & 31;
        const int sb    = (group << 3) + (quad << 1) + (l >> 4);

        const uint8_t ql_b = ql[group * 64 + (quad & 1) * 32 + l];
        const uint8_t qh_b = qh[group * 32 + l];

        const int raw4  = (quad < 2) ? (ql_b & 0x0F) : (ql_b >> 4);
        const int high2 = (qh_b >> (quad * 2)) & 0x03;
        const int val6  = (raw4 | (high2 << 4)) - 32;

        out256[e] = d * static_cast<float>(scales[sb]) * static_cast<float>(val6);
    }
}

static inline float dot_int8_fp32(const int8_t* w_row, const float* x, int K) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        acc0 += static_cast<float>(w_row[k + 0]) * x[k + 0];
        acc1 += static_cast<float>(w_row[k + 1]) * x[k + 1];
        acc2 += static_cast<float>(w_row[k + 2]) * x[k + 2];
        acc3 += static_cast<float>(w_row[k + 3]) * x[k + 3];
    }
    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (; k < K; ++k) {
        sum += static_cast<float>(w_row[k]) * x[k];
    }
    return sum;
}

} // namespace

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    validate_w_q6k(W_q6k, "dequant_q6k_to_fp16");
    const int out = W_q6k.rows;
    const int K   = W_q6k.cols;
    if (W_fp16.rows != out || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(out, K, Dtype::FP16);
    }
    if (out == 0 || K == 0) return;
    const int bpr = K / 256;
    const char* src = static_cast<const char*>(W_q6k.data);
    uint16_t* dst = static_cast<uint16_t*>(W_fp16.data);

    auto dequant_row = [&](int r) {
        float buf[256];
        const char* rsrc = src + static_cast<size_t>(r) * bpr * 210;
        uint16_t* rdst = dst + static_cast<size_t>(r) * K;
        for (int b = 0; b < bpr; ++b) {
            local_dequant_q6k_block(rsrc + b * 210, buf);
            for (int i = 0; i < 256; ++i) {
                rdst[b * 256 + i] = fp32_to_fp16_bits(buf[i]);
            }
        }
    };

    if (out >= 8) {
        detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
            dequant_row(static_cast<int>(r));
        });
    } else {
        for (int r = 0; r < out; ++r) dequant_row(r);
    }
}

void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q6k(W_q6k, "linear_forward_q6k_fp16");
    const int out = W_q6k.rows;
    const int K   = W_q6k.cols;
    if (x.size() != K) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x size mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: bias size mismatch");
    }
    if (y.rows != out || y.cols != 1 || (y.dtype != Dtype::FP16 && y.dtype != Dtype::FP32)) {
        y.resize(out, 1, x.dtype == Dtype::FP32 ? Dtype::FP32 : Dtype::FP16);
    }
    if (out == 0) return;

    std::vector<float> x_f32(K);
    if (x.dtype == Dtype::FP16) {
        const uint16_t* xp = static_cast<const uint16_t*>(x.data);
        for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
    } else if (x.dtype == Dtype::FP32) {
        std::memcpy(x_f32.data(), x.data, K * sizeof(float));
    } else {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x must be FP16 or FP32");
    }

    std::vector<float> y_f32(out);
    jit::gemv_q6_k(W_q6k.data, x_f32.data(), y_f32.data(), out, K);

    if (bias && bias->size() > 0) {
        if (bias->dtype == Dtype::FP16) {
            const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
        } else if (bias->dtype == Dtype::FP32) {
            const float* bp = static_cast<const float*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
        }
    }

    if (y.dtype == Dtype::FP16) {
        uint16_t* yp = static_cast<uint16_t*>(y.data);
        for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
    } else {
        std::memcpy(y.data, y_f32.data(), out * sizeof(float));
    }
}

void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q6k(W_q6k, "linear_forward_batched_q6k_fp16");
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q6k.rows;
    if (W_q6k.cols != K) {
        throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: shape mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: bias size mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out || (Y_BD.dtype != Dtype::FP16 && Y_BD.dtype != Dtype::FP32)) {
        Y_BD.resize(B, out, X_BD.dtype == Dtype::FP32 ? Dtype::FP32 : Dtype::FP16);
    }
    if (B == 0 || out == 0) return;

    for (int b = 0; b < B; ++b) {
        std::vector<float> x_f32(K);
        if (X_BD.dtype == Dtype::FP16) {
            const uint16_t* xp = static_cast<const uint16_t*>(X_BD.data) + static_cast<size_t>(b) * K;
            for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
        } else if (X_BD.dtype == Dtype::FP32) {
            const float* xp = static_cast<const float*>(X_BD.data) + static_cast<size_t>(b) * K;
            std::memcpy(x_f32.data(), xp, K * sizeof(float));
        } else {
            throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: X must be FP16 or FP32");
        }

        std::vector<float> y_f32(out);
        jit::gemv_q6_k(W_q6k.data, x_f32.data(), y_f32.data(), out, K);

        if (bias && bias->size() > 0) {
            if (bias->dtype == Dtype::FP16) {
                const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
            } else if (bias->dtype == Dtype::FP32) {
                const float* bp = static_cast<const float*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
            }
        }

        if (Y_BD.dtype == Dtype::FP16) {
            uint16_t* yp = static_cast<uint16_t*>(Y_BD.data) + static_cast<size_t>(b) * out;
            for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
        } else {
            float* yp = static_cast<float*>(Y_BD.data) + static_cast<size_t>(b) * out;
            std::memcpy(yp, y_f32.data(), out * sizeof(float));
        }
    }
}

void matmul_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                       const Tensor& X, Tensor& Y) {
    validate_w_int8(W_int8, scales, "matmul_int8w_fp16");
    const int out = W_int8.rows;
    const int K   = W_int8.cols;
    if (X.size() != K) {
        throw std::runtime_error("brotensor: matmul_int8w_fp16: X size mismatch");
    }
    if (Y.rows != out || Y.cols != 1 || (Y.dtype != Dtype::FP16 && Y.dtype != Dtype::FP32)) {
        Y.resize(out, 1, X.dtype == Dtype::FP32 ? Dtype::FP32 : Dtype::FP16);
    }
    if (out == 0) return;

    std::vector<float> x_f32(K);
    if (X.dtype == Dtype::FP16) {
        const uint16_t* xp = static_cast<const uint16_t*>(X.data);
        for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
    } else if (X.dtype == Dtype::FP32) {
        std::memcpy(x_f32.data(), X.data, K * sizeof(float));
    } else {
        throw std::runtime_error("brotensor: matmul_int8w_fp16: X must be FP16 or FP32");
    }

    const int8_t* w_ptr = static_cast<const int8_t*>(W_int8.data);
    const float* s_ptr = static_cast<const float*>(scales.data);
    std::vector<float> y_f32(out);

    auto compute_row = [&](int r) {
        const int8_t* w_row = w_ptr + static_cast<size_t>(r) * K;
        y_f32[r] = s_ptr[r] * dot_int8_fp32(w_row, x_f32.data(), K);
    };

    if (out >= 8) {
        detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
            compute_row(static_cast<int>(r));
        });
    } else {
        for (int r = 0; r < out; ++r) compute_row(r);
    }

    if (Y.dtype == Dtype::FP16) {
        uint16_t* yp = static_cast<uint16_t*>(Y.data);
        for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
    } else {
        std::memcpy(Y.data, y_f32.data(), out * sizeof(float));
    }
}

void linear_forward_batched_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                                       const Tensor* bias, const Tensor& X_BD,
                                       Tensor& Y_BD) {
    validate_w_int8(W_int8, scales, "linear_forward_batched_int8w_fp16");
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_int8.rows;
    if (W_int8.cols != K) {
        throw std::runtime_error("brotensor: linear_forward_batched_int8w_fp16: shape mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_batched_int8w_fp16: bias size mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out || (Y_BD.dtype != Dtype::FP16 && Y_BD.dtype != Dtype::FP32)) {
        Y_BD.resize(B, out, X_BD.dtype == Dtype::FP32 ? Dtype::FP32 : Dtype::FP16);
    }
    if (B == 0 || out == 0) return;

    const int8_t* w_ptr = static_cast<const int8_t*>(W_int8.data);
    const float* s_ptr = static_cast<const float*>(scales.data);

    std::vector<float> bias_f32;
    if (bias && bias->size() > 0) {
        bias_f32.resize(out);
        if (bias->dtype == Dtype::FP16) {
            const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
            for (int i = 0; i < out; ++i) bias_f32[i] = fp16_bits_to_fp32(bp[i]);
        } else if (bias->dtype == Dtype::FP32) {
            std::memcpy(bias_f32.data(), bias->data, out * sizeof(float));
        } else {
            throw std::runtime_error("brotensor: linear_forward_batched_int8w_fp16: bias must be FP16 or FP32");
        }
    }

    for (int b = 0; b < B; ++b) {
        std::vector<float> x_f32(K);
        if (X_BD.dtype == Dtype::FP16) {
            const uint16_t* xp = static_cast<const uint16_t*>(X_BD.data) + static_cast<size_t>(b) * K;
            for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
        } else if (X_BD.dtype == Dtype::FP32) {
            const float* xp = static_cast<const float*>(X_BD.data) + static_cast<size_t>(b) * K;
            std::memcpy(x_f32.data(), xp, K * sizeof(float));
        } else {
            throw std::runtime_error("brotensor: linear_forward_batched_int8w_fp16: X must be FP16 or FP32");
        }

        std::vector<float> y_f32(out);
        auto compute_row = [&](int r) {
            const int8_t* w_row = w_ptr + static_cast<size_t>(r) * K;
            float val = s_ptr[r] * dot_int8_fp32(w_row, x_f32.data(), K);
            if (!bias_f32.empty()) val += bias_f32[r];
            y_f32[r] = val;
        };

        if (out >= 8) {
            detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
                compute_row(static_cast<int>(r));
            });
        } else {
            for (int r = 0; r < out; ++r) compute_row(r);
        }

        if (Y_BD.dtype == Dtype::FP16) {
            uint16_t* yp = static_cast<uint16_t*>(Y_BD.data) + static_cast<size_t>(b) * out;
            for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
        } else {
            float* yp = static_cast<float*>(Y_BD.data) + static_cast<size_t>(b) * out;
            std::memcpy(yp, y_f32.data(), out * sizeof(float));
        }
    }
}

} // namespace brotensor::detail::cpu
