// CPU twin of src/cuda/linear_wq.cu: quantize_rows and
// linear_forward_batched_wq_ex in FP32 (the parity oracle).

#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

void linear_forward_batched_ex(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                               const ::brotensor::Tensor& X, int act, int epilogue,
                               ::brotensor::Tensor* workspace, ::brotensor::Tensor& Y);

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

float decode(Dtype fmt, uint8_t b) {
    return fmt == Dtype::INT8 ? static_cast<float>(static_cast<int8_t>(b)) : fp8_e4m3_bits_to_fp32(b);
}

}  // namespace

void quantize_rows(const ::brotensor::Tensor& W, Dtype fmt, ::brotensor::Tensor& Wq, ::brotensor::Tensor& scales) {
    constexpr const char* op = "quantize_rows";
    if (fmt != Dtype::INT8 && fmt != Dtype::FP8_E4M3) fail(op, "fmt must be INT8 or FP8_E4M3");
    if (W.dtype != Dtype::FP32) fail(op, "W must be FP32 on the CPU");
    const int N = W.rows, K = W.cols;
    if (Wq.rows != N || Wq.cols != K || Wq.dtype != fmt) Wq.resize(N, K, fmt);
    if (scales.rows != N || scales.cols != 1 || scales.dtype != Dtype::FP32) scales.resize(N, 1, Dtype::FP32);
    const float* w = W.host_f32();
    auto* q = static_cast<uint8_t*>(Wq.data);
    float* s = scales.host_f32_mut();
    const float qmax = fmt == Dtype::INT8 ? 127.0f : 448.0f;
    for (int r = 0; r < N; ++r) {
        const float* wr = w + static_cast<std::size_t>(r) * K;
        float mx = 0.0f;
        for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(wr[k]));
        const float scale = mx / qmax;
        s[r] = scale;
        uint8_t* qr = q + static_cast<std::size_t>(r) * K;
        for (int k = 0; k < K; ++k) {
            const float v = scale > 0.0f ? wr[k] / scale : 0.0f;
            if (fmt == Dtype::INT8)
                qr[k] = static_cast<uint8_t>(static_cast<int8_t>(std::clamp(std::nearbyint(v), -127.0f, 127.0f)));
            else
                qr[k] = fp32_to_fp8_e4m3_bits(v);
        }
    }
}

void dequantize_rows(const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& scales, Dtype out,
                     ::brotensor::Tensor& W) {
    constexpr const char* op = "dequantize_rows";
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    if (out != Dtype::FP32) fail(op, "out must be FP32 on the CPU");
    const int N = Wq.rows, K = Wq.cols;
    if (W.rows != N || W.cols != K || W.dtype != Dtype::FP32) W.resize(N, K, Dtype::FP32);
    const auto* q = static_cast<const uint8_t*>(Wq.data);
    const float* s = scales.host_f32();
    float* w = W.host_f32_mut();
    for (int r = 0; r < N; ++r)
        for (int k = 0; k < K; ++k) {
            const std::size_t i = static_cast<std::size_t>(r) * K + k;
            w[i] = decode(Wq.dtype, q[i]) * s[r];
        }
}

void linear_forward_batched_wq_ex(const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& scales,
                                  const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
                                  int epilogue, ::brotensor::Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_wq_ex";
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    const int N = Wq.rows, K = Wq.cols;
    ::brotensor::Tensor W = ::brotensor::Tensor::mat(N, K);
    const auto* q = static_cast<const uint8_t*>(Wq.data);
    const float* s = scales.host_f32();
    float* w = W.host_f32_mut();
    for (int r = 0; r < N; ++r)
        for (int k = 0; k < K; ++k) {
            const std::size_t i = static_cast<std::size_t>(r) * K + k;
            w[i] = decode(Wq.dtype, q[i]) * s[r];
        }
    ::brotensor::detail::cpu::linear_forward_batched_ex(W, bias, X, act, epilogue, nullptr, Y);
}

}  // namespace brotensor::detail::cpu
