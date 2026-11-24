// CUDA LSTM training (lstm_forward_train, lstm_backward, FP32) backed by
// Brass MIR -> PTX (MirKind::Lstm + MirKind::Norm NORM_COL_SUM) and CUDA GEMMs.

#include "cu_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void linear_forward_batched(const Tensor& W, const Tensor& bias, const Tensor& X_BD, Tensor& Y_BD);
void linear_backward_batched(const Tensor& W, const Tensor& X_BD, const Tensor& dY_BD,
                             Tensor& dX_BD, Tensor& dW, Tensor& dB);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need_f32(const char* op, const Tensor& t, const char* what) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(what) + " must be FP32 (lstm is FP32-only)");
}

const Tensor* opt(const char* op, const Tensor* t, const char* what, int rows, int cols) {
    if (!t || t->size() == 0) return nullptr;
    need_f32(op, *t, what);
    if (t->rows != rows || t->cols != cols) fail(op, std::string(what) + " has wrong shape");
    return t;
}

void ensure(Tensor& t, int rows, int cols, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) t.resize(rows, cols, Dtype::FP32);
}

void* row_ptr(const Tensor& t, long long row) {
    return static_cast<char*>(t.data) + static_cast<std::size_t>(row) * t.cols * sizeof(float);
}

void col_sum_accum(const Tensor& X, Tensor& out) {
    const std::uint32_t rows = static_cast<std::uint32_t>(X.rows);
    const std::uint32_t cols = static_cast<std::uint32_t>(X.cols);
    if (rows == 0 || cols == 0) return;
    const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
    CUfunction fn = mir_function(
        {MirKind::Norm, std::uint32_t(NORM_COL_SUM), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t groups = std::min((cols + 31u) / 32u, 65535u);
    mir_dispatch_norm(fn, groups, X.data, nullptr, out.data, nullptr, nullptr, nullptr, nullptr,
                      rows, cols, 0.0f, 0.0f);
}

}  // namespace

void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                        const Tensor* b_ih, const Tensor* b_hh,
                        const Tensor* h0, const Tensor* c0, int T, int B,
                        Tensor& Y, Tensor& gates, Tensor& C,
                        Tensor* hT, Tensor* cT) {
    constexpr const char* op = "lstm_forward_train";
    need_f32(op, X, "X");
    need_f32(op, W_ih, "W_ih");
    need_f32(op, W_hh, "W_hh");
    const int H = W_hh.cols, I = W_ih.cols, G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");
    const Tensor* bih = opt(op, b_ih, "b_ih", G, 1);
    const Tensor* bhh = opt(op, b_hh, "b_hh", G, 1);
    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);
    const ::brotensor::Device dev = X.device;
    ensure(Y, T * B, H, dev);
    ensure(gates, T * B, G, dev);
    ensure(C, T * B, H, dev);
    if (hT) ensure(*hT, B, H, dev);
    if (cT) ensure(*cT, B, H, dev);

    Tensor zero_b = Tensor::zeros_on(dev, G, 1, Dtype::FP32);
    detail::cuda::linear_forward_batched(W_ih, bih ? *bih : zero_b, X, gates);

    const std::uint8_t flags = static_cast<std::uint8_t>((bhh ? 1u : 0u) | (c0p ? 2u : 0u));
    CUfunction fn = mir_function(
        {MirKind::Lstm, std::uint32_t(LS_FWD), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = mir_groups_1d(static_cast<std::uint32_t>(B * H), 1);
    const void* bhh_p = bhh ? bhh->data : nullptr;
    const void* c0_p = c0p ? c0p->data : nullptr;
    Tensor g_tmp = Tensor::empty_on(dev, B, G, Dtype::FP32);
    for (int t = 0; t < T; ++t) {
        const void* hp = t > 0 ? row_ptr(Y, static_cast<long long>(t - 1) * B) : (h0p ? h0p->data : nullptr);
        if (hp != nullptr) {
            Tensor h_view = Tensor::view(dev, const_cast<void*>(hp), B, H, Dtype::FP32);
            Tensor g_step = Tensor::view(dev, row_ptr(gates, static_cast<long long>(t) * B), B, G, Dtype::FP32);
            detail::cuda::linear_forward_batched(W_hh, zero_b, h_view, g_tmp);
            ::brotensor::add_inplace(g_step, g_tmp);
        }
        mir_dispatch_lstm(
            fn, gx, gates.data, C.data, Y.data, bhh_p, c0_p,
            nullptr, nullptr, nullptr, nullptr,
            static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(t * B));
    }
    if (hT) ::brotensor::copy_d2d(Y, (T - 1) * B * H, *hT, 0, B * H);
    if (cT) ::brotensor::copy_d2d(C, (T - 1) * B * H, *cT, 0, B * H);
}

void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                   const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C, const Tensor& dY,
                   int T, int B, Tensor& dX, Tensor& dW_ih, Tensor& dW_hh,
                   Tensor* db_ih, Tensor* db_hh, Tensor* dh0, Tensor* dc0) {
    constexpr const char* op = "lstm_backward";
    for (const Tensor* t : {&X, &W_ih, &W_hh, &Y, &gates, &C, &dY,
                            static_cast<const Tensor*>(&dW_ih), static_cast<const Tensor*>(&dW_hh)}) {
        need_f32(op, *t, "every operand");
    }
    const int H = W_hh.cols, I = W_ih.cols, G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");
    if (dY.rows != T * B || dY.cols != H) fail(op, "dY must be (T*B, H)");
    if (Y.rows != T * B || Y.cols != H || C.rows != T * B || C.cols != H ||
        gates.rows != T * B || gates.cols != G) {
        fail(op, "Y / C / gates must be the forward's caches");
    }
    if (dW_ih.rows != G || dW_ih.cols != I) fail(op, "dW_ih must be (4H, I), zeroed");
    if (dW_hh.rows != G || dW_hh.cols != H) fail(op, "dW_hh must be (4H, H), zeroed");
    for (Tensor* db : {db_ih, db_hh}) {
        if (db && (db->dtype != Dtype::FP32 || db->rows != G || db->cols != 1)) {
            fail(op, "db_ih/db_hh must be (4H, 1)");
        }
    }
    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);
    const ::brotensor::Device dev = X.device;
    ensure(dX, T * B, I, dev);
    if (dh0) ensure(*dh0, B, H, dev);
    if (dc0) ensure(*dc0, B, H, dev);

    Tensor dZ = Tensor::empty_on(dev, T * B, G, Dtype::FP32);
    Tensor dh = Tensor::zeros_on(dev, B, H, Dtype::FP32);
    Tensor dc = Tensor::zeros_on(dev, B, H, Dtype::FP32);
    const std::uint8_t flags = static_cast<std::uint8_t>(c0p ? 2u : 0u);
    CUfunction fn = mir_function(
        {MirKind::Lstm, std::uint32_t(LS_BWD), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = mir_groups_1d(static_cast<std::uint32_t>(B * H), 1);
    const void* c0_p = c0p ? c0p->data : nullptr;
    for (int t = T - 1; t >= 0; --t) {
        mir_dispatch_lstm(
            fn, gx, gates.data, C.data, nullptr, nullptr, c0_p,
            dY.data, dZ.data, dh.data, dc.data,
            static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(t * B));
        Tensor dZt = Tensor::view(dev, row_ptr(dZ, static_cast<long long>(t) * B), B, G, Dtype::FP32);
        detail::cuda::matmul(dZt, W_hh, dh);
    }
    if (dh0) ::brotensor::copy_d2d(dh, 0, *dh0, 0, B * H);
    if (dc0) ::brotensor::copy_d2d(dc, 0, *dc0, 0, B * H);

    Tensor tmp_db = Tensor::empty_on(dev, G, 1, Dtype::FP32);
    detail::cuda::linear_backward_batched(W_ih, X, dZ, dX, dW_ih, db_ih ? *db_ih : tmp_db);
    Tensor tmp_dh = Tensor::empty_on(dev, (T > 1 ? (T - 1) * B : B), H, Dtype::FP32);
    if (T > 1) {
        Tensor Y_view = Tensor::view(dev, Y.data, (T - 1) * B, H, Dtype::FP32);
        Tensor dZ_tail = Tensor::view(dev, row_ptr(dZ, B), (T - 1) * B, G, Dtype::FP32);
        detail::cuda::linear_backward_batched(W_hh, Y_view, dZ_tail, tmp_dh, dW_hh, tmp_db);
    }
    if (h0p) {
        Tensor dZ0 = Tensor::view(dev, dZ.data, B, G, Dtype::FP32);
        detail::cuda::linear_backward_batched(W_hh, *h0p, dZ0, tmp_dh, dW_hh, tmp_db);
    }
    if (db_hh) col_sum_accum(dZ, *db_hh);
}

void fill_cuda_vtable_lstm(::brotensor::detail::OpsVTable& v) {
    v.lstm_forward_train = &lstm_forward_train;
    v.lstm_backward = &lstm_backward;
}

}  // namespace brotensor::detail::cuda
