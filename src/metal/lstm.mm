// Metal LSTM training: lstm_forward_train and lstm_backward (BPTT), FP32.
//
// PyTorch nn.LSTM layout: W_ih (4H, I), W_hh (4H, H), gates [i | f | g | o];
// gates holds post-activation values, C cell states, Y hidden states (T*B rows,
// row t*B + b); parameter gradients accumulate, dX / dh0 / dc0 are overwritten.
//
// Matrix products ride the simdgroup GEMM (fp16_matmul.h / gemm_fp32.mm);
// pointwise forward/backward steps and bias column sums ride single-source
// brass MIR kernels (src/kernels/mir_lstm.cpp, mir_norm.cpp -> mtl_mir_cache.mm).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#import "fp16_matmul.h"
#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

namespace {

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

void ensure(Tensor& t, int rows, int cols, Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) {
        t.resize(rows, cols, Dtype::FP32);
    }
}

NSUInteger row_off(const Tensor& t, long long row) {
    return buffer_offset_for(t) +
           static_cast<NSUInteger>(static_cast<std::size_t>(row) * t.cols * sizeof(float));
}

// C(m, n) (+)= A(m, k) op B, FP32; tb=false: B stored (n, k), tb=true: B stored (k, n).
void mm(id<MTLBuffer> a, NSUInteger oa, int lda, bool ta,
        id<MTLBuffer> b, NSUInteger ob, int ldb, bool tb,
        id<MTLBuffer> c, NSUInteger oc, int ldc,
        int m, int n, int k, bool acc,
        id<MTLBuffer> bias = nil, NSUInteger obias = 0) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    metal_impl::AbtMixed g;
    g.A = a; g.ofs_A = oa; g.lda = static_cast<std::uint64_t>(lda); g.transA = ta;
    g.B = b; g.ofs_B = ob; g.ldb = static_cast<std::uint64_t>(ldb); g.transB = tb;
    g.C = c; g.ofs_C = oc; g.ldc = static_cast<std::uint64_t>(ldc);
    g.bias = bias; g.ofs_bias = obias;
    g.M = m; g.N = n; g.K = k;
    g.in = g.out = metal_impl::kAbtF32;
    g.epilogue = acc ? metal_impl::kAbtAccumulate : metal_impl::kAbtStore;
    metal_impl::launch_matmul_abt_mixed(g);
}

void col_sum_accum(const Tensor& X, Tensor& out) {
    const std::uint32_t rows = static_cast<std::uint32_t>(X.rows);
    const std::uint32_t cols = static_cast<std::uint32_t>(X.cols);
    if (rows == 0 || cols == 0) return;
    const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
    const MirKernel mk = mir_kernel(
        {MirKind::Norm, std::uint32_t(NORM_COL_SUM), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t groups = std::min<std::uint32_t>((cols + 31u) / 32u, 65535u);
    mir_dispatch_norm(mk, groups, "lstm_backward",
                      X.data, nullptr, out.data, nullptr, nullptr, nullptr, nullptr,
                      rows, cols, 0.0f, 0.0f);
}

}  // namespace

void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                        const Tensor* b_ih, const Tensor* b_hh,
                        const Tensor* h0, const Tensor* c0, int T, int B,
                        Tensor& Y, Tensor& gates, Tensor& C,
                        Tensor* hT, Tensor* cT) {
    constexpr const char* op = "lstm_forward_train";
    require_fp32(op, X, "X");
    require_fp32(op, W_ih, "W_ih");
    require_fp32(op, W_hh, "W_hh");
    const int H = W_hh.cols, I = W_ih.cols, G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");

    const Tensor* bih = opt(op, b_ih, "b_ih", G, 1);
    const Tensor* bhh = opt(op, b_hh, "b_hh", G, 1);
    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);

    const Device dev = X.device;
    ensure(Y, T * B, H, dev);
    ensure(gates, T * B, G, dev);
    ensure(C, T * B, H, dev);
    if (hT) ensure(*hT, B, H, dev);
    if (cT) ensure(*cT, B, H, dev);

    // gates = X @ W_ih^T + b_ih for all T*B rows at once.
    mm(buffer_for(X), buffer_offset_for(X), I, false,
       buffer_for(W_ih), buffer_offset_for(W_ih), I, false,
       buffer_for(gates), buffer_offset_for(gates), G,
       T * B, G, I, false,
       bih ? buffer_for(*bih) : nil, bih ? buffer_offset_for(*bih) : 0);

    const std::uint8_t flags = static_cast<std::uint8_t>((bhh ? 1u : 0u) | (c0p ? 2u : 0u));
    const MirKernel mk = mir_kernel(
        {MirKind::Lstm, std::uint32_t(LS_FWD), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = mir_groups_1d(static_cast<std::uint32_t>(B * H), 1);
    const void* bhh_p = bhh ? bhh->data : nullptr;
    const void* c0_p = c0p ? c0p->data : nullptr;

    id<MTLBuffer> bWhh = buffer_for(W_hh);
    const NSUInteger oWhh = buffer_offset_for(W_hh);
    id<MTLBuffer> bGates = buffer_for(gates);
    id<MTLBuffer> bY = buffer_for(Y);

    for (int t = 0; t < T; ++t) {
        if (t > 0) {
            mm(bY, row_off(Y, static_cast<long long>(t - 1) * B), H, false,
               bWhh, oWhh, H, false,
               bGates, row_off(gates, static_cast<long long>(t) * B), G,
               B, G, H, true);
        } else if (h0p) {
            mm(buffer_for(*h0p), buffer_offset_for(*h0p), H, false,
               bWhh, oWhh, H, false,
               bGates, row_off(gates, 0), G,
               B, G, H, true);
        }
        mir_dispatch_lstm(mk, gx, op, gates.data, C.data, Y.data, bhh_p, c0_p,
                          nullptr, nullptr, nullptr, nullptr,
                          static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(H),
                          static_cast<std::uint32_t>(t * B));
    }

    if (hT) ::brotensor::copy_d2d(Y, static_cast<long long>(T - 1) * B * H, *hT, 0, B * H);
    if (cT) ::brotensor::copy_d2d(C, static_cast<long long>(T - 1) * B * H, *cT, 0, B * H);
}

void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh,
                   const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C,
                   const Tensor& dY, int T, int B,
                   Tensor& dX, Tensor& dW_ih, Tensor& dW_hh,
                   Tensor* db_ih, Tensor* db_hh, Tensor* dh0, Tensor* dc0) {
    constexpr const char* op = "lstm_backward";
    for (const Tensor* t : {&X, &W_ih, &W_hh, &Y, &gates, &C, &dY,
                            static_cast<const Tensor*>(&dW_ih),
                            static_cast<const Tensor*>(&dW_hh)}) {
        require_fp32(op, *t, "every operand");
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
    const Device dev = X.device;
    ensure(dX, T * B, I, dev);
    if (dh0) ensure(*dh0, B, H, dev);
    if (dc0) ensure(*dc0, B, H, dev);

    Tensor dZ = Tensor::empty_on(dev, T * B, G, Dtype::FP32);
    Tensor dh = Tensor::zeros_on(dev, B, H, Dtype::FP32);
    Tensor dc = Tensor::zeros_on(dev, B, H, Dtype::FP32);

    const std::uint8_t flags = static_cast<std::uint8_t>(c0p ? 2u : 0u);
    const MirKernel mk = mir_kernel(
        {MirKind::Lstm, std::uint32_t(LS_BWD), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = mir_groups_1d(static_cast<std::uint32_t>(B * H), 1);
    const void* c0_p = c0p ? c0p->data : nullptr;

    id<MTLBuffer> bdZ = buffer_for(dZ);
    id<MTLBuffer> bWhh = buffer_for(W_hh);
    const NSUInteger oWhh = buffer_offset_for(W_hh);
    id<MTLBuffer> bdh = buffer_for(dh);
    const NSUInteger odh = buffer_offset_for(dh);

    for (int t = T - 1; t >= 0; --t) {
        mir_dispatch_lstm(mk, gx, op, gates.data, C.data, nullptr, nullptr, c0_p,
                          dY.data, dZ.data, dh.data, dc.data,
                          static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(H),
                          static_cast<std::uint32_t>(t * B));
        // dh_{t-1} = dZ_t @ W_hh (W_hh stored (K = 4H, N = H)).
        mm(bdZ, row_off(dZ, static_cast<long long>(t) * B), G, false,
           bWhh, oWhh, H, true,
           bdh, odh, H, B, H, G, false);
    }

    if (dh0) ::brotensor::copy_d2d(dh, 0, *dh0, 0, B * H);
    if (dc0) ::brotensor::copy_d2d(dc, 0, *dc0, 0, B * H);

    // dX = dZ @ W_ih; dW_ih += dZ^T @ X; dW_hh += dZ_{t>0}^T @ Y_{t-1} (+ dZ_0^T @ h0).
    mm(bdZ, buffer_offset_for(dZ), G, false,
       buffer_for(W_ih), buffer_offset_for(W_ih), I, true,
       buffer_for(dX), buffer_offset_for(dX), I, T * B, I, G, false);
    mm(bdZ, buffer_offset_for(dZ), G, true,
       buffer_for(X), buffer_offset_for(X), I, true,
       buffer_for(dW_ih), buffer_offset_for(dW_ih), I, G, I, T * B, true);
    mm(bdZ, row_off(dZ, B), G, true,
       buffer_for(Y), buffer_offset_for(Y), H, true,
       buffer_for(dW_hh), buffer_offset_for(dW_hh), H, G, H, (T - 1) * B, true);
    if (h0p) {
        mm(bdZ, buffer_offset_for(dZ), G, true,
           buffer_for(*h0p), buffer_offset_for(*h0p), H, true,
           buffer_for(dW_hh), buffer_offset_for(dW_hh), H, G, H, B, true);
    }
    if (db_ih) col_sum_accum(dZ, *db_ih);
    if (db_hh) col_sum_accum(dZ, *db_hh);
}

}  // namespace brotensor::detail::metal
