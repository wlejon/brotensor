// Vulkan LSTM training: lstm_forward_train and lstm_backward (BPTT), FP32
// as on CUDA (src/cuda/lstm.cu) and the CPU reference
// (src/cpu/lstm.cpp): PyTorch layout W_ih (4H, I), W_hh (4H, H), gates
// [i | f | g | o]; gates holds the post-activation values, C the cell
// states, Y the hidden states (T B rows, row t B + b); parameter gradients
// accumulate, dX / dh0 / dc0 are overwritten.
//
// Forward: one GEMM for X W_ih^T + b_ih over all T B rows, then per step an
// accumulating GEMM of h_{t-1} W_hh^T into that step's rows and the pointwise
// cell (shaders/lstm.comp). Backward: per step (in reverse) the pointwise
// gradient into dZ and dh_{t-1} = dZ_t W_hh, then one GEMM each for
// dX = dZ W_ih, dW_ih += dZ^T X, dW_hh += dZ^T [h0; Y] and the bias column
// sums.

#include "detail/gemm.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void column_sum_accumulate(const Tensor& X, Tensor& out);   // ops_norm.cpp

namespace {

struct LsPush {
    std::uint64_t gates, c, y, bhh, c0, dy, dz, dh, dc;
    std::uint32_t b, h, row0;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need_f32(const char* op, const Tensor& t, const char* what) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(what) + " must be FP32");
}

// An optional tensor: present iff non-null and non-empty, FP32 (rows, cols).
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

std::uint64_t row_addr(const Tensor& t, long long row) {
    return addr(t.data) + static_cast<std::uint64_t>(row) * t.cols * 4u;
}

// C(m, n) (+)= A(m, k) op B, FP32; nt: B stored (n, k), else (k, n).
void mm(DeviceCtx& d, const char* op, std::uint64_t a, int lda, bool ta, std::uint64_t b, int ldb, bool nb,
        std::uint64_t c, int ldc, int m, int n, int k, bool acc, std::uint64_t bias = 0) {
    if (m == 0 || n == 0 || k == 0) return;
    GemmArgs g;
    g.op = op;
    g.a = a; g.lda = lda; g.ta = ta;
    g.b = b; g.ldb = ldb; g.nb = nb;
    g.c = c; g.ldc = ldc;
    g.bias = bias;
    g.m = m; g.n = n; g.k = k;
    g.epi = acc ? EPI_ACCUM : EPI_STORE;
    gemm(d, g);
}

}  // namespace

void lstm_forward_train(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh, const Tensor* b_ih,
                        const Tensor* b_hh, const Tensor* h0, const Tensor* c0, int T, int B, Tensor& Y,
                        Tensor& gates, Tensor& C, Tensor* hT, Tensor* cT) {
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
    DeviceCtx& d = device_of(X);

    // gates = X W_ih^T + b_ih for every step at once (pre-activations).
    mm(d, op, addr(X.data), I, false, addr(W_ih.data), I, false, addr(gates.data), G, T * B, G, I, false,
       bih ? addr(bih->data) : 0);
    const Kernel& k = d.pipelines().get(ShaderId::lstm, {std::uint32_t(LS_FWD)});
    LsPush pc{};
    pc.gates = addr(gates.data); pc.c = addr(C.data); pc.y = addr(Y.data);
    pc.bhh = bhh ? addr(bhh->data) : 0;
    pc.c0 = c0p ? addr(c0p->data) : 0;
    pc.b = static_cast<std::uint32_t>(B);
    pc.h = static_cast<std::uint32_t>(H);
    for (int t = 0; t < T; ++t) {
        const std::uint64_t hprev = t > 0 ? row_addr(Y, static_cast<long long>(t - 1) * B) : (h0p ? addr(h0p->data) : 0);
        if (hprev) mm(d, op, hprev, H, false, addr(W_hh.data), H, false, row_addr(gates, static_cast<long long>(t) * B), G,
                      B, G, H, true);
        pc.row0 = static_cast<std::uint32_t>(t * B);
        launch(d, k, pc, groups_1d(static_cast<std::uint64_t>(B) * H, k));
    }
    if (hT) ::brotensor::copy_d2d(Y, static_cast<long long>(T - 1) * B * H, *hT, 0, B * H);
    if (cT) ::brotensor::copy_d2d(C, static_cast<long long>(T - 1) * B * H, *cT, 0, B * H);
}

void lstm_backward(const Tensor& X, const Tensor& W_ih, const Tensor& W_hh, const Tensor* h0, const Tensor* c0,
                   const Tensor& Y, const Tensor& gates, const Tensor& C, const Tensor& dY, int T, int B, Tensor& dX,
                   Tensor& dW_ih, Tensor& dW_hh, Tensor* db_ih, Tensor* db_hh, Tensor* dh0, Tensor* dc0) {
    constexpr const char* op = "lstm_backward";
    for (const Tensor* t : {&X, &W_ih, &W_hh, &Y, &gates, &C, &dY, static_cast<const Tensor*>(&dW_ih),
                            static_cast<const Tensor*>(&dW_hh)}) {
        need_f32(op, *t, "every operand");
    }
    const int H = W_hh.cols, I = W_ih.cols, G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");
    if (dY.rows != T * B || dY.cols != H) fail(op, "dY must be (T*B, H)");
    if (Y.rows != T * B || Y.cols != H || C.rows != T * B || C.cols != H || gates.rows != T * B || gates.cols != G) {
        fail(op, "Y / C / gates must be the forward's caches");
    }
    if (dW_ih.rows != G || dW_ih.cols != I) fail(op, "dW_ih must be (4H, I), zeroed");
    if (dW_hh.rows != G || dW_hh.cols != H) fail(op, "dW_hh must be (4H, H), zeroed");
    for (Tensor* db : {db_ih, db_hh}) {
        if (db && (db->dtype != Dtype::FP32 || db->rows != G || db->cols != 1)) fail(op, "db_ih/db_hh must be (4H, 1)");
    }
    const Tensor* h0p = opt(op, h0, "h0", B, H);
    const Tensor* c0p = opt(op, c0, "c0", B, H);
    const ::brotensor::Device dev = X.device;
    ensure(dX, T * B, I, dev);
    if (dh0) ensure(*dh0, B, H, dev);
    if (dc0) ensure(*dc0, B, H, dev);
    DeviceCtx& d = device_of(X);

    Tensor dZ = Tensor::empty_on(dev, T * B, G, Dtype::FP32);
    Tensor dh = Tensor::zeros_on(dev, B, H, Dtype::FP32);   // dL/dh_t arriving from step t + 1
    Tensor dc = Tensor::zeros_on(dev, B, H, Dtype::FP32);
    const Kernel& k = d.pipelines().get(ShaderId::lstm, {std::uint32_t(LS_BWD)});
    LsPush pc{};
    pc.gates = addr(gates.data); pc.c = addr(C.data);
    pc.c0 = c0p ? addr(c0p->data) : 0;
    pc.dy = addr(dY.data); pc.dz = addr(dZ.data); pc.dh = addr(dh.data); pc.dc = addr(dc.data);
    pc.b = static_cast<std::uint32_t>(B);
    pc.h = static_cast<std::uint32_t>(H);
    for (int t = T - 1; t >= 0; --t) {
        pc.row0 = static_cast<std::uint32_t>(t * B);
        launch(d, k, pc, groups_1d(static_cast<std::uint64_t>(B) * H, k));
        // dh_{t-1} = dZ_t W_hh: W_hh stored (K = 4H, N = H).
        mm(d, op, row_addr(dZ, static_cast<long long>(t) * B), G, false, addr(W_hh.data), H, true, addr(dh.data), H, B,
           H, G, false);
    }
    if (dh0) ::brotensor::copy_d2d(dh, 0, *dh0, 0, B * H);
    if (dc0) ::brotensor::copy_d2d(dc, 0, *dc0, 0, B * H);

    // dX = dZ W_ih; dW_ih += dZ^T X; dW_hh += dZ_{t>0}^T Y_{t-1} (+ dZ_0^T h0).
    mm(d, op, addr(dZ.data), G, false, addr(W_ih.data), I, true, addr(dX.data), I, T * B, I, G, false);
    mm(d, op, addr(dZ.data), G, true, addr(X.data), I, true, addr(dW_ih.data), I, G, I, T * B, true);
    mm(d, op, row_addr(dZ, B), G, true, addr(Y.data), H, true, addr(dW_hh.data), H, G, H, (T - 1) * B, true);
    if (h0p) mm(d, op, addr(dZ.data), G, true, addr(h0p->data), H, true, addr(dW_hh.data), H, G, H, B, true);
    if (db_ih) column_sum_accumulate(dZ, *db_ih);
    if (db_hh) column_sum_accumulate(dZ, *db_hh);
}

void fill_vulkan_vtable_lstm(::brotensor::detail::OpsVTable& v) {
    v.lstm_forward_train = &vulkan::lstm_forward_train;
    v.lstm_backward = &vulkan::lstm_backward;
}

}  // namespace brotensor::detail::vulkan
