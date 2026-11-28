#pragma once

// Shared host-side validation, shape checks, and workspace sizing for
// materialised-probability attention backwards across CUDA, Vulkan, and Metal.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::gpu_common {

[[noreturn]] inline void attn_bwd_fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

inline void ensure_f32_2d(Tensor& t, int rows, int cols, Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) {
        t.resize(rows, cols, Dtype::FP32);
    }
}

struct AttentionBwdSpec {
    const char* op;
    const Tensor *dO, *X, *C;
    const Tensor *Qh, *Kh, *Vh, *Attnh, *Yc;
    const Tensor *Wq, *Wk, *Wv, *Wo;
    const float* kmask = nullptr;
    const float* qmask = nullptr;
    int H = 1;
    Tensor* dX;
    Tensor* dC = nullptr;
    Tensor *dWq, *dWk, *dWv, *dWo;
    Tensor *dbq = nullptr, *dbk = nullptr, *dbv = nullptr, *dbo = nullptr;
};

inline void validate_attention_bwd_shapes(const AttentionBwdSpec& s,
                                          int Lq, int Lk, int D, int Dx, int Dc, int hd) {
    const char* op = s.op;
    for (const Tensor* t : {s.dO, s.X, s.C, s.Qh, s.Kh, s.Vh, s.Attnh, s.Yc,
                            s.Wq, s.Wk, s.Wv, s.Wo,
                            static_cast<const Tensor*>(s.dWq), static_cast<const Tensor*>(s.dWk),
                            static_cast<const Tensor*>(s.dWv), static_cast<const Tensor*>(s.dWo)}) {
        if (t->dtype != Dtype::FP32) attn_bwd_fail(op, "every operand must be FP32");
    }
    auto shape = [&](const Tensor& t, int r, int c, const char* what) {
        if (t.rows != r || t.cols != c) {
            attn_bwd_fail(op, std::string(what) + " has the wrong shape");
        }
    };
    shape(*s.dO, Lq, D, "dO");
    shape(*s.Qh, s.H * Lq, hd, "Qh");
    shape(*s.Kh, s.H * Lk, hd, "Kh");
    shape(*s.Vh, s.H * Lk, hd, "Vh");
    shape(*s.Attnh, s.H * Lq, Lk, "Attnh");
    shape(*s.Yc, Lq, D, "Yconcat");
    shape(*s.Wq, D, Dx, "Wq");
    shape(*s.Wk, D, Dc, "Wk");
    shape(*s.Wv, D, Dc, "Wv");
    shape(*s.Wo, D, D, "Wo");
    shape(*s.dWq, D, Dx, "dWq");
    shape(*s.dWk, D, Dc, "dWk");
    shape(*s.dWv, D, Dc, "dWv");
    shape(*s.dWo, D, D, "dWo");
    for (Tensor* b : {s.dbq, s.dbk, s.dbv, s.dbo}) {
        if (b && (b->dtype != Dtype::FP32 || b->size() != D)) {
            attn_bwd_fail(op, "a bias gradient must have D FP32 elements");
        }
    }
}

/// Validates and resizes outputs for materialized attention backward.
/// Returns true if kernel launches should proceed, or false on empty/Lk==0 early exit.
inline bool prepare_attention_bwd(const AttentionBwdSpec& s,
                                  int& Lq, int& Lk, int& D, int& Dx, int& Dc, int& hd) {
    const Tensor &X = *s.X, &C = *s.C;
    Lq = X.rows;
    Lk = C.rows;
    Dx = X.cols;
    Dc = C.cols;
    D = s.Wq->rows;
    const int H = s.H;
    const Device dev = X.device;
    ensure_f32_2d(*s.dX, Lq, Dx, dev);
    if (s.dC) ensure_f32_2d(*s.dC, Lk, Dc, dev);
    if (Lq == 0 || D == 0 || H == 0) return false;
    if (H < 0 || D % H != 0) attn_bwd_fail(s.op, "num_heads must divide D");
    hd = D / H;
    if (Lk == 0) {
        s.dX->zero();
        if (s.dC && s.dC->size() > 0) s.dC->zero();
        return false;
    }
    validate_attention_bwd_shapes(s, Lq, Lk, D, Dx, Dc, hd);
    return true;
}

}  // namespace brotensor::gpu_common
