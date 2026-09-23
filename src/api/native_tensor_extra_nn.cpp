// native_tensor_extra_nn.cpp — the dense / recurrent / attention half of the
// never-bound ops: linearForwardBatchedEx, linearForwardBatchedFp16Act,
// matmulAbt, the LSTM training pair, GQA flash attention, the packed-QKV
// encoder attention pair, the Transformer-XL relative-position bias and the
// two RoPE variants (per-head tables, in-place over a packed QKV buffer).
//
// Same mechanism as native_tensor_extra.cpp: every operand is checked against
// the dims it is handed before the op runs, and index operands whose values
// address memory (sequence bounds, RoPE positions) are read and range-checked
// on the host first.

#include "native_tensor_extra_decl.h"
#include "native_tensor_extra_util.h"
#include "api_internal.h"
#include "native_register.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace brotensor::api;
using namespace brotensor::api::extra;

namespace {

// A LinearActivation value (brotensor/ops/linear.h).
bool needAct(const char* label, int32_t act) {
    return needRange(label, "act", act, brotensor::kLinearActNone, brotensor::kLinearActQuickGelu);
}

// A JS number carrying a `long long` element stride: whole and non-negative.
bool strideArg(const char* label, const char* name, double v, int64_t& out) {
    if (!(v >= 0.0) || v != std::floor(v) || v > 9007199254740991.0) {
        setError(std::string(label) + ": " + name + " must be a non-negative whole number");
        return false;
    }
    out = static_cast<int64_t>(v);
    return true;
}

// Elements a batch of `batch` slices of `slice` elements, `stride` apart, spans.
int64_t spanOf(int64_t batch, int64_t stride, int64_t slice) {
    if (batch <= 0 || slice <= 0) return 0;
    const int64_t head = prod({batch - 1, stride});
    if (head == INT64_MAX || head > INT64_MAX - slice) return INT64_MAX;
    return head + slice;
}

} // namespace

extern "C" {

// ---- batched linear with a fused epilogue ------------------------------------

void bro_tensor_linearForwardBatchedEx(void* W, uint64_t bias_bits, void* X, int32_t act, int32_t epilogue,
                                       uint64_t workspace_bits, void* Y) {
    constexpr const char* lb = "linearForwardBatchedEx";
    if (!need(lb, {W, X, Y})) return;
    auto* w = toTensor(W);
    auto* x = toTensor(X);
    auto* y = toTensor(Y);
    auto* b = tensorFromValue(bias_bits);
    auto* ws = tensorFromValue(workspace_bits);
    if (!needFloat(lb, "W", w) || !needSameDtype(lb, "X", x, w) || !needAct(lb, act)) return;
    if (w->empty()) { setError(std::string(lb) + ": W is empty"); return; }
    if (x->cols != w->cols) {
        setError(std::string(lb) + ": X has " + std::to_string(x->cols) + " columns, W has " + std::to_string(w->cols));
        return;
    }
    if (!optElems(lb, "bias", b, w->rows, w)) return;
    const int32_t base = epilogue & ~static_cast<int32_t>(brotensor::kLinearEpiFastAccum);
    if (base == brotensor::kLinearEpiAccumulate) {
        if (!needSameDtype(lb, "Y", y, w)) return;
        if (y->rows != x->rows || y->cols != w->rows) {
            setError(std::string(lb) + ": the accumulate epilogue needs Y already (" + std::to_string(x->rows) +
                     ", " + std::to_string(w->rows) + ")");
            return;
        }
    } else if (base == brotensor::kLinearEpiGeglu) {
        if (act != 0 || (w->rows & 1)) {
            setError(std::string(lb) + ": the GeGLU epilogue needs act 0 and an even W row count");
            return;
        }
    } else if (base != brotensor::kLinearEpiStore) {
        setError(std::string(lb) + ": unknown epilogue " + std::to_string(epilogue));
        return;
    }
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_ex(*w, b, *x, act, epilogue, ws, *y);
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_linearForwardBatchedFp16Act(void* W, uint64_t bias_bits, void* X, int32_t act, void* Y) {
    constexpr const char* lb = "linearForwardBatchedFp16Act";
    if (!need(lb, {W, X, Y})) return;
    auto* w = toTensor(W);
    auto* x = toTensor(X);
    auto* b = tensorFromValue(bias_bits);
    if (w->dtype != brotensor::Dtype::FP16 && w->dtype != brotensor::Dtype::BF16) {
        setError(std::string(lb) + ": W is " + dtypeToString(w->dtype) + ", expected fp16 or bf16");
        return;
    }
    if (!needSameDtype(lb, "X", x, w) || !needAct(lb, act)) return;
    if (x->cols != w->cols) {
        setError(std::string(lb) + ": X has " + std::to_string(x->cols) + " columns, W has " + std::to_string(w->cols));
        return;
    }
    if (!optElems(lb, "bias", b, w->rows, w)) return;
    BROTENSOR_API_TRY
        brotensor::linear_forward_batched_fp16_act(*w, b, *x, act, *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

// ---- batched A @ B^T ---------------------------------------------------------

void bro_tensor_matmulAbt(void* A, void* B, void* C, int32_t batch, int32_t M, int32_t N, int32_t K,
                          double strideA, double strideB, double strideC, uint64_t bias_bits, int32_t act) {
    constexpr const char* lb = "matmulAbt";
    if (!need(lb, {A, B, C})) return;
    auto* a = toTensor(A);
    auto* b = toTensor(B);
    auto* c = toTensor(C);
    auto* bias = tensorFromValue(bias_bits);
    if (a->dtype != brotensor::Dtype::FP16 && a->dtype != brotensor::Dtype::BF16) {
        setError(std::string(lb) + ": A is " + dtypeToString(a->dtype) + ", expected fp16 or bf16");
        return;
    }
    if (!needSameDtype(lb, "B", b, a) || !needSameDtype(lb, "C", c, a) || !needAct(lb, act)) return;
    if (batch < 1 || M < 0 || N < 0 || K < 0) {
        setError(std::string(lb) + ": batch must be >= 1 and M, N, K >= 0");
        return;
    }
    int64_t sA = 0, sB = 0, sC = 0;
    if (!strideArg(lb, "strideA", strideA, sA) || !strideArg(lb, "strideB", strideB, sB) ||
        !strideArg(lb, "strideC", strideC, sC)) return;
    // C is not resized by the op: it must already hold every slice it writes.
    if (!needElems(lb, "A", a, spanOf(batch, sA, prod({M, K}))) ||
        !needElems(lb, "B", b, spanOf(batch, sB, prod({N, K}))) ||
        !needElems(lb, "C", c, spanOf(batch, sC, prod({M, N}))) ||
        !optElems(lb, "bias", bias, N, a)) return;
    BROTENSOR_API_TRY
        brotensor::matmul_abt(*a, *b, *c, batch, M, N, K, sA, sB, sC, bias, act);
    BROTENSOR_API_CATCH(lb)
}

} // extern "C"

// ---- LSTM (training forward + BPTT) ------------------------------------------

namespace {

// The shape the LSTM pair shares: FP32 X / W_ih / W_hh, H and I from the
// weights, X (T*B, I). Answers H and I.
bool lstmShapes(const char* lb, brotensor::Tensor* X, brotensor::Tensor* Wih, brotensor::Tensor* Whh,
                int32_t T, int32_t B, int64_t& H, int64_t& I) {
    constexpr auto F = brotensor::Dtype::FP32;
    if (!needDtype(lb, "X", X, F) || !needDtype(lb, "W_ih", Wih, F) || !needDtype(lb, "W_hh", Whh, F)) return false;
    if (T < 1 || B < 1) { setError(std::string(lb) + ": T and B must be >= 1"); return false; }
    if (Wih->rows < 4 || Wih->rows % 4 != 0) { setError(std::string(lb) + ": W_ih must have 4*H rows"); return false; }
    H = Wih->rows / 4;
    I = Wih->cols;
    if (Whh->rows != Wih->rows || Whh->cols != H) {
        setError(std::string(lb) + ": W_hh must be (4H, H) = (" + std::to_string(4 * H) + ", " + std::to_string(H) + ")");
        return false;
    }
    if (X->cols != I || !needElems(lb, "X", X, prod({T, B, I}))) {
        if (X->cols != I) setError(std::string(lb) + ": X must be (T*B, I) with I = W_ih.cols = " + std::to_string(I));
        return false;
    }
    return true;
}

} // namespace

extern "C" {

void bro_tensor_lstmForwardTrain(void* X, void* W_ih, void* W_hh, uint64_t b_ih_bits, uint64_t b_hh_bits,
                                 uint64_t h0_bits, uint64_t c0_bits, int32_t T, int32_t B,
                                 void* Y, void* gates, void* C, uint64_t hT_bits, uint64_t cT_bits) {
    constexpr const char* lb = "lstmForwardTrain";
    if (!need(lb, {X, W_ih, W_hh, Y, gates, C})) return;
    int64_t H = 0, I = 0;
    if (!lstmShapes(lb, toTensor(X), toTensor(W_ih), toTensor(W_hh), T, B, H, I)) return;
    auto* bih = tensorFromValue(b_ih_bits);
    auto* bhh = tensorFromValue(b_hh_bits);
    auto* h0 = tensorFromValue(h0_bits);
    auto* c0 = tensorFromValue(c0_bits);
    if (!optF32Elems(lb, "b_ih", bih, 4 * H) || !optF32Elems(lb, "b_hh", bhh, 4 * H) ||
        !optF32Elems(lb, "h0", h0, prod({B, H})) || !optF32Elems(lb, "c0", c0, prod({B, H}))) return;
    BROTENSOR_API_TRY
        brotensor::lstm_forward_train(*toTensor(X), *toTensor(W_ih), *toTensor(W_hh), bih, bhh, h0, c0, T, B,
                                      *toTensor(Y), *toTensor(gates), *toTensor(C),
                                      tensorFromValue(hT_bits), tensorFromValue(cT_bits));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_lstmBackward(void* X, void* W_ih, void* W_hh, uint64_t h0_bits, uint64_t c0_bits,
                             void* Y, void* gates, void* C, void* dY, int32_t T, int32_t B,
                             void* dX, void* dW_ih, void* dW_hh, uint64_t db_ih_bits, uint64_t db_hh_bits,
                             uint64_t dh0_bits, uint64_t dc0_bits) {
    constexpr const char* lb = "lstmBackward";
    if (!need(lb, {X, W_ih, W_hh, Y, gates, C, dY, dX, dW_ih, dW_hh})) return;
    int64_t H = 0, I = 0;
    if (!lstmShapes(lb, toTensor(X), toTensor(W_ih), toTensor(W_hh), T, B, H, I)) return;
    constexpr auto F = brotensor::Dtype::FP32;
    const int64_t TB = prod({T, B});
    auto* h0 = tensorFromValue(h0_bits);
    auto* c0 = tensorFromValue(c0_bits);
    auto* dbih = tensorFromValue(db_ih_bits);
    auto* dbhh = tensorFromValue(db_hh_bits);
    if (!optF32Elems(lb, "h0", h0, prod({B, H})) || !optF32Elems(lb, "c0", c0, prod({B, H})) ||
        !needDtype(lb, "Y", toTensor(Y), F) || !needElems(lb, "Y", toTensor(Y), prod({TB, H})) ||
        !needDtype(lb, "gates", toTensor(gates), F) || !needElems(lb, "gates", toTensor(gates), prod({TB, 4 * H})) ||
        !needDtype(lb, "C", toTensor(C), F) || !needElems(lb, "C", toTensor(C), prod({TB, H})) ||
        !needDtype(lb, "dY", toTensor(dY), F) || !needElems(lb, "dY", toTensor(dY), prod({TB, H})) ||
        // The parameter gradients accumulate, so they are read: check them too.
        !needDtype(lb, "dW_ih", toTensor(dW_ih), F) || !needElems(lb, "dW_ih", toTensor(dW_ih), prod({4 * H, I})) ||
        !needDtype(lb, "dW_hh", toTensor(dW_hh), F) || !needElems(lb, "dW_hh", toTensor(dW_hh), prod({4 * H, H})) ||
        !optF32Elems(lb, "db_ih", dbih, 4 * H) || !optF32Elems(lb, "db_hh", dbhh, 4 * H)) return;
    BROTENSOR_API_TRY
        brotensor::lstm_backward(*toTensor(X), *toTensor(W_ih), *toTensor(W_hh), h0, c0,
                                 *toTensor(Y), *toTensor(gates), *toTensor(C), *toTensor(dY), T, B,
                                 *toTensor(dX), *toTensor(dW_ih), *toTensor(dW_hh), dbih, dbhh,
                                 tensorFromValue(dh0_bits), tensorFromValue(dc0_bits));
    BROTENSOR_API_CATCH(lb)
}

// ---- GQA flash attention -------------------------------------------------------

void bro_tensor_flashAttentionGqaForward(void* Q, void* K, void* V, uint64_t mask_bits, int32_t numQHeads,
                                         int32_t numKvHeads, bool causal, void* O) {
    constexpr const char* lb = "flashAttentionGqaForward";
    if (!need(lb, {Q, K, V, O})) return;
    auto* q = toTensor(Q);
    auto* k = toTensor(K);
    auto* v = toTensor(V);
    if (!needFloat(lb, "Q", q) || !needSameDtype(lb, "K", k, q) || !needSameDtype(lb, "V", v, q)) return;
    if (numQHeads < 1 || numKvHeads < 1 || numQHeads % numKvHeads != 0) {
        setError(std::string(lb) + ": numKvHeads must divide numQHeads (both >= 1)");
        return;
    }
    if (q->cols % numQHeads != 0) { setError(std::string(lb) + ": Q.cols must be numQHeads * headDim"); return; }
    const int64_t hd = q->cols / numQHeads;
    if (k->cols != numKvHeads * hd || v->cols != k->cols || v->rows != k->rows) {
        setError(std::string(lb) + ": K and V must both be (Lk, numKvHeads*headDim) with headDim " + std::to_string(hd));
        return;
    }
    if (k->rows < q->rows || (causal && k->rows != q->rows)) {
        setError(std::string(lb) + ": needs Lk >= Lq (and Lk == Lq when causal)");
        return;
    }
    if (!optF32Elems(lb, "mask", tensorFromValue(mask_bits), k->rows)) return;
    BROTENSOR_API_TRY
        brotensor::flash_attention_gqa_forward(*q, *k, *v, maskPtr(mask_bits), numQHeads, numKvHeads, causal,
                                               *toTensor(O));
    BROTENSOR_API_CATCH(lb)
}

} // extern "C"

// ---- packed-QKV encoder attention ----------------------------------------------

namespace {

// QKV (L, 3*numHeads*headDim) float storage, seqBounds (L, 2) whose row r is
// a sequence [b0, b1) inside [0, L) containing r. Answers the INT32 operand
// the op reads (seqBounds itself, or a converted copy in `scratch`).
const brotensor::Tensor* packedBounds(const char* lb, brotensor::Tensor* qkv, int32_t numHeads,
                                      brotensor::Tensor* sb, brotensor::Tensor& scratch) {
    if (!needFloat(lb, "QKV", qkv)) return nullptr;
    if (numHeads < 1 || qkv->cols % (3 * static_cast<int64_t>(numHeads)) != 0 || qkv->cols == 0) {
        setError(std::string(lb) + ": QKV.cols must be 3 * numHeads * headDim");
        return nullptr;
    }
    const int L = qkv->rows;
    if (sb->rows != L || sb->cols != 2) {
        setError(std::string(lb) + ": seqBounds must be (" + std::to_string(L) + ", 2)");
        return nullptr;
    }
    std::vector<int32_t> vals;
    if (!hostInt32(lb, "seqBounds", sb, prod({L, 2}), vals)) return nullptr;
    for (int r = 0; r < L; ++r) {
        const int32_t b0 = vals[2 * static_cast<size_t>(r)], b1 = vals[2 * static_cast<size_t>(r) + 1];
        if (b0 < 0 || b0 > r || b1 <= r || b1 > L) {
            setError(std::string(lb) + ": seqBounds row " + std::to_string(r) + " is [" + std::to_string(b0) + ", " +
                     std::to_string(b1) + "), which must lie in [0, " + std::to_string(L) + ") and contain the row");
            return nullptr;
        }
    }
    return &int32Operand(*sb, vals, scratch);
}

} // namespace

extern "C" {

void bro_tensor_flashAttentionPackedQkvForward(void* QKV, void* seqBounds, int32_t numHeads, int32_t window,
                                               void* O) {
    constexpr const char* lb = "flashAttentionPackedQkvForward";
    if (!need(lb, {QKV, seqBounds, O})) return;
    BROTENSOR_API_TRY
        brotensor::Tensor scratch;
        const brotensor::Tensor* sb = packedBounds(lb, toTensor(QKV), numHeads, toTensor(seqBounds), scratch);
        if (!sb) return;
        brotensor::flash_attention_packed_qkv_forward(*toTensor(QKV), *sb, numHeads, window, *toTensor(O));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_flashAttentionPackedQkvBackward(void* QKV, void* dO, void* seqBounds, int32_t numHeads,
                                                int32_t window, void* dQKV) {
    constexpr const char* lb = "flashAttentionPackedQkvBackward";
    if (!need(lb, {QKV, dO, seqBounds, dQKV})) return;
    auto* qkv = toTensor(QKV);
    auto* dout = toTensor(dO);
    if (!needSameDtype(lb, "dO", dout, qkv)) return;
    if (dout->rows != qkv->rows || static_cast<int64_t>(dout->cols) * 3 != qkv->cols) {
        setError(std::string(lb) + ": dO must be (L, numHeads*headDim), a third of QKV's width");
        return;
    }
    BROTENSOR_API_TRY
        brotensor::Tensor scratch;
        const brotensor::Tensor* sb = packedBounds(lb, qkv, numHeads, toTensor(seqBounds), scratch);
        if (!sb) return;
        brotensor::flash_attention_packed_qkv_backward(*qkv, *dout, *sb, numHeads, window, *toTensor(dQKV));
    BROTENSOR_API_CATCH(lb)
}

// ---- Transformer-XL relative-position bias -------------------------------------

void bro_tensor_relPosBiasXlForward(void* Qv, void* Pk, int32_t numHeads, int32_t headDim, void* Bias) {
    constexpr const char* lb = "relPosBiasXlForward";
    if (!need(lb, {Qv, Pk, Bias})) return;
    auto* q = toTensor(Qv);
    auto* pk = toTensor(Pk);
    constexpr auto F = brotensor::Dtype::FP32;
    if (!needDtype(lb, "Qv", q, F) || !needDtype(lb, "Pk", pk, F)) return;
    if (numHeads < 1 || headDim < 1 || q->cols != static_cast<int64_t>(numHeads) * headDim) {
        setError(std::string(lb) + ": Qv must be (T, numHeads*headDim)");
        return;
    }
    const int64_t T = q->rows;
    if (T < 1) { setError(std::string(lb) + ": Qv is empty"); return; }
    if (pk->cols != q->cols || pk->rows < 2 * T - 1) {
        setError(std::string(lb) + ": Pk must be (2T-1, numHeads*headDim) = (" + std::to_string(2 * T - 1) + ", " +
                 std::to_string(q->cols) + ")");
        return;
    }
    if (!needExtent(lb, "numHeads*T*T", prod({numHeads, T, T}))) return;
    BROTENSOR_API_TRY
        brotensor::rel_pos_bias_xl_forward(*q, *pk, numHeads, headDim, *toTensor(Bias));
    BROTENSOR_API_CATCH(lb)
}

// ---- RoPE: per-head tables, packed QKV in place --------------------------------

void bro_tensor_ropeApplyPerhead(void* X, void* cosTbl, void* sinTbl, int32_t headDim, int32_t numHeads, void* Y) {
    constexpr const char* lb = "ropeApplyPerhead";
    if (!need(lb, {X, cosTbl, sinTbl, Y})) return;
    auto* x = toTensor(X);
    if (!needFloat(lb, "X", x)) return;
    if (headDim < 2 || (headDim & 1) || numHeads < 1 || x->cols != static_cast<int64_t>(numHeads) * headDim) {
        setError(std::string(lb) + ": X must be (L, numHeads*headDim) with headDim even");
        return;
    }
    const int64_t n = prod({x->rows, numHeads, headDim / 2});
    if (!optF32Elems(lb, "cosTbl", toTensor(cosTbl), n) || !optF32Elems(lb, "sinTbl", toTensor(sinTbl), n)) return;
    BROTENSOR_API_TRY
        brotensor::rope_apply_perhead(*x, *toTensor(cosTbl), *toTensor(sinTbl), headDim, numHeads, *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_ropeQkvPackedInplace(void* QKV, void* cosTbl, void* sinTbl, void* pos, int32_t numHeads,
                                     int32_t headDim) {
    constexpr const char* lb = "ropeQkvPackedInplace";
    if (!need(lb, {QKV, cosTbl, sinTbl, pos})) return;
    auto* qkv = toTensor(QKV);
    auto* ct = toTensor(cosTbl);
    auto* st = toTensor(sinTbl);
    auto* ps = toTensor(pos);
    if (!needFloat(lb, "QKV", qkv)) return;
    if (headDim < 2 || (headDim & 1) || numHeads < 1 ||
        qkv->cols != 3 * static_cast<int64_t>(numHeads) * headDim) {
        setError(std::string(lb) + ": QKV must be (L, 3*numHeads*headDim) with headDim even");
        return;
    }
    constexpr auto F = brotensor::Dtype::FP32;
    if (!needDtype(lb, "cosTbl", ct, F) || !needDtype(lb, "sinTbl", st, F)) return;
    if (ct->cols != headDim / 2 || st->cols != headDim / 2) {
        setError(std::string(lb) + ": cosTbl / sinTbl must be (P, headDim/2)");
        return;
    }
    if (ps->rows != qkv->rows || ps->cols != 1) {
        setError(std::string(lb) + ": pos must be (" + std::to_string(qkv->rows) + ", 1)");
        return;
    }
    BROTENSOR_API_TRY
        std::vector<int32_t> vals;
        if (!hostInt32(lb, "pos", ps, qkv->rows, vals)) return;
        if (!needIndicesIn(lb, "pos", vals, qkv->rows, 0, std::min(ct->rows, st->rows))) return;
        brotensor::Tensor scratch;
        brotensor::rope_qkv_packed_inplace(*qkv, *ct, *st, int32Operand(*ps, vals, scratch), numHeads, headDim);
    BROTENSOR_API_CATCH(lb)
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_extra_nn(std::string* error) {
    using namespace brotensor::api::reg;
    return
        fn("__bro_native.tensor.linearForwardBatchedEx", p(&bro_tensor_linearForwardBatchedEx), "void",
           {kTensorCls, kDyn, kTensorCls, "i32", "i32", kDyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearForwardBatchedFp16Act", p(&bro_tensor_linearForwardBatchedFp16Act), "void",
           {kTensorCls, kDyn, kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.matmulAbt", p(&bro_tensor_matmulAbt), "void",
           {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "f64", "f64", "f64", kDyn, "i32"},
           error) &&
        fn("__bro_native.tensor.lstmForwardTrain", p(&bro_tensor_lstmForwardTrain), "void",
           {kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, kDyn, kDyn, "i32", "i32",
            kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn}, error) &&
        fn("__bro_native.tensor.lstmBackward", p(&bro_tensor_lstmBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, kTensorCls, kTensorCls, kTensorCls, kTensorCls,
            "i32", "i32", kTensorCls, kTensorCls, kTensorCls, kDyn, kDyn, kDyn, kDyn}, error) &&
        fn("__bro_native.tensor.flashAttentionGqaForward", p(&bro_tensor_flashAttentionGqaForward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kDyn, "i32", "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionPackedQkvForward", p(&bro_tensor_flashAttentionPackedQkvForward),
           "void", {kTensorCls, kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionPackedQkvBackward", p(&bro_tensor_flashAttentionPackedQkvBackward),
           "void", {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.relPosBiasXlForward", p(&bro_tensor_relPosBiasXlForward), "void",
           {kTensorCls, kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeApplyPerhead", p(&bro_tensor_ropeApplyPerhead), "void",
           {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeQkvPackedInplace", p(&bro_tensor_ropeQkvPackedInplace), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, "i32", "i32"}, error);
}

} // namespace brotensor::api
