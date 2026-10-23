// native_tensor_extra.cpp — the brotensor ops that never had a JS binding:
// the elementwise / reduction / index half, the INT32 upload + download pair
// the index-taking ops need, and the group's registration entry point (which
// chains native_tensor_extra_nn.cpp and native_tensor_extra_conv.cpp).
//
// Everything here is a free function on `bro.tensor`; js/tensor_extra.js is
// the wrapper half that validates arguments and reads the error slot back.
// Every body checks each operand against the dims it is handed before the op
// runs (the ops trust their caller's dims; from JS that trust is an
// out-of-bounds read or write) — see native_tensor_extra_util.h.

#include "native_tensor_extra_decl.h"
#include "native_tensor_extra_util.h"
#include "api_internal.h"
#include "native_register.h"
#include <brotensor/trace_autodiff.h>
#include "../jit/trace_dag.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace brotensor::api;
using namespace brotensor::api::extra;

namespace brotensor::api::extra {

bool hostInt32(const char* label, const char* name, const brotensor::Tensor* t, int64_t n,
               std::vector<int32_t>& out) {
    if (!needElems(label, name, t, n)) return false;
    out.clear();
    if (t->empty()) return true;
    const brotensor::Tensor host = t->to(brotensor::Device::CPU);
    const auto count = static_cast<size_t>(host.size());
    out.resize(count);
    if (host.dtype == brotensor::Dtype::INT32) {
        std::memcpy(out.data(), host.host_raw(), count * sizeof(int32_t));
        return true;
    }
    if (host.dtype == brotensor::Dtype::FP32) {
        const float* f = host.host_f32();
        for (size_t i = 0; i < count; ++i) {
            const float v = f[i];
            if (!(v == std::floor(v)) || v < -2147483648.0f || v >= 2147483648.0f) {
                setError(std::string(label) + ": " + name + "[" + std::to_string(i) +
                         "] is not a whole number that fits INT32");
                return false;
            }
            out[i] = static_cast<int32_t>(v);
        }
        return true;
    }
    setError(std::string(label) + ": " + name + " is " + dtypeToString(host.dtype) + ", expected int32 (or whole-number fp32)");
    return false;
}

const brotensor::Tensor& int32Operand(const brotensor::Tensor& t, const std::vector<int32_t>& vals,
                                      brotensor::Tensor& scratch) {
    if (t.dtype == brotensor::Dtype::INT32) return t;
    scratch = brotensor::Tensor::from_raw_bytes_on(t.device, vals.data(), t.rows, t.cols,
                                                   brotensor::Dtype::INT32, vals.size() * sizeof(int32_t));
    return scratch;
}

bool needIndicesIn(const char* label, const char* name, const std::vector<int32_t>& vals, int64_t n,
                   int64_t lo, int64_t hi) {
    for (int64_t i = 0; i < n && i < static_cast<int64_t>(vals.size()); ++i) {
        const int64_t v = vals[static_cast<size_t>(i)];
        if (v < lo || v >= hi) {
            setError(std::string(label) + ": " + name + "[" + std::to_string(i) + "] = " + std::to_string(v) +
                     " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + ")");
            return false;
        }
    }
    return true;
}

} // namespace brotensor::api::extra

namespace {

thread_local std::vector<int32_t> tl_download_i32;

void emptyBuffer(bronze_native_buffer* out) {
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
}

// The FP32-only elementwise pair shape: x FP32, dY the same size.
bool fp32Pair(const char* label, void* x, const char* xn, void* dY) {
    if (!need(label, {x, dY})) return false;
    return needDtype(label, xn, toTensor(x), brotensor::Dtype::FP32) &&
           needDtype(label, "dY", toTensor(dY), brotensor::Dtype::FP32) &&
           needSameSize(label, "dY", toTensor(dY), toTensor(x));
}

} // namespace

extern "C" {

// ---- INT32 upload / download -----------------------------------------------

void bro_tensor_uploadInt32(void* self, const int32_t* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) {
        setError("uploadInt32: not a GpuTensor / Int32Array");
        return;
    }
    if (data_len > static_cast<uint32_t>(INT_MAX)) {
        setError("uploadInt32: too many elements");
        return;
    }
    int r = t->rows, c = t->cols;
    if (static_cast<int64_t>(r) * c != static_cast<int64_t>(data_len)) {
        r = static_cast<int>(data_len);
        c = 1;
    }
    BROTENSOR_API_TRY
        // The typed-array view is only valid for this call: copy it into the
        // tensor before anything else runs.
        *t = brotensor::Tensor::from_raw_bytes_on(brotensor::default_device(), data, r, c,
                                                  brotensor::Dtype::INT32,
                                                  static_cast<size_t>(data_len) * sizeof(int32_t));
    BROTENSOR_API_CATCH("uploadInt32")
}

void bro_tensor_downloadInt32(void* self, bronze_native_buffer* out) {
    if (!out) return;
    emptyBuffer(out);
    auto* t = toTensor(self);
    if (!t) {
        setError("downloadInt32: not a GpuTensor");
        return;
    }
    if (t->empty()) return;
    if (t->dtype != brotensor::Dtype::INT32) {
        setError(std::string("downloadInt32: tensor is ") + dtypeToString(t->dtype) + ", expected int32");
        return;
    }
    BROTENSOR_API_TRY
        const brotensor::Tensor host = t->to(brotensor::Device::CPU);
        tl_download_i32.resize(static_cast<size_t>(host.size()));
        std::memcpy(tl_download_i32.data(), host.host_raw(), tl_download_i32.size() * sizeof(int32_t));
        out->data = tl_download_i32.data();
        out->length = static_cast<uint32_t>(tl_download_i32.size());
    BROTENSOR_API_CATCH("downloadInt32")
}

// ---- bias adds + axpby -----------------------------------------------------

void bro_tensor_addChannelBiasInplace(void* y, void* bias, int32_t C, int32_t L) {
    constexpr const char* lb = "addChannelBiasInplace";
    if (!need(lb, {y, bias})) return;
    auto* Y = toTensor(y);
    auto* B = toTensor(bias);
    if (!needFloat(lb, "y", Y) || !needSameDtype(lb, "bias", B, Y)) return;
    const int64_t n = prod({C, L});
    if (n < 0) { setError(std::string(lb) + ": negative C or L"); return; }
    if (n != Y->size()) {
        setError(std::string(lb) + ": y holds " + std::to_string(Y->size()) + " elements, C*L is " + std::to_string(n));
        return;
    }
    if (!needElems(lb, "bias", B, C)) return;
    BROTENSOR_API_TRY
        brotensor::add_channel_bias_inplace(*Y, *B, C, L);
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_addRowBiasInplace(void* y, void* bias) {
    constexpr const char* lb = "addRowBiasInplace";
    if (!need(lb, {y, bias})) return;
    auto* Y = toTensor(y);
    auto* B = toTensor(bias);
    if (!needFloat(lb, "Y", Y) || !needSameDtype(lb, "bias", B, Y)) return;
    if (B->size() != Y->cols) {
        setError(std::string(lb) + ": bias holds " + std::to_string(B->size()) + " elements, Y has " +
                 std::to_string(Y->cols) + " columns");
        return;
    }
    BROTENSOR_API_TRY
        brotensor::add_row_bias_inplace(*Y, *B);
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_axpbyInplace(void* y, void* x, double a, double b) {
    constexpr const char* lb = "axpbyInplace";
    if (!need(lb, {y, x})) return;
    auto* Y = toTensor(y);
    auto* X = toTensor(x);
    if (!needFloat(lb, "y", Y) || !needSameDtype(lb, "x", X, Y) || !needSameSize(lb, "x", X, Y)) return;
    BROTENSOR_API_TRY
        brotensor::axpby_inplace(*Y, *X, static_cast<float>(a), static_cast<float>(b));
    BROTENSOR_API_CATCH(lb)
}

// ---- sin / cos / rsqrt (FP32 elementwise) ----------------------------------

void bro_tensor_sinForward(void* x, void* y) {
    if (!need("sinForward", {x, y}) || !needDtype("sinForward", "x", toTensor(x), brotensor::Dtype::FP32)) return;
    BROTENSOR_API_TRY
        brotensor::sin_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("sinForward")
}

void bro_tensor_sinBackward(void* x, void* dY, void* dX) {
    if (!need("sinBackward", {dX}) || !fp32Pair("sinBackward", x, "x", dY)) return;
    BROTENSOR_API_TRY
        brotensor::sin_backward(*toTensor(x), *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("sinBackward")
}

void bro_tensor_cosForward(void* x, void* y) {
    if (!need("cosForward", {x, y}) || !needDtype("cosForward", "x", toTensor(x), brotensor::Dtype::FP32)) return;
    BROTENSOR_API_TRY
        brotensor::cos_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("cosForward")
}

void bro_tensor_cosBackward(void* x, void* dY, void* dX) {
    if (!need("cosBackward", {dX}) || !fp32Pair("cosBackward", x, "x", dY)) return;
    BROTENSOR_API_TRY
        brotensor::cos_backward(*toTensor(x), *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("cosBackward")
}

void bro_tensor_rsqrtForward(void* x, void* y) {
    if (!need("rsqrtForward", {x, y}) || !needDtype("rsqrtForward", "x", toTensor(x), brotensor::Dtype::FP32)) return;
    BROTENSOR_API_TRY
        brotensor::rsqrt_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("rsqrtForward")
}

void bro_tensor_rsqrtBackward(void* y, void* dY, void* dX) {
    if (!need("rsqrtBackward", {dX}) || !fp32Pair("rsqrtBackward", y, "y", dY)) return;
    BROTENSOR_API_TRY
        brotensor::rsqrt_backward(*toTensor(y), *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("rsqrtBackward")
}

// ---- pixel norm --------------------------------------------------------------

void bro_tensor_pixelNormForward(void* X, double eps, void* Y) {
    if (!need("pixelNormForward", {X, Y}) ||
        !needDtype("pixelNormForward", "X", toTensor(X), brotensor::Dtype::FP32)) return;
    BROTENSOR_API_TRY
        brotensor::pixel_norm_forward(*toTensor(X), static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("pixelNormForward")
}

void bro_tensor_pixelNormBackward(void* X, void* dY, double eps, void* dX) {
    if (!need("pixelNormBackward", {dX}) || !fp32Pair("pixelNormBackward", X, "X", dY)) return;
    BROTENSOR_API_TRY
        brotensor::pixel_norm_backward(*toTensor(X), *toTensor(dY), static_cast<float>(eps), *toTensor(dX));
    BROTENSOR_API_CATCH("pixelNormBackward")
}

// ---- thresholds + row counts -------------------------------------------------

void bro_tensor_thresholdU8(void* X, double t, void* Y) {
    if (!need("thresholdU8", {X, Y}) || !needFloat("thresholdU8", "X", toTensor(X))) return;
    BROTENSOR_API_TRY
        brotensor::threshold_u8(*toTensor(X), static_cast<float>(t), *toTensor(Y));
    BROTENSOR_API_CATCH("thresholdU8")
}

void bro_tensor_rowsCountAbove(void* X, double tLo, double tHi, void* counts) {
    if (!need("rowsCountAbove", {X, counts}) || !needFloat("rowsCountAbove", "X", toTensor(X))) return;
    BROTENSOR_API_TRY
        brotensor::rows_count_above(*toTensor(X), static_cast<float>(tLo), static_cast<float>(tHi),
                                    *toTensor(counts));
    BROTENSOR_API_CATCH("rowsCountAbove")
}

// ---- softmax ---------------------------------------------------------------

void bro_tensor_softmaxRowsForward(void* X, void* Y, int32_t rows, int32_t cols) {
    constexpr const char* lb = "softmaxRowsForward";
    if (!need(lb, {X, Y})) return;
    auto* x = toTensor(X);
    if (!needFloat(lb, "X", x) || !needElems(lb, "X", x, prod({rows, cols}))) return;
    BROTENSOR_API_TRY
        brotensor::softmax_rows_forward(*x, *toTensor(Y), rows, cols);
    BROTENSOR_API_CATCH(lb)
}

double bro_tensor_softmaxXent(void* logits, void* target, void* probs, void* dLogits, uint64_t mask_bits) {
    constexpr const char* lb = "softmaxXent";
    if (!need(lb, {logits, target, probs, dLogits})) return 0.0;
    auto* L = toTensor(logits);
    auto* T = toTensor(target);
    if (!needDtype(lb, "logits", L, brotensor::Dtype::FP32) || !needDtype(lb, "target", T, brotensor::Dtype::FP32) ||
        !needSameSize(lb, "target", T, L)) return 0.0;
    auto* M = tensorFromValue(mask_bits);
    if (!optF32Elems(lb, "mask", M, L->size())) return 0.0;
    BROTENSOR_API_TRY
        return brotensor::softmax_xent(*L, *T, *toTensor(probs), *toTensor(dLogits),
                                       M ? static_cast<const float*>(M->data) : nullptr);
    BROTENSOR_API_CATCH(lb)
    return 0.0;
}

// ---- strided copy + row scatter ----------------------------------------------

void bro_tensor_copyD2DStrided(void* src, int32_t srcOff, int32_t srcPitch, void* dst, int32_t dstOff,
                               int32_t dstPitch, int32_t width, int32_t height) {
    constexpr const char* lb = "copyD2DStrided";
    if (!need(lb, {src, dst})) return;
    auto* S = toTensor(src);
    auto* D = toTensor(dst);
    if (!needSameDtype(lb, "dst", D, S)) return;
    if (width < 0 || height < 0 || srcOff < 0 || dstOff < 0) {
        setError(std::string(lb) + ": offsets, width and height must be non-negative");
        return;
    }
    if (srcPitch < width || dstPitch < width) {
        setError(std::string(lb) + ": pitch must be >= width");
        return;
    }
    if (width == 0 || height == 0) return;
    const int64_t srcNeed = static_cast<int64_t>(srcOff) + static_cast<int64_t>(height - 1) * srcPitch + width;
    const int64_t dstNeed = static_cast<int64_t>(dstOff) + static_cast<int64_t>(height - 1) * dstPitch + width;
    if (!needElems(lb, "src", S, srcNeed) || !needElems(lb, "dst", D, dstNeed)) return;
    BROTENSOR_API_TRY
        brotensor::copy_d2d_strided(*S, srcOff, srcPitch, *D, dstOff, dstPitch, width, height);
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_scatterRows(void* Y, void* Idx, void* X) {
    constexpr const char* lb = "scatterRows";
    if (!need(lb, {Y, Idx, X})) return;
    auto* y = toTensor(Y);
    auto* x = toTensor(X);
    if (!needFloat(lb, "Y", y) || !needSameDtype(lb, "X", x, y)) return;
    if (x->empty()) { setError(std::string(lb) + ": X must be allocated (it is updated in place)"); return; }
    if (x->cols != y->cols) {
        setError(std::string(lb) + ": X has " + std::to_string(x->cols) + " columns, Y has " + std::to_string(y->cols));
        return;
    }
    auto* I = toTensor(Idx);
    if (I->rows != y->rows || I->cols != 1) {
        setError(std::string(lb) + ": Idx must be (" + std::to_string(y->rows) + ", 1), one row index per Y row");
        return;
    }
    BROTENSOR_API_TRY
        std::vector<int32_t> idx;
        if (!hostInt32(lb, "Idx", I, y->rows, idx)) return;
        if (!needIndicesIn(lb, "Idx", idx, y->rows, 0, x->rows)) return;
        brotensor::Tensor scratch;
        brotensor::scatter_rows(*y, int32Operand(*I, idx, scratch), *x);
    BROTENSOR_API_CATCH(lb)
}

// ---- segment softmax statistics ------------------------------------------------

void bro_tensor_segmentSoftmaxStats(void* logits, void* segOffsets, void* out) {
    constexpr const char* lb = "segmentSoftmaxStats";
    if (!need(lb, {logits, segOffsets, out})) return;
    auto* L = toTensor(logits);
    auto* O = toTensor(segOffsets);
    if (!needFloat(lb, "logits", L)) return;
    if (O->cols != 1 || O->rows < 1) { setError(std::string(lb) + ": segOffsets must be (S+1, 1)"); return; }
    BROTENSOR_API_TRY
        std::vector<int32_t> off;
        if (!hostInt32(lb, "segOffsets", O, O->rows, off)) return;
        const int64_t K = L->size();
        for (size_t s = 0; s < off.size(); ++s) {
            const bool bad = off[s] < 0 || off[s] > K || (s > 0 && off[s] < off[s - 1]);
            if (bad) {
                setError(std::string(lb) + ": segOffsets must be non-decreasing within [0, " + std::to_string(K) +
                         "]; entry " + std::to_string(s) + " is " + std::to_string(off[s]));
                return;
            }
        }
        brotensor::Tensor scratch;
        brotensor::segment_softmax_stats(*L, int32Operand(*O, off, scratch), *toTensor(out));
    BROTENSOR_API_CATCH(lb)
}

// ---- masked-diffusion token selection ------------------------------------------

void bro_tensor_maskedDiffusionScores(void* logits, void* tokens, int32_t T, int32_t C, int32_t V,
                                      int32_t maskId, double guidanceScale, double layerPenalty,
                                      double positionTemperature, double classTemperature,
                                      double classTopFrac, uint64_t seed_bits,
                                      void* pred, void* scores, void* confidence) {
    constexpr const char* lb = "maskedDiffusionScores";
    if (!need(lb, {logits, tokens, pred, scores, confidence})) return;
    if (T < 1 || C < 1 || V < 1) { setError(std::string(lb) + ": T, C and V must be >= 1"); return; }
    if (!needRange(lb, "maskId", maskId, 0, static_cast<int64_t>(V) - 1)) return;
    auto* L = toTensor(logits);
    const int64_t R = guidanceScale != 0.0 ? 2 * static_cast<int64_t>(T) : T;
    if (!needDtype(lb, "logits", L, brotensor::Dtype::FP32) || !needElems(lb, "logits", L, prod({R, C, V}))) return;
    if (!needExtent(lb, "C*T", prod({C, T}))) return;
    const uint64_t seed = bronze::embed::toUint64(bronze::Value{seed_bits});
    BROTENSOR_API_TRY
        std::vector<int32_t> tok;
        auto* tk = toTensor(tokens);
        if (!hostInt32(lb, "tokens", tk, prod({C, T}), tok)) return;
        brotensor::Tensor scratch;
        const brotensor::Tensor& tokI = int32Operand(*tk, tok, scratch);
        brotensor::masked_diffusion_scores(*L, tokI, T, C, V, maskId, static_cast<float>(guidanceScale),
                                           static_cast<float>(layerPenalty),
                                           static_cast<float>(positionTemperature),
                                           static_cast<float>(classTemperature),
                                           static_cast<float>(classTopFrac), seed,
                                           *toTensor(pred), *toTensor(scores), *toTensor(confidence));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_maskedDiffusionCommit(void* pred, void* idx, int32_t k, int32_t step, void* tokens,
                                      void* unmaskStep) {
    constexpr const char* lb = "maskedDiffusionCommit";
    if (!need(lb, {pred, idx, tokens, unmaskStep})) return;
    auto* P = toTensor(pred);
    auto* Tk = toTensor(tokens);
    auto* U = toTensor(unmaskStep);
    if (!needDtype(lb, "pred", P, brotensor::Dtype::INT32) || !needDtype(lb, "tokens", Tk, brotensor::Dtype::INT32) ||
        !needDtype(lb, "unmaskStep", U, brotensor::Dtype::INT32)) return;
    if (!needSameSize(lb, "tokens", Tk, P) || !needSameSize(lb, "unmaskStep", U, P)) return;
    if (k < 0) { setError(std::string(lb) + ": k must be >= 0"); return; }
    if (k == 0) return;
    BROTENSOR_API_TRY
        auto* I = toTensor(idx);
        brotensor::Tensor scratch;
        const brotensor::Tensor* use = I;
        if (I->dtype == brotensor::Dtype::INT32) {
            if (!needElems(lb, "idx", I, k)) return;
        } else {
            std::vector<int32_t> vals;
            if (!hostInt32(lb, "idx", I, k, vals)) return;
            use = &int32Operand(*I, vals, scratch);
        }
        brotensor::masked_diffusion_commit(*P, *use, k, step, *Tk, *U);
    BROTENSOR_API_CATCH(lb)
}

// ---- verified fused backward trace JIT --------------------------------------

namespace {
std::mutex g_jit_bwd_mu;
int32_t g_next_jit_bwd_id = 1;
std::unordered_map<int32_t, brotensor::jit::VerifiedBackwardTrace> g_jit_bwd_table;
} // namespace

int32_t bro_tensor_compileJitBackward(const int32_t* ops, uint32_t ops_len,
                                      const float* scalars, uint32_t scalars_len,
                                      int32_t num_inputs, int32_t rows, int32_t cols) {
    constexpr const char* lb = "compileJitBackward";
    if (!ops || ops_len == 0 || (ops_len % 5) != 0) {
        setError(std::string(lb) + ": invalid ops table");
        return -1;
    }
    const uint32_t n_nodes = ops_len / 5;
    if (!scalars || scalars_len < n_nodes * 2) {
        setError(std::string(lb) + ": invalid scalars table");
        return -1;
    }
    if (num_inputs < 1 || rows < 1 || cols < 1) {
        setError(std::string(lb) + ": num_inputs, rows, and cols must be >= 1");
        return -1;
    }
    BROTENSOR_API_TRY
        static float dummy_buf[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        brotensor::jit::TraceDAG fwd;
        for (uint32_t i = 0; i < n_nodes; ++i) {
            const int32_t op_code = ops[i * 5 + 0];
            const int32_t arity = ops[i * 5 + 1];
            std::vector<int> ins;
            for (int32_t k = 0; k < arity && k < 3; ++k) {
                ins.push_back(ops[i * 5 + 2 + k]);
            }
            const float sc1 = scalars[i * 2 + 0];
            const float sc2 = scalars[i * 2 + 1];
            const auto kind = static_cast<brotensor::jit::TraceOpKind>(op_code);
            void* buf = (kind == brotensor::jit::TraceOpKind::Input) ? dummy_buf : nullptr;
            fwd.add_node(kind, ins, sc1, buf, brotensor::default_device(),
                         brotensor::Dtype::FP32, rows, cols, sc2);
        }
        fwd.nodes().back().is_live_at_end = true;

        brotensor::jit::VerifiedBackwardTrace compiled =
            brotensor::jit::build_verified_backward_trace(fwd);
        std::lock_guard<std::mutex> lock(g_jit_bwd_mu);
        const int32_t id = g_next_jit_bwd_id++;
        g_jit_bwd_table[id] = std::move(compiled);
        return id;
    BROTENSOR_API_CATCH(lb)
    return -1;
}

void bro_tensor_getJitBackwardReport(int32_t handle_id, bronze_native_buffer* out) {
    if (!out) return;
    emptyBuffer(out);
    std::lock_guard<std::mutex> lock(g_jit_bwd_mu);
    auto it = g_jit_bwd_table.find(handle_id);
    if (it == g_jit_bwd_table.end()) {
        setError("getJitBackwardReport: invalid handle_id");
        return;
    }
    const auto& v = it->second.verification;
    tl_download_i32.assign({
        v.mirEquivProven ? 1 : 0,
        v.spirvPtxEquivProven ? 1 : 0,
        v.nanSafeProven ? 1 : 0,
        static_cast<int32_t>(v.forwardNodeCount),
        static_cast<int32_t>(v.rawBackwardNodeCount),
        static_cast<int32_t>(v.loweredBackwardNodeCount),
        static_cast<int32_t>(v.unoptimizedMirInsts),
        static_cast<int32_t>(v.optimizedMirInsts),
        static_cast<int32_t>(v.spirvWordCount),
        static_cast<int32_t>(v.ptxByteCount),
    });
    out->data = tl_download_i32.data();
    out->length = static_cast<uint32_t>(tl_download_i32.size());
}

void bro_tensor_runJitBackwardInput(int32_t handle_id, int32_t input_idx,
                                    void* x0, uint64_t x1_bits, uint64_t x2_bits, uint64_t x3_bits,
                                    void* grad_out, void* dx_out) {
    constexpr const char* lb = "runJitBackwardInput";
    if (!need(lb, {x0, grad_out, dx_out})) return;
    brotensor::jit::VerifiedBackwardTrace trace_copy;
    {
        std::lock_guard<std::mutex> lock(g_jit_bwd_mu);
        auto it = g_jit_bwd_table.find(handle_id);
        if (it == g_jit_bwd_table.end()) {
            setError(std::string(lb) + ": invalid handle_id");
            return;
        }
        trace_copy = it->second;
    }
    std::vector<brotensor::Tensor> primals;
    primals.push_back(brotensor::Tensor::view(
        toTensor(x0)->device, toTensor(x0)->data, toTensor(x0)->rows, toTensor(x0)->cols, toTensor(x0)->dtype));
    const uint64_t opt_bits[3] = {x1_bits, x2_bits, x3_bits};
    for (size_t i = 0; i < 3 && primals.size() < trace_copy.num_inputs; ++i) {
        auto* ti = tensorFromValue(opt_bits[i]);
        if (!ti) {
            setError(std::string(lb) + ": missing primal input tensor");
            return;
        }
        primals.push_back(brotensor::Tensor::view(ti->device, ti->data, ti->rows, ti->cols, ti->dtype));
    }
    BROTENSOR_API_TRY
        brotensor::Tensor g = trace_copy.execute_for_input(input_idx, primals, *toTensor(grad_out));
        brotensor::copy_d2d(g, 0, *toTensor(dx_out), 0, g.size());
    BROTENSOR_API_CATCH(lb)
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_extra(std::string* error) {
    using namespace brotensor::api::reg;
    return
        fn("__bro_native.tensor.uploadInt32", p(&bro_tensor_uploadInt32), "void",
           {kTensorCls, "i32[]"}, error) &&
        fn("__bro_native.tensor.downloadInt32", p(&bro_tensor_downloadInt32), "i32[]",
           {kTensorCls}, error) &&
        fn("__bro_native.tensor.addChannelBiasInplace", p(&bro_tensor_addChannelBiasInplace), "void",
           {kTensorCls, kTensorCls, "i32", "i32"}, error) &&
        fn("__bro_native.tensor.addRowBiasInplace", p(&bro_tensor_addRowBiasInplace), "void",
           {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.axpbyInplace", p(&bro_tensor_axpbyInplace), "void",
           {kTensorCls, kTensorCls, "f64", "f64"}, error) &&
        fn("__bro_native.tensor.sinForward", p(&bro_tensor_sinForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.sinBackward", p(&bro_tensor_sinBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.cosForward", p(&bro_tensor_cosForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.cosBackward", p(&bro_tensor_cosBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.rsqrtForward", p(&bro_tensor_rsqrtForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.rsqrtBackward", p(&bro_tensor_rsqrtBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.pixelNormForward", p(&bro_tensor_pixelNormForward), "void",
           {kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.pixelNormBackward", p(&bro_tensor_pixelNormBackward), "void",
           {kTensorCls, kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.thresholdU8", p(&bro_tensor_thresholdU8), "void",
           {kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.rowsCountAbove", p(&bro_tensor_rowsCountAbove), "void",
           {kTensorCls, "f64", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.softmaxRowsForward", p(&bro_tensor_softmaxRowsForward), "void",
           {kTensorCls, kTensorCls, "i32", "i32"}, error) &&
        fn("__bro_native.tensor.softmaxXent", p(&bro_tensor_softmaxXent), "f64",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kDyn}, error) &&
        fn("__bro_native.tensor.copyD2DStrided", p(&bro_tensor_copyD2DStrided), "void",
           {kTensorCls, "i32", "i32", kTensorCls, "i32", "i32", "i32", "i32"}, error) &&
        fn("__bro_native.tensor.scatterRows", p(&bro_tensor_scatterRows), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.segmentSoftmaxStats", p(&bro_tensor_segmentSoftmaxStats), "void",
           {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.maskedDiffusionScores", p(&bro_tensor_maskedDiffusionScores), "void",
           {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "f64", "f64", "f64", "f64", "f64", kDyn,
            kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.maskedDiffusionCommit", p(&bro_tensor_maskedDiffusionCommit), "void",
           {kTensorCls, kTensorCls, "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.compileJitBackward", p(&bro_tensor_compileJitBackward), "i32",
           {"i32[]", "f32[]", "i32", "i32", "i32"}, error) &&
        fn("__bro_native.tensor.getJitBackwardReport", p(&bro_tensor_getJitBackwardReport), "i32[]",
           {"i32"}, error) &&
        fn("__bro_native.tensor.runJitBackwardInput", p(&bro_tensor_runJitBackwardInput), "void",
           {"i32", "i32", kTensorCls, kDyn, kDyn, kDyn, kTensorCls, kTensorCls}, error) &&
        registerTensorNatives_extra_nn(error) &&
        registerTensorNatives_extra_conv(error);
}

} // namespace brotensor::api
