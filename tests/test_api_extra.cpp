// bro.tensor checks for the ops that never had a binding before
// (src/api/native_tensor_extra*.cpp, js/tensor_extra.js).
//
// Each op is checked against a small hand-computed expectation on whatever
// device brotensor picked, and each family carries at least one undersized
// operand (or out-of-range index) that must throw an Error from the native's
// size check instead of reaching the kernel. The 16-bit-only ops (matmulAbt,
// linearForwardBatchedFp16Act, and linearForwardBatchedEx on a GPU) run at
// FP16 through `cast`.

#include "api_test_helpers.h"

#include <string>

using brotensor_api_test::runJs;

namespace {

// Shared helpers, prepended to every block.
const char* kPrelude = R"JS(
    const tensor = globalThis.bro.tensor;
    const onCpu = tensor.backend === "cpu";
    // A tensor (r, c) holding `data` (FP32), cast to `dt` when given.
    const mk = (r, c, data, dt) => {
        const x = tensor.createTensor(r, c);
        if (data) x.upload(data);
        if (dt && dt !== "fp32") {
            const y = tensor.createTensor(1, 1);
            tensor.cast(x, y, dt);
            return y;
        }
        return x;
    };
    const i32 = (r, c, data) => { const x = tensor.createTensor(r, c, "int32"); x.uploadInt32(data); return x; };
    const out = () => tensor.createTensor(1, 1);
    const near = (got, want, tol, label) => {
        if (got.length < want.length) throw new Error(label + ": got " + got.length + " values, want " + want.length);
        for (let i = 0; i < want.length; i++) {
            if (!(Math.abs(got[i] - want[i]) <= tol)) {
                throw new Error(label + "[" + i + "] = " + got[i] + ", want " + want[i] + " (all: " + Array.from(got) + ")");
            }
        }
    };
    // The call must throw an Error (not a TypeError from a wrapper) whose
    // message contains `needle`.
    const throws = (f, needle, label) => {
        let msg = null;
        try { f(); } catch (e) { msg = String(e && e.message); }
        if (msg === null) throw new Error(label + " did not throw");
        if (needle && msg.indexOf(needle) < 0) throw new Error(label + " threw '" + msg + "', expected '" + needle + "'");
    };
)JS";

int block(const char* name, const char* body) {
    return runJs(name, std::string("(function () {\n") + kPrelude + body + "\n    return \"OK\";\n})();\n");
}

} // namespace

int run_api_extra_tests() {
    int f = 0;

    // ---- INT32 storage, bias adds, axpby -------------------------------------
    f += block("test_api_extra.int32_bias_axpby", R"JS(
        const ix = i32(2, 2, [1, -2, 3, 2147483647]);
        if (ix.dtype() !== "int32" || ix.rows !== 2 || ix.cols !== 2) throw new Error("uploadInt32 shape/dtype");
        near(ix.downloadInt32(), [1, -2, 3, 2147483647], 0, "downloadInt32");
        throws(() => mk(1, 2, [1, 2]).downloadInt32(), "expected int32", "downloadInt32 of fp32");

        // Channel bias over a (C=2, L=3) block.
        const y = mk(2, 3, [0, 0, 0, 1, 1, 1]);
        tensor.addChannelBiasInplace(y, mk(2, 1, [1, 2]), 2, 3);
        near(y.download(), [1, 1, 1, 3, 3, 3], 1e-6, "addChannelBiasInplace");
        throws(() => tensor.addChannelBiasInplace(y, mk(1, 1, [1]), 2, 3), "bias holds 1", "addChannelBiasInplace short bias");
        throws(() => tensor.addChannelBiasInplace(y, mk(2, 1, [1, 2]), 4, 3), "C*L", "addChannelBiasInplace C*L past y");

        const Y = mk(2, 3, [0, 0, 0, 1, 1, 1]);
        tensor.addRowBiasInplace(Y, mk(3, 1, [1, 2, 3]));
        near(Y.download(), [1, 2, 3, 2, 3, 4], 1e-6, "addRowBiasInplace");
        throws(() => tensor.addRowBiasInplace(Y, mk(2, 1, [1, 2])), "bias holds 2", "addRowBiasInplace short bias");

        const a = mk(1, 2, [1, 2]);
        tensor.axpbyInplace(a, mk(1, 2, [3, 4]), 2, -1);
        near(a.download(), [-1, 0], 1e-6, "axpbyInplace");
        throws(() => tensor.axpbyInplace(a, mk(1, 1, [3]), 2, -1), "elements", "axpbyInplace short x");
    )JS");

    // ---- sin / cos / rsqrt / pixel norm ---------------------------------------
    f += block("test_api_extra.elementwise", R"JS(
        const xs = [0, 0.5, 1, 2];
        const x = mk(1, 4, xs), dY = mk(1, 4, [1, 2, 3, 4]);
        const y = out(), dX = out();
        tensor.sinForward(x, y);
        near(y.download(), xs.map(Math.sin), 1e-5, "sinForward");
        tensor.sinBackward(x, dY, dX);
        near(dX.download(), xs.map((v, i) => (i + 1) * Math.cos(v)), 1e-5, "sinBackward");
        tensor.cosForward(x, y);
        near(y.download(), xs.map(Math.cos), 1e-5, "cosForward");
        tensor.cosBackward(x, dY, dX);
        near(dX.download(), xs.map((v, i) => -(i + 1) * Math.sin(v)), 1e-5, "cosBackward");

        const p = mk(1, 3, [1, 4, 0.25]);
        tensor.rsqrtForward(p, y);
        near(y.download(), [1, 0.5, 2], 1e-5, "rsqrtForward");
        tensor.rsqrtBackward(y, mk(1, 3, [1, 1, 1]), dX);
        near(dX.download(), [-0.5, -0.0625, -4], 1e-5, "rsqrtBackward");

        throws(() => tensor.sinBackward(x, mk(1, 2, [1, 2]), dX), "elements", "sinBackward short dY");
        throws(() => tensor.rsqrtBackward(y, mk(1, 1, [1]), dX), "elements", "rsqrtBackward short dY");
        throws(() => tensor.cosForward(mk(1, 2, [1, 2], "fp16"), y), "expected fp32", "cosForward fp16");

        // Pixel norm of one row [3, 4]: r = 1/sqrt(12.5).
        const X = mk(1, 2, [3, 4]);
        tensor.pixelNormForward(X, 0, y);
        const r = 1 / Math.sqrt(12.5);
        near(y.download(), [3 * r, 4 * r], 1e-5, "pixelNormForward");
        // dY = [1, 0]: s = 3, dX_c = r*dY_c - r^3*X_c/2*s.
        tensor.pixelNormBackward(X, mk(1, 2, [1, 0]), 0, dX);
        near(dX.download(), [r - 4.5 * r * r * r, -6 * r * r * r], 1e-5, "pixelNormBackward");
        throws(() => tensor.pixelNormBackward(X, mk(1, 1, [1]), 0, dX), "elements", "pixelNormBackward short dY");
    )JS");

    // ---- threshold / counts / softmax ------------------------------------------
    f += block("test_api_extra.threshold_softmax", R"JS(
        const X = mk(2, 3, [0.1, 0.6, 0.9, -1, 0.5, 2]);
        const M = out();
        tensor.thresholdU8(X, 0.5, M);
        if (M.dtype() !== "int8") throw new Error("thresholdU8 dtype " + M.dtype());
        near(M.downloadInt8(), [0, 1, 1, 0, 0, 1], 0, "thresholdU8");
        const counts = out();
        tensor.rowsCountAbove(X, 0, 0.7, counts);
        if (counts.rows !== 2 || counts.cols !== 2) throw new Error("rowsCountAbove shape");
        near(counts.downloadInt32(), [3, 1, 2, 1], 0, "rowsCountAbove");
        throws(() => tensor.thresholdU8(i32(1, 2, [1, 2]), 0, M), "expected fp32", "thresholdU8 int32 input");

        const S = mk(2, 2, [0, 0, 0, Math.log(3)]);
        const P = out();
        tensor.softmaxRowsForward(S, P, 2, 2);
        near(P.download(), [0.5, 0.5, 0.25, 0.75], 1e-5, "softmaxRowsForward");
        throws(() => tensor.softmaxRowsForward(S, P, 3, 2), "the dims need 6", "softmaxRowsForward rows past X");

        const probs = out(), dL = out();
        const loss = tensor.softmaxXent(mk(1, 2, [0, 0]), mk(1, 2, [1, 0]), probs, dL, null);
        near([loss], [Math.log(2)], 1e-5, "softmaxXent loss");
        near(probs.download(), [0.5, 0.5], 1e-5, "softmaxXent probs");
        near(dL.download(), [-0.5, 0.5], 1e-5, "softmaxXent dLogits");
        const lossM = tensor.softmaxXent(mk(1, 2, [0, 0]), mk(1, 2, [1, 0]), probs, dL, mk(2, 1, [1, 0]));
        near([lossM], [0], 1e-5, "softmaxXent masked loss");
        throws(() => tensor.softmaxXent(mk(1, 2, [0, 0]), mk(1, 2, [1, 0]), probs, dL, mk(1, 1, [1])),
               "mask holds 1", "softmaxXent short mask");
        throws(() => tensor.softmaxXent(mk(1, 2, [0, 0]), mk(1, 1, [1]), probs, dL, null), "elements", "softmaxXent short target");
    )JS");

    // ---- strided copy, row scatter, segment stats -------------------------------
    f += block("test_api_extra.copy_scatter_segments", R"JS(
        const src = mk(2, 4, [0, 1, 2, 3, 4, 5, 6, 7]);
        const dst = mk(2, 3, [9, 9, 9, 9, 9, 9]);
        tensor.copyD2DStrided(src, 1, 4, dst, 0, 3, 2, 2);
        near(dst.download(), [1, 2, 9, 5, 6, 9], 0, "copyD2DStrided");
        throws(() => tensor.copyD2DStrided(src, 1, 4, dst, 0, 3, 2, 3), "src holds 8", "copyD2DStrided rows past src");
        throws(() => tensor.copyD2DStrided(src, 0, 4, dst, 2, 3, 2, 2), "dst holds 6", "copyD2DStrided rows past dst");

        const X = mk(3, 2, [0, 0, 0, 0, 0, 0]);
        const Yr = mk(2, 2, [1, 2, 3, 4]);
        tensor.scatterRows(Yr, mk(2, 1, [2, 0]), X);        // FP32 whole-number Idx
        near(X.download(), [3, 4, 0, 0, 1, 2], 0, "scatterRows fp32 Idx");
        tensor.scatterRows(Yr, i32(2, 1, [1, 2]), X);       // INT32 Idx
        near(X.download(), [3, 4, 1, 2, 3, 4], 0, "scatterRows int32 Idx");
        throws(() => tensor.scatterRows(Yr, i32(2, 1, [0, 3]), X), "outside [0, 3)", "scatterRows row past X");
        throws(() => tensor.scatterRows(Yr, i32(1, 1, [0]), X), "Idx must be", "scatterRows short Idx");

        const logits = mk(3, 1, [0, 0, 5]);
        const stats = out();
        tensor.segmentSoftmaxStats(logits, i32(3, 1, [0, 2, 3]), stats);
        if (stats.rows !== 2 || stats.cols !== 4) throw new Error("segmentSoftmaxStats shape");
        near(stats.download(), [0.5, 0, 1, 2 / 255, 1, 1, 0, 2 / 255], 1e-5, "segmentSoftmaxStats");
        throws(() => tensor.segmentSoftmaxStats(logits, i32(3, 1, [0, 4, 3]), stats), "non-decreasing", "segmentSoftmaxStats offset past K");
    )JS");

    // ---- masked-diffusion scores + commit --------------------------------------
    f += block("test_api_extra.masked_diffusion", R"JS(
        // C = 1 codebook, T = 2 frames, V = 3, mask id 2; both cells masked.
        const logits = mk(2, 3, [1, 0, 0, 0, 2, 0]);
        const tokens = i32(1, 2, [2, 2]);
        const pred = out(), scores = out(), conf = out();
        tensor.maskedDiffusionScores(logits, tokens, 2, 1, 3, 2, null, pred, scores, conf);
        near(pred.downloadInt32(), [0, 1], 0, "maskedDiffusionScores pred");
        const c0 = 1 - Math.log(Math.E + 2), c1 = 2 - Math.log(Math.E * Math.E + 2);
        near(conf.download(), [c0, c1], 1e-5, "maskedDiffusionScores confidence");
        near(scores.download(), [c0, c1], 1e-5, "maskedDiffusionScores scores");
        throws(() => tensor.maskedDiffusionScores(logits, tokens, 3, 1, 3, 2, null, pred, scores, conf),
               "the dims need 9", "maskedDiffusionScores T past logits");
        throws(() => tensor.maskedDiffusionScores(logits, i32(1, 1, [2]), 2, 1, 3, 2, null, pred, scores, conf),
               "tokens holds 1", "maskedDiffusionScores short tokens");

        // Commit cell 1 at step 5.
        const unmask = i32(1, 2, [0, 0]);
        tensor.maskedDiffusionCommit(pred, i32(1, 1, [1]), 1, 5, tokens, unmask);
        near(tokens.downloadInt32(), [2, 1], 0, "maskedDiffusionCommit tokens");
        near(unmask.downloadInt32(), [0, 5], 0, "maskedDiffusionCommit unmaskStep");
        throws(() => tensor.maskedDiffusionCommit(pred, i32(1, 1, [1]), 2, 5, tokens, unmask), "idx holds 1", "maskedDiffusionCommit k past idx");
        throws(() => tensor.maskedDiffusionCommit(pred, i32(1, 1, [1]), 1, 5, i32(1, 1, [2]), unmask), "elements", "maskedDiffusionCommit short tokens");
    )JS");

    // ---- dense: linearForwardBatchedEx / Fp16Act, matmulAbt ----------------------
    f += block("test_api_extra.dense", R"JS(
        const dt = onCpu ? "fp32" : "fp16";
        const W = mk(2, 3, [1, 0, 0, 0, 1, 1], dt);
        const b = mk(2, 1, [1, -1], dt);
        const X = mk(1, 3, [1, 2, 3], dt);
        const Y = out();
        tensor.linearForwardBatchedEx(W, b, X, tensor.LinearActivation.none, tensor.LinearEpilogue.store, null, Y);
        near(Y.download(), [2, 4], 1e-2, "linearForwardBatchedEx store");
        tensor.linearForwardBatchedEx(W, b, X, 0, tensor.LinearEpilogue.accumulate, null, Y);
        near(Y.download(), [4, 8], 1e-2, "linearForwardBatchedEx accumulate");
        throws(() => tensor.linearForwardBatchedEx(W, b, mk(1, 2, [1, 2], dt), 0, 0, null, Y), "columns", "linearForwardBatchedEx narrow X");
        throws(() => tensor.linearForwardBatchedEx(W, mk(1, 1, [1], dt), X, 0, 0, null, Y), "bias holds 1", "linearForwardBatchedEx short bias");

        if (!onCpu) {
            // relu([1, -2]) through the fused epilogue.
            const W2 = mk(2, 3, [1, 0, 0, 0, -1, 0], "fp16");
            tensor.linearForwardBatchedFp16Act(W2, null, mk(1, 3, [1, 2, 3], "fp16"), tensor.LinearActivation.relu, Y);
            near(Y.download(), [1, 0], 1e-2, "linearForwardBatchedFp16Act");
            throws(() => tensor.linearForwardBatchedFp16Act(W2, mk(1, 1, [1], "fp16"), mk(1, 3, [1, 2, 3], "fp16"), 1, Y),
                   "bias holds 1", "linearForwardBatchedFp16Act short bias");
        }

        // C = A @ B^T with B = I.
        const A = mk(2, 2, [1, 2, 3, 4], "fp16");
        const B = mk(2, 2, [1, 0, 0, 1], "fp16");
        const C = mk(2, 2, [0, 0, 0, 0], "fp16");
        tensor.matmulAbt(A, B, C, 1, 2, 2, 2, 4, 4, 4, null, 0);
        near(C.download(), [1, 2, 3, 4], 1e-2, "matmulAbt");
        // Two slices of A with bias [10, 20]: slice 1 is A's second copy.
        const A2 = mk(4, 2, [1, 2, 3, 4, 5, 6, 7, 8], "fp16");
        const C2 = mk(4, 2, [0, 0, 0, 0, 0, 0, 0, 0], "fp16");
        tensor.matmulAbt(A2, B, C2, 2, 2, 2, 2, 4, 0, 4, mk(2, 1, [10, 20], "fp16"), 0);
        near(C2.download(), [11, 22, 13, 24, 15, 26, 17, 28], 1e-1, "matmulAbt batched + bias");
        throws(() => tensor.matmulAbt(A, B, C, 2, 2, 2, 2, 4, 4, 4, null, 0), "A holds 4", "matmulAbt batch past A");
        throws(() => tensor.matmulAbt(A, B, mk(1, 2, [0, 0], "fp16"), 1, 2, 2, 2, 4, 4, 4, null, 0), "C holds 2", "matmulAbt short C");
    )JS");

    // ---- LSTM training forward + BPTT -------------------------------------------
    f += block("test_api_extra.lstm", R"JS(
        // I = H = 1, T = B = 1, every gate pre-activation z = 1 (x = 1).
        const X = mk(1, 1, [1]);
        const Wih = mk(4, 1, [1, 1, 1, 1]);
        const Whh = mk(4, 1, [0, 0, 0, 0]);
        const Y = out(), gates = out(), C = out(), hT = out(), cT = out();
        tensor.lstmForwardTrain(X, Wih, Whh, null, null, null, null, 1, 1, Y, gates, C, hT, cT);
        const s = 1 / (1 + Math.exp(-1)), g = Math.tanh(1);
        const c = s * g, h = s * Math.tanh(c);
        near(gates.download(), [s, s, g, s], 1e-5, "lstm gates");
        near(C.download(), [c], 1e-5, "lstm C");
        near(Y.download(), [h], 1e-5, "lstm Y");
        near(hT.download(), [h], 1e-5, "lstm hT");
        near(cT.download(), [c], 1e-5, "lstm cT");

        // dY = 1 on h.
        const tc = Math.tanh(c);
        const dzo = tc * s * (1 - s);
        const dc = s * (1 - tc * tc);
        const dzi = dc * g * s * (1 - s);
        const dzg = dc * s * (1 - g * g);
        const dX = out(), dWih = mk(4, 1, [0, 0, 0, 0]), dWhh = mk(4, 1, [0, 0, 0, 0]), dbih = mk(4, 1, [0, 0, 0, 0]);
        tensor.lstmBackward(X, Wih, Whh, null, null, Y, gates, C, mk(1, 1, [1]), 1, 1,
                            dX, dWih, dWhh, dbih, null, null, null);
        near(dWih.download(), [dzi, 0, dzg, dzo], 1e-5, "lstm dW_ih");
        near(dbih.download(), [dzi, 0, dzg, dzo], 1e-5, "lstm db_ih");
        near(dX.download(), [dzi + dzg + dzo], 1e-5, "lstm dX");

        throws(() => tensor.lstmBackward(X, Wih, Whh, null, null, Y, mk(1, 2, [0, 0]), C, mk(1, 1, [1]), 1, 1,
                                         dX, dWih, dWhh, null, null, null, null), "gates holds 2", "lstmBackward short gates");
        throws(() => tensor.lstmBackward(X, Wih, Whh, null, null, Y, gates, C, mk(1, 1, [1]), 1, 1,
                                         dX, mk(2, 1, [0, 0]), dWhh, null, null, null, null), "dW_ih holds 2", "lstmBackward short dW_ih");
        throws(() => tensor.lstmForwardTrain(X, Wih, Whh, null, null, null, null, 2, 1, Y, gates, C, null, null),
               "X holds 1", "lstmForwardTrain T past X");
        throws(() => tensor.lstmForwardTrain(X, Wih, Whh, mk(2, 1, [0, 0]), null, null, null, 1, 1, Y, gates, C, null, null),
               "b_ih holds 2", "lstmForwardTrain short b_ih");
    )JS");

    // ---- attention: GQA, packed QKV, rel-pos bias --------------------------------
    f += block("test_api_extra.attention", R"JS(
        // Two query heads over one KV head, head_dim 1. Q = 0: uniform weights.
        const Q = mk(1, 2, [0, 0]);
        const K = mk(2, 1, [0, 0]), V = mk(2, 1, [2, 6]);
        const O = out();
        tensor.flashAttentionGqaForward(Q, K, V, null, 2, 1, false, O);
        near(O.download(), [4, 4], 1e-3, "flashAttentionGqaForward");
        tensor.flashAttentionGqaForward(Q, K, V, mk(2, 1, [1, 0]), 2, 1, false, O);
        near(O.download(), [2, 2], 1e-3, "flashAttentionGqaForward masked");
        throws(() => tensor.flashAttentionGqaForward(Q, K, V, mk(1, 1, [1]), 2, 1, false, O), "mask holds 1", "flashAttentionGqaForward short mask");
        throws(() => tensor.flashAttentionGqaForward(Q, mk(2, 2, [0, 0, 0, 0]), V, null, 2, 1, false, O), "K and V", "flashAttentionGqaForward wide K");

        // Packed QKV: sequences rows [0,2) and [2,3), one head of dim 2, Q = K = 0.
        const QKV = mk(3, 6, [0, 0, 0, 0, 1, 2,
                              0, 0, 0, 0, 3, 4,
                              0, 0, 0, 0, 5, 6]);
        const bounds = i32(3, 2, [0, 2, 0, 2, 2, 3]);
        tensor.flashAttentionPackedQkvForward(QKV, bounds, 1, 0, O);
        near(O.download(), [2, 3, 2, 3, 5, 6], 1e-3, "flashAttentionPackedQkvForward");
        const dQKV = out();
        tensor.flashAttentionPackedQkvBackward(QKV, mk(3, 2, [1, 1, 1, 1, 1, 1]), bounds, 1, 0, dQKV);
        near(dQKV.download(), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 1e-3, "flashAttentionPackedQkvBackward");
        throws(() => tensor.flashAttentionPackedQkvForward(QKV, i32(3, 2, [0, 2, 0, 2, 0, 2]), 1, 0, O), "contain the row", "packed bounds miss row");
        throws(() => tensor.flashAttentionPackedQkvForward(QKV, i32(3, 2, [0, 2, 0, 2, 2, 4]), 1, 0, O), "seqBounds row 2", "packed bounds past L");
        throws(() => tensor.flashAttentionPackedQkvBackward(QKV, mk(3, 1, [1, 1, 1]), bounds, 1, 0, dQKV), "dO must be", "packed backward narrow dO");

        // Transformer-XL bias, T = 2, one head of dim 1: Bias[q,k] = Qv[q]*Pk[1-q+k].
        const Bias = out();
        tensor.relPosBiasXlForward(mk(2, 1, [1, 2]), mk(3, 1, [10, 20, 30]), 1, 1, Bias);
        near(Bias.download(), [20, 30, 20, 40], 1e-3, "relPosBiasXlForward");
        throws(() => tensor.relPosBiasXlForward(mk(2, 1, [1, 2]), mk(2, 1, [10, 20]), 1, 1, Bias), "Pk must be", "relPosBiasXlForward short Pk");
    )JS");

    // ---- RoPE variants ---------------------------------------------------------------
    f += block("test_api_extra.rope", R"JS(
        // Per-head tables: head 0 rotates by 90 degrees, head 1 by 0.
        const X = mk(1, 4, [1, 0, 0, 1]);
        const Y = out();
        tensor.ropeApplyPerhead(X, mk(2, 1, [0, 1]), mk(2, 1, [1, 0]), 2, 2, Y);
        near(Y.download(), [0, 1, 0, 1], 1e-5, "ropeApplyPerhead");
        throws(() => tensor.ropeApplyPerhead(X, mk(1, 1, [0]), mk(2, 1, [1, 0]), 2, 2, Y), "cosTbl holds 1", "ropeApplyPerhead short cos");

        // Packed QKV in place: row 0 at position 1 (90 degrees), row 1 at 0.
        const QKV = mk(2, 6, [1, 0, 1, 0, 7, 8,
                              1, 0, 1, 0, 7, 8]);
        const cos = mk(2, 1, [1, 0]), sin = mk(2, 1, [0, 1]);
        tensor.ropeQkvPackedInplace(QKV, cos, sin, i32(2, 1, [1, 0]), 1, 2);
        near(QKV.download(), [0, 1, 0, 1, 7, 8, 1, 0, 1, 0, 7, 8], 1e-5, "ropeQkvPackedInplace");
        throws(() => tensor.ropeQkvPackedInplace(QKV, cos, sin, i32(2, 1, [2, 0]), 1, 2), "outside [0, 2)", "ropeQkvPackedInplace pos past table");
        throws(() => tensor.ropeQkvPackedInplace(QKV, cos, sin, i32(1, 1, [0]), 1, 2), "pos must be", "ropeQkvPackedInplace short pos");
    )JS");

    // ---- StyleGAN3-R: bias_act, upfirdn2d, modulated conv, filtered lrelu -----------
    f += block("test_api_extra.stylegan", R"JS(
        const r2 = Math.SQRT2;
        const X = mk(1, 2, [-1, 1]);
        const b = mk(2, 1, [0.5, 0.5]);
        const Y = out();
        tensor.biasActForward(X, b, 1, 2, 1, 1, 0.2, r2, -1, Y);
        near(Y.download(), [-0.1 * r2, 1.5 * r2], 1e-5, "biasActForward");
        const dX = out(), dB = mk(2, 1, [0, 0]);
        tensor.biasActBackward(mk(1, 2, [1, 1]), X, b, 1, 2, 1, 1, 0.2, r2, -1, dX, dB);
        near(dX.download(), [0.2 * r2, r2], 1e-5, "biasActBackward dX");
        near(dB.download(), [0.2 * r2, r2], 1e-5, "biasActBackward dB");
        throws(() => tensor.biasActForward(X, mk(1, 1, [0.5]), 1, 2, 1, 1, 0.2, r2, -1, Y), "b holds 1", "biasActForward short b");
        throws(() => tensor.biasActBackward(mk(1, 2, [1, 1]), X, b, 1, 2, 1, 1, 0.2, r2, -1, dX, mk(1, 1, [0])), "dB holds 1", "biasActBackward short dB");
        throws(() => tensor.biasActForward(X, null, 1, 2, 2, 1, 0.2, r2, -1, Y), "X holds 2", "biasActForward HW past X");

        // upfirdn2d: 2x zero-insert upsample along W with a 1x1 unit filter.
        const f1 = mk(1, 1, [1]);
        const U = out();
        tensor.upfirdn2dForward(mk(1, 2, [3, 5]), f1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, false, 1, U);
        near(U.download(), [3, 0, 5, 0], 1e-5, "upfirdn2dForward");
        const dU = out();
        tensor.upfirdn2dBackward(mk(1, 4, [1, 1, 1, 1]), f1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, false, 1, dU);
        near(dU.download(), [1, 1], 1e-5, "upfirdn2dBackward");
        // A 2-tap filter, correlate (flip = true): Y[w] = X[w] + 2*X[w+1].
        tensor.upfirdn2dForward(mk(1, 3, [1, 2, 3]), mk(1, 2, [1, 2]), 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, true, 1, U);
        near(U.download(), [5, 8], 1e-5, "upfirdn2dForward 2-tap");
        throws(() => tensor.upfirdn2dForward(mk(1, 1, [3]), f1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, false, 1, U), "X holds 1", "upfirdn2dForward short X");
        throws(() => tensor.upfirdn2dBackward(mk(1, 2, [1, 1]), f1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, false, 1, dU), "dY holds 2", "upfirdn2dBackward short dY");

        // Modulated 1x1 conv, no demodulation: Y = X*W*s = 2*3*4.
        const Xm = mk(1, 1, [2]), Wm = mk(1, 1, [3]), sm = mk(1, 1, [4]);
        const dcoef = out(), Ym = out();
        tensor.modulatedConv2dForward(Xm, Wm, sm, 1, 1, 1, 1, 1, 1, 1, 0, 0, false, 0, dcoef, Ym);
        near(Ym.download(), [24], 1e-4, "modulatedConv2dForward");
        near(dcoef.download(), [1], 1e-6, "modulatedConv2dForward dcoef");
        const dXm = out(), dWm = mk(1, 1, [0]), dsm = out();
        tensor.modulatedConv2dBackward(Xm, Wm, sm, dcoef, mk(1, 1, [1]), 1, 1, 1, 1, 1, 1, 1, 0, 0, false, 0, dXm, dWm, dsm);
        near(dXm.download(), [12], 1e-4, "modulatedConv2dBackward dX");
        near(dWm.download(), [8], 1e-4, "modulatedConv2dBackward dW");
        near(dsm.download(), [6], 1e-4, "modulatedConv2dBackward ds");
        // Demodulated: w'' = w'/|w'| so Y = X = 2.
        tensor.modulatedConv2dForward(Xm, Wm, sm, 1, 1, 1, 1, 1, 1, 1, 0, 0, true, 0, dcoef, Ym);
        near(Ym.download(), [2], 1e-4, "modulatedConv2dForward demodulated");
        throws(() => tensor.modulatedConv2dForward(Xm, Wm, mk(1, 1, [4]), 2, 1, 1, 1, 1, 1, 1, 0, 0, false, 0, dcoef, Ym), "X holds 1", "modulatedConv2dForward N past X");
        throws(() => tensor.modulatedConv2dForward(Xm, Wm, sm, 1, 1, 1, 1, 2, 1, 1, 0, 0, false, 0, dcoef, Ym), "W holds 1", "modulatedConv2dForward C_out past W");
        throws(() => tensor.modulatedConv2dBackward(Xm, Wm, sm, tensor.createTensor(0, 1), mk(1, 1, [1]), 1, 1, 1, 1, 1, 1, 1, 0, 0, false, 0, dXm, dWm, dsm),
               "dcoef holds 0", "modulatedConv2dBackward empty dcoef");
        throws(() => tensor.modulatedConv2dBackward(Xm, Wm, sm, dcoef, mk(1, 1, [1]), 1, 1, 1, 1, 1, 1, 1, 1, 0, false, 0, dXm, dWm, dsm),
               "dY holds 1", "modulatedConv2dBackward padded dY");

        // filtered_lrelu with unit filters, up = down = 1: Y = lrelu(X) (gain 1).
        const Xf = mk(1, 2, [-1, 2]);
        const upBuf = out(), actBuf = out(), Yf = out();
        tensor.filteredLreluForward(Xf, f1, f1, null, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1, 0.2, -1, upBuf, actBuf, Yf);
        near(Yf.download(), [-0.2, 2], 1e-5, "filteredLreluForward");
        const dXf = out();
        tensor.filteredLreluBackward(mk(1, 2, [1, 1]), Xf, f1, f1, null, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1, 0.2, -1, null, dXf, null);
        near(dXf.download(), [0.2, 1], 1e-5, "filteredLreluBackward");
        throws(() => tensor.filteredLreluBackward(mk(1, 1, [1]), Xf, f1, f1, null, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1, 0.2, -1, null, dXf, null),
               "dY holds 1", "filteredLreluBackward short dY");
        throws(() => tensor.filteredLreluForward(Xf, f1, f1, null, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 1, 0.2, -1, upBuf, actBuf, Yf),
               "X holds 2", "filteredLreluForward C past X");
    )JS");

    // ---- deformable conv, pixel shuffle, unpatchify ---------------------------------
    f += block("test_api_extra.spatial", R"JS(
        // 1x1 deformable conv, weight 2, bias 1, over a 2x2 image.
        const X = mk(1, 4, [1, 2, 3, 4]);
        const Wt = mk(1, 1, [2]), bias = mk(1, 1, [1]);
        const Y = out();
        tensor.deformConv2dForward(X, mk(1, 8, [0, 0, 0, 0, 0, 0, 0, 0]), null, Wt, bias,
                                   1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, Y);
        near(Y.download(), [3, 5, 7, 9], 1e-4, "deformConv2dForward");
        // Shift every tap one row down: row 1 samples outside (zero padding).
        tensor.deformConv2dForward(X, mk(1, 8, [1, 1, 1, 1, 0, 0, 0, 0]), null, Wt, bias,
                                   1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, Y);
        near(Y.download(), [7, 9, 1, 1], 1e-4, "deformConv2dForward offset");
        // A 0.5 modulator halves the conv term.
        tensor.deformConv2dForward(X, mk(1, 8, [0, 0, 0, 0, 0, 0, 0, 0]), mk(1, 4, [0.5, 0.5, 0.5, 0.5]), Wt, bias,
                                   1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, Y);
        near(Y.download(), [2, 3, 4, 5], 1e-4, "deformConv2dForward mask");
        throws(() => tensor.deformConv2dForward(X, mk(1, 7, [0, 0, 0, 0, 0, 0, 0]), null, Wt, bias,
                                                1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, Y), "offset holds 7", "deformConv2dForward short offset");
        throws(() => tensor.deformConv2dForward(X, mk(1, 8, [0, 0, 0, 0, 0, 0, 0, 0]), mk(1, 2, [1, 1]), Wt, bias,
                                                1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, Y), "mask holds 2", "deformConv2dForward short mask");

        // Pixel shuffle: C_in = 4 -> C_out = 1 is a plain depth-to-space.
        tensor.pixelShuffleUpsample2xForward(mk(1, 4, [1, 2, 3, 4]), 1, 4, 1, 1, 1, Y);
        near(Y.download(), [1, 2, 3, 4], 1e-5, "pixelShuffleUpsample2xForward shuffle");
        // C_in = C_out: a 2x nearest upsample.
        tensor.pixelShuffleUpsample2xForward(mk(1, 1, [7]), 1, 1, 1, 1, 1, Y);
        near(Y.download(), [7, 7, 7, 7], 1e-5, "pixelShuffleUpsample2xForward nearest");
        throws(() => tensor.pixelShuffleUpsample2xForward(mk(1, 2, [1, 2]), 1, 4, 1, 1, 1, Y), "X holds 2", "pixelShuffleUpsample2xForward short X");

        // Unpatchify one 2x2 patch of 2 channels, keeping channel 0.
        const tokens = mk(1, 8, [0, 1, 2, 3, 4, 5, 6, 7]);
        tensor.patchUnpackForward(tokens, 1, 1, 2, 2, 1, false, Y);
        near(Y.download(), [0, 2, 4, 6], 1e-5, "patchUnpackForward block-major");
        tensor.patchUnpackForward(tokens, 1, 1, 2, 2, 1, true, Y);
        near(Y.download(), [0, 1, 2, 3], 1e-5, "patchUnpackForward channel-major");
        throws(() => tensor.patchUnpackForward(mk(1, 6, [0, 1, 2, 3, 4, 5]), 1, 1, 2, 2, 1, false, Y), "tokens must be", "patchUnpackForward narrow tokens");
    )JS");

    return f;
}
