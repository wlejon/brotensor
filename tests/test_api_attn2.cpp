// JS-API checks for the second attention family (src/api/js/tensor_attn2.js):
// decomposed 2D rel-pos self-attention (global + windowed), packed
// variable-length flash attention (forward + backward), the gated delta rule
// (chunked + step) and M-RoPE.
//
// Every op here has an FP32 path on both the CPU backend and CUDA/Metal
// (gated_delta_rule is FP32-only everywhere; the varlen / rel-pos / M-RoPE GPU
// kernels take FP32 alongside FP16/BF16), so none of these blocks needs a
// backend gate — the numbers below are the same on every backend.
//
// The INT32 streams (cu_seqlens, M-RoPE position ids) keep the old binding's
// "GpuTensor viewed as INT32 storage" convention, so the helper builds them by
// uploading the little-endian bytes through uploadInt8 — the storage lands on
// the default device, which is exactly what the ops want (device pointer on
// CUDA/Metal, host pointer on CPU).

#include "api_test_helpers.h"

int run_api_attn2_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- decomposed 2D relative-position self-attention ---------------------
    f += runJs("test_api_attn2_rel_pos", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-3);
        const finite = (t, label) => { const d = t.download(); for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite"); return d; };

        // 2x2 token grid, D = 4, 2 heads => head_dim = 2.
        const D = 4, nH = 2, gh = 2, gw = 2, L = gh * gw;
        const X = tensor.createTensor(L, D);
        X.upload([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        const Z = tensor.createTensor(D, D);                       // zeros
        const I = tensor.createTensor(D, D);
        I.upload([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const relH = tensor.createTensor(2 * gh - 1, D / nH);       // (3, 2)
        const relW = tensor.createTensor(2 * gw - 1, D / nH);
        const O = tensor.createTensor(1, 1);

        // Wq = Wk = 0 with no qkv biases: the projected query is zero, so both
        // the QK dot and the decomposed bias vanish and every token attends
        // uniformly. With Wv = Wo = I that makes every output row the column
        // mean of X.
        tensor.selfAttentionDecomposedRelPosForward(X, Z, null, Z, null, I, null, I, null, relH, relW, nH, gh, gw, 0.5, O);
        if (O.rows !== L || O.cols !== D) throw new Error("decomposedRelPos O shape: " + O.rows + "x" + O.cols);
        const od = finite(O, "selfAttentionDecomposedRelPosForward O");
        const mean = [7, 8, 9, 10];
        for (let r = 0; r < L; r++) {
            for (let c = 0; c < D; c++) {
                if (!same(od[r * D + c], mean[c])) throw new Error("decomposedRelPos uniform: " + Array.from(od));
            }
        }

        // Same inputs through the windowed form with window == grid: one
        // window, so it must reproduce the global result exactly.
        const Ow = tensor.createTensor(1, 1);
        tensor.selfAttentionDecomposedRelPosWindowedForward(X, Z, null, Z, null, I, null, I, null, relH, relW, nH, gh, gw, 2, 0.5, Ow);
        const owd = finite(Ow, "selfAttentionDecomposedRelPosWindowedForward O");
        for (let i = 0; i < owd.length; i++) {
            if (!same(owd[i], od[i])) throw new Error("windowed(window=grid) != global: " + Array.from(owd));
        }

        // window = 1: every token is its own window, so the softmax is over a
        // single key and the output is V = X @ Wv = X, whatever the weights or
        // the bias tables are. rel-pos tables are window-sized: (2*1-1, head_dim).
        const relH1 = tensor.createTensor(1, D / nH); relH1.upload([0.5, -0.25]);
        const relW1 = tensor.createTensor(1, D / nH); relW1.upload([0.125, 0.75]);
        tensor.selfAttentionDecomposedRelPosWindowedForward(X, I, null, I, null, I, null, I, null, relH1, relW1, nH, gh, gw, 1, 0.5, Ow);
        const ow1 = finite(Ow, "windowed(window=1) O");
        const xd = X.download();
        for (let i = 0; i < ow1.length; i++) {
            if (!same(ow1[i], xd[i], 1e-3)) throw new Error("windowed(window=1) != X: " + Array.from(ow1));
        }

        // The optional projection biases: with Wq = 0 the query is bq alone, so
        // the decomposed bias is live. Finiteness + shape only.
        const bq = tensor.createTensor(D, 1); bq.upload([0.1, 0.2, 0.3, 0.4]);
        const bk = tensor.createTensor(D, 1); bk.upload([0.05, -0.05, 0.1, -0.1]);
        const bv = tensor.createTensor(D, 1); bv.upload([0, 0, 0, 0]);
        const bo = tensor.createTensor(D, 1); bo.upload([1, 1, 1, 1]);
        relH.upload([0.1, -0.2, 0.3, 0.4, -0.5, 0.6]);
        relW.upload([0.2, 0.1, -0.3, 0.25, 0.5, -0.1]);
        tensor.selfAttentionDecomposedRelPosForward(X, Z, bq, Z, bk, I, bv, I, bo, relH, relW, nH, gh, gw, 0.5, O);
        const ob = finite(O, "decomposedRelPos with biases O");
        if (O.rows !== L || O.cols !== D) throw new Error("decomposedRelPos biased O shape");
        // bo = 1 is added to every output row, so the biased result cannot be
        // the unbiased one.
        let moved = false;
        for (let i = 0; i < ob.length; i++) if (!same(ob[i], od[i])) moved = true;
        if (!moved) throw new Error("decomposedRelPos biases had no effect");

        // Argument validation happens in JS, before the native.
        let threw = false;
        try { tensor.selfAttentionDecomposedRelPosForward(X, Z, 5, Z, null, I, null, I, null, relH, relW, nH, gh, gw, 0.5, O); }
        catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("decomposedRelPos with a bad bq did not throw");
        return "OK";
    })();
    )JS");

    // ---- packed variable-length flash attention -----------------------------
    f += runJs("test_api_attn2_varlen", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-3);
        const finite = (t, label) => { const d = t.download(); for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite"); return d; };
        // A GpuTensor whose raw storage is a little-endian INT32 buffer.
        const i32 = (arr) => {
            const b = new Int8Array(arr.length * 4);
            for (let i = 0; i < arr.length; i++) {
                const v = arr[i] | 0;
                b[4 * i] = v & 0xff;
                b[4 * i + 1] = (v >> 8) & 0xff;
                b[4 * i + 2] = (v >> 16) & 0xff;
                b[4 * i + 3] = (v >> 24) & 0xff;
            }
            const t = tensor.createTensor(arr.length * 4, 1, "int8");
            t.uploadInt8(b);
            return t;
        };

        // Two packed sequences: rows [0,2) and [2,3). 1 head, head_dim 2.
        const nH = 1, hd = 2, D = nH * hd, batch = 2;
        const Q = tensor.createTensor(3, D);           // zeros
        const K = tensor.createTensor(3, D);           // zeros => uniform softmax
        const V = tensor.createTensor(3, D); V.upload([1, 2, 3, 4, 5, 6]);
        const cuQ = i32([0, 2, 3]), cuK = i32([0, 2, 3]);
        const O = tensor.createTensor(1, 1);
        tensor.flashAttentionVarlenForward(Q, K, V, cuQ, cuK, batch, 2, 2, nH, hd, false, O);
        if (O.rows !== 3 || O.cols !== D) throw new Error("varlen O shape: " + O.rows + "x" + O.cols);
        const od = finite(O, "flashAttentionVarlenForward O");
        // Sequence 0 averages V rows 0,1; sequence 1 sees only V row 2 — and
        // nothing crosses the boundary.
        const want = [2, 3, 2, 3, 5, 6];
        for (let i = 0; i < want.length; i++) {
            if (!same(od[i], want[i], 1e-2)) throw new Error("varlen forward: " + Array.from(od));
        }

        // Backward with dO = 1: uniform probabilities put exactly one unit of
        // upstream gradient on each V row, and dQ / dK vanish because K and Q
        // are zero.
        const dO = tensor.createTensor(3, D); dO.upload([1, 1, 1, 1, 1, 1]);
        const dQ = tensor.createTensor(1, 1), dK = tensor.createTensor(1, 1), dV = tensor.createTensor(1, 1);
        tensor.flashAttentionVarlenBackward(Q, K, V, O, dO, cuQ, cuK, batch, 2, 2, nH, hd, false, dQ, dK, dV);
        if (dV.rows !== 3 || dV.cols !== D || dQ.rows !== 3 || dQ.cols !== D) throw new Error("varlen backward shapes");
        const dvd = finite(dV, "flashAttentionVarlenBackward dV");
        const dqd = finite(dQ, "flashAttentionVarlenBackward dQ");
        const dkd = finite(dK, "flashAttentionVarlenBackward dK");
        for (let i = 0; i < dvd.length; i++) {
            if (!same(dvd[i], 1, 1e-2)) throw new Error("varlen backward dV: " + Array.from(dvd));
            if (!same(dqd[i], 0, 1e-2) || !same(dkd[i], 0, 1e-2)) throw new Error("varlen backward dQ/dK: " + Array.from(dqd) + " / " + Array.from(dkd));
        }

        // Causal within a sequence: query 0 of sequence 0 sees only key 0.
        tensor.flashAttentionVarlenForward(Q, K, V, cuQ, cuK, batch, 2, 2, nH, hd, true, O);
        const oc = finite(O, "causal varlen O");
        if (!same(oc[0], 1, 1e-2) || !same(oc[1], 2, 1e-2)) throw new Error("causal varlen row 0: " + Array.from(oc));
        if (!same(oc[4], 5, 1e-2) || !same(oc[5], 6, 1e-2)) throw new Error("causal varlen row 2: " + Array.from(oc));

        let threw = false;
        try { tensor.flashAttentionVarlenForward(Q, K, V, 7, cuK, batch, 2, 2, nH, hd, false, O); }
        catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("varlen forward with a bad cuSeqQ did not throw");
        return "OK";
    })();
    )JS");

    // ---- gated delta rule ---------------------------------------------------
    f += runJs("test_api_attn2_gated_delta", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-3);
        const finite = (t, label) => { const d = t.download(); for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite"); return d; };
        const mk = (r, c, vals) => { const t = tensor.createTensor(r, c); if (vals) t.upload(vals); return t; };

        // One head, d_k = d_v = 1, one token: the recurrence in closed form.
        //   alpha = exp(-softplus(a) * exp(logA)),  b = sigmoid(betaRaw)
        //   S = alpha*s0 + b*(v - alpha*s0*k)*k,    o = S*q
        const a = 0.5, braw = 0.3, la = -0.1, s0 = 0.7, q = 1.3, k = 0.9, v = 2.1;
        const alpha = Math.exp(-Math.log(1 + Math.exp(a)) * Math.exp(la));
        const bs = 1 / (1 + Math.exp(-braw));
        const Spre = alpha * s0;
        const S = Spre + bs * (v - Spre * k) * k;
        const o = S * q;

        const O1 = tensor.createTensor(1, 1);
        const st1 = mk(1, 1, [s0]);
        tensor.gatedDeltaRuleChunked(mk(1, 1, [q]), mk(1, 1, [k]), mk(1, 1, [v]),
                                     mk(1, 1, [a]), mk(1, 1, [braw]), mk(1, 1, [la]),
                                     1, 1, 1, st1, O1);
        if (O1.rows !== 1 || O1.cols !== 1) throw new Error("gatedDeltaRuleChunked O shape");
        const o1 = finite(O1, "gatedDeltaRuleChunked O");
        if (!same(o1[0], o, 1e-4)) throw new Error("gatedDeltaRuleChunked closed form: " + o1[0] + " want " + o);
        const s1 = finite(st1, "gatedDeltaRuleChunked state");
        if (!same(s1[0], S, 1e-4)) throw new Error("gatedDeltaRuleChunked state: " + s1[0] + " want " + S);

        // The step form is the same rule, so a single token reproduces it.
        const O1s = tensor.createTensor(1, 1);
        const st1s = mk(1, 1, [s0]);
        tensor.gatedDeltaRuleStep(mk(1, 1, [q]), mk(1, 1, [k]), mk(1, 1, [v]),
                                  mk(1, 1, [a]), mk(1, 1, [braw]), mk(1, 1, [la]),
                                  1, 1, 1, st1s, O1s);
        const o1s = finite(O1s, "gatedDeltaRuleStep O");
        if (!same(o1s[0], o, 1e-4)) throw new Error("gatedDeltaRuleStep closed form: " + o1s[0]);

        // Two tokens, d_k = d_v = 2: the chunked prefill must equal two
        // sequential steps over the same state.
        const dk = 2, dv = 2, L = 2;
        const Qv = [0.5, -0.3, 0.2, 0.9], Kv = [1.0, 0.4, -0.2, 0.7], Vv = [0.3, 1.2, -0.8, 0.5];
        const Av = [0.2, -0.4], Bv = [0.1, 0.6], LAv = [-0.2];
        const stC = mk(1, dk * dv);                       // zeros
        const OC = tensor.createTensor(1, 1);
        tensor.gatedDeltaRuleChunked(mk(L, dk, Qv), mk(L, dk, Kv), mk(L, dv, Vv),
                                     mk(L, 1, Av), mk(L, 1, Bv), mk(1, 1, LAv),
                                     1, dk, dv, stC, OC);
        if (OC.rows !== L || OC.cols !== dv) throw new Error("gatedDeltaRuleChunked O shape (L=2): " + OC.rows + "x" + OC.cols);
        const ocd = finite(OC, "gatedDeltaRuleChunked O (L=2)");

        const stS = mk(1, dk * dv);                       // zeros
        const rows = [];
        for (let t = 0; t < L; t++) {
            const Os = tensor.createTensor(1, 1);
            tensor.gatedDeltaRuleStep(mk(1, dk, Qv.slice(t * dk, t * dk + dk)),
                                      mk(1, dk, Kv.slice(t * dk, t * dk + dk)),
                                      mk(1, dv, Vv.slice(t * dv, t * dv + dv)),
                                      mk(1, 1, [Av[t]]), mk(1, 1, [Bv[t]]), mk(1, 1, LAv),
                                      1, dk, dv, stS, Os);
            const d = finite(Os, "gatedDeltaRuleStep O");
            for (let c = 0; c < dv; c++) rows.push(d[c]);
        }
        for (let i = 0; i < rows.length; i++) {
            if (!same(rows[i], ocd[i], 1e-4)) throw new Error("chunked != sequential steps: " + Array.from(ocd) + " vs " + rows);
        }
        const scd = finite(stC, "chunked state"), ssd = finite(stS, "stepped state");
        for (let i = 0; i < scd.length; i++) {
            if (!same(scd[i], ssd[i], 1e-4)) throw new Error("chunked state != stepped state: " + Array.from(scd) + " vs " + Array.from(ssd));
        }

        let threw = false;
        try { tensor.gatedDeltaRuleStep(null, mk(1, 1, [k]), mk(1, 1, [v]), mk(1, 1, [a]), mk(1, 1, [braw]), mk(1, 1, [la]), 1, 1, 1, st1s, O1s); }
        catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("gatedDeltaRuleStep(null, ...) did not throw");
        return "OK";
    })();
    )JS");

    // ---- M-RoPE -------------------------------------------------------------
    f += runJs("test_api_attn2_mrope", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-3);
        const finite = (t, label) => { const d = t.download(); for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite"); return d; };
        const i32 = (arr) => {
            const b = new Int8Array(arr.length * 4);
            for (let i = 0; i < arr.length; i++) {
                const v = arr[i] | 0;
                b[4 * i] = v & 0xff;
                b[4 * i + 1] = (v >> 8) & 0xff;
                b[4 * i + 2] = (v >> 16) & 0xff;
                b[4 * i + 3] = (v >> 24) & 0xff;
            }
            const t = tensor.createTensor(arr.length * 4, 1, "int8");
            t.uploadInt8(b);
            return t;
        };
        const mk = (r, c, vals) => { const t = tensor.createTensor(r, c); if (vals) t.upload(vals); return t; };

        // Degenerate single-axis M-RoPE (d_h = d_w = 0, posT = 0..L-1) is
        // exactly ropeApply with the same tables.
        const L = 4, headDim = 4, nH = 2, dT = headDim / 2;
        const xs = [];
        for (let i = 0; i < L * nH * headDim; i++) xs.push(Math.sin(i * 0.7));
        const X = mk(L, nH * headDim, xs);
        const cosv = [], sinv = [];
        for (let p = 0; p < L; p++) {
            for (let i = 0; i < dT; i++) {
                const theta = p * Math.exp(-(2 * i) / headDim * Math.log(10000));
                cosv.push(Math.cos(theta));
                sinv.push(Math.sin(theta));
            }
        }
        const cosT = mk(L, dT, cosv), sinT = mk(L, dT, sinv);
        const dummy = mk(1, 1, [1]);
        const posT = i32([0, 1, 2, 3]);

        const Yref = tensor.createTensor(1, 1);
        tensor.ropeApply(X, cosT, sinT, headDim, nH, Yref);
        const yr = finite(Yref, "ropeApply Y");

        const Y = tensor.createTensor(1, 1);
        tensor.ropeApplyMrope(X, cosT, sinT, dummy, dummy, dummy, dummy, posT, null, null, headDim, nH, dT, 0, 0, Y);
        if (Y.rows !== L || Y.cols !== nH * headDim) throw new Error("mrope Y shape: " + Y.rows + "x" + Y.cols);
        const yd = finite(Y, "ropeApplyMrope Y");
        for (let i = 0; i < yd.length; i++) {
            if (!same(yd[i], yr[i], 1e-4)) throw new Error("mrope degenerate != ropeApply at " + i + ": " + yd[i] + " vs " + yr[i]);
        }

        // Two live axes: head_dim 4 = 2*(d_t=1 + d_h=1), one head, per-axis
        // position streams that disagree, checked against the pair rotation.
        const L2 = 2, hd2 = 4, dt2 = 1, dh2 = 1;
        const x2 = [0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.1, -0.2];
        const X2 = mk(L2, hd2, x2);
        const ct = [0.6, -0.8], st = [0.8, 0.6];       // (2,1) tables
        const ch = [0.0, 1.0], sh = [1.0, 0.0];
        const cosT2 = mk(2, dt2, ct), sinT2 = mk(2, dt2, st);
        const cosH2 = mk(2, dh2, ch), sinH2 = mk(2, dh2, sh);
        const pT = [0, 1], pH = [1, 0];
        const Y2 = tensor.createTensor(1, 1);
        tensor.ropeApplyMrope(X2, cosT2, sinT2, cosH2, sinH2, dummy, dummy,
                              i32(pT), i32(pH), null, hd2, 1, dt2, dh2, 0, Y2);
        const y2 = finite(Y2, "ropeApplyMrope Y (two axes)");
        for (let r = 0; r < L2; r++) {
            const base = r * hd2;
            const pairs = [[ct[pT[r]], st[pT[r]]], [ch[pH[r]], sh[pH[r]]]];
            for (let p = 0; p < 2; p++) {
                const c = pairs[p][0], s = pairs[p][1];
                const x0 = x2[base + 2 * p], x1 = x2[base + 2 * p + 1];
                if (!same(y2[base + 2 * p], x0 * c - x1 * s, 1e-4) ||
                    !same(y2[base + 2 * p + 1], x0 * s + x1 * c, 1e-4)) {
                    throw new Error("mrope two-axis rotation at row " + r + " pair " + p + ": " + Array.from(y2));
                }
            }
        }

        let threw = false;
        try { tensor.ropeApplyMrope(X, cosT, sinT, dummy, dummy, dummy, dummy, 3, null, null, headDim, nH, dT, 0, 0, Y); }
        catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("ropeApplyMrope with a bad posT did not throw");
        return "OK";
    })();
    )JS");

    return f;
}
