// test_api_batched.cpp — JS-level checks for the restored "batched" group of
// bro.tensor (src/api/js/tensor_batched.js over
// src/api/native_tensor_batched.cpp).
//
// Everything here is FP32 and runs on any backend, except the
// linearForwardBatchedFp16 block: 16-bit storage is a GPU-only slot (the CPU
// backend registers none and the dispatcher throws), so that block is gated on
// tensor.backend !== "cpu".

#include "api_test_helpers.h"

int run_api_batched_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- batched dense + activation forwards / backwards -------------------
    f += runJs("test_api_batched.dense", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const near = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-4 : tol);
        const finite = (arr, what) => {
            for (let i = 0; i < arr.length; i++) {
                if (!isFinite(arr[i])) throw new Error(what + ": non-finite at " + i);
            }
        };

        // linearForwardBatched: W(2,3), bias(2,1), X_BD(2,3) -> Y_BD(2,2).
        //   W = [[1,2,3],[4,5,6]], bias = [1,-1]
        //   X[0] = [1,0,0] -> [1*1+1, 4*1-1] = [2, 3]
        //   X[1] = [0,1,1] -> [2+3+1, 5+6-1] = [6, 10]
        const W = tensor.createTensor(2, 3);
        const bias = tensor.createTensor(2, 1);
        const X = tensor.createTensor(2, 3);
        const Y = tensor.createTensor(2, 2);
        W.upload([1, 2, 3, 4, 5, 6]);
        bias.upload([1, -1]);
        X.upload([1, 0, 0, 0, 1, 1]);
        tensor.linearForwardBatched(W, bias, X, Y);
        if (Y.rows !== 2 || Y.cols !== 2) throw new Error("linearForwardBatched: shape " + Y.rows + "x" + Y.cols);
        const y = Y.download();
        finite(y, "linearForwardBatched");
        if (!near(y[0], 2) || !near(y[1], 3) || !near(y[2], 6) || !near(y[3], 10)) {
            throw new Error("linearForwardBatched: " + Array.from(y));
        }

        // reluForwardBatched over (2,2).
        const A = tensor.createTensor(2, 2);
        const R = tensor.createTensor(2, 2);
        A.upload([-1, 2, -3, 4]);
        tensor.reluForwardBatched(A, R);
        const r = R.download();
        if (r[0] !== 0 || r[1] !== 2 || r[2] !== 0 || r[3] !== 4) {
            throw new Error("reluForwardBatched: " + Array.from(r));
        }

        // tanhForwardBatched: tanh(0)=0, tanh(1)=0.7615942, tanh(-1)=-0.7615942.
        const Tin = tensor.createTensor(2, 2);
        const Tout = tensor.createTensor(2, 2);
        Tin.upload([0, 1, -1, 0.5]);
        tensor.tanhForwardBatched(Tin, Tout);
        const t = Tout.download();
        finite(t, "tanhForwardBatched");
        if (!near(t[0], 0) || !near(t[1], 0.7615942) || !near(t[2], -0.7615942) || !near(t[3], 0.4621172)) {
            throw new Error("tanhForwardBatched: " + Array.from(t));
        }

        // addInplaceBatched: R += A.
        tensor.addInplaceBatched(R, A);
        const ra = R.download();
        if (ra[0] !== -1 || ra[1] !== 4 || ra[2] !== -3 || ra[3] !== 8) {
            throw new Error("addInplaceBatched: " + Array.from(ra));
        }

        // linearBackwardBatched(W, X_BD, dY_BD, dX_BD, dW, dB) with dY = I(2).
        //   dX[b][j] = sum_o W[o,j]*dY[b][o]  -> dX = [[1,2,3],[4,5,6]]
        //   dW[o,j] += sum_b dY[b][o]*X[b][j] -> [[1,0,0],[0,1,1]]
        //   dB[o]   += sum_b dY[b][o]         -> [1,1]
        const dY = tensor.createTensor(2, 2);
        const dX = tensor.createTensor(2, 3);
        const dW = tensor.createTensor(2, 3);   // zeros: the op accumulates
        const dB = tensor.createTensor(2, 1);
        dY.upload([1, 0, 0, 1]);
        tensor.linearBackwardBatched(W, X, dY, dX, dW, dB);
        const dx = dX.download(), dw = dW.download(), db = dB.download();
        finite(dx, "linearBackwardBatched dX");
        if (!near(dx[0], 1) || !near(dx[1], 2) || !near(dx[2], 3) ||
            !near(dx[3], 4) || !near(dx[4], 5) || !near(dx[5], 6)) {
            throw new Error("linearBackwardBatched dX: " + Array.from(dx));
        }
        if (!near(dw[0], 1) || !near(dw[1], 0) || !near(dw[2], 0) ||
            !near(dw[3], 0) || !near(dw[4], 1) || !near(dw[5], 1)) {
            throw new Error("linearBackwardBatched dW: " + Array.from(dw));
        }
        if (!near(db[0], 1) || !near(db[1], 1)) {
            throw new Error("linearBackwardBatched dB: " + Array.from(db));
        }

        // reluBackwardBatched reads the forward INPUT: dX = dY*(X>0).
        const rbX = tensor.createTensor(2, 2);
        const rbDY = tensor.createTensor(2, 2);
        const rbDX = tensor.createTensor(2, 2);
        rbX.upload([-1, 2, -3, 4]);
        rbDY.upload([5, 7, 9, 11]);
        tensor.reluBackwardBatched(rbX, rbDY, rbDX);
        const rb = rbDX.download();
        if (rb[0] !== 0 || rb[1] !== 7 || rb[2] !== 0 || rb[3] !== 11) {
            throw new Error("reluBackwardBatched: " + Array.from(rb));
        }

        // tanhBackwardBatched reads the forward OUTPUT: dX = dY*(1-Y*Y).
        const tbY = tensor.createTensor(2, 2);
        const tbDY = tensor.createTensor(2, 2);
        const tbDX = tensor.createTensor(2, 2);
        tbY.upload([0, 0.5, -0.5, 1]);
        tbDY.upload([1, 1, 2, 1]);
        tensor.tanhBackwardBatched(tbY, tbDY, tbDX);
        const tb = tbDX.download();
        finite(tb, "tanhBackwardBatched");
        if (!near(tb[0], 1) || !near(tb[1], 0.75) || !near(tb[2], 1.5) || !near(tb[3], 0)) {
            throw new Error("tanhBackwardBatched: " + Array.from(tb));
        }

        // Argument checking is the wrapper's job.
        let threw = false;
        try { tensor.reluForwardBatched(A, null); } catch (e) { threw = true; }
        if (!threw) throw new Error("reluForwardBatched accepted a null output");

        return "OK";
    })();
    )JS");

    // ---- row gather / scatter / top-k --------------------------------------
    f += runJs("test_api_batched.rows", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const near = (a, b) => Math.abs(a - b) <= 1e-5;

        // X(3,4): rows [1,5,2,0] / [9,3,4,8] / [0,0,7,1].
        const X = tensor.createTensor(3, 4);
        X.upload([1, 5, 2, 0, 9, 3, 4, 8, 0, 0, 7, 1]);

        // topKRows(X, 2, Vals, Idx): descending per row.
        //   row0 -> 5 (col 1), 2 (col 2)
        //   row1 -> 9 (col 0), 8 (col 3)
        //   row2 -> 7 (col 2), 1 (col 3)
        const Vals = tensor.createTensor(3, 2);
        const Idx = tensor.createTensor(3, 2, "int32");
        tensor.topKRows(X, 2, Vals, Idx);
        if (Vals.rows !== 3 || Vals.cols !== 2) throw new Error("topKRows Vals shape " + Vals.rows + "x" + Vals.cols);
        if (Idx.rows !== 3 || Idx.cols !== 2) throw new Error("topKRows Idx shape " + Idx.rows + "x" + Idx.cols);
        if (Idx.dtype() !== "int32") throw new Error("topKRows Idx dtype " + Idx.dtype());
        const v = Vals.download();
        if (!near(v[0], 5) || !near(v[1], 2) || !near(v[2], 9) ||
            !near(v[3], 8) || !near(v[4], 7) || !near(v[5], 1)) {
            throw new Error("topKRows Vals: " + Array.from(v));
        }

        // k = 1 gives an (3,1) INT32 index column — exactly the shape
        // gatherRows / scatterRowsAdd want, so the two chain without a host
        // round trip. Here it is the per-row argmax column: [1, 0, 2].
        const V1 = tensor.createTensor(3, 1);
        const I1 = tensor.createTensor(3, 1, "int32");
        tensor.topKRows(X, 1, V1, I1);
        const v1 = V1.download();
        if (!near(v1[0], 5) || !near(v1[1], 9) || !near(v1[2], 7)) {
            throw new Error("topKRows k=1 Vals: " + Array.from(v1));
        }
        if (I1.dtype() !== "int32" || I1.rows !== 3 || I1.cols !== 1) {
            throw new Error("topKRows k=1 Idx: " + I1.dtype() + " " + I1.rows + "x" + I1.cols);
        }

        // gatherRows(table, I1, Y): rows 1, 0, 2 of the table.
        const table = tensor.createTensor(4, 2);
        table.upload([10, 11, 20, 21, 30, 31, 40, 41]);
        const G = tensor.createTensor(3, 2);
        tensor.gatherRows(table, I1, G);
        if (G.rows !== 3 || G.cols !== 2) throw new Error("gatherRows shape " + G.rows + "x" + G.cols);
        const g = G.download();
        if (!near(g[0], 20) || !near(g[1], 21) || !near(g[2], 10) ||
            !near(g[3], 11) || !near(g[4], 30) || !near(g[5], 31)) {
            throw new Error("gatherRows: " + Array.from(g));
        }

        // An index column built in JS arrives FP32 (upload() always does);
        // the native converts it. Idx = [2,0,1] -> rows 2, 0, 1.
        const If = tensor.createTensor(3, 1);
        If.upload([2, 0, 1]);
        const G2 = tensor.createTensor(3, 2);
        tensor.gatherRows(table, If, G2);
        const g2 = G2.download();
        if (!near(g2[0], 30) || !near(g2[1], 31) || !near(g2[2], 10) ||
            !near(g2[3], 11) || !near(g2[4], 20) || !near(g2[5], 21)) {
            throw new Error("gatherRows (fp32 idx): " + Array.from(g2));
        }

        // scatterRowsAdd(dY, I1, 4, dX): dX(4,2) zeroed, then dY rows summed
        // onto dX[I1[m]]. I1 = [1,0,2] -> dX = [[2,2],[1,1],[3,3],[0,0]].
        const dY = tensor.createTensor(3, 2);
        dY.upload([1, 1, 2, 2, 3, 3]);
        const dXs = tensor.createTensor(4, 2);
        tensor.scatterRowsAdd(dY, I1, 4, dXs);
        if (dXs.rows !== 4 || dXs.cols !== 2) throw new Error("scatterRowsAdd shape " + dXs.rows + "x" + dXs.cols);
        const s = dXs.download();
        if (!near(s[0], 2) || !near(s[1], 2) || !near(s[2], 1) || !near(s[3], 1) ||
            !near(s[4], 3) || !near(s[5], 3) || !near(s[6], 0) || !near(s[7], 0)) {
            throw new Error("scatterRowsAdd: " + Array.from(s));
        }

        // k > C is rejected by the op, and the wrapper rethrows it.
        let threw = false;
        try { tensor.topKRows(X, 99, Vals, Idx); } catch (e) { threw = true; }
        if (!threw) throw new Error("topKRows accepted k > C");

        return "OK";
    })();
    )JS");

    // ---- batched LayerNorm with training caches ----------------------------
    f += runJs("test_api_batched.layernorm", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const near = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-4 : tol);
        const finite = (arr, what) => {
            for (let i = 0; i < arr.length; i++) {
                if (!isFinite(arr[i])) throw new Error(what + ": non-finite at " + i);
            }
        };

        // X(2,4): row0 [1,2,3,4] (mean 2.5, var 1.25, rstd 0.8944272)
        //         row1 [2,4,6,8] (mean 5.0, var 5.00, rstd 0.4472136)
        // Both rows normalize to the same Xhat pattern.
        const XHAT = [-1.3416408, -0.4472136, 0.4472136, 1.3416408];
        const D = 4;
        const X = tensor.createTensor(2, D);
        X.upload([1, 2, 3, 4, 2, 4, 6, 8]);
        const gamma = tensor.createTensor(D, 1);
        const beta = tensor.createTensor(D, 1);
        gamma.upload([1, 2, 3, 4]);
        beta.upload([0.5, 0.5, 0.5, 0.5]);

        const Y = tensor.createTensor(2, D);
        const Xhat = tensor.createTensor(2, D);
        const Mean = tensor.createTensor(2, 1);
        const Rstd = tensor.createTensor(2, 1);
        tensor.layernormForwardBatchedWithCaches(X, gamma, beta, Y, Xhat, Mean, Rstd, 1e-5);

        if (Y.rows !== 2 || Y.cols !== D) throw new Error("layernormFwdCaches Y shape " + Y.rows + "x" + Y.cols);
        if (Mean.rows !== 2 || Mean.cols !== 1) throw new Error("layernormFwdCaches Mean shape " + Mean.rows + "x" + Mean.cols);
        if (Rstd.rows !== 2 || Rstd.cols !== 1) throw new Error("layernormFwdCaches Rstd shape " + Rstd.rows + "x" + Rstd.cols);

        const m = Mean.download(), rs = Rstd.download(), xh = Xhat.download(), y = Y.download();
        finite(m, "Mean"); finite(rs, "Rstd"); finite(xh, "Xhat"); finite(y, "Y");
        if (!near(m[0], 2.5) || !near(m[1], 5.0)) throw new Error("layernormFwdCaches Mean: " + Array.from(m));
        if (!near(rs[0], 0.8944272) || !near(rs[1], 0.4472136)) throw new Error("layernormFwdCaches Rstd: " + Array.from(rs));
        for (let r = 0; r < 2; r++) {
            for (let i = 0; i < D; i++) {
                const got = xh[r * D + i];
                if (!near(got, XHAT[i])) throw new Error("layernormFwdCaches Xhat[" + r + "][" + i + "]=" + got);
            }
        }
        // Y = gamma*Xhat + beta.
        const GAMMA = [1, 2, 3, 4];
        for (let r = 0; r < 2; r++) {
            for (let i = 0; i < D; i++) {
                const want = GAMMA[i] * XHAT[i] + 0.5;
                if (!near(y[r * D + i], want)) throw new Error("layernormFwdCaches Y[" + r + "][" + i + "]=" + y[r * D + i]);
            }
        }

        // Backward with gamma = 1 and dY = 1: the row-mean and row-projection
        // terms cancel, so dX is 0; dBeta accumulates R copies of dY and
        // dGamma accumulates sum_r Xhat[r][i] = 2*XHAT[i].
        const gOnes = tensor.createTensor(D, 1);
        gOnes.upload([1, 1, 1, 1]);
        const dY = tensor.createTensor(2, D);
        dY.upload([1, 1, 1, 1, 1, 1, 1, 1]);
        const dX = tensor.createTensor(2, D);
        const dGamma = tensor.createTensor(D, 1);   // zeros: the op accumulates
        const dBeta = tensor.createTensor(D, 1);
        tensor.layernormBackwardBatchedWithCaches(dY, Xhat, gOnes, Rstd, dX, dGamma, dBeta);

        if (dX.rows !== 2 || dX.cols !== D) throw new Error("layernormBwdCaches dX shape " + dX.rows + "x" + dX.cols);
        const dx = dX.download(), dg = dGamma.download(), dbt = dBeta.download();
        finite(dx, "dX"); finite(dg, "dGamma"); finite(dbt, "dBeta");
        for (let i = 0; i < dx.length; i++) {
            if (!near(dx[i], 0, 1e-3)) throw new Error("layernormBwdCaches dX[" + i + "]=" + dx[i]);
        }
        for (let i = 0; i < D; i++) {
            if (!near(dbt[i], 2)) throw new Error("layernormBwdCaches dBeta: " + Array.from(dbt));
            if (!near(dg[i], 2 * XHAT[i], 1e-3)) throw new Error("layernormBwdCaches dGamma: " + Array.from(dg));
        }

        return "OK";
    })();
    )JS");

    // ---- 16-bit batched linear (GPU-only slot) -----------------------------
    f += runJs("test_api_batched.fp16", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        if (tensor.backend === "cpu") return "OK";   // no FP16 slot on CPU
        const near = (a, b) => Math.abs(a - b) <= 0.02;

        // Same numbers as the FP32 case, in FP16 storage throughout.
        //   W = [[1,2,3],[4,5,6]], X[0] = [1,0,0], X[1] = [0,1,1]
        //   no bias -> [[1,4],[5,11]];  bias [1,-1] -> [[2,3],[6,10]]
        const W32 = tensor.createTensor(2, 3);
        const X32 = tensor.createTensor(2, 3);
        const b32 = tensor.createTensor(2, 1);
        W32.upload([1, 2, 3, 4, 5, 6]);
        X32.upload([1, 0, 0, 0, 1, 1]);
        b32.upload([1, -1]);

        const W = tensor.createTensor(2, 3, "fp16");
        const X = tensor.createTensor(2, 3, "fp16");
        const bias = tensor.createTensor(2, 1, "fp16");
        tensor.cast(W32, W, "fp16");
        tensor.cast(X32, X, "fp16");
        tensor.cast(b32, bias, "fp16");

        const Y = tensor.createTensor(2, 2, "fp16");
        tensor.linearForwardBatchedFp16(W, null, X, Y);
        if (Y.rows !== 2 || Y.cols !== 2) throw new Error("linearForwardBatchedFp16 shape " + Y.rows + "x" + Y.cols);
        const y = Y.download();
        for (let i = 0; i < y.length; i++) {
            if (!isFinite(y[i])) throw new Error("linearForwardBatchedFp16: non-finite at " + i);
        }
        if (!near(y[0], 1) || !near(y[1], 4) || !near(y[2], 5) || !near(y[3], 11)) {
            throw new Error("linearForwardBatchedFp16 (no bias): " + Array.from(y));
        }

        tensor.linearForwardBatchedFp16(W, bias, X, Y);
        const yb = Y.download();
        if (!near(yb[0], 2) || !near(yb[1], 3) || !near(yb[2], 6) || !near(yb[3], 10)) {
            throw new Error("linearForwardBatchedFp16 (bias): " + Array.from(yb));
        }

        // A non-GpuTensor bias is rejected by the wrapper, not the native.
        let threw = false;
        try { tensor.linearForwardBatchedFp16(W, 7, X, Y); } catch (e) { threw = true; }
        if (!threw) throw new Error("linearForwardBatchedFp16 accepted a non-tensor bias");

        return "OK";
    })();
    )JS");

    return f;
}
