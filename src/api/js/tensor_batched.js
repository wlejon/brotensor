// tensor_batched.js — the restored "batched" slice of bro.tensor: the B-row
// batched dense / activation forwards (linearForwardBatched,
// linearForwardBatchedFp16, reluForwardBatched, tanhForwardBatched,
// addInplaceBatched), their training backwards (linearBackwardBatched,
// reluBackwardBatched, tanhBackwardBatched), the row gather / scatter / top-k
// primitives (gatherRows, scatterRowsAdd, topKRows) and batched LayerNorm with
// its training caches (layernormForwardBatchedWithCaches,
// layernormBackwardBatchedWithCaches).
//
// Signatures are the QuickJS binding's (tensor_bindings.cpp): same argument
// order, same optionality — `bias|null` on linearForwardBatchedFp16 is the one
// nullable slot, and it crosses to the native as a raw value, so every
// argument is checked here. Hand-written like js/tensor.js and js/tensor_ext.js
// and compiled into its own module (bronze_tensor_batched_main), mounted by
// api.cpp after those two, so `bro.tensor` and `bro.tensor.GpuTensor` already
// exist. Each wrapper reads the native's error back with chk() and throws it.
(function () {
    'use strict';

    const bro = globalThis.bro;
    const ns_tensor = bro.tensor;
    const GpuTensor = ns_tensor.GpuTensor;
    const fn = (obj, name, value) =>
        Object.defineProperty(obj, name, { value, writable: true, enumerable: true, configurable: true });

    const chk = () => {
        const e = __bro_native.tensor.takeError();
        if (e) throw new Error(e);
    };
    // A required GpuTensor.
    const T = (v, label, name) => {
        if (!(v instanceof GpuTensor)) throw new TypeError(label + ": " + name + " must be a GpuTensor");
        return v;
    };
    // An optional GpuTensor: null / undefined pass through as null.
    const opt = (v, label, name) => {
        if (v === undefined || v === null) return null;
        if (!(v instanceof GpuTensor)) throw new TypeError(label + ": " + name + " must be null or a GpuTensor");
        return v;
    };
    const int = (v, def) => (v === undefined || v === null ? def : (v | 0));
    const num = (v, def) => (v === undefined || v === null ? def : +v);

    // ---- batched forwards ------------------------------------------------
    // B independent forward passes in one launch. Tensors carrying B rows are
    // (B, D) row-major: W is (out,in), bias (out,1), X_BD (B,in), Y_BD (B,out)
    // and resized. W may be FP32/FP16/BF16; the activations stay FP32.
    fn(ns_tensor, "linearForwardBatched", function linearForwardBatched(W, bias, X_BD, Y_BD) {
        const L = "linearForwardBatched(W,bias,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatched(T(W, L, "W"), T(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    // 16-bit storage throughout (FP16 or BF16 for W, bias, X and the produced
    // Y). Inference-only, and GPU-only — the CPU backend registers no slot.
    fn(ns_tensor, "linearForwardBatchedFp16", function linearForwardBatchedFp16(W, bias, X_BD, Y_BD) {
        const L = "linearForwardBatchedFp16(W,bias|null,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatchedFp16(T(W, L, "W"), opt(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "reluForwardBatched", function reluForwardBatched(X_BD, Y_BD) {
        const L = "reluForwardBatched(X_BD,Y_BD)";
        __bro_native.tensor.reluForwardBatched(T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "tanhForwardBatched", function tanhForwardBatched(X_BD, Y_BD) {
        const L = "tanhForwardBatched(X_BD,Y_BD)";
        __bro_native.tensor.tanhForwardBatched(T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "addInplaceBatched", function addInplaceBatched(Y_BD, X_BD) {
        const L = "addInplaceBatched(Y_BD,X_BD)";
        __bro_native.tensor.addInplaceBatched(T(Y_BD, L, "Y_BD"), T(X_BD, L, "X_BD"));
        chk();
    });

    // ---- batched-training backwards ---------------------------------------
    // dW and dB accumulate (project convention): the caller zeros them.
    fn(ns_tensor, "linearBackwardBatched", function linearBackwardBatched(W, X_BD, dY_BD, dX_BD, dW, dB) {
        const L = "linearBackwardBatched(W,X_BD,dY_BD,dX_BD,dW,dB)";
        __bro_native.tensor.linearBackwardBatched(T(W, L, "W"), T(X_BD, L, "X_BD"), T(dY_BD, L, "dY_BD"),
            T(dX_BD, L, "dX_BD"), T(dW, L, "dW"), T(dB, L, "dB"));
        chk();
    });
    // relu reads the forward *input*: dX = dY*(X>0).
    fn(ns_tensor, "reluBackwardBatched", function reluBackwardBatched(X_BD, dY_BD, dX_BD) {
        const L = "reluBackwardBatched(X_BD,dY_BD,dX_BD)";
        __bro_native.tensor.reluBackwardBatched(T(X_BD, L, "X_BD"), T(dY_BD, L, "dY_BD"), T(dX_BD, L, "dX_BD"));
        chk();
    });
    // tanh reads the forward *output*: dX = dY*(1-Y*Y).
    fn(ns_tensor, "tanhBackwardBatched", function tanhBackwardBatched(Y_BD, dY_BD, dX_BD) {
        const L = "tanhBackwardBatched(Y_BD,dY_BD,dX_BD)";
        __bro_native.tensor.tanhBackwardBatched(T(Y_BD, L, "Y_BD"), T(dY_BD, L, "dY_BD"), T(dX_BD, L, "dX_BD"));
        chk();
    });

    // ---- row gather / scatter / top-k -------------------------------------
    // Idx is an (M,1) index tensor — INT32 as topKRows writes it, or an FP32
    // tensor of whole numbers, which the native converts. Out-of-range values
    // are the caller's problem (the ops don't bounds-check).
    fn(ns_tensor, "gatherRows", function gatherRows(X, Idx, Y) {
        const L = "gatherRows(X,Idx,Y)";
        __bro_native.tensor.gatherRows(T(X, L, "X"), T(Idx, L, "Idx"), T(Y, L, "Y"));
        chk();
    });
    // The adjoint: dX is (R,C), zeroed and then scatter-added into. R is the
    // forward X's row count, which dY and Idx alone don't give.
    fn(ns_tensor, "scatterRowsAdd", function scatterRowsAdd(dY, Idx, R, dX) {
        const L = "scatterRowsAdd(dY,Idx,R,dX)";
        __bro_native.tensor.scatterRowsAdd(T(dY, L, "dY"), T(Idx, L, "Idx"), int(R, 0), T(dX, L, "dX"));
        chk();
    });
    // Per-row top-k, descending, ties to the smaller column index.
    // Vals: (R,k) FP32, Idx: (R,k) INT32 — both resized + dtype-set.
    fn(ns_tensor, "topKRows", function topKRows(X, k, Vals, Idx) {
        const L = "topKRows(X,k,Vals,Idx)";
        __bro_native.tensor.topKRows(T(X, L, "X"), int(k, 1), T(Vals, L, "Vals"), T(Idx, L, "Idx"));
        chk();
    });

    // ---- batched LayerNorm with training caches ---------------------------
    // X (R,D), gamma/beta (D,), Y/Xhat (R,D) resized, Mean/Rstd (R,1) FP32
    // whatever X's dtype is.
    fn(ns_tensor, "layernormForwardBatchedWithCaches", function layernormForwardBatchedWithCaches(X, gamma, beta, Y, Xhat, Mean, Rstd, eps) {
        const L = "layernormForwardBatchedWithCaches(X,gamma,beta,Y,Xhat,Mean,Rstd,eps)";
        __bro_native.tensor.layernormForwardBatchedWithCaches(T(X, L, "X"), T(gamma, L, "gamma"), T(beta, L, "beta"),
            T(Y, L, "Y"), T(Xhat, L, "Xhat"), T(Mean, L, "Mean"), T(Rstd, L, "Rstd"), num(eps, 0.00001));
        chk();
    });
    // dX is overwritten; dGamma and dBeta accumulate into (D,) tensors the
    // caller sized and zeroed — they are not resized here.
    fn(ns_tensor, "layernormBackwardBatchedWithCaches", function layernormBackwardBatchedWithCaches(dY, Xhat, gamma, Rstd, dX, dGamma, dBeta) {
        const L = "layernormBackwardBatchedWithCaches(dY,Xhat,gamma,Rstd,dX,dGamma,dBeta)";
        __bro_native.tensor.layernormBackwardBatchedWithCaches(T(dY, L, "dY"), T(Xhat, L, "Xhat"), T(gamma, L, "gamma"),
            T(Rstd, L, "Rstd"), T(dX, L, "dX"), T(dGamma, L, "dGamma"), T(dBeta, L, "dBeta"));
        chk();
    });
})();
