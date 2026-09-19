// tensor_ext.js — the rest of bro.tensor's public shape: the attention family
// (single-head, multi-head, self / cross, T5-style bias, flash), resblock,
// masked mean-pool, the fused softmax + cross-entropy losses and the concat /
// split ops. Hand-written like js/tensor.js, and compiled into a second
// module (bronze_tensor_ext_main) that api.cpp mounts right after it, so
// `bro.tensor` and `bro.tensor.GpuTensor` already exist here.
//
// Signatures are the QuickJS binding's (tensor_bindings.cpp,
// tensor_bindings_attention.cpp): `mask|null` slots take a length-N FP32
// GpuTensor or null, the projection biases and the optional gradient outputs
// take a GpuTensor or null, and the three options-object entry points
// (flashAttentionQkvoBackward, resblockForward, resblockBackward) keep their
// object form. Every wrapper checks its arguments here, because the natives
// receive nullable slots as raw values, and reads back the native's error
// after the call (native_tensor_decl.h takeError).
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
    // An array of GpuTensors (concat parts, split targets).
    const list = (v, label, name) => {
        if (!Array.isArray(v)) throw new TypeError(label + ": " + name + " must be an array of GpuTensors");
        for (let i = 0; i < v.length; i++) {
            if (!(v[i] instanceof GpuTensor)) throw new TypeError(label + ": " + name + "[" + i + "] is not a GpuTensor");
        }
        return v;
    };
    const int = (v, def) => (v === undefined || v === null ? def : (v | 0));
    const num = (v, def) => (v === undefined || v === null ? def : +v);

    // ---- single-head attention (FP32, training) ---------------------------------
    fn(ns_tensor, "attentionForward", function attentionForward(X, Wq, Wk, Wv, Wo, mask, Q, K, V, Attn, Y_pre_Wo, O) {
        const L = "attentionForward(X,Wq,Wk,Wv,Wo,mask|null,Q,K,V,Attn,Y_pre_Wo,O)";
        __bro_native.tensor.attentionForward(T(X, L, "X"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), T(Attn, L, "Attn"), T(Y_pre_Wo, L, "Y_pre_Wo"), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "attentionBackward", function attentionBackward(dO, X, Q, K, V, Attn, Y_pre_Wo, Wq, Wk, Wv, Wo, mask, dX, dWq, dWk, dWv, dWo) {
        const L = "attentionBackward(dO,X,Q,K,V,Attn,Y_pre_Wo,Wq,Wk,Wv,Wo,mask|null,dX,dWq,dWk,dWv,dWo)";
        __bro_native.tensor.attentionBackward(T(dO, L, "dO"), T(X, L, "X"), T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"),
            T(Attn, L, "Attn"), T(Y_pre_Wo, L, "Y_pre_Wo"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), T(dX, L, "dX"), T(dWq, L, "dWq"), T(dWk, L, "dWk"), T(dWv, L, "dWv"), T(dWo, L, "dWo"));
        chk();
    });

    // ---- multi-head self-attention (FP32, training) ---------------------------
    fn(ns_tensor, "mhaForward", function mhaForward(X, Wq, Wk, Wv, Wo, mask, numHeads, Qh, Kh, Vh, Attnh, Yconcat, O) {
        const L = "mhaForward(X,Wq,Wk,Wv,Wo,mask|null,numHeads,Qh,Kh,Vh,Attnh,Yconcat,O)";
        __bro_native.tensor.mhaForward(T(X, L, "X"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"), T(Attnh, L, "Attnh"),
            T(Yconcat, L, "Yconcat"), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "mhaBackward", function mhaBackward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, mask, numHeads, dX, dWq, dWk, dWv, dWo) {
        const L = "mhaBackward(dO,X,Qh,Kh,Vh,Attnh,Yconcat,Wq,Wk,Wv,Wo,mask|null,numHeads,dX,dWq,dWk,dWv,dWo)";
        __bro_native.tensor.mhaBackward(T(dO, L, "dO"), T(X, L, "X"), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"),
            T(Attnh, L, "Attnh"), T(Yconcat, L, "Yconcat"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(dX, L, "dX"), T(dWq, L, "dWq"), T(dWk, L, "dWk"), T(dWv, L, "dWv"), T(dWo, L, "dWo"));
        chk();
    });

    // ---- self-attention: inference (dtype of X) + FP32 training caches ---------
    fn(ns_tensor, "selfAttentionForward", function selfAttentionForward(X, Wq, Wk, Wv, Wo, mask, numHeads, O) {
        const L = "selfAttentionForward(X,Wq,Wk,Wv,Wo,mask|null,numHeads,O)";
        __bro_native.tensor.selfAttentionForward(T(X, L, "X"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "selfAttentionForwardTrain", function selfAttentionForwardTrain(X, Wq, Wk, Wv, Wo, mask, numHeads, Qh, Kh, Vh, Attnh, Yconcat, O) {
        const L = "selfAttentionForwardTrain(X,Wq,Wk,Wv,Wo,mask|null,numHeads,Qh,Kh,Vh,Attnh,Yconcat,O)";
        __bro_native.tensor.selfAttentionForwardTrain(T(X, L, "X"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"), T(Attnh, L, "Attnh"),
            T(Yconcat, L, "Yconcat"), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "selfAttentionBackward", function selfAttentionBackward(dO, X, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, mask, numHeads, dX, dWq, dWk, dWv, dWo) {
        const L = "selfAttentionBackward(dO,X,Qh,Kh,Vh,Attnh,Yconcat,Wq,Wk,Wv,Wo,mask|null,numHeads,dX,dWq,dWk,dWv,dWo)";
        __bro_native.tensor.selfAttentionBackward(T(dO, L, "dO"), T(X, L, "X"), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"),
            T(Attnh, L, "Attnh"), T(Yconcat, L, "Yconcat"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(dX, L, "dX"), T(dWq, L, "dWq"), T(dWk, L, "dWk"), T(dWv, L, "dWv"), T(dWo, L, "dWo"));
        chk();
    });
    // T5-style: additive per-head pre-softmax bias, (numHeads*L, L) FP32 or null;
    // scale multiplies QK before the bias (1.0 for T5).
    fn(ns_tensor, "selfAttentionBiasForward", function selfAttentionBiasForward(X, Wq, Wk, Wv, Wo, mask, attnBias, numHeads, scale, O) {
        const L = "selfAttentionBiasForward(X,Wq,Wk,Wv,Wo,mask|null,attnBias|null,numHeads,scale,O)";
        __bro_native.tensor.selfAttentionBiasForward(T(X, L, "X"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), opt(attnBias, L, "attnBias"), int(numHeads, 1), num(scale, 1.0), T(O, L, "O"));
        chk();
    });

    // ---- cross-attention -------------------------------------------------------
    fn(ns_tensor, "crossAttentionForward", function crossAttentionForward(X, Ctx, Wq, Wk, Wv, Wo, mask, numHeads, O) {
        const L = "crossAttentionForward(X,Ctx,Wq,Wk,Wv,Wo,mask|null,numHeads,O)";
        __bro_native.tensor.crossAttentionForward(T(X, L, "X"), T(Ctx, L, "Ctx"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"),
            T(Wo, L, "Wo"), opt(mask, L, "mask"), int(numHeads, 1), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "crossAttentionForwardWithAttn", function crossAttentionForwardWithAttn(X, Ctx, Wq, Wk, Wv, Wo, mask, attnLogitBias, numHeads, O, AttnAvg) {
        const L = "crossAttentionForwardWithAttn(X,Ctx,Wq,Wk,Wv,Wo,mask|null,attnLogitBias|null,numHeads,O,AttnAvg)";
        __bro_native.tensor.crossAttentionForwardWithAttn(T(X, L, "X"), T(Ctx, L, "Ctx"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"),
            T(Wo, L, "Wo"), opt(mask, L, "mask"), opt(attnLogitBias, L, "attnLogitBias"), int(numHeads, 1), T(O, L, "O"), T(AttnAvg, L, "AttnAvg"));
        chk();
    });
    fn(ns_tensor, "crossAttentionForwardTrain", function crossAttentionForwardTrain(X, Ctx, Wq, Wk, Wv, Wo, mask, numHeads, Qh, Kh, Vh, Attnh, Yconcat, O) {
        const L = "crossAttentionForwardTrain(X,Ctx,Wq,Wk,Wv,Wo,mask|null,numHeads,Qh,Kh,Vh,Attnh,Yconcat,O)";
        __bro_native.tensor.crossAttentionForwardTrain(T(X, L, "X"), T(Ctx, L, "Ctx"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"),
            T(Wo, L, "Wo"), opt(mask, L, "mask"), int(numHeads, 1), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"), T(Attnh, L, "Attnh"),
            T(Yconcat, L, "Yconcat"), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "crossAttentionBackward", function crossAttentionBackward(dO, X, Ctx, Qh, Kh, Vh, Attnh, Yconcat, Wq, Wk, Wv, Wo, mask, numHeads, dX, dCtx, dWq, dWk, dWv, dWo) {
        const L = "crossAttentionBackward(dO,X,Ctx,Qh,Kh,Vh,Attnh,Yconcat,Wq,Wk,Wv,Wo,mask|null,numHeads,dX,dCtx,dWq,dWk,dWv,dWo)";
        __bro_native.tensor.crossAttentionBackward(T(dO, L, "dO"), T(X, L, "X"), T(Ctx, L, "Ctx"), T(Qh, L, "Qh"), T(Kh, L, "Kh"), T(Vh, L, "Vh"),
            T(Attnh, L, "Attnh"), T(Yconcat, L, "Yconcat"), T(Wq, L, "Wq"), T(Wk, L, "Wk"), T(Wv, L, "Wv"), T(Wo, L, "Wo"),
            opt(mask, L, "mask"), int(numHeads, 1), T(dX, L, "dX"), T(dCtx, L, "dCtx"), T(dWq, L, "dWq"), T(dWk, L, "dWk"), T(dWv, L, "dWv"), T(dWo, L, "dWo"));
        chk();
    });

    // ---- flash attention -------------------------------------------------------
    fn(ns_tensor, "flashAttentionForward", function flashAttentionForward(Q, K, V, mask, numHeads, causal, O) {
        const L = "flashAttentionForward(Q,K,V,mask|null,numHeads,causal,O)";
        __bro_native.tensor.flashAttentionForward(T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), opt(mask, L, "mask"), int(numHeads, 1), !!causal, T(O, L, "O"));
        chk();
    });
    // Sliding-window causal; window <= 0 is unbounded causal.
    fn(ns_tensor, "flashAttentionWindowedForward", function flashAttentionWindowedForward(Q, K, V, mask, numHeads, window, O) {
        const L = "flashAttentionWindowedForward(Q,K,V,mask|null,numHeads,window,O)";
        __bro_native.tensor.flashAttentionWindowedForward(T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), opt(mask, L, "mask"), int(numHeads, 1), int(window, 0), T(O, L, "O"));
        chk();
    });
    fn(ns_tensor, "flashAttentionBackward", function flashAttentionBackward(Q, K, V, O, dO, mask, numHeads, causal, dQ, dK, dV) {
        const L = "flashAttentionBackward(Q,K,V,O,dO,mask|null,numHeads,causal,dQ,dK,dV)";
        __bro_native.tensor.flashAttentionBackward(T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), T(O, L, "O"), T(dO, L, "dO"),
            opt(mask, L, "mask"), int(numHeads, 1), !!causal, T(dQ, L, "dQ"), T(dK, L, "dK"), T(dV, L, "dV"));
        chk();
    });
    fn(ns_tensor, "flashAttentionQkvoForward", function flashAttentionQkvoForward(X, Ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo, mask, numHeads, causal, O) {
        const L = "flashAttentionQkvoForward(X,Ctx|null,Wq,bq|null,Wk,bk|null,Wv,bv|null,Wo,bo|null,mask|null,numHeads,causal,O)";
        __bro_native.tensor.flashAttentionQkvoForward(T(X, L, "X"), opt(Ctx, L, "Ctx"), T(Wq, L, "Wq"), opt(bq, L, "bq"), T(Wk, L, "Wk"), opt(bk, L, "bk"),
            T(Wv, L, "Wv"), opt(bv, L, "bv"), T(Wo, L, "Wo"), opt(bo, L, "bo"), opt(mask, L, "mask"), int(numHeads, 1), !!causal, T(O, L, "O"));
        chk();
    });
    // flashAttentionQkvoBackward(opts): required X, Wq, Wk, Wv, Wo, dO, numHeads,
    // dX, dWq, dWk, dWv, dWo; optional Ctx, bq, bk, bv, bo, mask, causal, dCtx,
    // dbq, dbk, dbv, dbo (null/false by default).
    fn(ns_tensor, "flashAttentionQkvoBackward", function flashAttentionQkvoBackward(o) {
        const L = "flashAttentionQkvoBackward(opts)";
        if (o === null || typeof o !== "object") throw new TypeError(L + " — opts is an object; see docs");
        __bro_native.tensor.flashAttentionQkvoBackward(T(o.X, L, "X"), opt(o.Ctx, L, "Ctx"), T(o.Wq, L, "Wq"), opt(o.bq, L, "bq"),
            T(o.Wk, L, "Wk"), opt(o.bk, L, "bk"), T(o.Wv, L, "Wv"), opt(o.bv, L, "bv"), T(o.Wo, L, "Wo"), opt(o.bo, L, "bo"),
            opt(o.mask, L, "mask"), int(o.numHeads, 1), !!o.causal, T(o.dO, L, "dO"),
            T(o.dX, L, "dX"), opt(o.dCtx, L, "dCtx"), T(o.dWq, L, "dWq"), opt(o.dbq, L, "dbq"), T(o.dWk, L, "dWk"), opt(o.dbk, L, "dbk"),
            T(o.dWv, L, "dWv"), opt(o.dbv, L, "dbv"), T(o.dWo, L, "dWo"), opt(o.dbo, L, "dbo"));
        chk();
    });
    fn(ns_tensor, "flashAttentionProjectKv", function flashAttentionProjectKv(ctx, Wk, bk, Wv, bv, K_out, V_out) {
        const L = "flashAttentionProjectKv(ctx,Wk,bk|null,Wv,bv|null,K_out,V_out)";
        __bro_native.tensor.flashAttentionProjectKv(T(ctx, L, "ctx"), T(Wk, L, "Wk"), opt(bk, L, "bk"), T(Wv, L, "Wv"), opt(bv, L, "bv"), T(K_out, L, "K_out"), T(V_out, L, "V_out"));
        chk();
    });
    fn(ns_tensor, "flashAttentionQWithKvCachedForward", function flashAttentionQWithKvCachedForward(X, K, V, Wq, bq, Wo, bo, mask, numHeads, causal, O) {
        const L = "flashAttentionQWithKvCachedForward(X,K,V,Wq,bq|null,Wo,bo|null,mask|null,numHeads,causal,O)";
        __bro_native.tensor.flashAttentionQWithKvCachedForward(T(X, L, "X"), T(K, L, "K"), T(V, L, "V"), T(Wq, L, "Wq"), opt(bq, L, "bq"),
            T(Wo, L, "Wo"), opt(bo, L, "bo"), opt(mask, L, "mask"), int(numHeads, 1), !!causal, T(O, L, "O"));
        chk();
    });

    // ---- resblock (options object) --------------------------------------------
    // resblockForward(opts): required X, gamma1, beta1, W1, gamma2, beta2, W2, Y,
    // N, C_in, C_out, H, W; optional b1, t_emb_shift, b2, Wskip, bskip,
    // numGroups (32), eps (1e-5).
    fn(ns_tensor, "resblockForward", function resblockForward(o) {
        const L = "resblockForward(opts)";
        if (o === null || typeof o !== "object") throw new TypeError(L + " — see docs");
        __bro_native.tensor.resblockForward(T(o.X, L, "X"), T(o.gamma1, L, "gamma1"), T(o.beta1, L, "beta1"), T(o.W1, L, "W1"),
            opt(o.b1, L, "b1"), opt(o.t_emb_shift, L, "t_emb_shift"), T(o.gamma2, L, "gamma2"), T(o.beta2, L, "beta2"), T(o.W2, L, "W2"),
            opt(o.b2, L, "b2"), opt(o.Wskip, L, "Wskip"), opt(o.bskip, L, "bskip"),
            int(o.N, 0), int(o.C_in, 0), int(o.C_out, 0), int(o.H, 0), int(o.W, 0), int(o.numGroups, 32), num(o.eps, 0.00001), T(o.Y, L, "Y"));
        chk();
    });
    // resblockBackward(opts): the forward inputs plus dY, dX, dGamma1, dBeta1,
    // dW1, dGamma2, dBeta2, dW2 (required) and db1, dt_emb_shift, db2, dWskip,
    // dbskip (optional, accumulated iff given).
    fn(ns_tensor, "resblockBackward", function resblockBackward(o) {
        const L = "resblockBackward(opts)";
        if (o === null || typeof o !== "object") throw new TypeError(L + " — see docs");
        __bro_native.tensor.resblockBackward(T(o.X, L, "X"), T(o.gamma1, L, "gamma1"), T(o.beta1, L, "beta1"), T(o.W1, L, "W1"),
            opt(o.b1, L, "b1"), opt(o.t_emb_shift, L, "t_emb_shift"), T(o.gamma2, L, "gamma2"), T(o.beta2, L, "beta2"), T(o.W2, L, "W2"),
            opt(o.b2, L, "b2"), opt(o.Wskip, L, "Wskip"), opt(o.bskip, L, "bskip"),
            int(o.N, 0), int(o.C_in, 0), int(o.C_out, 0), int(o.H, 0), int(o.W, 0), int(o.numGroups, 32), num(o.eps, 0.00001),
            T(o.dY, L, "dY"), T(o.dX, L, "dX"), T(o.dGamma1, L, "dGamma1"), T(o.dBeta1, L, "dBeta1"), T(o.dW1, L, "dW1"),
            opt(o.db1, L, "db1"), opt(o.dt_emb_shift, L, "dt_emb_shift"), T(o.dGamma2, L, "dGamma2"), T(o.dBeta2, L, "dBeta2"), T(o.dW2, L, "dW2"),
            opt(o.db2, L, "db2"), opt(o.dWskip, L, "dWskip"), opt(o.dbskip, L, "dbskip"));
        chk();
    });

    // ---- pooling + losses ------------------------------------------------------
    fn(ns_tensor, "maskedMeanPoolForward", function maskedMeanPoolForward(X, mask, y) {
        const L = "maskedMeanPoolForward(X,mask|null,y)";
        __bro_native.tensor.maskedMeanPoolForward(T(X, L, "X"), opt(mask, L, "mask"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "maskedMeanPoolBackward", function maskedMeanPoolBackward(dY, mask, K, dX) {
        const L = "maskedMeanPoolBackward(dY,mask|null,K,dX)";
        __bro_native.tensor.maskedMeanPoolBackward(T(dY, L, "dY"), opt(mask, L, "mask"), int(K, 0), T(dX, L, "dX"));
        chk();
    });
    // softmaxXentFused(logits, target, mask|null, probs, dLogits) -> loss
    fn(ns_tensor, "softmaxXentFused", function softmaxXentFused(logits, target, mask, probs, dLogits) {
        const L = "softmaxXentFused(logits,target,mask|null,probs,dLogits)";
        const loss = __bro_native.tensor.softmaxXentFused(T(logits, L, "logits"), T(target, L, "target"), opt(mask, L, "mask"), T(probs, L, "probs"), T(dLogits, L, "dLogits"));
        chk();
        return loss;
    });
    // headOffsets: a GpuTensor holding a device INT32 buffer of n_heads+1 cumulative offsets.
    fn(ns_tensor, "softmaxXentFusedBatched", function softmaxXentFusedBatched(logits_BL, target_BL, mask, headOffsets, nHeads, probs_BL, dLogits_BL, lossPerSample) {
        const L = "softmaxXentFusedBatched(logits_BL,target_BL,mask|null,headOffsets,nHeads,probs_BL,dLogits_BL,lossPerSample)";
        __bro_native.tensor.softmaxXentFusedBatched(T(logits_BL, L, "logits_BL"), T(target_BL, L, "target_BL"), opt(mask, L, "mask"),
            T(headOffsets, L, "headOffsets"), int(nHeads, 1), T(probs_BL, L, "probs_BL"), T(dLogits_BL, L, "dLogits_BL"), T(lossPerSample, L, "lossPerSample"));
        chk();
    });

    // ---- concat / split --------------------------------------------------------
    fn(ns_tensor, "concatRows", function concatRows(parts, out) {
        const L = "concatRows([parts...], out)";
        __bro_native.tensor.concatRows(list(parts, L, "parts"), T(out, L, "out"));
        chk();
    });
    fn(ns_tensor, "splitRows", function splitRows(input, parts) {
        const L = "splitRows(in, [parts...])";
        __bro_native.tensor.splitRows(T(input, L, "in"), list(parts, L, "parts"));
        chk();
    });
    fn(ns_tensor, "concatBatchedRows", function concatBatchedRows(parts, out) {
        const L = "concatBatchedRows([parts...], out)";
        __bro_native.tensor.concatBatchedRows(list(parts, L, "parts"), T(out, L, "out"));
        chk();
    });
    fn(ns_tensor, "concatNchwChannels", function concatNchwChannels(parts, N, H, W, C_per_part, out) {
        const L = "concatNchwChannels([parts...], N, H, W, [C_per_part...], out)";
        if (!Array.isArray(C_per_part) && !(C_per_part instanceof Int32Array)) throw new TypeError(L + ": C_per_part must be an array of ints");
        __bro_native.tensor.concatNchwChannels(list(parts, L, "parts"), int(N, 0), int(H, 0), int(W, 0), Int32Array.from(C_per_part), T(out, L, "out"));
        chk();
    });
    fn(ns_tensor, "concatNchwChannelsBackward", function concatNchwChannelsBackward(dY, N, H, W, C_per_part, parts) {
        const L = "concatNchwChannelsBackward(dY, N, H, W, [C_per_part...], [parts...])";
        if (!Array.isArray(C_per_part) && !(C_per_part instanceof Int32Array)) throw new TypeError(L + ": C_per_part must be an array of ints");
        __bro_native.tensor.concatNchwChannelsBackward(T(dY, L, "dY"), int(N, 0), int(H, 0), int(W, 0), Int32Array.from(C_per_part), list(parts, L, "parts"));
        chk();
    });
})();
