// tensor_attn2.js — the second attention family of bro.tensor: the SAM /
// ViTDet decomposed 2D relative-position self-attention (global + windowed),
// packed variable-length flash attention (forward + backward), the Qwen3-Next
// gated delta rule (chunked prefill + streaming step) and Qwen-VL M-RoPE.
// Hand-written like js/tensor.js and js/tensor_ext.js, compiled into its own
// module (bronze_tensor_attn2_main) that api.cpp mounts after them, so
// `bro.tensor` and `bro.tensor.GpuTensor` already exist here.
//
// Signatures are the QuickJS binding's (tensor_bindings_attention.cpp), same
// argument order and the same optional slots: `bq|null` and friends take a
// GpuTensor or null, and the index streams (cuSeqQ / cuSeqK, posT / posH /
// posW) are an INT32 or whole-number FP32 GpuTensor, or null. The wrappers
// check argument types here, because the natives receive the nullable slots
// as raw values; the natives read the streams back and range-check them, and
// each wrapper rethrows the native's error afterwards.
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
    // (No entry point in this group takes an array of GpuTensors, so the
    // sibling modules' `list` helper has no use here.)
    const int = (v, def) => (v === undefined || v === null ? def : (v | 0));
    const num = (v, def) => (v === undefined || v === null ? def : +v);

    // ---- decomposed 2D rel-pos self-attention (SAM / ViTDet) -------------------
    // A token maps to grid coords (t/gridW, t%gridW) over a gridH*gridW patch
    // grid, so X.rows === gridH*gridW. The bias reads the projected query and is
    // factored into the two length-grid tables:
    //   relPosH: (2*gridH-1, headDim).   relPosW: (2*gridW-1, headDim).
    // scale multiplies the Q.K dot only (typically 1/sqrt(headDim)).
    fn(ns_tensor, "selfAttentionDecomposedRelPosForward", function selfAttentionDecomposedRelPosForward(
            X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, relPosH, relPosW, numHeads, gridH, gridW, scale, O) {
        const L = "selfAttentionDecomposedRelPosForward(X,Wq,bq|null,Wk,bk|null,Wv,bv|null,Wo,bo|null,relPosH,relPosW,numHeads,gridH,gridW,scale,O)";
        __bro_native.tensor.selfAttentionDecomposedRelPosForward(
            T(X, L, "X"), T(Wq, L, "Wq"), opt(bq, L, "bq"), T(Wk, L, "Wk"), opt(bk, L, "bk"),
            T(Wv, L, "Wv"), opt(bv, L, "bv"), T(Wo, L, "Wo"), opt(bo, L, "bo"),
            T(relPosH, L, "relPosH"), T(relPosW, L, "relPosW"),
            int(numHeads, 1), int(gridH, 0), int(gridW, 0), num(scale, 1.0), T(O, L, "O"));
        chk();
    });
    // The windowed block: window x window tiles run independently, the grid is
    // zero-padded up to a multiple of `window` and cropped back. relPosH/relPosW
    // are sized for the WINDOW here — (2*window-1, headDim).
    fn(ns_tensor, "selfAttentionDecomposedRelPosWindowedForward", function selfAttentionDecomposedRelPosWindowedForward(
            X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, relPosH, relPosW, numHeads, gridH, gridW, window, scale, O) {
        const L = "selfAttentionDecomposedRelPosWindowedForward(X,Wq,bq|null,Wk,bk|null,Wv,bv|null,Wo,bo|null,relPosH,relPosW,numHeads,gridH,gridW,window,scale,O)";
        __bro_native.tensor.selfAttentionDecomposedRelPosWindowedForward(
            T(X, L, "X"), T(Wq, L, "Wq"), opt(bq, L, "bq"), T(Wk, L, "Wk"), opt(bk, L, "bk"),
            T(Wv, L, "Wv"), opt(bv, L, "bv"), T(Wo, L, "Wo"), opt(bo, L, "bo"),
            T(relPosH, L, "relPosH"), T(relPosW, L, "relPosW"),
            int(numHeads, 1), int(gridH, 0), int(gridW, 0), int(window, 0), num(scale, 1.0), T(O, L, "O"));
        chk();
    });

    // ---- packed variable-length flash attention -------------------------------
    // Q is (totalTokensQ, numHeads*headDim), K/V are (totalTokensK, ...); the
    // per-sequence boundaries are the INT32 prefix sums cuSeqQ / cuSeqK, each of
    // length batch+1, held in a GpuTensor read as INT32 storage (null is only
    // valid with batch === 0). No cross-sequence attention.
    fn(ns_tensor, "flashAttentionVarlenForward", function flashAttentionVarlenForward(
            Q, K, V, cuSeqQ, cuSeqK, batch, maxQ, maxK, numHeads, headDim, causal, O) {
        const L = "flashAttentionVarlenForward(Q,K,V,cuSeqQ,cuSeqK,batch,maxQ,maxK,numHeads,headDim,causal,O)";
        __bro_native.tensor.flashAttentionVarlenForward(
            T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), opt(cuSeqQ, L, "cuSeqQ"), opt(cuSeqK, L, "cuSeqK"),
            int(batch, 0), int(maxQ, 0), int(maxK, 0), int(numHeads, 1), int(headDim, 0), !!causal, T(O, L, "O"));
        chk();
    });
    // Recompute-based backward: consumes no forward caches (O is kept for API
    // symmetry). dQ/dK/dV are OVERWRITTEN, not accumulated.
    fn(ns_tensor, "flashAttentionVarlenBackward", function flashAttentionVarlenBackward(
            Q, K, V, O, dO, cuSeqQ, cuSeqK, batch, maxQ, maxK, numHeads, headDim, causal, dQ, dK, dV) {
        const L = "flashAttentionVarlenBackward(Q,K,V,O,dO,cuSeqQ,cuSeqK,batch,maxQ,maxK,numHeads,headDim,causal,dQ,dK,dV)";
        __bro_native.tensor.flashAttentionVarlenBackward(
            T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), T(O, L, "O"), T(dO, L, "dO"),
            opt(cuSeqQ, L, "cuSeqQ"), opt(cuSeqK, L, "cuSeqK"),
            int(batch, 0), int(maxQ, 0), int(maxK, 0), int(numHeads, 1), int(headDim, 0), !!causal,
            T(dQ, L, "dQ"), T(dK, L, "dK"), T(dV, L, "dV"));
        chk();
    });

    // ---- gated delta rule (linear attention — Qwen3-Next) ---------------------
    // Q/K: (L, numHeads*d_k).  V: (L, numHeads*d_v).  aRaw/beta: (L, numHeads)
    // FP32 raw gate inputs (softplus / sigmoid are applied inside the op).
    // logA: (numHeads, 1).  state: (numHeads, d_v*d_k) FP32, read AND updated in
    // place.  O: (L, numHeads*d_v).
    fn(ns_tensor, "gatedDeltaRuleChunked", function gatedDeltaRuleChunked(
            Q, K, V, aRaw, beta, logA, numHeads, d_k, d_v, state, O) {
        const L = "gatedDeltaRuleChunked(Q,K,V,aRaw,beta,logA,numHeads,d_k,d_v,state,O)";
        __bro_native.tensor.gatedDeltaRuleChunked(
            T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), T(aRaw, L, "aRaw"), T(beta, L, "beta"), T(logA, L, "logA"),
            int(numHeads, 1), int(d_k, 0), int(d_v, 0), T(state, L, "state"), T(O, L, "O"));
        chk();
    });
    // Same math for L_step new tokens against an existing state.
    fn(ns_tensor, "gatedDeltaRuleStep", function gatedDeltaRuleStep(
            Q, K, V, aRaw, beta, logA, numHeads, d_k, d_v, state, O) {
        const L = "gatedDeltaRuleStep(Q,K,V,aRaw,beta,logA,numHeads,d_k,d_v,state,O)";
        __bro_native.tensor.gatedDeltaRuleStep(
            T(Q, L, "Q"), T(K, L, "K"), T(V, L, "V"), T(aRaw, L, "aRaw"), T(beta, L, "beta"), T(logA, L, "logA"),
            int(numHeads, 1), int(d_k, 0), int(d_v, 0), T(state, L, "state"), T(O, L, "O"));
        chk();
    });

    // ---- M-RoPE (Qwen2.5-VL / Qwen3-VL multimodal rotary) ---------------------
    // headDim splits into three contiguous sub-ranges of widths 2*d_t, 2*d_h,
    // 2*d_w (in that order), each rotated by its own position stream.
    //   X, Y: (L, numHeads*headDim).   cos_a/sin_a: (maxPos_a, d_a) FP32.
    //   posT/posH/posW: length-L INT32 streams in a GpuTensor read as INT32
    //   storage, or null for an axis with d_a === 0.
    fn(ns_tensor, "ropeApplyMrope", function ropeApplyMrope(
            X, cosT, sinT, cosH, sinH, cosW, sinW, posT, posH, posW, headDim, numHeads, d_t, d_h, d_w, Y) {
        const L = "ropeApplyMrope(X,cosT,sinT,cosH,sinH,cosW,sinW,posT,posH,posW,headDim,numHeads,d_t,d_h,d_w,Y)";
        __bro_native.tensor.ropeApplyMrope(
            T(X, L, "X"), T(cosT, L, "cosT"), T(sinT, L, "sinT"), T(cosH, L, "cosH"), T(sinH, L, "sinH"),
            T(cosW, L, "cosW"), T(sinW, L, "sinW"),
            opt(posT, L, "posT"), opt(posH, L, "posH"), opt(posW, L, "posW"),
            int(headDim, 0), int(numHeads, 1), int(d_t, 0), int(d_h, 0), int(d_w, 0), T(Y, L, "Y"));
        chk();
    });
})();
