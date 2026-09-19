// tensor_int8.js — the INT8 / GGUF-quant slice of bro.tensor: the W8A16
// family (an INT8 weight matrix plus a per-output-row FP32 dequant scale, FP16
// activations) and the GGUF k-quant family (Q4_K / Q6_K / Q8_0 weights).
// Hand-written like js/tensor_ext.js and compiled into its own module
// (bronze_tensor_int8_main) that api.cpp mounts after js/tensor.js, so
// `bro.tensor` and `bro.tensor.GpuTensor` already exist here.
//
// Signatures are the QuickJS binding's (tensor_bindings_int8.cpp): the
// argument order, the optional slots and their defaults are unchanged, and the
// two entry points that took an options object (resblockForwardInt8wFp16,
// flashAttentionQkvoInt8wFp16) keep that object form. Every wrapper checks its
// arguments here — the natives receive the nullable slots as raw values — and
// reads back the native's error after the call.
//
// The whole group is GPU-only: the CPU backend leaves the FP16 / INT8-W8A16 /
// GGUF-quant vtable slots null, so these throw "not implemented on CPU" there.
// quantizeInt8PerRowHost is the one exception — a pure host helper over typed
// arrays, with no device involved.
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

    // ---- GGUF k-quant dequant (Q4_K / Q6_K / Q8_0 -> FP16) --------------------
    // A one-shot dequant of a block-quantised weight, for tests and for callers
    // that reuse one FP16 weight across many matmuls.
    fn(ns_tensor, "dequantQ4kToFp16", function dequantQ4kToFp16(W_q, W_fp16) {
        const L = "dequantQ4kToFp16(W_q4k, W_fp16)";
        __bro_native.tensor.dequantQ4kToFp16(T(W_q, L, "W_q4k"), T(W_fp16, L, "W_fp16"));
        chk();
    });
    fn(ns_tensor, "dequantQ6kToFp16", function dequantQ6kToFp16(W_q, W_fp16) {
        const L = "dequantQ6kToFp16(W_q6k, W_fp16)";
        __bro_native.tensor.dequantQ6kToFp16(T(W_q, L, "W_q6k"), T(W_fp16, L, "W_fp16"));
        chk();
    });
    fn(ns_tensor, "dequantQ8_0ToFp16", function dequantQ8_0ToFp16(W_q, W_fp16) {
        const L = "dequantQ8_0ToFp16(W_q8_0, W_fp16)";
        __bro_native.tensor.dequantQ8_0ToFp16(T(W_q, L, "W_q8_0"), T(W_fp16, L, "W_fp16"));
        chk();
    });

    // ---- GGUF k-quant weight-only linear -------------------------------------
    // Single token: x is (in,1) FP16, y is (out,1) FP16, bias optional.
    fn(ns_tensor, "linearForwardQ4kFp16", function linearForwardQ4kFp16(W_q, bias, x, y) {
        const L = "linearForwardQ4kFp16(W_q4k,bias|null,x,y)";
        __bro_native.tensor.linearForwardQ4kFp16(T(W_q, L, "W_q4k"), opt(bias, L, "bias"), T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "linearForwardQ6kFp16", function linearForwardQ6kFp16(W_q, bias, x, y) {
        const L = "linearForwardQ6kFp16(W_q6k,bias|null,x,y)";
        __bro_native.tensor.linearForwardQ6kFp16(T(W_q, L, "W_q6k"), opt(bias, L, "bias"), T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "linearForwardQ8_0Fp16", function linearForwardQ8_0Fp16(W_q, bias, x, y) {
        const L = "linearForwardQ8_0Fp16(W_q8_0,bias|null,x,y)";
        __bro_native.tensor.linearForwardQ8_0Fp16(T(W_q, L, "W_q8_0"), opt(bias, L, "bias"), T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    // Batched: X_BD is (B,in) FP16, Y_BD is (B,out) FP16.
    fn(ns_tensor, "linearForwardBatchedQ4kFp16", function linearForwardBatchedQ4kFp16(W_q, bias, X_BD, Y_BD) {
        const L = "linearForwardBatchedQ4kFp16(W_q4k,bias|null,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatchedQ4kFp16(T(W_q, L, "W_q4k"), opt(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "linearForwardBatchedQ6kFp16", function linearForwardBatchedQ6kFp16(W_q, bias, X_BD, Y_BD) {
        const L = "linearForwardBatchedQ6kFp16(W_q6k,bias|null,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatchedQ6kFp16(T(W_q, L, "W_q6k"), opt(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "linearForwardBatchedQ8_0Fp16", function linearForwardBatchedQ8_0Fp16(W_q, bias, X_BD, Y_BD) {
        const L = "linearForwardBatchedQ8_0Fp16(W_q8_0,bias|null,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatchedQ8_0Fp16(T(W_q, L, "W_q8_0"), opt(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });

    // ---- W8A16 host quantiser -------------------------------------------------
    // quantizeInt8PerRowHost(W_fp16:Uint16Array, out, in) ->
    //   { weights: Int8Array (out*in), scales: Float32Array (out) }
    // Pure host work — no device, no GpuTensor. Callers prepare W8A16 weights
    // with it once at load time. scale[row] = max(|w|)/127 (0 for an all-zero
    // row); quantised w = clamp(round(w/scale), -127, 127). Both typed arrays
    // are fresh copies. The native computes both halves in one pass and hands
    // back the weights; the scales come back through the companion native.
    fn(ns_tensor, "quantizeInt8PerRowHost", function quantizeInt8PerRowHost(W_fp16, out, inDim) {
        const L = "quantizeInt8PerRowHost(W_fp16:Uint16Array, out, in)";
        if (!(W_fp16 instanceof Uint16Array)) {
            throw new TypeError(L + ": W_fp16 must be a Uint16Array (binary16 bit pattern)");
        }
        const o = int(out, 0);
        const i = int(inDim, 0);
        if (o <= 0 || i <= 0) throw new RangeError(L + ": out, in must be > 0");
        if (W_fp16.length < o * i) throw new RangeError(L + ": W_fp16 view too small for (out, in)");
        const weights = __bro_native.tensor.quantizeInt8PerRowHost(W_fp16, o, i);
        chk();
        const scales = __bro_native.tensor.quantizeInt8PerRowHostScales();
        chk();
        return { weights: weights, scales: scales };
    });

    // ---- W8A16 dense ----------------------------------------------------------
    // Y(out,B) = dequant(W_int8, scales)(out,in) @ X(in,B); X / Y FP16.
    fn(ns_tensor, "matmulInt8wFp16", function matmulInt8wFp16(W_int8, scales, X, Y) {
        const L = "matmulInt8wFp16(W_int8,scales,X,Y)";
        __bro_native.tensor.matmulInt8wFp16(T(W_int8, L, "W_int8"), T(scales, L, "scales"), T(X, L, "X"), T(Y, L, "Y"));
        chk();
    });
    // Y_BD(B,out) = X_BD(B,in) @ dequant(W_int8)^T + bias.
    fn(ns_tensor, "linearForwardBatchedInt8wFp16", function linearForwardBatchedInt8wFp16(W_int8, scales, bias, X_BD, Y_BD) {
        const L = "linearForwardBatchedInt8wFp16(W_int8,scales,bias|null,X_BD,Y_BD)";
        __bro_native.tensor.linearForwardBatchedInt8wFp16(T(W_int8, L, "W_int8"), T(scales, L, "scales"),
            opt(bias, L, "bias"), T(X_BD, L, "X_BD"), T(Y_BD, L, "Y_BD"));
        chk();
    });

    // ---- W8A16 convolution ----------------------------------------------------
    fn(ns_tensor, "conv2dInt8wFp16Forward", function conv2dInt8wFp16Forward(X, W_int8, scales, bias, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, Y) {
        const L = "conv2dInt8wFp16Forward(X,W_int8,scales,bias|null,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,dH,dW,groups,Y)";
        __bro_native.tensor.conv2dInt8wFp16Forward(T(X, L, "X"), T(W_int8, L, "W_int8"), T(scales, L, "scales"),
            opt(bias, L, "bias"), int(N, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0), int(kH, 0), int(kW, 0),
            int(sH, 1), int(sW, 1), int(pH, 0), int(pW, 0), int(dH, 1), int(dW, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });
    // NCTHW / OICTHW, the Qwen3-VL patch-embed variant of conv3dForward.
    fn(ns_tensor, "conv3dInt8wFp16Forward", function conv3dInt8wFp16Forward(X, W_int8, scales, bias, N, C_in, T_in, H, W, C_out, kT, kH, kW, sT, sH, sW, pT, pH, pW, dT, dH, dW, groups, Y) {
        const L = "conv3dInt8wFp16Forward(X,W_int8,scales,bias|null,N,C_in,T,H,W,C_out,kT,kH,kW,sT,sH,sW,pT,pH,pW,dT,dH,dW,groups,Y)";
        __bro_native.tensor.conv3dInt8wFp16Forward(T(X, L, "X"), T(W_int8, L, "W_int8"), T(scales, L, "scales"),
            opt(bias, L, "bias"), int(N, 0), int(C_in, 0), int(T_in, 0), int(H, 0), int(W, 0), int(C_out, 0),
            int(kT, 0), int(kH, 0), int(kW, 0), int(sT, 1), int(sH, 1), int(sW, 1),
            int(pT, 0), int(pH, 0), int(pW, 0), int(dT, 1), int(dH, 1), int(dW, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });

    // ---- W8A16 diffusion resblock (options object) ----------------------------
    // resblockForwardInt8wFp16(opts): required X, gamma1, beta1, W1_int8, s1,
    // gamma2, beta2, W2_int8, s2, Y, N, C_in, C_out, H, W; optional b1,
    // t_emb_shift, b2, Wskip_int8, sskip, bskip, numGroups (32), eps (1e-5).
    fn(ns_tensor, "resblockForwardInt8wFp16", function resblockForwardInt8wFp16(o) {
        const L = "resblockForwardInt8wFp16(opts)";
        if (o === null || typeof o !== "object") throw new TypeError(L + " — see docs");
        __bro_native.tensor.resblockForwardInt8wFp16(T(o.X, L, "X"), T(o.gamma1, L, "gamma1"), T(o.beta1, L, "beta1"),
            T(o.W1_int8, L, "W1_int8"), T(o.s1, L, "s1"), opt(o.b1, L, "b1"), opt(o.t_emb_shift, L, "t_emb_shift"),
            T(o.gamma2, L, "gamma2"), T(o.beta2, L, "beta2"), T(o.W2_int8, L, "W2_int8"), T(o.s2, L, "s2"),
            opt(o.b2, L, "b2"), opt(o.Wskip_int8, L, "Wskip_int8"), opt(o.sskip, L, "sskip"), opt(o.bskip, L, "bskip"),
            int(o.N, 0), int(o.C_in, 0), int(o.C_out, 0), int(o.H, 0), int(o.W, 0), int(o.numGroups, 32),
            num(o.eps, 0.00001), T(o.Y, L, "Y"));
        chk();
    });

    // ---- W8A16 flash-attention triplet ----------------------------------------
    fn(ns_tensor, "flashAttentionProjectKvInt8wFp16", function flashAttentionProjectKvInt8wFp16(ctx, Wk_int8, sk, bk, Wv_int8, sv, bv, K_out, V_out) {
        const L = "flashAttentionProjectKvInt8wFp16(ctx,Wk_int8,sk,bk|null,Wv_int8,sv,bv|null,K_out,V_out)";
        __bro_native.tensor.flashAttentionProjectKvInt8wFp16(T(ctx, L, "ctx"), T(Wk_int8, L, "Wk_int8"), T(sk, L, "sk"),
            opt(bk, L, "bk"), T(Wv_int8, L, "Wv_int8"), T(sv, L, "sv"), opt(bv, L, "bv"),
            T(K_out, L, "K_out"), T(V_out, L, "V_out"));
        chk();
    });
    fn(ns_tensor, "flashAttentionQWithKvCachedInt8wFp16", function flashAttentionQWithKvCachedInt8wFp16(X, K, V, Wq_int8, sq, bq, Wo_int8, so, bo, mask, numHeads, causal, O) {
        const L = "flashAttentionQWithKvCachedInt8wFp16(X,K,V,Wq_int8,sq,bq|null,Wo_int8,so,bo|null,mask|null,numHeads,causal,O)";
        __bro_native.tensor.flashAttentionQWithKvCachedInt8wFp16(T(X, L, "X"), T(K, L, "K"), T(V, L, "V"),
            T(Wq_int8, L, "Wq_int8"), T(sq, L, "sq"), opt(bq, L, "bq"),
            T(Wo_int8, L, "Wo_int8"), T(so, L, "so"), opt(bo, L, "bo"),
            opt(mask, L, "mask"), int(numHeads, 1), !!causal, T(O, L, "O"));
        chk();
    });
    // flashAttentionQkvoInt8wFp16(opts): required X, Wq_int8, sq, Wk_int8, sk,
    // Wv_int8, sv, Wo_int8, so, O; optional Ctx, bq, bk, bv, bo, mask,
    // numHeads (1), causal (false).
    fn(ns_tensor, "flashAttentionQkvoInt8wFp16", function flashAttentionQkvoInt8wFp16(o) {
        const L = "flashAttentionQkvoInt8wFp16(opts)";
        if (o === null || typeof o !== "object") throw new TypeError(L + " — see docs");
        __bro_native.tensor.flashAttentionQkvoInt8wFp16(T(o.X, L, "X"), opt(o.Ctx, L, "Ctx"),
            T(o.Wq_int8, L, "Wq_int8"), T(o.sq, L, "sq"), opt(o.bq, L, "bq"),
            T(o.Wk_int8, L, "Wk_int8"), T(o.sk, L, "sk"), opt(o.bk, L, "bk"),
            T(o.Wv_int8, L, "Wv_int8"), T(o.sv, L, "sv"), opt(o.bv, L, "bv"),
            T(o.Wo_int8, L, "Wo_int8"), T(o.so, L, "so"), opt(o.bo, L, "bo"),
            opt(o.mask, L, "mask"), int(o.numHeads, 1), !!o.causal, T(o.O, L, "O"));
        chk();
    });

    // ---- W8A16 T5-style bias self-attention -----------------------------------
    // The quantised twin of selfAttentionBiasForward: each (D,D) INT8 weight
    // carries an FP32 (D,1) per-output-row scale, activations stay FP16.
    // attnBias is an optional (numHeads*L, L) FP32 additive pre-softmax bias;
    // scale multiplies QK before it (1.0 for T5).
    fn(ns_tensor, "selfAttentionBiasInt8wFp16", function selfAttentionBiasInt8wFp16(X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv, Wo_int8, so, mask, attnBias, numHeads, scale, O) {
        const L = "selfAttentionBiasInt8wFp16(X,Wq_int8,sq,Wk_int8,sk,Wv_int8,sv,Wo_int8,so,mask|null,attnBias|null,numHeads,scale,O)";
        __bro_native.tensor.selfAttentionBiasInt8wFp16(T(X, L, "X"),
            T(Wq_int8, L, "Wq_int8"), T(sq, L, "sq"), T(Wk_int8, L, "Wk_int8"), T(sk, L, "sk"),
            T(Wv_int8, L, "Wv_int8"), T(sv, L, "sv"), T(Wo_int8, L, "Wo_int8"), T(so, L, "so"),
            opt(mask, L, "mask"), opt(attnBias, L, "attnBias"), int(numHeads, 1), num(scale, 1.0), T(O, L, "O"));
        chk();
    });
})();
