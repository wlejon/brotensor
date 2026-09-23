// tensor_extra.js — the brotensor ops that never had a JS binding (neither in
// the QuickJS binding nor in the bronze port). Compiled into its own module
// (bronze_tensor_extra_main) that api.cpp mounts last, so `bro.tensor` and
// `bro.tensor.GpuTensor` already exist here. Every wrapper checks its own
// arguments and reads the native's error back after the call.
//
// Argument order follows the C++ op (brotensor/ops/*.h), with the op's
// snake_case name in camelCase. The natives check every operand against the
// dims they are handed (native_tensor_extra*.cpp), so an undersized tensor is
// an Error here, never an out-of-bounds read.
//
// INT32 operands (indices, positions, sequence bounds, token grids) are
// GpuTensors holding INT32 — GpuTensor.prototype.uploadInt32 makes one — or,
// where the operand is only read, an FP32 tensor of whole numbers, which the
// native converts.
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
    const bool = (v, def) => (v === undefined || v === null ? def : !!v);
    // A counter-based RNG seed: BigInt or Number, passed through as-is.
    const seed = (v, label) => {
        if (v === undefined || v === null) return 0;
        if (typeof v === "number" || typeof v === "bigint") return v;
        throw new TypeError(label + ": seed must be a Number or BigInt");
    };

    // ---- INT32 storage ---------------------------------------------------------
    // uploadInt32(data): INT32 storage from an Int32Array or an array of ints.
    // Keeps the tensor's (rows, cols) when their product matches data.length,
    // else becomes (data.length, 1) — upload()'s rule.
    fn(GpuTensor.prototype, "uploadInt32", function uploadInt32(data) {
        if (data === undefined || data === null) throw new TypeError("uploadInt32: data is required");
        const src = data instanceof Int32Array ? data : Int32Array.from(data);
        __bro_native.tensor.uploadInt32(this, src);
        chk();
    });
    // downloadInt32() -> Int32Array of an INT32 tensor's values.
    fn(GpuTensor.prototype, "downloadInt32", function downloadInt32() {
        const out = __bro_native.tensor.downloadInt32(this);
        chk();
        return out;
    });

    // ---- bias adds + axpby -------------------------------------------------------
    fn(ns_tensor, "addChannelBiasInplace", function addChannelBiasInplace(y, bias, C, L) {
        const Lb = "addChannelBiasInplace(y,bias,C,L)";
        __bro_native.tensor.addChannelBiasInplace(T(y, Lb, "y"), T(bias, Lb, "bias"), int(C, 0), int(L, 0));
        chk();
    });
    fn(ns_tensor, "addRowBiasInplace", function addRowBiasInplace(Y, bias) {
        const Lb = "addRowBiasInplace(Y,bias)";
        __bro_native.tensor.addRowBiasInplace(T(Y, Lb, "Y"), T(bias, Lb, "bias"));
        chk();
    });
    fn(ns_tensor, "axpbyInplace", function axpbyInplace(y, x, a, b) {
        const Lb = "axpbyInplace(y,x,a,b)";
        __bro_native.tensor.axpbyInplace(T(y, Lb, "y"), T(x, Lb, "x"), num(a, 1), num(b, 1));
        chk();
    });

    // ---- sin / cos / rsqrt ---------------------------------------------------------
    // FP32 only. The backwards read the forward INPUT x, except rsqrt's, which
    // reads the forward OUTPUT y (dX = -0.5*dY*y^3).
    fn(ns_tensor, "sinForward", function sinForward(x, y) {
        const Lb = "sinForward(x,y)";
        __bro_native.tensor.sinForward(T(x, Lb, "x"), T(y, Lb, "y"));
        chk();
    });
    fn(ns_tensor, "sinBackward", function sinBackward(x, dY, dX) {
        const Lb = "sinBackward(x,dY,dX)";
        __bro_native.tensor.sinBackward(T(x, Lb, "x"), T(dY, Lb, "dY"), T(dX, Lb, "dX"));
        chk();
    });
    fn(ns_tensor, "cosForward", function cosForward(x, y) {
        const Lb = "cosForward(x,y)";
        __bro_native.tensor.cosForward(T(x, Lb, "x"), T(y, Lb, "y"));
        chk();
    });
    fn(ns_tensor, "cosBackward", function cosBackward(x, dY, dX) {
        const Lb = "cosBackward(x,dY,dX)";
        __bro_native.tensor.cosBackward(T(x, Lb, "x"), T(dY, Lb, "dY"), T(dX, Lb, "dX"));
        chk();
    });
    fn(ns_tensor, "rsqrtForward", function rsqrtForward(x, y) {
        const Lb = "rsqrtForward(x,y)";
        __bro_native.tensor.rsqrtForward(T(x, Lb, "x"), T(y, Lb, "y"));
        chk();
    });
    fn(ns_tensor, "rsqrtBackward", function rsqrtBackward(y, dY, dX) {
        const Lb = "rsqrtBackward(y,dY,dX)";
        __bro_native.tensor.rsqrtBackward(T(y, Lb, "y"), T(dY, Lb, "dY"), T(dX, Lb, "dX"));
        chk();
    });

    // ---- pixel norm ------------------------------------------------------------------
    fn(ns_tensor, "pixelNormForward", function pixelNormForward(X, eps, Y) {
        const Lb = "pixelNormForward(X,eps,Y)";
        __bro_native.tensor.pixelNormForward(T(X, Lb, "X"), num(eps, 1e-8), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "pixelNormBackward", function pixelNormBackward(X, dY, eps, dX) {
        const Lb = "pixelNormBackward(X,dY,eps,dX)";
        __bro_native.tensor.pixelNormBackward(T(X, Lb, "X"), T(dY, Lb, "dY"), num(eps, 1e-8), T(dX, Lb, "dX"));
        chk();
    });

    // ---- thresholds + counts -----------------------------------------------------------
    fn(ns_tensor, "thresholdU8", function thresholdU8(X, t, Y) {
        const Lb = "thresholdU8(X,t,Y)";
        __bro_native.tensor.thresholdU8(T(X, Lb, "X"), num(t, 0), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "rowsCountAbove", function rowsCountAbove(X, tLo, tHi, counts) {
        const Lb = "rowsCountAbove(X,tLo,tHi,counts)";
        __bro_native.tensor.rowsCountAbove(T(X, Lb, "X"), num(tLo, 0), num(tHi, 0), T(counts, Lb, "counts"));
        chk();
    });

    // ---- softmax -----------------------------------------------------------------------
    fn(ns_tensor, "softmaxRowsForward", function softmaxRowsForward(X, Y, rows, cols) {
        const Lb = "softmaxRowsForward(X,Y,rows,cols)";
        T(X, Lb, "X");
        __bro_native.tensor.softmaxRowsForward(X, T(Y, Lb, "Y"), int(rows, X.rows), int(cols, X.cols));
        chk();
    });
    fn(ns_tensor, "softmaxXent", function softmaxXent(logits, target, probs, dLogits, mask) {
        const Lb = "softmaxXent(logits,target,probs,dLogits,mask|null)";
        const loss = __bro_native.tensor.softmaxXent(T(logits, Lb, "logits"), T(target, Lb, "target"), T(probs, Lb, "probs"),
            T(dLogits, Lb, "dLogits"), opt(mask, Lb, "mask"));
        chk();
        return loss;
    });

    // ---- strided copy, row scatter, segment stats -----------------------------------------
    fn(ns_tensor, "copyD2DStrided", function copyD2DStrided(src, srcOff, srcPitch, dst, dstOff, dstPitch, width, height) {
        const Lb = "copyD2DStrided(src,srcOff,srcPitch,dst,dstOff,dstPitch,width,height)";
        __bro_native.tensor.copyD2DStrided(T(src, Lb, "src"), int(srcOff, 0), int(srcPitch, 0), T(dst, Lb, "dst"), int(dstOff, 0),
            int(dstPitch, 0), int(width, 0), int(height, 0));
        chk();
    });
    fn(ns_tensor, "scatterRows", function scatterRows(Y, Idx, X) {
        const Lb = "scatterRows(Y,Idx,X)";
        __bro_native.tensor.scatterRows(T(Y, Lb, "Y"), T(Idx, Lb, "Idx"), T(X, Lb, "X"));
        chk();
    });
    fn(ns_tensor, "segmentSoftmaxStats", function segmentSoftmaxStats(logits, segOffsets, out) {
        const Lb = "segmentSoftmaxStats(logits,segOffsets,out)";
        __bro_native.tensor.segmentSoftmaxStats(T(logits, Lb, "logits"), T(segOffsets, Lb, "segOffsets"), T(out, Lb, "out"));
        chk();
    });

    // ---- masked-diffusion token selection ----------------------------------------------------
    // maskedDiffusionScores(logits, tokens, T, C, V, maskId, opts, pred, scores, confidence)
    // opts: { guidanceScale=0, layerPenalty=0, positionTemperature=0,
    //         classTemperature=0, classTopFrac=1, seed=0 }
    fn(ns_tensor, "maskedDiffusionScores", function maskedDiffusionScores(logits, tokens, Tn, C, V, maskId, opts, pred, scores, confidence) {
        const Lb = "maskedDiffusionScores(logits,tokens,T,C,V,maskId,opts,pred,scores,confidence)";
        const o = opts === undefined || opts === null ? {} : opts;
        if (typeof o !== "object") throw new TypeError(Lb + ": opts must be an object or null");
        __bro_native.tensor.maskedDiffusionScores(T(logits, Lb, "logits"), T(tokens, Lb, "tokens"), int(Tn, 0), int(C, 0), int(V, 0),
            int(maskId, 0), num(o.guidanceScale, 0), num(o.layerPenalty, 0), num(o.positionTemperature, 0),
            num(o.classTemperature, 0), num(o.classTopFrac, 1), seed(o.seed, Lb),
            T(pred, Lb, "pred"), T(scores, Lb, "scores"), T(confidence, Lb, "confidence"));
        chk();
    });
    fn(ns_tensor, "maskedDiffusionCommit", function maskedDiffusionCommit(pred, idx, k, step, tokens, unmaskStep) {
        const Lb = "maskedDiffusionCommit(pred,idx,k,step,tokens,unmaskStep)";
        __bro_native.tensor.maskedDiffusionCommit(T(pred, Lb, "pred"), T(idx, Lb, "idx"), int(k, 0), int(step, 0),
            T(tokens, Lb, "tokens"), T(unmaskStep, Lb, "unmaskStep"));
        chk();
    });

    // ---- dense -----------------------------------------------------------------------------
    // The LinearActivation / LinearEpilogue values (brotensor/ops/linear.h).
    fn(ns_tensor, "LinearActivation", Object.freeze({ none: 0, relu: 1, geluTanh: 2, geluExact: 3, silu: 4, quickGelu: 5 }));
    fn(ns_tensor, "LinearEpilogue", Object.freeze({ store: 0, accumulate: 1, geglu: 2, fastAccum: 16 }));

    fn(ns_tensor, "linearForwardBatchedEx", function linearForwardBatchedEx(W, bias, X_BD, act, epilogue, workspace, Y_BD) {
        const Lb = "linearForwardBatchedEx(W,bias|null,X_BD,act,epilogue,workspace|null,Y_BD)";
        __bro_native.tensor.linearForwardBatchedEx(T(W, Lb, "W"), opt(bias, Lb, "bias"), T(X_BD, Lb, "X_BD"), int(act, 0),
            int(epilogue, 0), opt(workspace, Lb, "workspace"), T(Y_BD, Lb, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "linearForwardBatchedFp16Act", function linearForwardBatchedFp16Act(W, bias, X_BD, act, Y_BD) {
        const Lb = "linearForwardBatchedFp16Act(W,bias|null,X_BD,act,Y_BD)";
        __bro_native.tensor.linearForwardBatchedFp16Act(T(W, Lb, "W"), opt(bias, Lb, "bias"), T(X_BD, Lb, "X_BD"), int(act, 0),
            T(Y_BD, Lb, "Y_BD"));
        chk();
    });
    fn(ns_tensor, "matmulAbt", function matmulAbt(A, B, C, batch, M, Nn, K, strideA, strideB, strideC, bias, act) {
        const Lb = "matmulAbt(A,B,C,batch,M,N,K,strideA,strideB,strideC,bias|null,act)";
        const m = int(M, 0), n = int(Nn, 0), k = int(K, 0);
        __bro_native.tensor.matmulAbt(T(A, Lb, "A"), T(B, Lb, "B"), T(C, Lb, "C"), int(batch, 1), m, n, k,
            num(strideA, m * k), num(strideB, n * k), num(strideC, m * n), opt(bias, Lb, "bias"), int(act, 0));
        chk();
    });

    // ---- LSTM ------------------------------------------------------------------------------
    fn(ns_tensor, "lstmForwardTrain", function lstmForwardTrain(X, W_ih, W_hh, b_ih, b_hh, h0, c0, Tn, B, Y, gates, C, hT, cT) {
        const Lb = "lstmForwardTrain(X,W_ih,W_hh,b_ih|null,b_hh|null,h0|null,c0|null,T,B,Y,gates,C,hT|null,cT|null)";
        __bro_native.tensor.lstmForwardTrain(T(X, Lb, "X"), T(W_ih, Lb, "W_ih"), T(W_hh, Lb, "W_hh"), opt(b_ih, Lb, "b_ih"),
            opt(b_hh, Lb, "b_hh"), opt(h0, Lb, "h0"), opt(c0, Lb, "c0"), int(Tn, 0), int(B, 0),
            T(Y, Lb, "Y"), T(gates, Lb, "gates"), T(C, Lb, "C"), opt(hT, Lb, "hT"), opt(cT, Lb, "cT"));
        chk();
    });
    fn(ns_tensor, "lstmBackward", function lstmBackward(X, W_ih, W_hh, h0, c0, Y, gates, C, dY, Tn, B, dX, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0) {
        const Lb = "lstmBackward(X,W_ih,W_hh,h0|null,c0|null,Y,gates,C,dY,T,B,dX,dW_ih,dW_hh,db_ih|null,db_hh|null,dh0|null,dc0|null)";
        __bro_native.tensor.lstmBackward(T(X, Lb, "X"), T(W_ih, Lb, "W_ih"), T(W_hh, Lb, "W_hh"), opt(h0, Lb, "h0"), opt(c0, Lb, "c0"),
            T(Y, Lb, "Y"), T(gates, Lb, "gates"), T(C, Lb, "C"), T(dY, Lb, "dY"), int(Tn, 0), int(B, 0),
            T(dX, Lb, "dX"), T(dW_ih, Lb, "dW_ih"), T(dW_hh, Lb, "dW_hh"), opt(db_ih, Lb, "db_ih"),
            opt(db_hh, Lb, "db_hh"), opt(dh0, Lb, "dh0"), opt(dc0, Lb, "dc0"));
        chk();
    });

    // ---- attention + RoPE ----------------------------------------------------------------------
    fn(ns_tensor, "flashAttentionGqaForward", function flashAttentionGqaForward(Q, K, V, mask, numQHeads, numKvHeads, causal, O) {
        const Lb = "flashAttentionGqaForward(Q,K,V,mask|null,numQHeads,numKvHeads,causal,O)";
        const nq = int(numQHeads, 1);
        __bro_native.tensor.flashAttentionGqaForward(T(Q, Lb, "Q"), T(K, Lb, "K"), T(V, Lb, "V"), opt(mask, Lb, "mask"), nq,
            int(numKvHeads, nq), bool(causal, false), T(O, Lb, "O"));
        chk();
    });
    fn(ns_tensor, "flashAttentionPackedQkvForward", function flashAttentionPackedQkvForward(QKV, seqBounds, numHeads, window, O) {
        const Lb = "flashAttentionPackedQkvForward(QKV,seqBounds,numHeads,window,O)";
        __bro_native.tensor.flashAttentionPackedQkvForward(T(QKV, Lb, "QKV"), T(seqBounds, Lb, "seqBounds"), int(numHeads, 1),
            int(window, 0), T(O, Lb, "O"));
        chk();
    });
    fn(ns_tensor, "flashAttentionPackedQkvBackward", function flashAttentionPackedQkvBackward(QKV, dO, seqBounds, numHeads, window, dQKV) {
        const Lb = "flashAttentionPackedQkvBackward(QKV,dO,seqBounds,numHeads,window,dQKV)";
        __bro_native.tensor.flashAttentionPackedQkvBackward(T(QKV, Lb, "QKV"), T(dO, Lb, "dO"), T(seqBounds, Lb, "seqBounds"),
            int(numHeads, 1), int(window, 0), T(dQKV, Lb, "dQKV"));
        chk();
    });
    fn(ns_tensor, "relPosBiasXlForward", function relPosBiasXlForward(Qv, Pk, numHeads, headDim, Bias) {
        const Lb = "relPosBiasXlForward(Qv,Pk,numHeads,headDim,Bias)";
        __bro_native.tensor.relPosBiasXlForward(T(Qv, Lb, "Qv"), T(Pk, Lb, "Pk"), int(numHeads, 1), int(headDim, 0), T(Bias, Lb, "Bias"));
        chk();
    });
    fn(ns_tensor, "ropeApplyPerhead", function ropeApplyPerhead(X, cosTbl, sinTbl, headDim, numHeads, Y) {
        const Lb = "ropeApplyPerhead(X,cosTbl,sinTbl,headDim,numHeads,Y)";
        __bro_native.tensor.ropeApplyPerhead(T(X, Lb, "X"), T(cosTbl, Lb, "cosTbl"), T(sinTbl, Lb, "sinTbl"), int(headDim, 0),
            int(numHeads, 1), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "ropeQkvPackedInplace", function ropeQkvPackedInplace(QKV, cosTbl, sinTbl, pos, numHeads, headDim) {
        const Lb = "ropeQkvPackedInplace(QKV,cosTbl,sinTbl,pos,numHeads,headDim)";
        __bro_native.tensor.ropeQkvPackedInplace(T(QKV, Lb, "QKV"), T(cosTbl, Lb, "cosTbl"), T(sinTbl, Lb, "sinTbl"),
            T(pos, Lb, "pos"), int(numHeads, 1), int(headDim, 0));
        chk();
    });

    // ---- StyleGAN3-R ---------------------------------------------------------------------------
    fn(ns_tensor, "biasActForward", function biasActForward(X, b, Nn, C, HW, act, alpha, gain, clamp, Y) {
        const Lb = "biasActForward(X,b|null,N,C,HW,act,alpha,gain,clamp,Y)";
        const a = int(act, 0);
        __bro_native.tensor.biasActForward(T(X, Lb, "X"), opt(b, Lb, "b"), int(Nn, 0), int(C, 0), int(HW, 0), a, num(alpha, 0.2),
            num(gain, a === 1 ? Math.SQRT2 : 1), num(clamp, -1), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "biasActBackward", function biasActBackward(dY, X, b, Nn, C, HW, act, alpha, gain, clamp, dX, dB) {
        const Lb = "biasActBackward(dY,X,b|null,N,C,HW,act,alpha,gain,clamp,dX,dB|null)";
        const a = int(act, 0);
        __bro_native.tensor.biasActBackward(T(dY, Lb, "dY"), T(X, Lb, "X"), opt(b, Lb, "b"), int(Nn, 0), int(C, 0), int(HW, 0), a,
            num(alpha, 0.2), num(gain, a === 1 ? Math.SQRT2 : 1), num(clamp, -1), T(dX, Lb, "dX"), opt(dB, Lb, "dB"));
        chk();
    });
    // upfirdn2d{Forward,Backward}(X|dY, f, N, C, H, W, fH, fW, upX, upY, downX,
    // downY, padX0, padX1, padY0, padY1, flipFilter, gain, Y|dX). fH / fW
    // default to f's shape; H, W are the forward INPUT dims in both.
    fn(ns_tensor, "upfirdn2dForward", function upfirdn2dForward(X, f, Nn, C, H, W, fH, fW, upX, upY, downX, downY,
                                                                padX0, padX1, padY0, padY1, flipFilter, gain, Y) {
        const Lb = "upfirdn2dForward(X,f,N,C,H,W,fH,fW,upX,upY,downX,downY,padX0,padX1,padY0,padY1,flipFilter,gain,Y)";
        T(f, Lb, "f");
        __bro_native.tensor.upfirdn2dForward(T(X, Lb, "X"), f, int(Nn, 0), int(C, 0), int(H, 0), int(W, 0), int(fH, f.rows),
            int(fW, f.cols), int(upX, 1), int(upY, 1), int(downX, 1), int(downY, 1), int(padX0, 0), int(padX1, 0),
            int(padY0, 0), int(padY1, 0), bool(flipFilter, false), num(gain, 1), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "upfirdn2dBackward", function upfirdn2dBackward(dY, f, Nn, C, H, W, fH, fW, upX, upY, downX, downY,
                                                                  padX0, padX1, padY0, padY1, flipFilter, gain, dX) {
        const Lb = "upfirdn2dBackward(dY,f,N,C,H,W,fH,fW,upX,upY,downX,downY,padX0,padX1,padY0,padY1,flipFilter,gain,dX)";
        T(f, Lb, "f");
        __bro_native.tensor.upfirdn2dBackward(T(dY, Lb, "dY"), f, int(Nn, 0), int(C, 0), int(H, 0), int(W, 0), int(fH, f.rows),
            int(fW, f.cols), int(upX, 1), int(upY, 1), int(downX, 1), int(downY, 1), int(padX0, 0), int(padX1, 0),
            int(padY0, 0), int(padY1, 0), bool(flipFilter, false), num(gain, 1), T(dX, Lb, "dX"));
        chk();
    });

    fn(ns_tensor, "modulatedConv2dForward", function modulatedConv2dForward(X, W, s, Nn, C_in, H, Wd, C_out, kH, kW, padH, padW, demodulate, eps, dcoef, Y) {
        const Lb = "modulatedConv2dForward(X,W,s,N,C_in,H,W,C_out,kH,kW,padH,padW,demodulate,eps,dcoef,Y)";
        __bro_native.tensor.modulatedConv2dForward(T(X, Lb, "X"), T(W, Lb, "W"), T(s, Lb, "s"), int(Nn, 0), int(C_in, 0), int(H, 0),
            int(Wd, 0), int(C_out, 0), int(kH, 1), int(kW, 1), int(padH, 0), int(padW, 0), bool(demodulate, true),
            num(eps, 1e-8), T(dcoef, Lb, "dcoef"), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "modulatedConv2dBackward", function modulatedConv2dBackward(X, W, s, dcoef, dY, Nn, C_in, H, Wd, C_out, kH, kW, padH, padW, demodulate, eps, dX, dW, ds) {
        const Lb = "modulatedConv2dBackward(X,W,s,dcoef,dY,N,C_in,H,W,C_out,kH,kW,padH,padW,demodulate,eps,dX,dW|null,ds)";
        __bro_native.tensor.modulatedConv2dBackward(T(X, Lb, "X"), T(W, Lb, "W"), T(s, Lb, "s"), T(dcoef, Lb, "dcoef"), T(dY, Lb, "dY"),
            int(Nn, 0), int(C_in, 0), int(H, 0), int(Wd, 0), int(C_out, 0), int(kH, 1), int(kW, 1), int(padH, 0),
            int(padW, 0), bool(demodulate, true), num(eps, 1e-8), T(dX, Lb, "dX"), opt(dW, Lb, "dW"), T(ds, Lb, "ds"));
        chk();
    });
    fn(ns_tensor, "filteredLreluForward", function filteredLreluForward(X, fu, fd, b, Nn, C, H, W, up, down, padX0, padX1, padY0, padY1, gain, slope, clamp, upBuf, actBuf, Y) {
        const Lb = "filteredLreluForward(X,fu,fd,b|null,N,C,H,W,up,down,padX0,padX1,padY0,padY1,gain,slope,clamp,upBuf,actBuf,Y)";
        __bro_native.tensor.filteredLreluForward(T(X, Lb, "X"), T(fu, Lb, "fu"), T(fd, Lb, "fd"), opt(b, Lb, "b"), int(Nn, 0), int(C, 0),
            int(H, 0), int(W, 0), int(up, 1), int(down, 1), int(padX0, 0), int(padX1, 0), int(padY0, 0), int(padY1, 0),
            num(gain, Math.SQRT2), num(slope, 0.2), num(clamp, -1), T(upBuf, Lb, "upBuf"), T(actBuf, Lb, "actBuf"),
            T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "filteredLreluBackward", function filteredLreluBackward(dY, X, fu, fd, b, Nn, C, H, W, up, down, padX0, padX1, padY0, padY1, gain, slope, clamp, upBuf, dX, dB) {
        const Lb = "filteredLreluBackward(dY,X,fu,fd,b|null,N,C,H,W,up,down,padX0,padX1,padY0,padY1,gain,slope,clamp,upBuf|null,dX,dB|null)";
        __bro_native.tensor.filteredLreluBackward(T(dY, Lb, "dY"), T(X, Lb, "X"), T(fu, Lb, "fu"), T(fd, Lb, "fd"), opt(b, Lb, "b"),
            int(Nn, 0), int(C, 0), int(H, 0), int(W, 0), int(up, 1), int(down, 1), int(padX0, 0), int(padX1, 0),
            int(padY0, 0), int(padY1, 0), num(gain, Math.SQRT2), num(slope, 0.2), num(clamp, -1),
            opt(upBuf, Lb, "upBuf"), T(dX, Lb, "dX"), opt(dB, Lb, "dB"));
        chk();
    });

    // ---- deformable conv, pixel shuffle, unpatchify ------------------------------------------------
    fn(ns_tensor, "deformConv2dForward", function deformConv2dForward(X, offset, mask, Wt, bias, Nn, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, deformGroups, Y) {
        const Lb = "deformConv2dForward(X,offset,mask|null,Wt,bias|null,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,dH,dW,groups,deformGroups,Y)";
        __bro_native.tensor.deformConv2dForward(T(X, Lb, "X"), T(offset, Lb, "offset"), opt(mask, Lb, "mask"), T(Wt, Lb, "Wt"),
            opt(bias, Lb, "bias"), int(Nn, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0), int(kH, 1), int(kW, 1),
            int(sH, 1), int(sW, 1), int(pH, 0), int(pW, 0), int(dH, 1), int(dW, 1), int(groups, 1),
            int(deformGroups, 1), T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "pixelShuffleUpsample2xForward", function pixelShuffleUpsample2xForward(X, Nn, C_in, H, W, C_out, Y) {
        const Lb = "pixelShuffleUpsample2xForward(X,N,C_in,H,W,C_out,Y)";
        __bro_native.tensor.pixelShuffleUpsample2xForward(T(X, Lb, "X"), int(Nn, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0),
            T(Y, Lb, "Y"));
        chk();
    });
    fn(ns_tensor, "patchUnpackForward", function patchUnpackForward(tokens, hp, wp, P, C_total, C_keep, channelMajor, Y) {
        const Lb = "patchUnpackForward(tokens,hp,wp,P,C_total,C_keep,channelMajor,Y)";
        const ct = int(C_total, 0);
        __bro_native.tensor.patchUnpackForward(T(tokens, Lb, "tokens"), int(hp, 0), int(wp, 0), int(P, 1), ct, int(C_keep, ct),
            bool(channelMajor, false), T(Y, Lb, "Y"));
        chk();
    });
})();
