// tensor_conv.js — the restored 2-D spatial family of bro.tensor: interp2d
// backward, pad2d / slice2d, max-pool + adaptive-avg-pool, transposed conv,
// conv3d, the Swin/SAM/Qwen-VL window + patch-merge gathers, batch-norm
// (train / backward / inference) and the two image-preprocessing helpers.
//
// Signatures are the QuickJS binding's (tensor_bindings_conv.cpp): the same
// positional order, the same `bias|null` slots and the same defaults for the
// stride / pad / dilation / groups tail (stride 1, pad 0, output-pad 0,
// dilation 1, groups 1). Compiled into its own module
// (bronze_tensor_conv_main) and mounted after js/tensor.js + js/tensor_ext.js,
// so `bro.tensor` and `bro.tensor.GpuTensor` already exist here.
//
// Every wrapper validates its arguments (the natives take nullable slots as
// raw values) and reads the native error slot back after the call.
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
    // A Uint8Array of packed image bytes (a plain array of byte values is
    // accepted and copied, the way the old binding's typed-array read was).
    const u8 = (v, label, name) => {
        if (v instanceof Uint8Array) return v;
        if (Array.isArray(v)) return Uint8Array.from(v);
        throw new TypeError(label + ": " + name + " must be a Uint8Array");
    };
    const int = (v, def) => (v === undefined || v === null ? def : (v | 0));
    const num = (v, def) => (v === undefined || v === null ? def : +v);

    // ---- interp2d backward ---------------------------------------------------
    // The adjoint of interp2dForward; mode 0 nearest / 1 bilinear (bicubic has
    // no backward). N, C, H_in, W_in, H_out, W_out, mode match the forward call.
    fn(ns_tensor, "interp2dBackward", function interp2dBackward(dY, N, C, H_in, W_in, H_out, W_out, mode, dX) {
        const L = "interp2dBackward(dY,N,C,H_in,W_in,H_out,W_out,mode,dX)";
        __bro_native.tensor.interp2dBackward(T(dY, L, "dY"), int(N, 0), int(C, 0), int(H_in, 0), int(W_in, 0),
            int(H_out, 0), int(W_out, 0), int(mode, 1), T(dX, L, "dX"));
        chk();
    });

    // ---- pad2d / slice2d -----------------------------------------------------
    // mode: 0 zero, 1 reflect, 2 replicate.
    fn(ns_tensor, "pad2dForward", function pad2dForward(X, N, C, H, W, padT, padB, padL, padR, mode, Y) {
        const L = "pad2dForward(X,N,C,H,W,padT,padB,padL,padR,mode,Y)";
        __bro_native.tensor.pad2dForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(padT, 0), int(padB, 0), int(padL, 0), int(padR, 0), int(mode, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "pad2dBackward", function pad2dBackward(dY, N, C, H, W, padT, padB, padL, padR, mode, dX) {
        const L = "pad2dBackward(dY,N,C,H,W,padT,padB,padL,padR,mode,dX)";
        __bro_native.tensor.pad2dBackward(T(dY, L, "dY"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(padT, 0), int(padB, 0), int(padL, 0), int(padR, 0), int(mode, 0), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "slice2dForward", function slice2dForward(X, N, C, H, W, h0, w0, H_out, W_out, Y) {
        const L = "slice2dForward(X,N,C,H,W,h0,w0,H_out,W_out,Y)";
        __bro_native.tensor.slice2dForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(h0, 0), int(w0, 0), int(H_out, 0), int(W_out, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "slice2dBackward", function slice2dBackward(dY, N, C, H, W, h0, w0, H_out, W_out, dX) {
        const L = "slice2dBackward(dY,N,C,H,W,h0,w0,H_out,W_out,dX)";
        __bro_native.tensor.slice2dBackward(T(dY, L, "dY"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(h0, 0), int(w0, 0), int(H_out, 0), int(W_out, 0), T(dX, L, "dX"));
        chk();
    });

    // ---- max pool / adaptive average pool ------------------------------------
    // Idx is an INT32 GpuTensor shaped like Y; it carries the winning input
    // offsets into maxPool2dBackward.
    fn(ns_tensor, "maxPool2dForward", function maxPool2dForward(X, N, C, H, W, kH, kW, sH, sW, padH, padW, Y, Idx) {
        const L = "maxPool2dForward(X,N,C,H,W,kH,kW,sH,sW,padH,padW,Y,Idx)";
        __bro_native.tensor.maxPool2dForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(kH, 0), int(kW, 0), int(sH, 1), int(sW, 1), int(padH, 0), int(padW, 0),
            T(Y, L, "Y"), T(Idx, L, "Idx"));
        chk();
    });
    fn(ns_tensor, "maxPool2dBackward", function maxPool2dBackward(dY, Idx, N, C, H, W, H_out, W_out, dX) {
        const L = "maxPool2dBackward(dY,Idx,N,C,H,W,H_out,W_out,dX)";
        __bro_native.tensor.maxPool2dBackward(T(dY, L, "dY"), T(Idx, L, "Idx"), int(N, 0), int(C, 0),
            int(H, 0), int(W, 0), int(H_out, 0), int(W_out, 0), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "adaptiveAvgPool2dForward", function adaptiveAvgPool2dForward(X, N, C, H, W, H_out, W_out, Y) {
        const L = "adaptiveAvgPool2dForward(X,N,C,H,W,H_out,W_out,Y)";
        __bro_native.tensor.adaptiveAvgPool2dForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(H_out, 0), int(W_out, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "adaptiveAvgPool2dBackward", function adaptiveAvgPool2dBackward(dY, N, C, H, W, H_out, W_out, dX) {
        const L = "adaptiveAvgPool2dBackward(dY,N,C,H,W,H_out,W_out,dX)";
        __bro_native.tensor.adaptiveAvgPool2dBackward(T(dY, L, "dY"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(H_out, 0), int(W_out, 0), T(dX, L, "dX"));
        chk();
    });

    // ---- transposed conv2d ---------------------------------------------------
    // Wt is input-channel-major: (C_in, (C_out/groups)*kH*kW). bias may be null.
    // dWt / dB ACCUMULATE — zero them before the call.
    fn(ns_tensor, "convTranspose2dForward", function convTranspose2dForward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, Y) {
        const L = "convTranspose2dForward(X,Wt,bias|null,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,Y)";
        __bro_native.tensor.convTranspose2dForward(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"),
            int(N, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0), int(kH, 0), int(kW, 0),
            int(sH, 1), int(sW, 1), int(pH, 0), int(pW, 0), int(opH, 0), int(opW, 0),
            int(dH, 1), int(dW, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "convTranspose2dBackwardInput", function convTranspose2dBackwardInput(Wt, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, dX) {
        const L = "convTranspose2dBackwardInput(Wt,dY,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,dX)";
        __bro_native.tensor.convTranspose2dBackwardInput(T(Wt, L, "Wt"), T(dY, L, "dY"),
            int(N, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0), int(kH, 0), int(kW, 0),
            int(sH, 1), int(sW, 1), int(pH, 0), int(pW, 0), int(opH, 0), int(opW, 0),
            int(dH, 1), int(dW, 1), int(groups, 1), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "convTranspose2dBackwardWeight", function convTranspose2dBackwardWeight(X, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, dWt) {
        const L = "convTranspose2dBackwardWeight(X,dY,N,C_in,H,W,C_out,kH,kW,sH,sW,pH,pW,opH,opW,dH,dW,groups,dWt)";
        __bro_native.tensor.convTranspose2dBackwardWeight(T(X, L, "X"), T(dY, L, "dY"),
            int(N, 0), int(C_in, 0), int(H, 0), int(W, 0), int(C_out, 0), int(kH, 0), int(kW, 0),
            int(sH, 1), int(sW, 1), int(pH, 0), int(pW, 0), int(opH, 0), int(opW, 0),
            int(dH, 1), int(dW, 1), int(groups, 1), T(dWt, L, "dWt"));
        chk();
    });
    fn(ns_tensor, "convTranspose2dBackwardBias", function convTranspose2dBackwardBias(dY, N, C_out, H_out, W_out, dB) {
        const L = "convTranspose2dBackwardBias(dY,N,C_out,H_out,W_out,dB)";
        __bro_native.tensor.convTranspose2dBackwardBias(T(dY, L, "dY"), int(N, 0), int(C_out, 0),
            int(H_out, 0), int(W_out, 0), T(dB, L, "dB"));
        chk();
    });

    // ---- conv3d (NCTHW, forward only) ---------------------------------------
    // Wt: (C_out, (C_in/groups)*kT*kH*kW), OICTHW. bias may be null.
    fn(ns_tensor, "conv3dForward", function conv3dForward(X, Wt, bias, N, C_in, T_in, H, W, C_out, kT, kH, kW, sT, sH, sW, pT, pH, pW, dT, dH, dW, groups, Y) {
        const L = "conv3dForward(X,Wt,bias|null,N,C_in,T,H,W,C_out,kT,kH,kW,sT,sH,sW,pT,pH,pW,dT,dH,dW,groups,Y)";
        __bro_native.tensor.conv3dForward(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"),
            int(N, 0), int(C_in, 0), int(T_in, 0), int(H, 0), int(W, 0), int(C_out, 0),
            int(kT, 0), int(kH, 0), int(kW, 0), int(sT, 1), int(sH, 1), int(sW, 1),
            int(pT, 0), int(pH, 0), int(pW, 0), int(dT, 1), int(dH, 1), int(dW, 1),
            int(groups, 1), T(Y, L, "Y"));
        chk();
    });

    // ---- window partition / reverse / 2x2 spatial merge ----------------------
    // windowReverseForward is the exact inverse (and the adjoint) of
    // windowPartitionForward; H and W must be multiples of `window`.
    fn(ns_tensor, "windowPartitionForward", function windowPartitionForward(X, N, C, H, W, window, Y) {
        const L = "windowPartitionForward(X,N,C,H,W,window,Y)";
        __bro_native.tensor.windowPartitionForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(window, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "windowReverseForward", function windowReverseForward(X, N, C, H, W, window, Y) {
        const L = "windowReverseForward(X,N,C,H,W,window,Y)";
        __bro_native.tensor.windowReverseForward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            int(window, 0), T(Y, L, "Y"));
        chk();
    });
    // channelMajor is optional and defaults to false: block-major
    // c_out = block*C + c_in (Qwen-VL patch merger). true gives
    // c_out = c_in*4 + block (torch pixel_unshuffle / Flux.2 VAE tail).
    fn(ns_tensor, "spatialMerge2x2Forward", function spatialMerge2x2Forward(X, N, C, H, W, Y, channelMajor) {
        const L = "spatialMerge2x2Forward(X,N,C,H,W,Y,channelMajor?)";
        __bro_native.tensor.spatialMerge2x2Forward(T(X, L, "X"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            T(Y, L, "Y"), !!channelMajor);
        chk();
    });

    // ---- batch norm ----------------------------------------------------------
    // Training forward: runningMean / runningVar are updated in place, and
    // savedMean / savedRstd ((C,1) each) feed batchNormBackward.
    fn(ns_tensor, "batchNormForward", function batchNormForward(X, gamma, beta, runningMean, runningVar, N, C, H, W, eps, momentum, Y, savedMean, savedRstd) {
        const L = "batchNormForward(X,gamma,beta,runningMean,runningVar,N,C,H,W,eps,momentum,Y,savedMean,savedRstd)";
        __bro_native.tensor.batchNormForward(T(X, L, "X"), T(gamma, L, "gamma"), T(beta, L, "beta"),
            T(runningMean, L, "runningMean"), T(runningVar, L, "runningVar"),
            int(N, 0), int(C, 0), int(H, 0), int(W, 0), num(eps, 0.00001), num(momentum, 0.1),
            T(Y, L, "Y"), T(savedMean, L, "savedMean"), T(savedRstd, L, "savedRstd"));
        chk();
    });
    // dGamma / dBeta ACCUMULATE — zero them before the call. dX is overwritten.
    fn(ns_tensor, "batchNormBackward", function batchNormBackward(X, gamma, savedMean, savedRstd, dY, N, C, H, W, dX, dGamma, dBeta) {
        const L = "batchNormBackward(X,gamma,savedMean,savedRstd,dY,N,C,H,W,dX,dGamma,dBeta)";
        __bro_native.tensor.batchNormBackward(T(X, L, "X"), T(gamma, L, "gamma"), T(savedMean, L, "savedMean"),
            T(savedRstd, L, "savedRstd"), T(dY, L, "dY"), int(N, 0), int(C, 0), int(H, 0), int(W, 0),
            T(dX, L, "dX"), T(dGamma, L, "dGamma"), T(dBeta, L, "dBeta"));
        chk();
    });
    fn(ns_tensor, "batchNormInference", function batchNormInference(X, gamma, beta, runningMean, runningVar, N, C, H, W, eps, Y) {
        const L = "batchNormInference(X,gamma,beta,runningMean,runningVar,N,C,H,W,eps,Y)";
        __bro_native.tensor.batchNormInference(T(X, L, "X"), T(gamma, L, "gamma"), T(beta, L, "beta"),
            T(runningMean, L, "runningMean"), T(runningVar, L, "runningVar"),
            int(N, 0), int(C, 0), int(H, 0), int(W, 0), num(eps, 0.00001), T(Y, L, "Y"));
        chk();
    });

    // ---- image preprocessing -------------------------------------------------
    // Per-channel (X - mean[c]) / std[c] on NCHW; mean / std are (C,1).
    fn(ns_tensor, "imageNormalize", function imageNormalize(X, mean, std, N, C, H, W, Y) {
        const L = "imageNormalize(X,mean,std,N,C,H,W,Y)";
        __bro_native.tensor.imageNormalize(T(X, L, "X"), T(mean, L, "mean"), T(std, L, "std"),
            int(N, 0), int(C, 0), int(H, 0), int(W, 0), T(Y, L, "Y"));
        chk();
    });
    // Packed uint8 HWC bytes (a decoder's output) into FP32 NCHW, with one
    // scale+bias pass: Y = src*scale + bias. [0,255]->[0,1] is scale 1/255,
    // bias 0; [0,255]->[-1,1] is scale 2/255, bias -1.
    fn(ns_tensor, "imageU8ToF32NhwcToNchw", function imageU8ToF32NhwcToNchw(src, N, H, W, C, scale, bias, Y) {
        const L = "imageU8ToF32NhwcToNchw(srcUint8,N,H,W,C,scale,bias,Y)";
        __bro_native.tensor.imageU8ToF32NhwcToNchw(u8(src, L, "src"), int(N, 0), int(H, 0), int(W, 0),
            int(C, 0), num(scale, 1.0), num(bias, 0.0), T(Y, L, "Y"));
        chk();
    });
})();
