// tensor_audio.js — the audio slice of bro.tensor: the spectral / FFT core and
// the complex helpers, STFT / iSTFT, the 1D convolution family (conv1d, its
// backward halves, the W8A16 conv1d, transposed conv, causal conv + its
// streaming update, pad1d), the vocoder / codec activations (snake, elu,
// leaky relu), codec quantization (VQ, FSQ), 1D resampling, the log / exp /
// round elementwise maps and the autoregressive logit sampler.
//
// These are the building blocks of Whisper / TTS / neural-codec / vocoder
// pipelines. Complex spectra are ordinary FP32 tensors with the bin axis
// stored interleaved [re, im, re, im, ...] — an (R, 2*C) tensor — so no new
// dtype is involved. Activations in the conv family are NCL: (N, C*L).
//
// Signatures are the QuickJS binding's (tensor_bindings_audio.cpp) — same
// argument order, same `bias|null` / `beta|null` slots, same defaults for the
// hyperparameters the old bodies initialised (stride 1, padding 0, dilation 1,
// groups 1, output padding 0, mode 0, elu alpha 1, leaky slope 0.01). Compiled
// into its own module (bronze_tensor_audio_main) that api.cpp mounts after
// js/tensor.js and js/tensor_ext.js, so bro.tensor and bro.tensor.GpuTensor
// already exist here. Every wrapper checks its arguments (the natives receive
// nullable slots as raw values) and reads back the native's error afterwards.
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
    // A counter-based RNG seed, as js/tensor.js takes one: Number or BigInt.
    const seed = (v, label, name) => {
        if (v === undefined || v === null) return 0;
        if (typeof v === "number" || typeof v === "bigint") return v;
        throw new TypeError(label + ": " + name + " must be a Number or BigInt");
    };

    // ---- spectral / FFT core ---------------------------------------------------
    // fft / ifft transform one interleaved-complex (R, 2*N) signal per row and
    // have no backward: the adjoint is the other transform times a scalar.
    fn(ns_tensor, "fft", function fft(x, y) {
        const L = "fft(x,y)";
        __bro_native.tensor.fft(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "ifft", function ifft(x, y) {
        const L = "ifft(x,y)";
        __bro_native.tensor.ifft(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    // rfft: REAL (R,len) -> interleaved-complex (R, 2*(len/2+1)).
    fn(ns_tensor, "rfft", function rfft(x, y) {
        const L = "rfft(x,y)";
        __bro_native.tensor.rfft(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    // irfft: the signal length is explicit — a C-bin half-spectrum is ambiguous.
    fn(ns_tensor, "irfft", function irfft(x, len, y) {
        const L = "irfft(x,L,y)";
        __bro_native.tensor.irfft(T(x, L, "x"), int(len, 0), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "rfftBackward", function rfftBackward(dY, len, dX) {
        const L = "rfftBackward(dY,L,dX)";
        __bro_native.tensor.rfftBackward(T(dY, L, "dY"), int(len, 0), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "irfftBackward", function irfftBackward(dY, dX) {
        const L = "irfftBackward(dY,dX)";
        __bro_native.tensor.irfftBackward(T(dY, L, "dY"), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "complexMul", function complexMul(a, b, y) {
        const L = "complexMul(a,b,y)";
        __bro_native.tensor.complexMul(T(a, L, "a"), T(b, L, "b"), T(y, L, "y"));
        chk();
    });
    // dA / dB accumulate — the caller pre-sizes and zeros them.
    fn(ns_tensor, "complexMulBackward", function complexMulBackward(a, b, dY, dA, dB) {
        const L = "complexMulBackward(a,b,dY,dA,dB)";
        __bro_native.tensor.complexMulBackward(T(a, L, "a"), T(b, L, "b"), T(dY, L, "dY"), T(dA, L, "dA"), T(dB, L, "dB"));
        chk();
    });
    fn(ns_tensor, "complexAbs", function complexAbs(z, y) {
        const L = "complexAbs(z,y)";
        __bro_native.tensor.complexAbs(T(z, L, "z"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "complexAbsBackward", function complexAbsBackward(z, dY, dZ) {
        const L = "complexAbsBackward(z,dY,dZ)";
        __bro_native.tensor.complexAbsBackward(T(z, L, "z"), T(dY, L, "dY"), T(dZ, L, "dZ"));
        chk();
    });
    fn(ns_tensor, "complexAngle", function complexAngle(z, y) {
        const L = "complexAngle(z,y)";
        __bro_native.tensor.complexAngle(T(z, L, "z"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "complexFromPolar", function complexFromPolar(mag, phase, y) {
        const L = "complexFromPolar(mag,phase,y)";
        __bro_native.tensor.complexFromPolar(T(mag, L, "mag"), T(phase, L, "phase"), T(y, L, "y"));
        chk();
    });

    // ---- STFT / iSTFT ----------------------------------------------------------
    // The spectrogram is interleaved-complex (N*frames, 2*(nFft/2+1)). stft and
    // istft are linear but NOT mutual adjoints once the window and the COLA
    // normalisation are folded in, so both backward ops are explicit.
    fn(ns_tensor, "stft", function stft(signal, window, N, nFft, hopLength, winLength, center, normalized, spec) {
        const L = "stft(signal,window,N,nFft,hopLength,winLength,center,normalized,spec)";
        __bro_native.tensor.stft(T(signal, L, "signal"), T(window, L, "window"), int(N, 1), int(nFft, 0),
            int(hopLength, 0), int(winLength, 0), !!center, !!normalized, T(spec, L, "spec"));
        chk();
    });
    fn(ns_tensor, "stftBackward", function stftBackward(dSpec, window, N, signalLen, nFft, hopLength, winLength, center, normalized, dSignal) {
        const L = "stftBackward(dSpec,window,N,signalLen,nFft,hopLength,winLength,center,normalized,dSignal)";
        __bro_native.tensor.stftBackward(T(dSpec, L, "dSpec"), T(window, L, "window"), int(N, 1), int(signalLen, 0),
            int(nFft, 0), int(hopLength, 0), int(winLength, 0), !!center, !!normalized, T(dSignal, L, "dSignal"));
        chk();
    });
    fn(ns_tensor, "istft", function istft(spec, window, N, signalLen, nFft, hopLength, winLength, center, normalized, signal) {
        const L = "istft(spec,window,N,signalLen,nFft,hopLength,winLength,center,normalized,signal)";
        __bro_native.tensor.istft(T(spec, L, "spec"), T(window, L, "window"), int(N, 1), int(signalLen, 0),
            int(nFft, 0), int(hopLength, 0), int(winLength, 0), !!center, !!normalized, T(signal, L, "signal"));
        chk();
    });
    fn(ns_tensor, "istftBackward", function istftBackward(dSignal, window, N, signalLen, nFft, hopLength, winLength, center, normalized, dSpec) {
        const L = "istftBackward(dSignal,window,N,signalLen,nFft,hopLength,winLength,center,normalized,dSpec)";
        __bro_native.tensor.istftBackward(T(dSignal, L, "dSignal"), T(window, L, "window"), int(N, 1), int(signalLen, 0),
            int(nFft, 0), int(hopLength, 0), int(winLength, 0), !!center, !!normalized, T(dSpec, L, "dSpec"));
        chk();
    });

    // ---- 1D padding + convolution family (NCL) ---------------------------------
    // mode: 0 zero, 1 reflect, 2 replicate.
    fn(ns_tensor, "pad1dForward", function pad1dForward(X, N, C, len, padLeft, padRight, mode, Y) {
        const L = "pad1dForward(X,N,C,L,padLeft,padRight,mode,Y)";
        __bro_native.tensor.pad1dForward(T(X, L, "X"), int(N, 1), int(C, 1), int(len, 0),
            int(padLeft, 0), int(padRight, 0), int(mode, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "pad1dBackward", function pad1dBackward(dY, N, C, len, padLeft, padRight, mode, dX) {
        const L = "pad1dBackward(dY,N,C,L,padLeft,padRight,mode,dX)";
        __bro_native.tensor.pad1dBackward(T(dY, L, "dY"), int(N, 1), int(C, 1), int(len, 0),
            int(padLeft, 0), int(padRight, 0), int(mode, 0), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "conv1d", function conv1d(X, Wt, bias, N, C_in, len, C_out, kL, stride, padding, dilation, groups, Y) {
        const L = "conv1d(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,padding,dilation,groups,Y)";
        __bro_native.tensor.conv1d(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"), int(N, 1), int(C_in, 1),
            int(len, 0), int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(dilation, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "conv1dBackwardInput", function conv1dBackwardInput(Wt, dY, N, C_in, len, C_out, kL, stride, padding, dilation, groups, dX) {
        const L = "conv1dBackwardInput(Wt,dY,N,C_in,L,C_out,kL,stride,padding,dilation,groups,dX)";
        __bro_native.tensor.conv1dBackwardInput(T(Wt, L, "Wt"), T(dY, L, "dY"), int(N, 1), int(C_in, 1), int(len, 0),
            int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(dilation, 1), int(groups, 1), T(dX, L, "dX"));
        chk();
    });
    // dWt accumulates — the caller zeros it.
    fn(ns_tensor, "conv1dBackwardWeight", function conv1dBackwardWeight(X, dY, N, C_in, len, C_out, kL, stride, padding, dilation, groups, dWt) {
        const L = "conv1dBackwardWeight(X,dY,N,C_in,L,C_out,kL,stride,padding,dilation,groups,dWt)";
        __bro_native.tensor.conv1dBackwardWeight(T(X, L, "X"), T(dY, L, "dY"), int(N, 1), int(C_in, 1), int(len, 0),
            int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(dilation, 1), int(groups, 1), T(dWt, L, "dWt"));
        chk();
    });
    // dB accumulates — the caller zeros it.
    fn(ns_tensor, "conv1dBackwardBias", function conv1dBackwardBias(dY, N, C_out, L_out, dB) {
        const L = "conv1dBackwardBias(dY,N,C_out,L_out,dB)";
        __bro_native.tensor.conv1dBackwardBias(T(dY, L, "dY"), int(N, 1), int(C_out, 1), int(L_out, 0), T(dB, L, "dB"));
        chk();
    });
    // W8A16: FP16 activations, INT8 per-output-row weights, FP32 scales. GPU-only.
    fn(ns_tensor, "conv1dInt8wFp16", function conv1dInt8wFp16(X, W_int8, scales, bias, N, C_in, len, C_out, kL, stride, padding, dilation, groups, Y) {
        const L = "conv1dInt8wFp16(X,W_int8,scales,bias|null,N,C_in,L,C_out,kL,stride,padding,dilation,groups,Y)";
        __bro_native.tensor.conv1dInt8wFp16(T(X, L, "X"), T(W_int8, L, "W_int8"), T(scales, L, "scales"), opt(bias, L, "bias"),
            int(N, 1), int(C_in, 1), int(len, 0), int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(dilation, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });
    // Transposed conv weights are input-channel-major: (C_in, (C_out/groups)*kL).
    fn(ns_tensor, "convTranspose1dForward", function convTranspose1dForward(X, Wt, bias, N, C_in, len, C_out, kL, stride, padding, outputPadding, dilation, groups, Y) {
        const L = "convTranspose1dForward(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,padding,outputPadding,dilation,groups,Y)";
        __bro_native.tensor.convTranspose1dForward(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"), int(N, 1), int(C_in, 1),
            int(len, 0), int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(outputPadding, 0), int(dilation, 1), int(groups, 1), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "convTranspose1dBackwardInput", function convTranspose1dBackwardInput(Wt, dY, N, C_in, len, C_out, kL, stride, padding, outputPadding, dilation, groups, dX) {
        const L = "convTranspose1dBackwardInput(Wt,dY,N,C_in,L,C_out,kL,stride,padding,outputPadding,dilation,groups,dX)";
        __bro_native.tensor.convTranspose1dBackwardInput(T(Wt, L, "Wt"), T(dY, L, "dY"), int(N, 1), int(C_in, 1), int(len, 0),
            int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(outputPadding, 0), int(dilation, 1), int(groups, 1), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "convTranspose1dBackwardWeight", function convTranspose1dBackwardWeight(X, dY, N, C_in, len, C_out, kL, stride, padding, outputPadding, dilation, groups, dWt) {
        const L = "convTranspose1dBackwardWeight(X,dY,N,C_in,L,C_out,kL,stride,padding,outputPadding,dilation,groups,dWt)";
        __bro_native.tensor.convTranspose1dBackwardWeight(T(X, L, "X"), T(dY, L, "dY"), int(N, 1), int(C_in, 1), int(len, 0),
            int(C_out, 1), int(kL, 1), int(stride, 1), int(padding, 0), int(outputPadding, 0), int(dilation, 1), int(groups, 1), T(dWt, L, "dWt"));
        chk();
    });
    fn(ns_tensor, "convTranspose1dBackwardBias", function convTranspose1dBackwardBias(dY, N, C_out, L_out, dB) {
        const L = "convTranspose1dBackwardBias(dY,N,C_out,L_out,dB)";
        __bro_native.tensor.convTranspose1dBackwardBias(T(dY, L, "dY"), int(N, 1), int(C_out, 1), int(L_out, 0), T(dB, L, "dB"));
        chk();
    });
    // `scratch` is a caller-owned GpuTensor reused as the left-padded-input buffer.
    fn(ns_tensor, "causalConv1d", function causalConv1d(X, Wt, bias, N, C_in, len, C_out, kL, stride, dilation, groups, scratch, Y) {
        const L = "causalConv1d(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,dilation,groups,scratch,Y)";
        __bro_native.tensor.causalConv1d(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"), int(N, 1), int(C_in, 1),
            int(len, 0), int(C_out, 1), int(kL, 1), int(stride, 1), int(dilation, 1), int(groups, 1), T(scratch, L, "scratch"), T(Y, L, "Y"));
        chk();
    });
    // `state` is the rolling (kL-1)*dilation-sample history — read AND overwritten.
    fn(ns_tensor, "causalConv1dUpdate", function causalConv1dUpdate(X, Wt, bias, N, C, L_step, kL, dilation, state, Y) {
        const L = "causalConv1dUpdate(X,Wt,bias|null,N,C,L_step,kL,dilation,state,Y)";
        __bro_native.tensor.causalConv1dUpdate(T(X, L, "X"), T(Wt, L, "Wt"), opt(bias, L, "bias"), int(N, 1), int(C, 1),
            int(L_step, 0), int(kL, 1), int(dilation, 1), T(state, L, "state"), T(Y, L, "Y"));
        chk();
    });

    // ---- vocoder / codec activations -------------------------------------------
    // snake carries per-channel learnable alpha (and optional beta); elu and
    // leaky relu are plain elementwise maps with a scalar parameter.
    fn(ns_tensor, "snakeForward", function snakeForward(X, alpha, beta, N, C, len, Y) {
        const L = "snakeForward(X,alpha,beta|null,N,C,L,Y)";
        __bro_native.tensor.snakeForward(T(X, L, "X"), T(alpha, L, "alpha"), opt(beta, L, "beta"),
            int(N, 1), int(C, 1), int(len, 0), T(Y, L, "Y"));
        chk();
    });
    // dBeta must be non-null exactly when beta is; dAlpha / dBeta accumulate.
    fn(ns_tensor, "snakeBackward", function snakeBackward(X, alpha, beta, dY, N, C, len, dX, dAlpha, dBeta) {
        const L = "snakeBackward(X,alpha,beta|null,dY,N,C,L,dX,dAlpha,dBeta|null)";
        __bro_native.tensor.snakeBackward(T(X, L, "X"), T(alpha, L, "alpha"), opt(beta, L, "beta"), T(dY, L, "dY"),
            int(N, 1), int(C, 1), int(len, 0), T(dX, L, "dX"), T(dAlpha, L, "dAlpha"), opt(dBeta, L, "dBeta"));
        chk();
    });
    fn(ns_tensor, "eluForward", function eluForward(x, alpha, y) {
        const L = "eluForward(x,alpha,y)";
        __bro_native.tensor.eluForward(T(x, L, "x"), num(alpha, 1.0), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "eluBackward", function eluBackward(x, dY, alpha, dX) {
        const L = "eluBackward(x,dY,alpha,dX)";
        __bro_native.tensor.eluBackward(T(x, L, "x"), T(dY, L, "dY"), num(alpha, 1.0), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "leakyReluForward", function leakyReluForward(x, negativeSlope, y) {
        const L = "leakyReluForward(x,negativeSlope,y)";
        __bro_native.tensor.leakyReluForward(T(x, L, "x"), num(negativeSlope, 0.01), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "leakyReluBackward", function leakyReluBackward(x, dY, negativeSlope, dX) {
        const L = "leakyReluBackward(x,dY,negativeSlope,dX)";
        __bro_native.tensor.leakyReluBackward(T(x, L, "x"), T(dY, L, "dY"), num(negativeSlope, 0.01), T(dX, L, "dX"));
        chk();
    });

    // ---- codec quantization ----------------------------------------------------
    // VQ-VAE residual-VQ and FSQ bottlenecks; the straight-through estimator
    // makes both backward ops a plain identity passthrough.
    fn(ns_tensor, "vqEncodeForward", function vqEncodeForward(x, codebook, indices, quantized) {
        const L = "vqEncodeForward(x,codebook,indices,quantized)";
        __bro_native.tensor.vqEncodeForward(T(x, L, "x"), T(codebook, L, "codebook"), T(indices, L, "indices"), T(quantized, L, "quantized"));
        chk();
    });
    fn(ns_tensor, "vqEncodeBackward", function vqEncodeBackward(dQuantized, dX) {
        const L = "vqEncodeBackward(dQuantized,dX)";
        __bro_native.tensor.vqEncodeBackward(T(dQuantized, L, "dQuantized"), T(dX, L, "dX"));
        chk();
    });
    // levels: a (D,1) INT32 GpuTensor of per-dim level counts.
    fn(ns_tensor, "fsqQuantizeForward", function fsqQuantizeForward(x, levels, quantized, packedIndices) {
        const L = "fsqQuantizeForward(x,levels,quantized,packedIndices)";
        __bro_native.tensor.fsqQuantizeForward(T(x, L, "x"), T(levels, L, "levels"), T(quantized, L, "quantized"), T(packedIndices, L, "packedIndices"));
        chk();
    });
    fn(ns_tensor, "fsqQuantizeBackward", function fsqQuantizeBackward(dQuantized, dX) {
        const L = "fsqQuantizeBackward(dQuantized,dX)";
        __bro_native.tensor.fsqQuantizeBackward(T(dQuantized, L, "dQuantized"), T(dX, L, "dX"));
        chk();
    });

    // ---- 1D resampling ---------------------------------------------------------
    // Arbitrary-scale resampling along the length axis of an NCL audio tensor.
    // mode: 0 = nearest, 1 = linear.
    fn(ns_tensor, "resample1dForward", function resample1dForward(X, N, C, L_in, L_out, mode, Y) {
        const L = "resample1dForward(X,N,C,L_in,L_out,mode,Y)";
        __bro_native.tensor.resample1dForward(T(X, L, "X"), int(N, 1), int(C, 1), int(L_in, 0), int(L_out, 0), int(mode, 0), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "resample1dBackward", function resample1dBackward(dY, N, C, L_in, L_out, mode, dX) {
        const L = "resample1dBackward(dY,N,C,L_in,L_out,mode,dX)";
        __bro_native.tensor.resample1dBackward(T(dY, L, "dY"), int(N, 1), int(C, 1), int(L_in, 0), int(L_out, 0), int(mode, 0), T(dX, L, "dX"));
        chk();
    });

    // ---- log / exp / round elementwise -----------------------------------------
    // log / exp backward read the raw forward input; round backward is the
    // straight-through estimator and needs only dY.
    fn(ns_tensor, "logForward", function logForward(x, y) {
        const L = "logForward(x,y)";
        __bro_native.tensor.logForward(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "logBackward", function logBackward(x, dY, dX) {
        const L = "logBackward(x,dY,dX)";
        __bro_native.tensor.logBackward(T(x, L, "x"), T(dY, L, "dY"), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "expForward", function expForward(x, y) {
        const L = "expForward(x,y)";
        __bro_native.tensor.expForward(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "expBackward", function expBackward(x, dY, dX) {
        const L = "expBackward(x,dY,dX)";
        __bro_native.tensor.expBackward(T(x, L, "x"), T(dY, L, "dY"), T(dX, L, "dX"));
        chk();
    });
    fn(ns_tensor, "roundForward", function roundForward(x, y) {
        const L = "roundForward(x,y)";
        __bro_native.tensor.roundForward(T(x, L, "x"), T(y, L, "y"));
        chk();
    });
    fn(ns_tensor, "roundBackward", function roundBackward(dY, dX) {
        const L = "roundBackward(dY,dX)";
        __bro_native.tensor.roundBackward(T(dY, L, "dY"), T(dX, L, "dX"));
        chk();
    });

    // ---- autoregressive logit sampling -----------------------------------------
    // Per-row next-token sampler (temperature / top-k / top-p, Philox RNG).
    // key / counter are Numbers or BigInts; indices comes back (N,1) INT32.
    fn(ns_tensor, "sampleLogits", function sampleLogits(logits, temperature, topK, topP, key, counter, indices) {
        const L = "sampleLogits(logits,temperature,topK,topP,key,counter,indices)";
        __bro_native.tensor.sampleLogits(T(logits, L, "logits"), num(temperature, 1.0), int(topK, 0), num(topP, 1.0),
            seed(key, L, "key"), seed(counter, L, "counter"), T(indices, L, "indices"));
        chk();
    });
    // The graph-capturable twin: counter is a (>=1,) INT32 GpuTensor advanced
    // in place on-device, scratch is FP32 with >= 3*N*V elements, and indices
    // is a caller-pre-sized (N,1) INT32 written in place.
    fn(ns_tensor, "sampleLogitsInto", function sampleLogitsInto(logits, temperature, topK, topP, key, counter, scratch, indices) {
        const L = "sampleLogitsInto(logits,temperature,topK,topP,key,counter,scratch,indices)";
        __bro_native.tensor.sampleLogitsInto(T(logits, L, "logits"), num(temperature, 1.0), int(topK, 0), num(topP, 1.0),
            seed(key, L, "key"), T(counter, L, "counter"), T(scratch, L, "scratch"), T(indices, L, "indices"));
        chk();
    });
})();
