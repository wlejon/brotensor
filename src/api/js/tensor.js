// tensor.js — the public shape of bro.tensor, bro.tensor.GpuTensor, assembled over the
// natives under __bro_native.tensor (native_tensor_decl.h states each C entry point,
// api.cpp registers them). Hand-written, like the rest of src/api/.
//
// Every native is spelled by its full dotted path at the point of use: that is the
// spelling the compiler lowers to a direct call. The roots this module reads as bare
// identifiers are listed in module.globals beside it.
//
// Errors: a native cannot throw, so one that fails records a message the
// wrapper reads back with __bro_native.tensor.takeError() and throws. The
// wrappers written since the port do that after every call (`chk()`); the
// older ones below still trust their natives.
//
// The attention family, losses, pooling and concat live in js/tensor_ext.js,
// a second compiled module mounted right after this one (api.cpp
// installTensorJS).
(function () {
    'use strict';

    if (typeof globalThis.bro === "undefined") {
        globalThis.bro = {};
    }
    const bro = globalThis.bro;

    const accessor = (obj, name, get, set) =>
        Object.defineProperty(obj, name, { get, set, enumerable: true, configurable: true });
    const fn = (obj, name, value) =>
        Object.defineProperty(obj, name, { value, writable: true, enumerable: true, configurable: true });
    const EMPTY_I8 = new Int8Array(0);
    const EMPTY_U16 = new Uint16Array(0);
    const mount = (root, name) => root[name] !== undefined ? root[name] : (root[name] = {});
    const toF32 = (v) => v instanceof Float32Array ? v : Float32Array.from(v);
    const toI8 = (v) => v instanceof Int8Array ? v : Int8Array.from(v);
    const toU16 = (v) => v instanceof Uint16Array ? v : Uint16Array.from(v);

    // ---- bro.tensor ----------------------------------------------------------
    const ns_tensor = mount(bro, "tensor");
    accessor(ns_tensor, "available",
        function () {
            return typeof __bro_native.tensor.available === "boolean" ? __bro_native.tensor.available : false;
        },
        undefined);
    accessor(ns_tensor, "backend",
        function () {
            return __bro_native.tensor.backend || "cpu";
        },
        undefined);
    const DTYPES = { fp32: 0, fp16: 1, int8: 2, int32: 3, bf16: 4, f64: 5 };
    fn(ns_tensor, "dtype", DTYPES);
    // A dtype argument the way the old binding took it: a name ("fp32"/"f32",
    // "fp16"/"f16", "bf16", "int8"/"i8", ...), a bro.tensor.dtype enum value,
    // or nothing (the default). Numbers map back to the name the native reads.
    const DTYPE_NAMES = ["fp32", "fp16", "int8", "int32", "bf16", "f64"];
    const dtypeName = (v, def) => {
        if (v === undefined || v === null) return def;
        if (typeof v === "string") return v;
        if (typeof v === "number") return DTYPE_NAMES[v | 0] !== undefined ? DTYPE_NAMES[v | 0] : def;
        throw new TypeError("dtype must be a string or a bro.tensor.dtype value");
    };
    // Throw what the last native recorded, if anything.
    const chk = () => {
        const e = __bro_native.tensor.takeError();
        if (e) throw new Error(e);
    };
    // A counter-based RNG seed: BigInt or Number, passed through as-is.
    const seed = (v, label) => {
        if (typeof v === "number" || typeof v === "bigint") return v;
        throw new TypeError(label + ": key / counter / state must be a Number or BigInt");
    };
    fn(ns_tensor, "init", function init() {
        if (ns_tensor.available) {
            __bro_native.tensor.init(); chk();
        }
        return undefined;
    });
    fn(ns_tensor, "sync", function sync() {
        __bro_native.tensor.sync(); chk();
    });
    fn(ns_tensor, "createTensor", function createTensor(rows, cols, dtype) {
        if (!ns_tensor.available) throw new Error("bro.tensor: compiled without BRO_WITH_TENSOR");
        if (rows === undefined || rows < 0 || (cols !== undefined && cols < 0)) throw new RangeError("createTensor: invalid dims");
        const r = rows | 0;
        const c = cols !== undefined ? (cols | 0) : 1;
        const res = __bro_native.tensor.createTensor(r, c, dtypeName(dtype, "fp32")); chk(); return res;
    });
    fn(ns_tensor, "linearForward", function linearForward(W, b, x, y) {
        if (W === undefined) throw new TypeError("bro.tensor.linearForward: W is required");
        if (b === undefined) throw new TypeError("bro.tensor.linearForward: b is required");
        if (x === undefined) throw new TypeError("bro.tensor.linearForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.linearForward: y is required");
        __bro_native.tensor.linearForward(W, b, x, y); chk();
    });
    fn(ns_tensor, "linearBackward", function linearBackward(W, x, dY, dX, dW, dB) {
        if (W === undefined) throw new TypeError("bro.tensor.linearBackward: W is required");
        if (x === undefined) throw new TypeError("bro.tensor.linearBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.linearBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.linearBackward: dX is required");
        if (dW === undefined) throw new TypeError("bro.tensor.linearBackward: dW is required");
        if (dB === undefined) throw new TypeError("bro.tensor.linearBackward: dB is required");
        __bro_native.tensor.linearBackward(W, x, dY, dX, dW, dB); chk();
    });
    fn(ns_tensor, "reluForward", function reluForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.reluForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.reluForward: y is required");
        __bro_native.tensor.reluForward(x, y); chk();
    });
    fn(ns_tensor, "reluBackward", function reluBackward(x, dY, dX) {
        if (x === undefined) throw new TypeError("bro.tensor.reluBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.reluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.reluBackward: dX is required");
        __bro_native.tensor.reluBackward(x, dY, dX); chk();
    });
    fn(ns_tensor, "tanhForward", function tanhForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.tanhForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.tanhForward: y is required");
        __bro_native.tensor.tanhForward(x, y); chk();
    });
    fn(ns_tensor, "tanhBackward", function tanhBackward(y, dY, dX) {
        if (y === undefined) throw new TypeError("bro.tensor.tanhBackward: y is required");
        if (dY === undefined) throw new TypeError("bro.tensor.tanhBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.tanhBackward: dX is required");
        __bro_native.tensor.tanhBackward(y, dY, dX); chk();
    });
    fn(ns_tensor, "sigmoidForward", function sigmoidForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.sigmoidForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.sigmoidForward: y is required");
        __bro_native.tensor.sigmoidForward(x, y); chk();
    });
    fn(ns_tensor, "sigmoidBackward", function sigmoidBackward(y, dY, dX) {
        if (y === undefined) throw new TypeError("bro.tensor.sigmoidBackward: y is required");
        if (dY === undefined) throw new TypeError("bro.tensor.sigmoidBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.sigmoidBackward: dX is required");
        __bro_native.tensor.sigmoidBackward(y, dY, dX); chk();
    });
    fn(ns_tensor, "addInplace", function addInplace(y, x) {
        if (y === undefined) throw new TypeError("bro.tensor.addInplace: y is required");
        if (x === undefined) throw new TypeError("bro.tensor.addInplace: x is required");
        __bro_native.tensor.addInplace(y, x); chk();
    });
    fn(ns_tensor, "addScalarInplace", function addScalarInplace(y, s) {
        if (y === undefined) throw new TypeError("bro.tensor.addScalarInplace: y is required");
        if (s === undefined) throw new TypeError("bro.tensor.addScalarInplace: s is required");
        __bro_native.tensor.addScalarInplace(y, s); chk();
    });
    fn(ns_tensor, "scaleInplace", function scaleInplace(y, s) {
        if (y === undefined) throw new TypeError("bro.tensor.scaleInplace: y is required");
        if (s === undefined) throw new TypeError("bro.tensor.scaleInplace: s is required");
        __bro_native.tensor.scaleInplace(y, s); chk();
    });
    fn(ns_tensor, "mulInplace", function mulInplace(y, x) {
        if (y === undefined) throw new TypeError("bro.tensor.mulInplace: y is required");
        if (x === undefined) throw new TypeError("bro.tensor.mulInplace: x is required");
        __bro_native.tensor.mulInplace(y, x); chk();
    });
    fn(ns_tensor, "clamp", function clamp(y, lo, hi) {
        if (y === undefined) throw new TypeError("bro.tensor.clamp: y is required");
        if (lo === undefined) throw new TypeError("bro.tensor.clamp: lo is required");
        if (hi === undefined) throw new TypeError("bro.tensor.clamp: hi is required");
        __bro_native.tensor.clamp(y, lo, hi); chk();
    });
    fn(ns_tensor, "buildSlotMask", function buildSlotMask(x, offset, K, stride, mask) {
        if (x === undefined) throw new TypeError("bro.tensor.buildSlotMask: x is required");
        if (offset === undefined) throw new TypeError("bro.tensor.buildSlotMask: offset is required");
        if (K === undefined) throw new TypeError("bro.tensor.buildSlotMask: K is required");
        if (stride === undefined) throw new TypeError("bro.tensor.buildSlotMask: stride is required");
        if (mask === undefined) throw new TypeError("bro.tensor.buildSlotMask: mask is required");
        __bro_native.tensor.buildSlotMask(x, offset, K, stride, mask); chk();
    });
    fn(ns_tensor, "copyD2D", function copyD2D(src, srcOff, dst, dstOff, n) {
        if (src === undefined) throw new TypeError("bro.tensor.copyD2D: src is required");
        if (srcOff === undefined) throw new TypeError("bro.tensor.copyD2D: srcOff is required");
        if (dst === undefined) throw new TypeError("bro.tensor.copyD2D: dst is required");
        if (dstOff === undefined) throw new TypeError("bro.tensor.copyD2D: dstOff is required");
        if (n === undefined) throw new TypeError("bro.tensor.copyD2D: n is required");
        __bro_native.tensor.copyD2D(src, srcOff, dst, dstOff, n); chk();
    });
    // cast(src, dst, outDtype): dst = src converted (resized + dtype-set on
    // src's device). FP32 <-> FP16 <-> BF16 plus a same-dtype copy.
    fn(ns_tensor, "cast", function cast(src, dst, outDtype) {
        if (!(src instanceof GpuTensor) || !(dst instanceof GpuTensor)) throw new TypeError("cast(src, dst, outDtype): src and dst must be GpuTensors");
        __bro_native.tensor.cast(src, dst, dtypeName(outDtype, "fp32"));
        chk();
    });
    fn(ns_tensor, "siluForward", function siluForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.siluForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.siluForward: y is required");
        __bro_native.tensor.siluForward(x, y); chk();
    });
    fn(ns_tensor, "siluBackward", function siluBackward(x, dY, dX) {
        if (x === undefined) throw new TypeError("bro.tensor.siluBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.siluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.siluBackward: dX is required");
        __bro_native.tensor.siluBackward(x, dY, dX); chk();
    });
    fn(ns_tensor, "geluForward", function geluForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.geluForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.geluForward: y is required");
        __bro_native.tensor.geluForward(x, y); chk();
    });
    fn(ns_tensor, "geluBackward", function geluBackward(x, dY, dX) {
        if (x === undefined) throw new TypeError("bro.tensor.geluBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.geluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.geluBackward: dX is required");
        __bro_native.tensor.geluBackward(x, dY, dX); chk();
    });
    fn(ns_tensor, "geluExactForward", function geluExactForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.geluExactForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.geluExactForward: y is required");
        __bro_native.tensor.geluExactForward(x, y); chk();
    });
    fn(ns_tensor, "geluExactBackward", function geluExactBackward(x, dY, dX) {
        if (x === undefined) throw new TypeError("bro.tensor.geluExactBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.geluExactBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.geluExactBackward: dX is required");
        __bro_native.tensor.geluExactBackward(x, dY, dX); chk();
    });
    fn(ns_tensor, "quickGeluForward", function quickGeluForward(x, y) {
        if (x === undefined) throw new TypeError("bro.tensor.quickGeluForward: x is required");
        if (y === undefined) throw new TypeError("bro.tensor.quickGeluForward: y is required");
        __bro_native.tensor.quickGeluForward(x, y); chk();
    });
    fn(ns_tensor, "quickGeluBackward", function quickGeluBackward(x, dY, dX) {
        if (x === undefined) throw new TypeError("bro.tensor.quickGeluBackward: x is required");
        if (dY === undefined) throw new TypeError("bro.tensor.quickGeluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.quickGeluBackward: dX is required");
        __bro_native.tensor.quickGeluBackward(x, dY, dX); chk();
    });
    fn(ns_tensor, "swigluForward", function swigluForward(X, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.swigluForward: X is required");
        if (Y === undefined) throw new TypeError("bro.tensor.swigluForward: Y is required");
        __bro_native.tensor.swigluForward(X, Y); chk();
    });
    fn(ns_tensor, "swigluBackward", function swigluBackward(X, dY, dX) {
        if (X === undefined) throw new TypeError("bro.tensor.swigluBackward: X is required");
        if (dY === undefined) throw new TypeError("bro.tensor.swigluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.swigluBackward: dX is required");
        __bro_native.tensor.swigluBackward(X, dY, dX); chk();
    });
    fn(ns_tensor, "gegluForward", function gegluForward(X, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.gegluForward: X is required");
        if (Y === undefined) throw new TypeError("bro.tensor.gegluForward: Y is required");
        __bro_native.tensor.gegluForward(X, Y); chk();
    });
    fn(ns_tensor, "gegluBackward", function gegluBackward(X, dY, dX) {
        if (X === undefined) throw new TypeError("bro.tensor.gegluBackward: X is required");
        if (dY === undefined) throw new TypeError("bro.tensor.gegluBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.gegluBackward: dX is required");
        __bro_native.tensor.gegluBackward(X, dY, dX); chk();
    });
    fn(ns_tensor, "gegluExactForward", function gegluExactForward(X, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.gegluExactForward: X is required");
        if (Y === undefined) throw new TypeError("bro.tensor.gegluExactForward: Y is required");
        __bro_native.tensor.gegluExactForward(X, Y); chk();
    });
    fn(ns_tensor, "gegluExactBackward", function gegluExactBackward(X, dY, dX) {
        if (X === undefined) throw new TypeError("bro.tensor.gegluExactBackward: X is required");
        if (dY === undefined) throw new TypeError("bro.tensor.gegluExactBackward: dY is required");
        if (dX === undefined) throw new TypeError("bro.tensor.gegluExactBackward: dX is required");
        __bro_native.tensor.gegluExactBackward(X, dY, dX); chk();
    });
    // softmaxForward(logits, probs, mask|null): the old third argument is a
    // length-N FP32 GpuTensor key mask (1 valid / 0 invalid). A number there
    // is the temperature the port introduced, kept for callers of that form.
    fn(ns_tensor, "softmaxForward", function softmaxForward(logits, probs, maskOrTemp) {
        if (logits === undefined || probs === undefined) throw new TypeError("bro.tensor.softmaxForward: logits, probs required");
        if (typeof maskOrTemp === "number") {
            __bro_native.tensor.softmaxForward(logits, probs, maskOrTemp); chk();
            return;
        }
        if (maskOrTemp !== undefined && maskOrTemp !== null && !(maskOrTemp instanceof GpuTensor)) {
            throw new TypeError("softmaxForward(logits, probs, mask|null): mask must be null or a GpuTensor");
        }
        __bro_native.tensor.softmaxForwardMasked(logits, probs, maskOrTemp === undefined ? null : maskOrTemp);
        chk();
    });
    fn(ns_tensor, "softmaxBackward", function softmaxBackward(probs, dProbs, dLogits) {
        if (probs === undefined) throw new TypeError("bro.tensor.softmaxBackward: probs is required");
        if (dProbs === undefined) throw new TypeError("bro.tensor.softmaxBackward: dProbs is required");
        if (dLogits === undefined) throw new TypeError("bro.tensor.softmaxBackward: dLogits is required");
        __bro_native.tensor.softmaxBackward(probs, dProbs, dLogits); chk();
    });
    // layernormForward(x, gamma, beta, y, xhat, eps) -> {mean, rstd}, the
    // scalar caches layernormBackward takes.
    fn(ns_tensor, "layernormForward", function layernormForward(x, gamma, beta, y, xhat, eps) {
        if (x === undefined || gamma === undefined || beta === undefined || y === undefined || xhat === undefined) {
            throw new TypeError("layernormForward(x, gamma, beta, y, xhat, eps)");
        }
        const stats = __bro_native.tensor.layernormForward(x, gamma, beta, y, xhat, eps === undefined ? 0.00001 : eps);
        chk();
        return { mean: stats[0], rstd: stats[1] };
    });
    fn(ns_tensor, "layernormBackward", function layernormBackward(dY, xhat, gamma, rstd, dX, dGamma, dBeta) {
        if (dY === undefined) throw new TypeError("bro.tensor.layernormBackward: dY is required");
        if (xhat === undefined) throw new TypeError("bro.tensor.layernormBackward: xhat is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.layernormBackward: gamma is required");
        if (rstd === undefined) throw new TypeError("bro.tensor.layernormBackward: rstd is required");
        if (dX === undefined) throw new TypeError("bro.tensor.layernormBackward: dX is required");
        if (dGamma === undefined) throw new TypeError("bro.tensor.layernormBackward: dGamma is required");
        if (dBeta === undefined) throw new TypeError("bro.tensor.layernormBackward: dBeta is required");
        __bro_native.tensor.layernormBackward(dY, xhat, gamma, rstd, dX, dGamma, dBeta); chk();
    });
    fn(ns_tensor, "layernormForwardInferenceBatched", function layernormForwardInferenceBatched(X_RD, gamma, beta, Y_RD, eps) {
        if (X_RD === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatched: X_RD is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatched: gamma is required");
        if (beta === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatched: beta is required");
        if (Y_RD === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatched: Y_RD is required");
        __bro_native.tensor.layernormForwardInferenceBatched(X_RD, gamma, beta, Y_RD, eps === undefined ? 0.00001 : eps); chk();
    });
    fn(ns_tensor, "layernormForwardInferenceBatchedFp16", function layernormForwardInferenceBatchedFp16(X_RD, gamma, beta, Y_RD, eps) {
        if (X_RD === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatchedFp16: X_RD is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatchedFp16: gamma is required");
        if (beta === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatchedFp16: beta is required");
        if (Y_RD === undefined) throw new TypeError("bro.tensor.layernormForwardInferenceBatchedFp16: Y_RD is required");
        __bro_native.tensor.layernormForwardInferenceBatchedFp16(X_RD, gamma, beta, Y_RD, eps === undefined ? 0.00001 : eps); chk();
    });
    fn(ns_tensor, "rmsNormForward", function rmsNormForward(X, gamma, eps, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.rmsNormForward: X is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.rmsNormForward: gamma is required");
        if (eps === undefined) throw new TypeError("bro.tensor.rmsNormForward: eps is required");
        if (Y === undefined) throw new TypeError("bro.tensor.rmsNormForward: Y is required");
        __bro_native.tensor.rmsNormForward(X, gamma, eps, Y); chk();
    });
    fn(ns_tensor, "rmsNormBackward", function rmsNormBackward(X, gamma, dY, eps, dX, dGamma) {
        if (X === undefined) throw new TypeError("bro.tensor.rmsNormBackward: X is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.rmsNormBackward: gamma is required");
        if (dY === undefined) throw new TypeError("bro.tensor.rmsNormBackward: dY is required");
        if (eps === undefined) throw new TypeError("bro.tensor.rmsNormBackward: eps is required");
        if (dX === undefined) throw new TypeError("bro.tensor.rmsNormBackward: dX is required");
        if (dGamma === undefined) throw new TypeError("bro.tensor.rmsNormBackward: dGamma is required");
        __bro_native.tensor.rmsNormBackward(X, gamma, dY, eps, dX, dGamma); chk();
    });
    fn(ns_tensor, "groupNormForward", function groupNormForward(X, gamma, beta, N, C, H, W, numGroups, eps, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.groupNormForward: X is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.groupNormForward: gamma is required");
        if (beta === undefined) throw new TypeError("bro.tensor.groupNormForward: beta is required");
        if (N === undefined) throw new TypeError("bro.tensor.groupNormForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.groupNormForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.groupNormForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.groupNormForward: W is required");
        if (numGroups === undefined) throw new TypeError("bro.tensor.groupNormForward: numGroups is required");
        if (eps === undefined) throw new TypeError("bro.tensor.groupNormForward: eps is required");
        if (Y === undefined) throw new TypeError("bro.tensor.groupNormForward: Y is required");
        __bro_native.tensor.groupNormForward(X, gamma, beta, N, C, H, W, numGroups, eps, Y); chk();
    });
    fn(ns_tensor, "groupNormBackward", function groupNormBackward(X, gamma, dY, N, C, H, W, numGroups, eps, dX, dGamma, dBeta) {
        if (X === undefined) throw new TypeError("bro.tensor.groupNormBackward: X is required");
        if (gamma === undefined) throw new TypeError("bro.tensor.groupNormBackward: gamma is required");
        if (dY === undefined) throw new TypeError("bro.tensor.groupNormBackward: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.groupNormBackward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.groupNormBackward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.groupNormBackward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.groupNormBackward: W is required");
        if (numGroups === undefined) throw new TypeError("bro.tensor.groupNormBackward: numGroups is required");
        if (eps === undefined) throw new TypeError("bro.tensor.groupNormBackward: eps is required");
        if (dX === undefined) throw new TypeError("bro.tensor.groupNormBackward: dX is required");
        if (dGamma === undefined) throw new TypeError("bro.tensor.groupNormBackward: dGamma is required");
        if (dBeta === undefined) throw new TypeError("bro.tensor.groupNormBackward: dBeta is required");
        __bro_native.tensor.groupNormBackward(X, gamma, dY, N, C, H, W, numGroups, eps, dX, dGamma, dBeta); chk();
    });
    fn(ns_tensor, "matmul", function matmul(A, B, C) {
        if (arguments.length === 0) throw new TypeError("matmul: arguments required");
        if (!A || !B || !C || !(A instanceof GpuTensor) || !(B instanceof GpuTensor) || !(C instanceof GpuTensor)) {
            throw new TypeError("matmul: arguments must be GpuTensor instances");
        }
        __bro_native.tensor.matmul(A, B, C); chk();
    });
    fn(ns_tensor, "matmulBackward", function matmulBackward(A, B, dC, dA, dB) {
        if (A === undefined) throw new TypeError("bro.tensor.matmulBackward: A is required");
        if (B === undefined) throw new TypeError("bro.tensor.matmulBackward: B is required");
        if (dC === undefined) throw new TypeError("bro.tensor.matmulBackward: dC is required");
        if (dA === undefined) throw new TypeError("bro.tensor.matmulBackward: dA is required");
        if (dB === undefined) throw new TypeError("bro.tensor.matmulBackward: dB is required");
        __bro_native.tensor.matmulBackward(A, B, dC, dA, dB); chk();
    });
    fn(ns_tensor, "ropeForward", function ropeForward(X, headDim, numHeads, seqOffset, thetaBase, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.ropeForward: X is required");
        if (headDim === undefined) throw new TypeError("bro.tensor.ropeForward: headDim is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.ropeForward: numHeads is required");
        if (seqOffset === undefined) throw new TypeError("bro.tensor.ropeForward: seqOffset is required");
        if (thetaBase === undefined) throw new TypeError("bro.tensor.ropeForward: thetaBase is required");
        if (Y === undefined) throw new TypeError("bro.tensor.ropeForward: Y is required");
        __bro_native.tensor.ropeForward(X, headDim, numHeads, seqOffset, thetaBase, Y); chk();
    });
    fn(ns_tensor, "ropeBackward", function ropeBackward(dY, headDim, numHeads, seqOffset, thetaBase, dX) {
        if (dY === undefined) throw new TypeError("bro.tensor.ropeBackward: dY is required");
        if (headDim === undefined) throw new TypeError("bro.tensor.ropeBackward: headDim is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.ropeBackward: numHeads is required");
        if (seqOffset === undefined) throw new TypeError("bro.tensor.ropeBackward: seqOffset is required");
        if (thetaBase === undefined) throw new TypeError("bro.tensor.ropeBackward: thetaBase is required");
        if (dX === undefined) throw new TypeError("bro.tensor.ropeBackward: dX is required");
        __bro_native.tensor.ropeBackward(dY, headDim, numHeads, seqOffset, thetaBase, dX); chk();
    });
    fn(ns_tensor, "ropeApply", function ropeApply(X, cosTbl, sinTbl, headDim, numHeads, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.ropeApply: X is required");
        if (cosTbl === undefined) throw new TypeError("bro.tensor.ropeApply: cosTbl is required");
        if (sinTbl === undefined) throw new TypeError("bro.tensor.ropeApply: sinTbl is required");
        if (headDim === undefined) throw new TypeError("bro.tensor.ropeApply: headDim is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.ropeApply: numHeads is required");
        if (Y === undefined) throw new TypeError("bro.tensor.ropeApply: Y is required");
        __bro_native.tensor.ropeApply(X, cosTbl, sinTbl, headDim, numHeads, Y); chk();
    });
    fn(ns_tensor, "ropeApplyBackward", function ropeApplyBackward(dY, headDim, numHeads, dX) {
        if (dY === undefined) throw new TypeError("bro.tensor.ropeApplyBackward: dY is required");
        if (headDim === undefined) throw new TypeError("bro.tensor.ropeApplyBackward: headDim is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.ropeApplyBackward: numHeads is required");
        if (dX === undefined) throw new TypeError("bro.tensor.ropeApplyBackward: dX is required");
        __bro_native.tensor.ropeApplyBackward(dY, headDim, numHeads, dX); chk();
    });
    fn(ns_tensor, "modulate", function modulate(X, scale, shift, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.modulate: X is required");
        if (scale === undefined) throw new TypeError("bro.tensor.modulate: scale is required");
        if (shift === undefined) throw new TypeError("bro.tensor.modulate: shift is required");
        if (Y === undefined) throw new TypeError("bro.tensor.modulate: Y is required");
        __bro_native.tensor.modulate(X, scale, shift, Y); chk();
    });
    fn(ns_tensor, "broadcastMul", function broadcastMul(X, v, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.broadcastMul: X is required");
        if (v === undefined) throw new TypeError("bro.tensor.broadcastMul: v is required");
        if (Y === undefined) throw new TypeError("bro.tensor.broadcastMul: Y is required");
        __bro_native.tensor.broadcastMul(X, v, Y); chk();
    });
    fn(ns_tensor, "sumRows", function sumRows(X, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.sumRows: X is required");
        if (Y === undefined) throw new TypeError("bro.tensor.sumRows: Y is required");
        __bro_native.tensor.sumRows(X, Y); chk();
    });
    fn(ns_tensor, "sumCols", function sumCols(X, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.sumCols: X is required");
        if (Y === undefined) throw new TypeError("bro.tensor.sumCols: Y is required");
        __bro_native.tensor.sumCols(X, Y); chk();
    });
    fn(ns_tensor, "argmaxRows", function argmaxRows(X, Idx) {
        if (X === undefined) throw new TypeError("bro.tensor.argmaxRows: X is required");
        if (Idx === undefined) throw new TypeError("bro.tensor.argmaxRows: Idx is required");
        __bro_native.tensor.argmaxRows(X, Idx); chk();
    });
    fn(ns_tensor, "attentionTokenMoments", function attentionTokenMoments(Attn, h_lat, w_lat, mass, centroid) {
        if (Attn === undefined) throw new TypeError("bro.tensor.attentionTokenMoments: Attn is required");
        if (h_lat === undefined) throw new TypeError("bro.tensor.attentionTokenMoments: h_lat is required");
        if (w_lat === undefined) throw new TypeError("bro.tensor.attentionTokenMoments: w_lat is required");
        if (mass === undefined) throw new TypeError("bro.tensor.attentionTokenMoments: mass is required");
        if (centroid === undefined) throw new TypeError("bro.tensor.attentionTokenMoments: centroid is required");
        __bro_native.tensor.attentionTokenMoments(Attn, h_lat, w_lat, mass, centroid); chk();
    });
    fn(ns_tensor, "buildCausalMaskRow", function buildCausalMaskRow(L, q, mask) {
        if (L === undefined) throw new TypeError("bro.tensor.buildCausalMaskRow: L is required");
        if (q === undefined) throw new TypeError("bro.tensor.buildCausalMaskRow: q is required");
        if (mask === undefined) throw new TypeError("bro.tensor.buildCausalMaskRow: mask is required");
        __bro_native.tensor.buildCausalMaskRow(L, q, mask); chk();
    });
    fn(ns_tensor, "flashAttentionDecode", function flashAttentionDecode(Q, K_cache, V_cache, validLen, numHeads, O, numKvHeads, attnSoftcap, window) {
        if (Q === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: Q is required");
        if (K_cache === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: K_cache is required");
        if (V_cache === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: V_cache is required");
        if (validLen === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: validLen is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: numHeads is required");
        if (O === undefined) throw new TypeError("bro.tensor.flashAttentionDecode: O is required");
        __bro_native.tensor.flashAttentionDecode(Q, K_cache, V_cache, validLen, numHeads, O, numKvHeads !== undefined, numKvHeads === undefined ? 0 : numKvHeads, attnSoftcap === undefined ? 0 : attnSoftcap, window === undefined ? 0 : window); chk();
    });
    fn(ns_tensor, "flashAttentionDecodeMasked", function flashAttentionDecodeMasked(Q, K_cache, V_cache, dMask, numHeads, O, numKvHeads, attnSoftcap, window) {
        if (Q === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: Q is required");
        if (K_cache === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: K_cache is required");
        if (V_cache === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: V_cache is required");
        if (dMask === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: dMask is required");
        if (numHeads === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: numHeads is required");
        if (O === undefined) throw new TypeError("bro.tensor.flashAttentionDecodeMasked: O is required");
        __bro_native.tensor.flashAttentionDecodeMasked(Q, K_cache, V_cache, dMask, numHeads, O, numKvHeads !== undefined, numKvHeads === undefined ? 0 : numKvHeads, attnSoftcap === undefined ? 0 : attnSoftcap, window === undefined ? 0 : window); chk();
    });
    fn(ns_tensor, "kvCacheAppend", function kvCacheAppend(K_new, V_new, curLen, K_cache, V_cache) {
        if (K_new === undefined) throw new TypeError("bro.tensor.kvCacheAppend: K_new is required");
        if (V_new === undefined) throw new TypeError("bro.tensor.kvCacheAppend: V_new is required");
        if (curLen === undefined) throw new TypeError("bro.tensor.kvCacheAppend: curLen is required");
        if (K_cache === undefined) throw new TypeError("bro.tensor.kvCacheAppend: K_cache is required");
        if (V_cache === undefined) throw new TypeError("bro.tensor.kvCacheAppend: V_cache is required");
        __bro_native.tensor.kvCacheAppend(K_new, V_new, curLen, K_cache, V_cache); chk();
    });
    fn(ns_tensor, "conv2dForward", function conv2dForward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, Y) {
        if (X === undefined || Wt === undefined || Y === undefined) throw new TypeError("bro.tensor.conv2dForward: X, Wt, Y required");
        __bro_native.tensor.conv2dForward(X, Wt, bias || null, N, C_in, H, W, C_out, kH, kW, sH || 1, sW || 1, pH || 0, pW || 0, dH || 1, dW || 1, groups || 1, Y); chk();
    });
    fn(ns_tensor, "conv2dBackwardInput", function conv2dBackwardInput(Wt, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, dX) {
        if (Wt === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: Wt is required");
        if (dY === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: N is required");
        if (C_in === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: C_in is required");
        if (H === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: W is required");
        if (C_out === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: C_out is required");
        if (kH === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: kH is required");
        if (kW === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: kW is required");
        if (sH === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: sH is required");
        if (sW === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: sW is required");
        if (pH === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: pH is required");
        if (pW === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: pW is required");
        if (dH === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: dH is required");
        if (dW === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: dW is required");
        if (groups === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: groups is required");
        if (dX === undefined) throw new TypeError("bro.tensor.conv2dBackwardInput: dX is required");
        __bro_native.tensor.conv2dBackwardInput(Wt, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, dX); chk();
    });
    fn(ns_tensor, "conv2dBackwardWeight", function conv2dBackwardWeight(X, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, dWt) {
        if (X === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: X is required");
        if (dY === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: N is required");
        if (C_in === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: C_in is required");
        if (H === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: W is required");
        if (C_out === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: C_out is required");
        if (kH === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: kH is required");
        if (kW === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: kW is required");
        if (sH === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: sH is required");
        if (sW === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: sW is required");
        if (pH === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: pH is required");
        if (pW === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: pW is required");
        if (dH === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: dH is required");
        if (dW === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: dW is required");
        if (groups === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: groups is required");
        if (dWt === undefined) throw new TypeError("bro.tensor.conv2dBackwardWeight: dWt is required");
        __bro_native.tensor.conv2dBackwardWeight(X, dY, N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, dH, dW, groups, dWt); chk();
    });
    fn(ns_tensor, "conv2dBackwardBias", function conv2dBackwardBias(dY, N, C_out, H_out, W_out, dB) {
        if (dY === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: N is required");
        if (C_out === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: C_out is required");
        if (H_out === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: H_out is required");
        if (W_out === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: W_out is required");
        if (dB === undefined) throw new TypeError("bro.tensor.conv2dBackwardBias: dB is required");
        __bro_native.tensor.conv2dBackwardBias(dY, N, C_out, H_out, W_out, dB); chk();
    });
    fn(ns_tensor, "upsampleNearest2xForward", function upsampleNearest2xForward(X, N, C, H, W, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: W is required");
        if (Y === undefined) throw new TypeError("bro.tensor.upsampleNearest2xForward: Y is required");
        __bro_native.tensor.upsampleNearest2xForward(X, N, C, H, W, Y); chk();
    });
    fn(ns_tensor, "upsampleNearest2xBackward", function upsampleNearest2xBackward(dY, N, C, H, W, dX) {
        if (dY === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: W is required");
        if (dX === undefined) throw new TypeError("bro.tensor.upsampleNearest2xBackward: dX is required");
        __bro_native.tensor.upsampleNearest2xBackward(dY, N, C, H, W, dX); chk();
    });
    fn(ns_tensor, "upsampleBilinear2xForward", function upsampleBilinear2xForward(X, N, C, H, W, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: W is required");
        if (Y === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xForward: Y is required");
        __bro_native.tensor.upsampleBilinear2xForward(X, N, C, H, W, Y); chk();
    });
    fn(ns_tensor, "upsampleBilinear2xBackward", function upsampleBilinear2xBackward(dY, N, C, H, W, dX) {
        if (dY === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: W is required");
        if (dX === undefined) throw new TypeError("bro.tensor.upsampleBilinear2xBackward: dX is required");
        __bro_native.tensor.upsampleBilinear2xBackward(dY, N, C, H, W, dX); chk();
    });
    fn(ns_tensor, "downsampleAvg2xForward", function downsampleAvg2xForward(X, N, C, H, W, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: W is required");
        if (Y === undefined) throw new TypeError("bro.tensor.downsampleAvg2xForward: Y is required");
        __bro_native.tensor.downsampleAvg2xForward(X, N, C, H, W, Y); chk();
    });
    fn(ns_tensor, "downsampleAvg2xBackward", function downsampleAvg2xBackward(dY, N, C, H, W, dX) {
        if (dY === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: dY is required");
        if (N === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: W is required");
        if (dX === undefined) throw new TypeError("bro.tensor.downsampleAvg2xBackward: dX is required");
        __bro_native.tensor.downsampleAvg2xBackward(dY, N, C, H, W, dX); chk();
    });
    fn(ns_tensor, "nchwToSequence", function nchwToSequence(X, N, C, H, W, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.nchwToSequence: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.nchwToSequence: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.nchwToSequence: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.nchwToSequence: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.nchwToSequence: W is required");
        if (Y === undefined) throw new TypeError("bro.tensor.nchwToSequence: Y is required");
        __bro_native.tensor.nchwToSequence(X, N, C, H, W, Y); chk();
    });
    fn(ns_tensor, "sequenceToNchw", function sequenceToNchw(X, N, C, H, W, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.sequenceToNchw: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.sequenceToNchw: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.sequenceToNchw: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.sequenceToNchw: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.sequenceToNchw: W is required");
        if (Y === undefined) throw new TypeError("bro.tensor.sequenceToNchw: Y is required");
        __bro_native.tensor.sequenceToNchw(X, N, C, H, W, Y); chk();
    });
    fn(ns_tensor, "interp2dForward", function interp2dForward(X, N, C, H_in, W_in, H_out, W_out, mode, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.interp2dForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.interp2dForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.interp2dForward: C is required");
        if (H_in === undefined) throw new TypeError("bro.tensor.interp2dForward: H_in is required");
        if (W_in === undefined) throw new TypeError("bro.tensor.interp2dForward: W_in is required");
        if (H_out === undefined) throw new TypeError("bro.tensor.interp2dForward: H_out is required");
        if (W_out === undefined) throw new TypeError("bro.tensor.interp2dForward: W_out is required");
        if (mode === undefined) throw new TypeError("bro.tensor.interp2dForward: mode is required");
        if (Y === undefined) throw new TypeError("bro.tensor.interp2dForward: Y is required");
        __bro_native.tensor.interp2dForward(X, N, C, H_in, W_in, H_out, W_out, mode, Y); chk();
    });
    fn(ns_tensor, "interp2dAlignCornersForward", function interp2dAlignCornersForward(X, N, C, H_in, W_in, H_out, W_out, mode, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: C is required");
        if (H_in === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: H_in is required");
        if (W_in === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: W_in is required");
        if (H_out === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: H_out is required");
        if (W_out === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: W_out is required");
        if (mode === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: mode is required");
        if (Y === undefined) throw new TypeError("bro.tensor.interp2dAlignCornersForward: Y is required");
        __bro_native.tensor.interp2dAlignCornersForward(X, N, C, H_in, W_in, H_out, W_out, mode, Y); chk();
    });
    fn(ns_tensor, "unfold2dForward", function unfold2dForward(X, N, C, H, W, kH, kW, sH, sW, padT, padB, padL, padR, mode, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.unfold2dForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.unfold2dForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.unfold2dForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.unfold2dForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.unfold2dForward: W is required");
        if (kH === undefined) throw new TypeError("bro.tensor.unfold2dForward: kH is required");
        if (kW === undefined) throw new TypeError("bro.tensor.unfold2dForward: kW is required");
        if (sH === undefined) throw new TypeError("bro.tensor.unfold2dForward: sH is required");
        if (sW === undefined) throw new TypeError("bro.tensor.unfold2dForward: sW is required");
        if (padT === undefined) throw new TypeError("bro.tensor.unfold2dForward: padT is required");
        if (padB === undefined) throw new TypeError("bro.tensor.unfold2dForward: padB is required");
        if (padL === undefined) throw new TypeError("bro.tensor.unfold2dForward: padL is required");
        if (padR === undefined) throw new TypeError("bro.tensor.unfold2dForward: padR is required");
        if (mode === undefined) throw new TypeError("bro.tensor.unfold2dForward: mode is required");
        if (Y === undefined) throw new TypeError("bro.tensor.unfold2dForward: Y is required");
        __bro_native.tensor.unfold2dForward(X, N, C, H, W, kH, kW, sH, sW, padT, padB, padL, padR, mode, Y); chk();
    });
    fn(ns_tensor, "l2NormalizeNchwForward", function l2NormalizeNchwForward(X, N, C, H, W, eps, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: X is required");
        if (N === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: W is required");
        if (eps === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: eps is required");
        if (Y === undefined) throw new TypeError("bro.tensor.l2NormalizeNchwForward: Y is required");
        __bro_native.tensor.l2NormalizeNchwForward(X, N, C, H, W, eps, Y); chk();
    });
    fn(ns_tensor, "convexUpsampleForward", function convexUpsampleForward(X, Mask, N, C, H, W, scale, Y) {
        if (X === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: X is required");
        if (Mask === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: Mask is required");
        if (N === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: N is required");
        if (C === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: C is required");
        if (H === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: H is required");
        if (W === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: W is required");
        if (scale === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: scale is required");
        if (Y === undefined) throw new TypeError("bro.tensor.convexUpsampleForward: Y is required");
        __bro_native.tensor.convexUpsampleForward(X, Mask, N, C, H, W, scale, Y); chk();
    });
    fn(ns_tensor, "mseVecForward", function mseVecForward(pred, target) {
        if (pred === undefined) throw new TypeError("bro.tensor.mseVecForward: pred is required");
        if (target === undefined) throw new TypeError("bro.tensor.mseVecForward: target is required");
        const res = __bro_native.tensor.mseVecForward(pred, target); chk(); return res;
    });
    fn(ns_tensor, "mseVecBackward", function mseVecBackward(pred, target, dPred) {
        if (pred === undefined) throw new TypeError("bro.tensor.mseVecBackward: pred is required");
        if (target === undefined) throw new TypeError("bro.tensor.mseVecBackward: target is required");
        if (dPred === undefined) throw new TypeError("bro.tensor.mseVecBackward: dPred is required");
        __bro_native.tensor.mseVecBackward(pred, target, dPred); chk();
    });
    fn(ns_tensor, "mseVecPerSample", function mseVecPerSample(pred, target, dPred, lossPerSample) {
        if (pred === undefined) throw new TypeError("bro.tensor.mseVecPerSample: pred is required");
        if (target === undefined) throw new TypeError("bro.tensor.mseVecPerSample: target is required");
        if (dPred === undefined) throw new TypeError("bro.tensor.mseVecPerSample: dPred is required");
        if (lossPerSample === undefined) throw new TypeError("bro.tensor.mseVecPerSample: lossPerSample is required");
        __bro_native.tensor.mseVecPerSample(pred, target, dPred, lossPerSample); chk();
    });
    fn(ns_tensor, "embeddingLookupForward", function embeddingLookupForward(table, idxAsInt32, B, out) {
        if (table === undefined) throw new TypeError("bro.tensor.embeddingLookupForward: table is required");
        if (idxAsInt32 === undefined) throw new TypeError("bro.tensor.embeddingLookupForward: idxAsInt32 is required");
        if (B === undefined) throw new TypeError("bro.tensor.embeddingLookupForward: B is required");
        if (out === undefined) throw new TypeError("bro.tensor.embeddingLookupForward: out is required");
        __bro_native.tensor.embeddingLookupForward(table, idxAsInt32, B, out); chk();
    });
    fn(ns_tensor, "embeddingLookupBackward", function embeddingLookupBackward(dOut, idxAsInt32, B, dTable) {
        if (dOut === undefined) throw new TypeError("bro.tensor.embeddingLookupBackward: dOut is required");
        if (idxAsInt32 === undefined) throw new TypeError("bro.tensor.embeddingLookupBackward: idxAsInt32 is required");
        if (B === undefined) throw new TypeError("bro.tensor.embeddingLookupBackward: B is required");
        if (dTable === undefined) throw new TypeError("bro.tensor.embeddingLookupBackward: dTable is required");
        __bro_native.tensor.embeddingLookupBackward(dOut, idxAsInt32, B, dTable); chk();
    });
    fn(ns_tensor, "sgdStep", function sgdStep(param, grad, velocity, lr, momentum) {
        if (param === undefined) throw new TypeError("bro.tensor.sgdStep: param is required");
        if (grad === undefined) throw new TypeError("bro.tensor.sgdStep: grad is required");
        if (velocity === undefined) throw new TypeError("bro.tensor.sgdStep: velocity is required");
        if (lr === undefined) throw new TypeError("bro.tensor.sgdStep: lr is required");
        if (momentum === undefined) throw new TypeError("bro.tensor.sgdStep: momentum is required");
        __bro_native.tensor.sgdStep(param, grad, velocity, lr, momentum); chk();
    });
    fn(ns_tensor, "adamStep", function adamStep(param, grad, m, v, lr, beta1, beta2, eps, step) {
        if (param === undefined) throw new TypeError("bro.tensor.adamStep: param is required");
        if (grad === undefined) throw new TypeError("bro.tensor.adamStep: grad is required");
        if (m === undefined) throw new TypeError("bro.tensor.adamStep: m is required");
        if (v === undefined) throw new TypeError("bro.tensor.adamStep: v is required");
        if (lr === undefined) throw new TypeError("bro.tensor.adamStep: lr is required");
        if (beta1 === undefined) throw new TypeError("bro.tensor.adamStep: beta1 is required");
        if (beta2 === undefined) throw new TypeError("bro.tensor.adamStep: beta2 is required");
        if (eps === undefined) throw new TypeError("bro.tensor.adamStep: eps is required");
        if (step === undefined) throw new TypeError("bro.tensor.adamStep: step is required");
        __bro_native.tensor.adamStep(param, grad, m, v, lr, beta1, beta2, eps, step); chk();
    });

    // ---- bro.tensor.GpuTensor ------------------------------------------------
    function GpuTensor() {
        throw new TypeError("bro.tensor.GpuTensor is not constructible: instances come from the natives that return one");
    }
    {
        const proto = __bro_native.tensor.GpuTensorProto;
        if (proto === undefined) throw new Error("bro.tensor.GpuTensor: native class prototype not published (registerNatives_tensor did not run)");
        Object.setPrototypeOf(proto, GpuTensor.prototype);
    }
    fn(mount(bro, "tensor"), "GpuTensor", GpuTensor);
    accessor(GpuTensor.prototype, "rows",
        function () {
            return __bro_native.tensor.GpuTensor_rows_get(this);
        },
        undefined);
    accessor(GpuTensor.prototype, "cols",
        function () {
            return __bro_native.tensor.GpuTensor_cols_get(this);
        },
        undefined);
    accessor(GpuTensor.prototype, "size",
        function () {
            return __bro_native.tensor.GpuTensor_size_get(this);
        },
        undefined);
    accessor(GpuTensor.prototype, "bytes",
        function () {
            return __bro_native.tensor.GpuTensor_bytes_get(this);
        },
        undefined);
    accessor(GpuTensor.prototype, "shape",
        function () {
            if (this._shape && Array.isArray(this._shape)) return this._shape.slice();
            return [this.rows, this.cols];
        },
        function (s) {
            if (Array.isArray(s)) this._shape = s.slice();
        });
    fn(GpuTensor.prototype, "zero", function zero() {
        __bro_native.tensor.GpuTensor_zero(this); chk();
    });
    // resize(rows, cols, dtype?): dtype a name or a bro.tensor.dtype value.
    fn(GpuTensor.prototype, "resize", function resize(rows, cols, dtype) {
        if (rows === undefined || cols === undefined) throw new TypeError("resize(rows, cols, dtype?)");
        __bro_native.tensor.GpuTensor_resize(this, rows | 0, cols | 0, dtypeName(dtype, "fp32")); chk();
    });
    fn(GpuTensor.prototype, "dtype", function dtype() {
        const res = __bro_native.tensor.GpuTensor_dtype(this); chk(); return res;
    });
    fn(GpuTensor.prototype, "clone", function clone() {
        const res = __bro_native.tensor.GpuTensor_clone(this); chk(); return res;
    });
    fn(GpuTensor.prototype, "upload", function upload(src) {
        if (src === undefined) throw new TypeError("bro.tensor.GpuTensor.prototype.upload: src is required");
        __bro_native.tensor.GpuTensor_upload(this, toF32(src)); chk();
    });
    // download()    -> a fresh Float32Array (any dtype, converted to FP32).
    // download(dst) -> fills the Float32Array `dst` in place and returns it;
    //                  dst.length must be >= size. The old binding's dst was a
    //                  bro.ai host tensor, a type this runtime no longer has.
    fn(GpuTensor.prototype, "download", function download(dst) {
        if (dst === undefined || dst === null) {
            const out = __bro_native.tensor.GpuTensor_download(this);
            chk();
            return out;
        }
        if (!(dst instanceof Float32Array)) throw new TypeError("download(dst): dst must be a Float32Array (or omitted)");
        const n = __bro_native.tensor.GpuTensor_size_get(this);
        if (dst.length < n) throw new RangeError("download(dst): dst holds " + dst.length + " elements, tensor has " + n);
        if (!__bro_native.tensor.GpuTensor_downloadInto(this, dst)) chk();
        return dst;
    });
    fn(GpuTensor.prototype, "uploadFp16", function uploadFp16(data) {
        if (data === undefined) throw new TypeError("bro.tensor.GpuTensor.prototype.uploadFp16: data is required");
        __bro_native.tensor.GpuTensor_uploadFp16(this, toU16(data)); chk();
    });
    fn(GpuTensor.prototype, "downloadFp16", function downloadFp16() {
        const out = __bro_native.tensor.GpuTensor_downloadFp16(this);
        chk();
        return out;
    });
    fn(GpuTensor.prototype, "uploadInt8", function uploadInt8(data) {
        if (data === undefined) throw new TypeError("bro.tensor.GpuTensor.prototype.uploadInt8: data is required");
        __bro_native.tensor.GpuTensor_uploadInt8(this, toI8(data)); chk();
    });
    fn(GpuTensor.prototype, "downloadInt8", function downloadInt8() {
        const res = __bro_native.tensor.GpuTensor_downloadInt8(this); chk(); return res;
    });

    // ---- counter-based RNG (Philox) + init -------------------------------------
    // key / counter / state: BigInt or Number. Y must be FP32 and pre-sized.
    const rngTarget = (Y, label) => {
        if (!(Y instanceof GpuTensor)) throw new TypeError(label + ": Y must be a GpuTensor");
    };
    fn(ns_tensor, "randUniform", function randUniform(key, counter, Y) {
        rngTarget(Y, "randUniform(key, counter, Y)");
        __bro_native.tensor.randUniform(seed(key, "randUniform"), seed(counter, "randUniform"), Y);
        chk();
    });
    fn(ns_tensor, "randn", function randn(key, counter, Y) {
        rngTarget(Y, "randn(key, counter, Y)");
        __bro_native.tensor.randn(seed(key, "randn"), seed(counter, "randn"), Y);
        chk();
    });
    fn(ns_tensor, "randBernoulli", function randBernoulli(p, key, counter, Y) {
        rngTarget(Y, "randBernoulli(p, key, counter, Y)");
        __bro_native.tensor.randBernoulli(+p, seed(key, "randBernoulli"), seed(counter, "randBernoulli"), Y);
        chk();
    });
    fn(ns_tensor, "randnTruncated", function randnTruncated(lo, hi, key, counter, Y) {
        rngTarget(Y, "randnTruncated(lo, hi, key, counter, Y)");
        __bro_native.tensor.randnTruncated(+lo, +hi, seed(key, "randnTruncated"), seed(counter, "randnTruncated"), Y);
        chk();
    });
    // xavierInit(W, rngState) -> the advanced state (a Number; the old binding
    // answered a BigInt). Thread it through successive inits.
    fn(ns_tensor, "xavierInit", function xavierInit(W, rngState) {
        if (!(W instanceof GpuTensor)) throw new TypeError("xavierInit(W, rngState): W must be a GpuTensor");
        const next = __bro_native.tensor.xavierInit(W, seed(rngState, "xavierInit"));
        chk();
        return next;
    });

    // ---- bro.tensor.SafetensorsFile --------------------------------------------
    // openSafetensors(path) mmaps the file and returns one of these; header()
    // and names() read metadata only, get() uploads one tensor, close()
    // releases the mapping early (GC does it otherwise).
    function SafetensorsFile() {
        throw new TypeError("bro.tensor.SafetensorsFile is not constructible: bro.tensor.openSafetensors returns one");
    }
    {
        const proto = __bro_native.tensor.SafetensorsFileProto;
        if (proto === undefined) throw new Error("bro.tensor.SafetensorsFile: native class prototype not published (registerNatives_tensor did not run)");
        Object.setPrototypeOf(proto, SafetensorsFile.prototype);
    }
    fn(ns_tensor, "SafetensorsFile", SafetensorsFile);
    const stOpen = (self, label) => {
        if (!__bro_native.tensor.SafetensorsFile_isOpen(self)) throw new Error(label + "() on a closed safetensors file");
    };
    accessor(SafetensorsFile.prototype, "count",
        function () {
            return __bro_native.tensor.SafetensorsFile_count_get(this);
        },
        undefined);
    fn(SafetensorsFile.prototype, "names", function names() {
        stOpen(this, "names");
        const n = __bro_native.tensor.SafetensorsFile_count_get(this);
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = __bro_native.tensor.SafetensorsFile_nameAt(this, i);
        return out;
    });
    // header() -> { name: { dtype: "F32"|"F16"|"BF16"|..., shape: number[], nbytes }, ... }
    fn(SafetensorsFile.prototype, "header", function header() {
        stOpen(this, "header");
        const n = __bro_native.tensor.SafetensorsFile_count_get(this);
        const out = {};
        for (let i = 0; i < n; i++) {
            out[__bro_native.tensor.SafetensorsFile_nameAt(this, i)] = {
                dtype: __bro_native.tensor.SafetensorsFile_dtypeAt(this, i),
                shape: Array.from(__bro_native.tensor.SafetensorsFile_shapeAt(this, i)),
                nbytes: __bro_native.tensor.SafetensorsFile_nbytesAt(this, i),
            };
        }
        return out;
    });
    // get(name, rows?, cols?, dtype?) -> GpuTensor. rows/cols omitted: the N-D
    // source flattens to (shape[0], numel/shape[0]). dtype: "native" (default,
    // the file's dtype) | "compute" (the backend's compute dtype) | "fp16";
    // the string may stand in place of, or after, rows/cols.
    fn(SafetensorsFile.prototype, "get", function get(name, a, b, c) {
        stOpen(this, "get");
        if (typeof name !== "string") throw new TypeError("get(name, rows?, cols?, dtype?)");
        let mode = "native";
        if (typeof a === "string") mode = a;
        if (typeof b === "string") mode = b;
        if (typeof c === "string") mode = c;
        let rows = 0, cols = 0;
        if (typeof a === "number" && typeof b === "number") {
            rows = a | 0;
            cols = b | 0;
        }
        const t = __bro_native.tensor.SafetensorsFile_get(this, name, rows, cols, mode);
        if (t === null) {
            const e = __bro_native.tensor.takeError() || "get: failed";
            throw e.indexOf("no tensor named") >= 0 ? new RangeError(e) : new TypeError(e);
        }
        if (rows === 0 && cols === 0) {
            const h = this.header();
            if (h && h[name] && Array.isArray(h[name].shape)) {
                t.shape = h[name].shape.slice();
            }
        }
        return t;
    });
    fn(SafetensorsFile.prototype, "close", function close() {
        __bro_native.tensor.SafetensorsFile_close(this); chk();
    });
    fn(ns_tensor, "openSafetensors", function openSafetensors(path) {
        if (typeof path !== "string") throw new TypeError("openSafetensors(path) — expected a string path");
        const f = __bro_native.tensor.openSafetensors(path);
        if (f === null) throw new TypeError(__bro_native.tensor.takeError() || ("openSafetensors: cannot open " + path));
        return f;
    });
    // saveSafetensors(path, { name: GpuTensor, ... }): FP32 and FP16 tensors,
    // shape stored as (rows, cols).
    fn(ns_tensor, "saveSafetensors", function saveSafetensors(path, tensors) {
        if (typeof path !== "string" || tensors === null || typeof tensors !== "object") {
            throw new TypeError("saveSafetensors(path, {name: tensor, ...})");
        }
        const names = Object.keys(tensors);
        const values = new Array(names.length);
        for (let i = 0; i < names.length; i++) {
            const v = tensors[names[i]];
            if (!(v instanceof GpuTensor)) throw new TypeError("saveSafetensors: value for '" + names[i] + "' is not a tensor");
            values[i] = v;
        }
        if (!__bro_native.tensor.saveSafetensors(path, names, values)) {
            throw new TypeError(__bro_native.tensor.takeError() || "saveSafetensors: failed");
        }
    });
})();
