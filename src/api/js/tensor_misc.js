// tensor_misc.js — the "misc" slice of bro.tensor: the gated-deltanet L2 norm
// pair, the three loss entry points the fused-loss family in tensor_ext.js left
// behind, and the diffusion sampler steps + sinusoidal timestep embedding.
//
// Signatures are the QuickJS binding's (tensor_bindings_activations.cpp,
// tensor_bindings_diffusion.cpp): same argument order, same optional arguments
// and defaults. `mask|null` slots take a length-N FP32 GpuTensor or null;
// softmaxXentSegment keeps its host Float32Array form (the op takes raw host
// pointers and writes probs/dLogits in place); mseScalar keeps its
// [loss, dPred] array result.
//
// Hand-written like js/tensor.js and js/tensor_ext.js, compiled into its own
// module (bronze_tensor_misc_main) that api.cpp mounts after them, so
// `bro.tensor` and `bro.tensor.GpuTensor` already exist here. Every wrapper
// checks its own arguments — the natives receive nullable slots as raw values —
// and reads the native's error back after the call.
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

    // A required Float32Array host buffer (softmaxXentSegment's four arrays).
    const F32 = (v, label, name) => {
        if (!(v instanceof Float32Array)) throw new TypeError(label + ": " + name + " must be a Float32Array");
        return v;
    };
    // "no mask" for an f32[] slot: an empty typed array, which the native reads
    // as a null pointer.
    const EMPTY_F32 = new Float32Array(0);
    const optF32 = (v, label, name) => {
        if (v === undefined || v === null) return EMPTY_F32;
        if (!(v instanceof Float32Array)) throw new TypeError(label + ": " + name + " must be null or a Float32Array");
        return v;
    };

    // ---- L2 norm (gated-deltanet per-head) ------------------------------------
    // Normalises each head's head_dim slice of an (L, numHeads*headDim) tensor
    // by sqrt(sum of squares + eps). Distinct from l2NormalizeNchwForward, which
    // works on the channel axis of an NCHW tensor.
    fn(ns_tensor, "l2NormForward", function l2NormForward(X, headDim, numHeads, eps, Y) {
        const L = "l2NormForward(X,headDim,numHeads,eps,Y)";
        __bro_native.tensor.l2NormForward(T(X, L, "X"), int(headDim, 0), int(numHeads, 1), num(eps, 0.000001), T(Y, L, "Y"));
        chk();
    });
    fn(ns_tensor, "l2NormBackward", function l2NormBackward(X, headDim, numHeads, eps, dY, dX) {
        const L = "l2NormBackward(X,headDim,numHeads,eps,dY,dX)";
        __bro_native.tensor.l2NormBackward(T(X, L, "X"), int(headDim, 0), int(numHeads, 1), num(eps, 0.000001),
            T(dY, L, "dY"), T(dX, L, "dX"));
        chk();
    });

    // ---- losses ---------------------------------------------------------------
    // bceWithLogitsFusedBatched(logits, target, mask|null, posWeight, probs, dLogits, lossPerSample)
    // Fused sigmoid + BCE over a (B,L) grid; lossPerSample is (B,1) summed over L.
    // posWeight === 1 gives standard unweighted BCE. mask is an optional (B,L)
    // FP32 GpuTensor (1 valid / 0 ignored).
    fn(ns_tensor, "bceWithLogitsFusedBatched", function bceWithLogitsFusedBatched(logits, target, mask, posWeight, probs, dLogits, lossPerSample) {
        const L = "bceWithLogitsFusedBatched(logits,target,mask|null,posWeight,probs,dLogits,lossPerSample)";
        __bro_native.tensor.bceWithLogitsFusedBatched(T(logits, L, "logits"), T(target, L, "target"),
            opt(mask, L, "mask"), num(posWeight, 1.0), T(probs, L, "probs"), T(dLogits, L, "dLogits"),
            T(lossPerSample, L, "lossPerSample"));
        chk();
    });

    // softmaxXentSegment(logits, target, probs, dLogits, n, mask|null) -> loss
    // Host Float32Array form of softmaxXent over the first n elements of each
    // buffer, so a caller can run xent on a segment of a larger buffer without
    // temporary tensors. probs and dLogits are written in place.
    fn(ns_tensor, "softmaxXentSegment", function softmaxXentSegment(logits, target, probs, dLogits, n, mask) {
        const L = "softmaxXentSegment(logits,target,probs,dLogits,n,mask|null)";
        const count = int(n, 0);
        const loss = __bro_native.tensor.softmaxXentSegment(F32(logits, L, "logits"), F32(target, L, "target"),
            F32(probs, L, "probs"), F32(dLogits, L, "dLogits"), count, optF32(mask, L, "mask"));
        chk();
        return loss;
    });

    // mseScalar(pred, target) -> [loss, dPred], with loss = 0.5*(pred-target)^2
    // and dPred = pred-target. Scalars, not tensors — the value-head MSE.
    fn(ns_tensor, "mseScalar", function mseScalar(pred, target) {
        const out = __bro_native.tensor.mseScalar(num(pred, 0), num(target, 0));
        chk();
        return [out[0], out[1]];
    });

    // ---- diffusion sampler steps ----------------------------------------------
    // ddimStep(x_t, eps_pred, alpha_t, alpha_prev, sigma_t, x_prev)
    //   x0_pred = (x_t - sqrt(1-alpha_t)*eps_pred) / sqrt(alpha_t)
    //   x_prev  = sqrt(alpha_prev)*x0_pred + sqrt(1-alpha_prev-sigma_t^2)*eps_pred
    // sigma_t = 0 is deterministic DDIM.
    fn(ns_tensor, "ddimStep", function ddimStep(x_t, eps_pred, alpha_t, alpha_prev, sigma_t, x_prev) {
        const L = "ddimStep(x_t,eps_pred,alpha_t,alpha_prev,sigma_t,x_prev)";
        __bro_native.tensor.ddimStep(T(x_t, L, "x_t"), T(eps_pred, L, "eps_pred"), num(alpha_t, 0),
            num(alpha_prev, 0), num(sigma_t, 0), T(x_prev, L, "x_prev"));
        chk();
    });

    // eulerStep(x_t, eps_pred, sigma_t, sigma_prev, x_prev)
    //   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
    // The kernel never interprets eps_pred, so this covers both the eps /
    // k-diffusion derivative form and flow-matching velocity.
    fn(ns_tensor, "eulerStep", function eulerStep(x_t, eps_pred, sigma_t, sigma_prev, x_prev) {
        const L = "eulerStep(x_t,eps_pred,sigma_t,sigma_prev,x_prev)";
        __bro_native.tensor.eulerStep(T(x_t, L, "x_t"), T(eps_pred, L, "eps_pred"), num(sigma_t, 0),
            num(sigma_prev, 0), T(x_prev, L, "x_prev"));
        chk();
    });

    // dpmpp2mStep(x_t, eps_pred, x0_prev, sigma_t, c_xt, c_x0t, c_x0prev, x_prev, x0_out)
    //   x0_t   = x_t - sigma_t*eps_pred
    //   x_prev = c_xt*x_t + c_x0t*x0_t + c_x0prev*x0_prev
    //   x0_out = x0_t   (copy into x0_prev for the next step)
    // First step, with no cached x0_prev: use eulerStep.
    fn(ns_tensor, "dpmpp2mStep", function dpmpp2mStep(x_t, eps_pred, x0_prev, sigma_t, c_xt, c_x0t, c_x0prev, x_prev, x0_out) {
        const L = "dpmpp2mStep(x_t,eps_pred,x0_prev,sigma_t,c_xt,c_x0t,c_x0prev,x_prev,x0_out)";
        __bro_native.tensor.dpmpp2mStep(T(x_t, L, "x_t"), T(eps_pred, L, "eps_pred"), T(x0_prev, L, "x0_prev"),
            num(sigma_t, 0), num(c_xt, 0), num(c_x0t, 0), num(c_x0prev, 0),
            T(x_prev, L, "x_prev"), T(x0_out, L, "x0_out"));
        chk();
    });

    // timestepEmbedding(timesteps, dim, maxPeriod, Y) — maxPeriod defaults to
    // 10000. timesteps is (N,1) FP32; Y is (N,dim) FP32, cos half first (the
    // diffusers flip_sin_to_cos=True layout).
    fn(ns_tensor, "timestepEmbedding", function timestepEmbedding(timesteps, dim, maxPeriod, Y) {
        const L = "timestepEmbedding(timesteps,dim,maxPeriod,Y)";
        __bro_native.tensor.timestepEmbedding(T(timesteps, L, "timesteps"), int(dim, 0), num(maxPeriod, 10000),
            T(Y, L, "Y"));
        chk();
    });
})();
