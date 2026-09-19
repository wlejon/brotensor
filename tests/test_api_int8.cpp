// test_api_int8.cpp — checks for the restored INT8 / GGUF-quant group of
// bro.tensor (src/api/native_tensor_int8.cpp + src/api/js/tensor_int8.js).
//
// Two blocks:
//   1. Backend-independent: every restored name is a function, the JS wrappers
//      reject wrong argument types with a TypeError before touching a native,
//      and quantizeInt8PerRowHost — the one host-only helper in the group —
//      produces the documented weights / scales.
//   2. GPU-only: the W8A16 ops themselves. The whole family (FP16 activations,
//      INT8 weights, GGUF block quants) is unimplemented on the CPU backend,
//      so this block returns early there.

#include "api_test_helpers.h"

int run_api_int8_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- surface + argument checking + the host quantiser ------------------
    f += runJs("test_api_int8_surface", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;

        const names = [
            "dequantQ4kToFp16", "dequantQ6kToFp16", "dequantQ8_0ToFp16",
            "linearForwardQ4kFp16", "linearForwardQ6kFp16", "linearForwardQ8_0Fp16",
            "linearForwardBatchedQ4kFp16", "linearForwardBatchedQ6kFp16", "linearForwardBatchedQ8_0Fp16",
            "conv3dInt8wFp16Forward", "quantizeInt8PerRowHost", "matmulInt8wFp16",
            "conv2dInt8wFp16Forward", "linearForwardBatchedInt8wFp16", "resblockForwardInt8wFp16",
            "flashAttentionProjectKvInt8wFp16", "flashAttentionQWithKvCachedInt8wFp16",
            "flashAttentionQkvoInt8wFp16", "selfAttentionBiasInt8wFp16"
        ];
        if (names.length !== 19) throw new Error("expected 19 restored names, listed " + names.length);
        for (let i = 0; i < names.length; i++) {
            if (typeof tensor[names[i]] !== "function") throw new Error("bro.tensor." + names[i] + " is not a function");
        }

        // The wrappers validate before the native call, so these throw a
        // TypeError on every backend.
        const typeErr = (label, f) => {
            let threw = false;
            try { f(); } catch (e) { threw = e instanceof TypeError; }
            if (!threw) throw new Error(label + " did not throw a TypeError");
        };
        typeErr("dequantQ4kToFp16(null,null)", () => tensor.dequantQ4kToFp16(null, null));
        typeErr("matmulInt8wFp16(1,2,3,4)", () => tensor.matmulInt8wFp16(1, 2, 3, 4));
        typeErr("linearForwardBatchedInt8wFp16 with a number weight", () => tensor.linearForwardBatchedInt8wFp16(1, 2, null, 3, 4));
        typeErr("conv2dInt8wFp16Forward(null,...)", () => tensor.conv2dInt8wFp16Forward(null, null, null, null, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, null));
        typeErr("conv3dInt8wFp16Forward(null,...)", () => tensor.conv3dInt8wFp16Forward(null, null, null, null, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, null));
        typeErr("resblockForwardInt8wFp16(null)", () => tensor.resblockForwardInt8wFp16(null));
        typeErr("resblockForwardInt8wFp16({})", () => tensor.resblockForwardInt8wFp16({}));
        typeErr("flashAttentionQkvoInt8wFp16(null)", () => tensor.flashAttentionQkvoInt8wFp16(null));
        typeErr("flashAttentionQkvoInt8wFp16({})", () => tensor.flashAttentionQkvoInt8wFp16({}));
        typeErr("flashAttentionProjectKvInt8wFp16(1,...)", () => tensor.flashAttentionProjectKvInt8wFp16(1, 2, 3, null, 4, 5, null, 6, 7));
        typeErr("flashAttentionQWithKvCachedInt8wFp16(1,...)", () => tensor.flashAttentionQWithKvCachedInt8wFp16(1, 2, 3, 4, 5, null, 6, 7, null, null, 1, false, 8));
        typeErr("selfAttentionBiasInt8wFp16(1,...)", () => tensor.selfAttentionBiasInt8wFp16(1, 2, 3, 4, 5, 6, 7, 8, 9, null, null, 1, 1.0, 10));
        typeErr("quantizeInt8PerRowHost with an Array", () => tensor.quantizeInt8PerRowHost([1, 2, 3, 4], 2, 2));
        typeErr("quantizeInt8PerRowHost with a Float32Array", () => tensor.quantizeInt8PerRowHost(new Float32Array(4), 2, 2));
        let rangeThrew = false;
        try { tensor.quantizeInt8PerRowHost(Uint16Array.from([0, 0]), 0, 2); } catch (e) { rangeThrew = e instanceof RangeError; }
        if (!rangeThrew) throw new Error("quantizeInt8PerRowHost(out=0) did not throw a RangeError");
        rangeThrew = false;
        try { tensor.quantizeInt8PerRowHost(Uint16Array.from([0, 0]), 2, 2); } catch (e) { rangeThrew = e instanceof RangeError; }
        if (!rangeThrew) throw new Error("quantizeInt8PerRowHost with a short view did not throw a RangeError");

        // The host quantiser is not device-dispatched, so it answers on any
        // backend. W = [[1, -4], [3, 0]] as binary16 bit patterns.
        const W = Uint16Array.from([0x3C00, 0xC400, 0x4200, 0x0000]);
        const q = tensor.quantizeInt8PerRowHost(W, 2, 2);
        if (!(q.weights instanceof Int8Array)) throw new Error("weights is not an Int8Array");
        if (!(q.scales instanceof Float32Array)) throw new Error("scales is not a Float32Array");
        if (q.weights.length !== 4) throw new Error("weights.length: " + q.weights.length);
        if (q.scales.length !== 2) throw new Error("scales.length: " + q.scales.length);
        // scale = max(|row|)/127; q = clamp(round(w/scale), -127, 127)
        if (Math.abs(q.scales[0] - 4 / 127) > 1e-6) throw new Error("scales[0]: " + q.scales[0]);
        if (Math.abs(q.scales[1] - 3 / 127) > 1e-6) throw new Error("scales[1]: " + q.scales[1]);
        if (q.weights[0] !== 32 || q.weights[1] !== -127) throw new Error("row 0 quants: " + q.weights[0] + "," + q.weights[1]);
        if (q.weights[2] !== 127 || q.weights[3] !== 0) throw new Error("row 1 quants: " + q.weights[2] + "," + q.weights[3]);
        // An all-zero row gets a zero scale rather than a division by zero.
        const qz = tensor.quantizeInt8PerRowHost(Uint16Array.from([0, 0]), 1, 2);
        if (qz.scales[0] !== 0 || qz.weights[0] !== 0 || qz.weights[1] !== 0) throw new Error("all-zero row: " + qz.scales[0]);

        return "OK";
    })();
    )JS");

    // ---- the W8A16 ops themselves (GPU only) ------------------------------
    f += runJs("test_api_int8_w8a16", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        // Every op below leaves a null vtable slot on the CPU backend.
        if (tensor.backend === "cpu") return "OK";

        const finite = (t, label) => {
            const d = t.download();
            for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite at " + i);
            return d;
        };
        const shape = (t, r, c, label) => {
            if (t.rows !== r || t.cols !== c) throw new Error(label + " shape: " + t.rows + "x" + t.cols + " want " + r + "x" + c);
        };
        // An FP16 tensor from plain FP32 values.
        const f16 = (rows, cols, vals) => {
            const f = tensor.createTensor(rows, cols);
            f.upload(vals);
            const h = tensor.createTensor(1, 1);
            tensor.cast(f, h, "fp16");
            return h;
        };
        // A W8A16 weight pair: INT8 (rows, cols) + FP32 (rows, 1) scales,
        // built through the same host quantiser callers use at load time.
        const qpair = (rows, cols, vals) => {
            const h = f16(rows, cols, vals);
            const q = tensor.quantizeInt8PerRowHost(h.downloadFp16(), rows, cols);
            const W = tensor.createTensor(rows, cols, "int8");
            W.uploadInt8(q.weights);
            const S = tensor.createTensor(rows, 1);
            S.upload(q.scales);
            return { W: W, S: S, q: q };
        };
        const ramp = (n, k) => { const a = []; for (let i = 0; i < n; i++) a.push(Math.sin(i * k) * 0.5); return a; };

        // ---- matmulInt8wFp16 against the dequantised weight -----------------
        const wm = qpair(4, 4, [1, -2, 0.5, 4,
                                2, 2, -2, 2,
                                0.5, 0.25, -0.5, 1,
                                3, 0, 0, -1]);
        const xv = [1, 2, -1, 0.5];
        const Xm = f16(4, 1, xv);
        const Ym = tensor.createTensor(1, 1);
        tensor.matmulInt8wFp16(wm.W, wm.S, Xm, Ym);
        const ym = finite(Ym, "matmulInt8wFp16 Y");
        shape(Ym, 4, 1, "matmulInt8wFp16 Y");
        for (let r = 0; r < 4; r++) {
            let acc = 0;
            for (let c = 0; c < 4; c++) acc += wm.q.weights[r * 4 + c] * wm.q.scales[r] * xv[c];
            if (Math.abs(ym[r] - acc) > 5e-2) throw new Error("matmulInt8wFp16 row " + r + ": " + ym[r] + " want " + acc);
        }

        // ---- linearForwardBatchedInt8wFp16 ----------------------------------
        const Xb = f16(2, 4, [1, 2, -1, 0.5, 0.5, -1, 2, 1]);
        const Yb = tensor.createTensor(1, 1);
        tensor.linearForwardBatchedInt8wFp16(wm.W, wm.S, null, Xb, Yb);
        shape(Yb, 2, 4, "linearForwardBatchedInt8wFp16 Y_BD");
        const yb = finite(Yb, "linearForwardBatchedInt8wFp16 Y_BD");
        // Row 0 of the batched form is the same product as the GEMV above.
        for (let r = 0; r < 4; r++) {
            let acc = 0;
            for (let c = 0; c < 4; c++) acc += wm.q.weights[r * 4 + c] * wm.q.scales[r] * xv[c];
            if (Math.abs(yb[r] - acc) > 5e-2) throw new Error("linearForwardBatchedInt8wFp16 [0," + r + "]: " + yb[r] + " want " + acc);
        }

        // ---- conv2d / conv3d, 1x1 kernels over a 2x2 image -------------------
        const wc = qpair(2, 1, [1, -2]);
        const Xc = f16(1, 4, [1, 2, 3, 4]);
        const Yc = tensor.createTensor(1, 1);
        tensor.conv2dInt8wFp16Forward(Xc, wc.W, wc.S, null, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, Yc);
        shape(Yc, 1, 8, "conv2dInt8wFp16Forward Y");
        finite(Yc, "conv2dInt8wFp16Forward Y");
        const Y3 = tensor.createTensor(1, 1);
        tensor.conv3dInt8wFp16Forward(Xc, wc.W, wc.S, null, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, Y3);
        shape(Y3, 1, 8, "conv3dInt8wFp16Forward Y");
        finite(Y3, "conv3dInt8wFp16Forward Y");

        // ---- the flash-attention triplet + the T5-bias variant ---------------
        const D = 4, L = 2;
        const X = f16(L, D, [1, 0, 0, 1, 0, 1, 1, 0]);
        const wq = qpair(D, D, ramp(D * D, 0.7));
        const wk = qpair(D, D, ramp(D * D, 1.1));
        const wv = qpair(D, D, ramp(D * D, 1.7));
        const wo = qpair(D, D, ramp(D * D, 2.3));
        const Kc = tensor.createTensor(1, 1), Vc = tensor.createTensor(1, 1);
        tensor.flashAttentionProjectKvInt8wFp16(X, wk.W, wk.S, null, wv.W, wv.S, null, Kc, Vc);
        shape(Kc, L, D, "flashAttentionProjectKvInt8wFp16 K_out");
        shape(Vc, L, D, "flashAttentionProjectKvInt8wFp16 V_out");
        finite(Kc, "flashAttentionProjectKvInt8wFp16 K_out");
        finite(Vc, "flashAttentionProjectKvInt8wFp16 V_out");

        const O1 = tensor.createTensor(1, 1);
        tensor.flashAttentionQWithKvCachedInt8wFp16(X, Kc, Vc, wq.W, wq.S, null, wo.W, wo.S, null, null, 1, false, O1);
        shape(O1, L, D, "flashAttentionQWithKvCachedInt8wFp16 O");
        finite(O1, "flashAttentionQWithKvCachedInt8wFp16 O");

        const O2 = tensor.createTensor(1, 1);
        tensor.flashAttentionQkvoInt8wFp16({
            X: X,
            Wq_int8: wq.W, sq: wq.S, Wk_int8: wk.W, sk: wk.S,
            Wv_int8: wv.W, sv: wv.S, Wo_int8: wo.W, so: wo.S,
            numHeads: 1, O: O2
        });
        shape(O2, L, D, "flashAttentionQkvoInt8wFp16 O");
        const o2 = finite(O2, "flashAttentionQkvoInt8wFp16 O");
        // Same composition as the cached-KV form, so the two agree.
        const o1 = O1.download();
        for (let i = 0; i < o1.length; i++) {
            if (Math.abs(o1[i] - o2[i]) > 5e-2) throw new Error("flashAttentionQkvoInt8wFp16 != QWithKvCached at " + i + ": " + o1[i] + " vs " + o2[i]);
        }

        const O3 = tensor.createTensor(1, 1);
        tensor.selfAttentionBiasInt8wFp16(X, wq.W, wq.S, wk.W, wk.S, wv.W, wv.S, wo.W, wo.S, null, null, 1, 1.0, O3);
        shape(O3, L, D, "selfAttentionBiasInt8wFp16 O");
        finite(O3, "selfAttentionBiasInt8wFp16 O");
        // With an additive bias and a key mask, still finite and the same shape.
        const bias = tensor.createTensor(L, L);
        bias.upload([50, 0, 50, 0]);
        const mask = tensor.createTensor(L, 1);
        mask.upload([1, 1]);
        tensor.selfAttentionBiasInt8wFp16(X, wq.W, wq.S, wk.W, wk.S, wv.W, wv.S, wo.W, wo.S, mask, bias, 1, 1.0, O3);
        finite(O3, "selfAttentionBiasInt8wFp16 O (bias + mask)");

        // ---- the W8A16 resblock ---------------------------------------------
        const C = 4, H = 2, Wd = 2;
        const Xr = f16(1, C * H * Wd, ramp(C * H * Wd, 0.9));
        const gm = f16(C, 1, [1, 1, 1, 1]);
        const bt = f16(C, 1, [0, 0, 0, 0]);
        const w1 = qpair(C, C * 9, ramp(C * C * 9, 0.13));
        const w2 = qpair(C, C * 9, ramp(C * C * 9, 0.17));
        const Yr = tensor.createTensor(1, 1);
        tensor.resblockForwardInt8wFp16({
            X: Xr, gamma1: gm, beta1: bt, W1_int8: w1.W, s1: w1.S,
            gamma2: gm, beta2: bt, W2_int8: w2.W, s2: w2.S,
            N: 1, C_in: C, C_out: C, H: H, W: Wd, numGroups: 1, Y: Yr
        });
        shape(Yr, 1, C * H * Wd, "resblockForwardInt8wFp16 Y");
        finite(Yr, "resblockForwardInt8wFp16 Y");

        // ---- GGUF k-quant entry points --------------------------------------
        // A Q4_K / Q6_K / Q8_0 tensor can only come from a GGUF file — the
        // public createTensor dtypes don't spell the block carriers — so what
        // is checked here is that each entry point reaches its op and the op
        // rejects a non-block-quantised weight.
        const rejects = (label, f) => {
            let threw = false;
            try { f(); } catch (e) { threw = true; }
            if (!threw) throw new Error(label + " accepted a weight that is not block-quantised");
        };
        const yq = tensor.createTensor(1, 1);
        rejects("dequantQ4kToFp16", () => tensor.dequantQ4kToFp16(Xm, yq));
        rejects("dequantQ6kToFp16", () => tensor.dequantQ6kToFp16(Xm, yq));
        rejects("dequantQ8_0ToFp16", () => tensor.dequantQ8_0ToFp16(Xm, yq));
        rejects("linearForwardQ4kFp16", () => tensor.linearForwardQ4kFp16(Xm, null, Xm, yq));
        rejects("linearForwardQ6kFp16", () => tensor.linearForwardQ6kFp16(Xm, null, Xm, yq));
        rejects("linearForwardQ8_0Fp16", () => tensor.linearForwardQ8_0Fp16(Xm, null, Xm, yq));
        rejects("linearForwardBatchedQ4kFp16", () => tensor.linearForwardBatchedQ4kFp16(Xm, null, Xb, yq));
        rejects("linearForwardBatchedQ6kFp16", () => tensor.linearForwardBatchedQ6kFp16(Xm, null, Xb, yq));
        rejects("linearForwardBatchedQ8_0Fp16", () => tensor.linearForwardBatchedQ8_0Fp16(Xm, null, Xb, yq));

        return "OK";
    })();
    )JS");

    return f;
}
