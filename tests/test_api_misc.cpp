// test_api_misc.cpp — JS-level checks for the "misc" restored group:
// l2NormForward / l2NormBackward, bceWithLogitsFusedBatched,
// softmaxXentSegment, mseScalar, ddimStep, eulerStep, dpmpp2mStep and
// timestepEmbedding. All nine are cheap to pin down numerically, so each is
// checked against the closed form computed in JS rather than just "it ran".
//
// The three sampler steps are FP32 on the CPU backend and FP16/BF16 on the GPU
// one, so the sampler tensors are built through `cast` at the backend's dtype
// and the tolerance widens accordingly.

#include "api_test_helpers.h"

int run_api_misc_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- l2NormForward / l2NormBackward ------------------------------------
    f += runJs("test_api_misc.l2Norm", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const close = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-5 : tol);

        // Two heads of head_dim 2 in one row: [3,4 | 5,12] -> [0.6,0.8 | 5/13,12/13].
        const X = tensor.createTensor(1, 4);
        const Y = tensor.createTensor(1, 4);
        X.upload([3, 4, 5, 12]);
        tensor.l2NormForward(X, 2, 2, 0, Y);
        const y = Y.download();
        if (!close(y[0], 0.6) || !close(y[1], 0.8) || !close(y[2], 5 / 13) || !close(y[3], 12 / 13)) {
            throw new Error("l2NormForward mismatch: " + Array.from(y));
        }

        // Backward on one head [3,4] with dY = [1,0]:
        //   n = 1/5, dot = 3, dX = n*(dY - x*n^2*dot) = [0.128, -0.096]
        const Xb = tensor.createTensor(1, 2);
        const dY = tensor.createTensor(1, 2);
        const dX = tensor.createTensor(1, 2);
        Xb.upload([3, 4]);
        dY.upload([1, 0]);
        tensor.l2NormBackward(Xb, 2, 1, 0, dY, dX);
        const dx = dX.download();
        if (!close(dx[0], 0.128) || !close(dx[1], -0.096)) {
            throw new Error("l2NormBackward mismatch: " + Array.from(dx));
        }

        // Argument checking is the wrapper's job.
        let threw = false;
        try { tensor.l2NormForward(null, 2, 1, 0, Y); } catch (e) { threw = true; }
        if (!threw) throw new Error("l2NormForward accepted a null X");

        return "OK";
    })();
    )JS");

    // ---- bceWithLogitsFusedBatched -----------------------------------------
    f += runJs("test_api_misc.bceWithLogitsFusedBatched", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const close = (a, b) => Math.abs(a - b) <= 1e-5;

        // B=1, L=2, logits [0,0], target [1,0], posWeight 1, no mask:
        //   probs   = [0.5, 0.5]
        //   dLogits = probs - target = [-0.5, 0.5]
        //   loss    = softplus(0) + softplus(0) = 2*ln2
        const logits = tensor.createTensor(1, 2);
        const target = tensor.createTensor(1, 2);
        const probs = tensor.createTensor(1, 2);
        const dLogits = tensor.createTensor(1, 2);
        const lossPerSample = tensor.createTensor(1, 1);
        logits.upload([0, 0]);
        target.upload([1, 0]);
        tensor.bceWithLogitsFusedBatched(logits, target, null, 1.0, probs, dLogits, lossPerSample);
        const p = probs.download(), d = dLogits.download(), l = lossPerSample.download();
        if (!close(p[0], 0.5) || !close(p[1], 0.5)) throw new Error("bce probs: " + Array.from(p));
        if (!close(d[0], -0.5) || !close(d[1], 0.5)) throw new Error("bce dLogits: " + Array.from(d));
        if (!close(l[0], 2 * Math.log(2))) throw new Error("bce loss: " + l[0]);

        // A mask zeroes the second column out of probs, gradient and loss.
        const mask = tensor.createTensor(1, 2);
        mask.upload([1, 0]);
        tensor.bceWithLogitsFusedBatched(logits, target, mask, 1.0, probs, dLogits, lossPerSample);
        const pm = probs.download(), dm = dLogits.download(), lm = lossPerSample.download();
        if (!close(pm[1], 0) || !close(dm[1], 0)) throw new Error("bce masked column: " + Array.from(pm));
        if (!close(lm[0], Math.log(2))) throw new Error("bce masked loss: " + lm[0]);

        // posWeight scales the positive term only: loss = 2*ln2 + ln2 = 3*ln2.
        tensor.bceWithLogitsFusedBatched(logits, target, null, 2.0, probs, dLogits, lossPerSample);
        const lw = lossPerSample.download();
        if (!close(lw[0], 3 * Math.log(2))) throw new Error("bce posWeight loss: " + lw[0]);

        return "OK";
    })();
    )JS");

    // ---- softmaxXentSegment + mseScalar ------------------------------------
    f += runJs("test_api_misc.hostLosses", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const close = (a, b) => Math.abs(a - b) <= 1e-5;

        // Segment of the first 2 of 4 slots: logits [0,0], one-hot target.
        //   probs = [0.5,0.5], loss = ln2, dLogits = probs - target.
        const logits = new Float32Array([0, 0, 7, 7]);
        const target = new Float32Array([1, 0, 0, 0]);
        const probs = new Float32Array([-1, -1, -1, -1]);
        const dLogits = new Float32Array([-1, -1, -1, -1]);
        const loss = tensor.softmaxXentSegment(logits, target, probs, dLogits, 2, null);
        if (!close(loss, Math.log(2))) throw new Error("softmaxXentSegment loss: " + loss);
        if (!close(probs[0], 0.5) || !close(probs[1], 0.5)) throw new Error("segment probs: " + Array.from(probs));
        if (!close(dLogits[0], -0.5) || !close(dLogits[1], 0.5)) throw new Error("segment dLogits: " + Array.from(dLogits));
        // Only the first n slots are touched.
        if (probs[2] !== -1 || dLogits[3] !== -1) throw new Error("segment wrote past n: " + Array.from(probs));

        // A mask drops the second class: the surviving class takes all the mass.
        const mask = new Float32Array([1, 0, 0, 0]);
        const loss2 = tensor.softmaxXentSegment(logits, target, probs, dLogits, 2, mask);
        if (!close(loss2, 0)) throw new Error("masked segment loss: " + loss2);
        if (!close(probs[0], 1) || !close(probs[1], 0)) throw new Error("masked segment probs: " + Array.from(probs));

        // Non-Float32Array arguments are a TypeError from the wrapper.
        let threw = false;
        try { tensor.softmaxXentSegment([0, 0], target, probs, dLogits, 2, null); } catch (e) { threw = true; }
        if (!threw) throw new Error("softmaxXentSegment accepted a plain array");

        // mseScalar(pred, target) -> [0.5*(d)^2, d]
        const r = tensor.mseScalar(3, 1);
        if (!Array.isArray(r) || r.length !== 2) throw new Error("mseScalar shape: " + r);
        if (!close(r[0], 2) || !close(r[1], 2)) throw new Error("mseScalar: " + r[0] + "," + r[1]);
        const r2 = tensor.mseScalar(-0.5, 0.5);
        if (!close(r2[0], 0.5) || !close(r2[1], -1)) throw new Error("mseScalar neg: " + r2[0] + "," + r2[1]);

        return "OK";
    })();
    )JS");

    // ---- diffusion sampler steps -------------------------------------------
    f += runJs("test_api_misc.samplerSteps", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        // The sampler kernels are FP32 on CPU and FP16/BF16 on the GPU backend.
        const isCpu = tensor.backend === "cpu";
        const dt = isCpu ? "fp32" : "fp16";
        const tol = isCpu ? 1e-5 : 5e-3;
        const close = (a, b) => Math.abs(a - b) <= tol;

        // A sampler-dtype tensor holding `vals`, built through cast so the FP16
        // path needs no hand-rolled half bits.
        const mk = (vals) => {
            const src = tensor.createTensor(1, vals.length);
            src.upload(vals);
            if (isCpu) return src;
            const dst = tensor.createTensor(1, vals.length, dt);
            tensor.cast(src, dst, dt);
            return dst;
        };
        const blank = (n) => tensor.createTensor(1, n, dt);

        const xv = [1, 2, -1, 0.5];
        const ev = [0.5, -0.25, 1, 0];
        const n = xv.length;

        // ddimStep: x0 = (x - sqrt(1-a_t)*eps)/sqrt(a_t)
        //           x_prev = sqrt(a_prev)*x0 + sqrt(1-a_prev-s^2)*eps
        {
            const a_t = 0.64, a_prev = 0.36, s_t = 0.0;
            const xp = blank(n);
            tensor.ddimStep(mk(xv), mk(ev), a_t, a_prev, s_t, xp);
            const got = xp.download();
            for (let i = 0; i < n; i++) {
                const x0 = (xv[i] - Math.sqrt(1 - a_t) * ev[i]) / Math.sqrt(a_t);
                const want = Math.sqrt(a_prev) * x0 + Math.sqrt(1 - a_prev - s_t * s_t) * ev[i];
                if (!close(got[i], want)) throw new Error("ddimStep[" + i + "]: " + got[i] + " want " + want);
            }
        }

        // eulerStep: x_prev = x + (sigma_prev - sigma_t)*eps
        {
            const s_t = 1.0, s_prev = 0.6;
            const xp = blank(n);
            tensor.eulerStep(mk(xv), mk(ev), s_t, s_prev, xp);
            const got = xp.download();
            for (let i = 0; i < n; i++) {
                const want = xv[i] + (s_prev - s_t) * ev[i];
                if (!close(got[i], want)) throw new Error("eulerStep[" + i + "]: " + got[i] + " want " + want);
            }
        }

        // dpmpp2mStep: x0_t = x - sigma_t*eps
        //              x_prev = c_xt*x + c_x0t*x0_t + c_x0prev*x0_prev
        //              x0_out = x0_t
        {
            const x0v = [0.25, 0.5, -0.5, 1];
            const s_t = 0.5, c_xt = 0.75, c_x0t = 0.5, c_x0prev = -0.25;
            const xp = blank(n);
            const x0o = blank(n);
            tensor.dpmpp2mStep(mk(xv), mk(ev), mk(x0v), s_t, c_xt, c_x0t, c_x0prev, xp, x0o);
            const gotX = xp.download(), gotX0 = x0o.download();
            for (let i = 0; i < n; i++) {
                const x0t = xv[i] - s_t * ev[i];
                const want = c_xt * xv[i] + c_x0t * x0t + c_x0prev * x0v[i];
                if (!close(gotX[i], want)) throw new Error("dpmpp2mStep x_prev[" + i + "]: " + gotX[i] + " want " + want);
                if (!close(gotX0[i], x0t)) throw new Error("dpmpp2mStep x0_out[" + i + "]: " + gotX0[i] + " want " + x0t);
            }
        }

        return "OK";
    })();
    )JS");

    // ---- timestepEmbedding --------------------------------------------------
    f += runJs("test_api_misc.timestepEmbedding", R"JS(
    (function () {
        const tensor = globalThis.bro.tensor;
        const close = (a, b) => Math.abs(a - b) <= 1e-5;

        // dim 4 -> half 2, freqs = [1, 1/sqrt(maxPeriod)]; cos half first.
        const ts = tensor.createTensor(2, 1);
        const Y = tensor.createTensor(2, 4);
        ts.upload([0, 1]);
        tensor.timestepEmbedding(ts, 4, 10000, Y);
        const y = Y.download();
        const f1 = Math.exp(-Math.log(10000) * 0.5); // = 0.01
        const want = [
            1, 1, 0, 0,
            Math.cos(1), Math.cos(f1), Math.sin(1), Math.sin(f1),
        ];
        for (let i = 0; i < want.length; i++) {
            if (!close(y[i], want[i])) throw new Error("timestepEmbedding[" + i + "]: " + y[i] + " want " + want[i]);
        }

        // maxPeriod is optional and defaults to 10000 — same answer.
        const Y2 = tensor.createTensor(2, 4);
        tensor.timestepEmbedding(ts, 4, undefined, Y2);
        const y2 = Y2.download();
        for (let i = 0; i < want.length; i++) {
            if (!close(y2[i], want[i])) throw new Error("timestepEmbedding default maxPeriod[" + i + "]: " + y2[i]);
        }

        // A different max_period moves the second frequency only.
        const Y3 = tensor.createTensor(2, 4);
        tensor.timestepEmbedding(ts, 4, 100, Y3);
        const y3 = Y3.download();
        if (!close(y3[4], Math.cos(1)) || !close(y3[5], Math.cos(0.1))) {
            throw new Error("timestepEmbedding maxPeriod=100: " + Array.from(y3));
        }

        // A bad dim reaches the op's own error, surfaced as a thrown Error.
        let threw = false;
        try { tensor.timestepEmbedding(ts, 0, 10000, Y); } catch (e) { threw = true; }
        if (!threw) throw new Error("timestepEmbedding accepted dim 0");

        return "OK";
    })();
    )JS");

    return f;
}
