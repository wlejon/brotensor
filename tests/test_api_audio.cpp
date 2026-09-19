// test_api_audio.cpp — JS-level checks for the restored audio group of
// bro.tensor (src/api/js/tensor_audio.js over src/api/native_tensor_audio*.cpp):
// the spectral / FFT core, STFT / iSTFT, the 1D convolution family, the
// vocoder / codec activations, codec quantization, 1D resampling, the log /
// exp / round elementwise maps and the autoregressive logit sampler.
//
// The realm is installed once by test_api.cpp; each block below is one
// evalScript that answers "OK" or throws.
//
// Two conventions worth knowing when reading the blocks:
//   * an INT32 tensor cannot be uploaded from JS (upload() always lands FP32
//     and cast has no FP32->INT32 pair), so `int32Col` mints one by creating
//     the column as INT32 and filling it through argmaxRows — whose output
//     dtype is opt-in: an already-INT32 Idx receives int32 indices, anything
//     else is written FP32 (src/cpu/public_reductions.cpp, the same contract
//     on CUDA);
//   * an INT32 tensor cannot be downloaded either, so `readIdx` reads indices
//     back through embeddingLookupForward against a table whose row k holds k.

#include "api_test_helpers.h"

#include <string>

namespace {

// Opens the IIFE and defines the helpers every block uses.
const char* kHead = R"JS(
(function () {
    const t = globalThis.bro.tensor;
    const up = (r, c, vals) => { const x = t.createTensor(r, c); x.upload(vals); return x; };
    const zeros = (r, c, dt) => t.createTensor(r, c, dt);
    const near = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-4 : tol);
    const eq = (got, want, what, tol) => {
        if (got.length < want.length) throw new Error(what + ": got " + got.length + " elements, want " + want.length);
        for (let i = 0; i < want.length; i++) {
            if (!near(got[i], want[i], tol)) throw new Error(what + "[" + i + "] = " + got[i] + ", want " + want[i]);
        }
    };
    const shape = (x, r, c, what) => {
        if (x.rows !== r || x.cols !== c) throw new Error(what + ": shape " + x.rows + "x" + x.cols + ", want " + r + "x" + c);
    };
    const finite = (a, what) => {
        if (a.length === 0) throw new Error(what + ": empty");
        for (let i = 0; i < a.length; i++) {
            if (!Number.isFinite(a[i])) throw new Error(what + "[" + i + "] is not finite: " + a[i]);
        }
    };
    // A (vals.length, 1) INT32 tensor holding vals, via a one-hot argmax.
    const int32Col = (vals, width) => {
        let w = width === undefined ? 0 : width;
        if (w === 0) { for (let i = 0; i < vals.length; i++) if (vals[i] + 1 > w) w = vals[i] + 1; }
        const buf = new Float32Array(vals.length * w);
        for (let r = 0; r < vals.length; r++) buf[r * w + vals[r]] = 1;
        const src = up(vals.length, w, buf);
        const idx = t.createTensor(vals.length, 1, "int32");
        t.argmaxRows(src, idx);
        if (idx.dtype() !== "int32") throw new Error("int32Col: argmaxRows gave " + idx.dtype());
        return idx;
    };
    // Read an (n,1) INT32 tensor back as floats (indices must be < K).
    const readIdx = (idx, n, K) => {
        const tb = new Float32Array(K);
        for (let k = 0; k < K; k++) tb[k] = k;
        const table = up(K, 1, tb);
        const out = t.createTensor(n, 1);
        t.embeddingLookupForward(table, idx, n, out);
        return out.download();
    };
)JS";

const char* kTail = R"JS(
    return "OK";
})();
)JS";

std::string js(const char* body) {
    return std::string(kHead) + body + kTail;
}

} // namespace

int run_api_audio_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- spectral / FFT core + the complex helpers --------------------------
    f += runJs("test_api_audio_spectral", js(R"JS(
    // fft of a unit impulse: every bin is 1+0i. Complex tensors are (R, 2*N).
    const x = up(1, 8, [1, 0, 0, 0, 0, 0, 0, 0]);
    const y = zeros(1, 8);
    t.fft(x, y);
    shape(y, 1, 8, "fft y");
    eq(y.download(), [1, 0, 1, 0, 1, 0, 1, 0], "fft");

    // ifft takes it straight back (1/N normalisation).
    const z = zeros(1, 8);
    t.ifft(y, z);
    eq(z.download(), [1, 0, 0, 0, 0, 0, 0, 0], "ifft");

    // rfft of a constant real signal: all the energy sits in the DC bin.
    const r = up(1, 4, [1, 1, 1, 1]);
    const spec = zeros(1, 6);
    t.rfft(r, spec);
    shape(spec, 1, 6, "rfft spec");
    eq(spec.download(), [4, 0, 0, 0, 0, 0], "rfft");

    // irfft needs the signal length: a 3-bin half-spectrum is ambiguous.
    const back = zeros(1, 4);
    t.irfft(spec, 4, back);
    eq(back.download(), [1, 1, 1, 1], "irfft");

    // rfft adjoint: a DC-only spectrum gradient spreads evenly over the signal.
    const dSpec = up(1, 6, [1, 0, 0, 0, 0, 0]);
    const dX = zeros(1, 4);
    t.rfftBackward(dSpec, 4, dX);
    eq(dX.download(), [1, 1, 1, 1], "rfftBackward");

    // irfft adjoint: a flat signal gradient lands (1/L * L) on the DC bin.
    const dSig = up(1, 4, [1, 1, 1, 1]);
    const dBins = zeros(1, 6);
    t.irfftBackward(dSig, dBins);
    shape(dBins, 1, 6, "irfftBackward dX");
    finite(dBins.download(), "irfftBackward");
    eq(dBins.download(), [1], "irfftBackward DC");

    // (1+2i)*(3+4i) = -5+10i.
    const a = up(1, 2, [1, 2]);
    const b = up(1, 2, [3, 4]);
    const m = zeros(1, 2);
    t.complexMul(a, b, m);
    eq(m.download(), [-5, 10], "complexMul");

    // dA = dY*conj(b), dB = dY*conj(a); both accumulate into zeroed buffers.
    const dY = up(1, 2, [1, 0]);
    const dA = zeros(1, 2);
    const dB = zeros(1, 2);
    t.complexMulBackward(a, b, dY, dA, dB);
    eq(dA.download(), [3, -4], "complexMulBackward dA");
    eq(dB.download(), [1, -2], "complexMulBackward dB");

    // |3+4i| = 5; d|z|/dz = z/|z|.
    const zc = up(1, 2, [3, 4]);
    const mag = zeros(1, 1);
    t.complexAbs(zc, mag);
    shape(mag, 1, 1, "complexAbs y");
    eq(mag.download(), [5], "complexAbs");
    const dMag = up(1, 1, [1]);
    const dZ = zeros(1, 2);
    t.complexAbsBackward(zc, dMag, dZ);
    eq(dZ.download(), [0.6, 0.8], "complexAbsBackward");

    // arg(0+1i) = pi/2.
    const zi = up(1, 2, [0, 1]);
    const ang = zeros(1, 1);
    t.complexAngle(zi, ang);
    eq(ang.download(), [Math.PI / 2], "complexAngle");

    // 2*exp(0i) = 2+0i.
    const pm = up(1, 1, [2]);
    const pp = up(1, 1, [0]);
    const pol = zeros(1, 2);
    t.complexFromPolar(pm, pp, pol);
    eq(pol.download(), [2, 0], "complexFromPolar");
)JS"));

    // ---- STFT / iSTFT --------------------------------------------------------
    f += runJs("test_api_audio_stft", js(R"JS(
    // One signal, n_fft 4, hop 2, rectangular window: 3 frames of 3 bins, so
    // the spectrogram is (N*frames, 2*bins) = (3, 6). Every sample is covered
    // by at least one frame, so the rectangular window is COLA here and
    // istft(stft(x)) is exact.
    const sig = up(1, 8, [1, 2, 3, 4, 5, 6, 7, 8]);
    const win = up(1, 4, [1, 1, 1, 1]);
    const spec = zeros(3, 6);
    t.stft(sig, win, 1, 4, 2, 4, false, false, spec);
    shape(spec, 3, 6, "stft spec");
    const sd = spec.download();
    finite(sd, "stft");
    // DC bin per frame = the frame's sum: [1..4], [3..6], [5..8].
    if (!near(sd[0], 10) || !near(sd[6], 18) || !near(sd[12], 26)) {
        throw new Error("stft DC bins: " + sd[0] + ", " + sd[6] + ", " + sd[12]);
    }

    const rec = zeros(1, 8);
    t.istft(spec, win, 1, 8, 4, 2, 4, false, false, rec);
    shape(rec, 1, 8, "istft signal");
    eq(rec.download(), [1, 2, 3, 4, 5, 6, 7, 8], "istft round trip", 1e-3);

    // Both adjoints are explicit ops (window + COLA are folded in).
    const ones18 = new Float32Array(18);
    for (let i = 0; i < 18; i++) ones18[i] = 1;
    const dSpec = up(3, 6, ones18);
    const dSignal = zeros(1, 8);
    t.stftBackward(dSpec, win, 1, 8, 4, 2, 4, false, false, dSignal);
    shape(dSignal, 1, 8, "stftBackward dSignal");
    finite(dSignal.download(), "stftBackward");

    const dSig2 = up(1, 8, [1, 1, 1, 1, 1, 1, 1, 1]);
    const dSpec2 = zeros(3, 6);
    t.istftBackward(dSig2, win, 1, 8, 4, 2, 4, false, false, dSpec2);
    shape(dSpec2, 3, 6, "istftBackward dSpec");
    finite(dSpec2.download(), "istftBackward");

    // normalized = true just rescales by 1/sqrt(n_fft) — still finite, and the
    // matching istft still round-trips.
    const specN = zeros(3, 6);
    t.stft(sig, win, 1, 4, 2, 4, false, true, specN);
    finite(specN.download(), "stft normalized");
    const recN = zeros(1, 8);
    t.istft(specN, win, 1, 8, 4, 2, 4, false, true, recN);
    eq(recN.download(), [1, 2, 3, 4, 5, 6, 7, 8], "istft normalized round trip", 1e-3);
)JS"));

    // ---- pad1d + the conv1d / transpose / causal family ----------------------
    f += runJs("test_api_audio_conv1d", js(R"JS(
    // pad1d: NCL, one channel, L = 3. mode 0 zero, 1 reflect, 2 replicate.
    const px = up(1, 3, [1, 2, 3]);
    const pz = zeros(1, 5);
    t.pad1dForward(px, 1, 1, 3, 1, 1, 0, pz);
    shape(pz, 1, 5, "pad1d zero");
    eq(pz.download(), [0, 1, 2, 3, 0], "pad1d zero");
    const pr = zeros(1, 5);
    t.pad1dForward(px, 1, 1, 3, 1, 1, 1, pr);
    eq(pr.download(), [2, 1, 2, 3, 2], "pad1d reflect");
    const prep = zeros(1, 5);
    t.pad1dForward(px, 1, 1, 3, 1, 1, 2, prep);
    eq(prep.download(), [1, 1, 2, 3, 3], "pad1d replicate");
    const pdY = up(1, 5, [1, 1, 1, 1, 1]);
    const pdX = zeros(1, 3);
    t.pad1dBackward(pdY, 1, 1, 3, 1, 1, 0, pdX);
    eq(pdX.download(), [1, 1, 1], "pad1dBackward zero");

    // conv1d: X (N, C_in*L), Wt (C_out, (C_in/groups)*kL) OIL.
    const X = up(1, 4, [1, 2, 3, 4]);
    const Wt = up(1, 2, [1, 1]);
    const Y = zeros(1, 3);
    t.conv1d(X, Wt, null, 1, 1, 4, 1, 2, 1, 0, 1, 1, Y);
    shape(Y, 1, 3, "conv1d Y");
    eq(Y.download(), [3, 5, 7], "conv1d");
    const bias = up(1, 1, [10]);
    const Yb = zeros(1, 3);
    t.conv1d(X, Wt, bias, 1, 1, 4, 1, 2, 1, 0, 1, 1, Yb);
    eq(Yb.download(), [13, 15, 17], "conv1d with bias");

    const dY = up(1, 3, [1, 1, 1]);
    const dX = zeros(1, 4);
    t.conv1dBackwardInput(Wt, dY, 1, 1, 4, 1, 2, 1, 0, 1, 1, dX);
    eq(dX.download(), [1, 2, 2, 1], "conv1dBackwardInput");
    const dWt = zeros(1, 2);            // accumulated — created zeroed
    t.conv1dBackwardWeight(X, dY, 1, 1, 4, 1, 2, 1, 0, 1, 1, dWt);
    eq(dWt.download(), [6, 9], "conv1dBackwardWeight");
    const dB = zeros(1, 1);
    t.conv1dBackwardBias(dY, 1, 1, 3, dB);
    eq(dB.download(), [3], "conv1dBackwardBias");

    // conv_transpose1d: weights are input-channel-major, L_out = 3 here.
    const tX = up(1, 2, [1, 2]);
    const tW = up(1, 2, [1, 1]);
    const tY = zeros(1, 3);
    t.convTranspose1dForward(tX, tW, null, 1, 1, 2, 1, 2, 1, 0, 0, 1, 1, tY);
    shape(tY, 1, 3, "convTranspose1d Y");
    eq(tY.download(), [1, 3, 2], "convTranspose1dForward");
    const tdY = up(1, 3, [1, 1, 1]);
    const tdX = zeros(1, 2);
    t.convTranspose1dBackwardInput(tW, tdY, 1, 1, 2, 1, 2, 1, 0, 0, 1, 1, tdX);
    eq(tdX.download(), [2, 2], "convTranspose1dBackwardInput");
    const tdW = zeros(1, 2);
    t.convTranspose1dBackwardWeight(tX, tdY, 1, 1, 2, 1, 2, 1, 0, 0, 1, 1, tdW);
    eq(tdW.download(), [3, 3], "convTranspose1dBackwardWeight");
    const tdB = zeros(1, 1);
    t.convTranspose1dBackwardBias(up(1, 3, [1, 2, 3]), 1, 1, 3, tdB);
    eq(tdB.download(), [6], "convTranspose1dBackwardBias");

    // causal conv: left-pad by dilation*(kL-1), then a valid conv1d.
    const scratch = zeros(1, 1);        // caller-owned, resized by the op
    const cY = zeros(1, 4);
    t.causalConv1d(X, Wt, null, 1, 1, 4, 1, 2, 1, 1, 1, scratch, cY);
    shape(cY, 1, 4, "causalConv1d Y");
    eq(cY.download(), [1, 3, 5, 7], "causalConv1d");

    // The streaming twin reproduces that output two samples at a time.
    const state = zeros(1, 1);          // (N, C*(kL-1)*dilation), zeroed
    const step1 = up(1, 2, [1, 2]);
    const sY1 = zeros(1, 2);
    t.causalConv1dUpdate(step1, Wt, null, 1, 1, 2, 2, 1, state, sY1);
    eq(sY1.download(), [1, 3], "causalConv1dUpdate step 1");
    const step2 = up(1, 2, [3, 4]);
    const sY2 = zeros(1, 2);
    t.causalConv1dUpdate(step2, Wt, null, 1, 1, 2, 2, 1, state, sY2);
    eq(sY2.download(), [5, 7], "causalConv1dUpdate step 2");
)JS"));

    // ---- W8A16 conv1d (GPU-only: no CPU conv2d_int8w slot) -------------------
    f += runJs("test_api_audio_conv1d_int8", js(R"JS(
    if (t.backend === "cpu") return "OK";
    const xf = up(1, 4, [1, 2, 3, 4]);
    const X = zeros(1, 4, "fp16");
    t.cast(xf, X, "fp16");
    const W = zeros(1, 2, "int8");
    W.uploadInt8(new Int8Array([1, 1]));
    const scales = up(1, 1, [1]);
    const Y = zeros(1, 3, "fp16");
    t.conv1dInt8wFp16(X, W, scales, null, 1, 1, 4, 1, 2, 1, 0, 1, 1, Y);
    shape(Y, 1, 3, "conv1dInt8wFp16 Y");
    const y = Y.download();
    finite(y, "conv1dInt8wFp16");
    eq(y, [3, 5, 7], "conv1dInt8wFp16", 0.05);
)JS"));

    // ---- vocoder / codec activations ----------------------------------------
    f += runJs("test_api_audio_activations", js(R"JS(
    // snake, plain (beta null): y = x + (1/alpha)*sin^2(alpha*x). NCL, C = 1.
    const X = up(1, 2, [0, Math.PI / 2]);
    const alpha = up(1, 1, [1]);
    const Y = zeros(1, 2);
    t.snakeForward(X, alpha, null, 1, 1, 2, Y);
    shape(Y, 1, 2, "snakeForward Y");
    eq(Y.download(), [0, Math.PI / 2 + 1], "snakeForward");

    // dy/dx = 1 + 2*a*r*sin*cos — 1 at both x = 0 and x = pi/2 (cos = 0 there).
    // dy/dalpha for plain snake adds -r^2*sin^2, i.e. -1 at pi/2.
    const dY = up(1, 2, [1, 1]);
    const dX = zeros(1, 2);
    const dAlpha = zeros(1, 1);         // accumulated — created zeroed
    t.snakeBackward(X, alpha, null, dY, 1, 1, 2, dX, dAlpha, null);
    eq(dX.download(), [1, 1], "snakeBackward dX");
    eq(dAlpha.download(), [-1], "snakeBackward dAlpha");

    // snakebeta: beta != null means dBeta must be non-null too.
    const beta = up(1, 1, [2]);
    const Yb = zeros(1, 2);
    t.snakeForward(X, alpha, beta, 1, 1, 2, Yb);
    eq(Yb.download(), [0, Math.PI / 2 + 0.5], "snakeForward beta");
    const dXb = zeros(1, 2);
    const dAlphaB = zeros(1, 1);
    const dBetaB = zeros(1, 1);
    t.snakeBackward(X, alpha, beta, dY, 1, 1, 2, dXb, dAlphaB, dBetaB);
    finite(dXb.download(), "snakeBackward beta dX");
    eq(dBetaB.download(), [-0.25], "snakeBackward dBeta");

    // elu(x) = x for x > 0, alpha*(exp(x)-1) otherwise.
    const ex = up(1, 3, [-1, 0, 2]);
    const ey = zeros(1, 3);
    t.eluForward(ex, 1.0, ey);
    eq(ey.download(), [Math.exp(-1) - 1, 0, 2], "eluForward");
    const edY = up(1, 3, [1, 1, 1]);
    const edX = zeros(1, 3);
    t.eluBackward(ex, edY, 1.0, edX);
    eq(edX.download(), [Math.exp(-1), 1, 1], "eluBackward");
    // alpha = 2 scales the negative branch.
    const ey2 = zeros(1, 3);
    t.eluForward(ex, 2.0, ey2);
    eq(ey2.download(), [2 * (Math.exp(-1) - 1), 0, 2], "eluForward alpha=2");

    // leaky relu.
    const lx = up(1, 2, [-2, 3]);
    const ly = zeros(1, 2);
    t.leakyReluForward(lx, 0.1, ly);
    eq(ly.download(), [-0.2, 3], "leakyReluForward");
    const ldY = up(1, 2, [1, 1]);
    const ldX = zeros(1, 2);
    t.leakyReluBackward(lx, ldY, 0.1, ldX);
    eq(ldX.download(), [0.1, 1], "leakyReluBackward");
)JS"));

    // ---- codec quantization + 1D resampling ----------------------------------
    f += runJs("test_api_audio_quant", js(R"JS(
    // VQ: each row snaps to its nearest codeword.
    const x = up(2, 2, [0.9, 0.1, 0.1, 0.9]);
    const codebook = up(2, 2, [1, 0, 0, 1]);
    const indices = t.createTensor(2, 1);
    const quantized = t.createTensor(2, 2);
    t.vqEncodeForward(x, codebook, indices, quantized);
    if (indices.dtype() !== "int32") throw new Error("vqEncodeForward: indices dtype " + indices.dtype());
    eq(readIdx(indices, 2, 2), [0, 1], "vqEncodeForward indices");
    eq(quantized.download(), [1, 0, 0, 1], "vqEncodeForward quantized");
    // Straight-through: dX = dQuantized.
    const dQ = up(2, 2, [1, 2, 3, 4]);
    const dX = zeros(2, 2);
    t.vqEncodeBackward(dQ, dX);
    eq(dX.download(), [1, 2, 3, 4], "vqEncodeBackward");

    // FSQ: 3 levels per dim, so [-1, 0, 1]; packed = i0 + L0*i1.
    const fx = up(1, 2, [0.9, -0.9]);
    const levels = int32Col([3, 3], 4);
    const fq = t.createTensor(1, 2);
    const packed = t.createTensor(1, 1);
    t.fsqQuantizeForward(fx, levels, fq, packed);
    eq(fq.download(), [1, -1], "fsqQuantizeForward quantized");
    if (packed.dtype() !== "int32") throw new Error("fsqQuantizeForward: packed dtype " + packed.dtype());
    eq(readIdx(packed, 1, 9), [2], "fsqQuantizeForward packed");
    const fdQ = up(1, 2, [5, 6]);
    const fdX = zeros(1, 2);
    t.fsqQuantizeBackward(fdQ, fdX);
    eq(fdX.download(), [5, 6], "fsqQuantizeBackward");

    // resample1d, align_corners=False: src = (dst+0.5)*(L_in/L_out) - 0.5.
    const rx = up(1, 2, [0, 1]);
    const rlin = zeros(1, 4);
    t.resample1dForward(rx, 1, 1, 2, 4, 1, rlin);
    shape(rlin, 1, 4, "resample1d linear Y");
    eq(rlin.download(), [0, 0.25, 0.75, 1], "resample1dForward linear");
    const rnear = zeros(1, 4);
    t.resample1dForward(rx, 1, 1, 2, 4, 0, rnear);
    eq(rnear.download(), [0, 0, 1, 1], "resample1dForward nearest");
    // Nearest adjoint: each input position collects its two output taps.
    const rdY = up(1, 4, [1, 1, 1, 1]);
    const rdX = zeros(1, 2);
    t.resample1dBackward(rdY, 1, 1, 2, 4, 0, rdX);
    eq(rdX.download(), [2, 2], "resample1dBackward nearest");
    const rdXl = zeros(1, 2);
    t.resample1dBackward(rdY, 1, 1, 2, 4, 1, rdXl);
    finite(rdXl.download(), "resample1dBackward linear");
)JS"));

    // ---- log / exp / round elementwise ---------------------------------------
    f += runJs("test_api_audio_elementwise", js(R"JS(
    const x = up(1, 2, [1, Math.E]);
    const y = zeros(1, 2);
    t.logForward(x, y);
    eq(y.download(), [0, 1], "logForward");
    const lx = up(1, 2, [1, 2]);
    const ones = up(1, 2, [1, 1]);
    const ldX = zeros(1, 2);
    t.logBackward(lx, ones, ldX);          // dX = dY / x
    eq(ldX.download(), [1, 0.5], "logBackward");

    const ex = up(1, 2, [0, 1]);
    const ey = zeros(1, 2);
    t.expForward(ex, ey);
    eq(ey.download(), [1, Math.E], "expForward");
    const edX = zeros(1, 2);
    t.expBackward(ex, ones, edX);          // dX = dY * exp(x)
    eq(edX.download(), [1, Math.E], "expBackward");

    // Round-half-to-even, like torch.round.
    const rx = up(1, 4, [0.5, 1.5, 2.5, -2.5]);
    const ry = zeros(1, 4);
    t.roundForward(rx, ry);
    eq(ry.download(), [0, 2, 2, -2], "roundForward");
    // Straight-through: dX = dY, and it needs no x.
    const rdY = up(1, 4, [1, 2, 3, 4]);
    const rdX = zeros(1, 4);
    t.roundBackward(rdY, rdX);
    eq(rdX.download(), [1, 2, 3, 4], "roundBackward");
)JS"));

    // ---- autoregressive logit sampling ---------------------------------------
    f += runJs("test_api_audio_sampling", js(R"JS(
    // One row, 4-token vocabulary, one logit far above the rest: whatever the
    // draw, the inverse-CDF lookup lands on token 2.
    const logits = up(1, 4, [0, 0, 100, 0]);
    const indices = t.createTensor(1, 1);
    t.sampleLogits(logits, 1.0, 0, 1.0, 1234, 0, indices);
    if (indices.dtype() !== "int32") throw new Error("sampleLogits: indices dtype " + indices.dtype());
    eq(readIdx(indices, 1, 4), [2], "sampleLogits");
    // temperature 0 is the deterministic argmax (no RNG consumed).
    const greedy = t.createTensor(1, 1);
    t.sampleLogits(logits, 0.0, 0, 1.0, 0, 0, greedy);
    eq(readIdx(greedy, 1, 4), [2], "sampleLogits greedy");

    // The graph-capturable twin: counter (INT32, >= 1 element), scratch
    // (FP32, >= 3*N*V) and indices ((N,1) INT32) are all caller-owned.
    const counter = int32Col([0], 1);
    const scratch = t.createTensor(1, 16);
    const into = int32Col([0], 1);
    t.sampleLogitsInto(logits, 1.0, 0, 1.0, 1234, counter, scratch, into);
    eq(readIdx(into, 1, 4), [2], "sampleLogitsInto");
    // Greedy leaves the counter alone and still picks the argmax.
    t.sampleLogitsInto(logits, 0.0, 0, 1.0, 1234, counter, scratch, into);
    eq(readIdx(into, 1, 4), [2], "sampleLogitsInto greedy");
)JS"));

    return f;
}
