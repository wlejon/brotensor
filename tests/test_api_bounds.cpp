// bro.tensor input-size contract: a native handed an operand smaller than its
// dims say (or a mismatched elementwise pair) throws a JS Error instead of
// reading or writing past the operand's storage.

#include "api_test_helpers.h"

#include <string>

using brotensor_api_test::runJs;

namespace {

// Every block gets `T` (bro.tensor), `mk(rows, cols, values?, dtype?)` and
// `throws(label, f, needle)`, which answers null when f throws an Error whose
// message contains `needle`, or a description of what happened instead.
const char* kPrelude = R"JS(
    const T = globalThis.bro.tensor;
    const mk = (r, c, vals, dt) => {
        const t = T.createTensor(r, c, dt);
        if (vals) t.upload(Float32Array.from(vals));
        return t;
    };
    const throws = (label, f, needle) => {
        try { f(); } catch (e) {
            if (needle && !String(e.message).includes(needle)) return label + ": wrong error: " + e.message;
            return null;
        }
        return label + ": did not throw";
    };
    const firstFail = (...rs) => rs.find((r) => r !== null) || "OK";
)JS";

std::string block(const char* body) {
    return std::string("(() => {") + kPrelude + body + "\n})()";
}

} // namespace

int run_api_bounds_tests() {
    int failures = 0;

    failures += runJs("bounds: elementwise pairs", block(R"JS(
        const x = mk(4, 1, [1, -2, 3, -4]);
        const small = mk(2, 1, [1, 1]);
        const dX = mk(4, 1);
        return firstFail(
            throws("reluBackward", () => T.reluBackward(x, small, dX), "dY"),
            throws("siluBackward", () => T.siluBackward(x, small, dX), "dY"),
            throws("tanhBackward", () => T.tanhBackward(x, small, dX), "dY"),
            throws("addInplace", () => T.addInplace(x, small), "x"),
            throws("mulInplace", () => T.mulInplace(x, small), "x"),
            throws("mseVecForward", () => T.mseVecForward(x, small), "target"),
            throws("softmaxBackward", () => T.softmaxBackward(x, small, dX), "dY"));
    )JS"));

    failures += runJs("bounds: dtype mismatch", block(R"JS(
        const a = mk(4, 1, [1, 2, 3, 4]);
        const h = mk(4, 1, null, "fp16");
        return firstFail(throws("addInplace fp32+fp16", () => T.addInplace(a, h), "fp16"));
    )JS"));

    failures += runJs("bounds: linear / matmul shapes", block(R"JS(
        const W = mk(3, 4);            // (out=3, in=4)
        const b = mk(3, 1);
        const xShort = mk(2, 1);
        const y = mk(3, 1);
        const A = mk(2, 3), B = mk(4, 2), C = mk(1, 1);
        return firstFail(
            throws("linearForward short x", () => T.linearForward(W, b, xShort, y), "x"),
            throws("linearForward short b", () => T.linearForward(W, mk(1, 1), mk(4, 1), y), "b"),
            throws("matmul A.cols != B.rows", () => T.matmul(A, B, C), "B.rows"),
            throws("linearBackward short dW", () => T.linearBackward(W, mk(4, 1), mk(3, 1), mk(4, 1), mk(2, 2), mk(3, 1)), "dW"));
    )JS"));

    failures += runJs("bounds: conv2d dims", block(R"JS(
        // X claims N=1, C_in=2, 4x4 = 32 elements; hand it 8.
        const X = mk(8, 1);
        const Wt = mk(3, 2 * 3 * 3);
        const Y = mk(1, 1);
        const Xok = mk(1, 32);
        return firstFail(
            throws("conv2dForward short X", () => T.conv2dForward(X, Wt, null, 1, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, Y), "X"),
            throws("conv2dForward short Wt", () => T.conv2dForward(Xok, mk(3, 2), null, 1, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, Y), "Wt"),
            // (The wrapper maps a stride of 0 to the default 1; a negative one reaches the native.)
            throws("conv2dForward stride -1", () => T.conv2dForward(Xok, Wt, null, 1, 2, 4, 4, 3, 3, 3, -1, 1, 1, 1, 1, 1, 1, Y), "positive"),
            throws("conv2dForward groups 3", () => T.conv2dForward(Xok, Wt, null, 1, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, Y), "groups"));
    )JS"));

    failures += runJs("bounds: NCHW resamples", block(R"JS(
        const X = mk(4, 1);
        const Y = mk(1, 1);
        return firstFail(
            throws("upsampleNearest2xForward", () => T.upsampleNearest2xForward(X, 1, 2, 4, 4, Y), "X"),
            throws("upsampleBilinear2xBackward", () => T.upsampleBilinear2xBackward(X, 1, 1, 2, 2, Y), "dY"),
            throws("downsampleAvg2xForward odd", () => T.downsampleAvg2xForward(mk(9, 1), 1, 1, 3, 3, Y), "even"),
            throws("interp2dForward", () => T.interp2dForward(X, 1, 1, 4, 4, 8, 8, 1, Y), "X"));
    )JS"));

    failures += runJs("bounds: copyD2D ranges", block(R"JS(
        const src = mk(4, 1, [1, 2, 3, 4]);
        const dst = mk(4, 1);
        T.copyD2D(src, 1, dst, 0, 3);
        const got = Array.from(dst.download());
        if (got.join(",") !== "2,3,4,0") return "in-range copy gave " + got;
        return firstFail(
            throws("copyD2D past src", () => T.copyD2D(src, 2, dst, 0, 3), "out of range"),
            throws("copyD2D past dst", () => T.copyD2D(src, 0, dst, 2, 3), "out of range"),
            throws("copyD2D negative", () => T.copyD2D(src, -1, dst, 0, 1), "non-negative"),
            throws("copyD2D dtype", () => T.copyD2D(src, 0, mk(4, 1, null, "fp16"), 0, 1), "fp16"));
    )JS"));

    failures += runJs("bounds: norms / rope / optimisers", block(R"JS(
        const X = mk(2, 4);
        const Y = mk(1, 1);
        return firstFail(
            throws("rmsNormForward short gamma", () => T.rmsNormForward(X, mk(2, 1), 1e-5, Y), "gamma"),
            throws("groupNormForward numGroups", () => T.groupNormForward(mk(1, 12), mk(3, 1), mk(3, 1), 1, 3, 2, 2, 2, 1e-5, Y), "divide"),
            throws("ropeForward cols", () => T.ropeForward(X, 4, 2, 0, 10000, Y), "columns"),
            throws("ropeForward odd headDim", () => T.ropeForward(mk(2, 3), 3, 1, 0, 10000, Y), "even"),
            throws("ropeApply short tables", () => T.ropeApply(X, mk(1, 1), mk(1, 1), 4, 1, Y), "cosTbl"),
            throws("adamStep short m", () => T.adamStep(X, mk(2, 4), mk(1, 1), mk(2, 4), 0.1, 0.9, 0.999, 1e-8, 1), "m"),
            throws("sgdStep short velocity", () => T.sgdStep(X, mk(2, 4), mk(1, 1), 0.1, 0.9), "velocity"));
    )JS"));

    failures += runJs("bounds: embedding indices", block(R"JS(
        const table = mk(3, 2, [0, 1, 10, 11, 20, 21]);
        const out = mk(1, 1);
        // FP32 indices go through a device INT32 buffer (they used to reach
        // the device kernel as a host pointer).
        T.embeddingLookupForward(table, mk(2, 1, [2, 0]), 2, out);
        const got = Array.from(out.download());
        if (got.join(",") !== "20,21,0,1") return "lookup gave " + got;
        return firstFail(
            throws("index out of range", () => T.embeddingLookupForward(table, mk(2, 1, [0, 3]), 2, out), "outside"),
            throws("negative index", () => T.embeddingLookupForward(table, mk(1, 1, [-1]), 1, out), "outside"),
            throws("short index tensor", () => T.embeddingLookupForward(table, mk(1, 1, [0]), 4, out), "indices"));
    )JS"));

    failures += runJs("bounds: attention shapes", block(R"JS(
        const X = mk(3, 4);
        const W = mk(4, 4);
        const O = mk(1, 1);
        return firstFail(
            throws("selfAttentionForward numHeads", () => T.selfAttentionForward(X, W, W, W, W, null, 3, O), "numHeads"),
            throws("selfAttentionForward short Wk", () => T.selfAttentionForward(X, W, mk(2, 2), W, W, null, 2, O), "Wk"),
            throws("selfAttentionForward short mask", () => T.selfAttentionForward(X, W, W, W, W, mk(1, 1), 2, O), "mask"));
    )JS"));

    failures += runJs("bounds: concat / split", block(R"JS(
        const a = mk(2, 1, [1, 2]);
        const h = mk(2, 1, null, "fp16");
        return firstFail(
            throws("concatRows mixed dtype", () => T.concatRows([a, h], mk(1, 1)), "fp16"),
            throws("splitRows overflow", () => T.splitRows(a, [mk(2, 1), mk(2, 1)]), "parts hold"),
            throws("concatBatchedRows rows", () => T.concatBatchedRows([mk(2, 2), mk(3, 2)], mk(1, 1)), "row count"));
    )JS"));

    return failures;
}
