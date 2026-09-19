// test_api_conv.cpp — the restored `conv` group of bro.tensor, exercised
// through the JS surface (js/tensor_conv.js -> native_tensor_conv.cpp).
//
// Every restored function is called; where a numeric answer is cheap to write
// down by hand (a padded 2x2 map, a max over a known 2x2, an adaptive pool to
// 1x1 = the mean, a unit-scale batch-norm inference, a 1x1 conv3d) the test
// pins the values, not just the shape. Nothing here is FP16 / INT8, and all 22
// ops are registered on the CPU backend, so no block is GPU-gated — the
// max-pool Idx tensor is INT32 and deliberately never downloaded (the CPU cast
// has no INT32 -> FP32 path; the backward pass is what proves it was filled).

#include "api_test_helpers.h"

#include <string>

namespace {

// The helpers every block shares: make/upload, exact-ish compare, shape check.
const char* kPrelude = R"JS(
    const tensor = globalThis.bro.tensor;
    const mk = (r, c, vals, dt) => {
        const t = tensor.createTensor(r, c, dt);
        if (vals) t.upload(vals);
        return t;
    };
    const eq = (t, exp, label, tol) => {
        const d = t.download();
        if (d.length !== exp.length) throw new Error(label + ": length " + d.length + " != " + exp.length);
        for (let i = 0; i < d.length; i++) {
            if (!(Math.abs(d[i] - exp[i]) < (tol || 1e-4))) {
                throw new Error(label + "[" + i + "] = " + d[i] + ", want " + exp[i] + " (got " + Array.from(d) + ")");
            }
        }
    };
    const shape = (t, r, c, label) => {
        if (t.rows !== r || t.cols !== c) {
            throw new Error(label + ": shape " + t.rows + "x" + t.cols + ", want " + r + "x" + c);
        }
    };
    const throwsType = (thunk, label) => {
        let threw = false;
        try { thunk(); } catch (e) { threw = (e instanceof TypeError); }
        if (!threw) throw new Error(label + ": expected a TypeError");
    };
)JS";

std::string js(const char* body) {
    return std::string("(function () {\n") + kPrelude + body + "\n    return \"OK\";\n})();\n";
}

} // namespace

int run_api_conv_tests() {
    using brotensor_api_test::runJs;
    int f = 0;

    // ---- pad2d / slice2d / interp2d backward --------------------------------
    f += runJs("test_api_conv.pad_slice_interp", js(R"JS(
    // pad2dForward: 1x1x2x2 [[1,2],[3,4]] padded by 1 on every side, zero mode.
    const X = mk(1, 4, [1, 2, 3, 4]);
    const Yp = mk(1, 16);
    tensor.pad2dForward(X, 1, 1, 2, 2, 1, 1, 1, 1, 0, Yp);
    shape(Yp, 1, 16, "pad2dForward Y");
    eq(Yp, [0,0,0,0,  0,1,2,0,  0,3,4,0,  0,0,0,0], "pad2dForward");

    // pad2dBackward: the zero-pad adjoint is the interior crop.
    const dXp = mk(1, 4);
    tensor.pad2dBackward(Yp, 1, 1, 2, 2, 1, 1, 1, 1, 0, dXp);
    eq(dXp, [1, 2, 3, 4], "pad2dBackward");

    // slice2dForward: the bottom-right 2x2 of a 3x3 counting map.
    const X3 = mk(1, 9, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const Ys = mk(1, 4);
    tensor.slice2dForward(X3, 1, 1, 3, 3, 1, 1, 2, 2, Ys);
    eq(Ys, [5, 6, 8, 9], "slice2dForward");

    // slice2dBackward: scatter it back into the same window, zeros elsewhere.
    const dXs = mk(1, 9);
    tensor.slice2dBackward(Ys, 1, 1, 3, 3, 1, 1, 2, 2, dXs);
    eq(dXs, [0,0,0,  0,5,6,  0,8,9], "slice2dBackward");

    // interp2dBackward: 1x1 -> 2x2 nearest; all four output pixels read the
    // single input pixel, so its gradient is their sum.
    const dY1 = mk(1, 4, [1, 1, 1, 1]);
    const dX1 = mk(1, 1);
    tensor.interp2dBackward(dY1, 1, 1, 1, 1, 2, 2, 0, dX1);
    eq(dX1, [4], "interp2dBackward");

    throwsType(() => tensor.pad2dForward("nope", 1, 1, 2, 2, 1, 1, 1, 1, 0, Yp), "pad2dForward(bad X)");
    throwsType(() => tensor.slice2dForward(X3, 1, 1, 3, 3, 1, 1, 2, 2, null), "slice2dForward(bad Y)");
)JS"));

    // ---- max pool / adaptive average pool -----------------------------------
    f += runJs("test_api_conv.pooling", js(R"JS(
    // maxPool2dForward over a known 2x2: the winner is 5 at flat offset 1.
    const X = mk(1, 4, [1, 5, 3, 2]);
    const Y = mk(1, 1);
    const Idx = tensor.createTensor(1, 1, "int32");
    tensor.maxPool2dForward(X, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, Y, Idx);
    shape(Y, 1, 1, "maxPool2dForward Y");
    shape(Idx, 1, 1, "maxPool2dForward Idx");
    eq(Y, [5], "maxPool2dForward");

    // maxPool2dBackward routes the whole gradient onto the winning pixel,
    // which is also the check that Idx carries the right offset.
    const dY = mk(1, 1, [7]);
    const dX = mk(1, 4);
    tensor.maxPool2dBackward(dY, Idx, 1, 1, 2, 2, 1, 1, dX);
    eq(dX, [0, 7, 0, 0], "maxPool2dBackward");

    // adaptiveAvgPool2d to 1x1 is the mean of the plane.
    const Xa = mk(1, 4, [1, 2, 3, 4]);
    const Ya = mk(1, 1);
    tensor.adaptiveAvgPool2dForward(Xa, 1, 1, 2, 2, 1, 1, Ya);
    eq(Ya, [2.5], "adaptiveAvgPool2dForward");

    // ... and its adjoint spreads dY evenly over the 4 contributing pixels.
    const dYa = mk(1, 1, [4]);
    const dXa = mk(1, 4);
    tensor.adaptiveAvgPool2dBackward(dYa, 1, 1, 2, 2, 1, 1, dXa);
    eq(dXa, [1, 1, 1, 1], "adaptiveAvgPool2dBackward");

    // A 2x2 pool over a 4x4 ramp: four 2x2 blocks, maxima 6, 8, 14, 16.
    const Xb = mk(1, 16, [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]);
    const Yb = mk(1, 4);
    const Ib = tensor.createTensor(1, 4, "int32");
    tensor.maxPool2dForward(Xb, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, Yb, Ib);
    eq(Yb, [6, 8, 14, 16], "maxPool2dForward 4x4");

    throwsType(() => tensor.maxPool2dForward(Xb, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, Yb, 3), "maxPool2dForward(bad Idx)");
)JS"));

    // ---- transposed conv2d + conv3d -----------------------------------------
    f += runJs("test_api_conv.transpose_and_3d", js(R"JS(
    // A 1x1 input through a 2x2 transposed kernel paints the kernel, scaled.
    //   X = [2], Wt = [1,2,3,4] (C_in, (C_out/groups)*kH*kW) -> Y = 2*Wt.
    const X = mk(1, 1, [2]);
    const Wt = mk(1, 4, [1, 2, 3, 4]);
    const Y = mk(1, 4);
    tensor.convTranspose2dForward(X, Wt, null, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, Y);
    shape(Y, 1, 4, "convTranspose2dForward Y");
    eq(Y, [2, 4, 6, 8], "convTranspose2dForward");

    // The same with a bias: +10 on every output pixel of channel 0.
    const bias = mk(1, 1, [10]);
    tensor.convTranspose2dForward(X, Wt, bias, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, Y);
    eq(Y, [12, 14, 16, 18], "convTranspose2dForward(bias)");

    // Backward to the input: dX = sum(Wt * dY) = 1+2+3+4 with dY all ones.
    const dY = mk(1, 4, [1, 1, 1, 1]);
    const dX = mk(1, 1);
    tensor.convTranspose2dBackwardInput(Wt, dY, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, dX);
    eq(dX, [10], "convTranspose2dBackwardInput");

    // Backward to the weights ACCUMULATES, so dWt starts zeroed: dWt = X*dY.
    const dY2 = mk(1, 4, [1, 2, 3, 4]);
    const dWt = mk(1, 4);
    dWt.zero();
    tensor.convTranspose2dBackwardWeight(X, dY2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, dWt);
    eq(dWt, [2, 4, 6, 8], "convTranspose2dBackwardWeight");

    // Backward to the bias: the sum of dY over the output plane, accumulated.
    const dB = mk(1, 1);
    dB.zero();
    tensor.convTranspose2dBackwardBias(dY2, 1, 1, 2, 2, dB);
    eq(dB, [10], "convTranspose2dBackwardBias");
    tensor.convTranspose2dBackwardBias(dY2, 1, 1, 2, 2, dB);
    eq(dB, [20], "convTranspose2dBackwardBias accumulates");

    // conv3d: a 1x1x1 kernel over a single voxel is a multiply.
    const X3 = mk(1, 1, [3]);
    const W3 = mk(1, 1, [2]);
    const Y3 = mk(1, 1);
    tensor.conv3dForward(X3, W3, null, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, Y3);
    eq(Y3, [6], "conv3dForward");
    const b3 = mk(1, 1, [1]);
    tensor.conv3dForward(X3, W3, b3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, Y3);
    eq(Y3, [7], "conv3dForward(bias)");

    // A 2x2x1 volume with a 1x1x1 kernel: element-wise scale, shape preserved.
    const Xv = mk(1, 4, [1, 2, 3, 4]);
    const Yv = mk(1, 4);
    tensor.conv3dForward(Xv, W3, null, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, Yv);
    shape(Yv, 1, 4, "conv3dForward volume Y");
    eq(Yv, [2, 4, 6, 8], "conv3dForward volume");

    throwsType(() => tensor.convTranspose2dForward(X, Wt, "bias", 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, Y),
               "convTranspose2dForward(bad bias)");
)JS"));

    // ---- window partition / reverse / 2x2 spatial merge ----------------------
    f += runJs("test_api_conv.window_and_merge", js(R"JS(
    // window=1 on a 2x2 plane splits every pixel into its own batch row.
    const X = mk(1, 4, [1, 2, 3, 4]);
    const Yw = mk(4, 1);
    tensor.windowPartitionForward(X, 1, 1, 2, 2, 1, Yw);
    shape(Yw, 4, 1, "windowPartitionForward Y");
    eq(Yw, [1, 2, 3, 4], "windowPartitionForward");

    // windowReverseForward is the exact inverse.
    const Yr = mk(1, 4);
    tensor.windowReverseForward(Yw, 1, 1, 2, 2, 1, Yr);
    shape(Yr, 1, 4, "windowReverseForward Y");
    eq(Yr, [1, 2, 3, 4], "windowReverseForward");

    // window=2 over a 4x4 plane: 4 windows of 4 pixels, row-major by window.
    const X4 = mk(1, 16, [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]);
    const Yw4 = mk(4, 4);
    tensor.windowPartitionForward(X4, 1, 1, 4, 4, 2, Yw4);
    eq(Yw4, [1,2,5,6,  3,4,7,8,  9,10,13,14,  11,12,15,16], "windowPartitionForward 4x4");
    const Yr4 = mk(1, 16);
    tensor.windowReverseForward(Yw4, 1, 1, 4, 4, 2, Yr4);
    eq(Yr4, [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16], "windowReverseForward 4x4");

    // spatialMerge2x2Forward, C=2 so the two orderings differ:
    //   channel 0 = [1,2,3,4], channel 1 = [5,6,7,8], block = dh*2 + dw.
    //   block-major   c_out = block*C + c_in -> [1,5, 2,6, 3,7, 4,8]
    //   channel-major c_out = c_in*4 + block -> [1,2,3,4, 5,6,7,8]
    const Xm = mk(1, 8, [1, 2, 3, 4, 5, 6, 7, 8]);
    const Ym = mk(1, 8);
    tensor.spatialMerge2x2Forward(Xm, 1, 2, 2, 2, Ym);
    shape(Ym, 1, 8, "spatialMerge2x2Forward Y");
    eq(Ym, [1, 5, 2, 6, 3, 7, 4, 8], "spatialMerge2x2Forward(block-major)");
    tensor.spatialMerge2x2Forward(Xm, 1, 2, 2, 2, Ym, true);
    eq(Ym, [1, 2, 3, 4, 5, 6, 7, 8], "spatialMerge2x2Forward(channel-major)");

    throwsType(() => tensor.windowPartitionForward(X, 1, 1, 2, 2, 1, undefined), "windowPartitionForward(bad Y)");
)JS"));

    // ---- batch norm + image preprocessing -----------------------------------
    f += runJs("test_api_conv.batchnorm_and_image", js(R"JS(
    // batchNormInference with unit scale and zero shift is the identity:
    //   running mean 0, running var 1, eps 0 -> rstd 1.
    const X = mk(1, 2, [1, 3]);
    const gamma = mk(2, 1, [1, 1]);
    const beta = mk(2, 1, [0, 0]);
    const rmean = mk(2, 1, [0, 0]);
    const rvar = mk(2, 1, [1, 1]);
    const Yi = mk(1, 2);
    tensor.batchNormInference(X, gamma, beta, rmean, rvar, 1, 2, 1, 1, 0, Yi);
    shape(Yi, 1, 2, "batchNormInference Y");
    eq(Yi, [1, 3], "batchNormInference(identity)");

    // ... and the affine pair really is applied: y = 2*x + 1.
    const gamma2 = mk(2, 1, [2, 2]);
    const beta2 = mk(2, 1, [1, 1]);
    tensor.batchNormInference(X, gamma2, beta2, rmean, rvar, 1, 2, 1, 1, 0, Yi);
    eq(Yi, [3, 7], "batchNormInference(affine)");

    // Training forward over N=2, C=1: mean 2, biased var 1, so rstd 1 and
    // Y = [-1, 1]; the saved caches feed the backward below.
    const Xt = mk(2, 1, [1, 3]);
    const g1 = mk(1, 1, [1]);
    const b1 = mk(1, 1, [0]);
    const rm = mk(1, 1, [0]);
    const rv = mk(1, 1, [1]);
    const Yt = mk(2, 1);
    const sMean = mk(1, 1);
    const sRstd = mk(1, 1);
    tensor.batchNormForward(Xt, g1, b1, rm, rv, 2, 1, 1, 1, 0, 0.1, Yt, sMean, sRstd);
    shape(Yt, 2, 1, "batchNormForward Y");
    shape(sMean, 1, 1, "batchNormForward savedMean");
    eq(Yt, [-1, 1], "batchNormForward Y");
    eq(sMean, [2], "batchNormForward savedMean");
    eq(sRstd, [1], "batchNormForward savedRstd");
    // The running stats moved off their initial values (momentum 0.1).
    const rmd = rm.download();
    if (!(Math.abs(rmd[0] - 0.2) < 1e-4)) throw new Error("batchNormForward runningMean: " + rmd[0]);
    const rvd = rv.download();
    if (!isFinite(rvd[0]) || rvd[0] <= 0) throw new Error("batchNormForward runningVar: " + rvd[0]);

    // Backward with dY all ones: dxhat = [1,1], sum dxhat = 2, M = 2 and
    // sum(dxhat*xhat) = 0, so dX = 0; dGamma = sum dY*xhat = 0, dBeta = 2.
    const dY = mk(2, 1, [1, 1]);
    const dX = mk(2, 1);
    const dGamma = mk(1, 1);
    const dBeta = mk(1, 1);
    dGamma.zero(); dBeta.zero();
    tensor.batchNormBackward(Xt, g1, sMean, sRstd, dY, 2, 1, 1, 1, dX, dGamma, dBeta);
    eq(dX, [0, 0], "batchNormBackward dX");
    eq(dGamma, [0], "batchNormBackward dGamma");
    eq(dBeta, [2], "batchNormBackward dBeta");

    // imageNormalize: per-channel (x - mean[c]) / std[c].
    const Xn = mk(1, 2, [4, 10]);
    const mean = mk(2, 1, [2, 4]);
    const std = mk(2, 1, [2, 3]);
    const Yn = mk(1, 2);
    tensor.imageNormalize(Xn, mean, std, 1, 2, 1, 1, Yn);
    eq(Yn, [1, 2], "imageNormalize");

    // imageU8ToF32NhwcToNchw: 1x2 RGB pixels, HWC in -> NCHW out.
    //   src (h=0,w=0) = 10,20,30   (h=0,w=1) = 40,50,60
    //   Y: channel 0 = [10,40], channel 1 = [20,50], channel 2 = [30,60].
    const src = new Uint8Array([10, 20, 30, 40, 50, 60]);
    const Yu = mk(1, 6);
    tensor.imageU8ToF32NhwcToNchw(src, 1, 1, 2, 3, 1.0, 0.0, Yu);
    shape(Yu, 1, 6, "imageU8ToF32NhwcToNchw Y");
    eq(Yu, [10, 40, 20, 50, 30, 60], "imageU8ToF32NhwcToNchw");

    // The two canonical scaling conventions on the 0/255 endpoints.
    const ends = new Uint8Array([0, 255]);
    const Y01 = mk(2, 1);
    tensor.imageU8ToF32NhwcToNchw(ends, 2, 1, 1, 1, 1 / 255, 0, Y01);
    eq(Y01, [0, 1], "imageU8ToF32NhwcToNchw [0,1]");
    const Ym11 = mk(2, 1);
    tensor.imageU8ToF32NhwcToNchw(ends, 2, 1, 1, 1, 2 / 255, -1, Ym11);
    eq(Ym11, [-1, 1], "imageU8ToF32NhwcToNchw [-1,1]");

    // A plain array of byte values is accepted too (copied to a Uint8Array).
    const Ya = mk(1, 6);
    tensor.imageU8ToF32NhwcToNchw([10, 20, 30, 40, 50, 60], 1, 1, 2, 3, 1.0, 0.0, Ya);
    eq(Ya, [10, 40, 20, 50, 30, 60], "imageU8ToF32NhwcToNchw(array)");

    // A short buffer is an error from the native, not a read past the end.
    let threw = false;
    try { tensor.imageU8ToF32NhwcToNchw(new Uint8Array([1, 2]), 1, 1, 2, 3, 1, 0, Ya); } catch (e) { threw = true; }
    if (!threw) throw new Error("imageU8ToF32NhwcToNchw with a short src did not throw");

    throwsType(() => tensor.imageU8ToF32NhwcToNchw("bytes", 1, 1, 2, 3, 1, 0, Ya), "imageU8ToF32NhwcToNchw(bad src)");
    throwsType(() => tensor.batchNormInference(X, gamma, beta, rmean, null, 1, 2, 1, 1, 0, Yi), "batchNormInference(bad runningVar)");
)JS"));

    return f;
}
