#include "api/api.h"
#include "api_test_helpers.h"
#include "embed/embed.h"
#include "eval/eval.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static void test_native_handles() {
    std::printf("test_native_handles...\n");
    brotensor::Tensor t = brotensor::Tensor::zeros(3, 4, brotensor::Dtype::FP32);
    bronze::Value val = brotensor::api::createGpuTensorValue(std::move(t));
    CHECK(!bronze::embed::isUndefined(val));
    CHECK(!bronze::embed::isNull(val));

    brotensor::Tensor* unwrapped = brotensor::api::getTensorFromHandle(val);
    CHECK(unwrapped != nullptr);
    CHECK(unwrapped->rows == 3);
    CHECK(unwrapped->cols == 4);
    CHECK(unwrapped->size() == 12);
}

static void test_js_api() {
    std::printf("test_js_api...\n");

    auto r_avail = bronze::eval::evalScript("globalThis.bro.tensor.available");
    CHECK(!r_avail.thrown);
    CHECK(bronze::embed::toBool(r_avail.value) == true);

    auto r_back = bronze::eval::evalScript("globalThis.bro.tensor.backend");
    CHECK(!r_back.thrown);
    CHECK(bronze::embed::isString(r_back.value));

    const char* script = R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;

        // 1. Create tensor and test accessors
        const t = tensor.createTensor(2, 3);
        if (t.rows !== 2 || t.cols !== 3 || t.size !== 6) {
            throw new Error("shape mismatch: " + t.rows + "x" + t.cols);
        }

        // 2. Upload and download
        t.upload([1, 2, 3, 4, 5, 6]);
        const d = t.download();
        if (d.length !== 6 || d[0] !== 1 || d[5] !== 6) {
            throw new Error("download mismatch");
        }

        // 3. Zero
        t.zero();
        const d_zero = t.download();
        if (d_zero[0] !== 0 || d_zero[5] !== 0) {
            throw new Error("zero mismatch");
        }

        // 4. Clone and resize
        t.upload([10, 20, 30, 40, 50, 60]);
        const c = t.clone();
        if (c.rows !== 2 || c.cols !== 3 || c.download()[1] !== 20) {
            throw new Error("clone mismatch");
        }
        c.resize(1, 6);
        if (c.rows !== 1 || c.cols !== 6 || c.size !== 6) {
            throw new Error("resize mismatch");
        }

        // 5. Activation: reluForward
        const x = tensor.createTensor(2, 2);
        const y = tensor.createTensor(2, 2);
        x.upload([-2, 3, -4, 5]);
        tensor.reluForward(x, y);
        const y_relu = y.download();
        if (y_relu[0] !== 0 || y_relu[1] !== 3 || y_relu[2] !== 0 || y_relu[3] !== 5) {
            throw new Error("relu mismatch: " + Array.from(y_relu));
        }

        // 6. Elementwise: addInplace
        tensor.addInplace(y, x);
        const y_add = y.download();
        if (y_add[0] !== -2 || y_add[1] !== 6 || y_add[2] !== -4 || y_add[3] !== 10) {
            throw new Error("addInplace mismatch: " + Array.from(y_add));
        }

        // 7. Matmul: (2x3) @ (3x2) -> (2x2)
        const A = tensor.createTensor(2, 3);
        const B = tensor.createTensor(3, 2);
        const C = tensor.createTensor(2, 2);
        A.upload([1, 2, 3, 4, 5, 6]);
        B.upload([7, 8, 9, 1, 2, 3]);
        tensor.matmul(A, B, C);
        const c_out = C.download();
        // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        if (c_out[0] !== 31 || c_out[1] !== 19) {
            throw new Error("matmul mismatch: " + Array.from(c_out));
        }

        // 8. Softmax
        const logits = tensor.createTensor(1, 3);
        const probs = tensor.createTensor(1, 3);
        logits.upload([0, 0, 0]);
        tensor.softmaxForward(logits, probs);
        const p_out = probs.download();
        if (Math.abs(p_out[0] - 1/3) > 1e-4) {
            throw new Error("softmax mismatch: " + Array.from(p_out));
        }

        // 9. Tanh and Clamp
        const act_in = tensor.createTensor(1, 2);
        const act_out = tensor.createTensor(1, 2);
        act_in.upload([0, 10]);
        tensor.tanhForward(act_in, act_out);
        const tanh_d = act_out.download();
        if (Math.abs(tanh_d[0]) > 1e-4 || Math.abs(tanh_d[1] - 1.0) > 1e-4) {
            throw new Error("tanh mismatch: " + Array.from(tanh_d));
        }
        tensor.clamp(act_in, -1, 1);
        const clamp_d = act_in.download();
        if (clamp_d[1] !== 1) {
            throw new Error("clamp mismatch: " + Array.from(clamp_d));
        }

        return "OK";
    })();
    )JS";

    auto r = bronze::eval::evalScript(script);
    if (r.thrown) {
        std::printf("  JS Exception thrown during test: %s\n",
                    bronze::embed::toUtf8(r.value).c_str());
        CHECK(!r.thrown);
    } else {
        std::printf("  JS tests passed successfully: %s\n",
                    bronze::embed::toUtf8(r.value).c_str());
    }
}

// Run one JS block; a throw or a non-"OK" result is a failure.
static void run_js(const char* name, const std::string& script) {
    std::printf("%s...\n", name);
    auto r = bronze::eval::evalScript(script);
    if (r.thrown) {
        std::printf("  JS Exception thrown: %s\n", bronze::embed::toUtf8(r.value).c_str());
        CHECK(!r.thrown);
        return;
    }
    const std::string out = bronze::embed::toUtf8(r.value);
    if (out != "OK") {
        std::printf("  unexpected result: %s\n", out.c_str());
        CHECK(out == "OK");
    }
}

// The methods the port had stubbed or narrowed (bro/docs/transition-drift.md
// H1, H2 and the tensor rows of the static handler diff), each against the
// old binding's contract.

// H1: openSafetensors(path) + the handle's count / names() / header() /
// get(name, rows?, cols?, dtype?) / close(), and saveSafetensors.
static void test_js_safetensors() {
    namespace st = brotensor::safetensors;
    const auto dir = std::filesystem::temp_directory_path();
    const auto fixture = dir / "brotensor_api_test.safetensors";
    const auto saved = dir / "brotensor_api_test_saved.safetensors";
    {
        const float alpha[6] = {1, 2, 3, 4, 5, 6};
        const uint16_t beta[4] = {0x3c00, 0x4000, 0x4200, 0x4400};  // 1, 2, 3, 4 in fp16
        std::vector<st::WriteEntry> entries(2);
        entries[0].name = "alpha";
        entries[0].dtype = st::Dtype::F32;
        entries[0].shape = {2, 3};
        entries[0].host_data = alpha;
        entries[0].bytes = sizeof(alpha);
        entries[1].name = "beta";
        entries[1].dtype = st::Dtype::F16;
        entries[1].shape = {4};
        entries[1].host_data = beta;
        entries[1].bytes = sizeof(beta);
        st::write_file(fixture.generic_string(), entries);
    }
    const std::string script = std::string(R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;
        const path = ")JS") + fixture.generic_string() + R"JS(";
        const savedPath = ")JS" + saved.generic_string() + R"JS(";
        const same = (a, b) => Math.abs(a - b) < 1e-4;

        let threw = false;
        try { tensor.openSafetensors(path + ".missing"); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("open of a missing file did not throw a TypeError");

        const f = tensor.openSafetensors(path);
        if (!(f instanceof tensor.SafetensorsFile)) throw new Error("handle is not a SafetensorsFile");
        if (f.count !== 2) throw new Error("count: " + f.count);
        const names = f.names();
        if (names.length !== 2 || names[0] !== "alpha" || names[1] !== "beta") throw new Error("names: " + names);
        const hdr = f.header();
        if (hdr.alpha.dtype !== "F32" || hdr.alpha.shape.length !== 2 || hdr.alpha.shape[0] !== 2 || hdr.alpha.shape[1] !== 3 || hdr.alpha.nbytes !== 24) {
            throw new Error("header.alpha: " + JSON.stringify(hdr.alpha));
        }
        if (hdr.beta.dtype !== "F16" || hdr.beta.shape[0] !== 4 || hdr.beta.nbytes !== 8) throw new Error("header.beta: " + JSON.stringify(hdr.beta));

        // native dtype, shape from the file
        const a = f.get("alpha");
        if (!(a instanceof tensor.GpuTensor) || a.rows !== 2 || a.cols !== 3 || a.dtype() !== "fp32") throw new Error("get(alpha): " + a.rows + "x" + a.cols + " " + a.dtype());
        const ad = a.download();
        for (let i = 0; i < 6; i++) if (!same(ad[i], i + 1)) throw new Error("get(alpha) data: " + Array.from(ad));
        // explicit 2D shape
        const a2 = f.get("alpha", 3, 2);
        if (a2.rows !== 3 || a2.cols !== 2) throw new Error("get(alpha,3,2): " + a2.rows + "x" + a2.cols);
        // dtype string in place of rows/cols, and after them
        const a16 = f.get("alpha", "fp16");
        if (a16.dtype() !== "fp16" || a16.size !== 6) throw new Error("get(alpha,'fp16'): " + a16.dtype());
        const a16b = f.get("alpha", 1, 6, "fp16");
        if (a16b.dtype() !== "fp16" || a16b.rows !== 1 || a16b.cols !== 6) throw new Error("get(alpha,1,6,'fp16')");
        const ad16 = a16.download();
        for (let i = 0; i < 6; i++) if (!same(ad16[i], i + 1)) throw new Error("fp16 upload data: " + Array.from(ad16));
        // "compute": the backend's compute dtype; an F16 source converts as needed
        const bc = f.get("beta", "compute");
        const expectCompute = tensor.backend === "cpu" ? "fp32" : "fp16";
        if (bc.dtype() !== expectCompute || bc.rows !== 4 || bc.cols !== 1) throw new Error("get(beta,'compute'): " + bc.dtype() + " " + bc.rows + "x" + bc.cols);
        const bd = bc.download();
        for (let i = 0; i < 4; i++) if (!same(bd[i], i + 1)) throw new Error("beta data: " + Array.from(bd));
        // unknown name -> RangeError
        threw = false;
        try { f.get("nope"); } catch (e) { threw = e instanceof RangeError; }
        if (!threw) throw new Error("get of an unknown name did not throw a RangeError");

        f.close();
        threw = false;
        try { f.names(); } catch (e) { threw = true; }
        if (!threw) throw new Error("names() after close did not throw");
        if (f.count !== 0) throw new Error("count after close: " + f.count);

        // saveSafetensors round trip (FP32 + FP16)
        const w = tensor.createTensor(2, 2);
        w.upload([1.5, -2, 3, 4]);
        const h = tensor.createTensor(1, 2, "fp16");
        tensor.cast(w, h, "fp16");
        tensor.saveSafetensors(savedPath, { w: w, h: h });
        const g = tensor.openSafetensors(savedPath);
        const gh = g.header();
        if (gh.w.dtype !== "F32" || gh.w.shape[0] !== 2 || gh.w.shape[1] !== 2) throw new Error("saved w header: " + JSON.stringify(gh.w));
        if (gh.h.dtype !== "F16" || gh.h.shape[0] !== 2 || gh.h.shape[1] !== 2) throw new Error("saved h header: " + JSON.stringify(gh.h));
        const wd = g.get("w").download();
        if (!same(wd[0], 1.5) || !same(wd[1], -2) || !same(wd[3], 4)) throw new Error("saved w data: " + Array.from(wd));
        const hd = g.get("h").download();
        if (!same(hd[0], 1.5) || !same(hd[1], -2)) throw new Error("saved h data: " + Array.from(hd));
        g.close();

        // 4D tensor shape preservation test
        const t4d = tensor.createTensor(2, 12);
        t4d.shape = [2, 3, 2, 2];
        tensor.saveSafetensors(savedPath, { t4d: t4d });
        const g2 = tensor.openSafetensors(savedPath);
        const gh2 = g2.header();
        if (!gh2.t4d || gh2.t4d.shape.length !== 4 || gh2.t4d.shape[0] !== 2 || gh2.t4d.shape[1] !== 3 || gh2.t4d.shape[2] !== 2 || gh2.t4d.shape[3] !== 2) {
            throw new Error("saved t4d header: " + JSON.stringify(gh2.t4d));
        }
        const t4d_loaded = g2.get("t4d");
        if (!t4d_loaded.shape || t4d_loaded.shape.length !== 4 || t4d_loaded.shape[1] !== 3) {
            throw new Error("loaded t4d shape: " + JSON.stringify(t4d_loaded.shape));
        }
        g2.close();
        // Round-trip saving the loaded 4D tensor
        tensor.saveSafetensors(savedPath, { t4d_copy: t4d_loaded });
        const g3 = tensor.openSafetensors(savedPath);
        const gh3 = g3.header();
        if (!gh3.t4d_copy || gh3.t4d_copy.shape.length !== 4 || gh3.t4d_copy.shape[1] !== 3) {
            throw new Error("re-saved t4d_copy header: " + JSON.stringify(gh3.t4d_copy));
        }
        g3.close();

        threw = false;
        try { tensor.saveSafetensors(savedPath, { w: 5 }); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("saveSafetensors with a non-tensor did not throw");
        return "OK";
    })();
    )JS";
    run_js("test_js_safetensors", script);
    std::error_code ec;
    std::filesystem::remove(fixture, ec);
    std::filesystem::remove(saved, ec);
}

// randn / randUniform / randBernoulli / randnTruncated / xavierInit: the
// counter-based RNG family (randn was a stub that only threw).
static void test_js_rng() {
    run_js("test_js_rng", R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;
        const Y = tensor.createTensor(8, 64);
        tensor.randn(7, 0, Y);
        const a = Y.download();
        let sum = 0, sq = 0, nonzero = 0;
        for (let i = 0; i < a.length; i++) { sum += a[i]; sq += a[i] * a[i]; if (a[i] !== 0) nonzero++; }
        const mean = sum / a.length, var_ = sq / a.length - mean * mean;
        if (nonzero < a.length / 2) throw new Error("randn left the tensor mostly zero");
        if (Math.abs(mean) > 0.25 || var_ < 0.6 || var_ > 1.5) throw new Error("randn moments: mean " + mean + " var " + var_);
        // deterministic in (key, counter); a BigInt key is accepted too
        tensor.randn(7, 0, Y);
        const b = Y.download();
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) throw new Error("randn is not deterministic");
        tensor.randn(BigInt(7), BigInt(0), Y);
        const c = Y.download();
        for (let i = 0; i < a.length; i++) if (a[i] !== c[i]) throw new Error("randn(BigInt) differs from randn(Number)");
        tensor.randn(8, 0, Y);
        const d = Y.download();
        let diff = 0;
        for (let i = 0; i < a.length; i++) if (a[i] !== d[i]) diff++;
        if (diff < a.length / 2) throw new Error("randn ignores the key");

        tensor.randUniform(1, 0, Y);
        const u = Y.download();
        for (let i = 0; i < u.length; i++) if (u[i] < 0 || u[i] >= 1) throw new Error("randUniform out of [0,1): " + u[i]);
        tensor.randBernoulli(1.0, 1, 0, Y);
        const ones = Y.download();
        for (let i = 0; i < ones.length; i++) if (ones[i] !== 1) throw new Error("randBernoulli(1.0) gave " + ones[i]);
        tensor.randBernoulli(0.0, 1, 0, Y);
        const zeros = Y.download();
        for (let i = 0; i < zeros.length; i++) if (zeros[i] !== 0) throw new Error("randBernoulli(0.0) gave " + zeros[i]);
        tensor.randnTruncated(-1, 1, 3, 0, Y);
        const t = Y.download();
        for (let i = 0; i < t.length; i++) if (t[i] < -1 || t[i] > 1) throw new Error("randnTruncated out of [-1,1]: " + t[i]);

        let threw = false;
        try { tensor.randn("7", 0, Y); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("randn with a string key did not throw");

        if (tensor.backend === "cpu") {
            const W = tensor.createTensor(4, 4);
            const next = tensor.xavierInit(W, 12345);
            if (typeof next !== "number" || next === 12345) throw new Error("xavierInit did not advance the state: " + next);
            const wd = W.download();
            let nz = 0;
            for (let i = 0; i < wd.length; i++) { if (wd[i] !== 0) nz++; if (Math.abs(wd[i]) > 1) throw new Error("xavier value out of range: " + wd[i]); }
            if (nz < 8) throw new Error("xavierInit left W zero");
        }
        return "OK";
    })();
    )JS");
}

// H2: download(dst) fills a caller-owned Float32Array in place; download()
// still answers a fresh one. Plus the dtype overloads (numeric enum values,
// the f32/f16 aliases) on createTensor / resize / cast, and cast itself.
static void test_js_download_and_dtypes() {
    run_js("test_js_download_and_dtypes", R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;
        const t = tensor.createTensor(2, 3);
        t.upload([1, 2, 3, 4, 5, 6]);
        const dst = new Float32Array(6);
        const ret = t.download(dst);
        if (ret !== dst) throw new Error("download(dst) did not return dst");
        for (let i = 0; i < 6; i++) if (dst[i] !== i + 1) throw new Error("download(dst) data: " + Array.from(dst));
        const big = new Float32Array(8);
        big[7] = 42;
        t.download(big);
        if (big[5] !== 6 || big[7] !== 42) throw new Error("download(dst) into a larger buffer");
        let threw = false;
        try { t.download(new Float32Array(5)); } catch (e) { threw = e instanceof RangeError; }
        if (!threw) throw new Error("download into a too-small buffer did not throw a RangeError");
        threw = false;
        try { t.download([0, 0, 0, 0, 0, 0]); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("download(array) did not throw a TypeError");
        const fresh = t.download();
        if (!(fresh instanceof Float32Array) || fresh.length !== 6 || fresh[2] !== 3) throw new Error("download() changed");

        // dtype overloads: numeric enum and the short aliases
        const h = tensor.createTensor(2, 3, tensor.dtype.fp16);
        if (h.dtype() !== "fp16") throw new Error("createTensor(dtype.fp16): " + h.dtype());
        h.resize(3, 2, "f32");
        if (h.dtype() !== "fp32" || h.rows !== 3) throw new Error("resize(.., 'f32'): " + h.dtype());
        threw = false;
        try { h.resize(3); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("resize(rows) without cols did not throw");

        // cast: FP32 -> FP16 -> FP32 round trip; dst is resized + dtype-set
        const c16 = tensor.createTensor(1, 1);
        tensor.cast(t, c16, tensor.dtype.fp16);
        if (c16.dtype() !== "fp16" || c16.rows !== 2 || c16.cols !== 3) throw new Error("cast to fp16: " + c16.dtype() + " " + c16.rows + "x" + c16.cols);
        const u16 = c16.downloadFp16();
        if (!(u16 instanceof Uint16Array) || u16.length !== 6 || u16[0] !== 0x3c00) throw new Error("downloadFp16 after cast: " + Array.from(u16));
        const back = tensor.createTensor(1, 1);
        tensor.cast(c16, back, "fp32");
        const bd = back.download();
        for (let i = 0; i < 6; i++) if (bd[i] !== i + 1) throw new Error("cast round trip: " + Array.from(bd));
        return "OK";
    })();
    )JS");
}

// layernormForward's {mean, rstd} (was a constant stub), softmaxForward's
// mask argument (the port had replaced it with a temperature), masked
// mean-pool, the fused softmax + cross-entropy loss, concat / split.
static void test_js_restored_ops() {
    run_js("test_js_restored_ops", R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-4);

        // layernormForward
        const x = tensor.createTensor(4, 1), g = tensor.createTensor(4, 1), b = tensor.createTensor(4, 1);
        const y = tensor.createTensor(4, 1), xhat = tensor.createTensor(4, 1);
        x.upload([1, 2, 3, 4]); g.upload([1, 1, 1, 1]); b.upload([0, 0, 0, 0]);
        const ln = tensor.layernormForward(x, g, b, y, xhat, 1e-5);
        if (!same(ln.mean, 2.5) || !same(ln.rstd, 1 / Math.sqrt(1.25 + 1e-5), 1e-3)) throw new Error("layernormForward stats: " + JSON.stringify(ln));
        const yd = y.download();
        if (!same(yd[0], -1.5 * ln.rstd, 1e-3) || !same(yd[3], 1.5 * ln.rstd, 1e-3)) throw new Error("layernormForward y: " + Array.from(yd));

        // softmaxForward with a key mask
        const logits = tensor.createTensor(1, 4), probs = tensor.createTensor(1, 4), mask = tensor.createTensor(4, 1);
        logits.upload([0, 0, 0, 0]); mask.upload([1, 1, 0, 0]);
        tensor.softmaxForward(logits, probs, mask);
        const pd = probs.download();
        if (!same(pd[0], 0.5) || !same(pd[1], 0.5) || !same(pd[2], 0) || !same(pd[3], 0)) throw new Error("masked softmax: " + Array.from(pd));
        tensor.softmaxForward(logits, probs, null);
        const pn = probs.download();
        if (!same(pn[2], 0.25)) throw new Error("unmasked softmax: " + Array.from(pn));
        let threw = false;
        try { tensor.softmaxForward(logits, probs, "mask"); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("softmaxForward with a bad mask did not throw");

        // maskedMeanPool forward + backward
        const X = tensor.createTensor(2, 3), pooled = tensor.createTensor(3, 1), rowMask = tensor.createTensor(2, 1);
        X.upload([1, 2, 3, 4, 5, 6]);
        tensor.maskedMeanPoolForward(X, null, pooled);
        const p0 = pooled.download();
        if (!same(p0[0], 2.5) || !same(p0[1], 3.5) || !same(p0[2], 4.5)) throw new Error("maskedMeanPool(null): " + Array.from(p0));
        rowMask.upload([1, 0]);
        tensor.maskedMeanPoolForward(X, rowMask, pooled);
        const p1 = pooled.download();
        if (!same(p1[0], 1) || !same(p1[1], 2) || !same(p1[2], 3)) throw new Error("maskedMeanPool(mask): " + Array.from(p1));
        const dY = tensor.createTensor(3, 1), dX = tensor.createTensor(2, 3);
        dY.upload([3, 6, 9]);
        tensor.maskedMeanPoolBackward(dY, rowMask, 2, dX);
        const dxd = dX.download();
        if (!same(dxd[0], 3) || !same(dxd[2], 9) || !same(dxd[3], 0)) throw new Error("maskedMeanPoolBackward: " + Array.from(dxd));

        // softmaxXentFused: uniform logits, one-hot target -> loss ln(2)
        const l2 = tensor.createTensor(2, 1), t2 = tensor.createTensor(2, 1), pr2 = tensor.createTensor(2, 1), dl2 = tensor.createTensor(2, 1);
        l2.upload([0, 0]); t2.upload([1, 0]);
        const loss = tensor.softmaxXentFused(l2, t2, null, pr2, dl2);
        if (!same(loss, Math.LN2, 1e-3)) throw new Error("softmaxXentFused loss: " + loss);
        const dld = dl2.download();
        if (!same(dld[0], -0.5) || !same(dld[1], 0.5)) throw new Error("softmaxXentFused dLogits: " + Array.from(dld));

        // concatRows / splitRows / concatBatchedRows / concatNchwChannels
        const a = tensor.createTensor(2, 1), c = tensor.createTensor(3, 1), out = tensor.createTensor(1, 1);
        a.upload([1, 2]); c.upload([3, 4, 5]);
        tensor.concatRows([a, c], out);
        if (out.size !== 5) throw new Error("concatRows size: " + out.size);
        const od = out.download();
        for (let i = 0; i < 5; i++) if (od[i] !== i + 1) throw new Error("concatRows data: " + Array.from(od));
        const s1 = tensor.createTensor(2, 1), s2 = tensor.createTensor(3, 1);
        tensor.splitRows(out, [s1, s2]);
        const s2d = s2.download();
        if (s2d[0] !== 3 || s2d[2] !== 5) throw new Error("splitRows: " + Array.from(s2d));
        const p = tensor.createTensor(2, 1), q = tensor.createTensor(2, 2), bo = tensor.createTensor(1, 1);
        p.upload([1, 2]); q.upload([3, 4, 5, 6]);
        tensor.concatBatchedRows([p, q], bo);
        const bod = bo.download();
        if (bo.rows !== 2 || bo.cols !== 3 || bod[0] !== 1 || bod[1] !== 3 || bod[3] !== 2 || bod[5] !== 6) throw new Error("concatBatchedRows: " + bo.rows + "x" + bo.cols + " " + Array.from(bod));
        const n1 = tensor.createTensor(1, 2), n2 = tensor.createTensor(1, 4), no = tensor.createTensor(1, 1);
        n1.upload([1, 2]); n2.upload([3, 4, 5, 6]);
        tensor.concatNchwChannels([n1, n2], 1, 1, 2, [1, 2], no);
        const nod = no.download();
        if (no.cols !== 6 || nod[0] !== 1 || nod[2] !== 3 || nod[5] !== 6) throw new Error("concatNchwChannels: " + Array.from(nod));
        const g1 = tensor.createTensor(1, 1), g2 = tensor.createTensor(1, 1);
        tensor.concatNchwChannelsBackward(no, 1, 1, 2, [1, 2], [g1, g2]);
        const g2d = g2.download();
        if (g2.cols !== 4 || g2d[0] !== 3 || g2d[3] !== 6) throw new Error("concatNchwChannelsBackward: " + Array.from(g2d));
        threw = false;
        try { tensor.concatRows([a, 3], out); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("concatRows with a non-tensor part did not throw");
        return "OK";
    })();
    )JS");
}

// The attention family (every entry was a `return 0` stub): the FP32
// training paths on any backend; the FP16 inference paths only where the
// backend has them.
static void test_js_attention() {
    run_js("test_js_attention", R"JS(
    (function() {
        const tensor = globalThis.bro.tensor;
        const same = (a, b, tol) => Math.abs(a - b) < (tol || 1e-3);
        const finite = (t, label) => { const d = t.download(); for (let i = 0; i < d.length; i++) if (!isFinite(d[i])) throw new Error(label + " not finite"); return d; };
        const D = 2, L = 2;
        const I = tensor.createTensor(D, D); I.upload([1, 0, 0, 1]);
        const X = tensor.createTensor(L, D); X.upload([1, 0, 0, 1]);

        // single-head: identity projections, so O = softmax(X X^T / 1) X per row
        const Q = tensor.createTensor(1, 1), K = tensor.createTensor(1, 1), V = tensor.createTensor(1, 1);
        const A = tensor.createTensor(1, 1), Yp = tensor.createTensor(1, 1), O = tensor.createTensor(1, 1);
        tensor.attentionForward(X, I, I, I, I, null, Q, K, V, A, Yp, O);
        const od = finite(O, "attentionForward O");
        if (O.rows !== L || O.cols !== D || A.rows !== L || A.cols !== L) throw new Error("attentionForward shapes: " + O.rows + "x" + O.cols);
        const ad = A.download();
        if (!same(ad[0] + ad[1], 1) || ad[0] <= ad[1]) throw new Error("attention weights: " + Array.from(ad));
        if (!same(od[0], ad[0]) || !same(od[1], ad[1])) throw new Error("attentionForward O: " + Array.from(od));
        // masking the second key sends all weight to the first
        const mask = tensor.createTensor(L, 1); mask.upload([1, 0]);
        tensor.attentionForward(X, I, I, I, I, mask, Q, K, V, A, Yp, O);
        const om = O.download();
        if (!same(om[0], 1) || !same(om[1], 0)) throw new Error("masked attentionForward: " + Array.from(om));
        // backward runs and fills the gradients (accumulated: caller zeros)
        const dO = tensor.createTensor(L, D); dO.upload([1, 1, 1, 1]);
        const dX = tensor.createTensor(L, D), dWq = tensor.createTensor(D, D), dWk = tensor.createTensor(D, D), dWv = tensor.createTensor(D, D), dWo = tensor.createTensor(D, D);
        tensor.attentionForward(X, I, I, I, I, null, Q, K, V, A, Yp, O);
        tensor.attentionBackward(dO, X, Q, K, V, A, Yp, I, I, I, I, null, dX, dWq, dWk, dWv, dWo);
        const dwo = finite(dWo, "attentionBackward dWo");
        if (dwo[0] === 0 && dwo[1] === 0 && dwo[2] === 0 && dwo[3] === 0) throw new Error("attentionBackward left dWo zero");

        // multi-head (1 head == single-head), the self-attention train wrappers and their backward
        const Qh = tensor.createTensor(1, 1), Kh = tensor.createTensor(1, 1), Vh = tensor.createTensor(1, 1);
        const Ah = tensor.createTensor(1, 1), Yc = tensor.createTensor(1, 1), Om = tensor.createTensor(1, 1);
        tensor.mhaForward(X, I, I, I, I, null, 1, Qh, Kh, Vh, Ah, Yc, Om);
        const omd = finite(Om, "mhaForward O");
        if (!same(omd[0], od[0]) || !same(omd[3], od[3])) throw new Error("mhaForward(1 head) != attentionForward: " + Array.from(omd));
        tensor.mhaBackward(dO, X, Qh, Kh, Vh, Ah, Yc, I, I, I, I, null, 1, dX, dWq, dWk, dWv, dWo);
        finite(dX, "mhaBackward dX");
        const Os = tensor.createTensor(1, 1);
        tensor.selfAttentionForwardTrain(X, I, I, I, I, null, 1, Qh, Kh, Vh, Ah, Yc, Os);
        const osd = finite(Os, "selfAttentionForwardTrain O");
        if (!same(osd[0], od[0])) throw new Error("selfAttentionForwardTrain: " + Array.from(osd));
        tensor.selfAttentionBackward(dO, X, Qh, Kh, Vh, Ah, Yc, I, I, I, I, null, 1, dX, dWq, dWk, dWv, dWo);
        finite(dX, "selfAttentionBackward dX");
        // cross-attention train + backward with Ctx == X
        const Oc = tensor.createTensor(1, 1), dCtx = tensor.createTensor(1, 1);
        tensor.crossAttentionForwardTrain(X, X, I, I, I, I, null, 1, Qh, Kh, Vh, Ah, Yc, Oc);
        const ocd = finite(Oc, "crossAttentionForwardTrain O");
        if (!same(ocd[0], od[0])) throw new Error("crossAttentionForwardTrain: " + Array.from(ocd));
        tensor.crossAttentionBackward(dO, X, X, Qh, Kh, Vh, Ah, Yc, I, I, I, I, null, 1, dX, dCtx, dWq, dWk, dWv, dWo);
        finite(dCtx, "crossAttentionBackward dCtx");
        // T5-style additive bias: a huge bias on key 0 pins the output to V row 0
        const bias = tensor.createTensor(L, L); bias.upload([50, 0, 50, 0]);
        const Ob = tensor.createTensor(1, 1);
        tensor.selfAttentionBiasForward(X, I, I, I, I, null, bias, 1, 1.0, Ob);
        const obd = finite(Ob, "selfAttentionBiasForward O");
        if (!same(obd[0], 1) || !same(obd[1], 0) || !same(obd[2], 1)) throw new Error("selfAttentionBiasForward: " + Array.from(obd));

        // flash attention core on pre-projected Q/K/V: causal row 0 == V row 0.
        // The CUDA kernel is FP16/BF16-only; the CPU one is FP32.
        const Xf = tensor.backend === "cpu" ? X : tensor.createTensor(1, 1);
        if (Xf !== X) tensor.cast(X, Xf, "fp16");
        const Of = tensor.createTensor(1, 1);
        tensor.flashAttentionForward(Xf, Xf, Xf, null, 1, false, Of);
        const ofd = finite(Of, "flashAttentionForward O");
        if (!same(ofd[0], od[0], 1e-2) || !same(ofd[1], od[1], 1e-2)) throw new Error("flashAttentionForward != attentionForward: " + Array.from(ofd) + " vs " + Array.from(od));
        tensor.flashAttentionForward(Xf, Xf, Xf, null, 1, true, Of);
        const ofc = Of.download();
        if (!same(ofc[0], 1, 1e-2) || !same(ofc[1], 0, 1e-2)) throw new Error("causal flashAttentionForward row 0: " + Array.from(ofc));
        tensor.flashAttentionWindowedForward(X, X, X, null, 1, 0, Of);
        const ofw = Of.download();
        if (!same(ofw[0], 1, 1e-2)) throw new Error("flashAttentionWindowedForward: " + Array.from(ofw));
        let threw = false;
        try { tensor.flashAttentionForward(X, X, X, 5, 1, false, Of); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("flashAttentionForward with a bad mask did not throw");
        threw = false;
        try { tensor.resblockForward(null); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("resblockForward(null) did not throw");
        threw = false;
        try { tensor.flashAttentionQkvoBackward({ X: X }); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) throw new Error("flashAttentionQkvoBackward without its required keys did not throw");

        if (tensor.backend !== "cpu") {
            // FP16 inference paths: the projected forms and resblock
            const X16 = tensor.createTensor(1, 1), I16 = tensor.createTensor(1, 1);
            tensor.cast(X, X16, "fp16"); tensor.cast(I, I16, "fp16");
            const O16 = tensor.createTensor(1, 1);
            tensor.selfAttentionForward(X16, I16, I16, I16, I16, null, 1, O16);
            const o16 = finite(O16, "selfAttentionForward O");
            if (!same(o16[0], od[0], 1e-2)) throw new Error("selfAttentionForward fp16: " + Array.from(o16));
            tensor.crossAttentionForward(X16, X16, I16, I16, I16, I16, null, 1, O16);
            finite(O16, "crossAttentionForward O");
            const Aavg = tensor.createTensor(1, 1);
            tensor.crossAttentionForwardWithAttn(X16, X16, I16, I16, I16, I16, null, null, 1, O16, Aavg);
            finite(Aavg, "crossAttentionForwardWithAttn AttnAvg");
            tensor.flashAttentionQkvoForward(X16, null, I16, null, I16, null, I16, null, I16, null, null, 1, false, O16);
            const oq = finite(O16, "flashAttentionQkvoForward O");
            if (!same(oq[0], od[0], 1e-2)) throw new Error("flashAttentionQkvoForward fp16: " + Array.from(oq));
            const Kc = tensor.createTensor(1, 1), Vc = tensor.createTensor(1, 1);
            tensor.flashAttentionProjectKv(X16, I16, null, I16, null, Kc, Vc);
            finite(Kc, "flashAttentionProjectKv K");
            tensor.flashAttentionQWithKvCachedForward(X16, Kc, Vc, I16, null, I16, null, null, 1, false, O16);
            const okv = finite(O16, "flashAttentionQWithKvCachedForward O");
            if (!same(okv[0], od[0], 1e-2)) throw new Error("flashAttentionQWithKvCachedForward: " + Array.from(okv));
            const dQ = tensor.createTensor(1, 1), dK = tensor.createTensor(1, 1), dV = tensor.createTensor(1, 1), dO16 = tensor.createTensor(1, 1);
            tensor.cast(dO, dO16, "fp16");
            tensor.flashAttentionBackward(X16, X16, X16, O16, dO16, null, 1, false, dQ, dK, dV);
            finite(dQ, "flashAttentionBackward dQ");
            const dX16 = tensor.createTensor(1, 1), dWq16 = tensor.createTensor(D, D, "fp16"), dWk16 = tensor.createTensor(D, D, "fp16"), dWv16 = tensor.createTensor(D, D, "fp16"), dWo16 = tensor.createTensor(D, D, "fp16");
            tensor.flashAttentionQkvoBackward({ X: X16, Wq: I16, Wk: I16, Wv: I16, Wo: I16, dO: dO16, numHeads: 1, dX: dX16, dWq: dWq16, dWk: dWk16, dWv: dWv16, dWo: dWo16 });
            finite(dX16, "flashAttentionQkvoBackward dX");
            // resblock: N=1, C_in=C_out=4, 2x2, one group
            const C = 4, H = 2, Wd = 2;
            const rx = tensor.createTensor(1, C * H * Wd); tensor.randn(1, 0, rx);
            const rx16 = tensor.createTensor(1, 1); tensor.cast(rx, rx16, "fp16");
            const ones = tensor.createTensor(C, 1); ones.upload([1, 1, 1, 1]);
            const zeros = tensor.createTensor(C, 1);
            const gm = tensor.createTensor(1, 1), bt = tensor.createTensor(1, 1); tensor.cast(ones, gm, "fp16"); tensor.cast(zeros, bt, "fp16");
            const w = tensor.createTensor(C, C * 9); tensor.randn(2, 0, w); tensor.scaleInplace(w, 0.1);
            const w16 = tensor.createTensor(1, 1); tensor.cast(w, w16, "fp16");
            const ry = tensor.createTensor(1, 1);
            tensor.resblockForward({ X: rx16, gamma1: gm, beta1: bt, W1: w16, gamma2: gm, beta2: bt, W2: w16, Y: ry, N: 1, C_in: C, C_out: C, H: H, W: Wd, numGroups: 1 });
            const ryd = finite(ry, "resblockForward Y");
            if (ry.cols !== C * H * Wd) throw new Error("resblockForward Y shape: " + ry.rows + "x" + ry.cols);
            const dYr = tensor.createTensor(1, 1); tensor.cast(rx, dYr, "fp16");
            const dXr = tensor.createTensor(1, 1), dG1 = tensor.createTensor(C, 1, "fp16"), dB1 = tensor.createTensor(C, 1, "fp16"), dW1 = tensor.createTensor(C, C * 9, "fp16");
            const dG2 = tensor.createTensor(C, 1, "fp16"), dB2 = tensor.createTensor(C, 1, "fp16"), dW2 = tensor.createTensor(C, C * 9, "fp16");
            tensor.resblockBackward({ X: rx16, gamma1: gm, beta1: bt, W1: w16, gamma2: gm, beta2: bt, W2: w16, N: 1, C_in: C, C_out: C, H: H, W: Wd, numGroups: 1,
                                      dY: dYr, dX: dXr, dGamma1: dG1, dBeta1: dB1, dW1: dW1, dGamma2: dG2, dBeta2: dB2, dW2: dW2 });
            finite(dXr, "resblockBackward dX");
        }
        return "OK";
    })();
    )JS");
}

int main() {
    // Unbuffered, so a crash inside a native still shows how far the run got
    // when stdout is a pipe (ctest, a shell capture).
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    brotensor::init();
    brotensor::api::installTensor();

    test_native_handles();
    test_js_api();
    test_js_safetensors();
    test_js_rng();
    test_js_download_and_dtypes();
    test_js_restored_ops();
    test_js_attention();

    // The restored free-function groups keep their checks beside themselves.
    g_failures += run_api_batched_tests();
    g_failures += run_api_attn2_tests();
    g_failures += run_api_audio_tests();
    g_failures += run_api_conv_tests();
    g_failures += run_api_int8_tests();
    g_failures += run_api_misc_tests();
    g_failures += run_api_extra_tests();
    g_failures += run_api_bounds_tests();

    if (g_failures > 0) {
        std::printf("FAILED: %d assertions failed\n", g_failures);
        return 1;
    }
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
