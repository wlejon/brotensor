#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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

int main() {
    brotensor::init();
    brotensor::api::installTensor();

    test_native_handles();
    test_js_api();

    if (g_failures > 0) {
        std::printf("FAILED: %d assertions failed\n", g_failures);
        return 1;
    }
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
