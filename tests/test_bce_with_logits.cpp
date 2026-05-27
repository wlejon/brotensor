// Standalone CPU-only unit test for bce_with_logits_fused_batched.
// Hand-verifies a few known values and checks the masking + pos_weight paths.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(float a, float b, float abs_eps, float rel_eps) {
    const float d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const float m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, ctx)                                     \
    do {                                                                       \
        const float _a = (actual);                                             \
        const float _e = (expected);                                           \
        if (!near_(_a, _e, 1e-6f, 1e-5f)) {                                    \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",    \
                        __FILE__, __LINE__, (ctx), _a, _e);                    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static Tensor cpu_mat(int r, int c) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

// z=0, y=0 -> loss = ln(2), dLogits = 0.5, probs = 0.5
// z=0, y=1 -> loss = ln(2), dLogits = -0.5, probs = 0.5
static void test_zero_logit() {
    const int B = 2, L = 1;
    Tensor logits = cpu_mat(B, L);
    Tensor target = cpu_mat(B, L);
    logits[0] = 0.0f; logits[1] = 0.0f;
    target[0] = 0.0f; target[1] = 1.0f;

    Tensor probs = cpu_mat(B, L);
    Tensor dLogits = cpu_mat(B, L);
    Tensor loss = cpu_mat(B, 1);
    brotensor::bce_with_logits_fused_batched(
        logits, target, nullptr, 1.0f, probs, dLogits, loss);

    const float ln2 = std::log(2.0f);
    EXPECT_NEAR(probs[0], 0.5f, "z=0,y=0 probs");
    EXPECT_NEAR(probs[1], 0.5f, "z=0,y=1 probs");
    EXPECT_NEAR(dLogits[0], 0.5f, "z=0,y=0 dLogits");
    EXPECT_NEAR(dLogits[1], -0.5f, "z=0,y=1 dLogits");
    EXPECT_NEAR(loss[0], ln2, "z=0,y=0 loss");
    EXPECT_NEAR(loss[1], ln2, "z=0,y=1 loss");
}

// Mask zeros out contributions on the masked element.
static void test_mask() {
    const int B = 1, L = 2;
    Tensor logits = cpu_mat(B, L);
    Tensor target = cpu_mat(B, L);
    logits[0] = 0.0f; logits[1] = 5.0f;  // 2nd element extreme
    target[0] = 0.0f; target[1] = 1.0f;
    std::vector<float> mask = {1.0f, 0.0f};

    Tensor probs = cpu_mat(B, L);
    Tensor dLogits = cpu_mat(B, L);
    Tensor loss = cpu_mat(B, 1);
    brotensor::bce_with_logits_fused_batched(
        logits, target, mask.data(), 1.0f, probs, dLogits, loss);

    const float ln2 = std::log(2.0f);
    EXPECT_NEAR(probs[1], 0.0f, "masked probs");
    EXPECT_NEAR(dLogits[1], 0.0f, "masked dLogits");
    EXPECT_NEAR(loss[0], ln2, "masked-only loss = ln2 from unmasked entry");
    EXPECT_NEAR(dLogits[0], 0.5f, "unmasked dLogits");
}

// pos_weight scales y=1 loss/grad terms.
// z=0, y=1, w=3 -> loss = 3*ln(2), dLogits = 0.5*(3) - 3 = -1.5
static void test_pos_weight() {
    const int B = 1, L = 1;
    Tensor logits = cpu_mat(B, L);
    Tensor target = cpu_mat(B, L);
    logits[0] = 0.0f;
    target[0] = 1.0f;

    Tensor probs = cpu_mat(B, L);
    Tensor dLogits = cpu_mat(B, L);
    Tensor loss = cpu_mat(B, 1);
    brotensor::bce_with_logits_fused_batched(
        logits, target, nullptr, 3.0f, probs, dLogits, loss);

    EXPECT_NEAR(probs[0], 0.5f, "pw probs");
    EXPECT_NEAR(loss[0], 3.0f * std::log(2.0f), "pw loss");
    EXPECT_NEAR(dLogits[0], -1.5f, "pw dLogits");
}

int main() {
    brotensor::init();
    std::printf("test_bce_with_logits\n");
    test_zero_logit();
    test_mask();
    test_pos_weight();
    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll bce_with_logits checks passed.\n");
    return 0;
}
