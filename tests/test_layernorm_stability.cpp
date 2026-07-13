// LayerNorm numerical stability on rows whose mean dwarfs their spread.
//
// A CLIP text row sits near 395 with a spread of ~0.007. Computing the variance
// as E[x^2] - E[x]^2 puts both terms at ~1.6e5, where FP32 has no resolution
// left to represent their difference: the result is rounding noise, negative
// about as often as positive, and a negative variance makes rstd NaN — which
// then poisons every downstream token. Summing squared deviations from the mean
// instead keeps every term non-negative, so the variance cannot go negative.
//
// The sweep below walks the offset from 0 up to the edge of FP32's mantissa and
// the spread from coarse to fine, so it covers both the well-conditioned rows
// and the ones where offset/spread is extreme. Two properties are asserted for
// every row: the output is finite, and it matches a double-precision reference
// computed from the same inputs.
//
// Note the reference divides by (var + eps) rather than assuming unit output
// variance. When a row's spread is small enough that eps dominates its variance
// — 0.001 steps here give var ~5e-6 against eps 1e-5 — layernorm is *supposed*
// to shrink it, and a test demanding unit variance would be wrong, not the op.
//
// CPU-only and self-contained: no GPU, no fixtures.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

int main() {
    brotensor::init();

    const int D = 8;
    const float eps = 1e-5f;

    // Each row is `offset + (j + 1) * step`: a large constant plus a small ramp.
    // 395.25 / 0.001 is the CLIP text case that surfaced this; the others sweep
    // either side of it.
    const float offsets[] = {0.0f, 1.0f, 100.0f, 395.25f, 1000.0f, 65504.0f};
    const float steps[]   = {1.0f, 0.1f, 0.001f};

    std::vector<float> gamma_h(static_cast<std::size_t>(D), 1.0f);
    std::vector<float> beta_h(static_cast<std::size_t>(D), 0.0f);
    Tensor gamma = Tensor::from_host_on(Device::CPU, gamma_h.data(), D, 1);
    Tensor beta  = Tensor::from_host_on(Device::CPU, beta_h.data(), D, 1);

    for (float offset : offsets) {
        for (float step : steps) {
            std::vector<float> host(static_cast<std::size_t>(D));
            for (int j = 0; j < D; ++j) {
                host[static_cast<std::size_t>(j)] =
                    offset + static_cast<float>(j + 1) * step;
            }

            Tensor X = Tensor::from_host_on(Device::CPU, host.data(), 1, D);
            Tensor Y;
            brotensor::layernorm_forward_inference_batched(X, gamma, beta, Y, eps);
            const std::vector<float> y = Y.to_host_vector();

            int nonfinite = 0;
            for (float v : y) if (!std::isfinite(v)) ++nonfinite;
            if (nonfinite != 0) {
                std::printf("  offset=%g step=%g: %d/%d non-finite\n",
                            offset, step, nonfinite, D);
            }
            CHECK(nonfinite == 0);
            if (nonfinite != 0) continue;

            // Reference in double, over the same FP32 inputs the kernel saw —
            // so a row whose ramp FP32 cannot represent is degenerate for the
            // reference too, and the comparison stays about the arithmetic.
            double mean = 0.0;
            for (float v : host) mean += static_cast<double>(v);
            mean /= D;
            double var = 0.0;
            for (float v : host) {
                const double d = static_cast<double>(v) - mean;
                var += d * d;
            }
            var /= D;
            const double rstd = 1.0 / std::sqrt(var + static_cast<double>(eps));

            // Accuracy is only a fair question where FP32 can hold the ramp in
            // the first place. At offset 65504 the spacing between neighbouring
            // FP32 values is ~0.0078, so a 0.001 step is already gone by the
            // time the input is stored — no algorithm recovers it, and the
            // finiteness check above is the whole contract for such a row.
            const float ulp = std::nextafter(offset, 1e30f) - offset;
            if (step < 4.0f * ulp) continue;

            double worst = 0.0;
            for (int j = 0; j < D; ++j) {
                const double want =
                    (static_cast<double>(host[static_cast<std::size_t>(j)]) - mean) * rstd;
                const double got = static_cast<double>(y[static_cast<std::size_t>(j)]);
                worst = std::max(worst, std::fabs(got - want));
            }

            // The kernel accumulates in FP32, so its mean carries a rounding
            // error of about |offset| * 2^-24 and rstd amplifies that into the
            // output. 0.05 absorbs it across this sweep while staying tight
            // enough to catch a variance that is actually wrong.
            if (worst > 0.05) {
                std::printf("  offset=%g step=%g: max |err| vs double = %g\n",
                            offset, step, worst);
            }
            CHECK(worst <= 0.05);
        }
    }

    if (g_failures == 0) std::printf("layernorm_stability: OK\n");
    else std::printf("layernorm_stability: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
