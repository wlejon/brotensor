// ─── Trace JIT: dtype, broadcast and op-coverage parity ──────────────────────
//
// The trace JIT originally emitted one shape of kernel: FP32 in, FP32 out, all
// operands the same full (rows, cols) buffer. None of the DiT/VAE surfaces it
// was meant to serve look like that — Qwen-Image 2.1 runs BF16 activations and
// multiplies them by (1, hidden) modulation rows. This suite is the parity
// gate for the three things that had to be added: typed I/O with FP32 math,
// row/scalar broadcast operands, and the tanh/sigmoid ops.
//
// Every case is checked against the eager brotensor op sequence it replaces.
// The tolerances are dtype-driven: a fused kernel keeps intermediates in FP32
// registers where the eager sequence round-trips them through BF16 memory, so
// the fused result is the more accurate of the two and the comparison has to
// admit one BF16 rounding step (~2^-8 relative).

#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace brotensor;

namespace {

int g_failures = 0;

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    float next_unit() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) / 9007199254740992.0);
    }
    float next_sym(float mag) { return (next_unit() * 2.0f - 1.0f) * mag; }
};

std::vector<float> random_rows(SplitMix64& rng, int rows, int cols, float mag) {
    std::vector<float> v(static_cast<std::size_t>(rows) * cols);
    for (auto& x : v) x = rng.next_sym(mag);
    return v;
}

// Build a device tensor of `dt` from FP32 host data, quantising through the
// dtype so both the JIT and the eager reference see the same bits.
Tensor make(Device dev, const std::vector<float>& host, int rows, int cols, Dtype dt) {
    if (dt == Dtype::FP32) return Tensor::from_host_on(dev, host.data(), rows, cols);
    std::vector<uint16_t> bits(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        bits[i] = (dt == Dtype::BF16) ? fp32_to_bf16_bits(host[i]) : fp32_to_fp16_bits(host[i]);
    }
    return (dt == Dtype::BF16) ? Tensor::from_host_bf16_on(dev, bits.data(), rows, cols)
                               : Tensor::from_host_fp16_on(dev, bits.data(), rows, cols);
}

std::vector<float> read(const Tensor& t) {
    if (t.dtype == Dtype::FP32) return t.to_host_vector();
    std::vector<uint16_t> bits = (t.dtype == Dtype::BF16) ? t.to_host_vector_bf16()
                                                          : t.to_host_vector_fp16();
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        out[i] = (t.dtype == Dtype::BF16) ? bf16_bits_to_fp32(bits[i])
                                          : fp16_bits_to_fp32(bits[i]);
    }
    return out;
}

const char* dtype_name(Dtype d) {
    switch (d) {
        case Dtype::FP32: return "fp32";
        case Dtype::BF16: return "bf16";
        case Dtype::FP16: return "fp16";
        default: return "?";
    }
}

// One BF16 rounding step is 2^-8 = 3.9e-3 relative; FP16 is 2^-11 = 4.9e-4.
// The eager sequence rounds each intermediate back to the storage dtype where
// the fused kernel keeps it in an FP32 register, so the two differ by a small
// number of those steps and the fused answer is the more accurate one — a
// tolerance of ~4 ulp is the right gate. A genuine codegen bug (wrong operand,
// wrong address, wrong activation) shows up as an O(1) error, not 4 ulp.
// FP32 comparisons still have to admit ex2.approx / tanh.approx.
float tol_for(Dtype d) {
    switch (d) {
        case Dtype::BF16: return 1.6e-2f;
        case Dtype::FP16: return 2e-3f;
        default: return 2e-5f;
    }
}

// Relative max error against the magnitude of the reference, so a tolerance
// expressed as "one rounding step" means what it says.
float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    float scale = 1e-6f;
    for (float x : b) scale = std::fmax(scale, std::fabs(x));
    float worst = 0.0f;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst / scale;
}

void check(const std::string& name, const std::vector<float>& got,
           const std::vector<float>& want, float tol) {
    if (got.size() != want.size()) {
        std::printf("  [FAIL] %s: size %zu != %zu\n", name.c_str(), got.size(), want.size());
        ++g_failures;
        return;
    }
    const float e = rel_err(got, want);
    if (!(e <= tol)) {
        std::printf("  [FAIL] %s: rel err %g > %g\n", name.c_str(), e, tol);
        ++g_failures;
    } else {
        std::printf("  [PASS] %s: rel err %g <= %g\n", name.c_str(), e, tol);
    }
}

void check_true(const std::string& name, bool cond, const std::string& detail) {
    if (!cond) {
        std::printf("  [FAIL] %s (%s)\n", name.c_str(), detail.c_str());
        ++g_failures;
    } else {
        std::printf("  [PASS] %s\n", name.c_str());
    }
}

// ── the DiT gated residual: x += gate[1,D] * y ──────────────────────────────

void test_gated_residual(Device dev, Dtype dt) {
    const int N = 512, D = 1024;
    SplitMix64 rng(0x9151);
    const std::vector<float> hx = random_rows(rng, N, D, 1.0f);
    const std::vector<float> hy = random_rows(rng, N, D, 1.0f);
    const std::vector<float> hg = random_rows(rng, 1, D, 1.0f);

    Tensor x = make(dev, hx, N, D, dt);
    Tensor y = make(dev, hy, N, D, dt);
    Tensor g = make(dev, hg, 1, D, dt);

    begin_trace();
    x += g * y;
    TraceHandle h = end_trace();
    sync_all();

    // Eager reference on separate buffers.
    Tensor ex = make(dev, hx, N, D, dt);
    Tensor tmp;
    broadcast_mul(y, g, tmp);
    add_inplace(ex, tmp);
    sync_all();

    const std::string tag = std::string("gated residual x += g*y [") + dtype_name(dt) + "]";
    check(tag, read(x), read(ex), tol_for(dt));
    check_true(tag + " is one launch", h.launch_count() == 1,
               "launches=" + std::to_string(h.launch_count()) + " fusion=" + h.fusion_name());
}

// ── SwiGLU tail: out = silu(gate) * proj ────────────────────────────────────

void test_swiglu(Device dev, Dtype dt) {
    const int N = 256, D = 3072;
    SplitMix64 rng(0x5717);
    const std::vector<float> hg = random_rows(rng, N, D, 2.0f);
    const std::vector<float> hp = random_rows(rng, N, D, 2.0f);

    Tensor g = make(dev, hg, N, D, dt);
    Tensor p = make(dev, hp, N, D, dt);

    begin_trace();
    Tensor out = jit::silu(g) * p;
    TraceHandle h = end_trace();
    sync_all();

    Tensor eg = make(dev, hg, N, D, dt);
    Tensor ep = make(dev, hp, N, D, dt);
    silu_forward(eg, eg);
    mul_inplace(eg, ep);
    sync_all();

    const std::string tag = std::string("swiglu silu(g)*p [") + dtype_name(dt) + "]";
    check(tag, read(out), read(eg), tol_for(dt));
    check_true(tag + " is one launch", h.launch_count() == 1, h.fusion_name());
}

// ── Euler step: x += dsigma * v ─────────────────────────────────────────────

void test_euler_step(Device dev, Dtype dt) {
    const int N = 1024, D = 64;
    SplitMix64 rng(0x2211);
    const std::vector<float> hx = random_rows(rng, N, D, 1.0f);
    const std::vector<float> hv = random_rows(rng, N, D, 1.0f);
    const float d_sigma = -0.0373f;

    Tensor x = make(dev, hx, N, D, dt);
    Tensor v = make(dev, hv, N, D, dt);

    begin_trace();
    x += v * d_sigma;
    TraceHandle h = end_trace();
    sync_all();

    Tensor ex = make(dev, hx, N, D, dt);
    Tensor ev = make(dev, hv, N, D, dt);
    scale_inplace(ev, d_sigma);
    add_inplace(ex, ev);
    sync_all();

    const std::string tag = std::string("euler x += dsigma*v [") + dtype_name(dt) + "]";
    check(tag, read(x), read(ex), tol_for(dt));
    check_true(tag + " is one launch", h.launch_count() == 1, h.fusion_name());
}

// ── tanh / sigmoid ──────────────────────────────────────────────────────────

void test_activations(Device dev, Dtype dt) {
    const int N = 128, D = 512;
    SplitMix64 rng(0x7788);
    const std::vector<float> ha = random_rows(rng, N, D, 3.0f);

    Tensor a = make(dev, ha, N, D, dt);

    begin_trace();
    Tensor t = jit::tanh(a);
    end_trace();
    sync_all();
    Tensor et = Tensor::empty_on(dev, N, D, dt);
    tanh_forward(a, et);
    sync_all();
    check(std::string("tanh [") + dtype_name(dt) + "]", read(t), read(et), tol_for(dt));

    begin_trace();
    Tensor s = jit::sigmoid(a);
    end_trace();
    sync_all();
    Tensor es = Tensor::empty_on(dev, N, D, dt);
    sigmoid_forward(a, es);
    sync_all();
    check(std::string("sigmoid [") + dtype_name(dt) + "]", read(s), read(es), tol_for(dt));
}

// ── a (1,1) scalar operand mixed with (N,D) ─────────────────────────────────

void test_scalar_broadcast(Device dev, Dtype dt) {
    const int N = 300, D = 128;
    SplitMix64 rng(0x1357);
    const std::vector<float> hx = random_rows(rng, N, D, 1.0f);
    const std::vector<float> hs(1, 0.375f);

    Tensor x = make(dev, hx, N, D, dt);
    Tensor s = make(dev, hs, 1, 1, dt);

    begin_trace();
    Tensor out = x * s;
    end_trace();
    sync_all();

    std::vector<float> want(hx.size());
    const float sv = read(s)[0];
    for (std::size_t i = 0; i < hx.size(); ++i) want[i] = read(x)[i] * sv;
    sync_all();

    check(std::string("scalar broadcast x*s [") + dtype_name(dt) + "]", read(out), want,
          tol_for(dt));
}

// ── RMSNorm followed by SiLU: the VAE resnet seam ───────────────────────────

void test_rmsnorm_silu(Device dev, Dtype dt) {
    const int N = 257, D = 512;
    SplitMix64 rng(0xBEEF);
    const std::vector<float> hx = random_rows(rng, N, D, 1.5f);
    std::vector<float> hg = random_rows(rng, 1, D, 0.4f);
    for (auto& v : hg) v += 1.0f;

    Tensor x = make(dev, hx, N, D, dt);
    Tensor gamma = make(dev, hg, 1, D, dt);

    begin_trace();
    Tensor out = jit::silu(jit::rms_norm(x, gamma, 1e-6f));
    TraceHandle h = end_trace();
    sync_all();

    Tensor en = Tensor::empty_on(dev, N, D, dt);
    rms_norm_forward(x, gamma, 1e-6f, en);
    silu_forward(en, en);
    sync_all();

    const std::string tag = std::string("rms_norm+silu [") + dtype_name(dt) + "]";
    check(tag, read(out), read(en), tol_for(dt));
    check_true(tag + " is one launch", h.launch_count() == 1, h.fusion_name());
}

// ── LayerNorm followed by a (1,D) modulation ────────────────────────────────

void test_layernorm_modulate(Device dev, Dtype dt) {
    const int N = 193, D = 768;
    SplitMix64 rng(0xFACE);
    const std::vector<float> hx = random_rows(rng, N, D, 1.0f);
    std::vector<float> hg = random_rows(rng, 1, D, 0.2f);
    for (auto& v : hg) v += 1.0f;
    const std::vector<float> hb = random_rows(rng, 1, D, 0.1f);
    const std::vector<float> hs = random_rows(rng, 1, D, 0.5f);
    const std::vector<float> hsh = random_rows(rng, 1, D, 0.3f);

    Tensor x = make(dev, hx, N, D, dt);
    Tensor gamma = make(dev, hg, 1, D, dt);
    Tensor beta = make(dev, hb, 1, D, dt);
    Tensor scale = make(dev, hs, 1, D, dt);
    Tensor shift = make(dev, hsh, 1, D, dt);

    begin_trace();
    Tensor ln = jit::layernorm(x, gamma, beta, 1e-6f);
    Tensor out = jit::modulate(ln, scale, shift);
    TraceHandle h = end_trace();
    sync_all();

    Tensor eln = Tensor::empty_on(dev, N, D, dt);
    layernorm_forward_inference_batched(x, gamma, beta, eln, 1e-6f);
    Tensor emod;
    modulate(eln, scale, shift, emod);
    sync_all();

    const std::string tag = std::string("layernorm+modulate [") + dtype_name(dt) + "]";
    check(tag, read(out), read(emod), tol_for(dt));
    check_true(tag + " is one launch", h.launch_count() == 1, h.fusion_name());
}

// ── a shape the vector entry cannot take ────────────────────────────────────
//
// (5, 6) FP32 is 30 elements: not a multiple of the 4-wide access, so the
// launcher has to fall back to the scalar entry. The answer must be identical.
void test_scalar_entry_fallback(Device dev) {
    const int N = 5, D = 6;
    SplitMix64 rng(0x4242);
    const std::vector<float> ha = random_rows(rng, N, D, 1.0f);
    const std::vector<float> hb = random_rows(rng, N, D, 1.0f);

    Tensor a = make(dev, ha, N, D, Dtype::FP32);
    Tensor b = make(dev, hb, N, D, Dtype::FP32);

    begin_trace();
    Tensor out = a * b + a;
    TraceHandle h = end_trace();
    sync_all();

    std::vector<float> want(ha.size());
    for (std::size_t i = 0; i < ha.size(); ++i) want[i] = ha[i] * hb[i] + ha[i];

    check("odd-length trace falls back to the scalar entry", read(out), want, 2e-5f);
    check_true("scalar entry reported", std::string(h.fusion_name()) == "elementwise-scalar",
               h.fusion_name());
}

void run_suite(Device dev, const char* label) {
    std::printf("\n=== %s ===\n", label);
    const bool gpu = dev.is_cuda();
    std::vector<Dtype> dtypes{Dtype::FP32};
    if (gpu) {
        dtypes.push_back(Dtype::BF16);
        dtypes.push_back(Dtype::FP16);
    }
    for (Dtype dt : dtypes) {
        test_gated_residual(dev, dt);
        test_swiglu(dev, dt);
        test_euler_step(dev, dt);
        test_activations(dev, dt);
        test_scalar_broadcast(dev, dt);
        test_rmsnorm_silu(dev, dt);
        test_layernorm_modulate(dev, dt);
    }
    test_scalar_entry_fallback(dev);
}

}  // namespace

int main() {
    brotensor::init();

    std::printf("================================================================\n");
    std::printf("  TRACE JIT — dtype / broadcast / op coverage\n");
    std::printf("================================================================\n");

#if !BROTENSOR_HAS_BRASS_JIT
    std::printf("[SKIP] brass JIT not enabled in this build.\n");
    return 0;
#else
    if (brotensor::is_available(Device::cuda())) {
        brotensor::set_default_device(Device::cuda());
        run_suite(Device::cuda(), "CUDA");
    } else {
        std::printf("[SKIP] no CUDA device; the typed and broadcast paths are CUDA-only.\n");
    }

    std::printf("\n================================================================\n");
    if (g_failures == 0) {
        std::printf("  [SUCCESS] all trace JIT dtype/broadcast cases passed\n");
    } else {
        std::printf("  [FAILURE] %d case(s) failed\n", g_failures);
    }
    std::printf("================================================================\n");
    return g_failures == 0 ? 0 : 1;
#endif
}
