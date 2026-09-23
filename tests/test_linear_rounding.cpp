// 16-bit linears round once: Y = round16(act(X·Wᵀ + bias)), the product
// accumulated in FP32, the bias and activation applied in FP32, one narrowing.
//
// The reference is FP64 math over the same 16-bit operands, rounded once to the
// output dtype. A kernel that is right lands on that value except where the
// exact result sits within FP32 accumulation error of a 16-bit rounding
// boundary — a vanishing fraction. A kernel that narrows X·Wᵀ first and adds
// the bias afterwards (two roundings) misses on a large fraction of elements,
// because the bias here is the same size as the product.
//
// Covered: the qkvo projections (flash_attention_project_kv, the helper the
// qkvo forward / backward share), linear_forward_batched_fp16(_act) at every
// batch size its dispatch splits on, and the quantized-weight batched linears
// (Q4_K / Q6_K / Q8_0 prefill). Pass `wmma` to force the WMMA GEMM in place of
// the mma.sync one (BROTENSOR_MMA_GEMM=0, read once at first use).

#include <brotensor/ops.h>
#include <brotensor/ops/quant.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

int g_failures = 0;

struct Rng {
    uint64_t s;
    double unit() {  // [-1, 1)
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) / static_cast<double>(1ull << 52) - 1.0;
    }
};

uint16_t to_bits(double x, Dtype dt) {
    const float f = static_cast<float>(x);
    return dt == Dtype::BF16 ? brotensor::fp32_to_bf16_bits(f) : brotensor::fp32_to_fp16_bits(f);
}
float from_bits(uint16_t b, Dtype dt) {
    return dt == Dtype::BF16 ? brotensor::bf16_bits_to_fp32(b) : brotensor::fp16_bits_to_fp32(b);
}

// A (r, c) matrix of 16-bit values in [-scale, scale): the bits for upload and
// their exact values for the reference.
struct Mat {
    int rows = 0, cols = 0;
    Dtype dt = Dtype::FP16;
    std::vector<uint16_t> bits;
    std::vector<double> val;
    Mat(int r, int c, Dtype d, Rng& rng, double scale) : rows(r), cols(c), dt(d), bits(size_t(r) * c), val(bits.size()) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bits[i] = to_bits(rng.unit() * scale, dt);
            val[i] = from_bits(bits[i], dt);
        }
    }
    Tensor upload() const {
        return dt == Dtype::BF16 ? Tensor::from_host_bf16_on(Device::CUDA, bits.data(), rows, cols)
                                 : Tensor::from_host_fp16_on(Device::CUDA, bits.data(), rows, cols);
    }
};

std::vector<uint16_t> download(const Tensor& t) {
    brotensor::sync_all();
    std::vector<uint16_t> h(static_cast<size_t>(t.rows) * t.cols);
    if (t.dtype == Dtype::BF16) t.copy_to_host_bf16(h.data());
    else t.copy_to_host_fp16(h.data());
    return h;
}

double act_ref(int act, double x) {
    switch (act) {
        case brotensor::kLinearActRelu: return x > 0.0 ? x : 0.0;
        case brotensor::kLinearActGeluExact: return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        default: return x;
    }
}

// Compares got (B, out) against act(X Wᵀ + b) in FP64. `W` holds the exact
// weight values (out, in) row-major.
//
// One rounding puts every element within half a unit in the last place of the
// exact value, plus what FP32 accumulation can move it. That allowance is
// 2·sqrt(K)·2⁻²⁴·Σ|x·w| (+ the FP32 bias add and activation): tens of times the
// accumulation error a K-term FP32 sum actually shows, yet a small fraction of
// a 16-bit unit except where the sum cancels. An element beyond the bound is a
// "miss"; a single-rounding kernel has none. Two roundings miss wherever the
// narrowed product and the bias round the wrong way together.
void check(const std::string& tag, const std::vector<uint16_t>& got, Dtype dt, const std::vector<double>& X,
           const std::vector<double>& W, const std::vector<double>* b, int B, int in, int out, int act = 0) {
    const double u32 = std::ldexp(1.0, -24);
    const int mant = dt == Dtype::BF16 ? 7 : 10, emin = dt == Dtype::BF16 ? -126 : -14;
    size_t miss = 0, differ = 0;
    double worst = 0.0;   // largest |got - exact| beyond the FP32 allowance, in units in the last place
    for (int r = 0; r < B; ++r)
        for (int o = 0; o < out; ++o) {
            double a = 0.0, s = 0.0;
            for (int k = 0; k < in; ++k) {
                const double t = X[size_t(r) * in + k] * W[size_t(o) * in + k];
                a += t;
                s += std::fabs(t);
            }
            if (b) a += (*b)[o];
            const double pre = a;
            a = act_ref(act, a);
            const double allow = (2.0 * std::sqrt(static_cast<double>(in)) * s + 4.0 * std::fabs(pre)) * u32 *
                                 (act ? 1.25 : 1.0);
            const double have = from_bits(got[size_t(r) * out + o], dt);
            if (got[size_t(r) * out + o] != to_bits(a, dt)) ++differ;
            const double mag = std::max(std::fabs(a), std::fabs(have));
            const int e = mag == 0.0 ? emin : std::max(std::ilogb(mag), emin);
            const double ulp = std::ldexp(1.0, e - mant);
            const double excess = (std::fabs(have - a) - allow) / ulp;
            worst = std::max(worst, excess);
            if (excess > 0.5) ++miss;
        }
    const bool ok = miss == 0;
    std::printf("  %-38s != round(exact) %6zu / %-7d  beyond 0.5 ulp + fp32 %6zu  worst %6.2f ulp  %s\n",
                tag.c_str(), differ, B * out, miss, worst, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

const char* dt_name(Dtype dt) { return dt == Dtype::BF16 ? "bf16" : "fp16"; }

// flash_attention_project_kv: K = ctx Wkᵀ + bk, V = ctx Wvᵀ + bv — the
// projection helper every qkvo op runs.
void run_project_kv(Dtype dt, uint64_t seed) {
    Rng rng{seed};
    const int Lk = 300, Dc = 768, D = 320;
    const Mat ctx(Lk, Dc, dt, rng, 2.0);
    const double ws = 1.5 / std::sqrt(static_cast<double>(Dc));
    const Mat Wk(D, Dc, dt, rng, ws), Wv(D, Dc, dt, rng, ws);
    const Mat bk(D, 1, dt, rng, 1.0), bv(D, 1, dt, rng, 1.0);
    const Tensor bkd = bk.upload(), bvd = bv.upload();
    Tensor K, V;
    brotensor::flash_attention_project_kv(ctx.upload(), Wk.upload(), &bkd, Wv.upload(), &bvd, K, V);
    check(std::string(dt_name(dt)) + " project_kv K", download(K), dt, ctx.val, Wk.val, &bk.val, Lk, Dc, D);
    check(std::string(dt_name(dt)) + " project_kv V", download(V), dt, ctx.val, Wv.val, &bv.val, Lk, Dc, D);
}

// linear_forward_batched_fp16(_act) across its dispatch: GEMV (B <= 4), the
// tensor-core GEMM, and N not a multiple of 8 (the naive kernel).
void run_linear(Dtype dt, uint64_t seed) {
    Rng rng{seed};
    struct Case { int B, in, out, act; };
    const Case cases[] = {
        {1, 640, 320, 0},   {3, 640, 320, 0},   {64, 640, 320, 0},
        {257, 768, 640, 0}, {257, 768, 640, brotensor::kLinearActGeluExact},
        {96, 512, 256, brotensor::kLinearActRelu}, {40, 320, 100, 0},
    };
    for (const Case& c : cases) {
        const Mat X(c.B, c.in, dt, rng, 2.0);
        const Mat W(c.out, c.in, dt, rng, 1.5 / std::sqrt(static_cast<double>(c.in)));
        const Mat b(c.out, 1, dt, rng, 1.0);
        const Tensor bd = b.upload();
        Tensor Y;
        brotensor::linear_forward_batched_fp16_act(W.upload(), &bd, X.upload(), c.act, Y);
        char tag[96];
        std::snprintf(tag, sizeof tag, "%s linear B=%d %dx%d act=%d", dt_name(dt), c.B, c.in, c.out, c.act);
        check(tag, download(Y), dt, X.val, W.val, &b.val, c.B, c.in, c.out, c.act);
    }
}

// Quantized weights from pseudo-random GGUF block bytes with the FP16 scale
// fields pinned (the formats have no host quantizer); the reference weight is
// the op's own dequant to FP16, which is exact for these formats.
Tensor make_quant_weight(int out, int in, Dtype dt, int block_bytes, int block_elems, uint32_t seed) {
    const size_t nblocks = static_cast<size_t>(out) * (in / block_elems);
    std::vector<uint8_t> bytes(nblocks * block_bytes);
    uint32_t lcg = seed;
    for (auto& v : bytes) { lcg = lcg * 1664525u + 1013904223u; v = static_cast<uint8_t>(lcg >> 24); }
    auto put = [](uint8_t* p, int off, float v) {
        const uint16_t h = brotensor::fp32_to_fp16_bits(v);
        p[off] = static_cast<uint8_t>(h & 0xFF);
        p[off + 1] = static_cast<uint8_t>(h >> 8);
    };
    for (size_t blk = 0; blk < nblocks; ++blk) {
        uint8_t* p = &bytes[blk * block_bytes];
        if (dt == Dtype::Q4_K) { put(p, 0, 0.004f); put(p, 2, 0.002f); }
        else if (dt == Dtype::Q8_0) put(p, 0, 0.0015f);
        else put(p, block_bytes - 2, 0.002f);   // Q6_K: d at the block's end
    }
    Tensor W = Tensor::empty_on(Device::CUDA, out, in, dt);
    cudaMemcpy(W.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    return W;
}

using QuantFn = void (*)(const Tensor&, const Tensor*, const Tensor&, Tensor&);
using DequantFn = void (*)(const Tensor&, Tensor&);

void run_quant(const char* name, Dtype qt, int block_bytes, int block_elems, QuantFn fn, DequantFn dq,
               uint64_t seed) {
    Rng rng{seed};
    const int B = 64, in = 768, out = 320;
    Tensor Wq = make_quant_weight(out, in, qt, block_bytes, block_elems, static_cast<uint32_t>(seed));
    Tensor W16;
    dq(Wq, W16);
    const std::vector<uint16_t> wb = download(W16);
    std::vector<double> W(wb.size());
    for (size_t i = 0; i < wb.size(); ++i) W[i] = brotensor::fp16_bits_to_fp32(wb[i]);
    const Mat X(B, in, Dtype::FP16, rng, 1.0);
    const Mat b(out, 1, Dtype::FP16, rng, 1.0);
    const Tensor bd = b.upload();
    Tensor Y;
    fn(Wq, &bd, X.upload(), Y);
    check(std::string(name) + " batched B=64 768x320", download(Y), Dtype::FP16, X.val, W, &b.val, B, in, out);
}

}  // namespace

int main(int argc, char** argv) {
    const bool wmma = argc > 1 && std::string(argv[1]) == "wmma";
    if (wmma) {
#ifdef _WIN32
        _putenv_s("BROTENSOR_MMA_GEMM", "0");
#else
        setenv("BROTENSOR_MMA_GEMM", "0", 1);
#endif
    }
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_linear_rounding (%s GEMM)\n", wmma ? "WMMA" : "default");
    uint64_t seed = 101;
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        run_project_kv(dt, seed++);
        run_linear(dt, seed++);
    }
    run_quant("q4k", Dtype::Q4_K, 144, 256, brotensor::linear_forward_batched_q4k_fp16, brotensor::dequant_q4k_to_fp16,
              seed++);
    run_quant("q6k", Dtype::Q6_K, 210, 256, brotensor::linear_forward_batched_q6k_fp16, brotensor::dequant_q6k_to_fp16,
              seed++);
    run_quant("q8_0", Dtype::Q8_0, 34, 32, brotensor::linear_forward_batched_q8_0_fp16,
              brotensor::dequant_q8_0_to_fp16, seed++);
    if (g_failures) {
        std::printf("test_linear_rounding: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_linear_rounding: OK\n");
    return 0;
}
