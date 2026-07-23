// Metal/CUDA parity for the quantized batched-linear PREFILL path.
//
// linear_forward_batched_{q4k,q6k,q8_0}_fp16 gained a large-batch fast path:
// for B >= kQuantPrefillMinB it dequantizes the weight once and runs a
// simdgroup GEMM, instead of re-decoding the weight per output token. This
// test validates that fast path against the trusted per-token GEMV path
// (linear_forward_*_fp16, always the fused-decode kernel) on the SAME
// quantized weight bytes — so it checks the new plumbing (strides, bias, batch
// layout, transient dequant scratch) end to end, on whichever GPU backend the
// binary was built with.
//
// The Q*_K/Q8_0 formats have no public host quantizer, so the weight is built
// from pseudo-random block bytes with only the FP16 scale field(s) pinned to a
// small finite constant (keeps decoded values bounded / non-NaN). The exact
// values are irrelevant: both paths decode identical bytes, so they must agree.

#include <brotensor/ops.h>
#include <brotensor/ops/quant.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

// The quantized weight is staged as raw host bytes and pushed into a device
// tensor. Metal's storage is host-addressable so a plain std::memcpy lands,
// but a CUDA Tensor::data is a bare cudaMalloc pointer — writing it from the
// host segfaults. Same shim as test_q4k_parity.cpp.
#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
static inline void cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n);
}
#define cudaMemcpyHostToDevice 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) {                                        \
    std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++g_failures; \
} } while (0)

namespace {

Device g_dev = Device::CPU;

uint16_t f16(float v) { return brotensor::fp32_to_fp16_bits(v); }

// Write a little-endian fp16 scale into a block byte buffer at `off`.
void put_f16(uint8_t* p, int off, float v) {
    const uint16_t b = f16(v);
    p[off] = static_cast<uint8_t>(b & 0xFF);
    p[off + 1] = static_cast<uint8_t>(b >> 8);
}

// Build a (out, in) quantized weight of `dt` with pseudo-random block bytes and
// finite pinned scale field(s). block_bytes/block_elems and the scale-field
// offsets are the GGUF layout for each format.
Tensor make_quant_weight(int out, int in, Dtype dt,
                         int block_bytes, int block_elems) {
    const int bpr = in / block_elems;
    const size_t nblocks = static_cast<size_t>(out) * bpr;
    std::vector<uint8_t> bytes(nblocks * block_bytes);
    uint32_t lcg = 0x1234567u;
    for (auto& b : bytes) { lcg = lcg * 1664525u + 1013904223u; b = uint8_t(lcg >> 24); }
    for (size_t blk = 0; blk < nblocks; ++blk) {
        uint8_t* p = &bytes[blk * block_bytes];
        if (dt == Dtype::Q4_K) {            // d @0, dmin @2
            put_f16(p, 0, 0.02f);
            put_f16(p, 2, 0.008f);
        } else if (dt == Dtype::Q8_0) {     // d @0
            put_f16(p, 0, 0.01f);
        } else {                            // Q6_K: d @ end (208)
            put_f16(p, block_bytes - 2, 0.01f);
        }
    }
    Tensor W = Tensor::empty_on(g_dev, out, in, dt);
    cudaMemcpy(W.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    return W;
}

using GemvFn    = void (*)(const Tensor&, const Tensor*, const Tensor&, Tensor&);
using BatchedFn = void (*)(const Tensor&, const Tensor*, const Tensor&, Tensor&);

void run_case(const char* name, Dtype dt, int block_bytes, int block_elems,
              GemvFn gemv, BatchedFn batched, bool with_bias) {
    const int OUT = 96, IN = block_elems * 3;   // 3 super-blocks per row
    const int B = 16;                            // >= kQuantPrefillMinB (prefill)

    Tensor W = make_quant_weight(OUT, IN, dt, block_bytes, block_elems);

    // Random FP16 activations (B, IN) and optional bias (OUT, 1).
    std::vector<uint16_t> Xh(static_cast<size_t>(B) * IN);
    uint32_t lcg = 0xABCDu;
    auto rnd = [&] { lcg = lcg * 1664525u + 1013904223u;
                     return (float(lcg >> 8) / float(1u << 24)) * 0.6f - 0.3f; };
    for (auto& v : Xh) v = f16(rnd());
    Tensor bias;
    if (with_bias) {
        std::vector<uint16_t> bh(OUT);
        for (auto& v : bh) v = f16(rnd());
        bias = Tensor::from_host_fp16_on(g_dev, bh.data(), OUT, 1);
    }
    const Tensor* biasp = with_bias ? &bias : nullptr;

    // got: batched prefill path (B=16).
    Tensor Xg = Tensor::from_host_fp16_on(g_dev, Xh.data(), B, IN);
    Tensor Yg;
    batched(W, biasp, Xg, Yg);
    CHECK(Yg.rows == B && Yg.cols == OUT && Yg.dtype == Dtype::FP16);
    brotensor::sync_all();
    std::vector<uint16_t> got(static_cast<size_t>(B) * OUT);
    Yg.copy_to_host_fp16(got.data());

    // ref: per-token fused-decode GEMV (always the trusted decode path).
    std::vector<uint16_t> ref(static_cast<size_t>(B) * OUT);
    for (int b = 0; b < B; ++b) {
        Tensor xb = Tensor::from_host_fp16_on(g_dev, &Xh[static_cast<size_t>(b) * IN], IN, 1);
        Tensor yb;
        gemv(W, biasp, xb, yb);
        brotensor::sync_all();
        yb.copy_to_host_fp16(&ref[static_cast<size_t>(b) * OUT]);
    }

    float max_abs = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref[i]);
        const float e = std::fabs(g - r);
        max_abs = std::max(max_abs, e);
        // Same operands, both FP32-accumulated but in different orders.
        if (e > 3e-2f + 5e-2f * std::fabs(r)) ++bad;
    }
    std::printf("  %-6s prefill vs gemv: max_abs=%g bad=%d/%zu\n",
                name, max_abs, bad, got.size());
    CHECK(bad == 0);
}

}  // namespace

int main() {
    brotensor::init();
    if (brotensor::is_available(Device::CUDA))       g_dev = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) g_dev = Device::Metal;
    else { std::printf("no GPU backend available - skipping\n"); return 0; }
    std::printf("test_quant_prefill_parity (device=%s)\n",
                g_dev == Device::CUDA ? "CUDA" : "Metal");

    for (bool bias : {false, true}) {
        run_case("q4k",  Dtype::Q4_K, 144, 256,
                 brotensor::linear_forward_q4k_fp16,
                 brotensor::linear_forward_batched_q4k_fp16, bias);
        run_case("q6k",  Dtype::Q6_K, 210, 256,
                 brotensor::linear_forward_q6k_fp16,
                 brotensor::linear_forward_batched_q6k_fp16, bias);
        run_case("q8_0", Dtype::Q8_0, 34, 32,
                 brotensor::linear_forward_q8_0_fp16,
                 brotensor::linear_forward_batched_q8_0_fp16, bias);
    }

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
