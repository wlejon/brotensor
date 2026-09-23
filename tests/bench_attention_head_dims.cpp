// Micro-benchmark: flash_attention_forward (non-causal, FP16 and BF16) across
// the head dims its callers use, at their real shapes:
//
//   hd 16          SAM mask decoder, token<->image (Lk / Lq 4096)
//   hd 32          Sana-class / DC-AE width, self attention
//   hd 40 / 80 / 160   SD1.5 UNet levels (8 heads at 320 / 640 / 1280 ch)
//   hd 64 / 72 / 128   DINOv3 + TripoSplat / PixArt / Flux-class DiTs
//   hd 112         Sana 1.6B cross attention (20 heads over the caption)
//   hd 512         SD VAE mid-block (one head over the whole latent)
//   hd 20          not a multiple of 8
//
// Useful for before/after comparison of the fused FlashAttention-2 kernel's
// head_dim coverage against the per-head fallback. Each row also runs a
// finite spot check so a fast-but-wrong kernel can't pass silently; accuracy
// lives in brotensor_test_vit_block_ops.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_attention_head_dims

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

Tensor upload_rand(int rows, int cols, Dtype dt, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<uint16_t> h(static_cast<size_t>(rows) * cols);
    for (auto& x : h) {
        const float f = d(rng);
        x = dt == Dtype::BF16 ? brotensor::fp32_to_bf16_bits(f) : brotensor::fp32_to_fp16_bits(f);
    }
    return dt == Dtype::BF16 ? Tensor::from_host_bf16_on(Device::CUDA, h.data(), rows, cols)
                             : Tensor::from_host_fp16_on(Device::CUDA, h.data(), rows, cols);
}

bool finite_spot(const Tensor& o) {
    std::vector<uint16_t> h(o.size());
    if (o.dtype == Dtype::BF16) o.copy_to_host_bf16(h.data());
    else o.copy_to_host_fp16(h.data());
    const size_t step = h.size() > 4096 ? h.size() / 4096 : 1;
    for (size_t i = 0; i < h.size(); i += step) {
        const float f = o.dtype == Dtype::BF16 ? brotensor::bf16_bits_to_fp32(h[i])
                                               : brotensor::fp16_bits_to_fp32(h[i]);
        if (!std::isfinite(f) || std::fabs(f) > 10.0f) return false;
    }
    return true;
}

void bench(const char* tag, int Lq, int Lk, int nh, int hd, Dtype dt) {
    std::mt19937 rng(42);
    const int D = nh * hd;
    Tensor Q = upload_rand(Lq, D, dt, rng, 1.0f);
    Tensor K = upload_rand(Lk, D, dt, rng, 1.0f);
    Tensor V = upload_rand(Lk, D, dt, rng, 1.0f);
    Tensor O;
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_forward(Q, K, V, nullptr, nh, /*causal=*/false, O);
    });
    const double flop = 4.0 * double(Lq) * Lk * D;
    std::printf("%-22s %-4s Lq=%5d Lk=%5d nh=%2d hd=%3d  %9.3f ms  %7.1f TFLOP/s  %s\n",
                tag, dt == Dtype::BF16 ? "bf16" : "fp16", Lq, Lk, nh, hd, ms,
                flop / (ms * 1e9), finite_spot(O) ? "ok" : "NOT FINITE");
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    bt_bench::spin_up();
    std::printf("brotensor_bench_attention_head_dims  (warmup %.0f ms/op, best of %d)\n",
                bt_bench::kWarmupMs, bt_bench::kSamples);
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        bench("sam tok->img",     7, 4096,  8,  16, dt);
        bench("sam img->tok",  4096,    7,  8,  16, dt);
        bench("sam hd32 tok->img", 7, 4096, 8,  32, dt);
        bench("hd16 self",     4096, 4096,  8,  16, dt);
        bench("hd32 self",     4096, 4096, 16,  32, dt);
        bench("hd20 self",     4096, 4096,  4,  20, dt);
        bench("sd15 L1 self",  4096, 4096,  8,  40, dt);
        bench("sd15 L2 self",  1024, 1024,  8,  80, dt);
        bench("sd15 L2 cross", 1024,   77,  8,  80, dt);
        bench("sd15 L3 self",   256,  256,  8, 160, dt);
        bench("sd15 L2 self 1k", 4096, 4096, 8,  80, dt);
        bench("sd15 L3 self 1k", 1024, 1024, 8, 160, dt);
        bench("sana1.6b cross", 1024,  300, 20, 112, dt);
        bench("hd112 self",    4096, 4096, 20, 112, dt);
        bench("vae mid",       4096, 4096,  1, 512, dt);
        bench("hd64 self",     4096, 4096, 16,  64, dt);
        bench("hd72 self",     4096, 4096, 16,  72, dt);
        bench("hd128 self",    4115, 4115, 24, 128, dt);
    }
    return 0;
}
