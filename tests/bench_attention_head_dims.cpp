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
// The qkvo rows time flash_attention_qkvo_forward / _backward (the four
// projections around the core) at SD1.5 and DiT shapes, with and without
// projection biases; `qkvo` as the argument runs only those.
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
#include <string>
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

// flash_attention_backward at the same shapes (dQ, dK, dV from Q, K, V, dO).
void bench_bwd(const char* tag, int Lq, int Lk, int nh, int hd, Dtype dt, bool causal = false) {
    std::mt19937 rng(43);
    const int D = nh * hd;
    Tensor Q = upload_rand(Lq, D, dt, rng, 1.0f);
    Tensor K = upload_rand(Lk, D, dt, rng, 1.0f);
    Tensor V = upload_rand(Lk, D, dt, rng, 1.0f);
    Tensor dO = upload_rand(Lq, D, dt, rng, 1.0f);
    Tensor dQ, dK, dV;
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_backward(Q, K, V, dO, dO, nullptr, nh, causal, dQ, dK, dV);
    });
    const double flop = 10.0 * double(Lq) * Lk * D;   // S, dP, dV, dQ, dK GEMMs
    std::printf("bwd %-18s %-4s Lq=%5d Lk=%5d nh=%2d hd=%3d%s %9.3f ms  %7.1f TFLOP/s  %s\n", tag,
                dt == Dtype::BF16 ? "bf16" : "fp16", Lq, Lk, nh, hd, causal ? " causal" : "       ", ms,
                flop / (ms * 1e9), finite_spot(dQ) && finite_spot(dK) && finite_spot(dV) ? "ok" : "NOT FINITE");
}

// flash_attention_varlen_backward: `nseq` sequences of `len` tokens.
void bench_varlen_bwd(const char* tag, int nseq, int len, int nh, int hd, Dtype dt, bool causal) {
    std::mt19937 rng(44);
    const int D = nh * hd, T = nseq * len;
    Tensor Q = upload_rand(T, D, dt, rng, 1.0f);
    Tensor K = upload_rand(T, D, dt, rng, 1.0f);
    Tensor V = upload_rand(T, D, dt, rng, 1.0f);
    Tensor dO = upload_rand(T, D, dt, rng, 1.0f);
    Tensor cuh = Tensor::zeros_on(Device::CPU, nseq + 1, 1, Dtype::INT32);
    for (int b = 0; b <= nseq; ++b) static_cast<int32_t*>(cuh.data)[b] = b * len;
    Tensor cud = cuh.to(Device::CUDA);
    const int32_t* c = static_cast<const int32_t*>(cud.data);
    Tensor dQ, dK, dV;
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_varlen_backward(Q, K, V, dO, dO, c, c, nseq, len, len, nh, hd, causal, dQ, dK,
                                                   dV);
    });
    const double flop = 10.0 * double(nseq) * len * len * D;
    std::printf("varlen bwd %-11s %-4s %2d x %5d  nh=%2d hd=%3d%s %9.3f ms  %7.1f TFLOP/s  %s\n", tag,
                dt == Dtype::BF16 ? "bf16" : "fp16", nseq, len, nh, hd, causal ? " causal" : "       ", ms,
                flop / (ms * 1e9), finite_spot(dQ) && finite_spot(dK) && finite_spot(dV) ? "ok" : "NOT FINITE");
}

// flash_attention_packed_qkv_backward: `nseq` sequences of `len` rows, fused QKV.
void bench_packed_bwd(const char* tag, int nseq, int len, int nh, int hd, Dtype dt, int window) {
    std::mt19937 rng(46);
    const int D = nh * hd, L = nseq * len;
    Tensor QKV = upload_rand(L, 3 * D, dt, rng, 1.0f);
    Tensor dO = upload_rand(L, D, dt, rng, 1.0f);
    Tensor bh = Tensor::zeros_on(Device::CPU, L, 2, Dtype::INT32);
    for (int r = 0; r < L; ++r) {
        static_cast<int32_t*>(bh.data)[2 * r] = r / len * len;
        static_cast<int32_t*>(bh.data)[2 * r + 1] = r / len * len + len;
    }
    Tensor bd = bh.to(Device::CUDA), dQKV;
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_packed_qkv_backward(QKV, dO, bd, nh, window, dQKV);
    });
    const double flop = 10.0 * double(nseq) * len * len * D;
    std::printf("packed bwd %-11s %-4s %2d x %5d  nh=%2d hd=%3d w=%3d %9.3f ms  %7.1f TFLOP/s  %s\n", tag,
                dt == Dtype::BF16 ? "bf16" : "fp16", nseq, len, nh, hd, window, ms, flop / (ms * 1e9),
                finite_spot(dQKV) ? "ok" : "NOT FINITE");
}

// The operands of one flash_attention_qkvo_* call. `biased` gives all four
// projections a bias (DiT-style); otherwise every bias is null.
struct QkvoOperands {
    Tensor X, C, Wq, Wk, Wv, Wo, bq, bk, bv, bo, dO;
    bool cross = false, biased = false;
    QkvoOperands(int Lq, int D, int Lk, int Dctx, bool cross_, bool biased_, Dtype dt, std::mt19937& rng)
        : cross(cross_), biased(biased_) {
        X = upload_rand(Lq, D, dt, rng, 1.0f);
        C = cross ? upload_rand(Lk, Dctx, dt, rng, 1.0f) : X;
        const float ws = 1.0f / std::sqrt(float(cross ? Dctx : D));
        Wq = upload_rand(D, D, dt, rng, 2.0f / std::sqrt(float(D)));
        Wk = upload_rand(D, cross ? Dctx : D, dt, rng, 2.0f * ws);
        Wv = upload_rand(D, cross ? Dctx : D, dt, rng, ws);
        Wo = upload_rand(D, D, dt, rng, 1.0f / std::sqrt(float(D)));
        if (biased) {
            bq = upload_rand(D, 1, dt, rng, 0.2f);
            bk = upload_rand(D, 1, dt, rng, 0.2f);
            bv = upload_rand(D, 1, dt, rng, 0.2f);
            bo = upload_rand(D, 1, dt, rng, 0.2f);
        }
        dO = upload_rand(Lq, D, dt, rng, 1.0f);
    }
    const Tensor* ctx() const { return cross ? &C : nullptr; }
    const Tensor* b(const Tensor& t) const { return biased ? &t : nullptr; }
};

// flash_attention_qkvo_forward: four projections around the attention core.
void bench_qkvo_fwd(const char* tag, int Lq, int D, int nh, int Lk, int Dctx, bool cross, bool biased, Dtype dt) {
    std::mt19937 rng(47);
    const QkvoOperands op(Lq, D, Lk, Dctx, cross, biased, dt, rng);
    Tensor O;
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_qkvo_forward(op.X, op.ctx(), op.Wq, op.b(op.bq), op.Wk, op.b(op.bk), op.Wv,
                                                op.b(op.bv), op.Wo, op.b(op.bo), nullptr, nh, false, O);
    });
    std::printf("qkvo fwd %-13s %-4s Lq=%5d Lk=%5d D=%4d nh=%2d hd=%3d%s %9.3f ms  %s\n", tag,
                dt == Dtype::BF16 ? "bf16" : "fp16", Lq, cross ? Lk : Lq, D, nh, D / nh, biased ? " bias" : "     ",
                ms, finite_spot(O) ? "ok" : "NOT FINITE");
}

// flash_attention_qkvo_backward: projections + attention core + projection grads.
void bench_qkvo_bwd(const char* tag, int Lq, int D, int nh, int Lk, int Dctx, bool cross, Dtype dt,
                    bool biased = false) {
    std::mt19937 rng(45);
    const QkvoOperands op(Lq, D, Lk, Dctx, cross, biased, dt, rng);
    const int Dk = cross ? Dctx : D;
    Tensor dX, dCtx;
    Tensor dWq = Tensor::zeros_on(Device::CUDA, D, D, dt), dWk = Tensor::zeros_on(Device::CUDA, D, Dk, dt);
    Tensor dWv = Tensor::zeros_on(Device::CUDA, D, Dk, dt), dWo = Tensor::zeros_on(Device::CUDA, D, D, dt);
    Tensor dbq = Tensor::zeros_on(Device::CUDA, D, 1, dt), dbk = Tensor::zeros_on(Device::CUDA, D, 1, dt);
    Tensor dbv = Tensor::zeros_on(Device::CUDA, D, 1, dt), dbo = Tensor::zeros_on(Device::CUDA, D, 1, dt);
    auto db = [&](Tensor& t) { return biased ? &t : nullptr; };
    const float ms = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_qkvo_backward(op.X, op.ctx(), op.Wq, op.b(op.bq), op.Wk, op.b(op.bk), op.Wv,
                                                 op.b(op.bv), op.Wo, op.b(op.bo), nullptr, nh, false, op.dO, dX,
                                                 cross ? &dCtx : nullptr, dWq, db(dbq), dWk, db(dbk), dWv,
                                                 db(dbv), dWo, db(dbo));
    });
    std::printf("qkvo bwd %-13s %-4s Lq=%5d Lk=%5d D=%4d nh=%2d hd=%3d%s %9.3f ms  %s\n", tag,
                dt == Dtype::BF16 ? "bf16" : "fp16", Lq, cross ? Lk : Lq, D, nh, D / nh, biased ? " bias" : "     ",
                ms, finite_spot(dX) ? "ok" : "NOT FINITE");
}

// The qkvo rows: SD1.5 levels (projection biases only on the output, as in
// the UNet; modelled here with and without) and a DiT block (all biased).
void bench_qkvo(bool fwd, Dtype dt) {
    struct Row { const char* tag; int Lq, D, nh, Lk, Dctx; bool cross; };
    const Row rows[] = {
        {"sd15 L1 self",  4096, 320, 8, 4096, 320, false},
        {"sd15 L2 cross", 1024, 640, 8,   77, 768, true},
        {"dit hd64 self", 1024, 1024, 16, 1024, 1024, false},
    };
    for (const bool biased : {false, true})
        for (const Row& r : rows) {
            if (fwd) bench_qkvo_fwd(r.tag, r.Lq, r.D, r.nh, r.Lk, r.Dctx, r.cross, biased, dt);
            else bench_qkvo_bwd(r.tag, r.Lq, r.D, r.nh, r.Lk, r.Dctx, r.cross, dt, biased);
        }
}

}  // namespace

int main(int argc, char** argv) {
    // Optional argument: `fwd` or `bwd` runs only that half; `qkvo` runs only
    // the flash_attention_qkvo_forward / _backward rows.
    const std::string only = argc > 1 ? argv[1] : "";
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    bt_bench::spin_up();
    std::printf("brotensor_bench_attention_head_dims  (warmup %.0f ms/op, best of %d)\n",
                bt_bench::kWarmupMs, bt_bench::kSamples);
    if (only == "qkvo") {
        for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) bench_qkvo(/*fwd=*/true, dt);
        for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) bench_qkvo(/*fwd=*/false, dt);
        return 0;
    }
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        if (only == "fwd") break;
        bench_bwd("sam tok->img",     7, 4096,  8,  16, dt);
        bench_bwd("hd16 self",     1024, 1024,  8,  16, dt);
        bench_bwd("hd32 self",     4096, 4096, 16,  32, dt);
        bench_bwd("hd20 self",     4096, 4096,  4,  20, dt);
        bench_bwd("sd15 L1 self",  4096, 4096,  8,  40, dt);
        bench_bwd("sd15 L1 cross", 4096,   77,  8,  40, dt);
        bench_bwd("sd15 L2 self",  1024, 1024,  8,  80, dt);
        bench_bwd("sd15 L3 self",   256,  256,  8, 160, dt);
        bench_bwd("hd64 self",     4096, 4096, 16,  64, dt);
        bench_bwd("hd64 causal",   2048, 2048, 16,  64, dt, true);
        bench_bwd("hd128 self",    2048, 2048, 16, 128, dt);
        bench_varlen_bwd("bert",      8,  512, 12, 64, dt, false);
        bench_varlen_bwd("causal",    4, 1024, 16, 64, dt, true);
        bench_varlen_bwd("short",    64,   48,  8, 32, dt, false);
        bench_packed_bwd("bert",      8,  512, 12, 64, dt, 0);
        bench_packed_bwd("short",    64,   48,  8, 32, dt, 0);
        bench_packed_bwd("l3",        1,  256,  8, 160, dt, 0);
        bench_qkvo(/*fwd=*/false, dt);
    }
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        if (only == "bwd") break;
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
        bench_qkvo(/*fwd=*/true, dt);
    }
    return 0;
}
