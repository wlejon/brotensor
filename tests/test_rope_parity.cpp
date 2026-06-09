// CPU↔GPU parity tests for brotensor::rope_forward and rope_backward.
//
// CHUNK 2. X / Y layout is (L, num_heads * head_dim). Tests cover seq_offset
// != 0, multiple heads, and a non-default theta_base.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstring>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

// Round an FP32 host tensor to FP16 and place it on the GPU backend.
Tensor to_fp16_gpu(const Tensor& f32cpu) {
    Tensor h = Tensor::zeros_on(Device::CPU, f32cpu.rows, f32cpu.cols,
                                brotensor::Dtype::FP16);
    const float* s = f32cpu.host_f32();
    uint16_t* d = h.host_fp16_mut();
    for (int i = 0; i < f32cpu.size(); ++i)
        d[i] = brotensor::fp32_to_fp16_bits(s[i]);
    return h.to(gpu_device());
}

void run_fwd(int L, int num_heads, int head_dim, int seq_offset,
             float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::rope_forward(X, head_dim, num_heads, seq_offset, theta_base, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::rope_forward(gX, head_dim, num_heads, seq_offset, theta_base, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "rope_fwd", 1e-4f, 1e-3f);
}

void run_bwd(int L, int num_heads, int head_dim, int seq_offset,
             float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::rope_backward(dY, head_dim, num_heads, seq_offset, theta_base, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::rope_backward(gdY, head_dim, num_heads, seq_offset, theta_base, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "rope_bwd", 1e-4f, 1e-3f);
}

// ─── rope_apply: explicit cos/sin tables ───────────────────────────────────

// Build (L, head_dim/2) cos/sin tables from the standard RoPE angle formula
// (pos = row + seq_offset). FP32, Device::CPU.
void build_rope_tables(int L, int head_dim, int seq_offset, float theta_base,
                       Tensor& cos_tbl, Tensor& sin_tbl) {
    const int half = head_dim / 2;
    cos_tbl = Tensor::mat(L, half);
    sin_tbl = Tensor::mat(L, half);
    for (int row = 0; row < L; ++row) {
        const int pos = row + seq_offset;
        for (int i = 0; i < half; ++i) {
            const float freq = std::exp(-static_cast<float>(2 * i) /
                                        static_cast<float>(head_dim) *
                                        std::log(theta_base));
            const float angle = static_cast<float>(pos) * freq;
            cos_tbl.ptr()[row * half + i] = std::cos(angle);
            sin_tbl.ptr()[row * half + i] = std::sin(angle);
        }
    }
}

void run_apply_fwd(int L, int num_heads, int head_dim, int seq_offset,
                   float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);
    Tensor cos_tbl, sin_tbl;
    build_rope_tables(L, head_dim, seq_offset, theta_base, cos_tbl, sin_tbl);

    Tensor cpu_Y;
    brotensor::rope_apply(X, cos_tbl, sin_tbl, head_dim, num_heads, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gC = cos_tbl.to(gpu_device());
    Tensor gS = sin_tbl.to(gpu_device());
    Tensor gpu_Y;
    brotensor::rope_apply(gX, gC, gS, head_dim, num_heads, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "rope_apply_fwd", 1e-4f, 1e-3f);

    // rope_apply with standard-formula tables must match rope_forward exactly.
    Tensor ref_Y;
    brotensor::rope_forward(X, head_dim, num_heads, seq_offset, theta_base, ref_Y);
    compare_tensors(ref_Y, cpu_Y, "rope_apply_vs_rope_forward", 1e-5f, 1e-4f);
}

void run_apply_bwd(int L, int num_heads, int head_dim, int seq_offset,
                   float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(dY, rng);
    Tensor cos_tbl, sin_tbl;
    build_rope_tables(L, head_dim, seq_offset, theta_base, cos_tbl, sin_tbl);

    Tensor cpu_dX;
    brotensor::rope_apply_backward(dY, cos_tbl, sin_tbl, head_dim, num_heads, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gC  = cos_tbl.to(gpu_device());
    Tensor gS  = sin_tbl.to(gpu_device());
    Tensor gpu_dX;
    brotensor::rope_apply_backward(gdY, gC, gS, head_dim, num_heads, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "rope_apply_bwd", 1e-4f, 1e-3f);
}

void run_apply_fwd_bf16(int L, int num_heads, int head_dim, int seq_offset,
                        float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);
    Tensor cos_tbl, sin_tbl;
    build_rope_tables(L, head_dim, seq_offset, theta_base, cos_tbl, sin_tbl);

    Tensor cpu_Y;
    brotensor::rope_apply(X, cos_tbl, sin_tbl, head_dim, num_heads, cpu_Y);

    // BF16 X; cos/sin tables stay FP32 on the GPU.
    Tensor gX = to_bf16_gpu(X);
    Tensor gC = cos_tbl.to(gpu_device());
    Tensor gS = sin_tbl.to(gpu_device());
    Tensor gpu_Y;
    brotensor::rope_apply(gX, gC, gS, head_dim, num_heads, gpu_Y);
    brotensor::sync_all();

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)),
                    "rope_apply_bf16_fwd", 3e-2f, 3e-2f);
}

// An offset *view* into a longer cos/sin table must produce byte-identical
// output to a freshly-built table for the same positions — for both FP32 and
// FP16 X. The AR-decode hot path (Qwen3-TTS Code Predictor) feeds rope_apply a
// view(rope_cos.data + pos_start*half, ...) rather than rebuilding the table
// per step; this confirms there is no "viewed-table edge" in the FP16 rope
// kernel. The kernel reads cos_tbl[row*half+i] from whatever float* it is
// handed and the tables are FP32 on every dtype path, so the view at
// data + pos_start*half carries the same angles as a fresh build for
// [pos_start, pos_start+n) — the outputs must match bit-for-bit.
void run_apply_view_parity(int n, int num_heads, int head_dim, int pos_start,
                           int P, float theta_base, uint64_t seed) {
    BT_CHECK(pos_start + n <= P);
    const int half = head_dim / 2;
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(n, num_heads * head_dim);
    fill_random(X, rng);

    Tensor longC, longS, freshC, freshS;
    build_rope_tables(P, head_dim, 0,         theta_base, longC,  longS);
    build_rope_tables(n, head_dim, pos_start, theta_base, freshC, freshS);

    Tensor gLongC  = longC.to(gpu_device());
    Tensor gLongS  = longS.to(gpu_device());
    Tensor gFreshC = freshC.to(gpu_device());
    Tensor gFreshS = freshS.to(gpu_device());
    Tensor vC = Tensor::view(gpu_device(),
        static_cast<float*>(gLongC.data) + static_cast<size_t>(pos_start) * half,
        n, half, brotensor::Dtype::FP32);
    Tensor vS = Tensor::view(gpu_device(),
        static_cast<float*>(gLongS.data) + static_cast<size_t>(pos_start) * half,
        n, half, brotensor::Dtype::FP32);

    auto run_pair = [&](const Tensor& gX) {
        Tensor Yfresh, Yview;
        brotensor::rope_apply(gX, gFreshC, gFreshS, head_dim, num_heads, Yfresh);
        brotensor::rope_apply(gX, vC,      vS,      head_dim, num_heads, Yview);
        brotensor::sync_all();
        Tensor hf = Yfresh.to(Device::CPU);
        Tensor hv = Yview.to(Device::CPU);
        BT_CHECK(hf.bytes() == hv.bytes());
        BT_CHECK(std::memcmp(hf.host_raw(), hv.host_raw(), hf.bytes()) == 0);
    };

    run_pair(X.to(gpu_device()));  // FP32 X
    run_pair(to_fp16_gpu(X));      // FP16 X — the case flagged for the CP
}

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(rope_fwd_1h_off0)    { run_fwd(8, 1, 16, 0, 10000.0f, 0x6000ull); }
BT_PARITY_TEST(rope_fwd_4h_off0)    { run_fwd(12, 4, 8, 0, 10000.0f, 0x6001ull); }
BT_PARITY_TEST(rope_fwd_offset)     { run_fwd(6, 2, 16, 37, 10000.0f, 0x6002ull); }
BT_PARITY_TEST(rope_fwd_small_dim)  { run_fwd(5, 3, 2, 11, 10000.0f, 0x6003ull); }
BT_PARITY_TEST(rope_fwd_theta500)   { run_fwd(7, 2, 8, 4, 500.0f, 0x6004ull); }

// ─── backward ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(rope_bwd_1h_off0)    { run_bwd(8, 1, 16, 0, 10000.0f, 0x6010ull); }
BT_PARITY_TEST(rope_bwd_4h_off0)    { run_bwd(12, 4, 8, 0, 10000.0f, 0x6011ull); }
BT_PARITY_TEST(rope_bwd_offset)     { run_bwd(6, 2, 16, 37, 10000.0f, 0x6012ull); }
BT_PARITY_TEST(rope_bwd_theta500)   { run_bwd(7, 2, 8, 4, 500.0f, 0x6013ull); }

// ─── rope_apply (explicit cos/sin tables) ──────────────────────────────────
BT_PARITY_TEST(rope_apply_fwd_1h)     { run_apply_fwd(8, 1, 16, 0, 10000.0f, 0x6020ull); }
BT_PARITY_TEST(rope_apply_fwd_4h)     { run_apply_fwd(12, 4, 8, 0, 10000.0f, 0x6021ull); }
BT_PARITY_TEST(rope_apply_fwd_offset) { run_apply_fwd(6, 2, 16, 37, 10000.0f, 0x6022ull); }
BT_PARITY_TEST(rope_apply_fwd_theta)  { run_apply_fwd(7, 3, 8, 4, 500.0f, 0x6023ull); }
BT_PARITY_TEST(rope_apply_bwd_4h)     { run_apply_bwd(12, 4, 8, 0, 10000.0f, 0x6030ull); }
BT_PARITY_TEST(rope_apply_bwd_offset) { run_apply_bwd(6, 2, 16, 37, 10000.0f, 0x6031ull); }
BT_PARITY_TEST(rope_apply_bf16_1h)    { run_apply_fwd_bf16(8, 1, 16, 0, 10000.0f, 0x6040ull); }
BT_PARITY_TEST(rope_apply_bf16_4h)    { run_apply_fwd_bf16(12, 4, 8, 5, 10000.0f, 0x6041ull); }

// ─── offset-view vs fresh-table parity (Code Predictor decode path) ─────────
BT_PARITY_TEST(rope_apply_view_decode)  { run_apply_view_parity(1, 4, 64, 3, 16, 1000000.0f, 0x6050ull); }
BT_PARITY_TEST(rope_apply_view_prefill) { run_apply_view_parity(2, 2, 128, 1, 16, 1000000.0f, 0x6051ull); }
BT_PARITY_TEST(rope_apply_view_mid)     { run_apply_view_parity(3, 3, 64, 5, 16, 10000.0f, 0x6052ull); }

int main() { return run_all("rope cpu/gpu parity"); }
