// CPU↔GPU parity tests for brotensor::rope_forward and rope_backward.
//
// CHUNK 2. X / Y layout is (L, num_heads * head_dim). Tests cover seq_offset
// != 0, multiple heads, and a non-default theta_base.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

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
    Tensor gX = to_bf16_cuda(X);
    Tensor gC = cos_tbl.to(Device::CUDA);
    Tensor gS = sin_tbl.to(Device::CUDA);
    Tensor gpu_Y;
    brotensor::rope_apply(gX, gC, gS, head_dim, num_heads, gpu_Y);
    brotensor::sync_all();

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)),
                    "rope_apply_bf16_fwd", 3e-2f, 3e-2f);
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

int main() { return run_all("rope cpu/gpu parity"); }
