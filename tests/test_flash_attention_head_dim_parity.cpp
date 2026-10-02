// CPU<->GPU parity for the flash-attention family at wide heads.
//
// The GPU kernels hold a per-thread register tile of the output row, so a
// head wider than one tile (1024 columns on CUDA and HIP) needs either a
// different path or column chunking, and on HIP the K/V tiles a kernel stages
// in LDS grow with head_dim until they no longer fit the 64 KB a block may
// take. This suite drives every forward entry point across those edges:
//
//   head_dim 256 / 320 FP32  — past the LDS staging budget at full tile depth
//   head_dim 512 / 1024      — the register tile's last chunk
//   head_dim 1100 / 1536     — two column chunks (and a ragged last one)
//
// for flash_attention_forward / _gqa_forward / _windowed_forward (causal,
// bidirectional, windowed, masked, GQA), _varlen_forward, _packed_qkv_forward,
// _decode and _decode_masked, against the FP32 CPU reference. Inputs are
// rounded to the GPU dtype first so both sides start from identical values.
//
// Coverage per backend: shapes CUDA accepts (FP16/BF16, head_dim <= 1024 on
// the scalar kernels) run on every GPU backend this binary has; FP32 inputs
// and the head_dims past CUDA's limits run where the backend takes them —
// currently HIP — and print a skip elsewhere.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

bool gpu_is_hip() { return gpu_device().is_hip(); }

// FP32 inputs and head_dims past CUDA's register-tile / LDS limits.
bool skip_unless_hip(const char* what) {
    if (gpu_is_hip()) return false;
    std::printf("    (skipped on this backend: %s)\n", what);
    return true;
}

float round_to(Dtype dt, float v) {
    if (dt == Dtype::FP16) return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
    if (dt == Dtype::BF16) return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
    return v;
}

Tensor make_cpu(Dtype dt, int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = round_to(dt, rng.next_unit() * scale);
    return t;
}

Tensor upload(Dtype dt, const Tensor& cpu) {
    if (dt == Dtype::FP16) return to_fp16_host(cpu).to(gpu_device());
    if (dt == Dtype::BF16) return to_bf16_host(cpu).to(gpu_device());
    return cpu.to(gpu_device());
}

Tensor download(Dtype dt, const Tensor& g) {
    Tensor h = download_to_host(g);
    if (dt == Dtype::FP16) return fp16_host_to_f32(h);
    if (dt == Dtype::BF16) return bf16_host_to_f32(h);
    return h;
}

float tol_for(Dtype dt) {
    if (dt == Dtype::FP16) return 1e-2f;
    if (dt == Dtype::BF16) return 3e-2f;
    return 2e-4f;
}

Tensor upload_int32(const std::vector<int32_t>& v, int rows, int cols) {
    Tensor h = Tensor::zeros_on(Device::CPU, rows, cols, Dtype::INT32);
    auto* p = static_cast<int32_t*>(h.host_raw_mut());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return h;
}

std::vector<float> key_mask(int Lk) {
    std::vector<float> m(static_cast<size_t>(Lk), 1.0f);
    for (int k = 0; k < Lk; k += 5) m[static_cast<size_t>(k)] = 0.0f;
    return m;
}

// Q scaled up so the softmax is peaked rather than near-uniform: a kernel
// that mixes up rows, columns or column chunks then moves the output.
constexpr float kQScale = 3.0f;
constexpr float kKVScale = 1.0f;

// flash_attention_windowed_forward covers forward / gqa_forward as special
// cases (window 0, causal on/off); the gqa / plain entry points get their own
// cases below so their wrappers are exercised too.
void run_windowed(Dtype dt, int Lq, int Lk, int H, int Hkv, int hd,
                  bool causal, int window, bool masked, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q = make_cpu(dt, Lq, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, Lk, Hkv * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, Lk, Hkv * hd, rng, kKVScale);
    std::vector<float> mask = key_mask(Lk);

    Tensor O_c;
    brotensor::flash_attention_windowed_forward(Q, K, V, masked ? mask.data() : nullptr,
                                                H, window, O_c, causal);

    Tensor gm = masked ? upload_mask(&mask) : Tensor{};
    Tensor gO;
    brotensor::flash_attention_windowed_forward(
        upload(dt, Q), upload(dt, K), upload(dt, V),
        masked ? static_cast<const float*>(gm.data) : nullptr, H, window, gO, causal);
    compare_tensors(O_c, download(dt, gO), "windowed.O", tol_for(dt), tol_for(dt));
}

void run_forward(Dtype dt, int Lq, int Lk, int H, int hd, bool causal, bool masked,
                 uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q = make_cpu(dt, Lq, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, Lk, H * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, Lk, H * hd, rng, kKVScale);
    std::vector<float> mask = key_mask(Lk);

    Tensor O_c;
    brotensor::flash_attention_forward(Q, K, V, masked ? mask.data() : nullptr, H, causal, O_c);

    Tensor gm = masked ? upload_mask(&mask) : Tensor{};
    Tensor gO;
    brotensor::flash_attention_forward(upload(dt, Q), upload(dt, K), upload(dt, V),
                                       masked ? static_cast<const float*>(gm.data) : nullptr,
                                       H, causal, gO);
    compare_tensors(O_c, download(dt, gO), "forward.O", tol_for(dt), tol_for(dt));
}

void run_gqa(Dtype dt, int L, int H, int Hkv, int hd, bool causal, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q = make_cpu(dt, L, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, L, Hkv * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, L, Hkv * hd, rng, kKVScale);

    Tensor O_c;
    brotensor::flash_attention_gqa_forward(Q, K, V, nullptr, H, Hkv, causal, O_c);
    Tensor gO;
    brotensor::flash_attention_gqa_forward(upload(dt, Q), upload(dt, K), upload(dt, V),
                                           nullptr, H, Hkv, causal, gO);
    compare_tensors(O_c, download(dt, gO), "gqa.O", tol_for(dt), tol_for(dt));
}

void run_varlen(Dtype dt, const std::vector<int32_t>& lens, int H, int hd, bool causal,
                uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<int32_t> cu(lens.size() + 1, 0);
    int max_len = 0;
    for (size_t i = 0; i < lens.size(); ++i) {
        cu[i + 1] = cu[i] + lens[i];
        if (lens[i] > max_len) max_len = lens[i];
    }
    const int total = cu.back();
    const int B = static_cast<int>(lens.size());
    Tensor Q = make_cpu(dt, total, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, total, H * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, total, H * hd, rng, kKVScale);

    Tensor O_c;
    brotensor::flash_attention_varlen_forward(Q, K, V, cu.data(), cu.data(), B, max_len, max_len,
                                              H, hd, causal, O_c);
    Tensor gcu = upload_int32(cu, B + 1, 1).to(gpu_device());
    const auto* dcu = static_cast<const int32_t*>(gcu.data);
    Tensor gO;
    brotensor::flash_attention_varlen_forward(upload(dt, Q), upload(dt, K), upload(dt, V),
                                              dcu, dcu, B, max_len, max_len, H, hd, causal, gO);
    compare_tensors(O_c, download(dt, gO), "varlen.O", tol_for(dt), tol_for(dt));
}

void run_packed(Dtype dt, const std::vector<int>& lens, int H, int hd, int window,
                uint64_t seed) {
    SplitMix64 rng(seed);
    int L = 0;
    for (int n : lens) L += n;
    std::vector<int32_t> bounds(static_cast<size_t>(2 * L));
    int start = 0;
    for (int n : lens) {
        for (int r = start; r < start + n; ++r) {
            bounds[static_cast<size_t>(2 * r)] = start;
            bounds[static_cast<size_t>(2 * r + 1)] = start + n;
        }
        start += n;
    }
    Tensor QKV = make_cpu(dt, L, 3 * H * hd, rng, kKVScale);
    for (int r = 0; r < L; ++r)  // peak the softmax through Q only
        for (int c = 0; c < H * hd; ++c)
            QKV.ptr()[static_cast<size_t>(r) * 3 * H * hd + c] =
                round_to(dt, QKV[r * 3 * H * hd + c] * kQScale);
    Tensor b_c = upload_int32(bounds, L, 2);

    Tensor O_c;
    brotensor::flash_attention_packed_qkv_forward(QKV, b_c, H, window, O_c);
    Tensor gO;
    brotensor::flash_attention_packed_qkv_forward(upload(dt, QKV), b_c.to(gpu_device()), H,
                                                  window, gO);
    compare_tensors(O_c, download(dt, gO), "packed.O", tol_for(dt), tol_for(dt));
}

void run_decode(Dtype dt, int Lq, int valid, int cap, int H, int Hkv, int hd, int window,
                uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q = make_cpu(dt, Lq, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, cap, Hkv * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, cap, Hkv * hd, rng, kKVScale);

    Tensor O_c;
    brotensor::flash_attention_decode(Q, K, V, valid, H, Hkv, O_c, 0.0f, window);
    Tensor gO;
    brotensor::flash_attention_decode(upload(dt, Q), upload(dt, K), upload(dt, V), valid, H, Hkv,
                                      gO, 0.0f, window);
    compare_tensors(O_c, download(dt, gO), "decode.O", tol_for(dt), tol_for(dt));
}

// Fixed-capacity cache filled to `valid`: the query sits at key valid-1, so a
// window counts back from there, not from the end of the buffer.
void run_decode_masked(Dtype dt, int valid, int cap, int H, int Hkv, int hd, int window,
                       uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q = make_cpu(dt, 1, H * hd, rng, kQScale);
    Tensor K = make_cpu(dt, cap, Hkv * hd, rng, kKVScale);
    Tensor V = make_cpu(dt, cap, Hkv * hd, rng, kKVScale);
    std::vector<float> mask(static_cast<size_t>(cap), 0.0f);
    for (int k = 0; k < valid; ++k) mask[static_cast<size_t>(k)] = 1.0f;

    Tensor O_c;
    brotensor::flash_attention_decode_masked(Q, K, V, mask.data(), H, Hkv, O_c, 0.0f, window);
    Tensor gm = upload_mask(&mask);
    Tensor gO;
    brotensor::flash_attention_decode_masked(upload(dt, Q), upload(dt, K), upload(dt, V),
                                             static_cast<const float*>(gm.data), H, Hkv, gO,
                                             0.0f, window);
    compare_tensors(O_c, download(dt, gO), "decode_masked.O", tol_for(dt), tol_for(dt));
}

} // namespace

// ─── Shapes every GPU backend accepts ─────────────────────────────────────

BT_PARITY_TEST(windowed_fp16_hd512_causal_mask) { run_windowed(Dtype::FP16, 24, 40, 2, 2, 512, true, 0, true, 0xA00); }
BT_PARITY_TEST(windowed_bf16_hd1024_causal_gqa) { run_windowed(Dtype::BF16, 20, 20, 4, 2, 1024, true, 0, false, 0xA01); }
BT_PARITY_TEST(windowed_fp16_hd1024_window8)    { run_windowed(Dtype::FP16, 30, 30, 2, 1, 1024, true, 8, false, 0xA02); }
BT_PARITY_TEST(windowed_fp16_hd512_bidir_win6)  { run_windowed(Dtype::FP16, 30, 30, 2, 2, 512, false, 6, true, 0xA03); }
BT_PARITY_TEST(forward_fp16_hd512_causal)       { run_forward(Dtype::FP16, 40, 40, 2, 512, true, false, 0xA04); }
BT_PARITY_TEST(forward_bf16_hd1024_causal_mask) { run_forward(Dtype::BF16, 24, 24, 1, 1024, true, true, 0xA05); }
BT_PARITY_TEST(gqa_fp16_hd512_causal)           { run_gqa(Dtype::FP16, 24, 4, 2, 512, true, 0xA06); }
BT_PARITY_TEST(varlen_fp16_hd512_causal)        { run_varlen(Dtype::FP16, {7, 19, 12}, 2, 512, true, 0xA07); }
BT_PARITY_TEST(varlen_bf16_hd1024)              { run_varlen(Dtype::BF16, {9, 21}, 1, 1024, false, 0xA08); }
BT_PARITY_TEST(decode_fp16_hd1024_gqa)          { run_decode(Dtype::FP16, 1, 77, 96, 4, 2, 1024, 0, 0xA09); }
BT_PARITY_TEST(decode_fp16_hd512_block_window)  { run_decode(Dtype::FP16, 4, 70, 70, 2, 1, 512, 16, 0xA0A); }
BT_PARITY_TEST(decode_masked_fp16_hd64_window)  { run_decode_masked(Dtype::FP16, 37, 128, 4, 2, 64, 8, 0xA0B); }
BT_PARITY_TEST(decode_masked_bf16_hd1024_window){ run_decode_masked(Dtype::BF16, 50, 96, 2, 1, 1024, 16, 0xA0C); }

// ─── FP32 inputs (the CUDA attention kernels take FP16/BF16 only) ─────────

BT_PARITY_TEST(windowed_fp32_hd256_causal_mask) {
    if (skip_unless_hip("FP32 attention")) return;
    run_windowed(Dtype::FP32, 20, 33, 2, 2, 256, true, 0, true, 0xA20);
}
BT_PARITY_TEST(windowed_fp32_hd320_bidir_mask_small) {
    if (skip_unless_hip("FP32 attention")) return;
    run_windowed(Dtype::FP32, 6, 50, 2, 1, 320, false, 0, true, 0xA21);
}
BT_PARITY_TEST(varlen_fp32_hd256) {
    if (skip_unless_hip("FP32 attention")) return;
    run_varlen(Dtype::FP32, {5, 17}, 2, 256, true, 0xA22);
}
BT_PARITY_TEST(packed_fp32_hd320_window) {
    if (skip_unless_hip("FP32 attention")) return;
    run_packed(Dtype::FP32, {9, 14}, 2, 320, 6, 0xA23);
}
BT_PARITY_TEST(decode_fp32_hd256) {
    if (skip_unless_hip("FP32 attention")) return;
    run_decode(Dtype::FP32, 1, 45, 45, 2, 2, 256, 0, 0xA24);
}

// ─── head_dim past one register tile (> 1024) ─────────────────────────────

// The qwenimage21 VAE mid-block shape: one head, wide, bidirectional, large
// enough for the GEMM path. Then the same with a key mask and with GQA.
BT_PARITY_TEST(forward_fp16_hd1536_bidir_dense) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_forward(Dtype::FP16, 96, 80, 1, 1536, false, false, 0xA40);
}
BT_PARITY_TEST(forward_fp32_hd1536_bidir_dense_mask) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_forward(Dtype::FP32, 72, 72, 1, 1536, false, true, 0xA41);
}
BT_PARITY_TEST(gqa_bf16_hd1100_bidir_dense) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_gqa(Dtype::BF16, 70, 4, 2, 1100, false, 0xA42);
}
BT_PARITY_TEST(forward_fp16_hd1536_causal) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_forward(Dtype::FP16, 33, 33, 2, 1536, true, false, 0xA43);
}
BT_PARITY_TEST(windowed_fp32_hd1100_window_mask) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_windowed(Dtype::FP32, 16, 40, 2, 1, 1100, true, 12, true, 0xA44);
}
BT_PARITY_TEST(windowed_bf16_hd2100_bidir_small) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_windowed(Dtype::BF16, 5, 30, 1, 1, 2100, false, 0, true, 0xA45);
}
BT_PARITY_TEST(varlen_fp16_hd1536_causal) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_varlen(Dtype::FP16, {11, 6}, 1, 1536, true, 0xA46);
}
BT_PARITY_TEST(packed_fp16_hd1536) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_packed(Dtype::FP16, {12, 7}, 1, 1536, 0, 0xA47);
}
BT_PARITY_TEST(decode_bf16_hd1536_gqa) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_decode(Dtype::BF16, 1, 60, 64, 4, 1, 1536, 0, 0xA48);
}
BT_PARITY_TEST(decode_masked_fp16_hd1100_window) {
    if (skip_unless_hip("head_dim > 1024")) return;
    run_decode_masked(Dtype::FP16, 41, 64, 2, 2, 1100, 10, 0xA49);
}

int main() { return run_all("flash_attention wide-head cpu/gpu parity"); }
