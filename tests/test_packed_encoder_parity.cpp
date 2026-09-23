// Packed variable-length encoder ops: CPU reference semantics + CPU<->GPU parity.
//
//   flash_attention_packed_qkv_forward — CPU checked against the existing
//       per-sequence flash_attention_windowed_forward (window) and
//       flash_attention_gqa_forward (full); GPU (FP16 / BF16 tensor-core path at
//       head_dim 64, generic kernel otherwise, FP32) against CPU.
//   rope_qkv_packed_inplace — CPU against rope_apply per sequence; GPU vs CPU.
//   segment_softmax_stats   — CPU against a hand computation; GPU vs CPU.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

float q16(float v) { return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v)); }
float qbf(float v) { return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v)); }

Tensor bounds_for(const std::vector<int>& lens) {
    int L = 0;
    for (int n : lens) L += n;
    Tensor b = Tensor::zeros_on(Device::CPU, L, 2, Dtype::INT32);
    int32_t* p = static_cast<int32_t*>(b.host_raw_mut());
    int s = 0;
    for (int n : lens) {
        for (int i = 0; i < n; ++i) {
            p[2 * (s + i)] = s;
            p[2 * (s + i) + 1] = s + n;
        }
        s += n;
    }
    return b;
}

Tensor rows_of(const Tensor& X, int r0, int n, int c0, int nc) {
    Tensor Y = Tensor::mat(n, nc);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < nc; ++c) Y(r, c) = X(r0 + r, c0 + c);
    return Y;
}

// CPU packed op vs the existing single-sequence attention ops.
void cpu_semantics(const std::vector<int>& lens, int H, int hd, int window, uint64_t seed) {
    SplitMix64 rng(seed);
    const Tensor b = bounds_for(lens);
    const int L = b.rows, D = H * hd;
    Tensor qkv = Tensor::mat(L, 3 * D);
    fill_random(qkv, rng, 1.5f);
    Tensor O;
    brotensor::flash_attention_packed_qkv_forward(qkv, b, H, window, O);
    int s = 0;
    for (int n : lens) {
        const Tensor q = rows_of(qkv, s, n, 0, D), k = rows_of(qkv, s, n, D, D), v = rows_of(qkv, s, n, 2 * D, D);
        Tensor ref;
        if (window > 0) {
            brotensor::flash_attention_windowed_forward(q, k, v, nullptr, H, window, ref, /*causal=*/false);
        } else {
            brotensor::flash_attention_gqa_forward(q, k, v, nullptr, H, H, /*causal=*/false, ref);
        }
        compare_tensors(ref, rows_of(O, s, n, 0, D), "packed_vs_per_seq", 1e-5f, 1e-4f);
        s += n;
    }
}

// GPU vs CPU at `dt` (inputs pre-rounded to dt so both sides see the same values).
void gpu_parity(const std::vector<int>& lens, int H, int hd, int window, Dtype dt, uint64_t seed) {
    SplitMix64 rng(seed);
    const Tensor b = bounds_for(lens);
    const int L = b.rows, D = H * hd;
    Tensor qkv = Tensor::mat(L, 3 * D);
    for (int i = 0; i < qkv.size(); ++i) {
        const float x = rng.next_unit() * 2.0f;
        qkv.ptr()[i] = dt == Dtype::FP16 ? q16(x) : dt == Dtype::BF16 ? qbf(x) : x;
    }
    Tensor O_c;
    brotensor::flash_attention_packed_qkv_forward(qkv, b, H, window, O_c);

    Tensor gq = dt == Dtype::FP16 ? to_fp16_gpu(qkv) : dt == Dtype::BF16 ? to_bf16_gpu(qkv) : qkv.to(gpu_device());
    Tensor gb = b.to(gpu_device());
    Tensor gO;
    brotensor::flash_attention_packed_qkv_forward(gq, gb, H, window, gO);
    Tensor back = download_to_host(gO);
    if (dt == Dtype::FP16) back = fp16_host_to_f32(back);
    if (dt == Dtype::BF16) back = bf16_host_to_f32(back);
    const float tol = dt == Dtype::FP16 ? 1e-2f : dt == Dtype::BF16 ? 3e-2f : 1e-5f;
    compare_tensors(O_c, back, "packed_attn_gpu", tol, tol);
}

BT_PARITY_TEST(cpu_matches_windowed_per_sequence) {
    cpu_semantics({1, 37, 130, 64, 5}, 2, 8, 16, 0x51);
    cpu_semantics({200}, 2, 8, 128, 0x52);
}
BT_PARITY_TEST(cpu_matches_full_per_sequence) {
    cpu_semantics({1, 37, 70, 3}, 3, 8, 0, 0x53);
}

// Lengths chosen so 64-row query blocks straddle sequence boundaries and a
// window cuts through tiles unaligned to 64.
const std::vector<int> kLens = {99, 1, 64, 130, 17, 200, 63, 65};

BT_PARITY_TEST(gpu_fp16_hd64_window) { gpu_parity(kLens, 4, 64, 128, Dtype::FP16, 0x61); }
BT_PARITY_TEST(gpu_fp16_hd64_full) { gpu_parity(kLens, 4, 64, 0, Dtype::FP16, 0x62); }
BT_PARITY_TEST(gpu_fp16_hd64_long) { gpu_parity({512, 300, 700}, 2, 64, 0, Dtype::FP16, 0x63); }
BT_PARITY_TEST(gpu_fp16_hd64_small_window) { gpu_parity(kLens, 2, 64, 6, Dtype::FP16, 0x64); }
BT_PARITY_TEST(gpu_bf16_hd64_window) { gpu_parity(kLens, 2, 64, 128, Dtype::BF16, 0x65); }
BT_PARITY_TEST(gpu_fp16_generic_hd16) { gpu_parity(kLens, 4, 16, 16, Dtype::FP16, 0x66); }
BT_PARITY_TEST(gpu_fp32_generic) { gpu_parity({9, 40, 3}, 2, 64, 0, Dtype::FP32, 0x67); }

// Backward: CPU against central finite differences of the CPU forward under
// the loss sum(O * G), then GPU against CPU.
double packed_loss(const Tensor& qkv, const Tensor& b, const Tensor& G, int H, int window) {
    Tensor O;
    brotensor::flash_attention_packed_qkv_forward(qkv, b, H, window, O);
    double s = 0;
    for (int i = 0; i < O.size(); ++i) s += static_cast<double>(O.ptr()[i]) * G.ptr()[i];
    return s;
}

void cpu_backward_fd(const std::vector<int>& lens, int H, int hd, int window, uint64_t seed) {
    SplitMix64 rng(seed);
    const Tensor b = bounds_for(lens);
    const int L = b.rows, D = H * hd;
    Tensor qkv = Tensor::mat(L, 3 * D), G = Tensor::mat(L, D);
    fill_random(qkv, rng, 1.0f);
    fill_random(G, rng, 1.0f);
    Tensor dqkv;
    brotensor::flash_attention_packed_qkv_backward(qkv, G, b, H, window, dqkv);
    BT_CHECK(dqkv.rows == L && dqkv.cols == 3 * D);
    const float eps = 1e-2f;
    float worst = 0.0f;
    for (int i = 0; i < qkv.size(); i += 7) {
        const float x = qkv.ptr()[i];
        qkv.ptr()[i] = x + eps;
        const double lp = packed_loss(qkv, b, G, H, window);
        qkv.ptr()[i] = x - eps;
        const double lm = packed_loss(qkv, b, G, H, window);
        qkv.ptr()[i] = x;
        const float fd = static_cast<float>((lp - lm) / (2.0 * eps));
        worst = std::max(worst, std::fabs(fd - dqkv.ptr()[i]) / (1.0f + std::fabs(fd)));
    }
    BT_CHECK(worst < 2e-3f);
}

void gpu_backward_parity(const std::vector<int>& lens, int H, int hd, int window, Dtype dt, uint64_t seed) {
    SplitMix64 rng(seed);
    const Tensor b = bounds_for(lens);
    const int L = b.rows, D = H * hd;
    Tensor qkv = Tensor::mat(L, 3 * D), G = Tensor::mat(L, D);
    auto round = [&](Tensor& t, float scale) {
        for (int i = 0; i < t.size(); ++i) {
            const float x = rng.next_unit() * scale;
            t.ptr()[i] = dt == Dtype::FP16 ? q16(x) : dt == Dtype::BF16 ? qbf(x) : x;
        }
    };
    round(qkv, 2.0f);
    round(G, 1.0f);
    Tensor ref;
    brotensor::flash_attention_packed_qkv_backward(qkv, G, b, H, window, ref);
    auto up = [&](const Tensor& t) {
        return dt == Dtype::FP16 ? to_fp16_gpu(t) : dt == Dtype::BF16 ? to_bf16_gpu(t) : t.to(gpu_device());
    };
    Tensor gd;
    brotensor::flash_attention_packed_qkv_backward(up(qkv), up(G), b.to(gpu_device()), H, window, gd);
    Tensor back = download_to_host(gd);
    if (dt == Dtype::FP16) back = fp16_host_to_f32(back);
    if (dt == Dtype::BF16) back = bf16_host_to_f32(back);
    const float tol = dt == Dtype::FP16 ? 2e-2f : dt == Dtype::BF16 ? 6e-2f : 1e-4f;
    compare_tensors(ref, back, "packed_attn_bwd_gpu", tol, tol);
}

BT_PARITY_TEST(backward_cpu_finite_difference) {
    cpu_backward_fd({1, 9, 23, 4}, 2, 8, 0, 0x81);
    cpu_backward_fd({30, 2, 17}, 2, 8, 6, 0x82);
}
BT_PARITY_TEST(backward_gpu_fp32) { gpu_backward_parity({9, 40, 3, 1}, 2, 64, 0, Dtype::FP32, 0x83); }
BT_PARITY_TEST(backward_gpu_fp32_window) { gpu_backward_parity(kLens, 2, 16, 16, Dtype::FP32, 0x84); }
BT_PARITY_TEST(backward_gpu_fp16_hd64_window) { gpu_backward_parity(kLens, 4, 64, 128, Dtype::FP16, 0x85); }
BT_PARITY_TEST(backward_gpu_fp16_hd64_full) { gpu_backward_parity(kLens, 4, 64, 0, Dtype::FP16, 0x86); }
BT_PARITY_TEST(backward_gpu_bf16_hd128) { gpu_backward_parity({70, 5, 33}, 2, 128, 0, Dtype::BF16, 0x87); }
BT_PARITY_TEST(backward_gpu_fp16_hd40) { gpu_backward_parity({70, 5, 33}, 3, 40, 8, Dtype::FP16, 0x88); }

BT_PARITY_TEST(rope_packed) {
    SplitMix64 rng(0x71);
    const int H = 3, hd = 8, half = hd / 2, D = H * hd;
    const std::vector<int> lens = {5, 1, 11};
    const Tensor b = bounds_for(lens);
    const int L = b.rows, P = 16;
    Tensor qkv = Tensor::mat(L, 3 * D);
    fill_random(qkv, rng);
    Tensor ct = Tensor::mat(P, half), st = Tensor::mat(P, half);
    for (int p = 0; p < P; ++p)
        for (int i = 0; i < half; ++i) {
            const float a = p * std::pow(100.0f, -2.0f * i / hd);
            ct(p, i) = std::cos(a);
            st(p, i) = std::sin(a);
        }
    Tensor pos = Tensor::zeros_on(Device::CPU, L, 1, Dtype::INT32);
    int32_t* pp = static_cast<int32_t*>(pos.host_raw_mut());
    for (int r = 0; r < L; ++r) pp[r] = r - static_cast<const int32_t*>(b.host_raw())[2 * r];

    Tensor cpu = qkv.clone();
    brotensor::rope_qkv_packed_inplace(cpu, ct, st, pos, H, hd);
    // Reference: rope_apply per sequence with that sequence's table rows.
    int s = 0;
    for (int n : lens) {
        const Tensor c = rows_of(ct, 0, n, 0, half), sn = rows_of(st, 0, n, 0, half);
        for (int sec = 0; sec < 3; ++sec) {
            const Tensor x = rows_of(qkv, s, n, sec * D, D);
            Tensor y;
            if (sec < 2) brotensor::rope_apply(x, c, sn, hd, H, y);
            else y = x.clone();
            compare_tensors(y, rows_of(cpu, s, n, sec * D, D), "rope_packed_cpu", 1e-6f, 1e-6f);
        }
        s += n;
    }
    // GPU FP32 and FP16.
    Tensor g32 = qkv.to(gpu_device());
    brotensor::rope_qkv_packed_inplace(g32, ct.to(gpu_device()), st.to(gpu_device()), pos.to(gpu_device()), H, hd);
    compare_tensors(cpu, download_to_host(g32), "rope_packed_gpu32", 1e-6f, 1e-6f);
    Tensor g16 = to_fp16_gpu(qkv);
    brotensor::rope_qkv_packed_inplace(g16, ct.to(gpu_device()), st.to(gpu_device()), pos.to(gpu_device()), H, hd);
    compare_tensors(cpu, fp16_host_to_f32(download_to_host(g16)), "rope_packed_gpu16", 2e-3f, 2e-3f);
}

BT_PARITY_TEST(segment_stats) {
    const std::vector<float> lg = {2.0f, 1.0f, 0.5f, /*seg1*/ 3.0f, /*seg2 empty*/ /*seg3*/ -1.0f, 4.0f,
                                   4.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.5f, -2.0f, 1.5f};
    const std::vector<int32_t> off = {0, 3, 4, 4, 14};
    Tensor l = Tensor::mat(static_cast<int>(lg.size()), 1);
    for (std::size_t i = 0; i < lg.size(); ++i) l[static_cast<int>(i)] = lg[i];
    Tensor o = Tensor::zeros_on(Device::CPU, static_cast<int>(off.size()), 1, Dtype::INT32);
    std::copy(off.begin(), off.end(), static_cast<int32_t*>(o.host_raw_mut()));
    Tensor cpu;
    brotensor::segment_softmax_stats(l, o, cpu);
    BT_CHECK(cpu.rows == 4 && cpu.cols == 4);
    // Hand check of segment 0: softmax(2, 1, 0.5).
    const float e0 = 1.0f, e1 = std::exp(-1.0f), e2 = std::exp(-1.5f), z = e0 + e1 + e2;
    const float p0 = e0 / z, p1 = e1 / z, p2 = e2 / z;
    const float ent = -(p0 * std::log(p0) + p1 * std::log(p1) + p2 * std::log(p2)) / std::log(3.0f);
    BT_CHECK(std::fabs(cpu(0, 0) - p0) < 1e-6f && std::fabs(cpu(0, 1) - (p0 - p1)) < 1e-6f);
    BT_CHECK(std::fabs(cpu(0, 2) - ent) < 1e-6f && std::fabs(cpu(0, 3) - 3.0f / 255.0f) < 1e-7f);
    // One element: top1 1, top2 0, entropy 0, k clamps to 2. Empty: zeros.
    BT_CHECK(cpu(1, 0) == 1.0f && cpu(1, 1) == 1.0f && std::fabs(cpu(1, 2)) < 1e-7f &&
             std::fabs(cpu(1, 3) - 2.0f / 255.0f) < 1e-7f);
    BT_CHECK(cpu(2, 0) == 0.0f && cpu(2, 3) == 0.0f);
    // Tied top: top1 - top2 == 0.
    BT_CHECK(std::fabs(cpu(3, 1)) < 1e-7f);

    Tensor g;
    brotensor::segment_softmax_stats(l.to(gpu_device()), o.to(gpu_device()), g);
    compare_tensors(cpu, download_to_host(g), "segment_stats_gpu32", 1e-6f, 1e-5f);
    Tensor g16;
    Tensor l16 = to_fp16_gpu(l);
    brotensor::segment_softmax_stats(l16, o.to(gpu_device()), g16);
    compare_tensors(cpu, fp16_host_to_f32(download_to_host(g16)), "segment_stats_gpu16", 2e-3f, 2e-3f);
}

}  // namespace

int main() { return run_all("packed encoder ops (attention / rope / segment stats) CPU<->GPU parity"); }
