// CPU↔GPU parity for rope_apply_mrope.
//
// FP32 cross-backend agreement. Tables are FP32; X/Y use the dispatch dtype.
// pos_t/h/w follow the cu_seqlens convention (host pointers on CPU, device
// pointers via upload_indices() on GPU).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

Tensor build_tables_pair(int max_pos, int num_pairs, int eff_head_dim,
                         float base, Tensor& sin_tbl) {
    Tensor cos_tbl = Tensor::mat(max_pos, num_pairs);
    sin_tbl = Tensor::mat(max_pos, num_pairs);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < num_pairs; ++i) {
            const float theta_i = std::exp(
                -(float)(2 * i) / (float)eff_head_dim * std::log(base));
            const float a = (float)p * theta_i;
            cos_tbl.ptr()[p * num_pairs + i] = std::cos(a);
            sin_tbl.ptr()[p * num_pairs + i] = std::sin(a);
        }
    }
    return cos_tbl;
}

void run_mrope(int L, int num_heads, int d_t, int d_h, int d_w,
               int max_pos_t, int max_pos_h, int max_pos_w,
               uint64_t seed) {
    const int head_dim = 2 * (d_t + d_h + d_w);
    SplitMix64 rng(seed);

    Tensor sin_t, sin_h, sin_w;
    Tensor cos_t = build_tables_pair(max_pos_t, d_t > 0 ? d_t : 1, head_dim,
                                     10000.0f, sin_t);
    Tensor cos_h = build_tables_pair(max_pos_h, d_h > 0 ? d_h : 1, head_dim,
                                     10000.0f, sin_h);
    Tensor cos_w = build_tables_pair(max_pos_w, d_w > 0 ? d_w : 1, head_dim,
                                     10000.0f, sin_w);

    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);

    std::vector<int32_t> pos_t(L), pos_h(L), pos_w(L);
    for (int i = 0; i < L; ++i) {
        pos_t[i] = (int32_t)(rng.next_u64() % (uint64_t)max_pos_t);
        pos_h[i] = (int32_t)(rng.next_u64() % (uint64_t)max_pos_h);
        pos_w[i] = (int32_t)(rng.next_u64() % (uint64_t)max_pos_w);
    }

    // CPU.
    Tensor Y_c;
    brotensor::rope_apply_mrope(
        X, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        d_t > 0 ? pos_t.data() : nullptr,
        d_h > 0 ? pos_h.data() : nullptr,
        d_w > 0 ? pos_w.data() : nullptr,
        head_dim, num_heads, d_t, d_h, d_w, Y_c);

    // GPU.
    Tensor gX  = X.to(gpu_device());
    Tensor gCT = cos_t.to(gpu_device());
    Tensor gST = sin_t.to(gpu_device());
    Tensor gCH = cos_h.to(gpu_device());
    Tensor gSH = sin_h.to(gpu_device());
    Tensor gCW = cos_w.to(gpu_device());
    Tensor gSW = sin_w.to(gpu_device());
    Tensor gPT = upload_indices(pos_t);
    Tensor gPH = upload_indices(pos_h);
    Tensor gPW = upload_indices(pos_w);
    const int32_t* d_pt = d_t > 0 ? static_cast<const int32_t*>(gPT.data) : nullptr;
    const int32_t* d_ph = d_h > 0 ? static_cast<const int32_t*>(gPH.data) : nullptr;
    const int32_t* d_pw = d_w > 0 ? static_cast<const int32_t*>(gPW.data) : nullptr;

    Tensor gY;
    brotensor::rope_apply_mrope(
        gX, gCT, gST, gCH, gSH, gCW, gSW,
        d_pt, d_ph, d_pw,
        head_dim, num_heads, d_t, d_h, d_w, gY);

    compare_tensors(Y_c, download_to_host(gY), "rope_apply_mrope", 1e-4f, 1e-3f);
}

BT_PARITY_TEST(mrope_degenerate_t_only) {
    // d_h = d_w = 0; equivalent to vanilla rope_apply.
    run_mrope(8, 2, 4, 0, 0, 16, 1, 1, 0xE001);
}
BT_PARITY_TEST(mrope_three_axis_small) {
    run_mrope(12, 2, 2, 3, 3, 16, 24, 24, 0xE002);
}
BT_PARITY_TEST(mrope_balanced_split) {
    // d_t = d_h = d_w = 4 -> head_dim = 24.
    run_mrope(10, 4, 4, 4, 4, 32, 32, 32, 0xE003);
}
BT_PARITY_TEST(mrope_longer_seq) {
    run_mrope(64, 2, 8, 12, 12, 128, 64, 64, 0xE004);
}

} // namespace

int main() {
    return run_all("test_rope_mrope_parity");
}
