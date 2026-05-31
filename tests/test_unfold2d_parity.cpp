// CPU↔GPU parity tests for brotensor::unfold2d_forward.
//
// Spatial-preserving neighborhood im2col, NCHW. Layout
//   Y[n, c, k, oy, ox] = X[n, c, oy*sh - pad_top + ky, ox*sw - pad_left + kx]
// with k = ky*kW + kx and out-of-range source resolved by mode
// (0 zero / 1 reflect / 2 replicate). FP32 on both sides; FP16/BF16 GPU-only.
//
// Beyond CPU↔GPU parity, one case checks the CPU result against a hand-rolled
// host reference for the DSINE get_unfold configuration (5×5, stride 1, pad 2,
// replicate) so the kernel's correctness is anchored, not just self-consistent.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

int src_idx(int coord, int L, int mode) {
    if (coord >= 0 && coord < L) return coord;
    if (mode == 0) return -1;
    if (mode == 2) return coord < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    int q = coord, period = 2 * (L - 1);
    q %= period; if (q < 0) q += period;
    return q < L ? q : period - q;
}

// Host reference unfold, computed straight from the contract.
std::vector<float> ref_unfold(const std::vector<float>& X,
                              int N, int C, int H, int W,
                              int kH, int kW, int sh, int sw,
                              int pt, int pb, int pl, int pr, int mode,
                              int& H_out, int& W_out) {
    H_out = (H + pt + pb - kH) / sh + 1;
    W_out = (W + pl + pr - kW) / sw + 1;
    const int kK = kH * kW;
    std::vector<float> Y(static_cast<size_t>(N) * C * kK * H_out * W_out, 0.0f);
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int ky = 0; ky < kH; ++ky)
          for (int kx = 0; kx < kW; ++kx) {
            const int k = ky * kW + kx;
            for (int oy = 0; oy < H_out; ++oy)
              for (int ox = 0; ox < W_out; ++ox) {
                const int sy = src_idx(oy * sh - pt + ky, H, mode);
                const int sx = src_idx(ox * sw - pl + kx, W, mode);
                float v = 0.0f;
                if (sy >= 0 && sx >= 0)
                    v = X[(((size_t)n * C + c) * H + sy) * W + sx];
                Y[(((size_t)n * C + (c * kK + k)) * H_out + oy) * W_out + ox] = v;
              }
          }
    return Y;
}

void run(int N, int C, int H, int W, int kH, int kW,
         int sh, int sw, int pt, int pb, int pl, int pr, int mode,
         uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::unfold2d_forward(X, N, C, H, W, kH, kW, sh, sw,
                                pt, pb, pl, pr, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::unfold2d_forward(gX, N, C, H, W, kH, kW, sh, sw,
                                pt, pb, pl, pr, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "unfold2d_fwd", 0.0f, 0.0f);

    // Anchor the CPU result against the independent host reference.
    int H_out = 0, W_out = 0;
    std::vector<float> Xv(X.size());
    for (int i = 0; i < X.size(); ++i) Xv[i] = X[i];
    std::vector<float> ref = ref_unfold(Xv, N, C, H, W, kH, kW, sh, sw,
                                        pt, pb, pl, pr, mode, H_out, W_out);
    BT_CHECK(cpu_Y.size() == static_cast<int>(ref.size()));
    for (int i = 0; i < cpu_Y.size(); ++i) BT_CHECK(cpu_Y[i] == ref[i]);
}

} // namespace

// DSINE get_unfold config: 5×5, stride 1, pad 2, replicate (same-size unfold).
BT_PARITY_TEST(unfold2d_dsine_5x5_replicate) { run(1, 3, 9, 11, 5, 5, 1, 1, 2, 2, 2, 2, 2, 0x9100ull); }
BT_PARITY_TEST(unfold2d_3x3_zero)            { run(1, 4, 8,  8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0x9101ull); }
BT_PARITY_TEST(unfold2d_3x3_reflect)         { run(2, 2, 7,  6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0x9102ull); }
BT_PARITY_TEST(unfold2d_2x2_stride2)         { run(1, 3, 8,  8, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0x9103ull); }
BT_PARITY_TEST(unfold2d_asym_pad)            { run(1, 2, 6,  5, 3, 3, 1, 1, 2, 0, 1, 1, 2, 0x9104ull); }

int main() { return run_all("unfold2d parity"); }
