// CPU↔GPU parity tests for brotensor::convex_upsample_forward.
//
// RAFT-style mask upsample, NCHW. Each low-res pixel expands to scale×scale;
// every fine pixel is a softmax-weighted blend of the 3×3 low-res neighborhood:
//   Y[n,c,k*y+sy,k*x+sx] = sum_m softmax_m(Mask[n,m,sy,sx,y,x]) * X[n,c,ny,nx]
// Mask flat channel = (m*k*k + sy*k + sx); neighbor m: ny=clamp(y-1+m/3),
// nx=clamp(x-1+m%3). FP32 both sides; plus an independent host reference,
// including the DSINE config (C=3, scale=8).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

std::vector<float> ref_upsample(const Tensor& X, const Tensor& Mask,
                                int N, int C, int H, int W, int scale) {
    const int HW = H * W, kk = scale * scale;
    const int oH = scale * H, oW = scale * W, oHW = oH * oW;
    std::vector<float> Y(static_cast<size_t>(N) * C * oHW, 0.0f);
    double w[9];
    for (int n = 0; n < N; ++n)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          const int pix = y * W + x;
          for (int sy = 0; sy < scale; ++sy)
            for (int sx = 0; sx < scale; ++sx) {
              const int sub = sy * scale + sx;
              double mx = -1e300;
              for (int m = 0; m < 9; ++m) {
                  const double v = Mask[((size_t)n * 9 * kk + (size_t)m * kk + sub) * HW + pix];
                  if (v > mx) mx = v;
              }
              double sum = 0.0;
              for (int m = 0; m < 9; ++m) {
                  const double e = std::exp(
                      (double)Mask[((size_t)n * 9 * kk + (size_t)m * kk + sub) * HW + pix] - mx);
                  w[m] = e; sum += e;
              }
              for (int m = 0; m < 9; ++m) w[m] /= sum;
              const int oy = scale * y + sy, ox = scale * x + sx;
              for (int c = 0; c < C; ++c) {
                  double acc = 0.0;
                  for (int m = 0; m < 9; ++m) {
                      const int ny = clampi(y - 1 + m / 3, 0, H - 1);
                      const int nx = clampi(x - 1 + m % 3, 0, W - 1);
                      acc += w[m] * X[((size_t)n * C + c) * HW + (size_t)ny * W + nx];
                  }
                  Y[((size_t)n * C + c) * oHW + (size_t)oy * oW + ox] = (float)acc;
              }
            }
        }
    return Y;
}

void run(int N, int C, int H, int W, int scale, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    Tensor Mask = Tensor::mat(N, 9 * scale * scale * H * W);
    fill_random(X, rng);
    fill_random(Mask, rng, 3.0f);  // wider logits → less uniform softmax

    Tensor cpu_Y;
    brotensor::convex_upsample_forward(X, Mask, N, C, H, W, scale, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gMask = Mask.to(gpu_device());
    Tensor gpu_Y;
    brotensor::convex_upsample_forward(gX, gMask, N, C, H, W, scale, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "convex_up", 1e-5f, 1e-4f);

    std::vector<float> ref = ref_upsample(X, Mask, N, C, H, W, scale);
    BT_CHECK(cpu_Y.size() == (int)ref.size());
    for (int i = 0; i < cpu_Y.size(); ++i)
        BT_CHECK(std::fabs(cpu_Y[i] - ref[i]) <= 1e-5f + 1e-4f * std::fabs(ref[i]));
}

} // namespace

BT_PARITY_TEST(convex_up_dsine_c3_k8) { run(1, 3, 6, 8, 8, 0x9300ull); }
BT_PARITY_TEST(convex_up_k2)          { run(1, 4, 5, 5, 2, 0x9301ull); }
BT_PARITY_TEST(convex_up_k4_n2)       { run(2, 2, 4, 6, 4, 0x9302ull); }
BT_PARITY_TEST(convex_up_1x1)         { run(1, 3, 1, 1, 4, 0x9303ull); }
BT_PARITY_TEST(convex_up_k1)          { run(1, 5, 7, 7, 1, 0x9304ull); }

int main() { return run_all("convex_upsample parity"); }
