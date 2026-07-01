// ─── CPU GroupNorm ops (CHUNK 3) ───────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/group_norm.cu — FP32 path
// only. NCHW activations; one tile per (sample, group).
//
// A tile spans channels_per_group * spatial elements, where
//   channels_per_group = C / num_groups   and   spatial = H * W.
//
// Forward:
//   mean    = (1/M) Σ x
//   var     = (1/M) Σ x² - mean²            (GPU's biased estimator)
//   rstd    = 1 / sqrt(var + eps)
//   y[c,s]  = (x - mean) * rstd * gamma[c] + beta[c]
//
// Backward (per tile; dx̂ = dY*γ_c, x̂ = (x-mean)*rstd):
//   sum1    = Σ dx̂
//   sum2    = Σ dx̂ · x̂
//   dX      = rstd * (dx̂ - (sum1 + x̂*sum2) / M)
//   dGamma_c += Σ_{s in tile,batch} dY · x̂
//   dBeta_c  += Σ_{s in tile,batch} dY
//
// ACCUMULATION (matches the GPU kernels):
//   group_norm_forward  — Y  OVERWRITTEN.
//   group_norm_backward — dX OVERWRITTEN; dGamma / dBeta ACCUMULATE (+=).
//                         The GPU atomicAdds into FP32 scratch then folds
//                         into the caller's dGamma/dBeta — caller zeros them.
//
// FP32 accumulation mirrors the GPU (single-precision sums); a wider double
// accumulator would diverge from the GPU and break parity.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu {

void group_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups,
                        float eps,
                        ::brotensor::Tensor& Y) {
    if (gamma.dtype != X.dtype || beta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_forward: gamma/beta dtype must match X");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward: num_groups must divide C");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (gamma.size() != C || beta.size() != C) {
        throw std::runtime_error("group_norm_forward: gamma/beta must have C elements");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    const float* gp = gamma.host_f32();
    const float* bp = beta.host_f32();
    float* Yp = Y.host_f32_mut();

    const int channels_per_group = C / num_groups;
    const int tile_size = channels_per_group * spatial;
    const int sample_stride = C * spatial;
    const float inv_M = 1.0f / static_cast<float>(tile_size);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int chan_base = g * channels_per_group;
            const float* x_tile = Xp + n * sample_stride + chan_base * spatial;
            float*       y_tile = Yp + n * sample_stride + chan_base * spatial;

            float sum = 0.0f, sumsq = 0.0f;
            for (int i = 0; i < tile_size; ++i) {
                const float v = x_tile[i];
                sum   += v;
                sumsq += v * v;
            }
            const float mean = sum * inv_M;
            const float var  = sumsq * inv_M - mean * mean;
            const float rstd = 1.0f / std::sqrt(var + eps);

            for (int i = 0; i < tile_size; ++i) {
                const int local_c = i / spatial;
                const int channel = chan_base + local_c;
                const float yn = (x_tile[i] - mean) * rstd;
                y_tile[i] = yn * gp[channel] + bp[channel];   // overwrite
            }
        }
    }
}

void group_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups,
                         float eps,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta) {
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: gamma/dY dtype must match X");
    }
    if (dGamma.dtype != X.dtype || dBeta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta dtype must match X");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_backward: num_groups must divide C");
    }
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta must be (C,1)");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (gamma.size() != C) {
        throw std::runtime_error("group_norm_backward: gamma must have C elements");
    }
    if (dY.rows != N || dY.cols != cols) {
        throw std::runtime_error("group_norm_backward: dY shape mismatch");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp  = X.host_f32();
    const float* gp  = gamma.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    float* dGp = dGamma.host_f32_mut();
    float* dBp = dBeta.host_f32_mut();

    const int channels_per_group = C / num_groups;
    const int tile_size = channels_per_group * spatial;
    const int sample_stride = C * spatial;
    const float inv_M = 1.0f / static_cast<float>(tile_size);

    // Scratch buffer for x̂, cached during pass 2 and reused in pass 3 so the
    // tile's data doesn't need to be re-read/recomputed a third time.
    std::vector<float> xhat_buf(static_cast<std::size_t>(tile_size));

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int chan_base = g * channels_per_group;
            const float* x_tile  = Xp  + n * sample_stride + chan_base * spatial;
            const float* dy_tile = dYp + n * sample_stride + chan_base * spatial;
            float*       dx_tile = dXp + n * sample_stride + chan_base * spatial;

            // Pass 1: mean, var, rstd over the tile.
            float sum = 0.0f, sumsq = 0.0f;
            for (int i = 0; i < tile_size; ++i) {
                const float v = x_tile[i];
                sum   += v;
                sumsq += v * v;
            }
            const float mean = sum * inv_M;
            const float var  = sumsq * inv_M - mean * mean;
            const float rstd = 1.0f / std::sqrt(var + eps);

            // Pass 2: sum1 = Σ dx̂, sum2 = Σ dx̂·x̂; accumulate dGamma/dBeta;
            // cache x̂ into xhat_buf for reuse in pass 3. Nested
            // (channel, spatial) loops keep local_c/channel loop-invariant
            // per channel instead of deriving it via division per element.
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int local_c = 0; local_c < channels_per_group; ++local_c) {
                const int channel = chan_base + local_c;
                const float gv = gp[channel];
                const int base = local_c * spatial;
                float dg = 0.0f, db = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    const int i = base + s;
                    const float dyv = dy_tile[i];
                    const float xh  = (x_tile[i] - mean) * rstd;
                    xhat_buf[i] = xh;
                    const float dxh = dyv * gv;
                    sum1 += dxh;
                    sum2 += dxh * xh;
                    dg += dyv * xh;
                    db += dyv;
                }
                dGp[channel] += dg;   // accumulate
                dBp[channel] += db;   // accumulate
            }

            // Pass 3: dX = rstd * (dx̂ - (sum1 + x̂*sum2) / M).
            for (int local_c = 0; local_c < channels_per_group; ++local_c) {
                const int channel = chan_base + local_c;
                const float gv = gp[channel];
                const int base = local_c * spatial;
                for (int s = 0; s < spatial; ++s) {
                    const int i = base + s;
                    const float dyv = dy_tile[i];
                    const float xh  = xhat_buf[i];
                    const float dxh = dyv * gv;
                    dx_tile[i] = rstd * (dxh - (sum1 + xh * sum2) * inv_M);  // overwrite
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
