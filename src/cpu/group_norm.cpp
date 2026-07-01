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
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <cstddef>
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

    // Each sample n owns its own slice of Y exclusively (X/gamma/beta are
    // read-only shared), so this parallelizes across n with no cross-thread
    // writes.
    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
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
    });
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

    // dGamma_c / dBeta_c accumulate over EVERY n (a fixed channel appears in
    // exactly one group but in all N samples' tiles), so a naive
    // parallel-for over n would race on dGp[channel]/dBp[channel] — that's a
    // shared reduction across the batch axis, not disjoint per n.
    //
    // Fix: each n instead writes its per-channel partial dg/db into row n of
    // a (N, C) scratch buffer (disjoint per n — every channel is touched by
    // exactly one group/local_c per n, so each element of the row is
    // written exactly once, no init needed). A final single-threaded pass
    // sums those partials down into dGp/dBp. Pass 1/2/3 (stats, partial
    // dg/db + x̂ caching, dX write) are otherwise fully independent per n and
    // run inside the parallel_for; x̂'s scratch buffer is declared LOCAL to
    // the lambda (one per n, sized tile_size) instead of shared across n, so
    // concurrent n's never touch the same buffer.
    std::vector<float> dg_partial(static_cast<std::size_t>(N) * C);
    std::vector<float> db_partial(static_cast<std::size_t>(N) * C);

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ni) {
        const int n = static_cast<int>(ni);
        float* dg_row = dg_partial.data() + static_cast<std::size_t>(n) * C;
        float* db_row = db_partial.data() + static_cast<std::size_t>(n) * C;
        // Local per-n scratch for x̂ — safe even though this lambda runs on
        // different threads, since each invocation gets its own stack/heap-
        // local buffer.
        std::vector<float> xhat_buf(static_cast<std::size_t>(tile_size));

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

            // Pass 2: sum1 = Σ dx̂, sum2 = Σ dx̂·x̂; stash this n's per-channel
            // dg/db into the partial buffers; cache x̂ into xhat_buf for
            // reuse in pass 3. Nested (channel, spatial) loops keep
            // local_c/channel loop-invariant per channel instead of
            // deriving it via division per element.
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
                dg_row[channel] = dg;   // disjoint per-n write
                db_row[channel] = db;   // disjoint per-n write
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
    });

    // Single-threaded: sum the per-n partials down into dGamma/dBeta.
    for (int c = 0; c < C; ++c) {
        float dg = 0.0f, db = 0.0f;
        for (int n = 0; n < N; ++n) {
            dg += dg_partial[static_cast<std::size_t>(n) * C + c];
            db += db_partial[static_cast<std::size_t>(n) * C + c];
        }
        dGp[c] += dg;   // accumulate
        dBp[c] += db;   // accumulate
    }
}

} // namespace brotensor::detail::cpu
