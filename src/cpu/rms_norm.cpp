// ─── CPU RMSNorm ops (CHUNK 2) ─────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/rms_norm.cu — FP32 path
// only. Per-row (one block per row on the GPU).
//
//   rms[b]    = sqrt(mean_j x[b, j]^2 + eps)
//   y[b, j]   = x[b, j] * gamma[j] / rms[b]
//
// Backward (rrms = 1/rms):
//   sum_xdy   = sum_j x_j * dY_j * gamma_j
//   coeff     = (1/D) * rrms^2 * sum_xdy
//   dX[b,j]   = rrms * (gamma_j * dY_j - x_j * coeff)
//   dGamma_j += sum_b dY[b,j] * x[b,j] * rrms
//
// ACCUMULATION: dX is OVERWRITTEN (the GPU writes dxrow[j] directly). dGamma
// ACCUMULATES (+=) — the GPU atomicAdds across the batch into dGamma, and the
// caller is responsible for zeroing dGamma beforehand.

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu {

void rms_norm_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma,
                      float eps, ::brotensor::Tensor& Y) {
    if (gamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_forward: gamma.dtype must match X.dtype");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (gamma.size() != D) {
        throw std::runtime_error("rms_norm_forward: gamma must have D elements");
    }
    if (Y.rows != B || Y.cols != D) Y.resize(B, D);
    if (B == 0 || D == 0) return;
    const float* Xp = X.host_f32();
    const float* gp = gamma.host_f32();
    float* Yp = Y.host_f32_mut();
    // Each row b owns Y's row b exclusively (X/gamma are read-only shared
    // inputs), so this parallelizes across b with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(B), [&](std::size_t bi) {
        const int b = static_cast<int>(bi);
        const float* xrow = Xp + static_cast<size_t>(b) * D;
        float*       yrow = Yp + static_cast<size_t>(b) * D;
        float sum = 0.0f;
        for (int j = 0; j < D; ++j) {
            const float v = xrow[j];
            sum += v * v;
        }
        const float rrms = 1.0f / std::sqrt(sum / static_cast<float>(D) + eps);
        for (int j = 0; j < D; ++j) {
            yrow[j] = xrow[j] * gp[j] * rrms;
        }
    });
}

void rms_norm_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma,
                       const ::brotensor::Tensor& dY, float eps,
                       ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma) {
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype ||
        dGamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_backward: dtypes must match");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (dY.rows != B || dY.cols != D) {
        throw std::runtime_error("rms_norm_backward: dY shape mismatch");
    }
    if (gamma.size() != D || dGamma.size() != D) {
        throw std::runtime_error("rms_norm_backward: gamma/dGamma size mismatch");
    }
    if (dX.rows != B || dX.cols != D) dX.resize(B, D);
    if (B == 0 || D == 0) return;
    const float* Xp = X.host_f32();
    const float* gp = gamma.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    float* dGp = dGamma.host_f32_mut();
    const float inv_D = 1.0f / static_cast<float>(D);

    // dGamma accumulates dGp[j] += ... summed over every row b — a shared
    // reduction across the batch axis, not disjoint per b. So the dX pass
    // (fully owned by row b) is parallelized, while rrms per row is cached
    // (one float per b, disjoint write) so the dGamma pass below can reuse
    // it without re-deriving it or touching any shared state during the
    // parallel section. dGamma itself is accumulated in a separate,
    // single-threaded pass afterwards.
    std::vector<float> rrms_cache(static_cast<std::size_t>(B));

    parallel_for(static_cast<std::size_t>(B), [&](std::size_t bi) {
        const int b = static_cast<int>(bi);
        const float* xrow  = Xp  + static_cast<size_t>(b) * D;
        const float* dyrow = dYp + static_cast<size_t>(b) * D;
        float*       dxrow = dXp + static_cast<size_t>(b) * D;

        float sum_xx = 0.0f;
        for (int j = 0; j < D; ++j) {
            const float v = xrow[j];
            sum_xx += v * v;
        }
        const float rrms = 1.0f / std::sqrt(sum_xx * inv_D + eps);
        rrms_cache[b] = rrms;   // disjoint per-b write

        float sum_xdy = 0.0f;
        for (int j = 0; j < D; ++j) {
            sum_xdy += xrow[j] * dyrow[j] * gp[j];
        }
        const float coeff = inv_D * rrms * rrms * sum_xdy;

        for (int j = 0; j < D; ++j) {
            const float g  = gp[j];
            const float dy = dyrow[j];
            const float x  = xrow[j];
            dxrow[j] = rrms * (g * dy - x * coeff);   // overwrite dX
        }
    });

    // Single-threaded: dGamma reduces across b, so it cannot be split across
    // threads without a race on dGp[j].
    for (int b = 0; b < B; ++b) {
        const float* xrow  = Xp  + static_cast<size_t>(b) * D;
        const float* dyrow = dYp + static_cast<size_t>(b) * D;
        const float rrms = rrms_cache[b];
        for (int j = 0; j < D; ++j) {
            dGp[j] += dyrow[j] * xrow[j] * rrms;   // accumulate dGamma
        }
    }
}

} // namespace brotensor::detail::cpu
