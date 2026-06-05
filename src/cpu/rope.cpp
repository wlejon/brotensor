// ─── CPU RoPE ops (CHUNK 2) ────────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/rope.cu — FP32 path only.
//
// Rotary position embedding: per head, rotate consecutive pairs
// (x_{2i}, x_{2i+1}) by angle theta = pos * theta_base^{-2i/head_dim}, where
// pos = row + seq_offset.
//
//   X / Y layout: (L, num_heads * head_dim), row-major. Within a row, heads
//   are contiguous head_dim-sized blocks; within a head the pairs are
//   (2i, 2i+1).
//   forward:  Y[2i]   = x0*c - x1*s ;  Y[2i+1] = x0*s + x1*c
//   backward: dX[2i]  = dy0*c + dy1*s ; dX[2i+1] = -dy0*s + dy1*c  (R(θ)^T)
//
// Both directions overwrite their output (the GPU kernels write directly).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline float rope_theta(int pair_i, int head_dim, float base) {
    // theta_i = base^{-2i/head_dim} = exp(-2i/hd * log(base)).
    return std::exp(-static_cast<float>(2 * pair_i) /
                    static_cast<float>(head_dim) * std::log(base));
}

} // namespace

void rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_forward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_forward: num_heads must be positive");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_forward: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols) Y.resize(L, X.cols);
    const int half = head_dim / 2;
    if (L * num_heads * half == 0) return;
    const int D = num_heads * head_dim;
    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();
    for (int row = 0; row < L; ++row) {
        const int pos = row + seq_offset;
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float theta =
                    static_cast<float>(pos) * rope_theta(i, head_dim, theta_base);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const float x0 = Xp[base_off + 2 * i];
                const float x1 = Xp[base_off + 2 * i + 1];
                Yp[base_off + 2 * i]     = x0 * c - x1 * s;
                Yp[base_off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

void rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, ::brotensor::Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_backward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_backward: num_heads must be positive");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    if (dX.rows != L || dX.cols != dY.cols) dX.resize(L, dY.cols);
    const int half = head_dim / 2;
    if (L * num_heads * half == 0) return;
    const int D = num_heads * head_dim;
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int row = 0; row < L; ++row) {
        const int pos = row + seq_offset;
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float theta =
                    static_cast<float>(pos) * rope_theta(i, head_dim, theta_base);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const float dy0 = dYp[base_off + 2 * i];
                const float dy1 = dYp[base_off + 2 * i + 1];
                // Inverse rotation (transpose of R(θ)).
                dXp[base_off + 2 * i]     =  dy0 * c + dy1 * s;
                dXp[base_off + 2 * i + 1] = -dy0 * s + dy1 * c;
            }
        }
    }
}

// ─── RoPE with explicit cos/sin tables ─────────────────────────────────────
//
// Ports src/cuda/rope.cu — rope_apply / rope_apply_backward, FP32 path.
// cos_tbl / sin_tbl are (L, head_dim/2): one rotation angle per (row, pair),
// shared across all heads. The caller owns position semantics (this is what
// supports 2D axial RoPE). Both directions overwrite their output.

namespace {

inline void check_tables(const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         const char* op, int L, int half) {
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": cos_tbl / sin_tbl must be FP32");
    }
    if (cos_tbl.size() != L * half || sin_tbl.size() != L * half) {
        throw std::runtime_error(std::string(op) +
                                 ": cos_tbl / sin_tbl must each be (L, head_dim/2)");
    }
}

} // namespace

void rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32) {
        throw std::runtime_error("rope_apply: X must be FP32 (CPU backend is FP32-only)");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    check_tables(cos_tbl, sin_tbl, "rope_apply", L, half);
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != Dtype::FP32) {
        Y.resize(L, X.cols, Dtype::FP32);
    }
    if (L * num_heads * half == 0) return;
    const int D = num_heads * head_dim;
    const float* Xp = X.host_f32();
    const float* Cp = cos_tbl.host_f32();
    const float* Sp = sin_tbl.host_f32();
    float* Yp = Y.host_f32_mut();
    for (int row = 0; row < L; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float c = Cp[row * half + i];
                const float s = Sp[row * half + i];
                const float x0 = Xp[base_off + 2 * i];
                const float x1 = Xp[base_off + 2 * i + 1];
                Yp[base_off + 2 * i]     = x0 * c - x1 * s;
                Yp[base_off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

// ─── RoPE with explicit PER-HEAD cos/sin tables ───────────────────────────
//
// Like rope_apply, but cos_tbl / sin_tbl are (L*num_heads, head_dim/2): one
// angle per (row, head, pair). Rows are head-minor within each token, i.e. the
// table row for (row, h) is (row*num_heads + h). Inference-only.

void rope_apply_perhead(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& cos_tbl,
                        const ::brotensor::Tensor& sin_tbl,
                        int head_dim, int num_heads, ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_perhead: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_perhead: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32) {
        throw std::runtime_error("rope_apply_perhead: X must be FP32 (CPU backend is FP32-only)");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_perhead: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) {
        throw std::runtime_error("rope_apply_perhead: cos_tbl / sin_tbl must be FP32");
    }
    if (cos_tbl.size() != L * num_heads * half || sin_tbl.size() != L * num_heads * half) {
        throw std::runtime_error(
            "rope_apply_perhead: cos_tbl / sin_tbl must each be (L*num_heads, head_dim/2)");
    }
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != Dtype::FP32) {
        Y.resize(L, X.cols, Dtype::FP32);
    }
    if (L * num_heads * half == 0) return;
    const int D = num_heads * head_dim;
    const float* Xp = X.host_f32();
    const float* Cp = cos_tbl.host_f32();
    const float* Sp = sin_tbl.host_f32();
    float* Yp = Y.host_f32_mut();
    for (int row = 0; row < L; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim;
            const int tbl_off  = (row * num_heads + h) * half;
            for (int i = 0; i < half; ++i) {
                const float c = Cp[tbl_off + i];
                const float s = Sp[tbl_off + i];
                const float x0 = Xp[base_off + 2 * i];
                const float x1 = Xp[base_off + 2 * i + 1];
                Yp[base_off + 2 * i]     = x0 * c - x1 * s;
                Yp[base_off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_backward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_backward: num_heads must be positive");
    }
    if (dY.dtype != Dtype::FP32) {
        throw std::runtime_error("rope_apply_backward: dY must be FP32 (CPU backend is FP32-only)");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    const int half = head_dim / 2;
    check_tables(cos_tbl, sin_tbl, "rope_apply_backward", L, half);
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != Dtype::FP32) {
        dX.resize(L, dY.cols, Dtype::FP32);
    }
    if (L * num_heads * half == 0) return;
    const int D = num_heads * head_dim;
    const float* dYp = dY.host_f32();
    const float* Cp = cos_tbl.host_f32();
    const float* Sp = sin_tbl.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int row = 0; row < L; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float c = Cp[row * half + i];
                const float s = Sp[row * half + i];
                const float dy0 = dYp[base_off + 2 * i];
                const float dy1 = dYp[base_off + 2 * i + 1];
                dXp[base_off + 2 * i]     =  dy0 * c + dy1 * s;
                dXp[base_off + 2 * i + 1] = -dy0 * s + dy1 * c;
            }
        }
    }
}

} // namespace brotensor::detail::cpu
