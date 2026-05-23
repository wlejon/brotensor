// ─── CPU M-RoPE (Qwen2.5-VL / Qwen3-VL) ────────────────────────────────────
//
// Three per-axis (t, h, w) position-ID streams index three per-axis cos/sin
// tables and apply pair-wise rotations to disjoint sub-ranges of each head.
// head_dim is split into three contiguous scalar sub-ranges of widths 2*d_t,
// 2*d_h, 2*d_w (in order t,h,w), with 2*(d_t + d_h + d_w) == head_dim.
//
//   X / Y      : (L, num_heads*head_dim)
//   cos_a/sin_a: (max_pos_a, d_a) FP32 — width = pairs per axis, NOT 2*d_a;
//                this matches rope_apply's convention (one cos per pair) and
//                lets the M-RoPE op reuse the same math without doubling table
//                width.
//   pos_a      : length-L INT32 host pointer on CPU; device pointer on GPU.
//
// CPU is FP32-only; X must be FP32.

#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_axis_tbl(const ::brotensor::Tensor& cos_a,
                           const ::brotensor::Tensor& sin_a,
                           const char* axis, int d_a) {
    if (d_a < 0) {
        throw std::runtime_error(std::string("rope_apply_mrope: d_") + axis +
                                 " must be non-negative");
    }
    if (d_a == 0) {
        // Empty axis: tables are not consulted; accept any (even empty) shape.
        return;
    }
    if (cos_a.dtype != Dtype::FP32 || sin_a.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " must be FP32");
    }
    if (cos_a.cols != d_a || sin_a.cols != d_a) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis +
                                 " must each have cols == d_" + axis);
    }
    if (cos_a.rows != sin_a.rows || cos_a.rows < 1) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " row count mismatch");
    }
}

inline void apply_axis(const float* Xp, float* Yp,
                       const float* cos_a, const float* sin_a,
                       const int32_t* pos_a, int max_pos_a,
                       int L, int D, int head_dim, int num_heads,
                       int pair_offset_in_head, int d_a) {
    if (d_a == 0) return;
    for (int row = 0; row < L; ++row) {
        const int pos = pos_a[row];
        if (pos < 0 || pos >= max_pos_a) {
            throw std::runtime_error(
                "rope_apply_mrope: position id out of range");
        }
        for (int h = 0; h < num_heads; ++h) {
            const int base_off = row * D + h * head_dim
                               + 2 * pair_offset_in_head;
            for (int i = 0; i < d_a; ++i) {
                const float c = cos_a[pos * d_a + i];
                const float s = sin_a[pos * d_a + i];
                const float x0 = Xp[base_off + 2 * i];
                const float x1 = Xp[base_off + 2 * i + 1];
                Yp[base_off + 2 * i]     = x0 * c - x1 * s;
                Yp[base_off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

} // namespace

void rope_apply_mrope(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& cos_t,
                      const ::brotensor::Tensor& sin_t,
                      const ::brotensor::Tensor& cos_h,
                      const ::brotensor::Tensor& sin_h,
                      const ::brotensor::Tensor& cos_w,
                      const ::brotensor::Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_mrope: head_dim must be a "
                                 "positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_mrope: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32) {
        throw std::runtime_error("rope_apply_mrope: X must be FP32 "
                                 "(CPU backend is FP32-only)");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_mrope: X.cols != "
                                 "num_heads * head_dim");
    }
    if (2 * (d_t + d_h + d_w) != head_dim) {
        throw std::runtime_error("rope_apply_mrope: 2*(d_t + d_h + d_w) "
                                 "must equal head_dim");
    }
    check_axis_tbl(cos_t, sin_t, "t", d_t);
    check_axis_tbl(cos_h, sin_h, "h", d_h);
    check_axis_tbl(cos_w, sin_w, "w", d_w);
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != Dtype::FP32) {
        Y.resize(L, X.cols, Dtype::FP32);
    }
    if (L == 0 || num_heads == 0 || head_dim == 0) return;
    if (d_t > 0 && !pos_t) throw std::runtime_error("rope_apply_mrope: pos_t null");
    if (d_h > 0 && !pos_h) throw std::runtime_error("rope_apply_mrope: pos_h null");
    if (d_w > 0 && !pos_w) throw std::runtime_error("rope_apply_mrope: pos_w null");

    const int D = num_heads * head_dim;
    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    // Y may not alias X here in general (output adopt allocates fresh), but
    // copy first so any pair we skip (e.g. d_a=0 sub-range) is identity-copied.
    if (Xp != Yp) {
        const int n = L * D;
        for (int i = 0; i < n; ++i) Yp[i] = Xp[i];
    }

    // Sub-range pair offsets within each head: [0, d_t) for t, then
    // [d_t, d_t + d_h) for h, then [d_t + d_h, d_t + d_h + d_w) for w.
    apply_axis(Xp, Yp, cos_t.host_f32(), sin_t.host_f32(),
               pos_t, d_t > 0 ? cos_t.rows : 0,
               L, D, head_dim, num_heads, /*pair_off=*/0, d_t);
    apply_axis(Xp, Yp, cos_h.host_f32(), sin_h.host_f32(),
               pos_h, d_h > 0 ? cos_h.rows : 0,
               L, D, head_dim, num_heads, /*pair_off=*/d_t, d_h);
    apply_axis(Xp, Yp, cos_w.host_f32(), sin_w.host_f32(),
               pos_w, d_w > 0 ? cos_w.rows : 0,
               L, D, head_dim, num_heads, /*pair_off=*/d_t + d_h, d_w);
}

} // namespace brotensor::detail::cpu
