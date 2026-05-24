// ─── CPU image preprocessing helpers ───────────────────────────────────────
//
// Two ops shared by vision models (CLIP, SAM, Depth Anything, DETR, …):
//
//   image_normalize           — per-channel (X - mean[c]) / std[c] on NCHW.
//                               The ImageNet / CLIP / SAM preprocess step,
//                               and the inverse of "renormalise to model
//                               distribution" in any decoder head.
//
//   image_u8_to_f32_nhwc_to_nchw — convert a packed uint8 HWC image buffer
//                               (e.g. straight from a JPEG decoder) into a
//                               FP32 NCHW tensor, applying a single
//                               scale+bias pass: Y = src * scale + bias.
//                               Covers the typical scaling conventions:
//                                 [0,255] -> [0,1]   : scale=1/255, bias=0
//                                 [0,255] -> [-1,1]  : scale=2/255, bias=-1
//
// Both are FP32-only on CPU (matches the rest of the backend).
//
// `image_u8_to_f32_nhwc_to_nchw` takes a raw `const uint8_t*` host pointer —
// pixel data essentially always originates host-side from an image decoder,
// and forcing a Tensor wrapper around uint8 bytes would require either a
// new UINT8 dtype or misusing INT8 (signed range). The op-table signature
// matches `embedding_lookup_forward(const int32_t*)` in spirit.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void image_normalize(const ::brotensor::Tensor& X,
                     const ::brotensor::Tensor& mean,
                     const ::brotensor::Tensor& std_,
                     int N, int C, int H, int W,
                     ::brotensor::Tensor& Y) {
    check_fp32(X,    "image_normalize", "X");
    check_fp32(mean, "image_normalize", "mean");
    check_fp32(std_, "image_normalize", "std");
    if (mean.size() != C || std_.size() != C) {
        throw std::runtime_error("image_normalize: mean/std must have C elements");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("image_normalize: X shape mismatch");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    const float* mp = mean.host_f32();
    const float* sp = std_.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int c = 0; c < C; ++c) {
        const float mu = mp[c];
        const float s  = sp[c];
        if (s == 0.0f) {
            throw std::runtime_error("image_normalize: std[c] == 0");
        }
        const float inv = 1.0f / s;
        for (int n = 0; n < N; ++n) {
            const float* x_chan = Xp + (n * C + c) * spatial;
            float*       y_chan = Yp + (n * C + c) * spatial;
            for (int s_idx = 0; s_idx < spatial; ++s_idx) {
                y_chan[s_idx] = (x_chan[s_idx] - mu) * inv;
            }
        }
    }
}

void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  ::brotensor::Tensor& Y) {
    if (src == nullptr && N > 0 && H > 0 && W > 0 && C > 0) {
        throw std::runtime_error("image_u8_to_f32_nhwc_to_nchw: src is null");
    }
    if (N < 0 || H < 0 || W < 0 || C < 0) {
        throw std::runtime_error("image_u8_to_f32_nhwc_to_nchw: negative dim");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    float* Yp = Y.host_f32_mut();

    // src indexed (n, h, w, c)_packed; Y indexed (n, c, h, w)_NCHW.
    for (int n = 0; n < N; ++n) {
        const uint8_t* src_n = src + n * spatial * C;
        float*         y_n   = Yp  + n * C * spatial;
        for (int c = 0; c < C; ++c) {
            float* y_chan = y_n + c * spatial;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const uint8_t v = src_n[(h * W + w) * C + c];
                    y_chan[h * W + w] = static_cast<float>(v) * scale + bias;
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
