// ─── CPU spatial 2x2 pixel-unshuffle (Qwen-VL merger / Flux.2 VAE) ──────────
//
// Pure gather. Stacks each 2x2 spatial block into the channel axis:
//   X: NCHW (N, C*H*W)             — H, W must be even.
//   Y: NCHW (N, 4*C*(H/2)*(W/2))
//   block = dh*2 + dw,             (h_in, w_in) = (2*h_out + dh, 2*w_out + dw).
//   channel_major=false: c_out = block*C + c_in  (Qwen-VL, block-major)
//   channel_major=true:  c_out = c_in*4 + block  (torch pixel_unshuffle / Flux.2)
// FP32 only on CPU (CPU backend is FP32-only by convention).

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

void spatial_merge_2x2_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               bool channel_major,
                               ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP32) {
        throw std::runtime_error("spatial_merge_2x2_forward: X must be FP32 "
                                 "(CPU backend is FP32-only)");
    }
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: negative dimension");
    }
    if ((H & 1) != 0 || (W & 1) != 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: H and W must be even");
    }
    const int H_out = H / 2;
    const int W_out = W / 2;
    const int C_out = 4 * C;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    const int HW = H * W;
    const int HW_out = H_out * W_out;

    for (int n = 0; n < N; ++n) {
        for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
                const int block = dh * 2 + dw;
                for (int c_in = 0; c_in < C; ++c_in) {
                    const int c_out =
                        channel_major ? c_in * 4 + block : block * C + c_in;
                    const float* xc = Xp + (n * C + c_in) * HW;
                    float*       yc = Yp + (n * C_out + c_out) * HW_out;
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        const int h_in = 2 * h_out + dh;
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            const int w_in = 2 * w_out + dw;
                            yc[h_out * W_out + w_out] = xc[h_in * W + w_in];
                        }
                    }
                }
            }
        }
    }
}

// ─── CPU DC-AE up-shortcut: repeat_interleave + 2x pixel-shuffle ────────────
//
// Y[n, c_out, 2h+i, 2w+j] = X[n, (4*c_out + 2i + j)/repeats, h, w],
// repeats = 4*C_out/C_in. See ops/spatial.h. FP32-only on CPU.

void pixel_shuffle_upsample_2x_forward(const ::brotensor::Tensor& X,
                                       int N, int C_in, int H, int W,
                                       int C_out,
                                       ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP32) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: X must be "
                                 "FP32 (CPU backend is FP32-only)");
    }
    if (N < 0 || C_in <= 0 || H < 0 || W < 0 || C_out <= 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: bad dimension");
    }
    if ((4 * C_out) % C_in != 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: C_in must "
                                 "divide 4*C_out");
    }
    const int repeats = (4 * C_out) / C_in;
    const int H_out = 2 * H;
    const int W_out = 2 * W;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || H == 0 || W == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();
    const int HW     = H * W;
    const int HW_out = H_out * W_out;

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float* yc = Yp + (static_cast<std::size_t>(n) * C_out + c_out) * HW_out;
            for (int h_out = 0; h_out < H_out; ++h_out) {
                const int i = h_out & 1;
                const int h = h_out >> 1;
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    const int j = w_out & 1;
                    const int w = w_out >> 1;
                    const int src_c = (4 * c_out + 2 * i + j) / repeats;
                    const float* xc =
                        Xp + (static_cast<std::size_t>(n) * C_in + src_c) * HW;
                    yc[h_out * W_out + w_out] = xc[h * W + w];
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
