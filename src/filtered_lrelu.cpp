// ─── filtered_lrelu (StyleGAN3 alias-free nonlinearity) ─────────────────────
//
// The public entry points are thin dispatchers: if the resolved backend
// registered a fused `filtered_lrelu_forward/backward` vtable slot (CUDA does),
// they call it; otherwise they fall back to the device-agnostic COMPOSITE below
// — a sequence of the public bias_act + upfirdn2d ops, which run on whatever
// backend the operands live on (this is the only path on CPU/Metal). The
// composite mirrors NVlabs `_filtered_lrelu_ref` EXACTLY, including order:
//
//   x = bias_act(x, b)                              # apply channel bias
//   x = upfirdn2d(x, fu, up=up,  pad=p, gain=up^2)  # upsample
//   x = bias_act(x, act=lrelu, gain, clamp)         # bias-free lrelu+clamp
//   x = upfirdn2d(x, fd, down=down)                 # downsample
//
// The bias is applied BEFORE the upsample (a linear bias_act), and the
// post-upsample bias_act carries only the lrelu — this is not interchangeable
// with biasing after the upsample. The backward reverses the chain.
//
// up_buf/act_buf are returned as caches: up_buf (the post-upsample tensor) is
// the input to the lrelu bias_act and is required by the backward; act_buf is
// kept for symmetry with the fused-kernel surface. A fused backend that does
// not reproduce these caches must keep whatever its own backward consumes —
// the cache contract is per-backend (see the CUDA kernel).

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <stdexcept>

namespace brotensor {

// Composite fallback, also reused by the CUDA backend for configs its fused
// kernel does not cover. Declared here (not in the public header) and called
// from src/cuda/filtered_lrelu.cu via a matching forward declaration.
void filtered_lrelu_forward_composite(const Tensor& X, const Tensor& fu,
                                      const Tensor& fd, const Tensor* b,
                                      int N, int C, int H, int W, int up, int down,
                                      int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                                      float gain, float slope, float clamp,
                                      Tensor& up_buf, Tensor& act_buf, Tensor& Y);
void filtered_lrelu_backward_composite(const Tensor& dY, const Tensor& X,
                                       const Tensor& fu, const Tensor& fd,
                                       const Tensor* b, int N, int C, int H, int W,
                                       int up, int down, int pad_x0, int pad_x1,
                                       int pad_y0, int pad_y1, float gain, float slope,
                                       float clamp, const Tensor& up_buf,
                                       Tensor& dX, Tensor* dB);

namespace {

constexpr int ACT_LINEAR = 0;
constexpr int ACT_LRELU  = 1;
constexpr float NO_CLAMP = -1.0f;

// Up-stage upfirdn2d output extent (down=1): (in*up + pad0 + pad1 - f) + 1.
inline int up_out(int in, int up, int pad0, int pad1, int fdim) {
    return in * up + pad0 + pad1 - fdim + 1;
}

} // namespace

// ─── public dispatchers ─────────────────────────────────────────────────────

void filtered_lrelu_forward(const Tensor& X, const Tensor& fu, const Tensor& fd,
                            const Tensor* b, int N, int C, int H, int W,
                            int up, int down, int pad_x0, int pad_x1,
                            int pad_y0, int pad_y1, float gain, float slope,
                            float clamp, Tensor& up_buf, Tensor& act_buf,
                            Tensor& Y) {
    const auto& v = detail::dispatch(X, fu, fd, up_buf, act_buf, Y);
    if (v.filtered_lrelu_forward) {
        detail::adopt_output(up_buf, X.device);
        detail::adopt_output(act_buf, X.device);
        detail::adopt_output(Y, X.device);
        v.filtered_lrelu_forward(X, fu, fd, b, N, C, H, W, up, down,
                                 pad_x0, pad_x1, pad_y0, pad_y1, gain, slope,
                                 clamp, up_buf, act_buf, Y);
        return;
    }
    filtered_lrelu_forward_composite(X, fu, fd, b, N, C, H, W, up, down,
                                     pad_x0, pad_x1, pad_y0, pad_y1, gain, slope,
                                     clamp, up_buf, act_buf, Y);
}

void filtered_lrelu_backward(const Tensor& dY, const Tensor& X,
                             const Tensor& fu, const Tensor& fd,
                             const Tensor* b, int N, int C, int H, int W,
                             int up, int down, int pad_x0, int pad_x1,
                             int pad_y0, int pad_y1, float gain, float slope,
                             float clamp, const Tensor& up_buf,
                             Tensor& dX, Tensor* dB) {
    const auto& v = detail::dispatch(dY, X, fu, fd, up_buf, dX);
    if (v.filtered_lrelu_backward) {
        detail::adopt_output(dX, X.device);
        if (dB) detail::adopt_output(*dB, X.device);
        v.filtered_lrelu_backward(dY, X, fu, fd, b, N, C, H, W, up, down,
                                  pad_x0, pad_x1, pad_y0, pad_y1, gain, slope,
                                  clamp, up_buf, dX, dB);
        return;
    }
    filtered_lrelu_backward_composite(dY, X, fu, fd, b, N, C, H, W, up, down,
                                      pad_x0, pad_x1, pad_y0, pad_y1, gain, slope,
                                      clamp, up_buf, dX, dB);
}

// ─── composite implementation (fallback / CPU / Metal) ──────────────────────

void filtered_lrelu_forward_composite(const Tensor& X, const Tensor& fu, const Tensor& fd,
                            const Tensor* b, int N, int C, int H, int W,
                            int up, int down, int pad_x0, int pad_x1,
                            int pad_y0, int pad_y1, float gain, float slope,
                            float clamp, Tensor& up_buf, Tensor& act_buf,
                            Tensor& Y) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;

    // 1. Channel bias (linear bias_act), at the input rate.
    Tensor pre;
    bias_act_forward(X, b, N, C, H * W, ACT_LINEAR, 0.0f, 1.0f, NO_CLAMP, pre);

    // 2. Upsample with gain = up^2.
    upfirdn2d_forward(pre, fu, N, C, H, W, fuH, fuW,
                      up, up, 1, 1, pad_x0, pad_x1, pad_y0, pad_y1,
                      /*flip=*/false, static_cast<float>(up * up), up_buf);

    const int Huo = up_out(H, up, pad_y0, pad_y1, fuH);
    const int Wuo = up_out(W, up, pad_x0, pad_x1, fuW);
    if (up_buf.cols != C * Huo * Wuo) {
        throw std::runtime_error("filtered_lrelu_forward: upsample dim mismatch");
    }

    // 3. Bias-free leaky ReLU + clamp at the 2x rate.
    bias_act_forward(up_buf, nullptr, N, C, Huo * Wuo, ACT_LRELU, slope,
                     gain, clamp, act_buf);

    // 4. Downsample (gain 1, no padding).
    upfirdn2d_forward(act_buf, fd, N, C, Huo, Wuo, fdH, fdW,
                      1, 1, down, down, 0, 0, 0, 0,
                      /*flip=*/false, 1.0f, Y);
}

void filtered_lrelu_backward_composite(const Tensor& dY, const Tensor& X,
                             const Tensor& fu, const Tensor& fd,
                             const Tensor* b, int N, int C, int H, int W,
                             int up, int down, int pad_x0, int pad_x1,
                             int pad_y0, int pad_y1, float gain, float slope,
                             float clamp, const Tensor& up_buf,
                             Tensor& dX, Tensor* dB) {
    const int fuH = fu.rows, fuW = fu.cols;
    const int fdH = fd.rows, fdW = fd.cols;
    const int Huo = up_out(H, up, pad_y0, pad_y1, fuH);
    const int Wuo = up_out(W, up, pad_x0, pad_x1, fuW);

    // up_buf is the post-upsample (pre-lrelu) tensor — the lrelu backward needs
    // it. The fused forward skips producing it (it's the buffer fusion avoids),
    // so when the caller hands us an uncommitted up_buf we recompute it here
    // from X exactly as the forward did (bias → up-FIR). The cache is thus
    // optional: a populated up_buf is used directly, an empty one is rebuilt.
    Tensor up_buf_local;
    const Tensor* up_ptr = &up_buf;
    if (up_buf.data == nullptr) {
        Tensor pre;
        bias_act_forward(X, b, N, C, H * W, ACT_LINEAR, 0.0f, 1.0f, NO_CLAMP, pre);
        upfirdn2d_forward(pre, fu, N, C, H, W, fuH, fuW,
                          up, up, 1, 1, pad_x0, pad_x1, pad_y0, pad_y1,
                          /*flip=*/false, static_cast<float>(up * up), up_buf_local);
        up_ptr = &up_buf_local;
    }

    // 4'. Through the downsample.
    Tensor d_act;
    upfirdn2d_backward(dY, fd, N, C, Huo, Wuo, fdH, fdW,
                       1, 1, down, down, 0, 0, 0, 0,
                       /*flip=*/false, 1.0f, d_act);

    // 3'. Through the leaky ReLU (no bias gradient here).
    Tensor d_up;
    bias_act_backward(d_act, *up_ptr, nullptr, N, C, Huo * Wuo, ACT_LRELU, slope,
                      gain, clamp, d_up, nullptr);

    // 2'. Through the upsample.
    Tensor d_pre;
    upfirdn2d_backward(d_up, fu, N, C, H, W, fuH, fuW,
                       up, up, 1, 1, pad_x0, pad_x1, pad_y0, pad_y1,
                       /*flip=*/false, static_cast<float>(up * up), d_pre);

    // 1'. Through the channel bias — dX (overwrite) and dB (accumulate).
    bias_act_backward(d_pre, X, b, N, C, H * W, ACT_LINEAR, 0.0f, 1.0f,
                      NO_CLAMP, dX, dB);
}

} // namespace brotensor
