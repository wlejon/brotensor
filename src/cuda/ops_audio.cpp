// CUDA Wave 7 audio & codec ops via single-source brass MIR kernels
// (MirKind::Audio / MirKind::Transpose) + CUDA conv2d / matmul:
//   conv_transpose1d_{forward,backward_input,backward_weight,backward_bias}
//   causal_conv1d_update, pad1d_{forward,backward}
//   snake_{forward,backward}, resample1d_{forward,backward}
//   vq_encode_{forward,backward}, fsq_quantize_{forward,backward}

#include "cu_mir_cache.h"
#include "../kernels/mir_audio.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in conv2d.cu / concat.cu / matmul.cu / ops_elementwise.cpp.
void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, Tensor& Y);
void conv2d_backward_bias(const Tensor& dY, int N, int C_out, int H_out, int W_out, Tensor& dB);
void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n);
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch, int width, int height);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void elu_forward(const Tensor& x, float alpha, Tensor& y);
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX);
void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y);
void leaky_relu_backward(const Tensor& x, const Tensor& dY, float negative_slope, Tensor& dX);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(const char* op, Dtype dt) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

void ensure(Tensor& t, int rows, int cols, Dtype dt) {
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

std::int32_t i32(long long v) noexcept {
    return static_cast<std::int32_t>(v);
}

std::size_t esize(Dtype dt) noexcept {
    return static_cast<std::size_t>(::brotensor::dtype_size_bytes(dt));
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

void* byte_off(void* p, std::size_t bytes) noexcept {
    return static_cast<char*>(p) + bytes;
}

const void* byte_off(const void* p, std::size_t bytes) noexcept {
    return static_cast<const char*>(p) + bytes;
}

void run_audio(std::uint32_t code, Dtype dt, std::uint8_t flags,
               std::uint64_t items, bool per_group,
               std::span<const std::uintptr_t> ptrs, std::span<const std::int32_t> ints) {
    if (items == 0) return;
    CUfunction fn = mir_function(MirKey{MirKind::Audio, code, dt, dt, 1, dt, flags});
    const std::uint32_t gx = per_group
        ? static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535))
        : mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kAudioBlock);
    mir_dispatch_args(fn, gx, kernels::kAudioBlock, ptrs, ints, {});
}

int convt1d_out_len(int L, int stride, int padding, int output_padding, int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1) + output_padding + 1;
}

void check_convt1d(const char* op, int C_in, int C_out, int kL, int stride, int padding,
                   int output_padding, int dilation, int groups) {
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >=1 and divide both C_in and C_out");
    need(op, kL >= 1 && stride >= 1 && dilation >= 1 && padding >= 0 && output_padding >= 0,
         "kL/stride/dilation must be >=1 and padding/output_padding >=0");
    need(op, output_padding < stride || output_padding < dilation,
         "output_padding must be < stride or < dilation");
}

void check_pad(const char* op, int N, int C, int L, int pad_left, int pad_right, int mode) {
    need(op, N >= 0 && C >= 1 && L >= 1, "C/L must be >=1 and N >=0");
    need(op, pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    need(op, mode >= 0 && mode <= 2, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    need(op, mode != 1 || (pad_left < L && pad_right < L),
         "reflect padding requires pad_left and pad_right < L");
}

void pad1d_dispatch(std::uint32_t code, const Tensor& src, int N, int C, int L,
                    int pad_left, int pad_right, int mode, Tensor& dst) {
    const int L_pad = L + pad_left + pad_right;
    const std::uintptr_t ptrs[] = {addr(src.data), addr(dst.data)};
    const std::int32_t ints[] = {i32(L), i32(pad_left), i32(L_pad), i32(N * C)};
    run_audio(code, src.dtype, static_cast<std::uint8_t>(mode),
              static_cast<std::uint64_t>(N) * C * (code == AU_PAD ? L_pad : L),
              false, ptrs, ints);
}

void check_snake(const char* op, const Tensor& X, const Tensor& alpha, const Tensor* beta,
                 int N, int C, int L) {
    require_fp(op, X.dtype);
    need(op, N >= 0 && C >= 0 && L >= 0, "N, C, L must be non-negative");
    need(op, X.rows == N && X.cols == static_cast<long long>(C) * L,
         "X must be shaped (N, C*L)");
    need(op, alpha.dtype == Dtype::FP32 && alpha.size() == C,
         "alpha must have C FP32 elements");
    need(op, !beta || (beta->dtype == Dtype::FP32 && beta->size() == C),
         "beta must have C FP32 elements");
}

void check_resample(const char* op, int N, int C, int L_in, int L_out, int mode) {
    need(op, N >= 0 && C >= 0 && L_in >= 0 && L_out >= 0,
         "N, C, L_in, L_out must be non-negative");
    need(op, mode == 0 || mode == 1, "mode must be 0 (nearest) or 1 (linear)");
    need(op, !(L_out > 0 && L_in == 0), "L_in must be > 0 when L_out > 0");
}

void straight_through(const char* op, const Tensor& dQ, Tensor& dX) {
    need(op, dQ.dtype == Dtype::FP32, "dQuantized must be FP32");
    ensure(dX, dQ.rows, dQ.cols, Dtype::FP32);
    if (dQ.size() == 0 || dX.data == dQ.data) return;
    copy_d2d(dQ, 0, dX, 0, static_cast<int>(dQ.size()));
}

}  // namespace

// Shared with ops_spectral.cpp (STFT center reflection padding).
void audio_pad1d(const Tensor& X, int rows, int L, int pad_left, int pad_right,
                 int mode, Tensor& Y) {
    pad1d_dispatch(AU_PAD, X, rows, 1, L, pad_left, pad_right, mode, Y);
}

// ─── conv_transpose1d ───────────────────────────────────────────────────────

void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL, int stride,
                              int padding, int output_padding, int dilation, int groups,
                              Tensor& Y) {
    constexpr const char* op = "conv_transpose1d_forward";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    const Dtype dt = X.dtype;
    require_fp(op, dt);
    need(op, Wt.dtype == dt && (!bias || bias->dtype == dt),
         "Wt and bias must share X's dtype");
    need(op, Wt.rows == C_in && Wt.cols == C_out / groups * kL,
         "Wt shape must be (C_in, (C_out/groups)*kL)");
    need(op, !bias || (bias->rows == C_out && bias->cols == 1),
         "bias shape must be (C_out, 1)");
    need(op, N >= 0 && L >= 1, "N must be >= 0 and L >= 1");
    need(op, X.size() >= static_cast<long long>(N) * C_in * L,
         "X is smaller than N*C_in*L");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    ensure(Y, N, C_out * L_out, dt);
    if (N == 0) return;

    const int Cgi = C_in / groups;
    const int Cgo = C_out / groups;
    const int M = Cgo * kL;
    const std::size_t es = esize(dt);
    const std::uint64_t per_image = static_cast<std::uint64_t>(C_out) * kL * L;
    const int nb = static_cast<int>(std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(N, (256ull << 20) / (per_image * es))));

    Tensor Wt_T = Tensor::empty_on(Wt.device, groups * M, Cgi, dt);
    CUfunction tr_fn = mir_function({MirKind::Transpose, static_cast<std::uint32_t>(es)});
    const std::uint32_t tr_gx = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>((M + 31) / 32));
    const std::uint32_t tr_gy = std::clamp(static_cast<std::uint32_t>((Cgi + 31) / 32), 1u, 65535u);
    for (int g = 0; g < groups; ++g) {
        mir_dispatch_transpose(
            tr_fn, tr_gx, tr_gy, 1u,
            byte_off(Wt.data, static_cast<std::size_t>(g) * Cgi * M * es),
            byte_off(Wt_T.data, static_cast<std::size_t>(g) * M * Cgi * es),
            1u, static_cast<std::uint32_t>(Cgi), static_cast<std::uint32_t>(M));
    }

    Tensor cols = Tensor::empty_on(X.device, nb, static_cast<int>(per_image), dt);
    for (int n0 = 0; n0 < N; n0 += nb) {
        const int nn = std::min(nb, N - n0);
        for (int n = 0; n < nn; ++n) {
            for (int g = 0; g < groups; ++g) {
                const Tensor w_v = Tensor::view(
                    Wt_T.device,
                    byte_off(Wt_T.data, static_cast<std::size_t>(g) * M * Cgi * es),
                    M, Cgi, dt);
                const Tensor x_v = Tensor::view(
                    X.device,
                    const_cast<void*>(byte_off(
                        X.data, (static_cast<std::size_t>(n0 + n) * C_in +
                                 static_cast<std::size_t>(g) * Cgi) * L * es)),
                    Cgi, L, dt);
                Tensor c_v = Tensor::view(
                    cols.device,
                    byte_off(cols.data, (static_cast<std::size_t>(n) * per_image +
                                         static_cast<std::size_t>(g) * M * L) * es),
                    M, L, dt);
                matmul(w_v, x_v, c_v);
            }
        }
        const std::uintptr_t ptrs[] = {
            addr(cols.data),
            bias ? addr(bias->data) : 0,
            addr(byte_off(Y.data, static_cast<std::size_t>(n0) * C_out * L_out * es)),
        };
        const std::int32_t ints[] = {
            i32(nn), i32(C_out), i32(L), i32(kL), i32(stride),
            i32(padding), i32(dilation), i32(Cgo), i32(L_out),
        };
        const std::uint8_t flags = static_cast<std::uint8_t>(bias ? 1u : 0u);
        run_audio(AU_COL2IM, dt, flags, static_cast<std::uint64_t>(nn) * C_out * L_out,
                  false, ptrs, ints);
    }
}

void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY, int N, int C_in, int L,
                                     int C_out, int kL, int stride, int padding,
                                     int output_padding, int dilation, int groups, Tensor& dX) {
    constexpr const char* op = "conv_transpose1d_backward_input";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    require_fp(op, dY.dtype);
    need(op, Wt.dtype == dY.dtype, "Wt dtype must match dY");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    need(op, Wt.rows == C_in && Wt.cols == C_out / groups * kL,
         "Wt shape must be (C_in, (C_out/groups)*kL)");
    need(op, dY.rows == N && dY.cols == C_out * L_out,
         "dY shape must be (N, C_out*L_out)");
    ensure(dX, N, C_in * L, dY.dtype);
    if (N == 0 || C_in * L == 0) return;
    const int Lc = (L_out + 2 * padding - dilation * (kL - 1) - 1) / stride + 1;
    if (Lc == L) {
        conv2d_forward(dY, Wt, nullptr, N, C_out, 1, L_out, C_in, 1, kL,
                       1, stride, 0, padding, 1, dilation, groups, dX);
        return;
    }
    Tensor t = Tensor::empty_on(dY.device, N, C_in * Lc, dY.dtype);
    conv2d_forward(dY, Wt, nullptr, N, C_out, 1, L_out, C_in, 1, kL,
                   1, stride, 0, padding, 1, dilation, groups, t);
    copy_d2d_strided(t, 0, Lc, dX, 0, L, L, N * C_in);
}

void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY, int N, int C_in, int L,
                                      int C_out, int kL, int stride, int padding,
                                      int output_padding, int dilation, int groups, Tensor& dWt) {
    constexpr const char* op = "conv_transpose1d_backward_weight";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    require_fp(op, X.dtype);
    need(op, dY.dtype == X.dtype && dWt.dtype == X.dtype,
         "X, dY and dWt must share one dtype");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    const int Cg_out = C_out / groups;
    need(op, dWt.rows == C_in && dWt.cols == Cg_out * kL,
         "dWt shape must be (C_in, (C_out/groups)*kL)");
    need(op, X.rows == N && X.cols == C_in * L, "X shape must be (N, C_in*L)");
    need(op, dY.rows == N && dY.cols == C_out * L_out,
         "dY shape must be (N, C_out*L_out)");
    if (C_in == 0 || Cg_out == 0 || N == 0) return;
    const std::uintptr_t ptrs[] = {addr(X.data), addr(dY.data), addr(dWt.data)};
    const std::int32_t ints[] = {
        i32(N), i32(C_in), i32(L), i32(C_out), i32(kL), i32(stride),
        i32(padding), i32(dilation), i32(C_in / groups), i32(Cg_out), i32(L_out),
    };
    run_audio(AU_CONVT_BWD_W, X.dtype, 0, static_cast<std::uint64_t>(C_in) * Cg_out * kL,
              true, ptrs, ints);
}

void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out, int L_out, Tensor& dB) {
    constexpr const char* op = "conv_transpose1d_backward_bias";
    need(op, dB.rows == C_out && dB.cols == 1 && dB.dtype == dY.dtype,
         "dB shape must be (C_out, 1) of dY's dtype");
    need(op, dY.rows == N && dY.cols == static_cast<long long>(C_out) * L_out,
         "dY shape must be (N, C_out*L_out)");
    conv2d_backward_bias(dY, N, C_out, 1, L_out, dB);
}

void causal_conv1d_update(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C,
                          int L_step, int kL, int dilation, Tensor& state, Tensor& Y) {
    constexpr const char* op = "causal_conv1d_update";
    require_fp(op, X.dtype);
    for (const Tensor* t : std::initializer_list<const Tensor*>{&Wt, bias, &state}) {
        need(op, !t || t->dtype == X.dtype, "all operands must share X's dtype");
    }
    need(op, kL >= 1 && dilation >= 1 && L_step >= 1 && N >= 0 && C >= 1,
         "kL/dilation/L_step/C must be >=1 and N >=0");
    need(op, Wt.rows == C && Wt.cols == kL,
         "Wt shape must be (C, kL) — one depthwise filter per channel");
    need(op, !bias || (bias->rows == C && bias->cols == 1), "bias shape must be (C, 1)");
    const int hist = (kL - 1) * dilation;
    need(op, state.rows == N && state.cols == C * hist,
         "state shape must be (N, C*(kL-1)*dilation)");
    need(op, X.size() >= static_cast<long long>(N) * C * L_step,
         "X is smaller than N*C*L_step");
    ensure(Y, N, C * L_step, X.dtype);
    if (N == 0) return;
    const std::uintptr_t ptrs[] = {
        addr(X.data), addr(Wt.data), bias ? addr(bias->data) : 0, addr(state.data), addr(Y.data)};
    const std::int32_t ints[] = {i32(C), i32(L_step), i32(kL), i32(dilation), i32(N * C)};
    run_audio(AU_CAUSAL, X.dtype, static_cast<std::uint8_t>(bias ? 1u : 0u),
              static_cast<std::uint64_t>(N) * C, false, ptrs, ints);
}

void pad1d_forward(const Tensor& X, int N, int C, int L, int pad_left, int pad_right,
                   int mode, Tensor& Y) {
    constexpr const char* op = "pad1d_forward";
    require_fp(op, X.dtype);
    check_pad(op, N, C, L, pad_left, pad_right, mode);
    need(op, X.rows == N && X.cols == C * L, "X shape must be (N, C*L)");
    ensure(Y, N, C * (L + pad_left + pad_right), X.dtype);
    if (N == 0) return;
    pad1d_dispatch(AU_PAD, X, N, C, L, pad_left, pad_right, mode, Y);
}

void pad1d_backward(const Tensor& dY, int N, int C, int L, int pad_left, int pad_right,
                    int mode, Tensor& dX) {
    constexpr const char* op = "pad1d_backward";
    require_fp(op, dY.dtype);
    check_pad(op, N, C, L, pad_left, pad_right, mode);
    need(op, dY.rows == N && dY.cols == C * (L + pad_left + pad_right),
         "dY shape must be (N, C*(L+pad_left+pad_right))");
    ensure(dX, N, C * L, dY.dtype);
    if (N == 0) return;
    pad1d_dispatch(AU_PAD_BWD, dY, N, C, L, pad_left, pad_right, mode, dX);
}

// ─── Snake ──────────────────────────────────────────────────────────────────

void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y) {
    constexpr const char* op = "snake_forward";
    check_snake(op, X, alpha, beta, N, C, L);
    ensure(Y, N, C * L, X.dtype);
    if (N == 0 || C * L == 0) return;
    const std::uint32_t total = static_cast<std::uint32_t>(static_cast<long long>(N) * C * L);
    const std::uintptr_t ptrs[] = {
        addr(X.data), addr(alpha.data), beta ? addr(beta->data) : 0, addr(Y.data)};
    const std::int32_t ints[] = {i32(C), i32(L), static_cast<std::int32_t>(total)};
    run_audio(AU_SNAKE, X.dtype, static_cast<std::uint8_t>(beta ? 1u : 0u),
              total, false, ptrs, ints);
}

void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta, const Tensor& dY,
                    int N, int C, int L, Tensor& dX, Tensor& dAlpha, Tensor* dBeta) {
    constexpr const char* op = "snake_backward";
    check_snake(op, X, alpha, beta, N, C, L);
    need(op, (beta == nullptr) == (dBeta == nullptr),
         "dBeta must be non-null exactly when beta is non-null");
    need(op, dY.dtype == X.dtype && dY.rows == N && dY.cols == static_cast<long long>(C) * L,
         "dY must be shaped (N, C*L) in X's dtype");
    need(op, dAlpha.rows == C && dAlpha.cols == 1 && dAlpha.dtype == Dtype::FP32,
         "dAlpha must be (C, 1) FP32");
    need(op, !dBeta || (dBeta->rows == C && dBeta->cols == 1 && dBeta->dtype == Dtype::FP32),
         "dBeta must be (C, 1) FP32");
    ensure(dX, N, C * L, X.dtype);
    if (N == 0 || C * L == 0) return;
    const std::uintptr_t ptrs[] = {
        addr(X.data), addr(alpha.data), beta ? addr(beta->data) : 0, addr(dY.data),
        addr(dX.data), addr(dAlpha.data), dBeta ? addr(dBeta->data) : 0,
    };
    const std::int32_t ints[] = {i32(C), i32(L), i32(N)};
    run_audio(AU_SNAKE_BWD, X.dtype, static_cast<std::uint8_t>(beta ? 1u : 0u),
              static_cast<std::uint64_t>(C), true, ptrs, ints);
}

// ─── resample1d ─────────────────────────────────────────────────────────────

void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out,
                        int mode, Tensor& Y) {
    constexpr const char* op = "resample1d_forward";
    require_fp(op, X.dtype);
    check_resample(op, N, C, L_in, L_out, mode);
    need(op, X.size() >= static_cast<long long>(N) * C * L_in,
         "X is smaller than N*C*L_in");
    ensure(Y, N, C * L_out, X.dtype);
    if (N == 0 || C * L_out == 0) return;
    const std::uintptr_t ptrs[] = {addr(X.data), addr(Y.data)};
    const std::int32_t ints[] = {i32(L_in), i32(L_out), i32(N * C)};
    run_audio(AU_RESAMPLE, X.dtype, static_cast<std::uint8_t>(mode),
              static_cast<std::uint64_t>(N) * C * L_out, false, ptrs, ints);
}

void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out,
                         int mode, Tensor& dX) {
    constexpr const char* op = "resample1d_backward";
    require_fp(op, dY.dtype);
    check_resample(op, N, C, L_in, L_out, mode);
    need(op, dY.size() >= static_cast<long long>(N) * C * L_out,
         "dY is smaller than N*C*L_out");
    ensure(dX, N, C * L_in, dY.dtype);
    if (N == 0 || C * L_in == 0) return;
    if (L_out == 0) {
        dX.zero();
        return;
    }
    const std::uintptr_t ptrs[] = {addr(dY.data), addr(dX.data)};
    const std::int32_t ints[] = {i32(L_in), i32(L_out), i32(N * C)};
    run_audio(AU_RESAMPLE_BWD, dY.dtype, static_cast<std::uint8_t>(mode),
              static_cast<std::uint64_t>(N) * C * L_in, false, ptrs, ints);
}

// ─── codec quantisers ───────────────────────────────────────────────────────

void vq_encode_forward(const Tensor& x, const Tensor& codebook,
                       Tensor& indices, Tensor& quantized) {
    constexpr const char* op = "vq_encode_forward";
    need(op, x.dtype == Dtype::FP32 && codebook.dtype == Dtype::FP32,
         "x and codebook must be FP32");
    const int N = x.rows, D = x.cols, K = codebook.rows;
    need(op, codebook.cols == D, "codebook must have the same column count as x");
    need(op, !(K == 0 && N != 0), "codebook must have at least one codeword");
    ensure(indices, N, 1, Dtype::INT32);
    ensure(quantized, N, D, Dtype::FP32);
    if (N == 0) return;
    const std::uintptr_t ptrs[] = {
        addr(x.data), addr(codebook.data), addr(quantized.data), addr(indices.data)};
    const std::int32_t ints[] = {i32(N), i32(D), i32(K)};
    run_audio(AU_VQ, Dtype::FP32, 0, static_cast<std::uint64_t>(N), true, ptrs, ints);
}

void fsq_quantize_forward(const Tensor& x, const Tensor& levels,
                          Tensor& quantized, Tensor& packed_indices) {
    constexpr const char* op = "fsq_quantize_forward";
    need(op, x.dtype == Dtype::FP32, "x must be FP32");
    need(op, levels.dtype == Dtype::INT32, "levels must be INT32");
    const int N = x.rows, D = x.cols;
    need(op, levels.size() == D, "levels must have D elements (one per column of x)");
    std::vector<std::int32_t> lv(static_cast<std::size_t>(D));
    if (D > 0) levels.copy_to_host_raw(lv.data(), lv.size() * sizeof(std::int32_t));
    for (std::int32_t l : lv) need(op, l >= 2, "every level count must be >= 2");
    ensure(quantized, N, D, Dtype::FP32);
    ensure(packed_indices, N, 1, Dtype::INT32);
    if (N == 0 || D == 0) return;
    const std::uintptr_t ptrs[] = {
        addr(x.data), addr(levels.data), addr(quantized.data), addr(packed_indices.data)};
    const std::int32_t ints[] = {i32(N), i32(D)};
    run_audio(AU_FSQ, Dtype::FP32, 0, static_cast<std::uint64_t>(N), false, ptrs, ints);
}

void vq_encode_backward(const Tensor& dQ, Tensor& dX) {
    straight_through("vq_encode_backward", dQ, dX);
}

void fsq_quantize_backward(const Tensor& dQ, Tensor& dX) {
    straight_through("fsq_quantize_backward", dQ, dX);
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_conv1d(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose1d_forward = &conv_transpose1d_forward;
    v.conv_transpose1d_backward_input = &conv_transpose1d_backward_input;
    v.conv_transpose1d_backward_weight = &conv_transpose1d_backward_weight;
    v.conv_transpose1d_backward_bias = &conv_transpose1d_backward_bias;
    v.causal_conv1d_update = &causal_conv1d_update;
    v.pad1d_forward = &pad1d_forward;
    v.pad1d_backward = &pad1d_backward;
}

void fill_cuda_vtable_vocoder(::brotensor::detail::OpsVTable& v) {
    v.snake_forward = &snake_forward;
    v.snake_backward = &snake_backward;
    v.elu_forward = &elu_forward;
    v.elu_backward = &elu_backward;
    v.leaky_relu_forward = &leaky_relu_forward;
    v.leaky_relu_backward = &leaky_relu_backward;
}

void fill_cuda_vtable_resample1d(::brotensor::detail::OpsVTable& v) {
    v.resample1d_forward = &resample1d_forward;
    v.resample1d_backward = &resample1d_backward;
}

void fill_cuda_vtable_codec_quant(::brotensor::detail::OpsVTable& v) {
    v.vq_encode_forward = &vq_encode_forward;
    v.vq_encode_backward = &vq_encode_backward;
    v.fsq_quantize_forward = &fsq_quantize_forward;
    v.fsq_quantize_backward = &fsq_quantize_backward;
}

}  // namespace brotensor::detail::cuda
