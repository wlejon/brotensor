// Vulkan audio / codec ops for brosoundml: the 1D convolution family
// (conv_transpose1d and its three backwards, causal_conv1d_update, pad1d),
// Snake / SnakeBeta, resample1d and the codec quantisers (VQ, FSQ).
// Contracts follow the CPU backend (src/cpu/conv1d.cpp, vocoder_activations.cpp,
// resample1d.cpp, codec_quant.cpp), widened to FP16 / BF16 signals where the
// op is pure data movement or elementwise (the per-channel Snake parameters
// and their gradients stay FP32; VQ / FSQ are FP32 as on every backend).
//
// The transposed convolution is a GEMM (W^T X, all kL taps of every input
// sample) and an overlap-add gather (AU_COL2IM); its input gradient is an ordinary
// strided, dilated, grouped conv1d of dY with the same weights (conv2d(), so
// the implicit-GEMM paths apply). Everything else is audio.comp.

#include "detail/gemm.h"
#include "detail/kernels.h"
#include "detail/spatial.h"
#include "shaders/op_codes.h"
#include "vk_mir_cache.h"
#include "../kernels/mir_audio.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void conv2d_backward_bias(const Tensor& dY, int N, int C_out, int H_out, int W_out, Tensor& dB);   // ops_conv.cpp
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch, Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height);                              // ops_copy.cpp

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void ensure(Tensor& t, int rows, int cols, Dtype dt) {
    if (t.rows != rows || t.cols != cols || t.dtype != dt) t.resize(rows, cols, dt);
}

std::int32_t i32(long long v) { return static_cast<std::int32_t>(v); }

void run_audio(DeviceCtx& d, std::uint32_t op, Dtype dt, std::uint8_t flags, std::uint64_t items, bool per_group,
               std::span<const std::uint64_t> ptrs, std::span<const std::int32_t> ints) {
    if (items == 0) return;
    const MirKernel mk = mir_kernel(d, MirKey{MirKind::Audio, op, dt, dt, 1, dt, flags});
    const std::uint32_t gx = per_group
        ? static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535))
        : mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kAudioBlock);
    mir_dispatch_args(d, mk, gx, ptrs, ints, {});
}

int convt1d_out_len(int L, int stride, int padding, int output_padding, int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1) + output_padding + 1;
}

void check_convt1d(const char* op, int C_in, int C_out, int kL, int stride, int padding, int output_padding,
                   int dilation, int groups) {
    need(op, groups >= 1 && C_in % groups == 0 && C_out % groups == 0,
         "groups must be >=1 and divide both C_in and C_out");
    need(op, kL >= 1 && stride >= 1 && dilation >= 1 && padding >= 0 && output_padding >= 0,
         "kL/stride/dilation must be >=1 and padding/output_padding >=0");
    need(op, output_padding < stride || output_padding < dilation, "output_padding must be < stride or < dilation");
}

// ─── conv_transpose1d ───────────────────────────────────────────────────────

// Y = col2im(W^T X): per image and group, cols (Cg_out kL, L) = Wt_g^T
// (Cg_out kL x Cg_in, stored (Cg_in, Cg_out kL)) X_g (Cg_in x L) through the
// GEMM (cooperative matrix for FP16), then each output sample gathers its
// kL / stride column products (AU_COL2IM). cols is in the signal's dtype, FP32
// for BF16 (rounding each product to BF16 before the overlap-add cost a few
// BF16 ulp); images go in batches that keep it under 256 MiB.
void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C_in, int L,
                              int C_out, int kL, int stride, int padding, int output_padding, int dilation, int groups,
                              Tensor& Y) {
    const char* op = "conv_transpose1d_forward";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, Wt.dtype == dt && (!bias || bias->dtype == dt), "Wt and bias must share X's dtype");
    need(op, Wt.rows == C_in && Wt.cols == C_out / groups * kL, "Wt shape must be (C_in, (C_out/groups)*kL)");
    need(op, !bias || (bias->rows == C_out && bias->cols == 1), "bias shape must be (C_out, 1)");
    need(op, N >= 0 && L >= 1, "N must be >= 0 and L >= 1");
    need(op, X.size() >= static_cast<long long>(N) * C_in * L, "X is smaller than N*C_in*L");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    ensure(Y, N, C_out * L_out, dt);
    if (N == 0) return;
    DeviceCtx& d = device_of(X);
    const int Cgi = C_in / groups, Cgo = C_out / groups, M = Cgo * kL;
    const std::uint64_t es = dt == Dtype::FP32 ? 4u : 2u;
    const Dtype cdt = dt == Dtype::BF16 ? Dtype::FP32 : dt;
    const std::uint64_t ces = cdt == Dtype::FP32 ? 4u : 2u;
    const std::uint64_t per_image = std::uint64_t(C_out) * kL * L;
    const int nb = static_cast<int>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(N, (256ull << 20) / (per_image * ces))));
    Tensor cols = Tensor::empty_on(Device::vulkan(d.index()), nb, static_cast<int>(per_image), cdt);
    for (int n0 = 0; n0 < N; n0 += nb) {
        const int nn = std::min(nb, N - n0);
        for (int g = 0; g < groups; ++g) {
            GemmArgs gm;
            gm.op = op;
            gm.da = gm.db = dt;
            gm.dc = cdt;
            gm.a = addr(Wt.data) + std::uint64_t(g) * Cgi * M * es;
            gm.ta = true;
            gm.lda = M;
            gm.b = addr(X.data) + (std::uint64_t(n0) * C_in + std::uint64_t(g) * Cgi) * L * es;
            gm.nb = true;
            gm.ldb = L;
            gm.c = addr(cols.data) + std::uint64_t(g) * M * L * ces;
            gm.ldc = L;
            gm.m = M; gm.n = L; gm.k = Cgi;
            gm.batch = nn;
            gm.sa = 0;
            gm.sb = static_cast<long long>(C_in) * L;
            gm.sc = static_cast<long long>(per_image);
            gemm(d, gm);
        }
        const std::uint64_t bias_ptr = bias ? addr(bias->data) : 0;
        const std::uint64_t ptrs[] = {
            addr(cols.data), bias_ptr, addr(Y.data) + std::uint64_t(n0) * C_out * L_out * es};
        const std::int32_t ints[] = {i32(nn), i32(C_out), i32(L), i32(kL), i32(stride),
                                     i32(padding), i32(dilation), i32(Cgo), i32(L_out)};
        const std::uint8_t flags = static_cast<std::uint8_t>((bias ? 1u : 0u) | (cdt != dt ? 0x10u : 0u));
        run_audio(d, AU_COL2IM, dt, flags, std::uint64_t(nn) * C_out * L_out, false, ptrs, ints);
    }
}

// dX[c_in, l] = sum over (oc, k) of dY[oc, l s - p + k d] W[c_in, oc, k]: a
// conv1d of dY (C_out channels) with W read as OIHW (C_in, C_out / groups, 1,
// kL). Its natural length (L_out + 2p - d (kL - 1) - 1) / s + 1 is L, or more
// when output_padding >= stride; then the first L columns are kept.
void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY, int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding, int output_padding, int dilation, int groups,
                                     Tensor& dX) {
    const char* op = "conv_transpose1d_backward_input";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    dt_code(dY.dtype, op);
    need(op, Wt.dtype == dY.dtype, "Wt dtype must match dY");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    need(op, Wt.rows == C_in && Wt.cols == C_out / groups * kL, "Wt shape must be (C_in, (C_out/groups)*kL)");
    need(op, dY.rows == N && dY.cols == C_out * L_out, "dY shape must be (N, C_out*L_out)");
    ensure(dX, N, C_in * L, dY.dtype);
    if (N == 0 || C_in * L == 0) return;
    DeviceCtx& d = device_of(dY);
    const int Lc = (L_out + 2 * padding - dilation * (kL - 1) - 1) / stride + 1;
    Conv2dArgs a;
    a.op = op;
    a.dt = dY.dtype;
    a.n = N; a.cin = C_out; a.h = 1; a.wd = L_out; a.cout = C_in; a.kh = 1; a.kw = kL;
    a.sh = 1; a.sw = stride; a.ph = 0; a.pw = padding; a.dh = 1; a.dw = dilation; a.groups = groups;
    a.x = addr(dY.data);
    a.w = addr(Wt.data);
    if (Lc == L) {
        a.y = addr(dX.data);
        conv2d(d, a);
        return;
    }
    Tensor t = Tensor::empty_on(Device::vulkan(d.index()), N, C_in * Lc, dY.dtype);
    a.y = addr(t.data);
    conv2d(d, a);
    vulkan::copy_d2d_strided(t, 0, Lc, dX, 0, L, L, N * C_in);
}

void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY, int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding, int output_padding, int dilation, int groups,
                                      Tensor& dWt) {
    const char* op = "conv_transpose1d_backward_weight";
    check_convt1d(op, C_in, C_out, kL, stride, padding, output_padding, dilation, groups);
    dt_code(X.dtype, op);
    need(op, dY.dtype == X.dtype && dWt.dtype == X.dtype, "X, dY and dWt must share one dtype");
    const int L_out = convt1d_out_len(L, stride, padding, output_padding, dilation, kL);
    need(op, L_out > 0, "non-positive output length");
    const int Cg_out = C_out / groups;
    need(op, dWt.rows == C_in && dWt.cols == Cg_out * kL, "dWt shape must be (C_in, (C_out/groups)*kL)");
    need(op, X.rows == N && X.cols == C_in * L, "X shape must be (N, C_in*L)");
    need(op, dY.rows == N && dY.cols == C_out * L_out, "dY shape must be (N, C_out*L_out)");
    if (C_in == 0 || Cg_out == 0 || N == 0) return;
    const std::uint64_t ptrs[] = {addr(X.data), addr(dY.data), addr(dWt.data)};
    const std::int32_t ints[] = {i32(N), i32(C_in), i32(L), i32(C_out), i32(kL), i32(stride),
                                 i32(padding), i32(dilation), i32(C_in / groups), i32(Cg_out), i32(L_out)};
    run_audio(device_of(X), AU_CONVT_BWD_W, X.dtype, 0, std::uint64_t(C_in) * Cg_out * kL, true, ptrs, ints);
}

void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out, int L_out, Tensor& dB) {
    const char* op = "conv_transpose1d_backward_bias";
    need(op, dB.rows == C_out && dB.cols == 1 && dB.dtype == dY.dtype, "dB shape must be (C_out, 1) of dY's dtype");
    need(op, dY.rows == N && dY.cols == static_cast<long long>(C_out) * L_out, "dY shape must be (N, C_out*L_out)");
    vulkan::conv2d_backward_bias(dY, N, C_out, 1, L_out, dB);   // dB += per-channel sums
}

void causal_conv1d_update(const Tensor& X, const Tensor& Wt, const Tensor* bias, int N, int C, int L_step, int kL,
                          int dilation, Tensor& state, Tensor& Y) {
    const char* op = "causal_conv1d_update";
    dt_code(X.dtype, op);
    for (const Tensor* t : std::initializer_list<const Tensor*>{&Wt, bias, &state}) need(op, !t || t->dtype == X.dtype, "all operands must share X's dtype");
    need(op, kL >= 1 && dilation >= 1 && L_step >= 1 && N >= 0 && C >= 1, "kL/dilation/L_step/C must be >=1 and N >=0");
    need(op, Wt.rows == C && Wt.cols == kL, "Wt shape must be (C, kL) — one depthwise filter per channel");
    need(op, !bias || (bias->rows == C && bias->cols == 1), "bias shape must be (C, 1)");
    const int hist = (kL - 1) * dilation;
    need(op, state.rows == N && state.cols == C * hist, "state shape must be (N, C*(kL-1)*dilation)");
    need(op, X.size() >= static_cast<long long>(N) * C * L_step, "X is smaller than N*C*L_step");
    ensure(Y, N, C * L_step, X.dtype);
    if (N == 0) return;
    const std::uint64_t ptrs[] = {addr(X.data), addr(Wt.data), bias ? addr(bias->data) : 0, addr(state.data), addr(Y.data)};
    const std::int32_t ints[] = {i32(C), i32(L_step), i32(kL), i32(dilation), i32(N * C)};
    run_audio(device_of(X), AU_CAUSAL, X.dtype, static_cast<std::uint8_t>(bias ? 1u : 0u),
              std::uint64_t(N) * C, false, ptrs, ints);
}

void check_pad(const char* op, int N, int C, int L, int pad_left, int pad_right, int mode) {
    need(op, N >= 0 && C >= 1 && L >= 1, "C/L must be >=1 and N >=0");
    need(op, pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    need(op, mode >= 0 && mode <= 2, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    need(op, mode != 1 || (pad_left < L && pad_right < L), "reflect padding requires pad_left and pad_right < L");
}

void pad1d(const char* /*op*/, std::uint32_t code, const Tensor& src, int N, int C, int L, int pad_left, int pad_right,
           int mode, Tensor& dst) {
    const int L_pad = L + pad_left + pad_right;
    const std::uint64_t ptrs[] = {addr(src.data), addr(dst.data)};
    const std::int32_t ints[] = {i32(L), i32(pad_left), i32(L_pad), i32(N * C)};
    run_audio(device_of(src), code, src.dtype, static_cast<std::uint8_t>(mode),
              std::uint64_t(N) * C * (code == AU_PAD ? L_pad : L), false, ptrs, ints);
}

void pad1d_forward(const Tensor& X, int N, int C, int L, int pad_left, int pad_right, int mode, Tensor& Y) {
    const char* op = "pad1d_forward";
    dt_code(X.dtype, op);
    check_pad(op, N, C, L, pad_left, pad_right, mode);
    need(op, X.rows == N && X.cols == C * L, "X shape must be (N, C*L)");
    ensure(Y, N, C * (L + pad_left + pad_right), X.dtype);
    if (N == 0) return;
    pad1d(op, AU_PAD, X, N, C, L, pad_left, pad_right, mode, Y);
}

void pad1d_backward(const Tensor& dY, int N, int C, int L, int pad_left, int pad_right, int mode, Tensor& dX) {
    const char* op = "pad1d_backward";
    dt_code(dY.dtype, op);
    check_pad(op, N, C, L, pad_left, pad_right, mode);
    need(op, dY.rows == N && dY.cols == C * (L + pad_left + pad_right), "dY shape must be (N, C*(L+pad_left+pad_right))");
    ensure(dX, N, C * L, dY.dtype);
    if (N == 0) return;
    pad1d(op, AU_PAD_BWD, dY, N, C, L, pad_left, pad_right, mode, dX);
}

// ─── Snake ──────────────────────────────────────────────────────────────────

void check_snake(const char* op, const Tensor& X, const Tensor& alpha, const Tensor* beta, int N, int C, int L) {
    dt_code(X.dtype, op);
    need(op, N >= 0 && C >= 0 && L >= 0, "N, C, L must be non-negative");
    need(op, X.rows == N && X.cols == static_cast<long long>(C) * L, "X must be shaped (N, C*L)");
    need(op, alpha.dtype == Dtype::FP32 && alpha.size() == C, "alpha must have C FP32 elements");
    need(op, !beta || (beta->dtype == Dtype::FP32 && beta->size() == C), "beta must have C FP32 elements");
}

void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta, int N, int C, int L, Tensor& Y) {
    const char* op = "snake_forward";
    check_snake(op, X, alpha, beta, N, C, L);
    ensure(Y, N, C * L, X.dtype);   // Y may be X (in place)
    if (N == 0 || C * L == 0) return;
    const std::uint32_t total = static_cast<std::uint32_t>(static_cast<long long>(N) * C * L);
    const std::uint64_t ptrs[] = {addr(X.data), addr(alpha.data), beta ? addr(beta->data) : 0, addr(Y.data)};
    const std::int32_t ints[] = {i32(C), i32(L), static_cast<std::int32_t>(total)};
    run_audio(device_of(X), AU_SNAKE, X.dtype, static_cast<std::uint8_t>(beta ? 1u : 0u), total, false, ptrs, ints);
}

void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta, const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta) {
    const char* op = "snake_backward";
    check_snake(op, X, alpha, beta, N, C, L);
    need(op, (beta == nullptr) == (dBeta == nullptr), "dBeta must be non-null exactly when beta is non-null");
    need(op, dY.dtype == X.dtype && dY.rows == N && dY.cols == static_cast<long long>(C) * L,
         "dY must be shaped (N, C*L) in X's dtype");
    need(op, dAlpha.rows == C && dAlpha.cols == 1 && dAlpha.dtype == Dtype::FP32, "dAlpha must be (C, 1) FP32");
    need(op, !dBeta || (dBeta->rows == C && dBeta->cols == 1 && dBeta->dtype == Dtype::FP32),
         "dBeta must be (C, 1) FP32");
    ensure(dX, N, C * L, X.dtype);
    if (N == 0 || C * L == 0) return;
    const std::uint64_t ptrs[] = {
        addr(X.data), addr(alpha.data), beta ? addr(beta->data) : 0, addr(dY.data),
        addr(dX.data), addr(dAlpha.data), dBeta ? addr(dBeta->data) : 0};
    const std::int32_t ints[] = {i32(C), i32(L), i32(N)};
    run_audio(device_of(X), AU_SNAKE_BWD, X.dtype, static_cast<std::uint8_t>(beta ? 1u : 0u),
              static_cast<std::uint64_t>(C), true, ptrs, ints);
}

// ─── resample1d ─────────────────────────────────────────────────────────────

void check_resample(const char* op, int N, int C, int L_in, int L_out, int mode) {
    need(op, N >= 0 && C >= 0 && L_in >= 0 && L_out >= 0, "N, C, L_in, L_out must be non-negative");
    need(op, mode == 0 || mode == 1, "mode must be 0 (nearest) or 1 (linear)");
    need(op, !(L_out > 0 && L_in == 0), "L_in must be > 0 when L_out > 0");
}

void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out, int mode, Tensor& Y) {
    const char* op = "resample1d_forward";
    dt_code(X.dtype, op);
    check_resample(op, N, C, L_in, L_out, mode);
    need(op, X.size() >= static_cast<long long>(N) * C * L_in, "X is smaller than N*C*L_in");
    ensure(Y, N, C * L_out, X.dtype);
    if (N == 0 || C * L_out == 0) return;
    const std::uint64_t ptrs[] = {addr(X.data), addr(Y.data)};
    const std::int32_t ints[] = {i32(L_in), i32(L_out), i32(N * C)};
    run_audio(device_of(X), AU_RESAMPLE, X.dtype, static_cast<std::uint8_t>(mode),
              std::uint64_t(N) * C * L_out, false, ptrs, ints);
}

void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out, int mode, Tensor& dX) {
    const char* op = "resample1d_backward";
    dt_code(dY.dtype, op);
    check_resample(op, N, C, L_in, L_out, mode);
    need(op, dY.size() >= static_cast<long long>(N) * C * L_out, "dY is smaller than N*C*L_out");
    ensure(dX, N, C * L_in, dY.dtype);
    if (N == 0 || C * L_in == 0) return;
    if (L_out == 0) { dX.zero(); return; }
    const std::uint64_t ptrs[] = {addr(dY.data), addr(dX.data)};
    const std::int32_t ints[] = {i32(L_in), i32(L_out), i32(N * C)};
    run_audio(device_of(dY), AU_RESAMPLE_BWD, dY.dtype, static_cast<std::uint8_t>(mode),
              std::uint64_t(N) * C * L_in, false, ptrs, ints);
}

// ─── codec quantisers ───────────────────────────────────────────────────────

void vq_encode_forward(const Tensor& x, const Tensor& codebook, Tensor& indices, Tensor& quantized) {
    const char* op = "vq_encode_forward";
    need(op, x.dtype == Dtype::FP32 && codebook.dtype == Dtype::FP32, "x and codebook must be FP32");
    const int N = x.rows, D = x.cols, K = codebook.rows;
    need(op, codebook.cols == D, "codebook must have the same column count as x");
    need(op, !(K == 0 && N != 0), "codebook must have at least one codeword");
    ensure(indices, N, 1, Dtype::INT32);
    ensure(quantized, N, D, Dtype::FP32);
    if (N == 0) return;
    const std::uint64_t ptrs[] = {addr(x.data), addr(codebook.data), addr(quantized.data), addr(indices.data)};
    const std::int32_t ints[] = {i32(N), i32(D), i32(K)};
    run_audio(device_of(x), AU_VQ, Dtype::FP32, 0, static_cast<std::uint64_t>(N), true, ptrs, ints);
}

void fsq_quantize_forward(const Tensor& x, const Tensor& levels, Tensor& quantized, Tensor& packed_indices) {
    const char* op = "fsq_quantize_forward";
    need(op, x.dtype == Dtype::FP32, "x must be FP32");
    need(op, levels.dtype == Dtype::INT32, "levels must be INT32");
    const int N = x.rows, D = x.cols;
    need(op, levels.size() == D, "levels must have D elements (one per column of x)");
    // Every level count must be >= 2; the levels are device data, read once.
    std::vector<std::int32_t> lv(static_cast<std::size_t>(D));
    if (D > 0) levels.copy_to_host_raw(lv.data(), lv.size() * sizeof(std::int32_t));
    for (std::int32_t l : lv) need(op, l >= 2, "every level count must be >= 2");
    ensure(quantized, N, D, Dtype::FP32);
    ensure(packed_indices, N, 1, Dtype::INT32);
    if (N == 0 || D == 0) return;
    const std::uint64_t ptrs[] = {addr(x.data), addr(levels.data), addr(quantized.data), addr(packed_indices.data)};
    const std::int32_t ints[] = {i32(N), i32(D)};
    run_audio(device_of(x), AU_FSQ, Dtype::FP32, 0, static_cast<std::uint64_t>(N), false, ptrs, ints);
}

// The straight-through backwards: dX = dQuantized.
void straight_through(const char* op, const Tensor& dQ, Tensor& dX) {
    need(op, dQ.dtype == Dtype::FP32, "dQuantized must be FP32");
    ensure(dX, dQ.rows, dQ.cols, Dtype::FP32);
    if (dQ.size() == 0 || dX.data == dQ.data) return;
    ::brotensor::copy_d2d(dQ, 0, dX, 0, static_cast<int>(dQ.size()));
}

void vq_encode_backward(const Tensor& dQ, Tensor& dX) { straight_through("vq_encode_backward", dQ, dX); }
void fsq_quantize_backward(const Tensor& dQ, Tensor& dX) { straight_through("fsq_quantize_backward", dQ, dX); }

}  // namespace

// pad1d over `rows` rows of one channel, for the STFT's centring
// (ops_spectral.cpp); no argument checks beyond the kernel's.
void audio_pad1d(DeviceCtx& d, const Tensor& X, int rows, int L, int pad_left, int pad_right, int mode, Tensor& Y) {
    (void)d;
    pad1d("stft", AU_PAD, X, rows, 1, L, pad_left, pad_right, mode, Y);
}

void fill_vulkan_vtable_audio(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose1d_forward = &conv_transpose1d_forward;
    v.conv_transpose1d_backward_input = &conv_transpose1d_backward_input;
    v.conv_transpose1d_backward_weight = &conv_transpose1d_backward_weight;
    v.conv_transpose1d_backward_bias = &conv_transpose1d_backward_bias;
    v.causal_conv1d_update = &causal_conv1d_update;
    v.pad1d_forward = &pad1d_forward;
    v.pad1d_backward = &pad1d_backward;
    v.snake_forward = &snake_forward;
    v.snake_backward = &snake_backward;
    v.resample1d_forward = &resample1d_forward;
    v.resample1d_backward = &resample1d_backward;
    v.vq_encode_forward = &vq_encode_forward;
    v.vq_encode_backward = &vq_encode_backward;
    v.fsq_quantize_forward = &fsq_quantize_forward;
    v.fsq_quantize_backward = &fsq_quantize_backward;
}

}  // namespace brotensor::detail::vulkan
