// CUDA decomposed 2D relative-position self-attention (SAM global + windowed)
// and Transformer-XL rel_pos_bias_xl_forward backed by Brass MIR -> PTX
// (MirKind::AttnSoftmax, Cast) + CUDA GEMMs / gather_rows.

#include "cu_mir_cache.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in batched_ops.cu / gemm.cu / matmul.cu / concat.cu / ops_spatial.cpp.
void linear_forward_batched(const Tensor& W, const Tensor& bias, const Tensor& X_BD, Tensor& Y_BD);
void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n);
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch, int width, int height);
void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

const Tensor* present(const Tensor* t) noexcept {
    return (t && t->data && t->size() > 0) ? t : nullptr;
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

void* byte_off(void* p, std::size_t bytes) noexcept {
    return static_cast<char*>(p) + bytes;
}

void cast_to_f32(const Tensor& src, Tensor& dst) {
    const long long n_ll = src.size();
    if (n_ll <= 0) return;
    const std::uint32_t n = static_cast<std::uint32_t>(n_ll);
    const int lanes = mir_vec_lanes_mixed(src.dtype, Dtype::FP32, n, addr(src.data), addr(dst.data));
    CUfunction fn = mir_function(
        {MirKind::Cast, 0u, src.dtype, Dtype::FP32, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_cast(fn, mir_groups_1d(n, lanes), src.data, dst.data, n);
}

void project_typed(const Tensor& X, const Tensor& W, const Tensor* b, Tensor& Y,
                   const Tensor* zero_bias_f32) {
    if (X.dtype == Dtype::FP32) {
        linear_forward_batched(W, b ? *b : *zero_bias_f32, X, Y);
    } else {
        linear_forward_batched_fp16(W, b, X, Y);
    }
}

void sam_attention(const char* op, const Tensor& X, const Tensor& Wq, const Tensor* bq,
                   const Tensor& Wk, const Tensor* bk, const Tensor& Wv, const Tensor* bv,
                   const Tensor& Wo, const Tensor* bo, const Tensor& rel_h,
                   const Tensor& rel_w, int H, int grid_h, int grid_w, int window,
                   float scale_in, Tensor& O) {
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    for (const Tensor* t : {&Wq, &Wk, &Wv, &Wo, &rel_h, &rel_w}) {
        need(op, t->dtype == dt, "Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    }
    const int L = X.rows, D = X.cols;
    need(op, H > 0 && D % H == 0, "num_heads must divide D");
    need(op, grid_h > 0 && grid_w > 0 && grid_h * grid_w == L,
         "grid_h*grid_w must equal X.rows");
    for (const Tensor* t : {&Wq, &Wk, &Wv, &Wo}) {
        need(op, t->rows == D && t->cols == D, "Wq/Wk/Wv/Wo must be (D, D)");
    }
    const int dh = D / H;
    const int gh = window > 0 ? window : grid_h, gw = window > 0 ? window : grid_w;
    need(op, rel_h.rows == 2 * gh - 1 && rel_h.cols == dh,
         window > 0 ? "rel_pos_h must be (2*window-1, head_dim)"
                    : "rel_pos_h must be (2*grid_h-1, head_dim)");
    need(op, rel_w.rows == 2 * gw - 1 && rel_w.cols == dh,
         window > 0 ? "rel_pos_w must be (2*window-1, head_dim)"
                    : "rel_pos_w must be (2*grid_w-1, head_dim)");
    for (const Tensor* b : {bq, bk, bv, bo}) {
        if (present(b)) {
            need(op, b->dtype == dt && b->size() == D,
                 "biases must have D entries of X's dtype");
        }
    }
    if (O.rows != L || O.cols != D || O.dtype != dt) O.resize(L, D, dt);
    if (L == 0 || D == 0) return;
    need(op, gh <= 0xffff && gw <= 0xffff, "grid too large");

    const ::brotensor::Device dev = X.device;
    const std::size_t es = static_cast<std::size_t>(::brotensor::dtype_size_bytes(dt));
    const float scale = std::isnan(scale_in)
        ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(dh)))
        : scale_in;

    int U = 1;
    const int Lu = gh * gw;
    Tensor Xw, back;
    if (window > 0) {
        const int nwh = (grid_h + window - 1) / window;
        const int nww = (grid_w + window - 1) / window;
        U = nwh * nww;
        std::vector<std::int32_t> fwd(static_cast<std::size_t>(U) * Lu);
        std::vector<std::int32_t> inv(static_cast<std::size_t>(L));
        for (int a = 0; a < nwh; ++a) {
            for (int b = 0; b < nww; ++b) {
                for (int lh = 0; lh < window; ++lh) {
                    for (int lw = 0; lw < window; ++lw) {
                        const int h = a * window + lh, w = b * window + lw;
                        const int r = (a * nww + b) * Lu + lh * window + lw;
                        const bool in = h < grid_h && w < grid_w;
                        fwd[static_cast<std::size_t>(r)] = in ? (h * grid_w + w) : L;
                        if (in) inv[static_cast<std::size_t>(h * grid_w + w)] = r;
                    }
                }
            }
        }
        Tensor Xp = Tensor::zeros_on(dev, L + 1, D, dt);
        copy_d2d(X, 0, Xp, 0, L * D);
        const Tensor fi = Tensor::from_raw_bytes_on(
            dev, fwd.data(), U * Lu, 1, Dtype::INT32, fwd.size() * sizeof(std::int32_t));
        back = Tensor::from_raw_bytes_on(
            dev, inv.data(), L, 1, Dtype::INT32, inv.size() * sizeof(std::int32_t));
        Xw = Tensor::empty_on(dev, U * Lu, D, dt);
        gather_rows(Xp, fi, Xw);
    }
    const Tensor& Xu = (window > 0) ? Xw : X;
    const int R = U * Lu;

    Tensor zero_d;
    if (dt == Dtype::FP32) zero_d = Tensor::zeros_on(dev, D, 1, Dtype::FP32);
    Tensor Q = Tensor::empty_on(dev, R, D, dt);
    Tensor K = Tensor::empty_on(dev, R, D, dt);
    Tensor V = Tensor::empty_on(dev, R, D, dt);
    project_typed(Xu, Wq, present(bq), Q, dt == Dtype::FP32 ? &zero_d : nullptr);
    project_typed(Xu, Wk, present(bk), K, dt == Dtype::FP32 ? &zero_d : nullptr);
    project_typed(Xu, Wv, present(bv), V, dt == Dtype::FP32 ? &zero_d : nullptr);

    Tensor Q_f32_buf;
    const Tensor* Q_f32p = &Q;
    if (dt != Dtype::FP32) {
        Q_f32_buf = Tensor::empty_on(dev, R, D, Dtype::FP32);
        cast_to_f32(Q, Q_f32_buf);
        Q_f32p = &Q_f32_buf;
    }

    const int ph = 2 * gh - 1, pw = 2 * gw - 1;
    Tensor Q_uh = Tensor::empty_on(dev, H * Lu, dh, Dtype::FP32);
    Tensor K_uh = Tensor::empty_on(dev, H * Lu, dh, dt);
    Tensor V_uh = Tensor::empty_on(dev, H * Lu, dh, dt);
    Tensor Bh = Tensor::empty_on(dev, H * Lu, ph, Dtype::FP32);
    Tensor Bw = Tensor::empty_on(dev, H * Lu, pw, Dtype::FP32);
    Tensor S = Tensor::empty_on(dev, H * Lu, Lu, Dtype::FP32);
    Tensor P_tmp;
    if (dt != Dtype::FP32) P_tmp = Tensor::empty_on(dev, H * Lu, Lu, dt);
    Tensor Y_uh = Tensor::empty_on(dev, H * Lu, dh, dt);
    Tensor Yc = Tensor::empty_on(dev, R, D, dt);

    Tensor zb_ph = Tensor::zeros_on(dev, ph, 1, Dtype::FP32);
    Tensor zb_pw = Tensor::zeros_on(dev, pw, 1, Dtype::FP32);
    Tensor zb_lu = Tensor::zeros_on(dev, Lu, 1, Dtype::FP32);
    CUfunction sm_fn = mir_function(
        {MirKind::AttnSoftmax, 2u, dt, Dtype::FP32, 1, Dtype::FP32, 0u});
    const std::uint32_t sm_bhs =
        static_cast<std::uint32_t>(gw) | (static_cast<std::uint32_t>(gh) << 16);
    const std::uint32_t sm_rows = static_cast<std::uint32_t>(H * Lu);

    for (int u = 0; u < U; ++u) {
        for (int h = 0; h < H; ++h) {
            copy_d2d_strided(*Q_f32p, u * Lu * D + h * dh, D, Q_uh, h * Lu * dh, dh, dh, Lu);
            copy_d2d_strided(K, u * Lu * D + h * dh, D, K_uh, h * Lu * dh, dh, dh, Lu);
            copy_d2d_strided(V, u * Lu * D + h * dh, D, V_uh, h * Lu * dh, dh, dh, Lu);
        }
        linear_forward_batched(rel_h, zb_ph, Q_uh, Bh);
        linear_forward_batched(rel_w, zb_pw, Q_uh, Bw);

        const std::size_t q_stride_f = static_cast<std::size_t>(Lu) * dh * sizeof(float);
        const std::size_t k_stride_dt = static_cast<std::size_t>(Lu) * dh * es;
        const std::size_t s_stride_f = static_cast<std::size_t>(Lu) * Lu * sizeof(float);
        const std::size_t p_stride_dt = static_cast<std::size_t>(Lu) * Lu * es;

        for (int h = 0; h < H; ++h) {
            const Tensor qh = Tensor::view(dev, byte_off(Q_uh.data, h * q_stride_f), Lu, dh, Dtype::FP32);
            const Tensor kh = Tensor::view(dev, byte_off(K_uh.data, h * k_stride_dt), Lu, dh, dt);
            Tensor sh = Tensor::view(dev, byte_off(S.data, h * s_stride_f), Lu, Lu, Dtype::FP32);
            linear_forward_batched(kh, zb_lu, qh, sh);
        }

        void* p_buf = (dt == Dtype::FP32) ? S.data : P_tmp.data;
        const std::uintptr_t ptrs[] = {
            addr(S.data), addr(p_buf), addr(Bh.data), addr(Bw.data), 0};
        const std::int32_t ints[] = {
            static_cast<std::int32_t>(sm_rows), Lu, Lu, 0, Lu, Lu, 0,
            static_cast<std::int32_t>(sm_bhs)};
        const float flts[] = {scale};
        mir_dispatch_args(sm_fn, std::clamp(sm_rows, 1u, 65535u), 256u, ptrs, ints, flts);

        for (int h = 0; h < H; ++h) {
            const Tensor ph_v = Tensor::view(dev, byte_off(p_buf, h * p_stride_dt), Lu, Lu, dt);
            const Tensor vh_v = Tensor::view(dev, byte_off(V_uh.data, h * k_stride_dt), Lu, dh, dt);
            Tensor yh_v = Tensor::view(dev, byte_off(Y_uh.data, h * k_stride_dt), Lu, dh, dt);
            matmul(ph_v, vh_v, yh_v);
        }
        for (int h = 0; h < H; ++h) {
            copy_d2d_strided(Y_uh, h * Lu * dh, dh, Yc, u * Lu * D + h * dh, D, dh, Lu);
        }
    }

    if (window > 0) {
        Tensor Ow = Tensor::empty_on(dev, R, D, dt);
        project_typed(Yc, Wo, present(bo), Ow, dt == Dtype::FP32 ? &zero_d : nullptr);
        gather_rows(Ow, back, O);
    } else {
        project_typed(Yc, Wo, present(bo), O, dt == Dtype::FP32 ? &zero_d : nullptr);
    }
}

}  // namespace

void self_attention_decomposed_rel_pos_forward(const Tensor& X, const Tensor& Wq, const Tensor* bq,
                                               const Tensor& Wk, const Tensor* bk,
                                               const Tensor& Wv, const Tensor* bv,
                                               const Tensor& Wo, const Tensor* bo,
                                               const Tensor& rel_pos_h, const Tensor& rel_pos_w,
                                               int num_heads, int grid_h, int grid_w,
                                               float scale, Tensor& O) {
    sam_attention("self_attention_decomposed_rel_pos_forward", X, Wq, bq, Wk, bk, Wv, bv,
                  Wo, bo, rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, 0, scale, O);
}

void self_attention_decomposed_rel_pos_windowed_forward(
    const Tensor& X, const Tensor& Wq, const Tensor* bq, const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv, const Tensor& Wo, const Tensor* bo,
    const Tensor& rel_pos_h, const Tensor& rel_pos_w, int num_heads, int grid_h, int grid_w,
    int window, float scale, Tensor& O) {
    constexpr const char* op = "self_attention_decomposed_rel_pos_windowed_forward";
    need(op, window > 0, "window must be >= 1");
    sam_attention(op, X, Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
                  num_heads, grid_h, grid_w, window, scale, O);
}

void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk, int num_heads,
                             int head_dim, Tensor& Bias) {
    constexpr const char* op = "rel_pos_bias_xl_forward";
    need(op, Qv.dtype == Dtype::FP32 && Pk.dtype == Dtype::FP32, "Qv and Pk must be FP32");
    const int T = Qv.rows, D = Qv.cols;
    need(op, num_heads > 0 && head_dim > 0 && num_heads * head_dim == D,
         "num_heads*head_dim must equal Qv.cols");
    need(op, Pk.cols == D && Pk.rows == 2 * T - 1, "Pk must be (2*Qv.rows - 1, Qv.cols)");
    if (Bias.rows != num_heads * T || Bias.cols != T || Bias.dtype != Dtype::FP32) {
        Bias.resize(num_heads * T, T, Dtype::FP32);
    }
    if (T == 0) return;
    const int P = 2 * T - 1;
    Tensor Qv_h = Tensor::empty_on(Qv.device, T, head_dim, Dtype::FP32);
    Tensor Pk_h = Tensor::empty_on(Qv.device, P, head_dim, Dtype::FP32);
    Tensor zb_p = Tensor::zeros_on(Qv.device, P, 1, Dtype::FP32);
    Tensor R_h = Tensor::empty_on(Qv.device, T, P, Dtype::FP32);
    for (int h = 0; h < num_heads; ++h) {
        copy_d2d_strided(Qv, h * head_dim, D, Qv_h, 0, head_dim, head_dim, T);
        copy_d2d_strided(Pk, h * head_dim, D, Pk_h, 0, head_dim, head_dim, P);
        linear_forward_batched(Pk_h, zb_p, Qv_h, R_h);
        if (T == 1) {
            copy_d2d(R_h, 0, Bias, h, 1);
        } else {
            copy_d2d_strided(R_h, T - 1, P - 1, Bias, h * T * T, T, T, T);
        }
    }
}

}  // namespace brotensor::detail::cuda
