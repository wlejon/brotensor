// Pure C++ ResBlock forward (FP16/BF16 & INT8W) and backward dispatchers
// backed by Brass MIR (MirKind::Gnorm with silu + MirKind::ConvAux), replacing src/cuda/resblock.cu.

#include "cu_mir_cache.h"
#include "../gpu_common/resblock.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y);
inline void conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           ::brotensor::Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, Y);
}

void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           ::brotensor::Tensor& dX);
inline void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                                  const ::brotensor::Tensor& dY,
                                  int N, int C_in, int H, int W,
                                  int C_out, int kH, int kW,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dil_h, int dil_w,
                                  ::brotensor::Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, dX);
}

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt);
inline void conv2d_backward_weight(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& dY,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dil_h, int dil_w,
                                   ::brotensor::Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, dWt);
}

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB);

void conv2d_int8w_fp16_forward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W_int8,
                               const ::brotensor::Tensor& scales,
                               const ::brotensor::Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w,
                               int groups,
                               ::brotensor::Tensor& Y);

void add_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);
void silu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void silu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX);
void group_norm_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta, int N, int C, int H, int W,
                        int num_groups, float eps, ::brotensor::Tensor& Y);
void group_norm_silu_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma,
                             const ::brotensor::Tensor& beta, int N, int C, int H, int W,
                             int num_groups, float eps, ::brotensor::Tensor& Y);
void group_norm_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY, int N, int C, int H, int W,
                         int num_groups, float eps, ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta);

namespace {

inline void ensure_cuda(::brotensor::Tensor& t) {
    if (t.data == nullptr) t.device = ::brotensor::Device::CUDA;
}

void dispatch_add_nc_shift(Dtype dt, void* Y, const void* shift, int N, int C, int spatial, int has_N) {
    const int total = N * C * spatial;
    if (total <= 0) return;
    CUfunction fn = mir_function(
        {MirKind::ConvAux, 5u, dt, Dtype::FP32, 1, Dtype::FP32, static_cast<std::uint8_t>(has_N ? 1u : 0u)});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(Y),
        reinterpret_cast<std::uintptr_t>(shift),
    };
    const std::int32_t ints[] = {N, C, spatial};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>((total + 255) / 256), 1u, 1u, 256u, ptrs, ints, {});
}

void dispatch_sum_hw_per_nc(Dtype dt, const void* dh2, void* d_shift, int N, int C, int spatial) {
    const int blocks = N * C;
    if (blocks <= 0) return;
    CUfunction fn = mir_function({MirKind::ConvAux, 6u, dt});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(dh2),
        reinterpret_cast<std::uintptr_t>(d_shift),
    };
    const std::int32_t ints[] = {N, C, spatial};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>(blocks), 1u, 1u, 256u, ptrs, ints, {});
}

}  // namespace

void resblock_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                      const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,
                      const ::brotensor::Tensor* t_emb_shift,
                      const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                      const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,
                      const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,
                      int N, int C_in, int C_out, int H, int Wd,
                      int num_groups, float eps,
                      ::brotensor::Tensor& Y) {
    const Dtype dt = X.dtype;
    gpu_common::validate_resblock_fp16_bf16(
        "resblock_forward", X, gamma1, beta1, W1, gamma2, beta2, W2, Wskip, C_in, C_out, num_groups);
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) Y.resize(N, out_cols, dt);
    if (N == 0 || spatial == 0) return;

    thread_local static ::brotensor::Tensor h1, h2, h3, skip_scratch;
    ensure_cuda(h1); ensure_cuda(h2); ensure_cuda(h3); ensure_cuda(skip_scratch);
    h1.resize(N, C_in  * spatial, dt);
    h2.resize(N, C_out * spatial, dt);
    h3.resize(N, C_out * spatial, dt);

    group_norm_silu_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps, h1);
    conv2d_forward(h1, W1, b1, N, C_in, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        const int has_N = gpu_common::parse_shift_has_n(t_emb_shift, dt, N, C_out, "resblock_forward");
        dispatch_add_nc_shift(dt, h2.data, t_emb_shift->data, N, C_out, spatial, has_N);
    }
    group_norm_silu_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps, h3);

    if (Wskip != nullptr) {
        conv2d_forward(X, *Wskip, bskip, N, C_in, H, Wd, C_out, 1, 1, 1, 1, 0, 0, 1, 1, skip_scratch);
    }
    conv2d_forward(h3, W2, b2, N, C_out, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, Y);
    add_inplace(Y, Wskip == nullptr ? X : skip_scratch);
}

void resblock_forward_int8w_fp16(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                                 const ::brotensor::Tensor& W1_int8, const ::brotensor::Tensor& s1,
                                 const ::brotensor::Tensor* b1,
                                 const ::brotensor::Tensor* t_emb_shift,
                                 const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                                 const ::brotensor::Tensor& W2_int8, const ::brotensor::Tensor& s2,
                                 const ::brotensor::Tensor* b2,
                                 const ::brotensor::Tensor* Wskip_int8, const ::brotensor::Tensor* sskip,
                                 const ::brotensor::Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int Wd,
                                 int num_groups, float eps,
                                 ::brotensor::Tensor& Y) {
    gpu_common::validate_resblock_int8w(
        "resblock_forward_int8w_fp16", X, gamma1, beta1, W1_int8, s1, b1,
        gamma2, beta2, W2_int8, s2, b2, Wskip_int8, sskip, bskip, C_in, C_out, num_groups);
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) Y.resize(N, out_cols, Dtype::FP16);
    if (N == 0 || spatial == 0) return;

    thread_local static ::brotensor::Tensor h1, h2, h3, skip_scratch;
    ensure_cuda(h1); ensure_cuda(h2); ensure_cuda(h3); ensure_cuda(skip_scratch);
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    group_norm_silu_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps, h1);
    conv2d_int8w_fp16_forward(h1, W1_int8, s1, b1, N, C_in, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        const int has_N = gpu_common::parse_shift_has_n(t_emb_shift, Dtype::FP16, N, C_out, "resblock_forward_int8w_fp16");
        dispatch_add_nc_shift(Dtype::FP16, h2.data, t_emb_shift->data, N, C_out, spatial, has_N);
    }
    group_norm_silu_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps, h3);

    if (Wskip_int8 != nullptr) {
        conv2d_int8w_fp16_forward(X, *Wskip_int8, *sskip, bskip, N, C_in, H, Wd, C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1, skip_scratch);
    }
    conv2d_int8w_fp16_forward(h3, W2_int8, s2, b2, N, C_out, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, Y);
    add_inplace(Y, Wskip_int8 == nullptr ? X : skip_scratch);
}

void resblock_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                       const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,
                       const ::brotensor::Tensor* t_emb_shift,
                       const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                       const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,
                       const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,
                       int N, int C_in, int C_out, int H, int Wd,
                       int num_groups, float eps,
                       const ::brotensor::Tensor& dY,
                       ::brotensor::Tensor& dX,
                       ::brotensor::Tensor& dGamma1, ::brotensor::Tensor& dBeta1,
                       ::brotensor::Tensor& dW1, ::brotensor::Tensor* db1,
                       ::brotensor::Tensor* dt_emb_shift,
                       ::brotensor::Tensor& dGamma2, ::brotensor::Tensor& dBeta2,
                       ::brotensor::Tensor& dW2, ::brotensor::Tensor* db2,
                       ::brotensor::Tensor* dWskip, ::brotensor::Tensor* dbskip) {
    (void)bskip;
    (void)b2;
    const Dtype dt = X.dtype;
    gpu_common::validate_resblock_fp16_bf16(
        "resblock_backward", X, gamma1, beta1, W1, gamma2, beta2, W2, Wskip, C_in, C_out, num_groups);
    const int spatial = H * Wd;
    if (dY.dtype != dt) throw std::runtime_error("resblock_backward: dY dtype must match X");
    if (dY.rows != N || dY.cols != C_out * spatial) throw std::runtime_error("resblock_backward: dY shape mismatch");
    if (dX.rows != N || dX.cols != C_in * spatial || dX.dtype != dt) dX.resize(N, C_in * spatial, dt);
    if (N == 0 || spatial == 0) return;

    ::brotensor::Tensor h1_pre_silu, h1, h2, h3_pre_silu, h3, dh3_pre_silu, dh2, dh1_pre_silu;
    ensure_cuda(h1_pre_silu); ensure_cuda(h1); ensure_cuda(h2);
    ensure_cuda(h3_pre_silu); ensure_cuda(h3);
    ensure_cuda(dh3_pre_silu); ensure_cuda(dh2); ensure_cuda(dh1_pre_silu);

    group_norm_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps, h1_pre_silu);
    silu_forward(h1_pre_silu, h1);
    conv2d_forward(h1, W1, b1, N, C_in, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        const int has_N = gpu_common::parse_shift_has_n(t_emb_shift, dt, N, C_out, "resblock_backward");
        dispatch_add_nc_shift(dt, h2.data, t_emb_shift->data, N, C_out, spatial, has_N);
    }
    group_norm_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps, h3_pre_silu);
    silu_forward(h3_pre_silu, h3);

    ::brotensor::Tensor dh3 = ::brotensor::Tensor::empty_on(::brotensor::Device::CUDA, N, C_out * spatial, dt);
    conv2d_backward_input(W2, dY, N, C_out, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh3);
    conv2d_backward_weight(h3, dY, N, C_out, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) conv2d_backward_bias(dY, N, C_out, H, Wd, *db2);

    silu_backward(h3_pre_silu, dh3, dh3_pre_silu);
    group_norm_backward(h2, gamma2, dh3_pre_silu, N, C_out, H, Wd, num_groups, eps, dh2, dGamma2, dBeta2);

    if (t_emb_shift && dt_emb_shift) {
        if (dt_emb_shift->dtype != dt) throw std::runtime_error("resblock_backward: dt_emb_shift dtype must match X");
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            dispatch_sum_hw_per_nc(dt, dh2.data, dt_emb_shift->data, N, C_out, spatial);
        } else {
            conv2d_backward_bias(dh2, N, C_out, H, Wd, *dt_emb_shift);
        }
    }

    ::brotensor::Tensor dh1 = ::brotensor::Tensor::empty_on(::brotensor::Device::CUDA, N, C_in * spatial, dt);
    conv2d_backward_input(W1, dh2, N, C_in, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh1);
    conv2d_backward_weight(h1, dh2, N, C_in, H, Wd, C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) conv2d_backward_bias(dh2, N, C_out, H, Wd, *db1);

    silu_backward(h1_pre_silu, dh1, dh1_pre_silu);
    group_norm_backward(X, gamma1, dh1_pre_silu, N, C_in, H, Wd, num_groups, eps, dX, dGamma1, dBeta1);

    if (Wskip == nullptr) {
        add_inplace(dX, dY);
    } else {
        if (Wskip->dtype != dt) throw std::runtime_error("resblock_backward: Wskip dtype must match X");
        ::brotensor::Tensor dX_skip = ::brotensor::Tensor::empty_on(::brotensor::Device::CUDA, N, C_in * spatial, dt);
        conv2d_backward_input(*Wskip, dY, N, C_in, H, Wd, C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX_skip);
        if (dWskip) conv2d_backward_weight(X, dY, N, C_in, H, Wd, C_out, 1, 1, 1, 1, 0, 0, 1, 1, *dWskip);
        if (dbskip) conv2d_backward_bias(dY, N, C_out, H, Wd, *dbskip);
        add_inplace(dX, dX_skip);
    }
}

}  // namespace brotensor::detail::cuda
