// Metal ResBlock forward (FP16/BF16 & INT8W) and backward backed by Brass MIR
// (group_norm_silu_forward + MirKind::ConvAux), with zero raw MSL shaders.

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#include <cstdint>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

// Forward declarations of sibling Metal ops implemented in other TUs (and of
// ops defined later in this file). Generated from the canonical op table.
#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

void group_norm_silu_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                             int N, int C, int H, int W, int num_groups, float eps,
                             Tensor& Y);

// The resblock body predates the grouped-conv API. These no-groups overloads
// forward to the canonical ops with groups = 1 (resblock convs are dense).
inline void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                           int N, int C_in, int H, int W, int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, Y);
}
inline void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                  int N, int C_in, int H, int W,
                                  int C_out, int kH, int kW,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int dil_h, int dil_w, Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, dX);
}
inline void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w, int pad_h, int pad_w,
                                   int dil_h, int dil_w, Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, 1, dWt);
}

namespace {

void launch_add_shift(Dtype dt, Tensor& Y, const Tensor& shift,
                      int N, int C, int spatial, int has_N) {
    const int total = N * C * spatial;
    if (total <= 0) return;
    const MirKernel mk = mir_kernel(
        {MirKind::ConvAux, 5u, dt, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(has_N ? 1u : 0u)});
    const void* ptrs[] = {Y.data, shift.data};
    const std::int32_t ints[] = {N, C, spatial};
    const std::uint32_t gx = static_cast<std::uint32_t>((static_cast<std::uint64_t>(total) + 255u) / 256u);
    mir_dispatch_args_3d(mk, gx, 1u, 1u, "resblock_add_shift", ptrs, ints, {});
}

void launch_sum_hw_per_NC(Dtype dt, const Tensor& dh2, Tensor& d_shift,
                          int N, int C, int spatial) {
    const int blocks = N * C;
    if (blocks <= 0 || spatial <= 0) return;
    const MirKernel mk = mir_kernel({MirKind::ConvAux, 6u, dt});
    const void* ptrs[] = {dh2.data, d_shift.data};
    const std::int32_t ints[] = {N, C, spatial};
    mir_dispatch_args_3d(mk, static_cast<std::uint32_t>(blocks), 1u, 1u,
                         "resblock_sum_hw", ptrs, ints, {});
}

} // namespace

void resblock_forward(const Tensor& X,
                          const Tensor& gamma1, const Tensor& beta1,
                          const Tensor& W1, const Tensor* b1,
                          const Tensor* t_emb_shift,
                          const Tensor& gamma2, const Tensor& beta2,
                          const Tensor& W2, const Tensor* b2,
                          const Tensor* Wskip, const Tensor* bskip,
                          int N, int C_in, int C_out, int H, int Wd,
                          int num_groups, float eps,
                          Tensor& Y) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("resblock_forward_gpu: X must be FP16 or BF16");
    }
    if (gamma1.dtype != dt || beta1.dtype != dt || W1.dtype != dt ||
        gamma2.dtype != dt || beta2.dtype != dt || W2.dtype != dt) {
        throw std::runtime_error("resblock_forward_gpu: all required tensors must share X's dtype (FP16 or BF16)");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_gpu: num_groups must divide C_in and C_out");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_forward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != dt) {
        Y.resize(N, out_cols, dt);
    }
    if (N == 0 || spatial == 0) return;

    thread_local Tensor h1 = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local Tensor h2 = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local Tensor h3 = Tensor::empty_on(Device::Metal, 0, 0);
    h1.resize(N, C_in  * spatial, dt);
    h2.resize(N, C_out * spatial, dt);
    h3.resize(N, C_out * spatial, dt);

    group_norm_silu_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps, h1);

    conv2d_forward(h1, W1, b1,
                       N, C_in, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       h2);

    if (t_emb_shift) {
        if (t_emb_shift->dtype != dt) {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift dtype must match X");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(dt, h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    group_norm_silu_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps, h3);

    thread_local Tensor skip_scratch = Tensor::empty_on(Device::Metal, 0, 0);
    if (Wskip != nullptr) {
        if (Wskip->dtype != dt) {
            throw std::runtime_error("resblock_forward_gpu: Wskip dtype must match X");
        }
        conv2d_forward(X, *Wskip, bskip,
                           N, C_in, H, Wd,
                           C_out, 1, 1,
                           /*stride*/1, 1,
                           /*pad*/0, 0,
                           /*dil*/1, 1,
                           skip_scratch);
    }

    conv2d_forward(h3, W2, b2,
                       N, C_out, H, Wd,
                       C_out, 3, 3,
                       /*stride*/1, 1,
                       /*pad*/1, 1,
                       /*dil*/1, 1,
                       Y);
    if (Wskip == nullptr) {
        add_inplace(Y, X);
    } else {
        add_inplace(Y, skip_scratch);
    }
}

void resblock_forward_int8w_fp16(const Tensor& X,
                                     const Tensor& gamma1, const Tensor& beta1,
                                     const Tensor& W1_int8, const Tensor& s1,
                                     const Tensor* b1,
                                     const Tensor* t_emb_shift,
                                     const Tensor& gamma2, const Tensor& beta2,
                                     const Tensor& W2_int8, const Tensor& s2,
                                     const Tensor* b2,
                                     const Tensor* Wskip_int8, const Tensor* sskip,
                                     const Tensor* bskip,
                                     int N, int C_in, int C_out, int H, int Wd,
                                     int num_groups, float eps,
                                     Tensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || gamma2.dtype != Dtype::FP16 ||
        beta2.dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: activation/norm tensors must be FP16");
    }
    if (W1_int8.dtype != Dtype::INT8 || W2_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1/W2 must be INT8");
    }
    if (s1.dtype != Dtype::FP32 || s2.dtype != Dtype::FP32) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: scales s1/s2 must be FP32");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: num_groups must divide C_in and C_out");
    }
    if (W1_int8.rows != C_out || W1_int8.cols != C_in * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W1_int8 shape mismatch");
    }
    if (W2_int8.rows != C_out || W2_int8.cols != C_out * 9) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: W2_int8 shape mismatch");
    }
    if (s1.rows != C_out || s1.cols != 1 || s2.rows != C_out || s2.cols != 1) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: s1/s2 must be (C_out, 1)");
    }
    if (b1 && b1->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b1 must be FP16");
    }
    if (b2 && b2->dtype != Dtype::FP16) {
        throw std::runtime_error("resblock_forward_int8w_fp16_gpu: b2 must be FP16");
    }
    if (Wskip_int8 == nullptr) {
        if (C_in != C_out) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip required when C_in != C_out");
        }
    } else {
        if (Wskip_int8->dtype != Dtype::INT8) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 must be INT8");
        }
        if (Wskip_int8->rows != C_out || Wskip_int8->cols != C_in) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: Wskip_int8 shape mismatch");
        }
        if (sskip == nullptr || sskip->dtype != Dtype::FP32 ||
            sskip->rows != C_out || sskip->cols != 1) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: sskip must be FP32 (C_out, 1)");
        }
        if (bskip && bskip->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: bskip must be FP16");
        }
    }
    const int spatial = H * Wd;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    if (N == 0 || spatial == 0) return;

    thread_local Tensor h1 = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local Tensor h2 = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local Tensor h3 = Tensor::empty_on(Device::Metal, 0, 0);
    h1.resize(N, C_in  * spatial, Dtype::FP16);
    h2.resize(N, C_out * spatial, Dtype::FP16);
    h3.resize(N, C_out * spatial, Dtype::FP16);

    group_norm_silu_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps, h1);

    conv2d_int8w_fp16_forward(h1, W1_int8, s1, b1,
                                  N, C_in, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  h2);

    if (t_emb_shift) {
        if (t_emb_shift->dtype != Dtype::FP16) {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift must be FP16");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_forward_int8w_fp16_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(Dtype::FP16, h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    group_norm_silu_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps, h3);

    thread_local Tensor skip_scratch = Tensor::empty_on(Device::Metal, 0, 0);
    if (Wskip_int8 != nullptr) {
        conv2d_int8w_fp16_forward(X, *Wskip_int8, *sskip, bskip,
                                      N, C_in, H, Wd,
                                      C_out, 1, 1,
                                      1, 1, 0, 0, 1, 1, /*groups*/1,
                                      skip_scratch);
    }

    conv2d_int8w_fp16_forward(h3, W2_int8, s2, b2,
                                  N, C_out, H, Wd,
                                  C_out, 3, 3,
                                  1, 1, 1, 1, 1, 1, /*groups*/1,
                                  Y);
    if (Wskip_int8 == nullptr) {
        add_inplace(Y, X);
    } else {
        add_inplace(Y, skip_scratch);
    }
}

void resblock_backward(const Tensor& X,
                           const Tensor& gamma1, const Tensor& beta1,
                           const Tensor& W1, const Tensor* b1,
                           const Tensor* t_emb_shift,
                           const Tensor& gamma2, const Tensor& beta2,
                           const Tensor& W2, const Tensor* b2,
                           const Tensor* Wskip, const Tensor* bskip,
                           int N, int C_in, int C_out, int H, int Wd,
                           int num_groups, float eps,
                           const Tensor& dY,
                           Tensor& dX,
                           Tensor& dGamma1, Tensor& dBeta1,
                           Tensor& dW1, Tensor* db1,
                           Tensor* dt_emb_shift,
                           Tensor& dGamma2, Tensor& dBeta2,
                           Tensor& dW2, Tensor* db2,
                           Tensor* dWskip, Tensor* dbskip) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("resblock_backward_gpu: X must be FP16 or BF16");
    }
    if (dY.dtype != dt || gamma1.dtype != dt || beta1.dtype != dt ||
        W1.dtype != dt || gamma2.dtype != dt || beta2.dtype != dt ||
        W2.dtype != dt) {
        throw std::runtime_error("resblock_backward_gpu: all required tensors must share X's dtype (FP16 or BF16)");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("resblock_backward_gpu: Wskip required when C_in != C_out");
    }
    const int spatial = H * Wd;
    if (dY.rows != N || dY.cols != C_out * spatial) {
        throw std::runtime_error("resblock_backward_gpu: dY shape mismatch");
    }
    if (dX.rows != N || dX.cols != C_in * spatial || dX.dtype != dt) {
        dX.resize(N, C_in * spatial, dt);
    }
    if (N == 0 || spatial == 0) return;

    Tensor h1_pre_silu = Tensor::empty_on(Device::Metal, 0, 0);
    Tensor h1 = Tensor::empty_on(Device::Metal, 0, 0);
    group_norm_forward(X, gamma1, beta1, N, C_in, H, Wd, num_groups, eps,
                           h1_pre_silu);
    silu_forward(h1_pre_silu, h1);

    Tensor h2 = Tensor::empty_on(Device::Metal, 0, 0);
    conv2d_forward(h1, W1, b1, N, C_in, H, Wd,
                       C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2);
    if (t_emb_shift) {
        if (t_emb_shift->dtype != dt) {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift dtype must match X");
        }
        int has_N = 0;
        if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) {
            has_N = 1;
        } else if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
                   (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
                   t_emb_shift->size() == C_out) {
            has_N = 0;
        } else {
            throw std::runtime_error("resblock_backward_gpu: t_emb_shift shape must be (N, C_out) or (C_out,)");
        }
        launch_add_shift(dt, h2, *t_emb_shift, N, C_out, spatial, has_N);
    }

    Tensor h3_pre_silu = Tensor::empty_on(Device::Metal, 0, 0);
    Tensor h3 = Tensor::empty_on(Device::Metal, 0, 0);
    group_norm_forward(h2, gamma2, beta2, N, C_out, H, Wd, num_groups, eps,
                           h3_pre_silu);
    silu_forward(h3_pre_silu, h3);

    Tensor dh3 = Tensor::empty_on(Device::Metal, N, C_out * spatial, dt);
    conv2d_backward_input(W2, dY, N, C_out, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh3);
    conv2d_backward_weight(h3, dY, N, C_out, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) {
        conv2d_backward_bias(dY, N, C_out, H, Wd, *db2);
    }

    Tensor dh3_pre_silu = Tensor::empty_on(Device::Metal, 0, 0);
    silu_backward(h3_pre_silu, dh3, dh3_pre_silu);

    Tensor dh2 = Tensor::empty_on(Device::Metal, 0, 0);
    group_norm_backward(h2, gamma2, dh3_pre_silu, N, C_out, H, Wd,
                            num_groups, eps, dh2, dGamma2, dBeta2);

    if (t_emb_shift && dt_emb_shift) {
        if (dt_emb_shift->dtype != dt) {
            throw std::runtime_error("resblock_backward_gpu: dt_emb_shift dtype must match X");
        }
        const bool has_N = (t_emb_shift->rows == N && t_emb_shift->cols == C_out);
        if (has_N) {
            launch_sum_hw_per_NC(dt, dh2, *dt_emb_shift, N, C_out, spatial);
        } else {
            conv2d_backward_bias(dh2, N, C_out, H, Wd, *dt_emb_shift);
        }
    }

    Tensor dh1 = Tensor::empty_on(Device::Metal, N, C_in * spatial, dt);
    conv2d_backward_input(W1, dh2, N, C_in, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh1);
    conv2d_backward_weight(h1, dh2, N, C_in, H, Wd,
                               C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) {
        conv2d_backward_bias(dh2, N, C_out, H, Wd, *db1);
    }

    Tensor dh1_pre_silu = Tensor::empty_on(Device::Metal, 0, 0);
    silu_backward(h1_pre_silu, dh1, dh1_pre_silu);

    group_norm_backward(X, gamma1, dh1_pre_silu, N, C_in, H, Wd,
                            num_groups, eps, dX, dGamma1, dBeta1);

    if (Wskip == nullptr) {
        add_inplace(dX, dY);
    } else {
        if (Wskip->dtype != dt) {
            throw std::runtime_error("resblock_backward_gpu: Wskip dtype must match X");
        }
        Tensor dX_skip = Tensor::empty_on(Device::Metal, N, C_in * spatial, dt);
        conv2d_backward_input(*Wskip, dY, N, C_in, H, Wd,
                                  C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX_skip);
        if (dWskip) {
            conv2d_backward_weight(X, dY, N, C_in, H, Wd,
                                       C_out, 1, 1, 1, 1, 0, 0, 1, 1, *dWskip);
        }
        if (dbskip) {
            conv2d_backward_bias(dY, N, C_out, H, Wd, *dbskip);
        }
        add_inplace(dX, dX_skip);
    }
}

} // namespace brotensor::detail::metal
