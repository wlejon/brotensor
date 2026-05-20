// CUDA concat / split / copy_d2d. Phase 2G port — kernel bodies unchanged.
// Also hosts `fill_cuda_vtable_specialised` — the Phase 2G master vtable
// fill function for this whole cluster.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;
using ::brotensor::dtype_size_bytes;

void concat_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    int total = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (const auto* p : parts) {
        if (!p) continue;
        total += p->size();
        if (!seen) { dt = p->dtype; seen = true; }
    }
    if (out.rows != total || out.cols != 1 || out.dtype != dt) {
        out.resize(total, 1, dt);
    }
    if (total == 0) return;

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    char* dst_base = reinterpret_cast<char*>(out.data);
    size_t off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(dst_base + off_bytes, p->data,
                                       elem * n,
                                       cudaMemcpyDeviceToDevice));
        off_bytes += elem * n;
    }
}

void split_rows(const Tensor& in, const std::vector<Tensor*>& parts) {
    const size_t elem = static_cast<size_t>(dtype_size_bytes(in.dtype));
    const char* src_base = reinterpret_cast<const char*>(in.data);
    size_t off_bytes = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(p->data, src_base + off_bytes,
                                       elem * static_cast<size_t>(n),
                                       cudaMemcpyDeviceToDevice));
        off_bytes += elem * static_cast<size_t>(n);
    }
}

void concat_batched_rows(const std::vector<const Tensor*>& parts,
                         Tensor& out) {
    if (parts.empty()) { out.resize(0, 0); return; }
    int B = 0;
    int total_cols = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (const auto* p : parts) {
        if (!p) continue;
        if (B == 0) B = p->rows;
        total_cols += p->cols;
        if (!seen) { dt = p->dtype; seen = true; }
    }
    if (out.rows != B || out.cols != total_cols || out.dtype != dt) {
        out.resize(B, total_cols, dt);
    }
    if (B == 0 || total_cols == 0) return;

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    const size_t dst_pitch = elem * total_cols;
    char* dst_base = reinterpret_cast<char*>(out.data);
    size_t col_off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_base + col_off_bytes, dst_pitch,
            p->data,                  elem * d,
            elem * d,                 B,
            cudaMemcpyDeviceToDevice));
        col_off_bytes += elem * d;
    }
}

void concat_nchw_channels(const std::vector<const Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          Tensor& out) {
    if (parts.size() != C_per_part.size()) {
        throw std::runtime_error("concat_nchw_channels: parts.size() != C_per_part.size()");
    }
    int total_C = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto* p = parts[i];
        const int Ci = C_per_part[i];
        if (!p) throw std::runtime_error("concat_nchw_channels: null part");
        if (!seen) { dt = p->dtype; seen = true; }
        else if (p->dtype != dt) {
            throw std::runtime_error("concat_nchw_channels: dtype mismatch across parts");
        }
        if (p->size() != static_cast<int>(static_cast<size_t>(N) * Ci * H * W)) {
            throw std::runtime_error("concat_nchw_channels: part size mismatch (expected N*C_i*H*W)");
        }
        total_C += Ci;
    }
    const int total_cols = total_C * H * W;
    if (out.rows != N || out.cols != total_cols || out.dtype != dt) {
        out.resize(N, total_cols, dt);
    }
    if (N == 0 || total_cols == 0) return;

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    const size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t dst_pitch = elem * static_cast<size_t>(total_C) * HW;
    char* dst_base = reinterpret_cast<char*>(out.data);
    size_t c_off = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const int Ci = C_per_part[i];
        if (Ci == 0) continue;
        const size_t width_bytes = elem * static_cast<size_t>(Ci) * HW;
        BROTENSOR_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_base + c_off * HW * elem, dst_pitch,
            parts[i]->data,                width_bytes,
            width_bytes,                   static_cast<size_t>(N),
            cudaMemcpyDeviceToDevice));
        c_off += static_cast<size_t>(Ci);
    }
}

void concat_nchw_channels_backward(const Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts) {
    if (parts.size() != C_per_part.size()) {
        throw std::runtime_error("concat_nchw_channels_backward: parts.size() != C_per_part.size()");
    }
    int total_C = 0;
    for (int Ci : C_per_part) total_C += Ci;
    const int expected_cols = total_C * H * W;
    if (dY.rows != N || dY.cols != expected_cols) {
        throw std::runtime_error("concat_nchw_channels_backward: dY shape mismatch (expected N x total_C*H*W)");
    }
    const Dtype dt = dY.dtype;
    if (dt != Dtype::FP32 && dt != Dtype::FP16) {
        throw std::runtime_error("concat_nchw_channels_backward: dY dtype must be FP16 or FP32");
    }

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    const size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t src_pitch = elem * static_cast<size_t>(total_C) * HW;
    const char* src_base = reinterpret_cast<const char*>(dY.data);

    size_t c_off = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        Tensor* p = parts[i];
        const int Ci = C_per_part[i];
        if (!p) throw std::runtime_error("concat_nchw_channels_backward: null part");
        const int cols = Ci * H * W;
        if (p->rows != N || p->cols != cols || p->dtype != dt) {
            p->resize(N, cols, dt);
        }
        if (Ci == 0 || N == 0 || HW == 0) {
            c_off += static_cast<size_t>(Ci);
            continue;
        }
        const size_t width_bytes = elem * static_cast<size_t>(Ci) * HW;
        BROTENSOR_CUDA_CHECK(cudaMemcpy2DAsync(
            p->data,                          width_bytes,
            src_base + c_off * HW * elem,     src_pitch,
            width_bytes,                       static_cast<size_t>(N),
            cudaMemcpyDeviceToDevice));
        c_off += static_cast<size_t>(Ci);
    }
}

void copy_d2d(const Tensor& src, int src_off,
              Tensor& dst,       int dst_off,
              int n) {
    if (n <= 0) return;
    const size_t elem = static_cast<size_t>(dtype_size_bytes(src.dtype));
    const char* src_base = reinterpret_cast<const char*>(src.data);
    char*       dst_base = reinterpret_cast<char*>(dst.data);
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        dst_base + static_cast<size_t>(dst_off) * elem,
        src_base + static_cast<size_t>(src_off) * elem,
        elem * static_cast<size_t>(n),
        cudaMemcpyDeviceToDevice));
}

// ─── Forward declarations for the rest of the specialised cluster ──────────
//
// Defined in batched_ops.cu / loss.cu / embedding.cu / kv_cache.cu /
// int8_quant.cu. All live in this namespace.

void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD);
void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD);
void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD);
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD);
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD, Tensor& dX_BD);
void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD, Tensor& dX_BD);
void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD, Tensor& dW, Tensor& dB);

float mse_vec_forward(const Tensor& pred, const Tensor& target);
void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred);
void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample);
float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask,
                         Tensor& probs, Tensor& dLogits);
void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL,
                                const float* d_mask_BL, const int* d_head_offsets,
                                int n_heads, Tensor& probs_BL, Tensor& dLogits_BL,
                                Tensor& loss_per_sample);

void embedding_lookup_forward(const Tensor& table, const int32_t* d_idx,
                              int B, Tensor& out);
void embedding_lookup_backward(const Tensor& dOut, const int32_t* d_idx,
                               int B, Tensor& dTable);

void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache);
void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_heads, Tensor& O);

void matmul_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                       const Tensor& X, Tensor& Y);
void linear_forward_batched_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD, Tensor& Y_BD);
void conv2d_int8w_fp16_forward(const Tensor& X, const Tensor& W_int8,
                               const Tensor& scales, const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y);

// ─── Master vtable fill ────────────────────────────────────────────────────

void fill_cuda_vtable_specialised(::brotensor::detail::OpsVTable& v) {
    // Batched (inference) + batched backward
    v.linear_forward_batched   = &linear_forward_batched;
    v.relu_forward_batched     = &relu_forward_batched;
    v.tanh_forward_batched     = &tanh_forward_batched;
    v.add_inplace_batched      = &add_inplace_batched;
    v.relu_backward_batched    = &relu_backward_batched;
    v.tanh_backward_batched    = &tanh_backward_batched;
    v.linear_backward_batched  = &linear_backward_batched;

    // Losses
    v.mse_vec_forward          = &mse_vec_forward;
    v.mse_vec_backward         = &mse_vec_backward;
    v.mse_vec_per_sample       = &mse_vec_per_sample;
    v.softmax_xent_fused       = &softmax_xent_fused;
    v.softmax_xent_fused_batched = &softmax_xent_fused_batched;

    // Embedding
    v.embedding_lookup_forward  = &embedding_lookup_forward;
    v.embedding_lookup_backward = &embedding_lookup_backward;

    // Concat / split / copy
    v.concat_rows                    = &concat_rows;
    v.split_rows                     = &split_rows;
    v.concat_batched_rows            = &concat_batched_rows;
    v.concat_nchw_channels           = &concat_nchw_channels;
    v.concat_nchw_channels_backward  = &concat_nchw_channels_backward;
    v.copy_d2d                       = &copy_d2d;

    // KV-cache + decode
    v.kv_cache_append          = &kv_cache_append;
    v.flash_attention_decode   = &flash_attention_decode;

    // INT8 weight-only paths
    v.matmul_int8w_fp16                 = &matmul_int8w_fp16;
    v.linear_forward_batched_int8w_fp16 = &linear_forward_batched_int8w_fp16;
    v.conv2d_int8w_fp16_forward         = &conv2d_int8w_fp16_forward;
}

} // namespace brotensor::detail::cuda
