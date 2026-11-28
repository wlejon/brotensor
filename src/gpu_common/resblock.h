#pragma once

// Shared host-side validation, shape checks, and t_emb_shift parsing for
// UNet ResBlock forward (FP16/BF16 and INT8W) and backward on CUDA and Metal.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::gpu_common {

inline int parse_shift_has_n(const Tensor* t_emb_shift, Dtype dt, int N, int C_out, const char* op) {
    if (t_emb_shift->dtype != dt) {
        throw std::runtime_error(std::string(op) + ": t_emb_shift dtype must match X");
    }
    if (t_emb_shift->rows == N && t_emb_shift->cols == C_out) return 1;
    if ((t_emb_shift->rows == C_out && t_emb_shift->cols == 1) ||
        (t_emb_shift->rows == 1 && t_emb_shift->cols == C_out) ||
        t_emb_shift->size() == C_out) {
        return 0;
    }
    throw std::runtime_error(std::string(op) + ": t_emb_shift shape must be (N, C_out) or (C_out,)");
}

inline void validate_resblock_fp16_bf16(const char* op, const Tensor& X,
                                        const Tensor& gamma1, const Tensor& beta1, const Tensor& W1,
                                        const Tensor& gamma2, const Tensor& beta2, const Tensor& W2,
                                        const Tensor* Wskip,
                                        int C_in, int C_out, int num_groups) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": X must be FP16 or BF16");
    }
    if (gamma1.dtype != dt || beta1.dtype != dt || W1.dtype != dt ||
        gamma2.dtype != dt || beta2.dtype != dt || W2.dtype != dt) {
        throw std::runtime_error(
            std::string(op) + ": all required tensors must share X's dtype (FP16 or BF16)");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error(std::string(op) + ": num_groups must divide C_in and C_out");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error(std::string(op) + ": Wskip required when C_in != C_out");
    }
    if (Wskip != nullptr && Wskip->dtype != dt) {
        throw std::runtime_error(std::string(op) + ": Wskip dtype must match X");
    }
}

inline void validate_resblock_int8w(const char* op, const Tensor& X,
                                    const Tensor& gamma1, const Tensor& beta1,
                                    const Tensor& W1_int8, const Tensor& s1, const Tensor* b1,
                                    const Tensor& gamma2, const Tensor& beta2,
                                    const Tensor& W2_int8, const Tensor& s2, const Tensor* b2,
                                    const Tensor* Wskip_int8, const Tensor* sskip, const Tensor* bskip,
                                    int C_in, int C_out, int num_groups) {
    if (X.dtype != Dtype::FP16 || gamma1.dtype != Dtype::FP16 ||
        beta1.dtype != Dtype::FP16 || gamma2.dtype != Dtype::FP16 || beta2.dtype != Dtype::FP16) {
        throw std::runtime_error(std::string(op) + ": activation/norm tensors must be FP16");
    }
    if (W1_int8.dtype != Dtype::INT8 || W2_int8.dtype != Dtype::INT8) {
        throw std::runtime_error(std::string(op) + ": W1/W2 must be INT8");
    }
    if (s1.dtype != Dtype::FP32 || s2.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": scales s1/s2 must be FP32");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error(std::string(op) + ": num_groups must divide C_in and C_out");
    }
    if (W1_int8.rows != C_out || W1_int8.cols != C_in * 9) {
        throw std::runtime_error(std::string(op) + ": W1_int8 shape mismatch");
    }
    if (W2_int8.rows != C_out || W2_int8.cols != C_out * 9) {
        throw std::runtime_error(std::string(op) + ": W2_int8 shape mismatch");
    }
    if (s1.rows != C_out || s1.cols != 1 || s2.rows != C_out || s2.cols != 1) {
        throw std::runtime_error(std::string(op) + ": s1/s2 must be (C_out, 1)");
    }
    if (b1 && b1->dtype != Dtype::FP16) {
        throw std::runtime_error(std::string(op) + ": b1 must be FP16");
    }
    if (b2 && b2->dtype != Dtype::FP16) {
        throw std::runtime_error(std::string(op) + ": b2 must be FP16");
    }
    if (Wskip_int8 == nullptr) {
        if (C_in != C_out) {
            throw std::runtime_error(std::string(op) + ": Wskip required when C_in != C_out");
        }
    } else {
        if (Wskip_int8->dtype != Dtype::INT8) {
            throw std::runtime_error(std::string(op) + ": Wskip_int8 must be INT8");
        }
        if (Wskip_int8->rows != C_out || Wskip_int8->cols != C_in) {
            throw std::runtime_error(std::string(op) + ": Wskip_int8 shape mismatch");
        }
        if (sskip == nullptr || sskip->dtype != Dtype::FP32 ||
            sskip->rows != C_out || sskip->cols != 1) {
            throw std::runtime_error(std::string(op) + ": sskip must be FP32 (C_out, 1)");
        }
        if (bskip && bskip->dtype != Dtype::FP16) {
            throw std::runtime_error(std::string(op) + ": bskip must be FP16");
        }
    }
}

}  // namespace brotensor::gpu_common
