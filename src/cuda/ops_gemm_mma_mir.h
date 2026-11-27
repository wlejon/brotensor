#pragma once

#include <brotensor/tensor.h>

#include <cstddef>

namespace brotensor::detail::cuda::mma_gemm {

struct MirSplitPlan {
    std::uint8_t cfg = 0;
    int splits = 1;
    int kps = 0;
};

MirSplitPlan plan_mir_gemm_mma(int m, int n, int k, int sms, bool allow_splitk);

void launch_mir_gemm_mma(Dtype dt, const void* a, const void* w, void* c,
                         int m, int n, int k, const void* bias,
                         int act, int epi, float* ws, std::size_t ws_floats,
                         int sms, bool fast_accum);

bool launch_mir_gemm_mma_batched(Dtype dt, const void* a, const void* w, void* c,
                                 int batch, int m, int n, int k,
                                 std::size_t sa, std::size_t sb, std::size_t sc,
                                 const void* bias, int act, int sms);

void launch_mir_gemm_mma_f32out(Dtype dt, const void* a, const void* w, float* c,
                                int m, int n, int k, int sms);

void launch_mir_splitk_reduce(Dtype dt, const float* partial, int splits,
                              int m, int n, void* c, const void* bias,
                              int act, int epi);

}  // namespace brotensor::detail::cuda::mma_gemm
