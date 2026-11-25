#pragma once

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::cuda {

// Dispatch C[batch](m, n_out) = op(A[batch], B[batch], bias) via MirKind::GemmSimt
// (plus MirKind::Glu when epi is EPI_GEGLU or EPI_SWIGLU).
void dispatch_mir_gemm_simt(const void* a, Dtype dt_a,
                            const void* b, Dtype dt_b,
                            void* c, Dtype dt_c,
                            const void* bias, Dtype dt_bias,
                            int batch, int m, int n, int k,
                            int lda, int ldb, int ldc,
                            std::int64_t sa, std::int64_t sb, std::int64_t sc,
                            bool ta, bool nb,
                            int epi, int act,
                            const Device& dev);

// Dispatch C(m, n_out) = op(A(m, k), B(n, k)^T, bias) via MirKind::Gemv.
// Caller must ensure bias (when non-null) has dtype == dt_c.
void dispatch_mir_gemv(const void* a, Dtype dt_a,
                       const void* b, Dtype dt_b,
                       void* c, Dtype dt_c,
                       const void* bias,
                       int m, int n, int k,
                       int lda, int ldb, int ldc,
                       int epi, int act);

// Apply MirKind::Glu (GLU_SWIGLU) over scratch r(m, n) -> y(m, n / 2) in dt.
void dispatch_mir_swiglu(const Tensor& r, Tensor& y, int m, int n);

}  // namespace brotensor::detail::cuda
