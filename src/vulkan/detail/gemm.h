#pragma once

// The one GEMM entry point every matrix op goes through (gemm.cpp). It picks
//   * gemv.comp       NT layout, at most 8 rows of A, one batch: the decode
//                     regime, bound by reading B once;
//   * gemm_cm.comp    FP16 / BF16 operands on a device with 16x16x16
//                     cooperative-matrix fragments (BF16 converted to FP16 at
//                     load, docs/vulkan.md "Dtype policy");
//   * gemm_simt.comp  everything else: FP32, FP32 activations against 16-bit
//                     weights, and 16-bit operands without cooperative matrix.
// C[z](M, N) = op(A[z]) op(B[z]) + epilogue, see shaders/gemm_common.glsl.

#include "device.h"

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::vulkan {

struct GemmArgs {
    std::uint64_t a = 0, b = 0, c = 0, bias = 0;   // device addresses; bias 0 = none
    ::brotensor::Dtype da = ::brotensor::Dtype::FP32, db = ::brotensor::Dtype::FP32,
                       dc = ::brotensor::Dtype::FP32;   // the bias shares dc, except for 16-bit
                                                         // operands with an FP32 result, where it
                                                         // shares da (SIMT)
    int m = 0, n = 0, k = 0;     // n: columns of the pre-epilogue result r (= rows of B in NT)
    int lda = 0, ldb = 0, ldc = 0;
    bool ta = false;             // A stored (K, M)
    bool nb = false;             // B stored (K, N) instead of (N, K)
    int batch = 1;
    long long sa = 0, sb = 0, sc = 0;   // batch strides in elements; 0 broadcasts
    int epi = 0;                 // EPI_* (op_codes.h); GLUs need the NT layout
    // Quantised B (QF_* of shaders/quant_decode.glsl, 0 = none): the
    // cooperative-matrix kernel only, NT layout, one batch, db ignored; ldb
    // is then B's row pitch in bytes and `scale` the INT8 per-row scales.
    int qb = 0;
    std::uint64_t scale = 0;
    int act = 0;                 // LACT_*
    const char* op = "gemm";     // for error messages
};

// Records the GEMM on `d`'s stream. Throws for an unsupported dtype mix (and,
// with qb set, on a device without cooperative matrix).
void gemm(DeviceCtx& d, const GemmArgs& g);

// Which kernel gemm() would use, for the tests and the benchmark: "gemv",
// "coopmat" or "simt".
const char* gemm_path(DeviceCtx& d, const GemmArgs& g);

// Test / benchmark hook, process-wide: 0 = automatic choice, 1 = never the
// GEMV kernel, 2 = SIMT only (as on a device without cooperative matrix).
void set_gemm_override(int mode);

// Benchmark hook: force cooperative-matrix tile configuration `index` (see
// gemm_cm_config_name), or -1 for the automatic choice. Returns the number of
// configurations.
int set_gemm_cm_config(int index);
const char* gemm_cm_config_name(int index);

}  // namespace brotensor::detail::vulkan
