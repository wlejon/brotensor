#pragma once

// Ampere/Ada Tensor Core row-major NN GEMM MIR builder:
//   C(M, N) = A(M, K) @ B(K, N)
// Uses 3-stage cp.async.cg pipeline, XOR-swizzled shared tiles, ldmatrix.x4
// for A, ldmatrix.x4.trans for B, and mma.sync.aligned.m16n8k16.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct GemmMmaNnSpec {
    Dtype dt = Dtype::FP16;
    std::uint8_t cfg = 0;  // 0..3 (shares kGemmMmaCfgs tile geometry)
    bool aligned_mnk = false;
    bool swizzle_l2 = false;
};

// Params:
//   ptrs (3): A(0), B(1), C(2)
//   ints (4): M(3), N(4), K(5), grid_m(6)
brass::Function* build_gemm_mma_nn_mir(brass::Module& mod, const GemmMmaNnSpec& spec);

}  // namespace brotensor::kernels
