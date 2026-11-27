#pragma once

// Short-M 16-bit Tensor Core weight-streaming GEMM MIR builder (9 <= M <= 64):
//   Y(M, N) = epilogue(X(M, K) @ W(N, K)^T + bias)
// 256 threads (8 warps), 128 output cols/CTA, 4-stage K=64 cp.async ring in
// dynamic shared memory, L2::evict_first cache policy for W, swizzled 16-byte
// shared fragment loads into m16n8k16 mma.sync, optional FP32 split-K partials.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct GemmSkinnySpec {
    Dtype dt = Dtype::FP16;
    std::uint8_t nt = 2;        // 2..8 (covers ROWS = 8 * nt >= M)
    std::uint8_t epi = 0;       // 0=store, 1=accum, 2=geglu, 3=swiglu
    std::uint8_t act = 0;       // 0..5
    bool has_bias = false;
    bool has_partial = false;
};

constexpr std::uint32_t gemm_skinny_smem_bytes(std::uint8_t nt) noexcept {
    const std::uint32_t rows = static_cast<std::uint32_t>(nt) * 8u;
    const std::uint32_t ring = 4u * (128u + rows) * 128u;
    const std::uint32_t tile = rows * 132u * 4u;
    return ring > tile ? ring : tile;
}

// Params:
//   ptrs (5): X(0), W(1), Y(2), bias(3), partial(4)
//   ints (4): M(5), N(6), K(7), chunks_per_split(8)
brass::Function* build_gemm_skinny_mir(brass::Module& mod, const GemmSkinnySpec& spec);

}  // namespace brotensor::kernels
