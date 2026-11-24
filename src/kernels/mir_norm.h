#pragma once

// Backend-neutral brass MIR kernel for row normalizations, row softmax,
// and their column reductions (LayerNorm/RMSNorm/L2/Softmax fwd+bwd + col sums).

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kNormBlock = 256;

struct NormMirFlags {
    bool has_g = false;
    bool has_b = false;
    bool has_xh = false;
    bool has_s0 = false;
    bool has_s1 = false;

    std::uint8_t pack() const noexcept {
        return static_cast<std::uint8_t>(
            (has_g ? 1 : 0) | (has_b ? 2 : 0) | (has_xh ? 4 : 0) |
            (has_s0 ? 8 : 0) | (has_s1 ? 16 : 0));
    }
    static NormMirFlags unpack(std::uint8_t m) noexcept {
        return {
            (m & 1) != 0, (m & 2) != 0, (m & 4) != 0, (m & 8) != 0, (m & 16) != 0};
    }
};

// Params (11): (ptr x, ptr y, ptr g, ptr b, ptr xh, ptr s0, ptr s1,
//               i32 rows, i32 d, f32 eps, f32 rstd).
brass::Function* build_norm_mir(brass::Module& mod, std::uint32_t mode, Dtype dt_x, Dtype dt_y,
                                Dtype dt_g, NormMirFlags flags, int lanes);

}  // namespace brotensor::kernels
