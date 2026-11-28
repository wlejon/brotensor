#pragma once

// Pure-C++ Brass MIR builder table for brotensor's Metal backend.
// Separated from mtl_mir_cache.mm (Objective-C++) so Linux/Windows CI can
// compile and verify Metal MIR -> MSL lowering via test_metal_mir_msl_emit.

#include <brotensor/tensor.h>

#include <cstdint>

namespace brass {
class Function;
class Module;
}  // namespace brass

namespace brotensor::detail::metal {

enum class MirKind : std::uint8_t {
    Unary = 0,
    Binary = 1,
    ActBwd = 2,
    Bias = 3,
    Cast = 4,
    Glu = 5,
    Rowvec = 6,
    SumRows = 7,
    ArgmaxRows = 8,
    SumCols = 9,
    Norm = 10,
    Rope = 11,
    FillBytes = 12,
    Copy2d = 13,
    Transpose = 14,
    ScatterAdd = 15,
    Gnorm = 16,
    Bnorm = 17,
    Sampler = 18,
    Philox = 19,
    Remap = 20,
    Resample = 21,
    ResampleBwd = 22,
    Lstm = 23,
    DeltaRule = 24,
    Xent = 25,
    Topk = 26,
    SelectSample = 27,
    SelectCounter = 28,
    SelectCommit = 29,
    SelectMd = 30,
    Misc = 31,
    Vision = 32,
    Audio = 33,
    Spectral = 34,
    AttnAux = 35,
    AttnBwd = 36,
    FaCombine = 37,
    AttnSoftmax = 38,
    ConvDirect = 39,
    WqRows = 40,
    Dequant = 41,
    Gemv = 42,
    GemmSimt = 43,
    ConvSimt = 44,
    GemvQ = 45,
    FaRows = 46,
    FaBwd = 47,
    ConvAux = 48,
    FilteredLrelu = 49,
    GemmCm = 50,
    ConvCm = 51,
};

struct MirKey {
    MirKind kind = MirKind::Unary;
    std::uint32_t op = 0;
    Dtype dt0 = Dtype::FP32;
    Dtype dt1 = Dtype::FP32;
    std::uint8_t lanes = 1;
    Dtype dt2 = Dtype::FP32;
    std::uint8_t flags = 0;
    std::uint64_t aux = 0;

    bool operator==(const MirKey& o) const noexcept = default;
};

inline std::uint8_t pack_norm_flags(bool has_g, bool has_b, bool has_xh,
                                    bool has_s0, bool has_s1) noexcept {
    return static_cast<std::uint8_t>(
        (has_g ? 1 : 0) | (has_b ? 2 : 0) | (has_xh ? 4 : 0) |
        (has_s0 ? 8 : 0) | (has_s1 ? 16 : 0));
}

struct MetalCmTile {
    std::uint32_t bm, bn;
};
MetalCmTile metal_cm_tile(std::uint32_t cfg_idx) noexcept;

std::uint32_t metal_mir_block_size(const MirKey& key);
brass::Function* build_metal_mir_function(brass::Module& mod, const MirKey& key);

}  // namespace brotensor::detail::metal
