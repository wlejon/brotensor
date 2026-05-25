// Shared Q6_K decode helpers. Single source of truth for the 6-bit element
// reconstruction used by both q6k.cu (dequant + GEMV) and q6k_wmma.cu
// (fused WMMA GEMM).
//
// Q6_K block layout (210 bytes, 256 elements):
//   [0..127]    uint8 ql[128]   -- low 4 bits, 256 nibbles
//   [128..191]  uint8 qh[64]    -- high 2 bits, packed 4-per-byte
//   [192..207]  int8  scales[16] -- 16 signed sub-block scales
//   [208..209]  fp16  d
//
// Elements emit in two groups of 128. Within a group of 128 the four
// "quads" (each 32 elements) are interleaved across ql/qh in the layout
// re-expressed in q6k_decode_element() below. Each sub-block of 16
// elements has its own int8 scale.

#pragma once
#include <cstdint>

namespace brotensor::detail::cuda::q6k {

constexpr int kBlockBytes    = 210;
constexpr int kBlockElems    = 256;
constexpr int kSubBlockElems = 16;
constexpr int kSubBlocks     = 16;   // kBlockElems / kSubBlockElems
constexpr int kQlOffset      = 0;
constexpr int kQhOffset      = 128;
constexpr int kScalesOffset  = 192;
constexpr int kDOffset       = 208;

// Given element index `e` in [0, 256), recover the signed 6-bit value and
// the sub-block index `sb` that selects the scale. Pulls bytes from the
// provided ql / qh arrays.
__device__ __forceinline__
void decode_element(int e, const uint8_t* ql, const uint8_t* qh,
                    int& sb_out, int& val6_out) {
    const int group = e >> 7;             // 0..1
    const int local = e - (group << 7);   // 0..127
    const int quad  = local >> 5;         // 0..3
    const int l     = local - (quad << 5); // 0..31

    const int sb = (group << 3) + (quad << 1) + (l >> 4);   // 0..15

    const uint8_t ql_b = ql[group * 64 + (quad & 1) * 32 + l];
    const uint8_t qh_b = qh[group * 32 + l];
    const int raw4  = (quad < 2) ? (ql_b & 0x0F) : (ql_b >> 4);
    const int high2 = (qh_b >> (quad * 2)) & 0x03;
    const int val6  = static_cast<int>(raw4 | (high2 << 4)) - 32;

    sb_out   = sb;
    val6_out = val6;
}

}  // namespace brotensor::detail::cuda::q6k
