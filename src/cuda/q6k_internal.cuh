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
#include "detail/load_align.cuh"

#include <cuda_runtime.h>   // __ldg
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

// ─── Register-resident form, four elements at a time ──────────────────────
//
// decode_element() re-derives the layout and issues two byte loads per
// element. A run of four consecutive elements starting at a multiple of four
// shares its group, quad and sub-block (4 divides both 32 and 16), and needs
// four *consecutive* ql bytes and four consecutive qh bytes — so the whole
// run costs two 4-byte reads instead of eight 1-byte ones.
//
// Those reads cannot be a single uint32 load: a block is 210 bytes and
// 210 % 4 == 2, so every other block start is only 2-byte aligned. Every
// offset involved is even, though, so a pair of uint16 loads is always legal.
// That is load_u32_align2(), in detail/load_align.cuh - Q8_0 has the same
// problem for the same reason (34-byte block), so it lives outside this
// header.
struct QuadDesc {
    int ql_off;      // byte offset of the run within ql
    int qh_off;      // byte offset of the run within qh
    int sb;          // sub-block index -> scales[sb]
    int qh_shift;    // bit position of this quad's 2 high bits
    bool low_nib;    // take the low nibble of ql (quad < 2) rather than the high
};

__device__ __forceinline__ QuadDesc quad_desc(int e0) {
    const int group = e0 >> 7;
    const int local = e0 - (group << 7);
    const int quad  = local >> 5;
    const int l     = local - (quad << 5);
    QuadDesc q;
    q.ql_off   = group * 64 + (quad & 1) * 32 + l;
    q.qh_off   = group * 32 + l;
    q.sb       = (group << 3) + (quad << 1) + (l >> 4);
    q.qh_shift = quad * 2;
    q.low_nib  = quad < 2;
    return q;
}

// c-th element (0..3) of the run described by `q`, from the two packed words.
__device__ __forceinline__ int decode_packed(const QuadDesc& q, uint32_t ql4,
                                             uint32_t qh4, int c) {
    const uint32_t qlb  = (ql4 >> (c * 8)) & 0xFFu;
    const uint32_t qhb  = (qh4 >> (c * 8)) & 0xFFu;
    const uint32_t raw4 = q.low_nib ? (qlb & 0x0Fu) : (qlb >> 4);
    const uint32_t hi2  = (qhb >> q.qh_shift) & 0x03u;
    return static_cast<int>(raw4 | (hi2 << 4)) - 32;
}

}  // namespace brotensor::detail::cuda::q6k
