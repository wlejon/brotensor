// Shared Q4_K decode helpers. Single source of truth for sc/m unpack used
// by both q4k.cu (dequant + GEMV) and q4k_wmma.cu (fused WMMA GEMM).
//
// Q4_K block layout (144 bytes, 256 elements):
//   [0..1]   fp16 d
//   [2..3]   fp16 dmin
//   [4..15]  uint8 scales[12]  -- eight packed (sc, m) 6-bit pairs
//   [16..143] uint8 qs[128]   -- 256 nibbles, 8 sub-blocks of 32
// Decoded value: y = (d * sc[is]) * nibble - (dmin * m[is]).

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace brotensor::detail::cuda::q4k {

// Recovers the j-th 6-bit sub-scale `sc` and sub-min `m` from the 12-byte
// packed scales array. j in [0, 8).
__device__ __forceinline__
void unpack_sc_m(int j, const uint8_t* scales, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = scales[j]     & 0x3Fu;
        m  = scales[j + 4] & 0x3Fu;
    } else {
        sc = (scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4);
        m  = (scales[j + 4] >> 4)    | ((scales[j - 0] >> 6) << 4);
    }
}

// Same recovery, but reading the 16-byte block header out of registers rather
// than memory. A block is 144 bytes = 9 * 16, so every block start is 16-byte
// aligned and `d`, `dmin` and all twelve scale bytes fit in one uint4 load —
// one request instead of the three-plus separate byte loads the memory form
// issues per sub-block. `h` is bytes [0..15] of the block: h.x = d | dmin<<16,
// h.y/h.z/h.w = scales[0..3] / [4..7] / [8..11].
__device__ __forceinline__ uint32_t scale_byte(const uint4& h, int s) {
    const uint32_t b = 4u + static_cast<uint32_t>(s);   // byte index in the block
    const uint32_t w = (b < 8u) ? h.y : ((b < 12u) ? h.z : h.w);
    return (w >> ((b & 3u) * 8u)) & 0xFFu;
}

__device__ __forceinline__
void unpack_sc_m(int j, const uint4& h, uint32_t& sc, uint32_t& m) {
    if (j < 4) {
        sc = scale_byte(h, j)     & 0x3Fu;
        m  = scale_byte(h, j + 4) & 0x3Fu;
    } else {
        sc = (scale_byte(h, j + 4) & 0x0Fu) | ((scale_byte(h, j - 4) >> 6) << 4);
        m  = (scale_byte(h, j + 4) >> 4)    | ((scale_byte(h, j)     >> 6) << 4);
    }
}

constexpr int kBlockBytes    = 144;
constexpr int kBlockElems    = 256;
constexpr int kSubBlockElems = 32;
constexpr int kSubBlocks     = 8;   // kBlockElems / kSubBlockElems
constexpr int kQsBytes       = 128;
constexpr int kDOffset       = 0;
constexpr int kDminOffset    = 2;
constexpr int kScalesOffset  = 4;
constexpr int kQsOffset      = 16;

}  // namespace brotensor::detail::cuda::q4k
