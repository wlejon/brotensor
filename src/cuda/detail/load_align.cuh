// Wide reads from GGUF block-quant payloads whose block size is not a
// multiple of four.
//
// The quant GEMVs all want to pull four consecutive weight bytes at a time:
// four elements share a scale and a sub-block, so one 4-byte read replaces
// four 1-byte ones, and request count is what these kernels are actually
// limited by. But Q6_K blocks are 210 bytes and Q8_0 blocks are 34, and
// neither divides by four - so every other block start is only 2-byte aligned
// and a single 32-bit load would fault on half the blocks.
//
// Every offset involved is even, though, so a pair of 16-bit loads is always
// legal and still halves the request count against reading byte by byte.

#pragma once

#include <cstdint>

namespace brotensor::detail::cuda {

// Four bytes at `p`, which is assumed 2-byte aligned but not 4-byte aligned.
__device__ __forceinline__ uint32_t load_u32_align2(const uint8_t* p) {
    const uint32_t lo = __ldg(reinterpret_cast<const uint16_t*>(p));
    const uint32_t hi = __ldg(reinterpret_cast<const uint16_t*>(p + 2));
    return lo | (hi << 16);
}

}  // namespace brotensor::detail::cuda
