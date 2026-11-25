#pragma once

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::cuda {

enum FaMode : std::uint32_t {
    kFaModeRows = 0,
    kFaModeVarlen = 1,
    kFaModePacked = 2,
    kFaModeLen = 3,
};

// Dispatch row-wise flash-attention forward via MirKind::FaRows
// (plus MirKind::FaCombine when split-K is enabled).
void dispatch_fa_rows(FaMode mode, Dtype dt, const Device& dev,
                      const void* q, const void* k, const void* v, void* o,
                      const float* mask, const std::int32_t* aux, const std::int32_t* aux2,
                      int lq, int lk, int ldq, int ldk, int ldo,
                      int hq, int hkv, int hd,
                      int window, bool causal, bool maskpos, int nseq, int q_offset,
                      float softcap);

// Dispatch 2-pass flash-attention backward (pass 0: stats + dQ; pass 1: dK + dV)
// via MirKind::FaBwd.
void dispatch_fa_bwd(FaMode mode, Dtype dt, const Device& dev,
                     const void* q, const void* k, const void* v, const void* g,
                     const float* mask, const std::int32_t* aux, const std::int32_t* aux2,
                     void* dq, void* dk, void* dv,
                     int lq, int lk, int ldq, int ldk, int ldg, int ldo,
                     int heads, int hd, int window, bool causal, int nseq);

}  // namespace brotensor::detail::cuda
