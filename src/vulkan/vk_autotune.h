#pragma once

// Per-machine Vulkan kernel micro-autotuner backed by KernelDiskCache.

#include "detail/attention.h"
#include "detail/device.h"
#include "detail/gemm.h"

#include <cstdint>

namespace brotensor::detail::vulkan {

struct GemmTuneCfg {
    std::uint32_t cfg_idx = 0;
    std::uint32_t swizzle_group = 0; // 0 = disabled, 4/8 = 2D L2 slab swizzle
    bool valid = false;
};

struct AttnTuneCfg {
    std::uint32_t path = 1;          // 0 = rows, 1 = coopmat
    std::uint32_t bc = 16;
    std::uint32_t nsg = 2;
    std::uint32_t head_swizzle = 0;
    std::uint32_t nsplit_target = 160;
    bool valid = false;
};

/// Returns true when live autotune benchmarking on cache misses is enabled
/// (via `brotensor::set_autotune_enabled(true)` or `BROTENSOR_AUTOTUNE=1`).
bool vk_autotune_live_enabled() noexcept;

/// Looks up a cached cooperative-matrix GEMM config for `(d, g)`. When none
/// is cached and `vk_autotune_live_enabled()` is true (and not capturing),
/// benchmarks candidate tile/swizzle configurations into a scratch buffer and
/// persists the fastest one to `KernelDiskCache`.
GemmTuneCfg autotune_gemm_cm(DeviceCtx& d, const GemmArgs& g, std::uint32_t nout,
                             GemmTuneCfg fallback);

/// Looks up (and optionally benchmarks) the fastest attention path / config
/// for `(d, p)`. Never mutates caller `O` buffer during live tuning.
AttnTuneCfg autotune_attention(DeviceCtx& d, const AttnProblem& p,
                               bool allow_rows, bool allow_cm,
                               AttnTuneCfg fallback);

} // namespace brotensor::detail::vulkan
