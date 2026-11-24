// CUDA direct 2D/3D forward & transposed-gather convolution dispatcher
// backed by Brass MIR -> PTX (MirKind::ConvDirect).

#include "cu_mir_cache.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;

void dispatch_conv_direct(std::uint32_t mode, Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(N) * static_cast<std::uint64_t>(Cout) *
        static_cast<std::uint64_t>(To) * static_cast<std::uint64_t>(Ho) *
        static_cast<std::uint64_t>(Wo);
    if (total == 0) return;
    if (total > 0xffffffffULL) {
        throw std::runtime_error("brotensor: conv: output too large for one dispatch");
    }
    CUfunction fn = mir_function(
        MirKey{MirKind::ConvDirect, mode, dt, Dtype::FP32, 1, Dtype::FP32,
               static_cast<std::uint8_t>(bias ? 1u : 0u)});
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>((total + 255u) / 256u, 1u, 65535u));
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(x),
        reinterpret_cast<std::uintptr_t>(w),
        reinterpret_cast<std::uintptr_t>(y),
        reinterpret_cast<std::uintptr_t>(bias),
    };
    const std::int32_t ints[] = {
        N, Cin, T, H, Wd, Cout, To, Ho, Wo, kT, kH, kW,
        st, sh, sw, pt, ph, pw, dt_, dh, dw, groups,
    };
    mir_dispatch_args(fn, gx, 256u, ptrs, ints, {});
}

}  // namespace brotensor::detail::cuda
