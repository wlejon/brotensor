#include <brotensor/tensor.h>

#include "cu_mir_cache.h"
#include "ops_fa_mir.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("brotensor: flash_attention_packed_qkv_forward: " + reason);
}

constexpr int kPkHd = 64;
constexpr int kPkBq = 64;
constexpr float kLog2e = 1.4426950408889634f;

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

}  // namespace

void flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV, const ::brotensor::Tensor& seq_bounds,
                                        int num_heads, int window, ::brotensor::Tensor& O) {
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail("QKV.cols must be 3 * num_heads * head_dim");
    if (seq_bounds.dtype != Dtype::INT32) fail("seq_bounds must be INT32");
    const int L = QKV.rows;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail("seq_bounds must be (L, 2)");
    const int D = QKV.cols / 3;
    const int hd = D / num_heads;
    const Dtype dt = QKV.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) fail("QKV must be FP32/FP16/BF16");
    if (O.rows != L || O.cols != D || O.dtype != dt) O.resize(L, D, dt);
    if (L == 0) return;

    if ((dt == Dtype::FP16 || dt == Dtype::BF16) && hd == kPkHd) {
        CUfunction fn = mir_function({MirKind::FaPackedMma, 0u, dt, dt, 1u});
        const int hw = window > 0 ? window / 2 : -1;
        const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(hd));
        const std::array<std::uintptr_t, 3> ptrs = {
            addr(QKV.data), addr(seq_bounds.data), addr(O.data)};
        const std::array<std::int32_t, 3> ints = {L, num_heads, hw};
        const std::array<float, 1> flts = {scale_log2};
        mir_dispatch_args_3d(fn, static_cast<std::uint32_t>((L + kPkBq - 1) / kPkBq),
                             static_cast<std::uint32_t>(num_heads), 1u, 128u, ptrs, ints, flts);
        return;
    }

    const std::size_t esz = (dt == Dtype::FP32) ? 4u : 2u;
    const char* base = static_cast<const char*>(QKV.data);
    dispatch_fa_rows(kFaModePacked, dt, QKV.device,
                     base, base + static_cast<std::size_t>(D) * esz,
                     base + static_cast<std::size_t>(2 * D) * esz,
                     O.data, nullptr, static_cast<const std::int32_t*>(seq_bounds.data), nullptr,
                     L, L, 3 * D, 3 * D, D,
                     num_heads, num_heads, hd,
                     window > 0 ? window : 0, false, false, 0, 0, 0.0f);
}

}  // namespace brotensor::detail::cuda
