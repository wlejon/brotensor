#include "flash_fused_internal.h"

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"
#include "../kernels/mir_fa_fused_mma.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace brotensor {
namespace flash_fused {

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

// Grow-only per-device scratch for split partials (never freed; graph-capture safe).
float* split_workspace(std::size_t floats, CUstream stream) {
    using namespace brotensor::detail::cuda;
    constexpr int MAX_DEVICES = 16;
    static float* buf[MAX_DEVICES] = {};
    static std::size_t cap[MAX_DEVICES] = {};
    const int dev = current_device();
    if (dev < 0 || dev >= MAX_DEVICES) return nullptr;
    if (cap[dev] >= floats) return buf[dev];
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    BROTENSOR_CUDA_CHECK(drv::cuStreamIsCapturing(stream, &st));
    if (st != CU_STREAM_CAPTURE_STATUS_NONE) return nullptr;
    const std::size_t want = floats > 2 * cap[dev] ? floats : 2 * cap[dev];
    buf[dev] = nullptr;
    cap[dev] = 0;
    CUdeviceptr dptr = 0;
    BROTENSOR_CUDA_CHECK(drv::cuMemAlloc(&dptr, want * sizeof(float)));
    buf[dev] = reinterpret_cast<float*>(static_cast<std::uintptr_t>(dptr));
    cap[dev] = want;
    return buf[dev];
}

int sm_count() {
    using namespace brotensor::detail::cuda;
    constexpr int MAX_DEVICES = 16;
    static int n[MAX_DEVICES] = {};
    const int dev = current_device();
    if (dev < 0 || dev >= MAX_DEVICES) return 80;
    if (!n[dev]) n[dev] = device_attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    return n[dev] ? n[dev] : 80;
}

void launch_common(Dtype dt, const void* Q, const void* K, const void* V, const float* mask, void* O,
                   int Lq, int Lk, int D, int num_heads, int num_kv_heads, int head_dim,
                   bool causal, int batch, CUstream stream) {
    using namespace brotensor::detail::cuda;
    const int br = static_cast<int>(kernels::kFaFusedBr);
    const int bc = static_cast<int>(kernels::fa_fused_bc(static_cast<std::uint32_t>(head_dim)));
    const float scale_log2 = 1.4426950408889634f / std::sqrt(static_cast<float>(head_dim));
    const int qtiles = (Lq + br - 1) / br;

    int splits = 1;
    const long long ctas = static_cast<long long>(qtiles) * num_heads * batch;
    const int n_tiles = (Lk + bc - 1) / bc;
    const int want = 2 * sm_count();
    if (ctas < want && n_tiles >= 8) {
        splits = static_cast<int>((want + ctas - 1) / ctas);
        splits = std::min(splits, std::min(n_tiles / 4, 32));
    }

    float* part_o = nullptr;
    float* part_ml = nullptr;
    if (splits > 1) {
        const std::size_t rows = static_cast<std::size_t>(batch) * splits * Lq;
        float* ws = split_workspace(rows * D + rows * num_heads * 2, stream);
        if (ws) {
            part_o = ws;
            part_ml = ws + rows * D;
        } else {
            splits = 1;
        }
    }

    const std::uint8_t flags = static_cast<std::uint8_t>(
        (causal ? 1u : 0u) | (mask ? 2u : 0u) | (splits > 1 ? 4u : 0u));
    CUfunction fn = mir_function({MirKind::FaFusedMma, static_cast<std::uint32_t>(head_dim),
                                  dt, Dtype::FP32, 1u, Dtype::FP32, flags});
    const std::array<std::uintptr_t, 7> ptrs = {
        addr(Q), addr(K), addr(V), addr(mask), addr(O), addr(part_o), addr(part_ml)};
    const std::array<std::int32_t, 8> ints = {
        Lq, Lk, D, num_kv_heads * head_dim, num_heads / num_kv_heads, splits, qtiles, num_heads};
    const std::array<float, 1> flts = {scale_log2};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>(qtiles),
                         static_cast<std::uint32_t>(num_heads),
                         static_cast<std::uint32_t>(batch * splits),
                         kernels::kFaFusedThreads, ptrs, ints, flts);

    if (splits > 1) {
        CUfunction cfn = mir_function({MirKind::FaAux, 0u, dt, dt, 1u});
        const std::array<std::uintptr_t, 3> cptrs = {addr(part_o), addr(part_ml), addr(O)};
        const std::array<std::int32_t, 5> cints = {Lq, D, num_heads, head_dim, splits};
        mir_dispatch_args_3d(cfn, static_cast<std::uint32_t>(Lq),
                             static_cast<std::uint32_t>(num_heads),
                             static_cast<std::uint32_t>(batch),
                             128u, cptrs, cints, {});
    }
}

}  // namespace

bool supported(int head_dim) {
    if (!brotensor::detail::cuda::supports_sm80()) return false;
    switch (head_dim) {
        case 16: case 32: case 40: case 56: case 64: case 72: case 80: case 96: case 112:
        case 128: case 160:
            return true;
        default:
            return false;
    }
}

void launch(const Fp16* Q, const Fp16* K, const Fp16* V,
            const float* mask, Fp16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            CUstream stream, int batch, int num_kv_heads) {
    const int nkv = num_kv_heads > 0 ? num_kv_heads : num_heads;
    launch_common(Dtype::FP16, Q, K, V, mask, O, Lq, Lk, D, num_heads, nkv, head_dim, causal, batch, stream);
}

void launch(const Bf16* Q, const Bf16* K,
            const Bf16* V,
            const float* mask, Bf16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            CUstream stream, int batch, int num_kv_heads) {
    const int nkv = num_kv_heads > 0 ? num_kv_heads : num_heads;
    launch_common(Dtype::BF16, Q, K, V, mask, O, Lq, Lk, D, num_heads, nkv, head_dim, causal, batch, stream);
}

}  // namespace flash_fused
}  // namespace brotensor
