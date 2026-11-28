#include "decode_gqa.h"

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda::decode_gqa {

namespace {

constexpr int kBlock = 128;
constexpr int kMaxSplits = 64;
constexpr float kLog2e = 1.44269504088896340736f;

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

// Grow-only per-device record scratch (never freed, outgrown buffers included);
// growth is refused during a graph capture (warm up eagerly first).
float* workspace(std::size_t floats, CUstream stream) {
    constexpr int MAX_DEVICES = 16;
    static float* buf[MAX_DEVICES] = {};
    static std::size_t capf[MAX_DEVICES] = {};
    const int dev = current_device();
    if (dev < 0 || dev >= MAX_DEVICES) throw std::runtime_error("flash_attention_decode_len: device out of range");
    if (capf[dev] >= floats) return buf[dev];
    CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
    BROTENSOR_CUDA_CHECK(drv::cuStreamIsCapturing(stream, &cs));
    if (cs != CU_STREAM_CAPTURE_STATUS_NONE) {
        throw std::runtime_error(
            "flash_attention_decode_len: workspace cannot grow during graph capture; run the op once eagerly first");
    }
    buf[dev] = nullptr;
    capf[dev] = 0;
    CUdeviceptr dptr = 0;
    BROTENSOR_CUDA_CHECK(drv::cuMemAlloc(&dptr, floats * sizeof(float)));
    buf[dev] = reinterpret_cast<float*>(static_cast<std::uintptr_t>(dptr));
    capf[dev] = floats;
    return buf[dev];
}

int sm_count() {
    constexpr int MAX_DEVICES = 16;
    static int n[MAX_DEVICES] = {};
    const int dev = current_device();
    if (dev < 0 || dev >= MAX_DEVICES) return 80;
    if (!n[dev]) n[dev] = device_attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    return n[dev] ? n[dev] : 80;
}

std::uint8_t encode_hd(int hd) {
    if (hd == 64) return 0u;
    if (hd == 128) return 1u;
    return 2u;
}

std::uint8_t encode_g(int g) {
    if (g == 1) return 0u;
    if (g == 2) return 1u;
    if (g == 4) return 2u;
    return 3u;
}

}  // namespace

bool supported(int head_dim, int group) {
    if (!supports_sm80()) return false;
    return (head_dim == 64 || head_dim == 128 || head_dim == 256) &&
           (group == 1 || group == 2 || group == 4 || group == 8);
}

void launch(const void* Q, const void* K, const void* V, const std::int32_t* d_len, void* O, int cap, int nq, int nkv,
            int hd, int rows, bool bf16, CUstream stream) {
    const Dtype dt = bf16 ? Dtype::BF16 : Dtype::FP16;
    const int G = nq / nkv;
    int n_splits = (2 * sm_count() + nkv - 1) / nkv;
    n_splits = std::max(1, std::min({n_splits, kMaxSplits, (cap + 31) / 32}));
    float* ws = workspace(static_cast<std::size_t>(rows) * nq * n_splits * (hd + 2), stream);

    const std::uint8_t hd_code = encode_hd(hd);
    const std::uint8_t g_code = encode_g(G);
    const std::uint8_t rc_code = (rows == 1) ? 0u : 1u;
    const std::uint8_t flags = static_cast<std::uint8_t>(hd_code | (g_code << 2) | (rc_code << 4));

    CUfunction pfn = mir_function({MirKind::DecodeGqa, 0u, dt, Dtype::FP32, 1u, Dtype::FP32, flags});
    const float inv_sqrt_log2 = kLog2e / std::sqrt(static_cast<float>(hd));
    const std::array<std::uintptr_t, 5> ptrs = {addr(Q), addr(K), addr(V), addr(d_len), addr(ws)};
    const std::array<float, 1> flts = {inv_sqrt_log2};

    const int rc = (rows == 1) ? 1 : (8 / G);
    for (int r0 = 0; r0 < rows; r0 += rc) {
        const std::array<std::int32_t, 6> ints = {cap, nkv * hd, n_splits, nq, rows, r0};
        mir_dispatch_args_3d(pfn, static_cast<std::uint32_t>(n_splits),
                             static_cast<std::uint32_t>(nkv), 1u, kBlock, ptrs, ints, flts);
    }

    CUfunction rfn = mir_function({MirKind::DecodeGqa, 1u, dt, dt, 1u});
    const std::array<std::uintptr_t, 2> rptrs = {addr(ws), addr(O)};
    const std::array<std::int32_t, 2> rints = {hd, n_splits};
    mir_dispatch_args_3d(rfn, static_cast<std::uint32_t>(rows * nq), 1u, 1u, kBlock, rptrs, rints, {});
}

}  // namespace brotensor::detail::cuda::decode_gqa
