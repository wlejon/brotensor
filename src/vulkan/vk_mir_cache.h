#pragma once

// Vulkan runtime pipeline cache for backend-neutral brass MIR kernels.

#include "detail/device.h"
#include "vulkan_jit.h"

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::vulkan {

enum class MirKind : std::uint8_t {
    Unary = 0,
    Binary = 1,
    ActBwd = 2,
    Bias = 3,
    Cast = 4,
    Glu = 5,
    Rowvec = 6,
    SumRows = 7,
    ArgmaxRows = 8,
    SumCols = 9,
    Norm = 10,
    Rope = 11,
};

struct MirKey {
    MirKind kind = MirKind::Unary;
    std::uint32_t op = 0;
    Dtype dt0 = Dtype::FP32;
    Dtype dt1 = Dtype::FP32;
    std::uint8_t lanes = 1;
    Dtype dt2 = Dtype::FP32;
    std::uint8_t flags = 0;

    bool operator==(const MirKey& o) const noexcept = default;
};

// Compiles on first use (per key) and interns pipelines per device context.
struct MirKernel {
    const brass::target::SpirvKernel* spv = nullptr;
    JitPipeline pipe{};
};

MirKernel mir_kernel(DeviceCtx& d, const MirKey& key);

// Chooses 128-bit (lanes=4 f32 / lanes=8 f16,bf16), 64-bit (lanes=4 f16,bf16),
// or scalar (lanes=1) based on element count and pointer alignment.
int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uint64_t a0, std::uint64_t a1 = 0,
                  std::uint64_t a2 = 0, std::uint64_t a3 = 0);
int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n, std::uint64_t a0, std::uint64_t a1);
int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g,
                       std::uint64_t b, std::uint64_t xh, std::uint64_t s0_mask);

inline std::uint32_t mir_groups_1d(std::uint32_t n_elems, int lanes, std::uint32_t block_x = 256) {
    const std::uint64_t nv = static_cast<std::uint64_t>(n_elems) / static_cast<std::uint64_t>(lanes);
    const std::uint64_t g = (nv + block_x - 1) / block_x;
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(g, 65535)));
}

void mir_dispatch_5(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                    std::uint64_t p0, std::uint64_t p1, std::uint32_t p2, float p3, float p4);
void mir_dispatch_act_bwd(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                          std::uint64_t p, std::uint64_t dy, std::uint64_t dx, std::uint32_t n, float a);
void mir_dispatch_bias(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t y, std::uint64_t bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l);
void mir_dispatch_cast(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t src, std::uint64_t dst, std::uint32_t n);
void mir_dispatch_glu(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                      std::uint64_t x, std::uint64_t dy, std::uint64_t y, std::uint32_t n, std::uint32_t dim);
void mir_dispatch_rowvec(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                         std::uint64_t x, std::uint64_t v, std::uint64_t w, std::uint64_t y,
                         std::uint32_t n, std::uint32_t dim);
void mir_dispatch_reduce(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                         std::uint64_t x, std::uint64_t y, std::uint32_t m, std::uint32_t n);
void mir_dispatch_norm(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g, std::uint64_t b,
                       std::uint64_t xh, std::uint64_t s0, std::uint64_t s1,
                       std::uint32_t rows, std::uint32_t dim, float eps, float rstd);
void mir_dispatch_rope(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t x, std::uint64_t y, std::uint64_t ct, std::uint64_t st,
                       std::uint64_t ch, std::uint64_t sh, std::uint64_t cw, std::uint64_t sw,
                       std::uint64_t pt, std::uint64_t ph, std::uint64_t pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base);

}  // namespace brotensor::detail::vulkan
