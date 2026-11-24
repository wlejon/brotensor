#pragma once

// Metal runtime pipeline cache for backend-neutral brass MIR kernels.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>

namespace brotensor::detail::metal {

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

struct MirEntry;

struct MirKernel {
    const MirEntry* entry = nullptr;
};

inline std::uint8_t pack_norm_flags(bool has_g, bool has_b, bool has_xh,
                                    bool has_s0, bool has_s1) noexcept {
    return static_cast<std::uint8_t>(
        (has_g ? 1 : 0) | (has_b ? 2 : 0) | (has_xh ? 4 : 0) |
        (has_s0 ? 8 : 0) | (has_s1 ? 16 : 0));
}

// Compiles on first use (per key) via brass::target::MslTarget and caches the
// compiled MTLComputePipelineState.
MirKernel mir_kernel(const MirKey& key);

// Chooses 128-bit (lanes=4 f32 / lanes=8 f16,bf16), 64-bit (lanes=4 f16,bf16),
// or scalar (lanes=1) based on element count and device pointer alignment.
int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uint64_t a0,
                  std::uint64_t a1 = 0, std::uint64_t a2 = 0,
                  std::uint64_t a3 = 0);
int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n,
                        std::uint64_t a0, std::uint64_t a1);
int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g,
                       std::uint64_t b, std::uint64_t xh,
                       std::uint64_t s0_mask);

inline std::uint32_t mir_groups_1d(std::uint32_t n_elems, int lanes,
                                   std::uint32_t block_x = 256) {
    const std::uint64_t nv =
        static_cast<std::uint64_t>(n_elems) / static_cast<std::uint64_t>(lanes);
    const std::uint64_t g = (nv + block_x - 1) / block_x;
    return static_cast<std::uint32_t>(
        std::max<std::uint64_t>(1, std::min<std::uint64_t>(g, 65535)));
}

void mir_dispatch_5(const MirKernel& mk, std::uint32_t gx,
                    const Tensor& b0, const Tensor& b1,
                    std::uint32_t p2, float p3, float p4);
void mir_dispatch_act_bwd(const MirKernel& mk, std::uint32_t gx,
                          const Tensor* p, const Tensor& dy, const Tensor& dx,
                          std::uint32_t n, float a);
void mir_dispatch_bias(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& y, const Tensor& bias,
                       std::uint32_t n, std::uint32_t dim, std::uint32_t l);
void mir_dispatch_cast(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& src, const Tensor& dst, std::uint32_t n);
void mir_dispatch_glu(const MirKernel& mk, std::uint32_t gx,
                      const Tensor& x, const Tensor* dy, const Tensor& y,
                      std::uint32_t n, std::uint32_t dim);
void mir_dispatch_rowvec(const MirKernel& mk, std::uint32_t gx,
                         const Tensor& x, const Tensor& v, const Tensor* w,
                         const Tensor& y, std::uint32_t n, std::uint32_t dim);
void mir_dispatch_reduce(const MirKernel& mk, std::uint32_t gx,
                         const Tensor& x, const Tensor& y,
                         std::uint32_t m, std::uint32_t n);
void mir_dispatch_norm(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* x, const void* y, const void* g,
                       const void* b, const void* xh, const void* s0,
                       const void* s1, std::uint32_t rows, std::uint32_t dim,
                       float eps, float rstd);
void mir_dispatch_rope(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* x, const void* y, const void* ct,
                       const void* st, const void* ch, const void* sh,
                       const void* cw, const void* sw, const void* pt,
                       const void* ph, const void* pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base);

}  // namespace brotensor::detail::metal
