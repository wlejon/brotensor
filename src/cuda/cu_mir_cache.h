#pragma once

// CUDA runtime JIT cache for backend-neutral brass MIR kernels (Waves 1 & 2).

#include <brotensor/tensor.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>

namespace brotensor::detail::cuda {

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
    FillBytes = 12,
    Copy2d = 13,
    Transpose = 14,
    ScatterAdd = 15,
    Gnorm = 16,
    Bnorm = 17,
    Sampler = 18,
    Philox = 19,
    Remap = 20,
    Resample = 21,
    ResampleBwd = 22,
    Lstm = 23,
    DeltaRule = 24,
    Xent = 25,
    Topk = 26,
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

inline std::uint8_t pack_norm_flags(bool has_g, bool has_b, bool has_xh,
                                    bool has_s0, bool has_s1) noexcept {
    return static_cast<std::uint8_t>(
        (has_g ? 1 : 0) | (has_b ? 2 : 0) | (has_xh ? 4 : 0) |
        (has_s0 ? 8 : 0) | (has_s1 ? 16 : 0));
}

// Lowers, compiles (on first use per key), and caches CUfunction via CudaJitEngine.
CUfunction mir_function(const MirKey& key);

// Chooses 128-bit (lanes=4 f32 / lanes=8 f16,bf16), 64-bit (lanes=4 f16,bf16),
// or scalar (lanes=1) based on element count and pointer alignment.
int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uintptr_t a0,
                  std::uintptr_t a1 = 0, std::uintptr_t a2 = 0, std::uintptr_t a3 = 0);
int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n,
                        std::uintptr_t a0, std::uintptr_t a1);
int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uintptr_t x, std::uintptr_t y, std::uintptr_t g,
                       std::uintptr_t b, std::uintptr_t xh, std::uintptr_t s0_mask);

inline std::uint32_t mir_groups_1d(std::uint32_t n_elems, int lanes, std::uint32_t block_x = 256) {
    const std::uint64_t nv = static_cast<std::uint64_t>(n_elems) / static_cast<std::uint64_t>(lanes);
    const std::uint64_t g = (nv + block_x - 1) / block_x;
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(g, 65535)));
}

void mir_dispatch_5(CUfunction fn, std::uint32_t gx,
                    const void* p0, const void* p1, std::uint32_t p2, float p3, float p4);
void mir_dispatch_act_bwd(CUfunction fn, std::uint32_t gx,
                          const void* p, const void* dy, const void* dx, std::uint32_t n, float a);
void mir_dispatch_bias(CUfunction fn, std::uint32_t gx,
                       const void* y, const void* bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l);
void mir_dispatch_cast(CUfunction fn, std::uint32_t gx,
                       const void* src, const void* dst, std::uint32_t n);
void mir_dispatch_glu(CUfunction fn, std::uint32_t gx,
                      const void* x, const void* dy, const void* y, std::uint32_t n, std::uint32_t dim);
void mir_dispatch_rowvec(CUfunction fn, std::uint32_t gx,
                         const void* x, const void* v, const void* w, const void* y,
                         std::uint32_t n, std::uint32_t dim);
void mir_dispatch_reduce(CUfunction fn, std::uint32_t gx,
                         const void* x, const void* y, std::uint32_t m, std::uint32_t n);
void mir_dispatch_norm(CUfunction fn, std::uint32_t gx,
                       const void* x, const void* y, const void* g, const void* b,
                       const void* xh, const void* s0, const void* s1,
                       std::uint32_t rows, std::uint32_t dim, float eps, float rstd);
void mir_dispatch_rope(CUfunction fn, std::uint32_t gx,
                       const void* x, const void* y, const void* ct, const void* st,
                       const void* ch, const void* sh, const void* cw, const void* sw,
                       const void* pt, const void* ph, const void* pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base);

// ── Wave 3: copy / transpose / fill / scatter / gnorm / bnorm ──
int mir_copy2d_lanes(std::uint32_t elem_bytes, std::uint32_t src_pitch, std::uint32_t dst_pitch,
                     std::uint32_t width, std::uintptr_t src_addr, std::uintptr_t dst_addr);
void mir_dispatch_fill(CUfunction fn, std::uint32_t gx, void* dst, std::uint32_t n, std::uint32_t val);
void mir_dispatch_copy2d(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                         const void* src, void* dst, std::uint32_t src_pitch, std::uint32_t dst_pitch,
                         std::uint32_t width, std::uint32_t height);
void mir_dispatch_transpose(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                            const void* src, void* dst,
                            std::uint32_t batch, std::uint32_t rows, std::uint32_t cols);
void mir_dispatch_scatter(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                          const void* idx, const void* keys, const void* keys_out,
                          const void* src, void* dst,
                          std::uint32_t m, std::uint32_t r, std::uint32_t c,
                          std::uint32_t p, std::uint32_t k, std::uint32_t j);
void mir_dispatch_gnorm(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                        const void* x, void* y, const void* g, const void* b,
                        const void* part, const void* rm, const void* rv,
                        std::uint32_t tile, std::uint32_t hw, std::uint32_t cpg,
                        std::uint32_t groups, std::uint32_t splits, std::uint32_t chunk,
                        std::uint32_t total, std::uint32_t c, float eps);
void mir_dispatch_bnorm(CUfunction fn, std::uint32_t gx,
                        const void* x, void* y, const void* g, const void* b,
                        void* rm, void* rv, void* sm, void* sr,
                        const void* dy, void* dg, void* db,
                        std::uint32_t n, std::uint32_t c, std::uint32_t hw,
                        float eps, float momentum);

// ── Wave 4: diffusion samplers / philox / remap / resample / resample_bwd ──
void mir_dispatch_sampler(CUfunction fn, std::uint32_t gx,
                          const void* x, const void* e, const void* p, void* y, void* z,
                          std::uint32_t n, std::uint32_t dim, std::uint32_t half,
                          std::uint32_t hw, std::uint32_t c,
                          float k0, float k1, float k2, float k3);
void mir_dispatch_philox(CUfunction fn, std::uint32_t gx,
                         void* y, std::uint64_t key, std::uint64_t counter, std::uint64_t n,
                         float p, float lo, float hi);
void mir_dispatch_remap(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                        const void* x, void* y, const void* idx,
                        std::uint32_t c, std::uint32_t cin, std::uint32_t hi, std::uint32_t wi,
                        std::uint32_t ho, std::uint32_t wo,
                        std::int32_t a0, std::int32_t a1, std::int32_t a2, std::int32_t a3,
                        std::int32_t a4, std::int32_t a5, std::int32_t a6);
void mir_dispatch_resample(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                           const void* x, void* y, const void* aux,
                           std::uint32_t c, std::uint32_t hi, std::uint32_t wi,
                           std::uint32_t ho, std::uint32_t wo,
                           std::uint32_t kh, std::uint32_t kw,
                           std::int32_t sh, std::int32_t sw, std::int32_t ph, std::int32_t pw);
void mir_dispatch_resample_bwd(CUfunction fn, std::uint32_t gx,
                               const void* dy, void* dx, const void* aux,
                               std::uint32_t planes, std::uint32_t hi, std::uint32_t wi,
                               std::uint32_t ho, std::uint32_t wo,
                               std::int32_t pt, std::int32_t pl);

// ── Wave 5: lstm / delta_rule / xent / topk ──
void mir_dispatch_lstm(CUfunction fn, std::uint32_t gx,
                       const void* gates, const void* c, const void* y,
                       const void* bhh, const void* c0,
                       const void* dy, void* dz, void* dh, void* dc,
                       std::uint32_t b, std::uint32_t h, std::uint32_t row0);
void mir_dispatch_delta(CUfunction fn, std::uint32_t gx,
                        const void* q, const void* k, const void* v,
                        const void* a, const void* b, const void* loga,
                        void* s, void* o,
                        std::uint32_t L, std::uint32_t H, std::uint32_t dk, std::uint32_t dv,
                        std::uint32_t ldq, std::uint32_t ldv);
void mir_dispatch_xent(CUfunction fn, std::uint32_t gx,
                       const void* logits, const void* target, const void* mask, const void* off,
                       void* probs, void* dlog, void* loss,
                       std::uint32_t rows, std::uint32_t n, std::uint32_t heads, float pos_weight);
void mir_dispatch_topk(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                       const void* x, void* vals, void* idx, const void* off,
                       std::uint32_t rows, std::uint32_t cols, std::uint32_t k, std::uint32_t nseg);

}  // namespace brotensor::detail::cuda
