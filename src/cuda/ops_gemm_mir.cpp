#include "ops_gemm_mir.h"

#include "cu_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace brotensor::detail::cuda {

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::size_t elem_sz(Dtype dt) {
    switch (dt) {
        case Dtype::FP32: return 4u;
        case Dtype::FP16:
        case Dtype::BF16: return 2u;
        default:
            throw std::runtime_error("ops_gemm_mir: unsupported floating dtype");
    }
}

std::uint64_t cdiv64(std::uint64_t a, std::uint64_t b) {
    return (a + b - 1u) / b;
}

const std::uint8_t* ptr_add_elems(const void* p, std::int64_t elems, Dtype dt) {
    return static_cast<const std::uint8_t*>(p) + static_cast<std::size_t>(elems) * elem_sz(dt);
}

std::uint8_t* ptr_add_elems_mut(void* p, std::int64_t elems, Dtype dt) {
    return static_cast<std::uint8_t*>(p) + static_cast<std::size_t>(elems) * elem_sz(dt);
}

}  // namespace

void dispatch_mir_gemm_simt(const void* a, Dtype dt_a,
                            const void* b, Dtype dt_b,
                            void* c, Dtype dt_c,
                            const void* bias, Dtype dt_bias,
                            int batch, int m, int n, int k,
                            int lda, int ldb, int ldc,
                            std::int64_t sa, std::int64_t sb, std::int64_t sc,
                            bool ta, bool nb,
                            int epi, int act,
                            const Device& dev) {
    if (batch <= 0 || m <= 0 || n <= 0) return;
    const bool geglu = (epi == EPI_GEGLU);
    const bool swiglu = (epi == EPI_SWIGLU);
    if ((geglu || swiglu) && batch != 1) {
        throw std::runtime_error("dispatch_mir_gemm_simt: batched GLU epilogue is unsupported");
    }

    Dtype dc = dt_c;
    void* c_out = c;
    int ldc_run = ldc;
    Tensor scratch;
    if (geglu) {
        scratch = Tensor::empty_on(dev, m, n, Dtype::FP32);
        dc = Dtype::FP32;
        c_out = scratch.data;
        ldc_run = n;
    }

    const std::uint64_t tiles_big = cdiv64(m, 128u) * cdiv64(n, 128u) * static_cast<std::uint64_t>(batch);
    const bool big128 = (!swiglu && tiles_big >= 160u);
    const std::uint64_t tile = big128 ? 128u : 64u;
    const std::uint32_t epi_run = static_cast<std::uint32_t>(geglu ? EPI_STORE : epi);
    const std::uint32_t op = epi_run | (static_cast<std::uint32_t>(act) << 4) |
                             (static_cast<std::uint32_t>(dt_bias) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (big128 ? 1u : 0u) | (ta ? 2u : 0u) | (nb ? 4u : 0u) | (bias ? 8u : 0u));

    CUfunction fn = mir_function({MirKind::GemmSimt, op, dt_a, dt_b, 1u, dc, flags});
    const std::uint64_t n_grid = swiglu ? static_cast<std::uint64_t>(n / 2) : static_cast<std::uint64_t>(n);
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv64(n_grid, tile));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv64(m, tile));
    if (gy > 65535u) {
        throw std::runtime_error("dispatch_mir_gemm_simt: matrix M exceeds CUDA grid.y limit");
    }

    const std::int64_t i32_max = std::numeric_limits<std::int32_t>::max();
    const bool large_stride = (batch > 1) && (sa > i32_max || sb > i32_max || sc > i32_max);
    const int step_z = large_stride ? 1 : 65535;
    for (int z0 = 0; z0 < batch; z0 += step_z) {
        const int nz = std::min(step_z, batch - z0);
        const std::uintptr_t ptrs[] = {
            addr(ptr_add_elems(a, static_cast<std::int64_t>(z0) * sa, dt_a)),
            addr(ptr_add_elems(b, static_cast<std::int64_t>(z0) * sb, dt_b)),
            addr(ptr_add_elems_mut(c_out, static_cast<std::int64_t>(z0) * sc, dc)),
            bias ? addr(bias) : 0u,
        };
        const std::int32_t ints[] = {
            m, n, k, lda, ldb, ldc_run,
            large_stride ? 0 : static_cast<std::int32_t>(sa),
            large_stride ? 0 : static_cast<std::int32_t>(sb),
            large_stride ? 0 : static_cast<std::int32_t>(sc),
        };
        mir_dispatch_args_3d(fn, gx, gy, static_cast<std::uint32_t>(nz), 256u, ptrs, ints, {});
    }

    if (geglu) {
        const std::uint32_t half = static_cast<std::uint32_t>(n / 2);
        if (static_cast<std::uint32_t>(ldc) != half) {
            throw std::runtime_error("dispatch_mir_gemm_simt: GLU output must be contiguous");
        }
        const std::uint32_t count = static_cast<std::uint32_t>(m) * half;
        CUfunction glu_fn = mir_function({MirKind::Glu, GLU_GEGLU_PAIRS, Dtype::FP32, dt_c, 1u});
        mir_dispatch_glu(glu_fn, mir_groups_1d(count, 1), scratch.data, nullptr, c, count, half);
    }
}

void dispatch_mir_gemv(const void* a, Dtype dt_a,
                       const void* b, Dtype dt_b,
                       void* c, Dtype dt_c,
                       const void* bias,
                       int m, int n, int k,
                       int lda, int ldb, int ldc,
                       int epi, int act) {
    if (m <= 0 || n <= 0) return;
    const bool glu = (epi == EPI_GEGLU || epi == EPI_SWIGLU);
    const int half_n = n / 2;
    const int nout = glu ? half_n : n;
    const std::uint64_t groups = cdiv64(static_cast<std::uint64_t>(nout), 4u);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535u));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv64(groups, gx));
    const int max_mb = 8;

    for (int r0 = 0; r0 < m; r0 += max_mb) {
        const int mb = std::min(max_mb, m - r0);
        const void* a_sub = ptr_add_elems(a, static_cast<std::int64_t>(r0) * lda, dt_a);
        void* c_sub = ptr_add_elems_mut(c, static_cast<std::int64_t>(r0) * ldc, dt_c);
        const bool vec = k > 0 && (k % 8 == 0) && (lda % 8 == 0) && (ldb % 8 == 0) &&
                         (addr(a_sub) % 16u == 0u) && (addr(b) % 16u == 0u);
        const std::uint32_t op = static_cast<std::uint32_t>(mb) |
                                 (static_cast<std::uint32_t>(epi) << 4) |
                                 (static_cast<std::uint32_t>(act) << 8);
        const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (bias ? 2u : 0u));
        CUfunction fn = mir_function({MirKind::Gemv, op, dt_a, dt_b, 1u, dt_c, flags});
        const std::uintptr_t ptrs[] = {addr(a_sub), addr(b), addr(c_sub), bias ? addr(bias) : 0u};
        const std::int32_t ints[] = {mb, n, k, lda, ldb, ldc, half_n};
        mir_dispatch_args_3d(fn, gx, gy, 1u, 256u, ptrs, ints, {});
    }
}

void dispatch_mir_swiglu(const Tensor& r, Tensor& y, int m, int n) {
    const std::uint32_t half = static_cast<std::uint32_t>(n / 2);
    const std::uint32_t count = static_cast<std::uint32_t>(m) * half;
    if (count == 0u) return;
    int lanes = 1;
    if ((half % 4u) == 0u) {
        const int ml = mir_vec_lanes(r.dtype, half, addr(r.data), addr(y.data));
        if ((half % static_cast<std::uint32_t>(ml)) == 0u) lanes = ml;
    }
    CUfunction fn = mir_function(
        {MirKind::Glu, GLU_SWIGLU, r.dtype, r.dtype, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_glu(fn, mir_groups_1d(count, lanes), r.data, nullptr, y.data, count, half);
}

}  // namespace brotensor::detail::cuda
