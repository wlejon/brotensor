// Metal GEMM dispatcher backed 100% by single-source Brass MIR kernels
// (zero raw MSL shaders):
//   - Decode GEMV (M in 1..8, NT, batch=1, matching A/C dtype): MirKind::Gemv
//   - FP16 x FP16 tiled simdgroup GEMM: MirKind::GemmCm (64x64 / 128x64 tiles,
//     32-wide simdgroup_matrix<half,8,8> -> simdgroup_float8x8)
//   - FP32 / BF16 / mixed / K == 0 / fallback: MirKind::GemmSimt (64x64 or 128x128)

#include "fp16_matmul.h"
#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace brotensor::metal_impl {

namespace {

std::uint64_t cdiv_u64(std::uint64_t a, std::uint64_t b) { return (a + b - 1u) / b; }

::brotensor::Dtype abt_to_dtype(AbtType t) {
    return t == kAbtF16 ? ::brotensor::Dtype::FP16
         : t == kAbtBF16 ? ::brotensor::Dtype::BF16
         : ::brotensor::Dtype::FP32;
}

std::size_t abt_elem_size(::brotensor::Dtype dt) {
    return dt == ::brotensor::Dtype::FP32 ? 4u : 2u;
}

bool try_mir_gemv(const AbtMixed& g, AbtType in_b) {
    if (g.transA || g.transB || g.batch != 1 || g.M < 1 || g.M > 8 || g.K <= 0 || g.alpha != 1.0f)
        return false;
    if (g.epilogue < kAbtStore || g.epilogue > kAbtSwiglu) return false;
    const ::brotensor::Dtype da = abt_to_dtype(g.in);
    const ::brotensor::Dtype db = abt_to_dtype(in_b);
    const ::brotensor::Dtype dc = abt_to_dtype(g.out);
    if (da != dc) return false;
    if (db != da && !(da == ::brotensor::Dtype::FP32 && db != ::brotensor::Dtype::FP32)) return false;
    if (g.lda > 0x7fffffffULL || g.ldb > 0x7fffffffULL || g.ldc > 0x7fffffffULL) return false;

    using namespace ::brotensor::detail::metal;
    const std::uint32_t nout =
        static_cast<std::uint32_t>((g.epilogue == kAbtGeglu || g.epilogue == kAbtSwiglu) ? g.N / 2 : g.N);
    if (nout == 0) return true;
    const bool vec = (g.ofs_A % 16u == 0u) && (g.ofs_B % 16u == 0u) &&
                     (g.lda % 8u == 0u) && (g.ldb % 8u == 0u) && (g.K % 8 == 0);
    const std::uint32_t op = static_cast<std::uint32_t>(g.M) |
                             (static_cast<std::uint32_t>(g.epilogue) << 4) |
                             (static_cast<std::uint32_t>(g.act) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (g.bias ? 2u : 0u));
    const MirKernel mk = mir_kernel({MirKind::Gemv, op, da, db, 1, dc, flags});
    const std::uint64_t groups = (static_cast<std::uint64_t>(nout) + 3u) / 4u;
    const std::uint32_t gx = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(1u, std::min<std::uint64_t>(groups, 65535u)));
    const std::uint32_t gy = static_cast<std::uint32_t>((groups + gx - 1u) / gx);
    const void* bufs[4] = {
        (__bridge const void*)g.A, (__bridge const void*)g.B,
        (__bridge const void*)g.C, (__bridge const void*)g.bias,
    };
    const std::size_t offs[4] = {g.ofs_A, g.ofs_B, g.ofs_C, g.ofs_bias};
    const std::int32_t ints[7] = {
        g.M, g.N, g.K,
        static_cast<std::int32_t>(g.lda), static_cast<std::int32_t>(g.ldb),
        static_cast<std::int32_t>(g.ldc), g.N / 2,
    };
    mir_dispatch_mtl_3d(mk, gx, gy, 1u, bufs, offs, 4, ints, {});
    return true;
}

std::uint32_t pick_cm_cfg(int M, std::uint32_t nout, int epi, int batch) {
    // Select 128x64 (cfg_idx=2, ~30 KiB shared) or 64x64 (cfg_idx=3, ~20 KiB shared)
    // so static threadgroup allocations stay strictly within Metal's 32 KiB cap.
    const std::uint64_t tiles2 =
        cdiv_u64(M, 128u) * cdiv_u64(nout, (epi == EPI_SWIGLU) ? 32u : 64u) *
        static_cast<std::uint64_t>(batch);
    return (M > 64 && tiles2 >= 24u) ? 2u : 3u;
}

template <typename EmitFn>
void dispatch_cm_interior_and_edges(std::uint32_t M, std::uint32_t N, std::uint32_t nout,
                                    int epi, const ::brotensor::detail::metal::MetalCmTile& cfg,
                                    std::uint32_t gx, std::uint32_t gy, bool direct_ok,
                                    EmitFn&& emit) {
    const std::uint32_t wout = (epi == EPI_SWIGLU) ? cfg.bn / 2u : cfg.bn;
    const std::uint32_t fx = direct_ok ? ((epi == EPI_SWIGLU ? nout : N) / wout) : 0u;
    const std::uint32_t fy = direct_ok ? (M / cfg.bm) : 0u;
    if (fx > 0u && fy > 0u) {
        emit(true, 0u, 0u, fx, fy);
        if (fx < gx) emit(false, 0u, fx, gx - fx, gy);
        if (fy < gy) emit(false, fy, 0u, fx, gy - fy);
    } else {
        emit(false, 0u, 0u, gx, gy);
    }
}

bool try_mir_gemm_cm(const AbtMixed& g, AbtType in_b) {
    if (g.K <= 0 || g.alpha != 1.0f) return false;
    if (g.in != kAbtF16 || in_b != kAbtF16) return false;
    if (g.out != kAbtF16 && g.out != kAbtF32) return false;
    if (g.epilogue < kAbtStore || g.epilogue > kAbtSwiglu) return false;
    if (g.lda > 0x7fffffffULL || g.ldb > 0x7fffffffULL || g.ldc > 0x7fffffffULL) return false;
    if (g.strideA > 0x7fffffffULL || g.strideB > 0x7fffffffULL || g.strideC > 0x7fffffffULL) return false;

    using namespace ::brotensor::detail::metal;
    const std::uint32_t nout =
        static_cast<std::uint32_t>((g.epilogue == kAbtGeglu || g.epilogue == kAbtSwiglu) ? g.N / 2 : g.N);
    if (nout == 0) return true;

    const std::uint32_t cfg_idx = pick_cm_cfg(g.M, nout, g.epilogue, g.batch);
    const MetalCmTile cfg = metal_cm_tile(cfg_idx);
    const std::uint32_t wout = (g.epilogue == kAbtSwiglu) ? cfg.bn / 2u : cfg.bn;
    const std::uint32_t gx = static_cast<std::uint32_t>(
        (g.epilogue == kAbtSwiglu) ? cdiv_u64(nout, wout) : cdiv_u64(g.N, cfg.bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv_u64(g.M, cfg.bm));
    if (gx > 65535u || gy > 65535u) return false;

    const int ea = g.transA ? g.M : g.K;
    const int eb = g.transB ? g.N : g.K;
    const bool vec = (g.ofs_A % 16u == 0u) && (g.ofs_B % 16u == 0u) &&
                     (g.lda % 8u == 0u) && (g.ldb % 8u == 0u) &&
                     (g.strideA % 8u == 0u) && (g.strideB % 8u == 0u) &&
                     (ea % 8 == 0) && (eb % 8 == 0);
    const ::brotensor::Dtype da = ::brotensor::Dtype::FP16;
    const ::brotensor::Dtype dc = abt_to_dtype(g.out);
    const std::uint32_t op = (cfg_idx & 3u) |
                             ((static_cast<std::uint32_t>(g.epilogue) & 0x0fu) << 4) |
                             ((static_cast<std::uint32_t>(g.act) & 0x0fu) << 8);
    // mir_gemm_cm uses `nb` for B stored (K, N) — exact match for AbtMixed::transB.
    const std::uint8_t base_flags = static_cast<std::uint8_t>(
        (g.transA ? 1u : 0u) | (g.transB ? 2u : 0u) | (vec ? 4u : 0u) |
        (g.bias ? 16u : 0u));
    const bool direct_ok =
        (dc == ::brotensor::Dtype::FP16) && (g.epilogue != EPI_GEGLU) &&
        (g.ofs_C % 16u == 0u) && (g.ldc % 8u == 0u) && (g.strideC % 8u == 0u) &&
        (!g.bias || g.ofs_bias % 16u == 0u);
    const std::size_t esc = abt_elem_size(dc);
    const void* bufs[6] = {
        (__bridge const void*)g.A, (__bridge const void*)g.B,
        (__bridge const void*)g.C, (__bridge const void*)g.bias,
        nullptr, nullptr,
    };

    auto emit = [&](bool direct, std::uint32_t tm0, std::uint32_t tn0,
                    std::uint32_t dx, std::uint32_t dy) {
        const std::uint8_t flags = static_cast<std::uint8_t>(base_flags | (direct ? 8u : 0u));
        const MirKernel mk = mir_kernel({MirKind::GemmCm, op, da, da, 1, dc, flags});
        const std::int32_t ints[12] = {
            g.M, g.N, g.K,
            static_cast<std::int32_t>(g.lda), static_cast<std::int32_t>(g.ldb),
            static_cast<std::int32_t>(g.ldc),
            static_cast<std::int32_t>(g.strideA), static_cast<std::int32_t>(g.strideB),
            static_cast<std::int32_t>(g.strideC), g.N / 2,
            static_cast<std::int32_t>(tm0), static_cast<std::int32_t>(tn0),
        };
        for (int z0 = 0; z0 < g.batch; z0 += 65535) {
            const int nz = std::min(65535, g.batch - z0);
            const std::size_t offs[6] = {
                g.ofs_A + static_cast<std::size_t>(z0) * g.strideA * 2u,
                g.ofs_B + static_cast<std::size_t>(z0) * g.strideB * 2u,
                g.ofs_C + static_cast<std::size_t>(z0) * g.strideC * esc,
                g.ofs_bias, 0u, 0u,
            };
            mir_dispatch_mtl_3d(mk, dx, dy, static_cast<std::uint32_t>(nz), bufs, offs, 6, ints, {});
        }
    };
    dispatch_cm_interior_and_edges(
        static_cast<std::uint32_t>(g.M), static_cast<std::uint32_t>(g.N), nout,
        g.epilogue, cfg, gx, gy, direct_ok, emit);
    return true;
}

void dispatch_simt_gemm(const AbtMixed& g, AbtType in_b) {
    if (g.alpha != 1.0f) {
        throw std::runtime_error("brotensor (Metal): GEMM alpha != 1 is not supported");
    }
    if (g.epilogue == kAbtGeglu || g.epilogue == kAbtSwiglu) {
        throw std::runtime_error("brotensor (Metal): SIMT GEMM does not support fused GLU epilogue directly");
    }
    using namespace ::brotensor::detail::metal;
    const ::brotensor::Dtype da = abt_to_dtype(g.in);
    const ::brotensor::Dtype db = abt_to_dtype(in_b);
    const ::brotensor::Dtype dc = abt_to_dtype(g.out);
    const ::brotensor::Dtype dt_bias = da;
    const int k_clamped = g.K < 0 ? 0 : g.K;
    const std::uint64_t big_tiles = cdiv_u64(g.M, 128u) * cdiv_u64(g.N, 128u) *
                                    static_cast<std::uint64_t>(g.batch);
    const bool big128 = big_tiles >= 160u && g.M >= 128 && g.N >= 128;
    const std::uint32_t tile = big128 ? 128u : 64u;
    const std::uint32_t op = static_cast<std::uint32_t>(g.epilogue) |
                             (static_cast<std::uint32_t>(g.act) << 4) |
                             (static_cast<std::uint32_t>(dt_bias) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (big128 ? 1u : 0u) | (g.transA ? 2u : 0u) | (g.transB ? 4u : 0u) |
        (g.bias ? 8u : 0u));
    const MirKernel mk = mir_kernel({MirKind::GemmSimt, op, da, db, 1, dc, flags});
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv_u64(g.N, tile));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv_u64(g.M, tile));
    const std::size_t esa = abt_elem_size(da);
    const std::size_t esb = abt_elem_size(db);
    const std::size_t esc = abt_elem_size(dc);
    for (int z0 = 0; z0 < g.batch; z0 += 65535) {
        const int nz = std::min(65535, g.batch - z0);
        const void* bufs[4] = {
            (__bridge const void*)g.A, (__bridge const void*)g.B,
            (__bridge const void*)g.C, (__bridge const void*)g.bias,
        };
        const std::size_t offs[4] = {
            g.ofs_A + static_cast<std::size_t>(z0) * g.strideA * esa,
            g.ofs_B + static_cast<std::size_t>(z0) * g.strideB * esb,
            g.ofs_C + static_cast<std::size_t>(z0) * g.strideC * esc,
            g.ofs_bias,
        };
        const std::int32_t ints[9] = {
            g.M, g.N, k_clamped,
            static_cast<std::int32_t>(g.lda), static_cast<std::int32_t>(g.ldb),
            static_cast<std::int32_t>(g.ldc), static_cast<std::int32_t>(g.strideA),
            static_cast<std::int32_t>(g.strideB), static_cast<std::int32_t>(g.strideC),
        };
        mir_dispatch_mtl_3d(mk, gx, gy, static_cast<std::uint32_t>(nz), bufs, offs, 4, ints, {});
    }
}

} // namespace

void launch_gemm_fp32(const AbtMixed& g) {
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0) return;
    const AbtType in_b = g.in_B < 0 ? g.in : static_cast<AbtType>(g.in_B);
    if (try_mir_gemv(g, in_b)) return;
    if (try_mir_gemm_cm(g, in_b)) return;
    dispatch_simt_gemm(g, in_b);
}

} // namespace brotensor::metal_impl

namespace brotensor::detail::metal {

bool try_mir_quant_gemm_cm(std::uint32_t qb,
                             const void* a, const void* b, const void* c,
                             const void* bias, const void* scale,
                             ::brotensor::Dtype dt_out,
                             int M, int N, int K,
                             int lda, int ldb_bytes, int ldc,
                             int act, int epi, const char* op_name) {
    if (M <= 0 || N <= 0) return true;
    if (K <= 0 || (dt_out != ::brotensor::Dtype::FP16 && dt_out != ::brotensor::Dtype::FP32)) {
        return false;
    }
    if (epi < EPI_STORE || epi > EPI_SWIGLU || epi == EPI_RESIDUAL_ADD) return false;
    const std::uint32_t nout =
        static_cast<std::uint32_t>((epi == EPI_GEGLU || epi == EPI_SWIGLU) ? N / 2 : N);
    if (nout == 0) return true;

    const std::uint32_t cfg_idx = brotensor::metal_impl::pick_cm_cfg(M, nout, epi, 1);
    const MetalCmTile cfg = metal_cm_tile(cfg_idx);
    const std::uint32_t wout = (epi == EPI_SWIGLU) ? cfg.bn / 2u : cfg.bn;
    const std::uint32_t gx = static_cast<std::uint32_t>(
        (epi == EPI_SWIGLU) ? brotensor::metal_impl::cdiv_u64(nout, wout)
                            : brotensor::metal_impl::cdiv_u64(N, cfg.bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(brotensor::metal_impl::cdiv_u64(M, cfg.bm));
    if (gx > 65535u || gy > 65535u) return false;

    const auto a_addr = reinterpret_cast<std::uintptr_t>(a);
    const auto b_addr = reinterpret_cast<std::uintptr_t>(b);
    const auto c_addr = reinterpret_cast<std::uintptr_t>(c);
    const auto bi_addr = reinterpret_cast<std::uintptr_t>(bias);
    const bool byte_fmt = (qb == QF_INT8 || qb == QF_FP8);
    const bool vec = (a_addr % 16u == 0u) && (lda % 8 == 0) && (K % 8 == 0) &&
                     (!byte_fmt || ((b_addr % 8u == 0u) && (ldb_bytes % 8 == 0)));
    const std::uint32_t op = (cfg_idx & 3u) |
                             ((static_cast<std::uint32_t>(epi) & 0x0fu) << 4) |
                             ((static_cast<std::uint32_t>(act) & 0x0fu) << 8) |
                             ((qb & 0x0fu) << 12);
    const std::uint8_t base_flags = static_cast<std::uint8_t>((vec ? 4u : 0u) | (bias ? 16u : 0u));
    const bool direct_ok = (dt_out == ::brotensor::Dtype::FP16) && (epi != EPI_GEGLU) &&
                           (c_addr % 16u == 0u) && (ldc % 8 == 0) &&
                           (!bias || bi_addr % 16u == 0u);
    const void* ptrs[6] = {a, b, c, bias, scale, nullptr};

    auto emit = [&](bool direct, std::uint32_t tm0, std::uint32_t tn0,
                    std::uint32_t dx, std::uint32_t dy) {
        const std::uint8_t flags = static_cast<std::uint8_t>(base_flags | (direct ? 8u : 0u));
        const MirKernel mk = mir_kernel({MirKind::GemmCm, op, ::brotensor::Dtype::FP16,
                                         ::brotensor::Dtype::FP16, 1, dt_out, flags});
        const std::int32_t ints[12] = {
            M, N, K, lda, ldb_bytes, ldc, 0, 0, 0, N / 2,
            static_cast<std::int32_t>(tm0), static_cast<std::int32_t>(tn0),
        };
        mir_dispatch_args_3d(mk, dx, dy, 1u, op_name, ptrs, ints, {});
    };
    brotensor::metal_impl::dispatch_cm_interior_and_edges(
        static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N), nout,
        epi, cfg, gx, gy, direct_ok, emit);
    return true;
}

} // namespace brotensor::detail::metal
