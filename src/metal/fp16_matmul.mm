// The A @ B^T launchers behind matmul_abt, batched linears, and attention GEMMs
// (see fp16_matmul.h).
//   - Decode GEMV (M in 1..8, NT, batch=1, alpha=1, matching A/C dtype):
//     MirKind::Gemv (src/kernels/mir_gemv.h) via launch_gemm_fp32
//   - Prefill / large GEMM (K > 0, M*N >= 1024 or alpha != 1):
//     Metal simdgroup_matrix 64x64 tiled GEMM (src/metal/gemm_fp32.mm)
//   - Small / K == 0 fallback:
//     MirKind::GemmSimt (src/kernels/mir_gemm_simt.h), with GLU epilogues via
//     FP32 scratch + MirKind::Glu preserving original dt_bias.

#include "fp16_matmul.h"
#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace brotensor::metal_impl {

namespace {

constexpr size_t kTiledMin = 1024;

::brotensor::Dtype to_dtype(AbtType t) {
    switch (t) {
        case kAbtF16:  return ::brotensor::Dtype::FP16;
        case kAbtBF16: return ::brotensor::Dtype::BF16;
        default:       return ::brotensor::Dtype::FP32;
    }
}

size_t elem_size(::brotensor::Dtype dt) { return dt == ::brotensor::Dtype::FP32 ? 4u : 2u; }

AbtMixed abt_plain(AbtType t, id<MTLBuffer> A, NSUInteger ofs_A, id<MTLBuffer> B, NSUInteger ofs_B,
                   id<MTLBuffer> C, NSUInteger ofs_C, int batch, int M, int N, int K, uint64_t strideA,
                   uint64_t strideB, uint64_t strideC, id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                   int act) {
    AbtMixed g;
    g.A = A; g.ofs_A = ofs_A; g.lda = static_cast<uint64_t>(K);
    g.B = B; g.ofs_B = ofs_B; g.ldb = static_cast<uint64_t>(K);
    g.C = C; g.ofs_C = ofs_C; g.ldc = static_cast<uint64_t>(N);
    if (has_bias) {
        g.bias = bias;
        g.ofs_bias = ofs_bias;
    }
    g.M = M; g.N = N; g.K = K < 0 ? 0 : K;
    g.in = g.out = t;
    g.act = act;
    g.batch = batch;
    g.strideA = strideA; g.strideB = strideB; g.strideC = strideC;
    return g;
}

void run_glu_mtl(std::uint32_t glu_op, ::brotensor::Dtype dt_out,
                 id<MTLBuffer> bX, NSUInteger oX, id<MTLBuffer> bY, NSUInteger oY,
                 int rows, int half_n) {
    using namespace ::brotensor::detail::metal;
    const std::uint32_t n = static_cast<std::uint32_t>(rows) * static_cast<std::uint32_t>(half_n);
    const int lanes = (glu_op != GLU_GEGLU_PAIRS)
                          ? mir_vec_lanes_mixed(::brotensor::Dtype::FP32, dt_out,
                                                static_cast<std::uint32_t>(half_n), oX, oY)
                          : 1;
    const MirKernel mk = mir_kernel(
        {MirKind::Glu, glu_op, ::brotensor::Dtype::FP32, dt_out, static_cast<std::uint8_t>(lanes)});
    const void* bufs[3] = {(__bridge const void*)bX, (__bridge const void*)bX,
                           (__bridge const void*)bY};
    const std::size_t offs[3] = {oX, oX, oY};
    const std::int32_t ints[2] = {static_cast<std::int32_t>(n), half_n};
    mir_dispatch_mtl_3d(mk, mir_groups_1d(n, lanes), 1u, 1u, bufs, offs, 3, ints, {});
}

void run_gemm_simt(const AbtMixed& g) {
    using namespace ::brotensor::detail::metal;
    const ::brotensor::Dtype da = to_dtype(g.in);
    const ::brotensor::Dtype db = to_dtype(g.in_B < 0 ? g.in : static_cast<AbtType>(g.in_B));
    const ::brotensor::Dtype dc_orig = to_dtype(g.out);
    const ::brotensor::Dtype dt_bias = da;  // AbtMixed contract: bias has A's dtype
    const bool geglu = (g.epilogue == kAbtGeglu);
    const bool swiglu = (g.epilogue == kAbtSwiglu);

    ::brotensor::Dtype dc = dc_orig;
    id<MTLBuffer> bC = g.C;
    NSUInteger oC = g.ofs_C;
    uint64_t ldc = g.ldc;
    uint64_t sc = g.strideC;
    ::brotensor::Tensor scratch;
    if (geglu) {
        scratch = ::brotensor::Tensor::empty_on(
            ::brotensor::Device::Metal, g.batch * g.M, g.N, ::brotensor::Dtype::FP32);
        dc = ::brotensor::Dtype::FP32;
        bC = buffer_for(scratch);
        oC = buffer_offset_for(scratch);
        ldc = static_cast<uint64_t>(g.N);
        sc = static_cast<uint64_t>(g.M) * static_cast<uint64_t>(g.N);
    }

    const std::uint32_t epi = static_cast<std::uint32_t>(geglu ? EPI_STORE : g.epilogue);
    const std::uint32_t op = epi | (static_cast<std::uint32_t>(g.act) << 4) |
                             (static_cast<std::uint32_t>(dt_bias) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (g.transA ? 2u : 0u) | (g.transB ? 4u : 0u) | (g.bias ? 8u : 0u));
    const MirKernel mk = mir_kernel({MirKind::GemmSimt, op, da, db, 1, dc, flags});
    const int n_out = swiglu ? (g.N / 2) : g.N;
    const std::uint32_t gx = static_cast<std::uint32_t>((n_out + 63) / 64);
    const std::uint32_t gy = static_cast<std::uint32_t>((g.M + 63) / 64);
    const size_t esa = elem_size(da), esb = elem_size(db), esc = elem_size(dc);
    const int k_clamped = g.K < 0 ? 0 : g.K;
    for (int z0 = 0; z0 < g.batch; z0 += 65535) {
        const int nz = std::min(65535, g.batch - z0);
        const void* bufs[4] = {
            (__bridge const void*)g.A, (__bridge const void*)g.B,
            (__bridge const void*)bC, (__bridge const void*)g.bias,
        };
        const std::size_t offs[4] = {
            g.ofs_A + static_cast<std::size_t>(z0) * g.strideA * esa,
            g.ofs_B + static_cast<std::size_t>(z0) * g.strideB * esb,
            oC + static_cast<std::size_t>(z0) * sc * esc,
            g.ofs_bias,
        };
        const std::int32_t ints[9] = {
            g.M, g.N, k_clamped,
            static_cast<std::int32_t>(g.lda), static_cast<std::int32_t>(g.ldb),
            static_cast<std::int32_t>(ldc), static_cast<std::int32_t>(g.strideA),
            static_cast<std::int32_t>(g.strideB), static_cast<std::int32_t>(sc),
        };
        mir_dispatch_mtl_3d(mk, gx, gy, static_cast<std::uint32_t>(nz), bufs, offs, 4, ints, {});
    }

    if (geglu) {
        const int half_n = g.N / 2;
        const size_t es_out = elem_size(dc_orig);
        for (int b = 0; b < g.batch; ++b) {
            run_glu_mtl(GLU_GEGLU_PAIRS, dc_orig,
                        bC, oC + static_cast<NSUInteger>(b) * sc * sizeof(float),
                        g.C, g.ofs_C + static_cast<NSUInteger>(b) * g.strideC * es_out,
                        g.M, half_n);
        }
    }
}

bool gemv_mir_eligible(const AbtMixed& g) {
    if (g.transA || g.transB || g.batch != 1 || g.M < 1 || g.M > 8 || g.K <= 0 || g.alpha != 1.0f)
        return false;
    if (g.epilogue < kAbtStore || g.epilogue > kAbtSwiglu) return false;
    const ::brotensor::Dtype da = to_dtype(g.in);
    const ::brotensor::Dtype db = to_dtype(g.in_B < 0 ? g.in : static_cast<AbtType>(g.in_B));
    const ::brotensor::Dtype dc = to_dtype(g.out);
    if (da != dc) return false;
    return db == da || (da == ::brotensor::Dtype::FP32 && db != ::brotensor::Dtype::FP32);
}

}  // namespace

void launch_matmul_abt_fp16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    launch_matmul_abt_mixed(abt_plain(kAbtF16, A, ofs_A, B, ofs_B, C, ofs_C, batch, M, N, K, strideA, strideB,
                                      strideC, bias, ofs_bias, has_bias, act));
}

void launch_matmul_abt_bf16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    launch_matmul_abt_mixed(abt_plain(kAbtBF16, A, ofs_A, B, ofs_B, C, ofs_C, batch, M, N, K, strideA, strideB,
                                      strideC, bias, ofs_bias, has_bias, act));
}

AbtType abt_type(::brotensor::Dtype dt) {
    switch (dt) {
        case ::brotensor::Dtype::FP16: return kAbtF16;
        case ::brotensor::Dtype::BF16: return kAbtBF16;
        case ::brotensor::Dtype::FP32: return kAbtF32;
        default: throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: dtype must be FP16, BF16 or FP32");
    }
}

void launch_matmul_abt_mixed(const AbtMixed& g) {
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0) return;
    if ((g.epilogue == kAbtGeglu || g.epilogue == kAbtSwiglu) && (g.N % 2) != 0)
        throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: GLU epilogue needs an even N");

    // SwiGLU epilogue: decode M <= 8 fuses into MirKind::Gemv; larger shapes
    // compute into FP32 scratch (preserving original bias dtype) and gate once.
    if (g.epilogue == kAbtSwiglu) {
        if (gemv_mir_eligible(g)) {
            launch_gemm_fp32(g);
            return;
        }
        ::brotensor::Tensor scratch = ::brotensor::Tensor::empty_on(
            ::brotensor::Device::Metal, g.batch * g.M, g.N, ::brotensor::Dtype::FP32);
        AbtMixed s = g;
        s.C = buffer_for(scratch);
        s.ofs_C = buffer_offset_for(scratch);
        s.ldc = static_cast<uint64_t>(g.N);
        s.strideC = static_cast<uint64_t>(g.M) * static_cast<uint64_t>(g.N);
        s.out = kAbtF32;
        s.epilogue = kAbtStore;
        launch_matmul_abt_mixed(s);
        const ::brotensor::Dtype dc_orig = to_dtype(g.out);
        const size_t es_out = elem_size(dc_orig);
        for (int b = 0; b < g.batch; ++b) {
            run_glu_mtl(GLU_SWIGLU, dc_orig,
                        s.C, s.ofs_C + static_cast<NSUInteger>(b) * s.strideC * sizeof(float),
                        g.C, g.ofs_C + static_cast<NSUInteger>(b) * g.strideC * es_out,
                        g.M, g.N / 2);
        }
        return;
    }

    const bool mixed_b = g.in_B >= 0 && g.in_B != g.in;
    if (g.K > 0 &&
        (g.alpha != 1.0f || mixed_b || g.M <= 8 ||
         static_cast<size_t>(g.M) * static_cast<size_t>(g.N) >= kTiledMin)) {
        launch_gemm_fp32(g);
        return;
    }
    run_gemm_simt(g);
}

}  // namespace brotensor::metal_impl
