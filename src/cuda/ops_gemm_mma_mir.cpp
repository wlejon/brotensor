#include "ops_gemm_mma_mir.h"

#include "cu_mir_cache.h"
#include "ops_gemm_mir.h"
#include "../kernels/mir_gemm_mma.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace brotensor::detail::cuda::mma_gemm {

namespace {

int cdiv_i(int a, int b) {
    return (a + b - 1) / b;
}

void dispatch_mma_kernel(Dtype dt, const void* a, const void* w, void* c,
                         int m, int n, int k, const void* bias,
                         int act, int epi, float* partial,
                         int batch, int sa, int sb, int sc,
                         std::uint8_t cfg, int splits, int kps, bool fast_accum) {
    const kernels::GemmMmaCfg& tcfg = kernels::kGemmMmaCfgs[cfg];
    const int bm = static_cast<int>(tcfg.bm);
    const int bn = static_cast<int>(tcfg.bn);
    const int gx = cdiv_i(n, bn);
    const int gy = cdiv_i(m, bm);
    const bool has_bias = (bias != nullptr);
    const bool has_partial = (partial != nullptr);
    const bool batched = (!has_partial && batch > 1);
    const bool aligned = (m % bm == 0) && (n % bn == 0) && (k % 32 == 0);
    const bool swizzle = (!has_partial && !batched && gy >= 8 && gx >= 4);

    const std::uint32_t op = static_cast<std::uint32_t>(cfg) |
                             (static_cast<std::uint32_t>(epi & 0x0f) << 4) |
                             (static_cast<std::uint32_t>(act & 0x0f) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (has_bias ? 1u : 0u) |
        (has_partial ? 2u : 0u) |
        (aligned ? 4u : 0u) |
        (swizzle ? 8u : 0u) |
        ((fast_accum && dt == Dtype::FP16) ? 16u : 0u) |
        (batched ? 32u : 0u));

    CUfunction fn = mir_function({MirKind::GemmMma, op, dt, dt, 1u, dt, flags});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(a),
        reinterpret_cast<std::uintptr_t>(w),
        reinterpret_cast<std::uintptr_t>(c),
        has_bias ? reinterpret_cast<std::uintptr_t>(bias) : 0u,
        has_partial ? reinterpret_cast<std::uintptr_t>(partial) : 0u,
    };
    const std::uint32_t gz = has_partial ? static_cast<std::uint32_t>(splits)
                                         : static_cast<std::uint32_t>(batch);
    if (batched) {
        const std::int32_t ints[] = {m, n, k, kps, gy, sa, sb, sc};
        mir_dispatch_args_3d(
            fn, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy), gz,
            tcfg.threads, ptrs, ints, {});
    } else {
        const std::int32_t ints[] = {m, n, k, kps, gy};
        mir_dispatch_args_3d(
            fn, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy), gz,
            tcfg.threads, ptrs, ints, {});
    }
}

}  // namespace

MirSplitPlan plan_mir_gemm_mma(int m, int n, int k, int sms, bool allow_splitk) {
    MirSplitPlan p;
    const int kt = cdiv_i(k, 32);
    p.kps = kt;

    const long long tiles0 = static_cast<long long>(cdiv_i(m, 128)) * cdiv_i(n, 128);
    if (tiles0 >= static_cast<long long>(sms)) {
        p.cfg = 0;  // 128x128
        return p;
    }
    const long long tiles1 = static_cast<long long>(cdiv_i(m, 64)) * cdiv_i(n, 128);
    if (tiles1 >= 2LL * sms) {
        p.cfg = 1;  // 64x128
        return p;
    }
    const long long tiles2 = static_cast<long long>(cdiv_i(m, 64)) * cdiv_i(n, 64);
    if (tiles2 >= static_cast<long long>(sms)) {
        p.cfg = 2;  // 64x64
        return p;
    }
    const long long tiles3 = static_cast<long long>(cdiv_i(m, 32)) * cdiv_i(n, 64);
    if (tiles3 >= static_cast<long long>(sms)) {
        p.cfg = 3;  // 32x64 (fills SMs in 1 wave without split-K partial traffic)
        return p;
    }

    p.cfg = 2;
    if (allow_splitk && tiles2 < static_cast<long long>(sms)) {
        const long long want = 2LL * sms;
        int s = static_cast<int>((want + tiles2 - 1) / tiles2);
        s = std::min(s, std::min(16, kt / 4));
        if (s >= 2) {
            p.kps = (kt + s - 1) / s;
            p.splits = (kt + p.kps - 1) / p.kps;
            return p;
        }
    }
    if (tiles3 > tiles2) p.cfg = 3;
    return p;
}

void launch_mir_splitk_reduce(Dtype dt, const float* partial, int splits,
                              int m, int n, void* c, const void* bias,
                              int act, int epi) {
    const bool has_bias = (bias != nullptr);
    const std::uint32_t op = static_cast<std::uint32_t>(epi & 0x0f) |
                             (static_cast<std::uint32_t>(act & 0x0f) << 4);
    const std::uint8_t flags = has_bias ? 1u : 0u;
    CUfunction fn = mir_function({MirKind::GemmSplitkReduce, op, dt, dt, 1u, dt, flags});
    const int out_cols = (epi >= 2) ? (n / 2) : n;
    const std::size_t total = static_cast<std::size_t>(m) * out_cols;
    const std::uint32_t threads = 256u;
    const std::uint32_t gx = static_cast<std::uint32_t>((total + threads - 1) / threads);
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(partial),
        reinterpret_cast<std::uintptr_t>(c),
        has_bias ? reinterpret_cast<std::uintptr_t>(bias) : 0u,
    };
    const std::int32_t ints[] = {m, n, splits};
    mir_dispatch_args(fn, gx, threads, ptrs, ints, {});
}

void launch_mir_gemm_mma(Dtype dt, const void* a, const void* w, void* c,
                         int m, int n, int k, const void* bias,
                         int act, int epi, float* ws, std::size_t ws_floats,
                         int sms, bool fast_accum) {
    if (m > 0 && m <= 8 && n > 0 && k > 0 && epi >= 0 && epi <= 3) {
        const int ldc = (epi >= 2) ? (n / 2) : n;
        dispatch_mir_gemv(a, dt, w, dt, c, dt, bias, m, n, k, k, k, ldc, epi, act);
        return;
    }
    MirSplitPlan p = plan_mir_gemm_mma(m, n, k, sms, ws != nullptr);
    if (p.splits > 1 && (!ws || ws_floats < static_cast<std::size_t>(p.splits) * m * n)) {
        p.splits = 1;
        p.kps = cdiv_i(k, 32);
    }
    float* partial = (p.splits > 1) ? ws : nullptr;
    dispatch_mma_kernel(dt, a, w, c, m, n, k, bias, act, epi, partial,
                        1, 0, 0, 0, p.cfg, p.splits, p.kps, fast_accum);
    if (partial) {
        launch_mir_splitk_reduce(dt, partial, p.splits, m, n, c, bias, act, epi);
    }
}

bool launch_mir_gemm_mma_batched(Dtype dt, const void* a, const void* w, void* c,
                                 int batch, int m, int n, int k,
                                 std::size_t sa, std::size_t sb, std::size_t sc,
                                 const void* bias, int act, int sms) {
    const std::size_t i32_max = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (sa > i32_max || sb > i32_max || sc > i32_max) return false;
    if (batch == 1) {
        launch_mir_gemm_mma(dt, a, w, c, m, n, k, bias, act, 0, nullptr, 0, sms, false);
        return true;
    }
    const int eff_sms = std::max(1, sms / batch);
    MirSplitPlan p = plan_mir_gemm_mma(m, n, k, eff_sms, false);
    const int kps = cdiv_i(k, 32);
    constexpr int kMaxZ = 65535;
    const auto* ap = static_cast<const std::uint16_t*>(a);
    const auto* wp = static_cast<const std::uint16_t*>(w);
    auto* cp = static_cast<std::uint16_t*>(c);
    for (int b0 = 0; b0 < batch; b0 += kMaxZ) {
        const int nb = std::min(kMaxZ, batch - b0);
        dispatch_mma_kernel(
            dt,
            ap + static_cast<std::size_t>(b0) * sa,
            wp + static_cast<std::size_t>(b0) * sb,
            cp + static_cast<std::size_t>(b0) * sc,
            m, n, k, bias, act, 0, nullptr, nb,
            static_cast<int>(sa), static_cast<int>(sb), static_cast<int>(sc),
            p.cfg, 1, kps, false);
    }
    return true;
}

void launch_mir_gemm_mma_f32out(Dtype dt, const void* a, const void* w, float* c,
                                int m, int n, int k, int sms) {
    MirSplitPlan p = plan_mir_gemm_mma(m, n, k, sms, false);
    dispatch_mma_kernel(dt, a, w, nullptr, m, n, k, nullptr, 0, 0, c,
                        1, 0, 0, 0, p.cfg, 1, cdiv_i(k, 32), false);
}

}  // namespace brotensor::detail::cuda::mma_gemm
