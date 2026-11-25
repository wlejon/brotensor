#include "ops_fa_mir.h"

#include "cu_mir_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace brotensor::detail::cuda {

namespace {

constexpr std::uint64_t kMaxStaticSmemBytes = 49152u;
constexpr float kLog2e = 1.44269504088896340736f;

struct BwdCfg {
    std::uint32_t nt, br, bc;
};

std::uint64_t bwd_shared_bytes(Dtype dt, std::uint32_t hdp, const BwdCfg& c) {
    const std::uint64_t esz = (dt == Dtype::FP32) ? 4u : 2u;
    return 2u * (c.br + c.bc) * hdp * esz + 8u * c.br * c.bc + 4u * c.bc +
           4u * (c.nt * 2u + c.br * 4u);
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::uint64_t cdiv64(std::uint64_t a, std::uint64_t b) {
    return (a + b - 1u) / b;
}

std::uint32_t pick_fa_rows_g(int hq, int hkv, int hd) {
    const std::uint32_t group = static_cast<std::uint32_t>(hq / hkv);
    const std::uint32_t hds = (static_cast<std::uint32_t>(hd) + 127u) / 128u;
    for (std::uint32_t g = std::min<std::uint32_t>(group, 16u); g >= 1u; --g) {
        if ((group % g) != 0u) continue;
        if (g * hds > 16u) continue;
        const std::uint64_t smem = (static_cast<std::uint64_t>(g) * hd +
                                    static_cast<std::uint64_t>(g) * 128u + 136u) * 4u;
        if (smem <= kMaxStaticSmemBytes) return g;
    }
    throw std::runtime_error("dispatch_fa_rows: head_dim exceeds MIR shared/register budget");
}

std::uint32_t pad_hd_for_cfg(std::uint32_t hd, const BwdCfg& cfg) {
    const std::uint32_t gq = cfg.nt / std::gcd(cfg.nt, cfg.br);
    const std::uint32_t gk = cfg.nt / std::gcd(cfg.nt, cfg.bc);
    const std::uint32_t mult = std::lcm(gq, gk);
    return ((hd + mult - 1u) / mult) * mult;
}

std::pair<BwdCfg, std::uint32_t> pick_fa_bwd_cfg(Dtype dt, std::uint32_t hd) {
    constexpr BwdCfg kCfgs[] = {
        {128u, 32u, 32u},
        {128u, 32u, 16u},
        {32u, 16u, 8u},
        {8u, 8u, 4u},
    };
    for (const BwdCfg& cfg : kCfgs) {
        const std::uint32_t hdp = pad_hd_for_cfg(hd, cfg);
        if (bwd_shared_bytes(dt, hdp, cfg) <= kMaxStaticSmemBytes) {
            return {cfg, hdp};
        }
    }
    throw std::runtime_error("dispatch_fa_bwd: head_dim exceeds 48 KB static shared memory limit");
}

}  // namespace

void dispatch_fa_rows(FaMode mode, Dtype dt, const Device& dev,
                      const void* q, const void* k, const void* v, void* o,
                      const float* mask, const std::int32_t* aux, const std::int32_t* aux2,
                      int lq, int lk, int ldq, int ldk, int ldo,
                      int hq, int hkv, int hd,
                      int window, bool causal, bool maskpos, int nseq, int q_offset,
                      float softcap) {
    if (lq <= 0 || hq <= 0 || hd <= 0) return;

    const std::uint32_t gqa_g = pick_fa_rows_g(hq, hkv, hd);
    const std::uint32_t h_groups = static_cast<std::uint32_t>(hq) / gqa_g;
    const bool vec = (hd % 8 == 0) && (ldk % 8 == 0) && (addr(k) % 16u == 0u);

    const std::uint64_t yz = static_cast<std::uint64_t>(lq) * h_groups;
    std::uint32_t nsplit = 1u;
    int chunk = lk;
    if (lk >= 512 && lq <= 4 && yz < 64u) {
        const std::uint32_t target = std::max<std::uint32_t>(
            2u, std::min<std::uint32_t>(64u, 256u / std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(yz))));
        std::uint64_t raw = cdiv64(static_cast<std::uint64_t>(lk), target);
        raw = cdiv64(raw, 128u) * 128u;
        if (raw < 256u) raw = 256u;
        chunk = static_cast<int>(raw);
        nsplit = static_cast<std::uint32_t>(cdiv64(static_cast<std::uint64_t>(lk), static_cast<std::uint32_t>(chunk)));
    }
    const bool split = nsplit > 1u;

    Tensor part;
    if (split) {
        part = Tensor::empty_on(dev, static_cast<int>(static_cast<std::uint64_t>(lq) * hq * nsplit),
                                hd + 2, Dtype::FP32);
    }

    const std::uint8_t flags = static_cast<std::uint8_t>(
        (mask ? 1u : 0u) |
        (split ? 2u : 0u) |
        (softcap > 0.0f ? 4u : 0u) |
        (vec ? 8u : 0u) |
        (maskpos ? 16u : 0u));
    CUfunction fn = mir_function({MirKind::FaRows, static_cast<std::uint32_t>(hd), dt, Dtype::FP32,
                                  static_cast<std::uint8_t>(gqa_g), Dtype::FP32, flags,
                                  static_cast<std::uint64_t>(mode)});

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const float flts[] = {scale, softcap};
    const int group = hq / hkv;
    for (int r0 = 0; r0 < lq; r0 += 65535) {
        const int row_chunk = std::min(65535, lq - r0);
        const std::uintptr_t ptrs[] = {
            addr(q), addr(k), addr(v), addr(o),
            mask ? addr(mask) : 0u,
            aux ? addr(aux) : 0u,
            aux2 ? addr(aux2) : 0u,
            split ? addr(part.data) : 0u,
        };
        const std::int32_t ints[] = {
            lq, lk, ldq, ldk, ldo, group, window, causal ? 1 : 0, nseq, chunk, hq, r0, q_offset,
        };
        mir_dispatch_args_3d(fn, nsplit, static_cast<std::uint32_t>(row_chunk), h_groups,
                             128u, ptrs, ints, flts);
    }

    if (split) {
        const std::uint32_t entries = static_cast<std::uint32_t>(lq) * static_cast<std::uint32_t>(hq);
        CUfunction cfn = mir_function({MirKind::FaCombine, 0u, dt, dt, 1u});
        const std::uintptr_t cptrs[] = {addr(part.data), addr(o)};
        const std::int32_t cints[] = {
            static_cast<std::int32_t>(entries), static_cast<std::int32_t>(nsplit), hd, hq, ldo,
        };
        const std::uint32_t cgx = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(entries, 65535u));
        mir_dispatch_args_3d(cfn, cgx, 1u, 1u, 256u, cptrs, cints, {});
    }
}

void dispatch_fa_bwd(FaMode mode, Dtype dt, const Device& dev,
                     const void* q, const void* k, const void* v, const void* g,
                     const float* mask, const std::int32_t* aux, const std::int32_t* aux2,
                     void* dq, void* dk, void* dv,
                     int lq, int lk, int ldq, int ldk, int ldg, int ldo,
                     int heads, int hd, int window, bool causal, int nseq) {
    if (heads <= 0 || hd <= 0 || (lq <= 0 && lk <= 0)) return;

    const auto [cfg, hdp] = pick_fa_bwd_cfg(dt, static_cast<std::uint32_t>(hd));
    const std::uint32_t op = (static_cast<std::uint32_t>(hd) & 0xffffu) |
                             ((hdp & 0xffffu) << 16);
    const std::uint64_t kaux = (static_cast<std::uint64_t>(cfg.br) & 0xffffu) |
                               ((static_cast<std::uint64_t>(cfg.bc) & 0xffffu) << 16);
    const std::uint8_t base_flags = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(mode) & 0xfu) | (mask ? 0x10u : 0u));

    const int stats_rows = std::max(1, heads * std::max(1, lq));
    Tensor stats = Tensor::empty_on(dev, stats_rows, 2, Dtype::FP32);

    const float inv_s = 1.0f / std::sqrt(static_cast<float>(hd));
    const float flts[] = {kLog2e * inv_s, inv_s};
    const std::int32_t ints[] = {lq, lk, ldq, ldk, ldg, ldo, window, causal ? 1 : 0, nseq};

    if (lq > 0) {
        CUfunction fn0 = mir_function({MirKind::FaBwd, op, dt, Dtype::FP32,
                                       static_cast<std::uint8_t>(cfg.nt), Dtype::FP32,
                                       base_flags, kaux});
        const std::uintptr_t ptrs0[] = {
            addr(q), addr(k), addr(v), addr(g),
            mask ? addr(mask) : 0u, aux ? addr(aux) : 0u, aux2 ? addr(aux2) : 0u,
            addr(stats.data), addr(dq), 0u,
        };
        mir_dispatch_args_3d(fn0, static_cast<std::uint32_t>(cdiv64(lq, cfg.br)),
                             static_cast<std::uint32_t>(heads), 1u, cfg.nt, ptrs0, ints, flts);
    }

    if (lk > 0) {
        CUfunction fn1 = mir_function({MirKind::FaBwd, op, dt, Dtype::FP32,
                                       static_cast<std::uint8_t>(cfg.nt), Dtype::FP32,
                                       static_cast<std::uint8_t>(base_flags | 0x20u), kaux});
        const std::uintptr_t ptrs1[] = {
            addr(q), addr(k), addr(v), addr(g),
            mask ? addr(mask) : 0u, aux ? addr(aux) : 0u, aux2 ? addr(aux2) : 0u,
            addr(stats.data), addr(dk), addr(dv),
        };
        mir_dispatch_args_3d(fn1, static_cast<std::uint32_t>(cdiv64(lk, cfg.bc)),
                             static_cast<std::uint32_t>(heads), 1u, cfg.nt, ptrs1, ints, flts);
    }
}

}  // namespace brotensor::detail::cuda
