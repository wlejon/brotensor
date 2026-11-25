#include "vk_autotune.h"

#include "detail/device.h"
#include "detail/kernels.h"
#include "vk_mir_cache.h"
#include "shaders/op_codes.h"
#include "../jit/kernel_disk_cache.h"

#include <brotensor/kernel_cache.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

struct DevTuneMap {
    std::mutex mu;
    std::unordered_map<std::string, GemmTuneCfg> gemm;
    std::unordered_map<std::string, AttnTuneCfg> attn;
};

DevTuneMap& tune_map() {
    static auto* m = new DevTuneMap();
    return *m;
}

std::string dev_prefix(DeviceCtx& d) {
    const auto& p = d.info().props;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "vk_at:%04x:%04x:%08x",
                  p.vendorID, p.deviceID, p.driverVersion);
    return std::string(buf);
}

struct StoredRecord {
    std::uint32_t magic;   // "BTAT" = 0x54415442
    std::uint32_t version; // 1
    std::uint32_t a, b, c, d, e;
};
constexpr std::uint32_t kTuneMagic = 0x54415442u;
constexpr std::uint32_t kTuneVer = 1u;

bool load_cached_record(const std::string& sig, StoredRecord& out) {
    std::vector<std::uint8_t> bytes;
    if (!jit::KernelDiskCache::instance().load_blob("autotune", sig, bytes)) return false;
    if (bytes.size() != sizeof(StoredRecord)) return false;
    std::memcpy(&out, bytes.data(), sizeof(out));
    return out.magic == kTuneMagic && out.version == kTuneVer;
}

void store_cached_record(const std::string& sig, std::uint32_t a, std::uint32_t b,
                         std::uint32_t c, std::uint32_t d, std::uint32_t e) {
    StoredRecord rec{kTuneMagic, kTuneVer, a, b, c, d, e};
    jit::KernelDiskCache::instance().store_blob("autotune", sig, &rec, sizeof(rec));
}

template <class LaunchFn>
double time_candidate(DeviceCtx& d, LaunchFn&& launch) {
    launch();
    d.stream().sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) launch();
    d.stream().sync();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 3.0;
}

} // namespace

bool vk_autotune_live_enabled() noexcept {
    return ::brotensor::autotune_enabled();
}

GemmTuneCfg autotune_gemm_cm(DeviceCtx& d, const GemmArgs& g, std::uint32_t nout,
                             GemmTuneCfg fallback) {
    if (g.m <= 0 || nout == 0 || g.k <= 0) return fallback;
    char sbuf[160];
    std::snprintf(sbuf, sizeof(sbuf), "%s:gemm:%dx%ux%d:b%d:%d_%d_%d:t%d%d:q%d:e%d:bi%d",
                  dev_prefix(d).c_str(), g.m, static_cast<unsigned>(nout), g.k, g.batch,
                  static_cast<int>(g.da), static_cast<int>(g.db), static_cast<int>(g.dc),
                  g.ta ? 1 : 0, g.nb ? 1 : 0, g.qb, g.epi, g.bias != 0 ? 1 : 0);
    const std::string sig(sbuf);

    {
        DevTuneMap& tm = tune_map();
        std::lock_guard<std::mutex> lk(tm.mu);
        if (auto it = tm.gemm.find(sig); it != tm.gemm.end()) return it->second;
    }

    StoredRecord rec{};
    if (load_cached_record(sig, rec) && rec.a < 4u) {
        GemmTuneCfg cfg{rec.a, rec.b, true};
        DevTuneMap& tm = tune_map();
        std::lock_guard<std::mutex> lk(tm.mu);
        tm.gemm[sig] = cfg;
        return cfg;
    }

    const double flops = 2.0 * g.m * double(nout) * g.k * g.batch;
    if (!vk_autotune_live_enabled() || d.stream().capturing() || flops < 1e8 ||
        g.epi == EPI_ACCUM || g.epi == EPI_RESIDUAL_ADD) {
        return fallback;
    }

    struct TileSpec { std::uint32_t bm, bn; };
    constexpr TileSpec kTiles[4] = {{256, 128}, {128, 128}, {128, 64}, {64, 64}};
    constexpr std::pair<std::uint32_t, std::uint32_t> kCands[] = {
        {0u, 0u}, {0u, 4u}, {1u, 0u}, {1u, 4u}, {2u, 4u}, {3u, 0u},
    };

    const long long out_elems = static_cast<long long>(g.m) * static_cast<long long>(g.ldc) *
                                (g.sc > 0 ? g.batch : 1);
    if (out_elems <= 0 || out_elems > 0x7fffffffLL) return fallback;

    ::brotensor::Tensor scratch = ::brotensor::Tensor::empty_on(
        ::brotensor::Device::vulkan(d.index()), static_cast<int>(out_elems), 1, g.dc);
    GemmArgs probe = g;
    probe.c = addr(scratch.data);

    // Temporarily run without re-entering autotune using explicit overrides/calls.
    const bool vec = probe.qb != 0
        ? (probe.a % 16 == 0 && probe.lda % 8 == 0 && probe.k % 8 == 0 &&
           ((probe.qb != QF_INT8 && probe.qb != QF_FP8) || (probe.b % 8 == 0 && probe.ldb % 8 == 0)))
        : (probe.a % 16 == 0 && probe.b % 16 == 0 && probe.lda % 8 == 0 && probe.ldb % 8 == 0 &&
           probe.sa % 8 == 0 && probe.sb % 8 == 0 && (probe.ta ? probe.m : probe.k) % 8 == 0 &&
           (probe.nb ? probe.n : probe.k) % 8 == 0);
    const bool scaled = (probe.da == Dtype::FP32 || probe.da == Dtype::BF16);
    ::brotensor::Tensor exps;
    std::uint64_t ascale = 0;
    if (scaled) {
        exps = ::brotensor::Tensor::empty_on(
            ::brotensor::Device::vulkan(d.index()),
            static_cast<int>(probe.m) * (probe.sa ? probe.batch : 1), 1, Dtype::INT32);
        ascale = addr(exps.data);
    }

    GemmTuneCfg best = fallback;
    double best_us = 1e30;
    const auto cdiv = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
    const std::uint32_t eb = (probe.dc == Dtype::FP32) ? 4u : 2u;

    for (const auto& [ci, sw] : kCands) {
        const auto& ts = kTiles[ci];
        const std::uint32_t wout = probe.epi == EPI_SWIGLU ? ts.bn / 2u : ts.bn;
        const std::uint32_t gx = static_cast<std::uint32_t>(cdiv(nout, wout));
        const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(probe.m, ts.bm));
        if (gx == 0 || gy == 0 || gx > 65535 || gy > 65535) continue;
        const bool direct = probe.da == Dtype::FP16 && probe.dc == Dtype::FP16 &&
                            probe.epi != EPI_GEGLU && probe.c % 16 == 0 && probe.ldc % 8 == 0 &&
                            probe.sc % 8 == 0 && (probe.bias == 0 || probe.bias % 16 == 0);
        const std::uint32_t fx = direct ? static_cast<std::uint32_t>(nout / wout) : 0u;
        const std::uint32_t fy = direct ? static_cast<std::uint32_t>(probe.m / ts.bm) : 0u;
        const std::uint32_t op = (ci & 3u) |
                                 ((static_cast<std::uint32_t>(probe.epi) & 0x0fu) << 4) |
                                 ((static_cast<std::uint32_t>(probe.act) & 0x0fu) << 8) |
                                 ((static_cast<std::uint32_t>(probe.qb) & 0x0fu) << 12);
        const std::uint8_t base_flags = static_cast<std::uint8_t>(
            (probe.ta ? 1u : 0u) | (probe.nb ? 2u : 0u) | (vec ? 4u : 0u) |
            (probe.bias ? 16u : 0u) | (scaled ? 32u : 0u));
        const bool use_sw = (sw != 0u) && ((fx > 0 && fy > 0) || (gx > 0 && gy > 0));
        auto get_mk = [&](bool dir, bool swz) {
            const std::uint8_t flags = static_cast<std::uint8_t>(
                base_flags | (dir ? 8u : 0u) | (swz ? 64u : 0u));
            return mir_kernel(d, {MirKind::GemmCm, op, probe.da, probe.db, 1, probe.dc, flags}, 32u);
        };
        const MirKernel mk_full = get_mk(fx > 0 && fy > 0, use_sw);
        const MirKernel mk_tail = get_mk(false, false);
        const std::int32_t ints_full[] = {
            probe.m, probe.n, probe.k, probe.lda, probe.ldb, probe.ldc,
            static_cast<std::int32_t>(probe.sa), static_cast<std::int32_t>(probe.sb),
            static_cast<std::int32_t>(probe.sc), probe.n / 2, 0, 0,
        };
        const std::uint64_t ptrs[] = {probe.a, probe.b, probe.c, probe.bias, probe.scale, ascale};
        try {
            const double us = time_candidate(d, [&] {
                if (fx > 0 && fy > 0) {
                    mir_dispatch_args_3d(d, mk_full, fx, fy, static_cast<std::uint32_t>(probe.batch), ptrs, ints_full, {});
                    if (fx < gx) {
                        const std::int32_t ints_rx[] = {
                            probe.m, probe.n, probe.k, probe.lda, probe.ldb, probe.ldc,
                            static_cast<std::int32_t>(probe.sa), static_cast<std::int32_t>(probe.sb),
                            static_cast<std::int32_t>(probe.sc), probe.n / 2, 0, static_cast<std::int32_t>(fx),
                        };
                        mir_dispatch_args_3d(d, mk_tail, gx - fx, gy, static_cast<std::uint32_t>(probe.batch), ptrs, ints_rx, {});
                    }
                    if (fy < gy) {
                        const std::int32_t ints_ry[] = {
                            probe.m, probe.n, probe.k, probe.lda, probe.ldb, probe.ldc,
                            static_cast<std::int32_t>(probe.sa), static_cast<std::int32_t>(probe.sb),
                            static_cast<std::int32_t>(probe.sc), probe.n / 2, static_cast<std::int32_t>(fy), 0,
                        };
                        mir_dispatch_args_3d(d, mk_tail, fx, gy - fy, static_cast<std::uint32_t>(probe.batch), ptrs, ints_ry, {});
                    }
                } else {
                    mir_dispatch_args_3d(d, mk_full, gx, gy, static_cast<std::uint32_t>(probe.batch), ptrs, ints_full, {});
                }
            });
            if (us < best_us) {
                best_us = us;
                best = GemmTuneCfg{ci, use_sw ? 4u : 0u, true};
            }
        } catch (...) {
            d.stream().sync();
        }
    }
    (void)eb;

    best.valid = true;
    store_cached_record(sig, best.cfg_idx, best.swizzle_group, 0u, 0u, 0u);
    {
        DevTuneMap& tm = tune_map();
        std::lock_guard<std::mutex> lk(tm.mu);
        tm.gemm[sig] = best;
    }
    return best;
}

AttnTuneCfg autotune_attention(DeviceCtx& d, const AttnProblem& p,
                               bool allow_rows, bool allow_cm,
                               AttnTuneCfg fallback) {
    if (p.lq <= 0 || p.lk <= 0 || p.hd <= 0) return fallback;
    char sbuf[160];
    std::snprintf(sbuf, sizeof(sbuf), "%s:attn:%d_%d:h%d_%d:d%d:dt%d:m%d:c%d",
                  dev_prefix(d).c_str(), p.lq, p.lk, p.hq, p.hkv, p.hd,
                  static_cast<int>(p.dt), p.mode, p.causal ? 1 : 0);
    const std::string sig(sbuf);

    {
        DevTuneMap& tm = tune_map();
        std::lock_guard<std::mutex> lk(tm.mu);
        if (auto it = tm.attn.find(sig); it != tm.attn.end()) return it->second;
    }

    StoredRecord rec{};
    if (load_cached_record(sig, rec) && rec.a <= 1u) {
        AttnTuneCfg cfg{rec.a, rec.b, rec.c, rec.d, rec.e, true};
        if ((cfg.path == 1u && allow_cm) || (cfg.path == 0u && allow_rows)) {
            DevTuneMap& tm = tune_map();
            std::lock_guard<std::mutex> lk(tm.mu);
            tm.attn[sig] = cfg;
            return cfg;
        }
    }

    if (!vk_autotune_live_enabled() || d.stream().capturing() || p.lk < 512) {
        return fallback;
    }

    std::vector<AttnTuneCfg> cands;
    if (allow_cm) {
        if (p.lq < 64) {
            cands.push_back({1u, 32u, 1u, 0u, 160u, true});
            cands.push_back({1u, 32u, 2u, 0u, 160u, true});
            cands.push_back({1u, 16u, 1u, 0u, 160u, true});
        } else {
            if (p.hd <= 96) {
                cands.push_back({1u, 32u, 4u, 1u, 160u, true});
                cands.push_back({1u, 32u, 4u, 0u, 160u, true});
                cands.push_back({1u, 32u, 2u, 1u, 160u, true});
                cands.push_back({1u, 16u, 2u, 1u, 160u, true});
                cands.push_back({1u, 16u, 2u, 0u, 160u, true});
            } else {
                cands.push_back({1u, 32u, 4u, 0u, 160u, true});
                cands.push_back({1u, 32u, 4u, 1u, 160u, true});
                cands.push_back({1u, 32u, 2u, 0u, 160u, true});
            }
        }
    }
    if (allow_rows && p.lq <= 16) {
        cands.push_back({0u, 0u, 0u, 0u, 80u, true});
        cands.push_back({0u, 0u, 0u, 0u, 160u, true});
    }
    if (cands.size() <= 1) return fallback;

    const long long out_elems = static_cast<long long>(p.lq) * static_cast<long long>(p.ldo);
    if (out_elems <= 0 || out_elems > 0x7fffffffLL) return fallback;
    ::brotensor::Tensor scratch = ::brotensor::Tensor::empty_on(
        ::brotensor::Device::vulkan(d.index()), p.lq, p.ldo, p.dt);
    AttnProblem probe = p;
    probe.o = addr(scratch.data);

    // Build shared check helper matching ops_attention.cpp using CM/Rows MIR directly.
    AttnTuneCfg best = fallback;
    double best_us = 1e30;
    const auto cdiv = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
    const float sc_log2 = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(p.hd)));
    const float sc_nat = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.hd)));

    for (const AttnTuneCfg& cand : cands) {
        try {
            double us = 1e30;
            if (cand.path == 1u) {
                const std::uint32_t br = cand.nsg * 16u;
                const bool vec = probe.hd % 8 == 0 && probe.q % 16 == 0 && probe.k % 16 == 0 &&
                                 probe.v % 16 == 0 && probe.ldq % 8 == 0 && probe.ldk % 8 == 0;
                const std::uint32_t nqb = static_cast<std::uint32_t>(cdiv(probe.lq, br));
                const std::uint64_t wgs = static_cast<std::uint64_t>(nqb) * probe.hq;
                std::uint64_t nsplit = 1;
                if (wgs < cand.nsplit_target && probe.lk > 16 * static_cast<int>(cand.bc)) {
                    nsplit = std::min<std::uint64_t>({cdiv(cand.nsplit_target, wgs), cdiv(probe.lk, 8 * cand.bc), 64});
                }
                const std::uint64_t chunk = cdiv(cdiv(probe.lk, nsplit), cand.bc) * cand.bc;
                nsplit = cdiv(probe.lk, chunk);
                const bool split = nsplit > 1;
                const bool direct = !split && probe.dt == Dtype::FP16 && probe.hd % 16 == 0 &&
                                    probe.o % 16 == 0 && probe.ldo % 8 == 0;
                const std::uint32_t op = (static_cast<std::uint32_t>(probe.hd) & 0xffffu) |
                                         ((cand.bc & 0xffu) << 16) |
                                         ((static_cast<std::uint32_t>(probe.mode) & 0x0fu) << 24);
                const std::uint8_t base_flags = static_cast<std::uint8_t>(
                    (probe.mask ? 1u : 0u) | (vec ? 2u : 0u) | (split ? 8u : 0u) |
                    (cand.head_swizzle ? 16u : 0u) | (probe.softcap > 0.0f ? 32u : 0u));
                ::brotensor::Tensor part;
                std::uint64_t part_addr = 0;
                if (split) {
                    const long long pn = static_cast<long long>(probe.lq) * probe.hq *
                                         static_cast<long long>(nsplit) * (probe.hd + 2);
                    part = ::brotensor::Tensor::empty_on(
                        ::brotensor::Device::vulkan(d.index()), static_cast<int>(pn), 1, Dtype::FP32);
                    part_addr = addr(part.data);
                }
                const std::uint64_t ptrs[] = {probe.q, probe.k, probe.v, probe.o, probe.mask, probe.aux, probe.aux2, part_addr};
                const float flts[] = {sc_log2, probe.softcap};
                const MirKernel mk_full = mir_kernel(
                    d, {MirKind::FaCm, op, probe.dt, Dtype::FP32, static_cast<std::uint8_t>(cand.nsg),
                        Dtype::FP32, static_cast<std::uint8_t>(base_flags | (direct ? 4u : 0u))}, 32u);
                const MirKernel mk_tail = mir_kernel(
                    d, {MirKind::FaCm, op, probe.dt, Dtype::FP32, static_cast<std::uint8_t>(cand.nsg),
                        Dtype::FP32, base_flags}, 32u);
                const MirKernel mk_comb = split
                    ? mir_kernel(d, {MirKind::FaCombine, 0u, probe.dt})
                    : MirKernel{};
                const std::uint32_t full = direct ? static_cast<std::uint32_t>(probe.lq) / br : 0u;
                us = time_candidate(d, [&] {
                    auto launch_sub = [&](const MirKernel& mk, std::uint32_t b0, std::uint32_t n) {
                        const std::int32_t ints[] = {
                            probe.lq, probe.lk, probe.ldq, probe.ldk, probe.ldo, probe.hq / probe.hkv,
                            std::max(0, probe.window), probe.causal ? 1 : 0, probe.nseq,
                            static_cast<std::int32_t>(chunk), probe.hq, static_cast<std::int32_t>(b0),
                            probe.q_offset,
                        };
                        const std::uint32_t gx = cand.head_swizzle ? static_cast<std::uint32_t>(probe.hq) : n;
                        const std::uint32_t gy = cand.head_swizzle ? n : static_cast<std::uint32_t>(probe.hq);
                        mir_dispatch_args_3d(d, mk, gx, gy, static_cast<std::uint32_t>(nsplit), ptrs, ints, flts);
                    };
                    if (full > 0) launch_sub(mk_full, 0, full);
                    if (full < nqb) launch_sub(mk_tail, full, nqb - full);
                    if (split) {
                        const std::uint32_t entries = static_cast<std::uint32_t>(static_cast<long long>(probe.lq) * probe.hq);
                        const std::uint64_t cptrs[] = {part_addr, probe.o};
                        const std::int32_t cints[] = {
                            static_cast<std::int32_t>(entries), static_cast<std::int32_t>(nsplit),
                            static_cast<std::int32_t>(probe.hd), static_cast<std::int32_t>(probe.hq),
                            static_cast<std::int32_t>(probe.ldo)};
                        mir_dispatch_args(d, mk_comb, std::min<std::uint32_t>(65535u, entries), cptrs, cints, {});
                    }
                });
            } else {
                const int group = probe.hq / probe.hkv;
                const int slots = (probe.hd + 127) / 128;
                int G = 1;
                for (int g = std::min(group, 16); g >= 1; --g) {
                    if (group % g == 0 && g * slots <= 16 && g * probe.hd <= 8192) { G = g; break; }
                }
                const std::uint64_t entries = static_cast<std::uint64_t>(probe.lq) * (probe.hq / G);
                std::uint64_t nsplit = 1;
                if (entries < cand.nsplit_target && probe.lk > 128) {
                    nsplit = std::min<std::uint64_t>({cdiv(cand.nsplit_target, entries), cdiv(probe.lk, 128), 64});
                }
                std::uint64_t chunk = cdiv(cdiv(probe.lk, nsplit), 128) * 128;
                nsplit = std::max<std::uint64_t>(1, cdiv(probe.lk, chunk));
                const bool vec = probe.hd % 8 == 0 && probe.k % 16 == 0 && probe.ldk % 8 == 0 &&
                                 (probe.dt != Dtype::FP32 || probe.ldk % 4 == 0);
                const bool split = nsplit > 1;
                const std::uint8_t flags = static_cast<std::uint8_t>(
                    (probe.mask != 0 ? 1u : 0u) | (split ? 2u : 0u) | (probe.softcap > 0.0f ? 4u : 0u) |
                    (vec ? 8u : 0u) | (probe.maskpos ? 16u : 0u));
                const MirKernel mk = mir_kernel(
                    d, {MirKind::FaRows, static_cast<std::uint32_t>(probe.hd), probe.dt, Dtype::FP32,
                        static_cast<std::uint8_t>(G), Dtype::FP32, flags, static_cast<std::uint64_t>(probe.mode)});
                ::brotensor::Tensor part;
                std::uint64_t part_addr = 0;
                if (split) {
                    const long long pn = static_cast<long long>(probe.lq) * probe.hq *
                                         static_cast<long long>(nsplit) * (probe.hd + 2);
                    part = ::brotensor::Tensor::empty_on(
                        ::brotensor::Device::vulkan(d.index()), static_cast<int>(pn), 1, Dtype::FP32);
                    part_addr = addr(part.data);
                }
                const MirKernel mk_comb = split
                    ? mir_kernel(d, {MirKind::FaCombine, 0u, probe.dt})
                    : MirKernel{};
                const std::uint64_t ptrs[] = {probe.q, probe.k, probe.v, probe.o, probe.mask, probe.aux, probe.aux2, part_addr};
                const float flts[] = {sc_nat, probe.softcap};
                const std::int32_t ints[] = {
                    probe.lq, probe.lk, probe.ldq, probe.ldk, probe.ldo, group,
                    std::max(0, probe.window), probe.causal ? 1 : 0, probe.nseq,
                    static_cast<std::int32_t>(chunk), probe.hq, 0, probe.q_offset,
                };
                us = time_candidate(d, [&] {
                    mir_dispatch_args_3d(
                        d, mk, static_cast<std::uint32_t>(nsplit),
                        static_cast<std::uint32_t>(probe.lq),
                        static_cast<std::uint32_t>(probe.hq / G), ptrs, ints, flts);
                    if (split) {
                        const std::uint32_t ce = static_cast<std::uint32_t>(static_cast<long long>(probe.lq) * probe.hq);
                        const std::uint64_t cptrs[] = {part_addr, probe.o};
                        const std::int32_t cints[] = {
                            static_cast<std::int32_t>(ce), static_cast<std::int32_t>(nsplit),
                            static_cast<std::int32_t>(probe.hd), static_cast<std::int32_t>(probe.hq),
                            static_cast<std::int32_t>(probe.ldo)};
                        mir_dispatch_args(d, mk_comb, std::min<std::uint32_t>(65535u, ce), cptrs, cints, {});
                    }
                });
            }
            if (us < best_us) {
                best_us = us;
                best = cand;
            }
        } catch (...) {
            d.stream().sync();
        }
    }

    best.valid = true;
    store_cached_record(sig, best.path, best.bc, best.nsg, best.head_swizzle, best.nsplit_target);
    {
        DevTuneMap& tm = tune_map();
        std::lock_guard<std::mutex> lk(tm.mu);
        tm.attn[sig] = best;
    }
    return best;
}

} // namespace brotensor::detail::vulkan
