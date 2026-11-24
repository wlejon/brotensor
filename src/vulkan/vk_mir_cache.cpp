#include "vk_mir_cache.h"

#include "../kernels/mir_elementwise.h"

#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace brotensor::detail::vulkan {

namespace {

struct MirKeyHash {
    std::size_t operator()(const MirKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.kind);
        h = (h * 1315423911u) ^ k.op;
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 8) ^ static_cast<std::size_t>(k.dt1);
        h = (h * 1315423911u) ^ k.lanes;
        return h;
    }
};

struct MirCache {
    std::mutex mu;
    std::unordered_map<MirKey, std::unique_ptr<brass::target::SpirvKernel>, MirKeyHash> map;
};

MirCache& mir_cache() {
    static auto* c = new MirCache();
    return *c;
}

const brass::target::SpirvKernel* compile_mir(const MirKey& k) {
    brass::Module mod("vk_mir");
    brass::Function* fn = nullptr;
    switch (k.kind) {
        case MirKind::Unary:  fn = kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Binary: fn = kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::ActBwd: fn = kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Bias:   fn = kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Cast:   fn = kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Glu:    fn = kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
    }
    brass::target::SpirvOptions opts;
    opts.stage = brass::target::SpirvStage::Compute;
    opts.local_size_x = kernels::kEwBlock;
    opts.local_size_spec_constants = true;
    auto spv = std::make_unique<brass::target::SpirvKernel>(brass::target::SpirvTarget::compile(*fn, opts));
    MirCache& cache = mir_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    auto [it, inserted] = cache.map.emplace(k, std::move(spv));
    (void)inserted;
    return it->second.get();
}

template <typename T>
void put_param(std::array<std::uint8_t, 64>& buf, const brass::target::SpirvKernel& sk, std::size_t idx, const T& val) {
    std::memcpy(buf.data() + sk.params[idx].offset, &val, sizeof(T));
}

}  // namespace

MirKernel mir_kernel(DeviceCtx& d, const MirKey& key) {
    const brass::target::SpirvKernel* spv = nullptr;
    {
        MirCache& cache = mir_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        if (auto it = cache.map.find(key); it != cache.map.end()) spv = it->second.get();
    }
    if (!spv) spv = compile_mir(key);
    return MirKernel{spv, jit_pipeline(d, *spv, kernels::kEwBlock)};
}

int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uint64_t a0, std::uint64_t a1, std::uint64_t a2) {
    const std::uint64_t m = a0 | a1 | a2;
    if (dt == Dtype::FP32) {
        return (n % 4u == 0u && (m % 16u == 0u)) ? 4 : 1;
    }
    if (n % 8u == 0u && (m % 16u == 0u)) return 8;
    if (n % 4u == 0u && (m % 8u == 0u)) return 4;
    return 1;
}

int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n, std::uint64_t a0, std::uint64_t a1) {
    if (dt0 == dt1) return mir_vec_lanes(dt0, n, a0, a1);
    const std::uint64_t m = a0 | a1;
    if (dt0 != Dtype::FP32 && dt1 != Dtype::FP32) {
        if (n % 8u == 0u && (m % 16u == 0u)) return 8;
        if (n % 4u == 0u && (m % 8u == 0u)) return 4;
        return 1;
    }
    const std::uint32_t r0 = (dt0 == Dtype::FP32) ? 16u : 8u;
    const std::uint32_t r1 = (dt1 == Dtype::FP32) ? 16u : 8u;
    if (n % 4u == 0u && (a0 % r0 == 0u) && (a1 % r1 == 0u) && (m % 8u == 0u)) return 4;
    return 1;
}

void mir_dispatch_5(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                    std::uint64_t p0, std::uint64_t p1, std::uint32_t p2, float p3, float p4) {
    std::array<std::uint8_t, 64> buf{};
    put_param(buf, *mk.spv, 0, p0);
    put_param(buf, *mk.spv, 1, p1);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(p2));
    put_param(buf, *mk.spv, 3, p3);
    put_param(buf, *mk.spv, 4, p4);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_act_bwd(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                          std::uint64_t p, std::uint64_t dy, std::uint64_t dx, std::uint32_t n, float a) {
    std::array<std::uint8_t, 64> buf{};
    put_param(buf, *mk.spv, 0, p);
    put_param(buf, *mk.spv, 1, dy);
    put_param(buf, *mk.spv, 2, dx);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 4, a);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_bias(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t y, std::uint64_t bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    std::array<std::uint8_t, 64> buf{};
    put_param(buf, *mk.spv, 0, y);
    put_param(buf, *mk.spv, 1, bias);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(dim));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(l));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_cast(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t src, std::uint64_t dst, std::uint32_t n) {
    std::array<std::uint8_t, 64> buf{};
    put_param(buf, *mk.spv, 0, src);
    put_param(buf, *mk.spv, 1, dst);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(n));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_glu(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                      std::uint64_t x, std::uint64_t dy, std::uint64_t y, std::uint32_t n, std::uint32_t dim) {
    std::array<std::uint8_t, 64> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, dy);
    put_param(buf, *mk.spv, 2, y);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(dim));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

}  // namespace brotensor::detail::vulkan
