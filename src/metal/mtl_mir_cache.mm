// Metal runtime pipeline cache for backend-neutral brass MIR kernels.
//
// Include Brass headers BEFORE Objective-C / Metal headers to avoid macro
// collisions (Objective-C defines `NO`, <mach/machine.h> collides with enums).

#include "../kernels/mir_elementwise.h"

#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>

#include "mtl_mir_cache.h"

#import "internal.h"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

struct MirEntry {
    brass::target::MslKernel msl;
    id<MTLComputePipelineState> pso = nil;
};

namespace {

struct MirKeyHash {
    std::size_t operator()(const MirKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.kind);
        h = (h * 1315423911u) ^ k.op;
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 8) ^
            static_cast<std::size_t>(k.dt1);
        h = (h * 1315423911u) ^ k.lanes;
        return h;
    }
};

struct MirCache {
    std::mutex mu;
    std::unordered_map<MirKey, std::unique_ptr<MirEntry>, MirKeyHash> map;
};

MirCache& mir_cache() {
    static auto* c = new MirCache();
    return *c;
}

const MirEntry* compile_mir(const MirKey& k) {
    brass::Module mod("mtl_mir");
    brass::Function* fn = nullptr;
    switch (k.kind) {
        case MirKind::Unary:  fn = kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Binary: fn = kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::ActBwd: fn = kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Bias:   fn = kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Cast:   fn = kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Glu:    fn = kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
    }
    brass::target::MslOptions opts;
    opts.workgroup_x = kernels::kEwBlock;
    auto entry = std::make_unique<MirEntry>();
    entry->msl = brass::target::MslTarget::compile(*fn, opts);
    @autoreleasepool {
        NSString* src = [NSString stringWithUTF8String:entry->msl.source.c_str()];
        NSString* name = [NSString stringWithUTF8String:entry->msl.entry_name.c_str()];
        entry->pso = compile_pipeline(src, name);
    }
    MirCache& cache = mir_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    auto [it, inserted] = cache.map.emplace(k, std::move(entry));
    (void)inserted;
    return it->second.get();
}

template <typename T>
void put_scalar(std::array<std::uint8_t, 64>& buf,
                const brass::target::MslKernel& msl,
                std::size_t idx, const T& val) {
    if (idx < msl.scalar_param_offsets.size()) {
        std::memcpy(buf.data() + msl.scalar_param_offsets[idx], &val, sizeof(T));
    }
}

void submit_mir(const MirEntry& e, std::uint32_t gx,
                const Tensor* const* bufs, std::size_t nbufs,
                const std::array<std::uint8_t, 64>& params) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:e.pso];
        for (std::size_t i = 0; i < nbufs; ++i) {
            const Tensor& t = *bufs[i];
            [enc setBuffer:buffer_for(t) offset:buffer_offset_for(t) atIndex:i];
        }
        if (e.msl.scalar_params_bytes > 0) {
            [enc setBytes:params.data()
                   length:e.msl.scalar_params_bytes
                  atIndex:e.msl.num_buffers];
        }
        [enc dispatchThreadgroups:MTLSizeMake(gx, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(e.msl.workgroup_x, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace

MirKernel mir_kernel(const MirKey& key) {
    const MirEntry* entry = nullptr;
    {
        MirCache& cache = mir_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        if (auto it = cache.map.find(key); it != cache.map.end()) entry = it->second.get();
    }
    if (!entry) entry = compile_mir(key);
    return MirKernel{entry};
}

int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uint64_t a0,
                  std::uint64_t a1, std::uint64_t a2) {
    const std::uint64_t m = a0 | a1 | a2;
    if (dt == Dtype::FP32) {
        return (n % 4u == 0u && (m % 16u == 0u)) ? 4 : 1;
    }
    if (n % 8u == 0u && (m % 16u == 0u)) return 8;
    if (n % 4u == 0u && (m % 8u == 0u)) return 4;
    return 1;
}

int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n,
                        std::uint64_t a0, std::uint64_t a1) {
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

void mir_dispatch_5(const MirKernel& mk, std::uint32_t gx,
                    const Tensor& b0, const Tensor& b1,
                    std::uint32_t p2, float p3, float p4) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 64> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(p2));
    put_scalar(buf, e.msl, 1, p3);
    put_scalar(buf, e.msl, 2, p4);
    const Tensor* bufs[2] = {&b0, &b1};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_act_bwd(const MirKernel& mk, std::uint32_t gx,
                          const Tensor* p, const Tensor& dy, const Tensor& dx,
                          std::uint32_t n, float a) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 64> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, a);
    const Tensor* bufs[3] = {p ? p : &dy, &dy, &dx};
    submit_mir(e, gx, bufs, 3, buf);
}

void mir_dispatch_bias(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& y, const Tensor& bias,
                       std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 64> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(l));
    const Tensor* bufs[2] = {&y, &bias};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_cast(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& src, const Tensor& dst, std::uint32_t n) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 64> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    const Tensor* bufs[2] = {&src, &dst};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_glu(const MirKernel& mk, std::uint32_t gx,
                      const Tensor& x, const Tensor* dy, const Tensor& y,
                      std::uint32_t n, std::uint32_t dim) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 64> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    const Tensor* bufs[3] = {&x, dy ? dy : &x, &y};
    submit_mir(e, gx, bufs, 3, buf);
}

}  // namespace brotensor::detail::metal
