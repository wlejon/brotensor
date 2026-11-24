// Metal runtime pipeline cache for backend-neutral brass MIR kernels.
//
// Include Brass headers BEFORE Objective-C / Metal headers to avoid macro
// collisions (Objective-C defines `NO`, <mach/machine.h> collides with enums).

#include "../kernels/mir_copy.h"
#include "../kernels/mir_delta.h"
#include "../kernels/mir_diffusion.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_gnorm.h"
#include "../kernels/mir_lstm.h"
#include "../kernels/mir_misc.h"
#include "../kernels/mir_norm.h"
#include "../kernels/mir_reduce.h"
#include "../kernels/mir_remap.h"
#include "../kernels/mir_resample.h"
#include "../kernels/mir_resample_bwd.h"
#include "../kernels/mir_rope.h"
#include "../kernels/mir_scatter.h"
#include "../kernels/mir_select.h"
#include "../kernels/mir_topk.h"
#include "../kernels/mir_vision.h"
#include "../kernels/mir_xent.h"

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
using metal_impl::pool_require;

struct MirEntry {
    brass::target::MslKernel msl;
    id<MTLComputePipelineState> pso = nil;
};

namespace {

struct MirKeyHash {
    std::size_t operator()(const MirKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.kind);
        h = (h * 1315423911u) ^ k.op;
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 16) ^
            (static_cast<std::size_t>(k.dt1) << 8) ^ static_cast<std::size_t>(k.dt2);
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.lanes) << 8) ^ k.flags;
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

std::uint32_t block_for(MirKind kind) {
    return (kind == MirKind::FillBytes) ? kernels::kFillBlock : kernels::kEwBlock;
}

const MirEntry* compile_mir(const MirKey& k) {
    brass::Module mod("mtl_mir");
    brass::Function* fn = nullptr;
    switch (k.kind) {
        case MirKind::Unary:      fn = kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Binary:     fn = kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::ActBwd:     fn = kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Bias:       fn = kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Cast:       fn = kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Glu:        fn = kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Rowvec:     fn = kernels::build_rowvec_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::SumRows:    fn = kernels::build_sum_rows_mir(mod, k.dt0, k.lanes); break;
        case MirKind::ArgmaxRows: fn = kernels::build_argmax_rows_mir(mod, k.dt0, k.flags != 0, k.lanes); break;
        case MirKind::SumCols:    fn = kernels::build_sum_cols_mir(mod, k.dt0, k.lanes); break;
        case MirKind::Norm:
            fn = kernels::build_norm_mir(
                mod, k.op, k.dt0, k.dt1, k.dt2,
                kernels::NormMirFlags::unpack(k.flags), k.lanes);
            break;
        case MirKind::Rope:       fn = kernels::build_rope_mir(mod, k.op, k.flags != 0, k.dt0); break;
        case MirKind::FillBytes:  fn = kernels::build_fill_bytes_mir(mod); break;
        case MirKind::Copy2d:     fn = kernels::build_copy2d_mir(mod, static_cast<int>(k.op), k.lanes); break;
        case MirKind::Transpose:  fn = kernels::build_transpose_mir(mod, static_cast<int>(k.op)); break;
        case MirKind::ScatterAdd: fn = kernels::build_scatter_add_mir(mod, k.op, k.dt0); break;
        case MirKind::Gnorm:      fn = kernels::build_gnorm_mir(mod, k.op, k.dt0, k.flags != 0, k.lanes); break;
        case MirKind::Bnorm:      fn = kernels::build_bnorm_mir(mod, k.op); break;
        case MirKind::Sampler:    fn = kernels::build_sampler_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Philox:     fn = kernels::build_philox_mir(mod, k.op); break;
        case MirKind::Remap:      fn = kernels::build_remap_mir(mod, k.op, k.lanes); break;
        case MirKind::Resample: {
            const bool align = (k.flags & 1u) != 0;
            const int imode = static_cast<int>(k.flags >> 1);
            fn = kernels::build_resample_mir(mod, k.op, k.dt0, align, imode);
            break;
        }
        case MirKind::ResampleBwd:
            fn = kernels::build_resample_bwd_mir(
                mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Lstm:
            fn = kernels::build_lstm_mir(
                mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::DeltaRule:
            fn = kernels::build_delta_rule_mir(
                mod, k.op, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Xent:
            fn = kernels::build_xent_mir(
                mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::Topk:
            fn = kernels::build_topk_mir(mod, k.op, k.dt0);
            break;
        case MirKind::SelectSample:
            fn = kernels::build_select_sample_mir(mod, k.flags != 0);
            break;
        case MirKind::SelectCounter:
            fn = kernels::build_select_counter_mir(mod);
            break;
        case MirKind::SelectCommit:
            fn = kernels::build_select_commit_mir(mod);
            break;
        case MirKind::SelectMd:
            fn = kernels::build_select_md_mir(
                mod, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0);
            break;
        case MirKind::Misc:
            fn = kernels::build_misc_mir(mod, k.op, k.dt0, k.flags != 0);
            break;
        case MirKind::Vision:
            fn = kernels::build_vision_mir(
                mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
    }
    brass::target::MslOptions opts;
    opts.workgroup_x = block_for(k.kind);
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
void put_scalar(std::array<std::uint8_t, 128>& buf,
                const brass::target::MslKernel& msl,
                std::size_t idx, const T& val) {
    if (idx < msl.scalar_param_offsets.size()) {
        std::memcpy(buf.data() + msl.scalar_param_offsets[idx], &val, sizeof(T));
    }
}

bool ptr_ok(std::uint64_t p, Dtype dt, int lanes) {
    if (p == 0) return true;
    const std::uint64_t req =
        static_cast<std::uint64_t>(lanes) * (dt == Dtype::FP32 ? 4u : 2u);
    return (p % req) == 0u;
}

void submit_mir(const MirEntry& e, std::uint32_t gx,
                const Tensor* const* bufs, std::size_t nbufs,
                const std::array<std::uint8_t, 128>& params) {
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

void submit_mir_ptrs(const MirEntry& e, std::uint32_t gx, std::uint32_t gy,
                     std::uint32_t gz, const char* op,
                     const void* const* ptrs, std::size_t nbufs,
                     const std::array<std::uint8_t, 128>& params) {
    @autoreleasepool {
        std::array<id<MTLBuffer>, 16> bufs{};
        std::array<NSUInteger, 16> offs{};
        NSUInteger fb_ofs = 0;
        id<MTLBuffer> fb_buf = nil;
        for (std::size_t i = 0; i < nbufs; ++i) {
            if (ptrs[i]) {
                bufs[i] = pool_require(ptrs[i], offs[i], op, "buffer");
                if (!fb_buf) { fb_buf = bufs[i]; fb_ofs = offs[i]; }
            }
        }
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:e.pso];
        for (std::size_t i = 0; i < nbufs; ++i) {
            [enc setBuffer:(bufs[i] ? bufs[i] : fb_buf)
                    offset:(bufs[i] ? offs[i] : fb_ofs)
                   atIndex:i];
        }
        if (e.msl.scalar_params_bytes > 0) {
            [enc setBytes:params.data()
                   length:e.msl.scalar_params_bytes
                  atIndex:e.msl.num_buffers];
        }
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
            threadsPerThreadgroup:MTLSizeMake(e.msl.workgroup_x,
                                              e.msl.workgroup_y,
                                              e.msl.workgroup_z)];
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
                  std::uint64_t a1, std::uint64_t a2, std::uint64_t a3) {
    const std::uint64_t m = a0 | a1 | a2 | a3;
    if (dt == Dtype::FP32) return (n % 4u == 0u && (m % 16u == 0u)) ? 4 : 1;
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
    if (n % 4u == 0u && ptr_ok(a0, dt0, 4) && ptr_ok(a1, dt1, 4)) return 4;
    return 1;
}

int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g,
                       std::uint64_t b, std::uint64_t xh,
                       std::uint64_t s0_mask) {
    const bool any_f32 = (dt_x == Dtype::FP32) || (dt_y == Dtype::FP32) ||
                         (g != 0 && dt_g == Dtype::FP32) ||
                         (b != 0 && dt_g == Dtype::FP32) || (s0_mask != 0);
    const int candidates[2] = {any_f32 ? 4 : 8, 4};
    for (int l : candidates) {
        if (d % static_cast<std::uint32_t>(l) != 0u) continue;
        if (ptr_ok(x, dt_x, l) && ptr_ok(y, dt_y, l) && ptr_ok(g, dt_g, l) &&
            ptr_ok(b, dt_g, l) && ptr_ok(xh, dt_x, l) &&
            ptr_ok(s0_mask, Dtype::FP32, l)) {
            return l;
        }
    }
    return 1;
}

int mir_copy2d_lanes(int elem_bytes, std::uint32_t width,
                     std::uint32_t src_pitch, std::uint32_t dst_pitch,
                     std::uint64_t src, std::uint64_t dst) {
    const std::uint32_t u = width | src_pitch | dst_pitch;
    if (((src | dst) % 16u) != 0u) return 1;
    if (elem_bytes == 4 && (u % 4u == 0u)) return 4;
    if (elem_bytes == 2 && (u % 8u == 0u)) return 8;
    return 1;
}

void mir_dispatch_5(const MirKernel& mk, std::uint32_t gx,
                    const Tensor& b0, const Tensor& b1,
                    std::uint32_t p2, float p3, float p4) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
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
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, a);
    const Tensor* bufs[3] = {p ? p : &dy, &dy, &dx};
    submit_mir(e, gx, bufs, 3, buf);
}

void mir_dispatch_bias(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& y, const Tensor& bias,
                       std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(l));
    const Tensor* bufs[2] = {&y, &bias};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_cast(const MirKernel& mk, std::uint32_t gx,
                       const Tensor& src, const Tensor& dst, std::uint32_t n) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    const Tensor* bufs[2] = {&src, &dst};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_glu(const MirKernel& mk, std::uint32_t gx,
                      const Tensor& x, const Tensor* dy, const Tensor& y,
                      std::uint32_t n, std::uint32_t dim) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    const Tensor* bufs[3] = {&x, dy ? dy : &x, &y};
    submit_mir(e, gx, bufs, 3, buf);
}

void mir_dispatch_rowvec(const MirKernel& mk, std::uint32_t gx,
                         const Tensor& x, const Tensor& v, const Tensor* w,
                         const Tensor& y, std::uint32_t n, std::uint32_t dim) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    const Tensor* bufs[4] = {&x, &v, w ? w : &v, &y};
    submit_mir(e, gx, bufs, 4, buf);
}

void mir_dispatch_reduce(const MirKernel& mk, std::uint32_t gx,
                         const Tensor& x, const Tensor& y,
                         std::uint32_t m, std::uint32_t n) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(m));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(n));
    const Tensor* bufs[2] = {&x, &y};
    submit_mir(e, gx, bufs, 2, buf);
}

void mir_dispatch_norm(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* x, const void* y, const void* g,
                       const void* b, const void* xh, const void* s0,
                       const void* s1, std::uint32_t rows, std::uint32_t dim,
                       float eps, float rstd) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(rows));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    put_scalar(buf, e.msl, 2, eps);
    put_scalar(buf, e.msl, 3, rstd);
    const void* ptrs[7] = {x, y, g, b, xh, s0, s1};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 7, buf);
}

void mir_dispatch_rope(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* x, const void* y, const void* ct,
                       const void* st, const void* ch, const void* sh,
                       const void* cw, const void* sw, const void* pt,
                       const void* ph, const void* pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(rows));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(heads));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(hd));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(dt));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(dh));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(dw));
    put_scalar(buf, e.msl, 6, offset);
    put_scalar(buf, e.msl, 7, ln_base);
    const void* ptrs[11] = {x, y, ct, st, ch, sh, cw, sw, pt, ph, pw};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 11, buf);
}

void mir_dispatch_fill(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* dst, std::uint32_t n, std::uint32_t val) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(val));
    const void* ptrs[1] = {dst};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 1, buf);
}

void mir_dispatch_copy2d(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                         const char* op, const void* src, const void* dst,
                         std::uint32_t src_pitch, std::uint32_t dst_pitch,
                         std::uint32_t width, std::uint32_t height) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(src_pitch));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dst_pitch));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(width));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(height));
    const void* ptrs[2] = {src, dst};
    submit_mir_ptrs(e, gx, gy, 1, op, ptrs, 2, buf);
}

void mir_dispatch_transpose(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                            std::uint32_t gz, const char* op,
                            const void* src, const void* dst,
                            std::uint32_t batch, std::uint32_t rows, std::uint32_t cols) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(batch));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(rows));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(cols));
    const void* ptrs[2] = {src, dst};
    submit_mir_ptrs(e, gx, gy, gz, op, ptrs, 2, buf);
}

void mir_dispatch_scatter(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                          const char* op, const void* idx, const void* keys,
                          const void* keys_out, const void* src, const void* dst,
                          std::uint32_t m, std::uint32_t r, std::uint32_t c,
                          std::uint32_t p, std::uint32_t k, std::uint32_t j) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(m));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(r));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(p));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(k));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(j));
    const void* ptrs[5] = {idx, keys, keys_out, src, dst};
    submit_mir_ptrs(e, gx, gy, 1, op, ptrs, 5, buf);
}

void mir_dispatch_gnorm(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                        const char* op, const void* x, const void* y,
                        const void* g, const void* b, const void* part,
                        const void* rm, const void* rv,
                        std::uint32_t tile, std::uint32_t hw, std::uint32_t cpg,
                        std::uint32_t groups, std::uint32_t splits, std::uint32_t chunk,
                        std::uint32_t total, std::uint32_t c, float eps) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(tile));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(hw));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(cpg));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(groups));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(splits));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(chunk));
    put_scalar(buf, e.msl, 6, static_cast<std::int32_t>(total));
    put_scalar(buf, e.msl, 7, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 8, eps);
    const void* ptrs[7] = {x, y, g, b, part, rm, rv};
    submit_mir_ptrs(e, gx, gy, 1, op, ptrs, 7, buf);
}

void mir_dispatch_bnorm(const MirKernel& mk, std::uint32_t gx, const char* op,
                        const void* x, const void* y, const void* g, const void* b,
                        const void* rm, const void* rv, const void* sm, const void* sr,
                        const void* dy, const void* dg, const void* db,
                        std::uint32_t n, std::uint32_t c, std::uint32_t hw,
                        float eps, float momentum) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(hw));
    put_scalar(buf, e.msl, 3, eps);
    put_scalar(buf, e.msl, 4, momentum);
    const void* ptrs[11] = {x, y, g, b, rm, rv, sm, sr, dy, dg, db};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 11, buf);
}

void mir_dispatch_sampler(const MirKernel& mk, std::uint32_t gx, const char* op,
                          const void* x, const void* e_ptr, const void* p,
                          const void* y, const void* z,
                          std::uint32_t n, std::uint32_t dim, std::uint32_t half,
                          std::uint32_t hw, std::uint32_t c,
                          float k0, float k1, float k2, float k3) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(dim));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(half));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(hw));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 5, k0);
    put_scalar(buf, e.msl, 6, k1);
    put_scalar(buf, e.msl, 7, k2);
    put_scalar(buf, e.msl, 8, k3);
    const void* ptrs[5] = {x, e_ptr, p, y, z};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 5, buf);
}

void mir_dispatch_philox(const MirKernel& mk, std::uint32_t gx, const char* op,
                         const void* y, std::uint64_t key, std::uint64_t counter,
                         std::uint64_t n, float p, float lo, float hi) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(key));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(key >> 32));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(counter));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(counter >> 32));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(n >> 32));
    put_scalar(buf, e.msl, 6, p);
    put_scalar(buf, e.msl, 7, lo);
    put_scalar(buf, e.msl, 8, hi);
    const void* ptrs[1] = {y};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 1, buf);
}

void mir_dispatch_remap(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                        std::uint32_t gz, const char* op,
                        const void* x, const void* y, const void* idx,
                        std::uint32_t c, std::uint32_t cin, std::uint32_t hi,
                        std::uint32_t wi, std::uint32_t ho, std::uint32_t wo,
                        std::int32_t a0, std::int32_t a1, std::int32_t a2,
                        std::int32_t a3, std::int32_t a4, std::int32_t a5,
                        std::int32_t a6) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(cin));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(hi));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(wi));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(ho));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(wo));
    put_scalar(buf, e.msl, 6, a0);
    put_scalar(buf, e.msl, 7, a1);
    put_scalar(buf, e.msl, 8, a2);
    put_scalar(buf, e.msl, 9, a3);
    put_scalar(buf, e.msl, 10, a4);
    put_scalar(buf, e.msl, 11, a5);
    put_scalar(buf, e.msl, 12, a6);
    const void* ptrs[3] = {x, y, idx};
    submit_mir_ptrs(e, gx, gy, gz, op, ptrs, 3, buf);
}

void mir_dispatch_resample(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                           std::uint32_t gz, const char* op,
                           const void* x, const void* y, const void* aux,
                           std::uint32_t c, std::uint32_t hi, std::uint32_t wi,
                           std::uint32_t ho, std::uint32_t wo, std::uint32_t kh,
                           std::uint32_t kw, std::int32_t sh, std::int32_t sw,
                           std::int32_t ph, std::int32_t pw) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(c));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(hi));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(wi));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(ho));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(wo));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(kh));
    put_scalar(buf, e.msl, 6, static_cast<std::int32_t>(kw));
    put_scalar(buf, e.msl, 7, sh);
    put_scalar(buf, e.msl, 8, sw);
    put_scalar(buf, e.msl, 9, ph);
    put_scalar(buf, e.msl, 10, pw);
    const void* ptrs[3] = {x, y, aux};
    submit_mir_ptrs(e, gx, gy, gz, op, ptrs, 3, buf);
}

void mir_dispatch_resample_bwd(const MirKernel& mk, std::uint32_t gx, const char* op,
                               const void* dy, const void* dx, const void* aux,
                               std::uint32_t planes, std::uint32_t hi, std::uint32_t wi,
                               std::uint32_t ho, std::uint32_t wo,
                               std::int32_t pt, std::int32_t pl) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(planes));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(hi));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(wi));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(ho));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(wo));
    put_scalar(buf, e.msl, 5, pt);
    put_scalar(buf, e.msl, 6, pl);
    const void* ptrs[3] = {dy, dx, aux};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 3, buf);
}

void mir_dispatch_lstm(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* gates, const void* c, const void* y,
                       const void* bhh, const void* c0, const void* dy,
                       const void* dz, const void* dh, const void* dc,
                       std::uint32_t b, std::uint32_t h, std::uint32_t row0) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(b));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(h));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(row0));
    const void* ptrs[9] = {gates, c, y, bhh, c0, dy, dz, dh, dc};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 9, buf);
}

void mir_dispatch_delta(const MirKernel& mk, std::uint32_t gx, const char* op,
                        const void* q, const void* k, const void* v,
                        const void* a, const void* b, const void* loga,
                        const void* s, const void* o,
                        std::uint32_t L, std::uint32_t H, std::uint32_t dk,
                        std::uint32_t dv, std::uint32_t ldq, std::uint32_t ldv) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(L));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(H));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(dk));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(dv));
    put_scalar(buf, e.msl, 4, static_cast<std::int32_t>(ldq));
    put_scalar(buf, e.msl, 5, static_cast<std::int32_t>(ldv));
    const void* ptrs[8] = {q, k, v, a, b, loga, s, o};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 8, buf);
}

void mir_dispatch_xent(const MirKernel& mk, std::uint32_t gx, const char* op,
                       const void* logits, const void* target, const void* mask,
                       const void* off, const void* probs, const void* dlog,
                       const void* loss, std::uint32_t rows, std::uint32_t n,
                       std::uint32_t heads, float pos_weight) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(rows));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(n));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(heads));
    put_scalar(buf, e.msl, 3, pos_weight);
    const void* ptrs[7] = {logits, target, mask, off, probs, dlog, loss};
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs, 7, buf);
}

void mir_dispatch_topk(const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                       const char* op, const void* x, const void* vals,
                       const void* idx, const void* off,
                       std::uint32_t rows, std::uint32_t cols,
                       std::uint32_t k, std::uint32_t nseg) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    put_scalar(buf, e.msl, 0, static_cast<std::int32_t>(rows));
    put_scalar(buf, e.msl, 1, static_cast<std::int32_t>(cols));
    put_scalar(buf, e.msl, 2, static_cast<std::int32_t>(k));
    put_scalar(buf, e.msl, 3, static_cast<std::int32_t>(nseg));
    const void* ptrs[4] = {x, vals, idx, off};
    submit_mir_ptrs(e, gx, gy, 1, op, ptrs, 4, buf);
}

void mir_dispatch_args(const MirKernel& mk, std::uint32_t gx, const char* op,
                       std::span<const void* const> ptrs,
                       std::span<const std::int32_t> ints,
                       std::span<const float> flts) {
    const MirEntry& e = *mk.entry;
    std::array<std::uint8_t, 128> buf{};
    std::size_t s = 0;
    for (std::int32_t v : ints) put_scalar(buf, e.msl, s++, v);
    for (float f : flts) put_scalar(buf, e.msl, s++, f);
    submit_mir_ptrs(e, gx, 1, 1, op, ptrs.data(), ptrs.size(), buf);
}

}  // namespace brotensor::detail::metal
