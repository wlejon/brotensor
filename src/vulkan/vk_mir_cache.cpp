#include "vk_mir_cache.h"

#include "../kernels/mir_attn_small.h"
#include "../kernels/mir_audio.h"
#include "../kernels/mir_conv_direct.h"
#include "../kernels/mir_copy.h"
#include "../kernels/mir_delta.h"
#include "../kernels/mir_diffusion.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_gemm_simt.h"
#include "../kernels/mir_gemv.h"
#include "../kernels/mir_gnorm.h"
#include "../kernels/mir_lstm.h"
#include "../kernels/mir_misc.h"
#include "../kernels/mir_norm.h"
#include "../kernels/mir_quant_helpers.h"
#include "../kernels/mir_reduce.h"
#include "../kernels/mir_remap.h"
#include "../kernels/mir_resample.h"
#include "../kernels/mir_resample_bwd.h"
#include "../kernels/mir_rope.h"
#include "../kernels/mir_scatter.h"
#include "../kernels/mir_select.h"
#include "../kernels/mir_spectral.h"
#include "../kernels/mir_topk.h"
#include "../kernels/mir_vision.h"
#include "../kernels/mir_xent.h"

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
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 16) ^
            (static_cast<std::size_t>(k.dt1) << 8) ^ static_cast<std::size_t>(k.dt2);
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.lanes) << 8) ^ k.flags;
        h = (h * 1315423911u) ^ static_cast<std::size_t>(k.aux ^ (k.aux >> 32));
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

std::uint32_t block_for(MirKind kind) {
    return (kind == MirKind::FillBytes) ? kernels::kFillBlock : kernels::kEwBlock;
}

const brass::target::SpirvKernel* compile_mir(const MirKey& k) {
    brass::Module mod("vk_mir");
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
                mod, k.op, k.dt0, k.dt1, k.dt2, kernels::NormMirFlags::unpack(k.flags), k.lanes);
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
            fn = kernels::build_resample_bwd_mir(mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Lstm:
            fn = kernels::build_lstm_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::DeltaRule:
            fn = kernels::build_delta_rule_mir(mod, k.op, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Xent:
            fn = kernels::build_xent_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
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
            fn = kernels::build_vision_mir(mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::Audio:
            fn = kernels::build_audio_mir(mod, k.op, k.dt0, k.flags & 0x0fu, (k.flags & 0x10u) != 0);
            break;
        case MirKind::Spectral:
            fn = kernels::build_spectral_mir(
                mod, k.op, k.flags & 0x03u, (k.flags & 0x04u) != 0, (k.flags & 0x08u) != 0);
            break;
        case MirKind::AttnAux:
            fn = kernels::build_attn_aux_mir(mod, k.op, k.dt0, k.flags != 0);
            break;
        case MirKind::AttnBwd:
            fn = kernels::build_attn_bwd_mir(mod, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::FaCombine:
            fn = kernels::build_fa_combine_mir(mod, k.dt0);
            break;
        case MirKind::AttnSoftmax:
            fn = kernels::build_attn_softmax_mir(
                mod, k.dt0, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0);
            break;
        case MirKind::ConvDirect:
            fn = kernels::build_conv_direct_mir(mod, k.op, k.dt0, k.flags != 0);
            break;
        case MirKind::GemmRowscale:
            fn = kernels::build_gemm_rowscale_mir(mod, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::WqRows:
            fn = kernels::build_wq_rows_mir(mod, k.op, k.dt0, k.flags != 0);
            break;
        case MirKind::Dequant:
            fn = kernels::build_dequant_mir(mod, k.op, k.flags != 0);
            break;
        case MirKind::Gemv:
            fn = kernels::build_gemv_mir(
                mod, k.dt0, k.dt1, k.dt2,
                k.op & 0x0fu, (k.op >> 4) & 0x0fu, (k.op >> 8) & 0x0fu,
                (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::GemmSimt: {
            const auto dt_bias = static_cast<Dtype>((k.op >> 8) & 0xffu);
            fn = kernels::build_gemm_simt_mir(
                mod, k.dt0, k.dt1, k.dt2, dt_bias,
                (k.flags & 1u) != 0, k.op & 0x0fu, (k.op >> 4) & 0x0fu,
                (k.flags & 2u) != 0, (k.flags & 4u) != 0, (k.flags & 8u) != 0);
            break;
        }
        case MirKind::ConvSimt: {
            const auto byte = [&](int s) -> int { return static_cast<int>((k.aux >> (s * 8)) & 0xffu); };
            fn = kernels::build_conv_simt_mir(
                mod, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0,
                byte(0), byte(1), byte(2), byte(3), byte(4), byte(5), byte(6), byte(7));
            break;
        }
    }
    brass::target::SpirvOptions opts;
    opts.stage = brass::target::SpirvStage::Compute;
    opts.local_size_x = block_for(k.kind);
    opts.local_size_spec_constants = true;
    auto spv = std::make_unique<brass::target::SpirvKernel>(brass::target::SpirvTarget::compile(*fn, opts));
    MirCache& cache = mir_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    auto [it, inserted] = cache.map.emplace(k, std::move(spv));
    (void)inserted;
    return it->second.get();
}

template <typename T>
void put_param(std::array<std::uint8_t, 128>& buf, const brass::target::SpirvKernel& sk, std::size_t idx, const T& val) {
    std::memcpy(buf.data() + sk.params[idx].offset, &val, sizeof(T));
}

bool ptr_ok(std::uint64_t p, Dtype dt, int lanes) {
    if (p == 0) return true;
    const std::uint64_t req = static_cast<std::uint64_t>(lanes) * (dt == Dtype::FP32 ? 4u : 2u);
    return (p % req) == 0u;
}

}  // namespace

MirKernel mir_kernel(DeviceCtx& d, const MirKey& key, std::uint32_t req_subgroup) {
    const brass::target::SpirvKernel* spv = nullptr;
    {
        MirCache& cache = mir_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        if (auto it = cache.map.find(key); it != cache.map.end()) spv = it->second.get();
    }
    if (!spv) spv = compile_mir(key);
    return MirKernel{spv, jit_pipeline(d, *spv, block_for(key.kind), req_subgroup)};
}

int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uint64_t a0, std::uint64_t a1,
                  std::uint64_t a2, std::uint64_t a3) {
    const std::uint64_t m = a0 | a1 | a2 | a3;
    if (dt == Dtype::FP32) return (n % 4u == 0u && (m % 16u == 0u)) ? 4 : 1;
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
    if (n % 4u == 0u && ptr_ok(a0, dt0, 4) && ptr_ok(a1, dt1, 4)) return 4;
    return 1;
}

int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g,
                       std::uint64_t b, std::uint64_t xh, std::uint64_t s0_mask) {
    const bool any_f32 = (dt_x == Dtype::FP32) || (dt_y == Dtype::FP32) ||
                         (g != 0 && dt_g == Dtype::FP32) ||
                         (b != 0 && dt_g == Dtype::FP32) || (s0_mask != 0);
    const int candidates[2] = {any_f32 ? 4 : 8, 4};
    for (int l : candidates) {
        if (d % static_cast<std::uint32_t>(l) != 0u) continue;
        if (ptr_ok(x, dt_x, l) && ptr_ok(y, dt_y, l) && ptr_ok(g, dt_g, l) &&
            ptr_ok(b, dt_g, l) && ptr_ok(xh, dt_x, l) && ptr_ok(s0_mask, Dtype::FP32, l)) {
            return l;
        }
    }
    return 1;
}

int mir_copy2d_lanes(int elem_bytes, std::uint32_t width, std::uint32_t src_pitch,
                     std::uint32_t dst_pitch, std::uint64_t src, std::uint64_t dst) {
    const std::uint32_t u = width | src_pitch | dst_pitch;
    if (((src | dst) % 16u) != 0u) return 1;
    if (elem_bytes == 4 && (u % 4u == 0u)) return 4;
    if (elem_bytes == 2 && (u % 8u == 0u)) return 8;
    return 1;
}

void mir_dispatch_5(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                    std::uint64_t p0, std::uint64_t p1, std::uint32_t p2, float p3, float p4) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, p0);
    put_param(buf, *mk.spv, 1, p1);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(p2));
    put_param(buf, *mk.spv, 3, p3);
    put_param(buf, *mk.spv, 4, p4);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_act_bwd(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                          std::uint64_t p, std::uint64_t dy, std::uint64_t dx, std::uint32_t n, float a) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, p);
    put_param(buf, *mk.spv, 1, dy);
    put_param(buf, *mk.spv, 2, dx);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 4, a);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_bias(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t y, std::uint64_t bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, y);
    put_param(buf, *mk.spv, 1, bias);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(dim));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(l));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_cast(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t src, std::uint64_t dst, std::uint32_t n) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, src);
    put_param(buf, *mk.spv, 1, dst);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(n));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_glu(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                      std::uint64_t x, std::uint64_t dy, std::uint64_t y, std::uint32_t n, std::uint32_t dim) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, dy);
    put_param(buf, *mk.spv, 2, y);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(dim));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_rowvec(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                         std::uint64_t x, std::uint64_t v, std::uint64_t w, std::uint64_t y,
                         std::uint32_t n, std::uint32_t dim) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, v);
    put_param(buf, *mk.spv, 2, w);
    put_param(buf, *mk.spv, 3, y);
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(dim));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_reduce(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                         std::uint64_t x, std::uint64_t y, std::uint32_t m, std::uint32_t n) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(m));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(n));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_norm(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t x, std::uint64_t y, std::uint64_t g, std::uint64_t b,
                       std::uint64_t xh, std::uint64_t s0, std::uint64_t s1,
                       std::uint32_t rows, std::uint32_t dim, float eps, float rstd) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, g);
    put_param(buf, *mk.spv, 3, b);
    put_param(buf, *mk.spv, 4, xh);
    put_param(buf, *mk.spv, 5, s0);
    put_param(buf, *mk.spv, 6, s1);
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(rows));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(dim));
    put_param(buf, *mk.spv, 9, eps);
    put_param(buf, *mk.spv, 10, rstd);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_rope(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t x, std::uint64_t y, std::uint64_t ct, std::uint64_t st,
                       std::uint64_t ch, std::uint64_t sh, std::uint64_t cw, std::uint64_t sw,
                       std::uint64_t pt, std::uint64_t ph, std::uint64_t pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, ct);
    put_param(buf, *mk.spv, 3, st);
    put_param(buf, *mk.spv, 4, ch);
    put_param(buf, *mk.spv, 5, sh);
    put_param(buf, *mk.spv, 6, cw);
    put_param(buf, *mk.spv, 7, sw);
    put_param(buf, *mk.spv, 8, pt);
    put_param(buf, *mk.spv, 9, ph);
    put_param(buf, *mk.spv, 10, pw);
    put_param(buf, *mk.spv, 11, static_cast<std::int32_t>(rows));
    put_param(buf, *mk.spv, 12, static_cast<std::int32_t>(heads));
    put_param(buf, *mk.spv, 13, static_cast<std::int32_t>(hd));
    put_param(buf, *mk.spv, 14, static_cast<std::int32_t>(dt));
    put_param(buf, *mk.spv, 15, static_cast<std::int32_t>(dh));
    put_param(buf, *mk.spv, 16, static_cast<std::int32_t>(dw));
    put_param(buf, *mk.spv, 17, offset);
    put_param(buf, *mk.spv, 18, ln_base);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_fill(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t dst, std::uint32_t n, std::uint32_t val) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, dst);
    put_param(buf, *mk.spv, 1, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(val));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_copy2d(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                         std::uint64_t src, std::uint64_t dst,
                         std::uint32_t src_pitch, std::uint32_t dst_pitch,
                         std::uint32_t width, std::uint32_t height) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, src);
    put_param(buf, *mk.spv, 1, dst);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(src_pitch));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(dst_pitch));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(width));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(height));
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy);
}

void mir_dispatch_transpose(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                            std::uint64_t src, std::uint64_t dst,
                            std::uint32_t batch, std::uint32_t rows, std::uint32_t cols) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, src);
    put_param(buf, *mk.spv, 1, dst);
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(batch));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(rows));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(cols));
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy, gz);
}

void mir_dispatch_scatter(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                          std::uint64_t idx, std::uint64_t keys, std::uint64_t keys_out,
                          std::uint64_t src, std::uint64_t dst,
                          std::uint32_t m, std::uint32_t r, std::uint32_t c,
                          std::uint32_t p, std::uint32_t k, std::uint32_t j) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, idx);
    put_param(buf, *mk.spv, 1, keys);
    put_param(buf, *mk.spv, 2, keys_out);
    put_param(buf, *mk.spv, 3, src);
    put_param(buf, *mk.spv, 4, dst);
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(m));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(r));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(p));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(k));
    put_param(buf, *mk.spv, 10, static_cast<std::int32_t>(j));
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy);
}

void mir_dispatch_gnorm(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                        std::uint64_t x, std::uint64_t y, std::uint64_t g, std::uint64_t b,
                        std::uint64_t part, std::uint64_t rm, std::uint64_t rv,
                        std::uint32_t tile, std::uint32_t hw, std::uint32_t cpg, std::uint32_t groups,
                        std::uint32_t splits, std::uint32_t chunk, std::uint32_t total, std::uint32_t c, float eps) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, g);
    put_param(buf, *mk.spv, 3, b);
    put_param(buf, *mk.spv, 4, part);
    put_param(buf, *mk.spv, 5, rm);
    put_param(buf, *mk.spv, 6, rv);
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(tile));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(hw));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(cpg));
    put_param(buf, *mk.spv, 10, static_cast<std::int32_t>(groups));
    put_param(buf, *mk.spv, 11, static_cast<std::int32_t>(splits));
    put_param(buf, *mk.spv, 12, static_cast<std::int32_t>(chunk));
    put_param(buf, *mk.spv, 13, static_cast<std::int32_t>(total));
    put_param(buf, *mk.spv, 14, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 15, eps);
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy);
}

void mir_dispatch_bnorm(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                        std::uint64_t x, std::uint64_t y, std::uint64_t g, std::uint64_t b,
                        std::uint64_t rm, std::uint64_t rv, std::uint64_t sm, std::uint64_t sr,
                        std::uint64_t dy, std::uint64_t dg, std::uint64_t db,
                        std::uint32_t n, std::uint32_t c, std::uint32_t hw, float eps, float momentum) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, g);
    put_param(buf, *mk.spv, 3, b);
    put_param(buf, *mk.spv, 4, rm);
    put_param(buf, *mk.spv, 5, rv);
    put_param(buf, *mk.spv, 6, sm);
    put_param(buf, *mk.spv, 7, sr);
    put_param(buf, *mk.spv, 8, dy);
    put_param(buf, *mk.spv, 9, dg);
    put_param(buf, *mk.spv, 10, db);
    put_param(buf, *mk.spv, 11, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 12, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 13, static_cast<std::int32_t>(hw));
    put_param(buf, *mk.spv, 14, eps);
    put_param(buf, *mk.spv, 15, momentum);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_sampler(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                          std::uint64_t x, std::uint64_t e, std::uint64_t p, std::uint64_t y, std::uint64_t z,
                          std::uint32_t n, std::uint32_t dim, std::uint32_t half, std::uint32_t hw, std::uint32_t c,
                          float k0, float k1, float k2, float k3) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, e);
    put_param(buf, *mk.spv, 2, p);
    put_param(buf, *mk.spv, 3, y);
    put_param(buf, *mk.spv, 4, z);
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(dim));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(half));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(hw));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 10, k0);
    put_param(buf, *mk.spv, 11, k1);
    put_param(buf, *mk.spv, 12, k2);
    put_param(buf, *mk.spv, 13, k3);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_philox(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                         std::uint64_t y, std::uint64_t key, std::uint64_t counter, std::uint64_t n,
                         float p, float lo, float hi) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, y);
    put_param(buf, *mk.spv, 1, static_cast<std::int32_t>(key));
    put_param(buf, *mk.spv, 2, static_cast<std::int32_t>(key >> 32));
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(counter));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(counter >> 32));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(n >> 32));
    put_param(buf, *mk.spv, 7, p);
    put_param(buf, *mk.spv, 8, lo);
    put_param(buf, *mk.spv, 9, hi);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_remap(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                        std::uint64_t x, std::uint64_t y, std::uint64_t idx,
                        std::uint32_t c, std::uint32_t cin, std::uint32_t hi, std::uint32_t wi,
                        std::uint32_t ho, std::uint32_t wo,
                        std::int32_t a0, std::int32_t a1, std::int32_t a2, std::int32_t a3,
                        std::int32_t a4, std::int32_t a5, std::int32_t a6) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, idx);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(cin));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(hi));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(wi));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(ho));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(wo));
    put_param(buf, *mk.spv, 9, a0);
    put_param(buf, *mk.spv, 10, a1);
    put_param(buf, *mk.spv, 11, a2);
    put_param(buf, *mk.spv, 12, a3);
    put_param(buf, *mk.spv, 13, a4);
    put_param(buf, *mk.spv, 14, a5);
    put_param(buf, *mk.spv, 15, a6);
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy, gz);
}

void mir_dispatch_resample(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                           std::uint64_t x, std::uint64_t y, std::uint64_t aux,
                           std::uint32_t c, std::uint32_t hi, std::uint32_t wi, std::uint32_t ho, std::uint32_t wo,
                           std::uint32_t kh, std::uint32_t kw,
                           std::int32_t sh, std::int32_t sw, std::int32_t ph, std::int32_t pw) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, y);
    put_param(buf, *mk.spv, 2, aux);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(c));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(hi));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(wi));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(ho));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(wo));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(kh));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(kw));
    put_param(buf, *mk.spv, 10, sh);
    put_param(buf, *mk.spv, 11, sw);
    put_param(buf, *mk.spv, 12, ph);
    put_param(buf, *mk.spv, 13, pw);
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy, gz);
}

void mir_dispatch_resample_bwd(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                               std::uint64_t dy, std::uint64_t dx, std::uint64_t aux,
                               std::uint32_t planes, std::uint32_t hi, std::uint32_t wi,
                               std::uint32_t ho, std::uint32_t wo, std::int32_t pt, std::int32_t pl) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, dy);
    put_param(buf, *mk.spv, 1, dx);
    put_param(buf, *mk.spv, 2, aux);
    put_param(buf, *mk.spv, 3, static_cast<std::int32_t>(planes));
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(hi));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(wi));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(ho));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(wo));
    put_param(buf, *mk.spv, 8, pt);
    put_param(buf, *mk.spv, 9, pl);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_lstm(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t gates, std::uint64_t c, std::uint64_t y, std::uint64_t bhh,
                       std::uint64_t c0, std::uint64_t dy, std::uint64_t dz, std::uint64_t dh, std::uint64_t dc,
                       std::uint32_t b, std::uint32_t h, std::uint32_t row0) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, gates);
    put_param(buf, *mk.spv, 1, c);
    put_param(buf, *mk.spv, 2, y);
    put_param(buf, *mk.spv, 3, bhh);
    put_param(buf, *mk.spv, 4, c0);
    put_param(buf, *mk.spv, 5, dy);
    put_param(buf, *mk.spv, 6, dz);
    put_param(buf, *mk.spv, 7, dh);
    put_param(buf, *mk.spv, 8, dc);
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(b));
    put_param(buf, *mk.spv, 10, static_cast<std::int32_t>(h));
    put_param(buf, *mk.spv, 11, static_cast<std::int32_t>(row0));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_delta(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                        std::uint64_t q, std::uint64_t k, std::uint64_t v, std::uint64_t a,
                        std::uint64_t b, std::uint64_t loga, std::uint64_t s, std::uint64_t o,
                        std::uint32_t L, std::uint32_t H, std::uint32_t dk, std::uint32_t dv,
                        std::uint32_t ldq, std::uint32_t ldv) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, q);
    put_param(buf, *mk.spv, 1, k);
    put_param(buf, *mk.spv, 2, v);
    put_param(buf, *mk.spv, 3, a);
    put_param(buf, *mk.spv, 4, b);
    put_param(buf, *mk.spv, 5, loga);
    put_param(buf, *mk.spv, 6, s);
    put_param(buf, *mk.spv, 7, o);
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(L));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(H));
    put_param(buf, *mk.spv, 10, static_cast<std::int32_t>(dk));
    put_param(buf, *mk.spv, 11, static_cast<std::int32_t>(dv));
    put_param(buf, *mk.spv, 12, static_cast<std::int32_t>(ldq));
    put_param(buf, *mk.spv, 13, static_cast<std::int32_t>(ldv));
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_xent(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::uint64_t logits, std::uint64_t target, std::uint64_t mask, std::uint64_t off,
                       std::uint64_t probs, std::uint64_t dlog, std::uint64_t loss,
                       std::uint32_t rows, std::uint32_t n, std::uint32_t heads, float pos_weight) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, logits);
    put_param(buf, *mk.spv, 1, target);
    put_param(buf, *mk.spv, 2, mask);
    put_param(buf, *mk.spv, 3, off);
    put_param(buf, *mk.spv, 4, probs);
    put_param(buf, *mk.spv, 5, dlog);
    put_param(buf, *mk.spv, 6, loss);
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(rows));
    put_param(buf, *mk.spv, 8, static_cast<std::int32_t>(n));
    put_param(buf, *mk.spv, 9, static_cast<std::int32_t>(heads));
    put_param(buf, *mk.spv, 10, pos_weight);
    jit_dispatch(d, mk.pipe, buf.data(), gx);
}

void mir_dispatch_topk(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx, std::uint32_t gy,
                       std::uint64_t x, std::uint64_t vals, std::uint64_t idx, std::uint64_t off,
                       std::uint32_t rows, std::uint32_t cols, std::uint32_t k, std::uint32_t nseg) {
    std::array<std::uint8_t, 128> buf{};
    put_param(buf, *mk.spv, 0, x);
    put_param(buf, *mk.spv, 1, vals);
    put_param(buf, *mk.spv, 2, idx);
    put_param(buf, *mk.spv, 3, off);
    put_param(buf, *mk.spv, 4, static_cast<std::int32_t>(rows));
    put_param(buf, *mk.spv, 5, static_cast<std::int32_t>(cols));
    put_param(buf, *mk.spv, 6, static_cast<std::int32_t>(k));
    put_param(buf, *mk.spv, 7, static_cast<std::int32_t>(nseg));
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy);
}

void mir_dispatch_args_3d(DeviceCtx& d, const MirKernel& mk,
                          std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                          std::span<const std::uint64_t> ptrs,
                          std::span<const std::int32_t> ints,
                          std::span<const float> flts) {
    std::array<std::uint8_t, 128> buf{};
    std::size_t idx = 0;
    for (std::uint64_t p : ptrs) put_param(buf, *mk.spv, idx++, p);
    for (std::int32_t v : ints) put_param(buf, *mk.spv, idx++, v);
    for (float f : flts) put_param(buf, *mk.spv, idx++, f);
    jit_dispatch(d, mk.pipe, buf.data(), gx, gy, gz);
}

void mir_dispatch_args(DeviceCtx& d, const MirKernel& mk, std::uint32_t gx,
                       std::span<const std::uint64_t> ptrs,
                       std::span<const std::int32_t> ints,
                       std::span<const float> flts) {
    mir_dispatch_args_3d(d, mk, gx, 1u, 1u, ptrs, ints, flts);
}

}  // namespace brotensor::detail::vulkan
