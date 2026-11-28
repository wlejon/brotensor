// Pure-C++ Brass MIR builder dispatch for Metal kernels.

#include "mtl_mir_builders.h"

#include "../kernels/mir_attn_small.h"
#include "../kernels/mir_audio.h"
#include "../kernels/mir_conv_aux.h"
#include "../kernels/mir_conv_cm.h"
#include "../kernels/mir_conv_direct.h"
#include "../kernels/mir_copy.h"
#include "../kernels/mir_delta.h"
#include "../kernels/mir_diffusion.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_fa_bwd.h"
#include "../kernels/mir_fa_rows.h"
#include "../kernels/mir_filtered_lrelu.h"
#include "../kernels/mir_gemm_cm.h"
#include "../kernels/mir_gemm_simt.h"
#include "../kernels/mir_gemv.h"
#include "../kernels/mir_gemv_q.h"
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

#include <stdexcept>

namespace brotensor::detail::metal {

MetalCmTile metal_cm_tile(std::uint32_t cfg_idx) noexcept {
    const auto& t = kernels::kGemmCmTiles[cfg_idx & 3u];
    return {t.bm, t.bn};
}

std::uint32_t metal_mir_block_size(const MirKey& k) {
    if (k.kind == MirKind::FillBytes) return kernels::kFillBlock;
    if (k.kind == MirKind::FaRows) return kernels::kFaRowsBlock;
    if (k.kind == MirKind::FaBwd) return static_cast<std::uint32_t>(k.lanes);
    if (k.kind == MirKind::FilteredLrelu) return (k.op < 2u) ? 128u : 256u;
    if (k.kind == MirKind::GemmCm || k.kind == MirKind::ConvCm) {
        return kernels::kGemmCmTiles[k.op & 3u].wg;
    }
    return kernels::kEwBlock;
}

brass::Function* build_metal_mir_function(brass::Module& mod, const MirKey& k) {
    switch (k.kind) {
        case MirKind::Unary:      return kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes);
        case MirKind::Binary:     return kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes);
        case MirKind::ActBwd:     return kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes);
        case MirKind::Bias:       return kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes);
        case MirKind::Cast:       return kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes);
        case MirKind::Glu:        return kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes);
        case MirKind::Rowvec:     return kernels::build_rowvec_mir(mod, k.op, k.dt0, k.lanes);
        case MirKind::SumRows:    return kernels::build_sum_rows_mir(mod, k.dt0, k.lanes);
        case MirKind::ArgmaxRows: return kernels::build_argmax_rows_mir(mod, k.dt0, k.flags != 0, k.lanes);
        case MirKind::SumCols:    return kernels::build_sum_cols_mir(mod, k.dt0, k.lanes);
        case MirKind::Norm:
            return kernels::build_norm_mir(
                mod, k.op, k.dt0, k.dt1, k.dt2,
                kernels::NormMirFlags::unpack(k.flags), k.lanes);
        case MirKind::Rope:       return kernels::build_rope_mir(mod, k.op, k.flags != 0, k.dt0);
        case MirKind::FillBytes:  return kernels::build_fill_bytes_mir(mod);
        case MirKind::Copy2d:     return kernels::build_copy2d_mir(mod, static_cast<int>(k.op), k.lanes);
        case MirKind::Transpose:  return kernels::build_transpose_mir(mod, static_cast<int>(k.op));
        case MirKind::ScatterAdd: return kernels::build_scatter_add_mir(mod, k.op, k.dt0);
        case MirKind::Gnorm:      return kernels::build_gnorm_mir(mod, k.op, k.dt0, k.flags != 0, k.lanes);
        case MirKind::Bnorm:      return kernels::build_bnorm_mir(mod, k.op);
        case MirKind::Sampler:    return kernels::build_sampler_mir(mod, k.op, k.dt0, k.dt1, k.lanes);
        case MirKind::Philox:     return kernels::build_philox_mir(mod, k.op);
        case MirKind::Remap:      return kernels::build_remap_mir(mod, k.op, k.lanes);
        case MirKind::Resample: {
            const bool align = (k.flags & 1u) != 0;
            const int imode = static_cast<int>(k.flags >> 1);
            return kernels::build_resample_mir(mod, k.op, k.dt0, align, imode);
        }
        case MirKind::ResampleBwd:
            return kernels::build_resample_bwd_mir(
                mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags));
        case MirKind::Lstm:
            return kernels::build_lstm_mir(
                mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::DeltaRule:
            return kernels::build_delta_rule_mir(
                mod, k.op, static_cast<std::uint32_t>(k.flags));
        case MirKind::Xent:
            return kernels::build_xent_mir(
                mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::Topk:
            return kernels::build_topk_mir(mod, k.op, k.dt0);
        case MirKind::SelectSample:
            return kernels::build_select_sample_mir(mod, k.flags != 0);
        case MirKind::SelectCounter:
            return kernels::build_select_counter_mir(mod);
        case MirKind::SelectCommit:
            return kernels::build_select_commit_mir(mod);
        case MirKind::SelectMd:
            return kernels::build_select_md_mir(
                mod, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0);
        case MirKind::Misc:
            return kernels::build_misc_mir(mod, k.op, k.dt0, k.flags != 0);
        case MirKind::Vision:
            return kernels::build_vision_mir(
                mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::Audio:
            return kernels::build_audio_mir(
                mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags & 0x0fu),
                (k.flags & 0x10u) != 0);
        case MirKind::Spectral:
            return kernels::build_spectral_mir(
                mod, k.op, static_cast<std::uint32_t>(k.flags & 0x03u),
                (k.flags & 0x04u) != 0, (k.flags & 0x08u) != 0);
        case MirKind::AttnAux:
            return kernels::build_attn_aux_mir(mod, k.op, k.dt0, k.flags != 0);
        case MirKind::AttnBwd:
            return kernels::build_attn_bwd_mir(mod, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::FaCombine:
            return kernels::build_fa_combine_mir(mod, k.dt0);
        case MirKind::AttnSoftmax:
            return kernels::build_attn_softmax_mir(
                mod, k.dt0, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0);
        case MirKind::ConvDirect:
            return kernels::build_conv_direct_mir(mod, k.op, k.dt0, k.flags != 0);
        case MirKind::WqRows:
            return kernels::build_wq_rows_mir(mod, k.op, k.dt0, k.flags != 0);
        case MirKind::Dequant:
            return kernels::build_dequant_mir(mod, k.op, k.flags != 0);
        case MirKind::Gemv:
            return kernels::build_gemv_mir(
                mod, k.dt0, k.dt1, k.dt2,
                k.op & 0x0fu, (k.op >> 4) & 0x0fu, (k.op >> 8) & 0x0fu,
                (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::GemmSimt: {
            const auto dt_bias = static_cast<Dtype>((k.op >> 8) & 0xffu);
            return kernels::build_gemm_simt_mir(
                mod, k.dt0, k.dt1, k.dt2, dt_bias,
                (k.flags & 1u) != 0, k.op & 0x0fu, (k.op >> 4) & 0x0fu,
                (k.flags & 2u) != 0, (k.flags & 4u) != 0, (k.flags & 8u) != 0);
        }
        case MirKind::ConvSimt: {
            const auto byte = [&](int s) -> int { return static_cast<int>((k.aux >> (s * 8)) & 0xffu); };
            return kernels::build_conv_simt_mir(
                mod, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0,
                byte(0), byte(1), byte(2), byte(3), byte(4), byte(5), byte(6), byte(7),
                k.op & 0x0fu);
        }
        case MirKind::GemvQ: {
            const std::uint32_t fmt = k.op & 0x0fu;
            const std::uint32_t mb = (k.op >> 4) & 0x0fu;
            const std::uint32_t lpr = k.lanes;
            const std::uint32_t unr = (k.op >> 8) & 0x0fu;
            const std::uint32_t nr = (k.op >> 12) & 0x0fu;
            const std::uint32_t act = (k.op >> 16) & 0x0fu;
            const std::uint32_t epi = (k.op >> 20) & 0x0fu;
            return kernels::build_gemv_q_mir(
                mod, k.dt0, fmt, mb, lpr, (k.flags & 1u) != 0, (k.flags & 2u) != 0,
                unr, nr, act, epi);
        }
        case MirKind::FaRows:
            return kernels::build_fa_rows_mir(
                mod, k.dt0, k.op, k.lanes, static_cast<std::uint32_t>(k.aux),
                (k.flags & 1u) != 0, (k.flags & 2u) != 0, (k.flags & 4u) != 0,
                (k.flags & 8u) != 0, (k.flags & 16u) != 0);
        case MirKind::FaBwd: {
            const std::uint32_t hd = k.op & 0xffffu;
            const std::uint32_t hdp = (k.op >> 16) & 0xffffu;
            const std::uint32_t br = static_cast<std::uint32_t>(k.aux & 0xffffu);
            const std::uint32_t bc = static_cast<std::uint32_t>((k.aux >> 16) & 0xffffu);
            const std::uint32_t mode = k.flags & 0x0fu;
            const bool has_mask = (k.flags & 0x10u) != 0;
            const std::uint32_t pass = (k.flags >> 5) & 0x01u;
            return kernels::build_fa_bwd_mir(mod, k.dt0, k.lanes, hd, hdp, br, bc, mode, has_mask, pass);
        }
        case MirKind::ConvAux:
            return kernels::build_conv_aux_mir(mod, k.op, k.dt0, k.flags);
        case MirKind::FilteredLrelu:
            return kernels::build_filtered_lrelu_mir(
                mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0, k.aux);
        case MirKind::GemmCm: {
            kernels::GemmCmSpec s;
            s.dt_a = k.dt0;
            s.dt_b = k.dt1;
            s.dt_c = k.dt2;
            s.cfg_idx = k.op & 3u;
            s.epi = (k.op >> 4) & 0x0fu;
            s.act = (k.op >> 8) & 0x0fu;
            s.qb = (k.op >> 12) & 0x0fu;
            s.ta = (k.flags & 1u) != 0;
            s.nb = (k.flags & 2u) != 0;
            s.vec = (k.flags & 4u) != 0;
            s.direct = (k.flags & 8u) != 0;
            s.has_bias = (k.flags & 16u) != 0;
            s.scale_a = (k.flags & 32u) != 0;
            s.swizzle = (k.flags & 64u) != 0;
            return kernels::build_gemm_cm_mir(mod, s);
        }
        case MirKind::ConvCm: {
            const auto byte = [&](int sh) -> int { return static_cast<int>((k.aux >> (sh * 8)) & 0xffu); };
            kernels::ConvCmSpec s;
            s.dt = k.dt0;
            s.cfg_idx = k.op & 3u;
            s.act = (k.op >> 4) & 0x0fu;
            s.bvec = (k.op >> 8) & 0x03u;
            s.qa = (k.op >> 12) & 0x0fu;
            s.avec = (k.flags & 1u) != 0;
            s.direct = (k.flags & 2u) != 0;
            s.has_bias = (k.flags & 4u) != 0;
            s.rowch = (k.flags & 8u) != 0;
            s.accum = (k.flags & 16u) != 0;
            s.kh = byte(0);
            s.kw = byte(1);
            s.sh = byte(2);
            s.sw = byte(3);
            s.ph = byte(4);
            s.pw = byte(5);
            s.dh = byte(6);
            s.dw = byte(7);
            return kernels::build_conv_cm_mir(mod, s);
        }
    }
    throw std::runtime_error("build_metal_mir_function: unknown MirKind");
}

}  // namespace brotensor::detail::metal
