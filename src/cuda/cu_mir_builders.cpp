#include "cu_mir_builders.h"

#include "../kernels/mir_attn_small.h"
#include "../kernels/mir_audio.h"
#include "../kernels/mir_conv_direct.h"
#include "../kernels/mir_copy.h"
#include "../kernels/mir_delta.h"
#include "../kernels/mir_diffusion.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_fa_bwd.h"
#include "../kernels/mir_fa_rows.h"
#include "../kernels/mir_filtered_lrelu.h"
#include "../kernels/mir_gemm_mma.h"
#include "../kernels/mir_gemm_mma_nn.h"
#include "../kernels/mir_gemm_simt.h"
#include "../kernels/mir_gemm_skinny.h"
#include "../kernels/mir_gemm_splitk.h"
#include "../kernels/mir_gemv.h"
#include "../kernels/mir_gemv_q.h"
#include "../kernels/mir_gnorm.h"
#include "../kernels/mir_linear_bwd.h"
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

#include <stdexcept>

namespace brotensor::detail::cuda {

#if BROTENSOR_HAS_BRASS_CUDA_JIT

void configure_mir_ptx_options(const MirKey& k, brass::target::PtxOptions& opts) {
    if (k.kind == MirKind::GemmMma || k.kind == MirKind::GemmMmaNn) {
        const auto cfg = static_cast<std::size_t>(k.op & 0x0fu);
        const kernels::GemmMmaCfg& tc = kernels::kGemmMmaCfgs[cfg < 4u ? cfg : 0u];
        opts.maxntid[0] = tc.threads;
        opts.minnctapersm = tc.min_ctas_per_sm;
    } else if (k.kind == MirKind::GemmSkinny) {
        opts.maxntid[0] = 256u;
        opts.minnctapersm = 1u;
    }
}

brass::Function* build_mir_kernel(brass::Module& mod, const MirKey& k) {
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
            return kernels::build_resample_bwd_mir(mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags));
        case MirKind::Lstm:
            return kernels::build_lstm_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::DeltaRule:
            return kernels::build_delta_rule_mir(mod, k.op, static_cast<std::uint32_t>(k.flags));
        case MirKind::Xent:
            return kernels::build_xent_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
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
            return kernels::build_vision_mir(mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
        case MirKind::Audio:
            return kernels::build_audio_mir(mod, k.op, k.dt0, k.flags & 0x0fu, (k.flags & 0x10u) != 0);
        case MirKind::Spectral:
            return kernels::build_spectral_mir(
                mod, k.op, k.flags & 0x03u, (k.flags & 0x04u) != 0, (k.flags & 0x08u) != 0);
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
        case MirKind::GemmRowscale:
            return kernels::build_gemm_rowscale_mir(mod, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
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
        case MirKind::FilteredLrelu:
            return kernels::build_filtered_lrelu_mir(
                mod, k.op, k.dt0, (k.flags & 1u) != 0, (k.flags & 2u) != 0, k.aux);
        case MirKind::GemmMma: {
            kernels::GemmMmaSpec spec;
            spec.dt = k.dt0;
            spec.cfg = static_cast<std::uint8_t>(k.op & 0x0fu);
            spec.epi = static_cast<std::uint8_t>((k.op >> 4) & 0x0fu);
            spec.act = static_cast<std::uint8_t>((k.op >> 8) & 0x0fu);
            spec.has_bias = (k.flags & 1u) != 0;
            spec.has_partial = (k.flags & 2u) != 0;
            spec.aligned_mnk = (k.flags & 4u) != 0;
            spec.swizzle_l2 = (k.flags & 8u) != 0;
            spec.fast_accum = (k.flags & 16u) != 0;
            spec.batched = (k.flags & 32u) != 0;
            return kernels::build_gemm_mma_mir(mod, spec);
        }
        case MirKind::GemmSplitkReduce: {
            kernels::GemmSplitkSpec spec;
            spec.dt = k.dt0;
            spec.epi = static_cast<std::uint8_t>(k.op & 0x0fu);
            spec.act = static_cast<std::uint8_t>((k.op >> 4) & 0x0fu);
            spec.has_bias = (k.flags & 1u) != 0;
            return kernels::build_gemm_splitk_reduce_mir(mod, spec);
        }
        case MirKind::GemmSkinny: {
            kernels::GemmSkinnySpec spec;
            spec.dt = k.dt0;
            spec.nt = static_cast<std::uint8_t>(k.op & 0x0fu);
            spec.epi = static_cast<std::uint8_t>((k.op >> 4) & 0x0fu);
            spec.act = static_cast<std::uint8_t>((k.op >> 8) & 0x0fu);
            spec.has_bias = (k.flags & 1u) != 0;
            spec.has_partial = (k.flags & 2u) != 0;
            return kernels::build_gemm_skinny_mir(mod, spec);
        }
        case MirKind::LinearBwd:
            return kernels::build_linear_bwd_mir(mod, k.op, k.dt0);
        case MirKind::GemmMmaNn: {
            kernels::GemmMmaNnSpec spec;
            spec.dt = k.dt0;
            spec.cfg = static_cast<std::uint8_t>(k.op & 0x0fu);
            spec.aligned_mnk = (k.flags & 1u) != 0;
            spec.swizzle_l2 = (k.flags & 2u) != 0;
            return kernels::build_gemm_mma_nn_mir(mod, spec);
        }
        default:
            throw std::runtime_error("build_mir_kernel: unhandled MirKind");
    }
}

#endif

}  // namespace brotensor::detail::cuda
