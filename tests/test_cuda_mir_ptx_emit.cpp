// GPU-less verification of brotensor's pure-C++ CUDA MIR -> PTX pipeline.
// Builds representative kernels across core and Wave 1-5 CUDA MirKinds, runs
// Brass GPU MIR optimization, and lowers via PtxTarget::emit_function (which
// validates every lowered PTX function with ptx::verify) with zero NVIDIA GPU
// or CUDA driver required.

#include "../src/cuda/cu_mir_builders.h"
#include "../src/cuda/cu_mir_cache.h"
#include "../src/jit/kernel_disk_cache.h"
#include "../src/vulkan/shaders/op_codes.h"

#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx_target.hpp>

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using brotensor::Dtype;
using namespace brotensor::detail::cuda;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

bool is_mma_loop_kind(MirKind kind) noexcept {
    return kind == MirKind::GemmMma || kind == MirKind::GemmMmaNn ||
           kind == MirKind::GemmSkinny || kind == MirKind::GemmQuantMma ||
           kind == MirKind::Conv2dInt8wMma || kind == MirKind::Conv2dMma ||
           kind == MirKind::Conv2dIgemmF32 || kind == MirKind::DecodeGqa ||
           kind == MirKind::FaFusedMma || kind == MirKind::FaPackedMma ||
           kind == MirKind::FaVarlenF32;
}

std::string emit_verified_ptx(const std::string& label, const MirKey& k) {
    brass::Module mod("cuda_mir_test");
    brass::Function* fn = build_mir_kernel(mod, k);
    check(fn != nullptr, label + ": build_mir_kernel returned null");
    if (!fn) return {};

    brass::GpuOptOptions gopt;
    gopt.opt_level = 2;
    gopt.enable_fma = true;
    gopt.enable_gpu_superopt = true;
    const bool mma_loop = is_mma_loop_kind(k.kind);
    gopt.enable_small_loop_unroll = !mma_loop;
    gopt.enable_load_coalesce = !mma_loop;
    brass::optimize_gpu_function(*fn, mod, gopt);

    brass::target::PtxOptions ptx_opts;
    ptx_opts.sm_arch = "sm_80";
    configure_mir_ptx_options(k, ptx_opts);

    std::string ptx = brass::target::PtxTarget::emit_function(*fn, ptx_opts);
    check(ptx.find(".visible .entry") != std::string::npos,
          label + ": missing .visible .entry in PTX");
    std::printf("  [PASS] %-38s (%zu bytes PTX)\n", label.c_str(), ptx.size());
    return ptx;
}

}  // namespace

int main() {
    std::printf("--- Kernel source fingerprint ---\n");
    const std::string_view fp = brotensor::jit::kernel_source_fingerprint();
    bool hex_ok = (fp.size() == 16);
    for (char c : fp) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) hex_ok = false;
    }
    check(hex_ok, "kernel_source_fingerprint is 16 lowercase hex digits");
    std::printf("  [PASS] kernel_source_fingerprint = %.*s\n",
                static_cast<int>(fp.size()), fp.data());

    std::printf("\n--- Core + Wave 1-5 CUDA MIR -> PTX verification ---\n");

    struct Case {
        const char* name;
        MirKey key;
        const char* expect_sub;
    };

    const std::uint64_t fl_aux =
        std::uint64_t{2} | (std::uint64_t{2} << 8) | (std::uint64_t{2} << 16) |
        (std::uint64_t{2} << 24) | (std::uint64_t{3} << 32) | (std::uint64_t{3} << 40);

    const std::uint64_t fa_bwd_aux =
        std::uint64_t{64} | (std::uint64_t{32} << 16);  // Br=64, Bc=32

    const std::vector<Case> cases = {
        // Core
        {"Unary silu f16x8",
         MirKey{MirKind::Unary, UOP_SILU, Dtype::FP16, Dtype::FP16, 8, Dtype::FP32, 0, 0},
         ".entry"},
        {"Binary axpby bf16x8",
         MirKey{MirKind::Binary, BOP_AXPBY, Dtype::BF16, Dtype::BF16, 8, Dtype::FP32, 0, 0},
         ".entry"},
        {"Norm RMS f16",
         MirKey{MirKind::Norm, NORM_RMS, Dtype::FP16, Dtype::FP16, 4, Dtype::FP16,
                pack_norm_flags(true, false, false, false, false), 0},
         "bar.sync"},
        {"RoPE theta bf16",
         MirKey{MirKind::Rope, ROPE_THETA, Dtype::BF16, Dtype::BF16, 1, Dtype::FP32, 0, 0},
         ".entry"},
        {"FilteredLrelu f16",
         MirKey{MirKind::FilteredLrelu, 2, Dtype::FP16, Dtype::FP16, 1, Dtype::FP32, 3, fl_aux},
         "bar.sync"},

        // Wave 1: GEMM / linear
        {"GemmMma fp16 cfg0",
         MirKey{MirKind::GemmMma, 0, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1 | 4, 0},
         "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
        {"GemmMma fp16 fast_accum",
         MirKey{MirKind::GemmMma, 0, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1 | 4 | 16, 0},
         "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"},
        {"GemmMma bf16 batched",
         MirKey{MirKind::GemmMma, 2, Dtype::BF16, Dtype::BF16, 1, Dtype::BF16, 32, 0},
         "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
        {"GemmSplitkReduce fp16 swiglu",
         MirKey{MirKind::GemmSplitkReduce, EPI_SWIGLU, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1, 0},
         ".entry"},
        {"GemmSkinny fp16 nt=2",
         MirKey{MirKind::GemmSkinny, 2, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1, 0},
         "createpolicy.fractional.L2::evict_first.b64"},
        {"LinearBwd dx bf16",
         MirKey{MirKind::LinearBwd, 0, Dtype::BF16, Dtype::BF16, 1, Dtype::FP32, 0, 0},
         "bar.sync"},
        {"GemmMmaNn bf16 cfg0",
         MirKey{MirKind::GemmMmaNn, 0, Dtype::BF16, Dtype::BF16, 1, Dtype::BF16, 1, 0},
         "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},

        // Wave 2: Quantized MMA
        {"GemmQuantMma INT8 bf16",
         MirKey{MirKind::GemmQuantMma, QF_INT8, Dtype::BF16, Dtype::FP32, 1, Dtype::FP32, 1 | 2, 0},
         "mma.sync"},
        {"GemmQuantMma Q8_0 fp16",
         MirKey{MirKind::GemmQuantMma, QF_Q8_0, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "cp.async"},
        {"GemmQuantMma Q4K fp16",
         MirKey{MirKind::GemmQuantMma, QF_Q4K, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 1, 0},
         "cp.async"},
        {"GemmQuantMma Q6K fp16",
         MirKey{MirKind::GemmQuantMma, QF_Q6K, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "mma.sync"},
        {"Conv2dInt8wMma cfg0",
         MirKey{MirKind::Conv2dInt8wMma, 0, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1 | 2, 0},
         "mma.sync"},
        {"Int8QuantAux conv fallback",
         MirKey{MirKind::Int8QuantAux, 1, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1, 0},
         ".entry"},

        // Wave 3: Conv2d / ConvAux
        {"Conv2dMma fp16 3x3p1s1",
         MirKey{MirKind::Conv2dMma, 0, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1 | 2, 0},
         "mma.sync"},
        {"Conv2dIgemmF32",
         MirKey{MirKind::Conv2dIgemmF32, 0, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 1 | 2, 0},
         ".entry"},
        {"ConvAux bwd_weight",
         MirKey{MirKind::ConvAux, 0, Dtype::FP16, Dtype::FP16, 1, Dtype::FP32, 0, 0},
         ".entry"},

        // Wave 4: Attention / FlashAttention
        {"DecodeGqa partials bf16",
         MirKey{MirKind::DecodeGqa, 0, Dtype::BF16, Dtype::BF16, 1, Dtype::FP32, 0, 0},
         "bar.sync"},
        {"FaFusedMma fp16 hd=64 causal",
         MirKey{MirKind::FaFusedMma, 64, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16, 1, 0},
         "mma.sync"},
        {"FaPackedMma bf16",
         MirKey{MirKind::FaPackedMma, 0, Dtype::BF16, Dtype::BF16, 1, Dtype::BF16, 0, 0},
         "mma.sync"},
        {"FaVarlenF32",
         MirKey{MirKind::FaVarlenF32, 0, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "smem_dyn"},
        {"FaAux bw_ds_tile",
         MirKey{MirKind::FaAux, 8, Dtype::FP16, Dtype::FP16, 1, Dtype::FP32, 3, 0},
         ".entry"},
        {"FaBwd pass0 packed f16",
         MirKey{MirKind::FaBwd, 64u | (64u << 16), Dtype::FP16, Dtype::FP16, 128, Dtype::FP32,
                static_cast<std::uint8_t>(FA_MODE_PACKED | 0x10u), fa_bwd_aux},
         "bar.sync"},
        {"FaRows causal split",
         MirKey{MirKind::FaRows, 64, Dtype::BF16, Dtype::BF16, 2, Dtype::FP32,
                static_cast<std::uint8_t>(FA_MODE_ROWS | (1u << 3) | (1u << 4)), 0},
         "shfl.sync"},
    };

    for (const Case& c : cases) {
        const std::string ptx = emit_verified_ptx(c.name, c.key);
        if (c.expect_sub && c.expect_sub[0] != '\0') {
            check(ptx.find(c.expect_sub) != std::string::npos,
                  std::string(c.name) + ": expected PTX substring '" + c.expect_sub + "'");
        }
    }

    if (g_failures > 0) {
        std::printf("\nFAILED (%d check(s))\n", g_failures);
        return 1;
    }
    std::printf("\nAll %zu GPU-less CUDA MIR -> PTX verification checks passed.\n", cases.size() + 1);
    return 0;
}
