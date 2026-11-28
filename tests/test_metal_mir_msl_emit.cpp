// GPU-less verification of brotensor's Metal Brass MIR -> MSL pipeline.
// Builds representative kernels across core Metal MirKinds plus all newly
// migrated MirKinds (ConvAux, FilteredLrelu, GemmCm, ConvCm), runs Brass GPU
// MIR optimization, lowers via MslTarget::compile, checks syntactic brace
// balance, verifies simdgroup_matrix emission for coopmat kernels, and checks
// static threadgroup allocation sizes against Metal's 32 KiB limit.

#include "../src/metal/mtl_mir_builders.h"
#include "../src/vulkan/shaders/op_codes.h"

#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Dtype;
using namespace brotensor::detail::metal;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

bool braces_balanced(const std::string& s) {
    int cur = 0, par = 0, sq = 0;
    for (char ch : s) {
        if (ch == '{') ++cur;
        else if (ch == '}') { if (--cur < 0) return false; }
        else if (ch == '(') ++par;
        else if (ch == ')') { if (--par < 0) return false; }
        else if (ch == '[') ++sq;
        else if (ch == ']') { if (--sq < 0) return false; }
    }
    return cur == 0 && par == 0 && sq == 0;
}

std::size_t static_threadgroup_bytes(const std::string& src) {
    std::size_t total = 0;
    std::size_t pos = 0;
    const std::string tag = "threadgroup ";
    while ((pos = src.find(tag, pos)) != std::string::npos) {
        pos += tag.size();
        const std::size_t semi = src.find(';', pos);
        const std::size_t lbr = src.find('[', pos);
        const std::size_t rbr = (lbr != std::string::npos) ? src.find(']', lbr + 1) : std::string::npos;
        if (semi == std::string::npos || lbr == std::string::npos || rbr == std::string::npos || rbr > semi) {
            continue;
        }
        const std::string decl = src.substr(pos, lbr - pos);
        const std::string count_str = src.substr(lbr + 1, rbr - lbr - 1);
        char* endp = nullptr;
        const unsigned long count = std::strtoul(count_str.c_str(), &endp, 10);
        if (endp == count_str.c_str() || count == 0) continue;
        std::size_t elem_bytes = 4;
        if (decl.find("int4") != std::string::npos || decl.find("float4") != std::string::npos) {
            elem_bytes = 16;
        } else if (decl.find("int64_t") != std::string::npos || decl.find("double") != std::string::npos) {
            elem_bytes = 8;
        } else if (decl.find("uint16_t") != std::string::npos || decl.find("half") != std::string::npos ||
                   decl.find("bfloat") != std::string::npos) {
            elem_bytes = 2;
        }
        total += static_cast<std::size_t>(count) * elem_bytes;
    }
    return total;
}

std::string emit_verified_msl(const std::string& label, const MirKey& k) {
    brass::Module mod("mtl_mir_test");
    brass::Function* fn = build_metal_mir_function(mod, k);
    check(fn != nullptr, label + ": build_metal_mir_function returned null");
    if (!fn) return {};

    brass::GpuOptOptions gopt;
    gopt.opt_level = 2;
    gopt.enable_fma = true;
    gopt.enable_gpu_superopt = true;
    gopt.enable_small_loop_unroll = true;
    gopt.enable_load_coalesce = true;
    brass::optimize_gpu_function(*fn, mod, gopt);

    brass::target::MslOptions opts;
    opts.workgroup_x = metal_mir_block_size(k);
    brass::target::MslKernel msl = brass::target::MslTarget::compile(*fn, opts);

    check(msl.source.find("#include <metal_stdlib>") != std::string::npos,
          label + ": missing #include <metal_stdlib>");
    check(msl.source.find("kernel void ") != std::string::npos,
          label + ": missing kernel void");
    check(braces_balanced(msl.source), label + ": unbalanced brackets/braces in MSL");
    const std::size_t tg_bytes = static_threadgroup_bytes(msl.source);
    check(tg_bytes <= 32768u,
          label + ": static threadgroup memory " + std::to_string(tg_bytes) + " > 32768");

    std::printf("  [PASS] %-38s (%zu bytes MSL, %zu B tgmem)\n",
                label.c_str(), msl.source.size(), tg_bytes);
    return msl.source;
}

}  // namespace

int main() {
    std::printf("--- Metal Brass MIR -> MSL verification ---\n");

    struct Case {
        const char* name;
        MirKey key;
        const char* expect_sub;
    };

    const std::uint64_t fl_aux =
        std::uint64_t{2} | (std::uint64_t{2} << 8) | (std::uint64_t{2} << 16) |
        (std::uint64_t{2} << 24) | (std::uint64_t{3} << 32) | (std::uint64_t{3} << 40);
    const std::uint64_t conv_aux =
        std::uint64_t{3} | (std::uint64_t{3} << 8) | (std::uint64_t{1} << 16) |
        (std::uint64_t{1} << 24) | (std::uint64_t{1} << 32) | (std::uint64_t{1} << 40) |
        (std::uint64_t{1} << 48) | (std::uint64_t{1} << 56);
    const std::uint64_t fa_bwd_aux = std::uint64_t{32} | (std::uint64_t{16} << 16);

    const std::vector<Case> cases = {
        {"Unary silu f16x8",
         MirKey{MirKind::Unary, UOP_SILU, Dtype::FP16, Dtype::FP16, 8, Dtype::FP32, 0, 0},
         "kernel void"},
        {"Norm col_sum f16",
         MirKey{MirKind::Norm, NORM_COL_SUM, Dtype::FP16, Dtype::FP16, 1, Dtype::FP16,
                pack_norm_flags(true, false, false, false, false), 0},
         "threadgroup_barrier"},
        {"Gnorm apply silu bf16",
         MirKey{MirKind::Gnorm, GN_APPLY, Dtype::BF16, Dtype::BF16, 4, Dtype::FP32, 1, 0},
         "kernel void"},
        {"Gemv fp32xbf16 m=4",
         MirKey{MirKind::Gemv, 4u, Dtype::FP32, Dtype::BF16, 1, Dtype::FP32, 3, 0},
         "kernel void"},
        {"GemmSimt fp32xfp16 big128",
         MirKey{MirKind::GemmSimt, static_cast<std::uint32_t>(EPI_STORE | (static_cast<int>(Dtype::FP32) << 8)),
                Dtype::FP32, Dtype::FP16, 1, Dtype::FP32, 1 | 4 | 8, 0},
         "threadgroup_barrier"},
        {"ConvSimt fp32 3x3",
         MirKey{MirKind::ConvSimt, 0u, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, 1 | 4, conv_aux},
         "threadgroup_barrier"},
        {"ConvAux bwd_weight_blk fp16",
         MirKey{MirKind::ConvAux, 0u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "kernel void"},
        {"ConvAux bwd_weight_thr bf16",
         MirKey{MirKind::ConvAux, 1u, Dtype::BF16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "kernel void"},
        {"ConvAux conv3d_int8w",
         MirKey{MirKind::ConvAux, 3u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 1, 0},
         "kernel void"},
        {"ConvAux resblock_add_shift bf16",
         MirKey{MirKind::ConvAux, 5u, Dtype::BF16, Dtype::FP32, 1, Dtype::FP32, 1, 0},
         "kernel void"},
        {"ConvAux resblock_sum_hw fp16",
         MirKey{MirKind::ConvAux, 6u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "threadgroup_barrier"},
        {"ConvAux splitk_fold_accum bf16",
         MirKey{MirKind::ConvAux, 7u, Dtype::BF16, Dtype::FP32, 1, Dtype::FP32, 0, 0},
         "kernel void"},
        {"FilteredLrelu polyu fp16",
         MirKey{MirKind::FilteredLrelu, 0u, Dtype::FP16, Dtype::FP16, 1, Dtype::FP32, 0, 0},
         "kernel void"},
        {"FilteredLrelu polyd bf16",
         MirKey{MirKind::FilteredLrelu, 1u, Dtype::BF16, Dtype::BF16, 1, Dtype::FP32, 0, 0},
         "kernel void"},
        {"FilteredLrelu main fp16",
         MirKey{MirKind::FilteredLrelu, 2u, Dtype::FP16, Dtype::FP16, 1, Dtype::FP32, 3, fl_aux},
         "threadgroup_barrier"},
        {"GemmCm fp16 cfg2 128x64 vec",
         MirKey{MirKind::GemmCm, 2u | (EPI_STORE << 4), Dtype::FP16, Dtype::FP16, 1, Dtype::FP16,
                4u | 16u, 0},
         "simdgroup_multiply_accumulate("},
        {"GemmCm fp16 cfg3 64x64 swiglu",
         MirKey{MirKind::GemmCm, 3u | (EPI_SWIGLU << 4), Dtype::FP16, Dtype::FP16, 1, Dtype::FP32,
                2u | 4u | 16u, 0},
         "simdgroup_multiply_accumulate("},
        {"ConvCm fp16 cfg2 bvec2 rowch",
         MirKey{MirKind::ConvCm, 2u | (2u << 8), Dtype::FP16, Dtype::FP32, 1, Dtype::FP32,
                1u | 4u | 8u, conv_aux},
         "simdgroup_multiply_accumulate("},
        {"ConvCm fp16 cfg3 scalar",
         MirKey{MirKind::ConvCm, 3u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32,
                1u | 4u, conv_aux},
         "simdgroup_multiply_accumulate("},
        {"FaRows fp16 causal",
         MirKey{MirKind::FaRows, 64u, Dtype::FP16, Dtype::FP16, 2, Dtype::FP32,
                static_cast<std::uint8_t>(FA_MODE_ROWS | (1u << 3) | (1u << 4)), 0},
         "kernel void"},
        {"FaBwd pass0 f16",
         MirKey{MirKind::FaBwd, 64u | (64u << 16), Dtype::FP16, Dtype::FP16, 128, Dtype::FP32,
                static_cast<std::uint8_t>(FA_MODE_PACKED | 0x10u), fa_bwd_aux},
         "threadgroup_barrier"},
    };

    for (const Case& c : cases) {
        const std::string msl = emit_verified_msl(c.name, c.key);
        if (c.expect_sub && c.expect_sub[0] != '\0') {
            check(msl.find(c.expect_sub) != std::string::npos,
                  std::string(c.name) + ": expected MSL substring '" + c.expect_sub + "'");
        }
    }

    if (g_failures > 0) {
        std::printf("\nFAILED (%d check(s))\n", g_failures);
        return 1;
    }
    std::printf("\nAll %zu GPU-less Metal MIR -> MSL verification checks passed.\n", cases.size());
    return 0;
}
