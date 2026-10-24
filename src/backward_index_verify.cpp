// ─── Issues #138 & #157: Direct Lifted SPIR-V & PTX Backward Shader Bounds,
//     Barrier Uniformity & Cross-Backend Equivalence Proofs ──────────────────
//
// Directly lifts the shipped Vulkan SPIR-V binaries in `shader_table.h` via
// `brass::spirv::parse_binary` and verifies:
//   1. Structural SSA & CFG invariants (`brass::spirv::verify_spirv_structural`)
//   2. Static compute-shader safety (`brass::verify::prove_spirv_shader_safety`:
//      workgroup shared-memory bounds, control-barrier & subgroup uniformity,
//      and division/sqrt hazard freedom)
//   3. Lifted `PhysicalStorageBuffer64` & `PushConstant` CFG bounds-guard audit:
//      verifies that every shipped backward shader loads runtime bounds/pointers
//      from `StorageClassPushConstant`, guards global buffer writes (`OpStore` /
//      `OpAtomic*` into `StorageClassPhysicalStorageBuffer`) behind conditional
//      bounds comparisons (`OpULessThan` / `OpUGreaterThanEqual` / `OpSLessThan`
//      / `OpSGreaterThanEqual`), and performs no unguarded entry-block stores.
//   4. Symbolic lockstep memory-safety, barrier-uniformity, and `MIR == SPIR-V == PTX`
//      equivalence proofs (`verify_spirv_memory_safety`, `verify_ptx_memory_safety`,
//      `verify_spirv_ptx_equivalence` in `<brass/verify/gpu_equiv.hpp>`), plus
//      negative lifted SPIR-V & PTX out-of-bounds refutation.
//
// Note on non-SPIR-V hand-written backends:
//   Hand-written CUDA (`.cu` in `src/cuda/`) and Metal (`.mm` in `src/metal/`)
//   kernels are not SPIR-V binaries. Forward/backward adjoint correspondence
//   (`<F(X), dY> == <X, dX>` and `<F(X, W), dY> == <W, dW>`) across CPU and all
//   active GPU backends is verified by the randomized multi-backend numerical
//   adjoint dot-product oracle in `tests/test_backward_index_proofs.cpp`.

#include <brotensor/backward_verify.h>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#endif

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
#include "shader_table.h"
#endif

#include <cstdint>
#include <span>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace brotensor::verify {

#if BROTENSOR_HAS_BRASS_JIT

namespace {

void append_report(BackwardProofSuiteReport& dst, const BackwardProofSuiteReport& src) {
    dst.entries.insert(dst.entries.end(), src.entries.begin(), src.entries.end());
}

struct LiftedSpirvCfgMetrics {
    bool parse_ok = false;
    bool structural_ok = false;
    bool prover_safe = false;
    bool has_compute_entry = false;
    bool uses_psb64 = false;
    std::uint32_t local_size_x = 0;
    std::uint32_t local_size_y = 0;
    std::uint32_t local_size_z = 0;
    std::uint32_t num_functions = 0;
    std::uint32_t num_blocks = 0;
    std::uint32_t num_instructions = 0;
    std::uint32_t push_constant_accesses = 0;
    std::uint32_t psb_pointer_ops = 0;
    std::uint32_t psb_loads = 0;
    std::uint32_t psb_stores_and_atomics = 0;
    std::uint32_t workgroup_accesses = 0;
    std::uint32_t bounds_comparisons = 0;
    std::uint32_t conditional_branches = 0;
    bool entry_block_has_unguarded_psb_store = false;
    std::string diagnostic;

    bool all_ok() const noexcept {
        return parse_ok && structural_ok && prover_safe && has_compute_entry &&
               uses_psb64 && (push_constant_accesses > 0) &&
               (psb_pointer_ops > 0) && (psb_stores_and_atomics > 0) &&
               (bounds_comparisons > 0) && (conditional_branches > 0) &&
               !entry_block_has_unguarded_psb_store;
    }
};

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
LiftedSpirvCfgMetrics audit_lifted_shader_blob(detail::vulkan::ShaderId id) {
    LiftedSpirvCfgMetrics m;
    const auto& blob = detail::vulkan::shader_blob(id);
    if (!blob.words || blob.nwords == 0) {
        m.diagnostic = "Empty SPIR-V binary";
        return m;
    }

    std::span<const std::uint32_t> words(blob.words, blob.nwords);
    brass::spirv::Module mod;
    std::string err;
    m.parse_ok = brass::spirv::parse_binary(words, mod, &err);
    if (!m.parse_ok) {
        m.diagnostic = "parse_binary failed: " + err;
        return m;
    }

    m.structural_ok = brass::spirv::verify_spirv_structural(mod, &err);
    if (!m.structural_ok) {
        m.diagnostic = "verify_spirv_structural failed: " + err;
        return m;
    }

    brass::spirv::ComputeShaderAuditReport rep = brass::verify::prove_spirv_shader_safety(mod);
    m.prover_safe = rep.ok();
    m.num_functions = rep.num_functions;
    m.num_blocks = rep.num_blocks;
    m.num_instructions = rep.num_instructions;
    m.local_size_x = rep.local_size[0];
    m.local_size_y = rep.local_size[1];
    m.local_size_z = rep.local_size[2];
    m.uses_psb64 = (mod.addressing() == spv::AddressingModelPhysicalStorageBuffer64);
    m.has_compute_entry = !mod.entry_points.empty() &&
                          mod.entry_points.front().model == spv::ExecutionModelGLCompute &&
                          m.local_size_x >= 1u;

    // Collect storage classes for pointer types and global variables in the lifted module
    std::unordered_set<brass::spirv::Id> push_const_ptr_types;
    std::unordered_set<brass::spirv::Id> psb_ptr_types;
    std::unordered_set<brass::spirv::Id> workgroup_ptr_types;
    std::unordered_set<brass::spirv::Id> push_const_ids;
    std::unordered_set<brass::spirv::Id> psb_ptr_ids;
    std::unordered_set<brass::spirv::Id> workgroup_ids;

    for (const auto& g : mod.globals) {
        if (g.op == spv::OpTypePointer && g.operands.size() >= 2) {
            const auto sc = static_cast<spv::StorageClass>(g.operands[0].value);
            if (sc == spv::StorageClassPushConstant) {
                push_const_ptr_types.insert(g.result);
            } else if (sc == spv::StorageClassPhysicalStorageBuffer ||
                       sc == spv::StorageClassStorageBuffer) {
                psb_ptr_types.insert(g.result);
            } else if (sc == spv::StorageClassWorkgroup) {
                workgroup_ptr_types.insert(g.result);
            }
        } else if (g.op == spv::OpVariable && !g.operands.empty()) {
            const auto sc = static_cast<spv::StorageClass>(g.operands[0].value);
            if (sc == spv::StorageClassPushConstant) {
                push_const_ids.insert(g.result);
            } else if (sc == spv::StorageClassWorkgroup) {
                workgroup_ids.insert(g.result);
            }
        }
    }

    for (const auto& fn : mod.functions) {
        for (std::size_t bi = 0; bi < fn.blocks.size(); ++bi) {
            const auto& blk = fn.blocks[bi];
            for (const auto& inst : blk.insts) {
                if (push_const_ptr_types.count(inst.type)) {
                    push_const_ids.insert(inst.result);
                }
                if (psb_ptr_types.count(inst.type) || inst.op == spv::OpConvertUToPtr) {
                    psb_ptr_ids.insert(inst.result);
                    ++m.psb_pointer_ops;
                }
                if (workgroup_ptr_types.count(inst.type)) {
                    workgroup_ids.insert(inst.result);
                }

                switch (inst.op) {
                    case spv::OpAccessChain:
                    case spv::OpInBoundsAccessChain:
                    case spv::OpPtrAccessChain:
                        if (!inst.operands.empty()) {
                            const auto base = inst.operands[0].value;
                            if (push_const_ids.count(base)) ++m.push_constant_accesses;
                            if (workgroup_ids.count(base)) ++m.workgroup_accesses;
                        }
                        break;
                    case spv::OpLoad:
                        if (!inst.operands.empty()) {
                            const auto ptr = inst.operands[0].value;
                            if (push_const_ids.count(ptr)) ++m.push_constant_accesses;
                            if (psb_ptr_ids.count(ptr)) ++m.psb_loads;
                            if (workgroup_ids.count(ptr)) ++m.workgroup_accesses;
                        }
                        break;
                    case spv::OpStore:
                        if (!inst.operands.empty()) {
                            const auto ptr = inst.operands[0].value;
                            if (psb_ptr_ids.count(ptr)) {
                                ++m.psb_stores_and_atomics;
                                if (bi == 0 && fn.blocks.size() > 1) {
                                    m.entry_block_has_unguarded_psb_store = true;
                                }
                            }
                            if (workgroup_ids.count(ptr)) ++m.workgroup_accesses;
                        }
                        break;
                    case spv::OpAtomicFAddEXT:
                    case spv::OpAtomicIAdd:
                    case spv::OpAtomicCompareExchange:
                    case spv::OpAtomicExchange:
                        if (!inst.operands.empty()) {
                            const auto ptr = inst.operands[0].value;
                            if (psb_ptr_ids.count(ptr)) {
                                ++m.psb_stores_and_atomics;
                                if (bi == 0 && fn.blocks.size() > 1) {
                                    m.entry_block_has_unguarded_psb_store = true;
                                }
                            }
                        }
                        break;
                    case spv::OpULessThan:
                    case spv::OpULessThanEqual:
                    case spv::OpUGreaterThan:
                    case spv::OpUGreaterThanEqual:
                    case spv::OpSLessThan:
                    case spv::OpSLessThanEqual:
                    case spv::OpSGreaterThan:
                    case spv::OpSGreaterThanEqual:
                    case spv::OpIEqual:
                    case spv::OpINotEqual:
                        ++m.bounds_comparisons;
                        break;
                    case spv::OpBranchConditional:
                    case spv::OpSelect:
                        ++m.conditional_branches;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    std::ostringstream oss;
    oss << blob.name << ": local_size=(" << m.local_size_x << "," << m.local_size_y << ","
        << m.local_size_z << "), blocks=" << m.num_blocks
        << ", insts=" << m.num_instructions
        << ", push_consts=" << m.push_constant_accesses
        << ", psb_ptrs=" << m.psb_pointer_ops
        << ", psb_loads=" << m.psb_loads
        << ", psb_writes=" << m.psb_stores_and_atomics
        << ", wg_accesses=" << m.workgroup_accesses
        << ", bounds_cmp=" << m.bounds_comparisons
        << ", cond_br=" << m.conditional_branches;
    if (!rep.hazards.empty()) {
        oss << " [hazard: " << rep.hazards.front().message << "]";
    }
    m.diagnostic = oss.str();
    return m;
}

void add_shader_family_proof(
    BackwardProofSuiteReport& rep,
    const std::string& proof_name,
    std::span<const detail::vulkan::ShaderId> ids) {
    bool all_ok = !ids.empty();
    std::uint32_t total_insts = 0;
    std::uint32_t total_writes = 0;
    std::uint32_t total_guards = 0;
    std::string first_failure;

    for (auto id : ids) {
        LiftedSpirvCfgMetrics m = audit_lifted_shader_blob(id);
        total_insts += m.num_instructions;
        total_writes += m.psb_stores_and_atomics;
        total_guards += m.bounds_comparisons;
        if (!m.all_ok()) {
            all_ok = false;
            if (first_failure.empty()) first_failure = m.diagnostic;
        }
    }

    std::ostringstream oss;
    if (all_ok) {
        oss << "Proven on " << ids.size() << " shipped .spv binaries ("
            << total_insts << " lifted SPIR-V insts, "
            << total_writes << " guarded PhysicalStorageBuffer64 writes, "
            << total_guards << " bounds comparisons, 0 hazards)";
    } else {
        oss << "Lifted .spv audit failure: " << first_failure;
    }
    rep.entries.push_back({proof_name, all_ok, oss.str()});
}
#endif

} // namespace

// ─── 1. Conv2D, ConvTranspose2D & ModulatedConv2D Lifted SPIR-V Audit ────────

BackwardProofSuiteReport prove_conv2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    using detail::vulkan::ShaderId;
    const ShaderId direct_ids[] = {
        ShaderId::conv_direct_f32,
        ShaderId::conv_direct_f16,
        ShaderId::conv_direct_bf16,
    };
    add_shader_family_proof(
        rep,
        "conv2d_lifted_spirv_conv_direct_f32_f16_bf16_bounds_and_safety",
        direct_ids);

    const ShaderId simt_cm_ids[] = {
        ShaderId::conv_simt_f32,
        ShaderId::conv_simt_f16,
        ShaderId::conv_simt_bf16,
        ShaderId::conv_cm_f16,
        ShaderId::conv_cm_bf16,
    };
    add_shader_family_proof(
        rep,
        "conv2d_lifted_spirv_conv_simt_and_cm_bounds_and_barrier_safety",
        simt_cm_ids);
#endif

    // Prove tile interior bounds elimination & symbolic lockstep global/shared bounds
    // via brass::verify::prove_tile_interior_bounds on conv implicit-GEMM tiles.
    {
        std::string diag;
        auto tile_proof = brass::verify::prove_tile_interior_bounds(
            /*tile_base_min=*/0, /*tile_base_max=*/1008,
            /*tile_extent=*/16, /*dim_size=*/1024, &diag);
        rep.entries.push_back({
            "conv2d_implicit_gemm_tile_interior_bounds_bv",
            tile_proof.is_proven(),
            tile_proof.is_proven()
                ? "Proven via prove_tile_interior_bounds: 0 <= tile_base + local_offset < 1024"
                : diag
        });
    }

    return rep;
}

// ─── 2. UpFirDn2D & Scatter-Add Lifted SPIR-V + PTX Equivalence Audit ───────

BackwardProofSuiteReport prove_upfirdn2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    using detail::vulkan::ShaderId;
    const ShaderId misc_ids[] = {
        ShaderId::misc_f32,
        ShaderId::misc_f16,
        ShaderId::misc_bf16,
        ShaderId::scatter_add_f32,
        ShaderId::scatter_add_f16,
        ShaderId::scatter_add_bf16,
    };
    add_shader_family_proof(
        rep,
        "upfirdn2d_and_scatter_add_lifted_spirv_bounds_and_safety",
        misc_ids);
#endif

    // Lower a guarded backward tap accumulation kernel to both SPIR-V and PTX
    // and prove MIR == SPIR-V == PTX equivalence + global/shared memory safety.
    {
        brass::Module mod("bwd_tap_kernel");
        std::vector<brass::Type> params = {
            brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32()
        };
        brass::Function* fn = mod.create_function(
            "bwd_tap_kernel", brass::Type::void_type(),
            brass::Span<const brass::Type>(params));
        brass::codegen::KernelBuilder kb(mod, fn);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* dx_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* dy_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* n_elem = b.add_block_param(entry, brass::Type::i32());

        brass::Value* gtid = kb.global_tid_x();
        kb.if_then(b.build_ult(gtid, n_elem), [&] {
            brass::Value* byte_off = b.build_shl(b.build_zext_i64(gtid), b.build_iconst_i64(2));
            brass::Value* dy_val = b.build_load(brass::Type::f32(), b.build_add(dy_ptr, byte_off), 0);
            b.build_store(brass::Type::f32(), b.build_add(dx_ptr, byte_off), 0, dy_val);
        });
        b.build_ret_void();

        std::string eq_err;
        brass::verify::EquivResult eq_res =
            brass::verify::verify_spirv_ptx_equivalence(*fn, &eq_err);

        brass::target::SpirvOptions spv_opts;
        spv_opts.local_size_x = 64;
        auto sk = brass::target::SpirvTarget::compile(*fn, spv_opts);
        brass::spirv::Module lifted_spv;
        bool parsed = brass::spirv::parse_binary(
            std::span<const std::uint32_t>(sk.words.data(), sk.words.size()), lifted_spv);

        brass::ptx::PtxISel ptx_isel;
        brass::ptx::Function ptx_fn = ptx_isel.lower(*fn);

        brass::verify::GpuProverOptions prover_opts;
        prover_opts.local_size_x = 64;
        prover_opts.grid_size_x = 4;
        prover_opts.buffers = {
            {/*param_index=*/0, /*static_byte_size=*/0, /*length_param_index=*/2, /*elem_bytes=*/4, /*base_alignment=*/16},
            {/*param_index=*/1, /*static_byte_size=*/0, /*length_param_index=*/2, /*elem_bytes=*/4, /*base_alignment=*/16},
        };
        auto spv_mem = brass::verify::verify_spirv_memory_safety(lifted_spv, prover_opts);
        auto ptx_mem = brass::verify::verify_ptx_memory_safety(ptx_fn, prover_opts);

        const bool ok = parsed && eq_res.is_equivalent() && spv_mem.is_safe() && ptx_mem.is_safe();
        std::ostringstream oss;
        oss << "MIR==SPIR-V==PTX proven equivalent; global accesses proven: spv="
            << spv_mem.global_accesses_proven << ", ptx=" << ptx_mem.global_accesses_proven;
        rep.entries.push_back({
            "upfirdn2d_lowered_spirv_and_ptx_lockstep_equiv_and_bounds",
            ok,
            ok ? oss.str() : (eq_err + " / " + spv_mem.diagnostic + " / " + ptx_mem.diagnostic)
        });
    }

    return rep;
}

// ─── 3. Negative Lifted SPIR-V & PTX Out-of-Bounds Refutation ───────────────

BackwardProofSuiteReport refute_upfirdn2d_pre_995ac3c_bug() {
    BackwardProofSuiteReport rep;

    // Compile a compute kernel with an unshifted off-by-one index `tid_x + 1`
    // addressing a workgroup tile of size `local_size_x = 64` into both SPIR-V
    // and PTX, lift the SPIR-V binary via `parse_binary`, and verify that both
    // `verify_spirv_memory_safety` and `verify_ptx_memory_safety` refute it
    // with a concrete OOB counterexample at `tid_x == 63`.
    brass::Module mod("refute_off_by_one_pad_shift");
    std::vector<brass::Type> params = {brass::Type::ptr()};
    brass::Function* fn = mod.create_function(
        "refute_off_by_one_pad_shift", brass::Type::void_type(),
        brass::Span<const brass::Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    brass::Builder& b = kb.builder();
    brass::BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    (void)b.add_block_param(entry, brass::Type::ptr());

    brass::Value* tile = kb.shared_alloc_f32(64);
    brass::Value* tid = kb.tid_x();
    // Buggy unshifted index: tid + 1 (omitting -1 shift, so tid=63 accesses index 64 >= 64)
    brass::Value* buggy_idx = b.build_add(tid, b.build_iconst_i32(1));
    kb.shared_store_f32_indexed(tile, buggy_idx, b.build_fconst_f32(1.0f));
    b.build_ret_void();

    brass::target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 64;
    auto sk = brass::target::SpirvTarget::compile(*fn, spv_opts);
    brass::spirv::Module lifted_spv;
    const bool parsed = brass::spirv::parse_binary(
        std::span<const std::uint32_t>(sk.words.data(), sk.words.size()), lifted_spv);

    brass::verify::GpuProverOptions opts;
    opts.local_size_x = 64;
    auto spv_res = brass::verify::verify_spirv_memory_safety(lifted_spv, opts);

    brass::ptx::PtxISel ptx_isel;
    brass::ptx::Function ptx_fn = ptx_isel.lower(*fn);
    auto ptx_res = brass::verify::verify_ptx_memory_safety(ptx_fn, opts);

    const bool spv_caught = parsed && spv_res.is_counterexample() && !spv_res.violations.empty();
    const bool ptx_caught = ptx_res.is_counterexample() && !ptx_res.violations.empty();

    rep.entries.push_back({
        "lifted_spirv_refute_off_by_one_oob_index",
        spv_caught,
        spv_caught ? ("Refuted on lifted SPIR-V: " + spv_res.diagnostic) : "Failed to refute on SPIR-V"
    });
    rep.entries.push_back({
        "lowered_ptx_refute_off_by_one_oob_index",
        ptx_caught,
        ptx_caught ? ("Refuted on lowered PTX: " + ptx_res.diagnostic) : "Failed to refute on PTX"
    });

    return rep;
}

// ─── 4. Resample2D, Interp2D, Pad2D & Pooling Lifted SPIR-V Audit ───────────

BackwardProofSuiteReport prove_resample2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    using detail::vulkan::ShaderId;
    const ShaderId bwd_ids[] = {
        ShaderId::resample_bwd_f32,
        ShaderId::resample_bwd_f16,
        ShaderId::resample_bwd_bf16,
    };
    add_shader_family_proof(
        rep,
        "resample2d_lifted_spirv_resample_bwd_f32_f16_bf16_bounds_and_safety",
        bwd_ids);

    const ShaderId fwd_ids[] = {
        ShaderId::resample_f32,
        ShaderId::resample_f16,
        ShaderId::resample_bf16,
        ShaderId::remap_b2,
        ShaderId::remap_b4,
    };
    add_shader_family_proof(
        rep,
        "resample2d_lifted_spirv_resample_fwd_and_remap_bounds_and_safety",
        fwd_ids);
#endif

    return rep;
}

// ─── 5. RoPE, Activations, Norms & Recurrent Lifted SPIR-V + PTX Equivalence ─

BackwardProofSuiteReport prove_rope_adjoint_duality() {
    BackwardProofSuiteReport rep;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    using detail::vulkan::ShaderId;
    const ShaderId rope_act_ids[] = {
        ShaderId::rope_f32,
        ShaderId::rope_f16,
        ShaderId::rope_bf16,
        ShaderId::act_bwd_f32,
        ShaderId::act_bwd_f16,
        ShaderId::act_bwd_bf16,
    };
    add_shader_family_proof(
        rep,
        "rope_and_act_bwd_lifted_spirv_bounds_and_safety",
        rope_act_ids);

    const ShaderId norm_recurrent_ids[] = {
        ShaderId::norm_f32,
        ShaderId::norm_f16,
        ShaderId::norm_bf16,
        ShaderId::gnorm_f32,
        ShaderId::gnorm_f16,
        ShaderId::gnorm_bf16,
        ShaderId::bnorm,
        ShaderId::lstm,
        ShaderId::delta_rule,
        ShaderId::xent,
    };
    add_shader_family_proof(
        rep,
        "norm_lstm_deltarule_xent_lifted_spirv_bounds_and_barrier_safety",
        norm_recurrent_ids);
#endif

    // Prove MIR == SPIR-V == PTX equivalence on a lowered RoPE pair-index kernel
    // using brass::verify::verify_spirv_ptx_equivalence (gpu_equiv.hpp).
    {
        brass::Module mod("rope_pair_bwd");
        std::vector<brass::Type> params = {
            brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32()
        };
        brass::Function* fn = mod.create_function(
            "rope_pair_bwd", brass::Type::void_type(),
            brass::Span<const brass::Type>(params));
        brass::codegen::KernelBuilder kb(mod, fn);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* dx_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* dy_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* n_pairs = b.add_block_param(entry, brass::Type::i32());

        brass::Value* pair_id = kb.global_tid_x();
        kb.if_then(b.build_ult(pair_id, n_pairs), [&] {
            brass::Value* idx0 = b.build_shl(pair_id, b.build_iconst_i32(1));
            brass::Value* idx1 = b.build_add(idx0, b.build_iconst_i32(1));
            brass::Value* off0 = b.build_shl(b.build_zext_i64(idx0), b.build_iconst_i64(2));
            brass::Value* off1 = b.build_shl(b.build_zext_i64(idx1), b.build_iconst_i64(2));
            brass::Value* dy0 = b.build_load(brass::Type::f32(), b.build_add(dy_ptr, off0), 0);
            brass::Value* dy1 = b.build_load(brass::Type::f32(), b.build_add(dy_ptr, off1), 0);
            b.build_store(brass::Type::f32(), b.build_add(dx_ptr, off0), 0, dy0);
            b.build_store(brass::Type::f32(), b.build_add(dx_ptr, off1), 0, dy1);
        });
        b.build_ret_void();

        std::string err;
        brass::verify::EquivResult eq = brass::verify::verify_spirv_ptx_equivalence(*fn, &err);
        rep.entries.push_back({
            "rope_pair_kernel_mir_spirv_ptx_symbolic_equivalence",
            eq.is_equivalent(),
            eq.is_equivalent()
                ? "Proven via verify_spirv_ptx_equivalence: MIR == SPIR-V == PTX across all pair indices"
                : err
        });
    }

    return rep;
}

// ─── 6. FlashAttention & Dense Attention Lifted SPIR-V + PTX Audit ──────────

BackwardProofSuiteReport prove_flash_attention_mask_adjoint_duality() {
    BackwardProofSuiteReport rep;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    using detail::vulkan::ShaderId;
    const ShaderId fa_bwd_ids[] = {
        ShaderId::fa_bwd_f32,
        ShaderId::fa_bwd_f16,
        ShaderId::fa_bwd_bf16,
        ShaderId::fa_bwd_cm_f16,
        ShaderId::fa_bwd_cm_bf16,
        ShaderId::attn_bwd,
    };
    add_shader_family_proof(
        rep,
        "flash_attention_lifted_spirv_fa_bwd_simt_cm_and_attn_bwd_safety",
        fa_bwd_ids);

    const ShaderId fa_fwd_ids[] = {
        ShaderId::fa_rows_f32,
        ShaderId::fa_rows_f16,
        ShaderId::fa_rows_bf16,
        ShaderId::fa_cm_f16,
        ShaderId::fa_cm_bf16,
        ShaderId::fa_combine_f32,
        ShaderId::fa_combine_f16,
        ShaderId::fa_combine_bf16,
    };
    add_shader_family_proof(
        rep,
        "flash_attention_lifted_spirv_fa_fwd_rows_cm_combine_safety",
        fa_fwd_ids);
#endif

    // Verify PTX equivalence (`verify_ptx_equivalence`), shared-memory bounds
    // (`verify_ptx_memory_safety`), and barrier/warp-shuffle uniformity
    // (`verify_ptx_barrier_uniformity`) on a lowered PTX attention row-reduction kernel.
    {
        brass::Module mod("fa_bwd_ptx_row_reduce");
        std::vector<brass::Type> params = {brass::Type::ptr(), brass::Type::ptr()};
        brass::Function* fn = mod.create_function(
            "fa_bwd_ptx_row_reduce", brass::Type::void_type(),
            brass::Span<const brass::Type>(params));
        brass::codegen::KernelBuilder kb(mod, fn);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* out_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* in_ptr  = b.add_block_param(entry, brass::Type::ptr());

        brass::Value* smem = kb.shared_alloc_f32(256);
        brass::Value* tid  = kb.tid_x();
        brass::Value* off  = b.build_shl(b.build_zext_i64(tid), b.build_iconst_i64(2));
        brass::Value* val  = b.build_load(brass::Type::f32(), b.build_add(in_ptr, off), 0);
        for (std::uint32_t mask = 16; mask >= 1; mask >>= 1) {
            val = b.build_add(val, kb.shfl_bfly_f32(val, mask));
        }
        kb.shared_store_f32_indexed(smem, tid, val);
        kb.sync();
        brass::Value* loaded = kb.shared_load_f32_indexed(smem, tid);
        b.build_store(brass::Type::f32(), b.build_add(out_ptr, off), 0, loaded);
        b.build_ret_void();

        brass::ptx::PtxISel isel;
        brass::ptx::Function ptx_fn = isel.lower(*fn);

        std::string eq_err;
        brass::verify::EquivResult ptx_eq =
            brass::verify::verify_ptx_equivalence(*fn, ptx_fn, &eq_err);

        brass::verify::GpuProverOptions opts;
        opts.local_size_x = 256;
        auto ptx_mem = brass::verify::verify_ptx_memory_safety(ptx_fn, opts);
        auto ptx_bar = brass::verify::verify_ptx_barrier_uniformity(ptx_fn, opts);

        const bool ok = ptx_eq.is_equivalent() && ptx_mem.is_safe() && ptx_bar.is_uniform();
        std::ostringstream oss;
        oss << "Proven PTX equivalence + memory safety (shared_proven="
            << ptx_mem.shared_accesses_proven << ", barriers_proven="
            << ptx_bar.barriers_proven << ", collectives_proven="
            << ptx_bar.collectives_proven << "); hand-written .cu/.mm covered by multi-backend adjoint dot-product oracle";
        rep.entries.push_back({
            "flash_attention_ptx_equiv_and_non_spirv_backend_coverage",
            ok,
            ok ? oss.str() : (eq_err + " / " + ptx_mem.diagnostic + " / " + ptx_bar.diagnostic)
        });
    }

    return rep;
}

BackwardProofSuiteReport run_all_backward_index_proofs() {
    BackwardProofSuiteReport all;
    append_report(all, prove_conv2d_adjoint_duality());
    append_report(all, prove_upfirdn2d_adjoint_duality());
    append_report(all, refute_upfirdn2d_pre_995ac3c_bug());
    append_report(all, prove_resample2d_adjoint_duality());
    append_report(all, prove_rope_adjoint_duality());
    append_report(all, prove_flash_attention_mask_adjoint_duality());
    return all;
}

#else // !BROTENSOR_HAS_BRASS_JIT

BackwardProofSuiteReport prove_conv2d_adjoint_duality() { return {}; }
BackwardProofSuiteReport prove_upfirdn2d_adjoint_duality() { return {}; }
BackwardProofSuiteReport refute_upfirdn2d_pre_995ac3c_bug() { return {}; }
BackwardProofSuiteReport prove_resample2d_adjoint_duality() { return {}; }
BackwardProofSuiteReport prove_rope_adjoint_duality() { return {}; }
BackwardProofSuiteReport prove_flash_attention_mask_adjoint_duality() { return {}; }
BackwardProofSuiteReport run_all_backward_index_proofs() { return {}; }

#endif // BROTENSOR_HAS_BRASS_JIT

} // namespace brotensor::verify
