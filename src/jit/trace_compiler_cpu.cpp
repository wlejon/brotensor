#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"
#include "kernel_disk_cache.h"
#include <brotensor/trace_autodiff.h>
#include "../cpu/cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/codegen/kernel_jit.hpp>
#include <brass/codegen/trace_builder.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/types.hpp>
#endif

#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace brotensor::jit {

FusionPattern TraceCompiler::classify_dag(const TraceDAG& dag) {
    const auto& nodes = dag.nodes();

    // Check for Fused Residual + RMSNorm: in-place Add followed by RMSNorm consuming it
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::RMSNorm && !n.inputs.empty()) {
            int in_id = n.inputs[0];
            if (in_id >= 0 && in_id < static_cast<int>(nodes.size())) {
                const auto& prev = nodes[in_id];
                if (prev.op == TraceOpKind::Add && prev.is_inplace) {
                    return FusionPattern::ResidualRMSNorm;
                }
            }
        }
    }

    // Check for Fused LayerNorm + Modulate: LayerNorm followed by Modulate / Affine
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::LayerNorm) {
            for (int out_id : n.outputs) {
                if (out_id >= 0 && out_id < static_cast<int>(nodes.size())) {
                    if (nodes[out_id].op == TraceOpKind::Modulate || nodes[out_id].op == TraceOpKind::Mul) {
                        return FusionPattern::LayerNormModulate;
                    }
                }
            }
        }
    }

    return FusionPattern::Elementwise;
}

static CudaTraceCompilerFn s_cuda_trace_compiler_fn = nullptr;

CudaTraceCompilerFn get_cuda_trace_compiler_hook() {
    return s_cuda_trace_compiler_fn;
}

void register_cuda_trace_compiler(CudaTraceCompilerFn fn) {
    s_cuda_trace_compiler_fn = fn;
}

static VulkanTraceCompilerFn s_vulkan_trace_compiler_fn = nullptr;

VulkanTraceCompilerFn get_vulkan_trace_compiler_hook() {
    return s_vulkan_trace_compiler_fn;
}

void register_vulkan_trace_compiler(VulkanTraceCompilerFn fn) {
    s_vulkan_trace_compiler_fn = fn;
}

TraceHandle TraceCompiler::compile_and_cache(const TraceDAG& raw_dag) {
    bool has_ext_ops = false;
    for (const auto& n : raw_dag.nodes()) {
        if (n.op == TraceOpKind::Const || n.op == TraceOpKind::Neg ||
            n.op == TraceOpKind::Abs || n.op == TraceOpKind::Exp ||
            n.op == TraceOpKind::Log || n.op == TraceOpKind::Sqrt ||
            n.op == TraceOpKind::Rsqrt) {
            has_ext_ops = true;
            break;
        }
    }
    TraceDAG lowered_holder;
    if (has_ext_ops) {
        lowered_holder = canonicalize_trace_dag_for_codegen(raw_dag);
    }
    const TraceDAG& dag = has_ext_ops ? lowered_holder : raw_dag;
    uint64_t hash = dag.compute_hash();

    // Check trace cache for previously compiled kernel
    auto cached = TraceCache::instance().lookup(hash);
    if (cached && cached->rebind_fn) {
        std::vector<void*> input_ptrs;
        std::vector<void*> output_ptrs;
        for (const auto& node : dag.nodes()) {
            if (node.op == TraceOpKind::Input) {
                input_ptrs.push_back(node.buffer);
            }
            if (node.is_live_at_end) {
                output_ptrs.push_back(node.buffer);
            }
        }
        auto re_bound = cached->rebind_fn(input_ptrs, output_ptrs);
        re_bound->signature_hash = hash;
        re_bound->is_cache_hit = true;
        re_bound->node_count = dag.node_count();
        re_bound->execute();
        return TraceHandle(re_bound);
    }

    FusionPattern pattern = classify_dag(dag);

    // Which compiler gets the DAG. CUDA has the PTX compiler and Vulkan the
    // SPIR-V one (brass MIR); a Vulkan DAG must never reach the CPU compiler
    // (host code over device addresses, unordered against the device
    // stream), so what the Vulkan compiler cannot fuse is replayed op by op
    // instead. Everything else is the CPU compiler's.
    bool is_cuda = false;
    bool is_vulkan = false;
    for (const auto& n : dag.nodes()) {
        if (n.device.is_cuda()) is_cuda = true;
        if (n.device.is_vulkan()) is_vulkan = true;
    }

    const auto compile_t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TraceHandleImpl> handle;
    if (is_cuda) {
        auto cuda_fn = get_cuda_trace_compiler_hook();
        if (!cuda_fn) {
            throw std::runtime_error("brotensor::jit: CUDA JIT compiler hook is not registered or CUDA is unavailable");
        }
        handle = cuda_fn(dag, pattern);
    } else if (is_vulkan) {
        if (auto vk_fn = get_vulkan_trace_compiler_hook()) handle = vk_fn(dag, pattern);
        if (!handle) handle = eager::compile_eager(dag);
    } else {
        handle = cpu::compile_cpu(dag, pattern);
    }

    handle->signature_hash = hash;
    handle->is_cuda = is_cuda;
    handle->node_count = dag.node_count();
    handle->is_cache_hit = false;
    // The CUDA and Vulkan compilers name their own fusion and time their own
    // builds, the eager replay counts its own launches; the CPU path is always
    // one fused call through brass's host codegen.
    if (handle->launch_count == 0) handle->launch_count = 1;
    if (!is_cuda && !is_vulkan) {
        if (handle->fusion_name == nullptr || handle->fusion_name[0] == '\0') {
            handle->fusion_name = "cpu-avx2-fused";
        }
        handle->compile_us = std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - compile_t0).count();
    }

    TraceCache::instance().insert(hash, handle);

    // Initial execution to populate outputs
    handle->execute();

    return TraceHandle(handle);
}

TraceHandle end_trace() {
    TraceDAG dag = TraceContext::current().end();
    return TraceCompiler::compile_and_cache(dag);
}

namespace cpu {

#if BROTENSOR_HAS_BRASS_JIT

using ElementwiseFn = void (*)(float* const* out_ptrs, const float* const* in_ptrs, int64_t n);

// What the host codegen below can actually be handed.
//
// It walks every operand as a flat run of `total_numel` FP32 values. A (1, D)
// broadcast row or a half-precision buffer would be read past the end of its
// allocation — a crash, or worse, silently wrong numbers — so anything that is
// not the trace's full shape in FP32 is refused here. end_trace() then throws
// and the caller falls back to its eager path, which is the contract the CUDA
// compiler already follows for expressions it cannot fuse.
static void require_cpu_fusable(const TraceDAG& dag) {
    int rows = 0, cols = 0;
    dag.dominant_shape(rows, cols);
    const int64_t dom_numel = static_cast<int64_t>(rows) * cols;
    for (const TraceNode& n : dag.nodes()) {
        if (n.dtype != Dtype::FP32) {
            throw std::runtime_error(
                "brotensor::jit: the CPU trace compiler is FP32 only");
        }
        const BroadcastKind bk = broadcast_of(n.rows, n.cols, rows, cols);
        const int64_t numel = static_cast<int64_t>(n.rows) * n.cols;
        if (bk == BroadcastKind::Full && numel != dom_numel) {
            throw std::runtime_error(
                "brotensor::jit: operand shape (" + std::to_string(n.rows) + ", " +
                std::to_string(n.cols) + ") cannot broadcast to dominant shape (" +
                std::to_string(rows) + ", " + std::to_string(cols) + ")");
        }
        if (n.is_live_at_end && bk != BroadcastKind::Full) {
            throw std::runtime_error(
                "brotensor::jit: a traced output must have the trace's full (rows, cols) shape");
        }
    }
}

// The emitted code dereferences every buffer on the host. CUDA device
// pointers and Vulkan buffer device addresses are not host addresses (and
// even where one happened to be readable, the host code would be unordered
// against the device stream that produces its inputs). Metal storage is
// MTLResourceStorageModeShared, whose contents pointer is a host address.
static void require_host_addressable(const TraceDAG& dag) {
    for (const TraceNode& n : dag.nodes()) {
        if (n.device.is_cuda() || n.device.is_vulkan()) {
            throw std::runtime_error(
                "brotensor::jit: the CPU trace compiler was handed a " +
                to_string(n.device) + " tensor; it only runs host-addressable memory");
        }
    }
}

static std::shared_ptr<TraceHandleImpl> compile_cpu_elementwise(const TraceDAG& dag) {
    require_cpu_fusable(dag);
    const auto& nodes = dag.nodes();

    int dom_rows = 0, dom_cols = 0;
    dag.dominant_shape(dom_rows, dom_cols);

    std::vector<int> input_node_ids;
    std::vector<BroadcastKind> input_bcast;
    std::vector<int> output_node_ids;
    std::unordered_map<int, int> node_to_in_idx;
    std::unordered_map<int, int> node_to_out_idx;

    bool any_bcast = false;
    bool any_row_bcast = false;
    bool any_col_bcast = false;
    bool has_where = false;

    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) {
            node_to_in_idx[n.id] = static_cast<int>(input_node_ids.size());
            input_node_ids.push_back(n.id);
            const BroadcastKind bk = broadcast_of(n.rows, n.cols, dom_rows, dom_cols);
            input_bcast.push_back(bk);
            if (bk != BroadcastKind::Full) any_bcast = true;
            if (bk == BroadcastKind::Row) any_row_bcast = true;
            if (bk == BroadcastKind::Col) any_col_bcast = true;
        }
        if (n.op == TraceOpKind::Where) {
            has_where = true;
        }
        if (n.is_live_at_end) {
            node_to_out_idx[n.id] = static_cast<int>(output_node_ids.size());
            output_node_ids.push_back(n.id);
        }
    }

    const int64_t total_numel = static_cast<int64_t>(dom_rows) * static_cast<int64_t>(dom_cols);
    const bool use_vec_loop = !any_row_bcast && !any_col_bcast && !has_where;

    using namespace brass;
    using namespace brass::codegen;

    auto build_fn = [&](Module& mod) {
    Function* fn = mod.create_function("trace_elementwise_kernel", Type::void_type(), {
        Type::ptr(), // float* const* out_ptrs
        Type::ptr(), // const float* const* in_ptrs
        Type::i64()  // int64_t n
    });

    TraceBuilder tb(mod, fn);

    BasicBlock* entry = tb.builder().append_block("entry");
    Value* p_out_ptrs = tb.builder().add_block_param(entry, Type::ptr());
    Value* p_in_ptrs = tb.builder().add_block_param(entry, Type::ptr());
    Value* n_val = tb.builder().add_block_param(entry, Type::i64());

    BasicBlock* shead = tb.builder().create_block("shead");
    BasicBlock* sbody = tb.builder().create_block("sbody");
    BasicBlock* exit_bb = tb.builder().create_block("exit");

    tb.position_at_end(entry);

    // Load input and output base pointers into local SSA values
    std::vector<Value*> in_ptrs_val;
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        Value* p = tb.builder().build_load(Type::ptr(), p_in_ptrs, static_cast<int32_t>(k * 8));
        in_ptrs_val.push_back(p);
    }
    std::vector<Value*> out_ptrs_val;
    for (size_t m = 0; m < output_node_ids.size(); ++m) {
        Value* p = tb.builder().build_load(Type::ptr(), p_out_ptrs, static_cast<int32_t>(m * 8));
        out_ptrs_val.push_back(p);
    }

    Value* zero_i64 = tb.builder().build_iconst_i64(0);
    Value* one_i64 = tb.builder().build_iconst_i64(1);

    if (use_vec_loop) {
        BasicBlock* vhead = tb.builder().create_block("vhead");
        BasicBlock* vbody = tb.builder().create_block("vbody");

        Value* two_i64 = tb.builder().build_iconst_i64(2);
        Value* eight_i64 = tb.builder().build_iconst_i64(8);
        Value* mask_eight = tb.builder().build_iconst_i64(~int64_t(7));
        Value* vec_n = tb.builder().build_and(n_val, mask_eight);
        Value* has_vec = tb.builder().build_sge(n_val, eight_i64);

        tb.builder().build_br_if(has_vec, vhead, {zero_i64}, shead, {zero_i64});

        // ── Vector AVX2 Loop (8-wide) ─────────────────────────────────────────
        fn->append_block(vhead);
        Value* iv_v = tb.builder().add_block_param(vhead, Type::i64());
        tb.position_at_end(vhead);
        Value* cond_v = tb.builder().build_slt(iv_v, vec_n);
        tb.builder().build_br_if(cond_v, vbody, {}, shead, {iv_v});

        fn->append_block(vbody);
        tb.position_at_end(vbody);
        Value* byte_off_v = tb.builder().build_shl(iv_v, two_i64);

        std::unordered_map<int, Value*> v_node_vals;
        for (size_t k = 0; k < input_node_ids.size(); ++k) {
            int nid = input_node_ids[k];
            if (input_bcast[k] == BroadcastKind::Scalar) {
                Value* sv = tb.load_f32(in_ptrs_val[k], 0);
                v_node_vals[nid] = tb.vbroadcast(Type::f32x8(), sv);
            } else {
                Value* addr = tb.builder().build_add(in_ptrs_val[k], byte_off_v);
                v_node_vals[nid] = tb.vload_f32x8(addr);
            }
        }

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::Input) continue;

            Value* res = nullptr;
            if (n.op == TraceOpKind::Add) {
                res = tb.vadd(v_node_vals[n.inputs[0]], v_node_vals[n.inputs[1]]);
            } else if (n.op == TraceOpKind::Sub) {
                res = tb.vsub(v_node_vals[n.inputs[0]], v_node_vals[n.inputs[1]]);
            } else if (n.op == TraceOpKind::Mul) {
                res = tb.vmul(v_node_vals[n.inputs[0]], v_node_vals[n.inputs[1]]);
            } else if (n.op == TraceOpKind::Div) {
                res = tb.vdiv(v_node_vals[n.inputs[0]], v_node_vals[n.inputs[1]]);
            } else if (n.op == TraceOpKind::AddScalar) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vadd(v_node_vals[n.inputs[0]], c);
            } else if (n.op == TraceOpKind::SubScalar) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vsub(v_node_vals[n.inputs[0]], c);
            } else if (n.op == TraceOpKind::MulScalar) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vmul(v_node_vals[n.inputs[0]], c);
            } else if (n.op == TraceOpKind::DivScalar) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vdiv(v_node_vals[n.inputs[0]], c);
            } else if (n.op == TraceOpKind::ScalarSub) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vsub(c, v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::ScalarDiv) {
                Value* c = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                res = tb.vdiv(c, v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::FMA) {
                res = tb.vfma(v_node_vals[n.inputs[0]], v_node_vals[n.inputs[1]], v_node_vals[n.inputs[2]]);
            } else if (n.op == TraceOpKind::Modulate) {
                Value* c1 = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(1.0f));
                res = tb.vfma(v_node_vals[n.inputs[0]], tb.vadd(v_node_vals[n.inputs[1]], c1), v_node_vals[n.inputs[2]]);
            } else if (n.op == TraceOpKind::Clamp) {
                Value* lo_v = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar));
                Value* hi_v = tb.vbroadcast(Type::f32x8(), tb.builder().build_fconst_f32(n.scalar2));
                res = tb.vmin(tb.vmax(v_node_vals[n.inputs[0]], lo_v), hi_v);
            } else if (n.op == TraceOpKind::SiLU) {
                res = tb.vsilu_f32x8(v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::GELU) {
                res = tb.vgelu_f32x8(v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::ReLU) {
                res = tb.vmax(v_node_vals[n.inputs[0]], tb.vzero(Type::f32x8()));
            } else if (n.op == TraceOpKind::Tanh) {
                res = tb.vtanh_f32x8(v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::Sigmoid) {
                res = tb.vsigmoid_f32x8(v_node_vals[n.inputs[0]]);
            } else if (n.op == TraceOpKind::Copy || n.op == TraceOpKind::Cast) {
                res = v_node_vals[n.inputs[0]];
            } else {
                throw std::runtime_error(
                    std::string("brotensor::jit: op '") + op_kind_name(n.op) +
                    "' has no CPU elementwise form");
            }
            v_node_vals[n.id] = res;

            if (n.is_live_at_end) {
                int out_idx = node_to_out_idx[n.id];
                Value* addr = tb.builder().build_add(out_ptrs_val[out_idx], byte_off_v);
                tb.vstore_f32x8(addr, res);
            }
        }

        Value* next_iv_v = tb.builder().build_add(iv_v, eight_i64);
        tb.builder().build_br(vhead, {next_iv_v});
    } else {
        tb.builder().build_br(shead, {zero_i64});
    }

    // ── Scalar Loop ──────────────────────────────────────────────────────────
    fn->append_block(shead);
    Value* iv_s = tb.builder().add_block_param(shead, Type::i64());
    tb.position_at_end(shead);
    Value* cond_s = tb.builder().build_slt(iv_s, n_val);
    tb.builder().build_br_if(cond_s, sbody, {}, exit_bb, {});

    fn->append_block(sbody);
    tb.position_at_end(sbody);

    Value* cols_i64 = (any_row_bcast || any_col_bcast)
                          ? tb.builder().build_iconst_i64(dom_cols)
                          : nullptr;
    Value* col_s = any_row_bcast ? tb.builder().build_umod(iv_s, cols_i64) : nullptr;
    Value* row_s = any_col_bcast ? tb.builder().build_udiv(iv_s, cols_i64) : nullptr;

    std::unordered_map<int, Value*> s_node_vals;
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        int nid = input_node_ids[k];
        if (input_bcast[k] == BroadcastKind::Scalar) {
            s_node_vals[nid] = tb.load_f32(in_ptrs_val[k], 0);
        } else if (input_bcast[k] == BroadcastKind::Row) {
            s_node_vals[nid] = tb.load_f32_indexed(in_ptrs_val[k], col_s, 4, 0);
        } else if (input_bcast[k] == BroadcastKind::Col) {
            s_node_vals[nid] = tb.load_f32_indexed(in_ptrs_val[k], row_s, 4, 0);
        } else {
            s_node_vals[nid] = tb.load_f32_indexed(in_ptrs_val[k], iv_s, 4, 0);
        }
    }

    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) continue;

        Value* res = nullptr;
        if (n.op == TraceOpKind::Add) {
            res = tb.add(s_node_vals[n.inputs[0]], s_node_vals[n.inputs[1]]);
        } else if (n.op == TraceOpKind::Sub) {
            res = tb.sub(s_node_vals[n.inputs[0]], s_node_vals[n.inputs[1]]);
        } else if (n.op == TraceOpKind::Mul) {
            res = tb.mul(s_node_vals[n.inputs[0]], s_node_vals[n.inputs[1]]);
        } else if (n.op == TraceOpKind::Div) {
            res = tb.div(s_node_vals[n.inputs[0]], s_node_vals[n.inputs[1]]);
        } else if (n.op == TraceOpKind::AddScalar) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.add(s_node_vals[n.inputs[0]], c);
        } else if (n.op == TraceOpKind::SubScalar) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.sub(s_node_vals[n.inputs[0]], c);
        } else if (n.op == TraceOpKind::MulScalar) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.mul(s_node_vals[n.inputs[0]], c);
        } else if (n.op == TraceOpKind::DivScalar) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.div(s_node_vals[n.inputs[0]], c);
        } else if (n.op == TraceOpKind::ScalarSub) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.sub(c, s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::ScalarDiv) {
            Value* c = tb.builder().build_fconst_f32(n.scalar);
            res = tb.div(c, s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::FMA) {
            res = tb.fma(s_node_vals[n.inputs[0]], s_node_vals[n.inputs[1]], s_node_vals[n.inputs[2]]);
        } else if (n.op == TraceOpKind::Modulate) {
            Value* c1 = tb.builder().build_fconst_f32(1.0f);
            res = tb.fma(s_node_vals[n.inputs[0]], tb.add(s_node_vals[n.inputs[1]], c1), s_node_vals[n.inputs[2]]);
        } else if (n.op == TraceOpKind::Clamp) {
            Value* lo_c = tb.builder().build_fconst_f32(n.scalar);
            Value* hi_c = tb.builder().build_fconst_f32(n.scalar2);
            res = tb.builder().build_fmin_f32(
                tb.builder().build_fmax_f32(s_node_vals[n.inputs[0]], lo_c), hi_c);
        } else if (n.op == TraceOpKind::Where) {
            Value* c_bits = tb.builder().build_bitcast_i32_f32(s_node_vals[n.inputs[0]]);
            Value* mag_bits = tb.builder().build_and(c_bits, tb.builder().build_iconst_i32(0x7FFFFFFF));
            Value* nz = tb.builder().build_ne(mag_bits, tb.builder().build_iconst_i32(0));
            res = tb.builder().build_select(nz, s_node_vals[n.inputs[1]], s_node_vals[n.inputs[2]]);
        } else if (n.op == TraceOpKind::SiLU) {
            res = tb.silu_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::GELU) {
            res = tb.gelu_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::ReLU) {
            res = tb.builder().build_fmax_f32(s_node_vals[n.inputs[0]], tb.builder().build_fconst_f32(0.0f));
        } else if (n.op == TraceOpKind::Tanh) {
            res = tb.tanh_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Sigmoid) {
            res = tb.sigmoid_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Copy || n.op == TraceOpKind::Cast) {
            res = s_node_vals[n.inputs[0]];
        } else {
            throw std::runtime_error(
                std::string("brotensor::jit: op '") + op_kind_name(n.op) +
                "' has no CPU elementwise form");
        }
        s_node_vals[n.id] = res;

        if (n.is_live_at_end) {
            int out_idx = node_to_out_idx[n.id];
            tb.store_f32_indexed(out_ptrs_val[out_idx], iv_s, res, 4, 0);
        }
    }

    Value* next_iv_s = tb.builder().build_add(iv_s, one_i64);
    tb.builder().build_br(shead, {next_iv_s});

    fn->append_block(exit_bb);
    tb.position_at_end(exit_bb);
    tb.builder().build_ret_void();
    };

    const std::string sig_key = "cpu_trace_ew_" + std::to_string(dag.compute_hash());
    KernelDiskCache::CpuCompileOptions c_opts;
    c_opts.enable_unroll = false;
    c_opts.enable_fp_reassociation = false;

    auto kernel_ptr = KernelDiskCache::instance().compile_or_load_cpu_fn<ElementwiseFn>(
        sig_key, "trace_elementwise_kernel", build_fn, c_opts);
    if (!kernel_ptr) {
        throw std::runtime_error("brotensor::jit: Failed to JIT compile CPU elementwise trace kernel");
    }

    auto make_bound_handle = [kernel_ptr, total_numel, any_bcast, input_node_ids, output_node_ids]
                             (const std::vector<void*>& in_buffers, const std::vector<void*>& out_buffers) {
        auto h = std::make_shared<TraceHandleImpl>();

        std::vector<const float*> in_f(in_buffers.size());
        for (size_t k = 0; k < in_buffers.size(); ++k) in_f[k] = static_cast<const float*>(in_buffers[k]);
        std::vector<float*> out_f(out_buffers.size());
        for (size_t m = 0; m < out_buffers.size(); ++m) out_f[m] = static_cast<float*>(out_buffers[m]);

        h->execute_fn = [kernel_ptr, total_numel, any_bcast, in_f, out_f]() {
            if (!any_bcast && total_numel >= 262144) {
                auto& pool = detail::cpu::ThreadPool::instance();
                int n_threads = pool.num_threads();
                int64_t chunk = (total_numel + n_threads - 1) / n_threads;
                chunk = (chunk + 7) & ~int64_t(7); // Align to 8-wide AVX2

                detail::cpu::parallel_for(n_threads, [&](std::size_t tid_sz) {
                    int64_t tid = static_cast<int64_t>(tid_sz);
                    int64_t start = tid * chunk;
                    if (start >= total_numel) return;
                    int64_t count = std::min(chunk, total_numel - start);

                    std::vector<const float*> chunk_in(in_f.size());
                    for (size_t k = 0; k < in_f.size(); ++k) {
                        chunk_in[k] = in_f[k] + start;
                    }
                    std::vector<float*> chunk_out(out_f.size());
                    for (size_t m = 0; m < out_f.size(); ++m) {
                        chunk_out[m] = out_f[m] + start;
                    }
                    kernel_ptr(chunk_out.data(), chunk_in.data(), count);
                });
            } else {
                kernel_ptr(out_f.data(), in_f.data(), total_numel);
            }
        };
        return h;
    };

    std::vector<void*> initial_in_buffers;
    for (int nid : input_node_ids) initial_in_buffers.push_back(nodes[nid].buffer);
    std::vector<void*> initial_out_buffers;
    for (int nid : output_node_ids) initial_out_buffers.push_back(nodes[nid].buffer);

    auto handle = make_bound_handle(initial_in_buffers, initial_out_buffers);
    handle->rebind_fn = [make_bound_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return make_bound_handle(in_b, out_b);
    };

    return handle;
}

std::shared_ptr<TraceHandleImpl> compile_cpu(const TraceDAG& dag, FusionPattern pattern) {
    const auto& nodes = dag.nodes();
    require_host_addressable(dag);

    for (const auto& n : nodes) {
        if (n.dtype != Dtype::FP32) {
            return eager::compile_eager(dag);
        }
    }

    auto in_idx_of = [&nodes](int node_id) -> int {
        int idx = 0;
        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::Input) {
                if (n.id == node_id) return idx;
                ++idx;
            }
        }
        return -1;
    };
    auto out_idx_of = [&nodes](int node_id) -> int {
        int idx = 0;
        for (const auto& n : nodes) {
            if (n.is_live_at_end) {
                if (n.id == node_id) return idx;
                ++idx;
            }
        }
        return -1;
    };

    if (pattern == FusionPattern::ResidualRMSNorm && nodes.size() == 5) {
        const TraceNode* node_rms = nullptr;
        const TraceNode* node_add = nullptr;
        const TraceNode* node_res = nullptr;
        const TraceNode* node_gamma = nullptr;

        for (const auto& n : nodes) {
            if (n.op != TraceOpKind::RMSNorm || n.inputs.empty()) continue;
            const TraceNode& add = nodes[n.inputs[0]];
            // The hand kernel below is exactly `x += res; rms_norm(x, gamma)`.
            // Anything else — no residual, no gain, or downstream ops — is not
            // this pattern.
            if (add.op != TraceOpKind::Add || !add.is_inplace || add.inputs.size() < 2) continue;
            if (n.inputs.size() < 2 || !n.outputs.empty() || !n.is_live_at_end || !add.is_live_at_end) continue;
            if (nodes[add.inputs[0]].op != TraceOpKind::Input ||
                nodes[add.inputs[1]].op != TraceOpKind::Input ||
                nodes[n.inputs[1]].op != TraceOpKind::Input) continue;
            node_rms = &n;
            node_add = &add;
            node_res = &nodes[add.inputs[1]];
            node_gamma = &nodes[n.inputs[1]];
            break;
        }

        if (node_rms && node_add && node_res && node_gamma) {
            int B = node_add->rows;
            int D = node_add->cols;
            float eps = node_rms->scalar > 0.0f ? node_rms->scalar : 1e-5f;

            auto make_handle = [B, D, eps](void* x_buf, const void* res_buf, const void* gamma_buf, void* y_buf) {
                auto h = std::make_shared<TraceHandleImpl>();
                h->execute_fn = [x_buf, res_buf, gamma_buf, y_buf, B, D, eps]() {
                    detail::cpu::jit::fused_residual_rmsnorm(
                        static_cast<float*>(x_buf),
                        static_cast<const float*>(res_buf),
                        static_cast<const float*>(gamma_buf),
                        eps,
                        static_cast<float*>(y_buf),
                        B, D
                    );
                };
                return h;
            };

            void* x_buf = node_add->buffer;
            const void* res_buf = node_res->buffer;
            const void* gamma_buf = node_gamma->buffer;
            void* y_buf = node_rms->buffer;

            const int in_res = in_idx_of(node_res->id);
            const int in_gamma = in_idx_of(node_gamma->id);
            const int out_x = out_idx_of(node_add->id);
            const int out_y = out_idx_of(node_rms->id);

            auto handle = make_handle(x_buf, res_buf, gamma_buf, y_buf);
            handle->rebind_fn = [make_handle, in_res, in_gamma, out_x, out_y](
                                    const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* res_p = (in_res >= 0 && static_cast<size_t>(in_res) < in_b.size()) ? in_b[in_res] : nullptr;
                const void* gamma_p = (in_gamma >= 0 && static_cast<size_t>(in_gamma) < in_b.size()) ? in_b[in_gamma] : nullptr;
                void* x_p = (out_x >= 0 && static_cast<size_t>(out_x) < out_b.size()) ? out_b[out_x] : nullptr;
                void* y_p = (out_y >= 0 && static_cast<size_t>(out_y) < out_b.size()) ? out_b[out_y] : nullptr;
                return make_handle(x_p, res_p, gamma_p, y_p);
            };
            return handle;
        }
    } else if (pattern == FusionPattern::LayerNormModulate) {
        const TraceNode* node_ln = nullptr;
        const TraceNode* node_x = nullptr;
        const TraceNode* node_gamma = nullptr;
        const TraceNode* node_beta = nullptr;
        const TraceNode* node_scale = nullptr;
        const TraceNode* node_shift = nullptr;
        const TraceNode* node_out = nullptr;

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::LayerNorm && !n.is_live_at_end && n.outputs.size() == 1 && n.inputs.size() >= 3) {
                if (nodes[n.inputs[0]].op == TraceOpKind::Input &&
                    nodes[n.inputs[1]].op == TraceOpKind::Input &&
                    nodes[n.inputs[2]].op == TraceOpKind::Input) {
                    node_ln = &n;
                    node_x = &nodes[n.inputs[0]];
                    node_gamma = &nodes[n.inputs[1]];
                    node_beta = &nodes[n.inputs[2]];
                }
            }
        }

        if (node_ln && nodes.size() == 7) {
            const TraceNode& mod = nodes[node_ln->outputs[0]];
            if (mod.op == TraceOpKind::Modulate && mod.is_live_at_end && mod.outputs.empty() &&
                mod.inputs.size() >= 3 && mod.inputs[0] == node_ln->id &&
                nodes[mod.inputs[1]].op == TraceOpKind::Input &&
                nodes[mod.inputs[2]].op == TraceOpKind::Input) {
                node_scale = &nodes[mod.inputs[1]];
                node_shift = &nodes[mod.inputs[2]];
                node_out = &mod;
            }
        } else if (node_ln && nodes.size() == 9) {
            const TraceNode& mul = nodes[node_ln->outputs[0]];
            if (mul.op == TraceOpKind::Mul && !mul.is_live_at_end && mul.outputs.size() == 1 && mul.inputs.size() == 2) {
                int other_mul_in = (mul.inputs[0] == node_ln->id) ? mul.inputs[1] : mul.inputs[0];
                const TraceNode& add_sc = nodes[other_mul_in];
                const TraceNode& add_sh = nodes[mul.outputs[0]];
                if (add_sc.op == TraceOpKind::AddScalar && !add_sc.is_live_at_end &&
                    add_sc.scalar == 1.0f && add_sc.inputs.size() == 1 &&
                    nodes[add_sc.inputs[0]].op == TraceOpKind::Input &&
                    add_sh.op == TraceOpKind::Add && add_sh.is_live_at_end &&
                    add_sh.outputs.empty() && add_sh.inputs.size() == 2) {
                    int shift_in = (add_sh.inputs[0] == mul.id) ? add_sh.inputs[1] : add_sh.inputs[0];
                    if (nodes[shift_in].op == TraceOpKind::Input) {
                        node_scale = &nodes[add_sc.inputs[0]];
                        node_shift = &nodes[shift_in];
                        node_out = &add_sh;
                    }
                }
            }
        }

        // The hand kernel dereferences all five operands, so a trace missing
        // any of them (a non-affine LayerNorm, a modulate with no shift) is
        // not this pattern and falls through to the generic path.
        if (node_ln && node_x && node_out && node_gamma && node_beta && node_scale && node_shift) {
            int R = node_ln->rows;
            int D = node_ln->cols;
            float eps = node_ln->scalar > 0.0f ? node_ln->scalar : 1e-5f;

            auto make_handle = [R, D, eps](const void* x, const void* g, const void* b,
                                           const void* scale, const void* shift, void* y) {
                auto h = std::make_shared<TraceHandleImpl>();
                h->execute_fn = [x, g, b, scale, shift, y, R, D, eps]() {
                    detail::cpu::jit::fused_layernorm_modulate(
                        static_cast<const float*>(x),
                        static_cast<const float*>(g),
                        static_cast<const float*>(b),
                        static_cast<const float*>(scale),
                        static_cast<const float*>(shift),
                        eps,
                        static_cast<float*>(y),
                        R, D
                    );
                };
                return h;
            };

            const void* x_buf = node_x->buffer;
            const void* g_buf = node_gamma->buffer;
            const void* b_buf = node_beta->buffer;
            const void* s_buf = node_scale->buffer;
            const void* sh_buf = node_shift->buffer;
            void* y_buf = node_out->buffer;

            const int in_x = in_idx_of(node_x->id);
            const int in_g = in_idx_of(node_gamma->id);
            const int in_b_idx = in_idx_of(node_beta->id);
            const int in_s = in_idx_of(node_scale->id);
            const int in_sh = in_idx_of(node_shift->id);
            const int out_y = out_idx_of(node_out->id);

            auto handle = make_handle(x_buf, g_buf, b_buf, s_buf, sh_buf, y_buf);
            handle->rebind_fn = [make_handle, in_x, in_g, in_b_idx, in_s, in_sh, out_y](
                                    const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* x = (in_x >= 0 && static_cast<size_t>(in_x) < in_b.size()) ? in_b[in_x] : nullptr;
                const void* g = (in_g >= 0 && static_cast<size_t>(in_g) < in_b.size()) ? in_b[in_g] : nullptr;
                const void* b = (in_b_idx >= 0 && static_cast<size_t>(in_b_idx) < in_b.size()) ? in_b[in_b_idx] : nullptr;
                const void* sc = (in_s >= 0 && static_cast<size_t>(in_s) < in_b.size()) ? in_b[in_s] : nullptr;
                const void* sh = (in_sh >= 0 && static_cast<size_t>(in_sh) < in_b.size()) ? in_b[in_sh] : nullptr;
                void* y = (out_y >= 0 && static_cast<size_t>(out_y) < out_b.size()) ? out_b[out_y] : nullptr;
                return make_handle(x, g, b, sc, sh, y);
            };
            return handle;
        }
    }

    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) {
            return eager::compile_eager(dag);
        }
    }

    return compile_cpu_elementwise(dag);
}

#else // !BROTENSOR_HAS_BRASS_JIT

std::shared_ptr<TraceHandleImpl> compile_cpu(const TraceDAG& /*dag*/, FusionPattern /*pattern*/) {
    throw std::runtime_error("brotensor::jit: Brass CPU JIT compiler is not enabled in this build");
}

#endif // BROTENSOR_HAS_BRASS_JIT

} // namespace cpu
} // namespace brotensor::jit
