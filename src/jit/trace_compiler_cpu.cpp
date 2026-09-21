#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"
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

TraceHandle TraceCompiler::compile_and_cache(const TraceDAG& dag) {
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

    bool is_cuda = false;
    for (const auto& n : dag.nodes()) {
        if (n.device.is_cuda()) {
            is_cuda = true;
            break;
        }
    }

    const auto compile_t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TraceHandleImpl> handle;
    if (is_cuda) {
        auto cuda_fn = get_cuda_trace_compiler_hook();
        if (!cuda_fn) {
            throw std::runtime_error("brotensor::jit: CUDA JIT compiler hook is not registered or CUDA is unavailable");
        }
        handle = cuda_fn(dag, pattern);
    } else {
        handle = cpu::compile_cpu(dag, pattern);
    }

    handle->signature_hash = hash;
    handle->is_cuda = is_cuda;
    handle->node_count = dag.node_count();
    handle->is_cache_hit = false;
    // The CUDA compiler names its own fusion and times its own PTX build; the
    // CPU path is always one fused call through brass's host codegen.
    if (handle->launch_count == 0) handle->launch_count = 1;
    if (!is_cuda) {
        handle->fusion_name = "cpu-avx2-fused";
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

static std::shared_ptr<TraceHandleImpl> compile_cpu_elementwise(const TraceDAG& dag) {
    const auto& nodes = dag.nodes();

    std::vector<int> input_node_ids;
    std::vector<int> output_node_ids;
    std::unordered_map<int, int> node_to_in_idx;
    std::unordered_map<int, int> node_to_out_idx;

    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) {
            node_to_in_idx[n.id] = static_cast<int>(input_node_ids.size());
            input_node_ids.push_back(n.id);
        }
        if (n.is_live_at_end) {
            node_to_out_idx[n.id] = static_cast<int>(output_node_ids.size());
            output_node_ids.push_back(n.id);
        }
    }

    int64_t total_numel = 0;
    if (!output_node_ids.empty()) {
        const auto& out_node = nodes[output_node_ids[0]];
        total_numel = static_cast<int64_t>(out_node.rows) * static_cast<int64_t>(out_node.cols);
    } else if (!input_node_ids.empty()) {
        const auto& in_node = nodes[input_node_ids[0]];
        total_numel = static_cast<int64_t>(in_node.rows) * static_cast<int64_t>(in_node.cols);
    }

    using namespace brass;
    using namespace brass::codegen;

    static uint64_t kernel_id_counter = 0;
    std::string mod_name = "mod_trace_cpu_" + std::to_string(++kernel_id_counter);
    Module mod(mod_name);

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

    BasicBlock* vhead = tb.builder().create_block("vhead");
    BasicBlock* vbody = tb.builder().create_block("vbody");
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
    Value* two_i64 = tb.builder().build_iconst_i64(2);
    Value* eight_i64 = tb.builder().build_iconst_i64(8);
    Value* mask_eight = tb.builder().build_iconst_i64(~int64_t(7));
    Value* vec_n = tb.builder().build_and(n_val, mask_eight);
    Value* has_vec = tb.builder().build_sge(n_val, eight_i64);

    tb.builder().build_br_if(has_vec, vhead, {zero_i64}, shead, {zero_i64});

    // ── Vector AVX2 Loop (8-wide) ─────────────────────────────────────────────
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
        Value* addr = tb.builder().build_add(in_ptrs_val[k], byte_off_v);
        v_node_vals[nid] = tb.vload_f32x8(addr);
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
        } else if (n.op == TraceOpKind::SiLU) {
            res = tb.vsilu_f32x8(v_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::GELU) {
            res = tb.vgelu_f32x8(v_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::ReLU) {
            res = tb.vrelu_f32x8(v_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Tanh) {
            res = tb.vtanh_f32x8(v_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Sigmoid) {
            res = tb.vsigmoid_f32x8(v_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Copy) {
            // The node exists so the store lands in the caller's buffer; the
            // value is already the one to write.
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

    // ── Scalar Remainder Loop ────────────────────────────────────────────────
    fn->append_block(shead);
    Value* iv_s = tb.builder().add_block_param(shead, Type::i64());
    tb.position_at_end(shead);
    Value* cond_s = tb.builder().build_slt(iv_s, n_val);
    tb.builder().build_br_if(cond_s, sbody, {}, exit_bb, {});

    fn->append_block(sbody);
    tb.position_at_end(sbody);

    std::unordered_map<int, Value*> s_node_vals;
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        int nid = input_node_ids[k];
        s_node_vals[nid] = tb.load_f32_indexed(in_ptrs_val[k], iv_s, 4, 0);
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
        } else if (n.op == TraceOpKind::SiLU) {
            res = tb.silu_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::GELU) {
            res = tb.gelu_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::ReLU) {
            res = tb.relu_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Tanh) {
            res = tb.tanh_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Sigmoid) {
            res = tb.sigmoid_f32(s_node_vals[n.inputs[0]]);
        } else if (n.op == TraceOpKind::Copy) {
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

    KernelOptions opts;
    opts.enable_optimizations = true;
    opts.enable_avx2 = true;
    opts.enable_fma = true;
    opts.enable_vectorize = true;

    auto jit = std::make_shared<KernelJit>(opts);
    KernelFunction kfn = jit->compile(*fn);
    if (!kfn.is_valid()) {
        throw std::runtime_error("brotensor::jit: Failed to JIT compile CPU elementwise trace kernel");
    }

    auto kernel_ptr = kfn.as<ElementwiseFn>();
    auto engine_ref = kfn.engine();

    auto make_bound_handle = [kernel_ptr, engine_ref, total_numel, input_node_ids, output_node_ids]
                             (const std::vector<void*>& in_buffers, const std::vector<void*>& out_buffers) {
        auto h = std::make_shared<TraceHandleImpl>();

        std::vector<const float*> in_f(in_buffers.size());
        for (size_t k = 0; k < in_buffers.size(); ++k) in_f[k] = static_cast<const float*>(in_buffers[k]);
        std::vector<float*> out_f(out_buffers.size());
        for (size_t m = 0; m < out_buffers.size(); ++m) out_f[m] = static_cast<float*>(out_buffers[m]);

        h->execute_fn = [kernel_ptr, engine_ref, total_numel, in_f, out_f]() {
            if (total_numel >= 262144) {
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

    if (pattern == FusionPattern::ResidualRMSNorm) {
        const TraceNode* node_rms = nullptr;
        const TraceNode* node_add = nullptr;
        const TraceNode* node_res = nullptr;
        const TraceNode* node_gamma = nullptr;

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::RMSNorm) {
                node_rms = &n;
                int add_id = n.inputs[0];
                node_add = &nodes[add_id];
                int res_id = node_add->inputs[1];
                node_res = &nodes[res_id];
                if (n.inputs.size() > 1) {
                    node_gamma = &nodes[n.inputs[1]];
                }
                break;
            }
        }

        if (node_rms && node_add && node_res) {
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
            const void* gamma_buf = node_gamma ? node_gamma->buffer : nullptr;
            void* y_buf = node_rms->buffer;

            auto handle = make_handle(x_buf, res_buf, gamma_buf, y_buf);
            handle->rebind_fn = [make_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* res_p = in_b.size() > 1 ? in_b[1] : nullptr;
                const void* gamma_p = in_b.size() > 2 ? in_b[2] : nullptr;
                void* x_p = out_b.size() > 0 ? out_b[0] : in_b[0];
                void* y_p = out_b.size() > 1 ? out_b[1] : (out_b.size() > 0 ? out_b[0] : nullptr);
                return make_handle(x_p, res_p, gamma_p, y_p);
            };
            return handle;
        }
    } else if (pattern == FusionPattern::LayerNormModulate) {
        const TraceNode* node_ln = nullptr;
        const TraceNode* node_gamma = nullptr;
        const TraceNode* node_beta = nullptr;
        const TraceNode* node_scale = nullptr;
        const TraceNode* node_shift = nullptr;
        const TraceNode* node_out = nullptr;

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::LayerNorm) {
                node_ln = &n;
                if (n.inputs.size() > 1) node_gamma = &nodes[n.inputs[1]];
                if (n.inputs.size() > 2) node_beta = &nodes[n.inputs[2]];
            }
            if (n.is_live_at_end && n.op != TraceOpKind::LayerNorm) {
                node_out = &n;
            }
            if (n.op == TraceOpKind::Modulate) {
                if (n.inputs.size() > 1) node_scale = &nodes[n.inputs[1]];
                if (n.inputs.size() > 2) node_shift = &nodes[n.inputs[2]];
            }
        }

        // Find scale/shift from expression ln * (1 + scale) + shift if not directly Modulate op
        if (!node_scale || !node_shift) {
            for (const auto& n : nodes) {
                if (n.op == TraceOpKind::AddScalar && n.inputs.size() > 0) {
                    node_scale = &nodes[n.inputs[0]];
                }
                if (n.op == TraceOpKind::Add && n.inputs.size() > 1) {
                    int rhs = n.inputs[1];
                    if (nodes[rhs].op == TraceOpKind::Input) {
                        node_shift = &nodes[rhs];
                    }
                }
            }
        }

        if (node_ln && node_out) {
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

            const void* x_buf = nodes[node_ln->inputs[0]].buffer;
            const void* g_buf = node_gamma ? node_gamma->buffer : nullptr;
            const void* b_buf = node_beta ? node_beta->buffer : nullptr;
            const void* s_buf = node_scale ? node_scale->buffer : nullptr;
            const void* sh_buf = node_shift ? node_shift->buffer : nullptr;
            void* y_buf = node_out->buffer;

            auto handle = make_handle(x_buf, g_buf, b_buf, s_buf, sh_buf, y_buf);
            handle->rebind_fn = [make_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* x = in_b.size() > 0 ? in_b[0] : nullptr;
                const void* g = in_b.size() > 1 ? in_b[1] : nullptr;
                const void* b = in_b.size() > 2 ? in_b[2] : nullptr;
                const void* sc = in_b.size() > 3 ? in_b[3] : nullptr;
                const void* sh = in_b.size() > 4 ? in_b[4] : nullptr;
                void* y = out_b.size() > 0 ? out_b[0] : nullptr;
                return make_handle(x, g, b, sc, sh, y);
            };
            return handle;
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
