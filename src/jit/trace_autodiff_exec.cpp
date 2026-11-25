#include <brotensor/trace_autodiff.h>
#include <brotensor/ops/reduction.h>
#include <brotensor/runtime.h>
#include "trace_dag.h"
#include "trace_plan.h"
#include "trace_compiler.h"
#include "spirv_emit.h"
#include "gpu_mir_emit_detail.h"

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/smt_equiv.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit {

namespace {

struct DominantMeta {
    Device dev = Device::CPU;
    Dtype dt = Dtype::FP32;
    int rows = 0;
    int cols = 0;
};

DominantMeta find_dominant_meta(const TraceDAG& dag) {
    DominantMeta m;
    int best = -1;
    for (const auto& n : dag.nodes()) {
        m.dev = n.device;
        if (n.dtype == Dtype::FP32) m.dt = n.dtype;
        int area = n.rows * n.cols;
        if (area > best) {
            best = area;
            m.rows = n.rows;
            m.cols = n.cols;
        }
    }
    return m;
}

#if BROTENSOR_HAS_BRASS_JIT
size_t count_mir_insts(const brass::Function& fn) {
    size_t n = 0;
    for (const auto* bb : fn.blocks()) {
        for (const auto* inst : *bb) { (void)inst; ++n; }
    }
    return n;
}

brass::Function* build_scalar_backward_mir_fn(brass::Module& mod, const char* name, const TraceDAG& lowered_dag) {
    std::vector<int> input_ids, output_ids;
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) input_ids.push_back(n.id);
        if (n.is_live_at_end) output_ids.push_back(n.id);
    }
    std::vector<brass::Type> params(input_ids.size(), brass::Type::f32());
    auto* fn = mod.create_function(name, brass::Type::f32(), brass::Span<const brass::Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    gpu_mir::Ctx c(kb);
    std::vector<brass::Value*> param_vals = gpu_mir::entry_params(c, fn);
    std::unordered_map<int, brass::Value*> vals;
    for (size_t i = 0; i < input_ids.size(); ++i) vals[input_ids[i]] = param_vals[i];
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) continue;
        std::vector<brass::Value*> ins;
        for (int in_id : n.inputs) ins.push_back(vals.at(in_id));
        vals[n.id] = gpu_mir::lane_op(c, n, ins);
    }
    brass::Value* ret_val = output_ids.empty() ? kb.const_f32(0.0f) : vals.at(output_ids.front());
    for (size_t i = 1; i < output_ids.size(); ++i) ret_val = c.b.build_add(ret_val, vals.at(output_ids[i]));
    c.b.build_ret(ret_val);
    return fn;
}

brass::Function* build_gpu_equiv_kernel_fn(brass::Module& mod, const TraceDAG& lowered_dag) {
    std::vector<int> input_ids, output_ids;
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) input_ids.push_back(n.id);
        if (n.is_live_at_end) output_ids.push_back(n.id);
    }
    std::vector<brass::Type> params(input_ids.size() + output_ids.size(), brass::Type::ptr());
    auto* fn = mod.create_function("bwd_trace_gpu_equiv", brass::Type::void_type(), brass::Span<const brass::Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    gpu_mir::Ctx c(kb);
    std::vector<brass::Value*> buf_params = gpu_mir::entry_params(c, fn);
    brass::Value* tid = kb.global_tid_x();
    std::unordered_map<int, brass::Value*> vals;
    for (size_t i = 0; i < input_ids.size(); ++i) vals[input_ids[i]] = kb.load_f32_indexed(buf_params[i], tid);
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) continue;
        std::vector<brass::Value*> ins;
        for (int in_id : n.inputs) ins.push_back(vals.at(in_id));
        vals[n.id] = gpu_mir::lane_op(c, n, ins);
    }
    for (size_t j = 0; j < output_ids.size(); ++j) {
        kb.store_f32_indexed(buf_params[input_ids.size() + j], tid, vals.at(output_ids[j]));
    }
    c.b.build_ret_void();
    return fn;
}
#endif

struct NormSpec {
    int fwd_node_id = -1;
    bool is_ln = false;
    int x_node_id = -1;
    int gamma_node_id = -1;
    int beta_node_id = -1;
    float eps = 1e-5f;
};

int input_slot_for_node(const TraceDAG& dag, int node_id) {
    int idx = 0;
    for (const auto& n : dag.nodes()) {
        if (n.op == TraceOpKind::Input) {
            if (n.id == node_id) return idx;
            ++idx;
        }
    }
    return -1;
}

Tensor run_dag_eval(const TraceDAG& raw_dag, int rows, int cols, Device dev) {
    TraceDAG dag = canonicalize_trace_dag_for_codegen(raw_dag);
    Tensor out = Tensor::empty_on(dev, rows, cols, Dtype::FP32);
    for (auto& n : dag.nodes()) {
        n.device = dev;
        if (n.op != TraceOpKind::Input) {
            n.rows = rows;
            n.cols = cols;
        }
        if (n.is_live_at_end) n.buffer = out.data;
    }
    TraceCompiler::compile_and_cache(dag);
    return out;
}

Tensor eval_fwd_value(const TraceDAG& fwd_dag, int node_id,
                      const std::vector<Tensor>& primal_inputs,
                      const Tensor* override_norm_val, int override_norm_id,
                      int dom_rows, int dom_cols, Device dev) {
    const TraceNode& target = fwd_dag.node(node_id);
    if (target.op == TraceOpKind::Input) {
        int slot = input_slot_for_node(fwd_dag, node_id);
        const Tensor& t = primal_inputs.at(static_cast<size_t>(slot));
        if (t.dtype == Dtype::FP32) return t;
        TraceDAG cast_dag;
        int in_s = cast_dag.add_node(TraceOpKind::Input, {}, 0.0f, t.data, dev, t.dtype, t.rows, t.cols);
        int cp = cast_dag.add_node(TraceOpKind::Copy, {in_s}, 0.0f, nullptr, dev, Dtype::FP32, t.rows, t.cols);
        cast_dag.nodes()[static_cast<size_t>(cp)].is_live_at_end = true;
        return run_dag_eval(cast_dag, t.rows, t.cols, dev);
    }
    if (override_norm_val && node_id == override_norm_id) {
        return *override_norm_val;
    }

    TraceDAG sub;
    std::unordered_map<int, int> remap;
    size_t in_cursor = 0;
    for (const auto& n : fwd_dag.nodes()) {
        if (n.id > node_id) break;
        if (n.op == TraceOpKind::Input) {
            const Tensor& pt = primal_inputs.at(in_cursor++);
            remap[n.id] = sub.add_node(TraceOpKind::Input, {}, 0.0f, pt.data, dev, pt.dtype, pt.rows, pt.cols);
        } else if (override_norm_val && n.id == override_norm_id) {
            remap[n.id] = sub.add_node(TraceOpKind::Input, {}, 0.0f, override_norm_val->data, dev,
                                       Dtype::FP32, dom_rows, dom_cols);
        } else {
            std::vector<int> ins;
            bool ok = true;
            for (int inp : n.inputs) {
                auto it = remap.find(inp);
                if (it == remap.end()) { ok = false; break; }
                ins.push_back(it->second);
            }
            if (!ok) continue;
            remap[n.id] = sub.add_node(n.op, ins, n.scalar, nullptr, dev, Dtype::FP32, dom_rows, dom_cols, n.scalar2);
        }
    }
    sub.nodes()[static_cast<size_t>(remap.at(node_id))].is_live_at_end = true;
    return run_dag_eval(sub, dom_rows, dom_cols, dev);
}

Tensor row_scale_stat(const Tensor& sum_x, float scale, Device dev) {
    TraceDAG dag;
    int s0 = dag.add_node(TraceOpKind::Input, {}, 0.0f, sum_x.data, dev, Dtype::FP32, sum_x.rows, 1);
    int y = dag.add_node(TraceOpKind::MulScalar, {s0}, scale, nullptr, dev, Dtype::FP32, sum_x.rows, 1);
    dag.nodes()[static_cast<size_t>(y)].is_live_at_end = true;
    return run_dag_eval(dag, sum_x.rows, 1, dev);
}

std::vector<Tensor> compute_norm_aux_tensors(const TraceDAG& fwd_dag,
                                             const std::vector<NormSpec>& specs,
                                             const std::vector<Tensor>& primal_inputs,
                                             const Tensor& grad_output) {
    std::vector<Tensor> aux;
    if (specs.empty()) return aux;
    const Device dev = grad_output.device;
    const int R = grad_output.rows;
    const int D = grad_output.cols;
    const float inv_d = 1.0f / static_cast<float>(D);

    // Build ordered list of final live outputs of fwd_dag to check for post-norm consumers.
    int final_live_id = -1;
    for (const auto& n : fwd_dag.nodes()) {
        if (n.is_live_at_end) final_live_id = n.id;
    }

    for (const NormSpec& ns : specs) {
        Tensor u = eval_fwd_value(fwd_dag, ns.x_node_id, primal_inputs, nullptr, -1, R, D, dev);
        Tensor gamma, beta;
        if (ns.gamma_node_id >= 0) {
            gamma = eval_fwd_value(fwd_dag, ns.gamma_node_id, primal_inputs, nullptr, -1, R, D, dev);
        }
        if (ns.beta_node_id >= 0) {
            beta = eval_fwd_value(fwd_dag, ns.beta_node_id, primal_inputs, nullptr, -1, R, D, dev);
        }

        Tensor mean_t, rstd_t;
        const float eps_safe = (ns.eps > 1e-12f) ? ns.eps : 1e-12f;
        if (ns.is_ln) {
            Tensor sx = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(u, sx);
            mean_t = row_scale_stat(sx, inv_d, dev);

            TraceDAG d2_dag;
            int iu = d2_dag.add_node(TraceOpKind::Input, {}, 0.0f, u.data, dev, Dtype::FP32, R, D);
            int im = d2_dag.add_node(TraceOpKind::Input, {}, 0.0f, mean_t.data, dev, Dtype::FP32, R, 1);
            int diff = d2_dag.add_node(TraceOpKind::Sub, {iu, im}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            int sq = d2_dag.add_node(TraceOpKind::Mul, {diff, diff}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            d2_dag.nodes()[static_cast<size_t>(sq)].is_live_at_end = true;
            Tensor diff2 = run_dag_eval(d2_dag, R, D, dev);

            Tensor ssq = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(diff2, ssq);

            TraceDAG rs_dag;
            int iss = rs_dag.add_node(TraceOpKind::Input, {}, 0.0f, ssq.data, dev, Dtype::FP32, R, 1);
            int var = rs_dag.add_node(TraceOpKind::MulScalar, {iss}, inv_d, nullptr, dev, Dtype::FP32, R, 1);
            int ve = rs_dag.add_node(TraceOpKind::AddScalar, {var}, eps_safe, nullptr, dev, Dtype::FP32, R, 1);
            int rs = rs_dag.add_node(TraceOpKind::Rsqrt, {ve}, 0.0f, nullptr, dev, Dtype::FP32, R, 1);
            rs_dag.nodes()[static_cast<size_t>(rs)].is_live_at_end = true;
            rstd_t = run_dag_eval(rs_dag, R, 1, dev);
        } else {
            TraceDAG u2_dag;
            int iu = u2_dag.add_node(TraceOpKind::Input, {}, 0.0f, u.data, dev, Dtype::FP32, R, D);
            int sq = u2_dag.add_node(TraceOpKind::Mul, {iu, iu}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            u2_dag.nodes()[static_cast<size_t>(sq)].is_live_at_end = true;
            Tensor u2 = run_dag_eval(u2_dag, R, D, dev);

            Tensor ssq = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(u2, ssq);

            TraceDAG rs_dag;
            int iss = rs_dag.add_node(TraceOpKind::Input, {}, 0.0f, ssq.data, dev, Dtype::FP32, R, 1);
            int ms = rs_dag.add_node(TraceOpKind::MulScalar, {iss}, inv_d, nullptr, dev, Dtype::FP32, R, 1);
            int ve = rs_dag.add_node(TraceOpKind::AddScalar, {ms}, eps_safe, nullptr, dev, Dtype::FP32, R, 1);
            int rs = rs_dag.add_node(TraceOpKind::Rsqrt, {ve}, 0.0f, nullptr, dev, Dtype::FP32, R, 1);
            rs_dag.nodes()[static_cast<size_t>(rs)].is_live_at_end = true;
            rstd_t = run_dag_eval(rs_dag, R, 1, dev);
        }

        // Compute xhat = (u - mean) * rstd (and norm_out if post-ops exist).
        Tensor norm_out;
        Tensor dy_norm = grad_output;
        const bool has_post = (ns.fwd_node_id != final_live_id);
        if (has_post) {
            TraceDAG no_dag;
            int iu = no_dag.add_node(TraceOpKind::Input, {}, 0.0f, u.data, dev, Dtype::FP32, R, D);
            int ir = no_dag.add_node(TraceOpKind::Input, {}, 0.0f, rstd_t.data, dev, Dtype::FP32, R, 1);
            int cur = iu;
            if (ns.is_ln) {
                int im = no_dag.add_node(TraceOpKind::Input, {}, 0.0f, mean_t.data, dev, Dtype::FP32, R, 1);
                cur = no_dag.add_node(TraceOpKind::Sub, {cur, im}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            }
            cur = no_dag.add_node(TraceOpKind::Mul, {cur, ir}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            if (ns.gamma_node_id >= 0) {
                int ig = no_dag.add_node(TraceOpKind::Input, {}, 0.0f, gamma.data, dev,
                                         Dtype::FP32, gamma.rows, gamma.cols);
                cur = no_dag.add_node(TraceOpKind::Mul, {cur, ig}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            }
            if (ns.beta_node_id >= 0) {
                int ib = no_dag.add_node(TraceOpKind::Input, {}, 0.0f, beta.data, dev,
                                         Dtype::FP32, beta.rows, beta.cols);
                cur = no_dag.add_node(TraceOpKind::Add, {cur, ib}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            }
            no_dag.nodes()[static_cast<size_t>(cur)].is_live_at_end = true;
            norm_out = run_dag_eval(no_dag, R, D, dev);

            // Build suffix TraceDAG starting at norm_out (slot 0) + needed external nodes -> final_live_id.
            TraceDAG post_dag;
            std::unordered_map<int, int> post_map;
            std::vector<Tensor> post_inputs;
            post_map[ns.fwd_node_id] = post_dag.add_node(TraceOpKind::Input, {}, 0.0f,
                                                         norm_out.data, dev, Dtype::FP32, R, D);
            post_inputs.push_back(norm_out);

            std::vector<Tensor> pre_cache;
            for (const auto& pn : fwd_dag.nodes()) {
                if (pn.id <= ns.fwd_node_id) continue;
                std::vector<int> ins;
                for (int inp : pn.inputs) {
                    auto it = post_map.find(inp);
                    if (it != post_map.end()) {
                        ins.push_back(it->second);
                    } else {
                        Tensor ext = eval_fwd_value(fwd_dag, inp, primal_inputs, &norm_out, ns.fwd_node_id, R, D, dev);
                        pre_cache.push_back(ext);
                        int ext_slot = post_dag.add_node(TraceOpKind::Input, {}, 0.0f,
                                                         pre_cache.back().data, dev, Dtype::FP32,
                                                         pre_cache.back().rows, pre_cache.back().cols);
                        post_map[inp] = ext_slot;
                        post_inputs.push_back(pre_cache.back());
                        ins.push_back(ext_slot);
                    }
                }
                int pid = post_dag.add_node(pn.op, ins, pn.scalar, nullptr, dev, Dtype::FP32, R, D, pn.scalar2);
                post_map[pn.id] = pid;
            }
            post_dag.nodes()[static_cast<size_t>(post_map.at(final_live_id))].is_live_at_end = true;
            TraceDAG post_bwd = canonicalize_trace_dag_for_codegen(post_dag.backward(0));
            size_t pc = 0;
            dy_norm = Tensor::empty_on(dev, R, D, Dtype::FP32);
            for (auto& bn : post_bwd.nodes()) {
                bn.device = dev;
                if (bn.op == TraceOpKind::Input) {
                    const Tensor& src = (pc < post_inputs.size()) ? post_inputs[pc++] : grad_output;
                    bn.buffer = src.data;
                    bn.rows = src.rows;
                    bn.cols = src.cols;
                    bn.dtype = src.dtype;
                } else {
                    bn.rows = R;
                    bn.cols = D;
                }
                if (bn.is_live_at_end) bn.buffer = dy_norm.data;
            }
            TraceCompiler::compile_and_cache(post_bwd);
        }

        // Compute c0 = mean(g_dy) [if LN] and c1 = mean(g_dy * xhat).
        Tensor c0_t, c1_t;
        if (ns.is_ln) {
            Tensor g_dy_t;
            if (ns.gamma_node_id >= 0) {
                TraceDAG gdy_dag;
                int idy = gdy_dag.add_node(TraceOpKind::Input, {}, 0.0f, dy_norm.data, dev, Dtype::FP32, R, D);
                int ig = gdy_dag.add_node(TraceOpKind::Input, {}, 0.0f, gamma.data, dev,
                                          Dtype::FP32, gamma.rows, gamma.cols);
                int mul = gdy_dag.add_node(TraceOpKind::Mul, {idy, ig}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
                gdy_dag.nodes()[static_cast<size_t>(mul)].is_live_at_end = true;
                g_dy_t = run_dag_eval(gdy_dag, R, D, dev);
            } else {
                g_dy_t = dy_norm;
            }
            Tensor c0_sum = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(g_dy_t, c0_sum);
            c0_t = row_scale_stat(c0_sum, inv_d, dev);

            TraceDAG prod_dag;
            int igdy = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, g_dy_t.data, dev, Dtype::FP32, R, D);
            int iu = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, u.data, dev, Dtype::FP32, R, D);
            int im = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, mean_t.data, dev, Dtype::FP32, R, 1);
            int ir = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, rstd_t.data, dev, Dtype::FP32, R, 1);
            int xhat = prod_dag.add_node(TraceOpKind::Mul,
                                         {prod_dag.add_node(TraceOpKind::Sub, {iu, im}, 0.0f, nullptr, dev, Dtype::FP32, R, D), ir},
                                         0.0f, nullptr, dev, Dtype::FP32, R, D);
            int pnode = prod_dag.add_node(TraceOpKind::Mul, {igdy, xhat}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            prod_dag.nodes()[static_cast<size_t>(pnode)].is_live_at_end = true;
            Tensor prod_t = run_dag_eval(prod_dag, R, D, dev);

            Tensor c1_sum = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(prod_t, c1_sum);
            c1_t = row_scale_stat(c1_sum, inv_d, dev);
        } else {
            TraceDAG prod_dag;
            int idy = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, dy_norm.data, dev, Dtype::FP32, R, D);
            int iu = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, u.data, dev, Dtype::FP32, R, D);
            int ir = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, rstd_t.data, dev, Dtype::FP32, R, 1);
            int xhat = prod_dag.add_node(TraceOpKind::Mul, {iu, ir}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            int gdy = idy;
            if (ns.gamma_node_id >= 0) {
                int ig = prod_dag.add_node(TraceOpKind::Input, {}, 0.0f, gamma.data, dev,
                                           Dtype::FP32, gamma.rows, gamma.cols);
                gdy = prod_dag.add_node(TraceOpKind::Mul, {idy, ig}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            }
            int pnode = prod_dag.add_node(TraceOpKind::Mul, {gdy, xhat}, 0.0f, nullptr, dev, Dtype::FP32, R, D);
            prod_dag.nodes()[static_cast<size_t>(pnode)].is_live_at_end = true;
            Tensor prod_t = run_dag_eval(prod_dag, R, D, dev);

            Tensor c1_sum = Tensor::empty_on(dev, R, 1, Dtype::FP32);
            brotensor::sum_rows(prod_t, c1_sum);
            c1_t = row_scale_stat(c1_sum, inv_d, dev);
        }

        if (ns.is_ln) aux.push_back(std::move(mean_t));
        aux.push_back(std::move(rstd_t));
        if (ns.is_ln) aux.push_back(std::move(c0_t));
        aux.push_back(std::move(c1_t));
    }
    return aux;
}

void bind_and_run_dag(const TraceDAG& template_dag,
                      const std::vector<Tensor>& primal_inputs,
                      const Tensor& grad_output,
                      const std::vector<Tensor>& aux_inputs,
                      std::vector<Tensor>& grad_inputs) {
    TraceDAG dag = template_dag;
    const Device target_dev = grad_output.device;

    size_t in_cursor = 0;
    size_t aux_cursor = 0;
    bool seen_grad_out = false;
    size_t out_cursor = 0;
    for (auto& n : dag.nodes()) {
        n.device = target_dev;
        if (n.op == TraceOpKind::Input) {
            const Tensor* src = nullptr;
            if (in_cursor < primal_inputs.size()) {
                src = &primal_inputs[in_cursor++];
            } else if (!seen_grad_out) {
                src = &grad_output;
                seen_grad_out = true;
            } else if (aux_cursor < aux_inputs.size()) {
                src = &aux_inputs[aux_cursor++];
            } else {
                throw std::runtime_error("VerifiedBackwardTrace::execute: unexpected extra Input node");
            }
            n.buffer = src->data;
            n.rows = src->rows;
            n.cols = src->cols;
            n.dtype = src->dtype;
        } else {
            n.rows = grad_output.rows;
            n.cols = grad_output.cols;
        }
        if (n.is_live_at_end) {
            if (out_cursor >= grad_inputs.size()) {
                throw std::runtime_error("VerifiedBackwardTrace::execute: output buffer count mismatch");
            }
            n.buffer = grad_inputs[out_cursor++].data;
        }
    }

    TraceHandle h = TraceCompiler::compile_and_cache(dag);
    (void)h;
}

Tensor reduce_grad_to_input_shape(const Tensor& g, const Tensor& primal_in, int dom_rows, int dom_cols) {
    if (primal_in.rows == dom_rows && primal_in.cols == dom_cols) return g;
    const Device dev = g.device;
    Tensor cur = g;
    if (primal_in.rows == 1 && dom_rows > 1) {
        Tensor red = Tensor::empty_on(dev, 1, dom_cols, Dtype::FP32);
        brotensor::sum_cols(cur, red);
        cur = std::move(red);
    }
    if (primal_in.cols == 1 && dom_cols > 1) {
        Tensor red = Tensor::empty_on(dev, cur.rows, 1, Dtype::FP32);
        brotensor::sum_rows(cur, red);
        cur = std::move(red);
    }
    return cur;
}

} // namespace

BackwardTraceVerificationReport verify_backward_trace_dag(const TraceDAG& bwd_dag) {
    BackwardTraceVerificationReport rep;
    rep.rawBackwardNodeCount = bwd_dag.node_count();
    TraceDAG lowered = canonicalize_trace_dag_for_codegen(bwd_dag);
    rep.loweredBackwardNodeCount = lowered.node_count();

#if BROTENSOR_HAS_BRASS_JIT
    {
        brass::Module mod("bwd_mir_equiv_mod");
        brass::Function* fn_unopt = build_scalar_backward_mir_fn(mod, "bwd_unopt", lowered);
        brass::Function* fn_opt = build_scalar_backward_mir_fn(mod, "bwd_opt", lowered);
        rep.unoptimizedMirInsts = count_mir_insts(*fn_unopt);
        gpu_mir::optimize_function(*fn_opt);
        rep.optimizedMirInsts = count_mir_insts(*fn_opt);

        brass::verify::EquivResult mir_res = brass::verify::verify_mir_equivalence(*fn_unopt, *fn_opt);
        rep.mirEquivProven = mir_res.is_equivalent();
        if (!rep.mirEquivProven && rep.diagnostic.empty()) {
            rep.diagnostic = "MIR equivalence failed: " + mir_res.diagnostic;
        }
    }

    std::string ptx_src;
    {
        brass::Module gpu_mod("bwd_gpu_equiv_mod");
        brass::Function* gpu_fn = build_gpu_equiv_kernel_fn(gpu_mod, lowered);
        gpu_mir::optimize_function(*gpu_fn);

        ptx_src = brass::target::PtxTarget::emit_function(*gpu_fn);
        rep.ptxByteCount = ptx_src.size();

        brass::verify::EquivResult gpu_res = brass::verify::verify_spirv_ptx_equivalence(*gpu_fn);
        rep.spirvPtxEquivProven = gpu_res.is_equivalent();
        if (!rep.spirvPtxEquivProven && rep.diagnostic.empty()) {
            rep.diagnostic = "SPIR-V/PTX equivalence failed: " + gpu_res.diagnostic;
        }
    }

    {
        plan::ElementwisePlan ew;
        if (plan::build_plan(lowered, ew)) {
            spirv::Kernels spv = spirv::emit_elementwise(lowered, ew);
            rep.spirvWordCount = spv.scalar.words.size();

            auto audit_words = [](const std::vector<uint32_t>& words, std::string& diag) -> bool {
                if (words.empty()) return false;
                brass::spirv::ComputeShaderAuditReport audit =
                    brass::spirv::prove_spirv_shader_safety(
                        std::span<const uint32_t>(words.data(), words.size()));
                if (!audit.ok()) {
                    if (!audit.hazards.empty()) diag = audit.hazards.front().message;
                    else diag = "SPIR-V audit failed";
                    return false;
                }
                for (const auto& d : audit.divisions) {
                    if (!d.proven_safe) {
                        diag = "Unguarded division/sqrt in SPIR-V backward shader: " + d.guard_reason;
                        return false;
                    }
                }
                return true;
            };

            std::string audit_diag;
            bool scalar_ok = audit_words(spv.scalar.words, audit_diag);
            bool vec_ok = !spv.vec.has_value() || audit_words(spv.vec->words, audit_diag);
            rep.nanSafeProven = scalar_ok && vec_ok && !ptx_src.empty();
            if (!rep.nanSafeProven && rep.diagnostic.empty()) {
                rep.diagnostic = audit_diag;
            }
        } else if (rep.diagnostic.empty()) {
            rep.diagnostic = "Failed to build ElementwisePlan for lowered backward TraceDAG";
        }
    }
#endif

    if (rep.all_proven() && rep.diagnostic.empty()) {
        rep.diagnostic = "OK";
    }
    return rep;
}

struct VerifiedBackwardTrace::Impl {
    TraceDAG forward_dag;
    TraceDAG raw_backward_dag;
    TraceDAG lowered_backward_dag;
    std::vector<TraceDAG> per_input_lowered_dags;
    std::vector<NormSpec> norm_specs;
};

VerifiedBackwardTrace build_verified_backward_trace(const TraceDAG& forward_dag) {
    VerifiedBackwardTrace res;
    res.impl = std::make_shared<VerifiedBackwardTrace::Impl>();
    res.impl->forward_dag = forward_dag;
    for (const auto& n : forward_dag.nodes()) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) {
            NormSpec ns;
            ns.fwd_node_id = n.id;
            ns.is_ln = (n.op == TraceOpKind::LayerNorm);
            ns.x_node_id = n.inputs[0];
            ns.gamma_node_id = (n.inputs.size() > 1) ? n.inputs[1] : -1;
            ns.beta_node_id = (n.inputs.size() > 2) ? n.inputs[2] : -1;
            ns.eps = n.scalar;
            res.impl->norm_specs.push_back(ns);
        }
    }
    res.impl->raw_backward_dag = forward_dag.backward(-1);
    res.impl->lowered_backward_dag = canonicalize_trace_dag_for_codegen(res.impl->raw_backward_dag);

    const DominantMeta m = find_dominant_meta(forward_dag);
    res.rows = m.rows;
    res.cols = m.cols;
    res.device = m.dev;
    res.dtype = Dtype::FP32;

    size_t n_in = 0;
    for (const auto& n : forward_dag.nodes()) {
        if (n.op == TraceOpKind::Input) ++n_in;
    }
    res.num_inputs = n_in;

    for (size_t i = 0; i < n_in; ++i) {
        TraceDAG single_raw = forward_dag.backward(static_cast<int>(i));
        res.impl->per_input_lowered_dags.push_back(canonicalize_trace_dag_for_codegen(single_raw));
    }

    res.verification = verify_backward_trace_dag(res.impl->raw_backward_dag);
    res.verification.forwardNodeCount = forward_dag.node_count();

#if BROTENSOR_HAS_BRASS_JIT
    {
        brass::Module gpu_mod("bwd_ptx_emit_mod");
        brass::Function* gpu_fn = build_gpu_equiv_kernel_fn(gpu_mod, res.impl->lowered_backward_dag);
        gpu_mir::optimize_function(*gpu_fn);
        res.ptx_source = brass::target::PtxTarget::emit_function(*gpu_fn);
    }
    plan::ElementwisePlan ew;
    if (plan::build_plan(res.impl->lowered_backward_dag, ew)) {
        spirv::Kernels spv = spirv::emit_elementwise(res.impl->lowered_backward_dag, ew);
        res.spirv_words = std::move(spv.scalar.words);
    }
#endif
    return res;
}

VerifiedBackwardTrace trace_and_verify_backward(
    const std::vector<Tensor>& example_inputs,
    const std::function<Tensor(const std::vector<Tensor>&)>& fn) {
    if (example_inputs.empty()) {
        throw std::invalid_argument("trace_and_verify_backward: example_inputs must not be empty");
    }
    if (is_tracing()) {
        abort_trace();
    }
    begin_trace();
    TraceDAG fwd_dag;
    try {
        for (const auto& in_t : example_inputs) {
            TraceContext::current().get_or_register_slot(in_t);
        }
        Tensor out = fn(example_inputs);
        fwd_dag = TraceContext::current().end();
        (void)out;
    } catch (...) {
        if (is_tracing()) abort_trace();
        throw;
    }
    return build_verified_backward_trace(fwd_dag);
}

std::vector<Tensor> VerifiedBackwardTrace::execute(
    const std::vector<Tensor>& primal_inputs,
    const Tensor& grad_output) const {
    if (!impl) {
        throw std::runtime_error("VerifiedBackwardTrace::execute: uninitialized backward trace");
    }
    if (primal_inputs.size() != num_inputs) {
        throw std::invalid_argument("VerifiedBackwardTrace::execute: primal_inputs count mismatch");
    }
    std::vector<Tensor> aux_tensors = compute_norm_aux_tensors(
        impl->forward_dag, impl->norm_specs, primal_inputs, grad_output);
    std::vector<Tensor> grads;
    grads.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        grads.push_back(Tensor::empty_on(grad_output.device, grad_output.rows, grad_output.cols, Dtype::FP32));
    }
    bind_and_run_dag(impl->lowered_backward_dag, primal_inputs, grad_output, aux_tensors, grads);
    for (size_t i = 0; i < num_inputs; ++i) {
        grads[i] = reduce_grad_to_input_shape(grads[i], primal_inputs[i], grad_output.rows, grad_output.cols);
    }
    return grads;
}

Tensor VerifiedBackwardTrace::execute_for_input(
    int input_idx,
    const std::vector<Tensor>& primal_inputs,
    const Tensor& grad_output) const {
    if (!impl || input_idx < 0 || static_cast<size_t>(input_idx) >= impl->per_input_lowered_dags.size()) {
        throw std::out_of_range("VerifiedBackwardTrace::execute_for_input: input_idx out of range");
    }
    if (primal_inputs.size() != num_inputs) {
        throw std::invalid_argument("VerifiedBackwardTrace::execute_for_input: primal_inputs count mismatch");
    }
    std::vector<Tensor> aux_tensors = compute_norm_aux_tensors(
        impl->forward_dag, impl->norm_specs, primal_inputs, grad_output);
    std::vector<Tensor> grads;
    grads.push_back(Tensor::empty_on(grad_output.device, grad_output.rows, grad_output.cols, Dtype::FP32));
    bind_and_run_dag(impl->per_input_lowered_dags[static_cast<size_t>(input_idx)],
                     primal_inputs, grad_output, aux_tensors, grads);
    return reduce_grad_to_input_shape(
        grads[0], primal_inputs[static_cast<size_t>(input_idx)], grad_output.rows, grad_output.cols);
}

} // namespace brotensor::jit
