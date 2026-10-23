// ─── Verified Fused Backward Trace JIT (TraceDAG::backward() Autodiff) ──────
//
// Reverse-mode automatic differentiation over `brotensor::jit::TraceDAG` with
// compile-time formal verification via Brass:
//   1. `mirEquivProven`: Formal equivalence proof (`brass::verify::verify_mir_equivalence`)
//      between unoptimized and optimized backward MIR functions.
//   2. `spirvPtxEquivProven`: Cross-ISA equivalence proof (`brass::verify::verify_spirv_ptx_equivalence`)
//      between the backward MIR, Vulkan SPIR-V, and NVIDIA PTX kernels.
//   3. `nanSafeProven`: Static compute shader safety proof (`brass::verify::prove_spirv_shader_safety`)
//      certifying absence of division-by-zero, domain violations, and NaN/Inf hazards.

#include <brotensor/trace_autodiff.h>
#include "trace_dag.h"
#include "trace_compiler.h"
#include "trace_plan.h"

#if BROTENSOR_HAS_BRASS_JIT
#include "spirv_emit.h"
#include "spirv_emit_detail.h"
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/types.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <brass/verify/smt_equiv.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit {

namespace {

constexpr float kSafeEps = 1e-12f;
constexpr float kGeluK0 = 0.7978845608f; // sqrt(2 / pi)
constexpr float kGeluK1 = 0.044715f;

struct ShapeMeta {
    Device dev = Device::CPU;
    Dtype dt = Dtype::FP32;
    int rows = 1, cols = 1;
};

ShapeMeta dominant_meta(const TraceDAG& dag) {
    ShapeMeta m;
    int r = 0, c = 0;
    dag.dominant_shape(r, c);
    m.rows = (r > 0) ? r : 1;
    m.cols = (c > 0) ? c : 1;
    for (const auto& n : dag.nodes()) {
        m.dev = n.device;
        m.dt = n.dtype;
        if (n.dtype == Dtype::FP32) break;
    }
    return m;
}

int emit_const(TraceDAG& dag, int anchor, float val, const ShapeMeta& m) {
    int zero = dag.add_node(TraceOpKind::MulScalar, {anchor}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return (val == 0.0f) ? zero : dag.add_node(TraceOpKind::AddScalar, {zero}, val, nullptr, m.dev, m.dt, m.rows, m.cols);
}

int emit_neg(TraceDAG& dag, int x, const ShapeMeta& m) {
    return dag.add_node(TraceOpKind::ScalarSub, {x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

int emit_abs(TraceDAG& dag, int x, const ShapeMeta& m) {
    int pos = dag.add_node(TraceOpKind::ReLU, {x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int neg_part = dag.add_node(TraceOpKind::ReLU, {emit_neg(dag, x, m)}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return dag.add_node(TraceOpKind::Add, {pos, neg_part}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

// Provably positive denominator: ReLU(x) + eps >= eps > 0 (`OpFAdd(NMax(x, 0), eps)`).
int emit_pos_denom(TraceDAG& dag, int nonneg_x, float eps, const ShapeMeta& m) {
    int rx = dag.add_node(TraceOpKind::ReLU, {nonneg_x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return dag.add_node(TraceOpKind::AddScalar, {rx}, eps, nullptr, m.dev, m.dt, m.rows, m.cols);
}

// Safe general division `num / den` for arbitrary signed `den` (including 0).
int emit_safe_div_signed(TraceDAG& dag, int anchor, int num, int den, const ShapeMeta& m) {
    const auto& dn = dag.node(den);
    if (dn.op == TraceOpKind::AddScalar && dn.scalar > 0.0f && !dn.inputs.empty() &&
        dag.node(dn.inputs[0]).op == TraceOpKind::ReLU) {
        return dag.add_node(TraceOpKind::Div, {num, den}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    }
    int pos_d = dag.add_node(TraceOpKind::ReLU, {den}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int neg_d = dag.add_node(TraceOpKind::ReLU, {emit_neg(dag, den, m)}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int abs_d = dag.add_node(TraceOpKind::Add, {pos_d, neg_d}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int safe_den = emit_pos_denom(dag, abs_d, kSafeEps, m);
    int c_zero = emit_const(dag, anchor, 0.0f, m);
    int c_pos1 = emit_const(dag, anchor, 1.0f, m);
    int c_neg1 = emit_const(dag, anchor, -1.0f, m);
    int neg_branch = dag.add_node(TraceOpKind::Where, {neg_d, c_neg1, c_zero}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int sign_d = dag.add_node(TraceOpKind::Where, {pos_d, c_pos1, neg_branch}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int signed_num = dag.add_node(TraceOpKind::Mul, {num, sign_d}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return dag.add_node(TraceOpKind::Div, {signed_num, safe_den}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

// Lowered Exp(x) using range reduction by 1/16 + Tanh + 4 squarings.
int emit_lowered_exp(TraceDAG& dag, int x, const ShapeMeta& m) {
    int xc = dag.add_node(TraceOpKind::Clamp, {x}, -88.0f, nullptr, m.dev, m.dt, m.rows, m.cols, 88.0f);
    int u = dag.add_node(TraceOpKind::MulScalar, {xc}, 1.0f / 32.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int t = dag.add_node(TraceOpKind::Tanh, {u}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int num = dag.add_node(TraceOpKind::AddScalar, {t}, 1.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int den = emit_pos_denom(dag, dag.add_node(TraceOpKind::ScalarSub, {t}, 1.0f, nullptr, m.dev, m.dt, m.rows, m.cols), kSafeEps, m);
    int e = dag.add_node(TraceOpKind::Div, {num, den}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    for (int i = 0; i < 4; ++i) e = dag.add_node(TraceOpKind::Mul, {e, e}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return e;
}

// Lowered Sqrt(x) via domain clamp + piecewise initial guess + 7 Newton-Raphson iterations.
int emit_lowered_sqrt(TraceDAG& dag, int anchor, int x, const ShapeMeta& m) {
    int rx = dag.add_node(TraceOpKind::ReLU, {x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int xs = emit_pos_denom(dag, rx, kSafeEps, m);
    int gt16 = dag.add_node(TraceOpKind::ReLU,
                            {dag.add_node(TraceOpKind::SubScalar, {xs}, 16.0f, nullptr, m.dev, m.dt, m.rows, m.cols)},
                            0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int guess_hi = dag.add_node(TraceOpKind::AddScalar,
                                {dag.add_node(TraceOpKind::MulScalar, {xs}, 0.125f, nullptr, m.dev, m.dt, m.rows, m.cols)},
                                2.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int guess_lo = dag.add_node(TraceOpKind::AddScalar,
                                {dag.add_node(TraceOpKind::MulScalar, {xs}, 0.5f, nullptr, m.dev, m.dt, m.rows, m.cols)},
                                0.5f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int y = dag.add_node(TraceOpKind::Where, {gt16, guess_hi, guess_lo}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    for (int iter = 0; iter < 7; ++iter) {
        int q = dag.add_node(TraceOpKind::Div, {xs, emit_pos_denom(dag, y, kSafeEps, m)}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
        y = dag.add_node(TraceOpKind::MulScalar, {dag.add_node(TraceOpKind::Add, {y, q}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 0.5f, nullptr, m.dev, m.dt, m.rows, m.cols);
    }
    return dag.add_node(TraceOpKind::Where, {rx, y, emit_const(dag, anchor, 0.0f, m)}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

int emit_lowered_rsqrt(TraceDAG& dag, int anchor, int x, const ShapeMeta& m) {
    int rx = dag.add_node(TraceOpKind::ReLU, {x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int s = emit_lowered_sqrt(dag, anchor, emit_pos_denom(dag, rx, kSafeEps, m), m);
    return dag.add_node(TraceOpKind::Div, {emit_const(dag, anchor, 1.0f, m), emit_pos_denom(dag, s, kSafeEps, m)}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

// Lowered Log(x) via 3 nested square roots (r = x^{1/8}) and 5-term atanh series.
int emit_lowered_log(TraceDAG& dag, int anchor, int x, const ShapeMeta& m) {
    int rx = dag.add_node(TraceOpKind::ReLU, {x}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int r1 = emit_lowered_sqrt(dag, anchor, emit_pos_denom(dag, rx, kSafeEps, m), m);
    int r2 = emit_lowered_sqrt(dag, anchor, r1, m);
    int r3 = emit_lowered_sqrt(dag, anchor, r2, m);
    int u = dag.add_node(TraceOpKind::Div,
                         {dag.add_node(TraceOpKind::SubScalar, {r3}, 1.0f, nullptr, m.dev, m.dt, m.rows, m.cols),
                          emit_pos_denom(dag, r3, 1.0f, m)},
                         0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int u2 = dag.add_node(TraceOpKind::Mul, {u, u}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    int p = dag.add_node(TraceOpKind::AddScalar, {dag.add_node(TraceOpKind::MulScalar, {u2}, 1.0f / 9.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 1.0f / 7.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    p = dag.add_node(TraceOpKind::AddScalar, {dag.add_node(TraceOpKind::Mul, {p, u2}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 1.0f / 5.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    p = dag.add_node(TraceOpKind::AddScalar, {dag.add_node(TraceOpKind::Mul, {p, u2}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 1.0f / 3.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    p = dag.add_node(TraceOpKind::AddScalar, {dag.add_node(TraceOpKind::Mul, {p, u2}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 1.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
    return dag.add_node(TraceOpKind::MulScalar, {dag.add_node(TraceOpKind::Mul, {u, p}, 0.0f, nullptr, m.dev, m.dt, m.rows, m.cols)}, 16.0f, nullptr, m.dev, m.dt, m.rows, m.cols);
}

} // namespace

TraceDAG canonicalize_trace_dag_for_codegen(const TraceDAG& dag) {
    bool needs_lowering = false;
    for (const auto& n : dag.nodes()) {
        switch (n.op) {
            case TraceOpKind::Const:
            case TraceOpKind::Neg:
            case TraceOpKind::Abs:
            case TraceOpKind::Exp:
            case TraceOpKind::Log:
            case TraceOpKind::Sqrt:
            case TraceOpKind::Rsqrt:
            case TraceOpKind::Div:
            case TraceOpKind::DivScalar:
            case TraceOpKind::ScalarDiv:
            case TraceOpKind::FMA:
            case TraceOpKind::Modulate:
            case TraceOpKind::GELU:
                needs_lowering = true;
                break;
            default:
                break;
        }
        if (needs_lowering) break;
    }
    if (!needs_lowering) return dag;

    TraceDAG out;
    std::unordered_map<int, int> remap;
    int anchor = -1;

    for (const auto& n : dag.nodes()) {
        ShapeMeta m{n.device, n.dtype, n.rows, n.cols};
        std::vector<int> ins;
        ins.reserve(n.inputs.size());
        for (int old_in : n.inputs) {
            ins.push_back(remap.at(old_in));
        }

        int new_id = -1;
        switch (n.op) {
            case TraceOpKind::Input: {
                new_id = out.add_node(TraceOpKind::Input, {}, 0.0f, n.buffer, n.device, n.dtype, n.rows, n.cols);
                if (anchor < 0) anchor = new_id;
                break;
            }
            case TraceOpKind::Const: {
                int a = (anchor >= 0) ? anchor : 0;
                new_id = emit_const(out, a, n.scalar, m);
                break;
            }
            case TraceOpKind::Neg: {
                new_id = emit_neg(out, ins[0], m);
                break;
            }
            case TraceOpKind::Abs: {
                new_id = emit_abs(out, ins[0], m);
                break;
            }
            case TraceOpKind::Exp: {
                new_id = emit_lowered_exp(out, ins[0], m);
                break;
            }
            case TraceOpKind::Sqrt: {
                new_id = emit_lowered_sqrt(out, (anchor >= 0) ? anchor : ins[0], ins[0], m);
                break;
            }
            case TraceOpKind::Rsqrt: {
                new_id = emit_lowered_rsqrt(out, (anchor >= 0) ? anchor : ins[0], ins[0], m);
                break;
            }
            case TraceOpKind::Log: {
                new_id = emit_lowered_log(out, (anchor >= 0) ? anchor : ins[0], ins[0], m);
                break;
            }
            case TraceOpKind::Div: {
                new_id = emit_safe_div_signed(out, (anchor >= 0) ? anchor : ins[0], ins[0], ins[1], m);
                break;
            }
            case TraceOpKind::DivScalar: {
                float s = n.scalar;
                if (std::fabs(s) < kSafeEps) s = (s < 0.0f) ? -kSafeEps : kSafeEps;
                new_id = out.add_node(TraceOpKind::MulScalar, {ins[0]}, 1.0f / s,
                                      nullptr, n.device, n.dtype, n.rows, n.cols);
                break;
            }
            case TraceOpKind::ScalarDiv: {
                int s_node = emit_const(out, (anchor >= 0) ? anchor : ins[0], n.scalar, m);
                new_id = emit_safe_div_signed(out, (anchor >= 0) ? anchor : ins[0], s_node, ins[0], m);
                break;
            }
            case TraceOpKind::FMA: {
                int ab = out.add_node(TraceOpKind::Mul, {ins[0], ins[1]}, 0.0f,
                                      nullptr, n.device, n.dtype, n.rows, n.cols);
                new_id = out.add_node(TraceOpKind::Add, {ab, ins[2]}, 0.0f,
                                      nullptr, n.device, n.dtype, n.rows, n.cols);
                break;
            }
            case TraceOpKind::Modulate: {
                int sp1 = out.add_node(TraceOpKind::AddScalar, {ins[1]}, 1.0f,
                                       nullptr, n.device, n.dtype, n.rows, n.cols);
                int x_sp1 = out.add_node(TraceOpKind::Mul, {ins[0], sp1}, 0.0f,
                                         nullptr, n.device, n.dtype, n.rows, n.cols);
                new_id = out.add_node(TraceOpKind::Add, {x_sp1, ins[2]}, 0.0f,
                                      nullptr, n.device, n.dtype, n.rows, n.cols);
                break;
            }
            case TraceOpKind::GELU: {
                int x = ins[0];
                int x2 = out.add_node(TraceOpKind::Mul, {x, x}, 0.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                int x3 = out.add_node(TraceOpKind::Mul, {x2, x}, 0.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                int kx3 = out.add_node(TraceOpKind::MulScalar, {x3}, kGeluK1, nullptr, n.device, n.dtype, n.rows, n.cols);
                int inner = out.add_node(TraceOpKind::Add, {x, kx3}, 0.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                int u = out.add_node(TraceOpKind::MulScalar, {inner}, kGeluK0, nullptr, n.device, n.dtype, n.rows, n.cols);
                int t = out.add_node(TraceOpKind::Tanh, {u}, 0.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                int one_p_t = out.add_node(TraceOpKind::AddScalar, {t}, 1.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                int half_x = out.add_node(TraceOpKind::MulScalar, {x}, 0.5f, nullptr, n.device, n.dtype, n.rows, n.cols);
                new_id = out.add_node(TraceOpKind::Mul, {half_x, one_p_t}, 0.0f, nullptr, n.device, n.dtype, n.rows, n.cols);
                break;
            }
            default: {
                if (n.is_inplace) {
                    int target = remap.at(n.inplace_target_slot);
                    int arg = (n.inputs.size() > 1) ? ins[1] : -1;
                    new_id = out.add_inplace_node(n.op, target, arg, n.buffer, n.scalar, n.scalar2);
                } else {
                    new_id = out.add_node(n.op, ins, n.scalar, nullptr,
                                          n.device, n.dtype, n.rows, n.cols, n.scalar2);
                }
                break;
            }
        }
        TraceNode& created = out.nodes()[static_cast<size_t>(new_id)];
        created.buffer = n.buffer;
        created.is_live_at_end = n.is_live_at_end;
        remap[n.id] = new_id;
    }

    for (auto& nd : out.nodes()) nd.outputs.clear();
    for (const auto& nd : out.nodes()) {
        for (int in_id : nd.inputs) {
            if (in_id >= 0 && in_id < static_cast<int>(out.nodes().size())) {
                out.nodes()[static_cast<size_t>(in_id)].outputs.push_back(nd.id);
            }
        }
    }
    return out;
}

TraceDAG TraceDAG::backward(int target_input_idx) const {
    if (nodes_.empty()) {
        throw std::runtime_error("TraceDAG::backward: cannot differentiate an empty TraceDAG");
    }

    const ShapeMeta m = dominant_meta(*this);
    TraceDAG bwd;
    std::unordered_map<int, int> fwd_to_bwd;
    std::vector<int> primal_input_ids;

    // 1. Copy primal Input nodes first so input slots [0..num_inputs-1] match forward_dag
    for (const auto& n : nodes_) {
        if (n.op == TraceOpKind::Input) {
            int bid = bwd.add_node(TraceOpKind::Input, {}, 0.0f, n.buffer,
                                   n.device, Dtype::FP32, n.rows, n.cols);
            fwd_to_bwd[n.id] = bid;
            primal_input_ids.push_back(n.id);
        }
    }
    if (primal_input_ids.empty()) {
        throw std::runtime_error("TraceDAG::backward: forward TraceDAG has no Input nodes");
    }
    if (target_input_idx >= static_cast<int>(primal_input_ids.size())) {
        throw std::invalid_argument("TraceDAG::backward: target_input_idx out of range");
    }

    // 2. Register upstream gradient `grad_output` (`dL/dy`) as the next Input node
    const int grad_out_slot = bwd.add_node(TraceOpKind::Input, {}, 0.0f, nullptr,
                                           m.dev, Dtype::FP32, m.rows, m.cols);
    const int anchor = fwd_to_bwd[primal_input_ids[0]];

    // 3. Replay forward intermediate nodes into `bwd` (without live_at_end) so
    //    backward derivative expressions can reference any primal intermediate
    for (const auto& n : nodes_) {
        if (n.op == TraceOpKind::Input) continue;
        std::vector<int> ins;
        ins.reserve(n.inputs.size());
        for (int in_id : n.inputs) ins.push_back(fwd_to_bwd.at(in_id));
        int bid = bwd.add_node(n.op, ins, n.scalar, nullptr,
                               n.device, Dtype::FP32, n.rows, n.cols, n.scalar2);
        fwd_to_bwd[n.id] = bid;
    }

    // 4. Seed adjoints: live-at-end output nodes receive `grad_out_slot`
    std::unordered_map<int, int> adj;
    bool seeded = false;
    for (const auto& n : nodes_) {
        if (n.is_live_at_end) {
            adj[n.id] = grad_out_slot;
            seeded = true;
        }
    }
    if (!seeded) {
        adj[nodes_.back().id] = grad_out_slot;
    }

    auto accum_grad = [&](int fwd_id, int contrib_bwd_slot) {
        auto it = adj.find(fwd_id);
        if (it == adj.end()) {
            adj[fwd_id] = contrib_bwd_slot;
        } else {
            adj[fwd_id] = bwd.add_node(TraceOpKind::Add, {it->second, contrib_bwd_slot},
                                       0.0f, nullptr, m.dev, Dtype::FP32, m.rows, m.cols);
        }
    };

    auto add_op = [&](TraceOpKind op, const std::vector<int>& ins, float s = 0.0f, float s2 = 0.0f) {
        return bwd.add_node(op, ins, s, nullptr, m.dev, Dtype::FP32, m.rows, m.cols, s2);
    };

    // 5. Reverse-mode traversal from last node to first
    for (int idx = static_cast<int>(nodes_.size()) - 1; idx >= 0; --idx) {
        const TraceNode& n = nodes_[static_cast<size_t>(idx)];
        auto it_g = adj.find(n.id);
        if (it_g == adj.end()) continue;
        const int dy = it_g->second;

        switch (n.op) {
            case TraceOpKind::Input:
            case TraceOpKind::Const:
                break;

            case TraceOpKind::Copy:
            case TraceOpKind::Cast:
            case TraceOpKind::AddScalar:
            case TraceOpKind::SubScalar:
                accum_grad(n.inputs[0], dy);
                break;

            case TraceOpKind::Add:
                accum_grad(n.inputs[0], dy);
                accum_grad(n.inputs[1], dy);
                break;

            case TraceOpKind::Sub: {
                accum_grad(n.inputs[0], dy);
                int ndy = add_op(TraceOpKind::ScalarSub, {dy}, 0.0f);
                accum_grad(n.inputs[1], ndy);
                break;
            }

            case TraceOpKind::ScalarSub:
            case TraceOpKind::Neg: {
                int ndy = add_op(TraceOpKind::ScalarSub, {dy}, 0.0f);
                accum_grad(n.inputs[0], ndy);
                break;
            }

            case TraceOpKind::Mul: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int b = fwd_to_bwd.at(n.inputs[1]);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, b}));
                accum_grad(n.inputs[1], add_op(TraceOpKind::Mul, {dy, a}));
                break;
            }

            case TraceOpKind::MulScalar:
                accum_grad(n.inputs[0], add_op(TraceOpKind::MulScalar, {dy}, n.scalar));
                break;

            case TraceOpKind::DivScalar: {
                float s = n.scalar;
                if (std::fabs(s) < kSafeEps) s = (s < 0.0f) ? -kSafeEps : kSafeEps;
                accum_grad(n.inputs[0], add_op(TraceOpKind::MulScalar, {dy}, 1.0f / s));
                break;
            }

            case TraceOpKind::Div: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int b = fwd_to_bwd.at(n.inputs[1]);
                int da = add_op(TraceOpKind::Div, {dy, b});
                int b2 = add_op(TraceOpKind::Mul, {b, b});
                int b2_safe = emit_pos_denom(bwd, b2, kSafeEps, m);
                int dy_a = add_op(TraceOpKind::Mul, {dy, a});
                int ndy_a = add_op(TraceOpKind::ScalarSub, {dy_a}, 0.0f);
                int db = add_op(TraceOpKind::Div, {ndy_a, b2_safe});
                accum_grad(n.inputs[0], da);
                accum_grad(n.inputs[1], db);
                break;
            }

            case TraceOpKind::ScalarDiv: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int a2 = add_op(TraceOpKind::Mul, {a, a});
                int a2_safe = emit_pos_denom(bwd, a2, kSafeEps, m);
                int num = add_op(TraceOpKind::MulScalar, {dy}, -n.scalar);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Div, {num, a2_safe}));
                break;
            }

            case TraceOpKind::FMA: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int b = fwd_to_bwd.at(n.inputs[1]);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, b}));
                accum_grad(n.inputs[1], add_op(TraceOpKind::Mul, {dy, a}));
                accum_grad(n.inputs[2], dy);
                break;
            }

            case TraceOpKind::Modulate: {
                int x = fwd_to_bwd.at(n.inputs[0]);
                int scale = fwd_to_bwd.at(n.inputs[1]);
                int sp1 = add_op(TraceOpKind::AddScalar, {scale}, 1.0f);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, sp1}));
                accum_grad(n.inputs[1], add_op(TraceOpKind::Mul, {dy, x}));
                accum_grad(n.inputs[2], dy);
                break;
            }

            case TraceOpKind::ReLU: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int ra = add_op(TraceOpKind::ReLU, {a});
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {ra, dy, zero}));
                break;
            }

            case TraceOpKind::Abs: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int pos = add_op(TraceOpKind::ReLU, {a});
                int na = add_op(TraceOpKind::ScalarSub, {a}, 0.0f);
                int neg_part = add_op(TraceOpKind::ReLU, {na});
                int ndy = add_op(TraceOpKind::ScalarSub, {dy}, 0.0f);
                int zero = emit_const(bwd, anchor, 0.0f, m);
                int neg_sel = add_op(TraceOpKind::Where, {neg_part, ndy, zero});
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {pos, dy, neg_sel}));
                break;
            }

            case TraceOpKind::Clamp: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int gt_lo = add_op(TraceOpKind::ReLU, {add_op(TraceOpKind::SubScalar, {a}, n.scalar)});
                int lt_hi = add_op(TraceOpKind::ReLU, {add_op(TraceOpKind::ScalarSub, {a}, n.scalar2)});
                int inside = add_op(TraceOpKind::Mul, {gt_lo, lt_hi});
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {inside, dy, zero}));
                break;
            }

            case TraceOpKind::Where: {
                int c = fwd_to_bwd.at(n.inputs[0]);
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[1], add_op(TraceOpKind::Where, {c, dy, zero}));
                accum_grad(n.inputs[2], add_op(TraceOpKind::Where, {c, zero, dy}));
                break;
            }

            case TraceOpKind::Sigmoid: {
                int sig = fwd_to_bwd.at(n.id);
                int one_m_sig = add_op(TraceOpKind::ScalarSub, {sig}, 1.0f);
                int dsig = add_op(TraceOpKind::Mul, {sig, one_m_sig});
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, dsig}));
                break;
            }

            case TraceOpKind::Tanh: {
                int th = fwd_to_bwd.at(n.id);
                int th2 = add_op(TraceOpKind::Mul, {th, th});
                int dtanh = add_op(TraceOpKind::ScalarSub, {th2}, 1.0f);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, dtanh}));
                break;
            }

            case TraceOpKind::SiLU: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int sig = add_op(TraceOpKind::Sigmoid, {a});
                int one_m_sig = add_op(TraceOpKind::ScalarSub, {sig}, 1.0f);
                int a_oms = add_op(TraceOpKind::Mul, {a, one_m_sig});
                int factor = add_op(TraceOpKind::AddScalar, {a_oms}, 1.0f);
                int dsilu = add_op(TraceOpKind::Mul, {sig, factor});
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, dsilu}));
                break;
            }

            case TraceOpKind::GELU: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int a2 = add_op(TraceOpKind::Mul, {a, a});
                int a3 = add_op(TraceOpKind::Mul, {a2, a});
                int inner = add_op(TraceOpKind::Add, {a, add_op(TraceOpKind::MulScalar, {a3}, kGeluK1)});
                int u = add_op(TraceOpKind::MulScalar, {inner}, kGeluK0);
                int t = add_op(TraceOpKind::Tanh, {u});
                int one_p_t = add_op(TraceOpKind::AddScalar, {t}, 1.0f);
                int left = add_op(TraceOpKind::MulScalar, {one_p_t}, 0.5f);

                int t2 = add_op(TraceOpKind::Mul, {t, t});
                int sech2 = add_op(TraceOpKind::ScalarSub, {t2}, 1.0f);
                int du = add_op(TraceOpKind::MulScalar,
                                {add_op(TraceOpKind::AddScalar,
                                        {add_op(TraceOpKind::MulScalar, {a2}, 3.0f * kGeluK1)},
                                        1.0f)},
                                kGeluK0);
                int half_a = add_op(TraceOpKind::MulScalar, {a}, 0.5f);
                int right = add_op(TraceOpKind::Mul, {half_a, add_op(TraceOpKind::Mul, {sech2, du})});
                int dgelu = add_op(TraceOpKind::Add, {left, right});
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, dgelu}));
                break;
            }

            case TraceOpKind::Exp: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int ac = add_op(TraceOpKind::Clamp, {a}, -88.0f, 88.0f);
                int ea = add_op(TraceOpKind::Exp, {ac});
                accum_grad(n.inputs[0], add_op(TraceOpKind::Mul, {dy, ea}));
                break;
            }

            case TraceOpKind::Log: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int ra = add_op(TraceOpKind::ReLU, {a});
                int as = emit_pos_denom(bwd, ra, kSafeEps, m);
                int q = add_op(TraceOpKind::Div, {dy, as});
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {ra, q, zero}));
                break;
            }

            case TraceOpKind::Sqrt: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int ra = add_op(TraceOpKind::ReLU, {a});
                int as = emit_pos_denom(bwd, ra, kSafeEps, m);
                int sa = add_op(TraceOpKind::Sqrt, {as});
                int sa_safe = emit_pos_denom(bwd, sa, kSafeEps, m);
                int q = add_op(TraceOpKind::MulScalar, {add_op(TraceOpKind::Div, {dy, sa_safe})}, 0.5f);
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {ra, q, zero}));
                break;
            }

            case TraceOpKind::Rsqrt: {
                int a = fwd_to_bwd.at(n.inputs[0]);
                int ra = add_op(TraceOpKind::ReLU, {a});
                int as = emit_pos_denom(bwd, ra, kSafeEps, m);
                int r = add_op(TraceOpKind::Rsqrt, {as});
                int r2 = add_op(TraceOpKind::Mul, {r, r});
                int r3 = add_op(TraceOpKind::Mul, {r2, r});
                int dr = add_op(TraceOpKind::MulScalar, {add_op(TraceOpKind::Mul, {dy, r3})}, -0.5f);
                int zero = emit_const(bwd, anchor, 0.0f, m);
                accum_grad(n.inputs[0], add_op(TraceOpKind::Where, {ra, dr, zero}));
                break;
            }

            default:
                throw std::runtime_error(std::string("TraceDAG::backward: unsupported op ") +
                                         op_kind_name(n.op));
        }
    }

    // 6. Emit final output Copy nodes for the requested primal inputs
    std::vector<int> out_slots;
    if (target_input_idx >= 0) {
        int fwd_in = primal_input_ids[static_cast<size_t>(target_input_idx)];
        auto it = adj.find(fwd_in);
        out_slots.push_back(add_op(TraceOpKind::Copy, {(it != adj.end()) ? it->second : emit_const(bwd, anchor, 0.0f, m)}));
    } else {
        for (int fwd_in : primal_input_ids) {
            auto it = adj.find(fwd_in);
            out_slots.push_back(add_op(TraceOpKind::Copy, {(it != adj.end()) ? it->second : emit_const(bwd, anchor, 0.0f, m)}));
        }
    }

    // 7. Dead-node elimination: keep all Input nodes + backward slice of `out_slots`
    std::vector<bool> keep(bwd.node_count(), false);
    for (const auto& n : bwd.nodes()) {
        if (n.op == TraceOpKind::Input) keep[static_cast<size_t>(n.id)] = true;
    }
    std::vector<int> stack = out_slots;
    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        if (cur < 0 || keep[static_cast<size_t>(cur)]) continue;
        keep[static_cast<size_t>(cur)] = true;
        for (int in_id : bwd.node(cur).inputs) {
            if (!keep[static_cast<size_t>(in_id)]) stack.push_back(in_id);
        }
    }

    TraceDAG pruned;
    std::unordered_map<int, int> prune_map;
    for (const auto& n : bwd.nodes()) {
        if (!keep[static_cast<size_t>(n.id)]) continue;
        std::vector<int> ins;
        for (int in_id : n.inputs) ins.push_back(prune_map.at(in_id));
        prune_map[n.id] = pruned.add_node(n.op, ins, n.scalar, n.buffer, n.device, n.dtype, n.rows, n.cols, n.scalar2);
    }
    for (auto& n : pruned.nodes()) {
        n.outputs.clear();
        n.is_live_at_end = false;
    }
    for (const auto& n : pruned.nodes()) {
        for (int in_id : n.inputs) pruned.nodes()[static_cast<size_t>(in_id)].outputs.push_back(n.id);
    }
    for (int os : out_slots) pruned.nodes()[static_cast<size_t>(prune_map.at(os))].is_live_at_end = true;
    return pruned;
}

#if BROTENSOR_HAS_BRASS_JIT
namespace {

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
    spirv::detail::Ctx c(kb);
    std::vector<brass::Value*> param_vals = spirv::detail::entry_params(c, fn);
    std::unordered_map<int, brass::Value*> vals;
    for (size_t i = 0; i < input_ids.size(); ++i) vals[input_ids[i]] = param_vals[i];
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) continue;
        std::vector<brass::Value*> ins;
        for (int in_id : n.inputs) ins.push_back(vals.at(in_id));
        vals[n.id] = spirv::detail::lane_op(c, n, ins);
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
    spirv::detail::Ctx c(kb);
    std::vector<brass::Value*> buf_params = spirv::detail::entry_params(c, fn);
    brass::Value* tid = kb.global_tid_x();
    std::unordered_map<int, brass::Value*> vals;
    for (size_t i = 0; i < input_ids.size(); ++i) vals[input_ids[i]] = kb.load_f32_indexed(buf_params[i], tid);
    for (const auto& n : lowered_dag.nodes()) {
        if (n.op == TraceOpKind::Input) continue;
        std::vector<brass::Value*> ins;
        for (int in_id : n.inputs) ins.push_back(vals.at(in_id));
        vals[n.id] = spirv::detail::lane_op(c, n, ins);
    }
    for (size_t j = 0; j < output_ids.size(); ++j) {
        kb.store_f32_indexed(buf_params[input_ids.size() + j], tid, vals.at(output_ids[j]));
    }
    c.b.build_ret_void();
    return fn;
}

} // namespace
#endif // BROTENSOR_HAS_BRASS_JIT

BackwardTraceVerificationReport verify_backward_trace_dag(const TraceDAG& bwd_dag) {
    BackwardTraceVerificationReport rep;
    rep.rawBackwardNodeCount = bwd_dag.node_count();
    TraceDAG lowered = canonicalize_trace_dag_for_codegen(bwd_dag);
    rep.loweredBackwardNodeCount = lowered.node_count();

#if BROTENSOR_HAS_BRASS_JIT
    // 1. MIR optimization equivalence proof (`verify_mir_equivalence`)
    {
        brass::Module mod("bwd_mir_equiv_mod");
        brass::Function* fn_unopt = build_scalar_backward_mir_fn(mod, "bwd_unopt", lowered);
        brass::Function* fn_opt = build_scalar_backward_mir_fn(mod, "bwd_opt", lowered);
        rep.unoptimizedMirInsts = count_mir_insts(*fn_unopt);
        spirv::detail::optimize_function(*fn_opt);
        rep.optimizedMirInsts = count_mir_insts(*fn_opt);

        brass::verify::EquivResult mir_res = brass::verify::verify_mir_equivalence(*fn_unopt, *fn_opt);
        rep.mirEquivProven = mir_res.is_equivalent();
        if (!rep.mirEquivProven && rep.diagnostic.empty()) {
            rep.diagnostic = "MIR equivalence failed: " + mir_res.diagnostic;
        }
    }

    // 2. Cross-ISA SPIR-V <-> PTX equivalence proof (`verify_spirv_ptx_equivalence`)
    std::string ptx_src;
    {
        brass::Module gpu_mod("bwd_gpu_equiv_mod");
        brass::Function* gpu_fn = build_gpu_equiv_kernel_fn(gpu_mod, lowered);
        spirv::detail::optimize_function(*gpu_fn);

        ptx_src = brass::target::PtxTarget::emit_function(*gpu_fn);
        rep.ptxByteCount = ptx_src.size();

        brass::verify::EquivResult gpu_res = brass::verify::verify_spirv_ptx_equivalence(*gpu_fn);
        rep.spirvPtxEquivProven = gpu_res.is_equivalent();
        if (!rep.spirvPtxEquivProven && rep.diagnostic.empty()) {
            rep.diagnostic = "SPIR-V/PTX equivalence failed: " + gpu_res.diagnostic;
        }
    }

    // 3. Static compute shader safety & NaN/division-by-zero proof (`prove_spirv_shader_safety`)
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
};

VerifiedBackwardTrace build_verified_backward_trace(const TraceDAG& forward_dag) {
    VerifiedBackwardTrace res;
    res.impl = std::make_shared<VerifiedBackwardTrace::Impl>();
    res.impl->forward_dag = forward_dag;
    res.impl->raw_backward_dag = forward_dag.backward(-1);
    res.impl->lowered_backward_dag = canonicalize_trace_dag_for_codegen(res.impl->raw_backward_dag);

    const ShapeMeta m = dominant_meta(forward_dag);
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
        spirv::detail::optimize_function(*gpu_fn);
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

namespace {

void bind_and_run_dag(const TraceDAG& template_dag,
                      const std::vector<Tensor>& primal_inputs,
                      const Tensor& grad_output,
                      std::vector<Tensor>& grad_inputs) {
    TraceDAG dag = template_dag;
    const Device target_dev = grad_output.device;

    size_t in_cursor = 0;
    size_t out_cursor = 0;
    for (auto& n : dag.nodes()) {
        n.device = target_dev;
        if (n.op == TraceOpKind::Input) {
            if (in_cursor < primal_inputs.size()) {
                const Tensor& p = primal_inputs[in_cursor++];
                n.buffer = p.data;
                n.rows = p.rows;
                n.cols = p.cols;
                n.dtype = p.dtype;
            } else {
                n.buffer = grad_output.data;
                n.rows = grad_output.rows;
                n.cols = grad_output.cols;
                n.dtype = grad_output.dtype;
            }
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

} // namespace

std::vector<Tensor> VerifiedBackwardTrace::execute(
    const std::vector<Tensor>& primal_inputs,
    const Tensor& grad_output) const {
    if (!impl) {
        throw std::runtime_error("VerifiedBackwardTrace::execute: uninitialized backward trace");
    }
    if (primal_inputs.size() != num_inputs) {
        throw std::invalid_argument("VerifiedBackwardTrace::execute: primal_inputs count mismatch");
    }
    std::vector<Tensor> grads;
    grads.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        grads.push_back(Tensor::empty_on(grad_output.device, grad_output.rows, grad_output.cols, Dtype::FP32));
    }
    bind_and_run_dag(impl->lowered_backward_dag, primal_inputs, grad_output, grads);
    return grads;
}

Tensor VerifiedBackwardTrace::execute_for_input(
    int input_idx,
    const std::vector<Tensor>& primal_inputs,
    const Tensor& grad_output) const {
    if (!impl || input_idx < 0 || static_cast<size_t>(input_idx) >= impl->per_input_lowered_dags.size()) {
        throw std::out_of_range("VerifiedBackwardTrace::execute_for_input: input_idx out of range");
    }
    std::vector<Tensor> grads;
    grads.push_back(Tensor::empty_on(grad_output.device, grad_output.rows, grad_output.cols, Dtype::FP32));
    bind_and_run_dag(impl->per_input_lowered_dags[static_cast<size_t>(input_idx)],
                     primal_inputs, grad_output, grads);
    return std::move(grads[0]);
}

} // namespace brotensor::jit
