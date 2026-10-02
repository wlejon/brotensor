// ─── Eager trace execution — backends with no trace compiler ────────────────
//
// The trace JIT compiles a DAG to host code (brass, CPU) or to PTX (CUDA).
// HIP has neither: there is no GPU code generator for AMD here, and the CPU
// compiler must never see a HIP DAG — its host code would walk device
// pointers outside the HIP stream's ordering (and fault outright on a
// discrete GPU). A HIP trace is therefore executed by replaying the DAG one
// node at a time through the ordinary device-dispatched ops, on the current
// stream, in program order.
//
// That keeps every trace correct on HIP — same results as the eager op
// sequence the trace replaces, same TraceHandle replay/rebind contract — but
// it is not fused: one launch per traced op, and an intermediate the fused
// kernel would keep in registers is a scratch tensor here (allocated per
// execute(), released after its last consumer). launch_count() and
// fusion_name() ("eager-unfused") say so.
//
// Operand rules mirror the fused compilers: every value carries its node's
// (rows, cols); a binary operand may be the full shape, a (1, cols) row or a
// (1, 1) scalar; a norm/modulate gain is a length-cols vector; an operand of
// another dtype is cast to the node's dtype first (math is FP32 inside each
// op either way). Anything else is rejected when the trace is compiled, so
// end_trace() throws and the caller takes its eager path.

#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"

#include <brotensor/ops.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::jit::eager {

namespace {

[[noreturn]] void reject(const TraceNode& n, const char* why) {
    throw std::runtime_error(std::string("brotensor::jit: eager trace execution cannot run ") +
                             op_kind_name(n.op) + " on " + to_string(n.device) + ": " + why);
}

bool is_unary(TraceOpKind op) {
    switch (op) {
        case TraceOpKind::AddScalar: case TraceOpKind::SubScalar:
        case TraceOpKind::MulScalar: case TraceOpKind::DivScalar:
        case TraceOpKind::ScalarSub: case TraceOpKind::SiLU:
        case TraceOpKind::GELU: case TraceOpKind::ReLU:
        case TraceOpKind::Tanh: case TraceOpKind::Sigmoid:
        case TraceOpKind::Copy:
            return true;
        default:
            return false;
    }
}

bool is_binary(TraceOpKind op) {
    return op == TraceOpKind::Add || op == TraceOpKind::Sub ||
           op == TraceOpKind::Mul || op == TraceOpKind::Div;
}

int64_t numel(int rows, int cols) { return static_cast<int64_t>(rows) * cols; }

// Shape and op checks, done once at compile time so a trace this executor
// cannot run fails in end_trace() rather than on some later replay.
void validate(const TraceDAG& dag) {
    const auto& nodes = dag.nodes();
    if (nodes.empty()) throw std::runtime_error("brotensor::jit: empty trace");
    const Device dev = nodes.front().device;
    for (const TraceNode& n : nodes) {
        if (n.device != dev) reject(n, "the trace spans more than one device");
        if (n.dtype != Dtype::FP32 && n.dtype != Dtype::FP16 && n.dtype != Dtype::BF16) {
            reject(n, "only FP32 / FP16 / BF16 values are supported");
        }
        if (n.op == TraceOpKind::Input) continue;
        auto operand = [&](std::size_t k) -> const TraceNode& {
            if (k >= n.inputs.size() || n.inputs[k] < 0 ||
                n.inputs[k] >= static_cast<int>(nodes.size())) {
                reject(n, "missing operand");
            }
            return nodes[static_cast<std::size_t>(n.inputs[k])];
        };
        auto vector_like = [&](const TraceNode& v) {
            return numel(v.rows, v.cols) == n.cols && (v.rows == 1 || v.cols == 1);
        };
        if (is_unary(n.op)) {
            const TraceNode& a = operand(0);
            if (numel(a.rows, a.cols) != numel(n.rows, n.cols)) reject(n, "operand shape mismatch");
        } else if (is_binary(n.op)) {
            for (std::size_t k = 0; k < 2; ++k) {
                const TraceNode& a = operand(k);
                const BroadcastKind bk = broadcast_of(a.rows, a.cols, n.rows, n.cols);
                if (bk == BroadcastKind::Full && numel(a.rows, a.cols) != numel(n.rows, n.cols)) {
                    reject(n, "operand is neither the full shape, a row nor a scalar");
                }
            }
        } else if (n.op == TraceOpKind::FMA) {
            for (std::size_t k = 0; k < 3; ++k) {
                if (numel(operand(k).rows, operand(k).cols) != numel(n.rows, n.cols)) {
                    reject(n, "FMA operands must be the full shape");
                }
            }
        } else if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm ||
                   n.op == TraceOpKind::Modulate) {
            if (numel(operand(0).rows, operand(0).cols) != numel(n.rows, n.cols)) {
                reject(n, "operand shape mismatch");
            }
            const std::size_t want = n.op == TraceOpKind::Modulate ? 3 : n.inputs.size();
            for (std::size_t k = 1; k < want; ++k) {
                if (!vector_like(operand(k))) reject(n, "gain / shift must be a length-cols vector");
            }
        } else {
            reject(n, "no eager form");
        }
    }
}

Tensor view_of(const TraceNode& n, void* buffer) {
    return Tensor::view(n.device, buffer, n.rows, n.cols, n.dtype);
}

// `t` reshaped to (rows, cols) without copying (same element count).
Tensor reshaped(const Tensor& t, int rows, int cols) {
    return Tensor::view(t.device, t.data, rows, cols, t.dtype);
}

Tensor as_dtype(const Tensor& t, Dtype dt) {
    if (t.dtype == dt) return reshaped(t, t.rows, t.cols);
    Tensor out;
    brotensor::cast(t, out, dt);
    return out;
}

// Copies `src` into `dst` unless they already are the same storage.
void assign(Tensor& dst, const Tensor& src) {
    if (dst.data == src.data) return;
    brotensor::copy_d2d(src, 0, dst, 0, static_cast<int>(dst.size()));
}

// A (1, cols) row or (1, 1) scalar operand spread to the full (rows, cols).
Tensor expand(const Tensor& small, int rows, int cols, Dtype dt) {
    Tensor full = Tensor::zeros_on(small.device, rows, cols, dt);
    Tensor s = as_dtype(small, dt);
    if (small.size() == 1) {
        Tensor col = reshaped(full, rows * cols, 1);
        brotensor::add_row_bias_inplace(col, reshaped(s, 1, 1));
    } else {
        brotensor::add_row_bias_inplace(full, reshaped(s, 1, cols));
    }
    return full;
}

class Program {
public:
    explicit Program(const TraceDAG& dag) : nodes_(dag.nodes()) {
        last_use_.assign(nodes_.size(), -1);
        for (const TraceNode& n : nodes_) {
            for (int in : n.inputs) {
                if (in >= 0) last_use_[static_cast<std::size_t>(in)] = n.id;
            }
        }
        for (const TraceNode& n : nodes_) {
            if (n.op == TraceOpKind::Input) continue;
            const bool needed = n.buffer != nullptr || last_use_[static_cast<std::size_t>(n.id)] >= 0;
            if (needed) ++launches_;
        }
    }

    std::size_t launches() const { return launches_; }

    // Input buffers in node order, then the buffers of live-at-end nodes in
    // node order — the order TraceCompiler::compile_and_cache rebinds in.
    std::shared_ptr<Program> rebound(const std::vector<void*>& in,
                                     const std::vector<void*>& out) const {
        auto p = std::make_shared<Program>(*this);
        std::size_t i = 0, o = 0;
        for (TraceNode& n : p->nodes_) {
            if (n.op == TraceOpKind::Input) {
                if (i < in.size()) n.buffer = in[i++];
            }
            if (n.is_live_at_end) {
                if (o < out.size()) n.buffer = out[o++];
            }
        }
        return p;
    }

    void run() const {
        std::vector<Tensor> vals(nodes_.size());
        for (const TraceNode& n : nodes_) {
            const std::size_t id = static_cast<std::size_t>(n.id);
            if (n.op == TraceOpKind::Input) {
                vals[id] = view_of(n, n.buffer);
                continue;
            }
            // A value nothing reads and nothing stores is dead code.
            if (n.buffer == nullptr && last_use_[id] < 0) continue;

            Tensor dst = n.buffer ? view_of(n, n.buffer)
                                  : Tensor::empty_on(n.device, n.rows, n.cols, n.dtype);
            evaluate(n, vals, dst);
            vals[id] = std::move(dst);

            for (int in : n.inputs) {
                if (in >= 0 && last_use_[static_cast<std::size_t>(in)] == n.id) {
                    vals[static_cast<std::size_t>(in)] = Tensor();
                }
            }
        }
    }

private:
    const Tensor& arg(const TraceNode& n, std::size_t k, const std::vector<Tensor>& vals) const {
        return vals[static_cast<std::size_t>(n.inputs[k])];
    }

    // Operand k as a full-shape tensor of the node's dtype.
    Tensor full_arg(const TraceNode& n, std::size_t k, const std::vector<Tensor>& vals) const {
        const Tensor& a = arg(n, k, vals);
        if (numel(a.rows, a.cols) != numel(n.rows, n.cols)) return expand(a, n.rows, n.cols, n.dtype);
        return reshaped(as_dtype(a, n.dtype), n.rows, n.cols);
    }

    // A gain / shift vector as (1, cols) in the node's dtype.
    Tensor row_arg(const TraceNode& n, std::size_t k, const std::vector<Tensor>& vals) const {
        return reshaped(as_dtype(arg(n, k, vals), n.dtype), 1, n.cols);
    }

    void evaluate(const TraceNode& n, const std::vector<Tensor>& vals, Tensor& dst) const {
        switch (n.op) {
            case TraceOpKind::Copy:
                assign(dst, full_arg(n, 0, vals));
                return;
            case TraceOpKind::AddScalar:
            case TraceOpKind::SubScalar:
                assign(dst, full_arg(n, 0, vals));
                brotensor::add_scalar_inplace(dst, n.op == TraceOpKind::AddScalar ? n.scalar : -n.scalar);
                return;
            case TraceOpKind::MulScalar:
            case TraceOpKind::DivScalar:
                assign(dst, full_arg(n, 0, vals));
                brotensor::scale_inplace(dst, n.op == TraceOpKind::MulScalar ? n.scalar : 1.0f / n.scalar);
                return;
            case TraceOpKind::ScalarSub:
                assign(dst, full_arg(n, 0, vals));
                brotensor::scale_inplace(dst, -1.0f);
                brotensor::add_scalar_inplace(dst, n.scalar);
                return;
            case TraceOpKind::SiLU:    brotensor::silu_forward(full_arg(n, 0, vals), dst); return;
            case TraceOpKind::GELU:    brotensor::gelu_forward(full_arg(n, 0, vals), dst); return;
            case TraceOpKind::ReLU:    brotensor::relu_forward(full_arg(n, 0, vals), dst); return;
            case TraceOpKind::Tanh:    brotensor::tanh_forward(full_arg(n, 0, vals), dst); return;
            case TraceOpKind::Sigmoid: brotensor::sigmoid_forward(full_arg(n, 0, vals), dst); return;
            case TraceOpKind::Add:
            case TraceOpKind::Sub:
            case TraceOpKind::Mul:
            case TraceOpKind::Div:
                binary(n, vals, dst);
                return;
            case TraceOpKind::FMA: {
                Tensor t = full_arg(n, 0, vals);
                Tensor prod = Tensor::empty_on(n.device, n.rows, n.cols, n.dtype);
                assign(prod, t);
                brotensor::mul_inplace(prod, full_arg(n, 1, vals));
                brotensor::add_inplace(prod, full_arg(n, 2, vals));
                assign(dst, prod);
                return;
            }
            case TraceOpKind::RMSNorm: {
                Tensor g = n.inputs.size() > 1 ? row_arg(n, 1, vals) : ones(n);
                into(dst, full_arg(n, 0, vals), [&](const Tensor& x, Tensor& y) {
                    brotensor::rms_norm_forward(x, g, n.scalar, y);
                });
                return;
            }
            case TraceOpKind::LayerNorm: {
                Tensor g = n.inputs.size() > 1 ? row_arg(n, 1, vals) : ones(n);
                Tensor b = n.inputs.size() > 2 ? row_arg(n, 2, vals)
                                               : Tensor::zeros_on(n.device, 1, n.cols, n.dtype);
                into(dst, full_arg(n, 0, vals), [&](const Tensor& x, Tensor& y) {
                    brotensor::layernorm_forward_inference_batched(x, g, b, y, n.scalar);
                });
                return;
            }
            case TraceOpKind::Modulate: {
                Tensor sc = row_arg(n, 1, vals);
                Tensor sh = row_arg(n, 2, vals);
                into(dst, full_arg(n, 0, vals), [&](const Tensor& x, Tensor& y) {
                    brotensor::modulate(x, sc, sh, y);
                });
                return;
            }
            default:
                reject(n, "no eager form");
        }
    }

    Tensor ones(const TraceNode& n) const {
        Tensor t = Tensor::zeros_on(n.device, 1, n.cols, n.dtype);
        brotensor::add_scalar_inplace(t, 1.0f);
        return t;
    }

    // Runs a row op whose output may not alias its input: through a scratch
    // tensor when dst is the input's storage (an in-place target).
    template <class F>
    void into(Tensor& dst, const Tensor& x, F&& op) const {
        if (dst.data != x.data) {
            op(x, dst);
            return;
        }
        Tensor tmp = Tensor::empty_on(dst.device, dst.rows, dst.cols, dst.dtype);
        op(x, tmp);
        assign(dst, tmp);
    }

    void binary(const TraceNode& n, const std::vector<Tensor>& vals, Tensor& dst) const {
        const Tensor& ra = arg(n, 0, vals);
        const Tensor& rb = arg(n, 1, vals);
        // Multiply by a row / scalar has a kernel of its own; no expansion.
        if (n.op == TraceOpKind::Mul) {
            const bool a_small = numel(ra.rows, ra.cols) != numel(n.rows, n.cols);
            const bool b_small = numel(rb.rows, rb.cols) != numel(n.rows, n.cols);
            if (a_small != b_small) {
                const Tensor big = full_arg(n, a_small ? 1 : 0, vals);
                const Tensor& small = a_small ? ra : rb;
                const Tensor v = as_dtype(small, n.dtype);
                Tensor tmp = dst.data == big.data
                                 ? Tensor::empty_on(n.device, n.rows, n.cols, n.dtype)
                                 : reshaped(dst, n.rows, n.cols);
                if (small.size() == 1) {
                    Tensor y = reshaped(tmp, n.rows * n.cols, 1);
                    brotensor::broadcast_mul(reshaped(big, n.rows * n.cols, 1), reshaped(v, 1, 1), y);
                } else {
                    brotensor::broadcast_mul(big, reshaped(v, 1, n.cols), tmp);
                }
                assign(dst, tmp);
                return;
            }
        }
        Tensor a = full_arg(n, 0, vals);
        Tensor b = full_arg(n, 1, vals);
        // dst = a op b, with dst possibly the storage of a (an in-place
        // target) or of b. Work in dst when it is a's storage or neither.
        Tensor acc = dst.data == b.data && dst.data != a.data
                         ? Tensor::empty_on(n.device, n.rows, n.cols, n.dtype)
                         : reshaped(dst, n.rows, n.cols);
        assign(acc, a);
        switch (n.op) {
            case TraceOpKind::Add: brotensor::add_inplace(acc, b); break;
            case TraceOpKind::Sub: brotensor::axpby_inplace(acc, b, 1.0f, -1.0f); break;
            case TraceOpKind::Mul: brotensor::mul_inplace(acc, b); break;
            case TraceOpKind::Div: brotensor::div_inplace(acc, b); break;
            default: reject(n, "not a binary op");
        }
        assign(dst, acc);
    }

    std::vector<TraceNode> nodes_;
    std::vector<int> last_use_;
    std::size_t launches_ = 0;
};

std::shared_ptr<TraceHandleImpl> make_handle(std::shared_ptr<const Program> prog) {
    auto h = std::make_shared<TraceHandleImpl>();
    h->execute_fn = [prog] { prog->run(); };
    h->rebind_fn = [prog](const std::vector<void*>& in, const std::vector<void*>& out) {
        return make_handle(prog->rebound(in, out));
    };
    h->launch_count = prog->launches();
    h->fusion_name = "eager-unfused";
    return h;
}

}  // namespace

std::shared_ptr<TraceHandleImpl> compile_eager(const TraceDAG& dag) {
    const auto t0 = std::chrono::steady_clock::now();
    validate(dag);
    auto h = make_handle(std::make_shared<const Program>(dag));
    h->compile_us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
    return h;
}

}  // namespace brotensor::jit::eager
