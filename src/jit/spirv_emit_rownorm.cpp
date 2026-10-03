// Row-norm trace kernel as brass MIR (spirv_emit.h).
//
// A workgroup holds `rpb` rows of `tpr` threads (plan::row_norm_geometry);
// each row's threads walk it in strides of tpr * lanes columns, so every
// full access moves 16 bytes. The workgroup index is 2-D when there are more
// row groups than one grid dimension takes (65535): group = y * nx + x.
//
//   RMS    pass 1  pre chain, sum of squares, pass-one outputs stored
//          pass 2  x (read back, or the pre chain again), * rstd, gamma,
//                  post chain, stores
//   Mean   pass 1  pre chain, sum, pass-one outputs stored
//          pass 1b x again, sum of squared deviations from the mean
//          pass 2  as above with (x - mean) * rstd, gamma, beta
//
// LayerNorm's variance is the two-pass sum of squared deviations, never
// E[x^2] - E[x]^2 (which cancels catastrophically once a row's mean dwarfs
// its spread — the rule in CLAUDE.md, the recipe of brass's own fused
// LayerNorm kernels). The extra pass re-reads a row the workgroup just
// touched, out of cache.
//
// Reductions: a 32-lane subgroup butterfly, then, when a row spans several
// 32-thread groups, one shared-memory round across that row's groups. Every
// thread folds the partials itself, so the statistic needs no broadcast.

#include "spirv_emit.h"
#include "spirv_emit_detail.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit::spirv {

using namespace detail;

namespace {

class RowNormBuilder {
public:
    RowNormBuilder(const TraceDAG& dag, const plan::RowNormPlan& plan, int lanes, const char* name)
        : dag_(dag), plan_(plan), ew_(plan.ew), lanes_(lanes), name_(name), mod_(std::string("mod_") + name) {
        plan::row_norm_geometry(ew_.cols, ew_.rows, lanes, tpr_, rpb_);
        warps_ = tpr_ / 32;
        step_ = tpr_ * lanes;
    }

    brass::target::SpirvKernel build() {
        const std::size_t n_out = ew_.outputs.size();
        const std::size_t n_in = ew_.inputs.size();
        std::vector<Type> params(n_out + n_in, Type::ptr());
        Function* fn = mod_.create_function(name_, Type::void_type(), brass::Span<const Type>(params));
        KernelBuilder kb(mod_, fn);
        Ctx c(kb);
        c_ = &c;
        Builder& b = c.b;
        const std::vector<Value*> p = entry_params(c, fn);
        out_ptr_.assign(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(n_out));
        in_ptr_.assign(p.begin() + static_cast<std::ptrdiff_t>(n_out), p.end());
        if (any_bf16(ew_)) alloc_bitcast_slot(c, static_cast<std::uint32_t>(tpr_ * rpb_));

        // Shared partials, one array per reduction so no barrier is needed
        // between them.
        const int nred = (plan_.reduce == plan::RowReduce::Mean) ? 2 : 1;
        std::vector<Value*> red;
        if (warps_ > 1) {
            for (int r = 0; r < nred; ++r) {
                red.push_back(kb.shared_alloc_f32(static_cast<std::uint32_t>(plan::kRowThreads / 32)));
            }
        }

        tid_ = kb.tid_x();
        Value* group = b.build_add(b.build_mul(kb.ctaid_y(), kb.nctaid_x()), kb.ctaid_x());
        Value* sub = tid_;
        local_ = c.i32(0);
        Value* row = group;
        if (rpb_ > 1) {
            local_ = b.build_lshr(tid_, c.i32(log2_exact(tpr_)));
            sub = b.build_and(tid_, c.i32(tpr_ - 1));
            row = b.build_add(b.build_mul(group, c.i32(rpb_)), local_);
        }
        // rpb divides the row count, so this exit is uniform across the
        // workgroup and the barriers below stay safe.
        kb.if_then(b.build_uge(row, c.i32(ew_.rows)), [&] { b.build_ret_void(); });

        c0_ = lanes_ > 1 ? b.build_mul(sub, c.i32(lanes_)) : sub;
        rowbase_ = b.build_mul(row, c.i32(ew_.cols));
        for (const auto& s : ew_.inputs) row_bytes(s.dtype);
        for (const auto& s : ew_.outputs) row_bytes(s.dtype);
        Value* D = c.i32(ew_.cols);
        const float inv_d = 1.0f / static_cast<float>(ew_.cols);

        Value* mean = nullptr;
        Value* rstd = nullptr;
        if (plan_.reduce == plan::RowReduce::Mean) {
            // pass 1: sum
            Value* sum = kb.for_range_reduce(c0_, D, c.i32(step_), c.f32(0.0f), [&](Value* col, Value* acc) {
                Offsets o;
                const std::vector<Value*> x = pass_one_x(o, col);
                for (Value* xj : x) acc = b.build_add(acc, xj);
                return acc;
            });
            mean = b.build_mul(reduce(sum, red.empty() ? nullptr : red[0]), c.f32(inv_d));
            // pass 1b: squared deviations about that mean
            Value* sq = kb.for_range_reduce(c0_, D, c.i32(step_), c.f32(0.0f), [&](Value* col, Value* acc) {
                Offsets o;
                for (Value* xj : reload_x(o, col, nullptr)) {
                    Value* d = b.build_sub(xj, mean);
                    acc = b.build_fma_f32(d, d, acc);
                }
                return acc;
            });
            Value* var = b.build_mul(reduce(sq, red.empty() ? nullptr : red[1]), c.f32(inv_d));
            rstd = kb.rsqrt_approx(b.build_add(var, c.f32(plan_.eps)));
        } else {
            Value* sq = kb.for_range_reduce(c0_, D, c.i32(step_), c.f32(0.0f), [&](Value* col, Value* acc) {
                Offsets o;
                for (Value* xj : pass_one_x(o, col)) acc = b.build_fma_f32(xj, xj, acc);
                return acc;
            });
            Value* ms = b.build_mul(reduce(sq, red.empty() ? nullptr : red[0]), c.f32(inv_d));
            rstd = kb.rsqrt_approx(b.build_add(ms, c.f32(plan_.eps)));
        }

        // pass 2: normalise, post chain, store
        kb.for_range(c0_, D, c.i32(step_), [&](Value* col) {
            Offsets o;
            ValueMap vals;
            const std::vector<Value*> x = reload_x(o, col, &vals);
            std::vector<Value*> gam, bet;
            if (plan_.gamma_input >= 0) {
                gam = vals.at(ew_.inputs[static_cast<std::size_t>(plan_.gamma_input)].node_id);
            }
            if (plan_.beta_input >= 0) {
                bet = vals.at(ew_.inputs[static_cast<std::size_t>(plan_.beta_input)].node_id);
            }
            std::vector<Value*> norm(static_cast<std::size_t>(lanes_));
            for (int j = 0; j < lanes_; ++j) {
                const std::size_t js = static_cast<std::size_t>(j);
                Value* v = mean ? b.build_mul(b.build_sub(x[js], mean), rstd) : b.build_mul(x[js], rstd);
                if (!gam.empty()) {
                    v = bet.empty() ? b.build_mul(v, gam[js]) : b.build_fma_f32(v, gam[js], bet[js]);
                }
                norm[js] = v;
            }
            vals[plan_.norm_node] = std::move(norm);
            vals = eval_chain(c, dag_, lanes_, std::move(vals), &plan_.post_ids);
            for (std::size_t m = 0; m < ew_.outputs.size(); ++m) {
                if (m < plan_.pre_output.size() && plan_.pre_output[m]) continue;   // written in pass one
                const plan::BufferSpec& s = ew_.outputs[m];
                store_lanes(c, s.dtype, lanes_, b.build_add(out_ptr_[m], full_off(o, col, s.dtype)),
                            vals.at(s.node_id));
            }
        });
        b.build_ret_void();
        return compile(*fn, static_cast<std::uint32_t>(tpr_ * rpb_));
    }

private:
    // Byte offsets materialised for the column one loop iteration is on.
    struct Offsets {
        std::unordered_map<int, Value*> col, full;
    };

    Value* row_bytes(Dtype dt) {
        const int e = plan::elem_bytes(dt);
        auto it = rowbyte_.find(e);
        if (it != rowbyte_.end()) return it->second;
        Value* v = byte_offset(*c_, rowbase_, e);
        rowbyte_[e] = v;
        return v;
    }

    Value* col_off(Offsets& o, Value* col, Dtype dt) {
        const int e = plan::elem_bytes(dt);
        auto it = o.col.find(e);
        if (it != o.col.end()) return it->second;
        Value* v = byte_offset(*c_, col, e);
        o.col[e] = v;
        return v;
    }

    Value* full_off(Offsets& o, Value* col, Dtype dt) {
        const int e = plan::elem_bytes(dt);
        auto it = o.full.find(e);
        if (it != o.full.end()) return it->second;
        Value* v = c_->b.build_add(row_bytes(dt), col_off(o, col, dt));
        o.full[e] = v;
        return v;
    }

    // Loads the inputs `want` selects at `col`.
    ValueMap load_inputs(Offsets& o, Value* col, const std::vector<char>& want) {
        ValueMap vals;
        for (std::size_t k = 0; k < ew_.inputs.size(); ++k) {
            if (!want.empty() && !want[k]) continue;
            const plan::BufferSpec& s = ew_.inputs[k];
            if (s.bcast == BroadcastKind::Scalar) {
                Value* v = load_lanes(*c_, s.dtype, 1, in_ptr_[k])[0];
                vals[s.node_id] = std::vector<Value*>(static_cast<std::size_t>(lanes_), v);
                continue;
            }
            Value* off = s.bcast == BroadcastKind::Row ? col_off(o, col, s.dtype) : full_off(o, col, s.dtype);
            vals[s.node_id] = load_lanes(*c_, s.dtype, lanes_, c_->b.build_add(in_ptr_[k], off));
        }
        return vals;
    }

    // Pass one: the pre chain at `col`, its outputs stored (final already —
    // and what lets the later passes read x back instead of replaying it).
    std::vector<Value*> pass_one_x(Offsets& o, Value* col) {
        ValueMap vals = eval_chain(*c_, dag_, lanes_, load_inputs(o, col, plan_.pre_input), &plan_.pre_ids);
        for (std::size_t m = 0; m < ew_.outputs.size(); ++m) {
            if (m >= plan_.pre_output.size() || !plan_.pre_output[m]) continue;
            const plan::BufferSpec& s = ew_.outputs[m];
            store_lanes(*c_, s.dtype, lanes_, c_->b.build_add(out_ptr_[m], full_off(o, col, s.dtype)),
                        vals.at(s.node_id));
        }
        return vals.at(plan_.x_node);
    }

    // x at `col` after pass one: read back from the output pass one wrote,
    // or (no pre output) the pre chain replayed from the inputs. With `vals`,
    // also loads what pass two needs and leaves it there.
    std::vector<Value*> reload_x(Offsets& o, Value* col, ValueMap* vals) {
        ValueMap local;
        ValueMap& v = vals ? *vals : local;
        if (plan_.x_output >= 0) {
            if (vals) v = load_inputs(o, col, plan_.post_input);
            const plan::BufferSpec& xs = ew_.outputs[static_cast<std::size_t>(plan_.x_output)];
            std::vector<Value*> x = load_lanes(
                *c_, xs.dtype, lanes_,
                c_->b.build_add(out_ptr_[static_cast<std::size_t>(plan_.x_output)], full_off(o, col, xs.dtype)));
            v[plan_.x_node] = x;
            return x;
        }
        // Without the reload the plan guarantees pass one stored nothing, so
        // the inputs are still what pass one read.
        v = load_inputs(o, col, vals ? plan_.post_input : plan_.pre_input);
        v = eval_chain(*c_, dag_, lanes_, std::move(v), &plan_.pre_ids);
        return v.at(plan_.x_node);
    }

    // The row total of `v` in every thread of the row.
    Value* reduce(Value* v, Value* red) {
        KernelBuilder& kb = c_->kb;
        Builder& b = c_->b;
        for (uint32_t m = 16; m >= 1; m >>= 1) v = b.build_add(v, kb.shfl_bfly_f32(v, m));
        if (warps_ == 1) return v;
        Value* lane = b.build_and(tid_, c_->i32(31));
        Value* warp = b.build_lshr(tid_, c_->i32(5));
        kb.if_then(b.build_eq(lane, c_->i32(0)), [&] { kb.shared_store_f32_indexed(red, warp, v); });
        kb.sync();
        Value* base = rpb_ == 1 ? c_->i32(0) : b.build_mul(local_, c_->i32(warps_));
        Value* total = nullptr;
        for (int w = 0; w < warps_; ++w) {
            Value* x = kb.shared_load_f32_indexed(red, b.build_add(base, c_->i32(w)));
            total = total ? b.build_add(total, x) : x;
        }
        return total;
    }

    const TraceDAG& dag_;
    const plan::RowNormPlan& plan_;
    const plan::ElementwisePlan& ew_;
    int lanes_;
    const char* name_;
    brass::Module mod_;
    int tpr_ = 32, rpb_ = 1, warps_ = 1, step_ = 32;
    Ctx* c_ = nullptr;
    std::vector<Value*> out_ptr_, in_ptr_;
    Value* tid_ = nullptr;
    Value* local_ = nullptr;
    Value* c0_ = nullptr;
    Value* rowbase_ = nullptr;
    std::unordered_map<int, Value*> rowbyte_;
};

}  // namespace

Kernels emit_row_norm(const TraceDAG& dag, const plan::RowNormPlan& plan) {
    Kernels k;
    if (plan.ew.vec > 1) k.vec = RowNormBuilder(dag, plan, plan.ew.vec, kEntryRowNorm).build();
    k.scalar = RowNormBuilder(dag, plan, 1, kEntryRowNormScalar).build();
    return k;
}

}  // namespace brotensor::jit::spirv
