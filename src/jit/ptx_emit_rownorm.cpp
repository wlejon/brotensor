#include "ptx_emit.h"
#include "ptx_emit_detail.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit::ptx {

using namespace brotensor::jit::ptx::detail;

namespace {

// ─── row-norm kernel ────────────────────────────────────────────────────────
//
// A block holds `rpb` rows of `tpr` threads; each group of tpr threads walks
// its row in strides of tpr * lanes columns, so every access moves 16 bytes.
// row_norm_geometry picks the pair — wide rows get all 256 threads and one
// row per block, narrow ones pack several rows into a block instead of idling
// most of it.
//
//   RMS    pass 1   pre chain (the nodes the reduction depends on), sum of
//                   squares, pass-one outputs stored
//          pass 2   x (read back, or the pre chain again — holding a whole
//                   row in registers would cap D), * rstd, gamma, post
//                   chain, stores
//   Mean   pass 1   pre chain, sum, pass-one outputs stored
//          pass 1b  x again, sum of squared deviations from the mean
//          pass 2   as above with (x - mean) * rstd, gamma, beta
//
// Each reduction is a warp butterfly plus, when a row spans several warps,
// one shared-memory round across that row's warps.
//
// LayerNorm's variance is the two-pass sum of squared deviations, never
// E[x^2] - E[x]^2 (which cancels catastrophically once a row's mean dwarfs
// its spread — the rule in CLAUDE.md), with the same structure as the Vulkan
// emitter (spirv_emit_rownorm.cpp). The extra pass re-reads a row the block
// just touched, out of L2.
//
// The pre chain is what lets `x += p; rms_norm(x)` fuse: the reduction needs
// the updated x that this same kernel writes. It also means the row is read
// more than once (twice for RMSNorm, three times for LayerNorm), but every
// read after the first is the same 8-64 KB the block just touched, so it
// comes out of L2 rather than HBM.
//
// D, the row count and eps are baked in as immediates. The trace signature
// already carries the shape, so a different shape is a different trace and a
// different compile — and the kernel loses every parameter decode and every
// runtime bound it would otherwise carry.

std::string emit_rn_entry(const TraceDAG& dag, const RowNormPlan& plan, int lanes,
                          const std::string& name) {
    const ElementwisePlan& ew = plan.ew;
    const int D = ew.cols;
    const int R = ew.rows;
    const bool mean = (plan.reduce == RowReduce::Mean);
    int tpr = kRowThreads, rpb = 1;
    row_norm_geometry(D, R, lanes, tpr, rpb);
    // tpr is never below 32, so a row's threads always fill whole warps and
    // the butterfly below is always a full-warp one.
    const int warps = tpr / 32;           // warps cooperating on one row
    const int step = tpr * lanes;

    Alloc a;
    std::stringstream b;

    const std::size_t n_out = ew.outputs.size();
    const std::size_t n_in = ew.inputs.size();

    std::vector<std::string> out_ptr(n_out), in_ptr(n_in);
    for (std::size_t m = 0; m < n_out; ++m) {
        out_ptr[m] = a.RD();
        b << "    ld.param.u64 " << out_ptr[m] << ", [" << name << "_out_" << m << "];\n";
    }
    for (std::size_t k = 0; k < n_in; ++k) {
        in_ptr[k] = a.RD();
        b << "    ld.param.u64 " << in_ptr[k] << ", [" << name << "_in_" << k << "];\n";
    }

    // tid splits into (which row of this block, which thread within the row).
    const std::string r_tid = a.R(), r_row = a.R(), r_c0 = a.R();
    const std::string r_sub = a.R(), r_local = a.R();
    b << "    mov.u32 " << r_tid << ", %tid.x;\n";
    b << "    mov.u32 " << r_row << ", %ctaid.x;\n";
    if (rpb == 1) {
        b << "    mov.u32 " << r_local << ", 0;\n";
        b << "    mov.u32 " << r_sub << ", " << r_tid << ";\n";
    } else {
        // tpr is a power of two, so both halves are a shift and a mask.
        int log_tpr = 0;
        while ((1 << log_tpr) < tpr) ++log_tpr;
        b << "    shr.u32 " << r_local << ", " << r_tid << ", " << log_tpr << ";\n";
        b << "    and.b32 " << r_sub << ", " << r_tid << ", " << (tpr - 1) << ";\n";
        b << "    mad.lo.u32 " << r_row << ", " << r_row << ", " << rpb << ", "
          << r_local << ";\n";
    }
    {
        // rpb divides R (row_norm_geometry guarantees it), so this exit is
        // uniform across the block and the barrier below stays safe.
        const std::string p = a.P();
        b << "    setp.ge.u32 " << p << ", " << r_row << ", " << R << ";\n";
        b << "    @" << p << " bra $L_" << name << "_done;\n";
    }
    // This thread's first column.
    if (lanes == 1) {
        b << "    mov.u32 " << r_c0 << ", " << r_sub << ";\n";
    } else {
        b << "    mul.lo.u32 " << r_c0 << ", " << r_sub << ", " << lanes << ";\n";
    }

    // Element index of column 0 of this row, and its byte offset per dtype.
    const std::string r_rowbase = a.R();
    b << "    mul.lo.u32 " << r_rowbase << ", " << r_row << ", " << D << ";\n";

    std::unordered_map<int, std::string> rowbyte;
    auto row_bytes = [&](Dtype dt) -> std::string {
        const int e = elem_bytes(dt);
        auto it = rowbyte.find(e);
        if (it != rowbyte.end()) return it->second;
        const std::string rd = a.RD();
        b << "    mul.wide.u32 " << rd << ", " << r_rowbase << ", " << e << ";\n";
        rowbyte[e] = rd;
        return rd;
    };
    // Every element size the trace touches, resolved before either loop so the
    // stride loops do not recompute a loop-invariant address.
    for (const BufferSpec& s : ew.inputs) row_bytes(s.dtype);
    for (const BufferSpec& s : ew.outputs) row_bytes(s.dtype);

    // Per-pass cache of the byte offsets materialised for the column the
    // stride loop is on. The two passes are separate PTX loops, so each gets
    // its own.
    struct Offsets {
        std::unordered_map<int, std::string> col, full;
    };

    auto col_off = [&](Offsets& o, const std::string& r_c, Dtype dt) -> std::string {
        const int e = elem_bytes(dt);
        auto it = o.col.find(e);
        if (it != o.col.end()) return it->second;
        const std::string rd = a.RD();
        b << "    mul.wide.u32 " << rd << ", " << r_c << ", " << e << ";\n";
        o.col[e] = rd;
        return rd;
    };
    auto full_off = [&](Offsets& o, const std::string& r_c, Dtype dt) -> std::string {
        const int e = elem_bytes(dt);
        auto it = o.full.find(e);
        if (it != o.full.end()) return it->second;
        const std::string rb = row_bytes(dt);
        const std::string co = col_off(o, r_c, dt);
        const std::string rd = a.RD();
        b << "    add.u64 " << rd << ", " << rb << ", " << co << ";\n";
        o.full[e] = rd;
        return rd;
    };

    // Loads the inputs `want` selects (empty = all) at the column in `r_c`.
    auto emit_inputs = [&](Offsets& o, const std::string& r_c,
                           const std::vector<char>& want) -> ValueMap {
        ValueMap vals;
        for (std::size_t k = 0; k < n_in; ++k) {
            if (!want.empty() && !want[k]) continue;
            const BufferSpec& s = ew.inputs[k];
            const int ln = (s.bcast == BroadcastKind::Scalar) ? 1 : lanes;
            std::string addr;
            if (s.bcast == BroadcastKind::Scalar) {
                addr = in_ptr[k];
            } else {
                const std::string off = (s.bcast == BroadcastKind::Row)
                                            ? col_off(o, r_c, s.dtype)
                                            : full_off(o, r_c, s.dtype);
                addr = a.RD();
                b << "    add.u64 " << addr << ", " << in_ptr[k] << ", " << off << ";\n";
            }
            std::vector<std::string> v = emit_load(b, a, s.dtype, ln, addr);
            if (s.bcast == BroadcastKind::Scalar) {
                const std::string only = v[0];
                v.assign(static_cast<std::size_t>(lanes), only);
            }
            vals[s.node_id] = std::move(v);
        }
        return vals;
    };

    // x at the column in `r_c` after pass one: read back from the output
    // pass one wrote, or (no pre output) the pre chain replayed from the
    // inputs. With `vals`, also loads what pass two needs and leaves it there.
    auto reload_x = [&](Offsets& o, const std::string& r_c,
                        ValueMap* vals) -> std::vector<std::string> {
        ValueMap local;
        ValueMap& v = vals ? *vals : local;
        if (plan.x_output >= 0) {
            if (vals) v = emit_inputs(o, r_c, plan.post_input);
            const BufferSpec& xs = ew.outputs[static_cast<std::size_t>(plan.x_output)];
            const std::string off = full_off(o, r_c, xs.dtype);
            const std::string addr = a.RD();
            b << "    add.u64 " << addr << ", " << out_ptr[static_cast<std::size_t>(plan.x_output)]
              << ", " << off << ";\n";
            std::vector<std::string> x = emit_load(b, a, xs.dtype, lanes, addr);
            v[plan.x_node] = x;
            return x;
        }
        // Without the reload the plan guarantees pass one stored nothing, so
        // the inputs are still what pass one read.
        v = emit_inputs(o, r_c, vals ? plan.post_input : plan.pre_input);
        v = emit_chain(b, a, dag, lanes, std::move(v), &plan.pre_ids);
        return v.at(plan.x_node);
    };

    // The row total of `acc` in every thread of the row: a warp butterfly,
    // then, when the row spans several warps, one shared-memory round across
    // ITS warps. `region` picks the slot set (one entry per warp of the
    // block, at most eight): squares at 0, sums after them, so the two
    // LayerNorm rounds touch different words and need no barrier between
    // them.
    auto block_reduce = [&](const std::string& acc, int region, const char* tag) {
        const std::string rb = a.R();
        for (int m = 16; m >= 1; m >>= 1) {
            const std::string t = a.F();
            b << "    mov.b32 " << rb << ", " << acc << ";\n";
            b << "    shfl.sync.bfly.b32 " << rb << ", " << rb << ", " << m << ", 31, -1;\n";
            b << "    mov.b32 " << t << ", " << rb << ";\n";
            b << "    add.f32 " << acc << ", " << acc << ", " << t << ";\n";
        }
        if (warps == 1) return;
        const int off = region * (kRowThreads / 32) * 4;
        const std::string r_warp = a.R(), r_lane = a.R(), r_sb = a.R(), r_sa = a.R();
        const std::string p_lane0 = a.P(), rt = a.R();
        b << "    shr.u32 " << r_warp << ", " << r_tid << ", 5;\n";
        b << "    and.b32 " << r_lane << ", " << r_tid << ", 31;\n";
        b << "    mov.u32 " << r_sb << ", rn_red;\n";
        if (off) b << "    add.u32 " << r_sb << ", " << r_sb << ", " << off << ";\n";
        b << "    setp.eq.u32 " << p_lane0 << ", " << r_lane << ", 0;\n";
        b << "    @!" << p_lane0 << " bra $L_" << name << "_" << tag << "_nostore;\n";
        b << "    mad.lo.u32 " << r_sa << ", " << r_warp << ", 4, " << r_sb << ";\n";
        b << "    mov.b32 " << rt << ", " << acc << ";\n";
        b << "    st.shared.b32 [" << r_sa << "], " << rt << ";\n";
        b << "$L_" << name << "_" << tag << "_nostore:\n";
        b << "    bar.sync 0;\n";

        // This row's warps start at (row within block) * warps. Every thread
        // folds them itself, so the result needs no second broadcast.
        const std::string r_base = a.R();
        if (rpb == 1) {
            b << "    mov.u32 " << r_base << ", " << r_sb << ";\n";
        } else {
            b << "    mad.lo.u32 " << r_base << ", " << r_local << ", " << (warps * 4)
              << ", " << r_sb << ";\n";
        }
        b << "    mov.f32 " << acc << ", 0f00000000;\n";
        for (int w = 0; w < warps; ++w) {
            const std::string ra = a.R(), rv = a.R(), fv = a.F();
            b << "    add.u32 " << ra << ", " << r_base << ", " << (w * 4) << ";\n";
            b << "    ld.shared.b32 " << rv << ", [" << ra << "];\n";
            b << "    mov.b32 " << fv << ", " << rv << ";\n";
            b << "    add.f32 " << acc << ", " << acc << ", " << fv << ";\n";
        }
    };

    // ── pass 1: the pre chain, then the sum (Mean) or sum of squares (RMS) ──
    const std::string f_acc = a.F();
    b << "    mov.f32 " << f_acc << ", 0f00000000;\n";
    {
        const std::string r_c = a.R(), p = a.P();
        b << "    mov.u32 " << r_c << ", " << r_c0 << ";\n";
        b << "$L_" << name << "_sum:\n";
        b << "    setp.ge.u32 " << p << ", " << r_c << ", " << D << ";\n";
        b << "    @" << p << " bra $L_" << name << "_sum_end;\n";

        Offsets o1;
        ValueMap vals =
            emit_chain(b, a, dag, lanes, emit_inputs(o1, r_c, plan.pre_input), &plan.pre_ids);
        const std::vector<std::string> x = vals.at(plan.x_node);
        for (int j = 0; j < lanes; ++j) {
            const std::string& xj = x[static_cast<std::size_t>(j)];
            if (mean) {
                b << "    add.f32 " << f_acc << ", " << f_acc << ", " << xj << ";\n";
            } else {
                b << "    fma.rn.f32 " << f_acc << ", " << xj << ", " << xj << ", " << f_acc << ";\n";
            }
        }
        // Outputs the pre chain produced are final here — writing them now is
        // what lets pass two read the value back instead of recomputing it.
        for (std::size_t m = 0; m < n_out; ++m) {
            if (m >= plan.pre_output.size() || !plan.pre_output[m]) continue;
            const BufferSpec& s = ew.outputs[m];
            const std::string off = full_off(o1, r_c, s.dtype);
            const std::string addr = a.RD();
            b << "    add.u64 " << addr << ", " << out_ptr[m] << ", " << off << ";\n";
            emit_store(b, a, s.dtype, lanes, addr, vals.at(s.node_id));
        }

        b << "    add.u32 " << r_c << ", " << r_c << ", " << step << ";\n";
        b << "    bra $L_" << name << "_sum;\n";
        b << "$L_" << name << "_sum_end:\n";
    }

    // ── reductions; for LayerNorm, pass 1b over the deviations ──────────────
    const std::string f_mean = a.F(), f_rstd = a.F();
    const float inv_d = 1.0f / static_cast<float>(D);
    if (mean) {
        block_reduce(f_acc, 1, "sum");
        b << "    mul.f32 " << f_mean << ", " << f_acc << ", " << fhex(inv_d) << ";\n";

        const std::string f_dev = a.F();
        b << "    mov.f32 " << f_dev << ", 0f00000000;\n";
        const std::string r_c = a.R(), p = a.P();
        b << "    mov.u32 " << r_c << ", " << r_c0 << ";\n";
        b << "$L_" << name << "_dev:\n";
        b << "    setp.ge.u32 " << p << ", " << r_c << ", " << D << ";\n";
        b << "    @" << p << " bra $L_" << name << "_dev_end;\n";
        Offsets o1b;
        const std::vector<std::string> x = reload_x(o1b, r_c, nullptr);
        for (int j = 0; j < lanes; ++j) {
            const std::string d = a.F();
            b << "    sub.f32 " << d << ", " << x[static_cast<std::size_t>(j)] << ", " << f_mean << ";\n";
            b << "    fma.rn.f32 " << f_dev << ", " << d << ", " << d << ", " << f_dev << ";\n";
        }
        b << "    add.u32 " << r_c << ", " << r_c << ", " << step << ";\n";
        b << "    bra $L_" << name << "_dev;\n";
        b << "$L_" << name << "_dev_end:\n";

        block_reduce(f_dev, 0, "dev");
        const std::string var = a.F();
        b << "    mul.f32 " << var << ", " << f_dev << ", " << fhex(inv_d) << ";\n";
        b << "    add.f32 " << var << ", " << var << ", " << fhex(plan.eps) << ";\n";
        b << "    rsqrt.approx.f32 " << f_rstd << ", " << var << ";\n";
    } else {
        block_reduce(f_acc, 0, "sq");
        const std::string ms = a.F();
        b << "    mov.f32 " << f_mean << ", 0f00000000;\n";
        b << "    mul.f32 " << ms << ", " << f_acc << ", " << fhex(inv_d) << ";\n";
        b << "    add.f32 " << ms << ", " << ms << ", " << fhex(plan.eps) << ";\n";
        b << "    rsqrt.approx.f32 " << f_rstd << ", " << ms << ";\n";
    }

    // ── pass 2: pre chain, normalise, post chain, store ─────────────────────
    {
        const std::string r_c = a.R(), p = a.P();
        b << "    mov.u32 " << r_c << ", " << r_c0 << ";\n";
        b << "$L_" << name << "_apply:\n";
        b << "    setp.ge.u32 " << p << ", " << r_c << ", " << D << ";\n";
        b << "    @" << p << " bra $L_" << name << "_apply_end;\n";

        Offsets o2;
        ValueMap vals;
        const std::vector<std::string> x = reload_x(o2, r_c, &vals);
        std::vector<std::string> gam, bet;
        if (plan.gamma_input >= 0) {
            gam = vals.at(ew.inputs[static_cast<std::size_t>(plan.gamma_input)].node_id);
        }
        if (plan.beta_input >= 0) {
            bet = vals.at(ew.inputs[static_cast<std::size_t>(plan.beta_input)].node_id);
        }

        std::vector<std::string> norm(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            const std::string f_n = a.F();
            const std::string& xj = x[static_cast<std::size_t>(j)];
            if (mean) {
                b << "    sub.f32 " << f_n << ", " << xj << ", " << f_mean << ";\n";
                b << "    mul.f32 " << f_n << ", " << f_n << ", " << f_rstd << ";\n";
            } else {
                b << "    mul.f32 " << f_n << ", " << xj << ", " << f_rstd << ";\n";
            }
            if (!gam.empty()) {
                if (!bet.empty()) {
                    b << "    fma.rn.f32 " << f_n << ", " << f_n << ", "
                      << gam[static_cast<std::size_t>(j)] << ", "
                      << bet[static_cast<std::size_t>(j)] << ";\n";
                } else {
                    b << "    mul.f32 " << f_n << ", " << f_n << ", "
                      << gam[static_cast<std::size_t>(j)] << ";\n";
                }
            }
            norm[static_cast<std::size_t>(j)] = f_n;
        }
        vals[plan.norm_node] = std::move(norm);

        vals = emit_chain(b, a, dag, lanes, std::move(vals), &plan.post_ids);

        for (std::size_t m = 0; m < n_out; ++m) {
            if (m < plan.pre_output.size() && plan.pre_output[m]) continue;  // written in pass one
            const BufferSpec& s = ew.outputs[m];
            const std::string off = full_off(o2, r_c, s.dtype);
            const std::string addr = a.RD();
            b << "    add.u64 " << addr << ", " << out_ptr[m] << ", " << off << ";\n";
            emit_store(b, a, s.dtype, lanes, addr, vals.at(s.node_id));
        }

        b << "    add.u32 " << r_c << ", " << r_c << ", " << step << ";\n";
        b << "    bra $L_" << name << "_apply;\n";
        b << "$L_" << name << "_apply_end:\n";
    }

    b << "$L_" << name << "_done:\n";
    b << "    ret;\n";

    std::stringstream ss;
    ss << ".visible .entry " << name << "(\n";
    for (std::size_t m = 0; m < n_out; ++m) {
        ss << "    .param .u64 " << name << "_out_" << m << ",\n";
    }
    for (std::size_t k = 0; k + 1 < n_in; ++k) {
        ss << "    .param .u64 " << name << "_in_" << k << ",\n";
    }
    ss << "    .param .u64 " << name << "_in_" << (n_in - 1) << "\n";
    ss << ") {\n";
    ss << declarations(a);
    ss << b.str();
    ss << "}\n\n";
    return ss.str();
}

}  // namespace

std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch) {
    std::stringstream ss;
    ss << ".version 7.8\n";
    ss << ".target " << arch << "\n";
    ss << ".address_size 64\n\n";
    // Per-CTA partials: one slot per warp for the sum of squares (RMS) or of
    // squared deviations (LayerNorm), and for LayerNorm one more set for the
    // plain sum. Shared by both entries.
    ss << ".shared .align 4 .b32 rn_red[" << (2 * (kRowThreads / 32)) << "];\n\n";
    ss << emit_rn_entry(dag, plan, plan.ew.vec, kEntryRowNorm);
    ss << emit_rn_entry(dag, plan, 1, kEntryRowNormScalar);
    return ss.str();
}

}  // namespace brotensor::jit::ptx
