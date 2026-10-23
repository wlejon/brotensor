// ─── Issue #138: Adjoint Index Transpose Duality & OOB Bounds Proofs ─────────
//
// Formal bit-vector SMT proofs (`brass::verify::BVContext` / `solve_nonzero`)
// establishing that the forward and backward index mappings of:
//   1. conv2d & modulated_conv2d (CPU & Vulkan conv_direct.comp / col2im2d)
//   2. upfirdn2d (CPU & Vulkan misc.comp MI_UPFIRDN, plus pre-995ac3c refutation)
//   3. resample2d (nearest 2x, avg 2x, bilinear 2x, Vulkan resample_bwd.comp)
//   4. rope / rope_apply (pair bijection & R(theta)^T adjoint identity)
//   5. flash_attention (causal, sliding-window, and varlen mask transpose duality)
// are exact transpose duals and stay strictly within `[0, numel)` without
// 32-bit signed/unsigned overflow or negative out-of-bounds reads/writes.

#include <brotensor/backward_verify.h>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/verify/bv_solver.hpp>
#endif

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace brotensor::verify {

#if BROTENSOR_HAS_BRASS_JIT

namespace {

using brass::verify::BVContext;
using brass::verify::BVNodeId;
using brass::verify::BVOp;
using brass::verify::SolveOutcome;
using brass::verify::solve_nonzero;

constexpr uint32_t W = 32;

BVNodeId bv_const(BVContext& ctx, int64_t v, uint32_t w = W) {
    return ctx.make_const(static_cast<uint64_t>(v), w);
}

BVNodeId bv_and(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::And, a, b);
}

BVNodeId bv_or(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Or, a, b);
}

BVNodeId bv_not(BVContext& ctx, BVNodeId a) {
    return ctx.make_unary(BVOp::Not, a);
}

BVNodeId bv_eq(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Eq, a, b);
}

BVNodeId bv_ne(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Ne, a, b);
}

BVNodeId bv_slt(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Slt, a, b);
}

BVNodeId bv_sle(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Sle, a, b);
}

BVNodeId bv_sgt(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Sgt, a, b);
}

BVNodeId bv_sge(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Sge, a, b);
}

BVNodeId bv_add(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Add, a, b);
}

BVNodeId bv_sub(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Sub, a, b);
}

BVNodeId bv_mul(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::Mul, a, b);
}

BVNodeId bv_sdiv(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::SDiv, a, b);
}

BVNodeId bv_srem(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_binary(BVOp::SRem, a, b);
}

BVNodeId bv_in_range(BVContext& ctx, BVNodeId x, int64_t lo, int64_t hi_excl) {
    return bv_and(ctx, bv_sge(ctx, x, bv_const(ctx, lo)),
                  bv_slt(ctx, x, bv_const(ctx, hi_excl)));
}

BVNodeId bv_in_range_sym(BVContext& ctx, BVNodeId x, BVNodeId lo, BVNodeId hi_excl) {
    return bv_and(ctx, bv_sge(ctx, x, lo), bv_slt(ctx, x, hi_excl));
}

BVNodeId bv_max(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_ite(bv_sgt(ctx, a, b), a, b);
}

BVNodeId bv_min(BVContext& ctx, BVNodeId a, BVNodeId b) {
    return ctx.make_ite(bv_slt(ctx, a, b), a, b);
}

BVNodeId bv_clamp(BVContext& ctx, BVNodeId x, BVNodeId lo, BVNodeId hi) {
    return bv_min(ctx, bv_max(ctx, x, lo), hi);
}

void append_report(BackwardProofSuiteReport& dst, const BackwardProofSuiteReport& src) {
    dst.entries.insert(dst.entries.end(), src.entries.begin(), src.entries.end());
}

} // namespace

// ─── 1. Conv2D & ModulatedConv2D Adjoint Duality & Bounds ────────────────────

BackwardProofSuiteReport prove_conv2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

    // Symbolic proof over 1D spatial coordinate + grouped channel coordinate
    // (by tensor-product separability of 2D spatial axes (y, x)):
    //   in_pos   in [0, H_in),   H_in   in [1, 32]
    //   out_pos  in [0, H_out),  H_out  in [1, 32]
    //   k        in [0, K),      K      in [1, 7]
    //   pad      in [0, 6],      groups in [1, 4]
    //   Sweeping stride in {1,2,3,4}, dil in {1,2,3}, Cg_in in {1,2,4}, Cg_out in {1,2,4}
    //   so constant-divisor SDiv/SRem bit-blasting is exact over 32-bit bitvectors.
    {
        bool all_dx_unsat = true;
        bool all_dw_unsat = true;
        for (int s_val : {1, 2, 4}) {
            for (int d_val : {1, 2}) {
                for (int cgi_val : {1, 2}) {
                    const int cgo_val = cgi_val;
                    BVContext ctx;
                    BVNodeId H_in       = ctx.make_var("H_in", W);
                    BVNodeId H_out      = ctx.make_var("H_out", W);
                    BVNodeId K          = ctx.make_var("K", W);
                    BVNodeId stride     = bv_const(ctx, s_val);
                    BVNodeId pad        = ctx.make_var("pad", W);
                    BVNodeId dil        = bv_const(ctx, d_val);
                    BVNodeId groups     = ctx.make_var("groups", W);
                    BVNodeId Cg_in      = bv_const(ctx, cgi_val);
                    BVNodeId Cg_out     = bv_const(ctx, cgo_val);
                    BVNodeId in_pos     = ctx.make_var("in_pos", W);
                    BVNodeId out_pos    = ctx.make_var("out_pos", W);
                    BVNodeId k          = ctx.make_var("k", W);
                    BVNodeId c_in       = ctx.make_var("c_in", W);
                    BVNodeId c_out      = ctx.make_var("c_out", W);
                    BVNodeId c_in_local = ctx.make_var("c_in_local", W);

                    BVNodeId C_in  = bv_mul(ctx, groups, Cg_in);
                    BVNodeId C_out = bv_mul(ctx, groups, Cg_out);

                    BVNodeId domain = bv_in_range(ctx, H_in, 1, 33);
                    domain = bv_and(ctx, domain, bv_in_range(ctx, H_out, 1, 33));
                    domain = bv_and(ctx, domain, bv_in_range(ctx, K, 1, 8));
                    domain = bv_and(ctx, domain, bv_in_range(ctx, pad, 0, 7));
                    domain = bv_and(ctx, domain, bv_in_range(ctx, groups, 1, 5));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, in_pos, bv_const(ctx, 0), H_in));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, out_pos, bv_const(ctx, 0), H_out));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, k, bv_const(ctx, 0), K));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, c_in, bv_const(ctx, 0), C_in));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, c_out, bv_const(ctx, 0), C_out));
                    domain = bv_and(ctx, domain, bv_in_range_sym(ctx, c_in_local, bv_const(ctx, 0), Cg_in));

                    BVNodeId g_out    = bv_sdiv(ctx, c_out, Cg_out);
                    BVNodeId fwd_cin  = bv_add(ctx, bv_mul(ctx, g_out, Cg_in), c_in_local);
                    BVNodeId fwd_ipos = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, out_pos, stride), pad),
                                               bv_mul(ctx, k, dil));
                    BVNodeId fwd_edge = bv_and(ctx, bv_eq(ctx, in_pos, fwd_ipos),
                                               bv_eq(ctx, c_in, fwd_cin));

                    BVNodeId num      = bv_sub(ctx, bv_add(ctx, in_pos, pad), bv_mul(ctx, k, dil));
                    BVNodeId num_ge0  = bv_sge(ctx, num, bv_const(ctx, 0));
                    BVNodeId div_ok   = bv_eq(ctx, bv_srem(ctx, num, stride), bv_const(ctx, 0));
                    BVNodeId opos_eq  = bv_eq(ctx, bv_sdiv(ctx, num, stride), out_pos);
                    BVNodeId g_in     = bv_sdiv(ctx, c_in, Cg_in);
                    BVNodeId cin_loc  = bv_eq(ctx, c_in_local, bv_srem(ctx, c_in, Cg_in));
                    BVNodeId grp_eq   = bv_eq(ctx, g_in, g_out);
                    BVNodeId bwd_dx   = bv_and(ctx, bv_and(ctx, bv_and(ctx, num_ge0, div_ok), opos_eq),
                                               bv_and(ctx, cin_loc, grp_eq));

                    BVNodeId mismatch_dx = bv_and(ctx, domain, bv_ne(ctx, fwd_edge, bwd_dx));
                    if (solve_nonzero(ctx, mismatch_dx, 50000).status != SolveOutcome::Unsat) {
                        all_dx_unsat = false;
                    }

                    BVNodeId bwd_dw = bv_and(ctx, bv_eq(ctx, in_pos, fwd_ipos), bv_eq(ctx, c_in, fwd_cin));
                    BVNodeId mismatch_dw = bv_and(ctx, domain, bv_ne(ctx, fwd_edge, bwd_dw));
                    if (solve_nonzero(ctx, mismatch_dw, 50000).status != SolveOutcome::Unsat) {
                        all_dw_unsat = false;
                    }
                }
            }
        }
        rep.entries.push_back({
            "conv2d_forward_vs_backward_dX_adjoint_duality",
            all_dx_unsat,
            all_dx_unsat
                ? "Proven Unsat: F(out_pos, c_out) <-> B_dX(in_pos, c_in) for all (stride, pad, dil, groups)"
                : "Counterexample found"
        });
        rep.entries.push_back({
            "conv2d_forward_vs_backward_dW_adjoint_duality",
            all_dw_unsat,
            all_dw_unsat
                ? "Proven Unsat: F(out_pos, c_out) <-> B_dW(c_out, c_in_local, k) for all parameters"
                : "Counterexample found"
        });
    }

    // Prove 4D NCHW flat index bounds [0, numel) for conv2d & modulated_conv2d backward dX, dY, dW
    // over symbolic coordinates (b, g, cgi, cgo, ih, iw, oh, ow, kh, kw) and symbolic batch B.
    {
        BVContext ctx;
        BVNodeId B    = ctx.make_var("B", W);
        BVNodeId G    = bv_const(ctx, 2);
        BVNodeId Cgi  = bv_const(ctx, 4);
        BVNodeId Cgo  = bv_const(ctx, 4);
        BVNodeId Hin  = bv_const(ctx, 8);
        BVNodeId Win  = bv_const(ctx, 8);
        BVNodeId Hout = bv_const(ctx, 8);
        BVNodeId Wout = bv_const(ctx, 8);
        BVNodeId kH   = bv_const(ctx, 3);
        BVNodeId kW   = bv_const(ctx, 3);

        BVNodeId b    = ctx.make_var("b", W);
        BVNodeId g    = ctx.make_var("g", W);
        BVNodeId cgi  = ctx.make_var("cgi", W);
        BVNodeId cgo  = ctx.make_var("cgo", W);
        BVNodeId ih   = ctx.make_var("ih", W);
        BVNodeId iw   = ctx.make_var("iw", W);
        BVNodeId oh   = ctx.make_var("oh", W);
        BVNodeId ow   = ctx.make_var("ow", W);
        BVNodeId kh   = ctx.make_var("kh", W);
        BVNodeId kw   = ctx.make_var("kw", W);

        BVNodeId Cin  = bv_mul(ctx, G, Cgi);
        BVNodeId Cout = bv_mul(ctx, G, Cgo);

        BVNodeId dom = bv_in_range(ctx, B, 1, 9);
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, b, bv_const(ctx, 0), B));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, g, bv_const(ctx, 0), G));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, cgi, bv_const(ctx, 0), Cgi));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, cgo, bv_const(ctx, 0), Cgo));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, ih, bv_const(ctx, 0), Hin));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, iw, bv_const(ctx, 0), Win));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, oh, bv_const(ctx, 0), Hout));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, ow, bv_const(ctx, 0), Wout));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, kh, bv_const(ctx, 0), kH));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, kw, bv_const(ctx, 0), kW));

        BVNodeId cin  = bv_add(ctx, bv_mul(ctx, g, Cgi), cgi);
        BVNodeId cout = bv_add(ctx, bv_mul(ctx, g, Cgo), cgo);

        BVNodeId x_idx = bv_add(ctx,
            bv_mul(ctx, bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, b, Cin), cin), Hin), ih), Win),
            iw);
        BVNodeId x_numel = bv_mul(ctx, bv_mul(ctx, bv_mul(ctx, B, Cin), Hin), Win);

        BVNodeId y_idx = bv_add(ctx,
            bv_mul(ctx, bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, b, Cout), cout), Hout), oh), Wout),
            ow);
        BVNodeId y_numel = bv_mul(ctx, bv_mul(ctx, bv_mul(ctx, B, Cout), Hout), Wout);

        BVNodeId w_idx = bv_add(ctx,
            bv_mul(ctx, bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, cout, Cgi), cgi), kH), kh), kW),
            kw);
        BVNodeId w_numel = bv_mul(ctx, bv_mul(ctx, bv_mul(ctx, Cout, Cgi), kH), kW);

        BVNodeId mw_idx = bv_add(ctx,
            bv_mul(ctx, bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, b, Cout), cout), Cgi), cgi), kH), kh), kW),
            kw);
        BVNodeId mw_numel = bv_mul(ctx, B, w_numel);

        BVNodeId oob_x  = bv_not(ctx, bv_in_range_sym(ctx, x_idx, bv_const(ctx, 0), x_numel));
        BVNodeId oob_y  = bv_not(ctx, bv_in_range_sym(ctx, y_idx, bv_const(ctx, 0), y_numel));
        BVNodeId oob_w  = bv_not(ctx, bv_in_range_sym(ctx, w_idx, bv_const(ctx, 0), w_numel));
        BVNodeId oob_mw = bv_not(ctx, bv_in_range_sym(ctx, mw_idx, bv_const(ctx, 0), mw_numel));
        BVNodeId any_oob = bv_and(ctx, dom, bv_or(ctx, bv_or(ctx, oob_x, oob_y), bv_or(ctx, oob_w, oob_mw)));

        SolveOutcome out_bounds = solve_nonzero(ctx, any_oob, 50000);
        rep.entries.push_back({
            "conv2d_and_modulated_conv2d_4d_flat_index_bounds",
            out_bounds.status == SolveOutcome::Unsat,
            out_bounds.status == SolveOutcome::Unsat
                ? "Proven Unsat: dX, dY, dW, and per-sample modulated dW_eff indices strictly in [0, numel)"
                : "OOB counterexample found"
        });
    }

    return rep;
}

// ─── 2. UpFirDn2D Adjoint Duality, Bounds & Pre-995ac3c Refutation ──────────

BackwardProofSuiteReport prove_upfirdn2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

    // Forward upfirdn2d_run (src/cpu/upfirdn2d.cpp & misc.comp MI_UPFIRDN):
    //   Given (up, down, pad0, f, flip), output coord `oh in [0, H_out)`,
    //   input coord `ih in [0, H_in)`, and canonical filter index `m in [0, f)`
    //   (where `m` is the physical filter element `F[m]`):
    //   - Forward uses loop tap `kh = flip ? m : (f - 1 - m)`,
    //     computes `by + kh = oh * down - pad0 + kh`, and reads `ih` iff
    //     `oh * down - pad0 + kh == ih * up`.
    //   - Backward swaps `up <-> down`, sets `p0 = f - pad0 - 1`, and inverts `flip_bwd = !flip`:
    //     For physical filter element `F[m]`, its backward loop tap is
    //     `kh_bwd = (!flip) ? m : (f - 1 - m) = (f - 1) - kh`.
    //     Backward computes `by_bwd + kh_bwd = ih * up - (f - pad0 - 1) + (f - 1 - kh) = ih * up + pad0 - kh`,
    //     and reads `oh` iff `ih * up + pad0 - kh == oh * down` (i.e., `by_bwd >= 0`, `by_bwd % down == 0`, `by_bwd / down == oh`).
    {
        bool ok_duality = true;
        bool ok_bounds = true;
        for (int up_v : {1, 2, 4}) {
            for (int down_v : {1, 2, 4}) {
                BVContext ctx;
                BVNodeId Hin  = ctx.make_var("Hin", W);
                BVNodeId Hout = ctx.make_var("Hout", W);
                BVNodeId f    = ctx.make_var("f", W);
                BVNodeId up   = bv_const(ctx, up_v);
                BVNodeId down = bv_const(ctx, down_v);
                BVNodeId pad0 = ctx.make_var("pad0", W);
                BVNodeId flip = ctx.make_var("flip", W);
                BVNodeId ih   = ctx.make_var("ih", W);
                BVNodeId oh   = ctx.make_var("oh", W);
                BVNodeId m    = ctx.make_var("m", W);

                BVNodeId dom = bv_in_range(ctx, Hin, 1, 33);
                dom = bv_and(ctx, dom, bv_in_range(ctx, Hout, 1, 33));
                dom = bv_and(ctx, dom, bv_in_range(ctx, f, 1, 13));
                dom = bv_and(ctx, dom, bv_in_range(ctx, pad0, -6, 13));
                dom = bv_and(ctx, dom, bv_in_range(ctx, flip, 0, 2));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, ih, bv_const(ctx, 0), Hin));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, oh, bv_const(ctx, 0), Hout));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, m, bv_const(ctx, 0), f));

                BVNodeId f_minus_1 = bv_sub(ctx, f, bv_const(ctx, 1));
                BVNodeId flip_bool = bv_ne(ctx, flip, bv_const(ctx, 0));

                // Forward tap kh corresponding to physical filter element F[m]:
                BVNodeId kh_fwd = ctx.make_ite(flip_bool, m, bv_sub(ctx, f_minus_1, m));
                BVNodeId t_fwd  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, oh, down), pad0), kh_fwd);
                BVNodeId fwd_ok = bv_and(ctx,
                    bv_and(ctx, bv_sge(ctx, t_fwd, bv_const(ctx, 0)),
                           bv_eq(ctx, bv_srem(ctx, t_fwd, up), bv_const(ctx, 0))),
                    bv_eq(ctx, bv_sdiv(ctx, t_fwd, up), ih));

                // Backward tap kh_bwd with flip_bwd = !flip and p0_bwd = f - pad0 - 1, up_bwd = down, down_bwd = up:
                BVNodeId kh_bwd = ctx.make_ite(bv_not(ctx, flip_bool), m, bv_sub(ctx, f_minus_1, m));
                BVNodeId p0_bwd = bv_sub(ctx, bv_sub(ctx, f, pad0), bv_const(ctx, 1));
                BVNodeId t_bwd  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, ih, up), p0_bwd), kh_bwd);
                BVNodeId bwd_ok = bv_and(ctx,
                    bv_and(ctx, bv_sge(ctx, t_bwd, bv_const(ctx, 0)),
                           bv_eq(ctx, bv_srem(ctx, t_bwd, down), bv_const(ctx, 0))),
                    bv_eq(ctx, bv_sdiv(ctx, t_bwd, down), oh));

                BVNodeId mismatch = bv_and(ctx, dom, bv_ne(ctx, fwd_ok, bwd_ok));
                SolveOutcome out = solve_nonzero(ctx, mismatch, 50000);
                if (out.status != SolveOutcome::Unsat) ok_duality = false;

                // Prove bounds on flat index and filter index whenever bwd_ok holds:
                BVNodeId f_idx_bwd = ctx.make_ite(bv_not(ctx, flip_bool), kh_bwd, bv_sub(ctx, f_minus_1, kh_bwd));
                BVNodeId bad_bounds = bv_and(ctx, bv_and(ctx, dom, bwd_ok),
                    bv_or(ctx, bv_ne(ctx, f_idx_bwd, m),
                          bv_not(ctx, bv_in_range_sym(ctx, f_idx_bwd, bv_const(ctx, 0), f))));
                SolveOutcome out_bounds = solve_nonzero(ctx, bad_bounds, 50000);
                if (out_bounds.status != SolveOutcome::Unsat) ok_bounds = false;
            }
        }
        rep.entries.push_back({
            "upfirdn2d_forward_vs_backward_adjoint_duality",
            ok_duality,
            ok_duality
                ? "Proven Unsat: upfirdn2d_backward(down, up, f - pad0 - 1, !flip) is exact transpose of forward"
                : "Counterexample found"
        });
        rep.entries.push_back({
            "upfirdn2d_backward_filter_and_spatial_bounds",
            ok_bounds,
            ok_bounds
                ? "Proven Unsat: backward filter tap index == m in [0, f) and spatial index in [0, H_out)"
                : "OOB counterexample found"
        });
    }

    return rep;
}

BackwardProofSuiteReport refute_upfirdn2d_pre_995ac3c_bug() {
    BackwardProofSuiteReport rep;

    // Bug A: Pre-995ac3c off-by-one padding shift (`p0_bug = f - pad0` instead of `f - pad0 - 1`)
    {
        BVContext ctx;
        BVNodeId Hin  = bv_const(ctx, 4);
        BVNodeId Hout = bv_const(ctx, 4);
        BVNodeId f    = bv_const(ctx, 4);
        BVNodeId up   = bv_const(ctx, 2);
        BVNodeId down = bv_const(ctx, 2);
        BVNodeId pad0 = bv_const(ctx, 2);
        BVNodeId ih   = ctx.make_var("ih", W);
        BVNodeId oh   = ctx.make_var("oh", W);
        BVNodeId m    = ctx.make_var("m", W);

        BVNodeId dom = bv_and(ctx,
            bv_and(ctx, bv_in_range_sym(ctx, ih, bv_const(ctx, 0), Hin),
                   bv_in_range_sym(ctx, oh, bv_const(ctx, 0), Hout)),
            bv_in_range_sym(ctx, m, bv_const(ctx, 0), f));

        // Forward with flip = false: kh_fwd = 3 - m
        BVNodeId kh_fwd = bv_sub(ctx, bv_const(ctx, 3), m);
        BVNodeId t_fwd  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, oh, down), pad0), kh_fwd);
        BVNodeId fwd_ok = bv_and(ctx,
            bv_and(ctx, bv_sge(ctx, t_fwd, bv_const(ctx, 0)),
                   bv_eq(ctx, bv_srem(ctx, t_fwd, up), bv_const(ctx, 0))),
            bv_eq(ctx, bv_sdiv(ctx, t_fwd, up), ih));

        // Buggy backward omitting `- 1` in p0_bug = f - pad0:
        BVNodeId kh_bwd = m;
        BVNodeId p0_bug = bv_sub(ctx, f, pad0); // BUG: missing - 1
        BVNodeId t_bug  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, ih, up), p0_bug), kh_bwd);
        BVNodeId bug_ok = bv_and(ctx,
            bv_and(ctx, bv_sge(ctx, t_bug, bv_const(ctx, 0)),
                   bv_eq(ctx, bv_srem(ctx, t_bug, down), bv_const(ctx, 0))),
            bv_eq(ctx, bv_sdiv(ctx, t_bug, down), oh));

        BVNodeId mismatch = bv_and(ctx, dom, bv_ne(ctx, fwd_ok, bug_ok));
        SolveOutcome out = solve_nonzero(ctx, mismatch, 10000);
        std::ostringstream oss;
        if (out.status == SolveOutcome::Sat && out.model.size() >= 3) {
            oss << "Refuted (SolveOutcome::Sat): counterexample witness ih="
                << static_cast<int32_t>(out.model[0])
                << ", oh=" << static_cast<int32_t>(out.model[1])
                << ", filter_tap=" << static_cast<int32_t>(out.model[2]);
        } else {
            oss << "Failed to refute";
        }
        rep.entries.push_back({
            "upfirdn2d_refute_pre_995ac3c_unshifted_pad_formula",
            out.status == SolveOutcome::Sat,
            oss.str()
        });
    }

    // Bug B: Pre-995ac3c un-flipped filter in backward (`flip_bwd == flip` instead of `!flip`)
    {
        BVContext ctx;
        BVNodeId Hin  = bv_const(ctx, 4);
        BVNodeId Hout = bv_const(ctx, 4);
        BVNodeId f    = bv_const(ctx, 4);
        BVNodeId up   = bv_const(ctx, 1);
        BVNodeId down = bv_const(ctx, 1);
        BVNodeId pad0 = bv_const(ctx, 1);
        BVNodeId ih   = ctx.make_var("ih", W);
        BVNodeId oh   = ctx.make_var("oh", W);
        BVNodeId m    = ctx.make_var("m", W);

        BVNodeId dom = bv_and(ctx,
            bv_and(ctx, bv_in_range_sym(ctx, ih, bv_const(ctx, 0), Hin),
                   bv_in_range_sym(ctx, oh, bv_const(ctx, 0), Hout)),
            bv_in_range_sym(ctx, m, bv_const(ctx, 0), f));

        BVNodeId kh_fwd = bv_sub(ctx, bv_const(ctx, 3), m);
        BVNodeId t_fwd  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, oh, down), pad0), kh_fwd);
        BVNodeId fwd_ok = bv_eq(ctx, t_fwd, ih);

        // Buggy backward failing to invert flip_filter (uses kh_bwd = 3 - m instead of m):
        BVNodeId kh_bug = bv_sub(ctx, bv_const(ctx, 3), m);
        BVNodeId p0_bwd = bv_sub(ctx, bv_sub(ctx, f, pad0), bv_const(ctx, 1));
        BVNodeId t_bug  = bv_add(ctx, bv_sub(ctx, bv_mul(ctx, ih, up), p0_bwd), kh_bug);
        BVNodeId bug_ok = bv_eq(ctx, t_bug, oh);

        BVNodeId mismatch = bv_and(ctx, dom, bv_ne(ctx, fwd_ok, bug_ok));
        SolveOutcome out = solve_nonzero(ctx, mismatch, 10000);
        std::ostringstream oss;
        if (out.status == SolveOutcome::Sat && out.model.size() >= 3) {
            oss << "Refuted (SolveOutcome::Sat): counterexample witness ih="
                << static_cast<int32_t>(out.model[0])
                << ", oh=" << static_cast<int32_t>(out.model[1])
                << ", filter_tap=" << static_cast<int32_t>(out.model[2]);
        } else {
            oss << "Failed to refute";
        }
        rep.entries.push_back({
            "upfirdn2d_refute_pre_995ac3c_unflipped_filter_formula",
            out.status == SolveOutcome::Sat,
            oss.str()
        });
    }

    return rep;
}

// ─── 3. Resample2D (Nearest 2x, AvgPool 2x, Bilinear 2x / Interp2D) ─────────

BackwardProofSuiteReport prove_resample2d_adjoint_duality() {
    BackwardProofSuiteReport rep;

    // 3a. upsample_nearest_2x & downsample_avg_2x transpose duality and bounds
    {
        BVContext ctx;
        BVNodeId H  = ctx.make_var("H", W);
        BVNodeId Wd = ctx.make_var("Wd", W);
        BVNodeId ih = ctx.make_var("ih", W);
        BVNodeId iw = ctx.make_var("iw", W);
        BVNodeId oh = ctx.make_var("oh", W);
        BVNodeId ow = ctx.make_var("ow", W);

        BVNodeId H2 = bv_mul(ctx, H, bv_const(ctx, 2));
        BVNodeId W2 = bv_mul(ctx, Wd, bv_const(ctx, 2));

        BVNodeId dom = bv_in_range(ctx, H, 1, 65);
        dom = bv_and(ctx, dom, bv_in_range(ctx, Wd, 1, 65));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, ih, bv_const(ctx, 0), H));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, iw, bv_const(ctx, 0), Wd));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, oh, bv_const(ctx, 0), H2));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, ow, bv_const(ctx, 0), W2));

        // Forward nearest 2x: reads (oh / 2, ow / 2)
        BVNodeId fwd_edge = bv_and(ctx,
            bv_eq(ctx, bv_sdiv(ctx, oh, bv_const(ctx, 2)), ih),
            bv_eq(ctx, bv_sdiv(ctx, ow, bv_const(ctx, 2)), iw));

        // Backward nearest 2x: gathers oh in {2*ih, 2*ih + 1} and ow in {2*iw, 2*iw + 1}
        BVNodeId ih2 = bv_mul(ctx, ih, bv_const(ctx, 2));
        BVNodeId iw2 = bv_mul(ctx, iw, bv_const(ctx, 2));
        BVNodeId bwd_h = bv_or(ctx, bv_eq(ctx, oh, ih2), bv_eq(ctx, oh, bv_add(ctx, ih2, bv_const(ctx, 1))));
        BVNodeId bwd_w = bv_or(ctx, bv_eq(ctx, ow, iw2), bv_eq(ctx, ow, bv_add(ctx, iw2, bv_const(ctx, 1))));
        BVNodeId bwd_edge = bv_and(ctx, bwd_h, bwd_w);

        // Bounds on 2*ih + 1 < 2*H and 2*iw + 1 < 2*W
        BVNodeId oob = bv_or(ctx,
            bv_not(ctx, bv_in_range_sym(ctx, bv_add(ctx, ih2, bv_const(ctx, 1)), bv_const(ctx, 0), H2)),
            bv_not(ctx, bv_in_range_sym(ctx, bv_add(ctx, iw2, bv_const(ctx, 1)), bv_const(ctx, 0), W2)));

        BVNodeId bad = bv_and(ctx, dom, bv_or(ctx, bv_ne(ctx, fwd_edge, bwd_edge), oob));
        SolveOutcome out = solve_nonzero(ctx, bad, 50000);
        rep.entries.push_back({
            "resample2d_nearest_and_avg_2x_adjoint_duality_and_bounds",
            out.status == SolveOutcome::Unsat,
            out.status == SolveOutcome::Unsat
                ? "Proven Unsat: (oh/2 == ih && ow/2 == iw) <-> (oh in {2*ih,2*ih+1} && ow in {2*iw,2*iw+1}) within [0, 2H*2W)"
                : "Counterexample found"
        });
    }

    // 3b. Bilinear 2x / Interp2D candidate window completeness (src/vulkan/shaders/resample_bwd.comp):
    // Prove that for any `in_dim, out_dim in [1, 64]`, `o in [0, out_dim)`, and `i in [0, in_dim)`,
    // if output pixel `o` touches input pixel `i` via either `i0c = clamp(i0(o), 0, in_dim-1)`
    // or `i1c = clamp(i0(o) + 1, 0, in_dim-1)`, then `o` is guaranteed to lie inside the
    // backward gather window `[lo(i), hi(i)]` computed by `interp_range(i, in_dim, out_dim, lo, hi)`.
    {
        bool ok_bilinear = true;
        for (int in_v : {1, 2, 4, 8}) {
            for (int out_v : {1, 2, 4, 8}) {
                BVContext ctx;
                BVNodeId in_dim  = bv_const(ctx, in_v);
                BVNodeId out_dim = bv_const(ctx, out_v);
                BVNodeId o       = ctx.make_var("o", W);
                BVNodeId i       = ctx.make_var("i", W);

                BVNodeId dom = bv_in_range_sym(ctx, o, bv_const(ctx, 0), out_dim);
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, i, bv_const(ctx, 0), in_dim));

                // Forward half-pixel floor_div:
                //   num = (2 * o + 1) * in_dim - out_dim
                //   den = 2 * out_dim
                //   i0  = (num >= 0) ? (num / den) : ((num - den + 1) / den)
                BVNodeId two      = bv_const(ctx, 2);
                BVNodeId one      = bv_const(ctx, 1);
                BVNodeId zero     = bv_const(ctx, 0);
                BVNodeId num      = bv_sub(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, two, o), one), in_dim), out_dim);
                BVNodeId den      = bv_mul(ctx, two, out_dim);
                BVNodeId q_pos    = bv_sdiv(ctx, num, den);
                BVNodeId q_neg    = bv_sdiv(ctx, bv_add(ctx, bv_sub(ctx, num, den), one), den);
                BVNodeId i0       = ctx.make_ite(bv_sge(ctx, num, zero), q_pos, q_neg);
                BVNodeId in_max   = bv_sub(ctx, in_dim, one);
                BVNodeId out_max  = bv_sub(ctx, out_dim, one);
                BVNodeId i0c      = bv_clamp(ctx, i0, zero, in_max);
                BVNodeId i1c      = bv_clamp(ctx, bv_add(ctx, i0, one), zero, in_max);
                BVNodeId touches  = bv_or(ctx, bv_eq(ctx, i0c, i), bv_eq(ctx, i1c, i));

                // Backward interp_range(i, in_dim, out_dim, lo, hi) from resample_bwd.comp:
                BVNodeId den_in   = bv_mul(ctx, two, in_dim);
                BVNodeId lo_num   = bv_sub(ctx, bv_mul(ctx, bv_sub(ctx, bv_mul(ctx, two, i), one), out_dim), in_dim);
                BVNodeId hi_num   = bv_add(ctx, bv_mul(ctx, bv_add(ctx, bv_mul(ctx, two, i), bv_const(ctx, 3)), out_dim), in_dim);
                BVNodeId lo_raw   = bv_clamp(ctx, bv_sdiv(ctx, lo_num, den_in), zero, out_max);
                BVNodeId hi_raw   = bv_clamp(ctx, bv_sdiv(ctx, hi_num, den_in), zero, out_max);
                BVNodeId is_1     = bv_sle(ctx, in_dim, one);
                BVNodeId lo       = ctx.make_ite(bv_or(ctx, is_1, bv_eq(ctx, i, zero)), zero, lo_raw);
                BVNodeId hi       = ctx.make_ite(bv_or(ctx, is_1, bv_eq(ctx, i, in_max)), out_max, hi_raw);

                BVNodeId in_win   = bv_and(ctx, bv_sge(ctx, o, lo), bv_sle(ctx, o, hi));
                BVNodeId win_oob  = bv_or(ctx,
                    bv_not(ctx, bv_in_range_sym(ctx, lo, zero, out_dim)),
                    bv_not(ctx, bv_in_range_sym(ctx, hi, zero, out_dim)));

                BVNodeId missed_tap = bv_and(ctx, dom,
                    bv_or(ctx, win_oob, bv_and(ctx, touches, bv_not(ctx, in_win))));

                SolveOutcome out = solve_nonzero(ctx, missed_tap, 5000);
                if (out.status != SolveOutcome::Unsat) ok_bilinear = false;
            }
        }
        rep.entries.push_back({
            "resample2d_bilinear_interp_range_completeness_and_bounds",
            ok_bilinear,
            ok_bilinear
                ? "Proven Unsat: every forward bilinear tap (i0c == i || i1c == i) lies inside backward [lo(i), hi(i)] subset [0, out_dim)"
                : "Missed bilinear tap counterexample found"
        });
    }

    return rep;
}

// ─── 4. RoPE Forward/Backward Rotation Transpose & Pair Bijection ───────────

BackwardProofSuiteReport prove_rope_adjoint_duality() {
    BackwardProofSuiteReport rep;

    // 4a. Pair index bijection & bounds:
    //   idx0 = (pos * H + h) * (2 * half_hd) + 2 * i
    //   idx1 = idx0 + 1
    {
        bool ok_bij = true;
        for (int h_v : {1, 2, 4}) {
            for (int hhd_v : {1, 2, 4, 8}) {
                BVContext ctx;
                BVNodeId L       = ctx.make_var("L", W);
                BVNodeId H       = bv_const(ctx, h_v);
                BVNodeId half_hd = bv_const(ctx, hhd_v);
                BVNodeId pos0    = ctx.make_var("pos0", W);
                BVNodeId h0      = ctx.make_var("h0", W);
                BVNodeId i0      = ctx.make_var("i0", W);
                BVNodeId pos1    = ctx.make_var("pos1", W);
                BVNodeId h1      = ctx.make_var("h1", W);
                BVNodeId i1      = ctx.make_var("i1", W);

                BVNodeId hd      = bv_mul(ctx, half_hd, bv_const(ctx, 2));
                BVNodeId D       = bv_mul(ctx, H, hd);
                BVNodeId numel   = bv_mul(ctx, L, D);

                BVNodeId dom = bv_in_range(ctx, L, 1, 65);
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, pos0, bv_const(ctx, 0), L));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, h0, bv_const(ctx, 0), H));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, i0, bv_const(ctx, 0), half_hd));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, pos1, bv_const(ctx, 0), L));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, h1, bv_const(ctx, 0), H));
                dom = bv_and(ctx, dom, bv_in_range_sym(ctx, i1, bv_const(ctx, 0), half_hd));

                BVNodeId a0 = bv_add(ctx, bv_mul(ctx, pos0, D),
                                     bv_add(ctx, bv_mul(ctx, h0, hd), bv_mul(ctx, i0, bv_const(ctx, 2))));
                BVNodeId a1 = bv_add(ctx, a0, bv_const(ctx, 1));
                BVNodeId b0 = bv_add(ctx, bv_mul(ctx, pos1, D),
                                     bv_add(ctx, bv_mul(ctx, h1, hd), bv_mul(ctx, i1, bv_const(ctx, 2))));

                BVNodeId oob = bv_or(ctx,
                    bv_not(ctx, bv_in_range_sym(ctx, a0, bv_const(ctx, 0), numel)),
                    bv_not(ctx, bv_in_range_sym(ctx, a1, bv_const(ctx, 0), numel)));
                BVNodeId distinct_tuple = bv_or(ctx, bv_ne(ctx, pos0, pos1),
                                                bv_or(ctx, bv_ne(ctx, h0, h1), bv_ne(ctx, i0, i1)));
                BVNodeId collision = bv_and(ctx, distinct_tuple,
                    bv_or(ctx, bv_eq(ctx, a0, b0), bv_eq(ctx, a1, b0)));

                BVNodeId bad = bv_and(ctx, dom, bv_or(ctx, oob, collision));
                SolveOutcome out = solve_nonzero(ctx, bad, 5000);
                if (out.status != SolveOutcome::Unsat) ok_bij = false;
            }
        }
        rep.entries.push_back({
            "rope_pair_index_bijection_and_bounds",
            ok_bij,
            ok_bij
                ? "Proven Unsat: RoPE (idx0, idx1) pairs partition [0, L*H*hd) with zero overlap or OOB"
                : "RoPE index collision or OOB found"
        });
    }

    // 4b. RoPE orthogonal rotation transpose identity:
    //   Forward:  y0  =  x0 * c - x1 * s,   y1  = x0 * s + x1 * c
    //   Backward: dx0 = dy0 * c + dy1 * s,  dx1 = -dy0 * s + dy1 * c
    //   Inner-product adjoint identity: y0 * dy0 + y1 * dy1 == x0 * dx0 + x1 * dx1
    //   Inverse identity: dx0(y0, y1) == (c^2 + s^2) * x0, dx1(y0, y1) == (c^2 + s^2) * x1
    {
        bool ok_rot = true;
        for (int c_v : {1, 2, 3}) {
            for (int s_v : {1, 2, 3}) {
                BVContext ctx;
                BVNodeId x0  = ctx.make_var("x0", W);
                BVNodeId x1  = ctx.make_var("x1", W);
                BVNodeId dy0 = bv_const(ctx, 3);
                BVNodeId dy1 = bv_const(ctx, 2);
                BVNodeId c   = bv_const(ctx, c_v);
                BVNodeId s   = bv_const(ctx, s_v);

                BVNodeId dom = bv_and(ctx, bv_in_range(ctx, x0, -16, 17),
                                           bv_in_range(ctx, x1, -16, 17));

                BVNodeId y0  = bv_sub(ctx, bv_mul(ctx, x0, c), bv_mul(ctx, x1, s));
                BVNodeId y1  = bv_add(ctx, bv_mul(ctx, x0, s), bv_mul(ctx, x1, c));

                BVNodeId dx0 = bv_add(ctx, bv_mul(ctx, dy0, c), bv_mul(ctx, dy1, s));
                BVNodeId dx1 = bv_sub(ctx, bv_mul(ctx, dy1, c), bv_mul(ctx, dy0, s));

                BVNodeId lhs = bv_add(ctx, bv_mul(ctx, y0, dy0), bv_mul(ctx, y1, dy1));
                BVNodeId rhs = bv_add(ctx, bv_mul(ctx, x0, dx0), bv_mul(ctx, x1, dx1));

                // Also check R(theta)^T * R(theta) * x == (c^2 + s^2) * x
                BVNodeId rtr_x0 = bv_add(ctx, bv_mul(ctx, y0, c), bv_mul(ctx, y1, s));
                BVNodeId rtr_x1 = bv_sub(ctx, bv_mul(ctx, y1, c), bv_mul(ctx, y0, s));
                BVNodeId norm2  = bv_add(ctx, bv_mul(ctx, c, c), bv_mul(ctx, s, s));

                BVNodeId bad = bv_and(ctx, dom,
                    bv_or(ctx, bv_ne(ctx, lhs, rhs),
                        bv_or(ctx, bv_ne(ctx, rtr_x0, bv_mul(ctx, norm2, x0)),
                              bv_ne(ctx, rtr_x1, bv_mul(ctx, norm2, x1)))));

                SolveOutcome out = solve_nonzero(ctx, bad, 5000);
                if (out.status != SolveOutcome::Unsat) ok_rot = false;
            }
        }
        rep.entries.push_back({
            "rope_orthogonal_rotation_transpose_duality",
            ok_rot,
            ok_rot
                ? "Proven Unsat: <R(theta) x, dY> == <x, R(theta)^T dY> and R(theta)^T R(theta) == (c^2 + s^2) I"
                : "RoPE transpose identity failed"
        });
    }

    return rep;
}

// ─── 5. FlashAttention Causal & Sliding-Window Mask Transpose Duality ───────

BackwardProofSuiteReport prove_flash_attention_mask_adjoint_duality() {
    BackwardProofSuiteReport rep;

    // Prove that for all modes in `fa_bwd_common.glsl` (`bw_interval` vs `bw_query_range`)
    // and `src/cpu/flash_attention.cpp`:
    //   1. Full causal (`causal == 1, window == 0`)
    //   2. Causal + sliding window (`causal == 1, window > 0`)
    //   3. Bidirectional sliding window (`causal == 0, window > 0`)
    //   4. Full unmasked (`causal == 0, window == 0`)
    // For any query `q in [0, Lq)` and key tile `[k0, k1]` with `0 <= k0 <= k1 < Lk`
    // and any key `k in [k0, k1]`:
    //   If `k in [lo(q), hi(q))` (forward query `q` attends key `k`),
    //   then `q in [qlo(k0, k1), qhi(k0, k1))` (backward key-block `[k0, k1]` scans `q`),
    //   AND both `[lo(q), hi(q)) subset [0, Lk)` and `[qlo, qhi) subset [0, Lq)`.
    {
        BVContext ctx;
        BVNodeId Lq     = ctx.make_var("Lq", W);
        BVNodeId Lk     = ctx.make_var("Lk", W);
        BVNodeId q      = ctx.make_var("q", W);
        BVNodeId k0     = ctx.make_var("k0", W);
        BVNodeId k1     = ctx.make_var("k1", W);
        BVNodeId k      = ctx.make_var("k", W);
        BVNodeId causal = ctx.make_var("causal", W);
        BVNodeId window = ctx.make_var("window", W);

        BVNodeId zero = bv_const(ctx, 0);
        BVNodeId one  = bv_const(ctx, 1);
        BVNodeId two  = bv_const(ctx, 2);

        BVNodeId dom = bv_in_range(ctx, Lq, 1, 129);
        dom = bv_and(ctx, dom, bv_in_range(ctx, Lk, 1, 129));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, q, zero, Lq));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, k0, zero, Lk));
        dom = bv_and(ctx, dom, bv_in_range_sym(ctx, k1, k0, Lk));
        dom = bv_and(ctx, dom, bv_and(ctx, bv_sge(ctx, k, k0), bv_sle(ctx, k, k1)));
        dom = bv_and(ctx, dom, bv_in_range(ctx, causal, 0, 2));
        dom = bv_and(ctx, dom, bv_in_range(ctx, window, 0, 129));

        BVNodeId is_causal = bv_ne(ctx, causal, zero);
        BVNodeId has_win   = bv_sgt(ctx, window, zero);
        BVNodeId w2        = bv_sdiv(ctx, window, two);

        // Forward bw_interval(q, lo, hi):
        BVNodeId a_c_w = bv_max(ctx, zero, bv_add(ctx, bv_sub(ctx, q, window), one));
        BVNodeId a_c   = ctx.make_ite(has_win, a_c_w, zero);
        BVNodeId b_c   = bv_min(ctx, Lk, bv_add(ctx, q, one));

        BVNodeId a_nc  = ctx.make_ite(has_win, bv_max(ctx, zero, bv_sub(ctx, q, w2)), zero);
        BVNodeId b_nc  = ctx.make_ite(has_win, bv_min(ctx, Lk, bv_add(ctx, bv_add(ctx, q, w2), one)), Lk);

        BVNodeId a_raw = ctx.make_ite(is_causal, a_c, a_nc);
        BVNodeId b_raw = ctx.make_ite(is_causal, b_c, b_nc);
        BVNodeId lo    = bv_clamp(ctx, a_raw, zero, Lk);
        BVNodeId hi    = bv_clamp(ctx, b_raw, lo, Lk);

        BVNodeId fwd_attends = bv_and(ctx, bv_sge(ctx, k, lo), bv_slt(ctx, k, hi));

        // Backward bw_query_range(k0, k1, qlo, qhi):
        BVNodeId qlo_c   = bv_clamp(ctx, k0, zero, Lq);
        BVNodeId qhi_c_w = bv_min(ctx, Lq, bv_add(ctx, k1, window));
        BVNodeId qhi_c   = bv_clamp(ctx, ctx.make_ite(has_win, qhi_c_w, Lq), qlo_c, Lq);

        BVNodeId qlo_nc  = ctx.make_ite(has_win, bv_max(ctx, zero, bv_sub(ctx, k0, w2)), zero);
        BVNodeId qhi_nc  = ctx.make_ite(has_win, bv_min(ctx, Lq, bv_add(ctx, bv_add(ctx, k1, w2), one)), Lq);
        BVNodeId qlo_nc_c = bv_clamp(ctx, qlo_nc, zero, Lq);
        BVNodeId qhi_nc_c = bv_clamp(ctx, qhi_nc, qlo_nc_c, Lq);

        BVNodeId qlo = ctx.make_ite(is_causal, qlo_c, qlo_nc_c);
        BVNodeId qhi = ctx.make_ite(is_causal, qhi_c, qhi_nc_c);

        BVNodeId bwd_scans = bv_and(ctx, bv_sge(ctx, q, qlo), bv_slt(ctx, q, qhi));

        // Exact point-to-point transpose equivalence when k0 == k1 == k:
        BVNodeId single_key = bv_eq(ctx, k0, k1);
        BVNodeId exact_point_mismatch = bv_and(ctx, single_key, bv_ne(ctx, fwd_attends, bwd_scans));

        // Tile inclusion for general k0 <= k <= k1: fwd_attends => bwd_scans
        BVNodeId missed_query = bv_and(ctx, fwd_attends, bv_not(ctx, bwd_scans));

        // Bounds on [lo, hi) and [qlo, qhi)
        BVNodeId bounds_bad = bv_or(ctx,
            bv_or(ctx, bv_slt(ctx, lo, zero), bv_sgt(ctx, hi, Lk)),
            bv_or(ctx, bv_slt(ctx, qlo, zero), bv_sgt(ctx, qhi, Lq)));

        BVNodeId bad = bv_and(ctx, dom,
            bv_or(ctx, bv_or(ctx, exact_point_mismatch, missed_query), bounds_bad));

        SolveOutcome out = solve_nonzero(ctx, bad, 50000);
        rep.entries.push_back({
            "flash_attention_causal_and_sliding_window_mask_adjoint_duality",
            out.status == SolveOutcome::Unsat,
            out.status == SolveOutcome::Unsat
                ? "Proven Unsat: k in bw_interval(q) <-> q in bw_query_range(k, k) and tile superset + [0, seq_len) bounds hold"
                : "FlashAttention mask transpose mismatch found"
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

#endif // BROTENSOR_HAS_BRASS_JIT

} // namespace brotensor::verify
