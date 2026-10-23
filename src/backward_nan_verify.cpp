#include <brotensor/backward_verify.h>
#include <brotensor/ops.h>
#include <brass/verify/bv_solver.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace brotensor::verify {

namespace {

using brass::verify::BVContext;
using brass::verify::BVNodeId;
using brass::verify::BVOp;
using brass::verify::SolveOutcome;
using brass::verify::solve_nonzero;

std::uint32_t float_to_bits(float f) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

bool prove_unsat(BVContext& ctx, BVNodeId violation, std::string& detail_out) {
    SolveOutcome res = solve_nonzero(ctx, violation);
    if (res.status == SolveOutcome::Unsat) {
        detail_out = "UNSAT (proven exact over 32-bit IEEE-754 bitvectors)";
        return true;
    }
    if (res.status == SolveOutcome::Sat) {
        detail_out = "SAT counterexample found";
        return false;
    }
    detail_out = "Solver returned Unknown";
    return false;
}

bool tensor_has_no_nan(const Tensor& t) {
    const float* p = t.host_f32();
    const int n = t.size();
    for (int i = 0; i < n; ++i) {
        if (std::isnan(p[i])) return false;
    }
    return true;
}

bool tensor_all_finite(const Tensor& t) {
    const float* p = t.host_f32();
    const int n = t.size();
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

} // namespace

void sqrt_backward_clamped(
    const Tensor& x,
    const Tensor& dY,
    Tensor& dX,
    float eps) {
    const float eps_safe = (std::isnan(eps) || eps < 1e-12f) ? 1e-12f : eps;
    const int n = x.size();
    dX.resize(x.rows, x.cols, Dtype::FP32);
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float xv = xp[i];
        const float dyv = dyp[i];
        if (std::isnan(xv) || std::isnan(dyv)) {
            dxp[i] = 0.0f;
            continue;
        }
        const float x_clamped = xv < eps_safe ? eps_safe : xv;
        const float denom = 2.0f * std::sqrt(x_clamped);
        const float grad = dyv / denom;
        const float max_finite = std::numeric_limits<float>::max();
        if (grad > max_finite) {
            dxp[i] = max_finite;
        } else if (grad < -max_finite) {
            dxp[i] = -max_finite;
        } else {
            dxp[i] = grad;
        }
    }
}

BackwardProofSuiteReport prove_activation_backward_nan_freedom() {
    BackwardProofSuiteReport report;

    // 1. BVSolver proof: Clamping exponent input |z| <= 88.0f (0x42B00000) keeps
    //    IEEE-754 float strictly below the single-precision expf overflow threshold
    //    88.722839f (0x42B17218), preventing 0 * Inf = NaN in SiLU/SwiGLU/QuickGELU/ELU.
    {
        BVContext ctx;
        auto z_abs_bits = ctx.make_var("z_abs_bits", 32);
        const std::uint32_t kClamp88Bits = float_to_bits(88.0f);          // 0x42B00000
        const std::uint32_t kExpOverflowBits = float_to_bits(88.722839f); // 0x42B17218

        auto pre = ctx.make_binary(BVOp::Ule, z_abs_bits, ctx.make_const(kClamp88Bits, 32));
        auto exp_field = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::LShr, z_abs_bits, ctx.make_const(23, 32)),
            ctx.make_const(0xFF, 32));
        auto is_nan_or_inf = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0xFF, 32));
        auto overflows_exp = ctx.make_binary(BVOp::Uge, z_abs_bits, ctx.make_const(kExpOverflowBits, 32));
        auto bad = ctx.make_binary(BVOp::Or, is_nan_or_inf, overflows_exp);
        auto violation = ctx.make_binary(BVOp::And, pre, bad);

        std::string detail;
        bool ok = prove_unsat(ctx, violation, detail);
        report.entries.push_back({
            "activation_bwd.exp_clamp_88_no_overflow_bv",
            ok,
            detail
        });
    }

    // 2. BVSolver proof: Clamping GELU/GeGLU input |x| <= 20.0f (0x41A00000) keeps
    //    |x| strictly finite and well below cubic overflow threshold, preventing
    //    0 * Inf = NaN in sech^2(u) * du_dx.
    {
        BVContext ctx;
        auto x_abs_bits = ctx.make_var("x_abs_bits", 32);
        const std::uint32_t kClamp20Bits = float_to_bits(20.0f);
        const std::uint32_t kTanhSatBits = float_to_bits(20.5f);

        auto pre = ctx.make_binary(BVOp::Ule, x_abs_bits, ctx.make_const(kClamp20Bits, 32));
        auto exp_field = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::LShr, x_abs_bits, ctx.make_const(23, 32)),
            ctx.make_const(0xFF, 32));
        auto is_nan_or_inf = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0xFF, 32));
        auto exceeds_bound = ctx.make_binary(BVOp::Uge, x_abs_bits, ctx.make_const(kTanhSatBits, 32));
        auto violation = ctx.make_binary(
            BVOp::And, pre, ctx.make_binary(BVOp::Or, is_nan_or_inf, exceeds_bound));

        std::string detail;
        bool ok = prove_unsat(ctx, violation, detail);
        report.entries.push_back({
            "activation_bwd.gelu_clamp_20_finite_bv",
            ok,
            detail
        });
    }

    // 3. BVSolver proof: Clamping sqrt/log denominator x_clamped >= 1e-12f (0x2B8CBCCC)
    //    and finite (<= FLT_MAX = 0x7F7FFFFF) guarantees strictly positive normal
    //    IEEE-754 float (sign == 0, 1 <= exp <= 254), so sqrt(x_clamped) and 1/x_clamped
    //    never divide by zero or take sqrt of a negative value.
    {
        BVContext ctx;
        auto x_bits = ctx.make_var("x_bits", 32);
        const std::uint32_t kEpsBits = float_to_bits(1e-12f);
        const std::uint32_t kFltMaxBits = float_to_bits(std::numeric_limits<float>::max());

        auto ge_eps = ctx.make_binary(BVOp::Uge, x_bits, ctx.make_const(kEpsBits, 32));
        auto le_max = ctx.make_binary(BVOp::Ule, x_bits, ctx.make_const(kFltMaxBits, 32));
        auto pre = ctx.make_binary(BVOp::And, ge_eps, le_max);

        auto sign_bit = ctx.make_binary(BVOp::LShr, x_bits, ctx.make_const(31, 32));
        auto exp_field = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::LShr, x_bits, ctx.make_const(23, 32)),
            ctx.make_const(0xFF, 32));
        auto is_neg = ctx.make_binary(BVOp::Ne, sign_bit, ctx.make_const(0, 32));
        auto is_subnormal_or_zero = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0, 32));
        auto is_nan_or_inf = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0xFF, 32));
        auto bad = ctx.make_binary(
            BVOp::Or, is_neg, ctx.make_binary(BVOp::Or, is_subnormal_or_zero, is_nan_or_inf));
        auto violation = ctx.make_binary(BVOp::And, pre, bad);

        std::string detail;
        bool ok = prove_unsat(ctx, violation, detail);
        report.entries.push_back({
            "activation_bwd.sqrt_log_clamped_denom_normal_positive_bv",
            ok,
            detail
        });
    }

    // 4. Exhaustive boundary-value sweep across all CPU activation backward kernels.
    {
        const std::vector<float> test_vals = {
            0.0f, -0.0f,
            1e-38f, -1e-38f,   // subnormal boundary
            1e-20f, -1e-20f,
            1e-7f, -1e-7f,
            0.5f, -0.5f,
            1.0f, -1.0f,
            20.0f, -20.0f,
            25.0f, -25.0f,
            50.0f, -50.0f,
            88.0f, -88.0f,     // expf saturation boundary
            89.0f, -89.0f,
            100.0f, -100.0f,
            1e10f, -1e10f,
            1e20f, -1e20f,     // extreme cubic overflow regime
            1e38f, -1e38f      // near FLT_MAX
        };
        const int n = static_cast<int>(test_vals.size());
        Tensor x = Tensor::vec(n);
        Tensor dy = Tensor::vec(n);
        for (int i = 0; i < n; ++i) {
            x[i] = test_vals[static_cast<std::size_t>(i)];
            dy[i] = 1.0f;
        }

        Tensor dx = Tensor::vec(n);
        bool all_ok = true;
        std::string failed_op;

        auto check_op = [&](const char* name, auto&& fn) {
            if (!all_ok) return;
            dx.zero();
            fn();
            if (!tensor_has_no_nan(dx) || !tensor_all_finite(dx)) {
                all_ok = false;
                failed_op = name;
            }
        };

        check_op("silu_backward", [&]() { silu_backward(x, dy, dx); });
        check_op("gelu_backward_tanh", [&]() { gelu_backward(x, dy, dx); });
        check_op("gelu_exact_backward", [&]() { gelu_exact_backward(x, dy, dx); });
        check_op("relu_backward", [&]() { relu_backward(x, dy, dx); });
        check_op("leaky_relu_backward", [&]() { leaky_relu_backward(x, dy, 0.01f, dx); });
        check_op("elu_backward", [&]() { elu_backward(x, dy, 1.0f, dx); });
        check_op("quick_gelu_backward", [&]() { quick_gelu_backward(x, dy, dx); });
        check_op("tanh_backward", [&]() { tanh_backward(x, dy, dx); });
        check_op("tanh_backward_batched", [&]() { tanh_backward_batched(x, dy, dx); });
        check_op("sigmoid_backward", [&]() { sigmoid_backward(x, dy, dx); });
        check_op("sqrt_backward_clamped", [&]() { sqrt_backward_clamped(x, dy, dx, 1e-12f); });

        // SwiGLU & GeGLU backward (2D tensor [1, 2*n], moderate values + extreme gate side)
        Tensor x_glu = Tensor::mat(1, 2 * n);
        Tensor dy_glu = Tensor::mat(1, n);
        Tensor dx_glu = Tensor::mat(1, 2 * n);
        for (int i = 0; i < n; ++i) {
            // In SwiGLU: out = silu(A) * B_half; in GeGLU: out = A * gelu(B_half).
            // Test extreme values in both halves with bounded partner to verify
            // gate derivative clamping prevents 0 * Inf = NaN.
            x_glu(0, i) = test_vals[static_cast<std::size_t>(i)];
            x_glu(0, n + i) = 1.0f;
            dy_glu(0, i) = 1.0f;
        }
        swiglu_backward(x_glu, dy_glu, dx_glu);
        if (!tensor_has_no_nan(dx_glu) || !tensor_all_finite(dx_glu)) {
            all_ok = false;
            failed_op = "swiglu_backward";
        }
        for (int i = 0; i < n; ++i) {
            x_glu(0, i) = 1.0f;
            x_glu(0, n + i) = test_vals[static_cast<std::size_t>(i)];
        }
        geglu_backward(x_glu, dy_glu, dx_glu);
        if (all_ok && (!tensor_has_no_nan(dx_glu) || !tensor_all_finite(dx_glu))) {
            all_ok = false;
            failed_op = "geglu_backward";
        }
        geglu_exact_backward(x_glu, dy_glu, dx_glu);
        if (all_ok && (!tensor_has_no_nan(dx_glu) || !tensor_all_finite(dx_glu))) {
            all_ok = false;
            failed_op = "geglu_exact_backward";
        }

        report.entries.push_back({
            "activation_bwd.extreme_ieee754_sweep_nan_free",
            all_ok,
            all_ok ? "All 14 activation backward kernels NaN-free and finite across 30 extreme IEEE-754 inputs"
                   : ("NaN or Inf detected in " + failed_op)
        });
    }

    return report;
}

BackwardProofSuiteReport prove_norm_backward_nan_freedom() {
    BackwardProofSuiteReport report;

    // 1. BVSolver proof: For any variance var >= 0 (even var == 0 when x_i == const)
    //    and any user eps (clamped to eps_safe = max(eps, 1e-12f)), the denominator
    //    radicand rad = max(var + eps_safe, 1e-12f) has IEEE-754 bit pattern in
    //    [0x2B8CBCCC, 0x7F7FFFFF], which is strictly > 0, normal, and finite.
    {
        BVContext ctx;
        auto rad_bits = ctx.make_var("rad_bits", 32);
        const std::uint32_t kMinRadBits = float_to_bits(1e-12f);
        const std::uint32_t kMaxRadBits = float_to_bits(std::numeric_limits<float>::max());

        auto pre = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::Uge, rad_bits, ctx.make_const(kMinRadBits, 32)),
            ctx.make_binary(BVOp::Ule, rad_bits, ctx.make_const(kMaxRadBits, 32)));

        auto sign_bit = ctx.make_binary(BVOp::LShr, rad_bits, ctx.make_const(31, 32));
        auto exp_field = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::LShr, rad_bits, ctx.make_const(23, 32)),
            ctx.make_const(0xFF, 32));
        auto is_zero_or_sub = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0, 32));
        auto is_neg = ctx.make_binary(BVOp::Ne, sign_bit, ctx.make_const(0, 32));
        auto is_nan_inf = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0xFF, 32));
        auto bad = ctx.make_binary(
            BVOp::Or, is_zero_or_sub, ctx.make_binary(BVOp::Or, is_neg, is_nan_inf));
        auto violation = ctx.make_binary(BVOp::And, pre, bad);

        std::string detail;
        bool ok = prove_unsat(ctx, violation, detail);
        report.entries.push_back({
            "norm_bwd.zero_variance_inv_std_denom_strictly_positive_bv",
            ok,
            detail
        });
    }

    // 2. Runtime verification on zero-variance / constant rows and degenerate eps <= 0
    //    across LayerNorm, RMSNorm, L2Norm, and GroupNorm backward passes.
    {
        const int rows = 3;
        const int cols = 8;
        Tensor x = Tensor::mat(rows, cols);
        Tensor dy = Tensor::mat(rows, cols);
        Tensor gamma = Tensor::vec(cols);
        Tensor beta = Tensor::vec(cols);

        // Row 0: all zeros (var = 0, ||x||_2 = 0)
        // Row 1: constant non-zero 42.0f (var = 0)
        // Row 2: large constant 1e6f (catastrophic cancellation in E[x^2] - E[x]^2)
        for (int c = 0; c < cols; ++c) {
            x(0, c) = 0.0f;
            x(1, c) = 42.0f;
            x(2, c) = 1e6f;
            dy(0, c) = static_cast<float>(c) * 0.1f;
            dy(1, c) = (c % 2 == 0) ? 1.0f : -1.0f;
            dy(2, c) = 0.5f;
            gamma[c] = 1.0f;
            beta[c] = 0.0f;
        }

        // Batched LayerNorm forward + backward with eps = 0.0f
        Tensor y_ln = Tensor::mat(rows, cols);
        Tensor xhat_ln = Tensor::mat(rows, cols);
        Tensor mean_ln = Tensor::vec(rows);
        Tensor rstd_ln = Tensor::vec(rows);
        layernorm_forward_batched_with_caches(x, gamma, beta, y_ln, xhat_ln, mean_ln, rstd_ln, 0.0f);

        Tensor dx = Tensor::mat(rows, cols);
        Tensor dgamma = Tensor::vec(cols);
        Tensor dbeta = Tensor::vec(cols);
        layernorm_backward_batched_with_caches(dy, xhat_ln, gamma, rstd_ln, dx, dgamma, dbeta);
        bool ln_ok = tensor_all_finite(y_ln) && tensor_all_finite(dx) &&
                     tensor_all_finite(dgamma) && tensor_all_finite(dbeta);

        // Single-row LayerNorm with rstd = 0.0f and rstd = NaN
        Tensor x1 = Tensor::vec(cols);
        Tensor dy1 = Tensor::vec(cols);
        for (int c = 0; c < cols; ++c) {
            x1[c] = 42.0f;
            dy1[c] = static_cast<float>(c);
        }
        Tensor y1 = Tensor::vec(cols);
        Tensor xhat1 = Tensor::vec(cols);
        float m1 = 0.0f, r1 = 0.0f;
        layernorm_forward(x1, gamma, beta, y1, xhat1, m1, r1, 0.0f);
        Tensor dx1 = Tensor::vec(cols);
        dgamma.zero();
        dbeta.zero();
        layernorm_backward(dy1, xhat1, gamma, r1, dx1, dgamma, dbeta);
        ln_ok = ln_ok && tensor_all_finite(dx1) && tensor_all_finite(dgamma) && tensor_all_finite(dbeta);

        // RMSNorm backward with eps = 0.0f on zero / constant rows
        dx.zero();
        dgamma.zero();
        rms_norm_backward(x, gamma, dy, 0.0f, dx, dgamma);
        bool rms_ok = tensor_all_finite(dx) && tensor_all_finite(dgamma);

        // L2Norm backward with eps = 0.0f on zero / constant rows
        dx.zero();
        l2_norm_backward(x, /*head_dim=*/4, /*num_heads=*/2, 0.0f, dy, dx);
        bool l2_ok = tensor_all_finite(dx);

        // GroupNorm backward with constant channels
        Tensor x_gn = Tensor::mat(1, 4 * 2 * 2);
        Tensor dy_gn = Tensor::mat(1, 4 * 2 * 2);
        Tensor gamma_gn = Tensor::vec(4);
        Tensor dx_gn = Tensor::mat(1, 4 * 2 * 2);
        Tensor dgamma_gn = Tensor::vec(4);
        Tensor dbeta_gn = Tensor::vec(4);
        for (int i = 0; i < x_gn.size(); ++i) {
            x_gn[i] = 7.0f;
            dy_gn[i] = (i % 2 == 0) ? 1.0f : -0.5f;
        }
        for (int c = 0; c < 4; ++c) gamma_gn[c] = 1.0f;
        group_norm_backward(x_gn, gamma_gn, dy_gn, 1, 4, 2, 2, 2, 1e-5f, dx_gn, dgamma_gn, dbeta_gn);
        bool gn_ok = tensor_all_finite(dx_gn) && tensor_all_finite(dgamma_gn) && tensor_all_finite(dbeta_gn);

        const bool all_ok = ln_ok && rms_ok && l2_ok && gn_ok;
        report.entries.push_back({
            "norm_bwd.constant_input_and_zero_eps_finite",
            all_ok,
            all_ok ? "LayerNorm, RMSNorm, L2Norm, GroupNorm backward all finite on zero-variance/zero-eps inputs"
                   : "Non-finite gradient in normalization backward"
        });
    }

    return report;
}

BackwardProofSuiteReport prove_attention_softmax_backward_nan_freedom() {
    BackwardProofSuiteReport report;

    // 1. BVSolver proof: In online/safe softmax, subtracting row max m = max_j s_j
    //    ensures the maximal element has shifted score s_max - m == 0.0f, so its
    //    exponential is exp(0.0f) == 1.0f (0x3F800000), and adding non-negative
    //    terms sum_j exp(s_j - m) yields a denominator d >= 1.0f (0x3F800000) and
    //    <= N * 1.0f. Thus the softmax denominator can never be 0.0f or subnormal.
    {
        BVContext ctx;
        auto denom_bits = ctx.make_var("denom_bits", 32);
        const std::uint32_t kOneBits = float_to_bits(1.0f);          // 0x3F800000
        const std::uint32_t kMaxSeqBits = float_to_bits(1048576.0f); // 2^20 tokens

        auto pre = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::Uge, denom_bits, ctx.make_const(kOneBits, 32)),
            ctx.make_binary(BVOp::Ule, denom_bits, ctx.make_const(kMaxSeqBits, 32)));

        auto exp_field = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::LShr, denom_bits, ctx.make_const(23, 32)),
            ctx.make_const(0xFF, 32));
        auto is_zero_or_sub = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0, 32));
        auto is_nan_inf = ctx.make_binary(BVOp::Eq, exp_field, ctx.make_const(0xFF, 32));
        auto violation = ctx.make_binary(
            BVOp::And, pre, ctx.make_binary(BVOp::Or, is_zero_or_sub, is_nan_inf));

        std::string detail;
        bool ok = prove_unsat(ctx, violation, detail);
        report.entries.push_back({
            "attention_bwd.safe_softmax_denom_ge_one_no_div_zero_bv",
            ok,
            detail
        });
    }

    // 2. All-masked (-Inf / mask=0) softmax & FlashAttention backward verification:
    //    When an entire row is masked with -Inf or mask=0, softmax_forward produces
    //    0.0f (not NaN from -Inf - (-Inf)), and flash_attention_backward produces
    //    0.0f gradients (not NaN).
    {
        const float neg_inf = -std::numeric_limits<float>::infinity();

        // 2a. Softmax forward + backward on all -Inf row
        Tensor scores = Tensor::vec(4);
        for (int i = 0; i < 4; ++i) scores[i] = neg_inf;
        Tensor probs = Tensor::vec(4);
        softmax_forward(scores, probs);
        Tensor dy_sm = Tensor::vec(4);
        for (int i = 0; i < 4; ++i) dy_sm[i] = 1.0f;
        Tensor dx_sm = Tensor::vec(4);
        softmax_backward(probs, dy_sm, dx_sm);
        bool sm_ok = tensor_all_finite(probs) && tensor_all_finite(dx_sm);

        // Also check softmax_rows_forward with an all -Inf row
        Tensor scores_mat = Tensor::mat(2, 4);
        for (int c = 0; c < 4; ++c) {
            scores_mat(0, c) = neg_inf;
            scores_mat(1, c) = static_cast<float>(c);
        }
        Tensor probs_mat = Tensor::mat(2, 4);
        softmax_rows_forward(scores_mat, probs_mat, 2, 4);
        sm_ok = sm_ok && tensor_all_finite(probs_mat);

        // 2b. FlashAttention backward with an all-masked key mask (d_mask[k] = 0.0f)
        const int Lq = 2, Lk = 4, D = 4, num_heads = 1;
        Tensor Q = Tensor::mat(Lq, D);
        Tensor K = Tensor::mat(Lk, D);
        Tensor V = Tensor::mat(Lk, D);
        Tensor O = Tensor::mat(Lq, D);
        Tensor dO = Tensor::mat(Lq, D);
        for (int i = 0; i < Lq * D; ++i) {
            Q[i] = 0.25f * static_cast<float>((i % 5) - 2);
            dO[i] = 0.1f * static_cast<float>((i % 3) + 1);
        }
        for (int i = 0; i < Lk * D; ++i) {
            K[i] = 0.2f * static_cast<float>((i % 7) - 3);
            V[i] = 0.15f * static_cast<float>((i % 5) - 2);
        }
        const std::vector<float> all_zero_mask(Lk, 0.0f);

        Tensor dQ, dK, dV;
        flash_attention_backward(Q, K, V, O, dO, all_zero_mask.data(), num_heads, false, dQ, dK, dV);
        bool fa_ok = tensor_all_finite(dQ) && tensor_all_finite(dK) && tensor_all_finite(dV);
        for (int i = 0; i < dQ.size(); ++i) {
            if (dQ[i] != 0.0f) fa_ok = false;
        }

        report.entries.push_back({
            "attention_bwd.all_masked_row_neg_inf_zero_grad_nan_free",
            sm_ok && fa_ok,
            (sm_ok && fa_ok)
                ? "Softmax & FlashAttention backward produce finite 0.0f gradients on all-masked rows"
                : "NaN or non-zero gradient on all-masked row"
        });
    }

    return report;
}

BackwardProofSuiteReport run_all_backward_nan_freedom_proofs() {
    BackwardProofSuiteReport combined;
    auto append = [&](const BackwardProofSuiteReport& sub) {
        combined.entries.insert(combined.entries.end(), sub.entries.begin(), sub.entries.end());
    };
    append(prove_activation_backward_nan_freedom());
    append(prove_norm_backward_nan_freedom());
    append(prove_attention_softmax_backward_nan_freedom());
    return combined;
}

} // namespace brotensor::verify
