// The CUDA trace JIT's row-norm PTX, checked as text: LayerNorm's variance
// must be the two-pass sum of squared deviations from the mean (CLAUDE.md's
// numerical rule), never E[x^2] - E[x]^2. The emitter is plain C++ writing
// PTX, so this runs everywhere; whether ptxas accepts the result and the
// device agrees needs a CUDA build on an NVIDIA GPU (test_cuda_jit.cpp).
//
// For every LayerNorm entry (vector and scalar; one warp per row and several;
// x as an input, as a stored pre-chain output that is read back, and as a
// pre chain that is replayed):
//   * the mean is the reduced sum times 1/D, defined before rsqrt;
//   * between the mean and rsqrt a loop subtracts the mean from x and
//     accumulates the square of that difference (sub.f32 d, x, mean;
//     fma.rn.f32 acc, d, d, acc), and the variance is that accumulator
//     times 1/D;
//   * nothing squares the mean (the one-pass form's mean * mean);
//   * with several warps per row, each of the two reductions has its own
//     shared round (two bar.sync), RMSNorm one.

#include "../src/jit/ptx_emit.h"
#include "../src/jit/trace_dag.h"
#include "../src/jit/trace_plan.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brotensor;
using namespace brotensor::jit;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

std::string fhex(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    std::stringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << bits;
    return ss.str();
}

enum class XKind { Input, StoredChain, ReplayedChain };

// x (rows x cols), optionally x = a + p first; then layer_norm / rms_norm
// with gamma (and beta for LayerNorm) as (1, cols) rows.
TraceDAG make_dag(bool layer, XKind xk, int rows, int cols) {
    static float dummy[4];
    TraceDAG dag;
    auto input = [&](int r, int c) {
        return dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, r, c);
    };
    int x = input(rows, cols);
    if (xk != XKind::Input) {
        const int p = input(rows, cols);
        x = dag.add_node(TraceOpKind::Add, {x, p}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        if (xk == XKind::StoredChain) dag.nodes()[static_cast<std::size_t>(x)].is_live_at_end = true;
    }
    std::vector<int> ins = {x, input(1, cols)};
    if (layer) ins.push_back(input(1, cols));
    const int n = dag.add_node(layer ? TraceOpKind::LayerNorm : TraceOpKind::RMSNorm, ins, 1e-5f, nullptr,
                               Device::CUDA, Dtype::FP32, rows, cols);
    dag.nodes()[static_cast<std::size_t>(n)].is_live_at_end = true;
    return dag;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    for (std::string l; std::getline(ss, l);) out.push_back(l);
    return out;
}

// The entries of a module, by their `.visible .entry` line.
std::vector<std::vector<std::string>> entries_of(const std::string& ptx) {
    std::vector<std::vector<std::string>> out;
    for (const std::string& l : lines_of(ptx)) {
        if (l.rfind(".visible .entry", 0) == 0) out.emplace_back();
        if (!out.empty()) out.back().push_back(l);
    }
    return out;
}

size_t count(const std::vector<std::string>& e, const std::string& needle) {
    size_t n = 0;
    for (const std::string& l : e) if (l.find(needle) != std::string::npos) ++n;
    return n;
}

int first_line(const std::vector<std::string>& e, const std::string& needle, int from = 0) {
    for (int i = from; i < static_cast<int>(e.size()); ++i)
        if (e[static_cast<std::size_t>(i)].find(needle) != std::string::npos) return i;
    return -1;
}

void check_layernorm_entry(const std::vector<std::string>& e, const std::string& label, int cols, bool multi_warp) {
    const std::string inv_d = fhex(1.0f / static_cast<float>(cols));
    const int rsqrt = first_line(e, "rsqrt.approx.f32");
    check(rsqrt > 0, label + ": rsqrt present");
    if (rsqrt <= 0) return;

    // mul.f32 MEAN, SUM, 1/D — the first use of 1/D.
    const std::regex mul_inv("^\\s*mul\\.f32 (%f\\d+), (%f\\d+), " + inv_d + ";");
    std::smatch m;
    int mean_line = -1, var_line = -1;
    std::string mean_reg, var_src;
    for (int i = 0; i < rsqrt; ++i) {
        if (!std::regex_search(e[static_cast<std::size_t>(i)], m, mul_inv)) continue;
        if (mean_line < 0) {
            mean_line = i;
            mean_reg = m[1];
        } else {
            var_line = i;
            var_src = m[2];
        }
    }
    check(mean_line >= 0 && var_line > mean_line, label + ": mean and variance both scale by 1/D before rsqrt");
    if (mean_line < 0 || var_line < 0) return;

    // Between them: sub.f32 D, X, MEAN then fma.rn.f32 ACC, D, D, ACC, inside
    // a loop (a label after the mean, a branch back to it before rsqrt).
    const std::regex dev_sub("^\\s*sub\\.f32 (%f\\d+), (%f\\d+), " + mean_reg + ";");
    int devs = 0;
    bool acc_feeds_var = true;
    for (int i = mean_line + 1; i < var_line; ++i) {
        if (!std::regex_search(e[static_cast<std::size_t>(i)], m, dev_sub)) continue;
        const std::string d = m[1];
        const std::regex sq("^\\s*fma\\.rn\\.f32 (%f\\d+), " + d + ", " + d + ", (%f\\d+);");
        std::smatch f;
        const bool ok = i + 1 < var_line && std::regex_search(e[static_cast<std::size_t>(i + 1)], f, sq) &&
                        f[1] == f[2];
        if (!ok) {
            acc_feeds_var = false;
            continue;
        }
        ++devs;
        // The accumulator is what the reduction folds and 1/D scales; the
        // reduction reuses the register, so the variance reads it directly.
        if (f[1] != var_src) acc_feeds_var = false;
    }
    check(devs >= 1 && acc_feeds_var,
          label + ": a pass accumulates (x - mean)^2 into the register the variance scales (" +
              std::to_string(devs) + " lanes)");

    int loop_label = -1;
    for (int i = mean_line + 1; i < var_line; ++i) {
        const std::string& l = e[static_cast<std::size_t>(i)];
        if (!l.empty() && l[0] == '$' && l.back() == ':') { loop_label = i; break; }
    }
    bool back_edge = false;
    if (loop_label >= 0) {
        const std::string name = e[static_cast<std::size_t>(loop_label)].substr(
            0, e[static_cast<std::size_t>(loop_label)].size() - 1);
        for (int i = loop_label + 1; i < var_line; ++i)
            if (e[static_cast<std::size_t>(i)].find("bra " + name + ";") != std::string::npos) back_edge = true;
    }
    check(back_edge, label + ": the deviations are summed in their own loop over the row");

    const std::regex mean_sq("^\\s*mul\\.f32 %f\\d+, " + mean_reg + ", " + mean_reg + ";");
    bool squares_mean = false;
    for (const std::string& l : e) if (std::regex_search(l, mean_sq)) squares_mean = true;
    check(!squares_mean, label + ": nothing computes mean * mean (no E[x^2] - E[x]^2)");

    if (multi_warp) {
        check(count(e, "bar.sync") == 2, label + ": two shared rounds (sum, then deviations)");
    } else {
        check(count(e, "bar.sync") == 0, label + ": one warp per row, no shared round");
    }
    check(count(e, "shfl.sync.bfly.b32") == 10, label + ": two warp butterflies");
}

void check_rms_entry(const std::vector<std::string>& e, const std::string& label, bool multi_warp) {
    const int rsqrt = first_line(e, "rsqrt.approx.f32");
    check(rsqrt > 0, label + ": rsqrt present");
    bool sub_before = false;
    for (int i = 0; i < rsqrt; ++i)
        if (e[static_cast<std::size_t>(i)].find("sub.f32") != std::string::npos) sub_before = true;
    check(!sub_before, label + ": one pass, sum of x^2");
    check(count(e, "shfl.sync.bfly.b32") == 5, label + ": one warp butterfly");
    check(count(e, "bar.sync") == (multi_warp ? 1u : 0u), label + ": shared rounds");
}

}  // namespace

int main() {
    std::printf("PTX row-norm emitter: two-pass LayerNorm variance (text only; not executed)\n");
    struct Shape { int rows, cols; };
    // 4096 columns: 256 threads per row (8 warps). 64 columns: one warp per
    // row and several rows per block in the vector entry, two warps per row
    // in the scalar one.
    const Shape shapes[] = {{64, 4096}, {256, 64}};
    const struct { XKind k; const char* name; } xs[] = {
        {XKind::Input, "x input"}, {XKind::StoredChain, "x = a + p stored"}, {XKind::ReplayedChain, "x = a + p replayed"}};

    for (const Shape& s : shapes) {
        for (const auto& x : xs) {
            TraceDAG dag = make_dag(true, x.k, s.rows, s.cols);
            plan::RowNormPlan plan;
            const bool planned = plan::build_row_norm_plan(dag, plan);
            const std::string base = "LayerNorm " + std::to_string(s.rows) + "x" + std::to_string(s.cols) + ", " +
                                     x.name;
            check(planned && plan.reduce == plan::RowReduce::Mean, base + ": planned");
            if (!planned) continue;
            if (x.k == XKind::StoredChain) check(plan.x_output >= 0, base + ": x is read back");
            if (x.k == XKind::ReplayedChain) check(plan.x_output < 0, base + ": x is replayed");
            const std::string ptx = ptx::emit_row_norm(dag, plan, "sm_80");
            const auto entries = entries_of(ptx);
            check(entries.size() == 2, base + ": vector and scalar entries");
            for (std::size_t i = 0; i < entries.size(); ++i) {
                // The vector entry's lane count follows the plan; whether a
                // row spans several warps depends on it.
                const int lanes = i == 0 ? plan.ew.vec : 1;
                int tpr = 0, rpb = 0;
                plan::row_norm_geometry(s.cols, s.rows, lanes, tpr, rpb);
                check_layernorm_entry(entries[i], base + (i == 0 ? " [vec]" : " [scalar]"), s.cols, tpr > 32);
            }
        }
        TraceDAG dag = make_dag(false, XKind::Input, s.rows, s.cols);
        plan::RowNormPlan plan;
        if (plan::build_row_norm_plan(dag, plan)) {
            const auto entries = entries_of(ptx::emit_row_norm(dag, plan, "sm_80"));
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const int lanes = i == 0 ? plan.ew.vec : 1;
                int tpr = 0, rpb = 0;
                plan::row_norm_geometry(s.cols, s.rows, lanes, tpr, rpb);
                check_rms_entry(entries[i], "RMSNorm " + std::to_string(s.cols) + (i == 0 ? " [vec]" : " [scalar]"),
                                tpr > 32);
            }
        } else {
            check(false, "RMSNorm planned");
        }
    }

    if (g_failures) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
