// ─── CPU per-row top-k ──────────────────────────────────────────────────────
//
// FP32 scalar host implementation. For each row of X(R, C), selects the k
// largest values and returns them in descending order in `Vals(R, k)` with
// their original column indices in `Idx(R, k)`. Ties are broken by smaller
// column index — deterministic and stable across runs.
//
// Used for: classification heads (top-5), NMS pre-filter (keep top-K box
// scores per class), beam-search candidates, retrieval rerankers. Not
// differentiable — no backward op.
//
// Algorithm: partial sort via a max-heap of (value, -index) tuples isn't a
// great fit for "tie -> smaller index wins" (we'd want a min-key composite),
// so for clarity we use a streaming-replacement strategy: maintain a working
// array of (value, index) for the current top-k, scan the row, and replace
// the current minimum if a candidate beats it. O(C * k) per row — k is
// always small (5, 100, etc.) in real use, so the simpler code wins.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   top_k_rows — Vals and Idx OVERWRITTEN.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) +
                 " must be FP32 (CPU backend is FP32-only)");
    }
}

// "(a, idx_a) precedes (b, idx_b)" in our descending-value, ascending-index
// ordering. Returns true iff a is strictly preferable to b (i.e. should
// appear earlier in the output).
inline bool prefers(float a, int idx_a, float b, int idx_b) {
    if (a != b) return a > b;
    return idx_a < idx_b;
}

} // namespace

void top_k_rows(const ::brotensor::Tensor& X, int k,
                ::brotensor::Tensor& Vals, ::brotensor::Tensor& Idx) {
    const char* op = "top_k_rows";
    check_fp32(X, op, "X");
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");

    if (Vals.rows != R || Vals.cols != k || Vals.dtype != Dtype::FP32) {
        Vals.resize(R, k, Dtype::FP32);
    }
    if (Idx.rows != R || Idx.cols != k || Idx.dtype != Dtype::INT32) {
        Idx.resize(R, k, Dtype::INT32);
    }
    if (R == 0 || k == 0) return;

    const float* Xp = X.host_f32();
    float* Vp = Vals.host_f32_mut();
    int32_t* Ip = static_cast<int32_t*>(Idx.data);

    // Per-row working arrays (size k). Keeping them on the stack via VLA
    // isn't portable on MSVC; allocate on the heap once and reuse.
    // k is small in practice but C can be huge — so we allocate per row.
    // The inner top-k is O(k) per candidate scanned.
    for (int r = 0; r < R; ++r) {
        const float* row = Xp + static_cast<long>(r) * C;
        float* out_v = Vp + static_cast<long>(r) * k;
        int32_t* out_i = Ip + static_cast<long>(r) * k;

        // Step 1: seed the working set with the first k elements (verbatim
        // order — ties get resolved on later replacements).
        for (int j = 0; j < k; ++j) {
            out_v[j] = row[j];
            out_i[j] = j;
        }
        // Track the current weakest entry — anything that beats it gets
        // swapped in. Scan the working set for the weakest each replace.
        // (For modest k this is faster + cleaner than a heap.)
        int weakest = 0;
        for (int j = 1; j < k; ++j) {
            if (prefers(out_v[weakest], out_i[weakest], out_v[j], out_i[j])) {
                weakest = j;
            }
        }

        // Step 2: scan the remainder. Replace the weakest if beaten.
        for (int c = k; c < C; ++c) {
            const float v = row[c];
            if (prefers(v, c, out_v[weakest], out_i[weakest])) {
                out_v[weakest] = v;
                out_i[weakest] = c;
                // Re-find the weakest after replacement.
                weakest = 0;
                for (int j = 1; j < k; ++j) {
                    if (prefers(out_v[weakest], out_i[weakest],
                                out_v[j], out_i[j])) {
                        weakest = j;
                    }
                }
            }
        }

        // Step 3: sort the k survivors into descending-value / ascending-
        // index order (insertion sort — small k).
        for (int i = 1; i < k; ++i) {
            const float v = out_v[i];
            const int32_t idx = out_i[i];
            int j = i;
            while (j > 0 && prefers(v, idx, out_v[j - 1], out_i[j - 1])) {
                out_v[j] = out_v[j - 1];
                out_i[j] = out_i[j - 1];
                --j;
            }
            out_v[j] = v;
            out_i[j] = idx;
        }
    }
}

} // namespace brotensor::detail::cpu
