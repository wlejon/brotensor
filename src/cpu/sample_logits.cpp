// ─── CPU autoregressive logit sampling (CHUNK 7, family F) ──────────────────
//
// FP32 scalar host implementation of `sample_logits` — the next-token sampler
// shared by autoregressive generation loops (brosoundml codec-LM decoding and
// the brolm language-model project). CPU is FP32-only; the GPU vtable slot for
// this op stays null.
//
// Per-row algorithm (one row of an (N, V) logit matrix at a time):
//   temperature scale -> softmax -> optional top-k filter -> optional top-p
//   (nucleus) filter -> renormalize over the kept set -> inverse-CDF draw with
//   a Philox-generated uniform in [0, 1). temperature == 0 short-circuits to a
//   deterministic argmax (no RNG consumed).
//
// ── INT32 output ────────────────────────────────────────────────────────────
//   indices — (N, 1) INT32 sampled token ids. Resized AND dtype-set to INT32,
//   accessed via host_raw / host_raw_mut cast to int32_t* (host_f32 throws on a
//   non-FP32 dtype).
//
// ── Philox (key, counter) ABI ───────────────────────────────────────────────
//   Standard Philox 4x32-10 counter-based generator (PyTorch / JAX compatible).
//   Row n is keyed by `key` and counter block {lo(counter+n), hi(counter+n),
//   0, 0}; the first of the four uint32 outputs becomes a uniform in [0, 1).
//   See ops.h for the full ABI contract.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// ─── Philox 4x32-10 counter-based RNG ───────────────────────────────────────
//
// The reference 4x32-10 generator from Salmon et al. "Parallel Random Numbers:
// As Easy as 1, 2, 3" — the same construction PyTorch's CUDAGeneratorImpl and
// JAX's threefry/philox use. State is a 128-bit counter (4x uint32) plus a
// 64-bit key (2x uint32). One call to philox4x32() produces four uint32s; we
// consume the first.

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;  // multiplier, lane 0
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;  // multiplier, lane 1
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // key bump, word 0 (golden ratio)
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // key bump, word 1 (sqrt(3)-1)

// 32x32 -> 64 multiply, returning the high and low 32-bit words.
inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product);
}

// One Philox round mixing the 128-bit counter under the current key words.
inline void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(kPhiloxM0, ctr[0], hi0, lo0);
    mulhilo32(kPhiloxM1, ctr[2], hi1, lo1);
    const uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    const uint32_t new1 = lo1;
    const uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    const uint32_t new3 = lo0;
    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

// Full Philox 4x32-10: ten rounds, bumping the key between rounds.
inline void philox4x32(uint32_t ctr[4], uint32_t key[2]) {
    for (int round = 0; round < 10; ++round) {
        philox_round(ctr, key);
        if (round < 9) {
            key[0] += kPhiloxW0;
            key[1] += kPhiloxW1;
        }
    }
}

// Draw one uniform in [0, 1) for substream `substream`, seeded by `key64`.
// The 64-bit key is split low-word-first into the two Philox key words; the
// substream index is the low 64 bits of the 128-bit counter (low-word-first),
// the upper 64 counter bits are zero. The first of the four uint32 outputs is
// mapped to [0, 1) via its top 24 bits / 2^24 — matching xavier_init's u01.
inline float philox_uniform(uint64_t key64, uint64_t substream) {
    uint32_t key[2] = {
        static_cast<uint32_t>(key64 & 0xFFFFFFFFull),
        static_cast<uint32_t>(key64 >> 32),
    };
    uint32_t ctr[4] = {
        static_cast<uint32_t>(substream & 0xFFFFFFFFull),
        static_cast<uint32_t>(substream >> 32),
        0u,
        0u,
    };
    philox4x32(ctr, key);
    return static_cast<float>(ctr[0] >> 8) / 16777216.0f;  // top 24 bits
}

} // namespace

// ─── sample_logits ──────────────────────────────────────────────────────────

void sample_logits(const ::brotensor::Tensor& logits, float temperature,
                   int top_k, float top_p, uint64_t key, uint64_t counter,
                   ::brotensor::Tensor& indices) {
    if (logits.dtype != ::brotensor::Dtype::FP32) {
        fail("sample_logits", "logits must be FP32 (CPU backend is FP32-only)");
    }
    if (temperature < 0.0f) {
        fail("sample_logits", "temperature must be >= 0");
    }
    if (top_k < 0) {
        fail("sample_logits", "top_k must be >= 0");
    }
    if (top_p < 0.0f) {
        fail("sample_logits", "top_p must be >= 0");
    }

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) {
        fail("sample_logits", "vocabulary size (logits.cols) must be > 0");
    }

    // indices: (N, 1) INT32 — resize AND dtype-set.
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }
    if (N == 0) return;

    const float* lp = logits.host_f32();
    int32_t* ip = static_cast<int32_t*>(indices.host_raw_mut());

    // Scratch reused across rows.
    std::vector<float> prob(static_cast<std::size_t>(V));
    std::vector<int>   order(static_cast<std::size_t>(V));

    for (int n = 0; n < N; ++n) {
        const float* row = lp + static_cast<std::size_t>(n) * V;

        // ── Greedy: temperature == 0 -> deterministic argmax, no RNG. ──
        if (temperature == 0.0f) {
            float best_v = -3.4028235e38f;   // -FLT_MAX
            int   best_i = 0;
            for (int v = 0; v < V; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best_i = v; }
            }
            ip[n] = static_cast<int32_t>(best_i);
            continue;
        }

        // ── 1. temperature scale + 2. softmax (numerically stable). ──
        float max_logit = -3.4028235e38f;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            if (s > max_logit) max_logit = s;
        }
        double sum = 0.0;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            const float e = std::exp(s - max_logit);
            prob[v] = e;
            sum += e;
        }
        const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
        for (int v = 0; v < V; ++v) prob[v] *= inv_sum;

        // ── 3. top-k filter: keep the top_k highest-probability tokens. ──
        // Only the top `bound` entries are ever consulted below (top-p can
        // only shrink the kept set further, never grow it past top_k), so a
        // partial_sort bounded to `bound` reproduces the same output as a
        // full sort of all V entries while avoiding O(V log V) work. The
        // comparator breaks ties by ascending index to match std::stable_sort
        // applied to the identity-ordered 0..V-1 sequence.
        int bound = V;
        if (top_k > 0 && top_k < bound) bound = top_k;

        for (int v = 0; v < V; ++v) order[v] = v;
        std::partial_sort(order.begin(), order.begin() + bound, order.end(),
                          [&](int a, int b) {
                              return prob[a] > prob[b] ||
                                     (prob[a] == prob[b] && a < b);
                          });

        int keep = bound;

        // ── 4. top-p (nucleus): keep the smallest high-prob set whose
        //       cumulative probability >= top_p, applied to the top-k
        //       survivors. top_p >= 1.0 disables it. ──
        if (top_p < 1.0f) {
            double cum = 0.0;
            int nucleus = 0;
            for (int r = 0; r < keep; ++r) {
                cum += prob[order[r]];
                ++nucleus;
                if (cum >= static_cast<double>(top_p)) break;
            }
            if (nucleus < 1) nucleus = 1;   // always keep at least one token
            keep = nucleus;
        }

        // ── 5. renormalize over the kept set. ──
        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += prob[order[r]];

        // ── 6. inverse-CDF draw with a Philox uniform for substream
        //       (counter + n). ──
        const float u = philox_uniform(key, counter + static_cast<uint64_t>(n));
        int chosen = order[0];
        if (kept_sum > 0.0) {
            const double target = static_cast<double>(u) * kept_sum;
            double acc = 0.0;
            chosen = order[keep - 1];   // fallback: last kept (covers u≈1).
            for (int r = 0; r < keep; ++r) {
                acc += prob[order[r]];
                if (target < acc) { chosen = order[r]; break; }
            }
        }
        ip[n] = static_cast<int32_t>(chosen);
    }
}

// ─── sample_logits_into ─────────────────────────────────────────────────────
//
// Graph-capturable variant: the Philox base counter lives in a device tensor
// (counter[0]) and is advanced by N on completion; scratch is caller-owned.
// On CPU there is no graph capture, so this is the same per-row pipeline run
// directly — its purpose here is to keep the op surface and the (key, counter)
// draw byte-identical to the CUDA/Metal capture path. Draw for row n uses
// substream (counter[0] + n), matching sample_logits with that base counter.

void sample_logits_into(const ::brotensor::Tensor& logits, float temperature,
                        int top_k, float top_p, uint64_t key,
                        ::brotensor::Tensor& counter,
                        ::brotensor::Tensor& scratch,
                        ::brotensor::Tensor& indices) {
    const char* op = "sample_logits_into";
    if (logits.dtype != ::brotensor::Dtype::FP32) {
        fail(op, "logits must be FP32 (CPU backend is FP32-only)");
    }
    if (temperature < 0.0f) fail(op, "temperature must be >= 0");
    if (top_k < 0)          fail(op, "top_k must be >= 0");
    if (top_p < 0.0f)       fail(op, "top_p must be >= 0");

    const int N = logits.rows;
    const int V = logits.cols;
    if (N > 0 && V == 0) fail(op, "vocabulary size (logits.cols) must be > 0");

    if (counter.dtype != ::brotensor::Dtype::INT32 ||
        static_cast<std::size_t>(counter.rows) * counter.cols < 1) {
        fail(op, "counter must be an INT32 tensor with >= 1 element");
    }
    const std::size_t nv = static_cast<std::size_t>(N) * V;
    if (scratch.dtype != ::brotensor::Dtype::FP32 ||
        static_cast<std::size_t>(scratch.rows) * scratch.cols < 3 * nv) {
        fail(op, "scratch must be FP32 with at least 3*N*V elements");
    }
    // indices must be a pre-sized (N,1) INT32 — never resized here (a resize
    // would allocate, which the CUDA path forbids mid-capture).
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        fail(op, "indices must be a pre-sized (N,1) INT32 tensor");
    }
    if (N == 0) return;

    int32_t* cp = static_cast<int32_t*>(counter.host_raw_mut());
    const uint64_t base =
        static_cast<uint64_t>(static_cast<uint32_t>(cp[0]));

    const float* lp = logits.host_f32();
    int32_t* ip = static_cast<int32_t*>(indices.host_raw_mut());

    std::vector<float> prob(static_cast<std::size_t>(V));
    std::vector<int>   order(static_cast<std::size_t>(V));

    for (int n = 0; n < N; ++n) {
        const float* row = lp + static_cast<std::size_t>(n) * V;

        if (temperature == 0.0f) {
            float best_v = -3.4028235e38f;
            int   best_i = 0;
            for (int v = 0; v < V; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best_i = v; }
            }
            ip[n] = static_cast<int32_t>(best_i);
            continue;
        }

        float max_logit = -3.4028235e38f;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            if (s > max_logit) max_logit = s;
        }
        double sum = 0.0;
        for (int v = 0; v < V; ++v) {
            const float s = row[v] / temperature;
            const float e = std::exp(s - max_logit);
            prob[v] = e;
            sum += e;
        }
        const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
        for (int v = 0; v < V; ++v) prob[v] *= inv_sum;

        // See the matching comment in sample_logits() above: bound to the
        // number of entries top-p/top-k can ever consult, and break ties by
        // ascending index to match std::stable_sort's behavior exactly.
        int bound = V;
        if (top_k > 0 && top_k < bound) bound = top_k;

        for (int v = 0; v < V; ++v) order[v] = v;
        std::partial_sort(order.begin(), order.begin() + bound, order.end(),
                          [&](int a, int b) {
                              return prob[a] > prob[b] ||
                                     (prob[a] == prob[b] && a < b);
                          });

        int keep = bound;

        if (top_p < 1.0f) {
            double cum = 0.0;
            int nucleus = 0;
            for (int r = 0; r < keep; ++r) {
                cum += prob[order[r]];
                ++nucleus;
                if (cum >= static_cast<double>(top_p)) break;
            }
            if (nucleus < 1) nucleus = 1;
            keep = nucleus;
        }

        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += prob[order[r]];

        const float u = philox_uniform(key, base + static_cast<uint64_t>(n));
        int chosen = order[0];
        if (kept_sum > 0.0) {
            const double target = static_cast<double>(u) * kept_sum;
            double acc = 0.0;
            chosen = order[keep - 1];
            for (int r = 0; r < keep; ++r) {
                acc += prob[order[r]];
                if (target < acc) { chosen = order[r]; break; }
            }
        }
        ip[n] = static_cast<int32_t>(chosen);
    }

    // Advance the base counter by the rows drawn (matches the device path so a
    // repeated call continues the same Philox stream). Greedy consumes no RNG,
    // so its counter is left untouched — mirroring the CUDA/Metal path.
    if (temperature != 0.0f)
        cp[0] = static_cast<int32_t>(
            static_cast<uint32_t>(base + static_cast<uint64_t>(N)));
}

} // namespace brotensor::detail::cpu
