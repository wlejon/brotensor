// Attention parity against the CPU backend (test_vulkan_common.h): the flash
// family (plain, GQA, windowed, varlen, packed QKV), the decode ops over a KV
// cache, kv_cache_append and the projection-fused flash ops, in FP32, FP16
// and BF16, through every kernel path the dispatcher can take (fa_cm, fa_rows,
// dense). Shapes include the awkward ones: GQA head
// layouts, packed QKV with restarting sequences, causal windows with
// Lk > Lq, fully masked rows (zero output), odd row counts, unaligned views,
// a cache longer than the query, head widths not a multiple of 16 (or 8).
//
// Inputs are rounded to the dtype first, so the CPU (FP32) and the GPU start
// from identical values. Tolerances: FP32 1e-4 (exp ~3 ulp, different
// summation order); 16-bit 2 ulp of the output dtype + 5e-4 for P rounded to
// FP16 in the cooperative-matrix kernel (observed: 6e-4 FP16, 4e-3 BF16).

#include "test_vulkan_common.h"

#include "detail/attention.h"

#include <cstdlib>
#include <functional>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

const Device kCpu = Device::CPU;

Tensor host(const std::vector<float>& v, int rows, int cols) { return Tensor::from_host_on(kCpu, v.data(), rows, cols); }

Tensor i32_on(Device d, const std::vector<int32_t>& v, int rows, int cols) {
    Tensor t = Tensor::empty_on(d, rows, cols, Dtype::INT32);
    if (!v.empty()) t.copy_from_host_raw(v.data(), v.size() * 4);
    return t;
}

void close_dt(Dtype dt, const std::vector<float>& got, const std::vector<float>& want, const std::string& tag,
              float f32_tol = 1e-4f) {
    if (dt == Dtype::FP32) {
        expect_close(got, want, f32_tol, f32_tol, tag);
    } else {
        const float t = 2.0f * dtype_eps(dt) + 5e-4f;
        expect_close(got, want, t, t, tag);
    }
}

std::string tagf(const char* what, Dtype dt, const char* path, int a, int b, int c, int d) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s %s [%s] %d/%d/%d/%d", what, dt_name(dt), path, a, b, c, d);
    return buf;
}

// FP32, FP16, BF16; VKT_ATTN_DT=f32|f16|bf16 picks one.
std::vector<Dtype> dtypes() {
    const char* e = std::getenv("VKT_ATTN_DT");
    std::vector<Dtype> all = {Dtype::FP32, Dtype::FP16, Dtype::BF16}, out;
    for (Dtype dt : all) if (!e || std::string(e) == dt_name(dt)) out.push_back(dt);
    return out;
}

// Runs `fn` once per forced kernel path (automatic, rows, coopmat, dense);
// the paths that do not apply to a problem fall back to the automatic choice.
// VKT_ATTN_PATH=<name> runs one of them only.
void each_path(const std::function<void(const char*)>& fn) {
    const char* names[] = {"auto", "rows", "cm", "dense"};
    const char* only = std::getenv("VKT_ATTN_PATH");
    for (int m = 0; m < 4; ++m) {
        if (only && std::string(only) != names[m]) continue;
        dv::set_attention_override(m);
        fn(names[m]);
    }
    dv::set_attention_override(0);
}

// ─── flash_attention_forward / gqa / windowed ──────────────────────────────

struct FaCase {
    int lq, lk, hq, hkv, hd;
    bool causal;
    int window;      // windowed op only
    int mask;        // 0 none, 1 random holes, 2 all masked, 3 one valid key
};

std::vector<float> make_mask(int lk, int kind, std::uint64_t seed) {
    std::vector<float> m(static_cast<std::size_t>(lk), 1.0f);
    Rng r(seed);
    for (int i = 0; i < lk; ++i) {
        if (kind == 1) m[i] = (r.next() % 3 == 0) ? 0.0f : 1.0f;
        if (kind == 2) m[i] = 0.0f;
        if (kind == 3) m[i] = i == lk / 2 ? 1.0f : 0.0f;
    }
    return m;
}

enum class FaOp { Forward, Gqa, Windowed };

void run_fa(FaOp which, const FaCase& c, Dtype dt, std::uint64_t seed) {
    const int Dq = c.hq * c.hd, Dkv = c.hkv * c.hd;
    const auto qv = random_values(std::size_t(c.lq) * Dq, seed, -1.5f, 1.5f, dt);
    const auto kv = random_values(std::size_t(c.lk) * Dkv, seed + 1, -1.5f, 1.5f, dt);
    const auto vv = random_values(std::size_t(c.lk) * Dkv, seed + 2, -1.0f, 1.0f, dt);
    const auto mv = make_mask(c.lk, c.mask, seed + 3);
    Tensor Oc;
    {
        Tensor Q = host(qv, c.lq, Dq), K = host(kv, c.lk, Dkv), V = host(vv, c.lk, Dkv);
        const float* m = c.mask ? mv.data() : nullptr;
        if (which == FaOp::Forward) brotensor::flash_attention_forward(Q, K, V, m, c.hq, c.causal, Oc);
        if (which == FaOp::Gqa) brotensor::flash_attention_gqa_forward(Q, K, V, m, c.hq, c.hkv, c.causal, Oc);
        if (which == FaOp::Windowed) brotensor::flash_attention_windowed_forward(Q, K, V, m, c.hq, c.window, Oc, c.causal);
    }
    const auto want = Oc.to_host_vector();
    Tensor Q = upload(qv, c.lq, Dq, dt), K = upload(kv, c.lk, Dkv, dt), V = upload(vv, c.lk, Dkv, dt);
    Tensor M = Tensor::from_host_on(vk(), mv.data(), c.lk, 1);
    const float* m = c.mask ? static_cast<const float*>(M.data) : nullptr;
    const char* what = which == FaOp::Forward ? "flash_attention_forward"
                       : which == FaOp::Gqa   ? "flash_attention_gqa_forward"
                                              : "flash_attention_windowed_forward";
    each_path([&](const char* path) {
        Tensor O;
        if (which == FaOp::Forward) brotensor::flash_attention_forward(Q, K, V, m, c.hq, c.causal, O);
        if (which == FaOp::Gqa) brotensor::flash_attention_gqa_forward(Q, K, V, m, c.hq, c.hkv, c.causal, O);
        if (which == FaOp::Windowed) brotensor::flash_attention_windowed_forward(Q, K, V, m, c.hq, c.window, O, c.causal);
        VKT_CHECK(O.dtype == dt && O.rows == c.lq && O.cols == Dq);
        const auto got = download(O);
        std::string t = tagf(what, dt, path, c.lq, c.lk, c.hq * 1000 + c.hkv, c.hd);
        if (c.causal) t += " causal";
        if (c.window) t += " w" + std::to_string(c.window);
        if (c.mask) t += " mask" + std::to_string(c.mask);
        close_dt(dt, got, want, t);
        if (c.mask == 2) {
            bool zero = true;
            for (float x : got) zero = zero && x == 0.0f;
            VKT_CHECK(zero);
        }
    });
}

void test_flash_forward() {
    std::printf("flash_attention_forward / gqa / windowed\n");
    const FaCase fwd[] = {
        {64, 64, 2, 2, 64, false, 0, 0},
        {130, 130, 3, 3, 128, true, 0, 0},     // odd rows, causal
        {77, 77, 2, 2, 80, false, 0, 1},       // hd 80, key mask
        {50, 200, 2, 2, 96, false, 0, 0},      // Lq != Lk
        {33, 70, 2, 2, 40, false, 0, 2},       // hd 40 (not a multiple of 16), fully masked
        {40, 40, 3, 3, 33, true, 0, 3},        // hd 33 (scalar loads), one valid key
        {3, 300, 4, 4, 64, false, 0, 1},       // few query rows (decode kernel)
        {96, 96, 1, 1, 160, false, 0, 0},      // wide head
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 100;
        for (const FaCase& c : fwd) run_fa(FaOp::Forward, c, dt, seed += 10);
    }
    const FaCase gqa[] = {
        {70, 70, 8, 2, 64, true, 0, 0},
        {70, 90, 6, 3, 128, false, 0, 1},
        {65, 65, 8, 1, 48, false, 0, 0},       // MQA
        {2, 129, 8, 2, 64, false, 0, 0},
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 300;
        for (const FaCase& c : gqa) run_fa(FaOp::Gqa, c, dt, seed += 10);
    }
    const FaCase win[] = {
        {60, 60, 4, 4, 64, true, 16, 0},       // sliding causal window
        {40, 100, 4, 2, 64, true, 32, 0},      // Lk > Lq: queries at the end
        {40, 100, 4, 2, 64, true, 0, 1},       // plain causal, Lk > Lq, mask
        {50, 50, 2, 2, 64, false, 9, 0},       // bidirectional band |q - k| <= 4
        {1, 513, 8, 2, 128, true, 100, 0},     // one decode step
        {37, 37, 2, 1, 96, true, 5, 1},
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 500;
        for (const FaCase& c : win) run_fa(FaOp::Windowed, c, dt, seed += 10);
    }
}

// Unaligned views: Q, K, V and O start a few elements into larger buffers
// (the scalar-load paths and the per-element epilogue).
void test_views() {
    std::printf("flash attention on unaligned views\n");
    for (Dtype dt : dtypes()) {
        const int L = 70, H = 2, hd = 64, D = H * hd, off = 3;
        const auto qv = random_values(std::size_t(L) * D, 41, -1.5f, 1.5f, dt);
        const auto kv = random_values(std::size_t(L) * D, 42, -1.5f, 1.5f, dt);
        const auto vv = random_values(std::size_t(L) * D, 43, -1.0f, 1.0f, dt);
        Tensor Oc;
        brotensor::flash_attention_forward(host(qv, L, D), host(kv, L, D), host(vv, L, D), nullptr, H, true, Oc);
        auto padded = [&](const std::vector<float>& v) {
            std::vector<float> p(v.size() + off, 0.0f);
            std::copy(v.begin(), v.end(), p.begin() + off);
            return upload(p, int(p.size()), 1, dt);
        };
        Tensor qb = padded(qv), kb = padded(kv), vb = padded(vv);
        Tensor ob = Tensor::zeros_on(vk(), int(qv.size()) + off, 1, dt);
        const std::size_t es = dt == Dtype::FP32 ? 4 : 2;
        auto view = [&](Tensor& b) { return Tensor::view(vk(), static_cast<char*>(b.data) + off * es, L, D, dt); };
        Tensor Q = view(qb), K = view(kb), V = view(vb), O = view(ob);
        each_path([&](const char* path) {
            brotensor::flash_attention_forward(Q, K, V, nullptr, H, true, O);
            const auto all = download(ob);
            close_dt(dt, std::vector<float>(all.begin() + off, all.end()), Oc.to_host_vector(),
                     tagf("flash_attention_forward view", dt, path, L, L, H, hd));
            VKT_CHECK(all[0] == 0.0f && all[1] == 0.0f && all[2] == 0.0f);   // nothing written before O
        });
    }
}

// ─── varlen / packed QKV ───────────────────────────────────────────────────

void test_varlen() {
    std::printf("flash_attention_varlen_forward\n");
    struct Case { std::vector<int32_t> cq, ck; int H, hd; bool causal; };
    const Case cases[] = {
        {{0, 5, 5, 45}, {0, 7, 7, 60}, 2, 64, false},          // an empty sequence, Lq != Lk
        {{0, 33, 50, 120}, {0, 33, 50, 120}, 3, 64, true},     // causal within each sequence
        {{0, 17, 90}, {0, 17, 90}, 2, 80, false},
        {{0, 64}, {0, 64}, 2, 48, false},                       // one sequence
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 700;
        for (const Case& c : cases) {
            const int B = int(c.cq.size()) - 1, tq = c.cq.back(), tk = c.ck.back(), D = c.H * c.hd;
            const auto qv = random_values(std::size_t(tq) * D, seed += 3, -1.5f, 1.5f, dt);
            const auto kv = random_values(std::size_t(tk) * D, seed + 1, -1.5f, 1.5f, dt);
            const auto vv = random_values(std::size_t(tk) * D, seed + 2, -1.0f, 1.0f, dt);
            Tensor Oc;
            brotensor::flash_attention_varlen_forward(host(qv, tq, D), host(kv, tk, D), host(vv, tk, D), c.cq.data(),
                                                      c.ck.data(), B, 0, 0, c.H, c.hd, c.causal, Oc);
            Tensor cq = i32_on(vk(), c.cq, B + 1, 1), ck = i32_on(vk(), c.ck, B + 1, 1);
            Tensor Q = upload(qv, tq, D, dt), K = upload(kv, tk, D, dt), V = upload(vv, tk, D, dt);
            each_path([&](const char* path) {
                Tensor O;
                brotensor::flash_attention_varlen_forward(Q, K, V, static_cast<const int32_t*>(cq.data),
                                                          static_cast<const int32_t*>(ck.data), B, 0, 0, c.H, c.hd,
                                                          c.causal, O);
                close_dt(dt, download(O), Oc.to_host_vector(),
                         tagf(c.causal ? "flash_attention_varlen_forward causal" : "flash_attention_varlen_forward", dt,
                              path, tq, tk, B, c.hd));
            });
        }
    }
}

void test_packed_qkv() {
    std::printf("flash_attention_packed_qkv_forward\n");
    struct Case { std::vector<int> lens; int H, hd, window; };
    const Case cases[] = {
        {{20, 1, 29}, 2, 64, 0},     // restarting sequences, a one-token sequence
        {{20, 1, 29}, 2, 64, 8},     // windowed band
        {{70}, 3, 48, 0},
        {{16, 16, 16, 16, 5}, 2, 128, 6},
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 900;
        for (const Case& c : cases) {
            std::vector<int32_t> bounds;
            int L = 0;
            for (int n : c.lens) {
                for (int i = 0; i < n; ++i) { bounds.push_back(L); bounds.push_back(L + n); }
                L += n;
            }
            const int D = c.H * c.hd;
            const auto qkv = random_values(std::size_t(L) * 3 * D, seed += 5, -1.5f, 1.5f, dt);
            Tensor bc = Tensor::empty_on(kCpu, L, 2, Dtype::INT32);
            std::memcpy(bc.data, bounds.data(), bounds.size() * 4);
            Tensor Oc;
            brotensor::flash_attention_packed_qkv_forward(host(qkv, L, 3 * D), bc, c.H, c.window, Oc);
            Tensor bg = i32_on(vk(), bounds, L, 2);
            Tensor X = upload(qkv, L, 3 * D, dt);
            each_path([&](const char* path) {
                Tensor O;
                brotensor::flash_attention_packed_qkv_forward(X, bg, c.H, c.window, O);
                close_dt(dt, download(O), Oc.to_host_vector(),
                         tagf("flash_attention_packed_qkv_forward", dt, path, L, int(c.lens.size()), c.window, c.hd));
            });
        }
    }
}

// ─── decode ────────────────────────────────────────────────────────────────

void test_decode() {
    std::printf("kv_cache_append / flash_attention_decode / _masked\n");
    struct Case { int lq, cap, valid, hq, hkv, hd; float softcap; int window; };
    const Case cases[] = {
        {1, 300, 257, 8, 2, 64, 0.0f, 0},
        {3, 300, 120, 8, 2, 64, 0.0f, 0},
        {1, 4100, 4000, 16, 4, 128, 0.0f, 0},      // long cache: split keys
        {1, 600, 600, 8, 8, 96, 30.0f, 0},         // soft-capping
        {2, 700, 650, 4, 1, 64, 0.0f, 128},        // sliding window, MQA
        {20, 300, 280, 4, 2, 64, 0.0f, 0},         // a block of queries (prefill kernel)
        {1, 64, 1, 4, 2, 40, 0.0f, 0},             // one valid key
    };
    for (Dtype dt : dtypes()) {
        std::uint64_t seed = 1100;
        for (const Case& c : cases) {
            const int Dq = c.hq * c.hd, Dkv = c.hkv * c.hd;
            const auto qv = random_values(std::size_t(c.lq) * Dq, seed += 7, -1.5f, 1.5f, dt);
            const auto kv = random_values(std::size_t(c.cap) * Dkv, seed + 1, -1.5f, 1.5f, dt);
            const auto vv = random_values(std::size_t(c.cap) * Dkv, seed + 2, -1.0f, 1.0f, dt);
            Tensor Oc;
            brotensor::flash_attention_decode(host(qv, c.lq, Dq), host(kv, c.cap, Dkv), host(vv, c.cap, Dkv), c.valid,
                                              c.hq, c.hkv, Oc, c.softcap, c.window);
            // Build the GPU cache by appending in two pieces.
            Tensor Kc = Tensor::zeros_on(vk(), c.cap, Dkv, dt), Vc = Tensor::zeros_on(vk(), c.cap, Dkv, dt);
            const int first = c.valid / 2;
            std::vector<float> k1(kv.begin(), kv.begin() + std::size_t(first) * Dkv);
            std::vector<float> v1(vv.begin(), vv.begin() + std::size_t(first) * Dkv);
            std::vector<float> k2(kv.begin() + std::size_t(first) * Dkv, kv.end());
            std::vector<float> v2(vv.begin() + std::size_t(first) * Dkv, vv.end());
            if (first > 0) brotensor::kv_cache_append(upload(k1, first, Dkv, dt), upload(v1, first, Dkv, dt), 0, Kc, Vc);
            brotensor::kv_cache_append(upload(k2, c.cap - first, Dkv, dt), upload(v2, c.cap - first, Dkv, dt), first, Kc, Vc);
            {
                const auto a = download(Kc), b = download(Vc);
                expect_equal_bits(a.data(), kv.data(), a.size() * 4, tagf("kv_cache_append K", dt, "", c.cap, Dkv, 0, 0));
                expect_equal_bits(b.data(), vv.data(), b.size() * 4, tagf("kv_cache_append V", dt, "", c.cap, Dkv, 0, 0));
            }
            Tensor Q = upload(qv, c.lq, Dq, dt);
            each_path([&](const char* path) {
                Tensor O;
                brotensor::flash_attention_decode(Q, Kc, Vc, c.valid, c.hq, c.hkv, O, c.softcap, c.window);
                close_dt(dt, download(O), Oc.to_host_vector(),
                         tagf("flash_attention_decode", dt, path, c.lq, c.valid, c.hq * 1000 + c.hkv, c.hd));
            });
            if (c.lq != 1) continue;
            // Masked: holes in the valid prefix, NaN in the masked V rows.
            std::vector<float> mv(std::size_t(c.cap), 0.0f);
            Rng r(seed + 5);
            for (int i = 0; i < c.valid; ++i) mv[i] = (r.next() % 4 == 0 && i != c.valid - 1) ? 0.0f : 1.0f;
            auto vnan = vv;
            for (int i = 0; i < c.cap; ++i)
                if (mv[i] == 0.0f) for (int j = 0; j < Dkv; ++j) vnan[std::size_t(i) * Dkv + j] = NAN;
            Tensor Om;
            brotensor::flash_attention_decode_masked(host(qv, 1, Dq), host(kv, c.cap, Dkv), host(vnan, c.cap, Dkv),
                                                     mv.data(), c.hq, c.hkv, Om, c.softcap, c.window);
            Tensor Vn = upload(vnan, c.cap, Dkv, dt);
            Tensor M = Tensor::from_host_on(vk(), mv.data(), c.cap, 1);
            each_path([&](const char* path) {
                Tensor O;
                brotensor::flash_attention_decode_masked(Q, Kc, Vn, static_cast<const float*>(M.data), c.hq, c.hkv, O,
                                                         c.softcap, c.window);
                close_dt(dt, download(O), Om.to_host_vector(),
                         tagf("flash_attention_decode_masked", dt, path, 1, c.cap, c.hq * 1000 + c.hkv, c.hd));
            });
        }
    }
}

// ─── projection-fused ──────────────────────────────────────────────────────

void test_fused() {
    std::printf("flash_attention_qkvo_forward / project_kv / q_with_kv_cached_forward\n");
    for (Dtype dt : dtypes()) {
        const int Lq = 40, Lk = 56, D = 128, Dc = 96, H = 2;
        auto W = [&](int r, int c, std::uint64_t s) { return random_values(std::size_t(r) * c, s, -0.15f, 0.15f, dt); };
        const auto x = random_values(std::size_t(Lq) * D, 1, -1.0f, 1.0f, dt);
        const auto ctx = random_values(std::size_t(Lk) * Dc, 2, -1.0f, 1.0f, dt);
        const auto wq = W(D, D, 3), wk = W(D, Dc, 4), wv = W(D, Dc, 5), wo = W(D, D, 6);
        const auto bq = W(D, 1, 7), bk = W(D, 1, 8), bv = W(D, 1, 9), bo = W(D, 1, 10);
        const auto mv = make_mask(Lk, 1, 11);
        Tensor Oc, Kc, Vc, Oq;
        {
            Tensor X = host(x, Lq, D), C = host(ctx, Lk, Dc);
            Tensor Wq = host(wq, D, D), Wk = host(wk, D, Dc), Wv = host(wv, D, Dc), Wo = host(wo, D, D);
            Tensor Bq = host(bq, D, 1), Bk = host(bk, D, 1), Bv = host(bv, D, 1), Bo = host(bo, D, 1);
            brotensor::flash_attention_qkvo_forward(X, &C, Wq, &Bq, Wk, &Bk, Wv, &Bv, Wo, &Bo, mv.data(), H, false, Oc);
            brotensor::flash_attention_project_kv(C, Wk, &Bk, Wv, &Bv, Kc, Vc);
            brotensor::flash_attention_q_with_kv_cached_forward(X, Kc, Vc, Wq, &Bq, Wo, &Bo, mv.data(), H, false, Oq);
        }
        Tensor X = upload(x, Lq, D, dt), C = upload(ctx, Lk, Dc, dt);
        Tensor Wq = upload(wq, D, D, dt), Wk = upload(wk, D, Dc, dt), Wv = upload(wv, D, Dc, dt), Wo = upload(wo, D, D, dt);
        Tensor Bq = upload(bq, D, 1, dt), Bk = upload(bk, D, 1, dt), Bv = upload(bv, D, 1, dt), Bo = upload(bo, D, 1, dt);
        Tensor M = Tensor::from_host_on(vk(), mv.data(), Lk, 1);
        const float* m = static_cast<const float*>(M.data);
        Tensor O, K, V, O2, Os;
        brotensor::flash_attention_qkvo_forward(X, &C, Wq, &Bq, Wk, &Bk, Wv, &Bv, Wo, &Bo, m, H, false, O);
        brotensor::flash_attention_project_kv(C, Wk, &Bk, Wv, &Bv, K, V);
        brotensor::flash_attention_q_with_kv_cached_forward(X, K, V, Wq, &Bq, Wo, &Bo, m, H, false, O2);
        // 16-bit: the projections round Q / K / V and the attention output.
        const float f = dt == Dtype::FP32 ? 1e-4f : 3.0f;
        auto close = [&](const Tensor& g, const Tensor& w, const char* what) {
            if (dt == Dtype::FP32) expect_close(download(g), w.to_host_vector(), f, f, tagf(what, dt, "", Lq, Lk, D, H));
            else expect_close(download(g), w.to_host_vector(), 4 * dtype_eps(dt) * f, 4 * dtype_eps(dt) * f,
                              tagf(what, dt, "", Lq, Lk, D, H));
        };
        close(O, Oc, "flash_attention_qkvo_forward cross");
        close(K, Kc, "flash_attention_project_kv K");
        close(V, Vc, "flash_attention_project_kv V");
        close(O2, Oq, "flash_attention_q_with_kv_cached_forward");
        // Self-attention (Ctx null), causal.
        Tensor Wk2 = upload(W(D, D, 12), D, D, dt), Wv2 = upload(W(D, D, 13), D, D, dt);
        Tensor Osc;
        brotensor::flash_attention_qkvo_forward(host(x, Lq, D), nullptr, host(wq, D, D), nullptr,
                                                host(download(Wk2), D, D), nullptr, host(download(Wv2), D, D), nullptr,
                                                host(wo, D, D), nullptr, nullptr, H, true, Osc);
        brotensor::flash_attention_qkvo_forward(X, nullptr, Wq, nullptr, Wk2, nullptr, Wv2, nullptr, Wo, nullptr,
                                                nullptr, H, true, Os);
        close(Os, Osc, "flash_attention_qkvo_forward self causal");
    }
}

}  // namespace

void run_attention_ops_tests();   // test_vulkan_attention_ops.cpp

void run_attention_tests() {
    test_flash_forward();
    test_views();
    test_varlen();
    test_packed_qkv();
    test_decode();
    test_fused();
    run_attention_ops_tests();
}

}  // namespace vkt
