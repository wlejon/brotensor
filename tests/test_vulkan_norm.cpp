// Vulkan parity for the transformer row ops against the CPU backend:
// LayerNorm (vector, batched inference with and without beta, FP16
// inference, FP32 -> FP16 output, training with caches, both backwards),
// RMSNorm (gamma in X's dtype and FP32, backward), per-head L2 norm, pixel
// norm, softmax (masked vector, rows, backward), every RoPE variant (theta,
// tables, offset table views, per-head, packed QKV with restarting positions
// over all heads, M-RoPE, GQA head counts), the GLU gates and modulate /
// broadcast_mul. FP32 / FP16 / BF16 storage; BF16 is computed directly (no
// conversion), the reference is the CPU op in FP32 on the rounded inputs.

#include "test_vulkan_common.h"

#include <brotensor/ops/fused.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace vkt {

namespace {

const Dtype kFloatTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

Tensor cpu_tensor(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

std::vector<float> rounded(std::vector<float> v, Dtype dt) {
    for (float& x : v) x = round_to(dt, x);
    return v;
}

void close_for(Dtype dt, const std::vector<float>& got, const std::vector<float>& want, float atol, float rtol,
               const std::string& t) {
    if (dt == Dtype::FP32) {
        expect_close(got, want, atol, rtol, t);
    } else {
        expect_close(got, rounded(want, dt), atol + 1e-6f, rtol + 2.0f * dtype_eps(dt), t);
    }
}

std::string tag(const char* op, Dtype dt, int r, int c, const char* extra = "") {
    return std::string(op) + " " + dt_name(dt) + " " + std::to_string(r) + "x" + std::to_string(c) + extra;
}

Tensor i32_tensor(const std::vector<std::int32_t>& v, Device d) {
    return Tensor::from_raw_bytes_on(d, v.data(), static_cast<int>(v.size()), 1, Dtype::INT32, v.size() * 4);
}

// Row shapes: single row, odd, a typical hidden size, wide rows, many rows.
const int kRowShapes[][2] = {{1, 1}, {3, 5}, {7, 768}, {2, 4097}, {300, 96}};

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void test_layernorm() {
    std::printf("layernorm\n");
    std::uint64_t seed = 1;
    for (Dtype dt : kFloatTypes) {
        for (const auto& s : kRowShapes) {
            const int R = s[0], D = s[1];
            // A large common offset: the case the two-pass variance exists for.
            const auto xv = random_values(std::size_t(R) * D, seed++, 380, 410, dt);
            const auto gv = random_values(std::size_t(D), seed++, 0.5f, 1.5f, dt);
            const auto bv = random_values(std::size_t(D), seed++, -0.5f, 0.5f, dt);
            Tensor xc = cpu_tensor(xv, R, D), gc = cpu_tensor(gv, D, 1), bc = cpu_tensor(bv, D, 1);
            Tensor xg = upload(xv, R, D, dt), gg = upload(gv, D, 1, dt), bg = upload(bv, D, 1, dt);
            // The CPU sums a row sequentially in FP32, so its mean of ~395
            // carries ~eps * sqrt(D) * 395 of rounding, and xhat that times
            // rstd (~1/9): the allowance below. The GPU's tree sum is closer
            // to the truth, which the double-precision check further down
            // holds to a tight bound.
            const float atol = 6e-6f * std::sqrt(float(D)) + 2e-5f;
            // Inference, with and without beta.
            Tensor yc, yg, nc, ng;
            brotensor::layernorm_forward_inference_batched(xc, gc, bc, yc, 1e-5f);
            brotensor::layernorm_forward_inference_batched(xg, gg, bg, yg, 1e-5f);
            VKT_CHECK(yg.dtype == dt && yg.rows == R && yg.cols == D);
            close_for(dt, download(yg), yc.to_host_vector(), atol, 2e-5f, tag("layernorm_inference", dt, R, D));
            if (dt == Dtype::FP32) {
                // Double-precision reference: two-pass variance on the GPU.
                const auto y = download(yg);
                std::vector<float> want(y.size());
                for (int r = 0; r < R; ++r) {
                    double m = 0, v = 0;
                    for (int i = 0; i < D; ++i) m += xv[std::size_t(r) * D + i];
                    m /= D;
                    for (int i = 0; i < D; ++i) v += (xv[std::size_t(r) * D + i] - m) * (xv[std::size_t(r) * D + i] - m);
                    const double rs = 1.0 / std::sqrt(v / D + 1e-5);
                    for (int i = 0; i < D; ++i) {
                        want[std::size_t(r) * D + i] = float(gv[i] * (xv[std::size_t(r) * D + i] - m) * rs + bv[i]);
                    }
                }
                // FP32 holds x ~ 395 to 3e-5, which x - mean carries times
                // rstd * gamma (< 0.2): a few of those is the floor.
                expect_close(y, want, 3e-5f, 1e-5f, tag("layernorm_inference vs double", dt, R, D));
            }
            brotensor::layernorm_forward_inference_batched(xc, gc, nc, 1e-5f);
            brotensor::layernorm_forward_inference_batched(xg, gg, ng, 1e-5f);
            close_for(dt, download(ng), nc.to_host_vector(), atol, 2e-5f, tag("layernorm_inference no beta", dt, R, D));
            if (dt == Dtype::FP16) {
                Tensor hg;
                brotensor::layernorm_forward_inference_batched_fp16(xg, gg, bg, hg, 1e-5f);
                close_for(dt, download(hg), yc.to_host_vector(), atol, 2e-5f, tag("layernorm_inference_fp16", dt, R, D));
            }
            if (dt == Dtype::FP32) {   // FP32 X into a pre-typed FP16 Y
                Tensor hy = Tensor::empty_on(vk(), R, D, Dtype::FP16);
                brotensor::layernorm_forward_inference_batched(xg, gg, bg, hy, 1e-5f);
                VKT_CHECK(hy.dtype == Dtype::FP16);
                close_for(Dtype::FP16, download(hy), yc.to_host_vector(), atol, 2e-5f,
                          tag("layernorm_inference f32 -> f16", dt, R, D));
            }
            // Training forward with caches, and its backward.
            Tensor y2c, hc, mc, rc, y2g, hgc, mg, rg;
            brotensor::layernorm_forward_batched_with_caches(xc, gc, bc, y2c, hc, mc, rc, 1e-5f);
            brotensor::layernorm_forward_batched_with_caches(xg, gg, bg, y2g, hgc, mg, rg, 1e-5f);
            VKT_CHECK(mg.dtype == Dtype::FP32 && rg.dtype == Dtype::FP32 && mg.rows == R);
            close_for(dt, download(y2g), y2c.to_host_vector(), atol, 2e-5f, tag("layernorm_caches y", dt, R, D));
            close_for(dt, download(hgc), hc.to_host_vector(), atol, 2e-5f, tag("layernorm_caches xhat", dt, R, D));
            expect_close(download(mg), mc.to_host_vector(), 3e-5f * std::sqrt(float(D)), 1e-6f,
                         tag("layernorm_caches mean", dt, R, D));
            expect_close(download(rg), rc.to_host_vector(), 0, 2e-5f, tag("layernorm_caches rstd", dt, R, D));
            const auto dyv = random_values(std::size_t(R) * D, seed++, -1, 1, dt);
            const auto dg0 = random_values(std::size_t(D), seed++, -1, 1, dt);
            const auto db0 = random_values(std::size_t(D), seed++, -1, 1, dt);
            // The backward consumes the forward's caches; feed both sides the
            // same (rounded) Xhat and Rstd so only the backward is compared.
            const auto xhv = rounded(download(hgc), dt);
            const auto rsv = download(rg);
            Tensor dxc, dGc = cpu_tensor(dg0, D, 1), dBc = cpu_tensor(db0, D, 1);
            brotensor::layernorm_backward_batched_with_caches(cpu_tensor(dyv, R, D), cpu_tensor(xhv, R, D), gc,
                                                              cpu_tensor(rsv, R, 1), dxc, dGc, dBc);
            Tensor dxg, dGg = upload(dg0, D, 1, dt), dBg = upload(db0, D, 1, dt);
            brotensor::layernorm_backward_batched_with_caches(upload(dyv, R, D, dt), upload(xhv, R, D, dt), gg,
                                                              upload(rsv, R, 1, Dtype::FP32), dxg, dGg, dBg);
            const float colat = 2e-6f * std::sqrt(float(R)) + 1e-6f;
            close_for(dt, download(dxg), dxc.to_host_vector(), 2e-4f, 2e-5f, tag("layernorm_backward dX", dt, R, D));
            close_for(dt, download(dGg), dGc.to_host_vector(), colat, 2e-5f, tag("layernorm_backward dGamma", dt, R, D));
            close_for(dt, download(dBg), dBc.to_host_vector(), colat, 2e-5f, tag("layernorm_backward dBeta", dt, R, D));
        }
        // The single-vector forms; mean and rstd come back to the host.
        const int n = 1000;
        const auto xv = random_values(n, seed++, -3, 5, dt);
        const auto gv = random_values(n, seed++, 0.5f, 1.5f, dt);
        const auto bv = random_values(n, seed++, -0.5f, 0.5f, dt);
        Tensor yc, hc, yg, hg;
        float mc = 0, rc = 0, mg = 0, rg = 0;
        brotensor::layernorm_forward(cpu_tensor(xv, n, 1), cpu_tensor(gv, n, 1), cpu_tensor(bv, n, 1), yc, hc, mc, rc, 1e-5f);
        brotensor::layernorm_forward(upload(xv, n, 1, dt), upload(gv, n, 1, dt), upload(bv, n, 1, dt), yg, hg, mg, rg, 1e-5f);
        close_for(dt, download(yg), yc.to_host_vector(), 2e-5f, 2e-5f, tag("layernorm_forward vector", dt, n, 1));
        expect_close({mg, rg}, {mc, rc}, 1e-6f, 2e-6f, tag("layernorm_forward mean/rstd", dt, n, 1));
        const auto dyv = random_values(n, seed++, -1, 1, dt);
        const auto xhv = rounded(download(hg), dt);
        Tensor dxc, dGc = Tensor::zeros_on(Device::cpu(), n, 1), dBc = Tensor::zeros_on(Device::cpu(), n, 1);
        brotensor::layernorm_backward(cpu_tensor(dyv, n, 1), cpu_tensor(xhv, n, 1), cpu_tensor(gv, n, 1), rg, dxc, dGc, dBc);
        Tensor dxg, dGg = Tensor::zeros_on(vk(), n, 1, dt), dBg = Tensor::zeros_on(vk(), n, 1, dt);
        brotensor::layernorm_backward(upload(dyv, n, 1, dt), upload(xhv, n, 1, dt), upload(gv, n, 1, dt), rg, dxg, dGg, dBg);
        close_for(dt, download(dxg), dxc.to_host_vector(), 2e-5f, 2e-5f, tag("layernorm_backward vector dX", dt, n, 1));
        close_for(dt, download(dGg), dGc.to_host_vector(), 1e-6f, 1e-6f, tag("layernorm_backward vector dGamma", dt, n, 1));
        close_for(dt, download(dBg), dBc.to_host_vector(), 0, 0, tag("layernorm_backward vector dBeta", dt, n, 1));
    }
}

// ─── RMSNorm, L2, pixel norm ───────────────────────────────────────────────

void test_rms_l2() {
    std::printf("rms_norm / l2_norm / pixel_norm\n");
    std::uint64_t seed = 100;
    for (Dtype dt : kFloatTypes) {
        for (const auto& s : kRowShapes) {
            const int R = s[0], D = s[1];
            const auto xv = random_values(std::size_t(R) * D, seed++, -2, 2, dt);
            const auto dyv = random_values(std::size_t(R) * D, seed++, -1, 1, dt);
            for (Dtype gdt : {dt, Dtype::FP32}) {
                const auto gv = random_values(std::size_t(D), seed++, 0.5f, 1.5f, gdt);
                const auto g0 = random_values(std::size_t(D), seed++, -1, 1, gdt);
                Tensor yc, yg;
                brotensor::rms_norm_forward(cpu_tensor(xv, R, D), cpu_tensor(gv, D, 1), 1e-6f, yc);
                brotensor::rms_norm_forward(upload(xv, R, D, dt), upload(gv, D, 1, gdt), 1e-6f, yg);
                VKT_CHECK(yg.dtype == dt);
                const char* g32 = gdt != dt ? " gamma f32" : "";
                close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 2e-6f, tag("rms_norm", dt, R, D, g32));
                Tensor dxc, dGc = cpu_tensor(g0, D, 1), dxg, dGg = upload(g0, D, 1, gdt);
                brotensor::rms_norm_backward(cpu_tensor(xv, R, D), cpu_tensor(gv, D, 1), cpu_tensor(dyv, R, D), 1e-6f,
                                             dxc, dGc);
                brotensor::rms_norm_backward(upload(xv, R, D, dt), upload(gv, D, 1, gdt), upload(dyv, R, D, dt), 1e-6f,
                                             dxg, dGg);
                close_for(dt, download(dxg), dxc.to_host_vector(), 2e-6f, 1e-5f, tag("rms_norm_backward dX", dt, R, D, g32));
                // The CPU's sequential sum of squares puts ~eps * sqrt(D) on
                // its rrms, which every dGamma term carries.
                close_for(gdt, download(dGg), dGc.to_host_vector(),
                          5e-7f * std::sqrt(float(D)) * R + 2e-6f * std::sqrt(float(R)) + 1e-6f, 1e-5f,
                          tag("rms_norm_backward dGamma", dt, R, D, g32));
                if (dt == Dtype::FP32) break;
            }
            Tensor pc, pg, pdc, pdg;
            brotensor::pixel_norm_forward(cpu_tensor(xv, R, D), 1e-8f, pc);
            brotensor::pixel_norm_forward(upload(xv, R, D, dt), 1e-8f, pg);
            close_for(dt, download(pg), pc.to_host_vector(), 1e-6f, 2e-6f, tag("pixel_norm", dt, R, D));
            brotensor::pixel_norm_backward(cpu_tensor(xv, R, D), cpu_tensor(dyv, R, D), 1e-8f, pdc);
            brotensor::pixel_norm_backward(upload(xv, R, D, dt), upload(dyv, R, D, dt), 1e-8f, pdg);
            close_for(dt, download(pdg), pdc.to_host_vector(), 2e-6f, 1e-5f, tag("pixel_norm_backward", dt, R, D));
        }
        // Per-head L2 norm: (L, heads * head_dim), GQA-like head counts.
        for (const auto& h : {std::array<int, 3>{5, 8, 64}, std::array<int, 3>{33, 2, 128}, std::array<int, 3>{1, 3, 6}}) {
            const int L = h[0], H = h[1], hd = h[2];
            const auto xv = random_values(std::size_t(L) * H * hd, seed++, -2, 2, dt);
            const auto dyv = random_values(std::size_t(L) * H * hd, seed++, -1, 1, dt);
            Tensor yc, yg, dc, dg;
            brotensor::l2_norm_forward(cpu_tensor(xv, L, H * hd), hd, H, 1e-6f, yc);
            brotensor::l2_norm_forward(upload(xv, L, H * hd, dt), hd, H, 1e-6f, yg);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 2e-6f, tag("l2_norm", dt, L, H * hd));
            brotensor::l2_norm_backward(cpu_tensor(xv, L, H * hd), hd, H, 1e-6f, cpu_tensor(dyv, L, H * hd), dc);
            brotensor::l2_norm_backward(upload(xv, L, H * hd, dt), hd, H, 1e-6f, upload(dyv, L, H * hd, dt), dg);
            close_for(dt, download(dg), dc.to_host_vector(), 2e-6f, 1e-5f, tag("l2_norm_backward", dt, L, H * hd));
        }
    }
}

// ─── softmax ───────────────────────────────────────────────────────────────

void test_softmax() {
    std::printf("softmax\n");
    std::uint64_t seed = 200;
    for (Dtype dt : kFloatTypes) {
        for (const auto& s : kRowShapes) {
            const int R = s[0], D = s[1];
            const auto xv = random_values(std::size_t(R) * D, seed++, -8, 8, dt);
            Tensor yc, yg;
            brotensor::softmax_rows_forward(cpu_tensor(xv, R, D), yc, R, D);
            brotensor::softmax_rows_forward(upload(xv, R, D, dt), yg, R, D);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-7f, 1e-5f, tag("softmax_rows", dt, R, D));
        }
        // In place, as the attention code calls it.
        {
            const auto xv = random_values(64 * 129, seed++, -4, 4, dt);
            Tensor yc, xg = upload(xv, 64, 129, dt);
            brotensor::softmax_rows_forward(cpu_tensor(xv, 64, 129), yc, 64, 129);
            brotensor::softmax_rows_forward(xg, xg, 64, 129);
            close_for(dt, download(xg), yc.to_host_vector(), 1e-7f, 1e-5f, tag("softmax_rows in place", dt, 64, 129));
        }
        // The vector form with a mask (entries < 0.5 excluded, written 0),
        // including a fully masked vector.
        for (int n : {1, 37, 5000}) {
            const auto xv = random_values(std::size_t(n), seed++, -10, 10, dt);
            std::vector<float> mv(static_cast<std::size_t>(n));
            Rng r(seed++);
            for (float& m : mv) m = (r.next() % 3 == 0) ? 0.0f : 1.0f;
            for (bool all_off : {false, true}) {
                if (all_off) std::fill(mv.begin(), mv.end(), 0.0f);
                Tensor mg = upload(mv, n, 1, Dtype::FP32);
                Tensor yc, yg;
                brotensor::softmax_forward(cpu_tensor(xv, n, 1), yc, mv.data());
                brotensor::softmax_forward(upload(xv, n, 1, dt), yg, static_cast<const float*>(mg.data));
                close_for(dt, download(yg), yc.to_host_vector(), 1e-7f, 1e-5f,
                          tag("softmax masked", dt, n, 1, all_off ? " all masked" : ""));
            }
            Tensor yc, yg;
            brotensor::softmax_forward(cpu_tensor(xv, n, 1), yc, nullptr);
            brotensor::softmax_forward(upload(xv, n, 1, dt), yg, nullptr);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-7f, 1e-5f, tag("softmax vector", dt, n, 1));
            const auto pv = rounded(download(yg), dt);
            const auto dv = random_values(std::size_t(n), seed++, -1, 1, dt);
            Tensor dc, dg;
            brotensor::softmax_backward(cpu_tensor(pv, n, 1), cpu_tensor(dv, n, 1), dc);
            brotensor::softmax_backward(upload(pv, n, 1, dt), upload(dv, n, 1, dt), dg);
            close_for(dt, download(dg), dc.to_host_vector(), 1e-6f, 1e-5f, tag("softmax_backward", dt, n, 1));
        }
    }
}

// ─── RoPE ──────────────────────────────────────────────────────────────────

void tables(int rows, int half, std::uint64_t seed, std::vector<float>& c, std::vector<float>& s) {
    Rng r(seed);
    c.resize(std::size_t(rows) * half);
    s.resize(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        const float a = r.uniform(-3.14159f, 3.14159f);
        c[i] = std::cos(a);
        s[i] = std::sin(a);
    }
}

void test_rope() {
    std::printf("rope\n");
    std::uint64_t seed = 300;
    // (L, heads, head_dim, offset): Q and GQA K head counts, odd L, decode
    // positions. The angle itself carries FP32 rounding of pos * inv_freq,
    // so the tolerance grows with the position.
    const int shapes[][4] = {{1, 1, 2, 0}, {37, 8, 64, 0}, {37, 2, 64, 0}, {5, 16, 128, 1000}, {3, 4, 128, 30000}, {11, 3, 6, 7}};
    for (Dtype dt : kFloatTypes) {
        for (const auto& sh : shapes) {
            const int L = sh[0], H = sh[1], hd = sh[2], off = sh[3];
            const int D = H * hd, half = hd / 2;
            const auto xv = random_values(std::size_t(L) * D, seed++, -2, 2, dt);
            const float atol = 8e-8f * float(off + L) * 2.0f + 2e-6f;
            for (bool bwd : {false, true}) {
                Tensor yc, yg;
                if (bwd) {
                    brotensor::rope_backward(cpu_tensor(xv, L, D), hd, H, off, 10000.0f, yc);
                    brotensor::rope_backward(upload(xv, L, D, dt), hd, H, off, 10000.0f, yg);
                } else {
                    brotensor::rope_forward(cpu_tensor(xv, L, D), hd, H, off, 1000000.0f, yc);
                    brotensor::rope_forward(upload(xv, L, D, dt), hd, H, off, 1000000.0f, yg);
                }
                VKT_CHECK(yg.dtype == dt && yg.rows == L && yg.cols == D);
                close_for(dt, download(yg), yc.to_host_vector(), atol, 2e-6f,
                          tag(bwd ? "rope_backward" : "rope_forward", dt, L, D, (" pos " + std::to_string(off)).c_str()));
            }
            // Tables: forward, backward, and an offset view into a longer table
            // (the decode pattern) against a fresh table of the same rows.
            std::vector<float> cv, sv;
            tables(L + 9, half, seed++, cv, sv);
            Tensor ctg = upload(cv, L + 9, half, Dtype::FP32), stg = upload(sv, L + 9, half, Dtype::FP32);
            const std::vector<float> cs(cv.begin() + 9 * half, cv.end()), ss(sv.begin() + 9 * half, sv.end());
            Tensor cview = Tensor::view(vk(), static_cast<float*>(ctg.data) + 9 * half, L, half, Dtype::FP32);
            Tensor sview = Tensor::view(vk(), static_cast<float*>(stg.data) + 9 * half, L, half, Dtype::FP32);
            Tensor yc, yg, yf, bc, bg;
            brotensor::rope_apply(cpu_tensor(xv, L, D), cpu_tensor(cs, L, half), cpu_tensor(ss, L, half), hd, H, yc);
            brotensor::rope_apply(upload(xv, L, D, dt), cview, sview, hd, H, yg);
            brotensor::rope_apply(upload(xv, L, D, dt), upload(cs, L, half, Dtype::FP32), upload(ss, L, half, Dtype::FP32),
                                  hd, H, yf);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 1e-6f, tag("rope_apply view", dt, L, D));
            const auto a = download(yg), b = download(yf);
            expect_equal_bits(a.data(), b.data(), a.size() * 4, tag("rope_apply view == fresh table", dt, L, D));
            brotensor::rope_apply_backward(cpu_tensor(xv, L, D), cpu_tensor(cs, L, half), cpu_tensor(ss, L, half), hd, H, bc);
            brotensor::rope_apply_backward(upload(xv, L, D, dt), cview, sview, hd, H, bg);
            close_for(dt, download(bg), bc.to_host_vector(), 1e-6f, 1e-6f, tag("rope_apply_backward", dt, L, D));
            // Per-head tables.
            std::vector<float> pcv, psv;
            tables(L * H, half, seed++, pcv, psv);
            Tensor pc, pg;
            brotensor::rope_apply_perhead(cpu_tensor(xv, L, D), cpu_tensor(pcv, L * H, half), cpu_tensor(psv, L * H, half),
                                          hd, H, pc);
            brotensor::rope_apply_perhead(upload(xv, L, D, dt), upload(pcv, L * H, half, Dtype::FP32),
                                          upload(psv, L * H, half, Dtype::FP32), hd, H, pg);
            close_for(dt, download(pg), pc.to_host_vector(), 1e-6f, 1e-6f, tag("rope_apply_perhead", dt, L, D));
        }
        // Packed QKV, in place over Q and K of every head; two packed
        // sequences whose positions restart at 0; V untouched.
        for (const auto& sh : {std::array<int, 3>{13, 4, 32}, std::array<int, 3>{40, 12, 64}}) {
            const int L = sh[0], H = sh[1], hd = sh[2], D = H * hd, half = hd / 2, P = 32;
            const auto qkv = random_values(std::size_t(L) * 3 * D, seed++, -2, 2, dt);
            std::vector<float> cv, sv;
            tables(P, half, seed++, cv, sv);
            std::vector<std::int32_t> pos(static_cast<std::size_t>(L));
            for (int r = 0; r < L; ++r) pos[std::size_t(r)] = r < L / 2 ? r : r - L / 2;
            Tensor qc = cpu_tensor(qkv, L, 3 * D);
            Tensor pcpu = i32_tensor(pos, Device::cpu());
            brotensor::rope_qkv_packed_inplace(qc, cpu_tensor(cv, P, half), cpu_tensor(sv, P, half), pcpu, H, hd);
            Tensor qg = upload(qkv, L, 3 * D, dt);
            brotensor::rope_qkv_packed_inplace(qg, upload(cv, P, half, Dtype::FP32), upload(sv, P, half, Dtype::FP32),
                                               i32_tensor(pos, vk()), H, hd);
            close_for(dt, download(qg), qc.to_host_vector(), 1e-6f, 1e-6f, tag("rope_qkv_packed_inplace", dt, L, 3 * D));
        }
        // M-RoPE: three position streams over sub-ranges of the pairs.
        {
            const int L = 23, H = 4, dt_ = 8, dh = 12, dw = 12, hd = 2 * (dt_ + dh + dw), D = H * hd, P = 40;
            const auto xv = random_values(std::size_t(L) * D, seed++, -2, 2, dt);
            std::vector<float> ct, st, ch, sh, cw, sw;
            tables(P, dt_, seed++, ct, st);
            tables(P, dh, seed++, ch, sh);
            tables(P, dw, seed++, cw, sw);
            std::vector<std::int32_t> pt(L), ph(L), pw(L);
            for (int r = 0; r < L; ++r) { pt[r] = r; ph[r] = (r * 7) % P; pw[r] = (r * 13 + 5) % P; }
            Tensor yc, yg;
            brotensor::rope_apply_mrope(cpu_tensor(xv, L, D), cpu_tensor(ct, P, dt_), cpu_tensor(st, P, dt_),
                                        cpu_tensor(ch, P, dh), cpu_tensor(sh, P, dh), cpu_tensor(cw, P, dw),
                                        cpu_tensor(sw, P, dw), pt.data(), ph.data(), pw.data(), hd, H, dt_, dh, dw, yc);
            Tensor gpt = i32_tensor(pt, vk()), gph = i32_tensor(ph, vk()), gpw = i32_tensor(pw, vk());
            brotensor::rope_apply_mrope(upload(xv, L, D, dt), upload(ct, P, dt_, Dtype::FP32), upload(st, P, dt_, Dtype::FP32),
                                        upload(ch, P, dh, Dtype::FP32), upload(sh, P, dh, Dtype::FP32),
                                        upload(cw, P, dw, Dtype::FP32), upload(sw, P, dw, Dtype::FP32),
                                        static_cast<const std::int32_t*>(gpt.data), static_cast<const std::int32_t*>(gph.data),
                                        static_cast<const std::int32_t*>(gpw.data), hd, H, dt_, dh, dw, yg);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 1e-6f, tag("rope_apply_mrope", dt, L, D));
        }
    }
}

// ─── GLU gates, modulate, broadcast_mul ────────────────────────────────────

void test_glu_rowvec() {
    std::printf("glu / modulate / broadcast_mul\n");
    std::uint64_t seed = 400;
    using Fwd = void (*)(const Tensor&, Tensor&);
    using Bwd = void (*)(const Tensor&, const Tensor&, Tensor&);
    struct G { const char* name; Fwd f; Bwd b; };
    const G gates[] = {
        {"swiglu", &brotensor::swiglu_forward, &brotensor::swiglu_backward},
        {"geglu", &brotensor::geglu_forward, &brotensor::geglu_backward},
        {"geglu_exact", &brotensor::geglu_exact_forward, &brotensor::geglu_exact_backward},
    };
    for (Dtype dt : kFloatTypes) {
        for (const G& g : gates) {
            for (const auto& s : {std::array<int, 2>{1, 2}, std::array<int, 2>{17, 3072}, std::array<int, 2>{300, 130}}) {
                const int R = s[0], C = s[1], D = C / 2;
                const auto xv = random_values(std::size_t(R) * C, seed++, -5, 5, dt);
                const auto dyv = random_values(std::size_t(R) * D, seed++, -1, 1, dt);
                Tensor yc, yg, dc, dg;
                g.f(cpu_tensor(xv, R, C), yc);
                g.f(upload(xv, R, C, dt), yg);
                VKT_CHECK(yg.rows == R && yg.cols == D && yg.dtype == dt);
                close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 1e-5f, tag(g.name, dt, R, C));
                g.b(cpu_tensor(xv, R, C), cpu_tensor(dyv, R, D), dc);
                g.b(upload(xv, R, C, dt), upload(dyv, R, D, dt), dg);
                close_for(dt, download(dg), dc.to_host_vector(), 2e-6f, 1e-5f, tag((std::string(g.name) + "_backward").c_str(), dt, R, C));
            }
        }
        for (const auto& s : kRowShapes) {
            const int L = s[0], D = s[1];
            const auto xv = random_values(std::size_t(L) * D, seed++, -2, 2, dt);
            const auto sc = random_values(std::size_t(D), seed++, -1, 1, dt);
            const auto sh = random_values(std::size_t(D), seed++, -1, 1, dt);
            Tensor mc, mg, bc, bg;
            brotensor::modulate(cpu_tensor(xv, L, D), cpu_tensor(sc, D, 1), cpu_tensor(sh, D, 1), mc);
            brotensor::modulate(upload(xv, L, D, dt), upload(sc, D, 1, dt), upload(sh, D, 1, dt), mg);
            close_for(dt, download(mg), mc.to_host_vector(), 1e-6f, 1e-6f, tag("modulate", dt, L, D));
            brotensor::broadcast_mul(cpu_tensor(xv, L, D), cpu_tensor(sc, D, 1), bc);
            brotensor::broadcast_mul(upload(xv, L, D, dt), upload(sc, D, 1, dt), bg);
            close_for(dt, download(bg), bc.to_host_vector(), 0, 0, tag("broadcast_mul", dt, L, D));
        }
        // The fused composites in ops/fused.h, over the ops above.
        {
            const int R = 9, D = 256;
            const auto xv = random_values(std::size_t(R) * D, seed++, -2, 2, dt);
            const auto gv = random_values(std::size_t(D), seed++, 0.5f, 1.5f, dt);
            const auto bv = random_values(std::size_t(D), seed++, -0.5f, 0.5f, dt);
            const auto sc = random_values(std::size_t(D), seed++, -1, 1, dt);
            const auto sh = random_values(std::size_t(D), seed++, -1, 1, dt);
            Tensor oc, og;
            brotensor::fused_layernorm_modulate(cpu_tensor(xv, R, D), cpu_tensor(gv, D, 1), cpu_tensor(bv, D, 1),
                                                cpu_tensor(sc, D, 1), cpu_tensor(sh, D, 1), 1e-6f, oc);
            brotensor::fused_layernorm_modulate(upload(xv, R, D, dt), upload(gv, D, 1, dt), upload(bv, D, 1, dt),
                                                upload(sc, D, 1, dt), upload(sh, D, 1, dt), 1e-6f, og);
            // 16-bit: the composite rounds the LayerNorm output (|y| < 3)
            // to the dtype before modulating it by up to 2x.
            const float rnd = dt == Dtype::FP32 ? 0.0f : 12.0f * dtype_eps(dt);
            close_for(dt, download(og), oc.to_host_vector(), 2e-5f + rnd, 2e-5f, tag("fused_layernorm_modulate", dt, R, D));
            Tensor hc = cpu_tensor(xv, R, D), hg = upload(xv, R, D, dt), rc, rg;
            const auto pv = random_values(std::size_t(R) * D, seed++, -1, 1, dt);
            brotensor::fused_residual_rmsnorm(hc, cpu_tensor(pv, R, D), cpu_tensor(gv, D, 1), 1e-6f, rc);
            brotensor::fused_residual_rmsnorm(hg, upload(pv, R, D, dt), upload(gv, D, 1, dt), 1e-6f, rg);
            close_for(dt, download(rg), rc.to_host_vector(), 1e-5f, 1e-5f + 2.0f * dtype_eps(dt), tag("fused_residual_rmsnorm", dt, R, D));
        }
    }
}

}  // namespace

void run_norm_tests() {
    test_layernorm();
    test_rms_l2();
    test_softmax();
    test_rope();
    test_glu_rowvec();
}

}  // namespace vkt
