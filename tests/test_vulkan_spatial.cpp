// Vulkan parity for the spatial and diffusion ops against the CPU backend:
// interpolation (every mode, half-pixel and corner-aligned, up / down / odd
// sizes), the 2x resamples and their backwards, pooling, padding, crop,
// unfold, window partition, the pixel (un)shuffles, unpatchify, row gather /
// scatter, the channel concat, convex upsample, the NCHW norms (GroupNorm,
// BatchNorm inference, L2), the sampler steps on real scheduler sequences,
// the timestep embedding, Philox noise and the image helpers. FP32 / FP16 /
// BF16 storage; pure data movement must be bit-exact.

#include "test_vulkan_common.h"

#include <cmath>
#include <string>

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
    if (dt == Dtype::FP32) expect_close(got, want, atol, rtol, t);
    else expect_close(got, rounded(want, dt), atol + 1e-6f, rtol + 2.0f * dtype_eps(dt), t);
}

void exact(const std::vector<float>& got, const std::vector<float>& want, const std::string& t) {
    expect_close(got, want, 0, 0, t);
}

std::string tg(const char* op, Dtype dt, const std::string& extra) {
    return std::string(op) + " " + dt_name(dt) + " " + extra;
}

std::string dims(int a, int b, int c, int d) {
    return std::to_string(a) + "x" + std::to_string(b) + "x" + std::to_string(c) + "x" + std::to_string(d);
}

// ─── resampling ────────────────────────────────────────────────────────────

void test_interp() {
    std::printf("interp2d\n");
    struct S { int n, c, hi, wi, ho, wo; };
    const S shapes[] = {{1, 3, 7, 9, 14, 18}, {2, 4, 20, 17, 7, 5}, {1, 2, 5, 5, 13, 3},
                        {1, 3, 6, 8, 6, 8},   {1, 2, 4, 6, 1, 1},   {1, 17, 9, 11, 12, 15}};
    std::uint64_t seed = 1;
    for (Dtype dt : kFloatTypes) {
        for (const S& s : shapes) {
            const auto xv = random_values(std::size_t(s.n) * s.c * s.hi * s.wi, seed++, -2, 2, dt);
            Tensor xc = cpu_tensor(xv, s.n, s.c * s.hi * s.wi), xg = upload(xv, s.n, s.c * s.hi * s.wi, dt);
            for (int align = 0; align < 2; ++align) {
                for (int mode = 0; mode < 4; ++mode) {
                    Tensor yc, yg;
                    if (align) {
                        brotensor::interp2d_align_corners_forward(xc, s.n, s.c, s.hi, s.wi, s.ho, s.wo, mode, yc);
                        brotensor::interp2d_align_corners_forward(xg, s.n, s.c, s.hi, s.wi, s.ho, s.wo, mode, yg);
                    } else {
                        brotensor::interp2d_forward(xc, s.n, s.c, s.hi, s.wi, s.ho, s.wo, mode, yc);
                        brotensor::interp2d_forward(xg, s.n, s.c, s.hi, s.wi, s.ho, s.wo, mode, yg);
                    }
                    VKT_CHECK(yg.rows == yc.rows && yg.cols == yc.cols && yg.dtype == dt);
                    const std::string t = tg(align ? "interp2d_align_corners" : "interp2d", dt,
                                             dims(s.hi, s.wi, s.ho, s.wo) + " mode " + std::to_string(mode));
                    if (mode == 0) exact(download(yg), rounded(yc.to_host_vector(), dt), t);
                    else close_for(dt, download(yg), yc.to_host_vector(), 4e-6f, 4e-6f, t);
                }
            }
        }
    }
    Tensor x = upload(random_values(16, 9, -1, 1, Dtype::FP32), 1, 16, Dtype::FP32), y;
    VKT_CHECK(throws([&] { brotensor::interp2d_forward(x, 1, 1, 4, 4, 8, 8, 4, y); }));
}

void test_resample_2x() {
    std::printf("2x resamples\n");
    std::uint64_t seed = 100;
    for (Dtype dt : kFloatTypes) {
        const int N = 2, C = 3, H = 6, W = 10;
        const auto xv = random_values(std::size_t(N) * C * H * W, seed++, -2, 2, dt);
        const auto dv = random_values(std::size_t(N) * C * 4 * H * W, seed++, -2, 2, dt);
        Tensor xc = cpu_tensor(xv, N, C * H * W), xg = upload(xv, N, C * H * W, dt);
        Tensor dc = cpu_tensor(dv, N, 4 * C * H * W), dg = upload(dv, N, 4 * C * H * W, dt);
        Tensor yc, yg;
        brotensor::upsample_nearest_2x(xc, N, C, H, W, yc);
        brotensor::upsample_nearest_2x(xg, N, C, H, W, yg);
        exact(download(yg), yc.to_host_vector(), tg("upsample_nearest_2x", dt, ""));
        brotensor::upsample_bilinear_2x(xc, N, C, H, W, yc);
        brotensor::upsample_bilinear_2x(xg, N, C, H, W, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 2e-6f, tg("upsample_bilinear_2x", dt, ""));
        brotensor::downsample_avg_2x(xc, N, C, H, W, yc);
        brotensor::downsample_avg_2x(xg, N, C, H, W, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 1e-6f, tg("downsample_avg_2x", dt, ""));
        brotensor::upsample_nearest_2x_backward(dc, N, C, H, W, yc);
        brotensor::upsample_nearest_2x_backward(dg, N, C, H, W, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 0, 0, tg("upsample_nearest_2x_backward", dt, ""));
        brotensor::downsample_avg_2x_backward(xc, N, C, H, W, yc);
        brotensor::downsample_avg_2x_backward(xg, N, C, H, W, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 0, 0, tg("downsample_avg_2x_backward", dt, ""));
    }
}

void test_pool() {
    std::printf("pooling\n");
    std::uint64_t seed = 200;
    for (Dtype dt : kFloatTypes) {
        struct A { int h, w, ho, wo; };
        for (const A& a : {A{7, 7, 3, 2}, A{10, 10, 1, 1}, A{5, 7, 5, 7}, A{9, 13, 4, 6}}) {
            const int N = 2, C = 3;
            const auto xv = random_values(std::size_t(N) * C * a.h * a.w, seed++, -2, 2, dt);
            Tensor xc = cpu_tensor(xv, N, C * a.h * a.w), xg = upload(xv, N, C * a.h * a.w, dt);
            Tensor yc, yg;
            brotensor::adaptive_avg_pool2d_forward(xc, N, C, a.h, a.w, a.ho, a.wo, yc);
            brotensor::adaptive_avg_pool2d_forward(xg, N, C, a.h, a.w, a.ho, a.wo, yg);
            close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 2e-6f,
                      tg("adaptive_avg_pool2d", dt, dims(a.h, a.w, a.ho, a.wo)));
        }
        struct M { int h, w, k, s, p; };
        for (const M& m : {M{8, 8, 2, 2, 0}, M{9, 11, 3, 2, 1}, M{7, 7, 3, 1, 1}, M{6, 5, 5, 3, 2}}) {
            const int N = 2, C = 4;
            // Distinct values so the argmax is unique.
            std::vector<float> xv(std::size_t(N) * C * m.h * m.w);
            Rng r(seed++);
            for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = round_to(dt, float(int(r.next() % 4001) - 2000) / 64.0f);
            Tensor xc = cpu_tensor(xv, N, C * m.h * m.w), xg = upload(xv, N, C * m.h * m.w, dt);
            Tensor yc, yg, ic, ig;
            brotensor::max_pool2d_forward(xc, N, C, m.h, m.w, m.k, m.k, m.s, m.s, m.p, m.p, yc, ic);
            brotensor::max_pool2d_forward(xg, N, C, m.h, m.w, m.k, m.k, m.s, m.s, m.p, m.p, yg, ig);
            const std::string t = tg("max_pool2d", dt, dims(m.h, m.w, m.k, m.s) + " p" + std::to_string(m.p));
            exact(download(yg), yc.to_host_vector(), t);
            VKT_CHECK(ig.dtype == Dtype::INT32 && ig.rows == ic.rows && ig.cols == ic.cols);
            std::vector<std::int32_t> a(static_cast<std::size_t>(ic.size())), b(a.size());
            ic.copy_to_host_raw(a.data(), a.size() * 4);
            ig.copy_to_host_raw(b.data(), b.size() * 4);
            expect_equal_bits(a.data(), b.data(), a.size() * 4, t + " Idx");
        }
    }
}

void test_convex() {
    std::printf("convex_upsample\n");
    std::uint64_t seed = 300;
    for (Dtype dt : kFloatTypes) {
        const int N = 2, C = 3, H = 5, W = 6, S = 4;
        const auto xv = random_values(std::size_t(N) * C * H * W, seed++, -2, 2, dt);
        const auto mv = random_values(std::size_t(N) * 9 * S * S * H * W, seed++, -3, 3, dt);
        Tensor xc = cpu_tensor(xv, N, C * H * W), xg = upload(xv, N, C * H * W, dt);
        Tensor mc = cpu_tensor(mv, N, 9 * S * S * H * W), mg = upload(mv, N, 9 * S * S * H * W, dt);
        Tensor yc, yg;
        brotensor::convex_upsample_forward(xc, mc, N, C, H, W, S, yc);
        brotensor::convex_upsample_forward(xg, mg, N, C, H, W, S, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 1e-5f, tg("convex_upsample", dt, ""));
    }
}

// ─── gathers ───────────────────────────────────────────────────────────────

void test_gathers() {
    std::printf("pad / slice / unfold / windows / shuffles\n");
    std::uint64_t seed = 400;
    for (Dtype dt : kFloatTypes) {
        const int N = 2, C = 3, H = 7, W = 9;
        const auto xv = random_values(std::size_t(N) * C * H * W, seed++, -2, 2, dt);
        Tensor xc = cpu_tensor(xv, N, C * H * W), xg = upload(xv, N, C * H * W, dt);
        Tensor yc, yg;
        struct P { int t, b, l, r; };
        for (int mode = 0; mode < 3; ++mode) {
            for (const P& p : {P{1, 1, 1, 1}, P{3, 0, 2, 5}, P{6, 2, 8, 0}}) {
                brotensor::pad2d_forward(xc, N, C, H, W, p.t, p.b, p.l, p.r, mode, yc);
                brotensor::pad2d_forward(xg, N, C, H, W, p.t, p.b, p.l, p.r, mode, yg);
                exact(download(yg), yc.to_host_vector(), tg("pad2d", dt, "mode " + std::to_string(mode) + " " +
                                                                           dims(p.t, p.b, p.l, p.r)));
            }
            brotensor::unfold2d_forward(xc, N, C, H, W, 3, 5, 2, 1, 1, 2, 2, 0, mode, yc);
            brotensor::unfold2d_forward(xg, N, C, H, W, 3, 5, 2, 1, 1, 2, 2, 0, mode, yg);
            exact(download(yg), yc.to_host_vector(), tg("unfold2d", dt, "3x5 mode " + std::to_string(mode)));
        }
        Tensor pz;
        VKT_CHECK(throws([&] { brotensor::pad2d_forward(xg, N, C, H, W, 7, 0, 0, 0, 1, pz); }));
        brotensor::slice2d_forward(xc, N, C, H, W, 2, 3, 4, 5, yc);
        brotensor::slice2d_forward(xg, N, C, H, W, 2, 3, 4, 5, yg);
        exact(download(yg), yc.to_host_vector(), tg("slice2d", dt, ""));
        Tensor dc, dg;
        brotensor::slice2d_backward(yc, N, C, H, W, 2, 3, 4, 5, dc);
        brotensor::slice2d_backward(upload(yc.to_host_vector(), yc.rows, yc.cols, dt), N, C, H, W, 2, 3, 4, 5, dg);
        exact(download(dg), dc.to_host_vector(), tg("slice2d_backward", dt, ""));

        const int H2 = 8, W2 = 12;
        const auto zv = random_values(std::size_t(N) * C * H2 * W2, seed++, -2, 2, dt);
        Tensor zc = cpu_tensor(zv, N, C * H2 * W2), zg = upload(zv, N, C * H2 * W2, dt);
        brotensor::window_partition_forward(zc, N, C, H2, W2, 4, yc);
        brotensor::window_partition_forward(zg, N, C, H2, W2, 4, yg);
        exact(download(yg), yc.to_host_vector(), tg("window_partition", dt, ""));
        Tensor rc, rg;
        brotensor::window_reverse_forward(yc, N, C, H2, W2, 4, rc);
        brotensor::window_reverse_forward(yg, N, C, H2, W2, 4, rg);
        exact(download(rg), rc.to_host_vector(), tg("window_reverse", dt, ""));
        exact(download(rg), zv, tg("window_reverse(partition) == identity", dt, ""));
        for (bool cm : {false, true}) {
            brotensor::spatial_merge_2x2_forward(zc, N, C, H2, W2, cm, yc);
            brotensor::spatial_merge_2x2_forward(zg, N, C, H2, W2, cm, yg);
            exact(download(yg), yc.to_host_vector(), tg("spatial_merge_2x2", dt, cm ? "channel-major" : "block-major"));
        }
        for (int cout : {3, 6, 12}) {   // repeats 4, 2, 1 over C_in = 3
            brotensor::pixel_shuffle_upsample_2x_forward(zc, N, C, H2, W2, cout, yc);
            brotensor::pixel_shuffle_upsample_2x_forward(zg, N, C, H2, W2, cout, yg);
            exact(download(yg), yc.to_host_vector(), tg("pixel_shuffle_upsample_2x", dt, "C_out " + std::to_string(cout)));
        }
        const int hp = 3, wp = 4, P = 2, Ct = 5, Ck = 3;
        const auto tv = random_values(std::size_t(hp) * wp * P * P * Ct, seed++, -2, 2, dt);
        Tensor tc = cpu_tensor(tv, hp * wp, P * P * Ct), tgk = upload(tv, hp * wp, P * P * Ct, dt);
        for (bool cm : {false, true}) {
            brotensor::patch_unpack_forward(tc, hp, wp, P, Ct, Ck, cm, yc);
            brotensor::patch_unpack_forward(tgk, hp, wp, P, Ct, Ck, cm, yg);
            exact(download(yg), yc.to_host_vector(), tg("patch_unpack", dt, cm ? "channel-major" : "block-major"));
        }
    }
}

void test_rows_and_concat() {
    std::printf("gather / scatter rows, concat_nchw_channels\n");
    for (Dtype dt : kFloatTypes) {
        const int R = 11, C = 37;
        const auto xv = random_values(std::size_t(R) * C, 500, -2, 2, dt);
        const std::vector<std::int32_t> idx = {3, 0, 10, 3, 7};
        Tensor ic = Tensor::from_raw_bytes_on(Device::cpu(), idx.data(), 5, 1, Dtype::INT32, idx.size() * 4);
        Tensor ig = Tensor::from_raw_bytes_on(vk(), idx.data(), 5, 1, Dtype::INT32, idx.size() * 4);
        Tensor xc = cpu_tensor(xv, R, C), xg = upload(xv, R, C, dt), yc, yg;
        brotensor::gather_rows(xc, ic, yc);
        brotensor::gather_rows(xg, ig, yg);
        exact(download(yg), yc.to_host_vector(), tg("gather_rows", dt, ""));
        const std::vector<std::int32_t> sidx = {9, 1, 4, 6, 2};
        Tensor sc = Tensor::from_raw_bytes_on(Device::cpu(), sidx.data(), 5, 1, Dtype::INT32, sidx.size() * 4);
        Tensor sg = Tensor::from_raw_bytes_on(vk(), sidx.data(), 5, 1, Dtype::INT32, sidx.size() * 4);
        brotensor::scatter_rows(yc, sc, xc);
        brotensor::scatter_rows(yg, sg, xg);
        exact(download(xg), xc.to_host_vector(), tg("scatter_rows", dt, ""));

        const int N = 2, H = 3, W = 5;
        const std::vector<int> cs = {2, 1, 4};
        std::vector<Tensor> pc, pg;
        for (std::size_t i = 0; i < cs.size(); ++i) {
            const auto v = random_values(std::size_t(N) * cs[i] * H * W, 510 + i, -2, 2, dt);
            pc.push_back(cpu_tensor(v, N, cs[i] * H * W));
            pg.push_back(upload(v, N, cs[i] * H * W, dt));
        }
        Tensor oc, og;
        brotensor::concat_nchw_channels({&pc[0], &pc[1], &pc[2]}, N, H, W, cs, oc);
        brotensor::concat_nchw_channels({&pg[0], &pg[1], &pg[2]}, N, H, W, cs, og);
        exact(download(og), oc.to_host_vector(), tg("concat_nchw_channels", dt, ""));
        {
            const auto av = random_values(5 * 3, 520, -2, 2, dt), bv = random_values(5 * 8, 521, -2, 2, dt);
            Tensor ac = cpu_tensor(av, 5, 3), bc = cpu_tensor(bv, 5, 8), ag = upload(av, 5, 3, dt),
                   bg = upload(bv, 5, 8, dt), cc, cg;
            brotensor::concat_batched_rows({&ac, nullptr, &bc, &ac}, cc);
            brotensor::concat_batched_rows({&ag, nullptr, &bg, &ag}, cg);
            VKT_CHECK(cg.rows == 5 && cg.cols == 14 && cg.dtype == dt);
            exact(download(cg), cc.to_host_vector(), tg("concat_batched_rows", dt, ""));
        }
        Tensor b0, b1, b2;
        brotensor::concat_nchw_channels_backward(og, N, H, W, cs, {&b0, &b1, &b2});
        exact(download(b0), pc[0].to_host_vector(), tg("concat_nchw_channels_backward part 0", dt, ""));
        exact(download(b2), pc[2].to_host_vector(), tg("concat_nchw_channels_backward part 2", dt, ""));
    }
}

// ─── NCHW norms ────────────────────────────────────────────────────────────

void test_norms() {
    std::printf("group / batch / L2 norms\n");
    std::uint64_t seed = 600;
    struct G { int n, c, h, w, groups; float lo, hi; };
    const G gs[] = {{1, 32, 8, 8, 8, -2, 2}, {2, 12, 5, 7, 3, -1, 3}, {1, 64, 33, 31, 32, 380, 410},
                    {2, 16, 64, 64, 4, -1, 1}, {1, 8, 1, 1, 2, -1, 1}};
    for (Dtype dt : kFloatTypes) {
        for (const G& g : gs) {
            const int cols = g.c * g.h * g.w;
            const auto xv = random_values(std::size_t(g.n) * cols, seed++, g.lo, g.hi, dt);
            const auto gv = random_values(g.c, seed++, 0.5f, 1.5f, dt), bv = random_values(g.c, seed++, -0.5f, 0.5f, dt);
            Tensor xc = cpu_tensor(xv, g.n, cols), xg = upload(xv, g.n, cols, dt);
            Tensor gc = cpu_tensor(gv, g.c, 1), gg = upload(gv, g.c, 1, dt);
            Tensor bc = cpu_tensor(bv, g.c, 1), bg = upload(bv, g.c, 1, dt);
            Tensor yc, yg;
            brotensor::group_norm_forward(xc, gc, bc, g.n, g.c, g.h, g.w, g.groups, 1e-5f, yc);
            brotensor::group_norm_forward(xg, gg, bg, g.n, g.c, g.h, g.w, g.groups, 1e-5f, yg);
            // The CPU sums a tile sequentially in FP32: its mean carries
            // ~eps sqrt(M) |x| of rounding (the large-offset case), xhat that
            // times rstd.
            const float scale = std::max(std::fabs(g.lo), std::fabs(g.hi));
            const float m = float(g.c / g.groups) * g.h * g.w;
            const float atol = 6e-8f * std::sqrt(m) * scale / ((g.hi - g.lo) * 0.29f) + 2e-5f;
            close_for(dt, download(yg), yc.to_host_vector(), atol, 2e-5f,
                      tg("group_norm", dt, dims(g.n, g.c, g.h, g.w) + " g" + std::to_string(g.groups)));
        }
        const int N = 2, C = 5, H = 6, W = 7;
        const auto xv = random_values(std::size_t(N) * C * H * W, seed++, -2, 2, dt);
        auto pv = [&](float lo, float hi) { return random_values(C, seed++, lo, hi, dt); };
        const auto g = pv(0.5f, 1.5f), b = pv(-0.5f, 0.5f), rm = pv(-1, 1), rv = pv(0.2f, 2);
        Tensor xc = cpu_tensor(xv, N, C * H * W), xg = upload(xv, N, C * H * W, dt), yc, yg;
        brotensor::batch_norm_inference(xc, cpu_tensor(g, C, 1), cpu_tensor(b, C, 1), cpu_tensor(rm, C, 1),
                                        cpu_tensor(rv, C, 1), N, C, H, W, 1e-5f, yc);
        brotensor::batch_norm_inference(xg, upload(g, C, 1, dt), upload(b, C, 1, dt), upload(rm, C, 1, dt),
                                        upload(rv, C, 1, dt), N, C, H, W, 1e-5f, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 2e-6f, tg("batch_norm_inference", dt, ""));
        brotensor::l2_normalize_nchw_forward(xc, N, C, H, W, 1e-6f, yc);
        brotensor::l2_normalize_nchw_forward(xg, N, C, H, W, 1e-6f, yg);
        close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 4e-6f, tg("l2_normalize_nchw", dt, ""));
    }
}

// ─── diffusion ─────────────────────────────────────────────────────────────

// alphas_cumprod of SD's scaled_linear schedule (beta 0.00085 .. 0.012 over
// 1000 steps), as the schedulers compute it in float.
std::vector<float> sd_alphas_cumprod() {
    std::vector<float> a(1000);
    double cp = 1.0;
    const double b0 = std::sqrt(0.00085), b1 = std::sqrt(0.012);
    for (int i = 0; i < 1000; ++i) {
        const double s = b0 + (b1 - b0) * i / 999.0;
        cp *= 1.0 - s * s;
        a[i] = static_cast<float>(cp);
    }
    return a;
}

void test_samplers() {
    std::printf("sampler steps\n");
    const auto ac = sd_alphas_cumprod();
    for (Dtype dt : kFloatTypes) {
        const int R = 4, C = 4096 + 3;
        const auto xv = random_values(std::size_t(R) * C, 700, -3, 3, dt);
        const auto ev = random_values(std::size_t(R) * C, 701, -1.5f, 1.5f, dt);
        const auto pv = random_values(std::size_t(R) * C, 702, -1, 1, dt);
        Tensor xc = cpu_tensor(xv, R, C), ec = cpu_tensor(ev, R, C), pc = cpu_tensor(pv, R, C);
        Tensor xg = upload(xv, R, C, dt), eg = upload(ev, R, C, dt), pg = upload(pv, R, C, dt);
        // The kernels evaluate the formulas unfused; the CPU backend is built
        // with -mfma and GCC contracts them, so FP32 results differ by an ulp
        // or two of the largest term and a 16-bit store by one step.
        int checked = 0;
        for (int step = 0; step < 50; step += 7) {   // DDIM, 50 steps, eta 0 and 1
            const int t = 999 - step * 20, tp = t - 20;
            const float at = ac[t], ap = tp >= 0 ? ac[tp] : 1.0f;
            for (float eta : {0.0f, 1.0f}) {
                const float sigma = eta * std::sqrt((1 - ap) / (1 - at) * (1 - at / ap));
                Tensor yc, yg;
                brotensor::ddim_step(xc, ec, at, ap, sigma, yc);
                brotensor::ddim_step(xg, eg, at, ap, sigma, yg);
                close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 6e-7f,
                          tg("ddim_step", dt, "t " + std::to_string(t) + " eta " + std::to_string(int(eta))));
                ++checked;
            }
        }
        for (int step = 0; step < 30; step += 6) {   // Euler on Karras-free sigmas
            const int t = 999 - step * 33, tp = std::max(0, t - 33);
            const float st = std::sqrt((1 - ac[t]) / ac[t]), sp = step == 29 ? 0.0f : std::sqrt((1 - ac[tp]) / ac[tp]);
            Tensor yc, yg;
            brotensor::euler_step(xc, ec, st, sp, yc);
            brotensor::euler_step(xg, eg, st, sp, yg);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 6e-7f, tg("euler_step", dt, "t " + std::to_string(t)));
        }
        for (int step = 1; step < 20; step += 5) {   // DPM++ 2M (data prediction, 20 steps)
            const int t = 999 - step * 50, tp = std::max(0, t - 50), tq = std::min(999, t + 50);
            auto lam = [&](int i) { return 0.5 * std::log(ac[i] / (1 - ac[i])); };
            const double h = lam(tp) - lam(t), hl = lam(t) - lam(tq), r = hl / h;
            const float sig_t = std::sqrt((1 - ac[t]) / ac[t]), sig_p = std::sqrt((1 - ac[tp]) / ac[tp]);
            const float c_xt = sig_p / sig_t;
            const float c_x0t = static_cast<float>(-std::expm1(-h) * (1 + 1 / (2 * r)));
            const float c_x0p = static_cast<float>(std::expm1(-h) / (2 * r));
            Tensor yc, yg, zc, zg;
            brotensor::dpmpp_2m_step(xc, ec, pc, sig_t, c_xt, c_x0t, c_x0p, yc, zc);
            brotensor::dpmpp_2m_step(xg, eg, pg, sig_t, c_xt, c_x0t, c_x0p, yg, zg);
            close_for(dt, download(yg), yc.to_host_vector(), 2e-6f, 6e-7f, tg("dpmpp_2m_step", dt, "t " + std::to_string(t)));
            close_for(dt, download(zg), zc.to_host_vector(), 2e-6f, 6e-7f,
                      tg("dpmpp_2m_step x0", dt, "t " + std::to_string(t)));
        }
        VKT_CHECK(checked > 0);
    }
}

void test_timestep_embedding() {
    std::printf("timestep_embedding\n");
    const std::vector<float> ts = {999.0f, 500.5f, 0.0f, 1.0f, 261.0f, 37.25f};
    for (Dtype dt : kFloatTypes) {
        for (int dim : {320, 321, 2}) {
            Tensor tc = cpu_tensor(rounded(ts, dt), 6, 1), tgk = upload(ts, 6, 1, dt);
            Tensor yc, yg = Tensor::empty_on(vk(), 6, dim, dt);
            brotensor::timestep_embedding(tc, dim, 10000.0f, yc);
            brotensor::timestep_embedding(tgk, dim, 10000.0f, yg);
            VKT_CHECK(yg.dtype == dt && yg.rows == 6 && yg.cols == dim);
            // The angle t * freq carries ulp(t) of the frequency, up to
            // 1000 * 2^-23 rad, on any backend (docs/vulkan.md, RoPE).
            close_for(dt, download(yg), yc.to_host_vector(), 2e-4f, 0, tg("timestep_embedding", dt, std::to_string(dim)));
        }
    }
}

void test_noise() {
    std::printf("Philox noise\n");
    for (int n : {1, 1000, 65536 + 17}) {
        Tensor yc = Tensor::empty_on(Device::cpu(), n, 1, Dtype::FP32), yg = Tensor::empty_on(vk(), n, 1, Dtype::FP32);
        const std::uint64_t key = 0x123456789abcdefULL, ctr = 0xfffffff0ULL;
        brotensor::rand_uniform(key, ctr, yc);
        brotensor::rand_uniform(key, ctr, yg);
        exact(yg.to_host_vector(), yc.to_host_vector(), "rand_uniform n " + std::to_string(n));
        brotensor::rand_bernoulli(0.3f, key, ctr, yc);
        brotensor::rand_bernoulli(0.3f, key, ctr, yg);
        exact(yg.to_host_vector(), yc.to_host_vector(), "rand_bernoulli n " + std::to_string(n));
        brotensor::randn(key, ctr, yc);
        brotensor::randn(key, ctr, yg);
        expect_close(yg.to_host_vector(), yc.to_host_vector(), 2e-6f, 2e-6f, "randn n " + std::to_string(n));
        brotensor::randn_truncated(-0.5f, 1.0f, key, ctr, yc);
        brotensor::randn_truncated(-0.5f, 1.0f, key, ctr, yg);
        expect_close(yg.to_host_vector(), yc.to_host_vector(), 2e-6f, 2e-6f, "randn_truncated n " + std::to_string(n));
    }
    Tensor y16 = Tensor::empty_on(vk(), 4, 4, Dtype::FP16);
    VKT_CHECK(throws([&] { brotensor::randn(1, 2, y16); }));
}

void test_image() {
    std::printf("image helpers\n");
    const int N = 2, C = 3, H = 5, W = 7;
    const auto xv = random_values(std::size_t(N) * C * H * W, 800, 0, 1, Dtype::FP32);
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f}, sd = {0.229f, 0.224f, 0.225f};
    Tensor yc, yg;
    brotensor::image_normalize(cpu_tensor(xv, N, C * H * W), cpu_tensor(mean, C, 1), cpu_tensor(sd, C, 1), N, C, H,
                               W, yc);
    brotensor::image_normalize(upload(xv, N, C * H * W, Dtype::FP32), upload(mean, C, 1, Dtype::FP32),
                               upload(sd, C, 1, Dtype::FP32), N, C, H, W, yg);
    exact(yg.to_host_vector(), yc.to_host_vector(), "image_normalize");
    std::vector<std::uint8_t> px(std::size_t(N) * H * W * C);
    Rng r(801);
    for (auto& b : px) b = static_cast<std::uint8_t>(r.next());
    Tensor src = Tensor::from_raw_bytes_on(vk(), px.data(), static_cast<int>(px.size()), 1, Dtype::INT8, px.size());
    brotensor::image_u8_to_f32_nhwc_to_nchw(px.data(), N, H, W, C, 2.0f / 255.0f, -1.0f, yc);
    yg = Tensor::empty_on(vk(), 1, 1, Dtype::FP32);
    brotensor::image_u8_to_f32_nhwc_to_nchw(static_cast<const std::uint8_t*>(src.data), N, H, W, C, 2.0f / 255.0f,
                                            -1.0f, yg);
    exact(yg.to_host_vector(), yc.to_host_vector(), "image_u8_to_f32_nhwc_to_nchw");
}

}  // namespace

void run_spatial_tests() {
    test_interp();
    test_resample_2x();
    test_pool();
    test_convex();
    test_gathers();
    test_rows_and_concat();
    test_norms();
    test_samplers();
    test_timestep_embedding();
    test_noise();
    test_image();
}

}  // namespace vkt
