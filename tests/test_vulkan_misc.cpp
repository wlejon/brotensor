// Vulkan parity for the chunk-6 small ops against the CPU backend: the
// embedding lookup, masked mean pooling (+ backward), the slot / causal masks,
// threshold_u8, rows_count_above, xavier_init (bit-exact, state included), the
// SGD / Adam steps, the MSE losses, attention_token_moments, StyleGAN3's
// bias_act / upfirdn2d (+ backwards) and the filtered_lrelu composite, and the
// gated delta rule (step and chunked, state carried across calls). FP32 /
// FP16 / BF16 where the op takes them; data movement and masks are exact.

#include "test_vulkan_common.h"

#include <brotensor/safetensors.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

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

std::string tg(const char* op, Dtype dt, const std::string& extra = "") {
    return std::string(op) + " " + dt_name(dt) + (extra.empty() ? "" : " " + extra);
}

Tensor upload_i32(const std::vector<std::int32_t>& v, int rows, int cols) {
    return Tensor::from_raw_bytes_on(vk(), v.data(), rows, cols, Dtype::INT32, v.size() * 4);
}

template <class T>
std::vector<T> raw(const Tensor& t) {
    std::vector<T> out(static_cast<std::size_t>(t.size()));
    t.copy_to_host_raw(out.data(), out.size() * sizeof(T));
    return out;
}

void test_embedding() {
    std::printf("embedding_lookup_forward\n");
    const int V = 37, D = 70, B = 23;
    std::vector<std::int32_t> idx(B);
    for (int b = 0; b < B; ++b) idx[b] = (b * 13 + 5) % V;
    idx[3] = idx[7];   // repeated index
    for (Dtype dt : kFloatTypes) {
        const auto tv = random_values(std::size_t(V) * D, 11, -3, 3, dt);
        Tensor tc = cpu_tensor(tv, V, D), tg_ = upload(tv, V, D, dt);
        Tensor oc, og;
        brotensor::embedding_lookup_forward(tc, idx.data(), B, oc);
        Tensor ig = upload_i32(idx, B, 1);
        brotensor::embedding_lookup_forward(tg_, static_cast<const int32_t*>(ig.data), B, og);
        VKT_CHECK(og.dtype == dt && og.rows == B && og.cols == D);
        expect_close(download(og), oc.to_host_vector(), 0, 0, tg("embedding_lookup_forward", dt));
    }
}

void test_mean_pool() {
    std::printf("masked_mean_pool\n");
    const int K = 9, D = 300;
    const std::vector<float> mask = {1, 0, 1, 1, 0, 1, 1, 1, 0};
    const std::vector<float> none(K, 0.0f);
    for (Dtype dt : kFloatTypes) {
        const auto xv = random_values(std::size_t(K) * D, 21, -2, 2, dt);
        Tensor xc = cpu_tensor(xv, K, D), xg = upload(xv, K, D, dt);
        for (int m = 0; m < 3; ++m) {
            const std::vector<float>* mv = m == 0 ? nullptr : m == 1 ? &mask : &none;
            Tensor mg = mv ? upload(*mv, K, 1, Dtype::FP32) : Tensor();
            const float* mc = mv ? mv->data() : nullptr;
            const float* mgp = mv ? static_cast<const float*>(mg.data) : nullptr;
            const std::string tag = m == 0 ? "no mask" : m == 1 ? "mask" : "all masked";
            Tensor yc, yg;
            brotensor::masked_mean_pool_forward(xc, mc, yc);
            brotensor::masked_mean_pool_forward(xg, mgp, yg);
            close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 1e-5f, tg("masked_mean_pool_forward", dt, tag));
            const auto dv = random_values(D, 22, -1, 1, dt);
            Tensor dc = cpu_tensor(dv, D, 1), dgv = upload(dv, D, 1, dt), dxc, dxg;
            brotensor::masked_mean_pool_backward(dc, mc, K, dxc);
            brotensor::masked_mean_pool_backward(dgv, mgp, K, dxg);
            close_for(dt, download(dxg), dxc.to_host_vector(), 1e-7f, 1e-6f, tg("masked_mean_pool_backward", dt, tag));
        }
    }
}

void test_masks_thresholds() {
    std::printf("masks and thresholds\n");
    {
        const int K = 6, stride = 5, off = 2;
        std::vector<float> x(40);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i * 7 % 3) == 0 ? 0.9f : 0.1f;
        Tensor mc, mg;
        brotensor::build_slot_mask(cpu_tensor(x, 40, 1), off, K, stride, mc);
        brotensor::build_slot_mask(upload(x, 40, 1, Dtype::FP32), off, K, stride, mg);
        expect_close(mg.to_host_vector(), mc.to_host_vector(), 0, 0, "build_slot_mask");
    }
    for (int q : {-1, 0, 7, 40}) {
        Tensor mc, mg = Tensor::empty_on(vk(), 1, 1, Dtype::FP32);
        brotensor::build_causal_mask_row(33, q, mc);
        brotensor::build_causal_mask_row(33, q, mg);
        expect_close(mg.to_host_vector(), mc.to_host_vector(), 0, 0, "build_causal_mask_row q=" + std::to_string(q));
    }
    for (Dtype dt : kFloatTypes) {
        const int R = 7, C = 1000;
        auto xv = random_values(std::size_t(R) * C, 31, -1, 1, dt);
        xv[5] = 0.25f;   // exactly at the threshold: not above
        Tensor xc = cpu_tensor(xv, R, C), xg = upload(xv, R, C, dt);
        Tensor yc, yg;
        brotensor::threshold_u8(xc, 0.25f, yc);
        brotensor::threshold_u8(xg, 0.25f, yg);
        VKT_CHECK(yg.dtype == Dtype::INT8);
        const auto a = raw<std::int8_t>(yg), b = raw<std::int8_t>(yc);
        expect_equal_bits(a.data(), b.data(), a.size(), tg("threshold_u8", dt));
        Tensor cc, cg;
        brotensor::rows_count_above(xc, -0.5f, 0.25f, cc);
        brotensor::rows_count_above(xg, -0.5f, 0.25f, cg);
        const auto ca = raw<std::int32_t>(cg), cb = raw<std::int32_t>(cc);
        expect_equal_bits(ca.data(), cb.data(), ca.size() * 4, tg("rows_count_above", dt));
    }
}

void test_init_optim_loss() {
    std::printf("xavier_init, optimisers, losses\n");
    {
        std::uint64_t sc = 0x1234567890abcdefULL, sg = sc;
        Tensor wc = Tensor::zeros_on(Device::cpu(), 67, 45), wg = Tensor::zeros_on(vk(), 67, 45);
        brotensor::xavier_init(wc, sc);
        brotensor::xavier_init(wg, sg);
        const auto a = wg.to_host_vector(), b = wc.to_host_vector();
        expect_equal_bits(a.data(), b.data(), a.size() * 4, "xavier_init values");
        VKT_CHECK(sc == sg);
    }
    const int n = 5000;
    const auto p = random_values(n, 41, -1, 1, Dtype::FP32), g = random_values(n, 42, -1, 1, Dtype::FP32);
    const auto m0 = random_values(n, 43, -0.1f, 0.1f, Dtype::FP32), v0 = random_values(n, 44, 0, 0.1f, Dtype::FP32);
    {
        Tensor pc = cpu_tensor(p, n, 1), gc = cpu_tensor(g, n, 1), vc = cpu_tensor(m0, n, 1);
        Tensor pg = upload(p, n, 1, Dtype::FP32), gg = upload(g, n, 1, Dtype::FP32), vg = upload(m0, n, 1, Dtype::FP32);
        for (int s = 0; s < 3; ++s) {
            brotensor::sgd_step(pc, gc, vc, 0.01f, 0.9f);
            brotensor::sgd_step(pg, gg, vg, 0.01f, 0.9f);
        }
        expect_close(pg.to_host_vector(), pc.to_host_vector(), 1e-7f, 1e-6f, "sgd_step param");
        expect_close(vg.to_host_vector(), vc.to_host_vector(), 1e-7f, 1e-6f, "sgd_step velocity");
    }
    {
        Tensor pc = cpu_tensor(p, n, 1), gc = cpu_tensor(g, n, 1), mc = cpu_tensor(m0, n, 1), vc = cpu_tensor(v0, n, 1);
        Tensor pg = upload(p, n, 1, Dtype::FP32), gg = upload(g, n, 1, Dtype::FP32);
        Tensor mg = upload(m0, n, 1, Dtype::FP32), vg = upload(v0, n, 1, Dtype::FP32);
        for (int s = 1; s <= 3; ++s) {
            brotensor::adam_step(pc, gc, mc, vc, 1e-3f, 0.9f, 0.999f, 1e-8f, s);
            brotensor::adam_step(pg, gg, mg, vg, 1e-3f, 0.9f, 0.999f, 1e-8f, s);
        }
        expect_close(pg.to_host_vector(), pc.to_host_vector(), 1e-6f, 1e-5f, "adam_step param");
        expect_close(mg.to_host_vector(), mc.to_host_vector(), 1e-7f, 1e-6f, "adam_step m");
        expect_close(vg.to_host_vector(), vc.to_host_vector(), 1e-7f, 1e-6f, "adam_step v");
    }
    {
        Tensor pc = cpu_tensor(p, 50, 100), tc = cpu_tensor(g, 50, 100);
        Tensor pg = upload(p, 50, 100, Dtype::FP32), tgg = upload(g, 50, 100, Dtype::FP32);
        const float lc = brotensor::mse_vec_forward(pc, tc), lg = brotensor::mse_vec_forward(pg, tgg);
        expect_close({lg}, {lc}, 0, 1e-5f, "mse_vec_forward");
        Tensor dc, dg;
        brotensor::mse_vec_backward(pc, tc, dc);
        brotensor::mse_vec_backward(pg, tgg, dg);
        expect_close(dg.to_host_vector(), dc.to_host_vector(), 0, 1e-6f, "mse_vec_backward");
        Tensor dc2, lc2, dg2, lg2;
        brotensor::mse_vec_per_sample(pc, tc, dc2, lc2);
        brotensor::mse_vec_per_sample(pg, tgg, dg2, lg2);
        expect_close(dg2.to_host_vector(), dc2.to_host_vector(), 0, 0, "mse_vec_per_sample dPred");
        expect_close(lg2.to_host_vector(), lc2.to_host_vector(), 0, 1e-6f, "mse_vec_per_sample loss");
    }
}

void test_moments() {
    std::printf("attention_token_moments\n");
    const int h = 24, w = 17, Lk = 77;
    for (Dtype dt : kFloatTypes) {
        auto av = random_values(std::size_t(h) * w * Lk, 51, 0, 0.01f, dt);
        for (int q = 0; q < h * w; ++q) av[std::size_t(q) * Lk + 3] = 0.0f;   // a token with no mass
        Tensor ac = cpu_tensor(av, h * w, Lk), ag = upload(av, h * w, Lk, dt);
        Tensor mc, cc, mg, cg;
        brotensor::attention_token_moments(ac, h, w, mc, cc);
        brotensor::attention_token_moments(ag, h, w, mg, cg);
        expect_close(mg.to_host_vector(), mc.to_host_vector(), 1e-6f, 1e-5f, tg("attention_token_moments mass", dt));
        expect_close(cg.to_host_vector(), cc.to_host_vector(), 1e-4f, 1e-5f, tg("attention_token_moments centroid", dt));
    }
}

void test_stylegan_ops() {
    std::printf("bias_act, upfirdn2d, filtered_lrelu\n");
    const int N = 2, C = 5, H = 9, W = 11;
    for (Dtype dt : kFloatTypes) {
        const auto xv = random_values(std::size_t(N) * C * H * W, 61, -2, 2, dt);
        const auto bv = random_values(C, 62, -0.5f, 0.5f, dt);
        Tensor xc = cpu_tensor(xv, N, C * H * W), xg = upload(xv, N, C * H * W, dt);
        Tensor bc = cpu_tensor(bv, C, 1), bg = upload(bv, C, 1, dt);
        for (int act = 0; act < 2; ++act) {
            for (float clampv : {-1.0f, 1.5f}) {
                const std::string tag = "act " + std::to_string(act) + " clamp " + std::to_string(clampv);
                Tensor yc, yg;
                brotensor::bias_act_forward(xc, &bc, N, C, H * W, act, 0.2f, 1.41421356f, clampv, yc);
                brotensor::bias_act_forward(xg, &bg, N, C, H * W, act, 0.2f, 1.41421356f, clampv, yg);
                close_for(dt, download(yg), yc.to_host_vector(), 1e-6f, 1e-6f, tg("bias_act_forward", dt, tag));
                const auto dyv = random_values(xv.size(), 63, -1, 1, dt);
                Tensor dyc = cpu_tensor(dyv, N, C * H * W), dyg = upload(dyv, N, C * H * W, dt);
                Tensor dxc, dxg;
                Tensor dbc = Tensor::zeros_on(Device::cpu(), C, 1), dbg = Tensor::zeros_on(vk(), C, 1, dt);
                brotensor::bias_act_backward(dyc, xc, &bc, N, C, H * W, act, 0.2f, 1.41421356f, clampv, dxc, &dbc);
                brotensor::bias_act_backward(dyg, xg, &bg, N, C, H * W, act, 0.2f, 1.41421356f, clampv, dxg, &dbg);
                close_for(dt, download(dxg), dxc.to_host_vector(), 1e-6f, 1e-6f, tg("bias_act_backward dX", dt, tag));
                close_for(dt, download(dbg), dbc.to_host_vector(), 1e-4f, 1e-5f, tg("bias_act_backward dB", dt, tag));
            }
        }
        // upfirdn2d: 2x up with a 6-tap filter, 2x down, odd padding, both flips.
        const int fH = 6, fW = 4;
        const auto fv = random_values(fH * fW, 64, -0.3f, 0.6f, dt);
        Tensor fc = cpu_tensor(fv, fH, fW), fg = upload(fv, fH, fW, dt);
        struct P { int ux, uy, dx, dy, px0, px1, py0, py1; };
        for (const P& p : {P{2, 2, 1, 1, 2, 1, 3, 2}, P{1, 1, 2, 2, 1, 2, 2, 3}, P{2, 1, 1, 2, 0, 3, 1, 1}}) {
            for (bool flip : {false, true}) {
                const std::string tag = "up " + std::to_string(p.ux) + "," + std::to_string(p.uy) + " down " +
                                        std::to_string(p.dx) + "," + std::to_string(p.dy) + (flip ? " flip" : "");
                Tensor yc, yg;
                brotensor::upfirdn2d_forward(xc, fc, N, C, H, W, fH, fW, p.ux, p.uy, p.dx, p.dy, p.px0, p.px1, p.py0,
                                             p.py1, flip, 4.0f, yc);
                brotensor::upfirdn2d_forward(xg, fg, N, C, H, W, fH, fW, p.ux, p.uy, p.dx, p.dy, p.px0, p.px1, p.py0,
                                             p.py1, flip, 4.0f, yg);
                close_for(dt, download(yg), yc.to_host_vector(), 1e-5f, 1e-5f, tg("upfirdn2d_forward", dt, tag));
                Tensor dxc, dxg;
                Tensor ygr = upload(rounded(yc.to_host_vector(), dt), yc.rows, yc.cols, dt);
                Tensor ycr = cpu_tensor(rounded(yc.to_host_vector(), dt), yc.rows, yc.cols);
                brotensor::upfirdn2d_backward(ycr, fc, N, C, H, W, fH, fW, p.ux, p.uy, p.dx, p.dy, p.px0, p.px1, p.py0,
                                              p.py1, flip, 4.0f, dxc);
                brotensor::upfirdn2d_backward(ygr, fg, N, C, H, W, fH, fW, p.ux, p.uy, p.dx, p.dy, p.px0, p.px1, p.py0,
                                              p.py1, flip, 4.0f, dxg);
                close_for(dt, download(dxg), dxc.to_host_vector(), 1e-4f, 1e-5f, tg("upfirdn2d_backward", dt, tag));
            }
        }
        if (dt == Dtype::FP32) {   // the public composite over bias_act + upfirdn2d
            const auto fu = random_values(4 * 4, 65, -0.2f, 0.5f, dt), fd = random_values(4 * 4, 66, -0.2f, 0.5f, dt);
            Tensor fuc = cpu_tensor(fu, 4, 4), fdc = cpu_tensor(fd, 4, 4), fug = upload(fu, 4, 4, dt),
                   fdg = upload(fd, 4, 4, dt);
            Tensor uc, ac, yc, ug, ag, yg;
            brotensor::filtered_lrelu_forward(xc, fuc, fdc, &bc, N, C, H, W, 2, 2, 2, 1, 2, 1, 1.41421356f, 0.2f, 256.0f,
                                              uc, ac, yc);
            brotensor::filtered_lrelu_forward(xg, fug, fdg, &bg, N, C, H, W, 2, 2, 2, 1, 2, 1, 1.41421356f, 0.2f,
                                              256.0f, ug, ag, yg);
            expect_close(download(yg), yc.to_host_vector(), 1e-5f, 1e-5f, "filtered_lrelu_forward (composite) f32");
            const auto dyv = random_values(yc.size(), 67, -1, 1, dt);
            Tensor dyc = cpu_tensor(dyv, yc.rows, yc.cols), dyg = upload(dyv, yc.rows, yc.cols, dt);
            Tensor dxc, dxg, dbc = Tensor::zeros_on(Device::cpu(), C, 1), dbg = Tensor::zeros_on(vk(), C, 1);
            brotensor::filtered_lrelu_backward(dyc, xc, fuc, fdc, &bc, N, C, H, W, 2, 2, 2, 1, 2, 1, 1.41421356f, 0.2f,
                                               256.0f, uc, dxc, &dbc);
            brotensor::filtered_lrelu_backward(dyg, xg, fug, fdg, &bg, N, C, H, W, 2, 2, 2, 1, 2, 1, 1.41421356f, 0.2f,
                                               256.0f, ug, dxg, &dbg);
            expect_close(download(dxg), dxc.to_host_vector(), 1e-5f, 1e-5f, "filtered_lrelu_backward (composite) dX f32");
            expect_close(download(dbg), dbc.to_host_vector(), 1e-4f, 1e-5f, "filtered_lrelu_backward (composite) dB f32");
        }
    }
}

// CPU reference state carried across several calls, both forms.
void test_delta_rule() {
    std::printf("gated_delta_rule\n");
    struct S { int H, dk, dv, L; };
    for (const S& s : {S{2, 128, 128, 7}, S{3, 64, 96, 33}, S{1, 160, 40, 5}, S{4, 16, 8, 1}}) {
        const int Dq = s.H * s.dk, Dv = s.H * s.dv;
        const auto st0 = random_values(std::size_t(s.H) * s.dv * s.dk, 71, -0.1f, 0.1f, Dtype::FP32);
        const auto la = random_values(s.H, 72, -1, 0.5f, Dtype::FP32);
        Tensor sc = cpu_tensor(st0, s.H, s.dv * s.dk), sg = upload(st0, s.H, s.dv * s.dk, Dtype::FP32);
        Tensor lac = cpu_tensor(la, s.H, 1), lag = upload(la, s.H, 1, Dtype::FP32);
        const std::string tag = "H" + std::to_string(s.H) + " dk" + std::to_string(s.dk) + " dv" +
                                std::to_string(s.dv) + " L" + std::to_string(s.L);
        for (int call = 0; call < 3; ++call) {
            const int L = call == 2 ? 1 : s.L;
            const std::uint64_t sd = 80 + 10 * call;
            auto q = random_values(std::size_t(L) * Dq, sd, -0.3f, 0.3f, Dtype::FP32);
            auto k = random_values(std::size_t(L) * Dq, sd + 1, -0.3f, 0.3f, Dtype::FP32);
            const auto v = random_values(std::size_t(L) * Dv, sd + 2, -1, 1, Dtype::FP32);
            const auto a = random_values(std::size_t(L) * s.H, sd + 3, -3, 3, Dtype::FP32);
            const auto b = random_values(std::size_t(L) * s.H, sd + 4, -3, 3, Dtype::FP32);
            Tensor oc, og;
            auto run = [&](Device dev, Tensor& st, Tensor& la_t, Tensor& o) {
                auto mk = [&](const std::vector<float>& x, int r, int c) {
                    return dev.is_cpu() ? cpu_tensor(x, r, c) : upload(x, r, c, Dtype::FP32);
                };
                Tensor Q = mk(q, L, Dq), K = mk(k, L, Dq), V = mk(v, L, Dv), A = mk(a, L, s.H), B = mk(b, L, s.H);
                if (L == 1) brotensor::gated_delta_rule_step(Q, K, V, A, B, la_t, s.H, s.dk, s.dv, st, o);
                else brotensor::gated_delta_rule_chunked(Q, K, V, A, B, la_t, s.H, s.dk, s.dv, st, o);
            };
            run(Device::cpu(), sc, lac, oc);
            run(vk(), sg, lag, og);
            expect_close(og.to_host_vector(), oc.to_host_vector(), 2e-5f, 1e-4f,
                         "gated_delta_rule O call " + std::to_string(call) + " " + tag);
        }
        expect_close(sg.to_host_vector(), sc.to_host_vector(), 2e-5f, 1e-4f, "gated_delta_rule state " + tag);
    }
}

// A BF16 checkpoint loaded with Vulkan as the default device: the loader's
// compute path uploads the raw BF16 bits and casts on the device to the
// compute dtype (FP16 on any GPU, the dtype policy's "cast once at upload"),
// upload() keeps the file dtype, Tensor::to keeps the dtype bit for bit, and
// the device cast agrees with the host conversion.
void test_bf16_load() {
    std::printf("BF16 checkpoint load on Vulkan\n");
    namespace st = brotensor::safetensors;
    const int R = 64, C = 96;
    auto vals = random_values(std::size_t(R) * C, 91, -70000.0f, 70000.0f, Dtype::FP32);
    vals[0] = 1e-6f; vals[1] = -3.0e-8f; vals[2] = 65504.0f; vals[3] = 1.0f / 3.0f;
    std::vector<std::uint16_t> bf(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) bf[i] = brotensor::fp32_to_bf16_bits(vals[i]);
    const auto path = std::filesystem::temp_directory_path() / "brotensor_vk_bf16_load.safetensors";
    st::write_file(path.string(), {st::WriteEntry{"w", st::Dtype::BF16, {R, C}, bf.data(), bf.size() * 2}});
    const st::File f = st::File::open(path.string());
    const st::TensorView& view = f.get("w");
    std::vector<float> as32(bf.size());
    std::vector<std::uint16_t> as16(bf.size());
    for (std::size_t i = 0; i < bf.size(); ++i) {
        as32[i] = brotensor::bf16_bits_to_fp32(bf[i]);
        as16[i] = brotensor::fp32_to_fp16_bits(as32[i]);
    }
    {
        brotensor::DeviceScope scope(vk());
        VKT_CHECK(brotensor::compute_dtype() == Dtype::FP16);
        Tensor w16, w32, wraw;
        st::upload_compute(view, R, C, w16);
        VKT_CHECK(w16.device == vk() && w16.dtype == Dtype::FP16);
        const auto got16 = [&] {
            std::vector<std::uint16_t> b(bf.size());
            w16.copy_to_host_raw(b.data(), b.size() * 2);
            return b;
        }();
        expect_equal_bits(got16.data(), as16.data(), as16.size() * 2, "upload_compute BF16 -> FP16 on Vulkan");
        st::upload_as(view, R, C, Dtype::FP32, w32);
        VKT_CHECK(w32.device == vk() && w32.dtype == Dtype::FP32);
        expect_close(w32.to_host_vector(), as32, 0, 0, "upload_as BF16 -> FP32 on Vulkan");
        st::upload(view, R, C, wraw);
        VKT_CHECK(wraw.device == vk() && wraw.dtype == Dtype::BF16);
        const auto rawb = [&] {
            std::vector<std::uint16_t> b(bf.size());
            wraw.copy_to_host_raw(b.data(), b.size() * 2);
            return b;
        }();
        expect_equal_bits(rawb.data(), bf.data(), bf.size() * 2, "upload BF16 keeps the file dtype on Vulkan");
    }
    Tensor host_bf = Tensor::from_host_bf16_on(Device::cpu(), bf.data(), R, C);
    Tensor moved = host_bf.to(vk());
    VKT_CHECK(moved.device == vk() && moved.dtype == Dtype::BF16);
    Tensor cast16;
    brotensor::cast(moved, cast16, Dtype::FP16);
    std::vector<std::uint16_t> c16(bf.size());
    cast16.copy_to_host_raw(c16.data(), c16.size() * 2);
    expect_equal_bits(c16.data(), as16.data(), as16.size() * 2, "Tensor::to(vulkan) + cast BF16 -> FP16");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace

void run_misc_tests() {
    test_bf16_load();
    test_embedding();
    test_mean_pool();
    test_masks_thresholds();
    test_init_optim_loss();
    test_moments();
    test_stylegan_ops();
    test_delta_rule();
}

}  // namespace vkt
