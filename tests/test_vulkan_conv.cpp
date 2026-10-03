// Vulkan parity for the convolutions against the CPU backend: conv2d_forward
// on every kernel path (cooperative-matrix implicit GEMM, SIMT implicit GEMM,
// direct) over 3x3 / 1x1 / 7x7 / 5x5 kernels, stride, dilation, padding
// beyond the kernel, odd image sizes, widths that are not a multiple of 8,
// grouped and depthwise convolutions, channel counts off the tile, batch > 1;
// conv3d (patch-embedding GEMM and direct), conv_transpose2d, the bias
// gradients and the ResBlock. FP32 / FP16 / BF16 storage; the reference is
// the CPU op in FP32 on the rounded inputs.

#include "test_vulkan_common.h"

#include "detail/spatial.h"

#include <brotensor/vulkan.h>

#include <cmath>
#include <string>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

const Dtype kFloatTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

Tensor cpu_tensor(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

std::vector<float> rounded(std::vector<float> v, Dtype dt) {
    for (float& x : v) x = round_to(dt, x);
    return v;
}

// Sum of |terms| bound: outputs are O(1) with weights ~ 1/sqrt(K); FP32 sums
// in another order differ by ~K ulp of the magnitude; 16-bit outputs round
// once more.
void close_conv(Dtype dt, const std::vector<float>& got, const std::vector<float>& want, int K,
                const std::string& t) {
    const float atol = 2e-6f * std::sqrt(float(K)) + 1e-5f;
    if (dt == Dtype::FP32) {
        expect_close(got, want, atol, 1e-5f, t);
    } else {
        expect_close(got, rounded(want, dt), atol + 1e-3f, 2.0f * dtype_eps(dt), t);
    }
}

struct Conv {
    int n, cin, h, w, cout, kh, kw, sh, sw, ph, pw, dh, dw, groups;
    bool bias;
    const char* name;
};

int out_dim(int in, int pad, int dil, int k, int stride) { return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1; }

const char* run_conv(const Conv& c, Dtype dt, std::uint64_t seed, int override_mode) {
    const int cgi = c.cin / c.groups;
    const int K = cgi * c.kh * c.kw;
    const float ws = 1.0f / std::sqrt(float(K));
    const auto xv = random_values(std::size_t(c.n) * c.cin * c.h * c.w, seed, -1.0f, 1.0f, dt);
    const auto wv = random_values(std::size_t(c.cout) * K, seed + 1, -ws, ws, dt);
    const auto bv = random_values(std::size_t(c.cout), seed + 2, -0.5f, 0.5f, dt);
    Tensor xc = cpu_tensor(xv, c.n, c.cin * c.h * c.w), wc = cpu_tensor(wv, c.cout, K), bc = cpu_tensor(bv, c.cout, 1);
    Tensor xg = upload(xv, c.n, c.cin * c.h * c.w, dt), wg = upload(wv, c.cout, K, dt), bg = upload(bv, c.cout, 1, dt);
    Tensor yc, yg;
    brotensor::conv2d_forward(xc, wc, c.bias ? &bc : nullptr, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw,
                              c.ph, c.pw, c.dh, c.dw, c.groups, yc);
    dv::set_conv_override(override_mode);
    brotensor::conv2d_forward(xg, wg, c.bias ? &bg : nullptr, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw,
                              c.ph, c.pw, c.dh, c.dw, c.groups, yg);
    dv::Conv2dArgs a;
    a.dt = dt; a.n = c.n; a.cin = c.cin; a.h = c.h; a.wd = c.w; a.cout = c.cout; a.kh = c.kh; a.kw = c.kw;
    a.sh = c.sh; a.sw = c.sw; a.ph = c.ph; a.pw = c.pw; a.dh = c.dh; a.dw = c.dw; a.groups = c.groups;
    a.w = reinterpret_cast<std::uintptr_t>(wg.data);
    a.x = reinterpret_cast<std::uintptr_t>(xg.data);
    const char* path = dv::conv2d_path(dv::device(0), a);
    dv::set_conv_override(0);
    const int ho = out_dim(c.h, c.ph, c.dh, c.kh, c.sh), wo = out_dim(c.w, c.pw, c.dw, c.kw, c.sw);
    VKT_CHECK(yg.rows == c.n && yg.cols == c.cout * ho * wo && yg.dtype == dt);
    close_conv(dt, download(yg), yc.to_host_vector(), K,
               std::string("conv2d ") + c.name + " " + dt_name(dt) + " [" + path + "]");
    return path;
}

const Conv kConvs[] = {
    // n, cin, h, w, cout, kh, kw, sh, sw, ph, pw, dh, dw, groups, bias
    {1, 64, 16, 16, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 64->64 16x16"},
    {2, 32, 13, 11, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 32->48 13x11 batch 2"},
    {1, 40, 24, 24, 72, 3, 3, 1, 1, 1, 1, 1, 1, 1, false, "3x3 40->72 no bias"},
    {1, 32, 17, 23, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, true, "3x3 stride 2 17x23"},
    {1, 24, 20, 20, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1, true, "3x3 dilation 2"},
    {1, 16, 9, 9, 24, 3, 3, 1, 1, 4, 3, 1, 1, 1, true, "3x3 padding beyond the kernel"},
    {2, 64, 16, 16, 128, 1, 1, 1, 1, 0, 0, 1, 1, 1, true, "1x1 64->128 batch 2"},
    {1, 48, 15, 15, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, true, "1x1 odd plane"},
    {1, 3, 32, 32, 64, 7, 7, 2, 2, 3, 3, 1, 1, 1, true, "7x7 stride 2 3->64 (stem)"},
    {1, 4, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 4->32 (conv_in)"},
    {1, 32, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 32->3 (conv_out)"},
    {1, 64, 12, 12, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2, true, "3x3 groups 2"},
    {2, 64, 10, 10, 128, 3, 3, 1, 1, 1, 1, 1, 1, 4, true, "3x3 groups 4 batch 2"},
    {1, 32, 14, 14, 32, 3, 3, 1, 1, 1, 1, 1, 1, 32, true, "3x3 depthwise"},
    {2, 16, 11, 9, 32, 5, 5, 2, 1, 2, 2, 1, 1, 16, true, "5x5 depthwise x2 stride (2,1)"},
    {1, 20, 9, 30, 36, 3, 1, 1, 1, 1, 0, 1, 1, 1, true, "3x1 kernel"},
    {1, 130, 10, 18, 140, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 130->140 (off the tile)"},
    {3, 8, 6, 5, 16, 2, 2, 2, 2, 0, 0, 1, 1, 1, true, "2x2 stride 2 batch 3"},
    // Stride 1 over rows of whole 16-byte chunks (the shifted vector loads):
    // large padding / dilation shifts, a 5x5 and a stride-(2, 1) kernel.
    {1, 16, 24, 24, 32, 3, 3, 1, 1, 3, 3, 3, 3, 1, true, "3x3 dilation 3 pad 3 24x24"},
    {2, 24, 16, 32, 40, 5, 5, 1, 1, 2, 2, 1, 1, 1, true, "5x5 16x32 batch 2"},
    {1, 32, 15, 16, 48, 3, 3, 2, 1, 1, 1, 1, 1, 1, true, "3x3 stride (2,1) 15x16"},
    {1, 32, 8, 8, 32, 3, 3, 1, 1, 9, 9, 1, 1, 1, true, "3x3 pad 9 on 8x8"},
};

void test_conv2d() {
    std::printf("conv2d\n");
    std::uint64_t seed = 100;
    for (Dtype dt : kFloatTypes) {
        for (const Conv& c : kConvs) run_conv(c, dt, seed += 3, 0);
    }
    // Every path on the same shapes: SIMT for the 16-bit types, direct for all.
    for (Dtype dt : kFloatTypes) {
        for (int i : {0, 1, 3, 6, 11, 16}) {
            if (dt != Dtype::FP32) run_conv(kConvs[i], dt, seed += 3, 1);
            run_conv(kConvs[i], dt, seed += 3, 2);
        }
    }
    // Large enough for full interior tiles (the fragment-form epilogue) and
    // edge strips in both directions.
    run_conv({1, 128, 40, 40, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 128->192 40x40"}, Dtype::FP16, 7, 0);
    run_conv({1, 128, 40, 40, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 128->192 40x40"}, Dtype::BF16, 8, 0);
    run_conv({2, 256, 32, 32, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 256->256 32x32 batch 2"}, Dtype::FP16, 9, 0);
    run_conv({1, 256, 64, 64, 128, 1, 1, 1, 1, 0, 0, 1, 1, 1, true, "1x1 256->128 64x64"}, Dtype::FP16, 10, 0);
    run_conv({1, 64, 33, 47, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 33x47 (rows wrap)"}, Dtype::FP16, 11, 0);
    run_conv({1, 64, 30, 30, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, true, "3x3 FP32 SIMT big tile"}, Dtype::FP32, 12, 0);

    // Errors: mismatched dtypes, groups not dividing, wrong weight size.
    Tensor x = upload(random_values(2 * 8 * 8, 1, -1, 1, Dtype::FP16), 1, 128, Dtype::FP16);
    Tensor w32 = upload(random_values(4 * 2 * 9, 2, -1, 1, Dtype::FP32), 4, 18, Dtype::FP32);
    Tensor w16 = upload(random_values(4 * 2 * 9, 2, -1, 1, Dtype::FP16), 4, 18, Dtype::FP16);
    Tensor y;
    VKT_CHECK(throws([&] { brotensor::conv2d_forward(x, w32, nullptr, 1, 2, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, y); }));
    VKT_CHECK(throws([&] { brotensor::conv2d_forward(x, w16, nullptr, 1, 2, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 3, y); }));
    VKT_CHECK(throws([&] { brotensor::conv2d_forward(x, w16, nullptr, 1, 2, 8, 8, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1, y); }));
}

void test_conv3d() {
    std::printf("conv3d\n");
    struct C3 { int n, cin, t, h, w, cout, kt, kh, kw, st, sh, sw, pt, ph, pw, groups; const char* name; };
    const C3 cases[] = {
        {6, 3, 2, 14, 14, 48, 2, 14, 14, 2, 14, 14, 0, 0, 0, 1, "patch embed (GEMM)"},
        {1, 4, 5, 9, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, "3x3x3 pad 1"},
        {2, 6, 4, 7, 7, 6, 2, 3, 3, 2, 2, 2, 0, 1, 1, 2, "2x3x3 stride 2 groups 2"},
    };
    std::uint64_t seed = 500;
    for (Dtype dt : kFloatTypes) {
        for (const C3& c : cases) {
            const int K = (c.cin / c.groups) * c.kt * c.kh * c.kw;
            const float ws = 1.0f / std::sqrt(float(K));
            const int xin = c.cin * c.t * c.h * c.w;
            const auto xv = random_values(std::size_t(c.n) * xin, seed++, -1, 1, dt);
            const auto wv = random_values(std::size_t(c.cout) * K, seed++, -ws, ws, dt);
            const auto bv = random_values(std::size_t(c.cout), seed++, -0.5f, 0.5f, dt);
            Tensor xc = cpu_tensor(xv, c.n, xin), wc = cpu_tensor(wv, c.cout, K), bc = cpu_tensor(bv, c.cout, 1);
            Tensor xg = upload(xv, c.n, xin, dt), wg = upload(wv, c.cout, K, dt), bg = upload(bv, c.cout, 1, dt);
            Tensor yc, yg;
            brotensor::conv3d_forward(xc, wc, &bc, c.n, c.cin, c.t, c.h, c.w, c.cout, c.kt, c.kh, c.kw, c.st, c.sh,
                                      c.sw, c.pt, c.ph, c.pw, 1, 1, 1, c.groups, yc);
            brotensor::conv3d_forward(xg, wg, &bg, c.n, c.cin, c.t, c.h, c.w, c.cout, c.kt, c.kh, c.kw, c.st, c.sh,
                                      c.sw, c.pt, c.ph, c.pw, 1, 1, 1, c.groups, yg);
            VKT_CHECK(yg.rows == yc.rows && yg.cols == yc.cols && yg.dtype == dt);
            close_conv(dt, download(yg), yc.to_host_vector(), K, std::string("conv3d ") + c.name + " " + dt_name(dt));
        }
    }
}

void test_conv_transpose2d() {
    std::printf("conv_transpose2d\n");
    struct CT { int n, cin, h, w, cout, kh, kw, sh, sw, ph, pw, oph, opw, dh, dw, groups; const char* name; };
    const CT cases[] = {
        {1, 32, 8, 8, 16, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, "2x2 stride 2 (SAM upscale)"},
        {2, 8, 7, 5, 12, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, "3x3 stride 2 output padding"},
        {1, 12, 6, 6, 8, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, 4, "4x4 stride 2 groups 4"},
        {1, 6, 5, 9, 4, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 1, "3x3 dilation 2"},
    };
    std::uint64_t seed = 700;
    for (Dtype dt : kFloatTypes) {
        for (const CT& c : cases) {
            const int cgo = c.cout / c.groups, kk = c.kh * c.kw;
            const auto xv = random_values(std::size_t(c.n) * c.cin * c.h * c.w, seed++, -1, 1, dt);
            const auto wv = random_values(std::size_t(c.cin) * cgo * kk, seed++, -0.3f, 0.3f, dt);
            const auto bv = random_values(std::size_t(c.cout), seed++, -0.5f, 0.5f, dt);
            Tensor xc = cpu_tensor(xv, c.n, c.cin * c.h * c.w), wc = cpu_tensor(wv, c.cin, cgo * kk),
                   bc = cpu_tensor(bv, c.cout, 1);
            Tensor xg = upload(xv, c.n, c.cin * c.h * c.w, dt), wg = upload(wv, c.cin, cgo * kk, dt),
                   bg = upload(bv, c.cout, 1, dt);
            Tensor yc, yg;
            brotensor::conv_transpose2d_forward(xc, wc, &bc, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw,
                                                c.ph, c.pw, c.oph, c.opw, c.dh, c.dw, c.groups, yc);
            brotensor::conv_transpose2d_forward(xg, wg, &bg, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw,
                                                c.ph, c.pw, c.oph, c.opw, c.dh, c.dw, c.groups, yg);
            VKT_CHECK(yg.rows == yc.rows && yg.cols == yc.cols && yg.dtype == dt);
            close_conv(dt, download(yg), yc.to_host_vector(), c.cin * kk,
                       std::string("conv_transpose2d ") + c.name + " " + dt_name(dt));
        }
    }
}

void test_backward_input() {
    std::printf("conv2d_backward_input\n");
    const Conv cases[] = {
        {2, 16, 9, 11, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1, false, "3x3"},
        {1, 12, 13, 10, 8, 3, 3, 2, 2, 1, 1, 1, 1, 4, false, "3x3 stride 2 groups 4"},
        {1, 6, 8, 8, 10, 5, 3, 1, 2, 2, 1, 2, 1, 2, false, "5x3 dilation (2,1) groups 2"},
    };
    std::uint64_t seed = 1200;
    for (Dtype dt : kFloatTypes) {
        for (const Conv& c : cases) {
            const int K = (c.cin / c.groups) * c.kh * c.kw;
            const int ho = out_dim(c.h, c.ph, c.dh, c.kh, c.sh), wo = out_dim(c.w, c.pw, c.dw, c.kw, c.sw);
            const auto wv = random_values(std::size_t(c.cout) * K, seed++, -0.3f, 0.3f, dt);
            const auto dyv = random_values(std::size_t(c.n) * c.cout * ho * wo, seed++, -1, 1, dt);
            Tensor wc = cpu_tensor(wv, c.cout, K), dyc = cpu_tensor(dyv, c.n, c.cout * ho * wo);
            Tensor wg = upload(wv, c.cout, K, dt), dyg = upload(dyv, c.n, c.cout * ho * wo, dt);
            Tensor dxc, dxg;
            brotensor::conv2d_backward_input(wc, dyc, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                                             c.dh, c.dw, c.groups, dxc);
            brotensor::conv2d_backward_input(wg, dyg, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                                             c.dh, c.dw, c.groups, dxg);
            VKT_CHECK(dxg.rows == dxc.rows && dxg.cols == dxc.cols && dxg.dtype == dt);
            close_conv(dt, download(dxg), dxc.to_host_vector(), c.cout * c.kh * c.kw,
                       std::string("conv2d_backward_input ") + c.name + " " + dt_name(dt));
        }
    }
}

void test_bias_grad() {
    std::printf("conv bias gradients\n");
    for (Dtype dt : kFloatTypes) {
        const int N = 3, C = 5, H = 7, W = 9;
        const auto dyv = random_values(std::size_t(N) * C * H * W, 900, -1, 1, dt);
        const auto b0 = random_values(C, 901, -1, 1, dt);
        Tensor dyc = cpu_tensor(dyv, N, C * H * W), dbc = cpu_tensor(b0, C, 1);
        Tensor dyg = upload(dyv, N, C * H * W, dt), dbg = upload(b0, C, 1, dt);
        brotensor::conv2d_backward_bias(dyc, N, C, H, W, dbc);
        brotensor::conv2d_backward_bias(dyg, N, C, H, W, dbg);
        close_conv(dt, download(dbg), dbc.to_host_vector(), N * H * W, std::string("conv2d_backward_bias ") + dt_name(dt));
        Tensor dbt = upload(b0, C, 1, dt);
        brotensor::conv_transpose2d_backward_bias(dyg, N, C, H, W, dbt);
        close_conv(dt, download(dbt), dbc.to_host_vector(), N * H * W,
                   std::string("conv_transpose2d_backward_bias ") + dt_name(dt));
    }
}

void test_resblock() {
    std::printf("resblock_forward\n");
    struct RB { int n, cin, cout, h, w, groups; bool skip, temb, temb_n; };
    const RB cases[] = {
        {1, 64, 64, 12, 12, 32, false, true, false},
        {2, 32, 64, 9, 10, 8, true, true, true},
        {1, 32, 64, 16, 16, 16, true, false, false},
        {2, 8, 8, 6, 6, 4, false, true, true},   // narrow: the direct kernel
    };
    std::uint64_t seed = 1000;
    for (Dtype dt : kFloatTypes) {
        for (const RB& r : cases) {
            const int hw = r.h * r.w;
            auto vec = [&](std::size_t n, float lo, float hi) { return random_values(n, seed++, lo, hi, dt); };
            const float s1 = 1.0f / std::sqrt(9.0f * r.cin), s2 = 1.0f / std::sqrt(9.0f * r.cout);
            const auto xv = vec(std::size_t(r.n) * r.cin * hw, -1, 1);
            const auto g1 = vec(r.cin, 0.5f, 1.5f), be1 = vec(r.cin, -0.2f, 0.2f);
            const auto w1 = vec(std::size_t(r.cout) * r.cin * 9, -s1, s1), b1 = vec(r.cout, -0.2f, 0.2f);
            const auto tv = vec(std::size_t(r.temb_n ? r.n : 1) * r.cout, -0.5f, 0.5f);
            const auto g2 = vec(r.cout, 0.5f, 1.5f), be2 = vec(r.cout, -0.2f, 0.2f);
            const auto w2 = vec(std::size_t(r.cout) * r.cout * 9, -s2, s2), b2 = vec(r.cout, -0.2f, 0.2f);
            const auto ws = vec(std::size_t(r.cout) * r.cin, -0.2f, 0.2f), bs = vec(r.cout, -0.2f, 0.2f);
            auto both = [&](const std::vector<float>& v, int rows, int cols) {
                return std::make_pair(cpu_tensor(v, rows, cols), upload(v, rows, cols, dt));
            };
            auto X = both(xv, r.n, r.cin * hw), G1 = both(g1, r.cin, 1), B1 = both(be1, r.cin, 1);
            auto W1 = both(w1, r.cout, r.cin * 9), Bi1 = both(b1, r.cout, 1);
            auto T = both(tv, r.temb_n ? r.n : r.cout, r.temb_n ? r.cout : 1);
            auto G2 = both(g2, r.cout, 1), B2 = both(be2, r.cout, 1), W2 = both(w2, r.cout, r.cout * 9),
                 Bi2 = both(b2, r.cout, 1), WS = both(ws, r.cout, r.cin), BS = both(bs, r.cout, 1);
            Tensor yc, yg;
            brotensor::resblock_forward(X.first, G1.first, B1.first, W1.first, &Bi1.first, r.temb ? &T.first : nullptr,
                                        G2.first, B2.first, W2.first, &Bi2.first, r.skip ? &WS.first : nullptr,
                                        r.skip ? &BS.first : nullptr, r.n, r.cin, r.cout, r.h, r.w, r.groups, 1e-5f, yc);
            brotensor::resblock_forward(X.second, G1.second, B1.second, W1.second, &Bi1.second,
                                        r.temb ? &T.second : nullptr, G2.second, B2.second, W2.second, &Bi2.second,
                                        r.skip ? &WS.second : nullptr, r.skip ? &BS.second : nullptr, r.n, r.cin,
                                        r.cout, r.h, r.w, r.groups, 1e-5f, yg);
            // A chain of two GroupNorms and two convolutions, the 16-bit
            // intermediates rounded at every stage (the CPU keeps FP32).
            const float tol = dt == Dtype::FP32 ? 2e-5f : (dt == Dtype::FP16 ? 5e-3f : 4e-2f);
            char tg[96];
            std::snprintf(tg, sizeof tg, "resblock %d->%d %dx%d n%d%s%s %s", r.cin, r.cout, r.h, r.w, r.n,
                          r.skip ? " skip" : "", r.temb ? (r.temb_n ? " temb(N)" : " temb") : "", dt_name(dt));
            expect_close(download(yg), yc.to_host_vector(), tol, tol, tg);
        }
    }
}

}  // namespace

void run_conv_tests() {
    test_conv2d();
    test_conv3d();
    test_conv_transpose2d();
    test_backward_input();
    test_bias_grad();
    test_resblock();
}

}  // namespace vkt
