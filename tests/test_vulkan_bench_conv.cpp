// `brotensor_test_vulkan --bench-conv`: conv2d_forward throughput on the
// VAE-decoder and SD U-Net shapes (FP16, NCHW), then the bandwidth-bound spatial ops (GroupNorm, the 2x upsamples,
// bilinear / bicubic interpolation) in GB/s against the 256 GB/s peak.
// Not part of ctest.
//
// Timing: wall clock around back-to-back launches with one device sync,
// median of 7 batches after warm-up (test_vulkan_bench.cpp's method). TF/s
// counts 2 C_out C_in kH kW H_out W_out N flops. GB/s counts the minimum
// traffic (each input read once, each output written once); GroupNorm
// actually reads X twice. BROTENSOR_VK_BENCH_SHAPE=<tag> runs one shape,
// BROTENSOR_VK_BENCH_PART=conv|band runs one table, BROTENSOR_VK_CONV_CFG=bm,bn,bk,wm,wn forces a tile,
// BROTENSOR_VK_BENCH_DTYPE=f32|bf16 changes the conv dtype and
// BROTENSOR_VK_BENCH_CONV_PATH=simt|direct forces a Vulkan conv path.

#include "test_vulkan_common.h"

#include "detail/spatial.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace vkt {

namespace {

template <class F>
double median_ms(Device dev, F&& launch_once) {
    for (int i = 0; i < 3; ++i) launch_once();
    brotensor::sync(dev);
    auto t0 = std::chrono::steady_clock::now();
    launch_once();
    brotensor::sync(dev);
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int reps = std::max(3, std::min(200, static_cast<int>(150.0 / std::max(one, 1e-3))));
    std::vector<double> t;
    for (int b = 0; b < 7; ++b) {
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) launch_once();
        brotensor::sync(dev);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

bool selected(const char* tag) {
    const char* only = std::getenv("BROTENSOR_VK_BENCH_SHAPE");
    return !only || std::string(only) == tag;
}

struct ConvShape {
    int n, cin, cout, h, w, k;
    const char* tag;
};

const ConvShape kConvShapes[] = {
    {1, 512, 512, 64, 64, 3, "vae-512c-64"},
    {1, 512, 512, 128, 128, 3, "vae-512c-128"},
    {1, 256, 256, 256, 256, 3, "vae-256c-256"},
    {1, 128, 128, 512, 512, 3, "vae-128c-512"},
    {1, 256, 128, 512, 512, 3, "vae-256to128-512"},
    {1, 512, 256, 256, 256, 1, "vae-1x1-512to256-256"},
    {2, 320, 320, 64, 64, 3, "unet-320c-64-b2"},
    {2, 640, 640, 32, 32, 3, "unet-640c-32-b2"},
    {2, 1280, 1280, 16, 16, 3, "unet-1280c-16-b2"},
};

Dtype bench_dtype() {
    const char* e = std::getenv("BROTENSOR_VK_BENCH_DTYPE");
    if (e && std::string(e) == "f32") return Dtype::FP32;
    if (e && std::string(e) == "bf16") return Dtype::BF16;
    return Dtype::FP16;
}

struct ConvProblem {
    Tensor X, W, B, Y;
    ConvProblem(Device dev, const ConvShape& s) {
        const int K = s.cin * s.k * s.k;
        const Dtype dt = bench_dtype();
        X = upload(random_values(std::size_t(s.n) * s.cin * s.h * s.w, 1, -1, 1, dt), s.n, s.cin * s.h * s.w, dt, dev);
        W = upload(random_values(std::size_t(s.cout) * K, 2, -0.05f, 0.05f, dt), s.cout, K, dt, dev);
        B = upload(random_values(std::size_t(s.cout), 3, -0.1f, 0.1f, dt), s.cout, 1, dt, dev);
    }
    void run(const ConvShape& s) {
        const int p = s.k / 2;
        brotensor::conv2d_forward(X, W, &B, s.n, s.cin, s.h, s.w, s.cout, s.k, s.k, 1, 1, p, p, 1, 1, 1, Y);
    }
};

void bench_conv() {
    std::printf("\nconv2d_forward, %s, stride 1, same padding (ms / TF/s)\n", dt_name(bench_dtype()));
    const char* path = std::getenv("BROTENSOR_VK_BENCH_CONV_PATH");
    if (path) brotensor::detail::vulkan::set_conv_override(std::string(path) == "direct" ? 2 : 1);
    std::printf("%-22s %16s\n", "shape", "vulkan");
    for (const ConvShape& s : kConvShapes) {
        if (!selected(s.tag)) continue;
        const double flop = 2.0 * s.n * s.cout * s.cin * s.k * s.k * double(s.h) * s.w;
        double ms = 0;
        ConvProblem p(vk(), s);
        try {
            ms = median_ms(vk(), [&] { p.run(s); });
        } catch (const std::exception&) {
            ms = 0;
        }
        char a[32];
        std::snprintf(a, sizeof a, "%.3f / %5.2f", ms, flop / (ms * 1e9));
        std::printf("%-22s %16s\n", s.tag, a);
    }
    brotensor::detail::vulkan::set_conv_override(0);
}

void bench_bandwidth() {
    std::printf("\nbandwidth-bound spatial ops, FP16 (ms / GB/s of minimum traffic, peak 256)\n");
    std::printf("%-30s %16s\n", "op", "vulkan");
    struct B { const char* tag; int c, h, w, ho, wo, kind; };   // kind: 0 GN, 1 up-nearest, 2 bilinear, 3 bicubic
    const B shapes[] = {
        {"groupnorm-512c-64", 512, 64, 64, 64, 64, 0},
        {"groupnorm-512c-128", 512, 128, 128, 128, 128, 0},
        {"groupnorm-256c-256", 256, 256, 256, 256, 256, 0},
        {"groupnorm-128c-512", 128, 512, 512, 512, 512, 0},
        {"upsample-nearest-512c-64", 512, 64, 64, 128, 128, 1},
        {"upsample-nearest-512c-128", 512, 128, 128, 256, 256, 1},
        {"upsample-nearest-256c-256", 256, 256, 256, 512, 512, 1},
        {"bilinear-512c-64to128", 512, 64, 64, 128, 128, 2},
        {"bilinear-256c-100to333", 256, 100, 100, 333, 333, 2},
        {"bicubic-64c-512to224", 64, 512, 512, 224, 224, 3},
    };
    for (const B& s : shapes) {
        if (!selected(s.tag)) continue;
        const double bytes = 2.0 * s.c * (double(s.h) * s.w + double(s.ho) * s.wo);
        double ms = 0;
        {
            const Device dev = vk();
            Tensor X = upload(random_values(std::size_t(s.c) * s.h * s.w, 4, -1, 1, Dtype::FP16), 1, s.c * s.h * s.w,
                              Dtype::FP16, dev);
            Tensor g = upload(random_values(s.c, 5, 0.5f, 1.5f, Dtype::FP16), s.c, 1, Dtype::FP16, dev);
            Tensor be = upload(random_values(s.c, 6, -0.5f, 0.5f, Dtype::FP16), s.c, 1, Dtype::FP16, dev);
            Tensor Y;
            try {
                ms = median_ms(dev, [&] {
                    if (s.kind == 0) brotensor::group_norm_forward(X, g, be, 1, s.c, s.h, s.w, 32, 1e-6f, Y);
                    else if (s.kind == 1) brotensor::upsample_nearest_2x(X, 1, s.c, s.h, s.w, Y);
                    else brotensor::interp2d_forward(X, 1, s.c, s.h, s.w, s.ho, s.wo, s.kind == 2 ? 1 : 3, Y);
                });
            } catch (const std::exception&) {
                ms = 0;
            }
        }
        char a[32];
        std::snprintf(a, sizeof a, "%.3f / %5.0f", ms, bytes / (ms * 1e6));
        std::printf("%-30s %16s\n", s.tag, a);
    }
}

}  // namespace

void run_conv_bench() {
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("conv / spatial benchmark on %s (%s)\n", info.name.c_str(), info.driver.c_str());
    const char* part = std::getenv("BROTENSOR_VK_BENCH_PART");   // conv | band
    if (!part || std::string(part) == "conv") bench_conv();
    if (!part || std::string(part) == "band") bench_bandwidth();
}

}  // namespace vkt
