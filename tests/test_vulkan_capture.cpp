// The device-neutral graph capture on Vulkan (CudaGraphCapture / CudaGraph,
// cuda_graph.h) and custom kernels (vulkan::register_shader / dispatch, built
// by brotensor_vulkan_add_shaders from tests/vulkan_shaders/).
// `brotensor_test_vulkan --only=capture`.

#include "test_vulkan_common.h"

#include "brotensor_test_vk_shaders.h"   // generated: brotensor_test_vk::Shader

#include <brotensor/cuda_graph.h>
#include <brotensor/runtime.h>
#include <brotensor/vulkan.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace vkt {

namespace {

namespace bt = ::brotensor;

void chain(const Tensor& x, Tensor& y, Tensor& tmp, int reps) {
    bt::copy_d2d(x, 0, y, 0, x.size());
    for (int i = 0; i < reps; ++i) {
        bt::scale_inplace(y, 0.75f);
        bt::add_scalar_inplace(y, 0.125f);
        bt::relu_forward(y, tmp);
        bt::axpby_inplace(tmp, x, 1.0f, 0.5f);
        bt::copy_d2d(tmp, 0, y, 0, y.size());
    }
}

std::vector<float> chain_ref(const std::vector<float>& x, int reps) {
    Tensor xc = Tensor::from_host_on(Device::cpu(), x.data(), static_cast<int>(x.size()), 1);
    Tensor y = Tensor::zeros_on(Device::cpu(), static_cast<int>(x.size()), 1), tmp;
    chain(xc, y, tmp, reps);
    return y.to_host_vector();
}

struct AxpyPush {
    std::uint64_t x, y;
    std::uint32_t n;
    float a;
};

}  // namespace

void test_neutral_capture() {
    std::printf("device-neutral capture (CudaGraphCapture on Vulkan)\n");
    VKT_CHECK(bt::graph_capture_available(vk()));
    VKT_CHECK(!bt::graph_capture_available(Device::cpu()));

    const int n = 2048, reps = 20;
    // Vulkan as the default device (a DeviceScope).
    {
        bt::DeviceScope scope(vk());
        VKT_CHECK(bt::graph_capture_available());
        auto xv = random_values(n, 5, -1, 1, Dtype::FP32);
        Tensor x = Tensor::from_host(xv.data(), n, 1);
        VKT_CHECK(x.device == vk());
        Tensor y = Tensor::zeros(n, 1), tmp = Tensor::zeros(n, 1);
        chain(x, y, tmp, reps);   // warm-up
        bt::sync_all();

        const auto st0 = bt::vulkan::stream_stats(vk());
        bt::CudaGraph g;
        {
            bt::CudaGraphCapture cap;
            VKT_CHECK(cap.device() == vk());
            chain(x, y, tmp, reps);
            // The Vulkan capture's rule: no host waits while recording.
            VKT_CHECK(throws([&] { bt::sync(vk()); }));
            g = cap.finish();
            VKT_CHECK(throws([&] { (void)cap.finish(); }));
        }
        VKT_CHECK(g.valid() && g.device() == vk());
        for (int t = 0; t < 2; ++t) {
            xv = random_values(n, 40 + t, -1, 1, Dtype::FP32);
            x.copy_from_host_raw(xv.data(), xv.size() * 4);
            g.launch();
            bt::sync_all();
            expect_close(y.to_host_vector(), chain_ref(xv, reps), 1e-6f, 1e-6f,
                         "neutral replay " + std::to_string(t));
        }
        const auto st1 = bt::vulkan::stream_stats(vk());
        VKT_CHECK(st1.launches == st0.launches + 2);
        g.reset();
        VKT_CHECK(!g.valid() && g.device() == Device::cpu());

        // An abandoned neutral capture is discarded and the device works on.
        {
            bt::CudaGraphCapture cap;
            bt::relu_forward(x, tmp);
        }
        bt::relu_forward(x, tmp);
        std::vector<float> want(xv);
        for (float& v : want) v = v > 0 ? v : 0;
        expect_close(tmp.to_host_vector(), want, 0, 0, "ops after an abandoned neutral capture");
    }

    // A named device, with the CPU as the default device.
    {
        bt::DeviceScope scope(Device::cpu());
        auto xv = random_values(n, 9, -1, 1, Dtype::FP32);
        Tensor x = upload(xv, n, 1, Dtype::FP32);
        Tensor y = Tensor::zeros_on(vk(), n, 1), tmp = Tensor::zeros_on(vk(), n, 1);
        chain(x, y, tmp, 3);
        bt::sync(vk());
        bt::CudaGraph g;
        {
            bt::CudaGraphCapture cap(vk());
            VKT_CHECK(cap.device() == vk());
            chain(x, y, tmp, 3);
            g = cap.finish();
        }
        xv = random_values(n, 10, -1, 1, Dtype::FP32);
        x.copy_from_host_raw(xv.data(), xv.size() * 4);
        g.launch();
        bt::sync(vk());
        expect_close(y.to_host_vector(), chain_ref(xv, 3), 1e-6f, 1e-6f, "named-device replay");
        VKT_CHECK(throws([&] { bt::CudaGraphCapture cap(Device::cpu()); }));
    }
}

void test_custom_kernels() {
    std::printf("custom kernels (register_shader / dispatch)\n");
    namespace sh = ::brotensor_test_vk;
    const auto h = sh::handle(sh::Shader::custom_axpy);
    VKT_CHECK(h != 0);
    VKT_CHECK(sh::handle(sh::Shader::custom_axpy) == h);
    VKT_CHECK(sh::handle(sh::Shader::custom_axpy_f64) != h);

    const std::uint32_t spec[] = {128};
    const auto ki = bt::vulkan::kernel_info(vk(), h, spec, 1);
    VKT_CHECK(ki.local[0] == 128 && ki.local[1] == 1);

    const int n = 100000;
    auto xv = random_values(n, 21, -2, 2, Dtype::FP32);
    auto yv = random_values(n, 22, -2, 2, Dtype::FP32);
    const float a = 1.75f;
    std::vector<float> want(yv);
    for (int i = 0; i < n; ++i) want[i] = a * xv[i] + yv[i];

    Tensor x = upload(xv, n, 1, Dtype::FP32), y = upload(yv, n, 1, Dtype::FP32);
    AxpyPush pc{bt::vulkan::address(x), bt::vulkan::address(y), static_cast<std::uint32_t>(n), a};
    bt::vulkan::dispatch(vk(), h, &pc, sizeof pc, 64, 1, 1, spec, 1);
    expect_close(y.to_host_vector(), want, 1e-6f, 1e-6f, "custom axpy");

    // Ordered with brotensor ops and recorded by a neutral capture.
    {
        Tensor y2 = upload(yv, n, 1, Dtype::FP32);
        AxpyPush p2{bt::vulkan::address(x), bt::vulkan::address(y2), static_cast<std::uint32_t>(n), a};
        bt::CudaGraph g;
        {
            bt::CudaGraphCapture cap(vk());
            bt::scale_inplace(y2, 2.0f);
            bt::vulkan::dispatch(vk(), h, &p2, sizeof p2, 64, 1, 1, spec, 1);
            g = cap.finish();
        }
        y2.copy_from_host_raw(yv.data(), yv.size() * 4);
        g.launch();
        bt::sync(vk());
        std::vector<float> w2(yv);
        for (int i = 0; i < n; ++i) w2[i] = a * xv[i] + 2.0f * yv[i];
        expect_close(y2.to_host_vector(), w2, 1e-6f, 1e-6f, "custom kernel in a capture");
    }

    if (bt::vulkan::device_info(vk()).shader_float64) {
        const auto h64 = sh::handle(sh::Shader::custom_axpy_f64);
        Tensor y3 = upload(yv, n, 1, Dtype::FP32);
        AxpyPush p3{bt::vulkan::address(x), bt::vulkan::address(y3), static_cast<std::uint32_t>(n), a};
        bt::vulkan::dispatch(vk(), h64, &p3, sizeof p3, 64, 1, 1, spec, 1);
        std::vector<float> w3(n);
        for (int i = 0; i < n; ++i) w3[i] = static_cast<float>(double(a) * xv[i] + yv[i]);
        expect_close(y3.to_host_vector(), w3, 1e-7f, 1e-7f, "custom double axpy");
    } else {
        std::printf("  (no shaderFloat64: double kernel skipped)\n");
    }

    // Misuse is an exception, not a device fault.
    VKT_CHECK(throws([&] { bt::vulkan::dispatch(vk(), h, &pc, 132, 1, 1, 1, spec, 1); }));
    VKT_CHECK(throws([&] { bt::vulkan::dispatch(vk(), 0xfffffu, &pc, sizeof pc, 1); }));
    VKT_CHECK(throws([&] { bt::vulkan::dispatch(Device::cpu(), h, &pc, sizeof pc, 1); }));
    const std::uint32_t junk[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    VKT_CHECK(throws([&] { (void)bt::vulkan::register_shader("junk", junk, 8); }));
}

void run_capture_tests() {
    test_neutral_capture();
    test_custom_kernels();
}

}  // namespace vkt
