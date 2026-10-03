// Vulkan backend: registration and device plumbing, transfers, the allocator,
// the pipeline limit guard, events, batching and record / replay graphs. The
// op parity half is test_vulkan_ops.cpp. Skips (exit 0) when no Vulkan device
// registers. `--expect-default-vulkan` is the BROTENSOR_DEFAULT_DEVICE=vulkan
// run registered in tests/CMakeLists.txt.

#include "test_vulkan_common.h"

// White-box: the pipeline guard test reaches into the backend.
#include "detail/device.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace vkt {

namespace {

std::uint64_t address(const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data); }

bool contains(const std::vector<Device>& v, Device d) { return std::find(v.begin(), v.end(), d) != v.end(); }

// Device::CUDA without a CUDA backend: HIP when it is registered, else
// Vulkan (detail::resolve_device_alias). Everything that acts on the device
// lands there; is_available(Device::CUDA) stays false. ctest runs this a
// second time with HIP hidden (brotensor_test_vulkan_cuda_alias).
void test_cuda_alias() {
    std::printf("Device::CUDA alias\n");
    if (brotensor::is_available(Device::CUDA)) {
        std::printf("  (CUDA backend present: alias is the identity, skipped)\n");
        return;
    }
    VKT_CHECK(!brotensor::is_available(Device::CUDA));
    const bool hip = brotensor::is_available(Device::HIP);
    const Device target = hip ? Device::HIP : vk();
    std::printf("  Device::CUDA -> %s\n", brotensor::to_string(target).c_str());
    Tensor t = Tensor::zeros_on(Device::CUDA, 3, 5);
    VKT_CHECK(t.device == target);
    Tensor v = Tensor::view(Device::CUDA, t.data, 3, 5, Dtype::FP32);
    VKT_CHECK(v.device == target);
    Tensor w;
    w.device = Device::CUDA;
    w.resize(2, 2);
    VKT_CHECK(w.device == target);
    brotensor::add_scalar_inplace(t, 2.0f);
    brotensor::sync(Device::CUDA);
    const auto tv = t.to_host_vector();
    VKT_CHECK(tv.size() == 15 && tv[7] == 2.0f);
    {
        brotensor::DeviceScope scope(Device::CUDA);
        VKT_CHECK(brotensor::default_device() == target);
        VKT_CHECK(Tensor::zeros(1, 1).device == target);
    }
    Tensor out;
    out.device = Device::CUDA;
    brotensor::relu_forward(t, out);
    VKT_CHECK(out.device == target && out.data != nullptr);
    if (!hip) {
        std::size_t free_b = 0, total_b = 0;
        VKT_CHECK(brotensor::device_mem_info(Device::CUDA, free_b, total_b) && total_b > 0);
        brotensor::vulkan::flush(Device::CUDA);
    }
    std::printf("  PASS  Device::CUDA alias\n");
}

void test_registration(bool expect_default_vulkan) {
    std::printf("registration\n");
    VKT_CHECK(brotensor::vulkan_device_count() >= 1);
    VKT_CHECK(brotensor::is_available(vk()));
    VKT_CHECK(!brotensor::is_available(Device::vulkan(brotensor::vulkan_device_count())));
    VKT_CHECK(contains(brotensor::available_devices(), vk()));
    VKT_CHECK(std::strcmp(brotensor::device_name(vk()), "vulkan") == 0);
    VKT_CHECK(brotensor::to_string(Device::vulkan(1)) == "vulkan:1");
    VKT_CHECK(vk().is_vulkan() && vk().is_gpu() && Device::VULKAN == vk());
    VKT_CHECK(brotensor::compute_dtype(vk()) == Dtype::FP16);
    const std::string prod = brotensor::device_product_name(vk());
    VKT_CHECK(!prod.empty());
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("  device: %s | %s | subgroup %u (%u..%u) | shared %u B | push %u B | %s queue | coopmat %d\n",
                info.name.c_str(), info.driver.c_str(), info.subgroup_size, info.min_subgroup_size,
                info.max_subgroup_size, info.max_shared_bytes, info.max_push_constant_bytes,
                info.graphics_queue ? "graphics" : "compute", info.cooperative_matrix ? 1 : 0);
    VKT_CHECK(info.graphics_queue);
    if (expect_default_vulkan) {
        VKT_CHECK(brotensor::default_device() == vk());
        Tensor t = Tensor::zeros(2, 3);
        VKT_CHECK(t.device == vk());
    } else {
        // Coverage is partial: Vulkan is never the default unless asked for.
        VKT_CHECK(!brotensor::default_device().is_vulkan());
    }
    {
        brotensor::DeviceScope scope(vk());
        Tensor t = Tensor::zeros(4, 4);
        VKT_CHECK(t.device == vk());
    }
    VKT_CHECK(!Tensor::zeros(1, 1).device.is_vulkan() || expect_default_vulkan);
    std::size_t free_b = 0, total_b = 0;
    VKT_CHECK(brotensor::device_mem_info(vk(), free_b, total_b));
    VKT_CHECK(total_b > 0 && free_b <= total_b);
    std::printf("  memory: %.1f GiB free of %.1f GiB\n", free_b / 1073741824.0, total_b / 1073741824.0);
    // Ops between devices are rejected by the dispatcher.
    Tensor a = Tensor::zeros_on(vk(), 2, 2), c = Tensor::zeros_on(Device::cpu(), 2, 2);
    VKT_CHECK(throws([&] { brotensor::add_inplace(a, c); }));
    // An op the backend does not implement names the device.
    bool named = false;
    try {
        Tensor dq, dk, dv;   // attention backwards: not on Vulkan yet
        brotensor::flash_attention_backward(a, a, a, a, a, nullptr, 1, false, dq, dk, dv);
    } catch (const std::exception& e) {
        named = std::string(e.what()).find("vulkan") != std::string::npos;
    }
    VKT_CHECK(named);
}

void test_transfers() {
    std::printf("transfers\n");
    // Sizes covering the inline-update path (<= 64 KiB), the mapped / staging
    // paths and a dedicated allocation larger than one 64 MiB staging chunk.
    for (std::size_t n : {std::size_t(1), std::size_t(1000), std::size_t(70000), std::size_t(40) << 20}) {
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i % 9973) * 0.5f - 7.0f;
        Tensor busy = Tensor::zeros_on(vk(), 1024, 1024);
        brotensor::relu_forward(busy, busy);          // stream not idle
        Tensor t = Tensor::from_host_on(vk(), v.data(), static_cast<int>(n), 1);
        const auto back = t.to_host_vector();
        expect_equal_bits(back.data(), v.data(), n * 4, "round trip " + std::to_string(n) + " floats");
    }
    // Device <-> device, clone, to(), and the CPU bounce.
    const auto v = random_values(3000, 1, -1, 1, Dtype::FP32);
    Tensor a = upload(v, 30, 100, Dtype::FP32);
    Tensor b = a.clone();
    VKT_CHECK(b.device == vk() && b.data != a.data);
    Tensor c = b.to(Device::cpu());
    expect_equal_bits(c.host_f32(), v.data(), v.size() * 4, "clone + to(cpu)");
    Tensor d = c.to(vk());
    expect_close(download(d), v, 0, 0, "cpu -> vulkan via to()");
    if (brotensor::is_available(Device::hip(0))) {
        Tensor h = d.to(Device::hip(0));
        Tensor back = h.to(vk());
        expect_close(download(back), v, 0, 0, "vulkan -> hip -> vulkan");
    }
    if (brotensor::vulkan_device_count() > 1) {
        Tensor p = d.to(Device::vulkan(1));
        expect_close(download(p), v, 0, 0, "vulkan:0 -> vulkan:1");
    }
    // Raw transfers through an unaligned view of an INT8 tensor.
    std::vector<std::uint8_t> bytes(100, 0xab);
    Tensor raw = Tensor::from_raw_bytes_on(vk(), bytes.data(), 100, 1, Dtype::INT8, 100);
    Tensor view = Tensor::view(vk(), static_cast<char*>(raw.data) + 3, 9, 1, Dtype::INT8);
    const std::uint8_t five[5] = {1, 2, 3, 4, 5};
    view.copy_from_host_raw(five, 5);
    std::vector<std::uint8_t> got(100);
    raw.copy_to_host_raw(got.data(), 100);
    std::vector<std::uint8_t> want(100, 0xab);
    std::copy(five, five + 5, want.begin() + 3);
    expect_equal_bits(got.data(), want.data(), 100, "unaligned raw upload into a view");
    // zero(): whole tensor, then an unaligned 9-byte window (byte kernel edges).
    view.zero();
    raw.copy_to_host_raw(got.data(), 100);
    std::fill(want.begin() + 3, want.begin() + 12, 0);
    expect_equal_bits(got.data(), want.data(), 100, "zero() on an unaligned view");
    a.zero();
    const auto z = download(a);
    VKT_CHECK(std::all_of(z.begin(), z.end(), [](float x) { return x == 0.0f; }));
    // Tensor::resize keeps storage when it fits.
    void* before = a.data;
    a.resize(10, 10);
    VKT_CHECK(a.data == before);
}

void test_allocator() {
    std::printf("allocator\n");
    brotensor::sync(vk());
    const auto s0 = brotensor::vulkan::memory_stats(vk());
    std::printf("  per-buffer limit %.2f GiB, sub-allocation up to %zu MiB, device memory %s\n",
                s0.max_buffer_bytes / 1073741824.0, s0.sub_alloc_max >> 20,
                s0.host_mapped ? "host-mapped" : "not mapped");
    VKT_CHECK(s0.max_buffer_bytes <= (std::size_t(4) << 30));

    // Small tensors share a block.
    {
        Tensor a = Tensor::empty_on(vk(), 100, 100), b = Tensor::empty_on(vk(), 3, 7);
        const auto s1 = brotensor::vulkan::memory_stats(vk());
        VKT_CHECK(s1.live_allocations == s0.live_allocations + 2);
        VKT_CHECK(s1.dedicated_blocks == s0.dedicated_blocks);
        const std::uint64_t lo = std::min(address(a), address(b)), hi = std::max(address(a), address(b));
        VKT_CHECK(hi - lo < s1.block_bytes);
        VKT_CHECK(address(a) % 256 == 0 && address(b) % 256 == 0);
    }
    // Release and reuse: the freed range is handed out again.
    {
        std::uint64_t first = 0;
        {
            Tensor t = Tensor::empty_on(vk(), 1024, 256);   // 1 MiB
            first = address(t);
        }
        Tensor u = Tensor::empty_on(vk(), 1024, 256);
        VKT_CHECK(address(u) == first);
        const auto s = brotensor::vulkan::memory_stats(vk());
        VKT_CHECK(s.live_allocations == s0.live_allocations + 1);
    }
    // Above the sub-allocation size: a dedicated block, returned on free once
    // the GPU is past its last use.
    {
        Tensor big = Tensor::empty_on(vk(), 64, 1 << 20);   // 256 MiB
        const auto s = brotensor::vulkan::memory_stats(vk());
        VKT_CHECK(s.dedicated_blocks == s0.dedicated_blocks + 1);
        brotensor::relu_forward(big, big);
    }
    brotensor::sync(vk());
    VKT_CHECK(brotensor::vulkan::memory_stats(vk()).dedicated_blocks == s0.dedicated_blocks);
    // Over 4 GiB in one tensor: refused (split it), not handed to the driver.
    std::string msg;
    try {
        Tensor huge = Tensor::empty_on(vk(), 1 << 20, 1100);   // 4.3 GiB of FP32
    } catch (const std::exception& e) {
        msg = e.what();
    }
    VKT_CHECK(msg.find("4 GiB") != std::string::npos);
    // 3 GiB is fine, and addressing past 2 GiB inside it works.
    {
        const int rows = 768, cols = 1 << 20;   // 3 GiB of FP32
        Tensor t = Tensor::empty_on(vk(), rows, cols);
        const int off = rows * cols - 4096;
        const auto v = random_values(4096, 5, -2, 2, Dtype::FP32);
        Tensor src = upload(v, 4096, 1, Dtype::FP32);
        brotensor::copy_d2d(src, 0, t, off, 4096);
        Tensor tail = Tensor::view(vk(), static_cast<char*>(t.data) + std::size_t(off) * 4, 4096, 1);
        brotensor::scale_inplace(tail, 2.0f);
        std::vector<float> want(v);
        for (float& x : want) x *= 2.0f;
        expect_close(tail.to_host_vector(), want, 0, 0, "3 GiB tensor, op at the last 16 KiB");
    }
    // trim returns empty blocks.
    const auto before = brotensor::vulkan::memory_stats(vk());
    VKT_CHECK(brotensor::device_mem_trim(vk(), 0));
    const auto after = brotensor::vulkan::memory_stats(vk());
    VKT_CHECK(after.block_bytes <= before.block_bytes);
    std::printf("  trim: %zu -> %zu sub-allocation blocks\n", before.blocks, after.blocks);
}

void test_pipeline_guard() {
    std::printf("pipeline limit guard\n");
    namespace dv = brotensor::detail::vulkan;
    auto& pipes = dv::device(0).pipelines();
    const std::uint32_t limit = brotensor::vulkan::device_info(vk()).max_shared_bytes;
    const dv::Kernel& ok = pipes.get(dv::ShaderId::test_shared_overflow, {1024});
    VKT_CHECK(ok.pipe != VK_NULL_HANDLE && ok.shared_bytes == 4096 && ok.local[0] == 64);
    // One float past the device limit: rejected before the driver sees it
    // (RADV would raise SIGFPE inside vkCreateComputePipelines).
    std::string msg;
    try {
        pipes.get(dv::ShaderId::test_shared_overflow, {limit / 4 + 1});
    } catch (const std::exception& e) {
        msg = e.what();
    }
    VKT_CHECK(msg.find("shared memory") != std::string::npos);
    if (!msg.empty()) std::printf("  rejected: %s\n", msg.c_str());
}

// y = relu(x * 1.5 + 0.25) - x, repeated `reps` times: a chain of small kernels.
void step(const Tensor& x, Tensor& y, Tensor& tmp, int reps) {
    brotensor::copy_d2d(x, 0, y, 0, x.size());
    for (int i = 0; i < reps; ++i) {
        brotensor::scale_inplace(y, 1.5f);
        brotensor::add_scalar_inplace(y, 0.25f);
        brotensor::relu_forward(y, tmp);
        brotensor::axpby_inplace(tmp, x, 1.0f, -1.0f);
        brotensor::copy_d2d(tmp, 0, y, 0, y.size());
    }
}

std::vector<float> step_ref(const std::vector<float>& x, int reps) {
    Tensor xc = Tensor::from_host_on(Device::cpu(), x.data(), static_cast<int>(x.size()), 1);
    Tensor y = Tensor::zeros_on(Device::cpu(), static_cast<int>(x.size()), 1), tmp;
    step(xc, y, tmp, reps);
    return y.to_host_vector();
}

void test_events_and_batching() {
    std::printf("events / batching\n");
    const auto st0 = brotensor::vulkan::stream_stats(vk());
    Tensor x = upload(random_values(4096, 3, -1, 1, Dtype::FP32), 4096, 1, Dtype::FP32);
    Tensor y = Tensor::zeros_on(vk(), 4096, 1), tmp = Tensor::zeros_on(vk(), 4096, 1);
    step(x, y, tmp, 60);   // 301 commands
    auto ev = brotensor::vulkan::Event::record(vk());
    VKT_CHECK(ev.valid());
    ev.wait();
    VKT_CHECK(ev.query());
    const auto st1 = brotensor::vulkan::stream_stats(vk());
    VKT_CHECK(st1.commands >= st0.commands + 301);
    // Batches are submitted when full, not once per op.
    VKT_CHECK(st1.submits - st0.submits <= 301 / st1.batch_limit + 2);
    std::printf("  301 commands -> %llu submissions (batch limit %u), %zu pipelines\n",
                static_cast<unsigned long long>(st1.submits - st0.submits), st1.batch_limit, st1.pipelines);
}

void test_graph() {
    std::printf("record / replay graph\n");
    const int n = 4096, reps = 60;
    auto xv = random_values(n, 77, -1, 1, Dtype::FP32);
    Tensor x = upload(xv, n, 1, Dtype::FP32);
    Tensor y = Tensor::zeros_on(vk(), n, 1), tmp = Tensor::zeros_on(vk(), n, 1);
    step(x, y, tmp, reps);   // warm-up: pipelines exist, outputs allocated
    brotensor::sync(vk());

    std::uint64_t held_addr = 0;
    brotensor::VulkanGraph g;
    {
        brotensor::VulkanGraphCapture cap(vk());
        step(x, y, tmp, reps);
        {
            // Allocated and freed inside the capture: the graph keeps it.
            Tensor scratch = Tensor::empty_on(vk(), 512, 512);
            held_addr = address(scratch);
            brotensor::relu_forward(x, tmp);   // overwritten by the next replay's step anyway
            brotensor::copy_d2d(x, 0, tmp, 0, n);
        }
        // Waiting for the GPU is not allowed while recording.
        VKT_CHECK(throws([&] { brotensor::sync(vk()); }));
        VKT_CHECK(throws([&] { (void)y.to_host_vector(); }));
        VKT_CHECK(throws([&] { (void)Tensor::from_host_on(vk(), xv.data(), n, 1); }));
        g = cap.finish();
    }
    VKT_CHECK(g.valid() && g.device() == vk());
    {
        Tensor again = Tensor::empty_on(vk(), 512, 512);
        VKT_CHECK(address(again) != held_addr);   // still owned by the graph
    }
    const auto st0 = brotensor::vulkan::stream_stats(vk());
    for (int it = 0; it < 3; ++it) {
        xv = random_values(n, 100 + it, -1, 1, Dtype::FP32);
        x.copy_from_host_raw(xv.data(), xv.size() * 4);
        g.launch();
        brotensor::sync(vk());
        expect_close(y.to_host_vector(), step_ref(xv, reps), 1e-6f, 1e-6f,
                     "replay " + std::to_string(it) + " matches a fresh CPU run");
    }
    const auto st1 = brotensor::vulkan::stream_stats(vk());
    VKT_CHECK(st1.launches == st0.launches + 3);
    // Replays record nothing; the only commands are the three uploads and
    // three downloads when they go in-stream (vkCmdUpdateBuffer / a staging
    // copy, e.g. with BROTENSOR_VK_MAPPED=0).
    VKT_CHECK(st1.commands - st0.commands <= 6);

    // Cost per kernel: eager (record + submit + wait) vs replay.
    using clk = std::chrono::steady_clock;
    const int iters = 50;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) { step(x, y, tmp, reps); brotensor::sync(vk()); }
    auto t1 = clk::now();
    for (int i = 0; i < iters; ++i) { g.launch(); brotensor::sync(vk()); }
    auto t2 = clk::now();
    const double kernels = 1.0 + 5.0 * reps;
    std::printf("  %d-command step: eager %.2f us/command, replay %.2f us/command (wall, incl. sync)\n",
                static_cast<int>(kernels),
                std::chrono::duration<double, std::micro>(t1 - t0).count() / iters / kernels,
                std::chrono::duration<double, std::micro>(t2 - t1).count() / iters / kernels);

    g.reset();
    VKT_CHECK(!g.valid());
    // An abandoned capture is discarded and the device keeps working.
    {
        brotensor::VulkanGraphCapture cap(vk());
        brotensor::relu_forward(x, tmp);
    }
    brotensor::relu_forward(x, tmp);
    std::vector<float> want(xv);
    for (float& v : want) v = v > 0 ? v : 0;
    expect_close(tmp.to_host_vector(), want, 0, 0, "ops after an abandoned capture");
}

}  // namespace

}  // namespace vkt

int main(int argc, char** argv) {
    bool expect_default_vulkan = false, bench = false, bench_attn = false, bench_conv = false, bench_quant = false, bench_audio = false, only = false;
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect-default-vulkan") == 0) expect_default_vulkan = true;
        if (std::strcmp(argv[i], "--bench-gemm") == 0) bench = true;
        if (std::strcmp(argv[i], "--bench-attention") == 0) bench_attn = true;
        if (std::strcmp(argv[i], "--bench-conv") == 0) bench_conv = true;
        if (std::strcmp(argv[i], "--bench-quant") == 0) bench_quant = true;
        if (std::strcmp(argv[i], "--bench-audio") == 0) bench_audio = true;
        if (std::strncmp(argv[i], "--only=", 7) == 0) { only = true; filter = argv[i] + 7; }
    }
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::vulkan(0))) {
        std::printf("SKIP: no Vulkan device registered\n");
        return 0;
    }
    try {
        if (bench) {
            vkt::run_gemm_bench();
            return 0;
        }
        if (bench_attn) {
            vkt::run_attention_bench();
            return 0;
        }
        if (bench_conv) {
            vkt::run_conv_bench();
            return 0;
        }
        if (bench_quant) {
            vkt::run_quant_bench();
            return 0;
        }
        if (bench_audio) {
            vkt::run_audio_bench();
            return 0;
        }
        if (only) {   // --only=ops|gemm|norm|attention|conv|spatial|quant|audio|misc|vision|capture: one group
            if (filter == "ops") vkt::run_op_tests();
            if (filter == "gemm") vkt::run_gemm_tests();
            if (filter == "norm") vkt::run_norm_tests();
            if (filter == "attention") vkt::run_attention_tests();
            if (filter == "conv") vkt::run_conv_tests();
            if (filter == "spatial") vkt::run_spatial_tests();
            if (filter == "quant") vkt::run_quant_tests();
            if (filter == "audio") vkt::run_audio_tests();
            if (filter == "misc") vkt::run_misc_tests();
            if (filter == "alias") vkt::test_cuda_alias();
            if (filter == "vision") vkt::run_vision_tests();
            if (filter == "capture") vkt::run_capture_tests();
            std::printf("%s: %d failure(s)\n", vkt::failures() ? "FAILED" : "OK", vkt::failures());
            return vkt::failures() ? 1 : 0;
        }
        vkt::test_registration(expect_default_vulkan);
        if (!expect_default_vulkan) {
            vkt::test_transfers();
            vkt::test_allocator();
            vkt::test_pipeline_guard();
            vkt::test_events_and_batching();
            vkt::test_graph();
            vkt::run_capture_tests();
            vkt::test_cuda_alias();
            vkt::run_op_tests();
            vkt::run_gemm_tests();
            vkt::run_norm_tests();
            vkt::run_attention_tests();
            vkt::run_conv_tests();
            vkt::run_spatial_tests();
            vkt::run_quant_tests();
            vkt::run_audio_tests();
            vkt::run_misc_tests();
            vkt::run_vision_tests();
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL  uncaught exception: %s\n", e.what());
        ++vkt::failures();
    }
    std::printf("%s: %d failure(s)\n", vkt::failures() ? "FAILED" : "OK", vkt::failures());
    return vkt::failures() ? 1 : 0;
}
