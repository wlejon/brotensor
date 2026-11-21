// Tests zero-copy brotensor::interop CPU + Vulkan BDA SPIR-V dispatch.
#include <brotensor/kernel_interop.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#ifdef BROTENSOR_TEST_HAS_BRASS_SPIRV
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/target/spirv_target.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

void test_cpu_interop() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    brotensor::Tensor t = brotensor::Tensor::empty_on(brotensor::Device::CPU, 2, 3);
    void* original_ptr = t.data;

    auto v = brotensor::interop::view(t);
    CHECK(!v.is_gpu());
    CHECK(v.host_ptr == original_ptr);
    CHECK(v.device_address == 0);
    CHECK(v.bytes == 6 * sizeof(float));
    CHECK(v.numel == 6);

    std::string err;
    CHECK(brotensor::interop::copy_bytes_to_tensor(input.data(), input.size() * sizeof(float), t, &err));
    CHECK(t.data == original_ptr);

    std::vector<float> out(6, 0.0f);
    CHECK(brotensor::interop::copy_bytes_from_tensor(t, out.data(), out.size() * sizeof(float), &err));
    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK(approx(out[i], input[i]));
    }

    // Overflow / null errors fail cleanly without crashing.
    CHECK(!brotensor::interop::copy_bytes_to_tensor(nullptr, 4, t, &err));
    CHECK(!brotensor::interop::copy_bytes_to_tensor(input.data(), 1024, t, &err));
    std::printf("brotensor::interop CPU: OK\n");
}

void test_vulkan_interop() {
#ifdef BROTENSOR_HAS_VULKAN
    brotensor::init();
    const brotensor::Device vk = brotensor::Device::vulkan(0);
    if (!brotensor::is_available(vk)) {
        std::printf("brotensor::interop Vulkan: skipped (no Vulkan device)\n");
        return;
    }
    CHECK(brotensor::interop::vulkan_spirv_dispatch_available(vk));

    constexpr int N = 128;
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = float(i) * 0.5f - 10.0f;

    brotensor::Tensor t_in = brotensor::Tensor::empty_on(vk, N, 1);
    brotensor::Tensor t_out = brotensor::Tensor::empty_on(vk, N, 1);
    void* out_orig_data = t_out.data;

    std::string err;
    CHECK(brotensor::interop::copy_bytes_to_tensor(h_in.data(), h_in.size() * sizeof(float), t_in, &err));
    CHECK(t_out.data == out_orig_data);

    auto vin = brotensor::interop::view(t_in);
    auto vout = brotensor::interop::view(t_out);
    CHECK(vin.is_gpu() && vin.is_vulkan() && vin.device_address != 0);
    CHECK(vout.is_gpu() && vout.is_vulkan() && vout.device_address != 0);
    CHECK(vout.host_ptr == nullptr);

#ifdef BROTENSOR_TEST_HAS_BRASS_SPIRV
    // Compile a simple SPIR-V BDA compute kernel: out[i] = in[i] * scale + bias
    brass::Module mod;
    brass::Function* fn = mod.create_function("affine_bda", brass::Type::void_type());
    fn->set_param_types(
        {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::f32(), brass::Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    auto& b = kb.builder();
    brass::BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    brass::Value* p_in = b.add_param(brass::Type::ptr());
    brass::Value* p_out = b.add_param(brass::Type::ptr());
    brass::Value* p_n = b.add_param(brass::Type::i32());
    brass::Value* p_scale = b.add_param(brass::Type::f32());
    brass::Value* p_bias = b.add_param(brass::Type::f32());

    brass::BasicBlock* body = b.append_block("body");
    brass::BasicBlock* done = b.append_block("done");
    b.position_at_end(entry);
    brass::Value* gid = kb.global_tid_x();
    b.build_br_if(b.build_slt(gid, p_n), body, done);
    b.position_at_end(body);
    brass::Value* idx64 = b.build_sext_i64(gid);
    brass::Value* x = b.build_load_indexed(brass::Type::f32(), p_in, idx64, 4);
    brass::Value* y = b.build_add(b.build_mul(x, p_scale), p_bias);
    b.build_store_indexed(brass::Type::f32(), p_out, idx64, 4, 0, y);
    b.build_br(done);
    b.position_at_end(done);
    b.build_ret_void();

    brass::target::SpirvOptions spv_opts;
    spv_opts.stage = brass::target::SpirvStage::Compute;
    spv_opts.local_size_x = 64;
    spv_opts.local_size_spec_constants = true;
    brass::target::SpirvKernel sk = brass::target::SpirvTarget::compile(*fn, spv_opts);
    CHECK(!sk.words.empty());
    CHECK(sk.params.size() == 5);

    std::vector<std::uint8_t> push(sk.push_constant_bytes, 0);
    const std::int32_t n_i32 = N;
    const float scale = 2.5f;
    const float bias = -1.25f;
    std::memcpy(push.data() + sk.params[0].offset, &vin.device_address, 8);
    std::memcpy(push.data() + sk.params[1].offset, &vout.device_address, 8);
    std::memcpy(push.data() + sk.params[2].offset, &n_i32, 4);
    std::memcpy(push.data() + sk.params[3].offset, &scale, 4);
    std::memcpy(push.data() + sk.params[4].offset, &bias, 4);

    brotensor::interop::SpirvDispatchDesc desc;
    desc.words = sk.words.data();
    desc.word_count = sk.words.size();
    desc.entry_name = sk.entry.c_str();
    desc.local_size[0] = 64;
    desc.groups[0] = (N + 63u) / 64u;
    desc.push_data = push.data();
    desc.push_bytes = static_cast<std::uint32_t>(push.size());
    desc.local_size_spec_constants = true;

    CHECK(brotensor::interop::dispatch_spirv(vk, desc, &err));
    brotensor::sync(vk);

    // Zero-copy guarantee: t_out kept its exact same device buffer pointer.
    CHECK(t_out.data == out_orig_data);

    std::vector<float> h_out(N, 0.0f);
    CHECK(brotensor::interop::copy_bytes_from_tensor(t_out, h_out.data(), h_out.size() * sizeof(float), &err));
    for (int i = 0; i < N; ++i) {
        CHECK(approx(h_out[i], h_in[i] * scale + bias));
    }
    std::printf("brotensor::interop Vulkan SPIR-V zero-copy dispatch: OK\n");
#else
    std::vector<float> h_out(N, 0.0f);
    CHECK(brotensor::interop::copy_bytes_from_tensor(t_in, h_out.data(), h_out.size() * sizeof(float), &err));
    for (int i = 0; i < N; ++i) CHECK(approx(h_out[i], h_in[i]));
    std::printf("brotensor::interop Vulkan H2D/D2H: OK\n");
#endif
#else
    std::printf("brotensor::interop Vulkan: not built in\n");
#endif
}

} // namespace

int main() {
    test_cpu_interop();
    test_vulkan_interop();
    brotensor::shutdown();
    return 0;
}
