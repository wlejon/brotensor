// Vulkan backend registration, called from brotensor::init() (src/init.cpp)
// when BROTENSOR_HAS_VULKAN is defined. Enumerates eligible devices (no
// VkDevice is created until a device is first used), fills the OpsVTable from
// each op file's fill function, and registers it with the Vulkan AllocVTable.
// Also home of the small public introspection API in <brotensor/vulkan.h>.

#include "detail/device.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

const ::brotensor::detail::AllocVTable& vulkan_alloc_table();
void fill_vulkan_vtable_elementwise(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_copy(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_reduce(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_linear(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_norm(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_rope(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_glu(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_attention(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_attention_proj(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_attention_bwd(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_topk(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_xent(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_conv(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_gnorm(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_spatial(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_diffusion(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_quant(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_quant_attention(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_audio(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_spectral(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_sampling(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_misc(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_delta(::brotensor::detail::OpsVTable& v);
void fill_vulkan_vtable_vision(::brotensor::detail::OpsVTable& v);
void register_vulkan_graph_backend();

namespace {

bool ends_with(const char* s, const char* suffix) {
    const std::size_t a = std::strlen(s), b = std::strlen(suffix);
    return a >= b && std::strcmp(s + a - b, suffix) == 0;
}

// dt_variant() offsets from a <base>_f32 id; every family in kDtypeFamilies
// must be listed in shaders.cmake as _f32, _f16, _bf16 in that order.
void check_shader_families() {
    for (ShaderId f32 : kDtypeFamilies) {
        const auto i = static_cast<std::uint32_t>(f32);
        const char* name = shader_blob(f32).name;
        const bool ok = ends_with(name, "_f32") && i + 2 < static_cast<std::uint32_t>(ShaderId::kCount);
        const std::string base = ok ? std::string(name, std::strlen(name) - 4) : std::string(name);
        if (!ok || shader_blob(static_cast<ShaderId>(i + 1)).name != base + "_f16" ||
            shader_blob(static_cast<ShaderId>(i + 2)).name != base + "_bf16") {
            throw std::logic_error("brotensor: vulkan: shader family " + base +
                                   " is not listed as _f32, _f16, _bf16 in shaders.cmake");
        }
    }
}

}  // namespace

}  // namespace brotensor::detail::vulkan

namespace {

bool vk_verbose() {
    const char* v = std::getenv("BROTENSOR_VK_VERBOSE");
    return v && *v && *v != '0';
}

void probe_and_register() {
    namespace dv = ::brotensor::detail::vulkan;
    const dv::Instance& inst = dv::Instance::get();
    if (inst.count() == 0) {
        if (vk_verbose()) {
            std::fprintf(stderr, "brotensor: Vulkan backend not registered: %s\n", inst.why().c_str());
        }
        return;
    }
    dv::check_shader_families();
    ::brotensor::detail::set_vulkan_device_count(inst.count());

    ::brotensor::detail::OpsVTable ops{};   // every slot starts null
    dv::fill_vulkan_vtable_elementwise(ops);
    dv::fill_vulkan_vtable_copy(ops);
    dv::fill_vulkan_vtable_reduce(ops);
    dv::fill_vulkan_vtable_linear(ops);
    dv::fill_vulkan_vtable_norm(ops);
    dv::fill_vulkan_vtable_rope(ops);
    dv::fill_vulkan_vtable_glu(ops);
    dv::fill_vulkan_vtable_attention(ops);
    dv::fill_vulkan_vtable_attention_proj(ops);
    dv::fill_vulkan_vtable_attention_bwd(ops);
    dv::fill_vulkan_vtable_topk(ops);
    dv::fill_vulkan_vtable_xent(ops);
    dv::fill_vulkan_vtable_conv(ops);
    dv::fill_vulkan_vtable_gnorm(ops);
    dv::fill_vulkan_vtable_spatial(ops);
    dv::fill_vulkan_vtable_diffusion(ops);
    dv::fill_vulkan_vtable_quant(ops);
    dv::fill_vulkan_vtable_quant_attention(ops);
    dv::fill_vulkan_vtable_audio(ops);
    dv::fill_vulkan_vtable_spectral(ops);
    dv::fill_vulkan_vtable_sampling(ops);
    dv::fill_vulkan_vtable_misc(ops);
    dv::fill_vulkan_vtable_delta(ops);
    dv::fill_vulkan_vtable_vision(ops);
    ::brotensor::detail::register_backend(::brotensor::DeviceType::VULKAN, ops,
                                          dv::vulkan_alloc_table());
    dv::register_vulkan_graph_backend();
    if (vk_verbose()) {
        for (int i = 0; i < inst.count(); ++i) {
            std::fprintf(stderr, "brotensor: vulkan:%d = %s (%s)\n", i, inst.phys(i).name.c_str(),
                         inst.phys(i).driver.c_str());
        }
    }
}

}  // namespace

// BROTENSOR_VK_VERBOSE=1 reports why the backend did not register (init()
// swallows the exception so a GPU-less host still runs).
extern "C" void brotensor_probe_and_register_vulkan() {
    try {
        probe_and_register();
    } catch (const std::exception& e) {
        if (vk_verbose()) std::fprintf(stderr, "brotensor: Vulkan backend not registered: %s\n", e.what());
        throw;
    }
}

// ─── public introspection (<brotensor/vulkan.h>) ───────────────────────────

namespace brotensor::vulkan {

namespace {

detail::vulkan::DeviceCtx& ctx_for(Device d, const char* who) {
    d = detail::resolve_device_alias(d);
    if (!d.is_vulkan() || !detail::is_registered(d)) {
        throw std::runtime_error(std::string("brotensor: ") + who + ": " + device_name(d) +
                                 " is not an available Vulkan device");
    }
    return detail::vulkan::device(d.index);
}

}  // namespace

void flush(Device d) { ctx_for(d, "vulkan::flush").stream().flush(); }

MemoryStats memory_stats(Device d) {
    const auto s = ctx_for(d, "vulkan::memory_stats").allocator().stats();
    MemoryStats m;
    m.blocks = s.blocks;
    m.block_bytes = s.block_bytes;
    m.dedicated_blocks = s.dedicated_blocks;
    m.dedicated_bytes = s.dedicated_bytes;
    m.used_bytes = s.used_bytes;
    m.live_allocations = s.live_allocations;
    m.deferred_frees = s.deferred_frees;
    m.max_buffer_bytes = s.max_buffer_bytes;
    m.sub_alloc_max = static_cast<std::size_t>(detail::vulkan::Allocator::kSubAllocMax);
    m.host_mapped = s.host_mapped;
    return m;
}

StreamStats stream_stats(Device d) {
    auto& c = ctx_for(d, "vulkan::stream_stats");
    const auto s = c.stream().stats();
    StreamStats out;
    out.submits = s.submits;
    out.commands = s.commands;
    out.launches = s.launches;
    out.batch_limit = s.batch_limit;
    out.pipelines = c.pipelines().size();
    return out;
}

DeviceInfo device_info(Device d) {
    d = detail::resolve_device_alias(d);
    const auto& inst = detail::vulkan::Instance::get();
    if (!d.is_vulkan() || d.index < 0 || d.index >= inst.count()) {
        throw std::runtime_error("brotensor: vulkan::device_info: " + to_string(d) +
                                 " is not an available Vulkan device");
    }
    const auto& p = inst.phys(d.index);
    DeviceInfo info;
    info.name = p.name;
    info.driver = p.driver;
    info.subgroup_size = p.subgroup_size;
    info.min_subgroup_size = p.min_subgroup;
    info.max_subgroup_size = p.max_subgroup;
    info.max_shared_bytes = p.props.limits.maxComputeSharedMemorySize;
    info.max_push_constant_bytes = p.props.limits.maxPushConstantsSize;
    info.graphics_queue = p.graphics_queue;
    info.cooperative_matrix = p.cooperative_matrix;
    info.subgroup_size_control = p.subgroup_size_control;
    info.shader_float64 = p.shader_float64;
    return info;
}

}  // namespace brotensor::vulkan
