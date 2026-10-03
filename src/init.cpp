// brotensor runtime: init(), default-device policy, DeviceScope, sync.
//
// The CPU backend self-registers from a static-init object in
// src/cpu/register.cpp. init() probes CUDA / Metal / Vulkan if the
// corresponding backend was compiled in. When a backend isn't built,
// BROTENSOR_HAS_CUDA / BROTENSOR_HAS_METAL are not defined so the probe
// branches compile out.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <atomic>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(BROTENSOR_HAS_VULKAN)
// Defined in src/vulkan/register.cpp.
extern "C" void brotensor_probe_and_register_vulkan();
#endif

#if defined(BROTENSOR_HAS_CUDA)
// Defined in src/cuda/init.cu.
extern "C" void brotensor_probe_and_register_cuda();
#endif

#if defined(BROTENSOR_HAS_METAL)
// Defined in src/metal/init.mm.
extern "C" void brotensor_probe_and_register_metal();
#endif

namespace brotensor {

namespace {

std::mutex& init_mutex() {
    static std::mutex m;
    return m;
}

std::atomic<bool>& init_done_flag() {
    static std::atomic<bool> f{false};
    return f;
}

// Global default device. Initialised lazily on first read.
std::atomic<Device>& global_default() {
    static std::atomic<Device> d{Device::CPU};
    return d;
}

std::atomic<bool>& global_default_set_flag() {
    static std::atomic<bool> f{false};
    return f;
}

// Thread-local DeviceScope override stack — we only need the *current* value
// since DeviceScope ctor saves the previous on the local stack frame.
thread_local std::optional<Device> tls_scope_override;

// CUDA first, then Metal, then Vulkan (the AMD GPU path, docs/vulkan.md).
Device pick_default_from_available() {
    if (detail::is_registered(Device::CUDA))  return Device::CUDA;
    if (detail::is_registered(Device::Metal)) return Device::Metal;
    if (detail::is_registered(Device::VULKAN)) return Device::VULKAN;
    return Device::CPU;
}

std::optional<Device> parse_env_device() {
    const char* env = std::getenv("BROTENSOR_DEFAULT_DEVICE");
    if (!env) return std::nullopt;
    std::string s(env);
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    }
    if (s == "cpu") return Device::cpu();
    if (s == "cuda" || s == "cuda:0") return Device::cuda(0);
    if (s.rfind("cuda:", 0) == 0) {
        int idx = std::atoi(s.c_str() + 5);
        return Device::cuda(idx);
    }
    for (const char* pfx : {"vulkan", "vk"}) {
        const std::string p(pfx);
        if (s == p) return Device::vulkan(0);
        if (s.rfind(p + ":", 0) == 0) return Device::vulkan(std::atoi(s.c_str() + p.size() + 1));
    }
    if (s == "metal" || s == "metal:0") return Device::metal(0);
    if (s.rfind("metal:", 0) == 0) {
        int idx = std::atoi(s.c_str() + 6);
        return Device::metal(idx);
    }
    return std::nullopt;
}

} // namespace

void init() {
    if (init_done_flag().load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(init_mutex());
    if (init_done_flag().load(std::memory_order_relaxed)) return;

#if defined(BROTENSOR_HAS_CUDA)
    try { brotensor_probe_and_register_cuda(); } catch (...) { /* no CUDA */ }
#endif
#if defined(BROTENSOR_HAS_METAL)
    try { brotensor_probe_and_register_metal(); } catch (...) { /* no Metal */ }
#endif
#if defined(BROTENSOR_HAS_VULKAN)
    try { brotensor_probe_and_register_vulkan(); } catch (...) { /* no Vulkan */ }
#endif

    // Determine default device once.
    if (!global_default_set_flag().load(std::memory_order_relaxed)) {
        if (auto envd = parse_env_device()) {
            const Device want = detail::resolve_device_alias(*envd);
            if (detail::is_registered(want)) {
                global_default().store(want, std::memory_order_relaxed);
            } else {
                global_default().store(pick_default_from_available(),
                                       std::memory_order_relaxed);
            }
        } else {
            global_default().store(pick_default_from_available(),
                                   std::memory_order_relaxed);
        }
        global_default_set_flag().store(true, std::memory_order_relaxed);
    }

    init_done_flag().store(true, std::memory_order_release);
}

void shutdown() {
    detail::cpu::ThreadPool::instance().shutdown();
}

Device default_device() {
    if (tls_scope_override.has_value()) return *tls_scope_override;
    // Lazy default: if init hasn't run, fall back to whatever's registered.
    if (!global_default_set_flag().load(std::memory_order_acquire)) {
        return pick_default_from_available();
    }
    return global_default().load(std::memory_order_acquire);
}

Dtype compute_dtype() {
    return compute_dtype(default_device());
}

Dtype compute_dtype(Device d) {
    return d.is_cpu() ? Dtype::FP32 : Dtype::FP16;
}

void set_default_device(Device d) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) {
        std::string m = "brotensor: set_default_device: backend ";
        m += device_name(d);
        m += " is not available";
        throw std::runtime_error(m);
    }
    global_default().store(d, std::memory_order_release);
    global_default_set_flag().store(true, std::memory_order_release);
}

std::vector<Device> available_devices() {
    std::vector<Device> out;
    if (detail::is_registered(Device::CPU)) {
        out.push_back(Device::CPU);
    }
    if (detail::is_registered(Device::CUDA)) {
        int count = cuda_device_count();
        if (count <= 0) count = 1;
        for (int i = 0; i < count; ++i) {
            out.push_back(Device::cuda(i));
        }
    }
    if (detail::is_registered(Device::Metal)) {
        out.push_back(Device::Metal);
    }
    for (int i = 0; i < vulkan_device_count(); ++i) {
        if (detail::is_registered(Device::vulkan(i))) out.push_back(Device::vulkan(i));
    }
    return out;
}

// Truthful about the hardware: not alias-resolved (detail/dispatch.h).
bool is_available(Device d) {
    return detail::is_registered(d);
}

// ─── DeviceScope ───────────────────────────────────────────────────────────

// Track whether tls_scope_override held a value before this scope was
// pushed. The Device prev_ field on the scope object stores the prior value
// when one existed; the parallel bool stack tracks "had a value".
namespace {
thread_local std::vector<bool> tls_scope_had_prev;
} // namespace

DeviceScope::DeviceScope(Device d) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) {
        std::string m = "brotensor: DeviceScope: backend ";
        m += device_name(d);
        m += " is not available";
        throw std::runtime_error(m);
    }
    if (tls_scope_override.has_value()) {
        prev_ = *tls_scope_override;
        tls_scope_had_prev.push_back(true);
    } else {
        prev_ = Device::CPU; // unused when had_prev=false
        tls_scope_had_prev.push_back(false);
    }
    tls_scope_override = d;
}

DeviceScope::~DeviceScope() {
    bool had = tls_scope_had_prev.empty() ? false : tls_scope_had_prev.back();
    if (!tls_scope_had_prev.empty()) tls_scope_had_prev.pop_back();
    if (had) tls_scope_override = prev_;
    else     tls_scope_override.reset();
}

// ─── sync ──────────────────────────────────────────────────────────────────

void sync(Device d) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) {
        std::string m = "brotensor: sync: backend ";
        m += device_name(d);
        m += " is not available";
        throw std::runtime_error(m);
    }
    detail::alloc_for(d).sync(d.index);
}

void sync_all() {
    for (Device d : available_devices()) {
        detail::alloc_for(d).sync(d.index);
    }
}

// ─── memory ────────────────────────────────────────────────────────────────

bool device_mem_info(Device d, std::size_t& free_bytes,
                     std::size_t& total_bytes) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) return false;
    const auto fn = detail::alloc_for(d).mem_info;
    if (fn == nullptr) return false;
    return fn(&free_bytes, &total_bytes, d.index);
}

bool device_mem_trim(Device d, std::size_t keep_bytes) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) return false;
    const auto fn = detail::alloc_for(d).mem_trim;
    if (fn == nullptr) return false;
    return fn(keep_bytes, d.index);
}

std::string device_product_name(Device d) {
    d = detail::resolve_device_alias(d);
    if (!detail::is_registered(d)) return {};
    const auto fn = detail::alloc_for(d).device_name;
    if (fn == nullptr) return {};
    char buf[256] = {0};
    if (!fn(buf, sizeof(buf), d.index)) return {};
    return std::string(buf);
}

} // namespace brotensor
