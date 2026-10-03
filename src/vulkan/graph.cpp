// VulkanGraph / VulkanGraphCapture (record once, replay many) and events.
// Public contract: include/brotensor/vulkan.h. Mechanics: the device's Stream
// records into a dedicated command buffer while capturing
// (Stream::begin_capture), the allocator hands every free during the capture
// to the graph instead of releasing it (Allocator::set_hold), and launch()
// resubmits the finished command buffer.

#include "detail/device.h"

#include <brotensor/vulkan.h>
#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor {

namespace {

int vulkan_index(Device d, const char* who) {
    d = detail::resolve_device_alias(d);
    if (!d.is_vulkan()) {
        throw std::runtime_error(std::string("brotensor: ") + who + ": " + device_name(d) +
                                 " is not a Vulkan device");
    }
    if (!detail::is_registered(d)) {
        throw std::runtime_error(std::string("brotensor: ") + who + ": " + device_name(d) +
                                 " is not available");
    }
    return d.index;
}

}  // namespace

// ─── events ────────────────────────────────────────────────────────────────

namespace vulkan {

Event Event::record(Device d) {
    Event e;
    e.device_ = Device::vulkan(vulkan_index(d, "vulkan::Event::record"));
    e.serial_ = detail::vulkan::device(e.device_.index).stream().work_serial();
    return e;
}

bool Event::query() const {
    if (!valid() || serial_ == 0) return true;
    auto& st = detail::vulkan::device(device_.index).stream();
    if (st.completed() >= serial_) return true;
    if (!st.capturing()) st.flush();
    return st.completed() >= serial_;
}

void Event::wait() const {
    if (!valid() || serial_ == 0) return;
    detail::vulkan::device(device_.index).stream().wait(serial_);
}

}  // namespace vulkan

// ─── VulkanGraph ───────────────────────────────────────────────────────────

struct VulkanGraph::Impl {
    int dev = 0;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::vector<void*> held;          // memory freed during the capture
    std::uint64_t last_launch = 0;

    ~Impl() {
        auto& d = detail::vulkan::device(dev);
        // A launch may still be running: wait before the buffer and the memory
        // it uses are released.
        if (last_launch) d.stream().wait(last_launch);
        d.stream().free_command_buffer(cb);
        for (void* p : held) d.allocator().free(p);
    }
};

VulkanGraph::VulkanGraph() = default;
VulkanGraph::~VulkanGraph() = default;
VulkanGraph::VulkanGraph(VulkanGraph&&) noexcept = default;
VulkanGraph& VulkanGraph::operator=(VulkanGraph&&) noexcept = default;

bool VulkanGraph::valid() const { return impl_ != nullptr; }

void VulkanGraph::launch() {
    if (!impl_) throw std::runtime_error("brotensor: VulkanGraph::launch: empty graph");
    impl_->last_launch = detail::vulkan::device(impl_->dev).stream().launch(impl_->cb);
}

void VulkanGraph::reset() { impl_.reset(); }

Device VulkanGraph::device() const {
    return impl_ ? Device::vulkan(impl_->dev) : Device::cpu();
}

// ─── VulkanGraphCapture ────────────────────────────────────────────────────

struct VulkanGraphCapture::Impl {
    int dev = 0;
    bool active = false;
    std::vector<void*> held;
};

VulkanGraphCapture::VulkanGraphCapture(Device d) : impl_(std::make_unique<Impl>()) {
    impl_->dev = vulkan_index(d, "VulkanGraphCapture");
    auto& ctx = detail::vulkan::device(impl_->dev);
    if (ctx.allocator().hold()) {
        throw std::runtime_error("brotensor: VulkanGraphCapture: a capture is already recording on " +
                                 to_string(Device::vulkan(impl_->dev)));
    }
    ctx.stream().begin_capture();
    ctx.allocator().set_hold(&impl_->held);
    impl_->active = true;
}

VulkanGraphCapture::~VulkanGraphCapture() {
    if (!impl_ || !impl_->active) return;
    // Abandoned (exception unwinding the captured region): discard the
    // recording and release what was freed during it the ordinary way.
    auto& ctx = detail::vulkan::device(impl_->dev);
    ctx.allocator().set_hold(nullptr);
    try {
        ctx.stream().abort_capture();
        for (void* p : impl_->held) ctx.allocator().free(p);
    } catch (...) {
    }
}

VulkanGraph VulkanGraphCapture::finish() {
    if (!impl_->active) throw std::runtime_error("brotensor: VulkanGraphCapture::finish called twice");
    auto& ctx = detail::vulkan::device(impl_->dev);
    ctx.allocator().set_hold(nullptr);
    impl_->active = false;
    VulkanGraph g;
    g.impl_ = std::make_unique<VulkanGraph::Impl>();
    g.impl_->dev = impl_->dev;
    g.impl_->held = std::move(impl_->held);
    g.impl_->cb = ctx.stream().end_capture();
    return g;
}

}  // namespace brotensor
