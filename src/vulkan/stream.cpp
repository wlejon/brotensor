// Per-device command recording, batching and submission. Design notes:
// detail/stream.h.

#include "detail/device.h"
#include "detail/stream.h"
#include "detail/vk_check.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

namespace {

constexpr VkPipelineStageFlags kWorkStages =
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkAccessFlags kWorkWrites = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags kWorkAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

}  // namespace

Stream::Stream(DeviceCtx& dev) : dev_(dev) {
    const DeviceFns& f = dev.fn();
    VkDevice d = dev.device();

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev.info().queue_family;
    BT_VK_CHECK(f.vkCreateCommandPool(d, &cpi, nullptr, &pool_));

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    for (Slot& s : slots_) BT_VK_CHECK(f.vkAllocateCommandBuffers(d, &ai, &s.cb));

    VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    st.initialValue = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &st;
    BT_VK_CHECK(f.vkCreateSemaphore(d, &sci, nullptr, &timeline_));

    if (const char* b = std::getenv("BROTENSOR_VK_BATCH")) {
        const long v = std::atol(b);
        if (v > 0) batch_limit_ = static_cast<std::uint32_t>(v);
    }
    stats_.batch_limit = batch_limit_;
}

Stream::~Stream() {
    const DeviceFns& f = dev_.fn();
    VkDevice d = dev_.device();
    if (open_slot_ >= 0) {
        f.vkEndCommandBuffer(slots_[open_slot_].cb);
        open_slot_ = -1;
    }
    f.vkDeviceWaitIdle(d);
    f.vkDestroySemaphore(d, timeline_, nullptr);
    f.vkDestroyCommandPool(d, pool_, nullptr);
}

void Stream::barrier(VkCommandBuffer cb, bool to_host) const {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = kWorkWrites;
    mb.dstAccessMask = to_host ? VkAccessFlags(VK_ACCESS_HOST_READ_BIT) : kWorkAccess;
    const VkPipelineStageFlags dst =
        to_host ? VkPipelineStageFlags(VK_PIPELINE_STAGE_HOST_BIT) : kWorkStages;
    dev_.fn().vkCmdPipelineBarrier(cb, kWorkStages, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Stream::throw_if_capturing(const char* what) const {
    if (capturing_.load(std::memory_order_acquire)) {
        throw std::runtime_error(std::string("brotensor: vulkan: ") + what +
                                 " is not allowed while a graph capture is recording "
                                 "(it needs the host to wait for the GPU)");
    }
}

VkCommandBuffer Stream::open_locked() {
    if (capture_cb_) return capture_cb_;
    if (open_slot_ >= 0) return slots_[open_slot_].cb;

    // Round-robin over the slots; a slot is reusable once its last submission
    // has completed, which bounds the batches in flight to kSlots.
    const int slot = static_cast<int>(next_serial_ % kSlots);
    Slot& s = slots_[slot];
    if (s.serial != 0) wait_value(s.serial);
    const DeviceFns& f = dev_.fn();
    BT_VK_CHECK(f.vkResetCommandBuffer(s.cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    BT_VK_CHECK(f.vkBeginCommandBuffer(s.cb, &bi));
    // Order this batch after everything submitted before it on the queue.
    barrier(s.cb, /*to_host=*/false);
    open_slot_ = slot;
    open_serial_ = next_serial_;
    work_serial_.store(open_serial_, std::memory_order_release);
    ops_in_batch_ = 0;
    need_barrier_ = false;
    bound_ = VK_NULL_HANDLE;
    return s.cb;
}

void Stream::before_command_locked(VkCommandBuffer cb) {
    if (need_barrier_) barrier(cb, /*to_host=*/false);
    need_barrier_ = true;
}

void Stream::after_command_locked() {
    ++stats_.commands;
    ++ops_in_batch_;
    if (ops_in_batch_ < batch_limit_) return;
    if (!capture_cb_) {
        submit_locked();
        return;
    }
    // A capture segment is full: close it (its results visible to the next
    // segment through that one's leading barrier) and record on into a new one.
    BT_VK_CHECK(dev_.fn().vkEndCommandBuffer(capture_cb_));
    capture_done_.push_back(capture_cb_);
    begin_capture_segment_locked();
}

void Stream::submit_cb_locked(VkCommandBuffer cb, std::uint64_t serial) {
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &serial;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &timeline_;
    BT_VK_CHECK(dev_.fn().vkQueueSubmit(dev_.queue(), 1, &si, VK_NULL_HANDLE));
    submitted_ = serial;
    ++stats_.submits;
}

void Stream::submit_locked() {
    if (open_slot_ < 0) return;
    Slot& s = slots_[open_slot_];
    barrier(s.cb, /*to_host=*/true);   // results visible to host reads after the wait
    BT_VK_CHECK(dev_.fn().vkEndCommandBuffer(s.cb));
    s.serial = open_serial_;
    open_slot_ = -1;
    submit_cb_locked(s.cb, open_serial_);
    ++next_serial_;
}

// ─── commands ──────────────────────────────────────────────────────────────

void Stream::dispatch(VkPipeline pipe, const void* push, std::uint32_t push_bytes,
                      std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    VkCommandBuffer cb = open_locked();
    before_command_locked(cb);
    const DeviceFns& f = dev_.fn();
    if (bound_ != pipe) {
        f.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        bound_ = pipe;
    }
    if (push_bytes) {
        f.vkCmdPushConstants(cb, dev_.pipelines().layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             push_bytes, push);
    }
    f.vkCmdDispatch(cb, gx, gy, gz);
    after_command_locked();
}

void Stream::copy(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off,
                  VkDeviceSize bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    VkCommandBuffer cb = open_locked();
    before_command_locked(cb);
    VkBufferCopy region{src_off, dst_off, bytes};
    dev_.fn().vkCmdCopyBuffer(cb, src, dst, 1, &region);
    after_command_locked();
}

void Stream::fill(VkBuffer dst, VkDeviceSize off, VkDeviceSize bytes, std::uint32_t value) {
    if (bytes == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    VkCommandBuffer cb = open_locked();
    before_command_locked(cb);
    dev_.fn().vkCmdFillBuffer(cb, dst, off, bytes, value);
    after_command_locked();
}

void Stream::update(VkBuffer dst, VkDeviceSize off, VkDeviceSize bytes, const void* data) {
    if (bytes == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    VkCommandBuffer cb = open_locked();
    before_command_locked(cb);
    dev_.fn().vkCmdUpdateBuffer(cb, dst, off, bytes, data);
    after_command_locked();
}

// ─── submission / waiting ──────────────────────────────────────────────────

void Stream::flush() {
    std::lock_guard<std::mutex> lk(mu_);
    submit_locked();
}

void Stream::wait_value(std::uint64_t value) const {
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline_;
    wi.pValues = &value;
    BT_VK_CHECK(dev_.fn().vkWaitSemaphores(dev_.device(), &wi, UINT64_MAX));
}

void Stream::wait(std::uint64_t serial) {
    if (serial == 0) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (serial > submitted_) {
            if (open_slot_ >= 0 && serial == open_serial_) {
                throw_if_capturing("waiting on recorded work");
                submit_locked();
            } else if (serial > submitted_) {
                return;  // nothing recorded under this serial yet
            }
        }
    }
    wait_value(serial);
    dev_.allocator().collect(completed());
}

void Stream::sync() {
    throw_if_capturing("sync");
    std::uint64_t target = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        submit_locked();
        target = submitted_;
    }
    if (target) wait_value(target);
    dev_.allocator().collect(completed());
}

std::uint64_t Stream::completed() const {
    std::uint64_t v = 0;
    BT_VK_CHECK(dev_.fn().vkGetSemaphoreCounterValue(dev_.device(), timeline_, &v));
    return v;
}

bool Stream::idle() const {
    std::uint64_t submitted = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (open_slot_ >= 0) return false;
        submitted = submitted_;
    }
    return completed() >= submitted;
}

// ─── capture / replay ──────────────────────────────────────────────────────

VkCommandBuffer Stream::begin_capture_segment_locked() {
    const DeviceFns& f = dev_.fn();
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    BT_VK_CHECK(f.vkAllocateCommandBuffers(dev_.device(), &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;   // launch() may overlap launches
    BT_VK_CHECK(f.vkBeginCommandBuffer(cb, &bi));
    // Ordered after everything submitted before it (the previous segment
    // included), as an eager batch is.
    barrier(cb, /*to_host=*/false);
    capture_cb_ = cb;
    ops_in_batch_ = 0;
    need_barrier_ = false;
    bound_ = VK_NULL_HANDLE;
    return cb;
}

void Stream::begin_capture() {
    std::lock_guard<std::mutex> lk(mu_);
    if (capture_cb_) throw std::runtime_error("brotensor: vulkan: a graph capture is already recording on this device");
    submit_locked();   // work recorded before the capture runs before it
    capture_done_.clear();
    begin_capture_segment_locked();
    capturing_.store(true, std::memory_order_release);
}

std::vector<VkCommandBuffer> Stream::end_capture() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!capture_cb_) throw std::runtime_error("brotensor: vulkan: no graph capture is recording");
    VkCommandBuffer cb = capture_cb_;
    barrier(cb, /*to_host=*/true);
    capture_cb_ = VK_NULL_HANDLE;
    capturing_.store(false, std::memory_order_release);
    bound_ = VK_NULL_HANDLE;
    need_barrier_ = false;
    ops_in_batch_ = 0;
    BT_VK_CHECK(dev_.fn().vkEndCommandBuffer(cb));
    std::vector<VkCommandBuffer> out = std::move(capture_done_);
    capture_done_.clear();
    out.push_back(cb);
    return out;
}

void Stream::abort_capture() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!capture_cb_) return;
    VkCommandBuffer cb = capture_cb_;
    capture_cb_ = VK_NULL_HANDLE;
    capturing_.store(false, std::memory_order_release);
    bound_ = VK_NULL_HANDLE;
    need_barrier_ = false;
    ops_in_batch_ = 0;
    dev_.fn().vkEndCommandBuffer(cb);
    capture_done_.push_back(cb);
    dev_.fn().vkFreeCommandBuffers(dev_.device(), pool_, static_cast<std::uint32_t>(capture_done_.size()),
                                   capture_done_.data());
    capture_done_.clear();
}

std::uint64_t Stream::launch(const std::vector<VkCommandBuffer>& cbs) {
    std::lock_guard<std::mutex> lk(mu_);
    if (capture_cb_) throw std::runtime_error("brotensor: vulkan: cannot launch a graph while a capture is recording");
    if (cbs.empty()) throw std::runtime_error("brotensor: vulkan: launch of an empty graph");
    submit_locked();
    // One submission per segment, each signalling its own serial, so no
    // single kernel-driver job holds the whole graph.
    std::uint64_t serial = 0;
    for (VkCommandBuffer cb : cbs) {
        serial = next_serial_++;
        work_serial_.store(serial, std::memory_order_release);
        submit_cb_locked(cb, serial);
    }
    ++stats_.launches;
    return serial;
}

void Stream::free_command_buffers(const std::vector<VkCommandBuffer>& cbs) {
    if (cbs.empty()) return;
    std::lock_guard<std::mutex> lk(mu_);
    dev_.fn().vkFreeCommandBuffers(dev_.device(), pool_, static_cast<std::uint32_t>(cbs.size()), cbs.data());
}

StreamStats Stream::stats() const {
    StreamStats s;
    {
        std::lock_guard<std::mutex> lk(mu_);
        s = stats_;
        s.submitted = submitted_;
    }
    s.completed = completed();
    return s;
}

}  // namespace brotensor::detail::vulkan
