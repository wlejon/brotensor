#pragma once

// Per-device command recording: the Vulkan backend's equivalent of a HIP
// stream.
//
// Every op on a device records into the device's one Stream. Commands go into
// an open command buffer (a "batch"), which is submitted to the graphics queue
// when it holds `batch_limit` commands, when the host needs a result
// (sync / a download / a host write to device memory), or on flush(). Each
// submission signals the device's timeline semaphore with a fresh serial, so
// "has the GPU finished the work recorded up to here" is one integer compare
// (completed() >= serial) and an event is just a serial.
//
// Ordering. A full memory barrier (compute + transfer, write -> read/write)
// separates consecutive commands and starts every command buffer, so the work
// is executed exactly in recording order across batches and graph launches,
// like a single in-order HIP stream. Later chunks may replace the blanket
// barrier with hazard tracking; nothing outside this file assumes more than
// "in order".
//
// Queue. The graphics (graphics + compute) family is used, not the async
// compute family: on RADV a dispatch chain that changes pipelines costs
// ~0.63 us per kernel on the graphics queue and ~1.8 us on the compute queue
// (vk-spike RESULTS.md).
//
// Threads. One mutex per device serialises recording, submission and the
// command pool. The serial counters are atomics so the allocator can tag a
// free without taking the stream lock (lock order is stream -> allocator,
// never the reverse).
//
// Capture (graph.cpp). While a capture is active, commands go into the
// capture's own command buffer instead and are never submitted; operations
// that need the host to wait (sync, downloads, uploads larger than an inline
// update) throw. The finished command buffers are submitted as-is by
// launch(), as many times as the caller likes: a pre-recorded command buffer
// replayed on the graphics queue costs ~0.74 us per dependent kernel, 2.5x
// cheaper than hipGraph replay on the same GPU. A capture is cut into
// segments of `batch_limit` commands, the same bound as an eager batch, and
// launch() submits each segment as its own queue submission: the amdgpu
// kernel driver resets the GPU when one submission runs past ~10 s, and a
// whole captured diffusion step (TripoSplat's flow model: two 3.6 s forwards
// at 8 steps) is past that as one command buffer.

#include "vk_fns.h"

#include <array>
#include <atomic>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace brotensor::detail::vulkan {

class DeviceCtx;

struct StreamStats {
    std::uint64_t submits   = 0;   // queue submissions (batches + launches)
    std::uint64_t commands  = 0;   // commands recorded (dispatch/copy/fill/update)
    std::uint64_t launches  = 0;   // graph launches
    std::uint64_t completed = 0;   // timeline value
    std::uint64_t submitted = 0;   // last serial handed to the queue
    std::uint32_t batch_limit = 0;
};

class Stream {
public:
    explicit Stream(DeviceCtx& dev);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // ── Commands (each is ordered after everything recorded before it) ──
    void dispatch(VkPipeline pipe, const void* push, std::uint32_t push_bytes,
                  std::uint32_t gx, std::uint32_t gy, std::uint32_t gz);
    void copy(VkBuffer src, VkDeviceSize src_off, VkBuffer dst,
              VkDeviceSize dst_off, VkDeviceSize bytes);
    // vkCmdFillBuffer: offset and size multiples of 4.
    void fill(VkBuffer dst, VkDeviceSize off, VkDeviceSize bytes, std::uint32_t value);
    // vkCmdUpdateBuffer: the data is copied into the command buffer now, so
    // the caller's memory is free on return. <= 65536 bytes, 4-byte multiples.
    void update(VkBuffer dst, VkDeviceSize off, VkDeviceSize bytes, const void* data);

    // ── Submission / waiting ──
    void flush();                       // submit the open batch, don't wait
    void sync();                        // flush + wait for everything
    void wait(std::uint64_t serial);    // flushes if `serial` is still open
    std::uint64_t completed() const;    // timeline value (GPU progress)
    // Serial whose completion covers every command recorded so far (the open
    // batch's serial while one is open). Lock free.
    std::uint64_t work_serial() const { return work_serial_.load(std::memory_order_acquire); }
    // True when nothing is recorded-but-unsubmitted or still running.
    bool idle() const;

    // ── Capture / replay (graph.cpp) ──
    void begin_capture();               // throws if already capturing
    std::vector<VkCommandBuffer> end_capture();   // ends recording, returns the segments
    void abort_capture();
    bool capturing() const { return capturing_.load(std::memory_order_acquire); }
    // Submits the segments in order, one submission each; returns the serial
    // the last one signals.
    std::uint64_t launch(const std::vector<VkCommandBuffer>& cbs);
    void free_command_buffers(const std::vector<VkCommandBuffer>& cbs);

    StreamStats stats() const;

private:
    VkCommandBuffer open_locked();      // the buffer commands record into
    void before_command_locked(VkCommandBuffer cb);
    void after_command_locked();
    void submit_locked();
    void submit_cb_locked(VkCommandBuffer cb, std::uint64_t serial);
    void wait_value(std::uint64_t value) const;
    void barrier(VkCommandBuffer cb, bool to_host) const;
    void throw_if_capturing(const char* what) const;
    VkCommandBuffer begin_capture_segment_locked();

    static constexpr int kSlots = 4;
    struct Slot {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        std::uint64_t serial = 0;      // serial of its last submission
    };

    DeviceCtx& dev_;
    mutable std::mutex mu_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::array<Slot, kSlots> slots_{};
    int open_slot_ = -1;                // slot being recorded, -1 if none
    std::uint64_t open_serial_ = 0;     // serial the open batch will signal
    std::uint64_t next_serial_ = 1;
    std::uint64_t submitted_ = 0;
    std::atomic<std::uint64_t> work_serial_{0};
    std::uint32_t ops_in_batch_ = 0;
    std::uint32_t batch_limit_ = 128;
    bool need_barrier_ = false;
    VkPipeline bound_ = VK_NULL_HANDLE;

    std::atomic<bool> capturing_{false};
    VkCommandBuffer capture_cb_ = VK_NULL_HANDLE;    // segment being recorded
    std::vector<VkCommandBuffer> capture_done_;      // finished segments

    StreamStats stats_{};
};

}  // namespace brotensor::detail::vulkan
