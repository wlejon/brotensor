#pragma once

// Vulkan-backend surface beyond the device-neutral API: batching control,
// events, record / replay graphs, and introspection. Device-neutral code never
// needs this header; ops on Device::vulkan(i) tensors work through ops.h like
// on any backend. Gate uses on BROTENSOR_HAS_VULKAN. Design: docs/vulkan.md.
//
// Batching. Ops on a Vulkan device are recorded into a command buffer that is
// submitted when it is full, when the host needs a result (sync(), a download,
// a host write), or on flush(). Nothing runs on the GPU before submission, so
// call flush() when the host has other work to do while the GPU computes.
//
// Graphs. VulkanGraphCapture / VulkanGraph mirror CudaGraphCapture / CudaGraph
// (cuda_graph.h) and have the same contract, and are built on the cheapest
// thing Vulkan offers: the captured ops are recorded once into a command
// buffer that launch() resubmits as-is (~0.74 us per dependent kernel on a
// Radeon 8060S, 2.5x cheaper than a hipGraph replay):
//
//   step();                                     // warm-up: outputs allocated
//   brotensor::sync(dev);
//   brotensor::VulkanGraph g;
//   {
//       brotensor::VulkanGraphCapture cap(dev); // ops now record into the graph
//       step();                                 // same tensors, same shapes
//       g = cap.finish();
//   }
//   for (int t = 0; t < T; ++t) {
//       write_new_inputs_in_place();            // copy_from_host_raw into inputs
//       g.launch();
//       brotensor::sync(dev);                   // before reading outputs
//   }
//
// Contract (as for CUDA graphs): kernels replay against the addresses and the
// scalar arguments they were captured with, so feed inputs by writing into the
// captured tensors in place and read outputs from theirs. While a capture is
// recording, nothing may make the host wait for the GPU: sync(), downloads and
// uploads of the capturing device throw. Memory freed while capturing is held
// by the graph until it is destroyed, so a replay never touches memory that
// has been handed to someone else; allocations made while capturing come from
// the ordinary pool. A capture covers one device and records every op issued
// on it from any thread.

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace brotensor {

namespace vulkan {

// Submit the ops recorded so far on `d` without waiting for them.
void flush(Device d);

// A point in a device's op sequence. record() marks everything enqueued on the
// device so far; the event is complete once all of it has run.
class Event {
public:
    Event() = default;
    static Event record(Device d);

    bool valid() const { return device_.is_vulkan(); }
    // True once complete. Submits the marked work if it is still unsubmitted,
    // so polling an event always makes progress.
    bool query() const;
    void wait() const;

    Device device() const { return device_; }
    std::uint64_t serial() const { return serial_; }

private:
    Device device_ = Device::cpu();
    std::uint64_t serial_ = 0;
};

struct MemoryStats {
    std::size_t blocks = 0;            // sub-allocation blocks held
    std::size_t block_bytes = 0;
    std::size_t dedicated_blocks = 0;  // allocations above the sub-allocation size
    std::size_t dedicated_bytes = 0;
    std::size_t used_bytes = 0;        // handed out (256-byte rounded)
    std::size_t live_allocations = 0;
    std::size_t deferred_frees = 0;    // dedicated blocks waiting for the GPU
    std::size_t max_buffer_bytes = 0;  // largest single tensor (<= 4 GiB)
    std::size_t sub_alloc_max = 0;     // requests up to this are sub-allocated
    bool host_mapped = false;          // device memory is host visible (UMA / ReBAR)
};
MemoryStats memory_stats(Device d);

struct StreamStats {
    std::uint64_t submits = 0;
    std::uint64_t commands = 0;
    std::uint64_t launches = 0;        // graph launches
    std::uint32_t batch_limit = 0;     // commands per batch (BROTENSOR_VK_BATCH)
    std::size_t pipelines = 0;         // pipelines created so far
};
StreamStats stream_stats(Device d);

struct DeviceInfo {
    std::string name;
    std::string driver;
    std::uint32_t subgroup_size = 0;
    std::uint32_t min_subgroup_size = 0;
    std::uint32_t max_subgroup_size = 0;
    std::uint32_t max_shared_bytes = 0;
    std::uint32_t max_push_constant_bytes = 0;
    bool graphics_queue = false;       // ops are submitted on a graphics+compute queue
    bool cooperative_matrix = false;   // VK_KHR_cooperative_matrix enabled
    bool subgroup_size_control = false;
};
DeviceInfo device_info(Device d);

}  // namespace vulkan

// A replayable recording of Vulkan ops. Move-only; owns the command buffer
// and the memory freed during its capture.
class VulkanGraph {
public:
    VulkanGraph();
    ~VulkanGraph();
    VulkanGraph(VulkanGraph&&) noexcept;
    VulkanGraph& operator=(VulkanGraph&&) noexcept;
    VulkanGraph(const VulkanGraph&) = delete;
    VulkanGraph& operator=(const VulkanGraph&) = delete;

    bool valid() const;
    // Submit the recorded ops once more, ordered after everything enqueued on
    // the device before it. Does not wait; sync before reading results.
    void launch();
    // Drop the recording (waits for its launches, then releases the memory
    // held since capture).
    void reset();
    Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class VulkanGraphCapture;
};

// RAII capture scope. Construction starts recording the device's ops into a
// new graph; finish() ends it. Destroyed without finish() (an exception
// unwinding the captured region), the capture is discarded.
class VulkanGraphCapture {
public:
    explicit VulkanGraphCapture(Device d = Device::vulkan(0));
    ~VulkanGraphCapture();
    VulkanGraphCapture(const VulkanGraphCapture&) = delete;
    VulkanGraphCapture& operator=(const VulkanGraphCapture&) = delete;

    VulkanGraph finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brotensor
