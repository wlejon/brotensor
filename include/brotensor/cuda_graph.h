#pragma once

// GPU graph capture / replay — amortise per-kernel launch overhead for tight,
// fixed-shape inference loops (e.g. an autoregressive decode step that issues
// dozens of tiny kernels per token). Capture the sequence once, then replay the
// whole thing with a single launch instead of re-issuing every kernel.
//
// Device-neutral despite the name (kept so existing capture sites compile
// unchanged): the classes are defined in brotensor's core (src/graph.cpp) and
// forward to the backend the capture's device belongs to — a cudaGraph on
// CUDA (src/cuda/cuda_graph.cu), a hipGraph on HIP (src/hip/graph.hip), a
// re-submitted command buffer on Vulkan (VulkanGraph, vulkan.h). Metal and the
// CPU have no graph capture: graph_capture_available() says whether the
// default device (or a given one) has one, and a capture on a device without
// it throws.
//
// Which device. CudaGraphCapture() captures on the default device
// (default_device(), so set_default_device / DeviceScope /
// BROTENSOR_DEFAULT_DEVICE pick it) when that is a GPU with graph capture;
// Device::cuda(i) is resolved through the CUDA -> HIP / Vulkan alias first.
// With a CPU or Metal default device it falls back to the CUDA or HIP
// backend's current device (what the capture did before it was neutral), and
// with neither of those to Vulkan device 0. CudaGraphCapture(Device) names it.
// Only ops on the capture's device are recorded: an op on another GPU runs
// eagerly (or, on CUDA / HIP, fails the capture as before).
//
// The capture stream and the allocator already honour the backend's current
// stream; CudaGraphCapture routes the bracketed ops onto a dedicated capture
// stream so they land in the graph rather than on the default stream.
//
// Allocation inside the capture: on CUDA it is a graph memory node from the
// stream-ordered pool (on wherever the device supports pools). On HIP every
// allocation made inside the capture comes from an arena the graph owns and
// every free inside it is deferred, so the memory a replay touches stays put
// for the graph's lifetime and the capture records no hipMallocAsync /
// hipFreeAsync memory nodes, which ROCm replays wrongly (see
// src/hip/detail/capture_arena.h). Either way a step may allocate and free its
// temporaries; the warm-up below still matters for buffers that must outlive
// the step (outputs, carried state), which should be allocated before capture.
//
// Usage — warm up once (so every output buffer is allocated), then capture an
// identical run that reuses those exact tensors, then replay:
//
//   step();                         // warm-up: ops allocate their outputs
//   brotensor::sync_all();
//   brotensor::CudaGraph g;
//   {
//       brotensor::CudaGraphCapture cap;   // ops now enqueue on the capture stream
//       step();                            // re-run, reusing the same tensors
//       g = cap.finish();                  // end capture + instantiate
//   }
//   for (int t = 0; t < T; ++t) {
//       write_new_inputs_in_place();       // update input buffers in place
//       g.launch();                        // single launch replays the step
//       brotensor::sync_all();             // before reading outputs to host
//   }
//
// Contract: feed new inputs by writing into the captured input buffers in place
// between launches and read outputs from their buffers after launch(); kernels
// replay against the device pointers they were captured with, so persistent
// tensors must be the same objects (same pointers and shapes) in the warm-up,
// the capture and every replay. Every entry point throws std::runtime_error on
// a CUDA / HIP error.
//
// Backend differences the contract leaves room for: CUDA and HIP capture only
// the capturing thread's work (ThreadLocal mode); a Vulkan capture records
// every op issued on its device from any thread, and while it records,
// sync(), downloads and uploads on that device throw. Memory freed inside a
// capture is held by the graph on HIP and Vulkan (see above and vulkan.h).

#include "tensor.h"

#include <memory>

namespace brotensor {

// True when `d` (alias-resolved) is a registered GPU whose backend has graph
// capture: CUDA, HIP, Vulkan. False for the CPU and Metal.
bool graph_capture_available(Device d);
// The same question for the device CudaGraphCapture() would pick.
bool graph_capture_available();

// An instantiated, replayable graph. Move-only; owns the backend's executable
// graph (cudaGraphExec_t, hipGraphExec_t + capture arena, Vulkan command buffer).
class CudaGraph {
public:
    CudaGraph();
    ~CudaGraph();
    CudaGraph(CudaGraph&&) noexcept;
    CudaGraph& operator=(CudaGraph&&) noexcept;
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    // True once a capture has been instantiated into this handle.
    bool valid() const;

    // Replay the captured sequence on the current stream. Does not synchronise
    // — call sync(device) / sync_all() before reading results to host.
    // Throws if the handle is empty.
    void launch();

    // Drop the captured graph (frees the executable graph and, on HIP, the
    // capture arena once the device is idle).
    void reset();

    // The device the graph was captured on (CPU when empty).
    Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CudaGraphCapture;
};

// RAII capture scope. Construction begins capture on the chosen device (CUDA /
// HIP: creates a capture stream and makes it the current stream; Vulkan:
// starts recording the device's ops into a command buffer); finish() ends
// capture and instantiates. If the scope is destroyed without finish() (e.g.
// an exception unwinds the captured region), the capture is aborted and the
// previous stream restored.
class CudaGraphCapture {
public:
    CudaGraphCapture();                    // the default device (see above)
    explicit CudaGraphCapture(Device d);   // a named GPU device
    ~CudaGraphCapture();
    CudaGraphCapture(const CudaGraphCapture&) = delete;
    CudaGraphCapture& operator=(const CudaGraphCapture&) = delete;

    // End capture, instantiate the graph, restore the previous current stream,
    // and return the replayable handle. Throws if called twice.
    CudaGraph finish();

    Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brotensor
