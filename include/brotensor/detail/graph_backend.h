#pragma once

// Backend side of the device-neutral graph capture (public API:
// include/brotensor/cuda_graph.h). CudaGraphCapture / CudaGraph are defined
// once, in src/graph.cpp; each GPU backend that can record and replay an op
// sequence registers a factory here from its probe (CUDA: cudaGraph, HIP:
// hipGraph + capture arena, Vulkan: a re-submitted command buffer), and the
// capture scope asks the factory of the backend its device belongs to.
// Nothing outside brotensor's backends includes this header.

#include <brotensor/tensor.h>

#include <memory>

namespace brotensor::detail {

// An instantiated recording. launch() enqueues one replay without waiting.
class GraphExec {
public:
    virtual ~GraphExec() = default;
    virtual void launch() = 0;
};

// An open capture on one device. Created by the backend's factory (which
// starts recording); finish() ends it and returns the replayable form; being
// destroyed without finish() aborts it and restores the backend's state.
class GraphRecorder {
public:
    virtual ~GraphRecorder() = default;
    virtual std::unique_ptr<GraphExec> finish() = 0;
    // The device index actually recording (resolves a factory's -1).
    virtual int device_index() const = 0;
};

// `device_index` < 0 means "the backend's current device" (HIP / CUDA:
// hipGetDevice / cudaGetDevice), which is what the pre-neutral capture did.
using GraphRecorderFactory = std::unique_ptr<GraphRecorder> (*)(int device_index);

void register_graph_backend(DeviceType dt, GraphRecorderFactory factory);

// Null when the backend has no graph capture (CPU, Metal) or is not built.
GraphRecorderFactory graph_backend(DeviceType dt);

}  // namespace brotensor::detail
