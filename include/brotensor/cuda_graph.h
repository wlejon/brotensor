#pragma once

// CUDA / HIP graph capture / replay — amortise per-kernel launch overhead for
// tight, fixed-shape inference loops (e.g. an autoregressive decode step that
// issues dozens of tiny kernels per token). Capture the sequence once, then
// replay the whole thing with a single cudaGraphLaunch / hipGraphLaunch instead
// of re-issuing every kernel.
//
// CUDA and HIP: the symbols are provided by the CUDA backend (cudaGraph*) and
// by the HIP backend (hipGraph*, src/hip/graph.hip); Metal has no graph
// capture. Gate calls on `BROTENSOR_HAS_CUDA || BROTENSOR_HAS_HIP` and capture
// only on a Device::CUDA / Device::HIP default device.
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

#include <memory>

namespace brotensor {

// An instantiated, replayable CUDA graph. Move-only; owns the cudaGraphExec_t.
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CudaGraphCapture;
};

// RAII capture scope. Construction creates a capture stream, makes it the
// current stream, and begins capture; finish() ends capture and instantiates.
// If the scope is destroyed without finish() (e.g. an exception unwinds the
// captured region), the capture is aborted and the previous stream restored.
class CudaGraphCapture {
public:
    CudaGraphCapture();
    ~CudaGraphCapture();
    CudaGraphCapture(const CudaGraphCapture&) = delete;
    CudaGraphCapture& operator=(const CudaGraphCapture&) = delete;

    // End capture, instantiate the graph, restore the previous current stream,
    // and return the replayable handle. Throws if called twice.
    CudaGraph finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brotensor
