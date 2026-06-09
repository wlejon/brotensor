#pragma once

// CUDA graph capture / replay — amortise per-kernel launch overhead for tight,
// fixed-shape inference loops (e.g. an autoregressive decode step that issues
// dozens of tiny kernels per token). Capture the sequence once, then replay the
// whole thing with a single cudaGraphLaunch instead of re-issuing every kernel.
//
// CUDA-only: the symbols are provided by the CUDA backend, so this header is
// usable only in a BROTENSOR_WITH_CUDA build (mirrors metal_interop.h on the
// Metal side). Gate calls on BROTENSOR_HAS_CUDA.
//
// The capture stream and the stream-ordered allocator already honour
// cuda_current_stream(); CudaGraphCapture routes the bracketed ops onto a
// dedicated capture stream so they land in the graph rather than on the default
// stream.
//
// Usage — warm up once (so every output buffer is allocated), then capture an
// identical run that reuses those exact tensors (no allocation during capture),
// then replay:
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
//       brotensor::sync(Device::CUDA);     // before reading outputs to host
//   }
//
// Contract: the captured run MUST reuse the warm-up tensor objects (same device
// pointers and shapes) so no allocation happens mid-capture; feed new inputs by
// writing into those buffers in place between launches and read outputs from
// their buffers after launch(). Every entry point throws std::runtime_error on
// a CUDA error.

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
    // — call sync(Device::CUDA) / sync_all() before reading results to host.
    // Throws if the handle is empty.
    void launch();

    // Drop the captured graph (frees the cudaGraphExec_t).
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
