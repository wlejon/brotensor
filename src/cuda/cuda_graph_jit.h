#pragma once

#include "cuda_driver.h"

#include <functional>
#include <memory>

namespace brotensor::detail::cuda::jit {

// CUDA Graph Runner for JIT Kernels and op sequences.
// Eliminates kernel launch and driver overhead by capturing kernel launches into
// an optimized CUDA Graph executable and replaying it with a single cuGraphLaunch call.
class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    CudaGraphRunner(CudaGraphRunner&& other) noexcept;
    CudaGraphRunner& operator=(CudaGraphRunner&& other) noexcept;

    // Begins stream capture on the given stream (or dedicated non-blocking stream if nullptr).
    void begin_capture(void* stream = nullptr);

    // Ends stream capture and captures the created CUgraph.
    void end_capture(void* stream = nullptr);

    // Instantiates the captured graph into an executable graph (CUgraphExec).
    void instantiate();

    // Replays the instantiated graph on the given stream (or capture stream if nullptr).
    void launch(void* stream = nullptr);

    template <typename Fn>
    void capture(Fn&& fn, void* stream = nullptr) {
        begin_capture(stream);
        try {
            fn();
        } catch (...) {
            reset();
            throw;
        }
        end_capture(stream);
        instantiate();
    }

    bool is_capturing() const noexcept { return capturing_; }
    bool is_captured() const noexcept { return graph_ != nullptr; }
    bool is_instantiated() const noexcept { return graph_exec_ != nullptr; }

    void reset() noexcept;

private:
    CUgraph graph_ = nullptr;
    CUgraphExec graph_exec_ = nullptr;
    CUstream owned_stream_ = nullptr;
    CUstream capture_stream_ = nullptr;
    void* prev_stream_ = nullptr;
    bool capturing_ = false;
};

}  // namespace brotensor::detail::cuda::jit
