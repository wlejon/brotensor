// CUDA graph capture / replay (public API in include/brotensor/cuda_graph.h).
//
// CudaGraphCapture brackets a fixed-shape op sequence: it creates a dedicated
// capture stream, makes it the current stream (so the stream-ordered allocator
// and every hot op enqueue onto it), and begins capture. finish() ends capture,
// instantiates the graph, and restores the previous current stream. CudaGraph
// replays the instantiated graph with one cudaGraphLaunch.

#include <brotensor/cuda_graph.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace brotensor {

// CUDA-internal current-stream hooks (defined in src/cuda/runtime.cu).
void* cuda_current_stream();
namespace detail::cuda { void cuda_set_stream(void*); }

// ─── CudaGraph ──────────────────────────────────────────────────────────────

struct CudaGraph::Impl {
    cudaGraphExec_t exec = nullptr;
    ~Impl() {
        if (exec) cudaGraphExecDestroy(exec);
    }
};

CudaGraph::CudaGraph() = default;
CudaGraph::~CudaGraph() = default;
CudaGraph::CudaGraph(CudaGraph&&) noexcept = default;
CudaGraph& CudaGraph::operator=(CudaGraph&&) noexcept = default;

bool CudaGraph::valid() const { return impl_ && impl_->exec != nullptr; }

void CudaGraph::reset() { impl_.reset(); }

void CudaGraph::launch() {
    if (!valid()) {
        throw std::runtime_error("brotensor: CudaGraph::launch: no captured graph");
    }
    auto stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    BROTENSOR_CUDA_CHECK(cudaGraphLaunch(impl_->exec, stream));
}

// ─── CudaGraphCapture ───────────────────────────────────────────────────────

struct CudaGraphCapture::Impl {
    cudaStream_t stream = nullptr;
    void* prev_stream = nullptr;
    bool capturing = false;
};

CudaGraphCapture::CudaGraphCapture() : impl_(new Impl) {
    BROTENSOR_CUDA_CHECK(cudaStreamCreate(&impl_->stream));
    impl_->prev_stream = cuda_current_stream();
    // Route subsequent ops (and pool allocations) onto the capture stream.
    detail::cuda::cuda_set_stream(impl_->stream);
    // ThreadLocal mode: only work on this thread's current stream is captured,
    // so unrelated host threads issuing CUDA work don't poison the capture.
    BROTENSOR_CUDA_CHECK(
        cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal));
    impl_->capturing = true;
}

CudaGraph CudaGraphCapture::finish() {
    if (!impl_ || !impl_->capturing) {
        throw std::runtime_error("brotensor: CudaGraphCapture::finish: not capturing");
    }
    cudaGraph_t graph = nullptr;
    BROTENSOR_CUDA_CHECK(cudaStreamEndCapture(impl_->stream, &graph));
    impl_->capturing = false;
    detail::cuda::cuda_set_stream(impl_->prev_stream);

    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        BROTENSOR_CUDA_CHECK(err);
    }

    CudaGraph g;
    g.impl_.reset(new CudaGraph::Impl);
    g.impl_->exec = exec;
    return g;
}

CudaGraphCapture::~CudaGraphCapture() {
    if (!impl_) return;
    if (impl_->capturing) {
        // Abort: drain the in-flight capture and discard the graph.
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(impl_->stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        detail::cuda::cuda_set_stream(impl_->prev_stream);
    }
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

}  // namespace brotensor
