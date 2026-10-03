// CUDA graph capture / replay (public API in include/brotensor/cuda_graph.h,
// device-neutral front in src/graph.cpp, which calls the recorder below).
//
// CudaGraphCapture brackets a fixed-shape op sequence: it creates a dedicated
// capture stream, makes it the current stream (so the stream-ordered allocator
// and every hot op enqueue onto it), and begins capture. finish() ends capture,
// instantiates the graph, and restores the previous current stream. CudaGraph
// replays the instantiated graph with one cudaGraphLaunch.

#include <brotensor/cuda_graph.h>
#include <brotensor/detail/graph_backend.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

namespace brotensor {

// CUDA-internal current-stream hooks (defined in src/cuda/runtime.cu).
void* cuda_current_stream();
namespace detail::cuda { void cuda_set_stream(void*); }

// ─── Recorder / executable graph ───────────────────────────────────────────
//
// The CUDA half of the neutral CudaGraphCapture / CudaGraph (src/graph.cpp):
// CudaRecorder is what CudaGraphCapture's constructor, finish() and destructor
// did before the capture became device-neutral.

namespace {

struct CudaGraphExecImpl final : detail::GraphExec {
    cudaGraphExec_t exec = nullptr;
    ~CudaGraphExecImpl() override {
        if (exec) cudaGraphExecDestroy(exec);
    }
    void launch() override {
        auto stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        BROTENSOR_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
};

struct CudaRecorder final : detail::GraphRecorder {
    cudaStream_t stream = nullptr;
    void* prev_stream = nullptr;
    int dev = 0;
    bool capturing = false;

    explicit CudaRecorder(int device_index);
    ~CudaRecorder() override;
    std::unique_ptr<detail::GraphExec> finish() override;
    int device_index() const override { return dev; }
};

CudaRecorder::CudaRecorder(int device_index) {
    // -1: the thread's current CUDA device (the pre-neutral behaviour).
    if (device_index < 0) {
        BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev));
    } else {
        dev = device_index;
        BROTENSOR_CUDA_CHECK(cudaSetDevice(dev));
    }
    // Non-blocking: a default (blocking) stream may not capture while any other
    // thread touches the legacy stream — every legacy-stream launch elsewhere
    // fails with cudaErrorStreamCaptureImplicit (906) and the capture itself is
    // then poisoned (901). Concurrent model threads (e.g. a wake-word detector
    // feeding on the legacy stream while an LLM/TTS thread captures its decode
    // step) make that a routine collision, not an edge case.
    BROTENSOR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    prev_stream = cuda_current_stream();
    // Route subsequent ops (and pool allocations) onto the capture stream.
    detail::cuda::cuda_set_stream(stream);
    // ThreadLocal mode: only work on this thread's current stream is captured,
    // so unrelated host threads issuing CUDA work don't poison the capture.
    BROTENSOR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    capturing = true;
}

std::unique_ptr<detail::GraphExec> CudaRecorder::finish() {
    if (!capturing) {
        throw std::runtime_error("brotensor: CudaGraphCapture::finish: not capturing");
    }
    cudaGraph_t graph = nullptr;
    BROTENSOR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    capturing = false;
    detail::cuda::cuda_set_stream(prev_stream);

    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        BROTENSOR_CUDA_CHECK(err);
    }
    auto g = std::make_unique<CudaGraphExecImpl>();
    g->exec = exec;
    return g;
}

CudaRecorder::~CudaRecorder() {
    if (capturing) {
        // Abort: drain the in-flight capture and discard the graph.
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        detail::cuda::cuda_set_stream(prev_stream);
    }
    if (stream) cudaStreamDestroy(stream);
}

std::unique_ptr<detail::GraphRecorder> make_cuda_recorder(int device_index) {
    return std::make_unique<CudaRecorder>(device_index);
}

}  // namespace

namespace detail::cuda {
void register_cuda_graph_backend() {
    ::brotensor::detail::register_graph_backend(DeviceType::CUDA, &make_cuda_recorder);
}
}  // namespace detail::cuda

}  // namespace brotensor
