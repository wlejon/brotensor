// CUDA graph capture / replay (pure C++ backed by the dynamic Driver API).

#include <brotensor/cuda_graph.h>
#include <brotensor/detail/graph_backend.h>

#include "cuda_driver.h"
#include "detail/cuda_check.h"

#include <memory>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();
namespace detail::cuda { void cuda_set_stream(void*); }

namespace {

struct CudaGraphExecImpl final : detail::GraphExec {
    CUgraphExec exec = nullptr;
    ~CudaGraphExecImpl() override {
        if (exec) (void)detail::cuda::drv::cuGraphExecDestroy(exec);
    }
    void launch() override {
        detail::cuda::ensure_context(-1);
        auto stream = reinterpret_cast<CUstream>(cuda_current_stream());
        BROTENSOR_CUDA_CHECK(detail::cuda::drv::cuGraphLaunch(exec, stream));
    }
};

struct CudaRecorder final : detail::GraphRecorder {
    CUstream stream = nullptr;
    void* prev_stream = nullptr;
    int dev = 0;
    bool capturing = false;

    explicit CudaRecorder(int device_index);
    ~CudaRecorder() override;
    std::unique_ptr<detail::GraphExec> finish() override;
    int device_index() const override { return dev; }
};

CudaRecorder::CudaRecorder(int device_index) {
    if (device_index < 0) {
        dev = detail::cuda::current_device();
    } else {
        dev = device_index;
        BROTENSOR_CUDA_CHECK(detail::cuda::set_device(dev));
    }
    BROTENSOR_CUDA_CHECK(detail::cuda::drv::cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    prev_stream = cuda_current_stream();
    detail::cuda::cuda_set_stream(stream);
    BROTENSOR_CUDA_CHECK(detail::cuda::drv::cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    capturing = true;
}

std::unique_ptr<detail::GraphExec> CudaRecorder::finish() {
    if (!capturing) {
        throw std::runtime_error("brotensor: CudaGraphCapture::finish: not capturing");
    }
    CUgraph graph = nullptr;
    BROTENSOR_CUDA_CHECK(detail::cuda::drv::cuStreamEndCapture(stream, &graph));
    capturing = false;
    detail::cuda::cuda_set_stream(prev_stream);

    CUgraphExec exec = nullptr;
    const CUresult err = detail::cuda::drv::cuGraphInstantiate(&exec, graph, 0);
    (void)detail::cuda::drv::cuGraphDestroy(graph);
    BROTENSOR_CUDA_CHECK(err);

    auto g = std::make_unique<CudaGraphExecImpl>();
    g->exec = exec;
    return g;
}

CudaRecorder::~CudaRecorder() {
    if (capturing) {
        CUgraph graph = nullptr;
        (void)detail::cuda::drv::cuStreamEndCapture(stream, &graph);
        if (graph) (void)detail::cuda::drv::cuGraphDestroy(graph);
        detail::cuda::cuda_set_stream(prev_stream);
    }
    if (stream) (void)detail::cuda::drv::cuStreamDestroy(stream);
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
