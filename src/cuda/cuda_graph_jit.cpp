#include "cuda_graph_jit.h"
#include "cuda_driver.h"
#include "cuda_jit_engine.h"
#include "detail/cuda_check.h"

#include <stdexcept>
#include <utility>

namespace brotensor {
void* cuda_current_stream();
namespace detail::cuda { void cuda_set_stream(void*); }
}

namespace brotensor::detail::cuda::jit {

CudaGraphRunner::CudaGraphRunner() = default;

CudaGraphRunner::~CudaGraphRunner() {
    reset();
}

CudaGraphRunner::CudaGraphRunner(CudaGraphRunner&& other) noexcept
    : graph_(other.graph_),
      graph_exec_(other.graph_exec_),
      owned_stream_(other.owned_stream_),
      capture_stream_(other.capture_stream_),
      prev_stream_(other.prev_stream_),
      capturing_(other.capturing_) {
    other.graph_ = nullptr;
    other.graph_exec_ = nullptr;
    other.owned_stream_ = nullptr;
    other.capture_stream_ = nullptr;
    other.prev_stream_ = nullptr;
    other.capturing_ = false;
}

CudaGraphRunner& CudaGraphRunner::operator=(CudaGraphRunner&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = other.graph_;
        graph_exec_ = other.graph_exec_;
        owned_stream_ = other.owned_stream_;
        capture_stream_ = other.capture_stream_;
        prev_stream_ = other.prev_stream_;
        capturing_ = other.capturing_;

        other.graph_ = nullptr;
        other.graph_exec_ = nullptr;
        other.owned_stream_ = nullptr;
        other.capture_stream_ = nullptr;
        other.prev_stream_ = nullptr;
        other.capturing_ = false;
    }
    return *this;
}

void CudaGraphRunner::reset() noexcept {
    if (capturing_ && capture_stream_) {
        CUgraph discarded = nullptr;
        (void)drv::cuStreamEndCapture(capture_stream_, &discarded);
        if (discarded) {
            (void)drv::cuGraphDestroy(discarded);
        }
        capturing_ = false;
        if (prev_stream_) {
            ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
            prev_stream_ = nullptr;
        }
    }
    if (graph_exec_) {
        (void)drv::cuGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        (void)drv::cuGraphDestroy(graph_);
        graph_ = nullptr;
    }
    if (owned_stream_) {
        (void)drv::cuStreamDestroy(owned_stream_);
        owned_stream_ = nullptr;
    }
    capture_stream_ = nullptr;
    prev_stream_ = nullptr;
}

void CudaGraphRunner::begin_capture(void* stream) {
    if (capturing_) {
        throw std::runtime_error("CudaGraphRunner: already capturing");
    }
    ensure_context(-1);
    if (graph_exec_) {
        (void)drv::cuGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        (void)drv::cuGraphDestroy(graph_);
        graph_ = nullptr;
    }

    if (stream) {
        capture_stream_ = reinterpret_cast<CUstream>(stream);
    } else {
        if (!owned_stream_) {
            BROTENSOR_CUDA_CHECK(drv::cuStreamCreate(&owned_stream_, CU_STREAM_NON_BLOCKING));
        }
        capture_stream_ = owned_stream_;
    }

    prev_stream_ = ::brotensor::cuda_current_stream();
    ::brotensor::detail::cuda::cuda_set_stream(capture_stream_);

    const CUresult err = drv::cuStreamBeginCapture(capture_stream_, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if (err != CUDA_SUCCESS) {
        ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
        prev_stream_ = nullptr;
        capture_stream_ = nullptr;
        BROTENSOR_CUDA_CHECK(err);
    }
    capturing_ = true;
}

void CudaGraphRunner::end_capture(void* stream) {
    if (!capturing_) {
        throw std::runtime_error("CudaGraphRunner: end_capture called but not capturing");
    }
    CUstream s = stream ? reinterpret_cast<CUstream>(stream) : capture_stream_;
    const CUresult err = drv::cuStreamEndCapture(s, &graph_);
    capturing_ = false;
    ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
    prev_stream_ = nullptr;

    BROTENSOR_CUDA_CHECK(err);
}

void CudaGraphRunner::instantiate() {
    if (!graph_) {
        throw std::runtime_error("CudaGraphRunner: cannot instantiate before capture");
    }
    if (graph_exec_) {
        (void)drv::cuGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }

    BROTENSOR_CUDA_CHECK(drv::cuGraphInstantiate(&graph_exec_, graph_, 0));
}

void CudaGraphRunner::launch(void* stream) {
    if (!graph_exec_) {
        throw std::runtime_error("CudaGraphRunner: cannot launch before instantiation");
    }
    ensure_context(-1);
    CUstream s = stream ? reinterpret_cast<CUstream>(stream)
                        : (capture_stream_ ? capture_stream_
                                           : reinterpret_cast<CUstream>(::brotensor::cuda_current_stream()));
    BROTENSOR_CUDA_CHECK(drv::cuGraphLaunch(graph_exec_, s));
}

}  // namespace brotensor::detail::cuda::jit
