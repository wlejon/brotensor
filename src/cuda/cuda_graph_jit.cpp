#include "cuda_graph_jit.h"
#include "cuda_jit_engine.h"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
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
        cudaGraph_t discarded = nullptr;
        cudaStreamEndCapture(capture_stream_, &discarded);
        if (discarded) {
            cudaGraphDestroy(discarded);
        }
        capturing_ = false;
        if (prev_stream_) {
            ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
            prev_stream_ = nullptr;
        }
    }
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    if (owned_stream_) {
        cudaStreamDestroy(owned_stream_);
        owned_stream_ = nullptr;
    }
    capture_stream_ = nullptr;
    prev_stream_ = nullptr;
}

void CudaGraphRunner::begin_capture(void* stream) {
    if (capturing_) {
        throw std::runtime_error("CudaGraphRunner: already capturing");
    }
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }

    if (stream) {
        capture_stream_ = reinterpret_cast<cudaStream_t>(stream);
    } else {
        if (!owned_stream_) {
            BROTENSOR_CUDA_CHECK(cudaStreamCreateWithFlags(&owned_stream_, cudaStreamNonBlocking));
        }
        capture_stream_ = owned_stream_;
    }

    prev_stream_ = ::brotensor::cuda_current_stream();
    ::brotensor::detail::cuda::cuda_set_stream(capture_stream_);

    cudaError_t err = cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
        prev_stream_ = nullptr;
        capture_stream_ = nullptr;
        throw std::runtime_error(std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(err));
    }
    capturing_ = true;
}

void CudaGraphRunner::end_capture(void* stream) {
    if (!capturing_) {
        throw std::runtime_error("CudaGraphRunner: end_capture called but not capturing");
    }
    cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream) : capture_stream_;
    cudaError_t err = cudaStreamEndCapture(s, &graph_);
    capturing_ = false;
    ::brotensor::detail::cuda::cuda_set_stream(prev_stream_);
    prev_stream_ = nullptr;

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(err));
    }
}

void CudaGraphRunner::instantiate() {
    if (!graph_) {
        throw std::runtime_error("CudaGraphRunner: cannot instantiate before capture");
    }
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }

    cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(err));
    }
}

void CudaGraphRunner::launch(void* stream) {
    if (!graph_exec_) {
        throw std::runtime_error("CudaGraphRunner: cannot launch before instantiation");
    }
    cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream)
                            : (capture_stream_ ? capture_stream_
                                               : reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()));
    BROTENSOR_CUDA_CHECK(cudaGraphLaunch(graph_exec_, s));
}

} // namespace brotensor::detail::cuda::jit
