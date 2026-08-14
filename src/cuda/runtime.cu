// CUDA backend runtime helpers.
//
// There is no public `cuda_init` / `cuda_set_stream` / etc. API — device
// lifecycle and sync go through `brotensor::init()` / `brotensor::sync()`.
// What lives here are CUDA-internal helpers:
//
//   * `cuda_check_throw` — backs the BROTENSOR_CUDA_CHECK macro defined in
//                          `src/cuda/detail/cuda_check.h`. Every CUDA TU
//                          reaches it through that header.
//   * `cuda_current_stream` — thread-local current stream used by hot ops
//                             (matmul, fp16 matmul, conv2d, flash attention
//                             fwd, int8w paths). Preserved verbatim from the
//                             pre-refactor runtime.cu so the hot-path stream
//                             behaviour does not regress. CUDA-internal now;
//                             not part of the public API.

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace brotensor::detail::cuda {

// ─── Error check throw ─────────────────────────────────────────────────────

void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    if (err == 0) return;
    const char* es = cudaGetErrorString(static_cast<cudaError_t>(err));
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "CUDA error %d (%s) at %s:%d in `%s`",
                  err, es ? es : "?", file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

// ─── Current stream (CUDA-internal) ────────────────────────────────────────

namespace {
constexpr int kMaxCudaStreams = 16;
thread_local cudaStream_t g_current_streams[kMaxCudaStreams] = {nullptr};

int resolve_device_index(int dev) {
    if (dev < 0) {
        int d = 0;
        if (cudaGetDevice(&d) == cudaSuccess && d >= 0 && d < kMaxCudaStreams) {
            return d;
        }
        return 0;
    }
    if (dev >= kMaxCudaStreams) return 0;
    return dev;
}
} // namespace

void* cuda_current_stream(int dev) {
    const int idx = resolve_device_index(dev);
    return reinterpret_cast<void*>(g_current_streams[idx]);
}

void* cuda_current_stream() {
    return cuda_current_stream(-1);
}

void cuda_set_stream(void* stream, int dev) {
    const int idx = resolve_device_index(dev);
    g_current_streams[idx] = reinterpret_cast<cudaStream_t>(stream);
}

void cuda_set_stream(void* stream) {
    cuda_set_stream(stream, -1);
}

} // namespace brotensor::detail::cuda

// Compatibility shim: many CUDA TUs forward-declare cuda_current_stream() in
// namespace brotensor and call it from there. Keep that name resolvable so
// each individual file doesn't have to re-qualify its forward decl.
namespace brotensor {
void* cuda_current_stream(int dev) { return ::brotensor::detail::cuda::cuda_current_stream(dev); }
void* cuda_current_stream() { return ::brotensor::detail::cuda::cuda_current_stream(-1); }
}
