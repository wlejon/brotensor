// HIP backend runtime helpers.
//
// Internal helpers for HIP stream management and error checking.

#include "detail/hip_check.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace brotensor::detail::hip {

// ─── Error check throw ─────────────────────────────────────────────────────

void hip_check_throw(hipError_t err, const char* expr_text, const char* file, int line) {
    if (err == hipSuccess) return;
    const char* es = hipGetErrorString(err);
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "HIP error %d (%s) at %s:%d in `%s`",
                  static_cast<int>(err), es ? es : "?", file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

// ─── Current stream (HIP-internal) ────────────────────────────────────────

namespace {
constexpr int kMaxHipStreams = 16;
thread_local hipStream_t g_current_streams[kMaxHipStreams] = {nullptr};

int resolve_device_index(int dev) {
    if (dev < 0) {
        int d = 0;
        if (hipGetDevice(&d) == hipSuccess && d >= 0 && d < kMaxHipStreams) {
            return d;
        }
        return 0;
    }
    if (dev >= kMaxHipStreams) return 0;
    return dev;
}
} // namespace

void* hip_current_stream(int dev) {
    const int idx = resolve_device_index(dev);
    return reinterpret_cast<void*>(g_current_streams[idx]);
}

void* hip_current_stream() {
    return hip_current_stream(-1);
}

void hip_set_stream(void* stream, int dev) {
    const int idx = resolve_device_index(dev);
    g_current_streams[idx] = reinterpret_cast<hipStream_t>(stream);
}

void hip_set_stream(void* stream) {
    hip_set_stream(stream, -1);
}

} // namespace brotensor::detail::hip

// Compatibility shim
namespace brotensor {
void* hip_current_stream(int dev) { return ::brotensor::detail::hip::hip_current_stream(dev); }
void* hip_current_stream() { return ::brotensor::detail::hip::hip_current_stream(-1); }
void hip_set_stream(void* stream, int dev) { ::brotensor::detail::hip::hip_set_stream(stream, dev); }
void hip_set_stream(void* stream) { ::brotensor::detail::hip::hip_set_stream(stream, -1); }

void* cuda_current_stream(int dev) { return ::brotensor::detail::hip::hip_current_stream(dev); }
void* cuda_current_stream() { return ::brotensor::detail::hip::hip_current_stream(-1); }
void cuda_set_stream(void* stream, int dev) { ::brotensor::detail::hip::hip_set_stream(stream, dev); }
void cuda_set_stream(void* stream) { ::brotensor::detail::hip::hip_set_stream(stream, -1); }
}

namespace brotensor::detail::cuda {
void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    ::brotensor::detail::hip::hip_check_throw(static_cast<hipError_t>(err), expr_text, file, line);
}
}
