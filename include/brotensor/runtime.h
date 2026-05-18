#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor {

// ─── CUDA Runtime helpers ──────────────────────────────────────────────────
//
// This header is safe to include from non-CUDA translation units. It does
// NOT include <cuda_runtime.h>. The check macro forwards to a thin wrapper
// implemented in runtime.cu — that's where the real cudaGetErrorString call
// happens. We want public headers to be plain C++ so that libraries
// consuming the foundation tensor type don't get pulled into nvcc.

// Idempotent. Selects device 0 unless the env var BROTENSOR_CUDA_DEVICE overrides
// it with a decimal device index. Safe to call multiple times.
void cuda_init();

// Wraps cudaDeviceSynchronize(). Throws std::runtime_error on failure.
void cuda_sync();

// ─── CUDA streams ──────────────────────────────────────────────────────────
//
// Thread-local "current stream" used by the hot compute ops (matmul, fp16
// matmul, conv2d, flash_attention forward) for their kernel launches.
// Setting the stream is purely additive — ops that don't query the current
// stream stay on the default (null) stream. `cudaStream_t` is an opaque
// pointer; the public header accepts `void*` to avoid pulling in
// <cuda_runtime.h>. Internally the implementations cast back to cudaStream_t.
// On Metal these are no-ops; Metal has its own queue model.
//
// Passing nullptr to cuda_set_stream restores the default (null) stream.
void  cuda_set_stream(void* stream);
void* cuda_current_stream();

// Wait for a single stream to drain (cudaStreamSynchronize). Throws on
// error. Pass nullptr to sync the default stream. On Metal this is a
// no-op (ops are submitted synchronously per command buffer).
void  cuda_stream_sync(void* stream);

// Implementation hook for BROTENSOR_CUDA_CHECK — translates a cudaError_t (passed
// in as int because we don't include cuda_runtime.h here) into a human
// readable error message and throws std::runtime_error if non-zero.
//
// `expr_text` is the stringified expression for diagnostic context;
// `file`/`line` describe the call site.
void cuda_check_throw(int err, const char* expr_text, const char* file, int line);

} // namespace brotensor

// Wrap any CUDA call. Safe to use from .cpp files (only forward-declared
// helpers are referenced; the int conversion is implicit from cudaError_t).
#define BROTENSOR_CUDA_CHECK(expr)                                                    \
    do {                                                                        \
        int _bga_err = static_cast<int>(expr);                                  \
        if (_bga_err != 0) {                                                    \
            ::brotensor::cuda_check_throw(                          \
                _bga_err, #expr, __FILE__, __LINE__);                           \
        }                                                                       \
    } while (0)
