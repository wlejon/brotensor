#pragma once

// ─── brotensor::jit — Automatic Tracing JIT Interface ────────────────────────
//
// Dynamic expression and operation sequence tracing powered by Brass.
// Automatically captures mathematical expressions and operation patterns,
// analyzes tensor liveness, and emits fused AVX2/FMA machine code on CPU or
// cooperative PTX kernels on NVIDIA RTX GPUs.

#include <brotensor/tensor.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace brotensor::jit {

class TraceHandleImpl;

class TraceHandle {
public:
    TraceHandle() = default;
    explicit TraceHandle(std::shared_ptr<TraceHandleImpl> impl) : impl_(std::move(impl)) {}

    void execute();
    uint64_t signature_hash() const;
    bool is_cuda() const;
    size_t node_count() const;
    double last_duration_us() const;
    bool is_cache_hit() const;

    explicit operator bool() const noexcept { return impl_ != nullptr; }

private:
    std::shared_ptr<TraceHandleImpl> impl_;
};

// Starts tracing thread-local operations.
void begin_trace();

// Stops tracing, compiles the captured DAG (or retrieves from cache), and returns a TraceHandle.
TraceHandle end_trace();

// Returns true if tracing is currently active on this thread.
bool is_tracing();

// RAII helper for tracing scopes.
class TraceScope {
public:
    TraceScope() { begin_trace(); }
    ~TraceScope() {
        if (is_tracing()) {
            try { end_trace(); } catch (...) {}
        }
    }
    TraceHandle end() { return end_trace(); }

    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
};

// ── Overloaded Operators for Tracing and Eager Execution ──────────────────────

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float s);
inline Tensor operator+(float s, const Tensor& a) { return a + s; }
inline Tensor operator+(const Tensor& a, int s) { return a + static_cast<float>(s); }
inline Tensor operator+(int s, const Tensor& a) { return a + static_cast<float>(s); }

Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, float s);
Tensor operator-(float s, const Tensor& a);
inline Tensor operator-(const Tensor& a, int s) { return a - static_cast<float>(s); }
inline Tensor operator-(int s, const Tensor& a) { return static_cast<float>(s) - a; }

Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, float s);
inline Tensor operator*(float s, const Tensor& a) { return a * s; }
inline Tensor operator*(const Tensor& a, int s) { return a * static_cast<float>(s); }
inline Tensor operator*(int s, const Tensor& a) { return a * static_cast<float>(s); }

Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, float s);
inline Tensor operator/(const Tensor& a, int s) { return a / static_cast<float>(s); }

Tensor& operator+=(Tensor& a, const Tensor& b);
Tensor& operator+=(Tensor& a, float s);
inline Tensor& operator+=(Tensor& a, int s) { return a += static_cast<float>(s); }

Tensor& operator*=(Tensor& a, const Tensor& b);
Tensor& operator*=(Tensor& a, float s);
inline Tensor& operator*=(Tensor& a, int s) { return a *= static_cast<float>(s); }

// ── Functional Activations & Normalizations ───────────────────────────────────

Tensor silu(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor relu(const Tensor& a);

Tensor rms_norm(const Tensor& x, const Tensor& gamma = Tensor(), float eps = 1e-5f);
Tensor layernorm(const Tensor& x, const Tensor& gamma = Tensor(), const Tensor& beta = Tensor(), float eps = 1e-5f);
Tensor modulate(const Tensor& x, const Tensor& scale, const Tensor& shift);

} // namespace brotensor::jit

namespace brotensor {
using jit::begin_trace;
using jit::end_trace;
using jit::is_tracing;
using jit::TraceScope;
using jit::TraceHandle;

using jit::operator+;
using jit::operator-;
using jit::operator*;
using jit::operator/;
using jit::operator+=;
using jit::operator*=;

using jit::silu;
using jit::gelu;
using jit::relu;
using jit::rms_norm;
using jit::layernorm;
using jit::modulate;
} // namespace brotensor
