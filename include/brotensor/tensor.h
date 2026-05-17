#pragma once

#include <cstddef>

namespace brotensor {

// ─── GpuTensor ─────────────────────────────────────────────────────────────
//
// Device-resident float32 tensor with row-major (rows, cols) shape. Storage
// is a raw device pointer (cudaMalloc on CUDA / MTLBuffer contents on Metal),
// freed by the matching device free on destruction when owning. Move-only —
// copying device buffers must be explicit (clone()).
//
// This header is safe to include from non-CUDA TUs (no <cuda_runtime.h>);
// the actual device API calls live in src/cuda/tensor.cu or src/metal/tensor.mm.

struct GpuTensor {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;

    GpuTensor() = default;
    GpuTensor(int r, int c);                   // cudaMalloc(r*c*sizeof(float))
    ~GpuTensor();

    // Move-only.
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    int  size() const { return rows * cols; }
    void zero();                                // cudaMemset to 0
    // Reallocates if shape differs from current; leaves contents undefined
    // (caller should zero() if needed). Existing storage is freed.
    void resize(int r, int c);

    // Device→device copy producing an owning duplicate.
    GpuTensor clone() const;

    // Non-owning view over an existing device pointer. The returned tensor's
    // destructor will NOT free `data`. Caller is responsible for lifetime.
    static GpuTensor view(float* data, int rows, int cols);

private:
    bool owns_ = false;
    void release_();
};

// ─── Host ↔ device transfers ───────────────────────────────────────────────
//
// brotensor doesn't own a host-tensor type. Callers pass raw float buffers
// and the source shape; for the upload path `dst` is resized to match (rows,
// cols). Downstream libraries that have their own CPU tensor type (e.g.
// brogameagent::nn::Tensor) provide thin inline glue at their own boundary.

// Host→device. dst is resized to (rows, cols). host must point to
// rows*cols floats.
void upload(const float* host, int rows, int cols, GpuTensor& dst);

// Device→host. host must point to at least src.size() floats. No allocation
// or shape change is performed on the host side.
void download(const GpuTensor& src, float* host);

} // namespace brotensor
