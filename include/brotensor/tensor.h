#pragma once

#include <cstddef>
#include <cstdint>

namespace brotensor {

// ─── Dtype ─────────────────────────────────────────────────────────────────
//
// brotensor's tensor type carries a dtype tag so ops can pick the right
// kernel without a parallel tensor type per precision. Storage stays as a
// single raw device pointer (`data`); FP16 ops reinterpret it as half* on
// the device side.
//
// Element sizes are fixed: FP32 = 4 bytes, FP16 = 2 bytes. Allocation,
// clone, zero, and resize all use dtype-aware byte counts.
enum class Dtype : int {
    FP32 = 0,
    FP16 = 1,
};

inline int dtype_size_bytes(Dtype dt) {
    return dt == Dtype::FP16 ? 2 : 4;
}

// ─── GpuTensor ─────────────────────────────────────────────────────────────
//
// Device-resident tensor with row-major (rows, cols) shape. Storage is a raw
// device pointer (cudaMalloc on CUDA / MTLBuffer contents on Metal), freed
// by the matching device free on destruction when owning. Move-only —
// copying device buffers must be explicit (clone()).
//
// `data` is typed as `float*` for ABI continuity with existing FP32 ops; for
// FP16 it aliases the same allocation and ops reinterpret it as half* on the
// device side. Use `data_fp16()` for the typed FP16 pointer on the host
// side. This header is safe to include from non-CUDA TUs.

struct GpuTensor {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;
    Dtype dtype = Dtype::FP32;

    GpuTensor() = default;
    explicit GpuTensor(int r, int c, Dtype dt = Dtype::FP32);
    ~GpuTensor();

    // Move-only.
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    int  size() const { return rows * cols; }
    std::size_t bytes() const {
        return static_cast<std::size_t>(size()) *
               static_cast<std::size_t>(dtype_size_bytes(dtype));
    }
    void zero();                                // memset to 0 over bytes()
    // Reallocates if (r, c, dt) differs from the current shape/dtype; leaves
    // contents undefined (caller should zero() if needed). Existing storage
    // is freed. dt defaults to FP32 — pass explicitly for FP16 tensors.
    void resize(int r, int c, Dtype dt = Dtype::FP32);

    // Device→device copy producing an owning duplicate (same dtype).
    GpuTensor clone() const;

    // Typed FP16 pointer alias for `data`. Caller is responsible for using
    // this only on FP16 tensors.
    uint16_t*       data_fp16()       { return reinterpret_cast<uint16_t*>(data); }
    const uint16_t* data_fp16() const { return reinterpret_cast<const uint16_t*>(data); }

    // Non-owning view over an existing device pointer. The returned tensor's
    // destructor will NOT free `data`. Caller is responsible for lifetime.
    static GpuTensor view(float* data, int rows, int cols);
    static GpuTensor view_fp16(uint16_t* data, int rows, int cols);

private:
    bool owns_ = false;
    void release_();
};

// ─── Host ↔ device transfers ───────────────────────────────────────────────
//
// brotensor doesn't own a host-tensor type. Callers pass raw buffers and the
// source shape; for upload, `dst` is resized to match (rows, cols) and the
// destination dtype. Downstream libraries that have their own CPU tensor
// type provide thin inline glue at their own boundary.

// Host→device, FP32. dst is resized to (rows, cols, FP32). host must point
// to rows*cols floats.
void upload(const float* host, int rows, int cols, GpuTensor& dst);

// Device→host, FP32. host must point to at least src.size() floats.
// src.dtype must be FP32.
void download(const GpuTensor& src, float* host);

// Host→device, FP16. dst is resized to (rows, cols, FP16). host must point
// to rows*cols half-precision values, encoded as uint16_t (IEEE 754 binary16
// bit pattern).
void upload_fp16(const uint16_t* host, int rows, int cols, GpuTensor& dst);

// Device→host, FP16. host must point to at least src.size() uint16_t slots.
// src.dtype must be FP16.
void download_fp16(const GpuTensor& src, uint16_t* host);

// ─── FP16 ↔ FP32 host-side conversion helpers ──────────────────────────────
//
// Pure-CPU IEEE 754 binary16 conversion. Useful for tests and small
// preprocessing where a GPU roundtrip would be wasteful. Not intended for
// hot loops.
uint16_t fp32_to_fp16_bits(float v);
float    fp16_bits_to_fp32(uint16_t bits);

} // namespace brotensor
