#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace brotensor {

// ─── Dtype ─────────────────────────────────────────────────────────────────
//
// brotensor's tensor type carries a dtype tag so ops can pick the right
// kernel without a parallel tensor type per precision. Storage stays as a
// single raw device pointer (`data`); FP16 ops reinterpret it as half* on
// the device side.
//
// Element sizes are fixed: FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte.
// Allocation, clone, zero, and resize all use dtype-aware byte counts.
// INT8 is currently only carried by weight-only quantised ops (W8A16
// matmul/conv2d); arithmetic ops only dispatch on FP32/FP16.
enum class Dtype : int {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

inline int dtype_size_bytes(Dtype dt) {
    switch (dt) {
        case Dtype::FP32: return 4;
        case Dtype::FP16: return 2;
        case Dtype::INT8: return 1;
    }
    return 4;
}

// ─── Tensor (host) ─────────────────────────────────────────────────────────
//
// Plain owned float buffer + shape on the CPU. No broadcasting, no views, no
// strides beyond row-major. Rank is fixed at 2 (matrix) or 1 (vector) —
// anything higher would imply we want a real tensor framework, which we
// don't. Batch dims go into dim 0 of a 2D tensor.
//
// Shape conventions:
//   Vector: shape = (N)        ; data[i]
//   Matrix: shape = (rows,cols); data[r*cols + c]
//
// Memory is std::vector<float>. CPU ops live in <brotensor/ops_cpu.h> and
// take Tensor by reference; GPU ops live in <brotensor/ops.h> and take
// GpuTensor.

struct Tensor {
    std::vector<float> data;
    int rows = 0;   // rank-1 tensors: rows = N, cols = 1
    int cols = 0;

    Tensor() = default;
    Tensor(int r, int c) : data(static_cast<size_t>(r) * c, 0.0f), rows(r), cols(c) {}

    static Tensor vec(int n) { return Tensor(n, 1); }
    static Tensor mat(int r, int c) { return Tensor(r, c); }

    int size() const { return rows * cols; }
    float*       ptr()       { return data.data(); }
    const float* ptr() const { return data.data(); }

    float&       operator()(int r, int c)       { return data[static_cast<size_t>(r) * cols + c]; }
    float        operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    float&       operator[](int i)       { return data[i]; }
    float        operator[](int i) const { return data[i]; }

    void zero() { std::memset(data.data(), 0, data.size() * sizeof(float)); }
    void resize(int r, int c) { rows = r; cols = c; data.assign(static_cast<size_t>(r) * c, 0.0f); }
};

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

// ─── Tensor (host) ↔ device convenience overloads ──────────────────────────
//
// Inline wrappers so callers holding a brotensor::Tensor can transfer without
// pulling out (ptr, rows, cols) by hand. Only compiled when a GPU backend is
// enabled; the raw-pointer overloads above are the underlying definitions.
#ifdef BROTENSOR_HAS_GPU
inline void upload(const Tensor& src, GpuTensor& dst) {
    upload(src.data.data(), src.rows, src.cols, dst);
}
inline void download(const GpuTensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        dst.resize(src.rows, src.cols);
    }
    download(src, dst.data.data());
}
#endif

// ─── FP16 ↔ FP32 host-side conversion helpers ──────────────────────────────
//
// Pure-CPU IEEE 754 binary16 conversion. Useful for tests and small
// preprocessing where a GPU roundtrip would be wasteful. Not intended for
// hot loops.
uint16_t fp32_to_fp16_bits(float v);
float    fp16_bits_to_fp32(uint16_t bits);

} // namespace brotensor
