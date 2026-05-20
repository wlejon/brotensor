#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brotensor {

// ─── Dtype ─────────────────────────────────────────────────────────────────
//
// brotensor's tensor type carries a dtype tag so ops can pick the right
// kernel without a parallel tensor type per precision. Storage stays as a
// single raw `void*` (`data`); typed access is via the host_f32 / host_fp16
// accessors. GPU backends reinterpret the same allocation for FP16 / INT8.
//
// Element sizes are fixed: FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte,
// INT32 = 4 bytes. Allocation, clone, zero, and resize all use dtype-aware
// byte counts. INT8 is currently only carried by weight-only quantised ops
// (W8A16 matmul/conv2d); arithmetic ops only dispatch on FP32/FP16. INT32 is
// likewise a pure storage carrier — used for device-resident index/offset
// buffers (e.g. per-head offset tables for softmax_xent_fused_batched); no
// arithmetic op dispatches on it.
enum class Dtype : int {
    FP32  = 0,
    FP16  = 1,
    INT8  = 2,
    INT32 = 3,
};

int dtype_size_bytes(Dtype);

// ─── Device ────────────────────────────────────────────────────────────────
//
// Runtime backend tag carried on every Tensor. CPU is always available;
// CUDA / Metal are registered at runtime by `brotensor::init()` if the
// corresponding backend was compiled into this binary. Multi-GPU within a
// single backend is deliberately out of scope for now (no Device::CUDA(idx)).
enum class Device { CPU, CUDA, Metal };

const char* device_name(Device);

// ─── Tensor ────────────────────────────────────────────────────────────────
//
// Unified tensor: a row-major (rows, cols) buffer tagged with both a Dtype
// and a Device. Storage is a single opaque `void*` allocated through the
// backend's alloc vtable (see detail/dispatch.h); the destructor frees via
// the same vtable. Rank is fixed at 2 (matrix) or 1 (vector — cols == 1).
//
// Move-only — copying device buffers must be explicit (clone()). The CPU
// backend allocates plain host memory through the same vtable interface so
// the storage layout is uniform across devices; for CPU tensors the typed
// host accessors (host_f32, host_fp16, at, to_host_vector) give ergonomic
// access without a device sync.
struct Tensor {
    void*  data   = nullptr;
    int    rows   = 0;     // rank-1 tensors: rows = N, cols = 1
    int    cols   = 0;
    Dtype  dtype  = Dtype::FP32;
    Device device = Device::CPU;

    Tensor() = default;
    ~Tensor();

    // Copyable + movable. The copy ctor / copy assignment perform a
    // device-aware deep copy — identical to clone() — so a Tensor can be
    // used with value semantics (caches, std::vector storage, by-value
    // params). clone() remains for call sites that want the copy to be
    // explicit. Copying a GPU-resident tensor allocates + copies on-device.
    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // ─── Factories ─────────────────────────────────────────────────────────
    //
    // zeros / empty allocate on the current default device (see runtime.h —
    // controlled by set_default_device() / DeviceScope, or the
    // BROTENSOR_DEFAULT_DEVICE env var). `zeros` memset-zeros the buffer
    // via the backend's memset_zero hook; `empty` leaves contents undefined.
    static Tensor zeros(int r, int c, Dtype dt = Dtype::FP32);
    static Tensor empty(int r, int c, Dtype dt = Dtype::FP32);

    // Explicit-device variants — bypass the thread-local default. Useful for
    // tests, multi-device pipelines, and any code that wants to pin storage
    // to a specific backend regardless of caller policy.
    static Tensor zeros_on(Device, int r, int c, Dtype dt = Dtype::FP32);
    static Tensor empty_on(Device, int r, int c, Dtype dt = Dtype::FP32);

    // Host (CPU) FP32 factories. Always allocate zero-filled storage pinned
    // to Device::CPU regardless of the current default device — a parameter-
    // bearing layer builds its weights on the host, then migrates the whole
    // layer with to(Device). `mat` is a (rows, cols) matrix; `vec` is a
    // rank-1 (n, 1) column vector.
    static Tensor mat(int r, int c) { return zeros_on(Device::CPU, r, c); }
    static Tensor vec(int n)        { return zeros_on(Device::CPU, n, 1); }

    // Host bootstrap. Allocates on the current default device and uploads
    // `r * c` floats (FP32) or uint16_t bit patterns (FP16) from `src`.
    // For non-CPU defaults this performs a host→device copy via the
    // backend's memcpy_h2d hook; for the CPU default it's a plain memcpy.
    static Tensor from_host(const float* src, int r, int c);
    static Tensor from_host_fp16(const uint16_t* src, int r, int c);

    // Variant that pins to a specific device, bypassing the default.
    static Tensor from_host_on(Device, const float* src, int r, int c);
    static Tensor from_host_fp16_on(Device, const uint16_t* src, int r, int c);

    // Non-owning view over an existing backend-resident pointer. The
    // returned tensor's destructor will NOT free `data`. Caller is
    // responsible for lifetime. Mirrors the legacy GpuTensor::view pattern.
    static Tensor view(Device, void* data, int rows, int cols, Dtype = Dtype::FP32);

    // ─── Migration ─────────────────────────────────────────────────────────

    // Returns a fresh tensor on `target` with the same shape/dtype/contents
    // as `*this`. No-op clone() if already on the target device. The source
    // tensor is unchanged. Uses the backend pair's memcpy_h2d / memcpy_d2h /
    // memcpy_d2d hooks as appropriate.
    Tensor to(Device target) const;

    // Device-preserving deep copy.
    Tensor clone() const;

    // ─── Mutators ──────────────────────────────────────────────────────────

    // memset-zero the buffer over bytes(). Dispatches through the backend's
    // memset_zero hook.
    void zero();

    // Reallocates if (r, c, dt) differs from current shape/dtype; leaves
    // contents undefined (call zero() afterwards if needed). Device is
    // preserved. Existing storage is freed.
    void resize(int r, int c, Dtype dt = Dtype::FP32);

    // ─── Accessors ─────────────────────────────────────────────────────────

    int          size()  const { return rows * cols; }
    std::size_t  bytes() const;
    bool         is_host() const { return device == Device::CPU; }
    bool         empty() const   { return data == nullptr || size() == 0; }

    // Host-side typed accessors. Throw std::runtime_error if device != CPU.
    // `host_f32` additionally throws if dtype != FP32; `host_fp16` if
    // dtype != FP16. `host_raw` is dtype-agnostic.
    float*       host_f32_mut();
    const float* host_f32() const;
    uint16_t*       host_fp16_mut();
    const uint16_t* host_fp16() const;
    void*        host_raw_mut();
    const void*  host_raw() const;

    // Element access helpers (host-only, FP32-only — convenience for tests).
    // Throw if device != CPU or dtype != FP32 or indices out of range.
    float& at(int r, int c);
    float  at(int r, int c) const;

    // Host (CPU) FP32 convenience accessors. Thin aliases over the typed
    // host accessors above — they throw via the same checks if device != CPU
    // or dtype != FP32. `ptr` is the raw row-major base pointer; operator()
    // is bounds-checked (r, c) access; operator[] is flat element access.
    float*       ptr()       { return host_f32_mut(); }
    const float* ptr() const { return host_f32(); }
    float& operator()(int r, int c)       { return at(r, c); }
    float  operator()(int r, int c) const { return at(r, c); }
    float& operator[](int i)       { return host_f32_mut()[i]; }
    float  operator[](int i) const { return host_f32()[i]; }

    // ─── Host roundtrip helpers ────────────────────────────────────────────
    //
    // `to_host_vector*` downloads (if on a GPU backend) and returns a
    // std::vector containing the buffer's contents in the matching scalar
    // type. The copy_to_host variants write into a caller-supplied buffer
    // of at least size() elements.
    std::vector<float>    to_host_vector() const;          // FP32 only
    std::vector<uint16_t> to_host_vector_fp16() const;     // FP16 only
    void copy_to_host(float* dst) const;                   // FP32 only
    void copy_to_host_fp16(uint16_t* dst) const;           // FP16 only

private:
    bool owns_ = false;
    void release_();
};

// ─── FP16 ↔ FP32 host-side conversion helpers ──────────────────────────────
//
// Pure-CPU IEEE 754 binary16 conversion. Useful for tests and small
// preprocessing where a GPU roundtrip would be wasteful. Not intended for
// hot loops.
uint16_t fp32_to_fp16_bits(float v);
float    fp16_bits_to_fp32(uint16_t bits);

} // namespace brotensor
