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
// Element sizes are fixed: FP32 = 4 bytes, FP16 = 2 bytes, BF16 = 2 bytes,
// INT8 = 1 byte, INT32 = 4 bytes. Allocation, clone, zero, and resize all use
// dtype-aware byte counts. BF16 (IEEE 754 bfloat16 — the high 16 bits of an
// FP32) is an arithmetic dtype carried only by the GPU backends; like FP16 it
// is stored as a uint16_t bit pattern on the host. Arithmetic ops dispatch on
// FP32/FP16/BF16. INT8 is currently only carried by weight-only quantised ops
// (W8A16 matmul/conv2d). INT32 is likewise a pure storage carrier — used for
// device-resident index/offset buffers (e.g. per-head offset tables for
// softmax_xent_fused_batched); no arithmetic op dispatches on it.
enum class Dtype : int {
    FP32  = 0,
    FP16  = 1,
    INT8  = 2,
    INT32 = 3,
    BF16  = 4,
    F64   = 5,
    // GGUF legacy quants — 32-element blocks, opaque storage carriers only.
    Q4_0  = 10,
    Q4_1  = 11,
    Q5_0  = 12,
    Q5_1  = 13,
    Q8_0  = 14,
    Q8_1  = 15,
    // GGUF K-quants — 256-element superblocks, opaque storage carriers only.
    Q2_K  = 20,
    Q3_K  = 21,
    Q4_K  = 22,
    Q5_K  = 23,
    Q6_K  = 24,
    Q8_K  = 25,
};

// Bytes per scalar element. Returns 0 for quant dtypes (they aren't
// element-addressable — use dtype_storage_bytes() instead).
int dtype_size_bytes(Dtype);

// Elements per block. 1 for non-quant types; 32 for the legacy GGUF quants
// (Q4_0..Q8_1); 256 for the K-quants.
int dtype_block_size(Dtype);

// Bytes per block. Equals dtype_size_bytes(d) for non-quant dtypes; for quant
// dtypes it's the on-disk block size (e.g. Q4_K = 144).
int dtype_block_bytes(Dtype);

// Byte count for a tensor of `numel` elements stored as `d`. For non-quant
// types it's numel * dtype_size_bytes(d). For quant types `numel` must be a
// multiple of dtype_block_size(d) (throws std::runtime_error otherwise) and
// the result is (numel / block_size) * block_bytes.
std::size_t dtype_storage_bytes(Dtype d, std::int64_t numel);

// True iff `d` is a quant block carrier (Q*_*).
bool dtype_is_quant(Dtype);

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
// Copyable and movable: the copy ctor / copy assignment perform a
// device-aware deep copy (identical to clone()); move transfers ownership
// of the underlying buffer. Copying a GPU-resident tensor therefore
// allocates and copies on-device — pass by reference on hot paths and use
// clone() where the copy should be explicit. The CPU backend allocates
// plain host memory through the same vtable interface so the storage layout
// is uniform across devices; for CPU tensors the typed host accessors
// (host_f32, host_fp16, at, to_host_vector) give ergonomic access without a
// device sync.
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
    static Tensor from_host_bf16(const uint16_t* src, int r, int c);
    // INT8 weights (W8A16): `r * c` int8_t values, e.g. the output of
    // quantize_int8_per_row_host paired with FP32 per-row dequant scales.
    static Tensor from_host_int8(const int8_t* src, int r, int c);

    // Variant that pins to a specific device, bypassing the default.
    static Tensor from_host_on(Device, const float* src, int r, int c);
    static Tensor from_host_fp16_on(Device, const uint16_t* src, int r, int c);
    static Tensor from_host_bf16_on(Device, const uint16_t* src, int r, int c);
    static Tensor from_host_int8_on(Device, const int8_t* src, int r, int c);

    // Non-owning view over an existing backend-resident pointer. The
    // returned tensor's destructor will NOT free `data`. Caller is
    // responsible for lifetime. Mirrors the legacy GpuTensor::view pattern.
    static Tensor view(Device, void* data, int rows, int cols, Dtype = Dtype::FP32);

    // Dtype-agnostic host bootstrap: allocates on `target` and copies
    // `nbytes` raw bytes from `src` — a plain memcpy for Device::CPU, a
    // single memcpy_h2d otherwise. Unlike from_host*_on, this works for any
    // Dtype including the opaque GGUF block-quant carriers, since it copies
    // bytes() rather than interpreting elements. `nbytes` must equal the
    // resulting tensor's bytes() (i.e. dtype_storage_bytes(dt, r*c)).
    static Tensor from_raw_bytes_on(Device target, const void* src,
                                     int r, int c, Dtype dt,
                                     std::size_t nbytes);

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

    // Reshapes to (r, c, dt); leaves contents undefined (call zero()
    // afterwards if needed). Device is preserved. Storage is kept whenever
    // the requested shape fits the existing allocation (capacity = the
    // high-water mark of this tensor's past sizes), so a scratch buffer
    // cycling through shapes stabilises at its largest size instead of
    // reallocating every call — which also keeps its device pointer stable,
    // a requirement for CUDA-graph-captured op sequences. Reallocates only
    // when growing past capacity. A no-op if the shape and dtype already
    // match. Throws std::runtime_error on a negative dimension, or if called
    // on a non-owning view (a tensor from view()) — reshaping a view would
    // silently sever it, so allocate a fresh tensor instead.
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
    uint16_t*       host_bf16_mut();
    const uint16_t* host_bf16() const;
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
    std::vector<uint16_t> to_host_vector_bf16() const;     // BF16 only
    void copy_to_host(float* dst) const;                   // FP32 only
    void copy_to_host_fp16(uint16_t* dst) const;           // FP16 only
    void copy_to_host_bf16(uint16_t* dst) const;           // BF16 only

private:
    bool owns_ = false;
    // Bytes actually allocated behind `data` when owns_ is true — resize()
    // keeps the existing storage whenever the requested size fits, so the
    // capacity is the high-water mark of past sizes. 0 for views, released,
    // and default-constructed tensors.
    std::size_t cap_bytes_ = 0;
    void release_();
};

// ─── FP16 / BF16 ↔ FP32 host-side conversion helpers ───────────────────────
//
// Pure-CPU conversion. `fp16` is IEEE 754 binary16; `bf16` is bfloat16 — the
// high 16 bits of an FP32 with round-to-nearest-even. Useful for tests and
// small preprocessing where a GPU roundtrip would be wasteful. Not intended
// for hot loops.
uint16_t fp32_to_fp16_bits(float v);
float    fp16_bits_to_fp32(uint16_t bits);
uint16_t fp32_to_bf16_bits(float v);
float    bf16_bits_to_fp32(uint16_t bits);

} // namespace brotensor
