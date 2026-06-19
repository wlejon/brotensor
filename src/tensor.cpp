// brotensor::Tensor implementation.
//
// All allocation, freeing, and inter-device memory traffic routes through the
// backend AllocVTable obtained from detail::alloc_for(device). The Tensor type
// itself is backend-agnostic; CUDA/Metal/CPU specifics live entirely behind
// the vtable indirection.

#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace brotensor {

// ─── env-gated weight-load profiler (BROTENSOR_TIME_LOAD=1) ──────────────────
// Attributes GPU weight-upload wall time to allocation vs H2D copy, accumulated
// across every from_host_*_on call and dumped once at process exit. Purely a
// diagnostic; the fast path (flag unset) is a single getenv-cached bool check.
namespace {
struct LoadProf {
    bool   on   = false;
    double alloc = 0.0, copy = 0.0, bytes = 0.0;
    long   calls = 0;
    LoadProf() { on = std::getenv("BROTENSOR_TIME_LOAD") != nullptr; }
    ~LoadProf() {
        if (!on || calls == 0) return;
        std::fprintf(stderr,
            "[brotensor load] %ld uploads, %.2f GB: alloc %.2f s, h2d %.2f s "
            "(%.2f GB/s copy)\n",
            calls, bytes / 1e9, alloc, copy,
            copy > 0 ? (bytes / 1e9) / copy : 0.0);
    }
};
LoadProf g_loadprof;
inline bool loadprof_on() { return g_loadprof.on; }
}  // namespace

// ─── dtype + device helpers ────────────────────────────────────────────────

int dtype_size_bytes(Dtype dt) {
    switch (dt) {
        case Dtype::FP32:  return 4;
        case Dtype::FP16:  return 2;
        case Dtype::BF16:  return 2;
        case Dtype::INT8:  return 1;
        case Dtype::INT32: return 4;
        case Dtype::F64:   return 8;
        // Quant dtypes are block-addressed; no per-element size.
        case Dtype::Q4_0: case Dtype::Q4_1:
        case Dtype::Q5_0: case Dtype::Q5_1:
        case Dtype::Q8_0: case Dtype::Q8_1:
        case Dtype::Q2_K: case Dtype::Q3_K:
        case Dtype::Q4_K: case Dtype::Q5_K:
        case Dtype::Q6_K: case Dtype::Q8_K:
            return 0;
    }
    return 0;
}

bool dtype_is_quant(Dtype dt) {
    switch (dt) {
        case Dtype::Q4_0: case Dtype::Q4_1:
        case Dtype::Q5_0: case Dtype::Q5_1:
        case Dtype::Q8_0: case Dtype::Q8_1:
        case Dtype::Q2_K: case Dtype::Q3_K:
        case Dtype::Q4_K: case Dtype::Q5_K:
        case Dtype::Q6_K: case Dtype::Q8_K:
            return true;
        default:
            return false;
    }
}

int dtype_block_size(Dtype dt) {
    switch (dt) {
        case Dtype::Q4_0: case Dtype::Q4_1:
        case Dtype::Q5_0: case Dtype::Q5_1:
        case Dtype::Q8_0: case Dtype::Q8_1:
            return 32;
        case Dtype::Q2_K: case Dtype::Q3_K:
        case Dtype::Q4_K: case Dtype::Q5_K:
        case Dtype::Q6_K: case Dtype::Q8_K:
            return 256;
        default:
            return 1;
    }
}

int dtype_block_bytes(Dtype dt) {
    switch (dt) {
        case Dtype::Q4_0: return 18;
        case Dtype::Q4_1: return 20;
        case Dtype::Q5_0: return 22;
        case Dtype::Q5_1: return 24;
        case Dtype::Q8_0: return 34;
        case Dtype::Q8_1: return 36;
        case Dtype::Q2_K: return 82;
        case Dtype::Q3_K: return 110;
        case Dtype::Q4_K: return 144;
        case Dtype::Q5_K: return 176;
        case Dtype::Q6_K: return 210;
        case Dtype::Q8_K: return 292;
        default:          return dtype_size_bytes(dt);
    }
}

std::size_t dtype_storage_bytes(Dtype d, std::int64_t numel) {
    if (numel < 0) {
        throw std::runtime_error("brotensor: dtype_storage_bytes: negative numel");
    }
    if (!dtype_is_quant(d)) {
        return static_cast<std::size_t>(numel) *
               static_cast<std::size_t>(dtype_size_bytes(d));
    }
    const int bs = dtype_block_size(d);
    const int bb = dtype_block_bytes(d);
    if (numel % bs != 0) {
        throw std::runtime_error(
            "brotensor: dtype_storage_bytes: numel not a multiple of block size");
    }
    return (static_cast<std::size_t>(numel) / static_cast<std::size_t>(bs)) *
           static_cast<std::size_t>(bb);
}

const char* device_name(Device d) {
    switch (d) {
        case Device::CPU:   return "CPU";
        case Device::CUDA:  return "CUDA";
        case Device::Metal: return "Metal";
    }
    return "?";
}

// ─── FP16 ↔ FP32 conversion (host-side IEEE 754 binary16) ──────────────────

uint16_t fp32_to_fp16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {
        // Inf or NaN.
        uint16_t out = static_cast<uint16_t>(sign | 0x7C00u);
        if (mant) out |= 0x0200u; // quiet NaN
        return out;
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        const uint32_t round = 1u << (shift - 1);
        uint32_t r = (mant + round) >> shift;
        return static_cast<uint16_t>(sign | r);
    }
    const uint32_t lsb   = (mant >> 13) & 1u;
    const uint32_t round = 0x00001000u + lsb - 1u;
    uint32_t r_mant = (mant + round) >> 13;
    uint32_t r_exp  = static_cast<uint32_t>(exp);
    if (r_mant & 0x400u) {
        r_mant = 0;
        r_exp += 1;
        if (r_exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (r_exp << 10) | r_mant);
}

float fp16_bits_to_fp32(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    uint32_t exp  = (bits >> 10) & 0x1Fu;
    uint32_t mant = bits & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3FFu;
            uint32_t e = (exp + (127 - 15) + 1) & 0xFFu;
            out = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t e = exp + (127 - 15);
        out = sign | (e << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// ─── BF16 ↔ FP32 conversion (host-side bfloat16) ───────────────────────────

uint16_t fp32_to_bf16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    // NaN: keep it a NaN (the rounding add below could otherwise carry a
    // NaN's mantissa into the exponent and produce an infinity).
    if (((x >> 23) & 0xFFu) == 0xFFu && (x & 0x7FFFFFu) != 0) {
        return static_cast<uint16_t>((x >> 16) | 0x0040u);
    }
    // Round to nearest, ties to even: add 0x7FFF + LSB-of-result.
    const uint32_t rounding_bias = 0x7FFFu + ((x >> 16) & 1u);
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}

float bf16_bits_to_fp32(uint16_t bits) {
    const uint32_t x = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ─── internal helpers ──────────────────────────────────────────────────────

namespace {

[[noreturn]] void throw_msg(const std::string& m) {
    throw std::runtime_error(m);
}

void check_host(const Tensor& t, const char* who) {
    if (t.device != Device::CPU) {
        std::string m = "brotensor: ";
        m += who;
        m += ": tensor is on ";
        m += device_name(t.device);
        m += ", not CPU";
        throw_msg(m);
    }
}

// Guard against negative dimensions before they are cast to std::size_t in
// bytes() — an unchecked (size_t)(-1) underflows to an astronomical count
// that sails past the bytes==0 short-circuits and reaches backend_alloc.
void check_dims(int r, int c, const char* who) {
    if (r < 0 || c < 0) {
        std::string m = "brotensor: ";
        m += who;
        m += ": negative dimension";
        throw_msg(m);
    }
}

void check_dtype(const Tensor& t, Dtype expected, const char* who) {
    if (t.dtype != expected) {
        std::string m = "brotensor: ";
        m += who;
        m += ": dtype mismatch";
        throw_msg(m);
    }
}

void* backend_alloc(Device d, std::size_t bytes) {
    if (bytes == 0) return nullptr;
    return detail::alloc_for(d).alloc(bytes);
}

void backend_free(Device d, void* p) {
    if (!p) return;
    detail::alloc_for(d).free(p);
}

void backend_zero(Device d, void* p, std::size_t bytes) {
    if (bytes == 0 || !p) return;
    detail::alloc_for(d).memset_zero(p, bytes);
}

} // namespace

// ─── Tensor lifetime ───────────────────────────────────────────────────────

Tensor::~Tensor() {
    release_();
}

void Tensor::release_() {
    if (owns_ && data) {
        // alloc_for can throw if the backend was somehow un-registered between
        // alloc and free; let it propagate — better than leaking a different
        // backend's pointer.
        backend_free(device, data);
    }
    data = nullptr;
    owns_ = false;
    cap_bytes_ = 0;
}

Tensor::Tensor(Tensor&& o) noexcept
    : data(o.data), rows(o.rows), cols(o.cols),
      dtype(o.dtype), device(o.device), owns_(o.owns_),
      cap_bytes_(o.cap_bytes_) {
    o.data = nullptr;
    o.rows = 0;
    o.cols = 0;
    o.owns_ = false;
    o.cap_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        release_();
        data   = o.data;
        rows    = o.rows;
        cols    = o.cols;
        dtype   = o.dtype;
        device  = o.device;
        owns_   = o.owns_;
        cap_bytes_ = o.cap_bytes_;
        o.data = nullptr;
        o.rows  = 0;
        o.cols  = 0;
        o.owns_ = false;
        o.cap_bytes_ = 0;
    }
    return *this;
}

// Copy ctor / assignment: device-aware deep copy via clone(). Implemented in
// terms of the move assignment so all the ownership bookkeeping lives in one
// place.
Tensor::Tensor(const Tensor& o) {
    *this = o.clone();
}

Tensor& Tensor::operator=(const Tensor& o) {
    if (this != &o) *this = o.clone();
    return *this;
}

std::size_t Tensor::bytes() const {
    return dtype_storage_bytes(
        dtype,
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols));
}

// ─── Factories ─────────────────────────────────────────────────────────────

Tensor Tensor::empty_on(Device d, int r, int c, Dtype dt) {
    check_dims(r, c, "empty_on");
    Tensor t;
    t.device = d;
    t.dtype  = dt;
    t.rows   = r;
    t.cols   = c;
    t.data  = backend_alloc(d, t.bytes());
    t.owns_  = (t.data != nullptr);
    t.cap_bytes_ = t.owns_ ? t.bytes() : 0;
    return t;
}

Tensor Tensor::zeros_on(Device d, int r, int c, Dtype dt) {
    Tensor t = empty_on(d, r, c, dt);
    backend_zero(d, t.data, t.bytes());
    return t;
}

Tensor Tensor::empty(int r, int c, Dtype dt) {
    return empty_on(default_device(), r, c, dt);
}

Tensor Tensor::zeros(int r, int c, Dtype dt) {
    return zeros_on(default_device(), r, c, dt);
}

namespace {
// Shared upload core for from_host_*_on. When BROTENSOR_TIME_LOAD is set it
// attributes host wall time to the empty_on (alloc enqueue) and the H2D copy.
Tensor from_host_typed(Device d, const void* src, int r, int c, Dtype dt) {
    using clock = std::chrono::steady_clock;
    const bool prof = loadprof_on() && d != Device::CPU;
    auto t0 = clock::now();
    Tensor t = Tensor::empty_on(d, r, c, dt);
    const std::size_t n = t.bytes();
    if (n == 0) return t;
    auto t1 = clock::now();
    if (d == Device::CPU) {
        std::memcpy(t.data, src, n);
    } else {
        detail::alloc_for(d).memcpy_h2d(t.data, src, n);
    }
    if (prof) {
        auto t2 = clock::now();
        g_loadprof.alloc += std::chrono::duration<double>(t1 - t0).count();
        g_loadprof.copy  += std::chrono::duration<double>(t2 - t1).count();
        g_loadprof.bytes += static_cast<double>(n);
        ++g_loadprof.calls;
    }
    return t;
}
}  // namespace

Tensor Tensor::from_host_on(Device d, const float* src, int r, int c) {
    return from_host_typed(d, src, r, c, Dtype::FP32);
}

Tensor Tensor::from_host_fp16_on(Device d, const uint16_t* src, int r, int c) {
    return from_host_typed(d, src, r, c, Dtype::FP16);
}

Tensor Tensor::from_host_bf16_on(Device d, const uint16_t* src, int r, int c) {
    return from_host_typed(d, src, r, c, Dtype::BF16);
}

Tensor Tensor::from_host_int8_on(Device d, const int8_t* src, int r, int c) {
    Tensor t = empty_on(d, r, c, Dtype::INT8);
    const std::size_t n = t.bytes();
    if (n == 0) return t;
    if (d == Device::CPU) {
        std::memcpy(t.data, src, n);
    } else {
        detail::alloc_for(d).memcpy_h2d(t.data, src, n);
    }
    return t;
}

Tensor Tensor::from_host(const float* src, int r, int c) {
    return from_host_on(default_device(), src, r, c);
}

Tensor Tensor::from_host_fp16(const uint16_t* src, int r, int c) {
    return from_host_fp16_on(default_device(), src, r, c);
}

Tensor Tensor::from_host_bf16(const uint16_t* src, int r, int c) {
    return from_host_bf16_on(default_device(), src, r, c);
}

Tensor Tensor::from_host_int8(const int8_t* src, int r, int c) {
    return from_host_int8_on(default_device(), src, r, c);
}

Tensor Tensor::view(Device d, void* data, int r, int c, Dtype dt) {
    Tensor t;
    t.device = d;
    t.dtype  = dt;
    t.rows   = r;
    t.cols   = c;
    t.data  = data;
    t.owns_  = false;
    return t;
}

// ─── Migration ─────────────────────────────────────────────────────────────

Tensor Tensor::clone() const {
    Tensor t = empty_on(device, rows, cols, dtype);
    const std::size_t n = bytes();
    if (n == 0 || !data) return t;
    if (device == Device::CPU) {
        std::memcpy(t.data, data, n);
    } else {
        detail::alloc_for(device).memcpy_d2d(t.data, data, n);
    }
    return t;
}

Tensor Tensor::to(Device target) const {
    if (target == device) return clone();
    Tensor t = empty_on(target, rows, cols, dtype);
    const std::size_t n = bytes();
    if (n == 0 || !data) return t;

    if (device == Device::CPU) {
        // CPU → GPU.
        detail::alloc_for(target).memcpy_h2d(t.data, data, n);
    } else if (target == Device::CPU) {
        // GPU → CPU.
        detail::alloc_for(device).memcpy_d2h(t.data, data, n);
    } else {
        // GPU → different GPU backend. Bounce through host.
        std::vector<unsigned char> staging(n);
        detail::alloc_for(device).memcpy_d2h(staging.data(), data, n);
        detail::alloc_for(target).memcpy_h2d(t.data, staging.data(), n);
    }
    return t;
}

// ─── Mutators ──────────────────────────────────────────────────────────────

void Tensor::zero() {
    backend_zero(device, data, bytes());
}

void Tensor::resize(int r, int c, Dtype dt) {
    check_dims(r, c, "resize");
    const std::size_t new_bytes = dtype_storage_bytes(
        dt,
        static_cast<std::int64_t>(r) * static_cast<std::int64_t>(c));
    if (r == rows && c == cols && dt == dtype && data != nullptr) return;
    // A non-owning view over real storage cannot be reshaped: reallocating
    // would silently allocate fresh owned memory and sever the view, leaving
    // callers with a tensor that no longer aliases what they passed to
    // view(). Reject it explicitly rather than converting it in place.
    // (A default-constructed / released tensor — owns_ == false but
    // data == nullptr — is not a view and resizes normally.)
    if (!owns_ && data != nullptr) {
        throw_msg("brotensor: resize: cannot reshape a non-owning view; "
                  "allocate a fresh tensor or re-view() with the new shape");
    }
    // Keep the existing storage whenever the new shape fits — the buffer
    // stabilises at its high-water mark and its device pointer stays put
    // (required when the op sequence using it is CUDA-graph captured).
    if (new_bytes > cap_bytes_ || !owns_) {
        release_();
        data = backend_alloc(device, new_bytes);
        owns_ = (data != nullptr);
        cap_bytes_ = owns_ ? new_bytes : 0;
    }
    rows  = r;
    cols  = c;
    dtype = dt;
}

// ─── Host accessors ────────────────────────────────────────────────────────

float* Tensor::host_f32_mut() {
    check_host(*this, "host_f32_mut");
    check_dtype(*this, Dtype::FP32, "host_f32_mut");
    return static_cast<float*>(data);
}
const float* Tensor::host_f32() const {
    check_host(*this, "host_f32");
    check_dtype(*this, Dtype::FP32, "host_f32");
    return static_cast<const float*>(data);
}

uint16_t* Tensor::host_fp16_mut() {
    check_host(*this, "host_fp16_mut");
    check_dtype(*this, Dtype::FP16, "host_fp16_mut");
    return static_cast<uint16_t*>(data);
}
const uint16_t* Tensor::host_fp16() const {
    check_host(*this, "host_fp16");
    check_dtype(*this, Dtype::FP16, "host_fp16");
    return static_cast<const uint16_t*>(data);
}

uint16_t* Tensor::host_bf16_mut() {
    check_host(*this, "host_bf16_mut");
    check_dtype(*this, Dtype::BF16, "host_bf16_mut");
    return static_cast<uint16_t*>(data);
}
const uint16_t* Tensor::host_bf16() const {
    check_host(*this, "host_bf16");
    check_dtype(*this, Dtype::BF16, "host_bf16");
    return static_cast<const uint16_t*>(data);
}

void* Tensor::host_raw_mut() {
    check_host(*this, "host_raw_mut");
    return data;
}
const void* Tensor::host_raw() const {
    check_host(*this, "host_raw");
    return data;
}

float& Tensor::at(int r, int c) {
    check_host(*this, "at");
    check_dtype(*this, Dtype::FP32, "at");
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw_msg("brotensor: at: index out of range");
    }
    return static_cast<float*>(data)[static_cast<std::size_t>(r) * cols + c];
}

float Tensor::at(int r, int c) const {
    check_host(*this, "at");
    check_dtype(*this, Dtype::FP32, "at");
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw_msg("brotensor: at: index out of range");
    }
    return static_cast<const float*>(data)[static_cast<std::size_t>(r) * cols + c];
}

// ─── Host roundtrip ────────────────────────────────────────────────────────

std::vector<float> Tensor::to_host_vector() const {
    if (dtype != Dtype::FP32) {
        throw_msg("brotensor: to_host_vector: dtype not FP32");
    }
    std::vector<float> out(static_cast<std::size_t>(rows) * cols);
    const std::size_t n = bytes();
    if (n == 0) return out;
    if (device == Device::CPU) {
        std::memcpy(out.data(), data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(out.data(), data, n);
    }
    return out;
}

std::vector<uint16_t> Tensor::to_host_vector_fp16() const {
    if (dtype != Dtype::FP16) {
        throw_msg("brotensor: to_host_vector_fp16: dtype not FP16");
    }
    std::vector<uint16_t> out(static_cast<std::size_t>(rows) * cols);
    const std::size_t n = bytes();
    if (n == 0) return out;
    if (device == Device::CPU) {
        std::memcpy(out.data(), data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(out.data(), data, n);
    }
    return out;
}

std::vector<uint16_t> Tensor::to_host_vector_bf16() const {
    if (dtype != Dtype::BF16) {
        throw_msg("brotensor: to_host_vector_bf16: dtype not BF16");
    }
    std::vector<uint16_t> out(static_cast<std::size_t>(rows) * cols);
    const std::size_t n = bytes();
    if (n == 0) return out;
    if (device == Device::CPU) {
        std::memcpy(out.data(), data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(out.data(), data, n);
    }
    return out;
}

void Tensor::copy_to_host(float* dst) const {
    if (dtype != Dtype::FP32) {
        throw_msg("brotensor: copy_to_host: dtype not FP32");
    }
    const std::size_t n = bytes();
    if (n == 0) return;
    if (device == Device::CPU) {
        std::memcpy(dst, data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(dst, data, n);
    }
}

void Tensor::copy_to_host_fp16(uint16_t* dst) const {
    if (dtype != Dtype::FP16) {
        throw_msg("brotensor: copy_to_host_fp16: dtype not FP16");
    }
    const std::size_t n = bytes();
    if (n == 0) return;
    if (device == Device::CPU) {
        std::memcpy(dst, data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(dst, data, n);
    }
}

void Tensor::copy_to_host_bf16(uint16_t* dst) const {
    if (dtype != Dtype::BF16) {
        throw_msg("brotensor: copy_to_host_bf16: dtype not BF16");
    }
    const std::size_t n = bytes();
    if (n == 0) return;
    if (device == Device::CPU) {
        std::memcpy(dst, data, n);
    } else {
        detail::alloc_for(device).memcpy_d2h(dst, data, n);
    }
}

} // namespace brotensor
