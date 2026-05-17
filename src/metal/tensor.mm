#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/device_buffer.h>

#import "internal.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace brotensor {

// ─── Pointer → MTLBuffer pool ──────────────────────────────────────────────
//
// GpuTensor's public API is just `float* data + rows + cols + dtype`. On
// Metal we allocate MTLBuffers with shared storage (unified memory on Apple
// Silicon) and use `[buf contents]` as the `data` pointer. To free the
// buffer when the GpuTensor is destroyed, we keep a global pointer→buffer
// map.

namespace metal_impl {

namespace {
struct Pool {
    std::mutex mu;
    std::unordered_map<const void*, void*> map; // void* holds an Obj-C +1 retain
};
Pool& pool() { static Pool p; return p; }
} // namespace

void pool_register(void* data_ptr, id<MTLBuffer> buf) {
    if (!data_ptr || !buf) return;
    auto& P = pool();
    void* retained = (__bridge_retained void*)buf;
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        id<MTLBuffer> prior = (__bridge_transfer id<MTLBuffer>)it->second;
        (void)prior;
        it->second = retained;
    } else {
        P.map.emplace(data_ptr, retained);
    }
}

id<MTLBuffer> pool_lookup(const void* data_ptr) {
    if (!data_ptr) return nil;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        return (__bridge id<MTLBuffer>)it->second;
    }
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return buf;
        }
    }
    return nil;
}

NSUInteger pool_lookup_offset(const void* data_ptr) {
    if (!data_ptr) return 0;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) return 0;
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return static_cast<NSUInteger>(p - base);
        }
    }
    return 0;
}

void pool_release(void* data_ptr) {
    if (!data_ptr) return;
    auto& P = pool();
    void* retained = nullptr;
    {
        std::lock_guard<std::mutex> lk(P.mu);
        auto it = P.map.find(data_ptr);
        if (it == P.map.end()) return;
        retained = it->second;
        P.map.erase(it);
    }
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)retained;
    (void)buf;
}

} // namespace metal_impl

namespace {

void* allocate_shared_bytes(std::size_t nbytes) {
    if (nbytes == 0) return nullptr;
    cuda_init();
    @autoreleasepool {
        id<MTLBuffer> buf = [metal_impl::device()
            newBufferWithLength:nbytes
                        options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error("Metal: failed to allocate MTLBuffer");
        }
        void* p = [buf contents];
        metal_impl::pool_register(p, buf);
        return p;
    }
}

} // namespace

GpuTensor::GpuTensor(int r, int c, Dtype dt)
    : data(nullptr), rows(r), cols(c), dtype(dt), owns_(false) {
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    if (n == 0) return;
    data = static_cast<float*>(allocate_shared_bytes(n * dtype_size_bytes(dt)));
    owns_ = true;
}

GpuTensor::~GpuTensor() {
    release_();
}

void GpuTensor::release_() {
    if (owns_ && data) {
        metal_impl::pool_release(data);
    }
    data = nullptr;
    rows = 0;
    cols = 0;
    dtype = Dtype::FP32;
    owns_ = false;
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : data(other.data), rows(other.rows), cols(other.cols),
      dtype(other.dtype), owns_(other.owns_) {
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    other.dtype = Dtype::FP32;
    other.owns_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        release_();
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        dtype = other.dtype;
        owns_ = other.owns_;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.dtype = Dtype::FP32;
        other.owns_ = false;
    }
    return *this;
}

void GpuTensor::zero() {
    if (size() == 0) return;
    std::memset(data, 0, bytes());
}

void GpuTensor::resize(int r, int c, Dtype dt) {
    if (r == rows && c == cols && dt == dtype && data != nullptr) return;
    release_();
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    rows = r;
    cols = c;
    dtype = dt;
    if (n == 0) return;
    data = static_cast<float*>(allocate_shared_bytes(n * dtype_size_bytes(dt)));
    owns_ = true;
}

GpuTensor GpuTensor::clone() const {
    GpuTensor out;
    if (size() == 0) {
        out.rows = rows;
        out.cols = cols;
        out.dtype = dtype;
        return out;
    }
    out.resize(rows, cols, dtype);
    std::memcpy(out.data, data, bytes());
    return out;
}

GpuTensor GpuTensor::view(float* data, int rows, int cols) {
    GpuTensor t;
    t.data = data;
    t.rows = rows;
    t.cols = cols;
    t.dtype = Dtype::FP32;
    t.owns_ = false;
    return t;
}

GpuTensor GpuTensor::view_fp16(uint16_t* data, int rows, int cols) {
    GpuTensor t;
    t.data = reinterpret_cast<float*>(data);
    t.rows = rows;
    t.cols = cols;
    t.dtype = Dtype::FP16;
    t.owns_ = false;
    return t;
}

void upload(const float* host, int rows, int cols, GpuTensor& dst) {
    if (dst.rows != rows || dst.cols != cols || dst.dtype != Dtype::FP32) {
        dst.resize(rows, cols, Dtype::FP32);
    }
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (n == 0) return;
    std::memcpy(dst.data, host, n * sizeof(float));
}

void download(const GpuTensor& src, float* host) {
    if (src.size() == 0) return;
    if (src.dtype != Dtype::FP32) {
        throw std::runtime_error("brotensor::download: src is not FP32 (use download_fp16)");
    }
    std::memcpy(host, src.data,
                static_cast<size_t>(src.size()) * sizeof(float));
}

void upload_fp16(const uint16_t* host, int rows, int cols, GpuTensor& dst) {
    if (dst.rows != rows || dst.cols != cols || dst.dtype != Dtype::FP16) {
        dst.resize(rows, cols, Dtype::FP16);
    }
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (n == 0) return;
    std::memcpy(dst.data, host, n * sizeof(uint16_t));
}

void download_fp16(const GpuTensor& src, uint16_t* host) {
    if (src.size() == 0) return;
    if (src.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor::download_fp16: src is not FP16");
    }
    std::memcpy(host, src.data,
                static_cast<size_t>(src.size()) * sizeof(uint16_t));
}

// ─── Host-side IEEE 754 binary16 helpers ───────────────────────────────────

uint16_t fp32_to_fp16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {
        uint16_t out = static_cast<uint16_t>(sign | 0x7C00u);
        if (mant) out |= 0x0200u;
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
    return static_cast<uint16_t>(sign | (r_exp << 10) | (r_mant & 0x3FFu));
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

} // namespace brotensor
