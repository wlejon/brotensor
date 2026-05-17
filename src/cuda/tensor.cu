#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/device_buffer.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace brotensor {

GpuTensor::GpuTensor(int r, int c, Dtype dt)
    : data(nullptr), rows(r), cols(c), dtype(dt), owns_(false) {
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    if (n == 0) return;
    cuda_init();
    void* p = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&p, n * static_cast<size_t>(dtype_size_bytes(dt))));
    data = static_cast<float*>(p);
    owns_ = true;
}

GpuTensor::~GpuTensor() {
    release_();
}

void GpuTensor::release_() {
    if (owns_ && data) {
        cudaFree(data);
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
    BROTENSOR_CUDA_CHECK(cudaMemset(data, 0, bytes()));
}

void GpuTensor::resize(int r, int c, Dtype dt) {
    if (r == rows && c == cols && dt == dtype && data != nullptr) return;
    release_();
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    rows = r;
    cols = c;
    dtype = dt;
    if (n == 0) return;
    cuda_init();
    void* p = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&p, n * static_cast<size_t>(dtype_size_bytes(dt))));
    data = static_cast<float*>(p);
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
    BROTENSOR_CUDA_CHECK(cudaMemcpy(out.data, data, bytes(),
                              cudaMemcpyDeviceToDevice));
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
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst.data, host, n * sizeof(float),
                                    cudaMemcpyHostToDevice));
}

void download(const GpuTensor& src, float* host) {
    if (src.size() == 0) return;
    if (src.dtype != Dtype::FP32) {
        throw std::runtime_error("brotensor::download: src is not FP32 (use download_fp16)");
    }
    BROTENSOR_CUDA_CHECK(cudaMemcpy(host, src.data,
                                    static_cast<size_t>(src.size()) * sizeof(float),
                                    cudaMemcpyDeviceToHost));
}

void upload_fp16(const uint16_t* host, int rows, int cols, GpuTensor& dst) {
    if (dst.rows != rows || dst.cols != cols || dst.dtype != Dtype::FP16) {
        dst.resize(rows, cols, Dtype::FP16);
    }
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (n == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(dst.data, host, n * sizeof(uint16_t),
                                    cudaMemcpyHostToDevice));
}

void download_fp16(const GpuTensor& src, uint16_t* host) {
    if (src.size() == 0) return;
    if (src.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor::download_fp16: src is not FP16");
    }
    BROTENSOR_CUDA_CHECK(cudaMemcpy(host, src.data,
                                    static_cast<size_t>(src.size()) * sizeof(uint16_t),
                                    cudaMemcpyDeviceToHost));
}

// ─── Host-side IEEE 754 binary16 helpers ───────────────────────────────────

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
        // Overflow → Inf.
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) {
        // Subnormal or underflow to zero.
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        const uint32_t round = 1u << (shift - 1);
        uint32_t r = (mant + round) >> shift;
        return static_cast<uint16_t>(sign | r);
    }
    // Normal — round-to-nearest-even on the discarded 13 bits.
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
            // Subnormal — normalize.
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

// ─── DeviceBuffer<T> backend hooks (CUDA) ──────────────────────────────────

void* device_alloc_bytes(std::size_t bytes, void** native_out) {
    if (native_out) *native_out = nullptr; // unused on CUDA
    if (bytes == 0) return nullptr;
    cuda_init();
    void* p = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

void device_free_bytes(void* device_ptr, void* /*native*/) {
    if (device_ptr) cudaFree(device_ptr);
}

void device_upload_bytes(void* device_ptr, const void* host, std::size_t bytes) {
    if (bytes == 0) return;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(device_ptr, host, bytes, cudaMemcpyHostToDevice));
}

} // namespace brotensor
