#include "native_tensor_decl.h"
#include "api_internal.h"
#include <vector>

using namespace brotensor::api;

namespace {
thread_local std::vector<float> tl_download_f32;
thread_local std::vector<uint16_t> tl_download_u16;
thread_local std::vector<int8_t> tl_download_i8;
} // namespace

extern "C" {

void* bro_tensor_GpuTensor_ctor(void) {
    return new GpuTensorHandle();
}

void bro_tensor_GpuTensor_dtor(void* self) {
    delete static_cast<GpuTensorHandle*>(self);
}

void* bro_tensor_createTensor(int32_t rows, int32_t cols, const char* dtype) {
    auto* h = new GpuTensorHandle();
    auto dt = parseDtype(dtype);
    if (rows > 0 && cols > 0) {
        h->tensor = brotensor::Tensor::zeros_on(brotensor::default_device(), rows, cols, dt);
    }
    return h;
}

int32_t bro_tensor_GpuTensor_rows_get(void* self) {
    auto* t = toTensor(self);
    return t ? t->rows : 0;
}

int32_t bro_tensor_GpuTensor_cols_get(void* self) {
    auto* t = toTensor(self);
    return t ? t->cols : 0;
}

int32_t bro_tensor_GpuTensor_size_get(void* self) {
    auto* t = toTensor(self);
    return t ? t->size() : 0;
}

int32_t bro_tensor_GpuTensor_bytes_get(void* self) {
    auto* t = toTensor(self);
    return t ? static_cast<int32_t>(t->bytes()) : 0;
}

void bro_tensor_GpuTensor_zero(void* self) {
    auto* t = toTensor(self);
    if (t && !t->empty()) {
        t->zero();
    }
}

void bro_tensor_GpuTensor_resize(void* self, int32_t rows, int32_t cols, const char* dtype) {
    auto* t = toTensor(self);
    if (!t) return;
    auto dt = parseDtype(dtype);
    if (rows < 0 || cols < 0) return;
    t->resize(rows, cols, dt);
}

const char* bro_tensor_GpuTensor_dtype(void* self) {
    auto* t = toTensor(self);
    return t ? dtypeToString(t->dtype) : "fp32";
}

void* bro_tensor_GpuTensor_clone(void* self) {
    auto* h = new GpuTensorHandle();
    auto* t = toTensor(self);
    if (t) {
        h->tensor = t->clone();
    }
    return h;
}

void bro_tensor_GpuTensor_upload(void* self, const float* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = t->rows;
    int c = t->cols;
    if (r * c != static_cast<int>(data_len)) {
        r = static_cast<int>(data_len);
        c = 1;
    }
    *t = brotensor::Tensor::from_host_on(brotensor::default_device(), data, r, c);
}

void bro_tensor_GpuTensor_download(void* self, bronze_native_buffer* out) {
    if (!out) return;
    auto* t = toTensor(self);
    if (!t || t->empty()) {
        out->data = nullptr;
        out->length = 0;
        out->release = nullptr;
        out->ctx = nullptr;
        return;
    }
    if (t->dtype == brotensor::Dtype::FP32) {
        tl_download_f32.resize(static_cast<size_t>(t->size()));
        t->copy_to_host(tl_download_f32.data());
        out->data = tl_download_f32.data();
        out->length = static_cast<uint32_t>(tl_download_f32.size());
    } else {
        auto host_t = t->to(brotensor::Device::CPU);
        brotensor::Tensor fp32_t = brotensor::Tensor::empty(t->rows, t->cols, brotensor::Dtype::FP32);
        brotensor::cast(host_t, fp32_t, brotensor::Dtype::FP32);
        tl_download_f32.resize(static_cast<size_t>(fp32_t.size()));
        fp32_t.copy_to_host(tl_download_f32.data());
        out->data = tl_download_f32.data();
        out->length = static_cast<uint32_t>(tl_download_f32.size());
    }
    out->release = nullptr;
    out->ctx = nullptr;
}

void bro_tensor_GpuTensor_uploadFp16(void* self, const uint16_t* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = t->rows;
    int c = t->cols;
    if (r * c != static_cast<int>(data_len)) {
        r = static_cast<int>(data_len);
        c = 1;
    }
    *t = brotensor::Tensor::from_host_fp16_on(brotensor::default_device(), data, r, c);
}

void bro_tensor_GpuTensor_downloadFp16(void* self, bronze_native_buffer* out) {
    if (!out) return;
    auto* t = toTensor(self);
    if (!t || t->empty()) {
        out->data = nullptr;
        out->length = 0;
        out->release = nullptr;
        out->ctx = nullptr;
        return;
    }
    if (t->dtype == brotensor::Dtype::FP16) {
        tl_download_u16.resize(static_cast<size_t>(t->size()));
        t->copy_to_host_fp16(tl_download_u16.data());
        out->data = tl_download_u16.data();
        out->length = static_cast<uint32_t>(tl_download_u16.size());
    } else {
        auto host_t = t->to(brotensor::Device::CPU);
        brotensor::Tensor fp16_t = brotensor::Tensor::empty(t->rows, t->cols, brotensor::Dtype::FP16);
        brotensor::cast(host_t, fp16_t, brotensor::Dtype::FP16);
        tl_download_u16.resize(static_cast<size_t>(fp16_t.size()));
        fp16_t.copy_to_host_fp16(tl_download_u16.data());
        out->data = tl_download_u16.data();
        out->length = static_cast<uint32_t>(tl_download_u16.size());
    }
    out->release = nullptr;
    out->ctx = nullptr;
}

void bro_tensor_GpuTensor_uploadInt8(void* self, const int8_t* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = t->rows;
    int c = t->cols;
    if (r * c != static_cast<int>(data_len)) {
        r = static_cast<int>(data_len);
        c = 1;
    }
    *t = brotensor::Tensor::from_host_int8_on(brotensor::default_device(), data, r, c);
}

void bro_tensor_GpuTensor_downloadInt8(void* self, bronze_native_buffer* out) {
    if (!out) return;
    auto* t = toTensor(self);
    if (!t || t->empty()) {
        out->data = nullptr;
        out->length = 0;
        out->release = nullptr;
        out->ctx = nullptr;
        return;
    }
    auto host_t = t->to(brotensor::Device::CPU);
    tl_download_i8.resize(host_t.bytes());
    std::memcpy(tl_download_i8.data(), host_t.host_raw(), host_t.bytes());
    out->data = tl_download_i8.data();
    out->length = static_cast<uint32_t>(tl_download_i8.size());
    out->release = nullptr;
    out->ctx = nullptr;
}

bool bro_tensor_available_get(void) {
    return true;
}

const char* bro_tensor_backend_get(void) {
    return brotensor::device_name(brotensor::default_device());
}

void bro_tensor_init(void) {
    brotensor::init();
}

void bro_tensor_sync(void) {
    brotensor::sync_all();
}

} // extern "C"
