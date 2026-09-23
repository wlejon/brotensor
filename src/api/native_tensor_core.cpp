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
        if (static_cast<int64_t>(rows) * cols > INT32_MAX) {
            setError("createTensor: rows*cols exceeds 2^31-1 elements");
            return h;
        }
        BROTENSOR_API_TRY
            h->tensor = brotensor::Tensor::zeros_on(brotensor::default_device(), rows, cols, dt);
        BROTENSOR_API_CATCH("createTensor")
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
        BROTENSOR_API_TRY
            t->zero();
        BROTENSOR_API_CATCH("zero")
    }
}

void bro_tensor_GpuTensor_resize(void* self, int32_t rows, int32_t cols, const char* dtype) {
    auto* t = toTensor(self);
    if (!t) return;
    auto dt = parseDtype(dtype);
    if (rows < 0 || cols < 0) {
        setError("resize: negative dimension");
        return;
    }
    if (static_cast<int64_t>(rows) * cols > INT32_MAX) {
        setError("resize: rows*cols exceeds 2^31-1 elements");
        return;
    }
    BROTENSOR_API_TRY
        t->resize(rows, cols, dt);
    BROTENSOR_API_CATCH("resize")
}

const char* bro_tensor_GpuTensor_dtype(void* self) {
    auto* t = toTensor(self);
    return t ? dtypeToString(t->dtype) : "fp32";
}

void* bro_tensor_GpuTensor_clone(void* self) {
    auto* h = new GpuTensorHandle();
    auto* src = toHandle(self);
    if (src) {
        BROTENSOR_API_TRY
            h->tensor = src->tensor.clone();
            h->shape = src->shape;
        BROTENSOR_API_CATCH("clone")
    }
    return h;
}

// upload keeps the tensor's (rows, cols) when the data fills it exactly and
// otherwise becomes a (data_len, 1) column.
static bool uploadShape(const char* label, const brotensor::Tensor* t, uint32_t data_len, int& r, int& c) {
    if (data_len > static_cast<uint32_t>(INT32_MAX)) {
        setError(std::string(label) + ": more than 2^31-1 elements");
        return false;
    }
    r = t->rows;
    c = t->cols;
    if (static_cast<int64_t>(r) * c != static_cast<int64_t>(data_len)) {
        r = static_cast<int>(data_len);
        c = 1;
    }
    return true;
}

void bro_tensor_GpuTensor_upload(void* self, const float* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = 0, c = 0;
    if (!uploadShape("upload", t, data_len, r, c)) return;
    BROTENSOR_API_TRY
        *t = brotensor::Tensor::from_host_on(brotensor::default_device(), data, r, c);
    BROTENSOR_API_CATCH("upload")
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
        // Convert on the host: the scratch tensor must live on the CPU too,
        // or cast() sees a mixed-device pair (Tensor::empty follows the
        // default device, CUDA once the backend is up).
        try {
            auto host_t = t->to(brotensor::Device::CPU);
            brotensor::Tensor fp32_t = brotensor::Tensor::empty_on(brotensor::Device::CPU, t->rows, t->cols, brotensor::Dtype::FP32);
            brotensor::cast(host_t, fp32_t, brotensor::Dtype::FP32);
            tl_download_f32.resize(static_cast<size_t>(fp32_t.size()));
            fp32_t.copy_to_host(tl_download_f32.data());
        } catch (const std::exception& e) {
            tl_download_f32.clear();
            setError(std::string("download: ") + e.what());
        }
        out->data = tl_download_f32.data();
        out->length = static_cast<uint32_t>(tl_download_f32.size());
    }
    out->release = nullptr;
    out->ctx = nullptr;
}

void bro_tensor_GpuTensor_uploadFp16(void* self, const uint16_t* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = 0, c = 0;
    if (!uploadShape("uploadFp16", t, data_len, r, c)) return;
    BROTENSOR_API_TRY
        *t = brotensor::Tensor::from_host_fp16_on(brotensor::default_device(), data, r, c);
    BROTENSOR_API_CATCH("uploadFp16")
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
        try {
            auto host_t = t->to(brotensor::Device::CPU);
            brotensor::Tensor fp16_t = brotensor::Tensor::empty_on(brotensor::Device::CPU, t->rows, t->cols, brotensor::Dtype::FP16);
            brotensor::cast(host_t, fp16_t, brotensor::Dtype::FP16);
            tl_download_u16.resize(static_cast<size_t>(fp16_t.size()));
            fp16_t.copy_to_host_fp16(tl_download_u16.data());
        } catch (const std::exception& e) {
            tl_download_u16.clear();
            setError(std::string("downloadFp16: ") + e.what());
        }
        out->data = tl_download_u16.data();
        out->length = static_cast<uint32_t>(tl_download_u16.size());
    }
    out->release = nullptr;
    out->ctx = nullptr;
}

void bro_tensor_GpuTensor_uploadInt8(void* self, const int8_t* data, uint32_t data_len) {
    auto* t = toTensor(self);
    if (!t || !data) return;
    int r = 0, c = 0;
    if (!uploadShape("uploadInt8", t, data_len, r, c)) return;
    BROTENSOR_API_TRY
        *t = brotensor::Tensor::from_host_int8_on(brotensor::default_device(), data, r, c);
    BROTENSOR_API_CATCH("uploadInt8")
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
    try {
        auto host_t = t->to(brotensor::Device::CPU);
        tl_download_i8.resize(host_t.bytes());
        std::memcpy(tl_download_i8.data(), host_t.host_raw(), host_t.bytes());
    } catch (const std::exception& e) {
        tl_download_i8.clear();
        setError(std::string("downloadInt8: ") + e.what());
    }
    out->data = tl_download_i8.data();
    out->length = static_cast<uint32_t>(tl_download_i8.size());
    out->release = nullptr;
    out->ctx = nullptr;
}

bool bro_tensor_GpuTensor_downloadInto(void* self, float* dst, uint32_t dst_len) {
    auto* t = toTensor(self);
    if (!t || !dst) {
        setError("download(dst): not a GpuTensor");
        return false;
    }
    if (t->empty()) return true;
    const auto n = static_cast<size_t>(t->size());
    if (dst_len < n) {
        setError("download(dst): dst holds " + std::to_string(dst_len) + " elements, tensor has " + std::to_string(n));
        return false;
    }
    BROTENSOR_API_TRY
        if (t->dtype == brotensor::Dtype::FP32) {
            t->copy_to_host(dst);
        } else {
            auto host_t = t->to(brotensor::Device::CPU);
            brotensor::Tensor fp32_t = brotensor::Tensor::empty_on(brotensor::Device::CPU, t->rows, t->cols, brotensor::Dtype::FP32);
            brotensor::cast(host_t, fp32_t, brotensor::Dtype::FP32);
            fp32_t.copy_to_host(dst);
        }
        return true;
    BROTENSOR_API_CATCH("download(dst)")
    return false;
}

const char* bro_tensor_takeError(void) {
    // The slot keeps the string alive until the next failure; the runtime
    // copies a `str` return into a JS string before anything else runs.
    static thread_local std::string taken;
    taken.swap(lastErrorSlot());
    lastErrorSlot().clear();
    return taken.c_str();
}

bool bro_tensor_available_get(void) {
    return true;
}

// The old binding published a lowercase identifier ("cuda" | "metal" |
// "unknown"), and everything written against bro.tensor compares against one:
// js/tensor.js falls back to "cpu", the docs spell the values lowercase and
// every caller gates with `backend === "cpu"` / `!== "cpu"`. device_name()
// answers a display name ("CPU", "CUDA", "Metal", "CUDA:1"), so lower-case it
// here rather than hand JS a string no gate can match.
const char* bro_tensor_backend_get(void) {
    static thread_local std::string name;
    name = brotensor::device_name(brotensor::default_device());
    for (char& c : name) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    }
    return name.c_str();
}

void bro_tensor_init(void) {
    BROTENSOR_API_TRY
        brotensor::init();
    BROTENSOR_API_CATCH("init")
}

void bro_tensor_sync(void) {
    BROTENSOR_API_TRY
        brotensor::sync_all();
    BROTENSOR_API_CATCH("sync")
}

} // extern "C"
