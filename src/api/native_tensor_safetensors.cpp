// bro.tensor.openSafetensors / saveSafetensors and the SafetensorsFile
// handle, ported from the QuickJS binding (tensor_bindings_safetensors.cpp):
//
//   openSafetensors(path) -> handle        (mmap; header only, no payload read)
//     handle.count / names() / header() / get(name, rows?, cols?, dtype?) / close()
//   saveSafetensors(path, { name: GpuTensor, ... })
//
// The handle owns the mmap'd brotensor::safetensors::File; `file` is null
// after close(), and the class destructor unmaps on GC. names()/header() are
// assembled in js/tensor.js from the per-index natives here (nameAt /
// dtypeAt / shapeAt / nbytesAt): a native returns scalars, typed arrays or
// handles, never a JS object. Paths go through api.h's path resolver.

#include "native_tensor_decl.h"
#include "api_internal.h"

#include <brotensor/safetensors.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace stns = brotensor::safetensors;
namespace ev = bronze::embed;
using namespace brotensor::api;
using Value = bronze::Value;

namespace {

struct SafetensorsFileHandle {
    std::unique_ptr<stns::File> file;
};

SafetensorsFileHandle* asFile(void* self) {
    return static_cast<SafetensorsFileHandle*>(self);
}

// The view at index i of an open file, or nullptr (with the error set).
const stns::TensorView* viewAt(void* self, int32_t i, const char* label) {
    auto* h = asFile(self);
    if (!h || !h->file) {
        setError(std::string(label) + "() on a closed safetensors file");
        return nullptr;
    }
    const auto& tensors = h->file->tensors();
    if (i < 0 || static_cast<size_t>(i) >= tensors.size()) {
        setError(std::string(label) + ": index out of range");
        return nullptr;
    }
    return &tensors[static_cast<size_t>(i)];
}

thread_local std::vector<double> tl_shape;

} // namespace

extern "C" {

void* bro_tensor_SafetensorsFile_ctor(void) {
    return new SafetensorsFileHandle();
}

void bro_tensor_SafetensorsFile_dtor(void* self) {
    delete asFile(self);
}

void* bro_tensor_openSafetensors(const char* path) {
    auto* h = new SafetensorsFileHandle();
    try {
        h->file = std::make_unique<stns::File>(stns::File::open(resolvePath(path ? path : "")));
    } catch (const std::exception& e) {
        delete h;
        setError(std::string("openSafetensors: ") + e.what());
        return nullptr;
    }
    return h;
}

bool bro_tensor_SafetensorsFile_isOpen(void* self) {
    auto* h = asFile(self);
    return h && h->file != nullptr;
}

int32_t bro_tensor_SafetensorsFile_count_get(void* self) {
    auto* h = asFile(self);
    return (h && h->file) ? static_cast<int32_t>(h->file->size()) : 0;
}

const char* bro_tensor_SafetensorsFile_nameAt(void* self, int32_t i) {
    const auto* tv = viewAt(self, i, "names");
    return tv ? tv->name.c_str() : "";
}

const char* bro_tensor_SafetensorsFile_dtypeAt(void* self, int32_t i) {
    const auto* tv = viewAt(self, i, "header");
    return tv ? stns::dtype_name(tv->dtype) : "";
}

void bro_tensor_SafetensorsFile_shapeAt(void* self, int32_t i, bronze_native_buffer* out) {
    if (!out) return;
    out->data = nullptr;
    out->length = 0;
    out->release = nullptr;
    out->ctx = nullptr;
    const auto* tv = viewAt(self, i, "header");
    if (!tv) return;
    tl_shape.clear();
    tl_shape.reserve(tv->shape.size());
    for (int64_t d : tv->shape) tl_shape.push_back(static_cast<double>(d));
    out->data = tl_shape.data();
    out->length = static_cast<uint32_t>(tl_shape.size());
}

double bro_tensor_SafetensorsFile_nbytesAt(void* self, int32_t i) {
    const auto* tv = viewAt(self, i, "header");
    return tv ? static_cast<double>(tv->nbytes) : 0.0;
}

// get(name, rows, cols, mode): the old body's shape rule — an explicit
// (rows, cols) pair when both were numbers, else (shape[0], numel/shape[0]);
// mode "compute" -> upload_compute, "fp16" -> upload_fp16, anything else ->
// upload (the file's own dtype).
void* bro_tensor_SafetensorsFile_get(void* self, const char* name, int32_t rows, int32_t cols, const char* mode) {
    auto* h = asFile(self);
    if (!h || !h->file) {
        setError("get() on a closed safetensors file");
        return nullptr;
    }
    const std::string key(name ? name : "");
    const stns::TensorView* tv = h->file->find(key);
    if (!tv) {
        setError("get: no tensor named '" + key + "'");
        return nullptr;
    }
    int r = rows;
    int c = cols;
    if (r <= 0 || c <= 0) {
        try {
            const int64_t numel = tv->numel();
            r = tv->shape.empty() ? 1 : static_cast<int>(tv->shape[0]);
            c = r > 0 ? static_cast<int>(numel / r) : 0;
        } catch (const std::exception& e) {
            setError(std::string("get: ") + e.what());
            return nullptr;
        }
    }
    auto* gt = new GpuTensorHandle();
    gt->shape = (rows <= 0 || cols <= 0) ? tv->shape : std::vector<int64_t>{rows, cols};
    try {
        if (mode && std::strcmp(mode, "compute") == 0) {
            stns::upload_compute(*tv, r, c, gt->tensor);
        } else if (mode && std::strcmp(mode, "fp16") == 0) {
            stns::upload_fp16(*tv, r, c, gt->tensor);
        } else {
            stns::upload(*tv, r, c, gt->tensor);
        }
    } catch (const std::exception& e) {
        delete gt;
        setError(std::string("get: ") + e.what());
        return nullptr;
    }
    return gt;
}

void bro_tensor_SafetensorsFile_close(void* self) {
    auto* h = asFile(self);
    if (h) h->file.reset();
}

bool bro_tensor_saveSafetensors(const char* path, uint64_t names_bits, uint64_t tensors_bits) {
    ev::Persistent names(bronze::Value{names_bits});
    ev::Persistent tensors(bronze::Value{tensors_bits});
    if (!ev::isObject(names.get()) || !ev::isObject(tensors.get())) {
        setError("saveSafetensors(path, {name: tensor, ...})");
        return false;
    }
    const auto n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(names.get(), "length")));

    std::vector<stns::WriteEntry> entries;
    // Backing host buffers, alive until write_file() returns; reserved so
    // the inner vectors never reallocate out from under host_data.
    std::vector<std::vector<float>> f32store;
    std::vector<std::vector<uint16_t>> f16store;
    entries.reserve(n);
    f32store.reserve(n);
    f16store.reserve(n);

    try {
        brotensor::sync_all();
        for (uint32_t i = 0; i < n; ++i) {
            const std::string key = ev::toUtf8(ev::getElement(names.get(), i));
            Value tVal = ev::getElement(tensors.get(), i);
            void* d = ev::handleData(tVal);
            auto* gh = d ? toHandle(d) : nullptr;
            auto* gt = gh ? &gh->tensor : nullptr;
            if (!gt) {
                setError("saveSafetensors: value for '" + key + "' is not a tensor");
                return false;
            }
            stns::WriteEntry e;
            e.name = key;

            // Preserve full n-dimensional tensor shape if present. The shape
            // array is read across several allocating getProperty/getElement
            // calls, so it lives in a Persistent: a raw Value is stale after
            // the first of them.
            bool shapeSet = false;
            ev::Persistent shapeProp(ev::getProperty(tVal, "shape"));
            if (ev::isObject(shapeProp.get())) {
                Value lenVal = ev::getProperty(shapeProp.get(), "length");
                if (ev::isNumber(lenVal)) {
                    uint32_t shapeLen = static_cast<uint32_t>(ev::toDouble(lenVal));
                    if (shapeLen > 0) {
                        std::vector<int64_t> customShape;
                        customShape.reserve(shapeLen);
                        int64_t numel = 1;
                        for (uint32_t s = 0; s < shapeLen; ++s) {
                            int64_t dim = static_cast<int64_t>(ev::toDouble(ev::getElement(shapeProp.get(), s)));
                            customShape.push_back(dim);
                            // A negative dim can multiply back to rows*cols
                            // ([-2, -3] for 6); never let it reach the header.
                            numel = dim < 0 ? -1 : numel * dim;
                            if (numel < 0) break;
                        }
                        if (numel == static_cast<int64_t>(gt->rows) * gt->cols) {
                            e.shape = std::move(customShape);
                            shapeSet = true;
                        }
                    }
                }
            }

            if (!shapeSet && gh && !gh->shape.empty()) {
                int64_t numel = 1;
                for (int64_t dim : gh->shape) numel *= dim;
                if (numel == static_cast<int64_t>(gt->rows) * gt->cols) {
                    e.shape = gh->shape;
                    shapeSet = true;
                }
            }

            if (!shapeSet) {
                e.shape = {gt->rows, gt->cols};
            }
            brotensor::Tensor host = gt->to(brotensor::Device::CPU);
            if (host.dtype == brotensor::Dtype::FP32) {
                f32store.push_back(host.to_host_vector());
                e.dtype = stns::Dtype::F32;
                e.host_data = f32store.back().data();
                e.bytes = f32store.back().size() * sizeof(float);
            } else if (host.dtype == brotensor::Dtype::FP16) {
                f16store.push_back(host.to_host_vector_fp16());
                e.dtype = stns::Dtype::F16;
                e.host_data = f16store.back().data();
                e.bytes = f16store.back().size() * sizeof(uint16_t);
            } else {
                setError("saveSafetensors: '" + key + "' has an unsupported dtype (FP32/FP16 only)");
                return false;
            }
            entries.push_back(std::move(e));
        }
        stns::write_file(resolvePath(path ? path : ""), entries);
    } catch (const std::exception& e) {
        setError(std::string("saveSafetensors: ") + e.what());
        return false;
    }
    return true;
}

} // extern "C"
