#pragma once

// GGUF reader for brotensor. Chunk 1 — loader + storage carrier only; no
// dequant kernels and no ops dispatch on quant dtypes.
//
// Format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// The file is mmap'd; TensorInfo::data points directly into the mapping, so
// it's valid only while the owning File is alive. Zero-copy on the read path.
// brotensor is 2D-only — gguf::shape_to_2d() collapses the GGUF
// (innermost-first) dim list to (rows, cols) for upload_raw().

#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brotensor::gguf {

enum class ValueType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
    F32 = 6, Bool = 7, String = 8, Array = 9,
    U64 = 10, I64 = 11, F64 = 12,
};

struct Value {
    ValueType type = ValueType::U32;
    // Only the field matching `type` (or, for arrays, `array_elem_type` on
    // each element) is meaningful.
    union Scalar {
        uint8_t  u8;  int8_t  i8;
        uint16_t u16; int16_t i16;
        uint32_t u32; int32_t i32;
        uint64_t u64; int64_t i64;
        float    f32; double  f64;
        bool     b;
    } scalar{};
    std::string         str;
    ValueType           array_elem_type = ValueType::U32;
    std::vector<Value>  array;
};

struct TensorInfo {
    std::string             name;
    // Dim order is innermost-first, as stored on disk. shape_to_2d() returns
    // (rows = product(shape[1..]), cols = shape[0]).
    std::vector<int64_t>    shape;
    uint32_t                ggml_type       = 0;
    brotensor::Dtype        dtype           = brotensor::Dtype::FP32;
    bool                    dtype_supported = true;
    const uint8_t*          data            = nullptr;
    std::size_t             nbytes          = 0;
    int64_t                 numel           = 1;
};

class File {
public:
    File() = default;
    ~File();

    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    static File open(const std::string& path);

    const TensorInfo* find_tensor(std::string_view name) const;
    const TensorInfo& get_tensor(std::string_view name) const;
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    const Value* find_meta(std::string_view key) const;
    const Value& get_meta(std::string_view key) const;
    const std::vector<std::pair<std::string, Value>>& metadata() const { return metadata_; }

    uint32_t version()   const { return version_; }
    uint32_t alignment() const { return alignment_; }
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    void release_() noexcept;

    void* file_    = nullptr;
    void* mapping_ = nullptr;
    void* base_    = nullptr;
    std::size_t file_size_ = 0;

    uint32_t version_   = 0;
    uint32_t alignment_ = 32;

    std::vector<std::pair<std::string, Value>>    metadata_;
    std::unordered_map<std::string, std::size_t>  meta_index_;
    std::vector<TensorInfo>                       tensors_;
    std::unordered_map<std::string, std::size_t>  tensor_index_;
};

// Convert GGUF shape (innermost-first) to brotensor 2D (rows, cols). cols is
// the innermost (contiguous) axis; rows is the product of the remaining axes.
// 1D input → (shape[0], 1). Empty shape throws.
std::pair<int, int> shape_to_2d(const std::vector<int64_t>& shape);

// Allocate `dst` on the current default device sized for `info.numel`
// elements at `info.dtype` (via dtype_storage_bytes) and copy info.nbytes
// bytes from the mmap into it. For quant dtypes, cols must be a multiple of
// dtype_block_size(dtype) (blocks run along the contiguous axis). Throws if
// info.dtype_supported is false or rows*cols != info.numel.
void upload_raw(const TensorInfo& info, int rows, int cols,
                brotensor::Tensor& dst);

}  // namespace brotensor::gguf
