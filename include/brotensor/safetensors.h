#pragma once

// safetensors reader + writer for brotensor.
//
// Format (from huggingface/safetensors):
//   bytes 0..7   : little-endian u64 header_size N
//   bytes 8..8+N : UTF-8 JSON header
//   bytes 8+N..  : raw tensor payload
//
// Header JSON is an object mapping tensor_name -> {
//     "dtype":        "F32" | "F16" | "BF16" | "I32" | "I64" | "U8" | "BOOL",
//     "shape":        [N1, N2, ...],
//     "data_offsets": [start, end]    // bytes relative to start of payload
// }
// plus an optional "__metadata__" entry (string -> string) that we skip.
//
// The file is mmap'd; TensorView::data points directly into the mapping, so
// it's valid only while the owning File is alive. Zero-copy on the read path.
//
// safetensors is a tensor-container format — a named collection of typed N-D
// arrays. It lives in brotensor because its natural output type is
// brotensor::Tensor; the upload* helpers below depend only on brotensor.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brotensor { struct Tensor; }

namespace brotensor::safetensors {

enum class Dtype {
    Unknown = 0,
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    BOOL,
};

const char* dtype_name(Dtype d);
int dtype_size_bytes(Dtype d);  // 0 for Unknown

struct TensorView {
    std::string name;
    Dtype dtype = Dtype::Unknown;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;  // into the mmap; valid while File lives
    std::size_t nbytes = 0;

    int64_t numel() const;          // product of shape
};

class File {
public:
    File() = default;
    ~File();

    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    // Open a .safetensors file. Throws std::runtime_error on I/O or parse
    // failure. The returned File owns the mmap and tensor index.
    static File open(const std::string& path);

    // Lookup by name. find() returns nullptr if missing; get() throws.
    const TensorView* find(std::string_view name) const;
    const TensorView& get(std::string_view name) const;

    const std::vector<TensorView>& tensors() const { return tensors_; }
    std::size_t size() const { return tensors_.size(); }

private:
    void release_() noexcept;

    // Platform-specific mmap handles. On Windows: file_ + mapping_ + base_.
    // On POSIX: file_ holds the fd, mapping_ is unused.
    void* file_    = nullptr;
    void* mapping_ = nullptr;
    void* base_    = nullptr;
    std::size_t file_size_ = 0;

    std::vector<TensorView> tensors_;
    std::unordered_map<std::string, std::size_t> index_;
};

// Upload a tensor view as a 2D (rows, cols) Tensor.
//
// brotensor is 2D-only by design; higher-rank tensors (e.g. conv weights
// [Cout, Cin, kH, kW]) are flattened by the caller to whatever 2D layout the
// consuming op expects.
//
// Requirements:
//   - view.dtype must be F32, F16, or BF16
//   - rows * cols * dtype_size_bytes(view.dtype) == view.nbytes
//   - dst is resized to (rows, cols) at the matching brotensor::Dtype
//     (a BF16 view produces a brotensor::Dtype::BF16 tensor, zero-converted)
//
// Throws std::runtime_error if the byte count or dtype is wrong.
void upload(const TensorView& view, int rows, int cols, brotensor::Tensor& dst);

// Like upload(), but always produces an FP16 Tensor. If the source view is
// F32, it is converted host-side via brotensor::fp32_to_fp16_bits before
// upload. If the source is F16, it's uploaded as-is.
void upload_fp16(const TensorView& view, int rows, int cols, brotensor::Tensor& dst);

// Upload a weight at brotensor's compute dtype — FP32 on the CPU backend,
// FP16 on a GPU backend (see brotensor::compute_dtype()). The source view
// (F16 or F32) is converted host-side as needed, so a single checkpoint
// serves either backend.
void upload_compute(const TensorView& view, int rows, int cols,
                    brotensor::Tensor& dst);

// Like upload_compute(), but first validates the view: its dtype must be
// F16 or F32 and its element count must equal rows*cols. Throws
// std::runtime_error tagged with `name` (a caller-supplied label) and the
// safetensors key on mismatch — the loader entry point a model component
// uses so a malformed checkpoint fails with a clear message.
void upload_compute_checked(const TensorView& view, int rows, int cols,
                            brotensor::Tensor& dst, const std::string& name);

// ─── Writer ────────────────────────────────────────────────────────────────
//
// Minimal safetensors writer. Builds the JSON header in insertion order and
// concatenates raw tensor bytes. All entries' `host_data` must point to
// `bytes` bytes of valid host memory matching `dtype`/`shape`. Tensor names
// must be unique. Throws std::runtime_error on I/O failure or invalid input.

struct WriteEntry {
    std::string             name;
    Dtype                   dtype = Dtype::F16;
    std::vector<int64_t>    shape;
    const void*             host_data = nullptr;
    std::size_t             bytes = 0;
};

void write_file(const std::string& path, const std::vector<WriteEntry>& entries);

}  // namespace brotensor::safetensors
