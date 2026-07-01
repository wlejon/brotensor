#include "brotensor/gguf.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace brotensor::gguf {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gguf: " + msg);
}

// ─── mmap (platform) — duplicated from safetensors.cpp by design ────────────

struct Mapping {
    void* file = nullptr;
    void* mapping = nullptr;
    void* base = nullptr;
    std::size_t size = 0;
};

Mapping map_file(const std::string& path) {
    Mapping m;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        fail("cannot open '" + path + "'");
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) {
        CloseHandle(fh);
        fail("GetFileSizeEx failed for '" + path + "'");
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        fail("CreateFileMapping failed for '" + path + "'");
    }
    void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(mh);
        CloseHandle(fh);
        fail("MapViewOfFile failed for '" + path + "'");
    }
    m.file    = fh;
    m.mapping = mh;
    m.base    = base;
    m.size    = static_cast<std::size_t>(sz.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open '" + path + "'");
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        fail("fstat failed for '" + path + "'");
    }
    void* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        fail("mmap failed for '" + path + "'");
    }
    m.file    = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
    m.mapping = nullptr;
    m.base    = base;
    m.size    = static_cast<std::size_t>(st.st_size);
#endif
    return m;
}

void unmap_file(Mapping& m) noexcept {
#ifdef _WIN32
    if (m.base)    UnmapViewOfFile(m.base);
    if (m.mapping) CloseHandle(m.mapping);
    if (m.file)    CloseHandle(m.file);
#else
    if (m.base && m.size) ::munmap(m.base, m.size);
    if (m.file) ::close(static_cast<int>(reinterpret_cast<intptr_t>(m.file)));
#endif
    m = Mapping{};
}

// ─── ggml type table ────────────────────────────────────────────────────────

struct GgmlTypeInfo {
    brotensor::Dtype dtype;
    bool             supported;
    int              block_size;
    int              type_size;
};

GgmlTypeInfo ggml_type_info(uint32_t t) {
    switch (t) {
        case 0:  return { brotensor::Dtype::FP32, true,  1,   4 };
        case 1:  return { brotensor::Dtype::FP16, true,  1,   2 };
        case 2:  return { brotensor::Dtype::Q4_0, true, 32,  18 };
        case 3:  return { brotensor::Dtype::Q4_1, true, 32,  20 };
        case 6:  return { brotensor::Dtype::Q5_0, true, 32,  22 };
        case 7:  return { brotensor::Dtype::Q5_1, true, 32,  24 };
        case 8:  return { brotensor::Dtype::Q8_0, true, 32,  34 };
        case 9:  return { brotensor::Dtype::Q8_1, true, 32,  36 };
        case 10: return { brotensor::Dtype::Q2_K, true, 256,  82 };
        case 11: return { brotensor::Dtype::Q3_K, true, 256, 110 };
        case 12: return { brotensor::Dtype::Q4_K, true, 256, 144 };
        case 13: return { brotensor::Dtype::Q5_K, true, 256, 176 };
        case 14: return { brotensor::Dtype::Q6_K, true, 256, 210 };
        case 15: return { brotensor::Dtype::Q8_K, true, 256, 292 };
        case 30: return { brotensor::Dtype::BF16, true,  1,   2 };
        default: return { brotensor::Dtype::FP32, false, 0,   0 };
    }
}

// ─── Reader ────────────────────────────────────────────────────────────────

struct Reader {
    const uint8_t* base = nullptr;
    std::size_t    pos  = 0;
    std::size_t    size = 0;

    void need(std::size_t n) const {
        if (pos + n > size) fail("unexpected EOF");
    }

    uint8_t read_u8() {
        need(1);
        return base[pos++];
    }
    int8_t read_i8() {
        return static_cast<int8_t>(read_u8());
    }
    uint16_t read_u16() {
        need(2);
        uint16_t v;
        std::memcpy(&v, base + pos, 2);
        pos += 2;
        return v;
    }
    int16_t read_i16() { uint16_t v = read_u16(); int16_t r; std::memcpy(&r, &v, 2); return r; }
    uint32_t read_u32() {
        need(4);
        uint32_t v;
        std::memcpy(&v, base + pos, 4);
        pos += 4;
        return v;
    }
    int32_t read_i32() { uint32_t v = read_u32(); int32_t r; std::memcpy(&r, &v, 4); return r; }
    uint64_t read_u64() {
        need(8);
        uint64_t v;
        std::memcpy(&v, base + pos, 8);
        pos += 8;
        return v;
    }
    int64_t read_i64() { uint64_t v = read_u64(); int64_t r; std::memcpy(&r, &v, 8); return r; }
    float read_f32()   { uint32_t v = read_u32(); float r;   std::memcpy(&r, &v, 4); return r; }
    double read_f64()  { uint64_t v = read_u64(); double r;  std::memcpy(&r, &v, 8); return r; }
    bool read_bool() {
        uint8_t b = read_u8();
        if (b != 0 && b != 1) fail("bool value not 0 or 1");
        return b != 0;
    }
    std::string read_str() {
        uint64_t len = read_u64();
        need(static_cast<std::size_t>(len));
        std::string s(reinterpret_cast<const char*>(base + pos), static_cast<std::size_t>(len));
        pos += static_cast<std::size_t>(len);
        return s;
    }
};

Value parse_value(Reader& r, ValueType type);

Value parse_scalar(Reader& r, ValueType type) {
    Value v;
    v.type = type;
    switch (type) {
        case ValueType::U8:   v.scalar.u8  = r.read_u8();  break;
        case ValueType::I8:   v.scalar.i8  = r.read_i8();  break;
        case ValueType::U16:  v.scalar.u16 = r.read_u16(); break;
        case ValueType::I16:  v.scalar.i16 = r.read_i16(); break;
        case ValueType::U32:  v.scalar.u32 = r.read_u32(); break;
        case ValueType::I32:  v.scalar.i32 = r.read_i32(); break;
        case ValueType::F32:  v.scalar.f32 = r.read_f32(); break;
        case ValueType::Bool: v.scalar.b   = r.read_bool();break;
        case ValueType::U64:  v.scalar.u64 = r.read_u64(); break;
        case ValueType::I64:  v.scalar.i64 = r.read_i64(); break;
        case ValueType::F64:  v.scalar.f64 = r.read_f64(); break;
        case ValueType::String:
            v.str = r.read_str();
            break;
        case ValueType::Array:
            // Caller dispatches arrays via parse_value.
            fail("internal: parse_scalar called with Array");
    }
    return v;
}

Value parse_value(Reader& r, ValueType type) {
    if (type != ValueType::Array) {
        return parse_scalar(r, type);
    }
    Value v;
    v.type = ValueType::Array;
    uint32_t elem_raw = r.read_u32();
    if (elem_raw > 12) fail("array element type out of range");
    v.array_elem_type = static_cast<ValueType>(elem_raw);
    uint64_t len = r.read_u64();
    v.array.reserve(static_cast<std::size_t>(len));
    for (uint64_t i = 0; i < len; ++i) {
        v.array.push_back(parse_value(r, v.array_elem_type));
    }
    return v;
}

std::size_t align_up(std::size_t x, std::size_t a) {
    if (a == 0) return x;
    return (x + (a - 1)) & ~(a - 1);
}

}  // namespace

// ─── shape_to_2d ────────────────────────────────────────────────────────────

std::pair<int, int> shape_to_2d(const std::vector<int64_t>& shape) {
    if (shape.empty()) fail("shape_to_2d: empty shape");
    for (int64_t d : shape) {
        if (d <= 0) fail("shape_to_2d: non-positive dimension");
        if (d > 0x7FFFFFFF) fail("shape_to_2d: dimension exceeds int range");
    }
    if (shape.size() == 1) {
        return { static_cast<int>(shape[0]), 1 };
    }
    int64_t rows = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        if (rows > (int64_t{0x7FFFFFFF}) / shape[i]) {
            fail("shape_to_2d: row product overflows int");
        }
        rows *= shape[i];
    }
    return { static_cast<int>(rows), static_cast<int>(shape[0]) };
}

// ─── File ──────────────────────────────────────────────────────────────────

File::~File() { release_(); }

File::File(File&& o) noexcept
    : file_(o.file_), mapping_(o.mapping_), base_(o.base_),
      file_size_(o.file_size_),
      version_(o.version_), alignment_(o.alignment_),
      metadata_(std::move(o.metadata_)),
      meta_index_(std::move(o.meta_index_)),
      tensors_(std::move(o.tensors_)),
      tensor_index_(std::move(o.tensor_index_)) {
    o.file_ = o.mapping_ = o.base_ = nullptr;
    o.file_size_ = 0;
    o.version_ = 0;
    o.alignment_ = 32;
}

File& File::operator=(File&& o) noexcept {
    if (this != &o) {
        release_();
        file_         = o.file_;
        mapping_      = o.mapping_;
        base_         = o.base_;
        file_size_    = o.file_size_;
        version_      = o.version_;
        alignment_    = o.alignment_;
        metadata_     = std::move(o.metadata_);
        meta_index_   = std::move(o.meta_index_);
        tensors_      = std::move(o.tensors_);
        tensor_index_ = std::move(o.tensor_index_);
        o.file_ = o.mapping_ = o.base_ = nullptr;
        o.file_size_ = 0;
        o.version_ = 0;
        o.alignment_ = 32;
    }
    return *this;
}

void File::release_() noexcept {
    Mapping m{ file_, mapping_, base_, file_size_ };
    unmap_file(m);
    file_ = mapping_ = base_ = nullptr;
    file_size_ = 0;
    version_ = 0;
    alignment_ = 32;
    metadata_.clear();
    meta_index_.clear();
    tensors_.clear();
    tensor_index_.clear();
}

File File::open(const std::string& path) {
    Mapping m = map_file(path);
    if (m.size < 24) {
        unmap_file(m);
        fail("file too small: '" + path + "'");
    }

    File f;
    f.file_      = m.file;
    f.mapping_   = m.mapping;
    f.base_      = m.base;
    f.file_size_ = m.size;
    // From here on, f's destructor unmaps on throw.

    Reader r{ static_cast<const uint8_t*>(m.base), 0, m.size };

    uint32_t magic = r.read_u32();
    if (magic != 0x46554747u) {
        fail("bad magic (not a GGUF file)");
    }
    uint32_t version = r.read_u32();
    if (version != 2 && version != 3) {
        fail("unsupported GGUF version " + std::to_string(version));
    }
    f.version_ = version;

    uint64_t tensor_count = r.read_u64();
    uint64_t meta_count   = r.read_u64();

    f.metadata_.reserve(static_cast<std::size_t>(meta_count));
    for (uint64_t i = 0; i < meta_count; ++i) {
        std::string key = r.read_str();
        uint32_t vt_raw = r.read_u32();
        if (vt_raw > 12) fail("metadata value type out of range");
        Value v = parse_value(r, static_cast<ValueType>(vt_raw));
        if (key == "general.alignment") {
            if (v.type != ValueType::U32) {
                fail("general.alignment must be u32");
            }
            if (v.scalar.u32 == 0 || (v.scalar.u32 & (v.scalar.u32 - 1)) != 0) {
                fail("general.alignment must be a positive power of two");
            }
            f.alignment_ = v.scalar.u32;
        }
        f.meta_index_.emplace(key, f.metadata_.size());
        f.metadata_.emplace_back(std::move(key), std::move(v));
    }

    // Tensor infos.
    struct PendingTensor {
        std::string             name;
        std::vector<int64_t>    shape;
        uint32_t                ggml_type = 0;
        uint64_t                offset    = 0;
    };
    std::vector<PendingTensor> pending;
    pending.reserve(static_cast<std::size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        PendingTensor pt;
        pt.name = r.read_str();
        uint32_t n_dims = r.read_u32();
        if (n_dims == 0 || n_dims > 8) fail("invalid tensor n_dims");
        pt.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            pt.shape[d] = static_cast<int64_t>(r.read_u64());
        }
        pt.ggml_type = r.read_u32();
        pt.offset    = r.read_u64();
        pending.push_back(std::move(pt));
    }

    // Align to alignment_ for the data blob.
    const std::size_t data_blob_start = align_up(r.pos, f.alignment_);
    if (data_blob_start > m.size) fail("data blob start past end of file");

    f.tensors_.reserve(pending.size());
    for (auto& pt : pending) {
        TensorInfo info;
        info.name      = std::move(pt.name);
        info.shape     = std::move(pt.shape);
        info.ggml_type = pt.ggml_type;

        int64_t numel = 1;
        for (int64_t d : info.shape) {
            if (d <= 0) fail("tensor '" + info.name + "' has non-positive dim");
            if (numel > INT64_MAX / d) fail("tensor numel overflows int64");
            numel *= d;
        }
        info.numel = numel;

        GgmlTypeInfo gt = ggml_type_info(info.ggml_type);
        info.dtype           = gt.dtype;
        info.dtype_supported = gt.supported;

        std::size_t nbytes = 0;
        if (gt.supported) {
            if (gt.block_size > 1 && (numel % gt.block_size) != 0) {
                fail("tensor '" + info.name + "' numel not multiple of block size");
            }
            nbytes = (static_cast<std::size_t>(numel) /
                      static_cast<std::size_t>(gt.block_size)) *
                     static_cast<std::size_t>(gt.type_size);
        } else if (gt.block_size > 0 && gt.type_size > 0) {
            nbytes = (static_cast<std::size_t>(numel) /
                      static_cast<std::size_t>(gt.block_size)) *
                     static_cast<std::size_t>(gt.type_size);
        }
        info.nbytes = nbytes;

        const std::size_t abs_off = data_blob_start + static_cast<std::size_t>(pt.offset);
        if (abs_off > m.size || abs_off + nbytes > m.size) {
            fail("tensor '" + info.name + "' data out of bounds");
        }
        info.data = static_cast<const uint8_t*>(m.base) + abs_off;

        f.tensor_index_.emplace(info.name, f.tensors_.size());
        f.tensors_.push_back(std::move(info));
    }

    return f;
}

const TensorInfo* File::find_tensor(std::string_view name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

const TensorInfo& File::get_tensor(std::string_view name) const {
    const TensorInfo* t = find_tensor(name);
    if (!t) fail("no tensor named '" + std::string(name) + "'");
    return *t;
}

const Value* File::find_meta(std::string_view key) const {
    auto it = meta_index_.find(key);
    if (it == meta_index_.end()) return nullptr;
    return &metadata_[it->second].second;
}

const Value& File::get_meta(std::string_view key) const {
    const Value* v = find_meta(key);
    if (!v) fail("no metadata key '" + std::string(key) + "'");
    return *v;
}

// ─── upload_raw ────────────────────────────────────────────────────────────

void upload_raw(const TensorInfo& info, int rows, int cols,
                brotensor::Tensor& dst) {
    if (!info.dtype_supported) {
        fail("upload_raw: tensor '" + info.name +
             "' has unsupported ggml_type " + std::to_string(info.ggml_type));
    }
    if (rows <= 0 || cols <= 0) {
        fail("upload_raw: rows/cols must be positive");
    }
    const int64_t prod = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (prod != info.numel) {
        fail("upload_raw: rows*cols (" + std::to_string(prod) +
             ") != info.numel (" + std::to_string(info.numel) + ")");
    }
    if (brotensor::dtype_is_quant(info.dtype)) {
        const int bs = brotensor::dtype_block_size(info.dtype);
        if (cols % bs != 0) {
            fail("upload_raw: cols not a multiple of block size for quant dtype");
        }
    }

    // info.data points into the mmap'd, host-resident GGUF file — a valid
    // H2D source as-is. Upload straight from it (single copy) instead of
    // staging through an intermediate CPU tensor and then migrating.
    const brotensor::Device target = brotensor::default_device();
    dst = brotensor::Tensor::from_raw_bytes_on(
        target, info.data, rows, cols, info.dtype, info.nbytes);
}

}  // namespace brotensor::gguf
