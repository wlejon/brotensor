// GGUF reader coverage test — complements test_gguf.cpp.
//
// Everything here is CPU-only: the reader (header parse, metadata KV parse,
// tensor-info table, shape_to_2d, upload_raw as a raw byte copy) is
// device-neutral, so it exercises fine on the always-on CPU tier. A
// DeviceScope pins uploads to Device::CPU so the byte-level assertions hold
// regardless of which backends the binary was built with.
//
// Fixtures are synthesised byte-by-byte, same approach as test_gguf.cpp.

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace gg = brotensor::gguf;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Expect `expr` to throw std::runtime_error.
#define CHECK_THROWS(expr) do { \
    bool threw_ = false; \
    try { expr; } catch (const std::runtime_error&) { threw_ = true; } \
    if (!threw_) { \
        std::fprintf(stderr, "FAIL %s:%d: expected throw: %s\n", \
                     __FILE__, __LINE__, #expr); \
        ++g_failures; \
    } \
} while (0)

namespace {

constexpr uint32_t kMagic = 0x46554747u;  // "GGUF"

constexpr uint8_t  u8c(int v)  { return static_cast<uint8_t>(v); }
constexpr uint16_t u16c(int v) { return static_cast<uint16_t>(v); }

// ─── little-endian byte writers ────────────────────────────────────────────

void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_i8(std::vector<uint8_t>& b, int8_t v) {
    put_u8(b, static_cast<uint8_t>(v));
}
void put_i16(std::vector<uint8_t>& b, int16_t v) {
    put_u16(b, static_cast<uint16_t>(v));
}
void put_i32(std::vector<uint8_t>& b, int32_t v) {
    put_u32(b, static_cast<uint32_t>(v));
}
void put_i64(std::vector<uint8_t>& b, int64_t v) {
    put_u64(b, static_cast<uint64_t>(v));
}
void put_f32(std::vector<uint8_t>& b, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    put_u32(b, v);
}
void put_f64(std::vector<uint8_t>& b, double d) {
    uint64_t v;
    std::memcpy(&v, &d, 8);
    put_u64(b, v);
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, static_cast<uint64_t>(s.size()));
    for (char c : s) b.push_back(static_cast<uint8_t>(c));
}

// ─── fixture assembly ──────────────────────────────────────────────────────

struct Fixture {
    std::vector<uint8_t> meta;
    std::vector<uint8_t> tinfo;
    std::vector<uint8_t> blob;
    uint64_t n_meta    = 0;
    uint64_t n_tensors = 0;
};

// Start a metadata KV: key + value-type tag. The caller appends the payload.
void meta_key(Fixture& fx, const std::string& key, uint32_t value_type) {
    put_str(fx.meta, key);
    put_u32(fx.meta, value_type);
    ++fx.n_meta;
}

void put_tinfo(Fixture& fx, const std::string& name,
               const std::vector<uint64_t>& dims, uint32_t ggml_type,
               uint64_t offset) {
    put_str(fx.tinfo, name);
    put_u32(fx.tinfo, static_cast<uint32_t>(dims.size()));
    for (uint64_t d : dims) put_u64(fx.tinfo, d);
    put_u32(fx.tinfo, ggml_type);
    put_u64(fx.tinfo, offset);
    ++fx.n_tensors;
}

// Reserve `n` bytes in the data blob filled with a recognisable pattern and
// return the blob-relative offset (which is what the tensor-info offset field
// wants — offsets are relative to the aligned start of the data blob).
uint64_t blob_pattern(Fixture& fx, std::size_t n, uint8_t seed) {
    const uint64_t off = static_cast<uint64_t>(fx.blob.size());
    for (std::size_t i = 0; i < n; ++i) {
        fx.blob.push_back(static_cast<uint8_t>(seed + static_cast<uint8_t>(i)));
    }
    return off;
}

// `align == 0` means: emit no padding and no data blob (used to exercise the
// "data blob start past end of file" check).
std::vector<uint8_t> assemble(const Fixture& fx, uint32_t magic,
                              uint32_t version, uint32_t align) {
    std::vector<uint8_t> b;
    put_u32(b, magic);
    put_u32(b, version);
    put_u64(b, fx.n_tensors);
    put_u64(b, fx.n_meta);
    b.insert(b.end(), fx.meta.begin(), fx.meta.end());
    b.insert(b.end(), fx.tinfo.begin(), fx.tinfo.end());
    if (align > 0) {
        while (b.size() % align != 0) b.push_back(0);
        b.insert(b.end(), fx.blob.begin(), fx.blob.end());
    }
    return b;
}

std::vector<std::filesystem::path> g_temps;

std::filesystem::path write_temp(const std::string& name,
                                 const std::vector<uint8_t>& bytes) {
    auto path = std::filesystem::temp_directory_path() /
                ("brotensor_gguf_extra_" + name + ".gguf");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create gguf fixture file");
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }
    f.close();
    g_temps.push_back(path);
    return path;
}

// ─── shared fixtures ───────────────────────────────────────────────────────

// Every metadata value type, plus arrays (including an empty array and a
// nested array-of-arrays), plus an explicit general.alignment of 16.
std::filesystem::path write_meta_fixture() {
    Fixture fx;

    meta_key(fx, "general.alignment", 4);  // U32 — must be a power of two
    put_u32(fx.meta, 16);

    meta_key(fx, "k.u8", 0);   put_u8(fx.meta, u8c(200));
    meta_key(fx, "k.i8", 1);   put_i8(fx.meta, static_cast<int8_t>(-5));
    meta_key(fx, "k.u16", 2);  put_u16(fx.meta, u16c(65000));
    meta_key(fx, "k.i16", 3);  put_i16(fx.meta, static_cast<int16_t>(-300));
    meta_key(fx, "k.u32", 4);  put_u32(fx.meta, 4000000000u);
    meta_key(fx, "k.i32", 5);  put_i32(fx.meta, -123456);
    meta_key(fx, "k.f32", 6);  put_f32(fx.meta, -2.5f);
    meta_key(fx, "k.true", 7); put_u8(fx.meta, u8c(1));
    meta_key(fx, "k.false", 7); put_u8(fx.meta, u8c(0));
    meta_key(fx, "k.str", 8);  put_str(fx.meta, "hello gguf");
    meta_key(fx, "k.u64", 10); put_u64(fx.meta, uint64_t{1} << 40);
    meta_key(fx, "k.i64", 11); put_i64(fx.meta, -(int64_t{1} << 40));
    meta_key(fx, "k.f64", 12); put_f64(fx.meta, 2.25);

    // Array of I32, len 3.
    meta_key(fx, "k.arr_i32", 9);
    put_u32(fx.meta, 5);        // elem type I32
    put_u64(fx.meta, 3);
    put_i32(fx.meta, 1);
    put_i32(fx.meta, -2);
    put_i32(fx.meta, 3);

    // Array of String, len 2.
    meta_key(fx, "k.arr_str", 9);
    put_u32(fx.meta, 8);        // elem type String
    put_u64(fx.meta, 2);
    put_str(fx.meta, "a");
    put_str(fx.meta, "bb");

    // Empty array of F32.
    meta_key(fx, "k.arr_empty", 9);
    put_u32(fx.meta, 6);        // elem type F32
    put_u64(fx.meta, 0);

    // Array of arrays of U8 — exercises parse_value's recursion.
    meta_key(fx, "k.arr_nested", 9);
    put_u32(fx.meta, 9);        // elem type Array
    put_u64(fx.meta, 2);
    for (int i = 0; i < 2; ++i) {
        put_u32(fx.meta, 0);    // inner elem type U8
        put_u64(fx.meta, 2);
        put_u8(fx.meta, u8c(10 + 2 * i));
        put_u8(fx.meta, u8c(11 + 2 * i));
    }

    // No tensors — the reader still aligns past the (empty) tensor-info table.
    return write_temp("meta", assemble(fx, kMagic, 3, 16));
}

// One tensor per supported ggml type (plus an unknown type and a few shapes)
// so the whole ggml type table is walked.
std::filesystem::path write_types_fixture() {
    Fixture fx;

    // F32, 1D [4] — real values, checked after upload.
    {
        const uint64_t off = static_cast<uint64_t>(fx.blob.size());
        for (int i = 0; i < 4; ++i) put_f32(fx.blob, static_cast<float>(i + 1));
        put_tinfo(fx, "f32.1d", { 4 }, 0, off);
    }
    // F16, 2D [2,2] — bit patterns 1.0, 2.0, 3.0, 4.0.
    {
        const uint64_t off = static_cast<uint64_t>(fx.blob.size());
        put_u16(fx.blob, u16c(0x3C00));
        put_u16(fx.blob, u16c(0x4000));
        put_u16(fx.blob, u16c(0x4200));
        put_u16(fx.blob, u16c(0x4400));
        put_tinfo(fx, "f16.2d", { 2, 2 }, 1, off);
    }
    // BF16, 1D [4] — bit patterns 1.0, 2.0, 3.0, 4.0.
    {
        const uint64_t off = static_cast<uint64_t>(fx.blob.size());
        put_u16(fx.blob, u16c(0x3F80));
        put_u16(fx.blob, u16c(0x4000));
        put_u16(fx.blob, u16c(0x4040));
        put_u16(fx.blob, u16c(0x4080));
        put_tinfo(fx, "bf16.1d", { 4 }, 30, off);
    }

    // 32-element-block legacy quants: one block each, except Q8_0 which gets
    // two blocks laid out 2D so the natural shape_to_2d → upload_raw path is
    // exercised for a quant carrier.
    put_tinfo(fx, "q4_0", { 32 },    2, blob_pattern(fx, 18, u8c(0x10)));
    put_tinfo(fx, "q4_1", { 32 },    3, blob_pattern(fx, 20, u8c(0x20)));
    put_tinfo(fx, "q5_0", { 32 },    6, blob_pattern(fx, 22, u8c(0x30)));
    put_tinfo(fx, "q5_1", { 32 },    7, blob_pattern(fx, 24, u8c(0x40)));
    put_tinfo(fx, "q8_0", { 32, 2 }, 8, blob_pattern(fx, 68, u8c(0x50)));
    put_tinfo(fx, "q8_1", { 32 },    9, blob_pattern(fx, 36, u8c(0x60)));

    // 256-element-superblock K-quants.
    put_tinfo(fx, "q2_k", { 256 }, 10, blob_pattern(fx,  82, u8c(0x70)));
    put_tinfo(fx, "q3_k", { 256 }, 11, blob_pattern(fx, 110, u8c(0x80)));
    put_tinfo(fx, "q4_k", { 256 }, 12, blob_pattern(fx, 144, u8c(0x90)));
    put_tinfo(fx, "q5_k", { 256 }, 13, blob_pattern(fx, 176, u8c(0xA0)));
    put_tinfo(fx, "q6_k", { 256 }, 14, blob_pattern(fx, 210, u8c(0xB0)));
    put_tinfo(fx, "q8_k", { 256 }, 15, blob_pattern(fx, 292, u8c(0xC0)));

    // Unknown ggml type: no dtype, no size, still bounds-checked at offset 0.
    put_tinfo(fx, "unknown", { 4 }, 99, 0);

    // 3D and 4D F32 shapes (for shape_to_2d flattening off a real file), and
    // the n_dims == 8 boundary.
    put_tinfo(fx, "f32.3d", { 2, 3, 4 }, 0, blob_pattern(fx, 96, u8c(0x01)));
    put_tinfo(fx, "f32.4d", { 2, 2, 2, 2 }, 0, blob_pattern(fx, 64, u8c(0x02)));
    put_tinfo(fx, "f32.8d", { 1, 1, 1, 1, 1, 1, 1, 1 }, 0,
              blob_pattern(fx, 4, u8c(0x03)));

    return write_temp("types", assemble(fx, kMagic, 3, 32));
}

// ─── tests ─────────────────────────────────────────────────────────────────

void test_metadata_scalar_types() {
    auto f = gg::File::open(write_meta_fixture().string());

    CHECK(f.version() == 3u);
    CHECK(f.alignment() == 16u);   // read from general.alignment
    CHECK(f.tensor_count() == 0u);
    CHECK(f.metadata().size() == 18u);

    const auto& u8 = f.get_meta("k.u8");
    CHECK(u8.type == gg::ValueType::U8);
    CHECK(u8.scalar.u8 == 200);

    const auto& i8 = f.get_meta("k.i8");
    CHECK(i8.type == gg::ValueType::I8);
    CHECK(i8.scalar.i8 == -5);

    const auto& u16 = f.get_meta("k.u16");
    CHECK(u16.type == gg::ValueType::U16);
    CHECK(u16.scalar.u16 == 65000);

    const auto& i16 = f.get_meta("k.i16");
    CHECK(i16.type == gg::ValueType::I16);
    CHECK(i16.scalar.i16 == -300);

    const auto& u32 = f.get_meta("k.u32");
    CHECK(u32.type == gg::ValueType::U32);
    CHECK(u32.scalar.u32 == 4000000000u);

    const auto& i32 = f.get_meta("k.i32");
    CHECK(i32.type == gg::ValueType::I32);
    CHECK(i32.scalar.i32 == -123456);

    const auto& f32 = f.get_meta("k.f32");
    CHECK(f32.type == gg::ValueType::F32);
    CHECK(std::fabs(f32.scalar.f32 - (-2.5f)) < 1e-9f);

    const auto& bt = f.get_meta("k.true");
    CHECK(bt.type == gg::ValueType::Bool);
    CHECK(bt.scalar.b);

    const auto& bf = f.get_meta("k.false");
    CHECK(bf.type == gg::ValueType::Bool);
    CHECK(!bf.scalar.b);

    const auto& s = f.get_meta("k.str");
    CHECK(s.type == gg::ValueType::String);
    CHECK(s.str == "hello gguf");

    const auto& u64 = f.get_meta("k.u64");
    CHECK(u64.type == gg::ValueType::U64);
    CHECK(u64.scalar.u64 == (uint64_t{1} << 40));

    const auto& i64 = f.get_meta("k.i64");
    CHECK(i64.type == gg::ValueType::I64);
    CHECK(i64.scalar.i64 == -(int64_t{1} << 40));

    const auto& f64 = f.get_meta("k.f64");
    CHECK(f64.type == gg::ValueType::F64);
    CHECK(std::fabs(f64.scalar.f64 - 2.25) < 1e-12);
}

void test_metadata_arrays_and_lookup() {
    auto f = gg::File::open(write_meta_fixture().string());

    const auto& ai = f.get_meta("k.arr_i32");
    CHECK(ai.type == gg::ValueType::Array);
    CHECK(ai.array_elem_type == gg::ValueType::I32);
    CHECK(ai.array.size() == 3u);
    if (ai.array.size() == 3u) {
        CHECK(ai.array[0].scalar.i32 == 1);
        CHECK(ai.array[1].scalar.i32 == -2);
        CHECK(ai.array[2].scalar.i32 == 3);
        CHECK(ai.array[0].type == gg::ValueType::I32);
    }

    const auto& as = f.get_meta("k.arr_str");
    CHECK(as.array_elem_type == gg::ValueType::String);
    CHECK(as.array.size() == 2u);
    if (as.array.size() == 2u) {
        CHECK(as.array[0].str == "a");
        CHECK(as.array[1].str == "bb");
    }

    const auto& ae = f.get_meta("k.arr_empty");
    CHECK(ae.type == gg::ValueType::Array);
    CHECK(ae.array_elem_type == gg::ValueType::F32);
    CHECK(ae.array.empty());

    const auto& an = f.get_meta("k.arr_nested");
    CHECK(an.array_elem_type == gg::ValueType::Array);
    CHECK(an.array.size() == 2u);
    if (an.array.size() == 2u) {
        CHECK(an.array[0].type == gg::ValueType::Array);
        CHECK(an.array[0].array_elem_type == gg::ValueType::U8);
        CHECK(an.array[0].array.size() == 2u);
        if (an.array[0].array.size() == 2u) {
            CHECK(an.array[0].array[0].scalar.u8 == 10);
            CHECK(an.array[0].array[1].scalar.u8 == 11);
        }
        CHECK(an.array[1].array.size() == 2u);
        if (an.array[1].array.size() == 2u) {
            CHECK(an.array[1].array[0].scalar.u8 == 12);
        }
    }

    // Key-miss behaviour.
    CHECK(f.find_meta("k.does_not_exist") == nullptr);
    CHECK_THROWS((void)f.get_meta("k.does_not_exist"));

    // Tensor-miss behaviour on a file with no tensors at all.
    CHECK(f.tensors().empty());
    CHECK(f.find_tensor("anything") == nullptr);
    CHECK_THROWS((void)f.get_tensor("anything"));
}

void test_ggml_type_table() {
    auto f = gg::File::open(write_types_fixture().string());
    CHECK(f.alignment() == 32u);  // default — fixture sets no general.alignment
    CHECK(f.tensor_count() == 19u);   // 18 typed + 1 unknown-type

    struct Expect {
        const char* name;
        Dtype       dtype;
        int64_t     numel;
        std::size_t nbytes;
    };
    const Expect want[] = {
        { "f32.1d", Dtype::FP32,   4,  16 },
        { "f16.2d", Dtype::FP16,   4,   8 },
        { "bf16.1d", Dtype::BF16,  4,   8 },
        { "q4_0",   Dtype::Q4_0,  32,  18 },
        { "q4_1",   Dtype::Q4_1,  32,  20 },
        { "q5_0",   Dtype::Q5_0,  32,  22 },
        { "q5_1",   Dtype::Q5_1,  32,  24 },
        { "q8_0",   Dtype::Q8_0,  64,  68 },
        { "q8_1",   Dtype::Q8_1,  32,  36 },
        { "q2_k",   Dtype::Q2_K, 256,  82 },
        { "q3_k",   Dtype::Q3_K, 256, 110 },
        { "q4_k",   Dtype::Q4_K, 256, 144 },
        { "q5_k",   Dtype::Q5_K, 256, 176 },
        { "q6_k",   Dtype::Q6_K, 256, 210 },
        { "q8_k",   Dtype::Q8_K, 256, 292 },
        { "f32.3d", Dtype::FP32,  24,  96 },
        { "f32.4d", Dtype::FP32,  16,  64 },
        { "f32.8d", Dtype::FP32,   1,   4 },
    };
    for (const auto& e : want) {
        const gg::TensorInfo* t = f.find_tensor(e.name);
        CHECK(t != nullptr);
        if (!t) continue;
        CHECK(t->dtype_supported);
        CHECK(t->dtype == e.dtype);
        CHECK(t->numel == e.numel);
        CHECK(t->nbytes == e.nbytes);
        CHECK(t->data != nullptr);
        // nbytes must agree with the dtype's own storage accounting.
        CHECK(t->nbytes == brotensor::dtype_storage_bytes(e.dtype, e.numel));
    }

    // Unknown ggml type: flagged unsupported, zero-sized, FP32 placeholder.
    const gg::TensorInfo& unk = f.get_tensor("unknown");
    CHECK(!unk.dtype_supported);
    CHECK(unk.ggml_type == 99u);
    CHECK(unk.dtype == Dtype::FP32);   // placeholder — dtype_supported is false
    CHECK(unk.numel == 4);
    CHECK(unk.nbytes == 0u);

    // shape_to_2d off the real (innermost-first) on-disk shapes.
    CHECK(gg::shape_to_2d(f.get_tensor("f32.1d").shape) == std::make_pair(4, 1));
    CHECK(gg::shape_to_2d(f.get_tensor("f16.2d").shape) == std::make_pair(2, 2));
    CHECK(gg::shape_to_2d(f.get_tensor("q8_0").shape)   == std::make_pair(2, 32));
    CHECK(gg::shape_to_2d(f.get_tensor("f32.3d").shape) == std::make_pair(12, 2));
    CHECK(gg::shape_to_2d(f.get_tensor("f32.4d").shape) == std::make_pair(8, 2));
    CHECK(gg::shape_to_2d(f.get_tensor("f32.8d").shape) == std::make_pair(1, 1));
}

void test_shape_to_2d() {
    CHECK(gg::shape_to_2d({ 5 })          == std::make_pair(5, 1));
    CHECK(gg::shape_to_2d({ 3, 2 })       == std::make_pair(2, 3));
    CHECK(gg::shape_to_2d({ 2, 3, 4 })    == std::make_pair(12, 2));
    CHECK(gg::shape_to_2d({ 2, 2, 2, 2 }) == std::make_pair(8, 2));

    CHECK_THROWS((void)gg::shape_to_2d({}));                    // empty
    CHECK_THROWS((void)gg::shape_to_2d({ 0 }));                 // non-positive
    CHECK_THROWS((void)gg::shape_to_2d({ -1, 2 }));             // negative
    CHECK_THROWS((void)gg::shape_to_2d({ int64_t{0x80000000} }));  // > INT_MAX
    // Row product (all dims but the innermost) overflows int.
    CHECK_THROWS((void)gg::shape_to_2d({ 2, 100000, 100000 }));
}

void test_upload_raw_dtypes() {
    brotensor::DeviceScope cpu(brotensor::Device::CPU);
    auto f = gg::File::open(write_types_fixture().string());

    // FP32 — values roundtrip.
    {
        const gg::TensorInfo& info = f.get_tensor("f32.1d");
        auto rc = gg::shape_to_2d(info.shape);
        brotensor::Tensor t;
        gg::upload_raw(info, rc.first, rc.second, t);
        CHECK(t.device == brotensor::Device::CPU);
        CHECK(t.rows == 4 && t.cols == 1);
        CHECK(t.dtype == Dtype::FP32);
        CHECK(t.bytes() == 16u);
        auto v = t.to_host_vector();
        CHECK(v.size() == 4u);
        for (int i = 0; i < 4; ++i) {
            CHECK(v[static_cast<std::size_t>(i)] == static_cast<float>(i + 1));
        }
    }

    // FP16 — bit patterns roundtrip.
    {
        const gg::TensorInfo& info = f.get_tensor("f16.2d");
        auto rc = gg::shape_to_2d(info.shape);
        brotensor::Tensor t;
        gg::upload_raw(info, rc.first, rc.second, t);
        CHECK(t.dtype == Dtype::FP16);
        CHECK(t.bytes() == 8u);
        auto v = t.to_host_vector_fp16();
        CHECK(v.size() == 4u);
        if (v.size() == 4u) {
            CHECK(v[0] == 0x3C00);
            CHECK(v[1] == 0x4000);
            CHECK(v[2] == 0x4200);
            CHECK(v[3] == 0x4400);
        }
    }

    // BF16 — bit patterns roundtrip.
    {
        const gg::TensorInfo& info = f.get_tensor("bf16.1d");
        brotensor::Tensor t;
        gg::upload_raw(info, 4, 1, t);
        CHECK(t.dtype == Dtype::BF16);
        CHECK(t.bytes() == 8u);
        auto v = t.to_host_vector_bf16();
        CHECK(v.size() == 4u);
        if (v.size() == 4u) {
            CHECK(v[0] == 0x3F80);
            CHECK(v[1] == 0x4000);
            CHECK(v[2] == 0x4040);
            CHECK(v[3] == 0x4080);
        }
    }

    // Q8_0, 2D: (rows=2, cols=32) — cols is a whole number of 32-elem blocks,
    // so the shape_to_2d result feeds upload_raw directly.
    {
        const gg::TensorInfo& info = f.get_tensor("q8_0");
        auto rc = gg::shape_to_2d(info.shape);
        brotensor::Tensor t;
        gg::upload_raw(info, rc.first, rc.second, t);
        CHECK(t.rows == 2 && t.cols == 32);
        CHECK(t.dtype == Dtype::Q8_0);
        CHECK(t.bytes() == 68u);
        const uint8_t* p = static_cast<const uint8_t*>(t.host_raw());
        CHECK(p != nullptr);
        if (p) {
            CHECK(p[0] == u8c(0x50));
            CHECK(p[67] == u8c(0x50 + 67));
        }
    }

    // Q4_K / Q6_K superblocks: a single 256-wide row.
    {
        const gg::TensorInfo& info = f.get_tensor("q4_k");
        brotensor::Tensor t;
        gg::upload_raw(info, 1, 256, t);
        CHECK(t.dtype == Dtype::Q4_K);
        CHECK(t.bytes() == 144u);
        const uint8_t* p = static_cast<const uint8_t*>(t.host_raw());
        if (p) CHECK(p[0] == u8c(0x90));
    }
    {
        const gg::TensorInfo& info = f.get_tensor("q6_k");
        brotensor::Tensor t;
        gg::upload_raw(info, 1, 256, t);
        CHECK(t.dtype == Dtype::Q6_K);
        CHECK(t.bytes() == 210u);
        const uint8_t* p = static_cast<const uint8_t*>(t.host_raw());
        if (p) CHECK(p[0] == u8c(0xB0));
    }
}

void test_upload_raw_errors() {
    brotensor::DeviceScope cpu(brotensor::Device::CPU);
    auto f = gg::File::open(write_types_fixture().string());

    brotensor::Tensor t;

    // Unsupported ggml type.
    CHECK_THROWS(gg::upload_raw(f.get_tensor("unknown"), 4, 1, t));

    // Non-positive rows / cols.
    const gg::TensorInfo& f32 = f.get_tensor("f32.1d");
    CHECK_THROWS(gg::upload_raw(f32, 0, 1, t));
    CHECK_THROWS(gg::upload_raw(f32, 4, 0, t));
    CHECK_THROWS(gg::upload_raw(f32, -4, 1, t));

    // rows * cols != numel.
    CHECK_THROWS(gg::upload_raw(f32, 3, 1, t));
    CHECK_THROWS(gg::upload_raw(f32, 4, 2, t));

    // Quant dtype whose cols isn't a whole number of blocks. shape_to_2d on a
    // 1D quant tensor yields (256, 1), which is exactly this case — a quant
    // tensor stored 1D must be uploaded as a single row instead.
    const gg::TensorInfo& q4k = f.get_tensor("q4_k");
    auto rc = gg::shape_to_2d(q4k.shape);
    CHECK(rc.first == 256 && rc.second == 1);
    CHECK_THROWS(gg::upload_raw(q4k, rc.first, rc.second, t));
    CHECK_THROWS(gg::upload_raw(q4k, 2, 128, t));   // 128 % 256 != 0

    // The failing calls must not have committed anything into `t`.
    CHECK(t.data == nullptr);
}

void test_file_move_semantics() {
    auto path = write_types_fixture().string();

    gg::File a = gg::File::open(path);
    CHECK(a.tensor_count() == 19u);

    // Move-construct: `b` takes the mapping, `a` is left empty but valid.
    gg::File b(std::move(a));
    CHECK(b.tensor_count() == 19u);
    CHECK(b.version() == 3u);
    CHECK(b.find_tensor("f32.1d") != nullptr);
    CHECK(a.tensor_count() == 0u);     // NOLINT — moved-from state is defined
    CHECK(a.version() == 0u);
    CHECK(a.alignment() == 32u);
    CHECK(a.find_tensor("f32.1d") == nullptr);

    // Move-assign onto a live File: releases the old mapping first.
    gg::File c = gg::File::open(write_meta_fixture().string());
    CHECK(c.alignment() == 16u);
    c = std::move(b);
    CHECK(c.tensor_count() == 19u);
    CHECK(c.alignment() == 32u);
    CHECK(c.get_tensor("f32.1d").numel == 4);
    CHECK(c.find_meta("k.u8") == nullptr);   // metadata came from the types file
    CHECK(b.tensor_count() == 0u);           // NOLINT — moved-from
}

void test_open_header_errors() {
    // Nonexistent path.
    CHECK_THROWS((void)gg::File::open(
        (std::filesystem::temp_directory_path() / "brotensor_gguf_nope.gguf")
            .string()));

    // Shorter than the 24-byte header.
    {
        std::vector<uint8_t> b(12, u8c(0xAB));
        auto p = write_temp("small", b);
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Bad magic.
    {
        Fixture fx;
        auto p = write_temp("badmagic", assemble(fx, 0xDEADBEEFu, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Unsupported version (only 2 and 3 are accepted).
    {
        Fixture fx;
        auto p = write_temp("badver", assemble(fx, kMagic, 1, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Version 2 is accepted.
    {
        Fixture fx;
        put_tinfo(fx, "t", { 2 }, 0, blob_pattern(fx, 8, u8c(0x11)));
        auto p = write_temp("v2", assemble(fx, kMagic, 2, 32));
        auto f = gg::File::open(p.string());
        CHECK(f.version() == 2u);
        CHECK(f.tensor_count() == 1u);
        CHECK(f.get_tensor("t").nbytes == 8u);
    }

    // Truncated: header claims one metadata KV, but the file stops there.
    {
        std::vector<uint8_t> b;
        put_u32(b, kMagic);
        put_u32(b, 3);
        put_u64(b, 0);   // tensor_count
        put_u64(b, 1);   // metadata_kv_count — nothing follows
        auto p = write_temp("trunc", b);
        CHECK_THROWS((void)gg::File::open(p.string()));
    }
}

void test_open_metadata_errors() {
    // Metadata value type out of range (> 12).
    {
        Fixture fx;
        meta_key(fx, "k.bad", 13);
        put_u32(fx.meta, 0);
        auto p = write_temp("badvt", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Array element type out of range (> 12).
    {
        Fixture fx;
        meta_key(fx, "k.badarr", 9);
        put_u32(fx.meta, 13);   // element type
        put_u64(fx.meta, 0);
        auto p = write_temp("badarr", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Bool payload that is neither 0 nor 1.
    {
        Fixture fx;
        meta_key(fx, "k.bool", 7);
        put_u8(fx.meta, u8c(2));
        auto p = write_temp("badbool", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // general.alignment must be U32...
    {
        Fixture fx;
        meta_key(fx, "general.alignment", 10);   // U64
        put_u64(fx.meta, 32);
        auto p = write_temp("align_type", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }
    // ...and a positive power of two.
    {
        Fixture fx;
        meta_key(fx, "general.alignment", 4);
        put_u32(fx.meta, 0);
        auto p = write_temp("align_zero", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }
    {
        Fixture fx;
        meta_key(fx, "general.alignment", 4);
        put_u32(fx.meta, 24);   // not a power of two
        auto p = write_temp("align_npot", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }
}

void test_open_tensor_errors() {
    // n_dims == 0.
    {
        Fixture fx;
        put_str(fx.tinfo, "t");
        put_u32(fx.tinfo, 0);   // n_dims
        fx.n_tensors = 1;
        auto p = write_temp("ndims0", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // n_dims > 8.
    {
        Fixture fx;
        put_str(fx.tinfo, "t");
        put_u32(fx.tinfo, 9);   // n_dims
        for (int i = 0; i < 9; ++i) put_u64(fx.tinfo, 1);
        put_u32(fx.tinfo, 0);
        put_u64(fx.tinfo, 0);
        fx.n_tensors = 1;
        auto p = write_temp("ndims9", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Non-positive dimension.
    {
        Fixture fx;
        put_tinfo(fx, "t", { 0 }, 0, blob_pattern(fx, 4, u8c(0x01)));
        auto p = write_temp("dim0", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // numel overflows int64 (2^62 * 8).
    {
        Fixture fx;
        put_tinfo(fx, "t", { uint64_t{1} << 62, 8 }, 0,
                  blob_pattern(fx, 4, u8c(0x01)));
        auto p = write_temp("numel_ovf", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Quant tensor whose numel isn't a multiple of the block size.
    {
        Fixture fx;
        put_tinfo(fx, "t", { 16 }, 2, blob_pattern(fx, 18, u8c(0x01)));  // Q4_0
        auto p = write_temp("blockmis", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Tensor data offset lands past the end of the file.
    {
        Fixture fx;
        put_tinfo(fx, "t", { 4 }, 0, uint64_t{1} << 30);
        blob_pattern(fx, 16, u8c(0x01));
        auto p = write_temp("oob", assemble(fx, kMagic, 3, 32));
        CHECK_THROWS((void)gg::File::open(p.string()));
    }

    // Data blob would start past EOF: one tensor info, file ends right after
    // it, and the aligned blob start rounds up beyond the file size.
    {
        Fixture fx;
        put_tinfo(fx, "a", { 1 }, 0, 0);
        auto bytes = assemble(fx, kMagic, 3, 0);   // no padding, no blob
        CHECK(bytes.size() % 32u != 0u);           // else the check can't fire
        auto p = write_temp("noblob", bytes);
        CHECK_THROWS((void)gg::File::open(p.string()));
    }
}

}  // namespace

int main() {
    test_metadata_scalar_types();
    test_metadata_arrays_and_lookup();
    test_ggml_type_table();
    test_shape_to_2d();
    test_upload_raw_dtypes();
    test_upload_raw_errors();
    test_file_move_semantics();
    test_open_header_errors();
    test_open_metadata_errors();
    test_open_tensor_errors();

    std::error_code ec;
    for (const auto& p : g_temps) std::filesystem::remove(p, ec);

    if (g_failures == 0) std::printf("gguf_extra: OK\n");
    else std::fprintf(stderr, "gguf_extra: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
