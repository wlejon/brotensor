// ─── safetensors: extended coverage (CPU-only) ─────────────────────────────
//
// Complements tests/test_safetensors.cpp. Targets the parts of
// src/safetensors.cpp the happy-path test never reaches:
//
//   1.  dtype_name / dtype_size_bytes over every Dtype (incl. Unknown).
//   2.  TensorView::numel() guards (negative dim, int64 overflow).
//   3.  File::open() failure paths — missing file, truncated file, bogus
//       header_size, every JSON parse error, offset/shape inconsistencies.
//   4.  Header oddities the parser must tolerate — __metadata__, unknown
//       entry fields of every JSON value kind, escaped tensor names.
//   5.  find() miss / get() throw, plus File move-construct + move-assign.
//   6.  upload() for F32/F16/BF16 and its rejection paths.
//   7.  upload_fp16() — F16 passthrough, F32 narrowing, BF16 rejection.
//   8.  upload_as() — all 3x3 source/target dtype combinations + rejections.
//   9.  upload_compute() / upload_compute_checked() incl. both check throws.
//   10. write_file() round-trip across all seven writable dtypes plus a
//       zero-element entry, re-read through File::open().
//   11. write_file() JSON name escaping.
//   12. write_file() validation + I/O failure paths.
//
// Everything runs on the always-available CPU backend (compute dtype FP32);
// no GPU device is touched.

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace st = brotensor::safetensors;
namespace fs = std::filesystem;

using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// ─── Fixtures / helpers ────────────────────────────────────────────────────

static fs::path tmp_path(const char* leaf) {
    return fs::temp_directory_path() / leaf;
}

static void remove_quiet(const fs::path& p) {
    std::error_code ec;
    fs::remove(p, ec);
}

// Write a safetensors-shaped file with an explicit (possibly wrong)
// header_size field, so the truncation / bad-length paths can be reached.
static void write_raw(const fs::path& p, uint64_t header_size,
                      const std::string& header,
                      const std::vector<uint8_t>& payload) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("test: cannot create fixture");
    f.write(reinterpret_cast<const char*>(&header_size), 8);
    if (!header.empty())
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!payload.empty())
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
}

// Write a well-formed container around `header` + `payload`, then try to open
// it. Returns true iff File::open() threw std::runtime_error.
static bool open_throws(const std::string& header,
                        const std::vector<uint8_t>& payload) {
    const fs::path p = tmp_path("brotensor_stx_err.safetensors");
    write_raw(p, static_cast<uint64_t>(header.size()), header, payload);
    bool threw = false;
    {
        try {
            st::File f = st::File::open(p.string());
            (void)f;
        } catch (const std::runtime_error&) {
            threw = true;
        }
    }
    remove_quiet(p);
    return threw;
}

static st::TensorView make_view(const char* name, st::Dtype dt,
                                std::vector<int64_t> shape,
                                const void* data, std::size_t nbytes) {
    st::TensorView v;
    v.name   = name;
    v.dtype  = dt;
    v.shape  = std::move(shape);
    v.data   = static_cast<const uint8_t*>(data);
    v.nbytes = nbytes;
    return v;
}

// Shared source payloads for the upload family: the values 1..6 (and 1..4),
// each exactly representable in FP32, FP16 and BF16, so every conversion in
// the matrix below is lossless and can be compared with ==.
static const float    kF32[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
static const uint16_t kF16[6] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
static const uint16_t kBF16[6] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0};

// ── 1. dtype tables ───────────────────────────────────────────────────────
static void test_dtype_tables() {
    CHECK(std::strcmp(st::dtype_name(st::Dtype::F32),  "F32")  == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::F16),  "F16")  == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::BF16), "BF16") == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::I32),  "I32")  == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::I64),  "I64")  == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::U8),   "U8")   == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::BOOL), "BOOL") == 0);
    CHECK(std::strcmp(st::dtype_name(st::Dtype::Unknown), "Unknown") == 0);
    // Out-of-range value takes the default arm.
    CHECK(std::strcmp(st::dtype_name(static_cast<st::Dtype>(99)), "Unknown") == 0);

    CHECK(st::dtype_size_bytes(st::Dtype::F32)  == 4);
    CHECK(st::dtype_size_bytes(st::Dtype::F16)  == 2);
    CHECK(st::dtype_size_bytes(st::Dtype::BF16) == 2);
    CHECK(st::dtype_size_bytes(st::Dtype::I32)  == 4);
    CHECK(st::dtype_size_bytes(st::Dtype::I64)  == 8);
    CHECK(st::dtype_size_bytes(st::Dtype::U8)   == 1);
    CHECK(st::dtype_size_bytes(st::Dtype::BOOL) == 1);
    CHECK(st::dtype_size_bytes(st::Dtype::Unknown) == 0);
    CHECK(st::dtype_size_bytes(static_cast<st::Dtype>(99)) == 0);
}

// ── 2. TensorView::numel guards ───────────────────────────────────────────
static void test_numel_guards() {
    st::TensorView v;
    CHECK(v.numel() == 1);            // empty shape is the scalar product

    v.shape = {2, 3, 4};
    CHECK(v.numel() == 24);

    v.shape = {2, -1};
    bool threw = false;
    try { (void)v.numel(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    v.shape = {INT64_MAX, 2};
    threw = false;
    try { (void)v.numel(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // A zero dim short-circuits the overflow guard rather than dividing by 0.
    v.shape = {0, INT64_MAX};
    CHECK(v.numel() == 0);
}

// ── 3. File::open failure paths ───────────────────────────────────────────
static void test_open_errors() {
    // Nonexistent path.
    bool threw = false;
    try {
        st::File f = st::File::open(
            (fs::temp_directory_path() / "brotensor_stx_does_not_exist.bin").string());
        (void)f;
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Fewer than 8 bytes: cannot even hold the header length.
    {
        const fs::path p = tmp_path("brotensor_stx_tiny.safetensors");
        {
            std::ofstream f(p, std::ios::binary | std::ios::trunc);
            const char four[4] = {'a', 'b', 'c', 'd'};
            f.write(four, 4);
        }
        threw = false;
        try { st::File f = st::File::open(p.string()); (void)f; }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        remove_quiet(p);
    }

    // header_size larger than the bytes that actually follow it.
    {
        const fs::path p = tmp_path("brotensor_stx_hdrlen.safetensors");
        write_raw(p, /*header_size*/ 4096, "{}", {});
        threw = false;
        try { st::File f = st::File::open(p.string()); (void)f; }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        remove_quiet(p);
    }

    // ── JSON parse errors (header_size is always correct here) ──
    CHECK(open_throws("not json at all", {}));                 // expected '{'
    CHECK(open_throws(R"({"a")", {}));                         // expected ':'
    CHECK(open_throws(R"({"abc)", {}));                        // unterminated string
    // ("\q" spelled out rather than in a raw literal: MSVC raises a bogus
    //  C4129 "unrecognized character escape" for backslashes in raw strings.)
    CHECK(open_throws("{\"a\\qb\":{}}", {}));                  // unsupported escape
    CHECK(open_throws(R"({5:{}})", {}));                       // expected string (key)
    CHECK(open_throws(
        R"({"t":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}})",
        std::vector<uint8_t>(8, 0)));                          // unsupported dtype
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[x],"data_offsets":[0,4]}})",
        std::vector<uint8_t>(4, 0)));                          // expected integer
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[99999999999999999999],"data_offsets":[0,4]}})",
        std::vector<uint8_t>(4, 0)));                          // integer out of range
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[0 4]}})",
        std::vector<uint8_t>(4, 0)));                          // expected ',' in offsets

    // data_offsets past the end of the payload.
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
        std::vector<uint8_t>(8, 0)));
    // data_offsets reversed (end < start).
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})",
        std::vector<uint8_t>(16, 0)));
    // nbytes disagrees with shape * dtype_size.
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
        std::vector<uint8_t>(8, 0)));
    // Negative dim in shape: numel() rejects it while indexing the header.
    CHECK(open_throws(
        R"({"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})",
        std::vector<uint8_t>(4, 0)));
}

// ── 4. header oddities the parser must tolerate ───────────────────────────
static void test_header_extras() {
    // __metadata__ (nested object, nested array, escaped string) is skipped;
    // unknown per-tensor fields of every JSON value kind are skipped too;
    // whitespace between tokens is ignored.
    const std::string header =
        "{ \"__metadata__\" : {\"format\":\"pt\", \"note\":\"a\\\"b\","
        " \"arr\":[1, 2, {\"x\":\"y\"}]} ,\n"
        " \"t\" : { \"note\":\"hi\", \"rank\":2, \"flag\":true, \"nil\":null,"
        " \"arr\":[1,2,3], \"obj\":{\"k\":\"v\"},"
        " \"dtype\":\"F32\", \"shape\":[2], \"data_offsets\":[0,8] } }";

    std::vector<uint8_t> payload(8, 0);
    const float vals[2] = {7.5f, -2.25f};
    std::memcpy(payload.data(), vals, 8);

    const fs::path p = tmp_path("brotensor_stx_extras.safetensors");
    write_raw(p, static_cast<uint64_t>(header.size()), header, payload);
    {
        st::File f = st::File::open(p.string());
        CHECK(f.size() == 1);                 // __metadata__ is not a tensor
        const st::TensorView* t = f.find("t");
        CHECK(t != nullptr);
        if (t) {
            CHECK(t->dtype == st::Dtype::F32);
            CHECK(t->nbytes == 8);
            CHECK(t->numel() == 2);
            const float* fp = reinterpret_cast<const float*>(t->data);
            CHECK(fp[0] == 7.5f && fp[1] == -2.25f);
        }
    }
    remove_quiet(p);

    // Every string escape the reader supports, inside a tensor name. Spelled
    // with ordinary (non-raw) literals: MSVC raises a bogus C4129
    // "unrecognized character escape" for backslashes inside raw strings.
    const std::string esc_header =
        "{\"a\\\"b\\\\c\\/d\\ne\\tf\\rg\\bh\\fi\":"
        "{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    const std::string esc_name = "a\"b\\c/d\ne\tf\rg\bh\fi";
    const fs::path ep = tmp_path("brotensor_stx_escname.safetensors");
    write_raw(ep, static_cast<uint64_t>(esc_header.size()), esc_header, {0xAB, 0xCD});
    {
        st::File f = st::File::open(ep.string());
        CHECK(f.size() == 1);
        const st::TensorView* t = f.find(esc_name);
        CHECK(t != nullptr);
        if (t) {
            CHECK(t->dtype == st::Dtype::U8);
            CHECK(t->nbytes == 2);
            CHECK(t->data[0] == 0xAB && t->data[1] == 0xCD);
        }
    }
    remove_quiet(ep);

    // An empty header object is a valid (zero-tensor) file.
    const fs::path zp = tmp_path("brotensor_stx_empty.safetensors");
    write_raw(zp, 2, "{}", {});
    {
        st::File f = st::File::open(zp.string());
        CHECK(f.size() == 0);
        CHECK(f.tensors().empty());
        CHECK(f.find("anything") == nullptr);
    }
    remove_quiet(zp);
}

// ── 5. find / get / move semantics ────────────────────────────────────────
static void test_lookup_and_move() {
    const std::string header =
        R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    std::vector<uint8_t> payload(8, 0);
    const float vals[2] = {3.f, 4.f};
    std::memcpy(payload.data(), vals, 8);

    const fs::path p = tmp_path("brotensor_stx_move.safetensors");
    write_raw(p, static_cast<uint64_t>(header.size()), header, payload);
    {
        st::File a = st::File::open(p.string());
        CHECK(a.size() == 1);
        CHECK(a.find("w") != nullptr);
        CHECK(a.find("missing") == nullptr);
        CHECK(&a.get("w") == a.find("w"));

        bool threw = false;
        try { (void)a.get("missing"); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);

        // Move-construct: the mapping and the index follow.
        st::File b(std::move(a));
        CHECK(b.size() == 1);
        CHECK(b.find("w") != nullptr);
        if (b.find("w")) {
            const float* fp = reinterpret_cast<const float*>(b.get("w").data);
            CHECK(fp[0] == 3.f && fp[1] == 4.f);
        }

        // Move-assign over a live File: c's own mapping is released first.
        st::File c = st::File::open(p.string());
        CHECK(c.size() == 1);
        c = std::move(b);
        CHECK(c.size() == 1);
        CHECK(c.find("w") != nullptr);
        if (c.find("w")) {
            const float* fp = reinterpret_cast<const float*>(c.get("w").data);
            CHECK(fp[0] == 3.f && fp[1] == 4.f);
        }

        // Move-assign into a default-constructed (never-mapped) File.
        st::File d;
        CHECK(d.size() == 0);
        d = std::move(c);
        CHECK(d.size() == 1);
        CHECK(d.tensors().size() == 1);
        CHECK(d.tensors()[0].name == "w");
    }
    remove_quiet(p);
}

// ── 6. upload() ───────────────────────────────────────────────────────────
static void test_upload() {
    const st::TensorView f32 = make_view("f32", st::Dtype::F32, {2, 3}, kF32, 24);
    const st::TensorView f16 = make_view("f16", st::Dtype::F16, {2, 3}, kF16, 12);
    const st::TensorView bf16 = make_view("bf16", st::Dtype::BF16, {2, 3}, kBF16, 12);

    Tensor t;
    st::upload(f32, 2, 3, t);
    CHECK(t.rows == 2 && t.cols == 3);
    CHECK(t.device == brotensor::Device::CPU);
    CHECK(t.dtype == brotensor::Dtype::FP32);
    CHECK(t.at(0, 0) == 1.f && t.at(1, 2) == 6.f);

    // F16 view stays FP16 (no conversion) — upload() preserves the disk dtype.
    Tensor h;
    st::upload(f16, 2, 3, h);
    CHECK(h.dtype == brotensor::Dtype::FP16);
    CHECK(h.rows == 2 && h.cols == 3);
    {
        const uint16_t* bits = h.host_fp16();
        for (int i = 0; i < 6; ++i) CHECK(bits[i] == kF16[i]);
    }

    // BF16 view stays BF16.
    Tensor b;
    st::upload(bf16, 2, 3, b);
    CHECK(b.dtype == brotensor::Dtype::BF16);
    {
        const uint16_t* bits = b.host_bf16();
        for (int i = 0; i < 6; ++i) CHECK(bits[i] == kBF16[i]);
    }

    // Non-positive extents.
    bool threw = false;
    try { st::upload(f32, 0, 3, t); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { st::upload(f32, 2, -1, t); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // rows*cols disagrees with nbytes.
    threw = false;
    try { st::upload(f32, 2, 2, t); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Integer dtypes are not uploadable.
    static const int32_t ints[6] = {1, 2, 3, 4, 5, 6};
    const st::TensorView i32 = make_view("i32", st::Dtype::I32, {2, 3}, ints, 24);
    threw = false;
    try { st::upload(i32, 2, 3, t); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 7. upload_fp16() ──────────────────────────────────────────────────────
static void test_upload_fp16() {
    const st::TensorView f16 = make_view("f16", st::Dtype::F16, {2, 3}, kF16, 12);
    const st::TensorView f32 = make_view("f32", st::Dtype::F32, {2, 3}, kF32, 24);
    const st::TensorView bf16 = make_view("bf16", st::Dtype::BF16, {2, 3}, kBF16, 12);

    // F16 source: raw bit patterns pass straight through.
    Tensor a;
    st::upload_fp16(f16, 2, 3, a);
    CHECK(a.dtype == brotensor::Dtype::FP16);
    CHECK(a.rows == 2 && a.cols == 3);
    {
        const uint16_t* bits = a.host_fp16();
        for (int i = 0; i < 6; ++i) CHECK(bits[i] == kF16[i]);
    }

    // F32 source: narrowed host-side to FP16 bit patterns.
    Tensor b;
    st::upload_fp16(f32, 3, 2, b);
    CHECK(b.dtype == brotensor::Dtype::FP16);
    CHECK(b.rows == 3 && b.cols == 2);
    {
        const uint16_t* bits = b.host_fp16();
        for (int i = 0; i < 6; ++i) {
            CHECK(bits[i] == brotensor::fp32_to_fp16_bits(kF32[i]));
            CHECK(brotensor::fp16_bits_to_fp32(bits[i]) == kF32[i]);
        }
    }

    // BF16 is NOT an accepted source for upload_fp16 (unlike upload_as).
    Tensor c;
    bool threw = false;
    try { st::upload_fp16(bf16, 2, 3, c); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { st::upload_fp16(f32, -2, 3, c); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { st::upload_fp16(f32, 2, 4, c); }   // byte count mismatch
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 8. upload_as(): every source x target combination ─────────────────────
static void test_upload_as() {
    const st::TensorView src[3] = {
        make_view("f32",  st::Dtype::F32,  {2, 3}, kF32,  24),
        make_view("f16",  st::Dtype::F16,  {2, 3}, kF16,  12),
        make_view("bf16", st::Dtype::BF16, {2, 3}, kBF16, 12),
    };
    const brotensor::Dtype want[3] = {
        brotensor::Dtype::FP32, brotensor::Dtype::FP16, brotensor::Dtype::BF16,
    };

    for (int s = 0; s < 3; ++s) {
        for (int w = 0; w < 3; ++w) {
            Tensor t;
            st::upload_as(src[s], 2, 3, want[w], t);
            CHECK(t.dtype == want[w]);
            CHECK(t.rows == 2 && t.cols == 3);
            CHECK(t.device == brotensor::Device::CPU);
            // Values 1..6 are exact in all three dtypes, so every conversion
            // in the matrix is lossless.
            for (int i = 0; i < 6; ++i) {
                float got = 0.f;
                if (want[w] == brotensor::Dtype::FP32) {
                    got = t.host_f32()[i];
                } else if (want[w] == brotensor::Dtype::FP16) {
                    got = brotensor::fp16_bits_to_fp32(t.host_fp16()[i]);
                } else {
                    got = brotensor::bf16_bits_to_fp32(t.host_bf16()[i]);
                }
                CHECK(got == kF32[i]);
            }
        }
    }

    Tensor t;
    // Non-arithmetic target dtype.
    bool threw = false;
    try { st::upload_as(src[0], 2, 3, brotensor::Dtype::INT8, t); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Non-float source dtype.
    static const uint8_t bytes6[6] = {1, 2, 3, 4, 5, 6};
    const st::TensorView u8 = make_view("u8", st::Dtype::U8, {2, 3}, bytes6, 6);
    threw = false;
    try { st::upload_as(u8, 2, 3, brotensor::Dtype::FP32, t); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { st::upload_as(src[0], 0, 0, brotensor::Dtype::FP32, t); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { st::upload_as(src[0], 3, 3, brotensor::Dtype::FP32, t); }  // byte mismatch
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 9. upload_compute / upload_compute_checked ────────────────────────────
static void test_upload_compute() {
    CHECK(brotensor::compute_dtype() == brotensor::Dtype::FP32);

    const st::TensorView f32 = make_view("f32", st::Dtype::F32, {2, 3}, kF32, 24);
    const st::TensorView f16 = make_view("f16", st::Dtype::F16, {6}, kF16, 12);

    Tensor a;
    st::upload_compute(f32, 2, 3, a);
    CHECK(a.dtype == brotensor::Dtype::FP32);
    CHECK(a.at(1, 2) == 6.f);

    Tensor b;
    st::upload_compute_checked(f16, 6, 1, b, "layer.weight");
    CHECK(b.dtype == brotensor::Dtype::FP32);
    CHECK(b.rows == 6 && b.cols == 1);
    for (int i = 0; i < 6; ++i) CHECK(b.at(i, 0) == kF32[i]);

    // Rejected source dtype (tagged with the caller-supplied label).
    static const int64_t longs[2] = {1, 2};
    const st::TensorView i64 = make_view("i64", st::Dtype::I64, {2}, longs, 16);
    Tensor c;
    bool threw = false;
    try { st::upload_compute_checked(i64, 2, 1, c, "layer.bias"); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Element count disagrees with rows*cols.
    threw = false;
    try { st::upload_compute_checked(f32, 4, 2, c, "layer.weight"); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 10. write_file() round-trip across every writable dtype ───────────────
static void test_write_roundtrip() {
    const float    f32v[6]  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const uint16_t f16v[4]  = {0x3c00, 0x4000, 0x4200, 0x4400};
    uint16_t       bf16v[4];
    for (int i = 0; i < 4; ++i)
        bf16v[i] = brotensor::fp32_to_bf16_bits(static_cast<float>(i + 1));
    const int32_t  i32v[3]  = {-7, 0, 123456};
    const int64_t  i64v[2]  = {-1, 9000000000LL};
    const uint8_t  u8v[5]   = {0, 1, 2, 254, 255};
    const uint8_t  boolv[2] = {1, 0};

    std::vector<st::WriteEntry> entries;
    auto add = [&](const char* name, st::Dtype dt, std::vector<int64_t> shape,
                   const void* p, std::size_t nbytes) {
        st::WriteEntry e;
        e.name      = name;
        e.dtype     = dt;
        e.shape     = std::move(shape);
        e.host_data = p;
        e.bytes     = nbytes;
        entries.push_back(std::move(e));
    };
    add("w.f32",  st::Dtype::F32,  {2, 3}, f32v,  sizeof(f32v));
    add("w.f16",  st::Dtype::F16,  {4},    f16v,  sizeof(f16v));
    add("w.bf16", st::Dtype::BF16, {4},    bf16v, sizeof(bf16v));
    add("w.i32",  st::Dtype::I32,  {3},    i32v,  sizeof(i32v));
    add("w.i64",  st::Dtype::I64,  {2},    i64v,  sizeof(i64v));
    add("w.u8",   st::Dtype::U8,   {5},    u8v,   sizeof(u8v));
    add("w.bool", st::Dtype::BOOL, {2},    boolv, sizeof(boolv));
    // Zero-element entry: no payload bytes, null host_data is legal.
    add("w.empty", st::Dtype::F32, {0}, nullptr, 0);

    const fs::path p = tmp_path("brotensor_stx_rt.safetensors");
    st::write_file(p.string(), entries);
    {
        st::File f = st::File::open(p.string());
        CHECK(f.size() == 8);

        const st::TensorView* a = f.find("w.f32");
        CHECK(a != nullptr);
        if (a) {
            CHECK(a->dtype == st::Dtype::F32);
            CHECK(a->shape.size() == 2 && a->shape[0] == 2 && a->shape[1] == 3);
            CHECK(a->nbytes == 24 && a->numel() == 6);
            CHECK(std::memcmp(a->data, f32v, sizeof(f32v)) == 0);
            Tensor t;
            st::upload(*a, 2, 3, t);
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.f && t.at(1, 2) == 6.f);
        }

        const st::TensorView* h = f.find("w.f16");
        CHECK(h != nullptr);
        if (h) {
            CHECK(h->dtype == st::Dtype::F16);
            CHECK(h->shape.size() == 1 && h->shape[0] == 4);
            CHECK(h->nbytes == 8);
            CHECK(std::memcmp(h->data, f16v, sizeof(f16v)) == 0);
            Tensor t;
            st::upload_compute(*h, 4, 1, t);   // F16 -> FP32 on the CPU backend
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.f && t.at(3, 0) == 4.f);
        }

        const st::TensorView* b = f.find("w.bf16");
        CHECK(b != nullptr);
        if (b) {
            CHECK(b->dtype == st::Dtype::BF16);
            CHECK(b->nbytes == 8);
            CHECK(std::memcmp(b->data, bf16v, sizeof(bf16v)) == 0);
            Tensor t;
            st::upload_as(*b, 4, 1, brotensor::Dtype::BF16, t);
            CHECK(t.dtype == brotensor::Dtype::BF16);
            const uint16_t* bits = t.host_bf16();
            for (int i = 0; i < 4; ++i) CHECK(bits[i] == bf16v[i]);
        }

        const st::TensorView* i32 = f.find("w.i32");
        CHECK(i32 != nullptr);
        if (i32) {
            CHECK(i32->dtype == st::Dtype::I32);
            CHECK(i32->nbytes == 12 && i32->numel() == 3);
            CHECK(std::memcmp(i32->data, i32v, sizeof(i32v)) == 0);
        }

        const st::TensorView* i64 = f.find("w.i64");
        CHECK(i64 != nullptr);
        if (i64) {
            CHECK(i64->dtype == st::Dtype::I64);
            CHECK(i64->nbytes == 16 && i64->numel() == 2);
            CHECK(std::memcmp(i64->data, i64v, sizeof(i64v)) == 0);
        }

        const st::TensorView* u8 = f.find("w.u8");
        CHECK(u8 != nullptr);
        if (u8) {
            CHECK(u8->dtype == st::Dtype::U8);
            CHECK(u8->nbytes == 5 && u8->numel() == 5);
            CHECK(std::memcmp(u8->data, u8v, sizeof(u8v)) == 0);
        }

        const st::TensorView* bl = f.find("w.bool");
        CHECK(bl != nullptr);
        if (bl) {
            CHECK(bl->dtype == st::Dtype::BOOL);
            CHECK(bl->nbytes == 2 && bl->numel() == 2);
            CHECK(bl->data[0] == 1 && bl->data[1] == 0);
        }

        const st::TensorView* z = f.find("w.empty");
        CHECK(z != nullptr);
        if (z) {
            CHECK(z->dtype == st::Dtype::F32);
            CHECK(z->shape.size() == 1 && z->shape[0] == 0);
            CHECK(z->nbytes == 0);
            CHECK(z->numel() == 0);
        }

        // Insertion order is preserved in the header, hence in tensors().
        CHECK(f.tensors().size() == 8);
        if (f.tensors().size() == 8) {
            CHECK(f.tensors()[0].name == "w.f32");
            CHECK(f.tensors()[7].name == "w.empty");
        }
    }
    remove_quiet(p);

    // Writing an empty entry list produces a valid, tensor-free file.
    const fs::path ep = tmp_path("brotensor_stx_rt_empty.safetensors");
    st::write_file(ep.string(), {});
    {
        st::File f = st::File::open(ep.string());
        CHECK(f.size() == 0);
    }
    remove_quiet(ep);
}

// ── 11. write_file() JSON name escaping ───────────────────────────────────
static void test_write_name_escaping() {
    // Every character the writer has a named escape for, and which the reader
    // can un-escape, so the name survives a round-trip byte for byte.
    const std::string names[7] = {
        "q\"uote",
        "back\\slash",
        "line\nbreak",
        "tab\there",
        "cr\rhere",
        "bs\bhere",
        "ff\fhere",
    };
    uint8_t vals[7];
    for (int i = 0; i < 7; ++i) vals[i] = static_cast<uint8_t>(i + 1);

    std::vector<st::WriteEntry> entries;
    for (int i = 0; i < 7; ++i) {
        st::WriteEntry e;
        e.name      = names[i];
        e.dtype     = st::Dtype::U8;
        e.shape     = {1};
        e.host_data = &vals[i];
        e.bytes     = 1;
        entries.push_back(std::move(e));
    }

    const fs::path p = tmp_path("brotensor_stx_escape.safetensors");
    st::write_file(p.string(), entries);
    {
        st::File f = st::File::open(p.string());
        CHECK(f.size() == 7);
        for (int i = 0; i < 7; ++i) {
            const st::TensorView* v = f.find(names[i]);
            CHECK(v != nullptr);
            if (v) {
                CHECK(v->name == names[i]);
                CHECK(v->nbytes == 1);
                CHECK(v->data[0] == static_cast<uint8_t>(i + 1));
            }
        }
    }
    remove_quiet(p);

    // A control character with no named escape is emitted as \uXXXX. (The
    // reader does not implement \u escapes — see the note in the test's
    // final report — so this checks the writer's output only.)
    std::string ctl = "ctl";
    ctl.push_back('\x01');
    ctl += "x";
    {
        uint8_t one = 42;
        st::WriteEntry e;
        e.name      = ctl;
        e.dtype     = st::Dtype::U8;
        e.shape     = {1};
        e.host_data = &one;
        e.bytes     = 1;
        const fs::path cp = tmp_path("brotensor_stx_ctl.safetensors");
        st::write_file(cp.string(), {e});
        std::ifstream in(cp, std::ios::binary);
        const std::string raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
        in.close();
        CHECK(raw.find("\\u0001") != std::string::npos);
        CHECK(raw.find('\x01') == std::string::npos);  // never emitted literally
        remove_quiet(cp);
    }
}

// ── 12. write_file() validation + I/O failures ────────────────────────────
static void test_write_errors() {
    const float v[2] = {1.f, 2.f};
    const fs::path p = tmp_path("brotensor_stx_werr.safetensors");

    auto entry = [&](const char* name, st::Dtype dt, std::vector<int64_t> shape,
                     const void* data, std::size_t nbytes) {
        st::WriteEntry e;
        e.name      = name;
        e.dtype     = dt;
        e.shape     = std::move(shape);
        e.host_data = data;
        e.bytes     = nbytes;
        return e;
    };
    auto write_throws = [&](const std::vector<st::WriteEntry>& es) {
        bool threw = false;
        try { st::write_file(p.string(), es); }
        catch (const std::runtime_error&) { threw = true; }
        remove_quiet(p);
        return threw;
    };

    // Empty tensor name.
    CHECK(write_throws({entry("", st::Dtype::F32, {2}, v, 8)}));

    // Duplicate tensor name.
    CHECK(write_throws({entry("dup", st::Dtype::F32, {2}, v, 8),
                        entry("dup", st::Dtype::F32, {2}, v, 8)}));

    // Null data with a non-zero byte count.
    CHECK(write_throws({entry("nul", st::Dtype::F32, {2}, nullptr, 8)}));

    // Dtype with no on-disk size.
    CHECK(write_throws({entry("bad", st::Dtype::Unknown, {2}, v, 8)}));

    // Negative dimension.
    CHECK(write_throws({entry("neg", st::Dtype::F32, {-2}, v, 8)}));

    // shape * dtype_size disagrees with bytes.
    CHECK(write_throws({entry("mism", st::Dtype::F32, {3}, v, 8)}));

    // A valid entry that lands after an invalid one is never written: the
    // whole call is rejected before the file is opened.
    CHECK(write_throws({entry("ok", st::Dtype::F32, {2}, v, 8),
                        entry("bad", st::Dtype::Unknown, {2}, v, 8)}));
    CHECK(!fs::exists(p));

    // Unopenable destination (parent directory does not exist).
    const fs::path nodir =
        fs::temp_directory_path() / "brotensor_stx_no_such_dir" / "out.safetensors";
    bool threw = false;
    try { st::write_file(nodir.string(), {entry("t", st::Dtype::F32, {2}, v, 8)}); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

int main() {
    brotensor::init();
    // Pin the whole test to the CPU backend: upload_as()/upload_compute()
    // branch on default_device(), and this test asserts the host conversion
    // path and reads results back with the host accessors.
    brotensor::DeviceScope cpu(brotensor::Device::CPU);

    std::printf("test_safetensors_extra (CPU FP32):\n");
    test_dtype_tables();
    test_numel_guards();
    test_open_errors();
    test_header_extras();
    test_lookup_and_move();
    test_upload();
    test_upload_fp16();
    test_upload_as();
    test_upload_compute();
    test_write_roundtrip();
    test_write_name_escaping();
    test_write_errors();

    if (g_failures == 0) {
        std::printf("  OK  all safetensors extra tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
