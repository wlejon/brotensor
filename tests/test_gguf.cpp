// GGUF loader smoke test. Builds a tiny in-memory fixture, parses it, and
// verifies header/metadata/tensor info + a Tensor upload roundtrip.

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace gg = brotensor::gguf;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_f32(std::vector<uint8_t>& b, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    put_u32(b, v);
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    for (char c : s) b.push_back(static_cast<uint8_t>(c));
}

std::filesystem::path write_fixture() {
    auto path = std::filesystem::temp_directory_path() / "brotensor_gguf_test.gguf";
    std::vector<uint8_t> buf;

    // Header.
    put_u32(buf, 0x46554747u);  // magic "GGUF"
    put_u32(buf, 3);            // version
    put_u64(buf, 2);            // tensor_count
    put_u64(buf, 3);            // metadata_kv_count

    // Metadata: general.architecture (string) = "llama"
    put_str(buf, "general.architecture");
    put_u32(buf, 8);  // String
    put_str(buf, "llama");

    // Metadata: llama.context_length (u32) = 2048
    put_str(buf, "llama.context_length");
    put_u32(buf, 4);  // U32
    put_u32(buf, 2048);

    // Metadata: test.scale (f32) = 0.125
    put_str(buf, "test.scale");
    put_u32(buf, 6);  // F32
    put_f32(buf, 0.125f);

    // Tensor info: "alpha", n_dims=2, dims=[3,2], type=F32 (0), offset=0
    put_str(buf, "alpha");
    put_u32(buf, 2);
    put_u64(buf, 3);
    put_u64(buf, 2);
    put_u32(buf, 0);   // F32
    put_u64(buf, 0);   // offset in data blob

    // Tensor info: "beta", n_dims=1, dims=[32], type=Q4_0 (2), offset=32
    put_str(buf, "beta");
    put_u32(buf, 1);
    put_u64(buf, 32);
    put_u32(buf, 2);   // Q4_0
    put_u64(buf, 32);  // offset (after alpha's 24 bytes + 8 pad)

    // Pad to alignment=32.
    while (buf.size() % 32 != 0) buf.push_back(0);

    // Data blob.
    // alpha: 6 floats 1..6
    const std::size_t data_start = buf.size();
    for (int i = 0; i < 6; ++i) put_f32(buf, static_cast<float>(i + 1));
    // 8 bytes of pad to reach offset 32 in the data blob.
    while (buf.size() - data_start < 32) buf.push_back(0);
    // beta: Q4_0 block = FP16 scale (2 bytes) + 16 packed nibbles (16 bytes) = 18 bytes
    put_u8(buf, 0xCD);  // first byte — asserted in the test
    put_u8(buf, 0x3C);  // FP16 scale ~1.0
    for (int i = 0; i < 16; ++i) put_u8(buf, static_cast<uint8_t>(0x80 | i));

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create gguf fixture file");
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return path;
}

}  // namespace

int main() {
    using brotensor::Dtype;

    // Dtype helper sanity (independent of file parsing).
    CHECK(brotensor::dtype_block_size(Dtype::Q4_0) == 32);
    CHECK(brotensor::dtype_block_bytes(Dtype::Q4_0) == 18);
    CHECK(brotensor::dtype_block_size(Dtype::Q4_K) == 256);
    CHECK(brotensor::dtype_block_bytes(Dtype::Q4_K) == 144);
    CHECK(brotensor::dtype_storage_bytes(Dtype::Q4_K, 256) == 144);
    CHECK(brotensor::dtype_storage_bytes(Dtype::FP32, 6) == 24);
    CHECK(brotensor::dtype_size_bytes(Dtype::F64) == 8);
    CHECK(brotensor::dtype_is_quant(Dtype::Q4_K));
    CHECK(!brotensor::dtype_is_quant(Dtype::FP32));
    {
        bool threw = false;
        try { (void)brotensor::dtype_storage_bytes(Dtype::Q4_K, 100); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    auto path = write_fixture();
    {
        auto f = gg::File::open(path.string());
        CHECK(f.version() == 3);
        CHECK(f.alignment() == 32);
        CHECK(f.tensor_count() == 2);

        const auto* arch = f.find_meta("general.architecture");
        CHECK(arch != nullptr);
        if (arch) {
            CHECK(arch->type == gg::ValueType::String);
            CHECK(arch->str == "llama");
        }
        const auto& ctx = f.get_meta("llama.context_length");
        CHECK(ctx.type == gg::ValueType::U32);
        CHECK(ctx.scalar.u32 == 2048u);
        const auto& scale = f.get_meta("test.scale");
        CHECK(scale.type == gg::ValueType::F32);
        CHECK(std::abs(scale.scalar.f32 - 0.125f) < 1e-9f);

        const auto* alpha = f.find_tensor("alpha");
        CHECK(alpha != nullptr);
        if (alpha) {
            CHECK(alpha->dtype == Dtype::FP32);
            CHECK(alpha->numel == 6);
            CHECK(alpha->nbytes == 24);
            CHECK(alpha->dtype_supported);
            // GGUF shape is innermost-first: [3, 2] → cols=3, rows=2.
            auto rc = gg::shape_to_2d(alpha->shape);
            CHECK(rc.first == 2 && rc.second == 3);

            brotensor::Tensor t;
            gg::upload_raw(*alpha, rc.first, rc.second, t);
            CHECK(t.rows == 2 && t.cols == 3);
            CHECK(t.dtype == Dtype::FP32);
            auto v = t.to_host_vector();
            CHECK(v.size() == 6);
            for (int i = 0; i < 6; ++i) {
                CHECK(v[static_cast<std::size_t>(i)] == static_cast<float>(i + 1));
            }
        }

        const auto* beta = f.find_tensor("beta");
        CHECK(beta != nullptr);
        if (beta) {
            CHECK(beta->dtype == Dtype::Q4_0);
            CHECK(beta->numel == 32);
            CHECK(beta->nbytes == 18);
            CHECK(beta->dtype_supported);
            CHECK(beta->ggml_type == 2u);

            brotensor::Tensor t;
            gg::upload_raw(*beta, 1, 32, t);
            CHECK(t.rows == 1 && t.cols == 32);
            CHECK(t.dtype == Dtype::Q4_0);
            CHECK(t.bytes() == 18);
            // On a non-CPU default device the bytes still match, but the
            // first-byte check only makes sense for CPU storage.
            if (t.device == brotensor::Device::CPU) {
                const uint8_t* p = static_cast<const uint8_t*>(t.host_raw());
                CHECK(p[0] == 0xCD);
            }
        }

        // Missing-tensor lookup.
        CHECK(f.find_tensor("nope") == nullptr);
        bool threw = false;
        try { (void)f.get_tensor("nope"); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // Garbage file: should throw.
    auto bad = std::filesystem::temp_directory_path() / "brotensor_gguf_bad.gguf";
    {
        std::ofstream bf(bad, std::ios::binary | std::ios::trunc);
        bf << "not a gguf file at all, really";
    }
    bool bad_threw = false;
    try { (void)gg::File::open(bad.string()); }
    catch (const std::runtime_error&) { bad_threw = true; }
    CHECK(bad_threw);

    auto expect_gguf_throws = [&](const std::vector<uint8_t>& raw) {
        std::ofstream bf(bad, std::ios::binary | std::ios::trunc);
        if (!raw.empty()) {
            bf.write(reinterpret_cast<const char*>(raw.data()),
                     static_cast<std::streamsize>(raw.size()));
        }
        bf.close();
        bool threw = false;
        try { (void)gg::File::open(bad.string()); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    };

    // Symbolic / adversarial GGUF parser verification:
    // 1. Huge 64-bit tensor_count / meta_count (UINT64_MAX) must fail cleanly without OOM reserve().
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, UINT64_MAX);
        put_u64(b, 0);
        expect_gguf_throws(b);
    }
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 0);
        put_u64(b, UINT64_MAX);
        expect_gguf_throws(b);
    }

    // 2. Huge 64-bit string length and array count (UINT64_MAX, SIZE_MAX - 8) must fail without OOB/OOM.
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 0);
        put_u64(b, 1);            // 1 metadata kv
        put_u64(b, UINT64_MAX);   // key string length = UINT64_MAX
        expect_gguf_throws(b);
    }
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 0);
        put_u64(b, 1);
        put_str(b, "arr");
        put_u32(b, 9);            // ValueType::Array
        put_u32(b, 0);            // elem_type = U8
        put_u64(b, UINT64_MAX);   // array length = UINT64_MAX
        expect_gguf_throws(b);
    }

    // 3. Deeply nested ValueType::Array (> 32 levels) must fail deterministically without stack overflow.
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 0);
        put_u64(b, 1);
        put_str(b, "deep");
        for (int d = 0; d < 40; ++d) {
            put_u32(b, 9);  // Array
            if (d < 39) {
                put_u64(b, 1);
            }
        }
        put_u32(b, 0);  // innermost elem_type = U8
        put_u64(b, 1);
        put_u8(b, 42);
        expect_gguf_throws(b);
    }

    // 4. Tensor nbytes 64-bit multiplication wraparound (`numel * type_size` wraps modulo 2^64 to 4):
    //    dim = 0x4000000000000001 (< INT64_MAX), ggml_type = 0 (FP32, 4 bytes).
    {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 1);  // 1 tensor
        put_u64(b, 0);  // 0 meta
        put_str(b, "wrap");
        put_u32(b, 1);  // n_dims = 1
        put_u64(b, 0x4000000000000001ull);
        put_u32(b, 0);  // F32 (4 bytes)
        put_u64(b, 0);  // offset = 0
        while (b.size() % 32 != 0) put_u8(b, 0);
        put_u32(b, 0);  // 4 bytes of data
        expect_gguf_throws(b);
    }

    // 5. Tensor offset wraparound (offset = UINT64_MAX, UINT64_MAX - 16) and non-power-of-two alignment:
    for (uint64_t bad_off : {uint64_t(UINT64_MAX), uint64_t(UINT64_MAX) - 16, uint64_t(1ull << 62)}) {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 1);
        put_u64(b, 0);
        put_str(b, "t");
        put_u32(b, 1);
        put_u64(b, 1);
        put_u32(b, 0);
        put_u64(b, bad_off);
        while (b.size() % 32 != 0) put_u8(b, 0);
        put_u32(b, 0);
        expect_gguf_throws(b);
    }
    for (uint32_t bad_align : {0u, 3u, 6u, 24u, 0xFFFFFFFFu}) {
        std::vector<uint8_t> b;
        put_u32(b, 0x46554747u);
        put_u32(b, 3);
        put_u64(b, 0);
        put_u64(b, 1);
        put_str(b, "general.alignment");
        put_u32(b, 4);  // U32
        put_u32(b, bad_align);
        expect_gguf_throws(b);
    }

    // 6. Truncated GGUF fixture at every byte offset [0 .. full.size() - 1] must never read OOB or succeed:
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<uint8_t> full((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
        in.close();
        CHECK(full.size() > 64);
        for (std::size_t cut = 0; cut < full.size(); ++cut) {
            std::vector<uint8_t> prefix(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(cut));
            expect_gguf_throws(prefix);
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(bad, ec);

    if (g_failures == 0) std::printf("gguf: OK\n");
    else std::fprintf(stderr, "gguf: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
