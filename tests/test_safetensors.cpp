// safetensors reader/writer coverage.
//
// Builds a tiny on-disk fixture by hand (exercises the reader against a
// known byte layout), round-trips through write_file()/File::open()
// (exercises the writer), and uploads views into brotensor Tensors on the
// always-available CPU backend.

#include "brotensor/json.h"
#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace st = brotensor::safetensors;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Build a minimal safetensors file with three tensors:
//   "alpha" : F32,  shape [2,3], 24 bytes
//   "beta"  : F16,  shape [4],   8 bytes
//   "delta" : BF16, shape [4],   8 bytes
static std::filesystem::path write_fixture() {
    auto path = std::filesystem::temp_directory_path() / "brotensor_st_test.safetensors";

    const std::string header =
        "{\"__metadata__\":{\"framework\":\"test\"},"
        "\"alpha\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"beta\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[24,32]},"
        "\"delta\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[32,40]}}";

    std::vector<uint8_t> payload(40, 0);
    // alpha: 6 floats 1.0..6.0
    for (int i = 0; i < 6; ++i) {
        float v = static_cast<float>(i + 1);
        std::memcpy(payload.data() + i * 4, &v, 4);
    }
    // beta: 4 fp16 bit patterns (1.0, 2.0, 3.0, 4.0)
    uint16_t halves[4] = {0x3c00, 0x4000, 0x4200, 0x4400};
    std::memcpy(payload.data() + 24, halves, 8);
    // delta: 4 bf16 bit patterns (1.0, 2.0, 3.0, 4.0) — the high 16 bits of
    // each FP32 value.
    uint16_t bhalves[4] = {0x3f80, 0x4000, 0x4040, 0x4080};
    std::memcpy(payload.data() + 32, bhalves, 8);

    uint64_t hdr_size = header.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create fixture file");
    f.write(reinterpret_cast<const char*>(&hdr_size), 8);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    return path;
}

int main() {
    auto path = write_fixture();

    // Scope the File so its mmap is released before we try to delete the
    // backing file. Windows refuses to remove a file with an open mapping.
    {
        auto file = st::File::open(path.string());
        CHECK(file.size() == 3);

        const auto* alpha = file.find("alpha");
        CHECK(alpha != nullptr);
        if (alpha) {
            CHECK(alpha->dtype == st::Dtype::F32);
            CHECK(alpha->shape.size() == 2);
            CHECK(alpha->shape[0] == 2 && alpha->shape[1] == 3);
            CHECK(alpha->nbytes == 24);
            CHECK(alpha->numel() == 6);
            const float* fp = reinterpret_cast<const float*>(alpha->data);
            for (int i = 0; i < 6; ++i) {
                CHECK(fp[i] == static_cast<float>(i + 1));
            }
            // upload into a CPU Tensor (compute dtype is FP32 on CPU).
            brotensor::Tensor t;
            st::upload(*alpha, 2, 3, t);
            CHECK(t.rows == 2 && t.cols == 3);
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.0f && t.at(1, 2) == 6.0f);
        }

        const auto* beta = file.find("beta");
        CHECK(beta != nullptr);
        if (beta) {
            CHECK(beta->dtype == st::Dtype::F16);
            CHECK(beta->shape.size() == 1 && beta->shape[0] == 4);
            CHECK(beta->nbytes == 8);
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(beta->data);
            CHECK(hp[0] == 0x3c00);
            CHECK(hp[3] == 0x4400);
            // upload_compute converts the F16 view to FP32 on the CPU backend.
            brotensor::Tensor t;
            st::upload_compute(*beta, 4, 1, t);
            CHECK(t.rows == 4 && t.cols == 1);
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.0f && t.at(3, 0) == 4.0f);
        }

        const auto* delta = file.find("delta");
        CHECK(delta != nullptr);
        if (delta) {
            CHECK(delta->dtype == st::Dtype::BF16);
            CHECK(delta->shape.size() == 1 && delta->shape[0] == 4);
            CHECK(delta->nbytes == 8);
            // upload_compute widens the BF16 view to FP32 on the CPU backend
            // (the path Flux-family checkpoints take). The values 1..4 are
            // exactly representable, so the widening is lossless here.
            brotensor::Tensor t;
            st::upload_compute(*delta, 4, 1, t);
            CHECK(t.rows == 4 && t.cols == 1);
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.0f && t.at(3, 0) == 4.0f);
            // upload_compute_checked accepts BF16 too.
            brotensor::Tensor t2;
            st::upload_compute_checked(*delta, 4, 1, t2, "delta");
            CHECK(t2.dtype == brotensor::Dtype::FP32);
            CHECK(t2.at(1, 0) == 2.0f && t2.at(2, 0) == 3.0f);
        }

        CHECK(file.find("nope") == nullptr);
        bool threw = false;
        try { (void)file.get("nope"); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // Writer round-trip: write_file() then read it back.
    auto rt_path = std::filesystem::temp_directory_path() / "brotensor_st_roundtrip.safetensors";
    // BF16 payload: round 1..4 to BF16 bit patterns (each exactly
    // representable, so the round-trip is lossless).
    uint16_t delta_bf16[4];
    for (int i = 0; i < 4; ++i) {
        delta_bf16[i] = brotensor::fp32_to_bf16_bits(static_cast<float>(i + 1));
    }
    {
        float gamma[6] = {10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
        std::vector<st::WriteEntry> entries;
        st::WriteEntry e;
        e.name = "gamma";
        e.dtype = st::Dtype::F32;
        e.shape = {3, 2};
        e.host_data = gamma;
        e.bytes = sizeof(gamma);
        entries.push_back(e);

        st::WriteEntry d;
        d.name = "delta";
        d.dtype = st::Dtype::BF16;
        d.shape = {4};
        d.host_data = delta_bf16;
        d.bytes = sizeof(delta_bf16);
        entries.push_back(d);

        st::write_file(rt_path.string(), entries);
    }
    {
        auto file = st::File::open(rt_path.string());
        CHECK(file.size() == 2);
        const auto* g = file.find("gamma");
        CHECK(g != nullptr);
        if (g) {
            CHECK(g->dtype == st::Dtype::F32);
            CHECK(g->shape.size() == 2 && g->shape[0] == 3 && g->shape[1] == 2);
            const float* fp = reinterpret_cast<const float*>(g->data);
            CHECK(fp[0] == 10.f && fp[5] == 60.f);
        }
        const auto* d = file.find("delta");
        CHECK(d != nullptr);
        if (d) {
            CHECK(d->dtype == st::Dtype::BF16);
            CHECK(d->shape.size() == 1 && d->shape[0] == 4);
            CHECK(d->nbytes == 8);
            // Raw BF16 bit patterns survive the write/read unchanged.
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(d->data);
            for (int i = 0; i < 4; ++i) CHECK(hp[i] == delta_bf16[i]);
            // upload_compute widens the BF16 view back to FP32.
            brotensor::Tensor t;
            st::upload_compute(*d, 4, 1, t);
            CHECK(t.dtype == brotensor::Dtype::FP32);
            CHECK(t.at(0, 0) == 1.0f && t.at(3, 0) == 4.0f);
        }
    }

    // Error path: garbage file should throw.
    auto bad_path = std::filesystem::temp_directory_path() / "brotensor_st_bad.safetensors";
    {
        std::ofstream bf(bad_path, std::ios::binary | std::ios::trunc);
        bf << "not a safetensors file";
    }
    bool bad_threw = false;
    try { (void)st::File::open(bad_path.string()); }
    catch (const std::runtime_error&) { bad_threw = true; }
    CHECK(bad_threw);

    // Symbolic / adversarial verification:
    // 1. numel() * dtype_size_bytes() 64-bit wraparound attack:
    //    shape = [4611686018427387905] (0x4000000000000001 < INT64_MAX), dtype = F32 (4 bytes).
    //    Unchecked `numel * 4` overflows 64-bit integer to `4` bytes and would match data_offsets [0,4]!
    {
        const std::string wrap_hdr =
            "{\"adv\":{\"dtype\":\"F32\",\"shape\":[4611686018427387905],\"data_offsets\":[0,4]}}";
        const uint64_t hlen = wrap_hdr.size();
        const uint32_t dummy_payload = 0;
        std::ofstream wf(bad_path, std::ios::binary | std::ios::trunc);
        wf.write(reinterpret_cast<const char*>(&hlen), 8);
        wf.write(wrap_hdr.data(), static_cast<std::streamsize>(wrap_hdr.size()));
        wf.write(reinterpret_cast<const char*>(&dummy_payload), 4);
        wf.close();
        bool wrap_threw = false;
        try { (void)st::File::open(bad_path.string()); }
        catch (const std::runtime_error&) { wrap_threw = true; }
        CHECK(wrap_threw);
    }

    // 2. Header size wraparound near UINT64_MAX (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF8):
    for (uint64_t adv_hlen : {uint64_t(UINT64_MAX), uint64_t(UINT64_MAX) - 7,
                              uint64_t(UINT64_MAX) - 8, uint64_t(1ull << 62)}) {
        std::ofstream wf(bad_path, std::ios::binary | std::ios::trunc);
        wf.write(reinterpret_cast<const char*>(&adv_hlen), 8);
        wf << "{}";
        wf.close();
        bool hthrew = false;
        try { (void)st::File::open(bad_path.string()); }
        catch (const std::runtime_error&) { hthrew = true; }
        CHECK(hthrew);
    }

    // 3. Truncated safetensors file at every byte offset [0 .. full_file.size() - 1]:
    {
        std::ifstream full_in(path, std::ios::binary);
        std::vector<char> full_bytes((std::istreambuf_iterator<char>(full_in)),
                                     std::istreambuf_iterator<char>());
        full_in.close();
        CHECK(full_bytes.size() > 16);
        for (std::size_t cut = 0; cut < full_bytes.size(); ++cut) {
            std::ofstream tf(bad_path, std::ios::binary | std::ios::trunc);
            if (cut > 0) tf.write(full_bytes.data(), static_cast<std::streamsize>(cut));
            tf.close();
            bool cut_threw = false;
            try { (void)st::File::open(bad_path.string()); }
            catch (const std::runtime_error&) { cut_threw = true; }
            CHECK(cut_threw);
        }
    }

    // 4. brotensor::json::parse recursion depth limit against stack-overflow DoS:
    {
        std::string deep(600, '[');
        deep += "1";
        deep += std::string(600, ']');
        bool deep_threw = false;
        try { (void)brotensor::json::parse(deep); }
        catch (const std::runtime_error&) { deep_threw = true; }
        CHECK(deep_threw);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(rt_path, ec);
    std::filesystem::remove(bad_path, ec);

    if (g_failures == 0) std::printf("safetensors: OK\n");
    else std::fprintf(stderr, "safetensors: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
