// safetensors reader/writer coverage.
//
// Builds a tiny on-disk fixture by hand (exercises the reader against a
// known byte layout), round-trips through write_file()/File::open()
// (exercises the writer), and uploads views into brotensor Tensors on the
// always-available CPU backend.

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
        st::write_file(rt_path.string(), entries);
    }
    {
        auto file = st::File::open(rt_path.string());
        CHECK(file.size() == 1);
        const auto* g = file.find("gamma");
        CHECK(g != nullptr);
        if (g) {
            CHECK(g->dtype == st::Dtype::F32);
            CHECK(g->shape.size() == 2 && g->shape[0] == 3 && g->shape[1] == 2);
            const float* fp = reinterpret_cast<const float*>(g->data);
            CHECK(fp[0] == 10.f && fp[5] == 60.f);
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

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(rt_path, ec);
    std::filesystem::remove(bad_path, ec);

    if (g_failures == 0) std::printf("safetensors: OK\n");
    else std::fprintf(stderr, "safetensors: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
