#include <brotensor/kernel_cache.h>
#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "../src/jit/kernel_disk_cache.h"
#include "../src/jit/trace_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using brotensor::jit::KernelDiskCache;
using brotensor::jit::TraceCache;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);           \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

struct TempDirGuard {
    fs::path dir;
    explicit TempDirGuard(const char* name) {
        dir = fs::temp_directory_path() / name;
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }
    ~TempDirGuard() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

} // namespace

int main() {
    brotensor::init();
    Device dev = brotensor::is_available(Device::VULKAN) ? Device::VULKAN
               : brotensor::is_available(Device::CUDA)   ? Device::CUDA
               : brotensor::is_available(Device::Metal)  ? Device::Metal
                                                         : Device::CPU;

    // 1. Autotune runtime toggle.
    brotensor::set_autotune_enabled(false);
    CHECK(!brotensor::autotune_enabled(), "autotune should be disabled");
    brotensor::set_autotune_enabled(true);
    CHECK(brotensor::autotune_enabled(), "autotune should be enabled");
    brotensor::set_autotune_enabled(false);

    // 2. Warmup + export_kernel_bundle -> fresh cache dir -> import_kernel_bundle -> warm cache hit.
    TempDirGuard cache_src("bt_bundle_src_cache");
    TempDirGuard cache_dst("bt_bundle_dst_cache");
    TempDirGuard bundle_dir("bt_bundle_pkg");
    const fs::path bundle_path = bundle_dir.dir / "kernels.bacb";

    auto& kdc = KernelDiskCache::instance();
    kdc.set_cache_dir(cache_src.dir);
    kdc.reset_stats();
    kdc.clear_loaded_libraries();
    TraceCache::instance().clear();

    CHECK(brotensor::kernel_cache_dir() == cache_src.dir.string(),
          "kernel_cache_dir should reflect override");

    brotensor::WarmupOptions wopts;
    wopts.profile = "minimal";
    wopts.autotune = false;
    wopts.flush = true;
    const brotensor::WarmupStats wstats = brotensor::warmup_kernels(dev, wopts);
    CHECK(wstats.kernels_compiled > 0, "warmup minimal should compile/write kernel artifacts");
    CHECK(wstats.elapsed_ms >= 0.0, "warmup elapsed_ms should be non-negative");

    const brotensor::KernelCacheStats st_src = brotensor::kernel_cache_stats();
    CHECK(st_src.entries > 0, "cacheStats entries should be > 0 after warmup");

    std::string err;
    CHECK(brotensor::export_kernel_bundle(bundle_path.string(), dev, &err),
          ("export_kernel_bundle failed: " + err).c_str());
    CHECK(fs::exists(bundle_path) && fs::file_size(bundle_path) > 0,
          "exported bundle file must exist and be non-empty");

    // Point disk cache at clean empty dir, import bundle, clear in-memory caches, and verify hit.
    kdc.set_cache_dir(cache_dst.dir);
    kdc.reset_stats();
    kdc.clear_loaded_libraries();
    TraceCache::instance().clear();

    brotensor::BundleImportReport rep;
    CHECK(brotensor::import_kernel_bundle(bundle_path.string(), &rep, &err),
          ("import_kernel_bundle failed: " + err).c_str());
    CHECK(rep.entries_imported > 0, "bundle import should import entries");

    {
        Tensor a = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor b = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor c = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor y;
        {
            brotensor::TraceScope ts;
            y = brotensor::silu(a * b + c);
        }
        (void)y;
    }
    const brotensor::KernelCacheStats st_dst = brotensor::kernel_cache_stats();
    CHECK(st_dst.hits > 0, "imported bundle should yield disk-cache hits without recompilation");

    // 3. Corrupt bundle rejection.
    const fs::path bad_bundle = bundle_dir.dir / "corrupt.bacb";
    {
        std::ofstream ofs(bad_bundle, std::ios::binary);
        ofs << "NOT_A_VALID_BACB_BUNDLE";
    }
    brotensor::BundleImportReport bad_rep;
    CHECK(!brotensor::import_kernel_bundle(bad_bundle.string(), &bad_rep, &err),
          "corrupt bundle must be rejected");

    kdc.clear_cache_dir_override();

    if (g_failures == 0) {
        std::cout << "[PASS] Kernel bundle, warmup, and autotune tests passed.\n";
        return 0;
    }
    return 1;
}
