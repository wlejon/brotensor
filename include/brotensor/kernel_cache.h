#pragma once

// Public kernel cache management, shippable bundle export/import, warmup,
// and per-device runtime autotuning controls.

#include "runtime.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace brotensor {

struct KernelCacheStats {
    std::string dir;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t writes = 0;
    std::size_t corruptions = 0;
    std::size_t entries = 0;
};

struct WarmupOptions {
    /// "minimal", "default", "llm", or "diffusion".
    std::string profile = "default";
    bool autotune = false;
    bool flush = true;
};

struct WarmupStats {
    std::size_t kernels_compiled = 0;
    std::size_t autotune_entries = 0;
    double elapsed_ms = 0.0;
};

struct BundleImportReport {
    std::size_t entries_imported = 0;
    std::size_t entries_skipped = 0;
    std::size_t bytes = 0;
};

/// Returns statistics and directory path for the active kernel disk cache.
KernelCacheStats kernel_cache_stats();

/// Returns the resolved directory used for persistent kernel caching.
std::string kernel_cache_dir();

/// Flushes pending cache writes and backend driver pipeline archives (e.g.,
/// Vulkan VkPipelineCache, Metal MTLBinaryArchive) to disk.
void flush_kernel_caches(Device dev = default_device());

/// Controls whether live per-machine micro-autotuning runs on cache misses
/// for eligible Vulkan kernels. Cached autotune decisions are always used
/// when present regardless of this setting. Can also be enabled via
/// `BROTENSOR_AUTOTUNE=1`.
void set_autotune_enabled(bool enabled) noexcept;
bool autotune_enabled() noexcept;

/// Pre-compiles representative kernel variants on `dev` according to `opts`.
WarmupStats warmup_kernels(Device dev = default_device(), const WarmupOptions& opts = {});

/// Exports portable disk-cache entries (plus device pipeline archives when
/// available) into a single checksum-verified `.bacb` bundle file.
bool export_kernel_bundle(const std::string& bundle_path,
                          Device dev = default_device(),
                          std::string* err = nullptr);

/// Validates `bundle_path` against this build's Brass codegen fingerprint and
/// imports its cached kernel artifacts into the local disk cache directory.
bool import_kernel_bundle(const std::string& bundle_path,
                          BundleImportReport* report = nullptr,
                          std::string* err = nullptr);

} // namespace brotensor
