#pragma once

// ─── Persistent On-Disk Superoptimized ML Kernel Cache ───────────────────────
//
// Caches compiled and superoptimized ML kernels on disk across:
//   - CPU SIMD (x86_64 AVX2/FMA and AArch64 NEON via brass::target::AotLinker
//     and brass::target::DynamicLibrary)
//   - Vulkan SPIR-V 1.5 compute shader binaries & push-constant layouts
//   - CUDA PTX assembly modules
//
// Cache directory resolution:
//   1. Explicit override via set_cache_dir() (if non-empty)
//   2. $BRO_ML_CACHE_DIR environment variable
//   3. $XDG_CACHE_HOME/brotensor/kernels
//   4. $HOME/.cache/brotensor/kernels
//   5. <temp_directory_path>/brotensor_kernel_cache
//
// Disable caching at runtime by setting BRO_ML_CACHE_DISABLE=1.
//
// Binary container format (.btkc):
//   [0..4)   Magic: "BTKC" (0x434B5442 little-endian)
//   [4..8)   Version: uint32_t (1)
//   [8..12)  BackendKind: uint32_t (1=CpuNative, 2=VulkanSpirv, 3=CudaPtx, 4=VulkanSpirvMeta)
//   [12..16) Reserved: uint32_t (0)
//   [16..24) Payload byte length: uint64_t
//   [24..32) Payload 64-bit FNV-1a checksum: uint64_t
//   [32..N)  Payload bytes

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(BROTENSOR_HAS_BRASS_JIT) || defined(BROTENSOR_HAS_VULKAN_TRACE_JIT) || defined(BRASS_HOST_BACKEND)
#ifndef BROTENSOR_HAS_BRASS_JIT
#define BROTENSOR_HAS_BRASS_JIT 1
#endif
#endif

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/mir/module.hpp>
#include <brass/target/dynamic_library.hpp>
#include <brass/target/spirv_target.hpp>
#endif

namespace brotensor::jit {

class KernelDiskCache {
public:
    enum class BackendKind : uint32_t {
        CpuNative       = 1,
        VulkanSpirv     = 2,
        CudaPtx         = 3,
        VulkanSpirvMeta = 4,
    };

    struct Stats {
        size_t hits        = 0;
        size_t misses      = 0;
        size_t writes      = 0;
        size_t corruptions = 0;
    };

    struct CpuCompileOptions {
        bool enable_loop_opt         = true;
        bool enable_avx2             = true;
        bool enable_fma              = true;
        bool enable_vectorize        = true;
        bool enable_unroll           = true;
        uint32_t unroll_factor       = 4;
        bool enable_fp_reassociation = true;
        bool enable_smt_guard_elim   = true;
        bool enable_mir_superopt     = true;
    };

    static KernelDiskCache& instance();

    // Returns false when BRO_ML_CACHE_DISABLE=1 is set in the environment.
    bool is_enabled() const noexcept;

    // Resolves the active cache directory on disk.
    std::filesystem::path cache_dir() const;

    // Sets or clears an explicit directory override (useful for isolated tests).
    void set_cache_dir(std::filesystem::path dir);
    void clear_cache_dir_override();

    // Computes the 32-character hex 128-bit FNV-1a cache key for (backend_tag, signature, abi_stamp).
    std::string compute_key(std::string_view backend_tag, std::string_view signature) const;

    // Returns the host CPU backend tag ("cpu_x64", "cpu_arm64", or "cpu_generic").
    static std::string_view host_cpu_backend_tag() noexcept;

    // Returns the path to the .btkc file for a given 128-bit hex key.
    std::filesystem::path entry_path_for_key(std::string_view key_hex) const;

    // Raw binary blob load/store with BTKC header and 64-bit FNV-1a verification.
    bool store_raw(BackendKind kind, std::string_view backend_tag, std::string_view signature,
                   const void* data, size_t size_bytes);
    bool load_raw(BackendKind kind, std::string_view backend_tag, std::string_view signature,
                  std::vector<uint8_t>& out_bytes);

    // Vulkan SPIR-V words load/store.
    bool store_spirv(std::string_view signature, const std::vector<uint32_t>& words);
    bool load_spirv(std::string_view signature, std::vector<uint32_t>& out_words);

    // CUDA PTX string load/store.
    bool store_ptx(std::string_view signature, std::string_view ptx);
    bool load_ptx(std::string_view signature, std::string& out_ptx);

#if BROTENSOR_HAS_BRASS_JIT
    // Full SpirvKernel metadata + SPIR-V words load/store for Vulkan trace compiler.
    bool store_spirv_kernel(std::string_view signature, const brass::target::SpirvKernel& kernel);
    bool load_spirv_kernel(std::string_view signature, brass::target::SpirvKernel& out_kernel);

    // Compiles or loads a superoptimized CPU SIMD kernel via brass::target::AotLinker
    // and brass::target::DynamicLibrary.
    // On cache hit: validates the .btkc file, loads the shared object via DynamicLibrary::open,
    // and returns the resolved symbol pointer without building or compiling MIR.
    // On cache miss (or corruption): invokes build_fn(mod), runs loop optimization,
    // SMT guard elimination, and MIR superoptimization, links an AOT shared library,
    // atomically writes the .btkc file, and loads the symbol via DynamicLibrary::open.
    void* compile_or_load_cpu_kernel(
        std::string_view signature,
        std::string_view entry_name,
        const std::function<void(brass::Module&)>& build_fn,
        const CpuCompileOptions& opts);

    void* compile_or_load_cpu_kernel(
        std::string_view signature,
        std::string_view entry_name,
        const std::function<void(brass::Module&)>& build_fn) {
        return compile_or_load_cpu_kernel(signature, entry_name, build_fn, CpuCompileOptions{});
    }

    template <typename FnPtr, typename BuildFn>
    FnPtr compile_or_load_cpu_fn(
        std::string_view signature,
        std::string_view entry_name,
        BuildFn&& build_fn,
        const CpuCompileOptions& opts) {
        return reinterpret_cast<FnPtr>(
            compile_or_load_cpu_kernel(signature, entry_name,
                                       std::function<void(brass::Module&)>(std::forward<BuildFn>(build_fn)),
                                       opts));
    }

    template <typename FnPtr, typename BuildFn>
    FnPtr compile_or_load_cpu_fn(
        std::string_view signature,
        std::string_view entry_name,
        BuildFn&& build_fn) {
        return compile_or_load_cpu_fn<FnPtr>(signature, entry_name, std::forward<BuildFn>(build_fn),
                                             CpuCompileOptions{});
    }
#endif

    Stats stats() const noexcept;
    void reset_stats() noexcept;

    // Releases in-memory loaded DynamicLibrary handles (only call in single-threaded test resets
    // after clearing cached function pointers).
    void clear_loaded_libraries();

private:
    KernelDiskCache() = default;

    mutable std::mutex mutex_;
    std::filesystem::path override_dir_;
#if BROTENSOR_HAS_BRASS_JIT
    std::vector<std::unique_ptr<brass::target::DynamicLibrary>> loaded_libs_;
    std::vector<std::shared_ptr<void>> fallback_engines_;
#endif

    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> writes_{0};
    mutable std::atomic<size_t> corruptions_{0};
};

using MlKernelDiskCache = KernelDiskCache;

} // namespace brotensor::jit
