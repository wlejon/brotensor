#pragma once

// ─── brotensor's On-Disk Kernel Cache ────────────────────────────────────────
//
// What brotensor compiles — CPU kernels (MIR -> a linked, loadable image),
// Vulkan SPIR-V kernels with their launch metadata, CUDA PTX — kept across
// processes in brass's artifact cache (brass docs/artifact_cache.md). brass
// owns the file format, the keys' hashing, atomic stores, the byte budget and
// "MIR -> cached loadable native module"; this file owns only brotensor's
// policy: where the directory is, its budget, what goes into each key, and the
// SpirvKernel encoding.
//
// Directory (first that applies):
//   1. set_cache_dir() (tests)
//   2. $BRO_ML_CACHE_DIR
//   3. $XDG_CACHE_HOME/brotensor/kernels
//   4. $HOME/.cache/brotensor/kernels
//   5. <temp_directory_path>/brotensor_kernel_cache
// Budget: set_max_bytes(), else $BRO_ML_CACHE_MAX_MB, else 256 MiB; the
// least recently used entries go first. BRO_ML_CACHE_DISABLE=1 turns the
// cache off (CPU kernels are then compiled in memory by brass's KernelJit).
//
// Every key is KeyBuilder(kKeyDomain) + add_brass_build() (brass's version,
// the binary holding brass — so any rebuild of brotensor is a new key space —
// the target and the host CPU's AVX/AVX2/FMA) + the kind ("cpu", "spirv",
// "ptx") + the caller's `signature`, hashed in full. A signature must name
// everything the build reads that the binary does not fix: the trace
// compilers pass TraceDAG::key_material() and their plan parameters, never a
// digest. CPU keys add the entry name, every CpuCompileOptions field and the
// pass budgets the build runs with.
//
// Files of the old format (*.btkc, the .rt/ load copies) are never read; they
// are deleted the first time a directory is opened.

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
#include <brass/cache/artifact_cache.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>
#endif

namespace brotensor::jit {

class KernelDiskCache {
public:
    // The key schema. Bump the suffix when what a key covers changes.
    static constexpr std::string_view kKeyDomain = "brotensor.kernel/2";
    static constexpr uint64_t kDefaultMaxBytes = uint64_t{256} << 20;

    struct Stats {
        size_t hits        = 0;   // loaded from disk
        size_t misses      = 0;   // compiled
        size_t writes      = 0;   // stored after a compile
        size_t corruptions = 0;   // an entry that was there but could not be used
    };

    // Everything here reaches the CPU kernel's code, so all of it is in the key.
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

    // False when BRO_ML_CACHE_DISABLE=1 (or true), or when brotensor was built
    // without brass.
    bool is_enabled() const noexcept;

    std::filesystem::path cache_dir() const;
    void set_cache_dir(std::filesystem::path dir);
    void clear_cache_dir_override();

    uint64_t max_bytes() const;
    // 0 restores the default (the environment, else kDefaultMaxBytes).
    void set_max_bytes(uint64_t bytes);

    // CUDA PTX text.
    bool store_ptx(std::string_view signature, std::string_view ptx);
    bool load_ptx(std::string_view signature, std::string& out_ptx);

#if BROTENSOR_HAS_BRASS_JIT
    // The key a PTX / SPIR-V blob of `kind` is stored under.
    static brass::cache::CacheKey blob_key(std::string_view kind, std::string_view signature);
    // The key a CPU kernel is stored under (NativeModuleCache adds the link
    // options to it): what compile_or_load_cpu_kernel builds from.
    static brass::cache::CacheKey cpu_key(std::string_view signature, std::string_view entry_name,
                                          const CpuCompileOptions& opts);
    // Where those live (tests).
    std::filesystem::path blob_entry_path(std::string_view kind, std::string_view signature) const;
    std::filesystem::path cpu_entry_path(std::string_view signature, std::string_view entry_name,
                                         const CpuCompileOptions& opts) const;

    // A SPIR-V kernel and everything needed to launch it.
    bool store_spirv_kernel(std::string_view signature, const brass::target::SpirvKernel& kernel);
    bool load_spirv_kernel(std::string_view signature, brass::target::SpirvKernel& out_kernel);

    // A CPU kernel: loaded from the cache, or built — build_fn fills the
    // module, then brotensor's passes run (SMT guard elimination, MIR
    // superoptimization, loop optimization, per `opts`) — compiled, linked,
    // stored and loaded by brass's NativeModuleCache. A module that calls
    // functions it does not define cannot be a cached image; it is compiled
    // in memory by KernelJit instead (also when the cache is disabled). The
    // code stays loaded until clear_loaded_libraries(), whatever happens to
    // its cache entry. Null when the kernel cannot be had; an exception from
    // build_fn propagates.
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

    // Walks the directory now, evicting down to the budget (tests,
    // maintenance); returns what the entries take afterwards.
    uint64_t trim();
#endif

    Stats stats() const noexcept;
    void reset_stats() noexcept;

    // Unloads every CPU kernel this cache handed out (only after the callers'
    // function pointers are gone: single-threaded test resets).
    void clear_loaded_libraries();

private:
    KernelDiskCache() = default;

    struct Stores;
    std::shared_ptr<Stores> stores() const;   // null when disabled

    mutable std::mutex mutex_;
    std::filesystem::path override_dir_;
    uint64_t override_max_bytes_ = 0;
    mutable std::shared_ptr<Stores> stores_;
    std::vector<std::shared_ptr<const void>> loaded_;   // NativeModules and KernelJit engines

    mutable Stats stats_;
};

using MlKernelDiskCache = KernelDiskCache;

} // namespace brotensor::jit
