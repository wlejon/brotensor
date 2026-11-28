#pragma once

#include "cuda_driver.h"
#include "../jit/kernel_disk_cache.h"

#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/target/ptx_target.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda::jit {

// ─── CudaJitEngine ──────────────────────────────────────────────────────────
//
// Driver-API PTX compiler with a lock-free CUfunction cache.
//
// Driver initialisation happens in the constructor, so the one-time setup is
// ordered by the language's guarantee on a function-local static — there is no
// initialised flag to guard and nothing to spin on.
//
// The cache itself is a fixed open-addressed table of write-once entries. An
// inserting thread claims a slot by compare-exchanging its key hash from 0,
// then publishes the CUfunction with release ordering; readers acquire it. Two
// threads that race on the same key both call cuModuleLoadDataEx, which is
// legal and costs one redundant compile — the loser's module is kept on the
// unload list and its function is dropped. Nothing blocks and nothing spins.
//
// Keys are compared by their full string as well as the hash, so a collision
// produces another probe, never a wrong kernel.
class CudaJitEngine {
public:
    static constexpr std::size_t kCapacity = 4096;  // power of two
    static constexpr int kMaxDevices = 16;

    static CudaJitEngine& for_device(int dev) {
        static std::array<std::once_flag, kMaxDevices> flags;
        static std::array<CudaJitEngine*, kMaxDevices> engines{};
        const int idx = (dev >= 0 && dev < kMaxDevices) ? dev : 0;
        std::call_once(flags[idx], [idx] { engines[idx] = new CudaJitEngine(idx); });
        return *engines[idx];
    }

    static CudaJitEngine& instance() {
        return for_device(current_device());
    }

    bool is_available() const { return available_; }
    int cc_major() const noexcept { return cc_major_; }
    int cc_minor() const noexcept { return cc_minor_; }
    int driver_version() const noexcept { return drv_ver_; }

    CUfunction get_function(const std::string& key, const std::string& ptx,
                            const std::string& entry_name) {
        if (!available_) {
            throw std::runtime_error("CUDA Driver JIT is not available on this system");
        }
        if (CUfunction hit = lookup_table(key)) return hit;

        auto& disk = ::brotensor::jit::KernelDiskCache::instance();
        std::string cubin_sig;
        CUmodule module = load_cached_raw_cubin(disk, entry_name, ptx, cubin_sig);
        if (!module) {
            module = compile_ptx_module(ptx, entry_name, "cuda_cubin", cubin_sig, /*direct_entry=*/{});
        }
        return extract_and_publish(key, module, entry_name);
    }

    // Loads or compiles PTX once into a single CUmodule and publishes two entry
    // points (e.g. vectorized + scalar entries) under key0 and key1.
    std::pair<CUfunction, CUfunction> get_function_pair(
        const std::string& key0, const std::string& entry0,
        const std::string& key1, const std::string& entry1,
        const std::string& ptx) {
        if (!available_) {
            throw std::runtime_error("CUDA Driver JIT is not available on this system");
        }
        CUfunction fn0 = lookup_table(key0);
        CUfunction fn1 = lookup_table(key1);
        if (fn0 && fn1) return {fn0, fn1};

        auto& disk = ::brotensor::jit::KernelDiskCache::instance();
        const std::string pair_label = entry0 + "+" + entry1;
        std::string cubin_sig;
        CUmodule module = load_cached_raw_cubin(disk, pair_label, ptx, cubin_sig);
        if (!module) {
            module = compile_ptx_module(ptx, pair_label, "cuda_cubin", cubin_sig, /*direct_entry=*/{});
        }
        if (!fn0) fn0 = extract_and_publish(key0, module, entry0);
        if (!fn1) fn1 = extract_and_publish(key1, module, entry1);
        return {fn0, fn1};
    }

    // Fast path for warm MIR kernels: loads a pre-compiled CUBIN blob keyed directly
    // by disk_sig (avoiding PTX disk read, entry parsing, and FNV-1a hashing).
    CUfunction try_load_direct_cubin(const std::string& mem_key, const std::string& disk_sig) {
        if (!available_) return nullptr;
        if (CUfunction hit = lookup_table(mem_key)) return hit;
        auto& disk = ::brotensor::jit::KernelDiskCache::instance();
        if (arch_sig_.empty() || !disk.is_enabled()) return nullptr;

        std::vector<std::uint8_t> blob;
        const std::string direct_sig = arch_sig_ + ":" + disk_sig;
        if (!disk.load_blob("cuda_mir_cubin", direct_sig, blob) || blob.size() <= 4u) {
            return nullptr;
        }
        std::uint32_t name_len = 0;
        std::memcpy(&name_len, blob.data(), sizeof(name_len));
        if (name_len == 0 || 4u + static_cast<std::size_t>(name_len) >= blob.size()) {
            return nullptr;
        }
        const std::string entry_name(reinterpret_cast<const char*>(blob.data() + 4), name_len);
        const std::uint8_t* cubin_ptr = blob.data() + 4 + name_len;

        CtxGuard ctx_guard(context_);
        CUmodule module = nullptr;
        if (drv::cuModuleLoadData(&module, cubin_ptr) != CUDA_SUCCESS || !module) {
            return nullptr;
        }
        CUfunction fn = nullptr;
        if (drv::cuModuleGetFunction(&fn, module, entry_name.c_str()) != CUDA_SUCCESS || !fn) {
            drv::cuModuleUnload(module);
            return nullptr;
        }
        retain_module(module);
        return publish_key(mem_key, fn);
    }

    // Compiles PTX to CUBIN once, stores packed [entry_name][cubin] under
    // `disk_sig` for future direct loads, and publishes `mem_key`.
    CUfunction get_function_cached(const std::string& mem_key, const std::string& disk_sig,
                                   const std::string& ptx, const std::string& entry_name) {
        if (!available_) {
            throw std::runtime_error("CUDA Driver JIT is not available on this system");
        }
        if (CUfunction hit = lookup_table(mem_key)) return hit;
        const std::string direct_sig = arch_sig_.empty() ? std::string() : (arch_sig_ + ":" + disk_sig);
        CUmodule module = compile_ptx_module(ptx, entry_name, "cuda_mir_cubin", direct_sig, entry_name);
        return extract_and_publish(mem_key, module, entry_name);
    }

private:
    struct Entry {
        std::atomic<std::uint64_t> key{0};
        std::atomic<CUfunction> fn{nullptr};
        std::string name;  // written before the key is published
    };

    struct CtxGuard {
        CUcontext prev = nullptr;
        bool restore = false;
        explicit CtxGuard(CUcontext target) {
            drv::cuCtxGetCurrent(&prev);
            if (prev != target) {
                drv::cuCtxSetCurrent(target);
                restore = (prev != nullptr);
            }
        }
        ~CtxGuard() {
            if (restore) drv::cuCtxSetCurrent(prev);
        }
    };

    explicit CudaJitEngine(int dev) {
        // Without a driver on the machine every drv:: call answers
        // CUDA_ERROR_NOT_INITIALIZED, so this is also the "no GPU" exit.
        if (drv::cuInit(0) != CUDA_SUCCESS) return;

        int dev_count = 0;
        if (drv::cuDeviceGetCount(&dev_count) != CUDA_SUCCESS || dev < 0 || dev >= dev_count) return;

        context_ = primary_context(dev);
        available_ = (context_ != nullptr);
        if (available_) {
            CUdevice cu_dev = 0;
            if (drv::cuDeviceGet(&cu_dev, dev) == CUDA_SUCCESS &&
                drv::cuDeviceGetAttribute(&cc_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev) == CUDA_SUCCESS &&
                drv::cuDeviceGetAttribute(&cc_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev) == CUDA_SUCCESS) {
                (void)drv::cuDriverGetVersion(&drv_ver_);
                char abuf[64];
                std::snprintf(abuf, sizeof(abuf), "sm_%d%d:drv%d", cc_major_, cc_minor_, drv_ver_);
                arch_sig_ = abuf;
            }
        }
    }

    ~CudaJitEngine() {
        // Process teardown; no other thread is running by this point.
        for (CUmodule mod : modules_) {
            if (mod) drv::cuModuleUnload(mod);
        }
    }

    static std::uint64_t key_hash(const std::string& s) {
        std::uint64_t h = 14695981039346656037ULL;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        return h ? h : 1ULL;
    }
    static std::size_t slot_of(std::uint64_t h) {
        return static_cast<std::size_t>(h) & (kCapacity - 1);
    }

    CUfunction lookup_table(const std::string& key) const {
        const std::uint64_t h = key_hash(key);
        std::size_t i = slot_of(h);
        for (std::size_t probe = 0; probe < kCapacity; ++probe) {
            const Entry& e = table_[i];
            const std::uint64_t cur = e.key.load(std::memory_order_acquire);
            if (cur == 0) return nullptr;
            if (cur == h) {
                const CUfunction fn = e.fn.load(std::memory_order_acquire);
                if (!fn) return nullptr;  // in-flight insert by another thread
                if (e.name == key) return fn;
            }
            i = (i + 1) & (kCapacity - 1);
        }
        return nullptr;
    }

    CUfunction publish_key(const std::string& key, CUfunction fn) {
        const std::uint64_t h = key_hash(key);
        std::size_t i = slot_of(h);
        for (std::size_t probe = 0; probe < kCapacity;) {
            Entry& e = table_[i];
            const std::uint64_t cur = e.key.load(std::memory_order_acquire);
            if (cur == h) {
                const CUfunction existing = e.fn.load(std::memory_order_acquire);
                if (existing) {
                    if (e.name == key) return existing;
                    ++probe;
                    i = (i + 1) & (kCapacity - 1);
                    continue;
                }
                return fn;  // another thread owns this key
            }
            if (cur == 0) {
                std::uint64_t expect = 0;
                if (e.key.compare_exchange_strong(expect, h, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
                    e.name = key;
                    CUfunction fn_expect = nullptr;
                    if (e.fn.compare_exchange_strong(fn_expect, fn, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
                        return fn;
                    }
                    return fn_expect;
                }
                continue;
            }
            ++probe;
            i = (i + 1) & (kCapacity - 1);
        }
        return fn;
    }

    CUfunction extract_and_publish(const std::string& mem_key, CUmodule module,
                                     const std::string& entry_name) {
        CUfunction fn = nullptr;
        const CUresult res = drv::cuModuleGetFunction(&fn, module, entry_name.c_str());
        if (res != CUDA_SUCCESS || !fn) {
            throw std::runtime_error("Failed to find kernel entry point '" + entry_name +
                                     "' in compiled PTX module");
        }
        return publish_key(mem_key, fn);
    }

    CUmodule load_cached_raw_cubin(::brotensor::jit::KernelDiskCache& disk,
                                   const std::string& label, const std::string& ptx,
                                   std::string& cubin_sig) {
        if (arch_sig_.empty() || !disk.is_enabled()) return nullptr;
        char hbuf[32];
        std::snprintf(hbuf, sizeof(hbuf), "%016llx:%zu",
                      static_cast<unsigned long long>(key_hash(ptx)), ptx.size());
        cubin_sig = arch_sig_ + ":" + label + ":" + hbuf;
        std::vector<std::uint8_t> cached_cubin;
        if (!disk.load_blob("cuda_cubin", cubin_sig, cached_cubin) || cached_cubin.empty()) {
            return nullptr;
        }
        CtxGuard ctx_guard(context_);
        CUmodule module = nullptr;
        if (drv::cuModuleLoadData(&module, cached_cubin.data()) != CUDA_SUCCESS || !module) {
            return nullptr;
        }
        retain_module(module);
        return module;
    }

    CUmodule compile_ptx_module(const std::string& ptx, const std::string& label,
                                const char* disk_kind, const std::string& disk_sig,
                                const std::string& direct_entry) {
        CtxGuard ctx_guard(context_);

        constexpr std::size_t LOG_SIZE = 8192;
        std::vector<char> error_log(LOG_SIZE, 0);
        std::vector<char> info_log(LOG_SIZE, 0);

        CUjit_option options[] = {
            CU_JIT_OPTIMIZATION_LEVEL,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
        };
        void* option_values[] = {
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(4)),
            error_log.data(),
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(LOG_SIZE)),
            info_log.data(),
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(LOG_SIZE))
        };

        CUmodule module = nullptr;
        auto& disk = ::brotensor::jit::KernelDiskCache::instance();
        if (!disk_sig.empty() && disk.is_enabled()) {
            CUlinkState link = nullptr;
            if (drv::cuLinkCreate(5, options, option_values, &link) == CUDA_SUCCESS && link) {
                void* cubin_ptr = nullptr;
                std::size_t cubin_size = 0;
                const CUresult add_res = drv::cuLinkAddData(
                    link, CU_JIT_INPUT_PTX, const_cast<char*>(ptx.c_str()),
                    ptx.size() + 1, label.c_str(), 0, nullptr, nullptr);
                if (add_res == CUDA_SUCCESS &&
                    drv::cuLinkComplete(link, &cubin_ptr, &cubin_size) == CUDA_SUCCESS &&
                    cubin_ptr && cubin_size > 0) {
                    if (drv::cuModuleLoadData(&module, cubin_ptr) == CUDA_SUCCESS && module) {
                        if (direct_entry.empty()) {
                            (void)disk.store_blob(disk_kind, disk_sig, cubin_ptr, cubin_size);
                        } else {
                            const std::uint32_t nlen = static_cast<std::uint32_t>(direct_entry.size());
                            std::vector<std::uint8_t> packed(4u + nlen + cubin_size);
                            std::memcpy(packed.data(), &nlen, 4u);
                            std::memcpy(packed.data() + 4u, direct_entry.data(), nlen);
                            std::memcpy(packed.data() + 4u + nlen, cubin_ptr, cubin_size);
                            (void)disk.store_blob(disk_kind, disk_sig, packed.data(), packed.size());
                        }
                    } else {
                        module = nullptr;
                    }
                }
                drv::cuLinkDestroy(link);
            }
        }

        const CUresult res = module
            ? CUDA_SUCCESS
            : drv::cuModuleLoadDataEx(&module, ptx.c_str(), 5, options, option_values);
        if (res != CUDA_SUCCESS || !module) {
            // A PTX that ptxas rejects is a code-generation bug, and the
            // exception that carries it is usually swallowed by a caller that
            // only reports "trace failed". Put the driver's own diagnosis on
            // stderr where it cannot be lost, and dump the offending PTX when
            // BROTENSOR_JIT_DEBUG is set so the reported line can be read.
            std::fprintf(stderr, "[brotensor::jit] PTX compile failed for '%s':\n%s\n",
                         label.c_str(), error_log.data());
            if (const char* dbg = std::getenv("BROTENSOR_JIT_DEBUG")) {
                if (dbg[0] != '0') std::fprintf(stderr, "%s\n", ptx.c_str());
            }
            std::fflush(stderr);
            throw std::runtime_error("CUDA JIT compilation failed for " + label + ": " +
                                     error_log.data());
        }
        retain_module(module);
        return module;
    }

    // Modules are only ever appended, and only unloaded at process exit. The
    // slot is claimed with a fetch_add so concurrent compiles do not collide.
    void retain_module(CUmodule mod) {
        const std::size_t i = module_count_.fetch_add(1, std::memory_order_relaxed);
        if (i < modules_.size()) modules_[i] = mod;
    }

    bool available_ = false;
    int cc_major_ = 0;
    int cc_minor_ = 0;
    int drv_ver_ = 0;
    CUcontext context_ = nullptr;
    std::string arch_sig_;
    Entry table_[kCapacity];
    std::array<CUmodule, kCapacity> modules_{};
    std::atomic<std::size_t> module_count_{0};
};

inline CUstream resolve_stream(void* stream) {
    if (stream) return reinterpret_cast<CUstream>(stream);
    return reinterpret_cast<CUstream>(::brotensor::cuda_current_stream());
}

#if BROTENSOR_HAS_BRASS_CUDA_JIT
inline void driver_max_ptx_version(int drv_ver, std::uint32_t& major, std::uint32_t& minor) noexcept {
    if (drv_ver >= 12090) { major = 8; minor = 8; }
    else if (drv_ver >= 12080) { major = 8; minor = 7; }
    else if (drv_ver >= 12060) { major = 8; minor = 5; }
    else if (drv_ver >= 12040) { major = 8; minor = 4; }
    else if (drv_ver >= 12020) { major = 8; minor = 2; }
    else if (drv_ver >= 12000) { major = 8; minor = 0; }
    else if (drv_ver >= 11080) { major = 7; minor = 8; }
    else if (drv_ver >= 11070) { major = 7; minor = 7; }
    else if (drv_ver >= 11060) { major = 7; minor = 6; }
    else if (drv_ver >= 11040) { major = 7; minor = 4; }
    else { major = 7; minor = 0; }
}

inline brass::target::PtxOptions make_ptx_opts(int dev = -1, bool require_sm80 = false) {
    brass::target::PtxOptions opts;
    if (dev < 0) dev = current_device();
    CudaJitEngine& eng = CudaJitEngine::for_device(dev);
    const int sm = eng.cc_major() * 10 + eng.cc_minor();
    if (require_sm80 || sm >= 80 || sm == 0) {
        opts.sm_arch = "sm_80";
        opts.ptx_version_major = 7;
        opts.ptx_version_minor = 4;
    } else if (sm >= 75) {
        opts.sm_arch = "sm_75";
        opts.ptx_version_major = 7;
        opts.ptx_version_minor = 0;
    } else {
        opts.sm_arch = "sm_70";
        opts.ptx_version_major = 7;
        opts.ptx_version_minor = 0;
    }
    if (eng.driver_version() > 0) {
        std::uint32_t max_maj = 7, max_min = 0;
        driver_max_ptx_version(eng.driver_version(), max_maj, max_min);
        if (opts.ptx_version_major > max_maj ||
            (opts.ptx_version_major == max_maj && opts.ptx_version_minor > max_min)) {
            opts.ptx_version_major = max_maj;
            opts.ptx_version_minor = max_min;
        }
    }
    return opts;
}
#endif

}  // namespace brotensor::detail::cuda::jit
