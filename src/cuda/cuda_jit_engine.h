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
#include <mutex>
#include <stdexcept>
#include <string>
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

    CUfunction get_function(const std::string& key, const std::string& ptx,
                            const std::string& entry_name) {
        if (!available_) {
            throw std::runtime_error("CUDA Driver JIT is not available on this system");
        }

        const std::uint64_t h = key_hash(key);
        std::size_t i = slot_of(h);
        for (std::size_t probe = 0; probe < kCapacity;) {
            Entry& e = table_[i];
            const std::uint64_t cur = e.key.load(std::memory_order_acquire);

            if (cur == h) {
                const CUfunction fn = e.fn.load(std::memory_order_acquire);
                if (fn) {
                    // `name` was written before the release store of `fn`, so
                    // acquiring a published function makes it safe to read.
                    if (e.name == key) return fn;
                    ++probe;
                    i = (i + 1) & (kCapacity - 1);
                    continue;  // 64-bit hash collision on a different key
                }
                // The slot is claimed but nothing is published yet: a compile
                // is in flight on another thread, and `name` cannot be read
                // safely to confirm it is ours. Compile uncached.
                return compile(ptx, entry_name);
            }

            if (cur == 0) {
                std::uint64_t expect = 0;
                if (e.key.compare_exchange_strong(expect, h, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
                    e.name = key;
                    return publish(e, ptx, entry_name);
                }
                continue;  // lost the claim — re-examine this same slot
            }

            ++probe;
            i = (i + 1) & (kCapacity - 1);
        }
        // Table full — compile without caching rather than fail.
        return compile(ptx, entry_name);
    }

private:
    struct Entry {
        std::atomic<std::uint64_t> key{0};
        std::atomic<CUfunction> fn{nullptr};
        std::string name;  // written before the key is published
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
            int cc_major = 0, cc_minor = 0, drv_ver = 0;
            if (drv::cuDeviceGet(&cu_dev, dev) == CUDA_SUCCESS &&
                drv::cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev) == CUDA_SUCCESS &&
                drv::cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev) == CUDA_SUCCESS) {
                (void)drv::cuDriverGetVersion(&drv_ver);
                char abuf[64];
                std::snprintf(abuf, sizeof(abuf), "sm_%d%d:drv%d", cc_major, cc_minor, drv_ver);
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

    CUfunction publish(Entry& e, const std::string& ptx, const std::string& entry_name) {
        CUfunction fn = compile(ptx, entry_name);
        CUfunction expect = nullptr;
        if (e.fn.compare_exchange_strong(expect, fn, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            return fn;
        }
        // Lost the publish race; the winner's function is equivalent.
        return expect;
    }

    CUfunction compile(const std::string& ptx, const std::string& entry_name) {
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
        std::string cubin_sig;
        if (!arch_sig_.empty() && disk.is_enabled()) {
            char hbuf[32];
            std::snprintf(hbuf, sizeof(hbuf), "%016llx:%zu",
                          static_cast<unsigned long long>(key_hash(ptx)), ptx.size());
            cubin_sig = arch_sig_ + ":" + entry_name + ":" + hbuf;
            std::vector<std::uint8_t> cached_cubin;
            if (disk.load_blob("cuda_cubin", cubin_sig, cached_cubin) && !cached_cubin.empty()) {
                if (drv::cuModuleLoadData(&module, cached_cubin.data()) != CUDA_SUCCESS) {
                    module = nullptr;
                }
            }
        }

        if (!module && !cubin_sig.empty()) {
            CUlinkState link = nullptr;
            if (drv::cuLinkCreate(5, options, option_values, &link) == CUDA_SUCCESS && link) {
                void* cubin_ptr = nullptr;
                std::size_t cubin_size = 0;
                const CUresult add_res = drv::cuLinkAddData(
                    link, CU_JIT_INPUT_PTX, const_cast<char*>(ptx.c_str()),
                    ptx.size() + 1, entry_name.c_str(), 0, nullptr, nullptr);
                if (add_res == CUDA_SUCCESS &&
                    drv::cuLinkComplete(link, &cubin_ptr, &cubin_size) == CUDA_SUCCESS &&
                    cubin_ptr && cubin_size > 0) {
                    if (drv::cuModuleLoadData(&module, cubin_ptr) == CUDA_SUCCESS) {
                        (void)disk.store_blob("cuda_cubin", cubin_sig, cubin_ptr, cubin_size);
                    } else {
                        module = nullptr;
                    }
                }
                drv::cuLinkDestroy(link);
            }
        }

        CUresult res = module
            ? CUDA_SUCCESS
            : drv::cuModuleLoadDataEx(&module, ptx.c_str(), 5, options, option_values);
        if (res != CUDA_SUCCESS) {
            // A PTX that ptxas rejects is a code-generation bug, and the
            // exception that carries it is usually swallowed by a caller that
            // only reports "trace failed". Put the driver's own diagnosis on
            // stderr where it cannot be lost, and dump the offending PTX when
            // BROTENSOR_JIT_DEBUG is set so the reported line can be read.
            std::fprintf(stderr, "[brotensor::jit] PTX compile failed for '%s':\n%s\n",
                         entry_name.c_str(), error_log.data());
            if (const char* dbg = std::getenv("BROTENSOR_JIT_DEBUG")) {
                if (dbg[0] != '0') std::fprintf(stderr, "%s\n", ptx.c_str());
            }
            std::fflush(stderr);
            throw std::runtime_error("CUDA JIT compilation failed for " + entry_name + ": " +
                                     error_log.data());
        }
        retain_module(module);

        CUfunction fn = nullptr;
        res = drv::cuModuleGetFunction(&fn, module, entry_name.c_str());
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to find kernel entry point '" + entry_name +
                                     "' in compiled PTX module");
        }
        return fn;
    }

    // Modules are only ever appended, and only unloaded at process exit. The
    // slot is claimed with a fetch_add so concurrent compiles do not collide.
    void retain_module(CUmodule mod) {
        const std::size_t i = module_count_.fetch_add(1, std::memory_order_relaxed);
        if (i < modules_.size()) modules_[i] = mod;
    }

    bool available_ = false;
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
inline brass::target::PtxOptions make_ptx_opts() {
    brass::target::PtxOptions opts;
    // The oldest architecture brotensor builds for (Ampere): the driver JITs
    // PTX forward to the GPU it runs on, never back, so a newer .target fails
    // to load on an older card (sm_89 PTX on an RTX 3060, sm_86: "SM version
    // specified by .target is higher than default"). These kernels use
    // nothing past sm_80.
    opts.sm_arch = "sm_80";
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 8;
    return opts;
}
#endif

}  // namespace brotensor::detail::cuda::jit
