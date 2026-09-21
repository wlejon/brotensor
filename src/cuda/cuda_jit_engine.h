#pragma once

#include "cuda_driver.h"

#include <cuda.h>
#include <cuda_runtime.h>
#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/target/ptx_target.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    static constexpr std::size_t kCapacity = 512;  // power of two

    static CudaJitEngine& instance() {
        static CudaJitEngine engine;
        return engine;
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

    CudaJitEngine() {
        // Without a driver on the machine every drv:: call answers
        // CUDA_ERROR_NOT_INITIALIZED, so this is also the "no GPU" exit.
        if (drv::cuInit(0) != CUDA_SUCCESS) return;

        int dev_count = 0;
        if (drv::cuDeviceGetCount(&dev_count) != CUDA_SUCCESS || dev_count <= 0) return;

        if (drv::cuCtxGetCurrent(&context_) != CUDA_SUCCESS || !context_) {
            CUdevice dev;
            if (drv::cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
                if (drv::cuDevicePrimaryCtxRetain(&context_, dev) == CUDA_SUCCESS) {
                    drv::cuCtxSetCurrent(context_);
                }
            }
        }
        available_ = (context_ != nullptr);
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
        CUcontext current_ctx = nullptr;
        drv::cuCtxGetCurrent(&current_ctx);
        if (!current_ctx) drv::cuCtxSetCurrent(context_);

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
        CUresult res = drv::cuModuleLoadDataEx(&module, ptx.c_str(), 5, options, option_values);
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
    opts.sm_arch = "sm_89";
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 8;
    return opts;
}
#endif

}  // namespace brotensor::detail::cuda::jit
