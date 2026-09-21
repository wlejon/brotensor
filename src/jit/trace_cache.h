#pragma once

#include <brotensor/jit/trace.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace brotensor::jit {

class TraceHandleImpl {
public:
    uint64_t signature_hash = 0;
    bool is_cuda = false;
    size_t node_count = 0;
    double last_duration_us = 0.0;
    bool is_cache_hit = false;

    // Kernel launches one execute() issues, and what the compiler called the
    // fusion it picked. Both are read by the benchmarks and the tests.
    size_t launch_count = 0;
    double compile_us = 0.0;
    const char* fusion_name = "unfused";

    // Callable kernel dispatcher with bound buffer pointers
    std::function<void()> execute_fn;

    // Factory/builder to re-bind pointers when reusing compiled artifact
    std::function<std::shared_ptr<TraceHandleImpl>(const std::vector<void*>& inputs,
                                                   const std::vector<void*>& outputs)> rebind_fn;

    void execute() {
        if (!execute_fn) {
            throw std::runtime_error("brotensor::jit: TraceHandle has no executable function bound");
        }
        auto t0 = std::chrono::steady_clock::now();
        execute_fn();
        auto t1 = std::chrono::steady_clock::now();
        last_duration_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
};

// ─── TraceCache ─────────────────────────────────────────────────────────────
//
// Compiled-trace lookup by DAG signature, with no lock on any path.
//
// The table is a fixed-capacity open-addressed array of write-once entries.
// An insert claims a slot by compare-exchanging its key from 0 to the
// signature; the winner then writes the owning shared_ptr and publishes it by
// storing the raw pointer with release ordering. A reader that sees a
// published pointer (acquire) is guaranteed to see the fully written
// shared_ptr beside it, and nothing ever rewrites an entry, so the copy it
// takes cannot race with a writer.
//
// Two threads that race on the same signature both compile; the loser's
// artifact is simply dropped and it uses the winner's. That costs a duplicate
// PTX compile in a case that does not arise in practice (a trace is compiled
// once, on the first step) and buys a table with no blocking and no spin.
//
// Capacity is fixed because a trace population is small and static: a model's
// worth of fused patterns is tens of entries. A full table degrades to "always
// recompile", never to a wrong answer.
class TraceCache {
public:
    static constexpr size_t kCapacity = 1024;  // power of two

    static TraceCache& instance();

    std::shared_ptr<TraceHandleImpl> lookup(uint64_t hash);
    void insert(uint64_t hash, std::shared_ptr<TraceHandleImpl> handle);

    size_t hit_count() const noexcept;
    size_t miss_count() const noexcept;

    // Drops every entry. Not safe against concurrent lookup/insert — it exists
    // for tests that want a cold cache, which run single-threaded.
    void clear();

private:
    TraceCache() = default;

    struct Entry {
        std::atomic<uint64_t> key{0};
        std::atomic<TraceHandleImpl*> published{nullptr};
        std::shared_ptr<TraceHandleImpl> owner;
    };

    // A signature of 0 means "empty", so fold it onto a different value.
    static uint64_t normalize(uint64_t h) { return h ? h : 0x9E3779B97F4A7C15ULL; }
    static size_t slot_of(uint64_t h) { return static_cast<size_t>(h) & (kCapacity - 1); }

    // A plain array, not a vector: the entries hold atomics, so the storage
    // must never be relocated.
    Entry table_[kCapacity];
    mutable std::atomic<size_t> hit_count_{0};
    mutable std::atomic<size_t> miss_count_{0};
};

}  // namespace brotensor::jit
