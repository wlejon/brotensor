#pragma once

#include <brotensor/jit/trace.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace brotensor::jit {

class TraceHandleImpl {
public:
    uint64_t signature_hash = 0;
    bool is_cuda = false;
    size_t node_count = 0;
    double last_duration_us = 0.0;
    bool is_cache_hit = false;

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

class TraceCache {
public:
    static TraceCache& instance();

    std::shared_ptr<TraceHandleImpl> lookup(uint64_t hash);
    void insert(uint64_t hash, std::shared_ptr<TraceHandleImpl> handle);

    size_t hit_count() const noexcept;
    size_t miss_count() const noexcept;
    void clear();

private:
    TraceCache() = default;
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, std::shared_ptr<TraceHandleImpl>> cache_;
    size_t hit_count_ = 0;
    size_t miss_count_ = 0;
};

} // namespace brotensor::jit
