#include "trace_cache.h"
#include "trace_dag.h"

#include <stdexcept>

namespace brotensor::jit {

void TraceHandle::execute() {
    if (!impl_) {
        throw std::runtime_error("brotensor::jit: cannot execute uninitialized TraceHandle");
    }
    impl_->execute();
}

uint64_t TraceHandle::signature_hash() const {
    return impl_ ? impl_->signature_hash : 0;
}

bool TraceHandle::is_cuda() const {
    return impl_ ? impl_->is_cuda : false;
}

size_t TraceHandle::node_count() const {
    return impl_ ? impl_->node_count : 0;
}

double TraceHandle::last_duration_us() const {
    return impl_ ? impl_->last_duration_us : 0.0;
}

bool TraceHandle::is_cache_hit() const {
    return impl_ ? impl_->is_cache_hit : false;
}

size_t TraceHandle::launch_count() const {
    return impl_ ? impl_->launch_count : 0;
}

double TraceHandle::compile_us() const {
    return impl_ ? impl_->compile_us : 0.0;
}

const char* TraceHandle::fusion_name() const {
    return impl_ ? impl_->fusion_name : "none";
}

TraceCache& TraceCache::instance() {
    static TraceCache cache;
    return cache;
}

std::shared_ptr<TraceHandleImpl> TraceCache::lookup(uint64_t hash) {
    const uint64_t k = normalize(hash);
    size_t i = slot_of(k);
    for (size_t probe = 0; probe < kCapacity; ++probe, i = (i + 1) & (kCapacity - 1)) {
        const uint64_t cur = table_[i].key.load(std::memory_order_acquire);
        if (cur == 0) break;  // empty slot ends the probe run
        if (cur != k) continue;
        // Claimed for this signature. A slot whose owner is not published yet
        // belongs to a compile still in flight on another thread; report a
        // miss and let this thread compile its own rather than spin.
        if (table_[i].published.load(std::memory_order_acquire) == nullptr) break;
        hit_count_.fetch_add(1, std::memory_order_relaxed);
        return table_[i].owner;
    }
    miss_count_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void TraceCache::insert(uint64_t hash, std::shared_ptr<TraceHandleImpl> handle) {
    if (!handle) return;
    const uint64_t k = normalize(hash);
    size_t i = slot_of(k);
    for (size_t probe = 0; probe < kCapacity; ++probe, i = (i + 1) & (kCapacity - 1)) {
        uint64_t expect = 0;
        if (table_[i].key.compare_exchange_strong(expect, k, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
            table_[i].owner = std::move(handle);
            table_[i].published.store(table_[i].owner.get(), std::memory_order_release);
            return;
        }
        if (expect == k) {
            // Someone else owns this signature. If they have not published yet
            // they are about to; either way their artifact is equivalent to
            // ours, so drop ours.
            return;
        }
    }
    // Table full: the trace stays uncached and will be recompiled next time.
}

size_t TraceCache::hit_count() const noexcept {
    return hit_count_.load(std::memory_order_relaxed);
}

size_t TraceCache::miss_count() const noexcept {
    return miss_count_.load(std::memory_order_relaxed);
}

void TraceCache::clear() {
    for (auto& e : table_) {
        e.published.store(nullptr, std::memory_order_relaxed);
        e.owner.reset();
        e.key.store(0, std::memory_order_relaxed);
    }
    hit_count_.store(0, std::memory_order_relaxed);
    miss_count_.store(0, std::memory_order_relaxed);
}

void begin_trace() {
    TraceContext::current().begin();
}

bool is_tracing() {
    return TraceContext::current().is_active();
}

void abort_trace() {
    TraceContext::current().discard();
}

}  // namespace brotensor::jit
