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

TraceCache& TraceCache::instance() {
    static TraceCache cache;
    return cache;
}

std::shared_ptr<TraceHandleImpl> TraceCache::lookup(uint64_t hash) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(hash);
    if (it != cache_.end()) {
        ++hit_count_;
        return it->second;
    }
    ++miss_count_;
    return nullptr;
}

void TraceCache::insert(uint64_t hash, std::shared_ptr<TraceHandleImpl> handle) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_[hash] = std::move(handle);
}

size_t TraceCache::hit_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return hit_count_;
}

size_t TraceCache::miss_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return miss_count_;
}

void TraceCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
    hit_count_ = 0;
    miss_count_ = 0;
}

void begin_trace() {
    TraceContext::current().begin();
}

bool is_tracing() {
    return TraceContext::current().is_active();
}

} // namespace brotensor::jit
