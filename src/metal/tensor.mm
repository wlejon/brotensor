// Metal AllocVTable + the pointer→MTLBuffer pool.
//
// All brotensor::Tensor allocation / freeing / memcpy / zero / sync on the
// Metal device routes through `metal_alloc_table()` (see
// <brotensor/detail/dispatch.h>); register.mm pairs it with the Metal
// OpsVTable and hands the pair to the dispatcher.
//
// The old split `GpuTensor` type — its ctor/dtor, upload/download free
// functions, and the host-side fp16 bit-conversion helpers — is gone. The
// unified `brotensor::Tensor` in src/tensor.cpp subsumes all of them; the
// conversion helpers moved there too.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#import "internal.h"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace brotensor {

// ─── Pointer → MTLBuffer pool ──────────────────────────────────────────────
//
// brotensor::Tensor stores only an opaque `void* data`. On Metal we allocate
// MTLBuffers with shared storage (unified memory on Apple Silicon) and use
// `[buf contents]` as the `data` pointer. To free the buffer when the Tensor
// is destroyed — and to recover the MTLBuffer when an op needs to bind it —
// we keep a global pointer→buffer map. Views into an existing buffer (a
// non-owning Tensor whose data points partway into a pooled allocation) are
// resolved by the range scan in pool_lookup / pool_lookup_offset.

namespace metal_impl {

namespace {
struct Pool {
    std::mutex mu;
    std::unordered_map<const void*, void*> map; // void* holds an Obj-C +1 retain
};
Pool& pool() { static Pool p; return p; }
} // namespace

void pool_register(void* data_ptr, id<MTLBuffer> buf) {
    if (!data_ptr || !buf) return;
    auto& P = pool();
    void* retained = (__bridge_retained void*)buf;
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        id<MTLBuffer> prior = (__bridge_transfer id<MTLBuffer>)it->second;
        (void)prior;
        it->second = retained;
    } else {
        P.map.emplace(data_ptr, retained);
    }
}

id<MTLBuffer> pool_lookup(const void* data_ptr) {
    if (!data_ptr) return nil;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        return (__bridge id<MTLBuffer>)it->second;
    }
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return buf;
        }
    }
    return nil;
}

NSUInteger pool_lookup_offset(const void* data_ptr) {
    if (!data_ptr) return 0;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) return 0;
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return static_cast<NSUInteger>(p - base);
        }
    }
    return 0;
}

void pool_release(void* data_ptr) {
    if (!data_ptr) return;
    auto& P = pool();
    void* retained = nullptr;
    {
        std::lock_guard<std::mutex> lk(P.mu);
        auto it = P.map.find(data_ptr);
        if (it == P.map.end()) return;
        retained = it->second;
        P.map.erase(it);
    }
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)retained;
    (void)buf;
}

} // namespace metal_impl

// ─── Metal AllocVTable ─────────────────────────────────────────────────────

namespace detail::metal {

void* metal_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    ::brotensor::cuda_init();
    @autoreleasepool {
        id<MTLBuffer> buf = [metal_impl::device()
            newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error("Metal: failed to allocate MTLBuffer");
        }
        void* p = [buf contents];
        metal_impl::pool_register(p, buf);
        return p;
    }
}

void metal_free(void* ptr) {
    if (ptr) metal_impl::pool_release(ptr);
}

// Apple Silicon unified memory: an MTLBuffer's contents pointer is directly
// host-addressable, so every transfer direction is a plain memcpy.
void metal_memcpy_h2d(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}
void metal_memcpy_d2h(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}
void metal_memcpy_d2d(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}

void metal_memset_zero(void* dst, std::size_t n) {
    if (n) std::memset(dst, 0, n);
}

// Ops submit their own command buffers and wait on them before returning, so
// by the time control reaches here the GPU is already caught up.
void metal_sync() {}

const ::brotensor::detail::AllocVTable& metal_alloc_table() {
    static const ::brotensor::detail::AllocVTable t = {
        &metal_alloc,
        &metal_free,
        &metal_memcpy_h2d,
        &metal_memcpy_d2h,
        &metal_memcpy_d2d,
        &metal_memset_zero,
        &metal_sync,
    };
    return t;
}

} // namespace detail::metal

} // namespace brotensor
