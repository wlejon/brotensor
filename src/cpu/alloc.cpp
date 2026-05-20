// ─── CPU AllocVTable ───────────────────────────────────────────────────────
//
// Plain host memory. malloc/free + memcpy/memset, sync is a no-op. All
// "device" pointers for the CPU backend are simply host pointers; the
// d2h / h2d / d2d hooks all collapse to std::memcpy so cross-device copies
// involving CPU work without any special casing in Tensor::to().

#include <brotensor/detail/dispatch.h>

#include <cstdlib>
#include <cstring>
#include <new>

namespace brotensor::detail::cpu {

void* cpu_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* p = std::malloc(bytes);
    if (!p) throw std::bad_alloc();
    return p;
}

void cpu_free(void* ptr) {
    if (ptr) std::free(ptr);
}

void cpu_memcpy_h2d(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}

void cpu_memcpy_d2h(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}

void cpu_memcpy_d2d(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
}

void cpu_memset_zero(void* dst, std::size_t n) {
    if (n) std::memset(dst, 0, n);
}

void cpu_sync() {
    // no-op — CPU ops are synchronous
}

const AllocVTable& cpu_alloc_table() {
    static const AllocVTable t = {
        &cpu_alloc,
        &cpu_free,
        &cpu_memcpy_h2d,
        &cpu_memcpy_d2h,
        &cpu_memcpy_d2d,
        &cpu_memset_zero,
        &cpu_sync,
    };
    return t;
}

} // namespace brotensor::detail::cpu
