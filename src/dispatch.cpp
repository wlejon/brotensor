// Backend registration + per-operand dispatch: the dispatcher core.
//
// Storage is two parallel std::array<>s indexed by static_cast<int>(Device).
// Registration happens during static init (CPU) and at brotensor::init()
// time (CUDA/Metal probe success). Both are single-threaded entry points
// before any op call, so no mutex is required.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail {

namespace {

constexpr int kNumDevices = 3; // CPU, CUDA, Metal

// Upper bound on operands a single dispatch call inspects. The widest op is
// resblock_backward (25 operands); 32 leaves headroom. dispatch_with_opts
// throws if a future op ever exceeds this rather than silently truncating.
constexpr std::size_t kMaxOperands = 32;

struct Slot {
    OpsVTable   ops{};
    AllocVTable alloc{};
    bool        registered = false;
};

std::array<Slot, kNumDevices>& slots() {
    static std::array<Slot, kNumDevices> s{};
    return s;
}

const char* dev_name(Device d) {
    switch (d) {
        case Device::CPU:   return "CPU";
        case Device::CUDA:  return "CUDA";
        case Device::Metal: return "Metal";
    }
    return "?";
}

[[noreturn]] void throw_unregistered(Device d) {
    std::string msg = "brotensor: backend ";
    msg += dev_name(d);
    msg += " not registered";
    throw std::runtime_error(msg);
}

// A tensor is "committed" to a device once it owns real storage. A freshly
// default-constructed (or resized-to-empty) tensor has data == nullptr and
// carries no device affinity yet — it is a wildcard that adopts the op's
// device. This is what lets callers pass an unsized output tensor to a GPU op
// without first pinning it to the right backend by hand.
inline bool committed(const Tensor& t) { return t.data != nullptr; }

[[noreturn]] void throw_device_mismatch(Device resolved, int idx, Device got) {
    std::string msg = "brotensor: dispatch: operand ";
    msg += std::to_string(idx);
    msg += " is on ";
    msg += dev_name(got);
    msg += " but the op resolved to ";
    msg += dev_name(resolved);
    throw std::runtime_error(msg);
}

// Resolve the op's device from the first committed operand; verify every other
// committed operand agrees (uncommitted operands are skipped). Returns the
// vtable for the resolved device.
const OpsVTable& dispatch_v(std::initializer_list<const Tensor*> ts) {
    Device dev = Device::CPU;
    bool found = false;
    for (const Tensor* t : ts) {
        if (t && committed(*t)) { dev = t->device; found = true; break; }
    }
    if (!found && ts.size() != 0 && *ts.begin() != nullptr) {
        // All operands empty — degenerate but not an error (e.g. a fresh
        // output passed before any input). Keep operand 0's tag.
        dev = (*ts.begin())->device;
    }
    int idx = 0;
    for (const Tensor* t : ts) {
        if (t && committed(*t) && t->device != dev) {
            throw_device_mismatch(dev, idx, t->device);
        }
        ++idx;
    }
    return ops_for(dev);
}

// Same resolution rule as dispatch_v, over a plain operand array — used by the
// optional-operand dispatch path so it needs no heap allocation. Null entries
// (skipped optional operands) are tolerated.
const OpsVTable& resolve_over(const Tensor* const* all, std::size_t count) {
    Device dev = Device::CPU;
    bool found = false;
    for (std::size_t i = 0; i < count; ++i) {
        if (all[i] && committed(*all[i])) {
            dev = all[i]->device;
            found = true;
            break;
        }
    }
    if (!found) {
        // No committed operand — adopt the first non-null operand's tag.
        for (std::size_t i = 0; i < count; ++i) {
            if (all[i]) { dev = all[i]->device; break; }
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (all[i] && committed(*all[i]) && all[i]->device != dev) {
            throw_device_mismatch(dev, static_cast<int>(i), all[i]->device);
        }
    }
    return ops_for(dev);
}

[[noreturn]] void throw_too_many_operands(std::size_t n) {
    throw std::runtime_error(
        "brotensor: dispatch: operand count " + std::to_string(n) +
        " exceeds the fixed dispatch buffer (" +
        std::to_string(kMaxOperands) + ")");
}

} // namespace

void register_backend(Device d, const OpsVTable& ops, const AllocVTable& alloc) {
    auto& s = slots()[static_cast<int>(d)];
    std::memcpy(&s.ops,   &ops,   sizeof(OpsVTable));
    std::memcpy(&s.alloc, &alloc, sizeof(AllocVTable));
    s.registered = true;
}

bool is_registered(Device d) {
    return slots()[static_cast<int>(d)].registered;
}

const OpsVTable& ops_for(Device d) {
    auto& s = slots()[static_cast<int>(d)];
    if (!s.registered) throw_unregistered(d);
    return s.ops;
}

const AllocVTable& alloc_for(Device d) {
    auto& s = slots()[static_cast<int>(d)];
    if (!s.registered) throw_unregistered(d);
    return s.alloc;
}

// ─── dispatch overloads ────────────────────────────────────────────────────
//
// Each forwards to dispatch_v, which resolves the device from the first
// committed operand and skips uncommitted (data == nullptr) wildcards.

const OpsVTable& dispatch(const Tensor& a) {
    return dispatch_v({&a});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b) {
    return dispatch_v({&a, &b});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c) {
    return dispatch_v({&a, &b, &c});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d) {
    return dispatch_v({&a, &b, &c, &d});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e) {
    return dispatch_v({&a, &b, &c, &d, &e});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f) {
    return dispatch_v({&a, &b, &c, &d, &e, &f});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f,
                          const Tensor& g) {
    return dispatch_v({&a, &b, &c, &d, &e, &f, &g});
}

const OpsVTable& dispatch(const Tensor& a, const Tensor& b, const Tensor& c,
                          const Tensor& d, const Tensor& e, const Tensor& f,
                          const Tensor& g, const Tensor& h) {
    return dispatch_v({&a, &b, &c, &d, &e, &f, &g, &h});
}

const OpsVTable& dispatch_with_opts(const Tensor& a,
                                    std::initializer_list<const Tensor*> opts) {
    const std::size_t count = 1 + opts.size();
    if (count > kMaxOperands) throw_too_many_operands(count);
    const Tensor* all[kMaxOperands];
    std::size_t n = 0;
    all[n++] = &a;
    for (const Tensor* p : opts) all[n++] = p;
    return resolve_over(all, n);
}

const OpsVTable& dispatch_with_opts(const Tensor& a, const Tensor& b,
                                    std::initializer_list<const Tensor*> opts) {
    const std::size_t count = 2 + opts.size();
    if (count > kMaxOperands) throw_too_many_operands(count);
    const Tensor* all[kMaxOperands];
    std::size_t n = 0;
    all[n++] = &a;
    all[n++] = &b;
    for (const Tensor* p : opts) all[n++] = p;
    return resolve_over(all, n);
}

// ─── output adoption ───────────────────────────────────────────────────────
//
// An uncommitted output tensor (data == nullptr) has no device affinity yet.
// The wrapper calls this after dispatch so the tensor is pinned to the op's
// device before the backend impl resizes/allocates it.
void adopt_output(Tensor& t, Device d) {
    if (t.data == nullptr) t.device = d;
}

[[noreturn]] void throw_not_implemented(const char* op_name, Device d) {
    std::string msg = "brotensor: ";
    msg += op_name;
    msg += ": not implemented on ";
    msg += dev_name(d);
    throw std::runtime_error(msg);
}

} // namespace brotensor::detail
