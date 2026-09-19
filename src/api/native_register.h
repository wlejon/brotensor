#pragma once

// Shared registration helpers for the per-group native files.
//
// api.cpp registers the core surface itself; each restored group
// (native_tensor_batched.cpp, native_tensor_audio*.cpp, native_tensor_conv.cpp,
// native_tensor_int8.cpp, native_tensor_misc.cpp, native_tensor_attention2.cpp)
// owns a `registerTensorNatives_<group>()` that api.cpp chains onto the core
// registration. Keeping each group's table beside its bodies is what lets the
// files stay under 1,000 lines.

#include "embed/embed.h"

#include <initializer_list>
#include <string>

namespace brotensor::api::reg {

namespace ev = bronze::embed;

// The two spellings every group's table uses.
inline constexpr const char* kTensorCls = "__bro_native.tensor.GpuTensor";
inline constexpr const char* kDyn = "dynamic";

inline ev::NativeSignature sig(const char* ret, std::initializer_list<const char*> params) {
    ev::NativeSignature s;
    s.returnType = ret;
    for (const char* p : params) s.paramTypes.emplace_back(p);
    s.kind = ev::NativeKind::Function;
    return s;
}

inline bool fn(const char* path, void* f, const char* ret,
               std::initializer_list<const char*> params, std::string* error) {
    return ev::registerNative(path, f, sig(ret, params), error);
}

template <typename F>
inline void* p(F* f) { return reinterpret_cast<void*>(f); }

} // namespace brotensor::api::reg
