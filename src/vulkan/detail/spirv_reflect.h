#pragma once

// Minimal SPIR-V reflection: just enough to check a compute shader against the
// device limits *before* vkCreateComputePipelines sees it.
//
// It resolves the workgroup size (LocalSize, LocalSizeId, or a WorkgroupSize
// built-in composite, with specialisation constants applied) and sums the size
// of every Workgroup-storage variable (GLSL `shared`), with array lengths taken
// from constants, specialisation constants (overridden by the supplied
// specialisation data) and OpSpecConstantOp integer arithmetic.
//
// Sizes follow the natural std430-like layout (vec3 rounds to 16 bytes), which
// is what drivers allocate or a little more. A shared array whose length this
// reflector cannot evaluate makes reflect() fail with a reason, so a new shader
// cannot slip past the guard silently; express shared-array lengths as
// constants, spec constants or + - * / << >> & | of them.

#include <cstdint>
#include <string>

namespace brotensor::detail::vulkan {

struct SpirvInfo {
    std::uint32_t local[3] = {1, 1, 1};
    std::uint64_t shared_bytes = 0;
};

// spec[i] is the value for constant_id i (i < nspec). Returns false with a
// reason on malformed or unevaluable input.
bool reflect_spirv(const std::uint32_t* words, std::size_t nwords,
                   const std::uint32_t* spec, std::uint32_t nspec,
                   SpirvInfo* out, std::string* why);

}  // namespace brotensor::detail::vulkan
