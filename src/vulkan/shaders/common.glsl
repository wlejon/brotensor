// Shared GLSL for the brotensor Vulkan backend. #include'd by every kernel.
//
// Tensors arrive as 64-bit buffer device addresses in the push-constant block
// (the backend uses no descriptor sets), and are accessed through the
// buffer_reference types below. Element index arithmetic stays 32-bit: no
// buffer exceeds 4 GiB (detail/allocator.h), so a byte offset of an element
// of >= 2 bytes always fits the driver's 64-bit address math.
//
// Dtype codes (match detail/kernels.h): 0 = FP32, 1 = FP16, 2 = BF16.
// All arithmetic is FP32. FP16 converts with float16_t(x), which is
// round-to-nearest-even; never packHalf2x16, which rounds toward zero on RADV.
// BF16 has no shader type on RADV (no VK_KHR_shader_bfloat16): it is carried
// as uint16 bits and converted here in registers, RNE with NaN kept quiet,
// bit-identical to brotensor::fp32_to_bf16_bits on the host.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32Buf { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer F16Buf { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16Buf { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Buf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32Buf { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8Buf  { uint8_t v[]; };

#define DT_F32  0
#define DT_F16  1
#define DT_BF16 2

float bf16_to_f32(uint16_t b) { return uintBitsToFloat(uint(b) << 16); }

uint16_t f32_to_bf16(float f) {
    uint u = floatBitsToUint(f);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return uint16_t((u >> 16) | 0x40u);          // quiet NaN
    }
    u += 0x7fffu + ((u >> 16) & 1u);                  // round to nearest even
    return uint16_t(u >> 16);
}

// `dt` is always a compile-time constant at the call sites, so the branches
// fold away.
float load_elem(const int dt, uint64_t base, uint i) {
    if (dt == DT_F32) return F32Buf(base).v[i];
    if (dt == DT_F16) return float(F16Buf(base).v[i]);
    return bf16_to_f32(U16Buf(base).v[i]);
}

void store_elem(const int dt, uint64_t base, uint i, float x) {
    if (dt == DT_F32) { F32Buf(base).v[i] = x; return; }
    if (dt == DT_F16) { F16Buf(base).v[i] = float16_t(x); return; }
    U16Buf(base).v[i] = f32_to_bf16(x);
}

// Grid-stride loop bound: total invocations in the dispatch's x dimension.
// (A macro: gl_WorkGroupSize is only readable after the kernel's layout.)
#define GRID_STRIDE_X (gl_NumWorkGroups.x * gl_WorkGroupSize.x)

// ─── Numerics shared by several kernels ────────────────────────────────────
//
// GLSL's built-ins are not IEEE-exact (Vulkan allows exp ~3 ulp, sin/cos an
// absolute 2^-11 on [-pi, pi]); where a built-in is used the tests' tolerance
// reflects that. tanh and erf are written out: tanh to stay finite for large
// |x| (exp overflow would give inf/inf) and accurate near 0, erf because GLSL
// has none.

float bt_tanh(float x) {
    const float ax = abs(x);
    if (ax < 0.25) {   // Maclaurin series; the first omitted term is < 1e-10
        const float x2 = x * x;
        return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0 +
                    x2 * (62.0 / 2835.0 + x2 * (-1382.0 / 155925.0))))));
    }
    const float t = exp(-2.0 * ax);
    const float r = (1.0 - t) / (1.0 + t);
    return x < 0.0 ? -r : r;
}

// erf: Abramowitz & Stegun 7.1.26 is too coarse (1.5e-7 absolute but poor
// relative accuracy near 0), so use the Maclaurin series near 0 and a
// rational (Numerical Recipes erfcc, fractional error < 1.2e-7) for the rest.
float bt_erf(float x) {
    const float ax = abs(x);
    float r;
    if (ax < 0.5) {
        const float x2 = x * x;
        // 2/sqrt(pi) * (x - x^3/3 + x^5/10 - x^7/42 + x^9/216 - x^11/1320)
        r = 1.1283791670955126 * ax *
            (1.0 + x2 * (-1.0 / 3.0 + x2 * (0.1 + x2 * (-1.0 / 42.0 + x2 * (1.0 / 216.0 +
             x2 * (-1.0 / 1320.0))))));
    } else {
        // erfc(x) = t * exp(-x^2 + P(t)), t = 1 / (1 + 0.5 x)
        const float t = 1.0 / (1.0 + 0.5 * ax);
        const float p = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                        t * (-0.82215223 + t * 0.17087277))))))));
        r = 1.0 - t * exp(-ax * ax + p);
    }
    return x < 0.0 ? -r : r;
}

float bt_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// tanh-approximation GELU, written through the identity 0.5 * (1 + tanh(u)) =
// sigmoid(2u): the same function as the CPU's 0.5*x*(1 + tanh(u)) without the
// cancellation in 1 + tanh(u) for negative x.
const float kGeluC = 0.7978845608028654;   // sqrt(2 / pi)
float bt_gelu_tanh(float x) {
    const float u = kGeluC * (x + 0.044715 * x * x * x);
    return x / (1.0 + exp(-2.0 * u));
}
float bt_gelu_tanh_grad(float x) {
    const float x2 = x * x;
    const float s = bt_sigmoid(2.0 * kGeluC * (x + 0.044715 * x2 * x));   // 0.5 * (1 + tanh u)
    const float d_inner = kGeluC * (1.0 + 3.0 * 0.044715 * x2);
    // 0.5 (1 + t) + 0.5 x (1 - t^2) u' with 1 + t = 2s, 1 - t^2 = 4 s (1 - s)
    return s + 2.0 * x * s * (1.0 - s) * d_inner;
}
