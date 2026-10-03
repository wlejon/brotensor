// Quantised-weight chunks for the kernels that stage weights in shared
// memory as FP16 (gemm_cm.comp, conv_cm.comp) and for dequant.comp: eight
// consecutive weights of one row, loaded raw by qload8() and decoded by
// qdecode8() (the GEMM's first rule, docs/vulkan.md: the decode waits until
// the chunk is stored, after the fragment math). Include after common.glsl.
//   QF_INT8  int8 with a per-row FP32 scale: w = q * scale[row]
//   QF_Q8_0  GGUF Q8_0, 34-byte blocks of 32: w = d q
//   QF_Q4K   GGUF Q4_K, 144-byte super-blocks of 256: w = d sc q - dmin m
//   QF_Q6K   GGUF Q6_K, 210-byte super-blocks of 256: w = d sc (q - 32)
// `row` is the address of the row's first byte; `k` (a multiple of 8) the
// chunk's first element (INT8 bytes at kmax and beyond read as zero). INT8 with vec = false takes any row length and
// alignment (byte loads); everything else needs the alignments the host
// checks (16-byte aligned Q4_K rows, even addresses for Q8_0 / Q6_K, 8-byte
// aligned INT8 rows when vec).

#include "op_codes.h"   // QF_*
#extension GL_EXT_control_flow_attributes : enable

layout(buffer_reference, std430, buffer_reference_align = 2) buffer QU16x4 { u16vec4 v; };   // 8 bytes, 2-aligned
layout(buffer_reference, std430, buffer_reference_align = 8) buffer QU2Buf { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer QU4Buf { uvec4 v[]; };

struct QRaw {
    uvec4 a;     // INT8: a.xy quants; Q4_K: the header
    uvec2 b;     // Q4_K: 8 bytes of nibbles; Q6_K: the scale pair (x), d (y)
    u16vec4 s0;  // Q8_0: the quants; Q6_K: ql (2-byte aligned data stays in 16-bit words
    u16vec4 s1;  // Q6_K: qh       until the decode packs them)
    uint c;      // INT8: scale bits; Q8_0: d
};

uvec2 pk16(u16vec4 v) { return uvec2(uint(v.x) | (uint(v.y) << 16), uint(v.z) | (uint(v.w) << 16)); }

QRaw qload8(const uint fmt, const bool vec, uint64_t row, uint64_t scale, uint ridx, uint k, uint kmax) {
    QRaw o;
    o.a = uvec4(0);
    o.b = uvec2(0);
    o.s0 = u16vec4(0);
    o.s1 = u16vec4(0);
    o.c = 0u;
    if (fmt == QF_INT8) {
        if (vec) {
            o.a.xy = QU2Buf(row).v[k / 8u];
        } else {
            uint w[2] = uint[2](0u, 0u);
            [[unroll]] for (uint e = 0; e < 8u; ++e) {
                if (k + e < kmax) w[e / 4u] |= uint(U8Buf(row).v[k + e]) << ((e % 4u) * 8u);
            }
            o.a.xy = uvec2(w[0], w[1]);
        }
        o.c = U32Buf(scale).v[ridx];
    } else if (fmt == QF_Q8_0) {
        const uint64_t blk = row + uint64_t(k / 32u) * 34u;
        o.s0 = QU16x4(blk + 2u + (k % 32u)).v;
        o.c = uint(U16Buf(blk).v[0]);
    } else if (fmt == QF_Q4K) {
        const uint64_t blk = row + uint64_t(k / 256u) * 144u;
        const uint j = (k % 256u) / 32u;
        o.a = QU4Buf(blk).v[0];
        o.b = QU2Buf(blk).v[2u + (j / 2u) * 4u + (k % 32u) / 8u];
    } else {
        const uint64_t blk = row + uint64_t(k / 256u) * 210u;
        const uint e = k % 256u, g = e / 128u, quad = (e % 128u) / 32u, l0 = e % 32u;
        o.s0 = QU16x4(blk + g * 64u + (quad & 1u) * 32u + l0).v;
        o.s1 = QU16x4(blk + 128u + g * 32u + l0).v;
        o.b.x = uint(U16Buf(blk + 192u + g * 8u + quad * 2u).v[0]);
        o.b.y = uint(U16Buf(blk + 208u).v[0]);
    }
    return o;
}

uint qbyte(uvec4 v, uint i) { return (v[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

// The eight weights of a chunk as floats.
void qdecode8(const uint fmt, QRaw o, uint k, out float w[8]) {
    if (fmt == QF_INT8 || fmt == QF_Q8_0) {
        const float s = fmt == QF_INT8 ? uintBitsToFloat(o.c) : float(uint16BitsToHalf(uint16_t(o.c)));
        const uvec2 q = fmt == QF_INT8 ? o.a.xy : pk16(o.s0);
        const vec4 lo = vec4(unpack8(int(q.x))), hi = vec4(unpack8(int(q.y)));
        [[unroll]] for (uint e = 0; e < 4u; ++e) { w[e] = lo[e] * s; w[4u + e] = hi[e] * s; }
    } else if (fmt == QF_Q4K) {
        const uint j = (k % 256u) / 32u;
        const uvec4 s = uvec4(o.a.y, o.a.z, o.a.w, 0u);
        uint sc, mn;
        if (j < 4u) {
            sc = qbyte(s, j) & 63u;
            mn = qbyte(s, j + 4u) & 63u;
        } else {
            sc = (qbyte(s, j + 4u) & 15u) | ((qbyte(s, j - 4u) >> 6) << 4);
            mn = (qbyte(s, j + 4u) >> 4) | ((qbyte(s, j) >> 6) << 4);
        }
        const float d = float(uint16BitsToHalf(uint16_t(o.a.x & 0xffffu)));
        const float dmin = float(uint16BitsToHalf(uint16_t(o.a.x >> 16)));
        const float ws = float(sc) * d, wm = float(mn) * dmin;
        const uint sh = (j & 1u) * 4u;
        const vec4 lo = vec4(unpack8((o.b.x >> sh) & 0x0f0f0f0fu)), hi = vec4(unpack8((o.b.y >> sh) & 0x0f0f0f0fu));
        [[unroll]] for (uint e = 0; e < 4u; ++e) { w[e] = ws * lo[e] - wm; w[4u + e] = ws * hi[e] - wm; }
    } else {
        const uint e0 = k % 256u, quad = (e0 % 128u) / 32u, l0 = e0 % 32u;
        const uint sbyte = (o.b.x >> ((l0 / 16u) * 8u)) & 0xffu;
        const float d = float(uint16BitsToHalf(uint16_t(o.b.y)));
        const float s = d * float(int(sbyte << 24) >> 24);
        const uvec2 qlw = pk16(o.s0), qhw = pk16(o.s1);
        [[unroll]] for (uint h = 0; h < 2u; ++h) {
            const uint ql = qlw[h], qh = qhw[h];
            const uint nib = quad >= 2u ? (ql >> 4) & 0x0f0f0f0fu : ql & 0x0f0f0f0fu;
            const uint hb = ((qh >> (2u * quad)) & 0x03030303u) << 4;
            const vec4 v = vec4(unpack8(nib | hb)) - vec4(32.0);
            [[unroll]] for (uint e = 0; e < 4u; ++e) w[h * 4u + e] = s * v[e];
        }
    }
}

// The same as eight FP16 values packed in a uvec4 (RNE conversion).
uvec4 qdecode8_f16(const uint fmt, QRaw o, uint k) {
    float w[8];
    qdecode8(fmt, o, k, w);
    return uvec4(packFloat2x16(f16vec2(w[0], w[1])), packFloat2x16(f16vec2(w[2], w[3])),
                 packFloat2x16(f16vec2(w[4], w[5])), packFloat2x16(f16vec2(w[6], w[7])));
}
