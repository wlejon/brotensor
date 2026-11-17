// FP8 E4M3 (OCP e4m3fn) for the byte-format weights (QF_FP8): decode as
// brotensor::fp8_e4m3_bits_to_fp32 on the host, bit for bit, and encode as
// brotensor::fp32_to_fp8_e4m3_bits. No FP32 subnormal is ever formed, so a
// device that flushes them decodes the same. Include after common.glsl.

const float FP8_NAN = uintBitsToFloat(0x7fc00000u);

// Four codes packed in a uint, the first code lowest.
vec4 fp8x4(uint w) {
    const uvec4 b = (uvec4(w) >> uvec4(0u, 8u, 16u, 24u)) & 0xffu;
    const uvec4 v = b & 0x7fu;
    // Normal codes: the exponent rebiased by 120, the mantissa moved up 20
    // bits; subnormal ones (exponent 0) are m / 8 * 2^-6.
    vec4 mag = mix(uintBitsToFloat((v << 20) + 0x3c000000u), vec4(v) * 0.001953125, lessThan(v, uvec4(8u)));
    mag = mix(mag, vec4(FP8_NAN), equal(v, uvec4(0x7fu)));
    return uintBitsToFloat(floatBitsToUint(mag) | ((b & 0x80u) << 24));
}

float fp8_decode(uint b) { return fp8x4(b & 0xffu).x; }

// Magnitude codes 0..126 ascend with their value: bracket by binary search,
// round to the nearer, ties to the even code; saturate at 448.
uint fp8_encode(float v) {
    if (isnan(v)) return 0x7fu;
    const uint sign = (floatBitsToUint(v) >> 31) << 7;
    const float a = abs(v);
    if (a >= 448.0) return sign | 0x7eu;
    uint lo = 0u, hi = 126u;
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if (fp8_decode(mid) <= a) lo = mid;
        else hi = mid;
    }
    const float dl = a - fp8_decode(lo);
    const float dh = fp8_decode(hi) - a;
    const uint code = dl < dh ? lo : dh < dl ? hi : ((lo & 1u) != 0u ? hi : lo);
    return sign | code;
}
