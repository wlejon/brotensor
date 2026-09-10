#pragma once

// brotensor detail/hash_rng.h — counter-based hash RNG shared by the CPU and
// CUDA backends (and by tests that need to reproduce an op's noise).
//
// A stateless, order-independent uniform generator: the uniform for element
// `index` of stream (`seed`, `domain`) is a pure function of those three
// integers, so a CPU loop and a CUDA grid produce bit-identical noise for
// bit-identical inputs. The mixer is splitmix64 (Steele/Lea/Flood) — a
// bijective 64-bit finaliser with full avalanche, which is exactly what a
// counter-based generator needs; the input is `seed ^ (domain + index)` so a
// caller varies `seed` per step and each op keeps a distinct `domain` for
// each of its noise streams (a position stream and a vocabulary stream must
// not collide at index 0).
//
// The uniform takes the high 24 bits of the hash to a float in [0, 1) — the
// same mapping the Philox ops use (rand_uniform, sample_logits), so the value
// set is identical: multiples of 2^-24, never 1.0.
//
// Every function is `inline` and, under nvcc, `__host__ __device__`, so the
// header compiles into both a .cpp and a .cu translation unit with identical
// integer arithmetic. Metal cannot include this header (MSL is compiled from
// source strings); src/metal/masked_diffusion.mm carries a verbatim copy.

#include <cstdint>

#if defined(__CUDACC__)
#define BROTENSOR_HASH_RNG_HD __host__ __device__
#else
#define BROTENSOR_HASH_RNG_HD
#endif

namespace brotensor::detail {

// Domain constants for the ops that draw from this generator. Arbitrary
// large odd constants, far apart, so `domain + index` never overlaps across
// streams for any index a tensor can hold.
inline constexpr std::uint64_t kHashDomainMaskedDiffusionPosition = 0x5851F42D4C957F2DULL;
inline constexpr std::uint64_t kHashDomainMaskedDiffusionClass    = 0x14057B7EF767814FULL;

// splitmix64 finaliser: one round of the SplitMix64 output function applied
// to `x` (including the golden-ratio increment, so x == 0 does not map to 0).
BROTENSOR_HASH_RNG_HD inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Raw 64-bit hash for element `index` of stream (seed, domain).
BROTENSOR_HASH_RNG_HD inline std::uint64_t hash_u64(std::uint64_t seed,
                                                    std::uint64_t domain,
                                                    std::uint64_t index) {
    return splitmix64(seed ^ (domain + index));
}

// Uniform in [0, 1): top 24 bits of the hash / 2^24. Exact in FP32.
BROTENSOR_HASH_RNG_HD inline float hash_uniform(std::uint64_t seed,
                                                std::uint64_t domain,
                                                std::uint64_t index) {
    return static_cast<float>(hash_u64(seed, domain, index) >> 40) *
           (1.0f / 16777216.0f);
}

}  // namespace brotensor::detail

#undef BROTENSOR_HASH_RNG_HD
