#pragma once

#if defined(__has_include)
#if __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
#include <type_traits>

namespace nvcuda::wmma {

using ::rocwmma::matrix_a;
using ::rocwmma::matrix_b;
using ::rocwmma::accumulator;
using ::rocwmma::row_major;
using ::rocwmma::col_major;
using ::rocwmma::mem_row_major;
using ::rocwmma::mem_col_major;
using ::rocwmma::layout_t;

template <typename T>
struct to_rocwmma { using type = T; };

template <>
struct to_rocwmma<__hip_bfloat16> { using type = hip_bfloat16; };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
using fragment = ::rocwmma::fragment<Use, M, N, K, typename to_rocwmma<T>::type, Layout>;

template <typename Frag, typename T>
__device__ inline void fill_fragment(Frag& frag, T val) {
    ::rocwmma::fill_fragment(frag, val);
}

template <typename Frag, typename T>
__device__ inline void load_matrix_sync(Frag& frag, const T* p, unsigned ldm) {
    if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        ::rocwmma::load_matrix_sync(frag, reinterpret_cast<const hip_bfloat16*>(p), ldm);
    } else {
        ::rocwmma::load_matrix_sync(frag, p, ldm);
    }
}

template <typename Frag, typename T, typename Layout>
__device__ inline void load_matrix_sync(Frag& frag, const T* p, unsigned ldm, Layout layout) {
    if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        ::rocwmma::load_matrix_sync(frag, reinterpret_cast<const hip_bfloat16*>(p), ldm, layout);
    } else {
        ::rocwmma::load_matrix_sync(frag, p, ldm, layout);
    }
}

template <typename DFrag, typename AFrag, typename BFrag, typename CFrag>
__device__ inline void mma_sync(DFrag& d, const AFrag& a, const BFrag& b, const CFrag& c) {
    ::rocwmma::mma_sync(d, a, b, c);
}

template <typename T, typename Frag, typename Layout>
__device__ inline void store_matrix_sync(T* p, const Frag& frag, unsigned ldm, Layout layout) {
    if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        ::rocwmma::store_matrix_sync(reinterpret_cast<hip_bfloat16*>(p), frag, ldm, layout);
    } else {
        ::rocwmma::store_matrix_sync(p, frag, ldm, layout);
    }
}

} // namespace nvcuda::wmma

#endif
#endif
