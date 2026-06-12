#pragma once

// Tiled FP16 matmul for C(M,N) = A(M,K) @ B(N,K)^T, FP16 storage / FP32 accum.
// Defined in src/metal/fp16_matmul.mm. Used by gemm.mm (Linear forward) and
// flash_attention.mm for QK^T / PV inner products.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace brotensor::metal_impl {

void launch_matmul_abt_fp16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K);

} // namespace brotensor::metal_impl
