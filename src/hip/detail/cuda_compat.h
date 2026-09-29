#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

// ─── CUDA Runtime Types & Constants Compatibility ──────────────────────────

using cudaStream_t    = hipStream_t;
using cudaEvent_t     = hipEvent_t;
using cudaError_t     = hipError_t;
using cudaDeviceProp  = hipDeviceProp_t;
using cudaMemPool_t   = hipMemPool_t;

#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize

#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync

// ─── Bfloat16 Helpers ──────────────────────────────────────────────────────

using nv_bfloat16   = hip_bfloat16;
using __nv_bfloat16 = hip_bfloat16;

inline __host__ __device__ hip_bfloat16 __float2bfloat16(float f) {
    return hip_bfloat16(f);
}

inline __host__ __device__ float __bfloat162float(hip_bfloat16 bf) {
    return float(bf);
}

inline __host__ __device__ float __nv_bfloat162float(hip_bfloat16 bf) {
    return float(bf);
}

inline __host__ __device__ hip_bfloat16 __float2bfloat16_rn(float f) {
    return hip_bfloat16::round_to_bfloat16(f);
}

inline __host__ __device__ hip_bfloat16 __float2bfloat16_rz(float f) {
    return hip_bfloat16(f, hip_bfloat16::truncate);
}

// ─── Warp Primitives with 64-bit Mask Handling ─────────────────────────────
// On AMD Wave32 / Wave64, HIP's template warp sync functions static_assert
// that sizeof(MaskT) == 8. In CUDA, masks are typically 32-bit (e.g. 0xffffffff).
// Providing exact non-template overloads for 32-bit integer types ensures
// promotion to unsigned long long (64-bit) without compilation failure.

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)

__device__ inline void __syncwarp(uint32_t mask) {
    __syncwarp(static_cast<unsigned long long>(mask));
}

__device__ inline void __syncwarp(int mask) {
    __syncwarp(static_cast<unsigned long long>(static_cast<uint32_t>(mask)));
}

template <typename T>
__device__ inline T __shfl_sync(uint32_t mask, T var, int srcLane, int width = 32) {
    return __shfl_sync(static_cast<unsigned long long>(mask), var, srcLane, width);
}

template <typename T>
__device__ inline T __shfl_sync(int mask, T var, int srcLane, int width = 32) {
    return __shfl_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), var, srcLane, width);
}

template <typename T>
__device__ inline T __shfl_down_sync(uint32_t mask, T var, unsigned int delta, int width = 32) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), var, delta, width);
}

template <typename T>
__device__ inline T __shfl_down_sync(int mask, T var, unsigned int delta, int width = 32) {
    return __shfl_down_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), var, delta, width);
}

template <typename T>
__device__ inline T __shfl_up_sync(uint32_t mask, T var, unsigned int delta, int width = 32) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), var, delta, width);
}

template <typename T>
__device__ inline T __shfl_up_sync(int mask, T var, unsigned int delta, int width = 32) {
    return __shfl_up_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), var, delta, width);
}

template <typename T>
__device__ inline T __shfl_xor_sync(uint32_t mask, T var, int laneMask, int width = 32) {
    return __shfl_xor_sync(static_cast<unsigned long long>(mask), var, laneMask, width);
}

template <typename T>
__device__ inline T __shfl_xor_sync(int mask, T var, int laneMask, int width = 32) {
    return __shfl_xor_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), var, laneMask, width);
}

__device__ inline unsigned int __ballot_sync(uint32_t mask, int predicate) {
    return static_cast<unsigned int>(__ballot_sync(static_cast<unsigned long long>(mask), predicate));
}

__device__ inline unsigned int __ballot_sync(int mask, int predicate) {
    return static_cast<unsigned int>(__ballot_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), predicate));
}

__device__ inline int __all_sync(uint32_t mask, int predicate) {
    return __all_sync(static_cast<unsigned long long>(mask), predicate);
}

__device__ inline int __all_sync(int mask, int predicate) {
    return __all_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), predicate);
}

__device__ inline int __any_sync(uint32_t mask, int predicate) {
    return __any_sync(static_cast<unsigned long long>(mask), predicate);
}

__device__ inline int __any_sync(int mask, int predicate) {
    return __any_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), predicate);
}

#endif // __HIPCC__ || __HIP_DEVICE_COMPILE__

// ─── rocWMMA Mapping (CUDA wmma -> rocwmma) ────────────────────────────────

#if defined(__has_include)
#if __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
namespace nvcuda {
namespace wmma {
    using namespace ::rocwmma;
}
}
#endif
#endif
