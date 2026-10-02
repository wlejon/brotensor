#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// The kernels assume Wave32, RDNA's 64 KB workgroup LDS and WMMA matrix
// cores: RDNA 3+ only (cmake/BrotensorHip.cmake rejects other targets; this
// catches a build that bypassed it).
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX8__) || defined(__GFX9__) || defined(__GFX10__))
#error "brotensor HIP kernels need an RDNA 3+ (gfx11xx / gfx12xx) target"
#endif

// ─── CUDA Runtime Types & Constants Compatibility ──────────────────────────

using cudaStream_t    = hipStream_t;
using cudaEvent_t     = hipEvent_t;
using cudaError_t     = hipError_t;
using cudaDeviceProp  = hipDeviceProp_t;
using cudaMemPool_t   = hipMemPool_t;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaGraph_t     = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;

#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphInstantiate(pExec, graph, flags) hipGraphInstantiateWithFlags(pExec, graph, flags)

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
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxBlocksPerMultiprocessor hipDeviceAttributeMaxBlocksPerMultiProcessor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaStreamDefault hipStreamDefault
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
template <typename T>
inline hipError_t cudaFuncSetAttribute(T* func, hipFuncAttribute attr, int value) {
    (void)func; (void)attr; (void)value;
    return hipSuccess;
}

// ─── Stream & Memory Runtime Bridge ───────────────────────────────────────

namespace brotensor {
void* cuda_current_stream();
void* cuda_current_stream(int dev);
void cuda_set_stream(void* stream);
void cuda_set_stream(void* stream, int dev);
}

namespace brotensor::detail::cuda {
void* cuda_current_stream();
void* cuda_current_stream(int dev);
void cuda_set_stream(void* stream);
void cuda_set_stream(void* stream, int dev);
void* cuda_alloc(std::size_t bytes);
void cuda_free(void* ptr);
void cuda_check_throw(int err, const char* expr_text, const char* file, int line);
}

// ─── Bfloat16 Helpers ──────────────────────────────────────────────────────

using nv_bfloat16   = __hip_bfloat16;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

inline __host__ __device__ float __nv_bfloat162float(__hip_bfloat16 bf) {
    return __bfloat162float(bf);
}

inline __host__ __device__ __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

inline __host__ __device__ __hip_bfloat16 __float2bfloat16_rz(float f) {
    return __float2bfloat16(f);
}

inline __host__ __device__ __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __float22bfloat162_rn(float2{a, b});
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

template <typename T>
__device__ inline unsigned int __match_any_sync(uint32_t mask, T value) {
    return __match_any_sync(static_cast<unsigned long long>(mask), value);
}

template <typename T>
__device__ inline unsigned int __match_any_sync(int mask, T value) {
    return __match_any_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), value);
}

template <typename T>
__device__ inline unsigned int __match_all_sync(uint32_t mask, T value, int* pred) {
    return __match_all_sync(static_cast<unsigned long long>(mask), value, pred);
}

template <typename T>
__device__ inline unsigned int __match_all_sync(int mask, T value, int* pred) {
    return __match_all_sync(static_cast<unsigned long long>(static_cast<uint32_t>(mask)), value, pred);
}

#endif // __HIPCC__ || __HIP_DEVICE_COMPILE__
