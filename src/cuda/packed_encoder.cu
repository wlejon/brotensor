// CUDA packed-encoder helpers: per-row-position RoPE over a fused QKV buffer
// and per-segment softmax statistics. CPU twins in src/cpu/packed_encoder.cpp.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

// flash_attention_packed.cu
void flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV,
                                        const ::brotensor::Tensor& seq_bounds,
                                        int num_heads, int window,
                                        ::brotensor::Tensor& O);
// flash_attention_packed_backward.cu
void flash_attention_packed_qkv_backward(const ::brotensor::Tensor& QKV, const ::brotensor::Tensor& dO,
                                         const ::brotensor::Tensor& seq_bounds, int num_heads, int window,
                                         ::brotensor::Tensor& dQKV);

namespace {

cudaStream_t cur_stream() { return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()); }

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

template <typename T> __device__ __forceinline__ float ld_f32(const T* p);
template <> __device__ __forceinline__ float ld_f32<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ __forceinline__ float ld_f32<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// One thread per rotated pair of one row; blockIdx.y = row. pairs = H*hd
// (H*hd/2 pairs in Q plus as many in K).
template <typename T>
__global__ void rope_qkv_packed_kernel(T* __restrict__ qkv, const float* __restrict__ cos_tbl,
                                       const float* __restrict__ sin_tbl,
                                       const int* __restrict__ pos, int D, int half) {
    const int r = blockIdx.y;
    const int pi = blockIdx.x * blockDim.x + threadIdx.x;
    if (pi >= D) return;  // D pairs per row across the Q and K sections
    const int sec = pi / (D / 2);
    const int pj = pi - sec * (D / 2);
    const int head = pj / half;
    const int i = pj - head * half;
    T* v = qkv + static_cast<size_t>(r) * 3 * D + sec * D + head * 2 * half + 2 * i;
    const size_t t = static_cast<size_t>(pos[r]) * half + i;
    const float c = cos_tbl[t], s = sin_tbl[t];
    const float x0 = ld_f32(v), x1 = ld_f32(v + 1);
    v[0] = from_f32<T>(x0 * c - x1 * s);
    v[1] = from_f32<T>(x0 * s + x1 * c);
}

// One warp per segment.
template <typename T>
__global__ void segment_softmax_stats_kernel(const T* __restrict__ logits,
                                             const int* __restrict__ off, int S,
                                             T* __restrict__ out) {
    const int s = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (s >= S) return;
    const int a = off[s], b = off[s + 1];
    T* row = out + static_cast<size_t>(s) * 4;
    if (b <= a) {
        if (lane < 4) row[lane] = from_f32<T>(0.0f);
        return;
    }
    float mx = __int_as_float(0xff800000);  // -inf
    for (int i = a + lane; i < b; i += 32) mx = fmaxf(mx, ld_f32(logits + i));
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.0f;
    for (int i = a + lane; i < b; i += 32) sum += expf(ld_f32(logits + i) - mx);
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    float ent = 0.0f, t1 = 0.0f, t2 = 0.0f;
    for (int i = a + lane; i < b; i += 32) {
        const float p = expf(ld_f32(logits + i) - mx) / sum;
        ent -= p * logf(fmaxf(p, 1e-9f));
        if (p > t1) {
            t2 = t1;
            t1 = p;
        } else if (p > t2) {
            t2 = p;
        }
    }
    for (int o = 16; o; o >>= 1) {
        ent += __shfl_xor_sync(0xffffffffu, ent, o);
        const float o1 = __shfl_xor_sync(0xffffffffu, t1, o);
        const float o2 = __shfl_xor_sync(0xffffffffu, t2, o);
        // merge two (top1, top2) pairs
        const float hi = fmaxf(t1, o1);
        const float lo = fmaxf(fminf(t1, o1), fmaxf(t2, o2));
        t1 = hi;
        t2 = lo;
    }
    if (lane == 0) {
        const float k = fmaxf(2.0f, static_cast<float>(b - a));
        row[0] = from_f32<T>(t1);
        row[1] = from_f32<T>(t1 - t2);
        row[2] = from_f32<T>(ent / logf(k));
        row[3] = from_f32<T>(k / 255.0f);
    }
}

void rope_qkv_packed_inplace(::brotensor::Tensor& QKV, const ::brotensor::Tensor& cos_tbl,
                             const ::brotensor::Tensor& sin_tbl, const ::brotensor::Tensor& pos,
                             int num_heads, int head_dim) {
    constexpr const char* op = "rope_qkv_packed_inplace";
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) fail(op, "cos/sin tables must be FP32");
    if (pos.dtype != Dtype::INT32 || pos.rows != QKV.rows) fail(op, "pos must be (L, 1) INT32");
    if (head_dim <= 0 || (head_dim & 1) || num_heads <= 0 || QKV.cols != 3 * num_heads * head_dim) {
        fail(op, "QKV.cols must be 3 * num_heads * head_dim with head_dim even");
    }
    const int half = head_dim / 2;
    if (cos_tbl.cols != half || sin_tbl.cols != half) fail(op, "cos_tbl / sin_tbl must be (P, head_dim/2)");
    const int L = QKV.rows;
    const int D = num_heads * head_dim;
    if (L == 0) return;
    const dim3 grid((D + 255) / 256, L);
    const float* c = static_cast<const float*>(cos_tbl.data);
    const float* s = static_cast<const float*>(sin_tbl.data);
    const int* p = static_cast<const int*>(pos.data);
    switch (QKV.dtype) {
        case Dtype::FP16:
            rope_qkv_packed_kernel<__half><<<grid, 256, 0, cur_stream()>>>(
                static_cast<__half*>(QKV.data), c, s, p, D, half);
            break;
        case Dtype::BF16:
            rope_qkv_packed_kernel<__nv_bfloat16><<<grid, 256, 0, cur_stream()>>>(
                static_cast<__nv_bfloat16*>(QKV.data), c, s, p, D, half);
            break;
        case Dtype::FP32:
            rope_qkv_packed_kernel<float><<<grid, 256, 0, cur_stream()>>>(
                static_cast<float*>(QKV.data), c, s, p, D, half);
            break;
        default:
            fail(op, "QKV must be FP32/FP16/BF16");
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void segment_softmax_stats(const ::brotensor::Tensor& logits, const ::brotensor::Tensor& seg_offsets,
                           ::brotensor::Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) fail(op, "seg_offsets must be (S+1, 1) INT32");
    const int S = seg_offsets.rows - 1;
    const Dtype dt = logits.dtype;
    if (out.rows != S || out.cols != 4 || out.dtype != dt) out.resize(S, 4, dt);
    if (S == 0) return;
    const int warps = 4;
    const dim3 grid((S + warps - 1) / warps);
    const int* off = static_cast<const int*>(seg_offsets.data);
    switch (dt) {
        case Dtype::FP16:
            segment_softmax_stats_kernel<__half><<<grid, 32 * warps, 0, cur_stream()>>>(
                static_cast<const __half*>(logits.data), off, S, static_cast<__half*>(out.data));
            break;
        case Dtype::BF16:
            segment_softmax_stats_kernel<__nv_bfloat16><<<grid, 32 * warps, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(logits.data), off, S,
                static_cast<__nv_bfloat16*>(out.data));
            break;
        case Dtype::FP32:
            segment_softmax_stats_kernel<float><<<grid, 32 * warps, 0, cur_stream()>>>(
                static_cast<const float*>(logits.data), off, S, static_cast<float*>(out.data));
            break;
        default:
            fail(op, "logits must be FP32/FP16/BF16");
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void fill_cuda_vtable_packed_encoder(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_packed_qkv_forward = &flash_attention_packed_qkv_forward;
    v.flash_attention_packed_qkv_backward = &flash_attention_packed_qkv_backward;
    v.rope_qkv_packed_inplace = &rope_qkv_packed_inplace;
    v.segment_softmax_stats = &segment_softmax_stats;
}

}  // namespace brotensor::detail::cuda
