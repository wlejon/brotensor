#pragma once

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace brotensor::detail::hip {

// Helper: load 4 bytes aligned to 2 bytes (for odd 2-byte aligned blocks like Q6_K 210B and Q8_0 34B)
__device__ __forceinline__ uint32_t load_u32_align2(const uint8_t* p) {
    const uint32_t lo = *reinterpret_cast<const uint16_t*>(p);
    const uint32_t hi = *reinterpret_cast<const uint16_t*>(p + 2);
    return lo | (hi << 16);
}

// ─── Q4_K layout & decode ──────────────────────────────────────────────────
namespace q4k {

constexpr int kBlockBytes    = 144;
constexpr int kBlockElems    = 256;
constexpr int kSubBlockElems = 32;
constexpr int kSubBlocks     = 8;   // kBlockElems / kSubBlockElems
constexpr int kQsBytes       = 128;
constexpr int kDOffset       = 0;
constexpr int kDminOffset    = 2;
constexpr int kScalesOffset  = 4;
constexpr int kQsOffset      = 16;

__device__ __forceinline__
void unpack_sc_m(int j, const uint8_t* scales, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = scales[j]     & 0x3Fu;
        m  = scales[j + 4] & 0x3Fu;
    } else {
        sc = (scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4);
        m  = (scales[j + 4] >> 4)    | ((scales[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ uint32_t scale_byte(const uint4& h, int s) {
    const uint32_t b = 4u + static_cast<uint32_t>(s);
    const uint32_t w = (b < 8u) ? h.y : ((b < 12u) ? h.z : h.w);
    return (w >> ((b & 3u) * 8u)) & 0xFFu;
}

__device__ __forceinline__
void unpack_sc_m(int j, const uint4& h, uint32_t& sc, uint32_t& m) {
    if (j < 4) {
        sc = scale_byte(h, j)     & 0x3Fu;
        m  = scale_byte(h, j + 4) & 0x3Fu;
    } else {
        sc = (scale_byte(h, j + 4) & 0x0Fu) | ((scale_byte(h, j - 4) >> 6) << 4);
        m  = (scale_byte(h, j + 4) >> 4)    | ((scale_byte(h, j)     >> 6) << 4);
    }
}

} // namespace q4k

// ─── Q8_0 layout ───────────────────────────────────────────────────────────
namespace q8_0 {

constexpr int kBlockBytes = 34;
constexpr int kBlockElems = 32;
constexpr int kDOffset    = 0;
constexpr int kQsOffset   = 2;

} // namespace q8_0

// ─── Q6_K layout & decode ──────────────────────────────────────────────────
namespace q6k {

constexpr int kBlockBytes    = 210;
constexpr int kBlockElems    = 256;
constexpr int kSubBlockElems = 16;
constexpr int kSubBlocks     = 16;
constexpr int kQlOffset      = 0;
constexpr int kQhOffset      = 128;
constexpr int kScalesOffset  = 192;
constexpr int kDOffset       = 208;

__device__ __forceinline__
void decode_element(int e, const uint8_t* ql, const uint8_t* qh,
                    int& sb_out, int& val6_out) {
    const int group = e >> 7;             // 0..1
    const int local = e - (group << 7);   // 0..127
    const int quad  = local >> 5;         // 0..3
    const int l     = local - (quad << 5); // 0..31

    const int sb = (group << 3) + (quad << 1) + (l >> 4);   // 0..15

    const uint8_t ql_b = ql[group * 64 + (quad & 1) * 32 + l];
    const uint8_t qh_b = qh[group * 32 + l];
    const int raw4  = (quad < 2) ? (ql_b & 0x0F) : (ql_b >> 4);
    const int high2 = (qh_b >> (quad * 2)) & 0x03;
    const int val6  = static_cast<int>(raw4 | (high2 << 4)) - 32;

    sb_out   = sb;
    val6_out = val6;
}

struct QuadDesc {
    int ql_off;
    int qh_off;
    int sb;
    int qh_shift;
    bool low_nib;
};

__device__ __forceinline__ QuadDesc quad_desc(int e0) {
    const int group = e0 >> 7;
    const int local = e0 - (group << 7);
    const int quad  = local >> 5;
    const int l     = local - (quad << 5);
    QuadDesc q;
    q.ql_off   = group * 64 + (quad & 1) * 32 + l;
    q.qh_off   = group * 32 + l;
    q.sb       = (group << 3) + (quad << 1) + (l >> 4);
    q.qh_shift = quad * 2;
    q.low_nib  = quad < 2;
    return q;
}

__device__ __forceinline__ int decode_packed(const QuadDesc& q, uint32_t ql4,
                                             uint32_t qh4, int c) {
    const uint32_t qlb  = (ql4 >> (c * 8)) & 0xFFu;
    const uint32_t qhb  = (qh4 >> (c * 8)) & 0xFFu;
    const uint32_t raw4 = q.low_nib ? (qlb & 0x0Fu) : (qlb >> 4);
    const uint32_t hi2  = (qhb >> q.qh_shift) & 0x03u;
    return static_cast<int>(raw4 | (hi2 << 4)) - 32;
}

} // namespace q6k

// ─── Function declarations ─────────────────────────────────────────────────

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16);
void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y);
void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD);

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16);
void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y);
void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD);

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16);
void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y);
void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD);

void matmul_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                       const Tensor& X, Tensor& Y);
void linear_forward_batched_int8w_fp16(const Tensor& W_int8, const Tensor& scales,
                                       const Tensor* bias, const Tensor& X_BD, Tensor& Y_BD);
void conv2d_int8w_fp16_forward(const Tensor& X, const Tensor& W_int8, const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W, int C_out, int kH, int kW,
                               int stride_h, int stride_w, int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups, Tensor& Y);

void fill_hip_vtable_quant(OpsVTable& v);

} // namespace brotensor::detail::hip
