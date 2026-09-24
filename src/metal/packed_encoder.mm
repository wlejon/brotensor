// Metal packed-encoder ops — the Metal twins of src/cuda/flash_attention_packed.cu,
// src/cuda/flash_attention_packed_backward.cu and src/cuda/packed_encoder.cu
// (CPU oracle: src/cpu/packed_encoder.cpp):
//
//   flash_attention_packed_qkv_forward  — bidirectional (optionally windowed)
//       attention off a fused (L, 3*H*hd) QKV with per-row sequence bounds.
//   flash_attention_packed_qkv_backward — its adjoint, recompute-based.
//   rope_qkv_packed_inplace             — RoPE on the Q/K sections, per-row pos.
//   segment_softmax_stats               — [top1, top1-top2, norm. entropy, k/255].
//
// FP16 / BF16 / FP32 storage, FP32 math throughout; each output rounds once.
//
// Attention: one simdgroup per (row, head). Keys are taken 32 at a time, one
// per lane (full q.k dot per lane against q staged in threadgroup memory),
// folded into an online softmax; the P @ V accumulation broadcasts each key's
// probability and every lane owns a strided slice of head_dim (<= 256).
//
// Backward, as on CUDA: pass 1 stores each (row, head)'s softmax max, sum and
// D = dO . O; pass 2 has every (row, head) act once as a query (dQ) and once
// as a key (dK, dV). The window band |r - j| <= window/2 inside one sequence
// is symmetric, so key r's queries are exactly query r's keys and pass 2 walks
// one range for both roles — deterministic, no atomics.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kPackedSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct PackedParams {
    uint  L, H, hd, D;
    int   hw;        // half window, or -1 for the whole sequence
    float scale;
};

constant constexpr uint PMAX = 8;   // head_dim <= 32 * PMAX
constant constexpr float kNeg = -1e30f;

// Row r's key range [lo, hi): its sequence, clipped to the window band.
inline void row_range(device const int* b, uint r, uint L, int hw, thread int& lo, thread int& hi) {
    const int s = max(b[2 * r], 0), e = min(b[2 * r + 1], int(L));
    lo = hw >= 0 ? max(s, int(r) - hw) : s;
    hi = hw >= 0 ? min(e, int(r) + hw + 1) : e;
}

// Online-softmax attention of query row r (q staged in qs) over [lo, hi).
// Returns the row's max / sum and leaves the unnormalised output in o.
template <typename T>
inline void attend_row(device const T* qkv, threadgroup const float* qs, uint h, int lo, int hi,
                       constant PackedParams& p, uint lane, thread float* o, thread float& m, thread float& l) {
    const ulong row3 = 3ul * p.D;
    m = kNeg;
    l = 0.0f;
    for (uint i = 0; i < PMAX; ++i) o[i] = 0.0f;
    for (int j0 = lo; j0 < hi; j0 += 32) {
        const int j = j0 + int(lane);
        float s = kNeg;
        if (j < hi) {
            device const T* k = qkv + ulong(j) * row3 + p.D + h * p.hd;
            float dot = 0.0f;
            for (uint d = 0; d < p.hd; ++d) dot += qs[d] * float(k[d]);
            s = dot * p.scale;
        }
        const float mn = max(m, simd_max(s));
        const float corr = exp(m - mn);
        const float e = j < hi ? exp(s - mn) : 0.0f;
        l = l * corr + simd_sum(e);
        for (uint i = 0; i < PMAX; ++i) o[i] *= corr;
        const int cnt = min(32, hi - j0);
        for (int t = 0; t < cnt; ++t) {
            const float pt = simd_shuffle(e, ushort(t));
            device const T* v = qkv + ulong(j0 + t) * row3 + 2 * p.D + h * p.hd;
            for (uint i = 0; i < PMAX; ++i) {
                const uint d = lane + 32 * i;
                if (d < p.hd) o[i] += pt * float(v[d]);
            }
        }
        m = mn;
    }
}

template <typename T>
kernel void k_packed_fwd(device const T*   qkv    [[buffer(0)]],
                         device const int* bounds [[buffer(1)]],
                         device T*         O      [[buffer(2)]],
                         constant PackedParams& p [[buffer(3)]],
                         uint tg   [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float qs[32 * PMAX];
    const uint r = tg / p.H, h = tg % p.H;
    device const T* q = qkv + ulong(r) * 3 * p.D + h * p.hd;
    for (uint d = lane; d < p.hd; d += 32) qs[d] = float(q[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int lo, hi;
    row_range(bounds, r, p.L, p.hw, lo, hi);
    float o[PMAX], m, l;
    attend_row(qkv, qs, h, lo, hi, p, lane, o, m, l);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device T* out = O + ulong(r) * p.D + h * p.hd;
    for (uint i = 0; i < PMAX; ++i) {
        const uint d = lane + 32 * i;
        if (d < p.hd) out[d] = T(o[i] * inv);
    }
}

// Pass 1 of the backward: stats[(r*H + h)*3 + {0,1,2}] = {max, sum, dO . O}.
template <typename T>
kernel void k_packed_bwd_stats(device const T*   qkv    [[buffer(0)]],
                               device const int* bounds [[buffer(1)]],
                               device const T*   dO     [[buffer(2)]],
                               device float*     stats  [[buffer(3)]],
                               constant PackedParams& p [[buffer(4)]],
                               uint tg   [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float qs[32 * PMAX];
    const uint r = tg / p.H, h = tg % p.H;
    device const T* q = qkv + ulong(r) * 3 * p.D + h * p.hd;
    for (uint d = lane; d < p.hd; d += 32) qs[d] = float(q[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int lo, hi;
    row_range(bounds, r, p.L, p.hw, lo, hi);
    float o[PMAX], m, l;
    attend_row(qkv, qs, h, lo, hi, p, lane, o, m, l);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device const T* g = dO + ulong(r) * p.D + h * p.hd;
    float dsum = 0.0f;
    for (uint i = 0; i < PMAX; ++i) {
        const uint d = lane + 32 * i;
        if (d < p.hd) dsum += float(g[d]) * o[i] * inv;
    }
    dsum = simd_sum(dsum);
    if (lane == 0) {
        device float* st = stats + (ulong(r) * p.H + h) * 3;
        st[0] = m;
        st[1] = l;
        st[2] = dsum;
    }
}

// Pass 2: dQ of row r as a query, dK / dV of row r as a key.
template <typename T>
kernel void k_packed_bwd_grads(device const T*     qkv    [[buffer(0)]],
                               device const int*   bounds [[buffer(1)]],
                               device const T*     dO     [[buffer(2)]],
                               device const float* stats  [[buffer(3)]],
                               device T*           dQKV   [[buffer(4)]],
                               constant PackedParams& p   [[buffer(5)]],
                               uint tg   [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float qr[32 * PMAX], kr[32 * PMAX], vr[32 * PMAX], gr[32 * PMAX];
    const uint r = tg / p.H, h = tg % p.H;
    const ulong row3 = 3ul * p.D;
    device const T* base = qkv + ulong(r) * row3 + h * p.hd;
    device const T* gro = dO + ulong(r) * p.D + h * p.hd;
    for (uint d = lane; d < p.hd; d += 32) {
        qr[d] = float(base[d]);
        kr[d] = float(base[p.D + d]);
        vr[d] = float(base[2 * p.D + d]);
        gr[d] = float(gro[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* st_r = stats + (ulong(r) * p.H + h) * 3;
    const float m_r = st_r[0], il_r = st_r[1] > 0.0f ? 1.0f / st_r[1] : 0.0f, D_r = st_r[2];

    int lo, hi;
    row_range(bounds, r, p.L, p.hw, lo, hi);
    float dq[PMAX], dk[PMAX], dv[PMAX];
    for (uint i = 0; i < PMAX; ++i) dq[i] = dk[i] = dv[i] = 0.0f;
    for (int j0 = lo; j0 < hi; j0 += 32) {
        const int j = j0 + int(lane);
        float ds_rj = 0.0f, ds_jr = 0.0f, p_jr = 0.0f;
        if (j < hi) {
            device const T* rowj = qkv + ulong(j) * row3 + h * p.hd;
            device const T* gj = dO + ulong(j) * p.D + h * p.hd;
            float s_rj = 0.0f, dp_rj = 0.0f, s_jr = 0.0f, dp_jr = 0.0f;
            for (uint d = 0; d < p.hd; ++d) {
                s_rj += qr[d] * float(rowj[p.D + d]);        // q_r . k_j
                dp_rj += gr[d] * float(rowj[2 * p.D + d]);   // dO_r . v_j
                s_jr += float(rowj[d]) * kr[d];              // q_j . k_r
                dp_jr += float(gj[d]) * vr[d];               // dO_j . v_r
            }
            const float p_rj = exp(s_rj * p.scale - m_r) * il_r;
            ds_rj = p_rj * (dp_rj - D_r) * p.scale;
            device const float* st_j = stats + (ulong(j) * p.H + h) * 3;
            const float il_j = st_j[1] > 0.0f ? 1.0f / st_j[1] : 0.0f;
            p_jr = exp(s_jr * p.scale - st_j[0]) * il_j;
            ds_jr = p_jr * (dp_jr - st_j[2]) * p.scale;
        }
        const int cnt = min(32, hi - j0);
        for (int t = 0; t < cnt; ++t) {
            const float a = simd_shuffle(ds_rj, ushort(t));
            const float b = simd_shuffle(ds_jr, ushort(t));
            const float c = simd_shuffle(p_jr, ushort(t));
            device const T* rowj = qkv + ulong(j0 + t) * row3 + h * p.hd;
            device const T* gj = dO + ulong(j0 + t) * p.D + h * p.hd;
            for (uint i = 0; i < PMAX; ++i) {
                const uint d = lane + 32 * i;
                if (d < p.hd) {
                    dq[i] += a * float(rowj[p.D + d]);   // dS_rj k_j
                    dk[i] += b * float(rowj[d]);         // dS_jr q_j
                    dv[i] += c * float(gj[d]);           // P_jr dO_j
                }
            }
        }
    }
    device T* out = dQKV + ulong(r) * row3 + h * p.hd;
    for (uint i = 0; i < PMAX; ++i) {
        const uint d = lane + 32 * i;
        if (d < p.hd) {
            out[d] = T(dq[i]);
            out[p.D + d] = T(dk[i]);
            out[2 * p.D + d] = T(dv[i]);
        }
    }
}

// One thread per rotated pair of one row (gid.y): D pairs across Q and K.
template <typename T>
kernel void k_rope_qkv_packed(device T*           qkv [[buffer(0)]],
                              device const float* ct  [[buffer(1)]],
                              device const float* stb [[buffer(2)]],
                              device const int*   pos [[buffer(3)]],
                              constant uint& D        [[buffer(4)]],
                              constant uint& half_hd  [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint pi = gid.x, r = gid.y;
    if (pi >= D) return;
    const uint sec = pi / (D / 2);
    const uint pj = pi - sec * (D / 2);
    const uint head = pj / half_hd, i = pj - head * half_hd;
    device T* v = qkv + ulong(r) * 3 * D + sec * D + head * 2 * half_hd + 2 * i;
    const ulong t = ulong(pos[r]) * half_hd + i;
    const float c = ct[t], s = stb[t];
    const float x0 = float(v[0]), x1 = float(v[1]);
    v[0] = T(x0 * c - x1 * s);
    v[1] = T(x0 * s + x1 * c);
}

// One simdgroup per segment.
template <typename T>
kernel void k_segment_softmax_stats(device const T*   logits [[buffer(0)]],
                                    device const int* off    [[buffer(1)]],
                                    device T*         out    [[buffer(2)]],
                                    constant uint& S         [[buffer(3)]],
                                    uint tg   [[threadgroup_position_in_grid]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint sgs  [[simdgroups_per_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint s = tg * sgs + sg;
    if (s >= S) return;
    const int a = off[s], b = off[s + 1];
    device T* row = out + ulong(s) * 4;
    if (b <= a) {
        if (lane < 4) row[lane] = T(0.0f);
        return;
    }
    float mx = kNeg;
    for (int i = a + int(lane); i < b; i += 32) mx = max(mx, float(logits[i]));
    mx = simd_max(mx);
    float sum = 0.0f;
    for (int i = a + int(lane); i < b; i += 32) sum += exp(float(logits[i]) - mx);
    sum = simd_sum(sum);
    float ent = 0.0f, t1 = 0.0f, t2 = 0.0f;
    for (int i = a + int(lane); i < b; i += 32) {
        const float pv = exp(float(logits[i]) - mx) / sum;
        ent -= pv * log(max(pv, 1e-9f));
        if (pv > t1) {
            t2 = t1;
            t1 = pv;
        } else if (pv > t2) {
            t2 = pv;
        }
    }
    ent = simd_sum(ent);
    for (ushort o = 16; o > 0; o >>= 1) {   // merge (top1, top2) pairs
        const float o1 = simd_shuffle_xor(t1, o), o2 = simd_shuffle_xor(t2, o);
        const float hi = max(t1, o1);
        const float lo = max(min(t1, o1), max(t2, o2));
        t1 = hi;
        t2 = lo;
    }
    if (lane == 0) {
        const float k = max(2.0f, float(b - a));
        row[0] = T(t1);
        row[1] = T(t1 - t2);
        row[2] = T(ent / log(k));
        row[3] = T(k / 255.0f);
    }
}

#define PACKED_INST(T, SUF)                                                                              \
template [[host_name("k_packed_fwd_" SUF)]] kernel void k_packed_fwd<T>(                                 \
    device const T*, device const int*, device T*, constant PackedParams&, uint, uint);                  \
template [[host_name("k_packed_bwd_stats_" SUF)]] kernel void k_packed_bwd_stats<T>(                     \
    device const T*, device const int*, device const T*, device float*, constant PackedParams&, uint, uint); \
template [[host_name("k_packed_bwd_grads_" SUF)]] kernel void k_packed_bwd_grads<T>(                     \
    device const T*, device const int*, device const T*, device const float*, device T*,                 \
    constant PackedParams&, uint, uint);                                                                 \
template [[host_name("k_rope_qkv_packed_" SUF)]] kernel void k_rope_qkv_packed<T>(                       \
    device T*, device const float*, device const float*, device const int*, constant uint&,             \
    constant uint&, uint2);                                                                              \
template [[host_name("k_segment_softmax_stats_" SUF)]] kernel void k_segment_softmax_stats<T>(           \
    device const T*, device const int*, device T*, constant uint&, uint, uint, uint, uint);
PACKED_INST(half, "f16")
PACKED_INST(bfloat, "bf16")
PACKED_INST(float, "f32")
)msl";

// Must match the MSL `PackedParams`.
struct PackedParams {
    uint32_t L, H, hd, D;
    int32_t hw;
    float scale;
};

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

const char* suffix(Dtype dt, const char* op) {
    switch (dt) {
        case Dtype::FP16: return "f16";
        case Dtype::BF16: return "bf16";
        case Dtype::FP32: return "f32";
        default: fail(op, "tensor must be FP32, FP16 or BF16");
    }
}

// Pipelines by kernel name; five kernels x three dtypes, compiled on first use.
id<MTLComputePipelineState> pso(const char* kernel, Dtype dt, const char* op) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cache = [NSMutableDictionary dictionary]; });
    NSString* name = [NSString stringWithFormat:@"%s_%s", kernel, suffix(dt, op)];
    @synchronized(cache) {
        id<MTLComputePipelineState> p = cache[name];
        if (!p) {
            p = compile_pipeline(kPackedSrc, name);
            cache[name] = p;
        }
        return p;
    }
}

constexpr int kMaxHeadDim = 256;   // 32 lanes * PMAX

struct PackedShape {
    int L, D, hd;
    PackedParams params;
};

PackedShape check_packed(const char* op, const Tensor& QKV, const Tensor& seq_bounds, int num_heads, int window) {
    if (seq_bounds.dtype != Dtype::INT32) fail(op, "seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail(op, "QKV.cols must be 3 * num_heads * head_dim");
    PackedShape s{};
    s.L = QKV.rows;
    s.D = QKV.cols / 3;
    s.hd = s.D / num_heads;
    if (seq_bounds.rows != s.L || seq_bounds.cols != 2) fail(op, "seq_bounds must be (L, 2)");
    if (s.hd > kMaxHeadDim) fail(op, "head_dim > 256 is not supported");
    s.params.L = static_cast<uint32_t>(s.L);
    s.params.H = static_cast<uint32_t>(num_heads);
    s.params.hd = static_cast<uint32_t>(s.hd);
    s.params.D = static_cast<uint32_t>(s.D);
    s.params.hw = window > 0 ? window / 2 : -1;
    s.params.scale = 1.0f / std::sqrt(static_cast<float>(s.hd));
    return s;
}

void bind(id<MTLComputeCommandEncoder> enc, const Tensor& t, NSUInteger index) {
    [enc setBuffer:buffer_for(t) offset:buffer_offset_for(t) atIndex:index];
}

}  // namespace

void flash_attention_packed_qkv_forward(const Tensor& QKV, const Tensor& seq_bounds, int num_heads, int window,
                                        Tensor& O) {
    constexpr const char* op = "flash_attention_packed_qkv_forward";
    const PackedShape s = check_packed(op, QKV, seq_bounds, num_heads, window);
    const Dtype dt = QKV.dtype;
    id<MTLComputePipelineState> p = pso("k_packed_fwd", dt, op);
    if (O.rows != s.L || O.cols != s.D || O.dtype != dt) O.resize(s.L, s.D, dt);
    if (s.L == 0 || s.D == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        bind(enc, QKV, 0);
        bind(enc, seq_bounds, 1);
        bind(enc, O, 2);
        [enc setBytes:&s.params length:sizeof(PackedParams) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(s.L) * num_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void flash_attention_packed_qkv_backward(const Tensor& QKV, const Tensor& dO, const Tensor& seq_bounds,
                                         int num_heads, int window, Tensor& dQKV) {
    constexpr const char* op = "flash_attention_packed_qkv_backward";
    const PackedShape s = check_packed(op, QKV, seq_bounds, num_heads, window);
    const Dtype dt = QKV.dtype;
    if (dO.dtype != dt) fail(op, "dO must share QKV's dtype");
    if (dO.rows != s.L || dO.cols != s.D) fail(op, "dO must be (L, num_heads*head_dim)");
    id<MTLComputePipelineState> p_stats = pso("k_packed_bwd_stats", dt, op);
    id<MTLComputePipelineState> p_grads = pso("k_packed_bwd_grads", dt, op);
    if (dQKV.rows != s.L || dQKV.cols != 3 * s.D || dQKV.dtype != dt) dQKV.resize(s.L, 3 * s.D, dt);
    if (s.L == 0 || s.D == 0) return;
    // 3 floats per (row, head); kept across calls to hold allocator pressure flat.
    thread_local static Tensor stats = Tensor::empty_on(Device::Metal, 0, 0);
    if (stats.rows != s.L * num_heads || stats.cols != 3) stats.resize(s.L * num_heads, 3, Dtype::FP32);
    const NSUInteger groups = static_cast<NSUInteger>(s.L) * num_heads;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p_stats];
        bind(enc, QKV, 0);
        bind(enc, seq_bounds, 1);
        bind(enc, dO, 2);
        bind(enc, stats, 3);
        [enc setBytes:&s.params length:sizeof(PackedParams) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:p_grads];
        bind(enc, QKV, 0);
        bind(enc, seq_bounds, 1);
        bind(enc, dO, 2);
        bind(enc, stats, 3);
        bind(enc, dQKV, 4);
        [enc setBytes:&s.params length:sizeof(PackedParams) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void rope_qkv_packed_inplace(Tensor& QKV, const Tensor& cos_tbl, const Tensor& sin_tbl, const Tensor& pos,
                             int num_heads, int head_dim) {
    constexpr const char* op = "rope_qkv_packed_inplace";
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) fail(op, "cos/sin tables must be FP32");
    if (pos.dtype != Dtype::INT32 || pos.rows != QKV.rows) fail(op, "pos must be (L, 1) INT32");
    if (head_dim <= 0 || (head_dim & 1) || num_heads <= 0 || QKV.cols != 3 * num_heads * head_dim) {
        fail(op, "QKV.cols must be 3 * num_heads * head_dim with head_dim even");
    }
    const uint32_t half_hd = static_cast<uint32_t>(head_dim / 2);
    if (cos_tbl.cols != static_cast<int>(half_hd) || sin_tbl.cols != static_cast<int>(half_hd)) {
        fail(op, "cos_tbl / sin_tbl must be (P, head_dim/2)");
    }
    id<MTLComputePipelineState> p = pso("k_rope_qkv_packed", QKV.dtype, op);
    const int L = QKV.rows;
    const uint32_t D = static_cast<uint32_t>(num_heads * head_dim);
    if (L == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        bind(enc, QKV, 0);
        bind(enc, cos_tbl, 1);
        bind(enc, sin_tbl, 2);
        bind(enc, pos, 3);
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&half_hd length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D, static_cast<NSUInteger>(L), 1)
            threadsPerThreadgroup:MTLSizeMake(D < 256 ? D : 256, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets, Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) fail(op, "seg_offsets must be (S+1, 1) INT32");
    const uint32_t S = static_cast<uint32_t>(seg_offsets.rows - 1);
    const Dtype dt = logits.dtype;
    id<MTLComputePipelineState> p = pso("k_segment_softmax_stats", dt, op);
    if (out.rows != static_cast<int>(S) || out.cols != 4 || out.dtype != dt) out.resize(static_cast<int>(S), 4, dt);
    if (S == 0) return;
    constexpr NSUInteger kWarps = 4;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        bind(enc, logits, 0);
        bind(enc, seg_offsets, 1);
        bind(enc, out, 2);
        [enc setBytes:&S length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((S + kWarps - 1) / kWarps, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * kWarps, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace brotensor::detail::metal
