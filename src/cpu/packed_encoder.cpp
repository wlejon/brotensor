// ─── CPU packed-encoder ops ─────────────────────────────────────────────────
//
// FP32 reference implementations of the ops a packed (variable-length,
// unpadded) encoder batch needs — the parity oracle for the CUDA kernels in
// src/cuda/flash_attention_packed.cu and src/cuda/packed_encoder.cu:
//
//   flash_attention_packed_qkv_forward — bidirectional (optionally windowed)
//       self-attention off a fused (L, 3*H*hd) QKV, per-row sequence bounds.
//   rope_qkv_packed_inplace            — RoPE on the Q/K sections, per-row pos.
//   segment_softmax_stats              — [top1, top1-top2, norm. entropy, k/255]
//                                        per variable-length logit segment.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void need_fp32(const ::brotensor::Tensor& t, const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
}

}  // namespace

void flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV,
                                        const ::brotensor::Tensor& seq_bounds,
                                        int num_heads, int window,
                                        ::brotensor::Tensor& O) {
    constexpr const char* op = "flash_attention_packed_qkv_forward";
    need_fp32(QKV, op, "QKV");
    if (seq_bounds.dtype != Dtype::INT32) fail(op, "seq_bounds must be INT32");
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) {
        fail(op, "QKV.cols must be 3 * num_heads * head_dim");
    }
    const int L = QKV.rows;
    if (seq_bounds.rows != L || seq_bounds.cols != 2) fail(op, "seq_bounds must be (L, 2)");
    const int D = QKV.cols / 3;
    const int hd = D / num_heads;
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP32) O.resize(L, D, Dtype::FP32);
    if (L == 0) return;

    const float* qkv = QKV.host_f32();
    const int32_t* b = static_cast<const int32_t*>(seq_bounds.host_raw());
    float* out = O.host_f32_mut();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const int hw = window > 0 ? window / 2 : -1;
    std::vector<float> s;
    for (int r = 0; r < L; ++r) {
        const int start = b[2 * r], end = b[2 * r + 1];
        if (start < 0 || end > L || r < start || r >= end) fail(op, "row outside its seq_bounds");
        const int lo = hw >= 0 ? std::max(start, r - hw) : start;
        const int hi = hw >= 0 ? std::min(end, r + hw + 1) : end;
        s.assign(static_cast<std::size_t>(hi - lo), 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            const float* q = qkv + static_cast<std::size_t>(r) * 3 * D + h * hd;
            float mx = -INFINITY;
            for (int j = lo; j < hi; ++j) {
                const float* k = qkv + static_cast<std::size_t>(j) * 3 * D + D + h * hd;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += q[d] * k[d];
                s[static_cast<std::size_t>(j - lo)] = dot * scale;
                mx = std::max(mx, dot * scale);
            }
            float sum = 0.0f;
            for (float& v : s) {
                v = std::exp(v - mx);
                sum += v;
            }
            float* o = out + static_cast<std::size_t>(r) * D + h * hd;
            for (int d = 0; d < hd; ++d) o[d] = 0.0f;
            for (int j = lo; j < hi; ++j) {
                const float* v = qkv + static_cast<std::size_t>(j) * 3 * D + 2 * D + h * hd;
                const float p = s[static_cast<std::size_t>(j - lo)] / sum;
                for (int d = 0; d < hd; ++d) o[d] += p * v[d];
            }
        }
    }
}

void rope_qkv_packed_inplace(::brotensor::Tensor& QKV, const ::brotensor::Tensor& cos_tbl,
                             const ::brotensor::Tensor& sin_tbl, const ::brotensor::Tensor& pos,
                             int num_heads, int head_dim) {
    constexpr const char* op = "rope_qkv_packed_inplace";
    need_fp32(QKV, op, "QKV");
    need_fp32(cos_tbl, op, "cos_tbl");
    need_fp32(sin_tbl, op, "sin_tbl");
    if (pos.dtype != Dtype::INT32 || pos.rows != QKV.rows) fail(op, "pos must be (L, 1) INT32");
    if (head_dim <= 0 || (head_dim & 1) || num_heads <= 0 || QKV.cols != 3 * num_heads * head_dim) {
        fail(op, "QKV.cols must be 3 * num_heads * head_dim with head_dim even");
    }
    const int half = head_dim / 2;
    if (cos_tbl.cols != half || sin_tbl.cols != half || sin_tbl.rows != cos_tbl.rows) {
        fail(op, "cos_tbl / sin_tbl must be (P, head_dim/2)");
    }
    const int L = QKV.rows;
    const int D = num_heads * head_dim;
    float* x = QKV.host_f32_mut();
    const float* ct = cos_tbl.host_f32();
    const float* st = sin_tbl.host_f32();
    const int32_t* p = static_cast<const int32_t*>(pos.host_raw());
    for (int r = 0; r < L; ++r) {
        const int pr = p[r];
        if (pr < 0 || pr >= cos_tbl.rows) fail(op, "pos outside the cos/sin table");
        float* row = x + static_cast<std::size_t>(r) * 3 * D;
        for (int sec = 0; sec < 2; ++sec) {
            for (int h = 0; h < num_heads; ++h) {
                float* v = row + sec * D + h * head_dim;
                for (int i = 0; i < half; ++i) {
                    const float c = ct[static_cast<std::size_t>(pr) * half + i];
                    const float s = st[static_cast<std::size_t>(pr) * half + i];
                    const float x0 = v[2 * i], x1 = v[2 * i + 1];
                    v[2 * i] = x0 * c - x1 * s;
                    v[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
}

void segment_softmax_stats(const ::brotensor::Tensor& logits,
                           const ::brotensor::Tensor& seg_offsets,
                           ::brotensor::Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    need_fp32(logits, op, "logits");
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) {
        fail(op, "seg_offsets must be (S+1, 1) INT32");
    }
    const int S = seg_offsets.rows - 1;
    if (out.rows != S || out.cols != 4 || out.dtype != Dtype::FP32) out.resize(S, 4, Dtype::FP32);
    const float* l = logits.host_f32();
    const int32_t* off = static_cast<const int32_t*>(seg_offsets.host_raw());
    float* o = out.host_f32_mut();
    std::vector<float> p;
    for (int s = 0; s < S; ++s) {
        const int a = off[s], b = off[s + 1];
        float* row = o + static_cast<std::size_t>(s) * 4;
        if (a < 0 || b < a || b > logits.size()) fail(op, "seg_offsets out of range");
        if (b == a) {
            row[0] = row[1] = row[2] = row[3] = 0.0f;
            continue;
        }
        const int n = b - a;
        float mx = -INFINITY;
        for (int i = a; i < b; ++i) mx = std::max(mx, l[i]);
        p.assign(static_cast<std::size_t>(n), 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] = std::exp(l[a + i] - mx);
            sum += p[static_cast<std::size_t>(i)];
        }
        float top1 = 0.0f, top2 = 0.0f, ent = 0.0f;
        for (float& v : p) {
            v /= sum;
            ent -= v * std::log(std::max(v, 1e-9f));
            if (v > top1) {
                top2 = top1;
                top1 = v;
            } else if (v > top2) {
                top2 = v;
            }
        }
        const float k = std::max(2.0f, static_cast<float>(n));
        row[0] = top1;
        row[1] = top1 - top2;
        row[2] = ent / std::log(k);
        row[3] = k / 255.0f;
    }
}

namespace {

float apply_act(float v, int act) {
    switch (act) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        case 3: return 0.5f * v * (1.0f + std::erf(v * 0.70710678118f));
        case 4: return v / (1.0f + std::exp(-v));
        case 5: return v / (1.0f + std::exp(-1.702f * v));
        default: return v;
    }
}

}  // namespace

void linear_forward_batched_ex(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                               const ::brotensor::Tensor& X, int act, int epilogue_flags,
                               ::brotensor::Tensor* /*workspace: CPU never splits*/,
                               ::brotensor::Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_ex";
    const int epilogue = epilogue_flags & ~16;  // kLinearEpiFastAccum: FP32 here regardless
    need_fp32(W, op, "W");
    need_fp32(X, op, "X");
    if (bias) need_fp32(*bias, op, "bias");
    const int N = W.rows, K = W.cols, M = X.rows;
    if (X.cols != K) fail(op, "X.cols must equal W.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail(op, "bias size must equal W.rows");
    if (epilogue < 0 || epilogue > 2) fail(op, "unknown epilogue");
    if (epilogue == 2 && (act != 0 || N % 2 != 0)) fail(op, "geglu needs act 0 and even W.rows");
    const int out_cols = epilogue == 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != Dtype::FP32) fail(op, "accumulate needs Y (B, out) FP32");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != Dtype::FP32) {
        Y.resize(M, out_cols, Dtype::FP32);
    }
    if (M == 0) return;
    const float* w = W.host_f32();
    const float* x = X.host_f32();
    const float* bb = bias ? bias->host_f32() : nullptr;
    float* y = Y.host_f32_mut();
    std::vector<float> r(static_cast<std::size_t>(N));
    for (int m = 0; m < M; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * K;
        for (int n = 0; n < N; ++n) {
            const float* wr = w + static_cast<std::size_t>(n) * K;
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
            r[static_cast<std::size_t>(n)] = apply_act(acc + (bb ? bb[n] : 0.0f), act);
        }
        float* yr = y + static_cast<std::size_t>(m) * out_cols;
        if (epilogue == 0) {
            for (int n = 0; n < N; ++n) yr[n] = r[static_cast<std::size_t>(n)];
        } else if (epilogue == 1) {
            for (int n = 0; n < N; ++n) yr[n] += r[static_cast<std::size_t>(n)];
        } else {
            for (int j = 0; j < out_cols; ++j)
                yr[j] = r[static_cast<std::size_t>(2 * j)] * apply_act(r[static_cast<std::size_t>(2 * j + 1)], 3);
        }
    }
}

}  // namespace brotensor::detail::cpu
