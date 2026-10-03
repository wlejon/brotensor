// Vulkan parity for the quantised-weight ops: the GGUF formats (Q8_0, Q4_K,
// Q6_K: dequantise, GEMV, batched linear on the GEMV and the cooperative-
// matrix prefill paths) and the INT8 W8A16 family (batched linear in FP16 and
// BF16, matmul, conv2d on the implicit-GEMM and fallback paths, conv3d).
//
// Weights are random values quantised on the host by the encoders below
// (block layouts as in brotensor/ops/quant.h, with negative Q6_K sub-block
// scales and nonzero Q4_K minimums so every field of a block is exercised);
// the reference decodes them on the host (checked against the CPU backend's
// dequant ops where it has one) and sums in double. Tolerances follow the
// format of the path: the GEMV keeps the decoded weight in FP32 and rounds
// once at the output; the prefill GEMM rounds each decoded weight to FP16
// as it enters shared memory, so an output may move by 2^-11 of
// sum |x w| plus the output rounding.

#include "test_vulkan_common.h"

#include "detail/quant.h"
#include "detail/spatial.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

std::uint16_t h16(float v) { return brotensor::fp32_to_fp16_bits(v); }
float f16(std::uint16_t b) { return brotensor::fp16_bits_to_fp32(b); }

// ─── host encoders / decoders ───────────────────────────────────────────────

void put16(std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); }
std::uint16_t get16(const std::uint8_t* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }

// Q8_0: 34 bytes per 32 values.
void enc_q8_0(const float* x, std::uint8_t* b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0f;
    put16(b, h16(d));
    const float dd = f16(h16(d));
    for (int i = 0; i < 32; ++i) {
        const int q = dd > 0 ? static_cast<int>(std::lround(x[i] / dd)) : 0;
        b[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(q, -127, 127)));
    }
}
void dec_q8_0(const std::uint8_t* b, float* y) {
    const float d = f16(get16(b));
    for (int i = 0; i < 32; ++i) y[i] = static_cast<float>(static_cast<std::int8_t>(b[2 + i])) * d;
}

// Q4_K: 144 bytes per 256: d, dmin, 12 bytes of 6-bit (scale, min), 128 of nibbles.
void q4k_sm(int j, const std::uint8_t* s, int& sc, int& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}
void enc_q4k(const float* x, std::uint8_t* b, Rng& r) {
    std::memset(b, 0, 144);
    float lo[8], hi[8];
    for (int j = 0; j < 8; ++j) {
        lo[j] = hi[j] = x[32 * j];
        for (int i = 0; i < 32; ++i) { lo[j] = std::min(lo[j], x[32 * j + i]); hi[j] = std::max(hi[j], x[32 * j + i]); }
    }
    float smax = 0, mmax = 0;
    for (int j = 0; j < 8; ++j) { smax = std::max(smax, (hi[j] - lo[j]) / 15.0f); mmax = std::max(mmax, -lo[j]); }
    const float d = smax / 63.0f, dmin = std::max(mmax, 1e-3f) / 63.0f;
    put16(b, h16(d));
    put16(b + 2, h16(dmin));
    const float dd = f16(h16(d)), dm = f16(h16(dmin));
    int sc[8], mn[8];
    for (int j = 0; j < 8; ++j) {
        sc[j] = std::clamp(static_cast<int>(std::lround((hi[j] - lo[j]) / 15.0f / dd)), 1, 63);
        mn[j] = std::clamp(static_cast<int>(std::lround(std::max(-lo[j], 0.0f) / dm)), 0, 63);
        if (r.next() % 7 == 0) mn[j] = static_cast<int>(r.next() % 64);   // exercise the high bits
    }
    std::uint8_t* s = b + 4;
    for (int j = 0; j < 4; ++j) {
        s[j] = static_cast<std::uint8_t>((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
        s[j + 4] = static_cast<std::uint8_t>((mn[j] & 63) | ((mn[j + 4] >> 4) << 6));
        s[j + 8] = static_cast<std::uint8_t>((sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4));
    }
    for (int j = 0; j < 8; ++j) {
        int scj, mj;
        q4k_sm(j, s, scj, mj);
        for (int i = 0; i < 32; ++i) {
            const float w = dd * scj;
            const int q = std::clamp(static_cast<int>(std::lround((x[32 * j + i] + dm * mj) / w)), 0, 15);
            std::uint8_t& byte = b[16 + 32 * (j / 2) + i];
            byte = static_cast<std::uint8_t>((j & 1) ? (byte & 0x0F) | (q << 4) : (byte & 0xF0) | q);
        }
    }
}
void dec_q4k(const std::uint8_t* b, float* y) {
    const float d = f16(get16(b)), dmin = f16(get16(b + 2));
    for (int j = 0; j < 8; ++j) {
        int sc, m;
        q4k_sm(j, b + 4, sc, m);
        for (int i = 0; i < 32; ++i) {
            const std::uint8_t byte = b[16 + 32 * (j / 2) + i];
            const int q = (j & 1) ? byte >> 4 : byte & 0xF;
            y[32 * j + i] = static_cast<float>(sc) * d * static_cast<float>(q) - static_cast<float>(m) * dmin;
        }
    }
}

// Q6_K: 210 bytes per 256: ql[128], qh[64], int8 sc[16], fp16 d.
void q6k_idx(int e, int& ql, int& qh, int& sh, bool& hi_nib, int& sb) {
    const int g = e / 128, local = e % 128, quad = local / 32, l = local % 32;
    ql = g * 64 + (quad & 1) * 32 + l;
    qh = 128 + g * 32 + l;
    sh = 2 * quad;
    hi_nib = quad >= 2;
    sb = g * 8 + quad * 2 + l / 16;
}
void enc_q6k(const float* x, std::uint8_t* b, Rng& r) {
    std::memset(b, 0, 210);
    float amax[16] = {};
    for (int e = 0; e < 256; ++e) {
        int ql, qh, sh, sb; bool hn;
        q6k_idx(e, ql, qh, sh, hn, sb);
        amax[sb] = std::max(amax[sb], std::fabs(x[e]));
    }
    float m = 0;
    for (float a : amax) m = std::max(m, a);
    const float d = std::max(m, 1e-4f) / (31.0f * 127.0f);
    put16(b + 208, h16(d));
    const float dd = f16(h16(d));
    int sc[16];
    for (int s = 0; s < 16; ++s) {
        sc[s] = std::clamp(static_cast<int>(std::lround(amax[s] / (dd * 31.0f))), 1, 127);
        if (r.next() % 2) sc[s] = -sc[s];   // negative scales: the sign must survive the decode
        b[192 + s] = static_cast<std::uint8_t>(static_cast<std::int8_t>(sc[s]));
    }
    for (int e = 0; e < 256; ++e) {
        int ql, qh, sh, sb; bool hn;
        q6k_idx(e, ql, qh, sh, hn, sb);
        const int v = std::clamp(static_cast<int>(std::lround(x[e] / (dd * sc[sb]))), -32, 31) + 32;
        b[ql] = static_cast<std::uint8_t>(hn ? (b[ql] & 0x0F) | ((v & 15) << 4) : (b[ql] & 0xF0) | (v & 15));
        b[qh] = static_cast<std::uint8_t>(b[qh] | (((v >> 4) & 3) << sh));
    }
}
void dec_q6k(const std::uint8_t* b, float* y) {
    const float d = f16(get16(b + 208));
    for (int e = 0; e < 256; ++e) {
        int ql, qh, sh, sb; bool hn;
        q6k_idx(e, ql, qh, sh, hn, sb);
        const int v = ((hn ? b[ql] >> 4 : b[ql] & 15) | (((b[qh] >> sh) & 3) << 4)) - 32;
        y[e] = d * static_cast<float>(static_cast<std::int8_t>(b[192 + sb])) * static_cast<float>(v);
    }
}

struct QMat {
    Dtype dt;
    int rows, k;
    std::vector<std::uint8_t> bytes;
    std::vector<float> w;   // decoded (FP32, before any FP16 rounding)
};

QMat make_gguf(Dtype dt, int rows, int k, std::uint64_t seed) {
    QMat m{dt, rows, k, {}, {}};
    const int blk = brotensor::dtype_block_size(dt), bb = brotensor::dtype_block_bytes(dt);
    const std::vector<float> src = random_values(std::size_t(rows) * k, seed, -1.0f, 1.0f, Dtype::FP32);
    m.bytes.resize(std::size_t(rows) * (k / blk) * bb);
    m.w.resize(src.size());
    Rng r(seed ^ 0x55);
    for (std::size_t b = 0; b < src.size() / blk; ++b) {
        std::uint8_t* p = m.bytes.data() + b * bb;
        const float* x = src.data() + b * blk;
        float* y = m.w.data() + b * blk;
        if (dt == Dtype::Q8_0) { enc_q8_0(x, p); dec_q8_0(p, y); }
        else if (dt == Dtype::Q4_K) { enc_q4k(x, p, r); dec_q4k(p, y); }
        else { enc_q6k(x, p, r); dec_q6k(p, y); }
    }
    return m;
}

Tensor upload_q(const QMat& m, Device d = vk()) {
    return Tensor::from_raw_bytes_on(d, m.bytes.data(), m.rows, m.k, m.dt, m.bytes.size());
}

struct I8Mat {
    int rows, k;
    std::vector<std::int8_t> q;
    std::vector<float> scale, w;
};

I8Mat make_int8(int rows, int k, std::uint64_t seed) {
    I8Mat m{rows, k, {}, {}, {}};
    const std::vector<float> src = random_values(std::size_t(rows) * k, seed, -0.5f, 0.5f, Dtype::FP16);
    std::vector<std::uint16_t> bits(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) bits[i] = h16(src[i]);
    m.q.resize(src.size());
    m.scale.resize(rows);
    brotensor::quantize_int8_per_row_host(bits.data(), rows, k, m.q.data(), m.scale.data());
    m.w.resize(src.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < k; ++c) m.w[std::size_t(r) * k + c] = float(m.q[std::size_t(r) * k + c]) * m.scale[r];
    return m;
}

// ─── checks ─────────────────────────────────────────────────────────────────

// Y(B, N) = X W^T + bias in double, and per element the sum of |x w| that
// bounds the rounding of the decoded weights.
void reference(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>* bias, int B,
               int N, int K, std::vector<float>& y, std::vector<float>& mag) {
    y.assign(std::size_t(B) * N, 0);
    mag.assign(y.size(), 0);
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n) {
            double s = 0, a = 0;
            for (int k = 0; k < K; ++k) {
                const double t = double(x[std::size_t(b) * K + k]) * w[std::size_t(n) * K + k];
                s += t;
                a += std::fabs(t);
            }
            if (bias) s += (*bias)[n];
            y[std::size_t(b) * N + n] = float(s);
            mag[std::size_t(b) * N + n] = float(a);
        }
}

// |got - want| <= wrel * mag + orel * |want| + 1e-6 elementwise.
void expect_bound(const std::vector<float>& got, const std::vector<float>& want, const std::vector<float>& mag,
                  float wrel, float orel, const std::string& tag) {
    if (got.size() != want.size()) {
        std::printf("  FAIL  %s: size %zu vs %zu\n", tag.c_str(), got.size(), want.size());
        ++failures();
        return;
    }
    double worst = 0;
    std::size_t at = 0;
    bool bad = false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(double(got[i]) - want[i]);
        const double tol = wrel * mag[i] + orel * std::fabs(want[i]) + 1e-6;
        if (!(diff <= tol)) { bad = true; at = i; }
        worst = std::max(worst, diff / tol);
    }
    if (bad) {
        std::printf("  FAIL  %s: at %zu got %.7g want %.7g (mag %.4g)\n", tag.c_str(), at, got[at], want[at], mag[at]);
        ++failures();
        return;
    }
    std::printf("  PASS  %s (worst %.2f of tolerance)\n", tag.c_str(), worst);
}

const char* qname(Dtype dt) { return dt == Dtype::Q8_0 ? "Q8_0" : dt == Dtype::Q4_K ? "Q4_K" : "Q6_K"; }

// The weight decode error a path may add: FP32 weights on the GEMV, FP16 in
// the GEMM's tiles.
float weight_rel(int B) { return B <= 8 ? 2e-6f : 1.0f / 2048.0f; }

void test_gguf_dequant(Dtype dt) {
    const int rows = 37, k = dt == Dtype::Q8_0 ? 32 * 7 : 256 * 3;
    const QMat m = make_gguf(dt, rows, k, 11);
    Tensor W = upload_q(m), Y;
    if (dt == Dtype::Q8_0) brotensor::dequant_q8_0_to_fp16(W, Y);
    else if (dt == Dtype::Q4_K) brotensor::dequant_q4k_to_fp16(W, Y);
    else brotensor::dequant_q6k_to_fp16(W, Y);
    VKT_CHECK(Y.rows == rows && Y.cols == k && Y.dtype == Dtype::FP16);
    std::vector<float> want(m.w.size());
    for (std::size_t i = 0; i < want.size(); ++i) want[i] = round_to(Dtype::FP16, m.w[i]);
    // One FP16 ulp: d sc q - dmin m may be fused differently.
    expect_close(download(Y), want, 0, 1.0f / 1024.0f, std::string("dequant ") + qname(dt));
    if (dt != Dtype::Q6_K) {   // the CPU backend decodes Q8_0 and Q4_K
        Tensor Wc = upload_q(m, Device::cpu()), Yc;
        if (dt == Dtype::Q8_0) brotensor::dequant_q8_0_to_fp16(Wc, Yc);
        else brotensor::dequant_q4k_to_fp16(Wc, Yc);
        expect_close(download(Y), download(Yc), 0, 1.0f / 1024.0f, std::string("dequant ") + qname(dt) + " vs CPU");
    }
}

void test_gguf_linear(Dtype dt, int N, int K, int B, bool with_bias, std::uint64_t seed) {
    const QMat m = make_gguf(dt, N, K, seed);
    Tensor W = upload_q(m);
    const std::vector<float> x = random_values(std::size_t(B) * K, seed + 1, -1, 1, Dtype::FP16);
    const std::vector<float> bias = random_values(N, seed + 2, -0.5f, 0.5f, Dtype::FP16);
    Tensor X = upload(x, B, K, Dtype::FP16), Bt = upload(bias, N, 1, Dtype::FP16), Y;
    std::vector<float> want, mag;
    reference(x, m.w, with_bias ? &bias : nullptr, B, N, K, want, mag);
    char tag[96];
    std::snprintf(tag, sizeof tag, "%s linear N=%d K=%d B=%d%s [%s]", qname(dt), N, K, B, with_bias ? " +bias" : "",
                  dv::quant_linear_path(dv::device(0), B));
    if (B == 1 && with_bias) {   // the GEMV op: x (K, 1) -> y (N, 1)
        Tensor xc = upload(x, K, 1, Dtype::FP16), y;
        if (dt == Dtype::Q8_0) brotensor::linear_forward_q8_0_fp16(W, &Bt, xc, y);
        else if (dt == Dtype::Q4_K) brotensor::linear_forward_q4k_fp16(W, &Bt, xc, y);
        else brotensor::linear_forward_q6k_fp16(W, &Bt, xc, y);
        VKT_CHECK(y.rows == N && y.cols == 1);
        expect_bound(download(y), want, mag, weight_rel(1), 1.0f / 1024.0f, std::string(tag) + " (gemv op)");
    }
    const Tensor* bp = with_bias ? &Bt : nullptr;
    if (dt == Dtype::Q8_0) brotensor::linear_forward_batched_q8_0_fp16(W, bp, X, Y);
    else if (dt == Dtype::Q4_K) brotensor::linear_forward_batched_q4k_fp16(W, bp, X, Y);
    else brotensor::linear_forward_batched_q6k_fp16(W, bp, X, Y);
    VKT_CHECK(Y.rows == B && Y.cols == N && Y.dtype == Dtype::FP16);
    expect_bound(download(Y), want, mag, weight_rel(B), 1.0f / 1024.0f, tag);
}

void test_gguf_errors() {
    const QMat m = make_gguf(Dtype::Q4_K, 4, 256, 3);
    Tensor W = upload_q(m), Y;
    Tensor X = upload(random_values(2 * 512, 1, -1, 1, Dtype::FP16), 2, 512, Dtype::FP16);
    VKT_CHECK(throws([&] { brotensor::linear_forward_batched_q4k_fp16(W, nullptr, X, Y); }));   // K mismatch
    Tensor X32 = upload(random_values(2 * 256, 1, -1, 1, Dtype::FP32), 2, 256, Dtype::FP32);
    VKT_CHECK(throws([&] { brotensor::linear_forward_batched_q4k_fp16(W, nullptr, X32, Y); }));   // FP32 X
    VKT_CHECK(throws([&] { brotensor::linear_forward_batched_q8_0_fp16(W, nullptr, X, Y); }));   // wrong format
}

// ─── INT8 ───────────────────────────────────────────────────────────────────

void test_int8_linear(int N, int K, int B, Dtype dt, bool with_bias, std::uint64_t seed) {
    const I8Mat m = make_int8(N, K, seed);
    Tensor W = Tensor::from_host_int8_on(vk(), m.q.data(), N, K);
    Tensor S = Tensor::from_host_on(vk(), m.scale.data(), N, 1);
    const std::vector<float> x = random_values(std::size_t(B) * K, seed + 1, -1, 1, dt);
    const std::vector<float> bias = random_values(N, seed + 2, -0.5f, 0.5f, dt);
    Tensor X = upload(x, B, K, dt), Bt = upload(bias, N, 1, dt), Y;
    std::vector<float> want, mag;
    reference(x, m.w, with_bias ? &bias : nullptr, B, N, K, want, mag);
    brotensor::linear_forward_batched_int8w_fp16(W, S, with_bias ? &Bt : nullptr, X, Y);
    VKT_CHECK(Y.rows == B && Y.cols == N && Y.dtype == dt);
    char tag[96];
    std::snprintf(tag, sizeof tag, "INT8 linear %s N=%d K=%d B=%d%s [%s]", dt_name(dt), N, K, B,
                  with_bias ? " +bias" : "", dv::quant_linear_path(dv::device(0), B));
    expect_bound(download(Y), want, mag, weight_rel(B), dt == Dtype::BF16 ? 1.0f / 128.0f : 1.0f / 1024.0f, tag);
}

void test_int8_matmul(int M, int K, int Nb, std::uint64_t seed) {
    const I8Mat m = make_int8(M, K, seed);
    Tensor W = Tensor::from_host_int8_on(vk(), m.q.data(), M, K);
    Tensor S = Tensor::from_host_on(vk(), m.scale.data(), M, 1);
    const std::vector<float> x = random_values(std::size_t(K) * Nb, seed + 1, -1, 1, Dtype::FP16);
    std::vector<float> xt(x.size());   // (Nb, K) for the reference
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < Nb; ++j) xt[std::size_t(j) * K + k] = x[std::size_t(k) * Nb + j];
    std::vector<float> yt, magt;
    reference(xt, m.w, nullptr, Nb, M, K, yt, magt);
    std::vector<float> want(yt.size()), mag(yt.size());
    for (int j = 0; j < Nb; ++j)
        for (int i = 0; i < M; ++i) {
            want[std::size_t(i) * Nb + j] = yt[std::size_t(j) * M + i];
            mag[std::size_t(i) * Nb + j] = magt[std::size_t(j) * M + i];
        }
    Tensor X = upload(x, K, Nb, Dtype::FP16), Y;
    brotensor::matmul_int8w_fp16(W, S, X, Y);
    VKT_CHECK(Y.rows == M && Y.cols == Nb);
    char tag[64];
    std::snprintf(tag, sizeof tag, "matmul_int8w M=%d K=%d Nb=%d", M, K, Nb);
    expect_bound(download(Y), want, mag, weight_rel(Nb), 1.0f / 1024.0f, tag);
}

struct ConvCase {
    int n, cin, h, w, cout, kh, kw, s, p, d, groups;
    const char* name;
};

void test_int8_conv2d(const ConvCase& c, std::uint64_t seed) {
    const int K = c.cin / c.groups * c.kh * c.kw;
    const I8Mat m = make_int8(c.cout, K, seed);
    const std::vector<float> x = random_values(std::size_t(c.n) * c.cin * c.h * c.w, seed + 1, -1, 1, Dtype::FP16);
    const std::vector<float> b = random_values(c.cout, seed + 2, -0.5f, 0.5f, Dtype::FP16);
    Tensor X = upload(x, c.n, c.cin * c.h * c.w, Dtype::FP16), Bt = upload(b, c.cout, 1, Dtype::FP16);
    Tensor W = Tensor::from_host_int8_on(vk(), m.q.data(), c.cout, K);
    Tensor S = Tensor::from_host_on(vk(), m.scale.data(), c.cout, 1);
    Tensor Y;
    brotensor::conv2d_int8w_fp16_forward(X, W, S, &Bt, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.s, c.s, c.p, c.p,
                                         c.d, c.d, c.groups, Y);
    // CPU reference in FP32 on the decoded weights.
    Tensor Xc = Tensor::from_host_on(Device::cpu(), x.data(), c.n, c.cin * c.h * c.w);
    Tensor Wc = Tensor::from_host_on(Device::cpu(), m.w.data(), c.cout, K);
    Tensor Bc = Tensor::from_host_on(Device::cpu(), b.data(), c.cout, 1), Yc;
    brotensor::conv2d_forward(Xc, Wc, &Bc, c.n, c.cin, c.h, c.w, c.cout, c.kh, c.kw, c.s, c.s, c.p, c.p, c.d, c.d,
                              c.groups, Yc);
    dv::Conv2dArgs a;
    a.dt = Dtype::FP16; a.n = c.n; a.cin = c.cin; a.h = c.h; a.wd = c.w; a.cout = c.cout; a.kh = c.kh; a.kw = c.kw;
    a.sh = a.sw = c.s; a.ph = a.pw = c.p; a.dh = a.dw = c.d; a.groups = c.groups;
    const std::string tag = std::string("conv2d_int8w ") + c.name + " [" + dv::conv2d_path(dv::device(0), a) + "]";
    // Weights in FP16 tiles: 2^-11 per product; |x| <= 1, |w| <= 0.5.
    const float atol = 0.5f * K / 2048.0f * 0.25f + 2e-3f;
    expect_close(download(Y), Yc.to_host_vector(), atol, 2e-3f, tag);
}

void test_int8_conv3d(bool patch, std::uint64_t seed) {
    const int N = patch ? 5 : 1, C = 3, T = 2, H = patch ? 14 : 6, W = patch ? 14 : 7, Co = patch ? 40 : 8;
    const int kT = 2, kH = patch ? 14 : 3, kW = patch ? 14 : 3, p = patch ? 0 : 1;
    const int K = C * kT * kH * kW;
    const I8Mat m = make_int8(Co, K, seed);
    const std::vector<float> x = random_values(std::size_t(N) * C * T * H * W, seed + 1, -1, 1, Dtype::FP16);
    const std::vector<float> b = random_values(Co, seed + 2, -0.5f, 0.5f, Dtype::FP16);
    Tensor X = upload(x, N, C * T * H * W, Dtype::FP16), Bt = upload(b, Co, 1, Dtype::FP16), Y;
    Tensor Wq = Tensor::from_host_int8_on(vk(), m.q.data(), Co, K);
    Tensor S = Tensor::from_host_on(vk(), m.scale.data(), Co, 1);
    brotensor::conv3d_int8w_fp16_forward(X, Wq, S, &Bt, N, C, T, H, W, Co, kT, kH, kW, kT, patch ? kH : 1,
                                         patch ? kW : 1, 0, p, p, 1, 1, 1, 1, Y);
    Tensor Xc = Tensor::from_host_on(Device::cpu(), x.data(), N, C * T * H * W);
    Tensor Wc = Tensor::from_host_on(Device::cpu(), m.w.data(), Co, K);
    Tensor Bc = Tensor::from_host_on(Device::cpu(), b.data(), Co, 1), Yc;
    brotensor::conv3d_forward(Xc, Wc, &Bc, N, C, T, H, W, Co, kT, kH, kW, kT, patch ? kH : 1, patch ? kW : 1, 0, p,
                              p, 1, 1, 1, 1, Yc);
    expect_close(download(Y), Yc.to_host_vector(), 0.5f * K / 2048.0f * 0.25f + 2e-3f, 2e-3f,
                 patch ? "conv3d_int8w patch embedding" : "conv3d_int8w 3x3x2 direct");
}

}  // namespace

void run_quant_attention_tests();   // test_vulkan_quant_attention.cpp

void run_quant_tests() {
    std::printf("\n[quantised weights]\n");
    for (Dtype dt : {Dtype::Q8_0, Dtype::Q4_K, Dtype::Q6_K}) test_gguf_dequant(dt);
    std::uint64_t seed = 100;
    for (Dtype dt : {Dtype::Q8_0, Dtype::Q4_K, Dtype::Q6_K}) {
        const int blk = brotensor::dtype_block_size(dt);
        // K: one block, an odd number of blocks, a long row; N odd and off every tile.
        const int ks[] = {blk, blk * 7, dt == Dtype::Q8_0 ? 32 * 33 : 256 * 16};
        for (int K : ks) {
            for (int B : {1, 2, 3, 4, 8}) test_gguf_linear(dt, 301, K, B, B != 3, seed++);
        }
        for (int B : {9, 17, 64, 200}) test_gguf_linear(dt, 301, blk * 7, B, B != 17, seed++);
        test_gguf_linear(dt, 1037, ks[2], 96, true, seed++);
        test_gguf_linear(dt, 8, blk, 1, true, seed++);   // fewer rows than a workgroup
    }
    test_gguf_errors();
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (int K : {64, 1000, 100, 4096}) {
            for (int B : {1, 3, 8, 9, 64}) test_int8_linear(203, K, B, dt, B != 3, seed++);
        }
    }
    test_int8_matmul(77, 96, 1, seed++);
    test_int8_matmul(77, 100, 5, seed++);
    test_int8_matmul(130, 256, 40, seed++);
    const ConvCase convs[] = {
        {1, 32, 13, 13, 48, 3, 3, 1, 1, 1, 1, "3x3 32->48 13x13"},
        {2, 64, 16, 16, 64, 1, 1, 1, 0, 1, 1, "1x1 64->64 16x16 n2"},
        {1, 24, 15, 17, 40, 3, 3, 2, 1, 1, 1, "3x3 s2 24->40 (K % 8 = 0, odd W)"},
        {1, 3, 20, 20, 32, 3, 3, 1, 1, 1, 1, "3x3 3->32 (K = 27)"},
        {1, 16, 12, 12, 16, 3, 3, 1, 2, 2, 16, "depthwise dilated"},
    };
    for (const ConvCase& c : convs) test_int8_conv2d(c, seed++);
    test_int8_conv3d(true, seed++);
    test_int8_conv3d(false, seed++);
    run_quant_attention_tests();
}

}  // namespace vkt
