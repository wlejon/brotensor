// Vulkan attention: the kernel dispatcher (detail/attention.h) and the flash
// family — flash_attention_forward / _gqa / _windowed / _varlen /
// _packed_qkv, the decode ops over a KV cache, kv_cache_append, and the
// projection-fused flash_attention_qkvo_forward / _project_kv /
// _q_with_kv_cached_forward. Contracts follow the HIP backend
// (src/hip/flash_attention*.hip) and the CPU reference: Q, K, V and O share
// one dtype (FP32 / FP16 / BF16), O is resized to (Lq, Dq) in that dtype,
// scale 1 / sqrt(head_dim), a key mask is valid where > 0.5, and a query row
// without a valid key comes out zero. Design: docs/vulkan.md "Attention".

#include "detail/attention.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void linear_forward_batched_ex(const Tensor& W, const Tensor* bias, const Tensor& X, int act, int epilogue,
                               Tensor* workspace, Tensor& Y);   // ops_linear.cpp

namespace {

struct FaPush {
    std::uint64_t q, k, v, o, mask, aux, aux2, part;
    std::uint32_t lq, lk, ldq, ldk, ldo, group, window, causal, nseq, chunk, hq, qb0;
    std::int32_t q_offset;
    float scale, softcap;
};

struct CombinePush {
    std::uint64_t part, o;
    std::uint32_t entries, nsplit, hd, hq, ldo;
};

std::atomic<int> g_override{0};
std::atomic<int> g_cm_bc{0}, g_cm_nsg{0};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }
std::uint32_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }
bool is16(Dtype t) { return t == Dtype::FP16 || t == Dtype::BF16; }

FaPush make_push(const AttnProblem& p) {
    FaPush pc{};
    pc.q = p.q; pc.k = p.k; pc.v = p.v; pc.o = p.o;
    pc.mask = p.mask; pc.aux = p.aux; pc.aux2 = p.aux2;
    pc.lq = static_cast<std::uint32_t>(p.lq);
    pc.lk = static_cast<std::uint32_t>(p.lk);
    pc.ldq = static_cast<std::uint32_t>(p.ldq);
    pc.ldk = static_cast<std::uint32_t>(p.ldk);
    pc.ldo = static_cast<std::uint32_t>(p.ldo);
    pc.group = static_cast<std::uint32_t>(p.hq / p.hkv);
    pc.window = static_cast<std::uint32_t>(std::max(0, p.window));
    pc.causal = p.causal ? 1u : 0u;
    pc.nseq = static_cast<std::uint32_t>(p.nseq);
    pc.hq = static_cast<std::uint32_t>(p.hq);
    pc.q_offset = p.q_offset;
    pc.softcap = p.softcap;
    return pc;
}

// ─── kernel choice ──────────────────────────────────────────────────────────

constexpr int kCmMaxHd = 256;
constexpr int kRowsMaxLq = 4;   // at or below this many query rows the decode kernel wins

bool cm_eligible(DeviceCtx& d, const AttnProblem& p) {
    return d.info().coopmat_f16 && is16(p.dt) && p.hd <= kCmMaxHd && p.softcap <= 0.0f && !p.maskpos;
}

bool dense_eligible(const AttnProblem& p) {
    return p.mode == FA_MODE_ROWS && !p.causal && p.window <= 0 && p.softcap <= 0.0f && !p.maskpos;
}

enum class Path { Rows, Cm, Dense };

Path choose(DeviceCtx& d, const AttnProblem& p) {
    if (p.nan_safe) return Path::Rows;
    static const int env_ov = [] {   // BROTENSOR_VK_FA_PATH=rows|cm|dense (benchmarking)
        const char* e = std::getenv("BROTENSOR_VK_FA_PATH");
        const std::string s = e ? e : "";
        return s == "rows" ? 1 : s == "cm" ? 2 : s == "dense" ? 3 : 0;
    }();
    int ov = g_override.load(std::memory_order_relaxed);
    if (ov == 0) ov = env_ov;
    if (ov == 1) return Path::Rows;
    if (ov == 2 && cm_eligible(d, p)) return Path::Cm;
    if (ov == 3 && dense_eligible(p)) return Path::Dense;
    if (cm_eligible(d, p) && p.lq > kRowsMaxLq) return Path::Cm;
    // FP32 (and 16-bit without cooperative matrix, or heads too wide for
    // fa_cm): big bidirectional problems are GEMM-shaped.
    if (dense_eligible(p) && p.lq >= 64 && p.lk >= 64) return Path::Dense;
    if (dense_eligible(p) && p.hd > kCmMaxHd && p.lq > kRowsMaxLq) return Path::Dense;
    return Path::Rows;
}

// ─── fa_cm.comp ─────────────────────────────────────────────────────────────

struct CmCfg { std::uint32_t bc, nsg; };

CmCfg pick_cm(const AttnProblem& p) {
    static const CmCfg forced = [] {
        CmCfg c{0, 0};
        if (const char* e = std::getenv("BROTENSOR_VK_FA_CFG")) {
            unsigned bc = 0, ns = 0;
            if (std::sscanf(e, "%u,%u", &bc, &ns) == 2) c = {bc, ns};
        }
        return c;
    }();
    CmCfg c{static_cast<std::uint32_t>(g_cm_bc.load()), static_cast<std::uint32_t>(g_cm_nsg.load())};
    if (c.bc == 0) c = forced;
    if (c.bc == 0) {
        // Measured (docs/vulkan.md "Attention"): heads up to 96 wide run best
        // with 16-key blocks and 32 query rows, 128-wide ones with 32 x 64 —
        // while K and V stay in the last-level cache. Every query block streams
        // its head's whole K / V, so once they outgrow it (24 MiB of K + V, the
        // 32 MB Infinity Cache of the RDNA 3.5 part this was measured on) the
        // 32-key blocks win: hd 64 x 16 heads at 7k-16k keys ran 8.5 -> 10.0,
        // 8.0 -> 10.2, 7.2 -> 10.3 and 3.0 -> 10.0 TF/s (TripoSplat's flow DiT
        // attends over 8-13k rows).
        const double kv_bytes = 2.0 * p.lk * p.hkv * p.hd * 2.0;
        if (p.hd <= 96) c = kv_bytes > 24.0 * 1024 * 1024 ? CmCfg{32, 2} : CmCfg{16, 2};
        else if (p.hd <= 128) c = {32, 4};
        else c = {16, 2};
    }
    if ((c.bc != 16 && c.bc != 32 && c.bc != 64) || (c.nsg != 1 && c.nsg != 2 && c.nsg != 4) ||
        c.nsg * 16 > 2 * c.bc) {
        fail(p.op, "BROTENSOR_VK_FA_CFG: bc in {16, 32, 64}, nsg in {1, 2, 4}, 16 nsg <= 2 bc");
    }
    return c;
}

void combine(DeviceCtx& d, const AttnProblem& p, std::uint64_t part, std::uint64_t nsplit) {
    const Kernel& kc = d.pipelines().get(dt_variant(ShaderId::fa_combine_f32, p.dt, p.op));
    const CombinePush cp{part, p.o, static_cast<std::uint32_t>(static_cast<long long>(p.lq) * p.hq),
                         static_cast<std::uint32_t>(nsplit), static_cast<std::uint32_t>(p.hd),
                         static_cast<std::uint32_t>(p.hq), static_cast<std::uint32_t>(p.ldo)};
    launch(d, kc, cp, static_cast<std::uint32_t>(std::min<long long>(65535, cp.entries)));
}

// FP32 partials for `nsplit` key splits: (lq * hq * nsplit) x (hd + 2).
Tensor partials(DeviceCtx& d, const AttnProblem& p, std::uint64_t nsplit) {
    const long long n = static_cast<long long>(p.lq) * p.hq * static_cast<long long>(nsplit) * (p.hd + 2);
    if (n > 0x7fffffffLL) fail(p.op, "attention partials too large");
    return Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(n), 1, Dtype::FP32);
}

void run_cm(DeviceCtx& d, const AttnProblem& p) {
    const CmCfg c = pick_cm(p);
    const std::uint32_t br = c.nsg * 16;
    const bool vec = p.hd % 8 == 0 && p.q % 16 == 0 && p.k % 16 == 0 && p.v % 16 == 0 && p.ldq % 8 == 0 &&
                     p.ldk % 8 == 0;
    const ShaderId id = p.dt == Dtype::FP16 ? ShaderId::fa_cm_f16 : ShaderId::fa_cm_bf16;
    const std::uint32_t nqb = static_cast<std::uint32_t>(cdiv(p.lq, br));
    // Split the keys when the query blocks alone would not fill the GPU
    // (~160 workgroups) and each split keeps at least 8 key blocks.
    const std::uint64_t wgs = static_cast<std::uint64_t>(nqb) * p.hq;
    std::uint64_t nsplit = 1;
    if (wgs < 160 && p.lk > 16 * static_cast<int>(c.bc)) {
        nsplit = std::min<std::uint64_t>({cdiv(160, wgs), cdiv(p.lk, 8 * c.bc), 64});
    }
    const std::uint64_t chunk = cdiv(cdiv(p.lk, nsplit), c.bc) * c.bc;
    nsplit = cdiv(p.lk, chunk);
    const bool split = nsplit > 1;
    const bool direct = !split && p.dt == Dtype::FP16 && p.hd % 16 == 0 && p.o % 16 == 0 && p.ldo % 8 == 0;
    auto kernel = [&](bool dir) -> const Kernel& {
        const std::uint32_t spec[] = {static_cast<std::uint32_t>(p.hd), c.bc, c.nsg, c.nsg * 32,
                                      static_cast<std::uint32_t>(p.mode), p.mask ? 1u : 0u, vec ? 1u : 0u,
                                      dir ? 1u : 0u, split ? 1u : 0u};
        return d.pipelines().get(id, spec, static_cast<std::uint32_t>(std::size(spec)), 32);
    };
    FaPush pc = make_push(p);
    pc.scale = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(p.hd)));
    pc.chunk = static_cast<std::uint32_t>(chunk);
    Tensor part;
    if (split) {
        part = partials(d, p, nsplit);
        pc.part = addr(part.data);
    }
    const std::uint32_t full = direct ? static_cast<std::uint32_t>(p.lq) / br : 0;
    auto dispatch = [&](const Kernel& k, std::uint32_t b0, std::uint32_t n) {
        for (std::uint32_t s = 0; s < n; s += 65535) {
            pc.qb0 = b0 + s;
            launch(d, k, pc, std::min<std::uint32_t>(65535, n - s), static_cast<std::uint32_t>(p.hq),
                   static_cast<std::uint32_t>(nsplit));
        }
    };
    if (full > 0) dispatch(kernel(true), 0, full);
    if (full < nqb) dispatch(kernel(false), full, nqb - full);
    if (split) combine(d, p, pc.part, nsplit);
}

// ─── fa_rows.comp (+ fa_combine.comp) ──────────────────────────────────────

void run_rows(DeviceCtx& d, const AttnProblem& p) {
    const int group = p.hq / p.hkv;
    const int slots = (p.hd + 127) / 128;
    // Query heads per workgroup: the largest divisor of the group whose
    // queries fit the shared stage and whose accumulators stay in registers.
    int G = 1;
    for (int g = std::min(group, 16); g >= 1; --g) {
        if (group % g == 0 && g * slots <= 16 && g * p.hd <= 8192) { G = g; break; }
    }
    if (p.hd > 8192) fail(p.op, "head_dim above 8192");
    const std::uint64_t entries = static_cast<std::uint64_t>(p.lq) * (p.hq / G);
    // Split the keys until the grid has BROTENSOR_VK_FA_WGS workgroups
    // (default 80: a long cache streams best in a few long splits; 160 and
    // up lost 5-30% on the 16k-32k decode shapes), at least one 128-key tile
    // per split.
    static const std::uint64_t target = [] {
        const char* e = std::getenv("BROTENSOR_VK_FA_WGS");
        return static_cast<std::uint64_t>(e && std::atoi(e) > 0 ? std::atoi(e) : 80);
    }();
    std::uint64_t nsplit = 1;
    if (entries < target && p.lk > 128) {
        nsplit = std::min<std::uint64_t>({cdiv(target, entries), cdiv(p.lk, 128), 64});
    }
    std::uint64_t chunk = cdiv(cdiv(p.lk, nsplit), 128) * 128;
    nsplit = std::max<std::uint64_t>(1, cdiv(p.lk, chunk));
    if (p.lk == 0) chunk = 128;

    const bool vec = p.hd % 8 == 0 && p.k % 16 == 0 && p.ldk % 8 == 0 &&
                     (p.dt != Dtype::FP32 || p.ldk % 4 == 0);
    const std::uint32_t spec[] = {static_cast<std::uint32_t>(p.hd), static_cast<std::uint32_t>(G),
                                  static_cast<std::uint32_t>(p.mode), p.mask ? 1u : 0u, nsplit > 1 ? 1u : 0u,
                                  p.softcap > 0.0f ? 1u : 0u, vec ? 1u : 0u, p.maskpos ? 1u : 0u};
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::fa_rows_f32, p.dt, p.op), spec,
                                        static_cast<std::uint32_t>(std::size(spec)));
    FaPush pc = make_push(p);
    pc.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.hd)));
    pc.chunk = static_cast<std::uint32_t>(chunk);
    Tensor part;
    if (nsplit > 1) {
        part = partials(d, p, nsplit);
        pc.part = addr(part.data);
    }
    for (int r0 = 0; r0 < p.lq; r0 += 65535) {
        pc.qb0 = static_cast<std::uint32_t>(r0);
        launch(d, k, pc, static_cast<std::uint32_t>(nsplit), static_cast<std::uint32_t>(std::min(65535, p.lq - r0)),
               static_cast<std::uint32_t>(p.hq / G));
    }
    if (nsplit > 1) combine(d, p, pc.part, nsplit);
}

}  // namespace

void set_attention_override(int mode) { g_override.store(mode, std::memory_order_relaxed); }

void set_attention_cm_config(int bc, int nsg) {
    g_cm_bc.store(bc, std::memory_order_relaxed);
    g_cm_nsg.store(nsg, std::memory_order_relaxed);
}

const char* attention_path(DeviceCtx& d, const AttnProblem& p) {
    switch (choose(d, p)) {
        case Path::Cm: return "coopmat";
        case Path::Dense: return "dense";
        default: return "rows";
    }
}

void attention(DeviceCtx& d, const AttnProblem& p) {
    if (p.hq <= 0 || p.hkv <= 0 || p.hq % p.hkv != 0 || p.hd <= 0) fail(p.op, "bad head configuration");
    if (p.lq <= 0) return;
    // 32-bit element indices in the kernels (no buffer exceeds 4 GiB).
    auto fits = [](long long rows, long long ld) { return rows * ld < (1LL << 32); };
    if (!fits(p.lq, p.ldq) || !fits(p.lk, p.ldk) || !fits(p.lq, p.ldo)) fail(p.op, "operand above 4 GiB");
    switch (choose(d, p)) {
        case Path::Cm: run_cm(d, p); break;
        case Path::Dense: dense_attention(d, p); break;
        default: run_rows(d, p); break;
    }
}

// ─── ops ────────────────────────────────────────────────────────────────────

namespace {

void need_qkv(const Tensor& Q, const Tensor& K, const Tensor& V, const char* op) {
    dt_code(Q.dtype, op);
    if (K.dtype != Q.dtype || V.dtype != Q.dtype) fail(op, "Q, K, V dtype must match");
}

void ensure_out(Tensor& O, int rows, int cols, Dtype dt) {
    if (O.rows != rows || O.cols != cols || O.dtype != dt) O.resize(rows, cols, dt);
}

AttnProblem base(const char* op, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, int hq, int hkv) {
    AttnProblem p;
    p.op = op;
    p.dt = Q.dtype;
    p.q = addr(Q.data); p.k = addr(K.data); p.v = addr(V.data); p.o = addr(O.data);
    p.lq = Q.rows;
    p.lk = K.rows;
    p.ldq = Q.cols; p.ldk = K.cols; p.ldo = O.cols;
    p.hq = hq;
    p.hkv = hkv;
    p.hd = Q.cols / hq;
    return p;
}

// Validates a (Q, K, V) head layout: Dq = hq * hd, Dkv = hkv * hd. Returns hkv.
int kv_heads(const Tensor& Q, const Tensor& K, const Tensor& V, int hq, const char* op) {
    if (hq <= 0 || Q.cols % hq != 0) fail(op, "num_heads must divide Q.cols");
    const int hd = Q.cols / hq;
    if (V.cols != K.cols || V.rows != K.rows) fail(op, "shape mismatch");
    if (hd == 0 || K.cols == 0 || K.cols % hd != 0) fail(op, "K/V width must be a head_dim multiple");
    const int hkv = K.cols / hd;
    if (hq % hkv != 0) fail(op, "num_heads must be a multiple of the KV heads");
    return hkv;
}

void zero_rows(Tensor& O) {
    if (O.size() > 0) O.zero();
}

}  // namespace

void flash_attention_windowed_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const float* d_mask,
                                      int num_heads, int window, Tensor& O, bool causal) {
    constexpr const char* op = "flash_attention_windowed_forward";
    need_qkv(Q, K, V, op);
    const int hkv = kv_heads(Q, K, V, num_heads, op);
    if ((causal || window > 0) && K.rows < Q.rows) fail(op, "causal / windowed attention requires Lk >= Lq");
    ensure_out(O, Q.rows, Q.cols, Q.dtype);
    if (Q.rows == 0 || Q.cols == 0) return;
    if (K.rows == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K, V, O, num_heads, hkv);
    p.mask = addr(d_mask);
    p.causal = causal;
    p.window = window;
    p.q_offset = K.rows - Q.rows;
    attention(device_of(Q), p);
}

void flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const float* d_mask, int num_heads,
                             bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_forward";
    need_qkv(Q, K, V, op);
    if (K.cols != Q.cols) fail(op, "shape mismatch");
    kv_heads(Q, K, V, num_heads, op);
    if (causal && Q.rows != K.rows) fail(op, "causal requires Lq == Lk");
    ensure_out(O, Q.rows, Q.cols, Q.dtype);
    if (Q.rows == 0 || Q.cols == 0) return;
    if (K.rows == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K, V, O, num_heads, num_heads);
    p.mask = addr(d_mask);
    p.causal = causal;
    attention(device_of(Q), p);
}

void flash_attention_gqa_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const float* d_mask,
                                 int num_q_heads, int num_kv_heads, bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_gqa_forward";
    need_qkv(Q, K, V, op);
    if (num_q_heads <= 0 || num_kv_heads <= 0) fail(op, "head counts must be positive");
    const int hkv = kv_heads(Q, K, V, num_q_heads, op);
    if (hkv != num_kv_heads) fail(op, "K/V width must be num_kv_heads * head_dim");
    if (K.rows < Q.rows) fail(op, "requires Lk >= Lq");
    if (causal && Q.rows != K.rows) fail(op, "causal requires Lq == Lk");
    ensure_out(O, Q.rows, Q.cols, Q.dtype);
    if (Q.rows == 0 || Q.cols == 0) return;
    if (K.rows == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K, V, O, num_q_heads, num_kv_heads);
    p.mask = addr(d_mask);
    p.causal = causal;
    attention(device_of(Q), p);
}

void flash_attention_varlen_forward(const Tensor& Q, const Tensor& K, const Tensor& V, const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k, int batch_size, int /*max_seqlen_q*/,
                                    int /*max_seqlen_k*/, int num_heads, int head_dim, bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_varlen_forward";
    need_qkv(Q, K, V, op);
    if (num_heads <= 0 || head_dim <= 0) fail(op, "num_heads / head_dim must be positive");
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != K.rows) fail(op, "shape mismatch");
    if (batch_size < 0) fail(op, "batch_size must be non-negative");
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) fail(op, "cu_seqlens_q/k required when batch_size > 0");
    ensure_out(O, Q.rows, D, Q.dtype);
    if (Q.rows == 0 || D == 0) return;
    // Rows outside every sequence (and sequences without keys) come out zero.
    if (batch_size == 0 || K.rows == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K, V, O, num_heads, num_heads);
    p.mode = FA_MODE_VARLEN;
    p.aux = addr(cu_seqlens_q);
    p.aux2 = addr(cu_seqlens_k);
    p.nseq = batch_size;
    p.causal = causal;
    attention(device_of(Q), p);
}

void flash_attention_packed_qkv_forward(const Tensor& QKV, const Tensor& seq_bounds, int num_heads, int window,
                                        Tensor& O) {
    constexpr const char* op = "flash_attention_packed_qkv_forward";
    dt_code(QKV.dtype, op);
    if (num_heads <= 0 || QKV.cols % (3 * num_heads) != 0) fail(op, "QKV.cols must be 3 * num_heads * head_dim");
    if (seq_bounds.dtype != Dtype::INT32 || seq_bounds.rows != QKV.rows || seq_bounds.cols != 2) {
        fail(op, "seq_bounds must be (L, 2) INT32");
    }
    const int L = QKV.rows, D = QKV.cols / 3;
    ensure_out(O, L, D, QKV.dtype);
    if (L == 0 || D == 0) return;
    const std::uint64_t base_addr = addr(QKV.data);
    const std::uint32_t es = esize(QKV.dtype);
    AttnProblem p;
    p.op = op;
    p.dt = QKV.dtype;
    p.q = base_addr;
    p.k = base_addr + static_cast<std::uint64_t>(D) * es;
    p.v = base_addr + 2ull * D * es;
    p.o = addr(O.data);
    p.lq = p.lk = L;
    p.ldq = p.ldk = 3 * D;
    p.ldo = D;
    p.hq = p.hkv = num_heads;
    p.hd = D / num_heads;
    p.mode = FA_MODE_PACKED;
    p.aux = addr(seq_bounds.data);
    p.window = window;
    attention(device_of(QKV), p);
}

// ─── decode over a KV cache ────────────────────────────────────────────────

void kv_cache_append(const Tensor& K_new, const Tensor& V_new, int cur_len, Tensor& K_cache, Tensor& V_cache) {
    constexpr const char* op = "kv_cache_append";
    dt_code(K_new.dtype, op);
    if (V_new.dtype != K_new.dtype || K_cache.dtype != K_new.dtype || V_cache.dtype != K_new.dtype) {
        fail(op, "all tensors must share one dtype");
    }
    if (K_new.cols != V_new.cols || K_new.cols != K_cache.cols || K_cache.cols != V_cache.cols) {
        fail(op, "column mismatch");
    }
    if (K_new.rows != V_new.rows) fail(op, "K_new/V_new row mismatch");
    if (K_cache.rows != V_cache.rows) fail(op, "K_cache/V_cache row mismatch");
    if (cur_len < 0 || cur_len + K_new.rows > K_cache.rows) fail(op, "cur_len + L_new exceeds cache capacity");
    const std::size_t n = static_cast<std::size_t>(K_new.rows) * K_new.cols * esize(K_new.dtype);
    if (n == 0) return;
    DeviceCtx& d = device_of(K_new);
    const std::size_t off = static_cast<std::size_t>(cur_len) * K_cache.cols * esize(K_new.dtype);
    auto copy = [&](const Tensor& src, Tensor& dst) {
        const Span a = d.allocator().resolve(addr(src.data), n);
        const Span b = d.allocator().resolve(addr(dst.data) + off, n);
        d.stream().copy(a.buf, a.offset, b.buf, b.offset, n);
    };
    copy(K_new, K_cache);
    copy(V_new, V_cache);
}

namespace {

void check_decode_heads(const Tensor& Q, const Tensor& K, const Tensor& V, int hq, int hkv, const char* op) {
    need_qkv(Q, K, V, op);
    if (V.cols != K.cols) fail(op, "K_cache.cols != V_cache.cols");
    if (hq <= 0 || hkv <= 0) fail(op, "num_q_heads / num_kv_heads must be positive");
    if (hq % hkv != 0) fail(op, "num_kv_heads must divide num_q_heads");
    if (Q.cols % hq != 0 || K.cols % hkv != 0) fail(op, "head_dim does not divide cols cleanly");
    if (K.cols / hkv != Q.cols / hq) fail(op, "head_dim mismatch between Q and K/V");
}

}  // namespace

void flash_attention_decode(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, int valid_len,
                            int num_q_heads, int num_kv_heads, Tensor& O, float attn_softcap, int window) {
    constexpr const char* op = "flash_attention_decode";
    check_decode_heads(Q, K_cache, V_cache, num_q_heads, num_kv_heads, op);
    if (valid_len < 0 || valid_len > K_cache.rows || valid_len > V_cache.rows) fail(op, "invalid valid_len");
    if (valid_len < Q.rows) fail(op, "valid_len must be >= Lq");
    ensure_out(O, Q.rows, Q.cols, Q.dtype);
    if (Q.rows == 0 || Q.cols == 0) return;
    if (valid_len == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K_cache, V_cache, O, num_q_heads, num_kv_heads);
    p.lk = valid_len;
    p.causal = true;
    p.window = window;
    p.q_offset = valid_len - Q.rows;
    p.softcap = attn_softcap;
    attention(device_of(Q), p);
}

void flash_attention_decode_masked(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, const float* d_mask,
                                   int num_q_heads, int num_kv_heads, Tensor& O, float attn_softcap, int window) {
    constexpr const char* op = "flash_attention_decode_masked";
    check_decode_heads(Q, K_cache, V_cache, num_q_heads, num_kv_heads, op);
    if (!d_mask) fail(op, "d_mask must not be null");
    if (Q.rows != 1) fail(op, "Q must be a single row (L_q == 1)");
    if (V_cache.rows != K_cache.rows) fail(op, "K_cache/V_cache row mismatch");
    ensure_out(O, 1, Q.cols, Q.dtype);
    if (Q.cols == 0) return;
    if (K_cache.rows == 0) return zero_rows(O);
    AttnProblem p = base(op, Q, K_cache, V_cache, O, num_q_heads, num_kv_heads);
    p.mask = addr(d_mask);
    p.softcap = attn_softcap;
    p.window = window;
    p.maskpos = window > 0;
    p.nan_safe = true;   // masked cache slots may be uninitialised (the CPU skips them too)
    attention(device_of(Q), p);
}

// ─── projection-fused ──────────────────────────────────────────────────────

namespace {

Tensor project(const Tensor& X, const Tensor& W, const Tensor* b) {
    Tensor Y = Tensor::empty_on(X.device, X.rows, W.rows, X.dtype);
    linear_forward_batched_ex(W, b, X, 0, 0, nullptr, Y);
    return Y;
}

}  // namespace

void flash_attention_project_kv(const Tensor& ctx, const Tensor& Wk, const Tensor* bk, const Tensor& Wv,
                                const Tensor* bv, Tensor& K_out, Tensor& V_out) {
    constexpr const char* op = "flash_attention_project_kv";
    if (Wk.cols != ctx.cols || Wv.rows != Wk.rows || Wv.cols != ctx.cols) fail(op, "Wk/Wv shape mismatch");
    ensure_out(K_out, ctx.rows, Wk.rows, ctx.dtype);
    ensure_out(V_out, ctx.rows, Wk.rows, ctx.dtype);
    if (ctx.rows == 0 || Wk.rows == 0) return;
    linear_forward_batched_ex(Wk, bk, ctx, 0, 0, nullptr, K_out);
    linear_forward_batched_ex(Wv, bv, ctx, 0, 0, nullptr, V_out);
}

void flash_attention_q_with_kv_cached_forward(const Tensor& X, const Tensor& K, const Tensor& V, const Tensor& Wq,
                                              const Tensor* bq, const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask, int num_heads, bool causal, Tensor& O) {
    constexpr const char* op = "flash_attention_q_with_kv_cached_forward";
    const int D = Wq.rows;
    if (Wq.cols != X.cols || Wo.rows != D || Wo.cols != D || K.cols != D || V.cols != D || V.rows != K.rows) {
        fail(op, "shape mismatch");
    }
    ensure_out(O, X.rows, D, X.dtype);
    if (X.rows == 0 || D == 0) return;
    const Tensor Q = project(X, Wq, bq);
    Tensor A = Tensor::empty_on(X.device, X.rows, D, X.dtype);   // internal calls skip adopt_output
    flash_attention_forward(Q, K, V, d_mask, num_heads, causal, A);
    linear_forward_batched_ex(Wo, bo, A, 0, 0, nullptr, O);
}

void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx, const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk, const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo, const float* d_mask, int num_heads, bool causal,
                                  Tensor& O) {
    constexpr const char* op = "flash_attention_qkvo_forward";
    const Tensor& kv_in = (Ctx && Ctx->size() > 0) ? *Ctx : X;
    const int D = Wq.rows;
    if (Wq.cols != X.cols || Wk.rows != D || Wv.rows != D || Wk.cols != kv_in.cols || Wv.cols != kv_in.cols ||
        Wo.rows != D || Wo.cols != D) {
        fail(op, "shape mismatch");
    }
    ensure_out(O, X.rows, D, X.dtype);
    if (X.rows == 0 || D == 0) return;
    const Tensor Q = project(X, Wq, bq), Kp = project(kv_in, Wk, bk), Vp = project(kv_in, Wv, bv);
    Tensor A = Tensor::empty_on(X.device, X.rows, D, X.dtype);   // internal calls skip adopt_output
    flash_attention_forward(Q, Kp, Vp, d_mask, num_heads, causal, A);
    linear_forward_batched_ex(Wo, bo, A, 0, 0, nullptr, O);
}

void fill_vulkan_vtable_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward = &flash_attention_forward;
    v.flash_attention_gqa_forward = &flash_attention_gqa_forward;
    v.flash_attention_windowed_forward = &flash_attention_windowed_forward;
    v.flash_attention_varlen_forward = &flash_attention_varlen_forward;
    v.flash_attention_packed_qkv_forward = &flash_attention_packed_qkv_forward;
    v.kv_cache_append = &kv_cache_append;
    v.flash_attention_decode = &flash_attention_decode;
    v.flash_attention_decode_masked = &flash_attention_decode_masked;
    v.flash_attention_project_kv = &flash_attention_project_kv;
    v.flash_attention_q_with_kv_cached_forward = &flash_attention_q_with_kv_cached_forward;
    v.flash_attention_qkvo_forward = &flash_attention_qkvo_forward;
}

}  // namespace brotensor::detail::vulkan
