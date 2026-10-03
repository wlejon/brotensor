// Vulkan-vs-HIP parity: harness, main, and the GEMM / quantised-weight / norm
// groups. See test_vulkan_hip_parity.h for the comparison rule; the attention,
// conv, RoPE / GLU / softmax / sampler and audio groups are in
// test_vulkan_hip_parity_ops.cpp.
//
//   brotensor_test_vulkan_hip_parity [--only=gemm|quant|norm|attention|conv|misc|audio] [--no-cpu]

#include "test_vulkan_hip_parity.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace vhp {

namespace {

struct Row {
    std::string op, shape;
    double vk_hip_abs = 0, vk_hip_rel = 0, cpu_hip = -1, cpu_vk = -1, atol = 0, rtol = 0;
    bool pass = true;
    std::string note;
};

std::vector<Row>& rows() {
    static std::vector<Row> r;
    return r;
}

bool g_cpu = true;

// max |a - b| (NaN / inf mismatches count as infinite); max |ref|.
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool fa = std::isfinite(a[i]), fb = std::isfinite(b[i]);
        if (!fa || !fb) {
            if (fa != fb || (std::isnan(a[i]) != std::isnan(b[i])) || (!std::isnan(a[i]) && a[i] != b[i])) return INFINITY;
            continue;
        }
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    }
    return m;
}

double max_abs(const std::vector<float>& a) {
    double m = 0;
    for (float x : a)
        if (std::isfinite(x)) m = std::max(m, double(std::fabs(x)));
    return m;
}

void print_row(const Row& r) {
    auto num = [](double v, char* buf, std::size_t n) {
        if (v < 0) std::snprintf(buf, n, "%9s", "-");
        else std::snprintf(buf, n, "%9.2e", v);
    };
    char a[16], b[16], c[16], d[16];
    num(r.vk_hip_abs, a, sizeof a);
    num(r.vk_hip_rel, b, sizeof b);
    num(r.cpu_hip, c, sizeof c);
    num(r.cpu_vk, d, sizeof d);
    std::printf("  %-4s %-34s %-30s %s %s  tol %.1e+%.1e*max  cpu: hip %s vk %s%s%s\n", r.pass ? "PASS" : "FAIL",
                r.op.c_str(), r.shape.c_str(), a, b, r.atol, r.rtol, c, d, r.note.empty() ? "" : "  ",
                r.note.c_str());
    std::fflush(stdout);
}

}  // namespace

std::string shp(const char* fmt, ...) {
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

void check_values(const std::string& op, const std::string& shape, const std::vector<float>& hip,
                  const std::vector<float>& vk, const std::vector<float>* cpu, double atol, double rtol) {
    Row r;
    r.op = op;
    r.shape = shape;
    r.atol = atol;
    r.rtol = rtol;
    const double range = std::max(max_abs(hip), 1e-30);
    r.vk_hip_abs = max_abs_diff(vk, hip);
    r.vk_hip_rel = r.vk_hip_abs / range;
    if (cpu) {
        r.cpu_hip = max_abs_diff(hip, *cpu);
        r.cpu_vk = max_abs_diff(vk, *cpu);
    }
    r.pass = r.vk_hip_abs <= atol + rtol * range;
    print_row(r);
    rows().push_back(r);
}

void check(const std::string& op, const std::string& shape, const GpuFn& gpu, const CpuFn& cpu, double atol,
           double rtol) {
    std::vector<float> h, v, c;
    try {
        h = gpu(kHip);
        v = gpu(kVk);
    } catch (const std::exception& e) {
        Row r;
        r.op = op;
        r.shape = shape;
        r.pass = false;
        r.vk_hip_abs = r.vk_hip_rel = INFINITY;
        r.note = std::string("threw: ") + e.what();
        print_row(r);
        rows().push_back(r);
        return;
    }
    bool have_cpu = false;
    if (cpu && g_cpu) {
        try {
            c = cpu();
            have_cpu = true;
        } catch (const std::exception&) {
            // The CPU backend lacks this form: no attribution column.
        }
    }
    check_values(op, shape, h, v, have_cpu ? &c : nullptr, atol, rtol);
}

void check_exact(const std::string& op, const std::string& shape, int mismatches, int n, int allowed) {
    Row r;
    r.op = op;
    r.shape = shape;
    r.vk_hip_abs = mismatches;
    r.vk_hip_rel = n ? double(mismatches) / n : 0;
    r.pass = mismatches <= allowed;
    r.note = shp("%d of %d differ (allowed %d)", mismatches, n, allowed);
    print_row(r);
    rows().push_back(r);
}

// ─── GEMM ───────────────────────────────────────────────────────────────────

namespace {

constexpr double kF16 = 1.0 / 1024.0, kBF16 = 1.0 / 128.0;

// FP32 CPU linear: Y = X W^T + b.
std::vector<float> cpu_linear(const std::vector<float>& w, const std::vector<float>* b, const std::vector<float>& x,
                              int N, int K, int B) {
    std::vector<float> zero(N, 0.0f);
    Tensor Y;
    brotensor::linear_forward_batched(up(w, N, K, Dtype::FP32, kCpu), up(b ? *b : zero, N, 1, Dtype::FP32, kCpu),
                                      up(x, B, K, Dtype::FP32, kCpu), Y);
    return Y.to_host_vector();
}

void gemm_16(Dtype dt, int N, int K, int B, bool bias, const char* what, std::uint64_t seed) {
    const auto w = rnd(std::size_t(N) * K, seed, -0.05f, 0.05f, dt);
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, dt);
    const auto b = rnd(N, seed + 2, -0.5f, 0.5f, dt);
    const double tol = dt == Dtype::BF16 ? 2 * kBF16 : 2 * kF16;
    check(shp("linear_batched_fp16 %s", vkt::dt_name(dt)), shp("%s N%d K%d B%d", what, N, K, B),
          [&](Device d) {
              Tensor Bt = up(b, N, 1, dt, d), Y;
              brotensor::linear_forward_batched_fp16(up(w, N, K, dt, d), bias ? &Bt : nullptr, up(x, B, K, dt, d), Y);
              return down(Y);
          },
          [&] { return cpu_linear(w, bias ? &b : nullptr, x, N, K, B); }, 0, tol);
}

// BF16 checkpoint weights cast at load to each device's compute dtype (FP16
// on both GPUs: the policy cast), then the FP16 GEMM.
void gemm_bf16_at_load(int N, int K, int B, std::uint64_t seed) {
    const auto w = rnd(std::size_t(N) * K, seed, -0.05f, 0.05f, Dtype::BF16);
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, Dtype::FP16);
    check("linear bf16 weight cast at load", shp("N%d K%d B%d -> %s", N, K, B,
                                                 vkt::dt_name(brotensor::compute_dtype(kVk))),
          [&](Device d) {
              Tensor Wb = up(w, N, K, Dtype::BF16, d), W, Y;
              brotensor::cast(Wb, W, brotensor::compute_dtype(d));
              brotensor::linear_forward_batched_fp16(W, nullptr, up(x, B, K, W.dtype, d), Y);
              return down(Y);
          },
          [&] { return cpu_linear(w, nullptr, x, N, K, B); }, 0, 2 * kF16);
}

void gemm_f32(int N, int K, int B, std::uint64_t seed) {
    const auto w = rnd(std::size_t(N) * K, seed, -0.1f, 0.1f, Dtype::FP32);
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, Dtype::FP32);
    const auto b = rnd(N, seed + 2, -0.5f, 0.5f, Dtype::FP32);
    check("linear_batched f32", shp("N%d K%d B%d", N, K, B),
          [&](Device d) {
              Tensor Y;
              brotensor::linear_forward_batched(up(w, N, K, Dtype::FP32, d), up(b, N, 1, Dtype::FP32, d),
                                                up(x, B, K, Dtype::FP32, d), Y);
              return down(Y);
          },
          [&] { return cpu_linear(w, &b, x, N, K, B); }, 0, 2e-5);
}

void matmul_case(Dtype dt, int M, int K, int N, std::uint64_t seed) {
    const auto a = rnd(std::size_t(M) * K, seed, -1.0f, 1.0f, dt);
    const auto b = rnd(std::size_t(K) * N, seed + 1, -0.1f, 0.1f, dt);
    check(shp("matmul %s", vkt::dt_name(dt)), shp("M%d K%d N%d", M, K, N),
          [&](Device d) {
              Tensor C;
              brotensor::matmul(up(a, M, K, dt, d), up(b, K, N, dt, d), C);
              return down(C);
          },
          [&] {
              Tensor C;
              brotensor::matmul(up(a, M, K, dt, kCpu), up(b, K, N, dt, kCpu), C);
              return C.to_host_vector();
          },
          0, dt == Dtype::FP32 ? 2e-5 : 2 * kF16);
}

// The GEMM's fused epilogues (brodiffusion GeGLU, brolm SwiGLU / residual).
void gemm_epilogue(int epi, int N, int K, int B, std::uint64_t seed) {
    const auto w = rnd(std::size_t(N) * K, seed, -0.05f, 0.05f, Dtype::FP16);
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, Dtype::FP16);
    const auto r0 = rnd(std::size_t(B) * N, seed + 2, -1.0f, 1.0f, Dtype::FP16);
    const char* name = epi == brotensor::kLinearEpiGeglu    ? "linear_ex geglu epilogue"
                       : epi == brotensor::kLinearEpiSwiglu ? "linear_ex swiglu epilogue"
                                                            : "linear_ex accumulate epilogue";
    check(name, shp("N%d K%d B%d", N, K, B),
          [&](Device d) {
              Tensor Y;
              if (epi == brotensor::kLinearEpiAccumulate) Y = up(r0, B, N, Dtype::FP16, d);
              brotensor::linear_forward_batched_ex(up(w, N, K, Dtype::FP16, d), nullptr, up(x, B, K, Dtype::FP16, d),
                                                   0, epi, nullptr, Y);
              return down(Y);
          },
          nullptr, 0, 3 * kF16);
}

}  // namespace

void run_gemm() {
    std::printf("\n[gemm]\n");
    // Qwen3-0.6B: fused QKV 4096 x 1024, MLP down 1024 x 3072; decode (B 1) and prefill.
    gemm_16(Dtype::FP16, 4096, 1024, 1, true, "qwen3 qkv decode", 11);
    gemm_16(Dtype::FP16, 4096, 1024, 128, true, "qwen3 qkv prefill", 12);
    gemm_16(Dtype::FP16, 1024, 3072, 128, false, "qwen3 mlp down", 13);
    gemm_16(Dtype::FP16, 1280, 320, 4096, true, "sd1.5 ff 64x64", 14);
    gemm_16(Dtype::BF16, 1024, 1024, 128, true, "bf16 operands", 15);
    gemm_16(Dtype::BF16, 3072, 1024, 7, false, "bf16 short batch", 16);
    gemm_bf16_at_load(2048, 1024, 64, 17);
    gemm_f32(1024, 512, 64, 18);
    gemm_f32(4096, 1024, 1, 19);
    matmul_case(Dtype::FP32, 128, 1024, 1024, 20);   // T5 FP32 activations
    matmul_case(Dtype::FP16, 256, 640, 640, 21);
    gemm_epilogue(brotensor::kLinearEpiGeglu, 2560, 320, 1024, 22);
    gemm_epilogue(brotensor::kLinearEpiSwiglu, 6144, 1024, 64, 23);
    gemm_epilogue(brotensor::kLinearEpiAccumulate, 1024, 1024, 64, 24);
}

// ─── quantised weights ──────────────────────────────────────────────────────

namespace {

std::uint16_t h16(float v) { return brotensor::fp32_to_fp16_bits(v); }

// Random GGUF blocks with sane block scales: every quant field random, so the
// decoders see the full bit range.
std::vector<std::uint8_t> random_gguf(Dtype dt, int rows, int k, std::uint64_t seed) {
    const int blk = brotensor::dtype_block_size(dt), bb = brotensor::dtype_block_bytes(dt);
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (k / blk) * bb);
    vkt::Rng r(seed);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(r.next());
    for (std::size_t o = 0; o < bytes.size(); o += bb) {
        auto put = [&](std::size_t at, float v) {
            const std::uint16_t h = h16(v);
            std::memcpy(&bytes[o + at], &h, 2);
        };
        if (dt == Dtype::Q8_0) put(0, r.uniform(0.002f, 0.01f));
        else if (dt == Dtype::Q4_K) { put(0, r.uniform(5e-4f, 1.5e-3f)); put(2, r.uniform(2e-4f, 8e-4f)); }
        else put(208, r.uniform(1e-4f, 3e-4f));
    }
    return bytes;
}

const char* qname(Dtype dt) { return dt == Dtype::Q8_0 ? "q8_0" : dt == Dtype::Q4_K ? "q4_k" : "q6_k"; }

void dequant(Dtype dt, const Tensor& W, Tensor& Y) {
    if (dt == Dtype::Q8_0) brotensor::dequant_q8_0_to_fp16(W, Y);
    else if (dt == Dtype::Q4_K) brotensor::dequant_q4k_to_fp16(W, Y);
    else brotensor::dequant_q6k_to_fp16(W, Y);
}

void gguf_case(Dtype dt, int N, int K, int B, std::uint64_t seed) {
    const auto bytes = random_gguf(dt, N, K, seed);
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, Dtype::FP16);
    const auto b = rnd(N, seed + 2, -0.5f, 0.5f, Dtype::FP16);
    auto wq = [&](Device d) { return Tensor::from_raw_bytes_on(d, bytes.data(), N, K, dt, bytes.size()); };
    // Decoded weights (HIP's dequant, which the dequant row checks) for the CPU column.
    std::vector<float> wdec;
    {
        Tensor Y;
        dequant(dt, wq(kHip), Y);
        wdec = down(Y);
    }
    if (B == 1) {
        check(shp("dequant_%s_to_fp16", qname(dt)), shp("N%d K%d", N, K),
              [&](Device d) {
                  Tensor Y;
                  dequant(dt, wq(d), Y);
                  return down(Y);
              },
              nullptr, 0, kF16);
    }
    check(shp("linear_batched_%s_fp16", qname(dt)), shp("N%d K%d B%d", N, K, B),
          [&](Device d) {
              Tensor Bt = up(b, N, 1, Dtype::FP16, d), Y;
              const Tensor W = wq(d), X = up(x, B, K, Dtype::FP16, d);
              if (dt == Dtype::Q8_0) brotensor::linear_forward_batched_q8_0_fp16(W, &Bt, X, Y);
              else if (dt == Dtype::Q4_K) brotensor::linear_forward_batched_q4k_fp16(W, &Bt, X, Y);
              else brotensor::linear_forward_batched_q6k_fp16(W, &Bt, X, Y);
              return down(Y);
          },
          [&] { return cpu_linear(wdec, &b, x, N, K, B); }, 0, 3 * kF16);
}

void int8_case(Dtype dt, int N, int K, int B, std::uint64_t seed) {
    const auto wsrc = rnd(std::size_t(N) * K, seed, -0.1f, 0.1f, Dtype::FP16);
    std::vector<std::uint16_t> bits(wsrc.size());
    for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = h16(wsrc[i]);
    std::vector<std::int8_t> q(wsrc.size());
    std::vector<float> scale(N);
    brotensor::quantize_int8_per_row_host(bits.data(), N, K, q.data(), scale.data());
    std::vector<float> wdec(q.size());
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < K; ++c) wdec[std::size_t(r) * K + c] = float(q[std::size_t(r) * K + c]) * scale[r];
    const auto x = rnd(std::size_t(B) * K, seed + 1, -1.0f, 1.0f, dt);
    const auto b = rnd(N, seed + 2, -0.5f, 0.5f, dt);
    check(shp("linear_batched_int8w %s", vkt::dt_name(dt)), shp("N%d K%d B%d", N, K, B),
          [&](Device d) {
              Tensor W = Tensor::from_host_int8_on(d, q.data(), N, K);
              Tensor S = Tensor::from_host_on(d, scale.data(), N, 1);
              Tensor Bt = up(b, N, 1, dt, d), Y;
              brotensor::linear_forward_batched_int8w_fp16(W, S, &Bt, up(x, B, K, dt, d), Y);
              return down(Y);
          },
          [&] { return cpu_linear(wdec, &b, x, N, K, B); }, 0, dt == Dtype::BF16 ? 2 * kBF16 : 3 * kF16);
}

}  // namespace

void run_quant() {
    std::printf("\n[quantised weights]\n");
    // Qwen3-0.6B projections: K 1024 (q/k/v/o, gate/up), K 3072 (down).
    std::uint64_t seed = 100;
    for (Dtype dt : {Dtype::Q8_0, Dtype::Q4_K, Dtype::Q6_K}) {
        gguf_case(dt, 2048, 1024, 1, seed++);
        gguf_case(dt, 1024, 3072, 1, seed++);
        gguf_case(dt, 2048, 1024, 4, seed++);
        gguf_case(dt, 2048, 1024, 96, seed++);
    }
    // SD / DiT INT8 W8A16 (HIP takes WMMA INT8 for the GEMM, Vulkan the FP16 tile).
    int8_case(Dtype::FP16, 1280, 320, 1, seed++);
    int8_case(Dtype::FP16, 1280, 320, 1024, seed++);
    int8_case(Dtype::FP16, 3072, 1152, 64, seed++);
    int8_case(Dtype::BF16, 1152, 1152, 64, seed++);
}

// ─── norms ──────────────────────────────────────────────────────────────────

void run_norm() {
    std::printf("\n[norms]\n");
    {   // SD1.5 ResBlock GroupNorm: 320 ch, 64x64, 32 groups.
        const int N = 1, C = 320, H = 64, W = 64, G = 32;
        const auto x = rnd(std::size_t(N) * C * H * W, 300, -2.0f, 3.0f, Dtype::FP16);
        const auto g = rnd(C, 301, 0.5f, 1.5f, Dtype::FP16), bt = rnd(C, 302, -0.5f, 0.5f, Dtype::FP16);
        for (Dtype dt : {Dtype::FP16, Dtype::FP32}) {
            auto run = [&](Device d) {
                Tensor Y;
                brotensor::group_norm_forward(up(x, N, C * H * W, dt, d), up(g, C, 1, dt, d), up(bt, C, 1, dt, d), N,
                                              C, H, W, G, 1e-5f, Y);
                return down(Y);
            };
            check(shp("group_norm_forward %s", vkt::dt_name(dt)), shp("C%d %dx%d g%d", C, H, W, G), run,
                  [&] { return run(kCpu); }, 0, dt == Dtype::FP32 ? 1e-5 : 2 * kF16);
        }
    }
    {   // LayerNorm: ViT-B / DiT rows.
        const int R = 1024, D = 768;
        const auto x = rnd(std::size_t(R) * D, 310, -3.0f, 3.0f, Dtype::FP16);
        const auto g = rnd(D, 311, 0.5f, 1.5f, Dtype::FP16), bt = rnd(D, 312, -0.5f, 0.5f, Dtype::FP16);
        check("layernorm_inference_batched_fp16", shp("R%d D%d", R, D),
              [&](Device d) {
                  Tensor Y;
                  brotensor::layernorm_forward_inference_batched_fp16(up(x, R, D, Dtype::FP16, d),
                                                                      up(g, D, 1, Dtype::FP16, d),
                                                                      up(bt, D, 1, Dtype::FP16, d), Y, 1e-6f);
                  return down(Y);
              },
              [&] {
                  Tensor Y;
                  brotensor::layernorm_forward_inference_batched(up(x, R, D, Dtype::FP32, kCpu),
                                                                 up(g, D, 1, Dtype::FP32, kCpu),
                                                                 up(bt, D, 1, Dtype::FP32, kCpu), Y, 1e-6f);
                  return Y.to_host_vector();
              },
              0, 2 * kF16);
        check("layernorm_inference_batched f32", shp("R%d D%d", R, D),
              [&](Device d) {
                  Tensor Y;
                  brotensor::layernorm_forward_inference_batched(up(x, R, D, Dtype::FP32, d),
                                                                 up(g, D, 1, Dtype::FP32, d),
                                                                 up(bt, D, 1, Dtype::FP32, d), Y, 1e-6f);
                  return down(Y);
              },
              nullptr, 0, 1e-5);
    }
    for (Dtype dt : {Dtype::FP16, Dtype::BF16, Dtype::FP32}) {   // Qwen3 RMSNorm, hidden 1024
        const int R = 256, D = 1024;
        const auto x = rnd(std::size_t(R) * D, 320, -3.0f, 3.0f, dt);
        const auto g = rnd(D, 321, 0.5f, 1.5f, dt);
        auto run = [&](Device d) {
            Tensor Y;
            brotensor::rms_norm_forward(up(x, R, D, dt, d), up(g, D, 1, dt, d), 1e-6f, Y);
            return down(Y);
        };
        check(shp("rms_norm_forward %s", vkt::dt_name(dt)), shp("R%d D%d", R, D), run, [&] { return run(kCpu); }, 0,
              dt == Dtype::FP32 ? 1e-5 : dt == Dtype::BF16 ? 2 * kBF16 : 2 * kF16);
    }
}

}  // namespace vhp

int main(int argc, char** argv) {
    using namespace vhp;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
        else if (std::strcmp(argv[i], "--no-cpu") == 0) g_cpu = false;
    }
    brotensor::init();
    if (!brotensor::is_available(kHip) || !brotensor::is_available(kVk)) {
        std::printf("SKIP: needs both a HIP and a Vulkan device\n");
        return 77;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::printf("Vulkan vs HIP parity. Columns: |vk-hip| max abs, max abs / max|hip|; cpu: max |gpu-cpu| (FP32 CPU on "
                "the same rounded inputs)\n");
    struct Group { const char* name; void (*fn)(); };
    const Group groups[] = {{"gemm", run_gemm}, {"quant", run_quant},         {"norm", run_norm},
                            {"attention", run_attention}, {"conv", run_conv}, {"misc", run_misc},
                            {"audio", run_audio}};
    for (const Group& g : groups) {
        if (!only.empty() && only != g.name) continue;
        const auto g0 = std::chrono::steady_clock::now();
        g.fn();
        std::printf("  (%s: %.1f s)\n", g.name,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count());
    }
    int fails = 0;
    for (const auto& r : rows()) fails += r.pass ? 0 : 1;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%zu checks, %d failed, %.1f s\n", rows().size(), fails, s);
    if (fails) {
        std::printf("Failed:\n");
        for (const auto& r : rows())
            if (!r.pass) std::printf("  %s %s\n", r.op.c_str(), r.shape.c_str());
    }
    return fails ? 1 : 0;
}
