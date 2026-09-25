// The A @ B^T launchers behind matmul_abt and the batched linears / attention
// GEMMs (see fp16_matmul.h). Every input type with K > 0 rides the simdgroup
// GEMM in gemm_fp32.mm (half tiles for FP16 x FP16, float tiles otherwise,
// FP32 accumulate throughout); small outputs and K == 0 take the
// one-thread-per-output naive kernel below, which carries the same contract.

#include "fp16_matmul.h"

#import "internal.h"

#include <mutex>
#include <stdexcept>

namespace brotensor::metal_impl {

namespace {

constexpr size_t kTiledMin = 1024;   // below this M*N (and M > 8), the naive path wins

// The public matmul_abt contract (row-major A (M, K), B (N, K), C (M, N) per
// batch matrix) as an AbtMixed.
AbtMixed abt_plain(AbtType t, id<MTLBuffer> A, NSUInteger ofs_A, id<MTLBuffer> B, NSUInteger ofs_B,
                   id<MTLBuffer> C, NSUInteger ofs_C, int batch, int M, int N, int K, uint64_t strideA,
                   uint64_t strideB, uint64_t strideC, id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                   int act) {
    AbtMixed g;
    g.A = A; g.ofs_A = ofs_A; g.lda = static_cast<uint64_t>(K);
    g.B = B; g.ofs_B = ofs_B; g.ldb = static_cast<uint64_t>(K);
    g.C = C; g.ofs_C = ofs_C; g.ldc = static_cast<uint64_t>(N);
    if (has_bias) {
        g.bias = bias;
        g.ofs_bias = ofs_bias;
    }
    g.M = M; g.N = N; g.K = K < 0 ? 0 : K;
    g.in = g.out = t;
    g.act = act;
    g.batch = batch;
    g.strideA = strideA; g.strideB = strideB; g.strideC = strideC;
    return g;
}

} // namespace

void launch_matmul_abt_fp16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    launch_matmul_abt_mixed(abt_plain(kAbtF16, A, ofs_A, B, ofs_B, C, ofs_C, batch, M, N, K, strideA, strideB,
                                      strideC, bias, ofs_bias, has_bias, act));
}

void launch_matmul_abt_bf16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act) {
    launch_matmul_abt_mixed(abt_plain(kAbtBF16, A, ofs_A, B, ofs_B, C, ofs_C, batch, M, N, K, strideA, strideB,
                                      strideC, bias, ofs_bias, has_bias, act));
}

// ─── Mixed-precision A @ B^T (see fp16_matmul.h) ────────────────────────────

namespace {

NSString* const kMixSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct MixParams {
    uint  M, N, K, has_bias;
    int   act, epi;
    float alpha;
    uint  trans;     // bit 0: A stored (K, M); bit 1: B stored (K, N)
    ulong lda, ldb, ldc, sA, sB, sC;
};

// Element (row, k) of an operand stored row-major, or transposed.
inline ulong mix_at(bool t, uint row, uint k, ulong ld) { return t ? ulong(k) * ld + row : ulong(row) * ld + k; }

// A & S 7.1.26, |err| < 1.5e-7 (MSL has no erf).
inline float mix_erf(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return x < 0.0f ? -y : y;
}
inline float mix_gelu_erf(float v) { return 0.5f * v * (1.0f + mix_erf(v * 0.70710678118654752f)); }
inline float mix_act(int act, float v) {
    switch (act) {
        case 1: return max(v, 0.0f);
        case 2: {
            const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: return mix_gelu_erf(v);
        case 4: return v / (1.0f + exp(-v));
        case 5: return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}

// r = act(alpha * acc + bias[n]); the epilogue then stores / adds / gates it.
template <typename TI>
inline float mix_pre(float acc, uint n, device const TI* bias, constant MixParams& p) {
    float v = acc * p.alpha;
    if (p.has_bias) v += float(bias[n]);
    return mix_act(p.act, v);
}

template <typename TI, typename TO>
kernel void k_abt_mixed_naive(device const TI* A    [[buffer(0)]],
                              device const TI* B    [[buffer(1)]],
                              device TO*       C    [[buffer(2)]],
                              device const TI* bias [[buffer(3)]],
                              constant MixParams& p [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const bool geglu = p.epi == 2;
    const uint cols = geglu ? p.N / 2 : p.N;
    if (gid.x >= p.M * cols) return;
    const uint m = gid.x / cols, j = gid.x % cols;
    const bool ta = (p.trans & 1u) != 0u, tb = (p.trans & 2u) != 0u;
    device const TI* Ab = A + (ulong)gid.y * p.sA;
    device const TI* Bb = B + (ulong)gid.y * p.sB;
    device TO* c = C + (ulong)gid.y * p.sC + (ulong)m * p.ldc + j;
    if (geglu) {
        const uint n0 = 2 * j;
        float a0 = 0.0f, a1 = 0.0f;
        for (uint k = 0; k < p.K; ++k) {
            const float a = float(Ab[mix_at(ta, m, k, p.lda)]);
            a0 += a * float(Bb[mix_at(tb, n0, k, p.ldb)]);
            a1 += a * float(Bb[mix_at(tb, n0 + 1, k, p.ldb)]);
        }
        *c = TO(mix_pre(a0, n0, bias, p) * mix_gelu_erf(mix_pre(a1, n0 + 1, bias, p)));
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) acc += float(Ab[mix_at(ta, m, k, p.lda)]) * float(Bb[mix_at(tb, j, k, p.ldb)]);
    float v = mix_pre(acc, j, bias, p);
    if (p.epi == 1) v += float(*c);
    *c = TO(v);
}

#define MIX_INST(TI, TO, SUF)                                                                           \
template [[host_name("k_abt_mixed_naive_" SUF)]] kernel void k_abt_mixed_naive<TI, TO>(                 \
    device const TI*, device const TI*, device TO*, device const TI*, constant MixParams&, uint2);
MIX_INST(half, half, "0_0")
MIX_INST(half, bfloat, "0_1")
MIX_INST(half, float, "0_2")
MIX_INST(bfloat, half, "1_0")
MIX_INST(bfloat, bfloat, "1_1")
MIX_INST(bfloat, float, "1_2")
MIX_INST(float, half, "2_0")
MIX_INST(float, bfloat, "2_1")
MIX_INST(float, float, "2_2")
)msl";

// Must match the MSL `MixParams` byte-for-byte.
struct MixParams {
    uint32_t M, N, K, has_bias;
    int32_t  act, epi;
    float    alpha;
    uint32_t trans;
    uint64_t lda, ldb, ldc, sA, sB, sC;
};

id<MTLComputePipelineState> pso_naive(int in, int out) {
    static std::once_flag once[3][3];
    static id<MTLComputePipelineState> pso[3][3];
    std::call_once(once[in][out], [&] {
        pso[in][out] = compile_pipeline(kMixSrc, [NSString stringWithFormat:@"k_abt_mixed_naive_%d_%d", in, out]);
    });
    return pso[in][out];
}

} // namespace

AbtType abt_type(::brotensor::Dtype dt) {
    switch (dt) {
        case ::brotensor::Dtype::FP16: return kAbtF16;
        case ::brotensor::Dtype::BF16: return kAbtBF16;
        case ::brotensor::Dtype::FP32: return kAbtF32;
        default: throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: dtype must be FP16, BF16 or FP32");
    }
}

void launch_matmul_abt_mixed(const AbtMixed& g) {
    if (g.batch <= 0 || g.M <= 0 || g.N <= 0) return;
    if (g.epilogue == kAbtGeglu && (g.N % 2) != 0)
        throw std::runtime_error("brotensor (Metal): matmul_abt_mixed: GeGLU needs an even N");
    MixParams p{};
    p.M = static_cast<uint32_t>(g.M);
    p.N = static_cast<uint32_t>(g.N);
    p.K = static_cast<uint32_t>(g.K < 0 ? 0 : g.K);
    p.has_bias = g.bias ? 1u : 0u;
    p.act = g.act;
    p.epi = g.epilogue;
    p.alpha = g.alpha;
    p.trans = (g.transA ? 1u : 0u) | (g.transB ? 2u : 0u);
    p.lda = g.lda;
    p.ldb = g.ldb;
    p.ldc = g.ldc;
    p.sA = g.strideA;
    p.sB = g.strideB;
    p.sC = g.strideC;
    id<MTLBuffer> bBias = g.bias ? g.bias : g.A;
    const NSUInteger oBias = g.bias ? g.ofs_bias : g.ofs_A;

    // Every K > 0 product with M <= 8 (the skinny decode kernel) or a large
    // enough output takes the simdgroup GEMM (gemm_fp32.mm), as does any B
    // typed apart from A, which the naive kernel does not carry.
    const bool mixed_b = g.in_B >= 0 && g.in_B != g.in;
    if (p.K > 0 &&
        (mixed_b || g.M <= 8 || static_cast<size_t>(g.M) * static_cast<size_t>(g.N) >= kTiledMin)) {
        launch_gemm_fp32(g);
        return;
    }
    id<MTLComputePipelineState> pso = pso_naive(g.in, g.out);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:g.A offset:g.ofs_A atIndex:0];
        [enc setBuffer:g.B offset:g.ofs_B atIndex:1];
        [enc setBuffer:g.C offset:g.ofs_C atIndex:2];
        [enc setBuffer:bBias offset:oBias atIndex:3];
        [enc setBytes:&p length:sizeof(MixParams) atIndex:4];
        const NSUInteger cols = g.epilogue == kAbtGeglu ? static_cast<NSUInteger>(g.N / 2)
                                                         : static_cast<NSUInteger>(g.N);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(g.M) * cols, static_cast<NSUInteger>(g.batch), 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::metal_impl
