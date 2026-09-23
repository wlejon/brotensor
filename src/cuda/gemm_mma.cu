// Tensor-core GEMM for the 16-bit linear — see gemm_mma.cuh.
//
// mma.sync.m16n8k16 with FP32 accumulation. K advances in 32-wide tiles
// through a STAGES-deep cp.async ring in shared memory; each tile row is 64
// bytes with its four 16-byte chunks XOR-swizzled by (row >> 1) & 3, which
// keeps every ldmatrix (8 rows x one chunk) bank-conflict free without
// padding. Two tile shapes:
//
//   large  128 x 128, 8 warps (64 x 32 warp tiles), 3 stages — 48 KB, 2 CTAs/SM
//   small   64 x  64, 4 warps (32 x 32 warp tiles), 4 stages — 32 KB
//
// The small shape keeps enough CTAs in flight to stream the weights at full
// bandwidth when M is a short packed batch; the large one takes over once it
// alone fills the machine.

#include "gemm_mma.cuh"

#include "detail/activations.cuh"
#include "detail/cuda_check.h"

#include <algorithm>
#include <cstdint>

namespace brotensor::detail::cuda::mma_gemm {

namespace {

constexpr int BK = 32;  // halves per k-tile row: 64 bytes = 4 x 16-byte chunks

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

// Element offset of 16-byte chunk `chunk` of row `row` in a [rows][BK] tile.
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3); }

template <typename T> struct Ops;
template <> struct Ops<__half> {
    using T2 = __half2;
    __device__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static float to_f(__half v) { return __half2float(v); }
    __device__ static __half from_f(float v) { return __float2half(v); }
    __device__ static T2 pack(float a, float b) { return __floats2half2_rn(a, b); }
    __device__ static float2 unpack(T2 v) { return __half22float2(v); }
};
template <> struct Ops<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    __device__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
    __device__ static T2 pack(float a, float b) { return __floats2bfloat162_rn(a, b); }
    __device__ static float2 unpack(T2 v) { return __bfloat1622float2(v); }
};

// The epilogue of one output column pair (col, col + 1) of one row, from FP32
// pre-bias values.
template <typename T>
__device__ __forceinline__ void store_pair(T* __restrict__ C, int row, int col, int N, float v0, float v1,
                                           const T* __restrict__ bias, int act, int epi) {
    using O = Ops<T>;
    if (bias) {
        v0 += O::to_f(bias[col]);
        v1 += O::to_f(bias[col + 1]);
    }
    if (epi == kGeglu) {
        C[static_cast<size_t>(row) * (N / 2) + col / 2] = O::from_f(v0 * act_gelu_exact(v1));
        return;
    }
    v0 = apply_linear_act(act, v0);
    v1 = apply_linear_act(act, v1);
    auto* p = reinterpret_cast<typename O::T2*>(C + static_cast<size_t>(row) * N + col);
    if (epi == kAccumulate) {
        const float2 old = O::unpack(*p);
        v0 += old.x;
        v1 += old.y;
    }
    *p = O::pack(v0, v1);
}

// FP16-accumulate mma (m16n8k16, D/C = two half2 registers: row g and row g+8
// of the thread's column pair). Twice the FP32-accumulate tensor rate on
// consumer parts (Ada / Ampere GeForce).
__device__ __forceinline__ void mma_f16acc(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Accumulator tiles, by accumulation mode:
//   kAccF32    FP32 throughout.
//   kAccHybrid each m16n8k16 step runs at the FP16-accumulate rate from a
//              zero C, and its 16-term FP16 result is folded into FP32 — the
//              rounding error stays that of a 16-term FP16 sum instead of
//              growing with K, for two FP32 adds per output pair per step.
template <typename T, int MODE, int MT, int NT> struct Acc;
template <typename T, int MT, int NT> struct Acc<T, kAccF32, MT, NT> {
    float c[MT][NT][4];
    __device__ void zero() {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;
    }
    __device__ void mma(int i, int j, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        Ops<T>::mma(c[i][j], a, b0, b1);
    }
    __device__ float2 get(int i, int j, int h) const { return make_float2(c[i][j][2 * h], c[i][j][2 * h + 1]); }
};
template <int MT, int NT> struct Acc<__half, kAccHybrid, MT, NT> {
    float c[MT][NT][4];
    __device__ void zero() {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;
    }
    __device__ void mma(int i, int j, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        uint32_t p[2] = {0u, 0u};
        mma_f16acc(p, a, b0, b1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&p[h]));
            c[i][j][2 * h] += v.x;
            c[i][j][2 * h + 1] += v.y;
        }
    }
    __device__ float2 get(int i, int j, int h) const { return make_float2(c[i][j][2 * h], c[i][j][2 * h + 1]); }
};

// partial != null: split-K — CTA z accumulates k-tiles [z*kps, (z+1)*kps) and
// writes its raw FP32 partial to partial[z] (M x N); splitk_reduce finishes.
// MIN_CTAS pins the register budget: the large tile needs <= 128 registers
// for its two CTAs per SM, which is worth ~25 % at large M.
template <typename T, int MODE, int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int MIN_CTAS>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MIN_CTAS)
    gemm_kernel(const T* __restrict__ A, const T* __restrict__ W, T* __restrict__ C, int M, int N, int K,
                const T* __restrict__ bias, int act, int epi, float* __restrict__ partial, int kps) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MT = WM / 16, NT = WN / 8;
    constexpr int A_CHUNKS = BM * 4 / THREADS, B_CHUNKS = BN * 4 / THREADS;
    static_assert(NT % 2 == 0 && A_CHUNKS >= 1 && B_CHUNKS >= 1, "tile shape");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* sA = reinterpret_cast<T*>(smem_raw);
    T* sB = sA + STAGES * BM * BK;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int KT_all = (K + BK - 1) / BK;
    const int kt0 = partial ? blockIdx.z * kps : 0;
    const int KT = partial ? min(kps, KT_all - kt0) : KT_all;
    A += static_cast<size_t>(kt0) * BK;  // this split's K offset; row strides stay K
    W += static_cast<size_t>(kt0) * BK;
    const int klim = K - kt0 * BK;

    auto load_stage = [&](int stage, int kt) {
        const int k0 = kt * BK;
        T* a = sA + stage * BM * BK;
        T* b = sB + stage * BN * BK;
#pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            const int c = tid + i * THREADS, row = c >> 2, ch = c & 3;
            const int gr = m0 + row, gk = k0 + ch * 8;
            const bool ok = gr < M && gk < klim;
            cp_async16(a + swz(row, ch), ok ? A + static_cast<size_t>(gr) * K + gk : A, ok);
        }
#pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            const int c = tid + i * THREADS, row = c >> 2, ch = c & 3;
            const int gr = n0 + row, gk = k0 + ch * 8;
            const bool ok = gr < N && gk < klim;
            cp_async16(b + swz(row, ch), ok ? W + static_cast<size_t>(gr) * K + gk : W, ok);
        }
    };

    Acc<T, MODE, MT, NT> acc;
    acc.zero();

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_stage(s, s);
        cp_async_commit();
    }

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk % STAGES, nk);
            cp_async_commit();
        }
        const T* a = sA + (kt % STAGES) * BM * BK;
        const T* b = sB + (kt % STAGES) * BN * BK;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t af[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ldsm_x4(af[i], a + swz(wm * WM + i * 16 + (lane & 15), kk * 2 + (lane >> 4)));
            }
            uint32_t bf[NT][2];
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
                const int mi = lane >> 3;
                uint32_t r[4];
                ldsm_x4(r, b + swz(wn * WN + j * 16 + (mi >> 1) * 8 + (lane & 7), kk * 2 + (mi & 1)));
                bf[2 * j][0] = r[0];
                bf[2 * j][1] = r[1];
                bf[2 * j + 1][0] = r[2];
                bf[2 * j + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) acc.mma(i, j, af[i], bf[j][0], bf[j][1]);
        }
    }
    cp_async_wait<0>();

    // Epilogue straight from the accumulators: each thread owns column pairs
    // (2t, 2t+1) of rows g and g + 8 in every 16 x 8 fragment.
    const int g = lane >> 2, t = lane & 3;
    float* P = partial ? partial + static_cast<size_t>(blockIdx.z) * M * N : nullptr;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int col = n0 + wn * WN + j * 8 + 2 * t;
        if (col >= N) continue;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * WM + i * 16 + g + h * 8;
                if (row >= M) continue;
                const float2 v = acc.get(i, j, h);
                if (P) {
                    *reinterpret_cast<float2*>(P + static_cast<size_t>(row) * N + col) = v;
                } else {
                    store_pair<T>(C, row, col, N, v.x, v.y, bias, act, epi);
                }
            }
        }
    }
#endif
}

// Sum the split partials in split order, then the epilogue. One thread per
// output column pair.
template <typename T>
__global__ void splitk_reduce_kernel(const float* __restrict__ partial, int splits, int M, int N,
                                     T* __restrict__ C, const T* __restrict__ bias, int act, int epi) {
    const size_t pairs = static_cast<size_t>(M) * (N / 2);
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const int row = static_cast<int>(p / (N / 2));
    const int col = static_cast<int>(p % (N / 2)) * 2;
    const size_t off = static_cast<size_t>(row) * N + col;
    const size_t stride = static_cast<size_t>(M) * N;
    float v0 = 0.0f, v1 = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float2 x = *reinterpret_cast<const float2*>(partial + s * stride + off);
        v0 += x.x;
        v1 += x.y;
    }
    store_pair<T>(C, row, col, N, v0, v1, bias, act, epi);
}

int sm_count_and_major(int& major) {
    static int cached_sms[16] = {0}, cached_major[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (cached_sms[dev] == 0) {
        cudaDeviceGetAttribute(&cached_sms[dev], cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&cached_major[dev], cudaDevAttrComputeCapabilityMajor, dev);
        if (cached_sms[dev] <= 0) cached_sms[dev] = 1;
    }
    major = cached_major[dev];
    return cached_sms[dev];
}

template <typename T, int MODE, int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int MIN_CTAS>
void run(const T* A, const T* W, T* C, int M, int N, int K, const T* bias, int act, int epi, float* partial,
         int splits, int kps, cudaStream_t stream) {
    constexpr int smem = STAGES * (BM + BN) * BK * static_cast<int>(sizeof(T));
    if constexpr (smem > 48 * 1024) {
        static bool opted_in = false;  // per process; the attribute is per function
        if (!opted_in) {
            BROTENSOR_CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<T, MODE, BM, BN, WARPS_M, WARPS_N, STAGES, MIN_CTAS>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            opted_in = true;
        }
    }
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, splits);
    gemm_kernel<T, MODE, BM, BN, WARPS_M, WARPS_N, STAGES, MIN_CTAS>
        <<<grid, WARPS_M * WARPS_N * 32, smem, stream>>>(A, W, C, M, N, K, bias, act, epi, partial, kps);
}

// Tile shape + K split for one product.
struct Plan {
    bool large = false;
    int splits = 1;
    int kps = 0;  // k-tiles per split
};

Plan make_plan(int M, int N, int K, int sms) {
    Plan p;
    const int KT = (K + BK - 1) / BK;
    p.kps = KT;
    const long long tiles_large = static_cast<long long>((M + 127) / 128) * ((N + 127) / 128);
    if (tiles_large >= sms) {
        p.large = true;
        return p;
    }
    // Small tiles. Under one tile per SM, split along K until ~2 CTAs per SM
    // are busy; every split keeps at least 4 k-tiles so the pipeline still
    // fills. (From one tile per SM up, the extra partial traffic and reduce
    // pass cost more than the idle SMs they would recover.)
    const long long tiles_small = static_cast<long long>((M + 63) / 64) * ((N + 63) / 64);
    const long long want = 2LL * sms;
    if (tiles_small < sms) {
        int s = static_cast<int>((want + tiles_small - 1) / tiles_small);
        s = std::min(s, std::min(16, KT / 4));
        if (s >= 2) {
            p.kps = (KT + s - 1) / s;
            p.splits = (KT + p.kps - 1) / p.kps;
        }
    }
    return p;
}

template <typename T, int MODE>
bool launch_t(const T* A, const T* W, T* C, int M, int N, int K, const T* bias, int act, int epi, float* ws,
              std::size_t ws_floats, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if ((K & 7) || (N & 7)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
    if (misaligned(A) || misaligned(W) || misaligned(C) || misaligned(ws)) return false;
    if (bias && (reinterpret_cast<uintptr_t>(bias) & 3)) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    Plan p = make_plan(M, N, K, sms);
    if (p.splits > 1 && (!ws || ws_floats < static_cast<std::size_t>(p.splits) * M * N)) {
        p.splits = 1;
        p.kps = (K + BK - 1) / BK;
    }
    float* partial = p.splits > 1 ? ws : nullptr;
    if (p.large) {
        run<T, MODE, 128, 128, 2, 4, 3, 2>(A, W, C, M, N, K, bias, act, epi, nullptr, 1, p.kps, stream);
    } else {
        run<T, MODE, 64, 64, 2, 2, 4, 1>(A, W, C, M, N, K, bias, act, epi, partial, p.splits, p.kps, stream);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    if (partial) {
        const std::size_t pairs = static_cast<std::size_t>(M) * (N / 2);
        const int threads = 256;
        splitk_reduce_kernel<T><<<static_cast<unsigned>((pairs + threads - 1) / threads), threads, 0, stream>>>(
            partial, p.splits, M, N, C, bias, act, epi);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

}  // namespace

std::size_t workspace_floats(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    int major = 0;
    const int sms = sm_count_and_major(major);
    const Plan p = make_plan(M, N, K, sms);
    return p.splits > 1 ? static_cast<std::size_t>(p.splits) * M * N : 0;
}

bool launch(const __half* A, const __half* W, __half* C, int M, int N, int K, const __half* bias, int act,
            int epi, float* ws, std::size_t ws_floats, cudaStream_t stream, int acc_mode) {
    if (acc_mode == kAccHybrid)
        return launch_t<__half, kAccHybrid>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream);
    return launch_t<__half, kAccF32>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

bool launch(const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* C, int M, int N, int K,
            const __nv_bfloat16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            cudaStream_t stream, int /*acc_mode: BF16 has no 16-bit-accumulate mma*/) {
    return launch_t<__nv_bfloat16, kAccF32>(A, W, C, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

}  // namespace brotensor::detail::cuda::mma_gemm
