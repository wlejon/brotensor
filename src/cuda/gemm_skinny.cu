// Short-M 16-bit linear on the tensor cores — see gemm_skinny.cuh.
//
// One CTA: 8 warps, 128 output columns (W rows), all 9..64 X rows. K runs in
// 64-wide chunks through a 4-deep cp.async ring in shared memory holding the
// CTA's 128 W rows and the X rows of each chunk.
//
// Why the shape: at a few dozen rows the product streams W once and does
// little else, and what limits that stream on an Ada part is how many bytes
// of one row a single load instruction covers. Sixteen-byte lanes four to a
// row (64 bytes per row per instruction — a 32-wide K tile, what a generic
// tile GEMM uses) top out near 540 GB/s on a 4090; eight to a row (128
// bytes) reach the ~945 GB/s the bus gives a plain sequential read. So the
// ring is filled eight lanes to a row and the chunk is 64 halves wide.
//
// Fragments: warp w owns W rows [16w, 16w + 16) as the mma's A operand (m16)
// and the X rows as its B operand (n8 per tile, up to eight tiles). Lane
// (g, t) = (lane / 4, lane % 4) reads one 16-byte piece of W row g, one of
// row g + 8 and one of X row g of every tile per 32-wide K step, and feeds
// them to the step's two m16n8k16 products under the K permutation
//     logical k (2t + j)     -> physical 8t + 4s + j
//     logical k (2t + 8 + j) -> physical 8t + 4s + 2 + j     (step s, j < 2)
// which visits each physical k of the 32 once, identically in both operands
// (a dot product does not care about the order of its terms). Rows are 128
// bytes in shared memory with their 16-byte pieces XOR-swizzled by
// (row & 1) << 2, so the eight lanes of a load phase (two rows, four pieces
// each) land on disjoint banks.
//
// W goes through L2 with an evict-first hint: it is read once, while X and
// the split partials are what L2 should keep. The accumulators leave through
// a shared [row][column] FP32 tile, so the epilogue and the partial store
// write whole rows of output columns at a time.

#include "gemm_skinny.cuh"

#include "detail/activations.cuh"
#include "detail/cuda_check.h"
#include "detail/smem_opt_in.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace brotensor::detail::cuda::gemm_skinny {

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kCtaCols = kWarps * 16;      // output columns (W rows) per CTA
constexpr int kChunk = 64;                 // K per chunk: 128 bytes per row
constexpr int kStages = 4;                 // cp.async ring depth
constexpr int kTileStride = kCtaCols + 4;  // floats per accumulator tile row

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16-byte async copy, L2 evict-first when `stream` (W) — zero-filled when
// !valid.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async16_stream(void* smem, const void* gmem, bool valid, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(smem_u32(smem)),
                 "l"(gmem), "r"(valid ? 16 : 0), "l"(policy));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Byte offset of 16-byte piece u of row `row` in a ring tile (128-byte rows).
__device__ __forceinline__ int piece(int row, int u) { return row * 128 + ((u ^ ((row & 1) << 2)) << 4); }

template <typename T> struct Ops;
template <> struct Ops<__half> {
    __device__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static float to_f(__half v) { return __half2float(v); }
    __device__ static __half from_f(float v) { return __float2half(v); }
};
template <> struct Ops<__nv_bfloat16> {
    __device__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
};

// The W row behind local column lr of CTA block cb, or -1 past the end. A
// SwiGLU block takes 64 gate rows and the 64 matching up rows, so both
// halves of every output it finishes are in the block.
__device__ __forceinline__ int w_row(int cb, int lr, int N, int epi) {
    if (epi == 3) {
        const int half = N / 2, j = cb * (kCtaCols / 2) + (lr & (kCtaCols / 2 - 1));
        if (j >= half) return -1;
        return lr < kCtaCols / 2 ? j : half + j;
    }
    const int n = cb * kCtaCols + lr;
    return n < N ? n : -1;
}

// One output element from FP32 sums plus bias. store / accumulate take
// (a = r[n]); GeGLU takes (a = r[2j], b = r[2j + 1]); SwiGLU takes
// (a = r[j], b = r[N/2 + j]). `col` is the output column.
template <typename T>
__device__ __forceinline__ void emit(T* __restrict__ Y, int m, int col, int out_cols, float a, float b, int act,
                                     int epi) {
    using O = Ops<T>;
    T* p = Y + static_cast<size_t>(m) * out_cols + col;
    float v;
    if (epi == 2) {
        v = a * act_gelu_exact(b);
    } else if (epi == 3) {
        v = act_silu(a) * b;
    } else {
        v = apply_linear_act(act, a);
        if (epi == 1) v += O::to_f(*p);
    }
    *p = O::from_f(v);
}

template <int NT>
constexpr int smem_bytes() {
    constexpr int ring = kStages * (kCtaCols + NT * 8) * 128;
    constexpr int tile = NT * 8 * kTileStride * 4;
    return ring > tile ? ring : tile;
}

template <typename T, int NT>
__global__ void __launch_bounds__(kThreads, 1)
    skinny_kernel(const T* __restrict__ X, const T* __restrict__ W, T* __restrict__ Y, int M, int N, int K,
                  const T* __restrict__ bias, int act, int epi, float* __restrict__ partial, int chunks_per_split) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    using O = Ops<T>;
    constexpr int ROWS = NT * 8;
    constexpr int STAGE = (kCtaCols + ROWS) * 128;  // bytes: W rows, then X rows
    extern __shared__ __align__(128) unsigned char smem_raw[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int cb = blockIdx.x;
    const int n_chunks = (K + kChunk - 1) / kChunk;
    const int c0 = blockIdx.y * chunks_per_split;
    const int KT = min(n_chunks, c0 + chunks_per_split) - c0;

    uint64_t evict_first;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(evict_first));

    // This thread's ring copies: W pieces c = tid + i * kThreads (row c / 8,
    // piece c % 8) — eight lanes per 128-byte row — and the X pieces alike.
    auto load_stage = [&](int stage, int c) {
        unsigned char* s = smem_raw + stage * STAGE;
        const int kc = c * kChunk;
#pragma unroll
        for (int i = 0; i < kCtaCols * 8 / kThreads; ++i) {
            const int p = tid + i * kThreads, row = p >> 3, u = p & 7;
            const int n = w_row(cb, row, N, epi), k = kc + u * 8;
            const bool ok = n >= 0 && k < K;
            cp_async16_stream(s + piece(row, u), ok ? W + static_cast<size_t>(n) * K + k : W, ok, evict_first);
        }
        for (int p = tid; p < ROWS * 8; p += kThreads) {
            const int row = p >> 3, u = p & 7, k = kc + u * 8;
            const bool ok = row < M && k < K;
            cp_async16(s + kCtaCols * 128 + piece(row, u), ok ? X + static_cast<size_t>(row) * K + k : X, ok);
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < KT) load_stage(s, c0 + s);
        cp_async_commit();
    }
    const int ra = warp * 16 + g, rb = ra + 8;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        {
            const int nk = kt + kStages - 1;
            if (nk < KT) load_stage(nk % kStages, c0 + nk);
            cp_async_commit();
        }
        const unsigned char* s = smem_raw + (kt % kStages) * STAGE;
        const unsigned char* xs = s + kCtaCols * 128;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int u = 4 * q + t;
            const uint4 wa = *reinterpret_cast<const uint4*>(s + piece(ra, u));
            const uint4 wb = *reinterpret_cast<const uint4*>(s + piece(rb, u));
            const uint32_t a0[4] = {wa.x, wb.x, wa.y, wb.y};
            const uint32_t a1[4] = {wa.z, wb.z, wa.w, wb.w};
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 xv = *reinterpret_cast<const uint4*>(xs + piece(j * 8 + g, u));
                O::mma(acc[j], a0, xv.x, xv.y);
                O::mma(acc[j], a1, xv.z, xv.w);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Accumulators -> tile[m][lr] (the ring is free now).
    float* tile = reinterpret_cast<float*>(smem_raw);
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int m = j * 8 + 2 * t;
        tile[m * kTileStride + ra] = acc[j][0];
        tile[(m + 1) * kTileStride + ra] = acc[j][1];
        tile[m * kTileStride + rb] = acc[j][2];
        tile[(m + 1) * kTileStride + rb] = acc[j][3];
    }
    __syncthreads();

    if (partial) {
        float* P = partial + static_cast<size_t>(blockIdx.y) * M * N;
        for (int i = tid; i < M * kCtaCols; i += kThreads) {
            const int m = i / kCtaCols, lr = i % kCtaCols;
            const int n = w_row(cb, lr, N, epi);
            if (n >= 0) P[static_cast<size_t>(m) * N + n] = tile[m * kTileStride + lr];
        }
        return;  // reduce_kernel sums the splits and finishes
    }
    auto bias_of = [&](int n) { return bias ? O::to_f(bias[n]) : 0.0f; };
    if (epi <= 1) {
        for (int i = tid; i < M * kCtaCols; i += kThreads) {
            const int m = i / kCtaCols, lr = i % kCtaCols;
            const int n = w_row(cb, lr, N, epi);
            if (n >= 0) emit<T>(Y, m, n, N, tile[m * kTileStride + lr] + bias_of(n), 0.0f, act, epi);
        }
    } else {
        constexpr int H = kCtaCols / 2;
        for (int i = tid; i < M * H; i += kThreads) {
            const int m = i / H, p = i % H;
            const float* row = tile + m * kTileStride;
            if (epi == 2) {
                const int n = cb * kCtaCols + 2 * p;
                if (n + 1 < N)
                    emit<T>(Y, m, n / 2, N / 2, row[2 * p] + bias_of(n), row[2 * p + 1] + bias_of(n + 1), 0, 2);
            } else {
                const int j = w_row(cb, p, N, 3);
                if (j >= 0) emit<T>(Y, m, j, N / 2, row[p] + bias_of(j), row[H + p] + bias_of(N / 2 + j), 0, 3);
            }
        }
    }
#endif
}

// Sum the split partials in split order, then the epilogue. One thread per
// output element.
template <typename T>
__global__ void reduce_kernel(const float* __restrict__ partial, int splits, int M, int N, T* __restrict__ Y,
                              const T* __restrict__ bias, int act, int epi) {
    using O = Ops<T>;
    const int out_cols = epi >= 2 ? N / 2 : N;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(M) * out_cols) return;
    const int m = static_cast<int>(i / out_cols), col = static_cast<int>(i % out_cols);
    const size_t stride = static_cast<size_t>(M) * N;
    auto sum = [&](int n) {
        const float* p = partial + static_cast<size_t>(m) * N + n;
        float v = 0.0f;
        for (int s = 0; s < splits; ++s) v += p[s * stride];
        return v + (bias ? O::to_f(bias[n]) : 0.0f);
    };
    if (epi <= 1) {
        emit<T>(Y, m, col, N, sum(col), 0.0f, act, epi);
    } else if (epi == 2) {
        emit<T>(Y, m, col, N / 2, sum(2 * col), sum(2 * col + 1), 0, 2);
    } else {
        emit<T>(Y, m, col, N / 2, sum(col), sum(N / 2 + col), 0, 3);
    }
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

int col_blocks(int N, int epi) {
    return epi == 3 ? (N / 2 + kCtaCols / 2 - 1) / (kCtaCols / 2) : (N + kCtaCols - 1) / kCtaCols;
}

// K chunks per CTA. One CTA fits an SM (the ring is ~96 KB), so K is split
// until the grid covers the SMs about once without spilling into a second,
// mostly idle wave.
int chunks_per_split(int N, int K, int epi, int sms) {
    const int n_chunks = (K + kChunk - 1) / kChunk, blocks = col_blocks(N, epi);
    const int splits = std::clamp(sms / blocks, 1, std::max(1, n_chunks / 4));
    return (n_chunks + splits - 1) / splits;
}

int split_count(int N, int K, int epi, int sms) {
    const int n_chunks = (K + kChunk - 1) / kChunk, cps = chunks_per_split(N, K, epi, sms);
    return (n_chunks + cps - 1) / cps;
}

bool shape_ok(int M, int N, int K, int epi) {
    if (M < 9 || M > 64 || N <= 0 || K <= 0 || (K & 7)) return false;
    if (epi < 0 || epi > 3 || (epi >= 2 && (N & 1))) return false;
    return true;
}

template <typename T, int NT>
void run(const T* X, const T* W, T* Y, int M, int N, int K, const T* bias, int act, int epi, float* partial,
         int splits, int cps, cudaStream_t stream) {
    constexpr int smem = smem_bytes<NT>();
    if constexpr (smem > 48 * 1024) {
        static std::atomic<std::uint32_t> opted_in{0};
        opt_in_dynamic_smem(skinny_kernel<T, NT>, smem, opted_in);
    }
    const dim3 grid(col_blocks(N, epi), splits);
    skinny_kernel<T, NT><<<grid, kThreads, smem, stream>>>(X, W, Y, M, N, K, bias, act, epi, partial, cps);
}

template <typename T>
bool launch_t(const T* X, const T* W, T* Y, int M, int N, int K, const T* bias, int act, int epi, float* ws,
              std::size_t ws_floats, cudaStream_t stream) {
    if (!shape_ok(M, N, K, epi)) return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
    if (misaligned(X) || misaligned(W)) return false;
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    int splits = split_count(N, K, epi, sms), cps = chunks_per_split(N, K, epi, sms);
    if (splits > 1 && (!ws || ws_floats < static_cast<std::size_t>(splits) * M * N)) {
        splits = 1;
        cps = (K + kChunk - 1) / kChunk;
    }
    float* partial = splits > 1 ? ws : nullptr;
    switch ((M + 7) / 8) {
        case 2: run<T, 2>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        case 3: run<T, 3>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        case 4: run<T, 4>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        case 5: run<T, 5>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        case 6: run<T, 6>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        case 7: run<T, 7>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
        default: run<T, 8>(X, W, Y, M, N, K, bias, act, epi, partial, splits, cps, stream); break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    if (partial) {
        const std::size_t outs = static_cast<std::size_t>(M) * (epi >= 2 ? N / 2 : N);
        const int threads = 256;
        reduce_kernel<T><<<static_cast<unsigned>((outs + threads - 1) / threads), threads, 0, stream>>>(
            partial, splits, M, N, Y, bias, act, epi);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

}  // namespace

std::size_t workspace_floats(int M, int N, int K, int epi) {
    if (!shape_ok(M, N, K, epi)) return 0;
    int major = 0;
    const int splits = split_count(N, K, epi, sm_count_and_major(major));
    return splits > 1 ? static_cast<std::size_t>(splits) * M * N : 0;
}

bool launch(const __half* X, const __half* W, __half* Y, int M, int N, int K, const __half* bias, int act, int epi,
            float* ws, std::size_t ws_floats, cudaStream_t stream) {
    return launch_t<__half>(X, W, Y, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

bool launch(const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K,
            const __nv_bfloat16* bias, int act, int epi, float* ws, std::size_t ws_floats, cudaStream_t stream) {
    return launch_t<__nv_bfloat16>(X, W, Y, M, N, K, bias, act, epi, ws, ws_floats, stream);
}

}  // namespace brotensor::detail::cuda::gemm_skinny
